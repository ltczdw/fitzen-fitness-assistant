# -*- coding: utf-8 -*-
"""记忆读后端 A/B 对比实验（实验 A）。

对比两种记忆读取表示（生产代码原路径，仅加一个表示层适配器，不改核心）：
  wiki    = 「Wiki 摘要寻址回读原文」：BM25+向量在 wiki 页摘要上召回 → page_id →
            回读该轮完整原文（生产路径，app/services/memory/wiki.py::WikiAdapter.resolve）。
  compact = 「compact_v1 滑动窗口摘要」：同样的摘要寻址选中同一批轮次，但注入 prompt 的
            是检索摘要文本本身，不回读原文（SummaryOnlyAdapter，只覆盖 resolve 的表示层）。
两臂共享同一份已写入数据与同一摘要，唯一差异是"回读原文 vs 注入摘要"。

切换方式：
  环境变量 FITZEN_MEMORY_AB_BACKEND=wiki|compact|both（或 --backend 参数，默认 both）。
  生产自身的开关是 settings.wiki_enabled / MEMORY_READ_BACKEND；本脚本在 wiki 臂走
  CompactMemory.load 原路径，两臂共用同一条"窗口 + 事实"装配代码。

隔离：在业务库实例上创建一次性 schema fitzen_memab_<hex>，全部读写发生在该 schema，
结束后 DROP CASCADE；不触碰任何业务数据（与 tests/test_wiki_full.py 同款模式）。
确定性：LLM 摘要与 embedding 均不联网——outbox 摘要用脚本内确定性抽取式摘要手写提交，
向量用 pipeline._embedding(…, dimensions=1024) 的确定性哈希向量（查询侧同函数打补丁），
rerank 未配置走规则降级。全程不使用任何真实密钥。

场景（覆盖四类能力，每场景 ≥12 轮使 gold 轮老化出 10 轮窗口）：
  cross_ref   跨轮引用：第 1 轮的深蹲/左膝信息在 11 轮后被问起
  truncation  长对话截断：第 2/3 轮的硬拉辅助装备与体脂率被 14 轮后的提问召回
  tool_obs    工具观察回放：第 1 轮工具观察（含 record_id 等结构化字段）老化后的召回
  entity      实体召回：显式事实（请记住…过敏）+ 老化轮中的计划实体

指标（每场景 × 每臂）：
  marker_hit_rate   gold 标记在最终装配 prompt 中的出现率（命中率口径）
  semantic_faithful gold 标记在"召回块"中逐字出现的比例（忠实度口径）
  tokens_*          ContextBudget 装配报告的 token 开销（recent/semantic/总计）
  latency_ms        load(窗口+事实) + 召回 + 装配 的端到端延迟（中位数/P95，--repeat 次）

用法（项目 venv，backend 目录，需要本机 PostgreSQL 可连）：
  .venv/Scripts/python.exe scripts/memory_ab_compare.py
  FITZEN_MEMORY_AB_BACKEND=wiki .venv/Scripts/python.exe scripts/memory_ab_compare.py --repeat 7
结果写入 backend/evidence/memory-ab-results.json。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db import session as db
from app.db.models import Base

SCHEMA = "fitzen_memab_" + uuid4().hex
_TURN_SUMMARIES: dict[str, str] = {}  # turn_id -> fixture 提供的确定性摘要


def _extractive_summary(messages: list[dict]) -> str:
    """兜底抽取式摘要：与生产 deepseek_summary 的密度相当，但完全确定性。"""
    parts = []
    for m in messages[:3]:
        role = m.get("role")
        if role == "user":
            parts.append(f"用户：{m.get('content', '')[:60]}")
        elif role == "assistant":
            parts.append(f"助手：{m.get('content', '')[:60]}")
        elif role == "tool":
            parts.append(f"工具{m.get('name', '')}返回：{m.get('content', '')[:60]}")
    return "；".join(parts)[:160]


# ---------------------------------------------------------------- 场景样例 ----
def _turn(user: str, assistant: str, tool: dict | None = None, summary: str = "") -> dict:
    return {"user": user, "assistant": assistant, "tool": tool, "summary": summary}


SCENARIOS: list[dict] = [
    {
        "name": "cross_ref",
        "cap": "跨轮引用：第 1 轮的深蹲/左膝在 11 轮后被问起",
        "probe": "我上次说膝盖不舒服，是哪个动作引起的？当时用了多大重量？",
        "markers": ["深蹲", "左膝", "80公斤"],
        "turns": [
            _turn("今天练了腿，深蹲 5组5次 80公斤，左膝有点不舒服",
                  "已记录：深蹲 5x5 80公斤。左膝不适建议降低重量并观察。",
                  summary="用户深蹲后左膝不适，助手建议降低重量观察。"),
            _turn("中午吃了一碗牛肉面", "好的，牛肉面大约500千卡。", summary="午餐牛肉面，约500千卡。"),
            _turn("下午加了个苹果", "苹果约95千卡。", summary="加餐苹果约95千卡。"),
            _turn("帮我查一下鸡胸肉的热量", "100克鸡胸肉约133千卡，蛋白质约24克。", summary="鸡胸肉每100克约133千卡。"),
            _turn("今天状态一般，跑了3公里", "3公里慢跑约消耗210千卡。", summary="慢跑3公里，约210千卡。"),
            _turn("晚上吃了沙拉", "沙拉注意酱料热量。", summary="晚餐沙拉，提醒酱料热量。"),
            _turn("卧推推了50公斤5组", "已记录卧推 5组 50公斤。", summary="卧推5组50公斤。"),
            _turn("引体向上做了8个", "已记录引体向上 8个。", summary="引体向上8个。"),
            _turn("明天练背", "明天练背安排划船与高位下拉。", summary="明日练背计划。"),
            _turn("今天喝了2升水", "饮水量达标，继续保持。", summary="饮水2升。"),
            _turn("周末想去爬山", "爬山属于中等强度有氧，注意补给。", summary="周末爬山计划。"),
            _turn("今天拉伸了15分钟", "拉伸15分钟有助恢复。", summary="拉伸15分钟。"),
        ],
    },
    {
        "name": "truncation",
        "cap": "长对话截断：第 2/3 轮信息被 14 轮后的提问召回",
        "probe": "我之前硬拉用了什么辅助装备？当时的体脂率是多少？",
        "markers": ["硬拉", "助力带", "22%"],
        "turns": [
            _turn("今天练了硬拉", "硬拉是很好的后链动作。", summary="用户练习硬拉。"),
            _turn("硬拉 100公斤 3组5次，用了助力带", "已记录：硬拉 100公斤 3x5，助力带辅助。",
                  summary="用户进行硬拉训练并使用助力带，助手已记录。"),
            _turn("顺便说下我体脂率 22%", "体脂率22%在正常范围上限附近。", summary="用户告知当前体脂率，助手给出范围评价。"),
            _turn("早餐吃了燕麦", "燕麦约150千卡/40克。", summary="早餐燕麦。"),
            _turn("中午吃了鸡胸肉饭", "约600千卡。", summary="午餐鸡胸肉饭约600千卡。"),
            _turn("下午游泳半小时", "自由泳半小时约300千卡。", summary="游泳半小时约300千卡。"),
            _turn("晚上加班没练", "注意休息。", summary="晚间未训练。"),
            _turn("第二天练了肩", "已记录肩部训练。", summary="肩部训练。"),
            _turn("侧平举 8公斤 4组", "已记录侧平举 4组 8公斤。", summary="侧平举8公斤4组。"),
            _turn("今天蛋白质够不够", "按你体重建议每公斤1.6克。", summary="蛋白质摄入建议。"),
            _turn("晚餐吃了三文鱼", "三文鱼富含omega-3。", summary="晚餐三文鱼。"),
            _turn("睡前一杯牛奶", "牛奶约120千卡。", summary="睡前牛奶约120千卡。"),
            _turn("周末休息", "休息日注意活动度。", summary="周末休息。"),
            _turn("周一开始增肌期", "增肌期热量盈余300千卡起步。", summary="下周开始增肌期。"),
        ],
    },
    {
        "name": "tool_obs",
        "cap": "工具观察回放：第 1 轮工具观察（含 record_id）老化后的召回",
        "probe": "上次工具帮我记录卧推的时候，返回的记录编号是多少？重量几公斤？",
        "markers": ["卧推", "12345", "60"],
        "turns": [
            _turn("记录卧推 60公斤 4组 每组8次",
                  "已保存训练记录。",
                  tool={"name": "workout_record",
                        "arguments": {"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8},
                        "result": {"status": "saved", "record_id": 12345, "exercise": "卧推",
                                   "weight_kg": 60, "sets": 4, "reps": 8}},
                  summary="用户记录卧推训练，工具保存成功。"),
            _turn("再记一下午饭：两个鸡蛋", "已记录两个鸡蛋约140千卡。",
                  tool={"name": "nutrition_record",
                        "arguments": {"foods": [{"name": "鸡蛋", "quantity": 2}]},
                        "result": {"status": "saved", "kcal": 140}},
                  summary="记录午餐两个鸡蛋，约140千卡。"),
            _turn("今天练完很累", "注意睡眠与恢复。", summary="用户反馈疲劳，建议恢复。"),
            _turn("补充问了下蛋白粉", "训练后30分钟内补充较好。", summary="蛋白粉补充时机建议。"),
            _turn("昨天没睡好", "睡眠不足会影响力量表现。", summary="睡眠不足提醒。"),
            _turn("今天做有氧半小时", "已记录有氧30分钟。", summary="有氧30分钟。"),
            _turn("椭圆机 20分钟", "约200千卡。", summary="椭圆机20分钟约200千卡。"),
            _turn("晚上吃了牛肉", "牛肉富含铁与蛋白质。", summary="晚餐牛肉。"),
            _turn("体重76公斤", "已记录体重76公斤。", summary="体重76公斤。"),
            _turn("下周想练腿", "腿部安排深蹲与腿举。", summary="下周练腿预告。"),
            _turn("今天状态不错", "保持节奏。", summary="状态良好。"),
            _turn("记录身高180", "已记录身高180厘米。", summary="身高180厘米。"),
        ],
    },
    {
        "name": "entity",
        "cap": "实体召回：显式事实（过敏）+ 老化轮中的计划实体",
        "probe": "我的过敏原是什么？我现在在跟哪个训练计划？",
        "markers": ["乳清蛋白", "过敏", "5/3/1"],
        "turns": [
            _turn("请记住我对乳清蛋白过敏", "已记住：你对乳清蛋白过敏。",
                  summary="用户明确：对乳清蛋白过敏。"),
            _turn("我开始跟 5/3/1 计划训练", "5/3/1 是经典的周期力量计划。", summary="用户开始新的周期性力量训练计划。"),
            _turn("今天深蹲日 70公斤", "已记录深蹲日。", summary="深蹲日70公斤。"),
            _turn("吃了鸡胸肉", "约133千卡/100克。", summary="鸡胸肉133千卡。"),
            _turn("今天做了瑜伽", "瑜伽算恢复活动。", summary="瑜伽恢复活动。"),
            _turn("卧推日 45公斤", "已记录卧推日。", summary="卧推日45公斤。"),
            _turn("硬拉日 90公斤", "已记录硬拉日。", summary="硬拉日90公斤。"),
            _turn("辅助训练做了引体", "引体向上是优秀的背部动作。", summary="辅助引体向上。"),
            _turn("今天体重75.5公斤", "已记录体重。", summary="体重75.5公斤。"),
            _turn("吃了三文鱼沙拉", "约450千卡。", summary="三文鱼沙拉约450千卡。"),
            _turn("今天休息", "休息也是训练的一部分。", summary="休息日。"),
            _turn("明天推日", "推日安排卧推与推举。", summary="明日推日预告。"),
        ],
    },
]

SYSTEM_PROMPT = "你是 FitZen 健身助手。依据记忆与知识回答，不得编造。"


# ---------------------------------------------------------------- 环境搭建 ----
def setup_schema():
    admin = create_engine(settings.database_url, isolation_level="AUTOCOMMIT",
                          connect_args={"connect_timeout": 5})
    with admin.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {SCHEMA}"))
    # 工作引擎必须用默认隔离级别：AUTOCOMMIT 下 connection.begin() 没有真实事务块，
    # join_transaction_mode='create_savepoint' 的 SAVEPOINT 会报 NoActiveSqlTransaction。
    worker = create_engine(settings.database_url,
                           connect_args={"connect_timeout": 5})
    engine = worker.execution_options(schema_translate_map={None: SCHEMA})
    Base.metadata.create_all(engine)
    connection = engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(bind=connection, expire_on_commit=False,
                           join_transaction_mode="create_savepoint")
    return admin, connection, transaction, factory


def hash_embedding_patch():
    from app.services.rag.pipeline import _embedding
    return lambda texts, **kw: [_embedding(t, dimensions=1024) for t in texts]


def drain_outbox() -> int:
    """把待索引轮次全部索引掉（fixture 摘要 + 确定性哈希向量）。

    生产只索引老化出窗的轮次；这里对场景内全部完成轮次入队，保证短场景可测。
    两臂共享同一份摘要，不影响 A/B 公平性。"""
    from app.services.memory import compact
    from app.services.rag.pipeline import _embedding

    n = 0
    while True:
        event = compact.claim_event()
        if event is None:
            break
        summary = (_TURN_SUMMARIES.get(event["turn_id"])
                   or _extractive_summary(event["messages"]))
        vector = _embedding(summary, dimensions=1024)
        assert compact.commit_index(event, summary, vector), "commit_index rejected event"
        n += 1
    return n


def write_conversation(cid: str, turns: list[dict]) -> None:
    from app.schemas.chat import ChatRequest
    from app.services.memory import compact

    for i, t in enumerate(turns):
        req = ChatRequest(message=t["user"], conversation_id=cid, request_id=f"{cid}-turn{i}")
        tid, replay = compact.begin_turn(req)
        assert replay is None, "benchmark request_id collision"
        if t.get("tool"):
            compact.append_observation(tid, t["tool"]["name"], t["tool"]["arguments"],
                                       t["tool"]["result"])
        compact.finish_turn(tid, t["assistant"], response={"answer": t["assistant"]})
        _TURN_SUMMARIES[tid] = t["summary"]


def enqueue_all(cid: str) -> None:
    from app.db.memory_models import MemoryTurn
    from app.services.memory import compact

    with compact.session() as s, s.begin():
        rows = list(s.scalars(select(MemoryTurn).where(
            MemoryTurn.conversation_id == cid, MemoryTurn.status == "completed",
            MemoryTurn.deleted_at.is_(None), MemoryTurn.index_status == "pending_window")))
        for row in rows:
            row.index_status = "pending"
            compact.enqueue(s, row)


# ---------------------------------------------------------------- 双臂读取 ----
async def build_context(cid: str, probe: str, arm: str) -> tuple[dict, dict]:
    """按生产同款路径装配 prompt 上下文。返回 (BudgetedMessages, state)。"""
    from app.services.context.budget import context_budget
    from app.services.memory import compact
    from app.services.memory.redis_memory import memory_service
    from app.services.rag.blocking import run_retrieval
    from app.services.rag.retrieval import retrieval_service

    state = await memory_service.load("demo-user", cid, "")  # 窗口 + 事实（不含召回）
    rows = compact.recent_rows(cid)
    excluded = [r.turn_id for r in rows]

    def retrieve():
        from app.services.memory.wiki import WikiAdapter
        adapter = WikiAdapter(excluded) if arm == "wiki" else SummaryOnlyAdapter(excluded)
        return retrieval_service.search(probe, adapter, 3)

    hits = await run_retrieval(retrieve)
    state["retrieved_history"] = [{"turn_id": h.chunk_id, "content": h.content,
                                   "score": h.score, **h.metadata} for h in hits]
    messages = context_budget.assemble(system=SYSTEM_PROMPT, query=probe,
                                       state=state, hits=[])
    return messages, state


class SummaryOnlyAdapter:
    """compact 臂适配器：与 WikiAdapter 完全相同的召回与校验，仅把回读内容换成摘要。

    即"compact_v1 滑动窗口摘要"表示：摘要用于定位，也用于注入，不回读原文。"""

    corpus = "memory"

    def __init__(self, exclude=()):
        from app.services.memory.wiki import WikiAdapter
        self._impl = WikiAdapter(exclude)

    def filters(self):
        return self._impl.filters()

    def lexical(self, query, k):
        return self._impl.lexical(query, k)

    def vector(self, vector, k):
        return self._impl.vector(vector, k)

    def resolve(self, candidates):
        from dataclasses import replace

        from app.services.memory.compact import session
        from app.db.wiki_models import WikiPage
        from app.services.rag.retrieval import RetrievedChunk

        raw = self._impl.resolve(candidates)
        with session() as s:
            summaries = {}
            for hit in raw:
                page = s.get(WikiPage, hit.chunk_id)
                summaries[hit.chunk_id] = (page.summary or "") if page else ""
        return [replace(hit, content=summaries.get(hit.chunk_id, ""), source="wiki-summary")
                for hit in raw]

    def validate(self, hits):
        return self._impl.validate(hits)


# ---------------------------------------------------------------- 指标 ----
def measure(messages, state, markers: list[str]) -> dict:
    contents = [m.get("content") or "" for m in messages]
    full_text = "\n".join(contents)
    retrieved_text = "\n".join(item.get("content") or ""
                               for item in state.get("retrieved_history", []))
    report = getattr(messages, "report", {}) or {}
    tokens_by_type = report.get("tokens_by_type", {})
    return {
        "markers_hit": [m for m in markers if m in full_text],
        "markers_missed": [m for m in markers if m not in full_text],
        "marker_hit_rate": sum(1 for m in markers if m in full_text) / len(markers),
        "semantic_faithful_rate": sum(1 for m in markers if m in retrieved_text) / len(markers),
        "retrieved_turns": [item.get("turn_id") for item in state.get("retrieved_history", [])],
        "retrieved_chars": len(retrieved_text),
        "used_tokens": report.get("used_tokens"),
        "tokens_by_type": tokens_by_type,
        "degraded_reason": state.get("degraded_reason"),
    }


def _median(values):
    vs = sorted(values)
    n = len(vs)
    return vs[n // 2] if n % 2 else (vs[n // 2 - 1] + vs[n // 2]) / 2


def _p95(values):
    vs = sorted(values)
    return vs[min(len(vs) - 1, int(len(vs) * 0.95))]


# ---------------------------------------------------------------- 主流程 ----
async def main() -> None:
    parser = argparse.ArgumentParser(description="记忆读后端 A/B 对比")
    parser.add_argument("--backend",
                        default=os.environ.get("FITZEN_MEMORY_AB_BACKEND", "both"),
                        choices=["wiki", "compact", "both"])
    parser.add_argument("--repeat", type=int, default=9, help="延迟测量重复次数")
    parser.add_argument("--output", default="evidence/memory-ab-results.json")
    args = parser.parse_args()
    arms = ["wiki", "compact"] if args.backend == "both" else [args.backend]

    admin, connection, transaction, factory = setup_schema()
    results = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "schema": SCHEMA, "arms": arms, "scenarios": []}
    try:
        with (patch.object(db, "DB_READY", True),
              patch.object(db, "SessionLocal", factory),
              patch.object(settings, "redis_enabled", False),
              patch.object(settings, "memory_read_backend", "compact_v1"),
              patch.object(settings, "wiki_enabled", True)):
            from app.services.memory import wiki as wiki_mod

            wiki_mod.initialize()
            from app.services.rag.embedding_client import embedding_client

            with patch.object(embedding_client, "embed", hash_embedding_patch()):
                for scenario in SCENARIOS:
                    cid = f"memab-{scenario['name']}"
                    write_conversation(cid, scenario["turns"])
                enqueue_all_each()
                drain_outbox()

                for scenario in SCENARIOS:
                    cid = f"memab-{scenario['name']}"
                    entry = {"name": scenario["name"], "cap": scenario["cap"],
                             "probe": scenario["probe"], "markers": scenario["markers"],
                             "arms": {}}
                    for arm in arms:
                        messages, state = await build_context(cid, scenario["probe"], arm)
                        m = measure(messages, state, scenario["markers"])
                        lat = []
                        for _ in range(args.repeat):
                            t0 = time.perf_counter()
                            await build_context(cid, scenario["probe"], arm)
                            lat.append((time.perf_counter() - t0) * 1000)
                        m["latency_ms_p50"] = round(_median(lat), 2)
                        m["latency_ms_p95"] = round(_p95(lat), 2)
                        m["assembled_preview"] = [
                            {"role": msg.get("role"), "content": (msg.get("content") or "")[:240]}
                            for msg in messages][:8]
                        entry["arms"][arm] = m
                    results["scenarios"].append(entry)
                    w, c = entry["arms"].get("wiki", {}), entry["arms"].get("compact", {})
                    print(f"[{scenario['name']}] {scenario['cap']}")
                    for arm in arms:
                        a = entry["arms"][arm]
                        print(f"  {arm:>7}: hit={a['marker_hit_rate']:.2f} "
                              f"missed={a['markers_missed']} "
                              f"semantic_faithful={a['semantic_faithful_rate']:.2f} "
                              f"tokens={a['used_tokens']} (semantic={a['tokens_by_type'].get('semantic', 0)}, "
                              f"recent={a['tokens_by_type'].get('recent', 0)}) "
                              f"retrieved_chars={a['retrieved_chars']} "
                              f"lat_p50={a['latency_ms_p50']}ms")
    finally:
        transaction.rollback()
        connection.close()
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {SCHEMA} CASCADE"))
        admin.dispose()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nresults -> {out_path}")


def enqueue_all_each() -> None:
    for scenario in SCENARIOS:
        enqueue_all(f"memab-{scenario['name']}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
