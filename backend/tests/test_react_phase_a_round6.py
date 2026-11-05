# -*- coding: utf-8 -*-
"""ReAct 阶段A第六轮固定复验：C6-01…C6-05 原合同闭环实例（round6 任务 §3，固定8例）。

只落实原合同批内隔离、调用事实/身份与异常终态/恢复关联；不新增问题编号。
预期来自第六轮合同，不取自实现现状。失败沿用 RA-08-R5/RA-14-R5/RA-04-R5 A/B 归属。
静态待验证点（graph:770-779 dispatched 估算、771 event_id 跨批、609起恢复无 call_facts、
938 仅查 call_facts is None）由 C6-02/C6-03/C6-05/C6-04 分别实测。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.services.agent.react_graph import (
    ReactDependencies,
    execute_react_batch,
    finalize_node,
)
from app.services.agent.react_prompts import RECOVERY_OBSERVATION_NOTICE

from tests.react_phase_a_support import (
    RecordingExecutor,
    assistant,
    batch_state,
    committed_receipt,
    ref_of,
    tool_call,
)

SRC_RECORD = "帮我记录午餐吃了牛肉面和可乐"
SRC_TWO_RECORDS = "早餐吃了燕麦粥,帮我记录。晚餐吃了鸡胸肉,也帮我记录"

BOUND_CANDIDATES = [{
    "id": "task-1", "task_type": "nutrition_record", "action": "record",
    "text": SRC_RECORD, "source_start": 0, "source_end": len(SRC_RECORD), "depends_on": [],
}]

ILLEGAL_METADATA_PAYLOAD = {"status": "completed", "commit_state": "committed",
                            "receipt_refs": ["r1"], "summary": float("nan")}


def _deps(executor=None, request_id="req-6", **options):
    return ReactDependencies(llm=object(), tool_executor=executor, user_id="app-user-1",
                             conversation_id="conv-1", request_id=request_id, **options)


def _run_batch(state, executor):
    return asyncio.run(execute_react_batch(state, _deps(executor)))


def _facts_by_call(result):
    facts = {}
    for fact in result.get("call_facts") or []:
        facts.setdefault(fact["tool_call_id"], []).append(fact)
    return facts


def _touch_by_call(executor):
    touched = {}
    for call in executor.calls:
        touched[call["name"]] = touched.get(call["name"], 0) + 1
    return touched


# ---------------------------------------------------------------------------
# C6-01：合法前项 → 非法写观察 → 合法独立后项（1例）
# ---------------------------------------------------------------------------
def test_c6_01_legal_prior_illegal_write_legal_later_all_touched():
    """三项实际触达；非法项受限/零可信提交；前后合法结果保留；每call一个tool结果；
    有限JSON。不以两项证明后项继续——中间非法项两侧都有真实独立合法项。"""
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "read_profile", {}),
        tool_call("t2", "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}}),
        tool_call("t3", "read_current_plan", {}),
    ]), candidates=BOUND_CANDIDATES)
    executor = RecordingExecutor({
        "read_profile": {"status": "completed", "result": {"profile": "ok"}},
        "record_nutrition": ILLEGAL_METADATA_PAYLOAD,
        "read_current_plan": {"status": "completed", "result": {"plan": "ok"}},
    })
    result = _run_batch(state, executor)
    # 三项全部实际触达
    assert len(executor.calls) == 3 and executor.names == [
        "read_profile", "record_nutrition", "read_current_plan"], (
        f"C6-01: 三项应全部实际触达: {executor.names}")
    # 非法项受限、零可信提交
    illegal_obs = result["observations"][1]
    assert illegal_obs["status"] != "completed" and result["committed_tasks"] == [], (
        f"C6-01: 非法项未受限: {illegal_obs}, committed={result['committed_tasks']}")
    # 前后合法结果保留
    assert result["observations"][0]["result"]["profile"] == "ok"
    assert result["observations"][2]["result"]["plan"] == "ok"
    # 每call一个tool结果、有限JSON
    tool_ids = [m.get("tool_call_id") for m in result["messages"] if m.get("role") == "tool"]
    assert sorted(tool_ids) == ["t1", "t2", "t3"], f"C6-01: 配对异常: {tool_ids}"
    for message in result["messages"]:
        if message.get("role") == "tool":
            json.loads(message["content"], parse_constant=lambda v: (_ for _ in ()).throw(
                ValueError(f"非有限JSON: {v}")))


# ---------------------------------------------------------------------------
# C6-02：真实派发与本地门禁/复用事实（1例，四场景依次）
# ---------------------------------------------------------------------------
def test_c6_02_call_facts_dispatched_matches_real_touch_four_paths():
    """四路径依次执行，call_facts.dispatched 与执行器真实触达逐call对照：
    合法读、合法写触达应记true；坏ref拒绝、同目标复用零触达应记false。
    有效目标/绑定/门禁来源应正确（不由领域error决定，不凭write+matched+有executor猜）。"""
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "read_profile", {}),                       # 场景a: 合法读（触达）
        tool_call("t2", "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}}),     # 场景b: 合法写（触达）
        tool_call("t3", "record_nutrition",
                  {"source_ref": {"start": 0, "end": 3, "exact_text": "帮我记XX"}}),  # 场景c: 坏ref拒绝
        tool_call("t4", "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}}),     # 场景d: 同目标复用（零触达）
    ]), candidates=BOUND_CANDIDATES)
    executor = RecordingExecutor({
        "read_profile": {"status": "completed", "result": {"profile": "ok"}},
        "record_nutrition": committed_receipt("rcpt-fixed"),
    })
    result = _run_batch(state, executor)
    facts = _facts_by_call(result)
    assert set(facts) == {"t1", "t2", "t3", "t4"}, f"C6-02: 每call应有事实: {sorted(facts)}"
    # 真实触达：t1 读、t2 写触达；t3 拒绝零触达；t4 复用零触达
    assert executor.names == ["read_profile", "record_nutrition"], (
        f"C6-02: 实际触达应只有合法读与合法写: {executor.names}")
    # 合同：读/写触达记dispatched=true
    assert facts["t1"][0]["dispatched"] is True, (
        f"C6-02: 合法读触达应记dispatched=true（实现按permission=read估算为false）: "
        f"{facts['t1']}")
    assert facts["t2"][0]["dispatched"] is True, (
        f"C6-02: 合法写触达应记dispatched=true: {facts['t2']}")
    assert facts["t3"][0]["dispatched"] is False, (
        f"C6-02: 坏ref拒绝应记false: {facts['t3']}")
    assert facts["t4"][0]["dispatched"] is False, (
        f"C6-02: 同目标复用应记false: {facts['t4']}")
    assert facts["t2"][0]["gate_reason"] is None, f"C6-02: 合法写无门禁原因: {facts['t2']}"
    assert facts["t4"][0]["gate_reason"] is None, f"C6-02: 复用无门禁原因: {facts['t4']}"
    # 有效目标/绑定来源正确：t2 的绑定来自应用（bound_task_id 为应用生成ID）
    assert facts["t2"][0]["bound_task_id"] and \
        facts["t2"][0]["bound_task_id"].startswith("react-record_nutrition-"), (
        f"C6-02: 绑定应来自应用: {facts['t2']}")
    assert facts["t1"][0]["permission"] == "read" and facts["t2"][0]["permission"] == "write"
    # T11校准：事实 observation 快照对应该 call 的规范结果（追加于原失败断言之后，
    # 保持第六轮首失败证据可比；第七轮实现字段，旧业务缺失时此处失败）
    for fact, obs in zip(result.get("call_facts") or [], result["observations"]):
        assert fact.get("observation") == obs, (
            f"C6-02: 事实observation快照未对应该call规范结果: {fact} vs {obs}")


# ---------------------------------------------------------------------------
# C6-03：跨批相同原call ID（1例）
# ---------------------------------------------------------------------------
def test_c6_03_same_call_id_across_batches_event_distinguishable():
    """两个合法批次分别使用同一原call ID（批内唯一性不变）；应用event_id跨批可区分、
    原call ID保留、各关联本批工具/目标/规范观察；不用同ID先前事实替代另一执行。"""
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}})]),
        candidates=BOUND_CANDIDATES)
    executor = RecordingExecutor({"record_nutrition": committed_receipt("rcpt-1")})
    first = _run_batch(state, executor)
    state.update(first)
    # T11校准（round5裁定口径修正）：第二批同ref/同目标 → 走复用路径（非新执行）。
    # 全序列一次真实写触达（第一批）、两个普通调用事件（第一批写+第二批复用）；
    # 两批各自产生 t1 的普通调用事实，event_id 必须不同且各自关联本次规范观察。
    assert executor.names == ["record_nutrition"], (
        f"C6-03前置: 全序列应恰一次真实写触达: {executor.names}")
    state["messages"].append(assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}})]))
    second = _run_batch(state, executor)
    facts_t1 = [f for f in second.get("call_facts") or [] if f["tool_call_id"] == "t1"]
    first_facts_t1 = [f for f in first.get("call_facts") or [] if f["tool_call_id"] == "t1"]
    assert facts_t1, "C6-03前置: 第二批应产生t1事实"
    assert len(first_facts_t1) == 1 and len(facts_t1) == 1, (
        f"C6-03: 两批各应产生一个普通调用事件: {first_facts_t1}, {facts_t1}")
    event_ids = {f["event_id"] for f in first_facts_t1} | {f["event_id"] for f in facts_t1}
    assert len(event_ids) == 2, (
        f"C6-03: event_id跨批不可区分（仅request_id+原call ID）: "
        f"{sorted(event_ids)}")
    assert {f["tool_call_id"] for f in facts_t1} == {"t1"}, "C6-03: 原call ID应保留"
    # 各自关联本次规范观察（第七轮实现字段；旧业务缺失时该断言在事件ID断言之后）
    assert first_facts_t1[0].get("observation") == first["observations"][0], (
        f"C6-03: 第一批事件未关联本次规范观察: {first_facts_t1[0]}")
    assert facts_t1[0].get("observation") == second["observations"][0], (
        f"C6-03: 第二批事件未关联本次规范观察: {facts_t1[0]}")


# ---------------------------------------------------------------------------
# C6-04：异常已有写状态、空或无关事实（4例 = 2载荷 × 2事实形态）
# ---------------------------------------------------------------------------
def _abnormal_state_with_facts(payload, call_facts):
    """真实mock写执行后，构造绑定/目标/调用事实关联丢失的异常状态，
    注入指定 call_facts 形态（空列表 / 仅另一真实合法只读调用的应用事实）。"""
    state = _write_call_state(BOUND_CANDIDATES)
    executor = RecordingExecutor({"record_nutrition": payload})
    result = asyncio.run(execute_react_batch(state, _deps(executor)))
    abnormal = {
        "messages": list(result["messages"]),
        "observations": list(result["observations"]),
        "bound_tasks": [],
        "goal_results": {},
        "committed_tasks": [],
        "covered_goals": [],
        "intent_candidates": [],
        "unbound_write_calls": [],
        "call_facts": call_facts,
        "pending_goals": [],
        "recovery_counter": 0,
        "tool_steps": result["tool_steps"],
        "consecutive_no_progress": 0,
        "terminated": False,
        "termination_reason": "",
        "status": "running",
        "protocol_error": "",
        "final_answer": "",
        "safety_level": "normal",
        "source_text": SRC_RECORD,
        "source_scope": None,
        "user_id": "app-user-1",
        "conversation_id": "conv-1",
        "request_id": "req-6",
    }
    return abnormal


def _unrelated_read_fact():
    """另一真实合法只读调用的应用事实（真实批次产生，非手造）。"""
    state = batch_state("你好", assistant(tool_calls=[tool_call("r1", "read_profile", {})]))
    executor = RecordingExecutor({"read_profile": {"status": "completed"}})
    result = asyncio.run(execute_react_batch(state, _deps(executor)))
    assert result.get("call_facts"), "R6前置: 只读事实应由真实批次产生"
    return result["call_facts"]


UNBOUND_WRITE_PAYLOADS = [
    {"status": "completed", "commit_state": "unknown"},
    {"status": "completed", "commit_state": "committed", "receipt_refs": ["r1"]},
]


@pytest.mark.parametrize("payload,fact_form", [
    (UNBOUND_WRITE_PAYLOADS[0], []),
    (UNBOUND_WRITE_PAYLOADS[1], []),
    (UNBOUND_WRITE_PAYLOADS[0], None),   # None 占位：由 _unrelated_read_fact() 填充
    (UNBOUND_WRITE_PAYLOADS[1], None),
], ids=["unknown-empty-facts", "committed-empty-facts",
        "unknown-unrelated-read-fact", "committed-unrelated-read-fact"])
def test_c6_04_abnormal_write_state_with_empty_or_unrelated_facts(payload, fact_form):
    """异常已有写状态 + call_facts 为空列表 / 仅另一真实合法只读事实：
    仍非complete、非空准确受限说明、不转发全部保存成功、不自动重试写。
    保留原字段缺失两例（call_facts为None的防御路径）不替换。"""
    if fact_form is None:
        fact_form = _unrelated_read_fact()
    abnormal = _abnormal_state_with_facts(payload, list(fact_form))
    # T11校准：finalize 前追加既定"全部保存成功"模型答复（核受限分支不转发）；
    # finalize 零执行器触达。
    finalize_executor = RecordingExecutor()
    abnormal["messages"].append({"role": "assistant", "content": "全部保存成功"})
    final = finalize_node(abnormal, _deps(finalize_executor))
    assert len(finalize_executor.calls) == 0, (
        f"C6-04/T11: finalize不得触达执行器: {finalize_executor.names}")
    assert final["status"] != "complete", (
        f"C6-04: 异常已有写状态（call_facts={fact_form!r}）判complete: {final['status']!r}")
    assert final["final_answer"].strip(), (
        f"C6-04: 受限说明为空: {final['final_answer']!r}")
    assert "全部保存成功" not in final["final_answer"], (
        f"C6-04: 转发肯定保存宣称: {final['final_answer']!r}")
    # 准确受限说明：关联不足/需要确认（追加于原失败断言之后）
    assert ("确认" in final["final_answer"] or "关联" in final["final_answer"]
            or "补充" in final["final_answer"]), (
        f"C6-04/T11: 受限说明未准确反映关联不足: {final['final_answer']!r}")


def _write_call_state(candidates):
    return batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}})]),
        candidates=candidates)


# ---------------------------------------------------------------------------
# C6-05：恢复观察与应用事实对应（1例）
# ---------------------------------------------------------------------------
def test_c6_05_recovery_observation_app_fact_association():
    """真实阻塞→可信提交→恰一次恢复流程：恢复实际触达、原call ID/序号及规范观察
    具有应用事实关联（等价关联结构可接受）；原阻塞与恢复身份可区分；不伪造
    tool_calls/第二tool结果；共用预算；成功后重选零重派保持。"""
    candidates = [
        {"id": "task-1", "task_type": "nutrition_record", "action": "record",
         "text": "早餐吃了燕麦粥", "source_start": 0, "source_end": 7, "depends_on": []},
        {"id": "goal-c1", "task_type": "nutrition_advice", "action": "question",
         "text": "帮我记录", "source_start": 8, "source_end": 12, "depends_on": ["task-1"]},
    ]

    def policy(nth, messages):
        if nth == 1:
            return {"role": "assistant", "content": "", "tool_calls": [
                tool_call("t1", "advise_nutrition",
                          {"source_ref": {"start": 8, "end": 12, "exact_text": "帮我记录"}})]}
        if nth == 2:
            return {"role": "assistant", "content": "", "tool_calls": [
                tool_call("t2", "record_nutrition",
                          {"source_ref": ref_of(SRC_TWO_RECORDS, "早餐吃了燕麦粥")})]}
        return {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t3", "advise_nutrition",
                      {"source_ref": {"start": 8, "end": 12, "exact_text": "帮我记录"}})]}

    llm = FakeLLMPolicy6(policy)
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-fixed"),
        "advise_nutrition": {"status": "completed", "result": {"advice": "ok"}},
    })
    deps = _deps(executor, max_tool_steps=16)
    state = batch_state(SRC_TWO_RECORDS, assistant(tool_calls=[
        tool_call("t1", "advise_nutrition",
                  {"source_ref": {"start": 8, "end": 12, "exact_text": "帮我记录"}})]),
        candidates=candidates)
    state.update(asyncio.run(execute_react_batch(state, deps)))  # turn1: advise blocked
    state["messages"].append(assistant(tool_calls=[
        tool_call("t2", "record_nutrition",
                  {"source_ref": ref_of(SRC_TWO_RECORDS, "早餐吃了燕麦粥")})]))
    state.update(asyncio.run(execute_react_batch(state, deps)))  # turn2: 提交+恢复
    # 恢复实际触达恰好一次
    advise_touches = [c for c in executor.calls if c["name"] == "advise_nutrition"]
    assert len(advise_touches) == 1, f"C6-05: 恢复应恰好触达一次: {executor.names}"
    # 恢复事件：原call ID/序号/规范观察（等价关联结构）
    events = []
    for message in state["messages"]:
        if message.get("role") == "user" and str(message.get("content", "")).startswith(
                RECOVERY_OBSERVATION_NOTICE):
            events.extend(json.loads(
                message["content"][len(RECOVERY_OBSERVATION_NOTICE):]))
    assert len(events) == 1 and events[0]["tool_call_id"] == "t1" and \
        events[0]["recovery_index"] == 1, f"C6-05: 恢复事件关联不完整: {events}"
    assert events[0]["observation"].get("status") == "completed", (
        f"C6-05: 恢复规范观察缺失: {events[0]}")
    # 原阻塞与恢复身份可区分：t1 原tool结果仍blocked，恢复观察为completed
    tool_msgs = {m.get("tool_call_id"): m for m in state["messages"]
                 if m.get("role") == "tool"}
    assert json.loads(tool_msgs["t1"]["content"])["status"] == "blocked_dependency", (
        "C6-05: 原阻塞tool结果应保持")
    # 成功后重选零重派（同goal-c1 ref 8:12）——T10校准：以实际总触达数为准，
    # 移除旧列表OR短路；重选前存总数、重选后读最新总数必须相等。
    total_touches_before_reselect = len(executor.calls)
    assert total_touches_before_reselect == 2, (
        f"C6-05前置: 三次普通调用+一次恢复, 执行器实际触达应为2(record+advise): "
        f"{executor.names}")
    state["messages"].append(assistant(tool_calls=[
        tool_call("t3", "advise_nutrition",
                  {"source_ref": {"start": 8, "end": 12, "exact_text": "帮我记录"}})]))
    state.update(asyncio.run(execute_react_batch(state, deps)))
    assert len(executor.calls) == total_touches_before_reselect, (
        f"C6-05/T10: 重选后实际触达应零增量: before={total_touches_before_reselect}, "
        f"after={len(executor.calls)}, names={executor.names}")
    # 共用预算：三次普通调用+一次恢复 tool_steps==4；无伪造第二tool结果
    assert state["tool_steps"] == 4, (
        f"C6-05/T10: tool_steps应为4（3普通+1恢复）: {state['tool_steps']}")
    tool_ids = [m.get("tool_call_id") for m in state["messages"] if m.get("role") == "tool"]
    assert sorted(tool_ids) == ["t1", "t2", "t3"], f"C6-05: tool结果异常: {tool_ids}"


class FakeLLMPolicy6:
    def __init__(self, policy):
        self.policy = policy
        self.calls = []

    async def complete_message(self, messages, temperature=0.0, tools=None, tool_choice=None):
        self.calls.append([dict(m) for m in messages])
        return dict(self.policy(len(self.calls), messages))
