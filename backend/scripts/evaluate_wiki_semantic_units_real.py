"""Real-provider quality comparison for Wiki semantic-unit context assembly.

Only synthetic, non-personal pages are written to a new isolated SQLite
database.  The application's existing settings loader supplies provider
credentials; this script never reads or prints .env contents, prompts, provider
bodies, or connection strings.  All provider calls use the existing LLMClient,
BudgetStore ledger, request/stage/deadline/retry limits and daily enforcement.

The locator plan is scripted to the known target pages so the comparison
measures the semantic-unit extraction and answer-assembly quality rather than
the locator.  Extraction and every final answer are real model calls.

Run:
  backend/.venv/Scripts/python backend/scripts/evaluate_wiki_semantic_units_real.py \
      --output evidence/.../real-quality.json
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import socket
import sys
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

BACKEND = Path(__file__).resolve().parents[1]
for path in (BACKEND, BACKEND / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.core import clock
from app.core.config import settings
from app.db.memory_models import MemoryTurn
from app.db.model_budget_models import ModelCallLedger, ModelQuota
from app.schemas.chat import ChatRequest
from app.services.agent import dialogue_context, semantics
from app.services.context import budget as budget_module
from app.services.context.budget import ContextBudgetExceeded, ContextBudgetManager
from app.services.llm.budget import BudgetStore, model_scope, store_override
from app.services.llm.client import LLMClient, LLMGatewayError
from app.services.memory import compact, semantic_units
from app.services.memory import wiki_context as wc
from app.services.telemetry import RequestMetrics, active_metrics


CASES = [
    {
        "id": "stable_allergy_old", "query": "我今天午餐可以吃花生酱吗？",
        "turns": [("我对花生过敏，这是长期禁忌。", "已记录。")], "ages": [500],
        "gt": "不得建议花生或含花生食物；说明这是长期过敏禁忌。",
        "window": 8000,
    },
    {
        "id": "stable_city_old", "query": "我住在哪个城市？",
        "turns": [("我长期住在杭州。", "已记录。")], "ages": [700],
        "gt": "回答杭州；长期稳定信息不因年龄被丢弃。", "window": 50000,
    },
    {
        "id": "stable_injury_old", "query": "我右膝有什么需要长期注意的限制？",
        "turns": [("我右膝半月板损伤，长期避免深蹲。", "已记录。")], "ages": [450],
        "gt": "回答右膝半月板损伤并需长期避免深蹲；不得因年龄丢弃。",
        "window": 8000,
    },
    {
        "id": "short_term_preference", "query": "我今天什么时候训练？",
        "turns": [("我通常晚上训练。", "已记录。"), ("这周我改成早上训练。", "收到。")],
        "ages": [400, 2],
        "gt": "当前应采用这周早上训练；不能把旧的通常晚上训练当成当前安排。",
        "window": 8000,
    },
    {
        "id": "current_substitution", "query": "我的目标体重是多少？",
        "turns": [("我的目标体重是70公斤。", "已记录。"), ("我把目标体重改成65公斤。", "收到。")],
        "ages": [300, 3],
        "gt": "当前目标是65公斤；旧70公斤只能作为旧值，不能冒充当前值。",
        "window": 8000,
    },
    {
        "id": "condition_negation", "query": "我今天膝盖有点疼，还能慢跑吗？",
        "turns": [("如果膝盖不疼，可以慢跑；如果疼就停止，不要硬撑。", "已记录。")],
        "ages": [10],
        "gt": "膝盖疼时应停止/不要慢跑；保留条件与否定的完整边界。",
        "window": 7500,
    },
    {
        "id": "numeric_precision", "query": "我上次卧推的重量和组次是多少？",
        "turns": [("我上次卧推60公斤，4组每组8次。", "已记录。")],
        "ages": [20], "gt": "重量60公斤、4组、每组8次；数字不得改写。",
        "window": 50000,
    },
    {
        "id": "large_page_short_fact",
        "query": "本周三训练什么？",
        "turns": [("背景说明" * 400 + "本周三只做两组徒手深蹲。", "收到。")],
        "ages": [1],
        "gt": "只回答本周三两组徒手深蹲；不把无关背景当成训练内容。",
        "window": 8000,
    },
    {
        "id": "deep_shared_dependency", "query": "我之前说周三练什么、在哪里练？",
        "turns": [("我在家训练，只有哑铃。", "已记录。"), ("周三练腿。", "收到。")],
        "ages": [30, 1], "select": [1], "dependencies": {1: [0]},
        "gt": "回答周三练腿，并保留在家、只有哑铃的前置条件；必要依赖不可丢。",
        "window": 8000,
    },
    {
        "id": "conflicting_records", "query": "我要不要吃布洛芬？",
        "turns": [("我吃布洛芬。", "已记录。"), ("我不吃布洛芬。", "收到。")],
        "ages": [5, 5],
        "gt": "指出两条记录冲突、当前无法确定；不能把其中一条说成唯一事实。",
        "window": 8000,
    },
    {
        "id": "explicit_history", "query": "我以前周三练什么？",
        "turns": [("我以前周三练腿。", "已记录。")], "ages": [500],
        "include_history": True,
        "gt": "作为历史回答以前周三练腿；不得把旧安排说成当前安排。",
        "window": 7500,
    },
    {
        "id": "old_short_term_state", "query": "我今天能训练吗？",
        "turns": [("我今天感冒发烧，暂停训练。", "已记录。")], "ages": [60],
        "gt": "不能仅凭60天前的发烧断言今天仍发烧；应提示核对当前症状/状态。",
        "window": 7500,
    },
    {
        "id": "unrelated_background", "query": "我能吃核桃吗？",
        "turns": [("无关背景：" + "天气情况" * 300, "收到。"),
                  ("我对核桃过敏。", "已记录。")],
        "ages": [5, 500], "select": [1],
        "gt": "回答不能吃核桃/避免核桃；不得把天气背景当成个人事实。",
        "window": 8000,
    },
    {
        "id": "required_over_budget", "query": "今天训练什么？",
        "turns": [("必要长文：" + "训练细节" * 300 + "今天只做拉伸。", "收到。")],
        "ages": [1], "required": True, "window": 5000,
        "gt": "明确说明必要原文超出本轮上下文预算，不能编造训练内容。",
    },
]


def input_hash(case):
    payload = {"query": case["query"], "turns": case["turns"]}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def add_turn(cid, message, answer):
    tid, _ = compact.begin_turn(ChatRequest(message=message, conversation_id=cid))
    compact.finish_turn(tid, answer, {"answer": answer})
    return tid


def backdate(turn_id, days):
    with compact.session() as s, s.begin():
        s.get(MemoryTurn, turn_id).completed_at = compact.now() - timedelta(days=days)


async def load(cid, query):
    legacy = SimpleNamespace(initialize=AsyncMock(return_value=False),
                             load=AsyncMock(return_value={}))
    return await compact.CompactMemory(legacy).load("demo-user", cid, query)


def page_id(memory, turn_id):
    return next(entry["page_id"] for entry in memory["wiki_catalog"]["entries"]
                if entry["turn_id"] == turn_id)


class ScriptedLocator:
    """Real client for extraction/answer; scripted locator for known target pages."""

    def __init__(self, real, plan):
        self.real = real
        self.plan = plan
        self.calls = []

    def configured(self):
        return self.real.configured()

    async def complete_structured(self, messages, schema, *, name, **kwargs):
        self.calls.append(name)
        if name == "return_wiki_read_plan":
            value = schema.model_validate(self.plan)
            if kwargs.get("validator"):
                kwargs["validator"](value)
            return value
        if name not in {"return_wiki_semantic_units", "return_dialogue_answer"}:
            raise AssertionError("unexpected model call: " + name)
        return await self.real.complete_structured(
            messages, schema, name=name, **kwargs)


def error_detail(exc):
    import traceback
    detail = {"type": type(exc).__name__}
    detail["frames"] = [{"file": Path(frame.filename).name, "line": frame.lineno,
                         "function": frame.name}
                        for frame in traceback.extract_tb(exc.__traceback__)]
    category = getattr(exc, "category", None)
    if category is not None:
        detail["category"] = getattr(category, "value", str(category))
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        detail["status_code"] = status
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        detail["code"] = code[:160]
    return detail


async def answer_arm(client, memory, query, ids, *, window, arm, quota):
    from app.services.memory import semantic_units as units_module
    state = copy.deepcopy(memory)
    if arm in {"whole_page", "full_fit"}:
        state["wiki_units"] = []
        state["wiki_selected_units"] = []
        with temporary_attr(settings, "wiki_unit_ranking", "density"):
            with temporary_attr(budget_module, "context_budget",
                                ContextBudgetManager(window=window, output=100, margin=0)):
                return await _answer(client, state, query, ids)

    with temporary_attr(budget_module, "context_budget",
                        ContextBudgetManager(window=window, output=100, margin=0)):
        if arm == "time_off":
            with temporary_attr(settings, "wiki_temporal_decay_enabled", False):
                return await _answer(client, state, query, ids)
        if arm == "value_rank":
            with temporary_attr(settings, "wiki_unit_ranking", "value"):
                return await _answer(client, state, query, ids)
        return await _answer(client, state, query, ids)


async def _answer(client, state, query, ids):
    understanding = {
        "dialogue_ids": list(ids), "reply_mode": "conversation",
        "tasks": [], "reply_purpose": "general",
        "safety": {"level": "normal", "evidence": "", "categories": [],
                   "reason": "synthetic quality sample"},
    }
    metrics = RequestMetrics()
    token = active_metrics.set(metrics)
    probe = {}
    original_assemble = ContextBudgetManager.assemble

    def assembled(self, *args, **kwargs):
        try:
            packed = original_assemble(self, *args, **kwargs)
            probe.setdefault("ok", []).append({
                "used": packed.report.get("used_tokens"),
                "budget": packed.report.get("input_budget")})
            return packed
        except ContextBudgetExceeded as exc:
            probe["error"] = {"reason": exc.reason, "used": exc.used,
                              "budget": exc.budget, "detail": exc.detail}
            raise

    ContextBudgetManager.assemble = assembled
    try:
        result = await dialogue_context.conversation_answer(
            client, ChatRequest(message=query, conversation_id="real-quality"),
            state, understanding)
        usage = metrics.summary()
        if result is None:
            result = {"answer": None, "status": "requires_grounding",
                      "model_used": bool(usage.get("model_calls"))}
        return {"result": result, "usage": usage, "probe": probe}
    finally:
        ContextBudgetManager.assemble = original_assemble
        active_metrics.reset(token)


@contextmanager
def temporary_attr(target, name, value):
    old = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, old)


def ledger_snapshot(factory):
    from sqlalchemy import select
    with factory() as session:
        rows = list(session.scalars(select(ModelCallLedger)))
        quotas = list(session.scalars(select(ModelQuota)))
    calls = [{
        "scope_id": row.scope_id, "kind": row.kind, "capability": row.capability,
        "model": row.model, "status": row.status, "reserved_tokens": row.reserved_tokens,
        "usage": {
            key: (row.usage or {}).get(key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    } for row in rows]
    return {
        "calls": len(calls),
        "by_status": {status: sum(item["status"] == status for item in calls)
                      for status in sorted({item["status"] for item in calls})},
        "by_capability": {cap: sum(item["capability"] == cap for item in calls)
                          for cap in sorted({item["capability"] for item in calls})},
        "total_tokens": sum(((item["usage"] or {}).get("total_tokens") or 0)
                            for item in calls),
        "rows": calls,
        "quotas": [{
            "kind": row.kind, "calls": row.calls, "tokens": row.tokens,
            "max_calls": row.max_calls, "max_tokens": row.max_tokens,
        } for row in quotas],
    }


async def run(output: Path, *, only: set[str] | None = None) -> int:
    from tests.quality_isolation import isolated_database
    import pytest

    patches = pytest.MonkeyPatch()
    from app.db import session as db
    payload = {
        "run": "real provider synthetic Wiki semantic-unit quality comparison",
        "provider": {
            "configured": bool(settings.llm_base_url and settings.llm_api_key
                               and settings.llm_model),
            "model": settings.llm_model,
            "base_host": urlsplit(settings.llm_base_url).hostname or "",
        },
        "personal_data": False,
        "samples": [],
        "errors": [],
        "status": "running",
        "limits": [
            "Locator plans are scripted to known target pages; extraction and answers are real.",
            "Synthetic pages only; no real user records, .env contents, secrets or connection strings.",
            "No embedding, rerank, MCP, tracing, background jobs, migration or deployment.",
            "Normal request/stage/deadline/retry and daily ledger enforcement; no daily waiver.",
        ],
    }
    host = urlsplit(settings.llm_base_url).hostname or ""
    original_dns = socket.getaddrinfo

    def dns(name, *args, **kwargs):
        if name in {host, host.encode("utf-8")}:
            return original_dns(name, *args, **kwargs)
        raise RuntimeError("non_provider_dns_blocked")

    original_async_init = None
    original_sync_send = None
    ledger = None
    database = None
    db_path = None
    try:
        patches.setattr(socket, "getaddrinfo", dns)
        import httpx
        original_async_init = httpx.AsyncClient.__init__

        def bounded_init(instance, *args, **kwargs):
            kwargs["trust_env"] = False
            kwargs["follow_redirects"] = False
            return original_async_init(instance, *args, **kwargs)

        patches.setattr(httpx.AsyncClient, "__init__", bounded_init)
        original_sync_send = httpx.Client.send
        patches.setattr(httpx.Client, "send", lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("sync_network_blocked")))

        with TemporaryDirectory() as directory:
            with isolated_database(patches, Path(directory), "sqlite") as isolated_engine:
                # The fixture owns the filename; report the database actually
                # bound to the isolated factory rather than an invented path.
                db_path = Path(isolated_engine.url.database)
                settings.wiki_enabled = True
                settings.wiki_semantic_units_enabled = True
                settings.wiki_source_read_max_tokens = 14000
                settings.wiki_semantic_unit_max_input_tokens = 16000
                settings.wiki_semantic_unit_max_pages = 8
                settings.wiki_unit_ranking = "density"
                settings.wiki_temporal_decay_enabled = True
                settings.wiki_temporal_half_life_days = 30
                settings.wiki_temporal_decay_floor = .85
                settings.memory_read_backend = "compact_v1"
                settings.redis_enabled = False
                settings.memory_background_enabled = False
                settings.memory_summary_enabled = False
                settings.planner_shadow_enabled = False
                settings.react_readonly_enabled = False
                settings.react_write_enabled = False
                settings.retrieval_embedding_api_key = ""
                settings.retrieval_embedding_base_url = ""
                settings.rerank_api_key = ""
                settings.rerank_base_url = ""
                settings.rerank_enabled = False
                settings.mcp_url = ""
                settings.langsmith_api_key = ""
                settings.langsmith_tracing = False
                settings.llm_max_output_tokens = min(settings.llm_max_output_tokens, 2048)
                settings.llm_context_window_tokens = 32768

                factory = db.SessionLocal
                engine = db.engine
                bound = getattr(factory, "kw", {}).get("bind")
                database = {
                    "backend": "sqlite",
                    "temporary_path": str(db_path),
                    "bound_engine_matches": bound is engine,
                    "db_ready": bool(db.DB_READY),
                    "created": db_path.exists(),
                    "removed_after": False,
                }
                quota = BudgetStore(factory)
                token = store_override.set(quota)
                client = LLMClient()
                try:
                    for index, case in enumerate(
                            [item for item in CASES
                             if not only or item["id"] in only], start=1):
                        cid = "real-quality-" + case["id"]
                        record = {
                            "id": case["id"], "query": case["query"],
                            "input_sha256": input_hash(case),
                            "gt": case["gt"], "turns": [],
                            "extraction": {}, "arms": {},
                        }
                        try:
                            tids = [add_turn(cid, message, answer)
                                    for message, answer in case["turns"]]
                            database["created"] = db_path.exists()
                            for tid, age in zip(tids, case["ages"]):
                                if age:
                                    backdate(tid, age)
                            memory = await load(cid, case["query"])
                            pages = [page_id(memory, tid) for tid in tids]
                            selected_indices = case.get("select", list(range(len(tids))))
                            selected = [pages[i] for i in selected_indices]
                            dependencies = {
                                pages[key]: [pages[value] for value in values]
                                for key, values in (case.get("dependencies") or {}).items()
                            }
                            plan = {
                                "page_ids": selected,
                                "required_page_ids": selected if case.get("required") else [],
                                "dependencies": dependencies,
                                "include_history": bool(case.get("include_history")),
                            }
                            record["turns"] = [
                                {"role": "user", "sha256": hashlib.sha256(
                                    message.encode("utf-8")).hexdigest()}
                                for message, _answer in case["turns"]
                            ]
                            record["plan"] = {
                                "selected_pages": len(selected),
                                "dependencies": {key: len(value)
                                                 for key, value in dependencies.items()},
                                "include_history": bool(case.get("include_history")),
                                "required": bool(case.get("required")),
                            }
                            locator = ScriptedLocator(client, plan)
                            as_of = clock.utc_now().isoformat()
                            with model_scope("wiki-quality-" + case["id"] + "-extract",
                                             kind="request", store=quota):
                                await wc.prepare(
                                    locator, case["query"], memory, semantics.PROMPT)
                            memory["wiki_as_of"] = as_of
                            record["extraction"] = {
                                "calls": list(locator.calls),
                                "source_read_tokens": memory.get("wiki_read_tokens"),
                                "source_read_limit": memory.get("wiki_read_limit"),
                                "unit_pages": len({u.get("page_id") for u in memory.get(
                                    "wiki_units") or []}),
                                "units": [{
                                    "claim": str(unit.get("claim") or "")[:240],
                                    "origin": unit.get("origin"),
                                    "completeness": unit.get("semantic_completeness"),
                                    "temporal": {
                                        key: (unit.get("temporal") or {}).get(key)
                                        for key in ("stability", "sensitivity", "source_at")
                                    },
                                } for unit in memory.get("wiki_units") or []],
                                "failures": list(memory.get("wiki_unit_failures") or []),
                            }
                            ids = [tids[i] + ":user" for i in selected_indices]
                            for arm in ("whole_page", "unit", "time_off", "value_rank",
                                        "full_fit"):
                                arm_name = "full_fit" if arm == "full_fit" else arm
                                window = (30000 if arm == "full_fit"
                                          else int(case.get("window") or 1200))
                                try:
                                    with model_scope(
                                            f"wiki-quality-{case['id']}-{arm_name}",
                                            kind="request", store=quota):
                                        outcome = await answer_arm(
                                            client, memory, case["query"], ids,
                                            window=window, arm=arm_name, quota=quota)
                                    result = outcome["result"]
                                    record["arms"][arm_name] = {
                                        "window": window,
                                        "answer": (result or {}).get("answer"),
                                        "status": (result or {}).get("status"),
                                        "error": (result or {}).get("error"),
                                        "model_used": (result or {}).get("model_used"),
                                        "usage": outcome["usage"],
                                        "context_probe": outcome.get("probe"),
                                    }
                                except Exception as exc:
                                    record["arms"][arm_name] = {
                                        "window": window, "error": error_detail(exc)}
                                    payload["errors"].append({
                                        "sample": case["id"], "arm": arm_name,
                                        "error": error_detail(exc)})
                                    if isinstance(exc, LLMGatewayError):
                                        raise
                            ledger = ledger_snapshot(factory)
                            record["ledger_after_sample"] = {
                                "calls": ledger["calls"],
                                "total_tokens": ledger["total_tokens"],
                                "by_status": ledger["by_status"],
                            }
                            payload["samples"].append(record)
                        except Exception as exc:
                            payload["errors"].append({
                                "sample": case["id"], "stage": "sample",
                                "error": error_detail(exc)})
                            record["error"] = error_detail(exc)
                            payload["samples"].append(record)
                            if isinstance(exc, LLMGatewayError):
                                raise
                    unavailable = any(
                        arm.get("status") == "unavailable"
                        for sample in payload["samples"]
                        for arm in sample.get("arms", {}).values())
                    payload["status"] = ("completed_with_failures"
                                         if unavailable or payload["errors"] else "completed")
                except Exception as exc:
                    payload["status"] = "blocked"
                    payload["stop_error"] = error_detail(exc)
                finally:
                    store_override.reset(token)
                    if ledger is None:
                        try:
                            ledger = ledger_snapshot(factory)
                        except Exception:
                            ledger = None
                    if database is not None:
                        database["removed_after"] = not db_path.exists()
                    payload["database"] = database
                    payload["ledger"] = ledger or {}
                    payload["budget"] = {
                        "daily_enforced": bool(quota.enforce_daily_limits),
                        "request_max_calls": settings.model_request_max_calls,
                        "request_max_tokens": settings.model_request_max_tokens,
                        "request_deadline_seconds": settings.model_request_deadline_seconds,
                        "stage_max_calls": settings.model_stage_max_calls,
                        "daily_max_calls": settings.model_daily_max_calls,
                        "daily_max_tokens": settings.model_daily_max_tokens,
                    }
    except Exception as exc:
        payload["status"] = "blocked"
        payload["stop_error"] = error_detail(exc)
    finally:
        if original_async_init is not None:
            import httpx
            httpx.AsyncClient.__init__ = original_async_init
        if original_sync_send is not None:
            import httpx
            httpx.Client.send = original_sync_send
        socket.getaddrinfo = original_dns
        patches.undo()

    if database is not None and db_path is not None:
        database["removed_after"] = not Path(db_path).exists()
        payload["database"] = database

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "samples": len(payload["samples"]),
        "errors": len(payload["errors"]),
        "ledger_calls": (payload.get("ledger") or {}).get("calls"),
        "ledger_tokens": (payload.get("ledger") or {}).get("total_tokens"),
        "database_removed": (payload.get("database") or {}).get("removed_after"),
    }))
    return 0 if payload["status"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--only", default="", help="comma-separated smoke subset")
    options = parser.parse_args()
    only = {item.strip() for item in options.only.split(",") if item.strip()} or None
    return asyncio.run(run(Path(options.output), only=only))


if __name__ == "__main__":
    raise SystemExit(main())
