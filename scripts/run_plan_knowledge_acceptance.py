"""One explicitly authorized, bounded, isolated PG/full-retrieval acceptance.

The user cancelled the earlier CNY4 limit for this run, allowing limited
unpriced retrieval calls. This does not modify or reuse any earlier ledger.
Only public source text and synthetic conversation fixtures leave the machine.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import date, datetime
import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "backend/tests")]
OUT = ROOT / "evidence/plan-knowledge-test-repair-20261023/live"
MAX_CALLS = 44


class AcceptanceStop(RuntimeError):
    pass


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class RequestLimit:
    """Count before dispatch; allow only exact configured endpoints and models."""
    def __init__(self, endpoints, output):
        self.endpoints, self.output = endpoints, output
        self.state = {"currency_limit_cancelled_by_user": True, "currency_cost": "unknown",
                      "physical_call_limit": MAX_CALLS, "halted": False, "calls": []}
        self.seen = set()
        save(output, self.state)

    def stop(self, code):
        self.state["halted"] = True
        self.state["stop_code"] = code
        save(self.output, self.state)
        raise AcceptanceStop(code)

    def reserve(self, request):
        if self.state["halted"]:
            raise AcceptanceStop("run_halted")
        if len(self.state["calls"]) >= MAX_CALLS:
            self.stop("physical_call_limit")
        identity = str(request.url)
        if request.method != "POST" or identity not in self.endpoints:
            self.stop("unapproved_endpoint")
        capability, model = self.endpoints[identity]
        body = request.content
        if len(body) > 262144:
            self.stop("request_size_limit")
        payload = json.loads(body)
        if payload.get("model") != model:
            self.stop("unapproved_model")
        if capability == "chat" and (payload.get("stream") or type(payload.get("max_tokens")) is not int
                or not 1 <= payload["max_tokens"] <= 2048):
            self.stop("chat_output_limit")
        if capability == "embedding" and (not isinstance(payload.get("input"), list)
                or not 1 <= len(payload["input"]) <= 20 or payload.get("dimensions") != 1024):
            self.stop("embedding_batch_limit")
        signature = hashlib.sha256(identity.encode() + body).hexdigest()
        if signature in self.seen:
            self.stop("physical_retry_blocked")
        self.seen.add(signature)
        record = {"number": len(self.state["calls"]) + 1, "capability": capability,
                  "request_bytes": len(body), "status": "dispatched_cost_unknown"}
        self.state["calls"].append(record)
        save(self.output, self.state)
        return record

    def finish(self, record, response):
        record["http_status"] = response.status_code
        if response.status_code != 200:
            self.stop("provider_http_error")
        data = response.json()
        usage = data.get("usage") if isinstance(data, dict) else None
        safe = {k: v for k, v in (usage or {}).items()
                if "token" in k and type(v) is int and v >= 0} if isinstance(usage, dict) else {}
        if not safe:
            self.stop("provider_usage_missing")
        record.update(status="response_received", usage=safe)
        save(self.output, self.state)
        print(json.dumps({"physical_call": record["number"], "capability": record["capability"],
                          "http_status": response.status_code, "usage": safe}), flush=True)


def endpoint_map(settings):
    from urllib.parse import urlsplit
    from app.services.rag.rerank_protocols import PROTOCOLS
    base, key = settings.rerank_base_url, settings.rerank_api_key
    if not base and not key and settings.rerank_provider == "dashscope" and settings.rerank_legacy_embedding_config:
        base, key = settings.retrieval_embedding_base_url, settings.retrieval_embedding_api_key
    if not all((base, key, settings.retrieval_embedding_base_url, settings.retrieval_embedding_api_key,
                settings.llm_base_url, settings.llm_api_key, settings.rerank_enabled)):
        raise AcceptanceStop("provider_configuration_missing")
    endpoints = {
        settings.llm_base_url.rstrip("/") + "/chat/completions": ("chat", settings.llm_model),
        settings.retrieval_embedding_base_url.rstrip("/") + "/embeddings": ("embedding", settings.retrieval_embedding_model),
        PROTOCOLS[settings.rerank_provider].endpoint(base): ("rerank", settings.rerank_model),
    }
    for endpoint in endpoints:
        p = urlsplit(endpoint)
        if p.scheme != "https" or not p.hostname or p.username or p.password or p.query or p.fragment:
            raise AcceptanceStop("provider_endpoint_invalid")
    if (settings.retrieval_embedding_dimensions != 1024 or settings.llm_model != "deepseek-flash"
            or urlsplit(settings.llm_base_url).hostname != "api.deepseek.com"):
        raise AcceptanceStop("provider_model_configuration_invalid")
    return endpoints


def seed_synthetic_context(factory):
    """Copy exactly two completed, synthetic training turns; never personal DBs."""
    from app.db.memory_models import MemoryTurn
    from app.db.models import WorkoutRecordModel, UserProfileModel
    source = ROOT / "evidence/ai-user-simulation-20261022-b5/retry-01/live/synthetic.sqlite"
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as old:
        old.row_factory = sqlite3.Row
        old.execute("PRAGMA query_only=ON")
        turns = list(old.execute("SELECT * FROM memory_turns WHERE request_id IN (?,?) ORDER BY started_at",
                                ("sim4-b5r1-workout_complete", "sim4-b5r1-workout_shorthand")))
        workouts = list(old.execute("SELECT * FROM workout_records WHERE exercise IN ('卧推','深蹲') ORDER BY id"))
    if len(turns) != 2 or len(workouts) != 2 or {float(r["weight_kg"]) for r in workouts} != {50., 72.}:
        raise AcceptanceStop("synthetic_fixture_mismatch")
    with factory() as session, session.begin():
        if session.get(UserProfileModel, "demo-user") is None:
            session.add(UserProfileModel(user_id="demo-user"))
        for row in workouts:
            values = {c.name: row[c.name] for c in WorkoutRecordModel.__table__.columns if c.name in row.keys()}
            values["workout_date"] = date.fromisoformat(values["workout_date"])
            values["created_at"] = datetime.fromisoformat(values["created_at"])
            values["completed"] = bool(values["completed"])
            session.add(WorkoutRecordModel(**values))
        for row in turns:
            values = {c.name: row[c.name] for c in MemoryTurn.__table__.columns if c.name in row.keys()}
            for name in ("messages", "response", "operations"):
                if values.get(name) is not None:
                    values[name] = json.loads(values[name])
            for name in ("started_at", "completed_at", "deleted_at", "hidden_at"):
                if values.get(name):
                    values[name] = datetime.fromisoformat(values[name])
            values["summary_embedding"] = None
            session.add(MemoryTurn(**values))
    return {"turns": 2, "workout_records": 2, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


async def chat_acceptance(report, factory, limiter):
    import httpx
    from sqlalchemy import text
    from app.api import chat
    from app.main import app
    from app.services.agent.agent import FitnessAgent
    from app.services.rag.async_http import close
    chat.agent = FitnessAgent()
    message = "先回顾刚才卧推和深蹲的重量，再给一个新手每周两次、每次20分钟、只有哑铃的增肌周计划。我没有伤病，只要建议，不保存。"
    payload = {"message": message, "conversation_id": "synthetic-b5r1-training",
               "request_id": "knowledge-full-mixed-20261023", "mode": "legacy"}
    def counts():
        with factory() as session:
            return {table: session.scalar(text("SELECT COUNT(*) FROM " + table))
                    for table in ("workout_records", "user_food_log", "nutrition_logs", "training_plans", "user_facts")}
    before = counts()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://simulation") as api:
            response = await api.post("/api/chat", json=payload)
            if response.status_code != 200:
                raise AcceptanceStop("application_http_error")
            result = response.json()
            report["response"] = {k: result.get(k) for k in ("answer", "intent", "model_used", "clarifications", "citations")}
            traces = result.get("metrics", {}).get("retrieval", [])
            report["retrieval"] = traces
            after = counts()
            answer = result.get("answer", "")
            outcomes = result.get("metrics", {}).get("task_outcomes", [])
            report["checks"] = {
                "recap_preserved": all(word in answer for word in ("卧推", "深蹲", "50", "72")),
                "plan_completed": any(t.get("task_type") == "training_plan" and t.get("status") == "complete" for t in outcomes),
                "no_business_change": before == after,
                "no_clarifications": not result.get("clarifications"),
                "hybrid_trace_complete": any(t.get("corpus") == "knowledge" and t.get("vector_count", 0) > 0
                    and t.get("lexical_count", 0) > 0 and t.get("degraded_reason") is None for t in traces),
                "original_model_evidence_gate_passed": any(t.get("quality_gate", {}).get("mode") == "model_rerank"
                    and t["quality_gate"].get("answerable") is True and t["quality_gate"].get("threshold") == .48 for t in traces),
            }
            report["task_outcomes"] = outcomes
            calls = len(limiter.state["calls"])
            replay = await api.post("/api/chat", json=payload)
            report["checks"]["replay_zero_calls_same_answer"] = (replay.status_code == 200
                and replay.json().get("answer") == answer and len(limiter.state["calls"]) == calls and counts() == after)
            report["passed"] = all(report["checks"].values())
    finally:
        await close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--allow-unpriced-limited", action="store_true")
    args = parser.parse_args()
    if not (args.live and args.allow_unpriced_limited):
        parser.error("Explicit live and user-authorized unpriced-limited flags required")
    logging.disable(logging.CRITICAL)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "started.json").open("x", encoding="utf-8") as stream:
        json.dump({"user_cancelled_original_cny4_limit": True, "physical_call_limit": MAX_CALLS,
                   "synthetic_and_public_only": True, "automatic_retry": False}, stream)
    import httpx
    import pytest
    from sqlalchemy import event
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings
    from app.db import session as db
    from app.services.llm.budget import BudgetStore, store_override, model_scope
    from app.services.rag.ingestion import backfill_missing_vectors
    from quality_isolation import TempPGDatabase, checked_engine
    from simulation_knowledge import prepare_public_knowledge, require_live_knowledge
    patch = pytest.MonkeyPatch()
    temp, engine, limiter = None, None, None
    report = {"paid_execution_complete": False, "passed": False, "currency_cost": "unknown"}
    try:
        endpoints = endpoint_map(settings)
        limiter = RequestLimit(endpoints, OUT / "request-ledger.json")
        sync_init, async_init = httpx.Client.__init__, httpx.AsyncClient.__init__
        sync_send, async_send = httpx.Client.send, httpx.AsyncClient.send
        def bounded_init(original, transport):
            def init(instance, *a, **kw):
                if not isinstance(kw.get("transport"), httpx.ASGITransport):
                    kw["transport"] = transport(retries=0)
                kw.update(trust_env=False, follow_redirects=False)
                kw.pop("proxy", None); kw.pop("mounts", None)
                original(instance, *a, **kw)
            return init
        def sync_dispatch(instance, request, *a, **kw):
            record = limiter.reserve(request)
            try:
                response = sync_send(instance, request, *a, **kw)
                response.read()
                limiter.finish(record, response)
                return response
            except BaseException as error:
                record["error_type"] = type(error).__name__
                limiter.stop("provider_sync_failure")
        async def async_dispatch(instance, request, *a, **kw):
            if isinstance(instance._transport, httpx.ASGITransport):
                return await async_send(instance, request, *a, **kw)
            record = limiter.reserve(request)
            try:
                response = await async_send(instance, request, *a, **kw)
                await response.aread()
                limiter.finish(record, response)
                return response
            except BaseException as error:
                record["error_type"] = type(error).__name__
                limiter.stop("provider_async_failure")
        patch.setattr(httpx.Client, "__init__", bounded_init(sync_init, httpx.HTTPTransport))
        patch.setattr(httpx.AsyncClient, "__init__", bounded_init(async_init, httpx.AsyncHTTPTransport))
        patch.setattr(httpx.Client, "send", sync_dispatch)
        patch.setattr(httpx.AsyncClient, "send", async_dispatch)
        for name in ("redis_enabled", "memory_background_enabled", "memory_summary_enabled", "langsmith_tracing",
                     "planner_shadow_enabled", "react_readonly_enabled", "react_write_enabled"):
            patch.setattr(settings, name, False)
        for name in ("mcp_url", "langsmith_api_key"):
            patch.setattr(settings, name, "")
        patch.setattr(settings, "memory_read_backend", "compact_v1")
        patch.setattr(settings, "model_http_max_attempts", 1)
        patch.setattr(settings, "llm_max_output_tokens", min(settings.llm_max_output_tokens, 2048))
        temp = TempPGDatabase()
        temp.create()
        report["temporary_database"] = temp.name
        engine = checked_engine(temp.url)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        patch.setattr(db, "engine", engine)
        patch.setattr(db, "SessionLocal", factory)
        patch.setattr(settings, "database_url", temp.url)
        if not db.init_db():
            raise AcceptanceStop("isolated_database_init_failed")
        patch.setattr(db, "DB_READY", True)
        report["knowledge_fixture"] = prepare_public_knowledge(factory, expected_database=temp.name)
        save(OUT / "result.json", report)
        quota = BudgetStore(factory)
        token = store_override.set(quota)
        try:
            with model_scope("public-knowledge-acceptance-20261023", kind="cli", store=quota, max_calls=MAX_CALLS):
                report["vector_backfill"] = asdict(backfill_missing_vectors(factory=factory))
                save(OUT / "result.json", report)
                if report["vector_backfill"]["failed"]:
                    raise AcceptanceStop("vector_backfill_failed")
                report["preflight"] = require_live_knowledge(factory, expected_database=temp.name)
                report["synthetic_fixture"] = seed_synthetic_context(factory)
                def audit_sql(conn, cursor, statement, parameters, context, executemany):
                    query = statement.lower().lstrip()
                    if query.startswith(("insert", "update", "delete", "truncate")) and re.search(
                            r"\b(workout_records|workout_sessions|workout_exercises|workout_sets|nutrition_logs|user_food_log|training_plans|user_facts)\b", query):
                        limiter.stop("unexpected_business_write")
                event.listen(engine, "before_cursor_execute", audit_sql)
                asyncio.run(chat_acceptance(report, factory, limiter))
                report["paid_execution_complete"] = True
        finally:
            store_override.reset(token)
    except BaseException as error:
        report["stop_type"] = type(error).__name__
        if isinstance(error, AcceptanceStop):
            report["stop_code"] = str(error)
        import traceback
        report["safe_frames"] = [{"file": Path(f.filename).name, "line": f.lineno}
                                 for f in traceback.extract_tb(error.__traceback__)]
        if limiter:
            limiter.state["halted"] = True
            save(limiter.output, limiter.state)
    finally:
        if engine is not None:
            engine.dispose()
        if temp is not None:
            try:
                temp.drop()
                report["temporary_database_removed"] = not temp.created
            except BaseException as error:
                report["cleanup_error_type"] = type(error).__name__
                report["temporary_database_removed"] = False
        patch.undo()
        report["physical_calls"] = len(limiter.state["calls"]) if limiter else 0
        save(OUT / "result.json", report)
    print(json.dumps({k: report.get(k) for k in ("passed", "paid_execution_complete", "physical_calls", "stop_code", "stop_type", "temporary_database_removed")}), flush=True)
    return 0 if report.get("passed") and report.get("temporary_database_removed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
