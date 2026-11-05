"""One new authorized synthetic-user run with real isolated PG/hybrid retrieval.

All prior ledgers stay immutable. The earlier CNY4 limit was explicitly
cancelled by the user, who also cancelled the fixed 100-call ceiling.
This finite scenario suite keeps application budgets and transport checks.
The daily call/token ceiling is waived only inside the freshly created UUID
temporary database unless --enforce-daily-budget is passed explicitly.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "backend/tests")]
import run_plan_knowledge_acceptance as shared
from run_ai_user_simulation import BATCH5_CASES, assess_batch5
from simulation_knowledge import prepare_public_knowledge, require_live_knowledge
from simulation_transport import RequestAudit

OUT = ROOT / "evidence/ai-simulation-repair-b6-20261023/live-01"
BUSINESS_TABLES = ("workout_records", "workout_sessions", "workout_session_exercises", "workout_session_sets",
    "user_food_log", "nutrition_logs", "training_plans", "user_profiles", "user_facts")


def allowed_tables(case):
    if case.get("rule") in {"remember", "replace_memory", "repeat_memory"}:
        return {"user_facts"}
    if case.get("rule") in {"workout", "mixed_write"}:
        return set(BUSINESS_TABLES[:4])
    if case.get("rule") == "food":
        return {"user_food_log"}
    return set()


def canonical(value):
    from datetime import date, datetime
    from decimal import Decimal
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def table_deltas(before, after):
    differences = {}
    for table in BUSINESS_TABLES:
        primary = "fact_id" if table == "user_facts" else "user_id" if table == "user_profiles" else "id"
        old = {r[primary]: r for r in before[table]}
        new = {r[primary]: r for r in after[table]}
        differences[table] = {"added": [r for k, r in new.items() if k not in old],
            "removed": [r for k, r in old.items() if k not in new],
            "changed": [{"before": r, "after": new[k]} for k, r in old.items() if k in new and r != new[k]]}
    return differences


def exact_changes_allowed(case, differences):
    for table, delta in differences.items():
        if any(delta.values()) and table not in allowed_tables(case):
            return False
        if delta["removed"]:
            return False
        for change in delta["changed"]:
            old, new = change["before"], change["after"]
            fields = {k for k in old if old[k] != new[k]}
            if not (table == "user_facts" and case.get("rule") == "replace_memory"
                    and old["fact_id"] in case.get("target_fact_ids", [])
                    and fields <= {"status", "valid_to"} and new["status"] == "superseded"
                    and new["valid_to"] is not None):
                return False
    return True


def cases():
    values = copy.deepcopy(BATCH5_CASES)
    index = next(i for i, c in enumerate(values) if c["id"] == "mixed_recap_plan") + 1
    values.insert(index, dict(id="mixed_prefix_constraints", cid="training", rule="mixed_read",
        message="仅提供建议，不保存。我没有伤病，家里只有哑铃。请先列出刚才卧推和深蹲的重量，再给一份新手增肌周计划：每周2次，每次20分钟。",
        includes=["卧推", "深蹲", "50", "72"], read_first=True))
    return values


def snapshot(factory):
    from sqlalchemy import text
    with factory() as s:
        tables = {}
        for table in BUSINESS_TABLES:
            rows = [{k: canonical(v) for k, v in row.items()} for row in s.execute(text("SELECT * FROM " + table)).mappings()]
            tables[table] = sorted(rows, key=lambda r: json.dumps(r, sort_keys=True))
        workouts = [dict(r) for r in s.execute(text("SELECT id,exercise,weight_kg,sets,reps FROM workout_records ORDER BY id")).mappings()]
        foods = [{**dict(r), "weight_g": float(r["weight_g"]), "record_time": str(r["record_time"])}
                 for r in s.execute(text("SELECT id,food_name,weight_g,meal_type,record_time FROM user_food_log ORDER BY id")).mappings()]
        facts = [dict(r) for r in s.execute(text("SELECT fact_id,field,source_quote,status,source_turn_id,source_message_id FROM user_facts ORDER BY fact_id")).mappings()]
        return {"workouts": workouts, "foods": foods, "facts": facts, "tables": tables,
                "active_facts": sum(r["status"] == "active" for r in facts)}


def assess(case, response, before, after, call_delta, dml_delta):
    checks = assess_batch5(case, response, before, after, call_delta, dml_delta)
    checks["exact_authorized_sql_changes"] = exact_changes_allowed(case, table_deltas(before["tables"], after["tables"]))
    if case["rule"] == "mixed_read":
        outcomes = response.get("metrics", {}).get("task_outcomes", [])
        checks["only_recap_and_plan_goals"] = (len(outcomes) == 2
            and sorted(o.get("task_type", "") for o in outcomes) == ["training_plan", "training_qa"])
        checks["both_goals_complete"] = bool(outcomes) and all(o.get("status") == "complete" for o in outcomes)
        traces = response.get("metrics", {}).get("retrieval", [])
        checks["real_hybrid_without_degradation"] = any(t.get("corpus") == "knowledge"
            and t.get("vector_count", 0) > 0 and t.get("lexical_count", 0) > 0
            and t.get("degraded_reason") is None for t in traces)
        checks["original_evidence_gate_passed"] = any(t.get("quality_gate", {}).get("answerable") is True
            and t["quality_gate"].get("threshold") == .48 for t in traces)
        checks["answer_generated_without_fallback"] = "answer_model_fallback" not in response.get("metrics", {}).get("degraded_reasons", [])
    return checks


async def run_cases(report, factory, limiter, sql):
    import httpx
    from app.api import chat
    from app.main import app
    from app.services.agent.agent import FitnessAgent
    from app.services.rag.async_http import close
    chat.agent = FitnessAgent()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://simulation") as api:
            for original in cases():
                case = copy.deepcopy(original)
                if limiter.state["halted"]:
                    raise shared.AcceptanceStop("run_halted")
                if case.get("confirm_if_unsaved"):
                    prior = next(t for t in report["turns"] if t["id"] == case["confirm_if_unsaved"])
                    if len(prior["after"]["foods"]) > len(prior["before"]["foods"]):
                        report["omitted"].append({"id": case["id"], "reason": "already_committed_no_duplicate_confirmation"})
                        continue
                if case.get("target_from"):
                    target = next(t for t in report["turns"] if t["id"] == case["target_from"])
                    old_ids = {f["fact_id"] for f in target["before"]["facts"]}
                    case["target_fact_ids"] = [f["fact_id"] for f in target["after"]["facts"] if f["fact_id"] not in old_ids]
                if case.get("replay_of"):
                    prior = next(t for t in report["turns"] if t["id"] == case["replay_of"])
                    payload = prior["request"]
                else:
                    payload = {"message": case["message"], "conversation_id": "synthetic-b6-" + case["cid"],
                               "request_id": "sim-b6-" + case["id"], "mode": "legacy"}
                # Identical independent user requests are legitimate. Wire
                # retry detection stays strict within this one application turn.
                limiter.seen.clear()
                sql["case"] = case
                before, calls_before, writes_before, events_before = snapshot(factory), len(limiter.state["calls"]), sql["dml"], len(sql["events"])
                streaming = case.get("stream", False) or (case["rule"] == "replay" and not case.get("json_replay"))
                record = {"id": case["id"], "request": payload, "before": before,
                          "transport": "sse" if streaming else "json"}
                try:
                    wire = await api.post("/api/chat/stream" if streaming else "/api/chat", json=payload)
                    if wire.status_code != 200:
                        raise shared.AcceptanceStop("application_http_error")
                    if streaming:
                        frames = [json.loads(line[6:]) for line in wire.text.splitlines() if line.startswith("data: ")]
                        response = {**frames[-1], "answer": "".join(f.get("content", "") for f in frames)}
                    else:
                        response = wire.json()
                    after = snapshot(factory)
                    checks = assess(case, response, before, after, len(limiter.state["calls"]) - calls_before, sql["dml"] - writes_before)
                    if case.get("meal"):
                        from app.core import clock
                        from datetime import timedelta
                        added = [r for r in after["foods"] if r["id"] not in {v["id"] for v in before["foods"]}]
                        checks["meal_and_date_preserved"] = bool(added) and all(r["meal_type"] == case["meal"]
                            and r["record_time"][:10] == (clock.today() + timedelta(days=case["day_offset"])).isoformat() for r in added)
                    if case.get("replay_of"):
                        checks["same_answer_as_original"] = response.get("answer") == prior["response"].get("answer")
                    record.update(after=after, response=response, checks=checks, passed=all(checks.values()))
                except BaseException as error:
                    record.update(after=snapshot(factory), passed=False, error_type=type(error).__name__)
                    if isinstance(error, shared.AcceptanceStop):
                        record["stop_code"] = str(error)
                record.update(physical_calls=len(limiter.state["calls"]) - calls_before, business_dml=sql["dml"] - writes_before)
                record["sql_delta"] = table_deltas(before["tables"], record["after"]["tables"])
                record["sql_operations"] = sql["events"][events_before:]
                report["turns"].append(record)
                shared.save(OUT / "result.json", report)
                print(json.dumps({"turn": case["id"], "automated_passed": record["passed"],
                                  "physical_calls": record["physical_calls"], "total_physical_calls": len(limiter.state["calls"])}), flush=True)
                if limiter.state["halted"]:
                    raise shared.AcceptanceStop("provider_or_accounting_halted")
        report["execution_complete"] = len(report["turns"]) + len(report["omitted"]) == len(cases())
    finally:
        await close()


def export_fixture(factory):
    """Keep public vectors and synthetic structured replies for later replay."""
    from sqlalchemy import select
    from app.db.models import KnowledgeChunkModel
    from app.db.model_budget_models import ModelResultCache
    with factory() as s, (OUT / "public-vectors.jsonl").open("x", encoding="utf-8") as f:
        count = 0
        for row in s.scalars(select(KnowledgeChunkModel).order_by(KnowledgeChunkModel.id)):
            if row.semantic_embedding is not None:
                f.write(json.dumps({"content_hash": row.content_hash, "embedding_version": row.embedding_version,
                    "vector": [float(v) for v in row.semantic_embedding]}, ensure_ascii=False) + "\n")
                count += 1
        cache = [{"capability": r.capability, "result": r.result} for r in s.scalars(
            select(ModelResultCache).where(ModelResultCache.capability != "embedding").order_by(ModelResultCache.created_at))]
    shared.save(OUT / "model-cache.json", cache)
    return {"public_vectors": count, "synthetic_model_results": len(cache)}


def export_accounting(factory):
    """Read-only, bounded SQL accounting export before the factory is gone.

    This is the real ledger/quota/source projection, not a reconstruction from
    provider usage. Failed calls keep their stored status/error and unknown
    usage; the query never mutates or resets a quota.
    """
    from sqlalchemy import func, select, text
    from app.db.memory_models import UserFact
    from app.db.model_budget_models import ModelCallLedger, ModelQuota
    limit = 2000
    with factory() as session:
        if session.bind is not None and session.bind.dialect.name == "postgresql":
            session.execute(text("SET TRANSACTION READ ONLY"))
        quota_count = session.scalar(select(func.count()).select_from(ModelQuota)) or 0
        ledger_count = session.scalar(select(func.count()).select_from(ModelCallLedger)) or 0
        if quota_count > limit or ledger_count > limit:
            raise RuntimeError("accounting_export_bounds_exceeded")
        quotas = [{key: canonical(value) for key, value in row._mapping.items()} for row in session.execute(
            select(ModelQuota.quota_id, ModelQuota.kind, ModelQuota.calls, ModelQuota.tokens,
                   ModelQuota.max_calls, ModelQuota.max_tokens, ModelQuota.created_at,
                   ModelQuota.expires_at).order_by(ModelQuota.quota_id).limit(limit))]
        ledger = [{key: canonical(value) for key, value in row._mapping.items()} for row in session.execute(
            select(ModelCallLedger.call_id, ModelCallLedger.scope_id, ModelCallLedger.kind,
                   ModelCallLedger.capability, ModelCallLedger.model, ModelCallLedger.stage_id,
                   ModelCallLedger.quota_ids, ModelCallLedger.reserved_tokens,
                   ModelCallLedger.status, ModelCallLedger.usage, ModelCallLedger.error_code,
                   ModelCallLedger.started_at, ModelCallLedger.finished_at)
            .order_by(ModelCallLedger.started_at, ModelCallLedger.call_id).limit(limit))]
        facts = [{key: canonical(value) for key, value in row._mapping.items()} for row in session.execute(
            select(UserFact.fact_id, UserFact.field, UserFact.source_turn_id,
                   UserFact.source_message_id, UserFact.source_quote, UserFact.status,
                   UserFact.valid_from, UserFact.valid_to, UserFact.memory_epoch)
            .order_by(UserFact.valid_from, UserFact.fact_id).limit(limit))]
    payload = {
        "readonly_export": True, "source": "isolated_simulation_sql",
        "bounded_limit": limit, "quota_count": quota_count, "ledger_count": ledger_count,
        "quotas": quotas, "ledger": ledger, "facts": facts,
    }
    shared.save(OUT / "task-quota-audit.json", payload)
    return {"quotas": len(quotas), "ledger": len(ledger), "facts": len(facts)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--allow-unpriced-limited", action="store_true")
    parser.add_argument("--enforce-daily-budget", action="store_true",
                        help="Keep the configured daily call/token limits for this run")
    args = parser.parse_args(argv)
    if not (args.live and args.allow_unpriced_limited):
        parser.error("Explicit limited live run required")
    return args


def budget_policy(settings, *, daily_enforced):
    return {"daily_enforced": bool(daily_enforced),
            "daily_enforcement": "enforced" if daily_enforced else "disabled_isolated_simulation",
            "daily_max_calls": settings.model_daily_max_calls,
            "daily_max_tokens": settings.model_daily_max_tokens,
            # Legacy fields kept for old reports/consumers. They describe the
            # frozen shared-request fallback, not the effective new chat policy.
            "request_policy": "per_task_v1",
            "request_root_enforcement": "aggregate_only",
            "request_root_calls_enforced": False,
            "request_root_tokens_enforced": False,
            "legacy_request_policy": "shared_request",
            "request_max_calls": settings.model_request_max_calls,
            "request_max_tokens": settings.model_request_max_tokens,
            # Uniform per-business-task policy: task_type no longer selects an
            # allowance. The deprecated standard/plan config fields are never
            # read here or by the task bridge.
            "task_policy": "uniform_v2",
            "task_max_calls": settings.model_task_max_calls,
            "task_max_tokens": settings.model_task_max_tokens,
            "prepare_max_calls": settings.model_prepare_max_calls,
            "stage_max_calls": settings.model_stage_max_calls,
            "stage_token_basis": "task_or_prepare.max_tokens; scope.max_tokens for shared scopes",
            "cli_max_calls": settings.model_cli_max_calls,
            "cli_max_tokens": settings.model_cli_max_tokens,
            "background_max_calls": settings.model_background_max_calls,
            "background_daily_max_calls": settings.model_background_daily_max_calls,
            "request_deadline_seconds": settings.model_request_deadline_seconds,
            "cli_deadline_seconds": settings.model_cli_deadline_seconds}


def simulation_budget_store(factory, args):
    from app.services.llm.budget import BudgetStore
    return BudgetStore(factory, enforce_daily_limits=args.enforce_daily_budget)


def finish_database(report, factory, engine, temp, *, fixture_ready):
    """Preserve paid public/synthetic and SQL accounting evidence before cleanup.

    Export runs while the isolated factory is still connected and before
    ``engine.dispose``/database drop. Each export and cleanup has its own
    independent exception path, so an accounting failure cannot skip cleanup
    and a cleanup failure cannot hide a successful export.
    """
    if factory is not None:
        if fixture_ready:
            try:
                report["retained_public_fixture"] = export_fixture(factory)
            except BaseException as error:
                report["export_error_type"] = type(error).__name__
        try:
            report["retained_accounting"] = export_accounting(factory)
        except BaseException as error:
            report["accounting_export_error_type"] = type(error).__name__
    try:
        if engine is not None:
            engine.dispose()
    except BaseException as error:
        report["dispose_error_type"] = type(error).__name__
    finally:
        if temp is not None:
            try:
                temp.drop(); report["temporary_database_removed"] = not temp.created
            except BaseException as error:
                report["cleanup_error_type"] = type(error).__name__
                report["temporary_database_removed"] = False


def main():
    args = parse_args()
    logging.disable(logging.CRITICAL)
    import httpx
    import pytest
    from sqlalchemy import event
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings
    from app.db import session as db
    from app.services.llm.budget import store_override, model_scope
    from app.services.rag.ingestion import backfill_missing_vectors
    from quality_isolation import TempPGDatabase, checked_engine
    OUT.mkdir(parents=True, exist_ok=True)
    policy = budget_policy(settings, daily_enforced=args.enforce_daily_budget)
    with (OUT / "started.json").open("x", encoding="utf-8") as f:
        json.dump({"user_requested_new_simulation": True, "old_currency_limit_cancelled": True,
                   "physical_limit": None, "fixed_100_limit_cancelled_by_user": True,
                   "live": bool(args.live), "allow_unpriced_limited": bool(args.allow_unpriced_limited),
                   "daily_budget_enforced": bool(args.enforce_daily_budget), "budget_policy": policy,
                   "old_ledgers_immutable": True}, f)
    # Preserve historical guards/ledgers. Use the configured normal output
    # allowance and no fixed suite-wide count; request/stage quotas still apply.
    patch, temp, engine, limiter = pytest.MonkeyPatch(), None, None, None
    factory, fixture_ready = None, False
    report = {"turns": [], "omitted": [], "execution_complete": False, "currency_cost": "unknown",
              "planned": len(cases()), "daily_budget_enforced": bool(args.enforce_daily_budget),
              "budget_policy": policy}
    try:
        limiter = RequestAudit(shared.endpoint_map(settings), OUT / "request-ledger.json",
                               max_output_tokens=settings.llm_max_output_tokens)
        sync_init, async_init = httpx.Client.__init__, httpx.AsyncClient.__init__
        sync_send, async_send = httpx.Client.send, httpx.AsyncClient.send
        def initializer(original, transport):
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
                response.read(); limiter.finish(record, response)
                retain_response(record, response)
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
                await response.aread(); limiter.finish(record, response)
                retain_response(record, response)
                return response
            except BaseException as error:
                record["error_type"] = type(error).__name__
                limiter.stop("provider_async_failure")
        def retain_response(record, response):
            if record["capability"] == "chat":
                # This runner sends only public knowledge and its own synthetic
                # conversations. Retain failed structured output too, without
                # provider URLs, headers, credentials or reasoning content.
                choices = response.json().get("choices", [])
                messages = [{k: c.get("message", {}).get(k) for k in ("content", "tool_calls")}
                            for c in choices]
                with (OUT / "synthetic-model-responses.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"call": record["number"], "messages": messages}, ensure_ascii=False) + "\n")
        patch.setattr(httpx.Client, "__init__", initializer(sync_init, httpx.HTTPTransport))
        patch.setattr(httpx.AsyncClient, "__init__", initializer(async_init, httpx.AsyncHTTPTransport))
        patch.setattr(httpx.Client, "send", sync_dispatch)
        patch.setattr(httpx.AsyncClient, "send", async_dispatch)
        for name in ("redis_enabled", "memory_background_enabled", "memory_summary_enabled", "langsmith_tracing", "planner_shadow_enabled", "react_readonly_enabled"):
            patch.setattr(settings, name, False)
        for name in ("mcp_url", "langsmith_api_key"):
            patch.setattr(settings, name, "")
        patch.setattr(settings, "react_write_enabled", True)
        patch.setattr(settings, "memory_read_backend", "compact_v1")
        patch.setattr(settings, "model_http_max_attempts", 1)
        os.environ["FITZEN_PG_DATABASE_AUDIT"] = str(OUT / "pg-lifecycle.jsonl")
        temp = TempPGDatabase(); temp.create()
        report["temporary_database"] = temp.name
        engine = checked_engine(temp.url)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        patch.setattr(db, "engine", engine); patch.setattr(db, "SessionLocal", factory)
        patch.setattr(settings, "database_url", temp.url)
        if not db.init_db():
            raise shared.AcceptanceStop("isolated_database_init_failed")
        patch.setattr(db, "DB_READY", True)
        fixture_ready = True
        from app.db.models import UserProfileModel
        with factory() as session, session.begin():
            session.add(UserProfileModel(user_id="demo-user"))
        report["knowledge_fixture"] = prepare_public_knowledge(factory, expected_database=temp.name)
        quota = simulation_budget_store(factory, args); token = store_override.set(quota)
        report["daily_budget_enforced"] = quota.enforce_daily_limits
        try:
            with model_scope("hybrid-b6-public-vectors", kind="cli", store=quota):
                report["vector_backfill"] = backfill_missing_vectors(factory=factory).as_dict()
            report["preflight"] = require_live_knowledge(factory, expected_database=temp.name)
            shared.save(OUT / "result.json", report)
            sql = {"dml": 0, "case": {}, "events": []}
            def target(statement):
                match = re.match(r'^\s*(insert\s+into|update|delete\s+from|truncate(?:\s+table)?)\s+"?([a-z_][a-z0-9_]*)', statement.lower())
                return (match[1].split()[0], match[2]) if match else (None, None)
            def audit(conn, cursor, statement, parameters, context, executemany):
                operation, table = target(statement)
                if table in BUSINESS_TABLES:
                    # Memory mutations have their own exact fact/source checks;
                    # the reused assessor's DML counter means training/food.
                    if table != "user_facts":
                        sql["dml"] += 1
                    if table not in allowed_tables(sql["case"]):
                        limiter.stop("unexpected_business_write")
            def observed(conn, cursor, statement, parameters, context, executemany):
                operation, table = target(statement)
                if table in BUSINESS_TABLES:
                    sql["events"].append({"table": table, "operation": operation, "rowcount": cursor.rowcount})
            event.listen(engine, "before_cursor_execute", audit)
            event.listen(engine, "after_cursor_execute", observed)
            asyncio.run(run_cases(report, factory, limiter, sql))
        finally:
            store_override.reset(token)
    except BaseException as error:
        report["error_type"] = type(error).__name__
        if isinstance(error, shared.AcceptanceStop):
            report["stop_code"] = str(error)
        import traceback
        report["safe_frames"] = [{"file": Path(f.filename).name, "line": f.lineno} for f in traceback.extract_tb(error.__traceback__)]
        if limiter:
            limiter.state["halted"] = True; shared.save(limiter.output, limiter.state)
    finally:
        finish_database(report, factory, engine, temp, fixture_ready=fixture_ready)
        patch.undo()
        report["physical_calls"] = len(limiter.state["calls"]) if limiter else 0
        report["automated_passed"] = sum(bool(t.get("passed")) for t in report["turns"])
        report["automated_failed"] = len(report["turns"]) - report["automated_passed"]
        shared.save(OUT / "result.json", report)
    print(json.dumps({k: report.get(k) for k in ("execution_complete", "automated_passed", "automated_failed", "physical_calls", "temporary_database_removed", "error_type", "stop_code")}), flush=True)
    return 0 if (report["execution_complete"] and report.get("temporary_database_removed")
                 and not report.get("error_type") and not report.get("export_error_type")
                 and not report.get("accounting_export_error_type")
                 and not report.get("cleanup_error_type")
                 and not report.get("dispose_error_type")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
