"""C2-3 phaseP-R1 support: real ASGI JSON/SSE, isolated SQLite/PostgreSQL, real
C1 ledger / two closed bridges / durable budget / MemoryTurn / privacy SQL as
the future execution oracle.

P3-01..06 corrections:
- The API owner is the normalised single person ``demo-user``; every HTTP
  business/ledger/budget difference is scoped to that owner and the real
  request_id (never a synthetic per-request user).
- ``require_c3_runtime`` only classifies the three actual target modules as
  missing_module (or a missing public interface as missing_interface); any
  other import/initialisation error stays unexpected.
- The provider MockTransport lives strictly on the LLM provider boundary and
  never clashes with the ASGITransport used by the HTTP harness.
- Fault helpers hold the caller engine, target an explicit SQL/commit window
  armed after the real chain exists, record hits and remove the listener.
- C1 helpers build real sources/digests/FK/nested leases and read the real
  hashed quota identity instead of拼接 raw scope strings.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import event, text

# --------------------------------------------------------------------- design

C3_MODULE_PATHS = {
    "runtime": "app.services.agent.react_write_runtime",
    "adapters": "app.services.agent.react_write_adapters",
    "checkpoint": "app.services.agent.react_checkpoint_store",
}
C3_INTERFACES = {
    "runtime": ("run_write", "write_lifecycle_scope", "finalize_write_response"),
    "adapters": ("write_catalog", "WriteAdapters"),
    "checkpoint": ("ReactCheckpointStore",),
}
RUNTIME_ORIGIN = {"key": "_react_runtime_origin", "protocol": 1,
                  "generation": "6bcac6d1-6791-4d5c-9003-ea4691cd2d30"}
RUNTIME_PREFIX = "c2-3-runtime:v1:"
CONTROLLER_TASK_ID = "react-controller-v1"
WRITE_MODES = ("legacy", "react_readonly", "react_write")
CONFIRMATION_SOURCES = ("确认保存", "确认记录")
CONFIRMATION_FIELDS = ("pending_request_id", "pending_op_id", "pending_digest")
OP_ID_RE = re.compile(r"^op2_[0-9a-f]{32}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
LOGICAL_LIMITS = {"supervisor_turns": 4, "tool_steps": 16, "no_progress": 2,
                  "supervisor_physical_attempts": 4}
PENDING_TTL_SECONDS = 900
API_OWNER = "demo-user"
FUTURE_BUSINESS = ("backend/app/services/agent/react_write_runtime.py",
                   "backend/app/services/agent/react_write_adapters.py",
                   "backend/app/services/agent/react_checkpoint_store.py")

# ----------------------------------------------------------------- gate

class C3MissingModule(RuntimeError):
    code = "missing_module"


class C3MissingInterface(RuntimeError):
    code = "missing_interface"


class C3Runtime:
    def __init__(self, modules):
        self.modules = modules

    @property
    def runtime(self):
        return self.modules["runtime"]

    @property
    def adapters(self):
        return self.modules["adapters"]

    @property
    def checkpoint(self):
        return self.modules["checkpoint"]


def require_c3_runtime() -> C3Runtime:
    """Import the future runtime modules and verify their public interfaces.

    Only a ModuleNotFoundError whose ``name`` is one of the three actual C3
    target modules maps to missing_module; a missing public attribute maps to
    missing_interface. Anything else (NameError, dependency import errors,
    initialisation/SQL/transport errors) propagates as a real failure and is
    classified unexpected by the runner.
    """
    modules = {}
    for name, path in C3_MODULE_PATHS.items():
        try:
            modules[name] = importlib.import_module(path)
        except ModuleNotFoundError as exc:
            if exc.name == path:
                raise C3MissingModule(f"missing_module:{path}") from exc
            raise
        except Exception:
            raise
    for name, attrs in C3_INTERFACES.items():
        for attr in attrs:
            if not hasattr(modules[name], attr):
                raise C3MissingInterface(f"missing_interface:{C3_MODULE_PATHS[name]}.{attr}")
            if not callable(getattr(modules[name], attr)):
                raise C3MissingInterface(f"missing_interface:{C3_MODULE_PATHS[name]}.{attr}")
    return C3Runtime(modules)


# ------------------------------------------------------------------ evidence

def emit(case, **values):
    print("C3_EVIDENCE=" + json.dumps({"case": case, **values}, ensure_ascii=False, allow_nan=False,
                                      default=str))


def check_all(case, checks):
    emit(case, checks=checks)
    non_bool = [key for key, value in checks.items() if not isinstance(value, bool)]
    assert not non_bool, f"{case}: non-bool check values: {non_bool}"
    assert all(checks.values()), f"{case}: {[k for k, v in checks.items() if not v]}"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def parse_iso(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def as_utc(value):
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def dec(value):
    return value if isinstance(value, Decimal) else Decimal(str(value))


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def ids_of(records):
    return {item["id"] for item in records}


def delta(records, before_ids):
    return [item for item in records if item["id"] not in before_ids]


def key_delta(rows, before_keys, key="id"):
    return [row for row in rows if row[key] not in before_keys]


# --------------------------------------------------------------- real rows

def _rows(engine, sql, params):
    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(text(sql), params)]


def workout_rows(engine, owner=API_OWNER):
    rows = _rows(engine, "SELECT id, user_id, workout_date, exercise, weight_kg, sets, reps, "
                         "set_details, idempotency_key FROM workout_records WHERE user_id=:u ORDER BY id",
                 {"u": owner})
    for row in rows:
        row["weight_kg"] = float(row["weight_kg"])
        row["workout_date"] = str(row["workout_date"])[:10]
    return rows


def nutrition_rows(engine, owner=API_OWNER):
    rows = _rows(engine, "SELECT id, user_id, meal_type, description, calories, protein_g, carbs_g, "
                         "fat_g, log_date FROM nutrition_logs WHERE user_id=:u ORDER BY id", {"u": owner})
    for row in rows:
        row["log_date"] = str(row["log_date"])[:10]
        for field in ("calories", "protein_g", "carbs_g", "fat_g"):
            row[field] = None if row[field] is None else float(row[field])
    return rows


def food_rows(engine, owner=API_OWNER):
    rows = _rows(engine, "SELECT id, user_id, idempotency_key, raw_food_name, food_name, weight_g, "
                         "meal_group_id, total_kcal, is_estimated, match_method FROM user_food_log "
                         "WHERE user_id=:u ORDER BY id", {"u": owner})
    for row in rows:
        row["weight_g"] = float(row["weight_g"])
        row["total_kcal"] = float(row["total_kcal"])
        row["is_estimated"] = as_bool(row["is_estimated"])
    return rows


def all_business_rows(engine, owner=API_OWNER):
    return workout_rows(engine, owner) + nutrition_rows(engine, owner) + food_rows(engine, owner)


def business_keys(engine, owner=API_OWNER):
    """Per-table id sets keyed by table so identical ids in different tables never cancel."""
    return {"workout_records": ids_of(workout_rows(engine, owner)),
            "nutrition_logs": ids_of(nutrition_rows(engine, owner)),
            "user_food_log": ids_of(food_rows(engine, owner))}


def ledger_snapshot(engine, request_id, owner=API_OWNER):
    result = {"request": None, "tasks": [], "operations": [], "events": []}
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT user_id, request_id, conversation_id, source, source_digest, status, revision, "
            "checkpoint, time_base, requested_mode, effective_mode, budget_scope_id, response, "
            "lease_worker_id, lease_token, lease_epoch, lease_deadline "
            "FROM agent_execution_requests WHERE user_id=:u AND request_id=:r"),
            {"u": owner, "r": request_id}).mappings().first()
        result["request"] = dict(row) if row else None
        for table, key, order in (
                ("agent_execution_tasks", "tasks", "ORDER BY task_id"),
                ("agent_execution_operations", "operations", "ORDER BY op_id"),
                ("agent_execution_events", "events", "ORDER BY sequence, event_id")):
            result[key] = _rows(
                engine, f"SELECT * FROM {table} WHERE user_id=:u AND request_id=:r {order}",
                {"u": owner, "r": request_id})
    return result


def runtime_event_keys(engine, request_id, owner=API_OWNER):
    return [item["event_key"] for item in ledger_snapshot(engine, request_id, owner)["events"]
            if str(item["event_key"]).startswith(RUNTIME_PREFIX)]


def budget_snapshot(engine, *, owner=API_OWNER, request_id=None):
    """Real quota/ledger/cache rows. For a request, the request-scope ledger rows
    carry the actual ``quota_ids`` (request + daily + each stage); every related
    parent/child quota row is read through that real relation instead of keeping
    only the request-scope quota. Cache presence is a boolean, never content."""
    with engine.connect() as conn:
        quotas = [dict(row) for row in conn.execute(text(
            "SELECT quota_id, kind, calls, tokens, max_calls, max_tokens, expires_at "
            "FROM model_quotas ORDER BY quota_id")).mappings().all()]
        calls = [dict(row) for row in conn.execute(text(
            "SELECT call_id, scope_id, kind, capability, model, stage_id, quota_ids, status, "
            "reserved_tokens FROM model_call_ledger ORDER BY started_at")).mappings().all()]
        caches = [dict(row) for row in conn.execute(text(
            "SELECT cache_id, scope_id, capability, owner_id FROM model_result_cache")).mappings().all()]
    def _quota_id_list(value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return []
        return [str(item) for item in value] if isinstance(value, (list, tuple)) else []

    for row in calls:
        row["quota_ids"] = _quota_id_list(row.get("quota_ids"))
    if request_id is not None:
        scope_id = real_scope_id(request_id)
        calls = [row for row in calls if row["scope_id"] == scope_id]
        related = {quota_id for row in calls for quota_id in row["quota_ids"]}
        quotas = [row for row in quotas if row["quota_id"] in related]
        caches = [row for row in caches if row["scope_id"] == scope_id]
    return {"quotas": quotas, "calls": calls, "caches": caches,
            "cache_present": bool(caches)}


def real_scope_id(request_id: str, kind: str = "request") -> str:
    """The real BudgetStore hash rule for the app's request scope name
    (``model_scope('request:' + request_id)`` -> sha256(kind:name))."""
    return "scope:" + hashlib.sha256((kind + ":request:" + request_id).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------- C1 helper

class C1RecipeHarness:
    """A real C1 request/task/op/lease recipe using actual dataclass fields,
    snapshot nested lease/revision and the database time_base."""

    def __init__(self, engine, *, source, request_id, task_type="workout_record", kind="workout",
                 user_id=API_OWNER, conversation_id=None, position=0, ttl=300, parameters=None):
        from sqlalchemy.orm import sessionmaker
        from app.services.agent import execution_ledger_store as ledger_api
        from tests.react_phase_c1_support import sync

        self.ledger = ledger_api
        self.engine = engine
        self.factory = sessionmaker(bind=engine, expire_on_commit=False)
        self.store = ledger_api.ExecutionLedgerStore(self.factory)
        self.sync = sync
        self.source = source
        self.task_type = task_type
        self.kind = kind
        self.position = position
        self.task_id = f"task-{position}"
        self.user_id = user_id
        self.request_id = request_id
        self.conversation_id = conversation_id or f"c3-conv-{uuid4().hex[:10]}"
        self.source_digest = sha256_text(source)
        if parameters is None:
            raise ValueError("C1RecipeHarness requires the full canonical record/batch parameters")
        with self.factory() as session:
            self.time_base = as_utc(self.store.ledger_statement_now(session)).isoformat()
        self.op_parameters = deepcopy(parameters)
        self.identity = ledger_api.RequestIdentity(
            user_id=self.user_id, request_id=self.request_id, conversation_id=self.conversation_id,
            source=source, source_digest=self.source_digest, requested_mode="react_write",
            effective_mode="react_write", schema_version=2, tool_version="react-tools-v2",
            time_base=self.time_base, budget_scope_id="scope:synthetic-c2-3", request_kind="tasks")
        self.snapshot = sync(self.store.ensure_request(self.identity))
        self.snapshot = sync(self.store.acquire_lease(self.user_id, self.request_id, "c3-worker", ttl))
        frozen_parameters = ({"records": [deepcopy(self.op_parameters)]} if kind == "workout"
                             else {"batches": [deepcopy(self.op_parameters)]})
        self.task = ledger_api.TaskSpec(
            task_id=self.task_id, source_start=0, source_end=len(source),
            source_digest=self.source_digest, task_type=task_type, dependencies=[],
            parameters=frozen_parameters, required=True)
        self.snapshot = sync(self.store.freeze_task(self.user_id, self.request_id, self.task,
                                                    expected_revision=self.snapshot["revision"],
                                                    lease=self.lease_proof()))
        self.snapshot = sync(self.store.reserve_operation(
            self.user_id, self.request_id, task_id=self.task_id, position=position, kind=kind,
            parameters=deepcopy(self.op_parameters), source_digest=self.source_digest,
            time_base=self.time_base, expected_revision=self.snapshot["revision"],
            lease=self.lease_proof()))
        self.snapshot = sync(self.store.mark_dispatch_intent(
            self.user_id, self.request_id, self.op_id(), expected_revision=self.snapshot["revision"],
            lease=self.lease_proof()))

    def op_id(self):
        material = ["ledger-v2", self.user_id, self.request_id, self.task_id, self.position, self.kind]
        return "op2_" + hashlib.sha256(canonical(material)).hexdigest()[:32]

    def lease_proof(self):
        lease = self.snapshot["lease"]
        return self.ledger.LeaseProof(worker_id=lease["worker_id"], token=lease["token"],
                                      epoch=lease["epoch"])

    def reload(self):
        self.snapshot = self.sync(self.store.load_request(self.user_id, self.request_id))
        return self.snapshot


def c1_execution_binding_from_env():
    return os.environ.get("FITZEN_C3_EXECUTION_BINDING")


# --------------------------------------------------------------- HTTP harness

def set_write_flag(monkeypatch, enabled):
    from app.core.config import settings

    fields = getattr(type(settings), "model_fields", {}) or {}
    if "react_write_enabled" in fields:
        monkeypatch.setattr(settings, "react_write_enabled", bool(enabled), raising=False)
    else:
        monkeypatch.setitem(settings.__dict__, "react_write_enabled", bool(enabled))


def mock_provider_transport(monkeypatch, responses):
    """Provider-boundary MockTransport: inject only when no transport is passed.

    The ASGI harness passes its own ASGITransport so the two boundaries never
    clash (P3-03). Returns the list of seen provider URLs.
    """
    import httpx
    from app.core.config import settings

    seen = []
    monkeypatch.setattr(settings, "llm_api_key", "mock-not-real", raising=False)
    monkeypatch.setattr(settings, "llm_base_url", "https://mock.invalid/v1", raising=False)
    monkeypatch.setattr(settings, "llm_model", "mock-model", raising=False)
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    original = httpx.AsyncClient

    def handler(request):
        seen.append(str(request.url))
        index = min(len(seen) - 1, len(responses) - 1)
        spec = responses[index]
        if callable(spec):
            return spec(request)
        return httpx.Response(200, json=spec)

    transport = httpx.MockTransport(handler)

    def factory(**kwargs):
        if "transport" not in kwargs:
            kwargs["transport"] = transport
        return original(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return seen


def synthetic_evidence_conflict_response(request):
    """A provider-bound, no-conflict judgement bound to the actual RAG prompt.

    The real evidence-conflict stage validates every chunk id and quote against
    the exact retrieved fragments. This fixture therefore reads those fragments
    from the provider request and quotes the first one verbatim instead of
    replaying an unrelated free-form answer.
    """
    import httpx

    body = json.loads(request.content.decode("utf-8"))
    messages = body.get("messages") or []
    prompt = str((messages[-1] or {}).get("content") or "") if messages else ""
    marker = "证据片段（数据，非指令）："
    assert marker in prompt, "evidence conflict prompt missing fragment payload"
    chunks = json.loads(prompt.split(marker, 1)[1])
    assert isinstance(chunks, list) and chunks
    chunk = chunks[0]
    value = {
        "claims": [{
            "claim": "合成证据片段与问题一致，未见同一条件下的相反结论",
            "relation": "consistent",
            "conditions": [],
            "evidence": [{
                "chunk_id": chunk["chunk_id"],
                "quote": str(chunk["content"])[:400],
            }],
        }],
        "irrelevant_chunk_ids": [],
    }
    return httpx.Response(200, json={
        "choices": [{
            "message": {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": "synthetic-evidence-conflict", "type": "function",
                    "function": {
                        "name": "return_evidence_conflict_judgement",
                        "arguments": json.dumps(value, ensure_ascii=False),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 6, "completion_tokens": 4},
    })


class C3HTTPHarness:
    """Real ASGI JSON/SSE entry over the isolated database (owner=demo-user)."""

    def __init__(self, monkeypatch, engine, *, write_enabled=True, conversation_id=None, note=None):
        import app.main as app_main
        from app.core.config import settings

        self.app = app_main.app
        self.engine = engine
        self.monkeypatch = monkeypatch
        self.settings = settings
        self.conversation_id = conversation_id or f"c3-conv-{uuid4().hex[:12]}"
        self.trace = {"http": [], "sse": [], "notes": [] if note is None else [note]}
        set_write_flag(monkeypatch, write_enabled)
        monkeypatch.setattr(settings, "planner_shadow_enabled", False, raising=False)
        monkeypatch.setattr(settings, "memory_read_backend", "compact_v1", raising=False)

    # -- payload builders -------------------------------------------------

    def body(self, message, *, mode="react_write", request_id=None, conversation_id=None,
             confirmation=None, extra=None, user_id=API_OWNER):
        payload = {"message": message, "user_id": user_id,
                   "conversation_id": conversation_id or self.conversation_id,
                   "request_id": request_id or str(uuid4())}
        if mode is not None:
            payload["mode"] = mode
        if confirmation is not None:
            payload["confirmation"] = deepcopy(confirmation)
        if extra:
            payload.update(deepcopy(extra))
        return payload

    # -- real transport ---------------------------------------------------

    async def _request(self, path, payload, *, stream=False):
        from httpx import ASGITransport, AsyncClient
        from evals.v2.safety import deny_network

        events = []
        body = ""
        with deny_network() as audit:
            async with AsyncClient(transport=ASGITransport(app=self.app),
                                   base_url="http://api.test") as client:
                if stream:
                    async with client.stream("POST", path, json=payload) as response:
                        status = response.status_code
                        if status == 200:
                            async for line in response.aiter_lines():
                                body += line + "\n"
                                if line.startswith("data:"):
                                    try:
                                        events.append(json.loads(line[5:].strip()))
                                    except json.JSONDecodeError:
                                        pass
                        else:
                            body = (await response.aread()).decode("utf-8", errors="replace")
                else:
                    response = await client.post(path, json=payload)
                    status = response.status_code
                    body = response.text
        self.trace["http" if not stream else "sse"].append(
            {"path": path, "status": status, "request_id": payload.get("request_id"),
             "cancelled": False,
             "audit": {"blocked_connections": audit["blocked_connections"],
                       "blocked_dns": audit["blocked_dns"]}})
        assert audit["blocked_connections"] == audit["blocked_dns"] == 0, audit
        return status, body, events

    def post(self, payload, path="/api/chat"):
        status, body, _ = asyncio.run(self._request(path, payload, stream=False))
        return status, self._decode(body)

    def post_json(self, message, **kwargs):
        return self.post(self.body(message, **kwargs))

    def post_raw(self, payload):
        return self.post(payload)

    def stream(self, message, **kwargs):
        """Normal SSE stream: returns (status:int, events:list, body:str)."""
        status, body, events = asyncio.run(self._request("/api/chat/stream",
                                                         self.body(message, **kwargs), stream=True))
        return status, events, body

    def stream_cancel(self, message, *, cancel_after_chunks=1, disconnect_when=None,
                      on_disconnect=None, **kwargs):
        """Real cancel path through the direct ASGI barrier; the trace carries
        the actual event/task state (never a parameter-derived flag)."""
        result = asyncio.run(asgi_stream_cancel(self.app, path="/api/chat/stream",
                                                payload=self.body(message, **kwargs),
                                                cancel_after_chunks=cancel_after_chunks,
                                                disconnect_when=disconnect_when,
                                                on_disconnect=on_disconnect))
        self.trace["sse"].append({"path": "/api/chat/stream", "status": result["status"],
                                  "cancelled": result["disconnect_sent"] is True,
                                  "cancel_state": {key: result[key] for key in
                                                   ("chunks_before_disconnect", "chunks_after_disconnect",
                                                    "task_done", "error", "sent_types")}})
        return result

    @staticmethod
    def _decode(body):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body}

    @staticmethod
    def metrics_of(payload):
        return (payload or {}).get("metrics") or {}

    @staticmethod
    def pending_targets(payload):
        return list((C3HTTPHarness.metrics_of(payload)).get("pending_confirmations") or [])


def _compiled_update_columns(context, table_name):
    """(where_values, set_columns) for a real Core/ORM UPDATE on the table: the
    WHERE column->value mapping plus the statement's own SET column names. SET
    columns come from the explicit ``_values`` mapping or from compiled parameter
    keys that are plain column names of the table (WHERE binds are qualified,
    e.g. ``memory_turns_turn_id``/``request_id_1``); text() SQL falls back to its
    SET clause. Dialect-independent, pure compilation introspection - no extra
    connection, so listeners never recurse."""
    where_values = _compiled_where_values(context, table_name)
    compiled = getattr(context, "compiled", None)
    statement_obj = getattr(compiled, "statement", None) if compiled is not None else None
    param_sets = [item for item in (getattr(context, "compiled_parameters", None) or [])
                  if isinstance(item, dict)]
    set_columns = []
    values = getattr(statement_obj, "_values", None) if statement_obj is not None else None
    if isinstance(values, dict):
        for column in values:
            name = getattr(column, "name", None)
            if isinstance(name, str) and name:
                set_columns.append(name)
    if not set_columns:
        table = getattr(statement_obj, "table", None) if statement_obj is not None else None
        column_names = {column.name for column in table.columns} if table is not None else set()
        if not column_names:
            if table_name == "memory_turns":
                from app.db.memory_models import MemoryTurn
                column_names = {column.name for column in MemoryTurn.__table__.columns}
            elif table_name == "agent_execution_requests":
                from app.db.execution_ledger_models import ExecutionRequestModel
                column_names = {column.name for column in ExecutionRequestModel.__table__.columns}
            elif table_name == "user_profiles":
                from app.db.models import UserProfileModel
                column_names = {column.name for column in UserProfileModel.__table__.columns}
        for item in param_sets:
            for key in item:
                if key in column_names and key not in set_columns:
                    set_columns.append(key)
    if not set_columns and statement_obj is not None and hasattr(statement_obj, "text"):
        text_sql = statement_obj.text
        lowered = text_sql.lower()
        set_index = lowered.find(" set ")
        where_index = lowered.find(" where ")
        if set_index >= 0:
            set_sql = text_sql[set_index + 5:where_index if where_index >= 0 else None]
            set_columns = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", set_sql)
    return where_values, set_columns


# ----------------------------------------------------------------- faults

def _compiled_where_values(context, table_name):
    """Real compiled WHERE column->value mapping for Core/ORM statements on the
    given table: walk ``_where_criteria`` and resolve each column comparison's
    BindParameter through ``compiled.bind_names`` into ``compiled_parameters``.
    Dialect-independent; returns {} when the shape is unknown. Shared by
    cas_window attribution and C3SQLFault matchers."""
    from sqlalchemy.sql import visitors as _visitors
    from sqlalchemy.sql.elements import (BinaryExpression as _Binary,
                                         BindParameter as _Bind)
    from sqlalchemy.sql.schema import Column as _Column

    compiled = getattr(context, "compiled", None)
    statement_obj = getattr(compiled, "statement", None) if compiled is not None else None
    param_sets = [item for item in (getattr(context, "compiled_parameters", None) or [])
                  if isinstance(item, dict)]
    bind_names = getattr(compiled, "bind_names", None) or {}
    values = {}
    for criterion in list(getattr(statement_obj, "_where_criteria", ()) or ()):
        for expression in _visitors.iterate(criterion):
            if not isinstance(expression, _Binary):
                continue
            left = expression.left
            if not isinstance(left, _Column):
                continue
            if getattr(getattr(left, "table", None), "name", None) != table_name:
                continue
            right = expression.right
            if not isinstance(right, _Bind):
                continue
            name = bind_names.get(right)
            if name is None:
                continue
            for item in param_sets:
                if name in item:
                    values[left.name] = item[name]
                    break
    return values


class C3SQLFault:
    """Real SQL-window fault that holds the caller engine, counts hits, records
    the target table/action and removes the listener on exit. Arm it after the
    real C1/bridge chain exists so the chosen occurrence is the intended one.
    """

    def __init__(self, engine, *, table, action="insert", occurrence=1, armed=True, match=None):
        self.engine = engine
        self.table = table
        self.action = action
        self.occurrence = occurrence
        self.armed = armed
        self.match = match
        self.reached = 0
        self.hits = []
        self._installed = None
        self.raised = False

    def where_values(self, context):
        """The statement's real WHERE column->value mapping for this fault's
        table, from the compiled statement (Core/ORM) or the named binds of
        text() SQL. Never a literal alias guess."""
        values = _compiled_where_values(context, self.table)
        if values:
            return values
        compiled = getattr(context, "compiled", None)
        statement_obj = getattr(compiled, "statement", None) if compiled is not None else None
        if statement_obj is not None and hasattr(statement_obj, "text"):
            text_sql = statement_obj.text
            where_index = text_sql.lower().find(" where ")
            where_sql = text_sql[where_index:] if where_index >= 0 else ""
            param_sets = [item for item in (getattr(context, "compiled_parameters", None) or [])
                          if isinstance(item, dict)]
            for column, bind in re.findall(
                    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*:([A-Za-z_][A-Za-z0-9_]*)", where_sql):
                for item in param_sets:
                    if bind in item:
                        values[column] = item[bind]
                        break
        return values

    def _hook(self, conn, cursor, statement, parameters, context, executemany):
        if not self.armed or self.raised:
            return
        lowered = statement.lower()
        if self.action == "select":
            matched = f"from {self.table}" in lowered or f"join {self.table}" in lowered
        else:
            matched = (f"{self.action} into {self.table}" in lowered
                       or f"{self.action} {self.table}" in lowered
                       or f"{self.action}  {self.table}" in lowered)
        if not matched:
            return
        if self.match is not None:
            try:
                if not self.match(statement, parameters, context):
                    return
            except TypeError:
                if not self.match(statement, parameters):
                    return
        self.reached += 1
        self.hits.append({"n": self.reached, "table": self.table, "action": self.action,
                          "verb": statement.lstrip().split(" ", 1)[0].upper(),
                          "statement": statement.strip()[:200]})
        if self.reached == self.occurrence:
            self.raised = True
            raise RuntimeError(f"c3_sql_fault:{self.table}:{self.action}")

    def install(self):
        event.listen(self.engine, "before_cursor_execute", self._hook)
        self._installed = True
        return self

    def remove(self):
        if self._installed:
            event.remove(self.engine, "before_cursor_execute", self._hook)
            self._installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
        return False

    def report(self):
        return {"table": self.table, "action": self.action, "occurrence": self.occurrence,
                "reached": self.reached, "raised": self.raised, "hits": self.hits}


class cas_window:
    """Real observation window (P3-04): while wrapped, records the rowcount of
    every UPDATE on the C1 request row, optionally scoped to the given real
    request ids and owner in the bound parameters. A committed save must contain
    CAS updates with rowcount exactly 1 (0 for fenced attempts), never a stub."""

    def __init__(self, engine, *, request_ids=None, owner=None):
        self.engine = engine
        self.request_ids = tuple(request_ids or ())
        self.owner = owner
        self.updates = []
        self._hook = None

    def _record(self, conn, cursor, statement, parameters, context, executemany):
        lowered = statement.lower()
        if not lowered.lstrip().startswith("update") or "agent_execution_requests" not in lowered:
            return
        from sqlalchemy.sql import visitors as _visitors
        from sqlalchemy.sql.elements import (BinaryExpression as _Binary,
                                             BindParameter as _Bind,
                                             TextClause as _Text)
        from sqlalchemy.sql.schema import Column as _Column

        compiled = getattr(context, "compiled", None)
        param_sets = [item for item in (getattr(context, "compiled_parameters", None) or [])
                      if isinstance(item, dict)]
        user_value = None
        request_value = None
        statement_obj = getattr(compiled, "statement", None) if compiled is not None else None
        if isinstance(statement_obj, _Text):
            # text() SQL: only the WHERE clause binds may define identity; the
            # SET payload (which can mention another rid) is never inspected.
            text_sql = statement_obj.text
            where_index = text_sql.lower().find(" where ")
            where_sql = text_sql[where_index:] if where_index >= 0 else ""
            user_match = re.search(r"\buser_id\s*=\s*:([A-Za-z_][A-Za-z0-9_]*)",
                                   where_sql, re.IGNORECASE)
            request_match = re.search(r"\brequest_id\s*=\s*:([A-Za-z_][A-Za-z0-9_]*)",
                                      where_sql, re.IGNORECASE)

            def _named_value(name):
                for item in param_sets:
                    if name in item:
                        return item[name]
                if isinstance(parameters, dict) and name in parameters:
                    return parameters[name]
                return None

            if user_match is not None:
                user_value = _named_value(user_match.group(1))
            if request_match is not None:
                request_value = _named_value(request_match.group(1))
        elif statement_obj is not None and hasattr(statement_obj, "_where_criteria"):
            # Core/ORM Update: the same real compiled WHERE mapping used by the
            # C3SQLFault matchers (dialect-independent bind_names resolution).
            values = _compiled_where_values(context, "agent_execution_requests")
            user_value = values.get("user_id")
            request_value = values.get("request_id")
        else:
            return
        if self.owner is not None and (user_value is None or str(user_value) != self.owner):
            return
        hit = None
        if self.request_ids:
            if request_value is None:
                return
            request_text = str(request_value)
            hit = request_text if request_text in self.request_ids else None
            if hit is None:
                return
        self.updates.append({"rowcount": cursor.rowcount, "request_id": hit,
                             "user_id": str(user_value) if user_value is not None else None,
                             "connection": id(conn.connection.driver_connection)})

    def __enter__(self):
        self._hook = self._record
        event.listen(self.engine, "after_cursor_execute", self._hook)
        return self

    def __exit__(self, *exc):
        event.remove(self.engine, "after_cursor_execute", self._hook)
        return False


async def asgi_stream_cancel(app, *, path, payload, cancel_after_chunks=1, timeout=15,
                             disconnect_when=None, on_disconnect=None):
    """Direct ASGI send/receive cancel barrier (P3-03): the client delivers
    ``http.disconnect`` while the application is still producing the stream.

    Two trigger modes:
    - ``disconnect_when`` (preferred): a watcher coroutine delivers the
      disconnect exactly when the real target condition becomes true (an
      entered provider await, an entered post-commit projection, ...), with no
      dependence on send-chunk cadence; ``on_disconnect`` runs at that instant
      to freeze external counters (it must not release application gates).
    - ``cancel_after_chunks`` (legacy mechanism contrast only): disconnect after
      the Nth streamed chunk when ``disconnect_when`` is None.
    The watcher and the app task are always cleaned up (cancel + await) before
    returning; ``target_reached`` reports whether the watcher fired."""
    import asyncio
    import json as _json

    body = _json.dumps(payload, ensure_ascii=False).encode("utf-8")
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
             "http_version": "1.1", "method": "POST", "scheme": "http", "path": path,
             "raw_path": path.encode("utf-8"), "query_string": b"", "root_path": "",
             "headers": [(b"host", b"api.test"), (b"content-type", b"application/json"),
                         (b"content-length", str(len(body)).encode())],
             "client": ("127.0.0.1", 12345), "server": ("api.test", 80)}
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put({"type": "http.request", "body": body, "more_body": False})
    sent = []
    box = {"chunks": 0, "disconnect_sent": False, "disconnect_at_chunk": None,
           "target_reached": False}

    async def receive():
        return await queue.get()

    async def _send_disconnect():
        if box["disconnect_sent"]:
            return
        box["disconnect_sent"] = True
        box["disconnect_at_chunk"] = box["chunks"]
        if on_disconnect is not None:
            on_disconnect()
        await queue.put({"type": "http.disconnect"})

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("more_body"):
            box["chunks"] += 1
            if disconnect_when is None and box["chunks"] >= cancel_after_chunks:
                await _send_disconnect()

    async def watcher():
        if disconnect_when is None:
            return
        while not box["disconnect_sent"]:
            if disconnect_when():
                box["target_reached"] = True
                await _send_disconnect()
                return
            await asyncio.sleep(0.01)

    task = asyncio.create_task(app(scope, receive, send))
    watch = asyncio.create_task(watcher())
    status = None
    error = None
    try:
        await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        task.cancel()
        error = "timeout"
    except asyncio.CancelledError:
        error = "cancelled"
    except Exception as exc:  # the app may surface the disconnect as a real error
        error = type(exc).__name__
    finally:
        watch.cancel()
        try:
            await watch
        except asyncio.CancelledError:
            pass
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    for message in sent:
        if message["type"] == "http.response.start":
            status = message.get("status")
    return {"status": status, "chunks_before_disconnect": box["disconnect_at_chunk"],
            "chunks_after_disconnect": max(0, box["chunks"] - (box["disconnect_at_chunk"] or 0)),
            "disconnect_sent": box["disconnect_sent"], "target_reached": box["target_reached"],
            "task_done": task.done(), "error": error,
            "sent_types": [m["type"] for m in sent][:8]}


def session_fault_factory(monkeypatch, engine, *, window="after_commit", occurrence=1, predicate=None):
    """Real Session-window fault wired into the app's session factory.

    The Nth matching ``after_commit``/``after_flush``/``before_commit`` event on
    the real ORM Session raises; ``predicate(session)`` can scope the window and
    the returned ``restore`` removes the listener. The caller verifies recovery
    behaviour against the already committed rows (the real commit boundary is
    the window, never a table-name first-write guess).
    """
    from sqlalchemy.orm import Session, sessionmaker
    from app.db import session as db

    state = {"n": 0, "raised": False}

    class FaultSession(Session):
        pass

    def hook(session):
        if state["raised"]:
            return
        if predicate is not None and not predicate(session):
            return
        state["n"] += 1
        if state["n"] == occurrence:
            state["raised"] = True
            raise RuntimeError("c3_session_fault:" + window)

    event.listen(FaultSession, window, hook)
    factory = sessionmaker(bind=engine, class_=FaultSession, expire_on_commit=False)
    monkeypatch.setattr(db, "SessionLocal", factory)

    def restore():
        event.remove(FaultSession, window, hook)

    return {"state": state, "session_class": FaultSession, "factory": factory, "restore": restore}


# ------------------------------------------------------------------- budget

def gateway_probe(monkeypatch, engine, *, scope_name, responses=None, calls=1):
    """Real gateway + real BudgetStore + MockTransport probe (no provider).

    ``scope_name`` is the real request id; the scope name mirrors the app
    (``model_scope('request:' + request_id)``).
    """
    responses = responses or [{"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                               "usage": {"prompt_tokens": 3, "completion_tokens": 2}}]
    seen = mock_provider_transport(monkeypatch, responses)
    from app.services.llm.budget import model_scope, paid_attempt, active_scope
    from app.services.llm.client import LLMClient

    rows = []
    for _ in range(calls):
        with model_scope("request:" + scope_name, kind="request"):
            client = LLMClient()
            result = asyncio.run(client.complete([{"role": "user", "content": "synthetic"}]))
            scope = active_scope.get()
            rows.append({"scope_id": scope.quota_id, "result": result})
    return {"seen": seen, "rows": rows, "budget": budget_snapshot(engine, request_id=scope_name)}


# ------------------------------------------------------------------- memory

def create_memory_turn(engine, conversation_id, request_id, *, hidden=False, deleted=False,
                       epoch=0, owner=API_OWNER):
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO memory_turns (turn_id, request_id, conversation_id, messages, status, "
            "memory_epoch, hidden_at, deleted_at, content_hash, index_status) "
            "VALUES (:turn_id,:request_id,:cid,:messages,'completed',:epoch,:hidden,:deleted,'','indexed')"),
            {"turn_id": str(uuid4()), "request_id": request_id, "cid": conversation_id,
             "messages": json.dumps([{"role": "user", "content": "synthetic"}], ensure_ascii=False),
             "epoch": epoch,
             "hidden": datetime.now(timezone.utc) if hidden else None,
             "deleted": datetime.now(timezone.utc) if deleted else None})


def hide_or_delete_conversation(conversation_id, *, purge=False):
    from app.services.memory import compact
    if purge:
        return compact.delete_memory(conversation_id)
    return compact.hide_conversation(conversation_id)


def existing_memory_turn(engine, conversation_id):
    """Read-only confirmation of the real turn created by the HTTP request; the
    oracle must never fabricate a missing precondition."""
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT turn_id, request_id, status FROM memory_turns WHERE conversation_id=:c"),
            {"c": conversation_id}).mappings().first()
    return dict(row) if row else None


def ensure_memory_turn(engine, conversation_id, request_id, *, owner=API_OWNER):
    """Use the real turn created by the application when present; only create a
    fixture when the real request did not create one (never duplicate)."""
    with engine.connect() as conn:
        existing = conn.execute(text(
            "SELECT turn_id FROM memory_turns WHERE conversation_id=:c"),
            {"c": conversation_id}).first()
    if existing is not None:
        return {"created": False, "turn_id": str(existing[0])}
    create_memory_turn(engine, conversation_id, request_id, owner=owner)
    return {"created": True, "turn_id": None}


def privacy_rows(engine, conversation_id):
    """Safe tombstone view: timestamps/epoch plus emptiness booleans for the
    content columns; never returns messages/response/summary/operations content."""
    with engine.connect() as conn:
        turn = conn.execute(text(
            "SELECT hidden_at, deleted_at, memory_epoch, messages, response, window_summary, "
            "retrieval_summary, operations FROM memory_turns WHERE conversation_id=:c"),
            {"c": conversation_id}).mappings().first()
    if turn is None:
        return None
    row = dict(turn)

    def _empty(value):
        if value is None:
            return True
        if isinstance(value, (list, tuple, dict, set)):
            return len(value) == 0
        return str(value).strip() in ("", "[]", "{}", "null", "None")

    row["messages_empty"] = _empty(row.get("messages"))
    row["response_empty"] = _empty(row.get("response"))
    row["window_summary_empty"] = _empty(row.get("window_summary"))
    row["retrieval_summary_empty"] = _empty(row.get("retrieval_summary"))
    row["operations_empty"] = _empty(row.get("operations"))
    for key in ("messages", "response", "window_summary", "retrieval_summary", "operations"):
        row.pop(key, None)
    return row


# -------------------------------------------------------------------- child

def c3_runner_path() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "react-phase-c2-3-evidence" / "run.py"


def subprocess_c3_call(action, packet, *, timeout=90, binding=None):
    env = dict(os.environ)
    if binding is not None:
        env["FITZEN_C3_EXECUTION_BINDING"] = str(binding)
    return subprocess.run([sys.executable, str(c3_runner_path()), action], input=json.dumps(packet),
                          capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                          cwd=Path(__file__).resolve().parents[1], env=env)


def child_json(result):
    try:
        return json.loads((result.stdout or "").strip())
    except json.JSONDecodeError:
        return {"error": "child_output_invalid", "stdout": (result.stdout or "")[-300:]}


# ------------------------------------------------- old child redirect (C3 plugin)

def redirect_old_children_to_c3(binding=None):
    """Only in this C3 plugin: point the old C1/C2-1/C2-2 child launchers at the
    new C3 runner under the same 155 binding. Old test/helper/runner bytes stay
    untouched; the original outputs (business_counts included) pass through."""
    import tests.react_phase_c1_support as c1_support
    import tests.react_phase_c2_1_support as c2_1_support
    import tests.react_phase_c2_2_support as c2_2_support

    def c1_recovery(url, backend, user_id, request_id):
        result = subprocess_c3_call("child-old-c1-read",
                                    {"url": url, "backend": backend, "user_id": user_id,
                                     "request_id": request_id}, binding=binding)
        assert result.returncode == 0, "C3 old C1-read child failed (see stable child code)"
        return child_json(result)

    def c2_1_bridge(url, backend, user_id, request_id, op_id, expected_revision, lease,
                    tool_call_id, batch_id):
        result = subprocess_c3_call("child-old-c2-1-workout",
                                    {"url": url, "backend": backend, "user_id": user_id,
                                     "request_id": request_id, "op_id": op_id,
                                     "expected_revision": expected_revision,
                                     "lease": {"worker_id": lease.worker_id, "token": lease.token,
                                               "epoch": lease.epoch},
                                     "tool_call_id": tool_call_id, "batch_id": batch_id}, binding=binding)
        assert result.returncode == 0, "C3 old C2-1 workout child failed (see stable child code)"
        return child_json(result)

    def c2_2_nutrition(url, backend, user_id, request_id, op_id, mode, **extra):
        result = subprocess_c3_call("child-old-c2-2-nutrition",
                                    {"url": url, "backend": backend, "user_id": user_id,
                                     "request_id": request_id, "op_id": op_id, "mode": mode, **extra},
                                    binding=binding)
        assert result.returncode == 0, "C3 old C2-2 nutrition child failed (see stable child code)"
        return child_json(result)

    c1_support.subprocess_recovery = c1_recovery
    c2_1_support.subprocess_bridge_call = c2_1_bridge
    c2_2_support.subprocess_nutrition_call = c2_2_nutrition


@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session):
    """Adapt after the old plugins' own sessionstart rebinding. pluggy executes
    hook implementations in LIFO order, so without an explicit priority a later
    registered plugin would run first and be overwritten by the old ones; the
    trylast marker makes this the final installation step."""
    if os.environ.get("FITZEN_C3_REDIRECT_OLD_CHILDREN") == "1":
        redirect_old_children_to_c3(os.environ.get("FITZEN_C3_EXECUTION_BINDING"))


def pytest_runtest_logreport(report):
    """Local report hook: save the real full nodeid and status of every runtest
    report (including subtest reports) for the formal identity gate."""
    path = os.environ.get("FITZEN_C3_IDENTITY_REPORT")
    if not path:
        return
    context = getattr(report, "context", None)
    subtest = bool(getattr(report, "_subtest", False) or context is not None)
    entry = {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
             "subtest": subtest, "context": str(context) if context is not None else None}
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """Local schema-lifecycle observer: records the real CREATE/DROP SCHEMA
    statements executed inside the child's temporary database (schema name and
    action only, never a URL or credential)."""
    path = os.environ.get("FITZEN_C3_SCHEMA_AUDIT")
    if not path:
        return
    import re as _re

    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    def observer(conn, cursor, statement, parameters, context, executemany):
        match = _re.match(r'\s*(CREATE|DROP)\s+SCHEMA\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?'
                          r'"?([A-Za-z0-9_]+)"?', statement, _re.I)
        if match:
            try:
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"action": match.group(1).upper(),
                                             "schema": match.group(2)}) + "\n")
            except OSError:
                pass

    event.listen(Engine, "before_cursor_execute", observer, propagate=True)


# ----------------------------------------------------------------- fixtures

@pytest.fixture
def c3_env(tmp_path, monkeypatch, backend):
    """Isolated SQLite/PostgreSQL database with the real application schema."""
    from tests.react_phase_c1_support import isolated
    from tests.react_phase_c2_2_support import ensure_catalog

    with isolated(monkeypatch, tmp_path / "c3", backend) as engine:
        ensure_catalog(engine)  # failures must surface, never be swallowed
        yield engine
