"""C1 contract/fixtures only. No implementation of the execution ledger.

The API below is an explicit new design interface, not an alias for old code.
Helpers use real production engine defaults and temporary quality_isolation DBs.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import os
import re
import socket
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import event, inspect as sa_inspect, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.engine.interfaces import Dialect

TABLES = ("agent_execution_requests", "agent_execution_tasks", "agent_execution_operations", "agent_execution_events")
MODEL_NAMES = ("ExecutionRequestModel", "ExecutionTaskModel", "ExecutionOperationModel", "ExecutionEventModel")
NEW_MODULES = ("app.db.execution_ledger_models", "app.db.execution_ledger_schema", "app.services.agent.execution_ledger_store")
IDENTITY = {
    "user_id": "c1-owner", "request_id": "c1-request", "conversation_id": "c1-conversation",
    "source": "synthetic ledger input", "source_digest": hashlib.sha256(b"synthetic ledger input").hexdigest(),
    "requested_mode": "react_readonly", "effective_mode": "react_readonly", "schema_version": 2,
    "tool_version": "react-tools-v2", "time_base": "2026-10-05T00:00:00+00:00",
    "budget_scope_id": "scope:synthetic-c1", "request_kind": "tasks",
}
INTERFACE = {
    "version": 2, "store_module": NEW_MODULES[2], "store_class": "ExecutionLedgerStore",
    "constructor": "ExecutionLedgerStore(factory: sqlalchemy.orm.sessionmaker)",
    "calling_convention": "All methods are synchronous def; factory returns real Session. No asyncio.run, SQL callback or domain handler parameter.",
    "dataclasses_in_store_module": {
        "RequestIdentity": {k: ("int" if k == "schema_version" else "str") for k in IDENTITY},
        "LeaseProof": {"worker_id": "str", "token": "str", "epoch": "int"},
        "TaskSpec": {"task_id": "str", "source_start": "int", "source_end": "int", "source_digest": "str",
                     "task_type": "str", "dependencies": "list[str]", "parameters": "dict[str,finite_JSON]", "required": "bool"},
        "Checkpoint": {"supervisor_turns": "int", "tool_steps": "int", "event_sequence": "int", "budget_scope_id": "str"},
        "EventInput": {"event_key": "str", "tool_call_id": "str", "batch_id": "str", "task_id": "str", "op_id": "str|None",
                       "phase": "str", "observation": "dict[str,finite_JSON]", "task_status": "str",
                       "operation_state": "str|None", "confirmed": "bool"},
    },
    "methods": {
        "ensure_request": "ensure_request(identity: RequestIdentity) -> RequestSnapshot",
        "load_request": "load_request(user_id: str, request_id: str) -> RequestSnapshot|None",
        "acquire_lease": "acquire_lease(user_id: str, request_id: str, worker_id: str, ttl_seconds: int) -> RequestSnapshot",
        "renew_lease": "renew_lease(user_id, request_id, *, expected_revision: int, lease: LeaseProof, ttl_seconds: int) -> RequestSnapshot",
        "release_lease": "release_lease(user_id, request_id, *, expected_revision: int, lease: LeaseProof) -> RequestSnapshot",
        "freeze_task": "freeze_task(user_id, request_id, task: TaskSpec, *, expected_revision: int, lease: LeaseProof) -> RequestSnapshot",
        "reserve_operation": "reserve_operation(user_id, request_id, *, task_id: str, position: int, kind: str, parameters: dict, source_digest: str, time_base: str, expected_revision: int, lease: LeaseProof) -> RequestSnapshot",
        "mark_dispatch_intent": "mark_dispatch_intent(user_id, request_id, op_id: str, *, expected_revision: int, lease: LeaseProof) -> RequestSnapshot",
        "append_observation": "append_observation(user_id, request_id, event: EventInput, *, checkpoint: Checkpoint, expected_revision: int, lease: LeaseProof) -> RequestSnapshot",
        "save_checkpoint": "save_checkpoint(user_id, request_id, checkpoint: Checkpoint, *, expected_revision: int, lease: LeaseProof) -> RequestSnapshot",
        "finish_request": "finish_request(user_id, request_id, *, status: str, response: dict, termination_reason: str, expected_revision: int, lease: LeaseProof) -> RequestSnapshot",
        "load_recovery_state": "load_recovery_state(user_id: str, request_id: str) -> RecoverySnapshot|None",
    },
    "errors": {"class": "ExecutionLedgerError", "attributes": {"code": "str", "reason": "safe str|None"},
               "codes": ["input_invalid", "identity_conflict", "revision_conflict", "lease_busy", "lease_lost", "ledger_unavailable", "state_corrupt", "commit_unknown", "completion_blocked", "terminal_frozen"]},
    "request_snapshot": {"identity": "RequestIdentity as detached dict", "status": "pending|running|limited|failed|completed", "revision": "nonnegative strict int",
                         "lease": "None or {worker_id,token,epoch,deadline:UTC ISO str}", "checkpoint": "Checkpoint dict",
                         "tasks": "detached list of frozen TaskSpec dicts + status/current_result", "operations": "detached list described below",
                         "events": "detached sequence-ordered list described below", "response": "None or finite JSON dict"},
    "operation_fields": ["op_id", "idempotency_key", "task_id", "position", "kind", "parameters", "parameters_digest", "source_digest", "time_base", "status", "confirmation_result"],
    "event_fields": ["event_id", "sequence", "event_key", "tool_call_id", "batch_id", "task_id", "op_id", "phase", "observation", "checkpoint", "task_status", "operation_state", "confirmed"],
    "recovery_extra_fields": {"pending_verification": "list of unchanged dispatch_intent/commit_unknown operation snapshots", "replay_allowed": False},
    "semantics": {
        "ttl_seconds": {"minimum": 1, "maximum": 300, "type": "strict int; bool/string/float rejected", "clock": "database time"},
        "cas": "Every successful changing command increments revision by one. New ensure starts0; successful acquire increments revision and epoch. Input/identity/lease/revision checks precede state changes. Same-input replay is zero-mutation and may return current snapshot despite old revision.",
        "precedence": "input_invalid before SQL; inaccessible owner identity_conflict; stale/expired proof lease_lost before revision_conflict; valid proof with stale revision revision_conflict.",
        "lease": "Live lease acquisition, even same worker, returns lease_busy; renew/release require proof and revision. New acquisition creates new token, epoch+1; renewal preserves token/epoch. Release clears lease; completed snapshots remain readable.",
        "finite_json": "Reject NaN/Infinity/set/non-string keys/arbitrary objects at all depths; no default=str. Integer counters/positions/versions reject bool/string/1.0.",
        "source": "Request source_digest is SHA256 UTF8 source; TaskSpec digest is SHA256 source slice at strict start/end. Immutable identity/task/op parameters/time_base conflict uses identity_conflict.",
        "checkpoint": "Initial counters0 and identity budget_scope_id; monotonic, immutable scope. New append allocates next sequence and checkpoint.event_sequence must equal it; save_checkpoint.event_sequence equals actual latest sequence.",
        "event_replay": "Stable caller event_key identifies append; exact EventInput+checkpoint replay returns current snapshot without revision/sequence/current-result rewind; changed payload identity_conflict.",
        "operations": "reserved -> dispatch_intent -> committed/commit_unknown; rejected only before intent; committed needs confirmed=True, completed observation, commit_state=committed and nonempty finite receipt_refs. Intent/unknown recovery never replays writes; only a new confirmed event may resolve unknown.",
        "completion": "completed blocked by any required unfinished task or reserved/dispatch_intent/commit_unknown op (completion_blocked). An explicit rejected required task cannot be called complete. No-task complete requires request_kind=readonly_no_tasks and termination_reason=readonly_no_tasks; never default empty-set success. Exact repeated finish keeps original response/IDs, differing response identity_conflict; all other terminal mutations terminal_frozen.",
        "failures": "Failure before commit ledger_unavailable with rollback; exception after commit attempted commit_unknown, never auto-retry or new IDs. Corrupt persisted critical shape state_corrupt, not empty state.",
    },
    "derived_identity": {
        "canonical": "json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')",
        "op_material": "['ledger-v2',user_id,request_id,task_id,position,kind]", "op_id": "'op2_'+SHA256(op_material)[:32]",
        "idempotency_key": "'ledger-v2:'+SHA256(op_material), length74 <= existing repo limit150",
        "event_id": "'ev2:'+SHA256(['ledger-v2-event',user_id,request_id,event_key])",
        "sequence": "DB-allocated contiguous per(owner,request), starts1; same call ID across batches distinct by event_key/sequence",
    },
    "schema_module": NEW_MODULES[1], "schema_method": "ensure_execution_ledger_schema(connection: sqlalchemy.engine.Connection) -> None; state_corrupt for incompatible new tables, no commit/drop/rebuild inside function",
    "models_module": NEW_MODULES[0], "models": dict(zip(TABLES, MODEL_NAMES)),
    "schema_minimum": {
        TABLES[0]: {"columns": [*IDENTITY, "status", "revision", "lease_worker_id", "lease_token", "lease_epoch", "lease_deadline", "checkpoint", "response"], "unique": [["user_id", "request_id"]], "foreign_keys": []},
        TABLES[1]: {"columns": ["user_id", "request_id", "task_id", "source_start", "source_end", "source_digest", "task_type", "dependencies", "parameters", "required", "status", "current_result"], "unique": [["user_id", "request_id", "task_id"]], "foreign_keys": [{"columns": ["user_id", "request_id"], "table": TABLES[0], "referred": ["user_id", "request_id"]}]},
        TABLES[2]: {"columns": ["user_id", "request_id", *[f for f in ["op_id", "idempotency_key", "task_id", "position", "kind", "parameters", "parameters_digest", "source_digest", "time_base", "status", "confirmation_result"]]], "unique": [["user_id", "request_id", "op_id"], ["user_id", "request_id", "task_id", "position"], ["idempotency_key"]], "foreign_keys": [{"columns": ["user_id", "request_id", "task_id"], "table": TABLES[1], "referred": ["user_id", "request_id", "task_id"]}]},
        TABLES[3]: {"columns": ["user_id", "request_id", "event_id", "sequence", "event_key", "tool_call_id", "batch_id", "task_id", "op_id", "phase", "observation", "checkpoint", "task_status", "operation_state", "confirmed"], "unique": [["user_id", "request_id", "event_id"], ["user_id", "request_id", "sequence"], ["user_id", "request_id", "event_key"]], "foreign_keys": [{"columns": ["user_id", "request_id", "task_id"], "table": TABLES[1], "referred": ["user_id", "request_id", "task_id"]}, {"columns": ["user_id", "request_id", "op_id"], "table": TABLES[2], "referred": ["user_id", "request_id", "op_id"]}]},
    },
}


def emit(case, **values):
    print("C1_EVIDENCE="+json.dumps({"case": case, **values}, ensure_ascii=False, allow_nan=False))


def check_all(case, checks):
    emit(case, checks=checks)
    assert all(checks.values()), f"{case}: {[k for k,v in checks.items() if not v]}"


def modules():
    return tuple(importlib.import_module(name) for name in NEW_MODULES)


def sync(value):
    if inspect.isawaitable(value):
        if inspect.iscoroutine(value):
            value.close()
        raise AssertionError("C1 API must be synchronous")
    return value


def outcome(function):
    try:
        return {"ok": True, "value": sync(function()), "code": None}
    except Exception as exc:
        error = modules()[2].ExecutionLedgerError
        if not isinstance(exc, error):
            # Preserve an unexpected failure without echoing arbitrary SQL or
            # URL text; remaining finite negative controls can still execute.
            return {"ok":False,"value":None,"code":"unexpected_exception","exception_type":type(exc).__name__}
        return {"ok": False, "value": None, "code": exc.code}


def race(engine, function):
    """Each worker holds its own actual physical connection across a barrier."""
    barrier = threading.Barrier(2)
    physical = []
    def worker(number):
        with engine.connect() as connection:
            physical.append(connection.connection.driver_connection)
            factory = sessionmaker(bind=connection, expire_on_commit=False)
            return function(number, factory, connection, barrier)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, i) for i in range(2)]
        results = [f.result(timeout=15) for f in futures]
    assert len(physical) == 2 and physical[0] is not physical[1]
    return results


class SQLAudit:
    def __init__(self, engine):
        self.statements = []
        self.engine = engine
        event.listen(engine, "before_cursor_execute", self.record)
    def record(self, conn, cursor, statement, parameters, context, executemany):
        self.statements.append(statement)
    def clear(self):
        self.statements.clear()
    @property
    def writes(self):
        return [s for s in self.statements if s.lstrip().split(" ", 1)[0].upper() in {"INSERT", "UPDATE", "DELETE", "REPLACE"}]
    def close(self):
        event.remove(self.engine, "before_cursor_execute", self.record)


@contextmanager
def fail_after_insert(engine, table=TABLES[3]):
    evidence = {"after_insert_reached": False}
    def fail(conn, cursor, statement, parameters, context, executemany):
        normalized = statement.lower().replace('"', '').strip()
        if normalized.startswith("insert into "+table.lower()):
            evidence["after_insert_reached"] = True
            raise RuntimeError("synthetic_after_insert_failure")
    event.listen(engine, "after_cursor_execute", fail)
    try:
        yield evidence
    finally:
        event.remove(engine, "after_cursor_execute", fail)


@contextmanager
def fail_sql(engine, prefix):
    evidence = {"reached": False}
    def fail(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith(prefix.upper()):
            evidence["reached"] = True
            raise RuntimeError("synthetic_sql_failure")
    event.listen(engine, "before_cursor_execute", fail)
    try:
        yield evidence
    finally:
        event.remove(engine, "before_cursor_execute", fail)


class UnknownCommitSession(Session):
    """Actual Session, with a post-DB-commit fault under either transaction API."""


@event.listens_for(UnknownCommitSession,"after_commit")
def _unknown_commit(session):
    raise RuntimeError("synthetic_after_real_commit")


def rows(engine, names=TABLES):
    # Actual rows, no coercion of API payloads. SQLAlchemy handles typed DB values.
    from sqlalchemy import MetaData, Table
    result = {}
    with engine.connect() as conn:
        for name in names:
            table = Table(name, MetaData(), autoload_with=conn)
            values = [dict(row) for row in conn.execute(select(table)).mappings()]
            result[name] = sorted(values, key=lambda item: repr(sorted(item.items())))
    return result


def schema_checks(engine):
    inspector, checks = sa_inspect(engine), {}
    for name, expected in INTERFACE["schema_minimum"].items():
        checks[name+"_columns"] = set(expected["columns"]) <= {c["name"] for c in inspector.get_columns(name)}
        unique = {tuple(inspector.get_pk_constraint(name)["constrained_columns"])}
        unique.update(tuple(u["column_names"]) for u in inspector.get_unique_constraints(name))
        unique.update(tuple(u["column_names"]) for u in inspector.get_indexes(name) if u.get("unique"))
        checks[name+"_unique"] = all(tuple(u) in unique for u in expected["unique"])
        fks = inspector.get_foreign_keys(name)
        checks[name+"_FK"] = all(any(f["constrained_columns"] == wanted["columns"] and f["referred_table"] == wanted["table"]
                                   and f["referred_columns"] == wanted["referred"] for f in fks) for wanted in expected["foreign_keys"])
        checks[name+"_no_old_FK"] = all(f["referred_table"] in TABLES for f in fks)
    return checks


_pg_bootstrap = ContextVar("c1_pg_bootstrap", default=None)
_pg_admin = ContextVar("c1_pg_admin", default=False)
_allowed_pg_databases = set()
_active_audit = None


@contextmanager
def network_fence():
    """Preserve old socket denial; allow only explicit local PG setup endpoint.

libpq DB connections are separately counted/identity-checked on checkout by the
existing checked_engine. No generic application/network exemption exists.
"""
    from evals.v2.safety import deny_network
    original = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo)
    with deny_network() as denied:
        denied_connect, denied_ex, denied_create, denied_dns = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo)
        audit = {"denied": denied, "allowed_pg_socket_connections": 0, "allowed_pg_dns": 0, "verified_db_checkouts": 0}
        def database_guard(dialect, record, cargs, cparams):
            if dialect.name != "postgresql":
                return
            database = cparams.get("dbname") or cparams.get("database")
            # PostgreSQL SQLAlchemy URL connections supply dbname as a parameter.
            # Refuse unverifiable raw DSNs rather than guessing or logging them.
            if database == "postgres" and _pg_admin.get():
                return
            if database not in _allowed_pg_databases or not re.fullmatch(r"fitzen_test_[0-9a-f]{32}", database or ""):
                raise RuntimeError("c1_unverified_database_connect")
        event.listen(Dialect, "do_connect", database_guard)
        def permitted(address):
            endpoint = _pg_bootstrap.get()
            return endpoint is not None and isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1", "localhost"} and address[1] == endpoint[1]
        def connect(sock, address):
            if permitted(address):
                audit["allowed_pg_socket_connections"] += 1
                return original[0](sock, address)
            return denied_connect(sock, address)
        def connect_ex(sock, address):
            if permitted(address):
                audit["allowed_pg_socket_connections"] += 1
                return original[1](sock, address)
            return denied_ex(sock, address)
        def create(address, *args, **kwargs):
            if permitted(address):
                audit["allowed_pg_socket_connections"] += 1
                return original[2](address, *args, **kwargs)
            return denied_create(address, *args, **kwargs)
        def dns(host, port, *args, **kwargs):
            if _pg_bootstrap.get() is not None and host in {"localhost", "127.0.0.1", "::1"}:
                audit["allowed_pg_dns"] += 1
                return original[3](host, port, *args, **kwargs)
            return denied_dns(host, port, *args, **kwargs)
        try:
            with patch.object(socket.socket, "connect", connect), patch.object(socket.socket, "connect_ex", connect_ex), patch.object(socket, "create_connection", create), patch.object(socket, "getaddrinfo", dns):
                yield audit
        finally:
            event.remove(Dialect, "do_connect", database_guard)


@contextmanager
def isolated(monkeypatch, directory, backend):
    from tests.quality_isolation import isolated_database, TempPGDatabase
    directory.mkdir(parents=True, exist_ok=True)
    token = None
    if backend == "postgres":
        from app.core.config import settings
        from sqlalchemy.engine import make_url
        if os.environ.get("FITZEN_DB_TESTS") != "1":
            pytest.skip("C1 PostgreSQL pending: explicit opt-in required; never counted as pass")
        parsed = make_url(settings.database_url)
        assert parsed.host in {"127.0.0.1", "::1", "localhost"}, "C1 isolation permits local PG test service only"
        token = _pg_bootstrap.set((parsed.host, parsed.port or 5432))
    with monkeypatch.context() as scoped:
        if backend == "postgres":
            original_init, original_create, original_drop = TempPGDatabase.__init__, TempPGDatabase.create, TempPGDatabase.drop
            def admin_call(function, self):
                admin_token = _pg_admin.set(True)
                try:
                    return function(self)
                finally:
                    _pg_admin.reset(admin_token)
            def initialize(self):
                return admin_call(original_init, self)
            def create(self):
                assert re.fullmatch(r"fitzen_test_[0-9a-f]{32}", self.name)
                _allowed_pg_databases.add(self.name)
                return admin_call(original_create, self)
            def drop(self):
                try:
                    return admin_call(original_drop, self)
                finally:
                    _allowed_pg_databases.discard(self.name)
            scoped.setattr(TempPGDatabase, "__init__", initialize)
            scoped.setattr(TempPGDatabase, "create", create)
            scoped.setattr(TempPGDatabase, "drop", drop)
        try:
          with isolated_database(scoped, directory, backend) as engine:
            if backend == "postgres":
                def count(connection, record, proxy):
                    session = _active_audit
                    if session is not None:
                        session["verified_db_checkouts"] += 1
                # Existing checked_engine already checks current_database first.
                event.listen(engine, "checkout", count)
            yield engine
        finally:
            if token is not None:
                _pg_bootstrap.reset(token)


class LedgerHarness:
    def __init__(self, engine):
        self.models, self.schema, self.api = modules()
        self.engine, self.factory = engine, sessionmaker(bind=engine, expire_on_commit=False)
        self.store = self.api.ExecutionLedgerStore(self.factory)
        assert all(not inspect.iscoroutinefunction(getattr(self.store, name)) for name in INTERFACE["methods"])
        self.identity = copy.deepcopy(IDENTITY)
        self.snapshot = sync(self.store.ensure_request(self.api.RequestIdentity(**self.identity)))
        self.snapshot = sync(self.store.acquire_lease(self.identity["user_id"], self.identity["request_id"], "worker-1", 300))
    @property
    def ids(self):
        return self.identity["user_id"], self.identity["request_id"]
    @property
    def proof(self):
        return self.api.LeaseProof(**{k: self.snapshot["lease"][k] for k in ("worker_id", "token", "epoch")})
    def call(self, name, *args, **kwargs):
        kwargs.setdefault("expected_revision", self.snapshot["revision"])
        if "lease" not in kwargs:
            kwargs["lease"] = self.proof
        self.snapshot = sync(getattr(self.store, name)(*self.ids, *args, **kwargs))
        return self.snapshot
    def task(self, task_id="task-1", *, task_type="workout_record", required=True):
        return self.api.TaskSpec(task_id=task_id, source_start=0, source_end=len(self.identity["source"]), source_digest=self.identity["source_digest"],
                                 task_type=task_type, dependencies=[], parameters={"weight_kg": 60, "reps": 8}, required=required)
    def freeze(self, **kwargs):
        return self.call("freeze_task", self.task(**kwargs))
    def reserve(self, position=0, **changes):
        payload = {"task_id": "task-1", "position": position, "kind": "workout", "parameters": {"weight_kg": 60, "reps": 8},
                   "source_digest": self.identity["source_digest"], "time_base": self.identity["time_base"], **changes}
        return self.call("reserve_operation", **payload)
    def checkpoint(self, *, turns=None, steps=None, sequence=None):
        current = self.snapshot["checkpoint"]
        if turns is None:
            turns = max(1,current["supervisor_turns"])
        if steps is None:
            steps = max(1,current["tool_steps"])
        if sequence is None:
            sequence = current["event_sequence"]
        return self.api.Checkpoint(supervisor_turns=turns, tool_steps=steps, event_sequence=sequence, budget_scope_id=self.identity["budget_scope_id"])
    def event(self, key="event-1", *, task_id="task-1", op_id=None, committed=False, unknown=False, **changes):
        observation = {"status": "completed" if committed else "partial", "result": {"synthetic_value": 7},
                       "commit_state": "committed" if committed else "unknown" if unknown else "none", "receipt_refs": ["synthetic-confirmation"] if committed else []}
        payload = dict(event_key=key, tool_call_id="same-call", batch_id="batch-1", task_id=task_id, op_id=op_id,
                       phase="confirmation" if committed else "observation", observation=observation, task_status="completed" if committed else "running",
                       operation_state="committed" if committed else "commit_unknown" if unknown else None, confirmed=committed)
        return self.api.EventInput(**{**payload, **changes})
    def append(self, event_input):
        return self.call("append_observation", event_input, checkpoint=self.checkpoint(sequence=self.snapshot["checkpoint"]["event_sequence"]+1))
    def operation(self, position=0):
        return next(o for o in self.snapshot["operations"] if o["position"] == position)
    def reload(self):
        self.snapshot = sync(self.api.ExecutionLedgerStore(self.factory).load_request(*self.ids))
        return self.snapshot


def subprocess_recovery(url, backend, user_id, request_id):
    runner = Path(__file__).resolve().parents[2]/"docs/react-phase-c1-codex-evidence/run.py"
    packet = {"url":url,"backend":backend,"user_id":user_id,"request_id":request_id}
    result = subprocess.run([sys.executable,str(runner),"child-read"], input=json.dumps(packet),
                            capture_output=True,text=True,encoding="utf-8",timeout=20,
                            cwd=Path(__file__).resolve().parents[1])
    # The child only emits a safe synthetic recovery summary or a stable code;
    # never include stderr/connection-string repr in a parent failure message.
    assert result.returncode==0, "C1 child recovery failed (see stable child code)"
    return json.loads(result.stdout)


def pytest_sessionstart(session):
    global _active_audit
    session._c1_network_context = network_fence()
    session._c1_network_audit = session._c1_network_context.__enter__()
    _active_audit = session._c1_network_audit


def pytest_collection_finish(session):
    path = os.environ.get("FITZEN_C1_COLLECTION_MANIFEST")
    if path:
        Path(path).write_text(json.dumps([item.nodeid for item in session.items], indent=2), encoding="utf-8")


def pytest_sessionfinish(session, exitstatus):
    global _active_audit
    audit = session._c1_network_audit
    session._c1_network_context.__exit__(None, None, None)
    _active_audit = None
    emit("NETWORK", blocked_connections=audit["denied"]["blocked_connections"], blocked_dns=audit["denied"]["blocked_dns"],
         allowed_pg_socket_connections=audit["allowed_pg_socket_connections"], allowed_pg_dns=audit["allowed_pg_dns"], verified_db_checkouts=audit["verified_db_checkouts"])
    if any(audit["denied"].values()):
        session.exitstatus = 1
