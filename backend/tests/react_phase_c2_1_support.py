"""C2-1 workout transaction bridge fixtures only. No bridge implementation.

The helper uses the real Base/create_database_engine/sessionmaker/quality
isolation via the frozen C1 support module and prepares requests/tasks/op with
the real C1 ExecutionLedgerStore API. The bridge import stays lazy so a missing
implementation is a genuine failure, never a skip.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session, sessionmaker

from tests.react_phase_c1_support import (
    fail_after_insert,
    fail_sql,
    isolated,
    network_fence,
    race,
    rows,
    sync,
)

C2_MODULE = "app.services.agent.workout_write_bridge"
BRIDGE_CLASS = "WorkoutWriteBridge"


def emit(case, **values):
    print("C2_EVIDENCE=" + json.dumps({"case": case, **values}, ensure_ascii=False, allow_nan=False))


def check_all(case, checks):
    emit(case, checks=checks)
    assert all(checks.values()), f"{case}: {[k for k, v in checks.items() if not v]}"

SUCCESS_SOURCE = "帮我记录今天卧推60公斤3组8次，谢谢"
SPAN_TEXT = "卧推60公斤3组8次"
SPAN_START = SUCCESS_SOURCE.index(SPAN_TEXT)
SPAN_END = SPAN_START + len(SPAN_TEXT)
FROZEN_DATE = "2026-10-07"
ALT_DATE = "2026-05-05"

BUSINESS_TABLES = (
    "workout_records",
    "workout_sessions",
    "workout_session_exercises",
    "workout_session_sets",
)


def bridge_module():
    return importlib.import_module(C2_MODULE)


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def op_id_for(user_id: str, request_id: str, task_id: str, position: int, kind: str) -> str:
    material = ["ledger-v2", user_id, request_id, task_id, position, kind]
    return "op2_" + hashlib.sha256(canonical(material)).hexdigest()[:32]


def frozen_record(**changes) -> dict:
    record = {
        "exercise": "卧推",
        "sets": 3,
        "reps": 8,
        "weight_kg": 60.0,
        "note": "",
        "difficulty": "未记录",
        "completed": True,
        "date": FROZEN_DATE,
        "set_details": None,
    }
    record.update(changes)
    return record


def bridge_outcome(function):
    """Capture a raised bridge error without echoing arbitrary exception text."""
    try:
        return {"ok": True, "value": function(), "raised": None}
    except Exception as exc:  # pragma: no cover - genuine protocol failure
        return {"ok": False, "value": None, "raised": type(exc).__name__}


NORMAL_SAFETY = {"level": "normal", "evidence": "", "categories": [],
                 "reason": "c2-1 synthetic request source"}
URGENT_SAFETY = {"level": "urgent", "evidence": "胸痛", "categories": ["current_symptom"],
                 "reason": "synthetic request reports current chest pain"}


def default_safety(source: str) -> dict:
    """The scripted assessment for this synthetic request source.

    Only the explicitly synthesized urgent request is urgent; every other
    request carries a normal assessment bound to its own source text. Missing
    or invalid safety is never turned into normal by this default — callers
    that need that negative pass an explicit assessment or bypass this helper.
    """
    if isinstance(source, str) and "胸痛" in source:
        return dict(URGENT_SAFETY)
    return dict(NORMAL_SAFETY)


_GATE_TASKS = {
    # The frozen C2-1-02 synthetic sources script the model's structured
    # decision for exactly that request. Negation/third-party/hypothetical are
    # real record intents with the model's own flags; questions and quoted
    # material are not record actions at all.
    "不要记录卧推60公斤3组8次": {"action": "record", "is_negated": True},
    "怎么记录卧推60公斤3组8次？": {"action": "question"},
    "如果今天卧推60公斤3组8次，帮我记录": {"action": "record", "is_hypothetical": True},
    "朋友昨天卧推60公斤3组8次，帮我记录": {"action": "record", "is_third_party": True},
    "帮我记录深蹲100公斤。卧推60公斤3组8次": {"action": "question"},
    "卧推60公斤3组8次": {"action": "question"},
    "资料：卧推60公斤3组8次": {"action": "question"},
    "帮我记录今天卧推60公斤3组8次，胸痛": {"action": "question"},
}


def scripted_understanding(source: str) -> dict:
    """One source-bound validated semantic result for the synthetic request."""
    script = dict(_GATE_TASKS.get(source) or {"action": "record"})
    text = source if isinstance(source, str) else ""
    return {
        "validated_tasks": [{
            "text": text,
            "source_start": 0,
            "source_end": len(text),
            "task_type": "workout_record",
            "action": script.pop("action"),
            "is_negated": False,
            "is_hypothetical": False,
            "is_third_party": False,
            "goal_role": "result",
            **script,
        }],
        "safety_assessed": True,
    }


def bind_request_semantics(source: str, *, tasks="auto", safety="auto") -> None:
    """Install the scripted safety/understanding for exactly this request source.

    ``"auto"`` uses the synthesized script for that source. A dict is stored
    verbatim. ``None`` removes that dimension instead of inventing a default,
    so missing/wrong-safety negative controls stay genuinely missing/wrong.
    """
    from app.services.agent.semantics import request_understanding

    current = dict(request_understanding.get() or {})
    value = dict(current.get(source) or {})
    if safety == "auto":
        value["safety"] = default_safety(source)
    elif safety is None:
        value.pop("safety", None)
    else:
        value["safety"] = dict(safety)
    if tasks == "auto":
        value.update(scripted_understanding(source))
    elif tasks is None:
        value.pop("validated_tasks", None)
    else:
        value["validated_tasks"] = copy.deepcopy(tasks)
    current[source] = value
    request_understanding.set(current)


@contextmanager
def assessed_safety(*sources, level="normal", evidence="", categories=None, reason=""):
    """Bind a source-bound SafetyAssessment for these synthetic requests.

    The real bridge requires the request-local model SafetyAssessment before a
    write. These tests script the semantic layer, so each harness binds exactly
    the assessment a real request would carry for its own source. A missing or
    malformed assessment is never replaced with a blanket normal default.
    """
    from app.services.agent.semantics import request_understanding

    payload = {"level": level, "evidence": evidence,
               "categories": list(categories or []),
               "reason": reason or "c2-1 synthetic request source"}
    current = dict(request_understanding.get() or {})
    for source in sources:
        if isinstance(source, str) and source:
            current[source] = {"safety": dict(payload), **(current.get(source) or {})}
    token = request_understanding.set(current)
    try:
        yield
    finally:
        request_understanding.reset(token)


def business_rows(engine) -> dict:
    return rows(engine, BUSINESS_TABLES)


def event_by_key(snapshot: dict, event_key: str) -> dict | None:
    if not snapshot:
        return None
    for item in snapshot["events"]:
        if item["event_key"] == event_key:
            return item
    return None


class CommitCountingSession(Session):
    """Real Session that counts commits and optionally faults after commit N."""

    state = {"commits": 0, "fail_at": None}

    @classmethod
    def reset(cls, fail_at=None):
        cls.state = {"commits": 0, "fail_at": fail_at}


@event.listens_for(CommitCountingSession, "after_commit")
def _count_commits(session):
    CommitCountingSession.state["commits"] += 1
    if CommitCountingSession.state["fail_at"] == CommitCountingSession.state["commits"]:
        raise RuntimeError("synthetic_after_commit")


@contextmanager
def fail_on_event_key(engine, marker: str):
    """Raise on the event INSERT whose bound parameters carry ``marker``."""
    evidence = {"reached": False}

    def fail(conn, cursor, statement, parameters, context, executemany):
        normalized = statement.lower().replace('"', "").strip()
        if normalized.startswith("insert into agent_execution_events") and marker in str(parameters):
            evidence["reached"] = True
            raise RuntimeError("synthetic_event_insert_failure")

    event.listen(engine, "before_cursor_execute", fail)
    try:
        yield evidence
    finally:
        event.remove(engine, "before_cursor_execute", fail)


@contextmanager
def delay_on_insert(engine, table: str, seconds: float):
    """Sleep after the first real INSERT into ``table`` (inside the open TX)."""
    import time

    evidence = {"reached": False}

    def slow(conn, cursor, statement, parameters, context, executemany):
        normalized = statement.lower().replace('"', "").strip()
        if not evidence["reached"] and normalized.startswith("insert into " + table.lower()):
            evidence["reached"] = True
            time.sleep(seconds)

    event.listen(engine, "after_cursor_execute", slow)
    try:
        yield evidence
    finally:
        event.remove(engine, "after_cursor_execute", slow)


class WorkoutBridgeHarness:
    """Real C1 ledger preparation plus the real bridge over one isolated engine."""

    def __init__(self, engine, *, task_type="workout_record", kind="workout", records=None,
                 source=None, ttl=300, session_class=None, user_id="c2-owner", request_id="c2-request",
                 task_id="task-1", position=0, auto_reserve=True, safety="auto", understanding="auto"):
        from app.services.agent import execution_ledger_store as ledger_api

        self.ledger = ledger_api
        self.engine = engine
        self.factory = sessionmaker(bind=engine, expire_on_commit=False)
        self.store = ledger_api.ExecutionLedgerStore(self.factory)
        if session_class is not None:
            self.bridge_factory = sessionmaker(bind=engine, class_=session_class, expire_on_commit=False)
        else:
            self.bridge_factory = self.factory
        self.source = source if source is not None else SUCCESS_SOURCE
        self.safety = default_safety(self.source) if safety == "auto" else (
            dict(safety) if isinstance(safety, dict) else None)
        bind_request_semantics(self.source, tasks=understanding, safety=self.safety)
        self.span_start = self.source.index(SPAN_TEXT) if SPAN_TEXT in self.source else SPAN_START
        self.span_end = self.span_start + len(SPAN_TEXT)
        if SPAN_TEXT not in self.source:
            self.span_start, self.span_end = 0, len(self.source)
        self.records = [dict(item) for item in (records or [frozen_record()])]
        self.task_type = task_type
        self.kind = kind
        self.task_id = task_id
        self.position = position
        self.identity = {
            "user_id": user_id,
            "request_id": request_id,
            "conversation_id": "c2-conversation",
            "source": self.source,
            "source_digest": hashlib.sha256(self.source.encode("utf-8")).hexdigest(),
            "requested_mode": "react_readonly",
            "effective_mode": "react_readonly",
            "schema_version": 2,
            "tool_version": "react-tools-v2",
            "time_base": "2026-10-07T00:00:00+00:00",
            "budget_scope_id": "scope:synthetic-c2",
            "request_kind": "tasks",
        }
        snapshot = sync(self.store.ensure_request(ledger_api.RequestIdentity(**self.identity)))
        snapshot = sync(self.store.acquire_lease(self.identity["user_id"], self.identity["request_id"], "c2-worker", ttl))
        self.snapshot = snapshot
        task = ledger_api.TaskSpec(
            task_id=self.task_id,
            source_start=self.span_start,
            source_end=self.span_end,
            source_digest=hashlib.sha256(self.source[self.span_start:self.span_end].encode("utf-8")).hexdigest(),
            task_type=self.task_type,
            dependencies=[],
            parameters={"records": copy.deepcopy(self.records)},
            required=True,
        )
        self.snapshot = sync(self.call_store("freeze_task", task))
        if auto_reserve:
            self.record_op(self.position)
        self.proof = self.lease_proof()

    @property
    def ids(self):
        return self.identity["user_id"], self.identity["request_id"]

    def lease_proof(self):
        lease = self.snapshot["lease"]
        return self.ledger.LeaseProof(worker_id=lease["worker_id"], token=lease["token"], epoch=lease["epoch"])

    def call_store(self, name, *args, **kwargs):
        kwargs.setdefault("expected_revision", self.snapshot["revision"])
        if "lease" not in kwargs:
            kwargs["lease"] = self.lease_proof()
        self.snapshot = sync(getattr(self.store, name)(*self.ids, *args, **kwargs))
        return self.snapshot

    def record_op(self, position, *, kind=None, parameters=None, task_id=None, dispatch=True):
        task_id = task_id or self.task_id
        self.snapshot = sync(self.call_store(
            "reserve_operation",
            task_id=task_id,
            position=position,
            kind=kind or self.kind,
            parameters=copy.deepcopy(parameters if parameters is not None else self.records[position]),
            source_digest=self.identity["source_digest"],
            time_base=self.identity["time_base"],
        ))
        op_id = self.op_id(position, task_id=task_id)
        if dispatch:
            self.snapshot = sync(self.call_store("mark_dispatch_intent", op_id))
        return op_id

    def op_id(self, position=0, *, task_id=None):
        return op_id_for(self.identity["user_id"], self.identity["request_id"], task_id or self.task_id, position, self.kind)

    def bridge(self):
        return getattr(bridge_module(), BRIDGE_CLASS)(self.bridge_factory)

    def commit(self, *, op_id=None, expected_revision=None, lease=None, tool_call_id="call-c2-0001", batch_id="batch-c2-1", **overrides):
        bridge = self.bridge()
        kwargs = dict(
            expected_revision=self.snapshot["revision"] if expected_revision is None else expected_revision,
            lease=self.lease_proof() if lease is None else lease,
            tool_call_id=tool_call_id,
            batch_id=batch_id,
        )
        kwargs.update(overrides)
        if self.safety is None:
            result = bridge.commit_workout_operation(
                self.identity["user_id"], self.identity["request_id"], op_id or self.op_id(self.position), **kwargs)
        else:
            with assessed_safety(self.source, **self.safety):
                result = bridge.commit_workout_operation(
                    self.identity["user_id"], self.identity["request_id"], op_id or self.op_id(self.position), **kwargs)
        self.reload()
        return result

    def recover(self, *, op_id=None):
        bridge = self.bridge()
        result = bridge.recover_workout_operation(self.identity["user_id"], self.identity["request_id"], op_id or self.op_id(self.position))
        self.reload()
        return result

    def reload(self):
        self.snapshot = sync(self.ledger.ExecutionLedgerStore(self.factory).load_request(*self.ids))
        return self.snapshot

    def event(self, event_key):
        return event_by_key(self.snapshot, event_key)

    def claim_event_key(self, op_id=None):
        return "c2-1-claim:v1:" + (op_id or self.op_id(self.position))

    def confirmation_event_key(self, op_id=None):
        return "c2-1-commit:v1:" + (op_id or self.op_id(self.position))

    def operation(self, op_id=None):
        target = op_id or self.op_id(self.position)
        return next(item for item in self.snapshot["operations"] if item["op_id"] == target)

    def task(self):
        return next(item for item in self.snapshot["tasks"] if item["task_id"] == self.task_id)


def subprocess_bridge_call(url, backend, user_id, request_id, op_id, expected_revision, lease, tool_call_id, batch_id):
    """Launch the versioned C2-2 runner child-workout entry for a fresh-process replay check."""
    import subprocess
    import sys

    runner = Path(__file__).resolve().parents[2] / "docs/react-phase-c2-2-evidence/run.py"
    packet = {
        "url": url, "backend": backend, "user_id": user_id, "request_id": request_id,
        "op_id": op_id, "expected_revision": expected_revision,
        "lease": {"worker_id": lease.worker_id, "token": lease.token, "epoch": lease.epoch},
        "tool_call_id": tool_call_id, "batch_id": batch_id,
    }
    result = subprocess.run(
        [sys.executable, str(runner), "child-workout"], input=json.dumps(packet),
        capture_output=True, text=True, encoding="utf-8", timeout=30,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, "C2 child bridge call failed (see stable child code)"
    return json.loads(result.stdout)


def _redirect_c1_recovery():
    """Only in this C2 plugin: point the C1 helper's child launcher at the versioned child-read."""
    import tests.react_phase_c1_support as c1_support

    def c2_subprocess_recovery(url, backend, user_id, request_id):
        import subprocess
        import sys

        runner = Path(__file__).resolve().parents[2] / "docs/react-phase-c2-2-evidence/run.py"
        packet = {"url": url, "backend": backend, "user_id": user_id, "request_id": request_id}
        result = subprocess.run(
            [sys.executable, str(runner), "child-read"], input=json.dumps(packet),
            capture_output=True, text=True, encoding="utf-8", timeout=30,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode == 0, "C2 child C1-recovery call failed (see stable child code)"
        return json.loads(result.stdout)

    c1_support.subprocess_recovery = c2_subprocess_recovery


_audit_state = {"context": None, "audit": None}


def _race_with_request_context(original):
    """Run C1's two real connection workers inside a copy of this request's
    context, so the scripted safety/semantic binding is visible in both threads
    exactly as the application's request context is."""
    import contextvars

    def race(engine, function):
        captured = contextvars.copy_context()

        def bound(*args, **kwargs):
            return captured.copy().run(function, *args, **kwargs)

        return original(engine, bound)

    return race


def pytest_sessionstart(session):
    import tests.react_phase_c1_support as c1_support

    c1_support.race = _race_with_request_context(c1_support.race)
    _redirect_c1_recovery()
    _audit_state["context"] = network_fence()
    _audit_state["audit"] = _audit_state["context"].__enter__()


def pytest_collection_finish(session):
    import os

    path = os.environ.get("FITZEN_C2_COLLECTION_MANIFEST")
    if path:
        Path(path).write_text(json.dumps([item.nodeid for item in session.items], indent=2), encoding="utf-8")


def pytest_sessionfinish(session, exitstatus):
    audit = _audit_state["audit"]
    _audit_state["context"].__exit__(None, None, None)
    _audit_state["context"] = None
    _audit_state["audit"] = None
    emit(
        "NETWORK",
        blocked_connections=audit["denied"]["blocked_connections"],
        blocked_dns=audit["denied"]["blocked_dns"],
        allowed_pg_socket_connections=audit["allowed_pg_socket_connections"],
        allowed_pg_dns=audit["allowed_pg_dns"],
        verified_db_checkouts=audit["verified_db_checkouts"],
    )
    if any(audit["denied"].values()):
        session.exitstatus = 1
