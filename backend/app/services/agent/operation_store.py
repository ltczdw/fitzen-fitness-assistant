"""Stable execution identity for retried requests (P1-2).

The first request persists its confirmed task plan, the parameters produced by
later entity extraction, and one entry per write operation *before* the
operation is dispatched. A retry of the same ``request_id`` reuses all of it,
so a second model ordering or a new UTC day cannot reinterpret an already
executed operation or write the same business action twice.

A ledger that cannot be read or persisted must never let a business write
through: accessors that gate a dispatch raise ``OperationStoreUnavailable``
and the callers fail closed. Operation ids are derived from the idempotency
key (which encodes request, task and position), so two intentionally identical
operations in one request stay distinct while a retry keeps the exact same
identities.
"""
from __future__ import annotations

import hashlib
import json
import logging
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from app.db.storage import IdempotencyConflict

logger = logging.getLogger(__name__)

current_operations: ContextVar["OperationStore | None"] = ContextVar("request_operations", default=None)


class OperationStoreUnavailable(RuntimeError):
    """The retry ledger could not be read or persisted.

    Raised by the accessors a write dispatch depends on, so no save tool can
    run against a plan, parameter set or reservation that was not durably
    recorded for the retry.
    """

    code = "operation_ledger_unavailable"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def operation_id_for_key(kind: str, key: str) -> str:
    """Positional identity: exactly one ledger entry per idempotency key.

    The key encodes request, task and record index, so identical content at
    different positions never collapses into one operation.
    """
    return f"{kind}:op:" + hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:32]


def workout_fingerprint(record: dict[str, Any]) -> str:
    return content_digest({
        "kind": "workout",
        "exercise": str(record.get("exercise") or "").strip(),
        "sets": int(record.get("sets") or 0),
        "reps": int(record.get("reps") or 0),
        "weight_kg": round(float(record.get("weight_kg") or 0.0), 4),
        "date": str(record.get("date") or ""),
        "set_details": record.get("set_details"),
        "completed": bool(record.get("completed", True)),
    })[:32]


def workout_operation_id(record: dict[str, Any]) -> str:
    key = str(record.get("idempotency_key") or "")
    if key:
        return operation_id_for_key("workout", key)
    return "workout:" + workout_fingerprint(record)


def food_fingerprint(entries: list[dict[str, Any]]) -> str:
    return content_digest({
        "kind": "food",
        "entries": [
            {field: entry.get(field) for field in (
                "requested_name", "weight_g", "quantity", "unit", "meal_type", "record_date")}
            for entry in entries
        ],
    })[:32]


def food_operation_id(entries: list[dict[str, Any]], *, key: str | None = None) -> str:
    if key:
        return operation_id_for_key("food", key)
    return "food:" + food_fingerprint(entries)


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


class OperationStore:
    """In-memory view of one persisted request state; all DB I/O is sync."""

    def __init__(self, request_id: str, *, persist: bool = True) -> None:
        self.request_id = request_id
        self.persist = persist
        self._state: dict[str, Any] | None = None

    @staticmethod
    def _default_state() -> dict[str, Any]:
        return {"version": 1, "plan": None, "time_base": None, "tasks": {}, "operations": {}}

    def load(self) -> dict[str, Any]:
        if self._state is not None:
            return self._state
        state = None
        if self.persist:
            from app.services.memory import compact
            try:
                state = compact.load_request_state(self.request_id)
            except Exception as exc:
                logger.exception("request_state_load_failed")
                raise OperationStoreUnavailable("request_state_read_failed") from exc
        merged = self._default_state()
        if isinstance(state, dict):
            if state.get("plan") is not None:
                merged["plan"] = state.get("plan")
            if state.get("time_base"):
                merged["time_base"] = str(state["time_base"])
            if isinstance(state.get("tasks"), dict):
                merged["tasks"] = state["tasks"]
            if isinstance(state.get("operations"), dict):
                merged["operations"] = state["operations"]
        self._state = merged
        return merged

    def _persist(self) -> None:
        """Persist the current state or raise ``OperationStoreUnavailable``.

        ``save_plan``, ``save_task_params`` and ``reserve_operation`` gate a
        business write and must let this propagate: with an unpersisted ledger
        a retry could replay or merge that write. ``mark_operation`` is the
        single deliberate exception because its write already happened.
        """
        if not self.persist or self._state is None:
            return
        from app.services.memory import compact
        try:
            saved = compact.save_request_state(self.request_id, self._state)
        except Exception as exc:
            logger.exception("request_state_save_failed")
            raise OperationStoreUnavailable("request_state_write_failed") from exc
        if saved is False:
            # The turn row is gone or already deleted; the ledger no longer has
            # an owner, so a retry could not find it either.
            logger.error("request_state_save_rejected", extra={"request_id": self.request_id})
            raise OperationStoreUnavailable("request_state_turn_unavailable")

    # -- plan -----------------------------------------------------------
    def stored_plan(self) -> list[dict[str, Any]] | None:
        plan = self.load().get("plan")
        return plan if isinstance(plan, list) else None

    def save_plan(self, tasks: list[dict[str, Any]], *, time_base: datetime | None = None) -> None:
        state = self.load()
        if state.get("plan") is not None:
            return
        state["plan"] = json.loads(json.dumps(list(tasks), ensure_ascii=False, default=str))
        if time_base is not None and not state.get("time_base"):
            state["time_base"] = time_base.isoformat()
        self._persist()

    def time_base(self) -> datetime | None:
        return _parse_time(self.load().get("time_base"))

    # -- task parameters (entity extraction fixed before dispatch) ------
    def task_params(self, task_id: str) -> dict[str, Any] | None:
        params = self.load().get("tasks", {}).get(str(task_id))
        return params if isinstance(params, dict) else None

    def save_task_params(self, task_id: str, params: dict[str, Any]) -> dict[str, Any]:
        state = self.load()
        existing = state["tasks"].get(str(task_id))
        if isinstance(existing, dict):
            return existing
        stored = json.loads(json.dumps(params, ensure_ascii=False, default=str))
        state["tasks"][str(task_id)] = stored
        self._persist()
        return stored

    # -- operations -----------------------------------------------------
    def operation(self, op_id: str) -> dict[str, Any] | None:
        op = self.load().get("operations", {}).get(op_id)
        return op if isinstance(op, dict) else None

    def reserve_operation(self, op_id: str, *, kind: str, key: str, fingerprint: str,
                          params: dict[str, Any]) -> dict[str, Any]:
        state = self.load()
        operations = state["operations"]
        existing = operations.get(op_id)
        if isinstance(existing, dict):
            if existing.get("fingerprint") != fingerprint:
                # Same position (key) reinterpreted with different parameters:
                # never silently reuse the old result or overwrite the entry.
                raise IdempotencyConflict(
                    f"operation {op_id} was already reserved with different parameters")
            return existing
        for other in operations.values():
            if (other.get("key") == key and other.get("fingerprint")
                    and other.get("fingerprint") != fingerprint):
                raise IdempotencyConflict(
                    f"idempotency key {key} was already used for another operation")
        entry = {
            "op_id": op_id, "kind": kind, "key": key, "fingerprint": fingerprint,
            "params": json.loads(json.dumps(params, ensure_ascii=False, default=str)),
            "status": "pending", "result": None, "error": None,
        }
        operations[op_id] = entry
        self._persist()
        return entry

    def mark_operation(self, op_id: str, status: str, *, result: Any = None,
                       error: str | None = None) -> None:
        """Best-effort status update; the business write already happened.

        A failed update is logged and the entry stays ``pending``/``unknown``,
        so a retry recovers it by idempotency key instead of replaying it.
        """
        try:
            state = self.load()
            entry = state["operations"].get(op_id)
            if not isinstance(entry, dict):
                return
            entry["status"] = status
            if result is not None:
                entry["result"] = json.loads(json.dumps(result, ensure_ascii=False, default=str))
            entry["error"] = error
            self._persist()
        except OperationStoreUnavailable:
            logger.exception("operation_status_update_failed",
                             extra={"op_id": op_id, "status": status})


def workout_operation_identity(record: dict[str, Any]) -> tuple[str, str]:
    return workout_operation_id(record), workout_fingerprint(record)


def food_operation_identity(entries: list[dict[str, Any]], *,
                            key: str | None = None, nutrition_overrides: dict | None = None) -> tuple[str, str]:
    fingerprint = food_fingerprint(entries)
    if nutrition_overrides:
        import hashlib
        fingerprint = hashlib.sha256((fingerprint + json.dumps(nutrition_overrides, sort_keys=True, ensure_ascii=False, allow_nan=False)).encode()).hexdigest()
    return food_operation_id(entries, key=key), fingerprint
