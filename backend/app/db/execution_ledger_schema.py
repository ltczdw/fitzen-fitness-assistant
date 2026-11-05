"""Shape creation/verification for the four C1 execution ledger tables.

The function creates missing ledger tables (idempotent) and verifies that
existing same-name tables have the frozen minimum columns, uniqueness and
foreign keys. It never drops, rebuilds or commits; an incompatible existing
shape raises a limited schema error so the caller's transaction rolls back.
"""
from __future__ import annotations

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Connection

from app.db.execution_ledger_models import (
    ExecutionEventModel,
    ExecutionOperationModel,
    ExecutionRequestModel,
    ExecutionTaskModel,
)

_REQUEST = "agent_execution_requests"
_TASK = "agent_execution_tasks"
_OPERATION = "agent_execution_operations"
_EVENT = "agent_execution_events"

_TABLES = {
    _REQUEST: ExecutionRequestModel.__table__,
    _TASK: ExecutionTaskModel.__table__,
    _OPERATION: ExecutionOperationModel.__table__,
    _EVENT: ExecutionEventModel.__table__,
}

_EXPECTED = {
    _REQUEST: {
        "columns": {
            "user_id", "request_id", "conversation_id", "source", "source_digest",
            "requested_mode", "effective_mode", "schema_version", "tool_version",
            "time_base", "budget_scope_id", "request_kind", "status", "revision",
            "lease_worker_id", "lease_token", "lease_epoch", "lease_deadline",
            "checkpoint", "response",
        },
        "unique": [("user_id", "request_id")],
        "foreign_keys": [],
    },
    _TASK: {
        "columns": {
            "user_id", "request_id", "task_id", "source_start", "source_end",
            "source_digest", "task_type", "dependencies", "parameters", "required",
            "status", "current_result",
        },
        "unique": [("user_id", "request_id", "task_id")],
        "foreign_keys": [
            {"columns": ["user_id", "request_id"], "table": _REQUEST, "referred": ["user_id", "request_id"]},
        ],
    },
    _OPERATION: {
        "columns": {
            "user_id", "request_id", "op_id", "idempotency_key", "task_id", "position",
            "kind", "parameters", "parameters_digest", "source_digest", "time_base",
            "status", "confirmation_result",
        },
        "unique": [
            ("user_id", "request_id", "op_id"),
            ("user_id", "request_id", "task_id", "position"),
            ("idempotency_key",),
        ],
        "foreign_keys": [
            {"columns": ["user_id", "request_id", "task_id"], "table": _TASK, "referred": ["user_id", "request_id", "task_id"]},
        ],
    },
    _EVENT: {
        "columns": {
            "user_id", "request_id", "event_id", "sequence", "event_key", "tool_call_id",
            "batch_id", "task_id", "op_id", "phase", "observation", "checkpoint",
            "task_status", "operation_state", "confirmed",
        },
        "unique": [
            ("user_id", "request_id", "event_id"),
            ("user_id", "request_id", "sequence"),
            ("user_id", "request_id", "event_key"),
        ],
        "foreign_keys": [
            {"columns": ["user_id", "request_id", "task_id"], "table": _TASK, "referred": ["user_id", "request_id", "task_id"]},
            {"columns": ["user_id", "request_id", "op_id"], "table": _OPERATION, "referred": ["user_id", "request_id", "op_id"]},
        ],
    },
}


class ExecutionLedgerSchemaError(RuntimeError):
    """Limited failure for an incompatible existing ledger table shape."""

    def __init__(self, reason: str):
        self.code = "state_corrupt"
        self.reason = reason
        super().__init__(reason)


def _unique_guarantees(inspector, name: str) -> set[tuple]:
    guarantees = {tuple(inspector.get_pk_constraint(name).get("constrained_columns") or ())}
    for constraint in inspector.get_unique_constraints(name):
        guarantees.add(tuple(constraint.get("column_names") or ()))
    for index in inspector.get_indexes(name):
        if index.get("unique"):
            guarantees.add(tuple(index.get("column_names") or ()))
    return guarantees


def _verify_table(inspector, name: str) -> None:
    expected = _EXPECTED[name]
    actual_columns = {column["name"] for column in inspector.get_columns(name)}
    missing = expected["columns"] - actual_columns
    if missing:
        raise ExecutionLedgerSchemaError(f"{name}: missing columns")
    guarantees = _unique_guarantees(inspector, name)
    for wanted in expected["unique"]:
        if tuple(wanted) not in guarantees:
            raise ExecutionLedgerSchemaError(f"{name}: missing unique shape")
    actual_fks = inspector.get_foreign_keys(name)
    for wanted in expected["foreign_keys"]:
        matched = any(
            fk.get("constrained_columns") == wanted["columns"]
            and fk.get("referred_table") == wanted["table"]
            and fk.get("referred_columns") == wanted["referred"]
            for fk in actual_fks
        )
        if not matched:
            raise ExecutionLedgerSchemaError(f"{name}: missing foreign key shape")


def verify_execution_ledger_schema(connection: Connection) -> None:
    """Read-only check; never creates or repairs a ledger table."""
    inspector = sa_inspect(connection)
    for name in _TABLES:
        if not inspector.has_table(name):
            raise ExecutionLedgerSchemaError("execution ledger table missing")
        _verify_table(inspector, name)


def ensure_execution_ledger_schema(connection: Connection) -> None:
    """Create missing ledger tables and verify the frozen minimum shape.

    Runs inside the caller's transaction: missing tables are created in
    dependency order, existing tables are only inspected. An incompatible
    existing table raises :class:`ExecutionLedgerSchemaError` without any
    ``DROP``/``ALTER``/commit so the caller can roll the whole transaction back.
    """
    for table in _TABLES.values():
        table.create(bind=connection, checkfirst=True)
    verify_execution_ledger_schema(connection)
