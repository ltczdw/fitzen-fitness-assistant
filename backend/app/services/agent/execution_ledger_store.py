"""Synchronous v2 execution ledger store (phase C1).

This module is a new, isolated persistence boundary for the execution ledger.
It never touches the legacy in-memory ``OperationStore``/``MemoryTurn`` paths
and is not wired into any chat, agent, runtime or write-handler entry point.
All methods are synchronous ``def`` and use only the caller-provided
``sessionmaker``; snapshots are detached plain dicts (deep copies of persisted
state), never live ORM objects or caller-referenced payloads.

Stable limited failures are raised as :class:`ExecutionLedgerError` with a
``code`` from the frozen interface; SQL text, connection strings and arbitrary
exception objects are never surfaced.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import secrets
import weakref
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, func, insert, literal, select, text, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.db.execution_ledger_models import (
    ExecutionEventModel,
    ExecutionOperationModel,
    ExecutionRequestModel,
    ExecutionTaskModel,
)

_TTL_MINIMUM = 1
_TTL_MAXIMUM = 300
_WORKOUT_EVENT_PREFIXES = ("c2-1-claim:v1:", "c2-1-commit:v1:")
_NUTRITION_EVENT_PREFIXES = ("c2-2-pending:v1:", "c2-2-confirmation-use:v1:", "c2-2-claim:v1:", "c2-2-commit:v1:")
# C2-3 runtime private family: its own namespace, origin key and fixed non-secret
# generation pinned by the public clarifications/spec v2.
_RUNTIME_EVENT_PREFIXES = ("c2-3-runtime:v1:",)
_CORRECTION_EVENT_PREFIX = "workout-name-correction:v1:"
_CORRECTION_ORIGIN_KEY = "_workout_name_correction_origin"
_CORRECTION_ORIGIN = {"protocol": 1, "generation": "eb94a531-c922-4fe7-b280-6e09e676bcd6"}
_RESERVED_EVENT_PREFIXES = (_WORKOUT_EVENT_PREFIXES + _NUTRITION_EVENT_PREFIXES
                            + _RUNTIME_EVENT_PREFIXES + (_CORRECTION_EVENT_PREFIX,))
# C2-1-09b bridge origin protocol: stamped by code on the dedicated path only.
# The generation is a fixed non-secret constant pinned in interface-freeze-v4;
# it must never change per process or per call.
_BRIDGE_ORIGIN_KEY = "_workout_bridge_origin"
_BRIDGE_ORIGIN_PROTOCOL = 2
_BRIDGE_ORIGIN_GENERATION = "60459400-f1c8-4ca3-875c-54d03b755834"
# C2-2-09b nutrition bridge origin protocol: separate family, own fixed
# generation pinned by interface-freeze-v2.
_NUTRITION_ORIGIN_KEY = "_nutrition_bridge_origin"
_NUTRITION_ORIGIN_PROTOCOL = 2
_NUTRITION_ORIGIN_GENERATION = "ea0c7b67-2ec9-4db9-a3a8-81a3475e4cda"
# C2-3 runtime family origin: separate from both bridges; protocol 1 and the
# fixed generation from the interface spec v2 (never per-process/per-call).
_RUNTIME_ORIGIN_KEY = "_react_runtime_origin"
_RUNTIME_ORIGIN_PROTOCOL = 1
_RUNTIME_ORIGIN_GENERATION = "6bcac6d1-6791-4d5c-9003-ea4691cd2d30"

_REQUEST_STATUSES = frozenset({"pending", "running", "limited", "failed", "completed"})
_TERMINAL_REQUEST_STATUSES = frozenset({"limited", "failed", "completed"})
_OPERATION_STATUSES = frozenset({"reserved", "dispatch_intent", "committed", "commit_unknown", "rejected"})
_EVENT_OPERATION_STATES = frozenset({"committed", "commit_unknown", "rejected"})
_UNFINISHED_OPERATION_STATUSES = frozenset({"reserved", "dispatch_intent", "commit_unknown"})
_FINISH_STATUSES = frozenset({"completed", "failed", "limited"})

_IDENTITY_FIELDS = (
    "user_id",
    "request_id",
    "conversation_id",
    "source",
    "source_digest",
    "requested_mode",
    "effective_mode",
    "schema_version",
    "tool_version",
    "time_base",
    "budget_scope_id",
    "request_kind",
)
_FROZEN_IDENTITY_FIELDS = tuple(
    field for field in _IDENTITY_FIELDS if field not in {"user_id", "request_id"}
)
_CHECKPOINT_FIELDS = ("supervisor_turns", "tool_steps", "event_sequence", "budget_scope_id")
_OPERATION_FIELDS = (
    "op_id",
    "idempotency_key",
    "task_id",
    "position",
    "kind",
    "parameters",
    "parameters_digest",
    "source_digest",
    "time_base",
    "status",
    "confirmation_result",
)
_EVENT_FIELDS = (
    "event_id",
    "sequence",
    "event_key",
    "tool_call_id",
    "batch_id",
    "task_id",
    "op_id",
    "phase",
    "observation",
    "checkpoint",
    "task_status",
    "operation_state",
    "confirmed",
)
_EVENT_PAYLOAD_FIELDS = (
    "tool_call_id",
    "batch_id",
    "task_id",
    "op_id",
    "phase",
    "observation",
    "task_status",
    "operation_state",
    "confirmed",
)


class ExecutionLedgerError(Exception):
    """Stable limited ledger failure exposed to callers."""

    def __init__(self, code: str, reason: str | None = None):
        self.code = code
        self.reason = reason
        super().__init__(code + (f": {reason}" if reason else ""))


class RequestSnapshot(dict):
    """Detached execution-ledger snapshot.

    Item access always returns a fresh deep copy, so callers may freely
    mutate whatever they read without ever altering the snapshot, the store
    or persisted state.
    """

    def __getitem__(self, key):
        return copy.deepcopy(dict.__getitem__(self, key))

    def get(self, key, default=None):
        if key in self:
            return copy.deepcopy(dict.__getitem__(self, key))
        return default


class RecoverySnapshot(RequestSnapshot):
    """Request snapshot extended with recovery-only pending intent fields."""


@dataclass
class RequestIdentity:
    user_id: str
    request_id: str
    conversation_id: str
    source: str
    source_digest: str
    requested_mode: str
    effective_mode: str
    schema_version: int
    tool_version: str
    time_base: str
    budget_scope_id: str
    request_kind: str


@dataclass
class LeaseProof:
    worker_id: str
    token: str
    epoch: int


@dataclass
class TaskSpec:
    task_id: str
    source_start: int
    source_end: int
    source_digest: str
    task_type: str
    dependencies: list
    parameters: dict
    required: bool


@dataclass
class Checkpoint:
    supervisor_turns: int
    tool_steps: int
    event_sequence: int
    budget_scope_id: str


@dataclass
class EventInput:
    event_key: str
    tool_call_id: str
    batch_id: str
    task_id: str
    op_id: str | None
    phase: str
    observation: dict
    task_status: str
    operation_state: str | None
    confirmed: bool


def _invalid(reason: str) -> ExecutionLedgerError:
    return ExecutionLedgerError("input_invalid", reason)


def _conflict(reason: str) -> ExecutionLedgerError:
    return ExecutionLedgerError("identity_conflict", reason)


def _corrupt(reason: str) -> ExecutionLedgerError:
    return ExecutionLedgerError("state_corrupt", reason)


def _unavailable() -> ExecutionLedgerError:
    return ExecutionLedgerError("ledger_unavailable")


def _strict_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid(f"{name}_must_be_nonempty_string")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise _invalid(f"{name}_must_be_bool")
    return value


def _strict_int(value: Any, name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid(f"{name}_must_be_strict_integer")
    if minimum is not None and value < minimum:
        raise _invalid(f"{name}_below_minimum")
    if maximum is not None and value > maximum:
        raise _invalid(f"{name}_above_maximum")
    return value


def _finite_json(value: Any, name: str, _depth: int = 0) -> Any:
    """Return a detached finite-JSON copy; reject anything else before SQL."""
    if _depth > 64:
        raise _invalid(f"{name}_too_deep")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise _invalid(f"{name}_non_finite_number")
        return value
    if isinstance(value, (list, tuple)):
        return [_finite_json(item, name, _depth + 1) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _invalid(f"{name}_non_string_key")
            result[key] = _finite_json(item, name, _depth + 1)
        return result
    raise _invalid(f"{name}_unsupported_type")


def _finite_persisted(value: Any, _depth: int = 0) -> bool:
    if _depth > 64:
        return False
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_persisted(item, _depth + 1) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_persisted(item, _depth + 1) for key, item in value.items())
    return False


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _op_identity(user_id: str, request_id: str, task_id: str, position: int, kind: str) -> tuple[str, str]:
    material = ["ledger-v2", user_id, request_id, task_id, position, kind]
    digest = _digest(material)
    return f"op2_{digest[:32]}", f"ledger-v2:{digest}"


def _event_identity(user_id: str, request_id: str, event_key: str) -> str:
    return "ev2:" + _digest(["ledger-v2-event", user_id, request_id, event_key])


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return _as_utc(value).isoformat()


class ExecutionLedgerStore:
    """Persistent v2 execution ledger over the caller-provided sessionmaker."""

    def __init__(self, factory):
        self._factory = factory

    # ------------------------------------------------------------------ reads

    def _session(self):
        return self._factory()

    @staticmethod
    def _safe_close(session) -> None:
        try:
            session.close()
        except Exception:
            pass

    @staticmethod
    def _safe_rollback(session) -> None:
        try:
            session.rollback()
        except Exception:
            pass

    def _get_request(self, session, user_id: str, request_id: str):
        try:
            return session.execute(
                select(ExecutionRequestModel).where(
                    ExecutionRequestModel.user_id == user_id,
                    ExecutionRequestModel.request_id == request_id,
                )
            ).scalar_one_or_none()
        except ExecutionLedgerError:
            raise
        except SQLAlchemyError:
            raise _unavailable() from None
        except (ValueError, TypeError, KeyError):
            raise _corrupt("persisted_row_unreadable") from None
        except Exception:
            raise _unavailable() from None

    def _load_snapshot(self, session, user_id: str, request_id: str):
        try:
            row = session.execute(
                select(ExecutionRequestModel).where(
                    ExecutionRequestModel.user_id == user_id,
                    ExecutionRequestModel.request_id == request_id,
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return self._build_snapshot(session, row)
        except ExecutionLedgerError:
            raise
        except SQLAlchemyError:
            raise _unavailable() from None
        except (ValueError, TypeError, KeyError):
            raise _corrupt("persisted_row_unreadable") from None
        except Exception:
            raise _unavailable() from None

    def _build_snapshot(self, session, row) -> dict:
        identity = {field: getattr(row, field) for field in _IDENTITY_FIELDS}
        if not isinstance(identity["schema_version"], int) or isinstance(identity["schema_version"], bool):
            raise _corrupt("request_schema_version_shape")
        status = row.status if row.status in _REQUEST_STATUSES else None
        if status is None:
            raise _corrupt("request_status_shape")
        revision = row.revision
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise _corrupt("request_revision_shape")
        checkpoint = _persisted_checkpoint(row.checkpoint)
        lease = self._persisted_lease(row)
        tasks = self._task_list(session, row.user_id, row.request_id)
        operations = self._operation_list(session, row.user_id, row.request_id)
        events = self._event_list(session, row.user_id, row.request_id)
        response = None
        if row.response is not None:
            if not isinstance(row.response, dict) or not _finite_persisted(row.response):
                raise _corrupt("request_response_shape")
            response = copy.deepcopy(row.response)
        return RequestSnapshot({
            "identity": identity,
            "status": status,
            "revision": revision,
            "lease": lease,
            "checkpoint": checkpoint,
            "tasks": tasks,
            "operations": operations,
            "events": events,
            "response": response,
        })

    def _persisted_lease(self, row) -> dict | None:
        if row.lease_token is None:
            return None
        if row.lease_worker_id is None or row.lease_epoch is None or row.lease_deadline is None:
            raise _corrupt("lease_shape")
        if isinstance(row.lease_epoch, bool) or not isinstance(row.lease_epoch, int):
            raise _corrupt("lease_epoch_shape")
        try:
            deadline = _iso_utc(row.lease_deadline)
        except Exception:
            raise _corrupt("lease_deadline_shape") from None
        return {
            "worker_id": row.lease_worker_id,
            "token": row.lease_token,
            "epoch": row.lease_epoch,
            "deadline": deadline,
        }

    def _task_list(self, session, user_id: str, request_id: str) -> list[dict]:
        rows = session.execute(
            select(ExecutionTaskModel)
            .where(ExecutionTaskModel.user_id == user_id, ExecutionTaskModel.request_id == request_id)
            .order_by(ExecutionTaskModel.task_id)
        ).scalars().all()
        tasks = []
        for row in rows:
            if row.required not in (True, False):
                raise _corrupt("task_required_shape")
            if not isinstance(row.dependencies, list) or any(not isinstance(item, str) for item in row.dependencies):
                raise _corrupt("task_dependencies_shape")
            if not isinstance(row.parameters, dict) or not _finite_persisted(row.parameters):
                raise _corrupt("task_parameters_shape")
            if row.current_result is not None and (
                not isinstance(row.current_result, dict) or not _finite_persisted(row.current_result)
            ):
                raise _corrupt("task_result_shape")
            tasks.append(
                {
                    "task_id": row.task_id,
                    "source_start": row.source_start,
                    "source_end": row.source_end,
                    "source_digest": row.source_digest,
                    "task_type": row.task_type,
                    "dependencies": copy.deepcopy(row.dependencies),
                    "parameters": copy.deepcopy(row.parameters),
                    "required": bool(row.required),
                    "status": row.status,
                    "current_result": copy.deepcopy(row.current_result),
                }
            )
        return tasks

    def _operation_list(self, session, user_id: str, request_id: str) -> list[dict]:
        rows = session.execute(
            select(ExecutionOperationModel)
            .where(ExecutionOperationModel.user_id == user_id, ExecutionOperationModel.request_id == request_id)
            .order_by(ExecutionOperationModel.task_id, ExecutionOperationModel.position, ExecutionOperationModel.op_id)
        ).scalars().all()
        operations = []
        for row in rows:
            if row.status not in _OPERATION_STATUSES:
                raise _corrupt("operation_status_shape")
            if not isinstance(row.parameters, dict) or not _finite_persisted(row.parameters):
                raise _corrupt("operation_parameters_shape")
            if row.confirmation_result is not None and (
                not isinstance(row.confirmation_result, dict) or not _finite_persisted(row.confirmation_result)
            ):
                raise _corrupt("operation_confirmation_shape")
            operations.append(
                {
                    "op_id": row.op_id,
                    "idempotency_key": row.idempotency_key,
                    "task_id": row.task_id,
                    "position": row.position,
                    "kind": row.kind,
                    "parameters": copy.deepcopy(row.parameters),
                    "parameters_digest": row.parameters_digest,
                    "source_digest": row.source_digest,
                    "time_base": row.time_base,
                    "status": row.status,
                    "confirmation_result": copy.deepcopy(row.confirmation_result),
                }
            )
        return operations

    def _event_list(self, session, user_id: str, request_id: str) -> list[dict]:
        rows = session.execute(
            select(ExecutionEventModel)
            .where(ExecutionEventModel.user_id == user_id, ExecutionEventModel.request_id == request_id)
            .order_by(ExecutionEventModel.sequence)
        ).scalars().all()
        events = []
        for row in rows:
            if not isinstance(row.observation, dict) or not _finite_persisted(row.observation):
                raise _corrupt("event_observation_shape")
            if row.operation_state is not None and row.operation_state not in _EVENT_OPERATION_STATES:
                raise _corrupt("event_operation_state_shape")
            events.append(
                {
                    "event_id": row.event_id,
                    "sequence": row.sequence,
                    "event_key": row.event_key,
                    "tool_call_id": row.tool_call_id,
                    "batch_id": row.batch_id,
                    "task_id": row.task_id,
                    "op_id": row.op_id,
                    "phase": row.phase,
                    "observation": _strip_runtime_origin(copy.deepcopy(row.observation)),
                    "checkpoint": _persisted_checkpoint(row.checkpoint),
                    "task_status": row.task_status,
                    "operation_state": row.operation_state,
                    "confirmed": bool(row.confirmed),
                }
            )
        return events

    # --------------------------------------------------------------- validate

    def _valid_identity(self, identity) -> dict:
        if not isinstance(identity, RequestIdentity):
            raise _invalid("identity_required")
        values = {field: getattr(identity, field) for field in _IDENTITY_FIELDS}
        for field in (
            "user_id", "request_id", "conversation_id", "source", "requested_mode",
            "effective_mode", "tool_version", "time_base", "budget_scope_id", "request_kind",
        ):
            _strict_str(values[field], field)
        digest = _strict_str(values["source_digest"], "source_digest")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise _invalid("source_digest_shape")
        expected = hashlib.sha256(values["source"].encode("utf-8")).hexdigest()
        if digest != expected:
            raise _invalid("source_digest_mismatch")
        values["schema_version"] = _strict_int(values["schema_version"], "schema_version", minimum=1)
        return values

    def _valid_task(self, task) -> dict:
        if not isinstance(task, TaskSpec):
            raise _invalid("task_spec_required")
        return {
            "task_id": _strict_str(task.task_id, "task_id"),
            "source_start": _strict_int(task.source_start, "source_start", minimum=0),
            "source_end": _strict_int(task.source_end, "source_end", minimum=0),
            "source_digest": _strict_str(task.source_digest, "source_digest"),
            "task_type": _strict_str(task.task_type, "task_type"),
            "dependencies": _finite_json(list(task.dependencies), "dependencies"),
            "parameters": _finite_json(dict(task.parameters), "parameters"),
            "required": _strict_bool(task.required, "required"),
        }

    def _valid_checkpoint(self, checkpoint) -> dict:
        if not isinstance(checkpoint, Checkpoint):
            raise _invalid("checkpoint_required")
        return {
            "supervisor_turns": _strict_int(checkpoint.supervisor_turns, "supervisor_turns", minimum=0),
            "tool_steps": _strict_int(checkpoint.tool_steps, "tool_steps", minimum=0),
            "event_sequence": _strict_int(checkpoint.event_sequence, "event_sequence", minimum=0),
            "budget_scope_id": _strict_str(checkpoint.budget_scope_id, "budget_scope_id"),
        }

    def _valid_event(self, event) -> dict:
        if not isinstance(event, EventInput):
            raise _invalid("event_input_required")
        op_id = event.op_id
        if op_id is not None:
            op_id = _strict_str(op_id, "op_id")
        operation_state = event.operation_state
        if operation_state is not None and operation_state not in _EVENT_OPERATION_STATES:
            raise _invalid("operation_state_unsupported")
        return {
            "event_key": _strict_str(event.event_key, "event_key"),
            "tool_call_id": _strict_str(event.tool_call_id, "tool_call_id"),
            "batch_id": _strict_str(event.batch_id, "batch_id"),
            "task_id": _strict_str(event.task_id, "task_id"),
            "op_id": op_id,
            "phase": _strict_str(event.phase, "phase"),
            "observation": _finite_json(event.observation, "observation"),
            "task_status": _strict_str(event.task_status, "task_status"),
            "operation_state": operation_state,
            "confirmed": _strict_bool(event.confirmed, "confirmed"),
        }

    def _valid_lease(self, lease) -> None:
        if not isinstance(lease, LeaseProof):
            raise _invalid("lease_proof_required")
        _strict_str(lease.worker_id, "worker_id")
        _strict_str(lease.token, "token")
        _strict_int(lease.epoch, "epoch", minimum=1)

    # ------------------------------------------------------------------- time

    @staticmethod
    def _now(session) -> datetime:
        value = session.execute(select(func.now())).scalar_one()
        if not isinstance(value, datetime):
            raise _unavailable()
        return _as_utc(value)

    def _require_lease(self, row, lease, now: datetime) -> None:
        self._valid_lease(lease)
        if (
            row.lease_token is None
            or row.lease_worker_id != lease.worker_id
            or row.lease_token != lease.token
            or row.lease_epoch != lease.epoch
        ):
            raise ExecutionLedgerError("lease_lost")
        if row.lease_deadline is None or _as_utc(row.lease_deadline) < now:
            raise ExecutionLedgerError("lease_lost")

    def _cas_request(self, session, row, expected_revision: int, lease: LeaseProof, *,
                     final_fence: bool = False, pending_deadline=None, **values) -> None:
        clauses = [
            ExecutionRequestModel.user_id == row.user_id,
            ExecutionRequestModel.request_id == row.request_id,
            ExecutionRequestModel.revision == expected_revision,
            ExecutionRequestModel.lease_worker_id == lease.worker_id,
            ExecutionRequestModel.lease_token == lease.token,
            ExecutionRequestModel.lease_epoch == lease.epoch,
        ]
        if final_fence:
            # C2-1: the last business write must compare the real statement
            # clock, never the transaction-start now() on PostgreSQL.
            if session.get_bind().dialect.name == "postgresql":
                clauses.append(ExecutionRequestModel.lease_deadline > func.clock_timestamp())
            else:
                clauses.append(ExecutionRequestModel.lease_deadline > func.now())
        if pending_deadline is not None:
            # C2-2: the required protected pending window must still contain the
            # database statement clock in the same final CAS SQL.
            if session.get_bind().dialect.name == "postgresql":
                clauses.append(literal(pending_deadline) > func.clock_timestamp())
            else:
                clauses.append(literal(pending_deadline) > func.now())
        result = session.execute(
            update(ExecutionRequestModel)
            .where(*clauses)
            .values(revision=expected_revision + 1, **values)
        )
        if result.rowcount == 1:
            return
        # 真实 rowcount0 诊断：必须按数据库已提交现状判定（并发 CAS 被阻塞后
        # 重试时，事务早期 ORM 身份映射里的旧行会造成误判），同时绝不触发
        # autoflush（保留本会话未提交状态）也不用实体加载覆盖/被覆盖会话内对象。
        # 列投影读取，缺行/存储异常映射与原 _get_request 完全一致。
        try:
            with session.no_autoflush:
                current = session.execute(
                    select(
                        ExecutionRequestModel.lease_worker_id,
                        ExecutionRequestModel.lease_token,
                        ExecutionRequestModel.lease_epoch,
                        ExecutionRequestModel.revision,
                    ).where(
                        ExecutionRequestModel.user_id == row.user_id,
                        ExecutionRequestModel.request_id == row.request_id,
                    )
                ).one_or_none()
        except ExecutionLedgerError:
            raise
        except SQLAlchemyError:
            raise _unavailable() from None
        except (ValueError, TypeError, KeyError):
            raise _corrupt("persisted_row_unreadable") from None
        except Exception:
            raise _unavailable() from None
        if current is None:
            raise _conflict("request_disappeared")
        current_worker_id, current_token, current_epoch, current_revision = current
        if (
            current_worker_id != lease.worker_id
            or current_token != lease.token
            or current_epoch != lease.epoch
        ):
            raise ExecutionLedgerError("lease_lost")
        if current_revision != expected_revision:
            raise ExecutionLedgerError("revision_conflict")
        if final_fence:
            raise ExecutionLedgerError("lease_lost")
        raise ExecutionLedgerError("revision_conflict")

    def _guarded_write(self, work):
        """Run ``work(session)`` then commit; map failures to limited codes."""
        session = self._session()
        try:
            try:
                value = work(session)
            except IntegrityError:
                self._safe_rollback(session)
                raise _conflict("unique_conflict") from None
            except ExecutionLedgerError:
                self._safe_rollback(session)
                raise
            except Exception:
                self._safe_rollback(session)
                raise _unavailable() from None
            try:
                session.commit()
            except Exception:
                self._safe_rollback(session)
                raise ExecutionLedgerError("commit_unknown") from None
            return value
        finally:
            self._safe_close(session)

    # ------------------------------------------------------------------ write

    def ensure_request(self, identity) -> dict:
        values = self._valid_identity(identity)
        user_id = values["user_id"]
        request_id = values["request_id"]
        session = self._session()
        try:
            row = self._get_request(session, user_id, request_id)
            if row is not None:
                self._compare_frozen_identity(row, values)
                return self._build_snapshot(session, row)
            initial = {
                "supervisor_turns": 0,
                "tool_steps": 0,
                "event_sequence": 0,
                "budget_scope_id": values["budget_scope_id"],
            }
            try:
                session.execute(
                    insert(ExecutionRequestModel).values(
                        **values, status="pending", revision=0, checkpoint=initial
                    )
                )
            except IntegrityError:
                self._safe_rollback(session)
                row = self._get_request(session, user_id, request_id)
                if row is None:
                    raise _unavailable() from None
                self._compare_frozen_identity(row, values)
                return self._build_snapshot(session, row)
            except Exception:
                self._safe_rollback(session)
                raise _unavailable() from None
            try:
                session.commit()
            except IntegrityError:
                self._safe_rollback(session)
                row = self._get_request(session, user_id, request_id)
                if row is None:
                    raise _unavailable() from None
                self._compare_frozen_identity(row, values)
                return self._build_snapshot(session, row)
            except Exception:
                self._safe_rollback(session)
                raise ExecutionLedgerError("commit_unknown") from None
            snapshot = self._load_snapshot(session, user_id, request_id)
            if snapshot is None:
                raise _unavailable()
            return snapshot
        finally:
            self._safe_close(session)

    def _compare_frozen_identity(self, row, values: dict) -> None:
        for field in _FROZEN_IDENTITY_FIELDS:
            if getattr(row, field) != values[field]:
                raise _conflict(f"frozen_identity_{field}")

    def load_request(self, user_id: str, request_id: str) -> dict | None:
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        session = self._session()
        try:
            return self._load_snapshot(session, user_id, request_id)
        finally:
            self._safe_close(session)

    def acquire_lease(self, user_id: str, request_id: str, worker_id: str, ttl_seconds: int) -> dict:
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        _strict_str(worker_id, "worker_id")
        _strict_int(ttl_seconds, "ttl_seconds", minimum=_TTL_MINIMUM, maximum=_TTL_MAXIMUM)

        def work(session):
            now = self._now(session)
            deadline = now + timedelta(seconds=ttl_seconds)
            token = secrets.token_hex(16)
            # managed react_write 终态防护：completed/failed/limited 后即使租约已清，
            # 迟到竞争者也不得重新取得（真实 CAS：WHERE 排除 → rowcount 0 → 下方
            # terminal_frozen）；非 managed/旧 C1/bridge 行保持原闭单语义。
            managed_terminal = and_(
                func.coalesce(ExecutionRequestModel.requested_mode, "") == "react_write",
                func.coalesce(ExecutionRequestModel.effective_mode, "") == "react_write",
                ExecutionRequestModel.status.in_(tuple(sorted(_TERMINAL_REQUEST_STATUSES))),
            )
            result = session.execute(
                update(ExecutionRequestModel)
                .where(
                    ExecutionRequestModel.user_id == user_id,
                    ExecutionRequestModel.request_id == request_id,
                    (
                        ExecutionRequestModel.lease_token.is_(None)
                        | (ExecutionRequestModel.lease_deadline < now)
                    ),
                    ~managed_terminal,
                )
                .values(
                    lease_worker_id=worker_id,
                    lease_token=token,
                    lease_epoch=func.coalesce(ExecutionRequestModel.lease_epoch, 0) + 1,
                    lease_deadline=deadline,
                    revision=ExecutionRequestModel.revision + 1,
                )
            )
            if result.rowcount != 1:
                row = self._get_request(session, user_id, request_id)
                if row is None:
                    raise _conflict("unknown_owner")
                if row.status in _TERMINAL_REQUEST_STATUSES:
                    raise ExecutionLedgerError("terminal_frozen")
                raise ExecutionLedgerError("lease_busy")
            snapshot = self._load_snapshot(session, user_id, request_id)
            if snapshot is None:
                raise _unavailable()
            return snapshot

        return self._guarded_write(work)

    def renew_lease(self, user_id, request_id, *, expected_revision: int, lease, ttl_seconds: int) -> dict:
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        _strict_int(expected_revision, "expected_revision", minimum=0)
        _strict_int(ttl_seconds, "ttl_seconds", minimum=_TTL_MINIMUM, maximum=_TTL_MAXIMUM)

        def work(session):
            row = self._get_request(session, user_id, request_id)
            if row is None:
                raise _conflict("unknown_owner")
            if row.status in _TERMINAL_REQUEST_STATUSES:
                raise ExecutionLedgerError("terminal_frozen")
            now = self._now(session)
            self._require_lease(row, lease, now)
            if row.revision != expected_revision:
                raise ExecutionLedgerError("revision_conflict")
            self._cas_request(
                session, row, expected_revision, lease,
                lease_deadline=now + timedelta(seconds=ttl_seconds),
            )
            return self._load_snapshot(session, user_id, request_id)

        return self._guarded_write(work)

    def release_lease(self, user_id, request_id, *, expected_revision: int, lease) -> dict:
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        _strict_int(expected_revision, "expected_revision", minimum=0)

        def work(session):
            row = self._get_request(session, user_id, request_id)
            if row is None:
                raise _conflict("unknown_owner")
            if row.status in _TERMINAL_REQUEST_STATUSES:
                raise ExecutionLedgerError("terminal_frozen")
            now = self._now(session)
            self._require_lease(row, lease, now)
            if row.revision != expected_revision:
                raise ExecutionLedgerError("revision_conflict")
            self._cas_request(
                session, row, expected_revision, lease,
                lease_worker_id=None, lease_token=None, lease_deadline=None,
            )
            return self._load_snapshot(session, user_id, request_id)

        return self._guarded_write(work)

    def freeze_task(self, user_id, request_id, task, *, expected_revision: int, lease) -> dict:
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        _strict_int(expected_revision, "expected_revision", minimum=0)
        values = self._valid_task(task)

        def work(session):
            row = self._get_request(session, user_id, request_id)
            if row is None:
                raise _conflict("unknown_owner")
            if row.status in _TERMINAL_REQUEST_STATUSES:
                raise ExecutionLedgerError("terminal_frozen")
            now = self._now(session)
            self._require_lease(row, lease, now)
            if row.revision != expected_revision:
                raise ExecutionLedgerError("revision_conflict")
            if values["source_start"] > values["source_end"]:
                raise _invalid("source_range")
            expected_digest = hashlib.sha256(
                row.source[values["source_start"]:values["source_end"]].encode("utf-8")
            ).hexdigest()
            if expected_digest != values["source_digest"]:
                raise _conflict("task_source_digest")
            existing = session.execute(
                select(ExecutionTaskModel).where(
                    ExecutionTaskModel.user_id == user_id,
                    ExecutionTaskModel.request_id == request_id,
                    ExecutionTaskModel.task_id == values["task_id"],
                )
            ).scalar_one_or_none()
            if existing is not None:
                if all(getattr(existing, field) == values[field] for field in (
                    "source_start", "source_end", "source_digest", "task_type",
                    "dependencies", "parameters", "required",
                )):
                    return self._build_snapshot(session, row)
                raise _conflict("frozen_task_changed")
            session.execute(
                insert(ExecutionTaskModel).values(
                    user_id=user_id,
                    request_id=request_id,
                    status="pending",
                    current_result=None,
                    **values,
                )
            )
            self._cas_request(session, row, expected_revision, lease)
            return self._load_snapshot(session, user_id, request_id)

        return self._guarded_write(work)

    def reserve_operation(
        self, user_id, request_id, *, task_id: str, position: int, kind: str,
        parameters: dict, source_digest: str, time_base: str,
        expected_revision: int, lease,
    ) -> dict:
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        _strict_str(task_id, "task_id")
        _strict_int(position, "position", minimum=0)
        _strict_str(kind, "kind")
        parameters_value = _finite_json(parameters, "parameters")
        if not isinstance(parameters_value, dict):
            raise _invalid("parameters_must_be_object")
        _strict_str(source_digest, "source_digest")
        _strict_str(time_base, "time_base")
        _strict_int(expected_revision, "expected_revision", minimum=0)

        def work(session):
            row = self._get_request(session, user_id, request_id)
            if row is None:
                raise _conflict("unknown_owner")
            if row.status in _TERMINAL_REQUEST_STATUSES:
                raise ExecutionLedgerError("terminal_frozen")
            now = self._now(session)
            self._require_lease(row, lease, now)
            if source_digest != row.source_digest or time_base != row.time_base:
                raise _conflict("frozen_source_or_time_base")
            task_row = session.execute(
                select(ExecutionTaskModel).where(
                    ExecutionTaskModel.user_id == user_id,
                    ExecutionTaskModel.request_id == request_id,
                    ExecutionTaskModel.task_id == task_id,
                )
            ).scalar_one_or_none()
            if task_row is None:
                raise _invalid("unknown_task")
            op_id, idempotency_key = _op_identity(user_id, request_id, task_id, position, kind)
            parameters_digest = _digest(parameters_value)
            existing = session.execute(
                select(ExecutionOperationModel).where(
                    ExecutionOperationModel.user_id == user_id,
                    ExecutionOperationModel.request_id == request_id,
                    ExecutionOperationModel.task_id == task_id,
                    ExecutionOperationModel.position == position,
                )
            ).scalar_one_or_none()
            if existing is not None:
                if (
                    existing.op_id == op_id
                    and existing.kind == kind
                    and existing.parameters == parameters_value
                    and existing.parameters_digest == parameters_digest
                    and existing.source_digest == source_digest
                    and existing.time_base == time_base
                ):
                    return self._build_snapshot(session, row)
                raise _conflict("operation_position_changed")
            session.execute(
                insert(ExecutionOperationModel).values(
                    user_id=user_id,
                    request_id=request_id,
                    op_id=op_id,
                    idempotency_key=idempotency_key,
                    task_id=task_id,
                    position=position,
                    kind=kind,
                    parameters=parameters_value,
                    parameters_digest=parameters_digest,
                    source_digest=source_digest,
                    time_base=time_base,
                    status="reserved",
                    confirmation_result=None,
                )
            )
            # 规范载荷文本：中文/实体保持真实可读（ensure_ascii=False），与同一
            # work(session) 事务/租约/最终 CAS 一体，不改变对象与 digest。
            session.execute(
                text(
                    "UPDATE agent_execution_operations SET parameters=:p "
                    "WHERE user_id=:u AND request_id=:r AND op_id=:o"
                ),
                {"p": json.dumps(parameters_value, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False),
                 "u": user_id, "r": request_id, "o": op_id},
            )
            self._cas_request(session, row, expected_revision, lease)
            return self._load_snapshot(session, user_id, request_id)

        return self._guarded_write(work)

    def mark_dispatch_intent(self, user_id, request_id, op_id: str, *, expected_revision: int, lease) -> dict:
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        _strict_str(op_id, "op_id")
        _strict_int(expected_revision, "expected_revision", minimum=0)

        def work(session):
            row = self._get_request(session, user_id, request_id)
            if row is None:
                raise _conflict("unknown_owner")
            if row.status in _TERMINAL_REQUEST_STATUSES:
                raise ExecutionLedgerError("terminal_frozen")
            now = self._now(session)
            self._require_lease(row, lease, now)
            if row.revision != expected_revision:
                raise ExecutionLedgerError("revision_conflict")
            operation = self._operation_row(session, user_id, request_id, op_id)
            if operation is None:
                raise _conflict("unknown_operation")
            if operation.status == "dispatch_intent":
                return self._build_snapshot(session, row)
            if operation.status != "reserved":
                raise ExecutionLedgerError("terminal_frozen")
            result = session.execute(
                update(ExecutionOperationModel)
                .where(
                    ExecutionOperationModel.user_id == user_id,
                    ExecutionOperationModel.request_id == request_id,
                    ExecutionOperationModel.op_id == op_id,
                    ExecutionOperationModel.status == "reserved",
                )
                .values(status="dispatch_intent")
            )
            if result.rowcount != 1:
                raise _conflict("operation_state_changed")
            self._cas_request(session, row, expected_revision, lease)
            return self._load_snapshot(session, user_id, request_id)

        return self._guarded_write(work)

    # ------------------------------------------- C2-1 session-scoped primitives

    def ledger_request_row(self, session, user_id, request_id, *, for_update: bool = False):
        statement = select(ExecutionRequestModel).where(
            ExecutionRequestModel.user_id == user_id,
            ExecutionRequestModel.request_id == request_id,
        )
        if for_update and session.get_bind().dialect.name == "postgresql":
            statement = statement.with_for_update()
        try:
            return session.execute(statement).scalar_one_or_none()
        except SQLAlchemyError:
            raise _unavailable() from None

    def ledger_statement_now(self, session) -> datetime:
        """Real statement clock: PostgreSQL clock_timestamp(), SQLite CURRENT_TIMESTAMP."""
        if session.get_bind().dialect.name == "postgresql":
            value = session.execute(text("SELECT clock_timestamp()")).scalar_one()
        else:
            value = session.execute(select(func.now())).scalar_one()
        if not isinstance(value, datetime):
            raise _unavailable()
        return _as_utc(value)

    def ledger_require_lease(self, row, lease, now: datetime) -> None:
        self._require_lease(row, lease, now)

    def ledger_cas_request(self, session, row, expected_revision: int, lease, *,
                           final_fence: bool = False, **values) -> None:
        self._cas_request(session, row, expected_revision, lease, final_fence=final_fence, **values)

    def ledger_operation_row(self, session, user_id: str, request_id: str, op_id: str):
        return self._operation_row(session, user_id, request_id, op_id)

    def ledger_task_row(self, session, user_id: str, request_id: str, task_id: str):
        return session.execute(
            select(ExecutionTaskModel).where(
                ExecutionTaskModel.user_id == user_id,
                ExecutionTaskModel.request_id == request_id,
                ExecutionTaskModel.task_id == task_id,
            )
        ).scalar_one_or_none()

    def ledger_event_row(self, session, user_id: str, request_id: str, event_key: str):
        return session.execute(
            select(ExecutionEventModel).where(
                ExecutionEventModel.user_id == user_id,
                ExecutionEventModel.request_id == request_id,
                ExecutionEventModel.event_key == event_key,
            )
        ).scalar_one_or_none()

    def ledger_append_event(self, session, user_id, request_id, event, *, checkpoint,
                            expected_revision: int, lease, final_fence: bool = False) -> dict:
        """Validated ordinary C1 append inside the caller's transaction; never
        commits. The ordinary entry has no caller authorization switch and
        always rejects reserved bridge keys."""
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        _strict_int(expected_revision, "expected_revision", minimum=0)
        event_values = self._valid_event(event)
        checkpoint_values = self._valid_checkpoint(checkpoint)
        return self._append_event_core(session, user_id, request_id, event_values, checkpoint_values,
                                       expected_revision, lease, final_fence=final_fence)

    def ledger_append_reserved_event(self, session, user_id, request_id, event, *, checkpoint,
                                     expected_revision: int, lease, capability,
                                     final_fence: bool = False) -> dict:
        """Workout bridge-dedicated path. Requires the local, non-serializable
        capability bound to the real workout bridge instance by object identity;
        after that the training origin protocol/generation is generated by code,
        never from input. Only the c2-1 workout namespace is writable here."""
        self._require_bound_capability(capability)
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        _strict_int(expected_revision, "expected_revision", minimum=0)
        event_values = self._valid_event(event)
        checkpoint_values = self._valid_checkpoint(checkpoint)
        if not event_values["event_key"].startswith(_WORKOUT_EVENT_PREFIXES):
            raise _invalid("reserved_key_required")
        return self._append_event_core(session, user_id, request_id, event_values, checkpoint_values,
                                       expected_revision, lease, final_fence=final_fence,
                                       reserved_origin=_bridge_origin_metadata())

    def ledger_append_nutrition_event(self, session, user_id, request_id, event, *, checkpoint,
                                      expected_revision: int, lease, capability,
                                      pending_deadline=None, final_fence: bool = False) -> dict:
        """Nutrition bridge-dedicated path. Requires the local, non-serializable
        capability bound to the real nutrition bridge instance by object
        identity; after that the nutrition origin protocol/generation is
        generated by code, never from input. Only the c2-2 nutrition namespace
        is writable here; ``pending_deadline`` (derived by the bridge from the
        validated protected draft) joins the final CAS SQL."""
        self._require_bound_nutrition_capability(capability)
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        _strict_int(expected_revision, "expected_revision", minimum=0)
        event_values = self._valid_event(event)
        checkpoint_values = self._valid_checkpoint(checkpoint)
        if not event_values["event_key"].startswith(_NUTRITION_EVENT_PREFIXES):
            raise _invalid("reserved_key_required")
        return self._append_event_core(session, user_id, request_id, event_values, checkpoint_values,
                                       expected_revision, lease, final_fence=final_fence,
                                       reserved_origin=_nutrition_origin_metadata(),
                                       origin_key=_NUTRITION_ORIGIN_KEY,
                                       pending_deadline=pending_deadline)

    def ledger_append_runtime_event(self, session, user_id, request_id, event, *, checkpoint,
                                    expected_revision: int, lease, capability,
                                    final_fence: bool = False) -> dict:
        """C2-3 runtime-family dedicated path. Requires the local capability bound
        to the real checkpoint-store owner by object identity; the runtime origin
        is stamped by code, never from input. Only the c2-3-runtime namespace is
        writable here, and the ordinary public/session paths never open it."""
        self._require_bound_runtime_capability(capability)
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        _strict_int(expected_revision, "expected_revision", minimum=0)
        event_values = self._valid_event(event)
        checkpoint_values = self._valid_checkpoint(checkpoint)
        if not event_values["event_key"].startswith(_RUNTIME_EVENT_PREFIXES):
            raise _invalid("reserved_key_required")
        return self._append_event_core(session, user_id, request_id, event_values, checkpoint_values,
                                       expected_revision, lease, final_fence=final_fence,
                                       reserved_origin=_runtime_origin_metadata(),
                                       origin_key=_RUNTIME_ORIGIN_KEY)

    def ledger_append_correction_event(self, session, user_id, request_id, event, *,
                                       checkpoint, expected_revision, lease, capability):
        bound = getattr(self, "_correction_capability", None)
        owner = getattr(self, "_correction_owner", None)
        if bound is None or capability is not bound or owner is None or owner() is None:
            raise _conflict("reserved_capability_required")
        values = self._valid_event(event)
        if not values["event_key"].startswith(_CORRECTION_EVENT_PREFIX):
            raise _invalid("reserved_key_required")
        return self._append_event_core(session, user_id, request_id, values,
            self._valid_checkpoint(checkpoint), expected_revision, lease, final_fence=True,
            reserved_origin=dict(_CORRECTION_ORIGIN), origin_key=_CORRECTION_ORIGIN_KEY)

    def _require_bound_capability(self, capability) -> None:
        bound = getattr(self, "_reserved_event_capability", None)
        owner = getattr(self, "_reserved_event_capability_owner", None)
        if bound is None or capability is not bound or owner is None or owner() is None:
            raise _conflict("reserved_capability_required")

    def _require_bound_nutrition_capability(self, capability) -> None:
        bound = getattr(self, "_nutrition_event_capability", None)
        owner = getattr(self, "_nutrition_event_capability_owner", None)
        if bound is None or capability is not bound or owner is None or owner() is None:
            raise _conflict("reserved_capability_required")

    def _require_bound_runtime_capability(self, capability) -> None:
        bound = getattr(self, "_runtime_event_capability", None)
        owner = getattr(self, "_runtime_event_capability_owner", None)
        if bound is None or capability is not bound or owner is None or owner() is None:
            raise _conflict("reserved_capability_required")

    def _append_event_core(self, session, user_id, request_id, event_values, checkpoint_values,
                           expected_revision, lease, *, final_fence: bool = False,
                           reserved_origin: dict | None = None, origin_key: str = _BRIDGE_ORIGIN_KEY,
                           pending_deadline=None) -> dict:
        if reserved_origin is None:
            if event_values["event_key"].startswith(_RESERVED_EVENT_PREFIXES):
                raise _conflict("reserved_event_key")
            public_observation = event_values["observation"]
        else:
            observation = event_values["observation"]
            if not isinstance(observation, dict):
                raise _invalid("observation_must_be_object")
            stamped = {key: value for key, value in observation.items() if key != origin_key}
            stamped[origin_key] = copy.deepcopy(reserved_origin)
            event_values["observation"] = stamped
            public_observation = {key: value for key, value in stamped.items() if key != origin_key}
        row = self._get_request(session, user_id, request_id)
        if row is None:
            raise _conflict("unknown_owner")
        if row.status in _TERMINAL_REQUEST_STATUSES:
            raise ExecutionLedgerError("terminal_frozen")
        if not final_fence:
            now = self._now(session)
            self._require_lease(row, lease, now)
        else:
            # The final-fence append is enforced by the real SQL CAS: the
            # lease deadline (and the pending deadline when provided) are
            # compared against the statement clock inside the WHERE, so the
            # Python pre-check must not short-circuit that evidence.
            self._valid_lease(lease)

        existing = session.execute(
            select(ExecutionEventModel).where(
                ExecutionEventModel.user_id == user_id,
                ExecutionEventModel.request_id == request_id,
                ExecutionEventModel.event_key == event_values["event_key"],
            )
        ).scalar_one_or_none()
        if existing is not None:
            same_payload = all(
                getattr(existing, field) == event_values[field] for field in _EVENT_PAYLOAD_FIELDS
            )
            if same_payload and _persisted_checkpoint(existing.checkpoint) == checkpoint_values:
                return {"event_id": existing.event_id, "sequence": existing.sequence,
                        "revision": row.revision, "created": False}
            raise _conflict("event_key_reused")

        current_checkpoint = _persisted_checkpoint(row.checkpoint)
        next_sequence = current_checkpoint["event_sequence"] + 1
        if checkpoint_values["budget_scope_id"] != current_checkpoint["budget_scope_id"]:
            raise _conflict("checkpoint_scope_changed")
        if checkpoint_values["event_sequence"] != next_sequence:
            raise _invalid("checkpoint_sequence_mismatch")
        if (
            checkpoint_values["supervisor_turns"] < current_checkpoint["supervisor_turns"]
            or checkpoint_values["tool_steps"] < current_checkpoint["tool_steps"]
        ):
            raise _invalid("checkpoint_counter_regression")

        task_row = session.execute(
            select(ExecutionTaskModel).where(
                ExecutionTaskModel.user_id == user_id,
                ExecutionTaskModel.request_id == request_id,
                ExecutionTaskModel.task_id == event_values["task_id"],
            )
        ).scalar_one_or_none()
        if task_row is None:
            raise _invalid("unknown_task")

        operation = None
        if event_values["op_id"] is not None:
            operation = self._operation_row(session, user_id, request_id, event_values["op_id"])
            if operation is None:
                raise _invalid("unknown_operation")

        self._validate_operation_event(event_values, operation)

        event_id = _event_identity(user_id, request_id, event_values["event_key"])
        session.execute(
            insert(ExecutionEventModel).values(
                user_id=user_id,
                request_id=request_id,
                event_id=event_id,
                sequence=next_sequence,
                checkpoint=checkpoint_values,
                **event_values,
            )
        )
        if operation is not None and event_values["operation_state"] is not None:
            session.execute(
                update(ExecutionOperationModel)
                .where(
                    ExecutionOperationModel.user_id == user_id,
                    ExecutionOperationModel.request_id == request_id,
                    ExecutionOperationModel.op_id == event_values["op_id"],
                )
                .values(
                    status=event_values["operation_state"],
                    confirmation_result=(
                        copy.deepcopy(public_observation)
                        if event_values["operation_state"] == "committed"
                        else operation.confirmation_result
                    ),
                )
            )
        session.execute(
            update(ExecutionTaskModel)
            .where(
                ExecutionTaskModel.user_id == user_id,
                ExecutionTaskModel.request_id == request_id,
                ExecutionTaskModel.task_id == event_values["task_id"],
            )
            .values(
                status=event_values["task_status"],
                current_result=copy.deepcopy(public_observation),
            )
        )
        self._cas_request(session, row, expected_revision, lease, final_fence=final_fence,
                          pending_deadline=pending_deadline, checkpoint=checkpoint_values)
        return {"event_id": event_id, "sequence": next_sequence,
                "revision": expected_revision + 1, "created": True}

    def append_observation(self, user_id, request_id, event, *, checkpoint, expected_revision: int, lease) -> dict:
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        _strict_int(expected_revision, "expected_revision", minimum=0)
        event_values = self._valid_event(event)
        checkpoint_values = self._valid_checkpoint(checkpoint)

        def work(session):
            self._append_event_core(session, user_id, request_id, event_values, checkpoint_values,
                                    expected_revision, lease)
            return self._load_snapshot(session, user_id, request_id)

        return self._guarded_write(work)

    def _operation_row(self, session, user_id: str, request_id: str, op_id: str):
        return session.execute(
            select(ExecutionOperationModel).where(
                ExecutionOperationModel.user_id == user_id,
                ExecutionOperationModel.request_id == request_id,
                ExecutionOperationModel.op_id == op_id,
            )
        ).scalar_one_or_none()

    @staticmethod
    def _validate_operation_event(event_values: dict, operation) -> None:
        state = event_values["operation_state"]
        if state is None:
            return
        if operation is None:
            raise _invalid("operation_state_without_operation")
        observation = event_values["observation"]
        if state == "committed":
            if operation.status not in {"dispatch_intent", "commit_unknown"}:
                raise _invalid("commit_requires_dispatch_intent")
            if event_values["confirmed"] is not True:
                raise _invalid("commit_requires_confirmation")
            if observation.get("commit_state") != "committed" or observation.get("status") != "completed":
                raise _invalid("commit_requires_completed_observation")
            receipts = observation.get("receipt_refs")
            if not isinstance(receipts, list) or not receipts:
                raise _invalid("commit_requires_receipts")
        elif state == "commit_unknown":
            if operation.status not in {"dispatch_intent", "commit_unknown"}:
                raise _invalid("unknown_requires_dispatch_intent")
            if event_values["confirmed"] is True:
                raise _invalid("unknown_cannot_be_confirmed")
        elif state == "rejected":
            if operation.status != "reserved":
                raise _invalid("reject_requires_reserved")
            if event_values["confirmed"] is True:
                raise _invalid("reject_cannot_be_confirmed")

    def save_checkpoint(self, user_id, request_id, checkpoint, *, expected_revision: int, lease) -> dict:
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        _strict_int(expected_revision, "expected_revision", minimum=0)
        checkpoint_values = self._valid_checkpoint(checkpoint)

        def work(session):
            row = self._get_request(session, user_id, request_id)
            if row is None:
                raise _conflict("unknown_owner")
            if row.status in _TERMINAL_REQUEST_STATUSES:
                raise ExecutionLedgerError("terminal_frozen")
            now = self._now(session)
            self._require_lease(row, lease, now)
            if row.revision != expected_revision:
                raise ExecutionLedgerError("revision_conflict")
            current_checkpoint = _persisted_checkpoint(row.checkpoint)
            if checkpoint_values["budget_scope_id"] != current_checkpoint["budget_scope_id"]:
                raise _conflict("checkpoint_scope_changed")
            if checkpoint_values["event_sequence"] != current_checkpoint["event_sequence"]:
                raise _invalid("checkpoint_sequence_mismatch")
            if (
                checkpoint_values["supervisor_turns"] < current_checkpoint["supervisor_turns"]
                or checkpoint_values["tool_steps"] < current_checkpoint["tool_steps"]
            ):
                raise _invalid("checkpoint_counter_regression")
            self._cas_request(session, row, expected_revision, lease, checkpoint=checkpoint_values)
            return self._load_snapshot(session, user_id, request_id)

        return self._guarded_write(work)

    def finish_request(
        self, user_id, request_id, *, status: str, response: dict,
        termination_reason: str, expected_revision: int, lease,
        journal_event=None, journal_checkpoint=None, capability=None,
    ) -> dict:
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        _strict_int(expected_revision, "expected_revision", minimum=0)
        if status not in _FINISH_STATUSES:
            raise _invalid("finish_status_unsupported")
        _strict_str(termination_reason, "termination_reason")
        response_value = _finite_json(response, "response")
        if not isinstance(response_value, dict):
            raise _invalid("response_must_be_object")

        def work(session):
            row = self._get_request(session, user_id, request_id)
            if row is None:
                raise _conflict("unknown_owner")
            if row.status in _TERMINAL_REQUEST_STATUSES:
                stored = copy.deepcopy(row.response)
                if row.status == status and stored is not None and _canonical(stored) == _canonical(response_value):
                    return self._build_snapshot(session, row)
                raise _conflict("terminal_response_changed")
            now = self._now(session)
            self._require_lease(row, lease, now)
            if row.revision != expected_revision:
                raise ExecutionLedgerError("revision_conflict")
            if status == "completed":
                self._check_completion(session, row)
            revision = expected_revision
            if journal_event is not None:
                # 规范 final_reply journal 与终态响应同一事务：避免 journal 成功而
                # 响应失败（或反之）的部分状态；失败整体回滚。
                appended = self.ledger_append_runtime_event(
                    session, user_id, request_id, journal_event,
                    checkpoint=journal_checkpoint, expected_revision=revision,
                    lease=lease, capability=capability)
                revision = int(appended["revision"])
            self._cas_request(
                session, row, revision, lease,
                status=status, response=response_value,
                lease_worker_id=None, lease_token=None, lease_deadline=None,
            )
            return self._load_snapshot(session, user_id, request_id)

        return self._guarded_write(work)

    def freeze_response(
        self, user_id, request_id, *, response: dict,
        expected_revision: int, lease,
        journal_event=None, journal_checkpoint=None, capability=None,
    ) -> dict:
        """非终态诚实响应冻结：保持请求状态与未处理目标不变，只以真实 CAS+lease
        写入 response（不写终态、不做完成检查），用于确认消费后原请求仍有未处理
        目标时的保守投影；绝不冒充 completed/limited。可选 journal_event 与响应
        在同一事务写入（规范 final_reply 证明，避免部分状态）。"""
        _strict_str(user_id, "user_id")
        _strict_str(request_id, "request_id")
        _strict_int(expected_revision, "expected_revision", minimum=0)
        response_value = _finite_json(response, "response")
        if not isinstance(response_value, dict):
            raise _invalid("response_must_be_object")

        def work(session):
            row = self._get_request(session, user_id, request_id)
            if row is None:
                raise _conflict("unknown_owner")
            if row.status in _TERMINAL_REQUEST_STATUSES:
                raise _conflict("terminal_frozen")
            now = self._now(session)
            self._require_lease(row, lease, now)
            if row.revision != expected_revision:
                raise ExecutionLedgerError("revision_conflict")
            revision = expected_revision
            if journal_event is not None:
                appended = self.ledger_append_runtime_event(
                    session, user_id, request_id, journal_event,
                    checkpoint=journal_checkpoint, expected_revision=revision,
                    lease=lease, capability=capability)
                revision = int(appended["revision"])
            self._cas_request(
                session, row, revision, lease,
                status=row.status, response=response_value,
                lease_worker_id=None, lease_token=None, lease_deadline=None,
            )
            return self._load_snapshot(session, user_id, request_id)

        return self._guarded_write(work)

    def _check_completion(self, session, row) -> None:
        tasks = session.execute(
            select(ExecutionTaskModel).where(
                ExecutionTaskModel.user_id == row.user_id,
                ExecutionTaskModel.request_id == row.request_id,
            )
        ).scalars().all()
        operations = session.execute(
            select(ExecutionOperationModel).where(
                ExecutionOperationModel.user_id == row.user_id,
                ExecutionOperationModel.request_id == row.request_id,
            )
        ).scalars().all()
        if row.request_kind == "readonly_no_tasks":
            if tasks or operations:
                raise ExecutionLedgerError("completion_blocked", "readonly_no_tasks_with_state")
            return
        if not tasks:
            raise ExecutionLedgerError("completion_blocked", "no_tasks")
        for task in tasks:
            if task.required in (True, "true", 1):
                if task.status != "completed" or task.current_result is None:
                    raise ExecutionLedgerError("completion_blocked", "required_task_unfinished")
        for operation in operations:
            if operation.status in _UNFINISHED_OPERATION_STATUSES:
                raise ExecutionLedgerError("completion_blocked", "unfinished_operation")

    def load_recovery_state(self, user_id: str, request_id: str) -> dict | None:
        snapshot = self.load_request(user_id, request_id)
        if snapshot is None:
            return None
        recovery = RecoverySnapshot(snapshot)
        recovery["pending_verification"] = [
            copy.deepcopy(operation)
            for operation in dict.__getitem__(snapshot, "operations")
            if operation["status"] in {"dispatch_intent", "commit_unknown"}
        ]
        recovery["replay_allowed"] = False
        return recovery


def _persisted_checkpoint(value: Any) -> dict:
    if not isinstance(value, dict) or set(value) != set(_CHECKPOINT_FIELDS):
        raise _corrupt("checkpoint_shape")
    result = {}
    for field in ("supervisor_turns", "tool_steps", "event_sequence"):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise _corrupt("checkpoint_integer_shape")
        result[field] = item
    result["budget_scope_id"] = value["budget_scope_id"]
    if not isinstance(result["budget_scope_id"], str) or not result["budget_scope_id"]:
        raise _corrupt("checkpoint_scope_shape")
    return result


def _bridge_origin_metadata() -> dict:
    """Fresh copy of the fixed, non-secret origin protocol/generation."""
    return {"protocol": _BRIDGE_ORIGIN_PROTOCOL, "generation": _BRIDGE_ORIGIN_GENERATION}


def _nutrition_origin_metadata() -> dict:
    """Fresh copy of the nutrition family origin protocol/generation."""
    return {"protocol": _NUTRITION_ORIGIN_PROTOCOL, "generation": _NUTRITION_ORIGIN_GENERATION}


def _runtime_origin_metadata() -> dict:
    """Fresh copy of the runtime family origin protocol/generation."""
    return {"protocol": _RUNTIME_ORIGIN_PROTOCOL, "generation": _RUNTIME_ORIGIN_GENERATION}


def _strip_runtime_origin(observation: dict) -> dict:
    """Public event projections never expose the private runtime origin key; the
    bridge families keep their own (unchanged) behavior."""
    observation.pop(_RUNTIME_ORIGIN_KEY, None)
    return observation


class _ReservedEventCapability:
    """Local, non-serializable capability bound to one bridge instance.

    This is a component-local interface boundary, not a defense against
    arbitrary Python access to private state or raw SQL. It is never persisted,
    logged or copied into events.
    """
    __slots__ = ("__weakref__",)

    def __reduce__(self):
        raise TypeError("reserved event capability is not serializable")


def _bind_correction_reserved_capability(store, bridge) -> _ReservedEventCapability:
    from app.services.agent.workout_correction import WorkoutCorrectionBridge
    if not isinstance(store, ExecutionLedgerStore) or not isinstance(bridge, WorkoutCorrectionBridge):
        raise TypeError("correction capability requires its real bridge")
    capability = _ReservedEventCapability()
    store._correction_capability = capability
    store._correction_owner = weakref.ref(bridge)
    return capability


def _bind_bridge_reserved_capability(store, bridge) -> _ReservedEventCapability:
    """Internal bridge-construction binding; not a public store method."""
    if not isinstance(store, ExecutionLedgerStore) or bridge is None:
        raise TypeError("reserved capability binding requires a real store and bridge instance")
    capability = _ReservedEventCapability()
    store._reserved_event_capability = capability
    store._reserved_event_capability_owner = weakref.ref(bridge)
    return capability


def _bind_nutrition_reserved_capability(store, bridge) -> _ReservedEventCapability:
    """Internal nutrition-bridge-construction binding; not a public store method.
    Separate family: a workout capability never opens the nutrition namespace
    and vice versa."""
    if not isinstance(store, ExecutionLedgerStore) or bridge is None:
        raise TypeError("reserved capability binding requires a real store and bridge instance")
    capability = _ReservedEventCapability()
    store._nutrition_event_capability = capability
    store._nutrition_event_capability_owner = weakref.ref(bridge)
    return capability


def _bind_runtime_reserved_capability(store, owner) -> _ReservedEventCapability:
    """Internal runtime-construction binding; not a public store method.

    The owner must be the real checkpoint-store object that owns this ledger
    store by object identity (``owner._ledger is store``); a fake owner or a
    different store's owner is rejected before any capability exists. Separate
    family: runtime never opens the two bridge namespaces and vice versa.
    """
    if not isinstance(store, ExecutionLedgerStore) or owner is None:
        raise TypeError("runtime capability binding requires a real store and owner instance")
    if getattr(owner, "_ledger", None) is not store:
        raise _conflict("identity_conflict")
    capability = _ReservedEventCapability()
    store._runtime_event_capability = capability
    store._runtime_event_capability_owner = weakref.ref(owner)
    return capability
