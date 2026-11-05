"""C2-1 workout record/ledger transaction bridge.

Synchronous, session-factory scoped. The bridge matches frozen C1 operations
with real workout records and mirrors: the claim event commits alone, then the
record+mirror+confirmation commit in one locked business transaction whose last
write re-validates the lease against the database statement clock.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re as _re
from datetime import date as date_type

from sqlalchemy.exc import IntegrityError

from app.db import repositories
from app.db.storage import IdempotencyConflict
from app.services.agent import execution_ledger_store as ledger_api
from app.services.agent.execution_ledger_store import ExecutionLedgerError
from app.services.agent.react_models import CommitState, DomainObservation, SourceScope, ToolStatus
from app.services.agent.react_tools import validate_source_ref
from app.services.security.write_guard import source_safety_blocked

_BRIDGE_VERSION = 1
_CLAIM_PREFIX = "c2-1-claim:v1:"
_COMMIT_PREFIX = "c2-1-commit:v1:"
_PROOF_TYPE = "workout_db_v1"
_RECORD_KEYS = frozenset({"exercise", "sets", "reps", "weight_kg", "note", "difficulty", "completed", "date", "set_details"})
_SET_KEYS = frozenset({"weight_kg", "reps", "difficulty", "completed", "note"})
_DATE_RE = _re.compile(r"\d{4}-\d{2}-\d{2}$")
_TERMINAL = frozenset({"limited", "failed", "completed"})


def _origin_matches(observation) -> bool:
    """The persisted declaration must carry the code-stamped origin protocol and
    the current fixed generation; a caller-declared marker alone is never proof."""
    if not isinstance(observation, dict):
        return False
    origin = observation.get(ledger_api._BRIDGE_ORIGIN_KEY)
    return (isinstance(origin, dict)
            and origin.get("protocol") == ledger_api._BRIDGE_ORIGIN_PROTOCOL
            and origin.get("generation") == ledger_api._BRIDGE_ORIGIN_GENERATION)


class _Refuse(Exception):
    def __init__(self, outcome, reason, detail=None, dispatched=False):
        self.outcome = outcome
        self.reason = reason
        self.detail = detail
        self.dispatched = dispatched
        super().__init__(outcome)


class WorkoutWriteBridge:
    def __init__(self, factory):
        self._factory = factory
        self._ledger = ledger_api.ExecutionLedgerStore(factory)
        # Local, non-serializable capability bound to this real bridge instance;
        # it is the only way to reach the reserved append path and is never
        # derived from a flag, string, dict or event field.
        self._reserved_capability = ledger_api._bind_bridge_reserved_capability(self._ledger, self)

    # ------------------------------------------------------------------ public

    def commit_workout_operation(self, user_id, request_id, op_id, *, expected_revision, lease,
                                 tool_call_id, batch_id) -> dict:
        try:
            self._validate_call_inputs(user_id, request_id, op_id, expected_revision, lease, tool_call_id, batch_id)
            return self._commit(user_id, request_id, op_id, expected_revision, lease, tool_call_id, batch_id)
        except _Refuse as refuse:
            return self._limited(refuse, user_id, request_id, op_id)
        except ExecutionLedgerError as exc:
            return self._limited(self._ledger_refusal(exc), user_id, request_id, op_id)
        except Exception:
            return self._limited(_Refuse("commit_unknown", "commit_unknown", "internal_error"), user_id, request_id, op_id)

    def recover_workout_operation(self, user_id, request_id, op_id) -> dict:
        try:
            for name, value in (("user_id", user_id), ("request_id", request_id), ("op_id", op_id)):
                if not isinstance(value, str) or not value:
                    raise _Refuse("rejected", "schema", f"{name}_invalid")
            return self._read_result(user_id, request_id, op_id, dispatched=False, found_outcome="recovered")
        except _Refuse as refuse:
            return self._limited(refuse, user_id, request_id, op_id)
        except Exception:
            return self._limited(_Refuse("commit_unknown", "commit_unknown", "recovery_read_failed"),
                                 user_id, request_id, op_id)

    # ------------------------------------------------------------- transaction

    def _commit(self, user_id, request_id, op_id, expected_revision, lease, tool_call_id, batch_id) -> dict:
        # Transaction 1: freeze validation + the single winning claim.
        session = self._factory()
        try:
            row = self._ledger.ledger_request_row(session, user_id, request_id, for_update=True)
            if row is None:
                raise _Refuse("rejected", "identity", "unknown_owner")
            op = self._ledger.ledger_operation_row(session, user_id, request_id, op_id)
            if op is None:
                raise _Refuse("rejected", "identity", "unknown_operation")
            if op.kind != "workout":
                raise _Refuse("rejected", "identity", "wrong_kind")
            confirmation = self._ledger.ledger_event_row(session, user_id, request_id, _COMMIT_PREFIX + op_id)
            if confirmation is not None:
                return self._replay(session, user_id, request_id, op, confirmation)
            self._ledger.ledger_require_lease(row, lease, self._ledger.ledger_statement_now(session))
            if row.revision != expected_revision:
                raise _Refuse("rejected", "revision", "stale_revision")
            if row.status in _TERMINAL:
                raise _Refuse("rejected", "identity", "terminal_request")
            if op.status in {"committed", "commit_unknown"}:
                raise _Refuse("commit_unknown", "operation_unverified", "operation_state_untrusted")
            if op.status != "dispatch_intent":
                raise _Refuse("rejected", "identity", "operation_not_dispatched")
            if self._ledger.ledger_event_row(session, user_id, request_id, _CLAIM_PREFIX + op_id) is not None:
                raise _Refuse("commit_unknown", "operation_unverified", "claim_without_confirmation")
            task = self._ledger.ledger_task_row(session, user_id, request_id, op.task_id)
            if task is None:
                raise _Refuse("rejected", "identity", "unknown_task")
            self._validate_source_and_task(row, task)
            self._frozen_record(task, op)
            foreign = repositories.find_workout_by_key_in_session(session, user_id, op.idempotency_key)
            if foreign is not None:
                raise _Refuse("commit_unknown", "operation_unverified", "existing_record_without_bridge_proof")
            claim = ledger_api.EventInput(
                event_key=_CLAIM_PREFIX + op_id, tool_call_id=tool_call_id, batch_id=batch_id,
                task_id=op.task_id, op_id=op_id, phase="write_claim",
                observation={"claim_state": "dispatched", "op_id": op_id, "position": op.position,
                             "idempotency_key": op.idempotency_key, "tool_call_id": tool_call_id},
                task_status="running", operation_state=None, confirmed=False,
            )
            try:
                self._ledger.ledger_append_reserved_event(session, user_id, request_id, claim,
                                                          checkpoint=self._next_checkpoint(row),
                                                          expected_revision=expected_revision, lease=lease,
                                                          capability=self._reserved_capability)
            except IntegrityError:
                self._rollback_quiet(session)
                raise _Refuse("commit_unknown", "operation_unverified", "claim_race")
            except ExecutionLedgerError as exc:
                self._rollback_quiet(session)
                raise self._ledger_refusal(exc)
            try:
                session.commit()
            except Exception:
                self._rollback_quiet(session)
                raise _Refuse("commit_unknown", "commit_unknown", "claim_commit_unknown")
        finally:
            self._close_quiet(session)
        # Transaction 2: the one business transaction.
        return self._business_transaction(user_id, request_id, op_id, lease, tool_call_id, batch_id)

    def _business_transaction(self, user_id, request_id, op_id, lease, tool_call_id, batch_id) -> dict:
        session = self._factory()
        ctx = {"dispatched": False}
        try:
            row = self._ledger.ledger_request_row(session, user_id, request_id, for_update=True)
            if row is None:
                raise _Refuse("commit_unknown", "commit_unknown", "request_vanished")
            op = self._ledger.ledger_operation_row(session, user_id, request_id, op_id)
            if op is None or op.status != "dispatch_intent":
                raise _Refuse("commit_unknown", "commit_unknown", "operation_changed")
            if self._ledger.ledger_event_row(session, user_id, request_id, _CLAIM_PREFIX + op_id) is None:
                raise _Refuse("commit_unknown", "commit_unknown", "claim_missing")
            self._ledger.ledger_require_lease(row, lease, self._ledger.ledger_statement_now(session))
            task = self._ledger.ledger_task_row(session, user_id, request_id, op.task_id)
            if task is None:
                raise _Refuse("commit_unknown", "commit_unknown", "task_vanished")
            record = self._frozen_record(task, op)
            ctx["dispatched"] = True
            try:
                created = repositories.create_workout_in_session(
                    session, user_id,
                    exercise=record["exercise"], sets=record["sets"], reps=record["reps"],
                    weight_kg=record["weight_kg"], note=record["note"], difficulty=record["difficulty"],
                    completed=record["completed"], date=date_type.fromisoformat(record["date"]),
                    set_details=record["set_details"], idempotency_key=op.idempotency_key)
            except IdempotencyConflict:
                self._rollback_quiet(session)
                raise _Refuse("commit_unknown", "operation_unverified", "existing_unverified_record", True)
            if (created.get("created") is not True or created.get("record_id") is None
                    or created.get("session_id") is None or created.get("exercise_id") is None):
                # A matched existing row (same key/parameters) is not this
                # bridge's proof: roll back any mirror refresh from this
                # transaction, keep the claim and the foreign content.
                self._rollback_quiet(session)
                raise _Refuse("commit_unknown", "operation_unverified", "existing_record_unverified", True)
            proof = {
                "proof_type": _PROOF_TYPE,
                "record_id": int(created["record_id"]),
                "session_id": int(created["session_id"]),
                "exercise_id": int(created["exercise_id"]),
                "set_ids": [int(item) for item in created["set_ids"]],
                "idempotency_key": op.idempotency_key,
                "position": op.position,
            }
            receipts = ["workout_db_v1", f"workout_record:{proof['record_id']}",
                        f"workout_session:{proof['session_id']}", f"workout_exercise:{proof['exercise_id']}"]
            receipts += [f"workout_set:{item}" for item in proof["set_ids"]]
            observation = DomainObservation(
                status=ToolStatus.COMPLETED, task_id=op.task_id, commit_state=CommitState.COMMITTED,
                result=proof, receipt_refs=receipts).to_dict()
            confirmation = ledger_api.EventInput(
                event_key=_COMMIT_PREFIX + op_id, tool_call_id=tool_call_id, batch_id=batch_id,
                task_id=op.task_id, op_id=op_id, phase="confirmation", observation=observation,
                task_status="completed", operation_state="committed", confirmed=True,
            )
            self._ledger.ledger_require_lease(row, lease, self._ledger.ledger_statement_now(session))
            self._ledger.ledger_append_reserved_event(session, user_id, request_id, confirmation,
                                                      checkpoint=self._next_checkpoint(row),
                                                      expected_revision=row.revision, lease=lease,
                                                      capability=self._reserved_capability, final_fence=True)
            try:
                session.commit()
            except Exception:
                self._rollback_quiet(session)
                return self._resolve_after_commit_fault(user_id, request_id, op_id)
            snapshot = self._safe_snapshot(user_id, request_id)
            return self._result("committed", observation=observation, dispatched=True, ledger_snapshot=snapshot,
                                user_id=user_id, request_id=request_id, op_id=op_id,
                                task_id=op.task_id, position=op.position)
        except _Refuse as refuse:
            refuse.dispatched = refuse.dispatched or ctx["dispatched"]
            raise
        except ExecutionLedgerError as exc:
            self._rollback_quiet(session)
            raise self._ledger_refusal(exc, ctx["dispatched"])
        except Exception:
            self._rollback_quiet(session)
            raise _Refuse("not_committed", "storage", "business_transaction_failed", ctx["dispatched"])
        finally:
            self._close_quiet(session)

    def _resolve_after_commit_fault(self, user_id, request_id, op_id) -> dict:
        try:
            return self._read_result(user_id, request_id, op_id, dispatched=True, found_outcome="recovered")
        except Exception:
            return self._limited(_Refuse("commit_unknown", "commit_unknown", "post_commit_read_failed", True),
                                 user_id, request_id, op_id)

    # ------------------------------------------------------------ read results

    def _read_result(self, user_id, request_id, op_id, *, dispatched, found_outcome) -> dict:
        def limited(outcome, reason, detail):
            return self._limited(_Refuse(outcome, reason, detail, dispatched), user_id, request_id, op_id)

        session = self._factory()
        try:
            op = self._ledger.ledger_operation_row(session, user_id, request_id, op_id)
            if op is None:
                return limited("rejected", "identity", "unknown_operation")
            if op.kind != "workout":
                return limited("rejected", "identity", "wrong_kind")
            confirmation = self._ledger.ledger_event_row(session, user_id, request_id, _COMMIT_PREFIX + op_id)
            if confirmation is not None:
                if self._trusted_proof(session, confirmation, op) is not None:
                    observation = DomainObservation.from_dict(confirmation.observation).to_dict()
                    return self._result(found_outcome, observation=observation, dispatched=dispatched,
                                        ledger_snapshot=self._safe_snapshot(user_id, request_id),
                                        user_id=user_id, request_id=request_id, op_id=op.op_id,
                                        task_id=op.task_id, position=op.position)
                return limited("commit_unknown", "operation_unverified", "confirmation_untrusted")
            if op.status in {"committed", "commit_unknown"}:
                return limited("commit_unknown", "operation_unverified", "operation_state_untrusted")
            if self._ledger.ledger_event_row(session, user_id, request_id, _CLAIM_PREFIX + op_id) is not None:
                return limited("commit_unknown", "operation_unverified", "claim_without_confirmation")
            if repositories.find_workout_by_key_in_session(session, user_id, op.idempotency_key) is not None:
                return limited("commit_unknown", "operation_unverified", "existing_record_without_bridge_proof")
            return limited("not_committed", "operation_unverified", "no_bridge_attempt")
        finally:
            self._close_quiet(session)

    def _replay(self, session, user_id, request_id, op, confirmation) -> dict:
        if self._trusted_proof(session, confirmation, op) is None:
            raise _Refuse("commit_unknown", "operation_unverified", "confirmation_untrusted")
        observation = DomainObservation.from_dict(confirmation.observation).to_dict()
        return self._result("replayed", observation=observation, dispatched=False,
                            ledger_snapshot=self._safe_snapshot(user_id, request_id),
                            user_id=user_id, request_id=request_id, op_id=op.op_id,
                            task_id=op.task_id, position=op.position)

    # ------------------------------------------------------------- validation

    def _validate_call_inputs(self, user_id, request_id, op_id, expected_revision, lease, tool_call_id, batch_id) -> None:
        for name, value in (("user_id", user_id), ("request_id", request_id), ("op_id", op_id),
                            ("tool_call_id", tool_call_id), ("batch_id", batch_id)):
            if not isinstance(value, str) or not value:
                raise _Refuse("rejected", "schema", f"{name}_invalid")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise _Refuse("rejected", "schema", "expected_revision_invalid")
        if not isinstance(lease, ledger_api.LeaseProof):
            raise _Refuse("rejected", "schema", "lease_type_invalid")

    def _validate_source_and_task(self, row, task) -> None:
        if task.task_type != "workout_record":
            raise _Refuse("rejected", "identity", "wrong_task_type")
        start, end = task.source_start, task.source_end
        if (isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int)
                or start < 0 or end < start or end > len(row.source)):
            raise _Refuse("rejected", "identity", "source_range_invalid")
        digest = hashlib.sha256(row.source[start:end].encode("utf-8")).hexdigest()
        if digest != task.source_digest:
            raise _Refuse("rejected", "identity", "task_source_digest_mismatch")
        safety_block = source_safety_blocked(row.source)
        if safety_block:
            raise _Refuse("rejected", "source", safety_block)
        self._validate_semantic_authorization(row, task)
        validation = validate_source_ref(
            SourceScope(text=row.source),
            {"start": start, "end": end, "exact_text": row.source[start:end]},
            "workout_record")
        if not validation.ok:
            raise _Refuse("rejected", "source", "source_ref_invalid")
        if validation.authorization != "authorized":
            raise _Refuse("rejected", "source", validation.reason or "source_rejected")

    @staticmethod
    def _validate_semantic_authorization(row, task) -> None:
        """Require the current request's validated record task to authorize this span.

        Safety is a separate gate. The write intent itself is the model's
        structured decision, so the bridge must see a source-bound validated
        workout_record task that is a real record write and whose span covers
        the frozen task. Negation/hypothetical/third-party flags and non-write
        actions are rejected here with the same reasons the planner gate uses.
        """
        from app.services.agent.semantics import get_understanding
        from app.services.agent.task_models import is_record_write
        from app.services.security.write_guard import authorize_write_span

        value = get_understanding(row.source)
        tasks = value.get("validated_tasks") if isinstance(value, dict) else None
        if not isinstance(tasks, list) or not tasks:
            raise _Refuse("rejected", "source", "source_authorization_unavailable")
        start, end = task.source_start, task.source_end
        record_reason = None
        non_record = False
        for item in tasks:
            if not isinstance(item, dict) or item.get("task_type") != "workout_record":
                continue
            item_start, item_end = item.get("source_start"), item.get("source_end")
            if (isinstance(item_start, bool) or not isinstance(item_start, int)
                    or isinstance(item_end, bool) or not isinstance(item_end, int)
                    or not (0 <= item_start <= start <= end <= item_end <= len(row.source))):
                continue
            if item.get("text") != row.source[item_start:item_end]:
                continue
            if not is_record_write(item):
                non_record = True
                continue
            authorization, reason = authorize_write_span(
                row.source[start:end], item, semantic_available=True)
            if authorization == "authorized":
                return
            record_reason = record_reason or reason or "source_unauthorized"
        if record_reason is not None:
            raise _Refuse("rejected", "source", record_reason)
        if non_record:
            raise _Refuse("rejected", "source", "source_not_record_write")
        raise _Refuse("rejected", "source", "source_authorization_missing")

    def _frozen_record(self, task, op) -> dict:
        parameters = task.parameters
        if not isinstance(parameters, dict):
            raise _Refuse("rejected", "identity", "task_parameters_invalid")
        records = parameters.get("records")
        if not isinstance(records, list):
            raise _Refuse("rejected", "identity", "task_records_missing")
        if isinstance(op.position, bool) or not isinstance(op.position, int) or not (0 <= op.position < len(records)):
            raise _Refuse("rejected", "identity", "position_out_of_range")
        if not isinstance(op.parameters, dict) or op.parameters != records[op.position]:
            raise _Refuse("rejected", "identity", "position_record_mismatch")
        return self._validate_record(copy.deepcopy(op.parameters))

    @staticmethod
    def _strict_int(value, minimum) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise _Refuse("rejected", "schema", "integer_field_invalid")
        return value

    @staticmethod
    def _strict_number(value) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise _Refuse("rejected", "schema", "number_field_invalid")
        return float(value)

    def _validate_record(self, value) -> dict:
        if not isinstance(value, dict) or set(value) - _RECORD_KEYS:
            raise _Refuse("rejected", "schema", "record_keys_invalid")
        exercise = value.get("exercise")
        if not isinstance(exercise, str) or not exercise.strip():
            raise _Refuse("rejected", "schema", "exercise_invalid")
        sets = self._strict_int(value.get("sets"), 1)
        reps = self._strict_int(value.get("reps"), 1)
        weight = self._strict_number(value.get("weight_kg"))
        note = value.get("note", "")
        if not isinstance(note, str):
            raise _Refuse("rejected", "schema", "note_invalid")
        difficulty = value.get("difficulty", "未记录")
        if not isinstance(difficulty, str) or not difficulty:
            raise _Refuse("rejected", "schema", "difficulty_invalid")
        completed = value.get("completed", True)
        if not isinstance(completed, bool):
            raise _Refuse("rejected", "schema", "completed_invalid")
        frozen_date = value.get("date")
        if not isinstance(frozen_date, str) or not _DATE_RE.fullmatch(frozen_date):
            raise _Refuse("rejected", "schema", "date_invalid")
        try:
            date_type.fromisoformat(frozen_date)
        except ValueError:
            raise _Refuse("rejected", "schema", "date_invalid") from None
        details = value.get("set_details")
        if details is not None:
            if not isinstance(details, list):
                raise _Refuse("rejected", "schema", "set_details_invalid")
            for item in details:
                if not isinstance(item, dict) or set(item) - _SET_KEYS:
                    raise _Refuse("rejected", "schema", "set_details_invalid")
                if "weight_kg" in item:
                    self._strict_number(item["weight_kg"])
                if "reps" in item:
                    self._strict_int(item["reps"], 1)
                if "difficulty" in item and (not isinstance(item["difficulty"], str) or not item["difficulty"]):
                    raise _Refuse("rejected", "schema", "set_details_invalid")
                if "completed" in item and not isinstance(item["completed"], bool):
                    raise _Refuse("rejected", "schema", "set_details_invalid")
                if "note" in item and not isinstance(item["note"], str):
                    raise _Refuse("rejected", "schema", "set_details_invalid")
        return {"exercise": exercise, "sets": sets, "reps": reps, "weight_kg": weight, "note": note,
                "difficulty": difficulty, "completed": completed, "date": frozen_date, "set_details": details}

    def _next_checkpoint(self, row) -> ledger_api.Checkpoint:
        current = row.checkpoint
        if not isinstance(current, dict):
            raise _Refuse("not_committed", "storage", "checkpoint_corrupt")
        values = []
        for field in ("supervisor_turns", "tool_steps", "event_sequence"):
            item = current.get(field)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise _Refuse("not_committed", "storage", "checkpoint_corrupt")
            values.append(item)
        return ledger_api.Checkpoint(supervisor_turns=max(1, values[0]), tool_steps=max(1, values[1]),
                                     event_sequence=values[2] + 1, budget_scope_id=row.budget_scope_id)

    def _trusted_proof(self, session, confirmation, op) -> dict | None:
        """Trusted bridge proof: canonical reserved key/phase plus a paired claim.

        Checks application owner/request/task/op/key/strict position, the
        canonical reserved keys and phases, a real claim before the
        confirmation (not necessarily adjacent) and proof/receipt consistency.
        """
        try:
            if not confirmation.confirmed or confirmation.operation_state != "committed":
                return None
            if confirmation.phase != "confirmation" or confirmation.event_key != _COMMIT_PREFIX + op.op_id:
                return None
            observation = confirmation.observation
            if not isinstance(observation, dict) or observation.get("commit_state") != "committed":
                return None
            if not _origin_matches(observation):
                return None
            result = observation.get("result")
            if not isinstance(result, dict) or result.get("proof_type") != _PROOF_TYPE:
                return None
            if confirmation.op_id != op.op_id or confirmation.task_id != op.task_id:
                return None
            if isinstance(op.position, bool) or not isinstance(op.position, int):
                return None
            if result.get("idempotency_key") != op.idempotency_key or result.get("position") != op.position:
                return None
            record_id = result.get("record_id")
            session_id = result.get("session_id")
            if isinstance(record_id, bool) or not isinstance(record_id, int):
                return None
            if isinstance(session_id, bool) or not isinstance(session_id, int):
                return None
            receipts = observation.get("receipt_refs")
            if not isinstance(receipts, list) or not receipts or not all(isinstance(item, str) and item.strip() for item in receipts):
                return None
            if (_PROOF_TYPE not in receipts or f"workout_record:{record_id}" not in receipts
                    or f"workout_session:{session_id}" not in receipts):
                return None
            claim = self._ledger.ledger_event_row(session, op.user_id, op.request_id, _CLAIM_PREFIX + op.op_id)
            if claim is None or claim.phase != "write_claim" or claim.event_key != _CLAIM_PREFIX + op.op_id:
                return None
            if not _origin_matches(claim.observation):
                return None
            if claim.op_id != op.op_id or claim.task_id != op.task_id:
                return None
            claim_sequence, confirmation_sequence = claim.sequence, confirmation.sequence
            if (isinstance(claim_sequence, bool) or not isinstance(claim_sequence, int)
                    or isinstance(confirmation_sequence, bool) or not isinstance(confirmation_sequence, int)
                    or claim_sequence >= confirmation_sequence):
                return None
            return result
        except Exception:
            return None

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _ledger_refusal(exc, dispatched=False) -> _Refuse:
        code = getattr(exc, "code", None)
        if code == "input_invalid":
            return _Refuse("rejected", "schema", code, dispatched)
        if code in {"identity_conflict", "terminal_frozen"}:
            return _Refuse("rejected", "identity", code, dispatched)
        if code in {"lease_busy", "lease_lost"}:
            return _Refuse("rejected", "lease", code, dispatched)
        if code == "revision_conflict":
            return _Refuse("rejected", "revision", code, dispatched)
        return _Refuse("not_committed", "storage", code or "ledger_error", dispatched)

    def _limited(self, refuse, user_id=None, request_id=None, op_id=None) -> dict:
        return self._result(refuse.outcome, reason=refuse.reason, detail=refuse.detail,
                            dispatched=refuse.dispatched, ledger_snapshot=self._safe_snapshot(user_id, request_id),
                            user_id=user_id, request_id=request_id, op_id=op_id)

    @staticmethod
    def _result(outcome, *, reason=None, detail=None, observation=None, dispatched=False,
                ledger_snapshot=None, user_id=None, request_id=None, op_id=None, task_id=None, position=None) -> dict:
        if observation is None:
            status = {"rejected": ToolStatus.REJECTED, "not_committed": ToolStatus.PARTIAL,
                      "commit_unknown": ToolStatus.COMMIT_UNKNOWN}.get(outcome, ToolStatus.UNAVAILABLE)
            observation = DomainObservation(status=status, task_id=task_id, error=detail).to_dict()
        return {"request_id": request_id, "op_id": op_id, "task_id": task_id, "position": position,
                "outcome": outcome, "dispatched": bool(dispatched), "reason": reason,
                "observation": observation, "ledger_snapshot": ledger_snapshot}

    def _safe_snapshot(self, user_id, request_id):
        try:
            if user_id is None or request_id is None:
                return None
            return self._ledger.load_request(user_id, request_id)
        except Exception:
            return None

    @staticmethod
    def _rollback_quiet(session) -> None:
        try:
            session.rollback()
        except Exception:
            pass

    @staticmethod
    def _close_quiet(session) -> None:
        try:
            session.close()
        except Exception:
            pass
