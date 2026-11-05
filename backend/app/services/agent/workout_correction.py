"""Source-bound workout-name updates; business mutation and C1 proof are atomic."""
from __future__ import annotations
from types import SimpleNamespace

from sqlalchemy import select

from app.db.execution_ledger_models import ExecutionRequestModel, ExecutionOperationModel
from app.db.memory_models import MemoryTurn
from app.db.models import UserProfileModel, WorkoutRecordModel
from app.services.agent import execution_ledger_store as ledger
from app.services.agent.react_models import DomainObservation

KIND = "workout_name_correction"
PROOF = "workout_name_update_v1"
FIELDS = ("exercise", "workout_date", "sets", "reps", "weight_kg", "difficulty",
          "completed", "note", "set_details")


class CorrectionError(ValueError):
    pass


def snapshot(record):
    return {key: (getattr(record, key).isoformat() if key == "workout_date"
                  else getattr(record, key)) for key in FIELDS}


def source_request(session, user_id, conversation_id, message_id, *, for_update=False):
    turn_id, separator, role = message_id.rpartition(":")
    if not separator or role not in {"user", "assistant"}:
        raise CorrectionError("correction_source_unavailable")
    # Owner/epoch/conversation/visibility are filtered in SQL before loading data.
    statement = select(MemoryTurn.messages,
        ExecutionRequestModel.user_id.label("owner_id"),
        ExecutionRequestModel.request_id.label("saved_request_id")).select_from(MemoryTurn).join(
        ExecutionRequestModel, ExecutionRequestModel.request_id == MemoryTurn.request_id
    ).join(UserProfileModel, UserProfileModel.user_id == ExecutionRequestModel.user_id).where(
        ExecutionRequestModel.user_id == user_id,
        ExecutionRequestModel.conversation_id == conversation_id,
        MemoryTurn.conversation_id == conversation_id, MemoryTurn.turn_id == turn_id,
        MemoryTurn.memory_epoch == UserProfileModel.memory_epoch,
        MemoryTurn.status == "completed", MemoryTurn.hidden_at.is_(None),
        MemoryTurn.deleted_at.is_(None))
    row = session.execute(statement.with_for_update() if for_update else statement).first()
    if row is None or not any(m.get("message_id") == message_id for m in row.messages or []):
        raise CorrectionError("correction_source_unavailable")
    return SimpleNamespace(user_id=row.owner_id, request_id=row.saved_request_id,
                           conversation_id=conversation_id)


class WorkoutCorrectionBridge:
    def __init__(self, factory):
        self.factory = factory
        self.store = ledger.ExecutionLedgerStore(factory)
        self.capability = ledger._bind_correction_reserved_capability(self.store, self)

    def resolve(self, user_id, conversation_id, correction):
        with self.factory() as session:
            source = source_request(session, user_id, conversation_id, correction["source_message_id"])
            ids = self._source_ids(session, source)
            statement = select(WorkoutRecordModel).where(
                WorkoutRecordModel.user_id == user_id, WorkoutRecordModel.id.in_(ids))
            original = correction.get("original_exercise")
            if original:
                statement = statement.where(WorkoutRecordModel.exercise == original)
            records = session.scalars(statement.order_by(WorkoutRecordModel.id).limit(13)).all()
            if len(records) != 1:
                raise CorrectionError("correction_target_ambiguous" if records else "correction_target_unavailable")
            return {"record_id": records[0].id, "expected": snapshot(records[0]),
                    "source_request_id": source.request_id, "correction": dict(correction)}

    def _source_ids(self, session, source):
        """At most twelve individually verified commit proofs, never arbitrary IDs."""
        from app.services.agent.workout_write_bridge import WorkoutWriteBridge
        operations = session.scalars(select(ExecutionOperationModel).where(
            ExecutionOperationModel.user_id == source.user_id,
            ExecutionOperationModel.request_id == source.request_id,
            ExecutionOperationModel.kind.in_(["workout", KIND]),
            ExecutionOperationModel.status == "committed").order_by(
                ExecutionOperationModel.position, ExecutionOperationModel.op_id).limit(13)).all()
        if len(operations)>12:
            raise CorrectionError("correction_target_ambiguous")
        original_bridge = WorkoutWriteBridge(self.factory)
        ids=set()
        for op in operations:
            prefix = ledger._CORRECTION_EVENT_PREFIX if op.kind == KIND else "c2-1-commit:v1:"
            event = self.store.ledger_event_row(session,source.user_id,source.request_id,prefix+op.op_id)
            proof = ((event.observation or {}).get("result") if self.verified(session,op,event) else None) \
                if op.kind == KIND else (original_bridge._trusted_proof(session,event,op) if event else None)
            if isinstance(proof,dict) and type(proof.get("record_id")) is int:
                ids.add(proof["record_id"])
        return ids

    def _read_result(self, user_id, request_id, op_id, *, dispatched=False, found_outcome="committed"):
        with self.factory() as session:
            op = self.store.ledger_operation_row(session, user_id, request_id, op_id)
            event = self.store.ledger_event_row(session, user_id, request_id,
                                               ledger._CORRECTION_EVENT_PREFIX + op_id)
            if not self.verified(session, op, event):
                return {"outcome": "commit_unknown"}
            request = self.store.ledger_request_row(session, user_id, request_id)
            try:
                source_request(session, user_id, request.conversation_id,
                               op.parameters["correction"]["source_message_id"])
            except CorrectionError:
                return {"outcome": "commit_unknown"}
            return {"outcome": "committed", "observation": dict(event.observation)}

    def verified(self, session, op, event):
        if op is None or event is None or op.kind != KIND or op.status != "committed":
            return False
        observation = event.observation or {}
        proof = observation.get("result") or {}
        request = self.store.ledger_request_row(session,op.user_id,op.request_id)
        task = self.store.ledger_task_row(session,op.user_id,op.request_id,op.task_id)
        if (request is None or task is None or task.parameters.get("correction_operation") != op.parameters
                or op.source_digest != request.source_digest or op.time_base != request.time_base):
            return False
        return (event.event_key == ledger._CORRECTION_EVENT_PREFIX + op.op_id
                and event.op_id == op.op_id and event.task_id == op.task_id
                and event.phase == "confirmation" and event.confirmed is True
                and event.operation_state == "committed"
                and observation.get("status") == "completed"
                and observation.get("commit_state") == "committed"
                and observation.get("task_id") == op.task_id
                and observation.get(ledger._CORRECTION_ORIGIN_KEY) == ledger._CORRECTION_ORIGIN
                and proof.get("proof_type") == PROOF
                and type(proof.get("record_id")) is int
                and proof.get("record_id") == op.parameters.get("record_id")
                and proof.get("parameters_digest") == op.parameters_digest
                and op.parameters_digest == ledger._digest(op.parameters)
                and proof.get("idempotency_key") == op.idempotency_key
                and observation.get("receipt_refs") == [PROOF, f"workout_record:{proof.get('record_id')}"])

    def commit(self, runtime, op_id):
        if runtime._policy_stop_check():
            return DomainObservation(status="unavailable",error="policy_stop")
        session = self.factory()
        committed = False
        try:
            request = self.store.ledger_request_row(session, runtime.user_id, runtime.request_id, for_update=True)
            self.store.ledger_require_lease(request, runtime.lease, self.store.ledger_statement_now(session))
            op = self.store.ledger_operation_row(session, runtime.user_id, runtime.request_id, op_id)
            existing = self.store.ledger_event_row(session, runtime.user_id, runtime.request_id,
                                                  ledger._CORRECTION_EVENT_PREFIX + op_id)
            if self.verified(session, op, existing):
                source_request(session,runtime.user_id,request.conversation_id,
                               op.parameters["correction"]["source_message_id"],for_update=True)
                return DomainObservation.from_dict(existing.observation)
            if op is None or op.kind != KIND or op.status != "dispatch_intent":
                raise CorrectionError("correction_operation_unavailable")
            task = self.store.ledger_task_row(session, runtime.user_id, runtime.request_id, op.task_id)
            params = op.parameters
            if (task is None or task.parameters.get("correction_operation") != params
                    or op.parameters_digest != ledger._digest(params)
                    or op.source_digest != request.source_digest or op.time_base != request.time_base):
                raise CorrectionError("correction_source_unavailable")
            from app.services.agent.workout_write_bridge import WorkoutWriteBridge
            WorkoutWriteBridge._validate_semantic_authorization(request, task)
            from app.services.agent.semantics import get_understanding
            targets = (get_understanding(request.source) or {}).get("validated_tasks") or []
            matching = [t for t in targets if t.get("source_start") == task.source_start
                        and t.get("source_end") == task.source_end
                        and t.get("workout_correction") == params["correction"]]
            if len(matching) != 1 or params["correction"]["exercise_quote"] not in request.source[task.source_start:task.source_end]:
                raise CorrectionError("correction_source_unavailable")
            origin = source_request(session, runtime.user_id, request.conversation_id,
                                    params["correction"]["source_message_id"],for_update=True)
            if origin.request_id != params["source_request_id"]:
                raise CorrectionError("correction_source_unavailable")
            if params["record_id"] not in self._source_ids(session,origin):
                raise CorrectionError("correction_source_unavailable")
            row = session.scalar(select(WorkoutRecordModel).where(
                WorkoutRecordModel.id == params["record_id"],
                WorkoutRecordModel.user_id == runtime.user_id).with_for_update())
            if row is None:
                raise CorrectionError("correction_target_unavailable")
            if snapshot(row) != params["expected"]:
                raise CorrectionError("correction_target_changed")
            from app.db.workout_session_repository import _record_relation_session, _resolve_bound_exercise
            mirror = _record_relation_session(session, row, runtime.user_id)
            exercise = (_resolve_bound_exercise(session, mirror, row, previous_name=row.exercise)
                        if mirror is not None else None)
            new_name = params["correction"]["exercise_quote"]
            previous_name = row.exercise
            if runtime._policy_stop_check():
                session.rollback()
                return DomainObservation(status="unavailable",error="policy_stop")
            row.exercise = new_name
            if mirror is not None:
                mirror.title = new_name
                if exercise is None:
                    raise CorrectionError("correction_mirror_unavailable")
                exercise.exercise_name = new_name
            session.flush()
            summary = (f"已将训练记录的动作名由“{previous_name}”改为“{new_name}”；"
                       f"{row.workout_date.isoformat()}，{row.weight_kg:g} kg × {row.reps} 次 × {row.sets} 组保持不变。")
            proof = {"proof_type": PROOF, "record_id": row.id,
                     "idempotency_key": op.idempotency_key, "parameters_digest": op.parameters_digest}
            observation = DomainObservation(status="completed", task_id=task.task_id,
                commit_state="committed", result=proof, summary=summary,
                receipt_refs=[PROOF, f"workout_record:{row.id}"])
            current = request.checkpoint
            checkpoint = ledger.Checkpoint(supervisor_turns=current["supervisor_turns"],
                tool_steps=current["tool_steps"],event_sequence=current["event_sequence"]+1,
                budget_scope_id=request.budget_scope_id)
            event = ledger.EventInput(event_key=ledger._CORRECTION_EVENT_PREFIX+op_id,
                tool_call_id="correction-commit",batch_id="workout-name-correction",
                task_id=task.task_id,op_id=op_id,phase="confirmation",observation=observation.to_dict(),
                task_status="completed",operation_state="committed",confirmed=True)
            self.store.ledger_append_correction_event(session, runtime.user_id, runtime.request_id, event,
                checkpoint=checkpoint,expected_revision=request.revision,lease=runtime.lease,capability=self.capability)
            committed = True
            session.commit()
            return observation
        except CorrectionError as exc:
            session.rollback()
            return DomainObservation(status="needs_clarification",error=str(exc),
                missing_fields=["correction_target"],summary={
                    "correction_target_changed":"这条记录已发生变化，请核对后重新指定要更正的动作。",
                }.get(str(exc),"无法确认要更正的原记录，本次没有修改；请明确是哪一条训练记录。"))
        except Exception:
            session.rollback()
            if committed:
                try:
                    truth = self._read_result(runtime.user_id,runtime.request_id,op_id)
                except Exception:
                    truth = {"outcome":"commit_unknown"}
                if truth.get("outcome") == "committed":
                    return DomainObservation.from_dict(truth["observation"])
            return DomainObservation(status="unavailable",error="correction_storage_unavailable",
                commit_state="unknown" if committed else None,
                summary="训练记录更正未能确认完成，请核对记录后再试。")
        finally:
            session.close()
