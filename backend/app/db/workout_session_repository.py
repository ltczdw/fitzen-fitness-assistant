from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select

from app.db import session as db_session
from app.schemas.workout_session import WorkoutSetPayload
from app.db.models import WorkoutRecordModel
from app.db.storage import require_ready, storage_session, WorkoutAssociationError, BusinessValidationError
from app.db.workout_session_models import WorkoutExerciseModel, WorkoutSessionModel, WorkoutSetModel


logger = logging.getLogger(__name__)


class WorkoutMetricError(BusinessValidationError):
    """A set must retain a consistent strength or cardio measurement kind."""







def list_sessions(user_id: str = "demo-user", limit: int = 30, start_date: date | None = None, end_date: date | None = None) -> list[dict[str, Any]]:
    require_ready()
    with storage_session() as session:
        statement = select(WorkoutSessionModel).where(WorkoutSessionModel.user_id == user_id)
        if start_date:
            statement = statement.where(WorkoutSessionModel.session_date >= start_date)
        if end_date:
            statement = statement.where(WorkoutSessionModel.session_date <= end_date)
        rows = session.scalars(statement.order_by(WorkoutSessionModel.session_date.desc(), WorkoutSessionModel.id.desc()).limit(limit)).all()
        return _serialize_many(session, rows)


def create_session(payload: dict[str, Any]) -> dict[str, Any]:
    require_ready()
    with storage_session() as session:
        row = WorkoutSessionModel(user_id=payload["user_id"], session_date=payload["session_date"], title=payload["title"], note=payload.get("note", ""))
        session.add(row)
        session.flush()
        _replace_exercises(session, row.id, payload["exercises"])
        session.commit()
        session.refresh(row)
        return _serialize(session, row)


def update_session(user_id: str, session_id: int, payload: dict[str, Any]) -> dict[str, Any] | None:
    require_ready()
    with storage_session() as session:
        row = session.scalar(select(WorkoutSessionModel).where(WorkoutSessionModel.id == session_id, WorkoutSessionModel.user_id == user_id))
        if row is None:
            return None
        # R012: the complete relation gate runs before any ORM mutation/flush/
        # autoflush so a damaged relation returns 409 with zero business DML.
        record = _session_relation_record(session, row, user_id)
        bound = _resolve_bound_exercise(session, row, record, previous_name=record.exercise) if record is not None else None
        row.session_date = payload["session_date"]
        row.title = payload["title"]
        row.note = payload.get("note", "")
        _apply_exercise_payload(session, row, payload["exercises"])
        session.flush()
        if record is not None:
            if bound is not None:
                if _exercise_exists(session, bound.id):
                    # R012: the bound source action keeps its identity through
                    # the full PUT; only that action projects the source record.
                    _sync_record_from_session(session, row, exercise_row=bound, record=record)
                else:
                    # The source action was explicitly omitted: delete its
                    # record and detach the session, keeping unrelated actions.
                    _delete_linked_record(session, row)
                    row.source_record_id = None
            else:
                # A record-linked session without a binding: adopt the single
                # new/unique action or fail closed, never rebind silently.
                adopted = _resolve_bound_exercise(session, row, record, previous_name=record.exercise)
                if adopted is not None:
                    _sync_record_from_session(session, row, exercise_row=adopted, record=record)
                else:
                    _delete_linked_record(session, row)
                    row.source_record_id = None
        session.commit()
        session.refresh(row)
        return _serialize(session, row)


def delete_session(user_id: str, session_id: int) -> bool:
    require_ready()
    with storage_session() as session:
        row = session.scalar(select(WorkoutSessionModel).where(WorkoutSessionModel.id == session_id, WorkoutSessionModel.user_id == user_id))
        if row is None:
            return False
        # R012: an explicit session delete removes its actions/sets and the
        # associated source record in the same transaction; a damaged relation
        # (including a foreign action binding on the record) is rejected before
        # any delete.
        record = _session_relation_record(session, row, user_id)
        if record is not None:
            _record_relation_session(session, record, user_id)
        _delete_linked_record(session, row)
        _delete_session_row(session, row)
        session.commit()
        return True


def delete_exercise(user_id: str, session_id: int, exercise_id: int) -> bool:
    """删除单个动作；最后一个动作被删除时同步删除空会话。"""
    require_ready()
    with storage_session() as session:
        session_row = session.scalar(select(WorkoutSessionModel).where(WorkoutSessionModel.id == session_id, WorkoutSessionModel.user_id == user_id))
        exercise_row = session.scalar(select(WorkoutExerciseModel).where(WorkoutExerciseModel.id == exercise_id, WorkoutExerciseModel.session_id == session_id))
        if session_row is None or exercise_row is None:
            return False
        record = _session_relation_record(session, session_row, user_id)
        bound = _resolve_bound_exercise(session, session_row, record, previous_name=record.exercise) if record is not None else None
        is_source = bound is not None and bound.id == exercise_id
        remaining = session.scalars(select(WorkoutExerciseModel.id).where(
            WorkoutExerciseModel.session_id == session_id,
            WorkoutExerciseModel.id != exercise_id)).all()
        session.execute(delete(WorkoutSetModel).where(WorkoutSetModel.exercise_id == exercise_id))
        session.delete(exercise_row)
        session.flush()
        if is_source:
            # Deleting the source action deletes its record; unrelated actions
            # keep their IDs/values and the session becomes detached.
            _delete_linked_record(session, session_row)
            if remaining:
                session_row.source_record_id = None
            else:
                session.delete(session_row)
        elif not remaining:
            session.delete(session_row)
        session.commit()
        return True


def add_set(user_id: str, session_id: int, exercise_id: int, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Append one set to an existing exercise after frontend confirmation."""
    require_ready()
    with storage_session() as session:
        exercise = session.scalar(
            select(WorkoutExerciseModel)
            .join(WorkoutSessionModel, WorkoutExerciseModel.session_id == WorkoutSessionModel.id)
            .where(
                WorkoutExerciseModel.id == exercise_id,
                WorkoutExerciseModel.session_id == session_id,
                WorkoutSessionModel.user_id == user_id,
            )
        )
        if exercise is None:
            return None
        session_row = session.scalar(select(WorkoutSessionModel).where(WorkoutSessionModel.id == exercise.session_id))
        # R012: the complete relation gate runs before any ORM mutation/flush so
        # a damaged relation returns 409 with zero business DML.
        record = _session_relation_record(session, session_row, user_id) if session_row is not None else None
        bound = _resolve_bound_exercise(session, session_row, record, previous_name=record.exercise) if record is not None else None
        last_index = session.scalars(
            select(WorkoutSetModel.set_index)
            .where(WorkoutSetModel.exercise_id == exercise_id)
            .order_by(WorkoutSetModel.set_index.desc())
            .limit(1)
        ).first()
        existing_sets = _ordered_sets(session, exercise_id)
        cardio = payload.get("duration_minutes") is not None
        if any((other.duration_minutes is not None) != cardio for other in existing_sets):
            raise WorkoutMetricError("新增段必须与该动作已有记录使用相同计量方式")
        row = WorkoutSetModel(
            exercise_id=exercise_id,
            set_index=(int(last_index) + 1) if last_index is not None else 0,
            distance_km=payload.get("distance_km"),
            duration_minutes=payload.get("duration_minutes"),
            weight_kg=payload["weight_kg"],
            reps=payload["reps"],
            difficulty=payload.get("difficulty", "未记录"),
            completed=payload.get("completed", True),
            note=payload.get("note", ""),
        )
        session.add(row)
        session.flush()
        if session_row is not None:
            # R012: adding a set to the record-backed exercise updates the
            # record's per-set snapshot in the same transaction.
            _sync_record_for_exercise(session, session_row, exercise, bound=bound)
        session.commit()
        session.refresh(row)
        return {"session_id": session_id, "exercise_id": exercise_id, "set": _serialize_set(row)}

def delete_set(user_id: str, set_id: int) -> dict[str, Any] | None:
    """Delete one set without deleting its parent exercise."""
    require_ready()
    with storage_session() as session:
        row = session.scalar(
            select(WorkoutSetModel)
            .join(WorkoutExerciseModel, WorkoutSetModel.exercise_id == WorkoutExerciseModel.id)
            .join(WorkoutSessionModel, WorkoutExerciseModel.session_id == WorkoutSessionModel.id)
            .where(WorkoutSetModel.id == set_id, WorkoutSessionModel.user_id == user_id)
        )
        if row is None:
            return None
        exercise_id = row.exercise_id
        exercise_row = session.scalar(select(WorkoutExerciseModel).where(WorkoutExerciseModel.id == exercise_id))
        session_row = session.scalar(select(WorkoutSessionModel).where(WorkoutSessionModel.id == exercise_row.session_id)) if exercise_row else None
        record = _session_relation_record(session, session_row, user_id) if session_row is not None else None
        bound = _resolve_bound_exercise(session, session_row, record, previous_name=record.exercise) if record is not None else None
        is_source = bound is not None and bound.id == exercise_id
        session.delete(row)
        session.flush()
        if is_source and exercise_row is not None and session_row is not None:
            if not _ordered_sets(session, exercise_id):
                # The source action lost its last set: apply the source-action
                # deletion contract instead of leaving an empty projection.
                session.delete(exercise_row)
                session.flush()
                _delete_linked_record(session, session_row)
                if _ordered_exercises(session, session_row.id):
                    session_row.source_record_id = None
                else:
                    session.delete(session_row)
            else:
                # R012: the remaining sets are still the record's per-set facts.
                _sync_record_from_session(session, session_row, exercise_row=exercise_row, record=record)
        session.commit()
        return {"deleted": True, "set_id": set_id, "exercise_id": exercise_id}

def update_set(user_id: str, set_id: int, payload: dict[str, Any]) -> dict[str, Any] | None:
    """只更新一组，避免客户端修改一个按钮时覆盖整条训练会话。"""
    require_ready()
    with storage_session() as session:
        row = session.scalar(
            select(WorkoutSetModel)
            .join(WorkoutExerciseModel, WorkoutSetModel.exercise_id == WorkoutExerciseModel.id)
            .join(WorkoutSessionModel, WorkoutExerciseModel.session_id == WorkoutSessionModel.id)
            .where(WorkoutSetModel.id == set_id, WorkoutSessionModel.user_id == user_id)
        )
        if row is None:
            return None
        exercise = session.scalar(select(WorkoutExerciseModel).where(WorkoutExerciseModel.id == row.exercise_id))
        if exercise is None:
            return None
        session_row = session.scalar(select(WorkoutSessionModel).where(WorkoutSessionModel.id == exercise.session_id))
        # R012: complete relation gate before mutating the set row.
        record = _session_relation_record(session, session_row, user_id) if session_row is not None else None
        bound = _resolve_bound_exercise(session, session_row, record, previous_name=record.exercise) if record is not None else None
        candidate = _serialize_set(row)
        candidate.pop("set_index", None)
        candidate.update({key: value for key, value in payload.items() if value is not None})
        try:
            WorkoutSetPayload.model_validate(candidate)
        except ValueError as exc:
            raise WorkoutMetricError("请填写完整有氧距离和时间，且不能混用力量指标") from exc
        cardio = candidate.get("duration_minutes") is not None
        if any((other.duration_minutes is not None) != cardio for other in _ordered_sets(session, exercise.id) if other.id != row.id):
            raise WorkoutMetricError("同一个动作的各段必须使用一致的计量方式")
        for field in ("difficulty", "completed", "weight_kg", "reps", "distance_km", "duration_minutes", "note"):
            if field in payload and payload[field] is not None:
                setattr(row, field, payload[field])
        session.flush()
        if session_row is not None:
            # R012: PATCH keeps every other set row and only rewrites the
            # record's detail snapshot from the current set rows.
            _sync_record_for_exercise(session, session_row, exercise, bound=bound)
        session.commit()
        return {"session_id": exercise.session_id, "exercise_id": exercise.id, "set": _serialize_set(row)}


def _replace_exercises(session, session_id: int, exercises: list[dict[str, Any]]) -> None:
    exercise_ids = session.scalars(select(WorkoutExerciseModel.id).where(WorkoutExerciseModel.session_id == session_id)).all()
    if exercise_ids:
        session.execute(delete(WorkoutSetModel).where(WorkoutSetModel.exercise_id.in_(exercise_ids)))
    session.execute(delete(WorkoutExerciseModel).where(WorkoutExerciseModel.session_id == session_id))
    for exercise_index, exercise in enumerate(exercises):
        exercise_row = WorkoutExerciseModel(
            session_id=session_id,
            exercise_name=exercise["exercise_name"],
            difficulty=exercise.get("difficulty", "未记录"),
            order_index=exercise_index,
            note=exercise.get("note", ""),
            source_record_id=exercise.get("source_record_id"),
        )
        session.add(exercise_row)
        session.flush()
        for set_index, item in enumerate(exercise["sets"]):
            session.add(WorkoutSetModel(exercise_id=exercise_row.id, set_index=set_index, distance_km=item.get("distance_km"), duration_minutes=item.get("duration_minutes"), weight_kg=item["weight_kg"], reps=item["reps"], difficulty=item.get("difficulty", "未记录"), completed=item.get("completed", True), note=item.get("note", "")))


# ---------------------------------------------------------------------------
# R012: record-backed sessions mirror the same training in one transaction
# ---------------------------------------------------------------------------

def _delete_session_row(session, row: WorkoutSessionModel) -> None:
    """Delete one session with its actions and sets (caller owns the commit)."""
    exercise_ids = session.scalars(select(WorkoutExerciseModel.id).where(WorkoutExerciseModel.session_id == row.id)).all()
    if exercise_ids:
        session.execute(delete(WorkoutSetModel).where(WorkoutSetModel.exercise_id.in_(exercise_ids)))
    session.execute(delete(WorkoutExerciseModel).where(WorkoutExerciseModel.session_id == row.id))
    session.delete(row)


def _record_for_session(session, session_row: WorkoutSessionModel) -> WorkoutRecordModel | None:
    if not getattr(session_row, "source_record_id", None):
        return None
    return session.scalar(select(WorkoutRecordModel).where(
        WorkoutRecordModel.id == session_row.source_record_id,
        WorkoutRecordModel.user_id == session_row.user_id))


def _load_record_by_id(session, record_id: int) -> WorkoutRecordModel | None:
    return session.scalar(select(WorkoutRecordModel).where(WorkoutRecordModel.id == record_id))


def _session_relation_record(session, session_row: WorkoutSessionModel, user_id: str | None = None) -> WorkoutRecordModel | None:
    """Complete preflight of a session's two-representation relation (R012).

    Returns the linked record, or None for a truly independent session (no
    session pointer and no action pointer). Raises ``WorkoutAssociationError``
    for dangling or inconsistent representations; callers run this before any
    ORM mutation/flush/autoflush so a damaged relation returns 409 with zero
    business DML.
    """
    if user_id is not None and str(session_row.user_id) != str(user_id):
        raise WorkoutAssociationError("训练会话不属于当前用户。")
    linked = None
    if session_row.source_record_id is not None:
        linked = _load_record_by_id(session, session_row.source_record_id)
        if linked is None or str(linked.user_id) != str(session_row.user_id):
            raise WorkoutAssociationError("训练会话的来源记录关联不一致，请刷新后重试。")
    pointers = session.scalars(select(WorkoutExerciseModel).where(
        WorkoutExerciseModel.session_id == session_row.id,
        WorkoutExerciseModel.source_record_id.isnot(None))).all()
    for exercise in pointers:
        if session_row.source_record_id is None:
            # An action pointer without its session pointer is a damaged second
            # representation, never an independent session.
            raise WorkoutAssociationError("训练会话的来源动作关联不一致，请刷新后重试。")
        if exercise.source_record_id != session_row.source_record_id:
            raise WorkoutAssociationError("训练会话的两种表示不一致，请刷新后重试。")
    return linked


def _record_relation_session(session, record: WorkoutRecordModel, user_id: str | None = None) -> WorkoutSessionModel | None:
    """Complete preflight of every representation pointing at a record (R012).

    Returns the linked session (or None when the record has no mirror at all);
    raises ``WorkoutAssociationError`` for cross-user, dangling or inconsistent
    relations.
    """
    if user_id is not None and str(record.user_id) != str(user_id):
        raise WorkoutAssociationError("训练记录不属于当前用户。")
    linked_sessions = session.scalars(select(WorkoutSessionModel).where(
        WorkoutSessionModel.source_record_id == record.id)).all()
    pointing = session.scalars(select(WorkoutExerciseModel).where(
        WorkoutExerciseModel.source_record_id == record.id)).all()
    owners = {row.id: row for row in linked_sessions}
    for session_row in linked_sessions:
        if str(session_row.user_id) != str(record.user_id):
            raise WorkoutAssociationError("训练记录的来源会话关联不一致，请刷新后重试。")
    if owners:
        pointers = session.scalars(select(WorkoutExerciseModel).where(
            WorkoutExerciseModel.session_id.in_(owners),
            WorkoutExerciseModel.source_record_id.isnot(None))).all()
        if any(exercise.source_record_id != record.id for exercise in pointers):
            raise WorkoutAssociationError("训练记录的两种表示不一致，请刷新后重试。")
    for exercise in pointing:
        # A valid owner is already in the source-record query above. Absence
        # proves a dangling/mismatched pointer without another SELECT by id.
        if exercise.session_id not in owners:
            raise WorkoutAssociationError("训练记录的来源动作关联不一致，请刷新后重试。")
    return linked_sessions[0] if linked_sessions else None


def _delete_linked_record(session, session_row: WorkoutSessionModel) -> None:
    record = _record_for_session(session, session_row)
    if record is not None:
        session.delete(record)


def _ordered_exercises(session, session_id: int) -> list[WorkoutExerciseModel]:
    return session.scalars(
        select(WorkoutExerciseModel)
        .where(WorkoutExerciseModel.session_id == session_id)
        .order_by(WorkoutExerciseModel.order_index, WorkoutExerciseModel.id)
    ).all()


def _ordered_sets(session, exercise_id: int) -> list[WorkoutSetModel]:
    return session.scalars(
        select(WorkoutSetModel)
        .where(WorkoutSetModel.exercise_id == exercise_id)
        .order_by(WorkoutSetModel.set_index, WorkoutSetModel.id)
    ).all()


def _exercise_exists(session, exercise_id: int) -> bool:
    return session.scalar(select(WorkoutExerciseModel.id).where(WorkoutExerciseModel.id == exercise_id)) is not None


def _binding_candidates(session, record_id: int) -> list[WorkoutExerciseModel]:
    return session.scalars(select(WorkoutExerciseModel).where(WorkoutExerciseModel.source_record_id == record_id)).all()


def _assert_binding_ownership(
    session,
    record: WorkoutRecordModel | None,
    session_row: WorkoutSessionModel | None = None,
) -> WorkoutExerciseModel | None:
    """Return the bound action or None; raise on any inconsistent binding (R012).

    A unique/FK constraint only proves structure: the current user, the linked
    session and the bound action must all agree. A missing binding returns None
    so legacy adoption can run; a corrupt binding never falls back to name or
    first action and is never silently unbound or rebound.
    """
    if record is None:
        return None
    linked = session_row
    if linked is None:
        linked = session.scalar(select(WorkoutSessionModel).where(WorkoutSessionModel.source_record_id == record.id))
    candidates = _binding_candidates(session, record.id)
    # The linked session's ownership is checked even when no action binding
    # exists: a cross-user or dangling session pointer is corruption, not a
    # legal legacy missing binding.
    if linked is None:
        if candidates:
            raise WorkoutAssociationError("训练记录的来源动作绑定不一致，请刷新后重试或带 ID 提交。")
        return None
    if (str(linked.user_id) != str(record.user_id)
            or linked.source_record_id != record.id):
        raise WorkoutAssociationError("训练记录的来源动作绑定不一致，请刷新后重试或带 ID 提交。")
    if not candidates:
        return None
    bound = [exercise for exercise in candidates if exercise.session_id == linked.id]
    for exercise in candidates:
        owner = session.scalar(select(WorkoutSessionModel).where(WorkoutSessionModel.id == exercise.session_id))
        if (owner is None
                or owner.source_record_id != record.id
                or str(owner.user_id) != str(record.user_id)):
            raise WorkoutAssociationError("训练记录的来源动作绑定不一致，请刷新后重试或带 ID 提交。")
    if len(candidates) != 1 or len(bound) != 1:
        raise WorkoutAssociationError("训练记录的来源动作绑定不一致，请刷新后重试或带 ID 提交。")
    return bound[0]


def _resolve_bound_exercise(
    session,
    session_row: WorkoutSessionModel,
    record: WorkoutRecordModel | None,
    previous_name: str | None = None,
) -> WorkoutExerciseModel | None:
    """The action bound to the record, adopting legacy data only with evidence.

    Corrupt bindings raise first (409, zero business DML); only a truly missing
    binding may adopt when the session has exactly one action, or when the
    record's pre-edit name matches exactly one action. Anything else is an
    identity conflict rather than a silent rebind.
    """
    if record is None:
        return None
    bound = _assert_binding_ownership(session, record, session_row)
    if bound is not None:
        return bound
    exercises = _ordered_exercises(session, session_row.id)
    if not exercises:
        return None
    if len(exercises) == 1:
        exercises[0].source_record_id = record.id
        return exercises[0]
    name = str(previous_name if previous_name is not None else record.exercise)
    matches = [exercise for exercise in exercises if exercise.exercise_name == name]
    if len(matches) == 1:
        matches[0].source_record_id = record.id
        return matches[0]
    raise WorkoutAssociationError("无法确定该记录对应的训练动作，请刷新后重试或带 ID 提交。")


def _delete_exercise_row(session, exercise: WorkoutExerciseModel) -> None:
    session.execute(delete(WorkoutSetModel).where(WorkoutSetModel.exercise_id == exercise.id))
    session.delete(exercise)


def mirror_sets_from_record(record: WorkoutRecordModel) -> list[dict[str, Any]]:
    """The mirror's set payload for a record: explicit details, else defaults."""
    details = None
    if getattr(record, "set_details", None):
        try:
            details = json.loads(record.set_details)
        except (TypeError, json.JSONDecodeError):
            details = None
    result: list[dict[str, Any]] = []
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict):
                continue
            weight = item.get("weight_kg")
            reps = item.get("reps")
            result.append({
                "weight_kg": float(record.weight_kg if weight is None else weight),
                "reps": int(record.reps if reps is None else reps),
                **({"distance_km": item["distance_km"], "duration_minutes": item["duration_minutes"]} if item.get("duration_minutes") is not None else {}),
                "difficulty": str(item.get("difficulty") or getattr(record, "difficulty", "未记录") or "未记录"),
                "completed": bool(item.get("completed", getattr(record, "completed", True))),
                "note": str(item.get("note", "") or ""),
            })
    if result:
        return result
    count = max(0, int(record.sets or 0))
    return [{
        "weight_kg": float(record.weight_kg or 0),
        "reps": int(record.reps or 1),
        "difficulty": str(getattr(record, "difficulty", "未记录") or "未记录"),
        "completed": bool(getattr(record, "completed", True)),
        "note": "",
    } for _ in range(count)]


def _sync_sets(session, exercise_id: int, sets: list[dict[str, Any]]) -> None:
    """Update sets in place by position so untouched rows keep their IDs."""
    existing = _ordered_sets(session, exercise_id)
    for index, item in enumerate(sets):
        if index < len(existing):
            target = existing[index]
            target.set_index = index
            target.distance_km = item.get("distance_km")
            target.duration_minutes = item.get("duration_minutes")
            target.weight_kg = item["weight_kg"]
            target.reps = item["reps"]
            target.difficulty = item.get("difficulty", "未记录") or "未记录"
            target.completed = bool(item.get("completed", True))
            target.note = item.get("note", "") or ""
        else:
            session.add(WorkoutSetModel(
                exercise_id=exercise_id,
                set_index=index,
                distance_km=item.get("distance_km"), duration_minutes=item.get("duration_minutes"), weight_kg=item["weight_kg"],
                reps=item["reps"],
                difficulty=item.get("difficulty", "未记录") or "未记录",
                completed=bool(item.get("completed", True)),
                note=item.get("note", "") or "",
            ))
    for extra in existing[len(sets):]:
        session.delete(extra)


def _set_payload_signature(item: dict[str, Any]) -> tuple:
    return (float(item["weight_kg"]), int(item["reps"]), item.get("difficulty", "未记录"),
            bool(item.get("completed", True)), item.get("note", "") or "", item.get("distance_km"), item.get("duration_minutes"))


def _set_row_signature(item: WorkoutSetModel) -> tuple:
    return (float(item.weight_kg), int(item.reps), getattr(item, "difficulty", "未记录") or "未记录",
            bool(item.completed), item.note or "", item.distance_km, item.duration_minutes)


def _match_legacy_sets(payload_sets: list[dict[str, Any]], existing_sets: list[WorkoutSetModel]) -> list[WorkoutSetModel | None]:
    """Match no-ID set payloads without inventing identities.

    Exact full-field matches anchor the request; otherwise the remaining slots
    pair only when the counts are equal. Unanchored insert/delete is a 409.
    """
    remaining = list(existing_sets)
    matches: list[WorkoutSetModel | None] = [None] * len(payload_sets)
    if not remaining:
        return matches
    anchored = False
    for index, item in enumerate(payload_sets):
        signature = _set_payload_signature(item)
        candidates = [item_row for item_row in remaining if _set_row_signature(item_row) == signature]
        if len(candidates) == 1:
            matches[index] = candidates[0]
            remaining.remove(candidates[0])
            anchored = True
        elif len(candidates) > 1:
            raise WorkoutAssociationError("无法确定组身份，请刷新后带 ID 提交。")
    unmatched = [index for index, match in enumerate(matches) if match is None]
    if unmatched:
        if len(unmatched) == len(remaining):
            for index, existing in zip(unmatched, remaining):
                matches[index] = existing
        elif not anchored:
            raise WorkoutAssociationError("无法确定组身份，请刷新后带 ID 提交。")
    return matches


def _exercise_payload_signature(item: dict[str, Any]) -> tuple:
    return (item["exercise_name"], item.get("difficulty", "未记录"), item.get("note", "") or "",
            tuple(_set_payload_signature(set_item) for set_item in item["sets"]))


def _match_legacy_exercises(payload_exercises: list[dict[str, Any]], existing: list[WorkoutExerciseModel], session) -> list[WorkoutExerciseModel | None]:
    """Match no-ID exercise payloads by unique exact content, else equal slots."""
    remaining = list(existing)
    matches: list[WorkoutExerciseModel | None] = [None] * len(payload_exercises)
    if not remaining:
        return matches
    anchored = False
    for index, item in enumerate(payload_exercises):
        signature = _exercise_payload_signature(item)
        candidates = [exercise for exercise in remaining
                      if (exercise.exercise_name, getattr(exercise, "difficulty", "未记录") or "未记录", exercise.note or "",
                          tuple(_set_row_signature(set_item) for set_item in _ordered_sets(session, exercise.id))) == signature]
        if len(candidates) == 1:
            matches[index] = candidates[0]
            remaining.remove(candidates[0])
            anchored = True
        elif len(candidates) > 1:
            raise WorkoutAssociationError("无法确定动作身份，请刷新后带 ID 提交。")
    unmatched = [index for index, match in enumerate(matches) if match is None]
    if unmatched:
        if len(unmatched) == len(remaining):
            for index, existing_item in zip(unmatched, remaining):
                matches[index] = existing_item
        elif not anchored:
            raise WorkoutAssociationError("无法确定动作身份，请刷新后带 ID 提交。")
    return matches


def _apply_set_payload(session, exercise: WorkoutExerciseModel, payload_sets: list[dict[str, Any]]) -> None:
    existing_sets = _ordered_sets(session, exercise.id)
    existing_by_id = {item.id: item for item in existing_sets}
    payload_ids = [item.get("id") for item in payload_sets if item.get("id") is not None]
    if len(payload_ids) != len(set(payload_ids)):
        raise WorkoutAssociationError("请求包含重复的组 ID，请刷新后重试。")
    for set_id in payload_ids:
        if set_id not in existing_by_id:
            raise WorkoutAssociationError("组 ID 不属于该动作，请刷新后重试。")
    if payload_ids:
        matches = [existing_by_id.get(item.get("id")) for item in payload_sets]
    else:
        matches = _match_legacy_sets(payload_sets, existing_sets)
    matched_ids = {item.id for item in matches if item is not None}
    for index, item in enumerate(payload_sets):
        target = matches[index]
        if target is None:
            session.add(WorkoutSetModel(
                exercise_id=exercise.id,
                set_index=index,
                distance_km=item.get("distance_km"), duration_minutes=item.get("duration_minutes"), weight_kg=item["weight_kg"],
                reps=item["reps"],
                difficulty=item.get("difficulty", "未记录"),
                completed=bool(item.get("completed", True)),
                note=item.get("note", ""),
            ))
        else:
            target.set_index = index
            target.distance_km = item.get("distance_km")
            target.duration_minutes = item.get("duration_minutes")
            target.weight_kg = item["weight_kg"]
            target.reps = item["reps"]
            target.difficulty = item.get("difficulty", "未记录")
            target.completed = bool(item.get("completed", True))
            target.note = item.get("note", "")
    for item_row in existing_sets:
        if item_row.id not in matched_ids:
            session.delete(item_row)


def _apply_exercise_payload(session, session_row: WorkoutSessionModel, payload_exercises: list[dict[str, Any]]) -> None:
    """Apply a full session PUT, preserving IDs and the source binding.

    Provided IDs are verified against the session/action; no-ID payloads use
    unique exact matches first and only fall back to equal-slot pairing, never
    to name/order guessing.
    """
    existing = _ordered_exercises(session, session_row.id)
    existing_by_id = {exercise.id: exercise for exercise in existing}
    payload_ids = [item.get("id") for item in payload_exercises if item.get("id") is not None]
    if len(payload_ids) != len(set(payload_ids)):
        raise WorkoutAssociationError("请求包含重复的动作 ID，请刷新后重试。")
    for exercise_id in payload_ids:
        if exercise_id not in existing_by_id:
            raise WorkoutAssociationError("动作 ID 不属于当前会话，请刷新后重试。")
    for item in payload_exercises:
        parent = existing_by_id.get(item.get("id"))
        set_ids = [set_item.get("id") for set_item in item["sets"] if set_item.get("id") is not None]
        if parent is None and set_ids:
            raise WorkoutAssociationError("新动作不能携带已有组 ID，请刷新后重试。")
        if parent is not None:
            parent_set_ids = {row.id for row in _ordered_sets(session, parent.id)}
            if any(set_id not in parent_set_ids for set_id in set_ids):
                raise WorkoutAssociationError("组 ID 不属于该动作，请刷新后重试。")
    if payload_ids:
        matches = [existing_by_id.get(item.get("id")) for item in payload_exercises]
    else:
        matches = _match_legacy_exercises(payload_exercises, existing, session)
    matched_ids = {item.id for item in matches if item is not None}
    for index, item in enumerate(payload_exercises):
        target = matches[index]
        if target is None:
            target = WorkoutExerciseModel(
                session_id=session_row.id,
                exercise_name=item["exercise_name"],
                difficulty=item.get("difficulty", "未记录"),
                note=item.get("note", ""),
                order_index=index,
            )
            session.add(target)
            session.flush()
        else:
            target.exercise_name = item["exercise_name"]
            target.difficulty = item.get("difficulty", "未记录")
            target.note = item.get("note", "")
            target.order_index = index
        _apply_set_payload(session, target, item["sets"])
    for exercise in existing:
        if exercise.id not in matched_ids:
            _delete_exercise_row(session, exercise)


def sync_mirror_session_from_record(
    session,
    session_row: WorkoutSessionModel,
    record: WorkoutRecordModel,
    exercise: WorkoutExerciseModel | None = None,
) -> None:
    """Record-side edits update the bound action in place (R012).

    The binding ID locates the source action; names and order are never used.
    Existing set rows are updated by position, extra rows are removed and new
    rows appended, so a single record edit does not rebuild unrelated rows.
    """
    session_row.session_date = record.workout_date
    session_row.title = record.exercise
    session_row.note = record.note or ""
    if exercise is None:
        exercise = _resolve_bound_exercise(session, session_row, record, previous_name=record.exercise)
    sets = mirror_sets_from_record(record)
    if exercise is None:
        # A record-linked session with no actions at all: create the bound one.
        _replace_exercises(session, session_row.id, [{
            "exercise_name": record.exercise,
            "difficulty": getattr(record, "difficulty", "未记录") or "未记录",
            "note": record.note or "",
            "sets": sets,
            "source_record_id": record.id,
        }])
        return
    exercise.exercise_name = record.exercise
    exercise.difficulty = getattr(record, "difficulty", "未记录") or "未记录"
    exercise.note = record.note or ""
    if exercise.source_record_id != record.id:
        exercise.source_record_id = record.id
    _sync_sets(session, exercise.id, sets)


def _sync_record_from_session(session, session_row: WorkoutSessionModel, exercise_row: WorkoutExerciseModel | None = None, record: WorkoutRecordModel | None = None) -> None:
    """Mirror-side edits rewrite the source record's per-set facts (R012).

    The top-level weight/reps are the current first set's compatibility
    summary (set_index order); sets is the current count. The record note and
    metadata stay untouched because mirror payloads never submitted them.
    """
    if not getattr(session_row, "source_record_id", None):
        return
    record = record if record is not None else _record_for_session(session, session_row)
    if record is None:
        return
    exercise = exercise_row if exercise_row is not None else _resolve_bound_exercise(session, session_row, record, previous_name=record.exercise)
    if exercise is None:
        return
    sets = _ordered_sets(session, exercise.id)
    details = [{
        "weight_kg": item.weight_kg,
        "reps": item.reps,
        **({"distance_km": item.distance_km, "duration_minutes": item.duration_minutes} if item.duration_minutes is not None else {}),
        "difficulty": getattr(item, "difficulty", "未记录") or "未记录",
        "completed": bool(item.completed),
        "note": item.note or "",
    } for item in sets]
    record.workout_date = session_row.session_date
    record.exercise = exercise.exercise_name
    record.sets = len(details)
    record.set_details = json.dumps(details, ensure_ascii=False) if details else None
    if details:
        record.weight_kg = float(details[0]["weight_kg"])
        record.reps = int(details[0]["reps"])


def _sync_record_for_exercise(
    session,
    session_row: WorkoutSessionModel,
    exercise_row: WorkoutExerciseModel,
    bound: WorkoutExerciseModel | None = None,
) -> None:
    """Sync only when the touched action is the record's bound action."""
    record = _record_for_session(session, session_row)
    if record is None:
        return
    if bound is None:
        bound = _resolve_bound_exercise(session, session_row, record, previous_name=record.exercise)
    if bound is None or bound.id != exercise_row.id:
        return
    _sync_record_from_session(session, session_row, exercise_row=exercise_row, record=record)


def _serialize(session, row: WorkoutSessionModel) -> dict[str, Any]:
    return _serialize_many(session, [row])[0]


def _serialize_many(session, rows: list[WorkoutSessionModel]) -> list[dict[str, Any]]:
    if not rows:
        return []
    exercises = session.scalars(select(WorkoutExerciseModel).where(
        WorkoutExerciseModel.session_id.in_([row.id for row in rows])
    ).order_by(WorkoutExerciseModel.order_index, WorkoutExerciseModel.id)).all()
    by_session: dict[int, list[WorkoutExerciseModel]] = {}
    for exercise in exercises:
        by_session.setdefault(exercise.session_id, []).append(exercise)
    sets_by_exercise: dict[int, list[WorkoutSetModel]] = {}
    if exercises:
        all_sets = session.scalars(select(WorkoutSetModel).where(
            WorkoutSetModel.exercise_id.in_([exercise.id for exercise in exercises])
        ).order_by(WorkoutSetModel.exercise_id, WorkoutSetModel.set_index, WorkoutSetModel.id)).all()
        for workout_set in all_sets:
            sets_by_exercise.setdefault(workout_set.exercise_id, []).append(workout_set)
    return [{
        "id": row.id, "user_id": row.user_id, "source_record_id": row.source_record_id,
        "session_date": row.session_date.isoformat(), "title": row.title, "note": row.note,
        "exercises": [{
            "id": exercise.id, "exercise_name": exercise.exercise_name,
            "difficulty": getattr(exercise, "difficulty", "未记录") or "未记录", "note": exercise.note,
            "sets": [_serialize_set(item) for item in sets_by_exercise.get(exercise.id, [])],
        } for exercise in by_session.get(row.id, [])],
    } for row in rows]


def _serialize_set(item: WorkoutSetModel) -> dict[str, Any]:
    return {"id": item.id, "set_index": item.set_index, "weight_kg": item.weight_kg, "reps": item.reps, **({"distance_km": item.distance_km, "duration_minutes": item.duration_minutes} if item.duration_minutes is not None else {}), "difficulty": getattr(item, "difficulty", "未记录") or "未记录", "completed": bool(item.completed), "note": item.note}






def _normalize_fallback_session(item: dict[str, Any]) -> dict[str, Any]:
    normalized = {**item}
    normalized["exercises"] = [
        {
            **exercise,
            "difficulty": exercise.get("difficulty", "未记录") or "未记录",
            "sets": [{**workout_set, "difficulty": workout_set.get("difficulty", "未记录") or "未记录", "completed": bool(workout_set.get("completed", True))} for workout_set in exercise.get("sets", [])],
        }
        for exercise in item.get("exercises", [])
    ]
    return normalized


def _session_signature(item: dict[str, Any]) -> str:
    normalized = _normalize_fallback_session(item)
    content = {
        "user_id": normalized.get("user_id"),
        "session_date": str(normalized.get("session_date", ""))[:10],
        "title": normalized.get("title", "训练会话"),
        "note": normalized.get("note", ""),
        "exercises": [
            {
                "exercise_name": exercise.get("exercise_name", ""),
                "difficulty": exercise.get("difficulty", "未记录") or "未记录",
                "note": exercise.get("note", "") or "",
                "sets": [
                    {
                        "set_index": int(workout_set.get("set_index", index)),
                        "weight_kg": float(workout_set.get("weight_kg", 0) or 0),
                        "reps": int(workout_set.get("reps", 0) or 0),
                        **({"distance_km": workout_set.get("distance_km"), "duration_minutes": workout_set["duration_minutes"]} if workout_set.get("duration_minutes") is not None else {}),
                        "difficulty": workout_set.get("difficulty", "未记录") or "未记录",
                        "completed": bool(workout_set.get("completed", True)),
                        "note": workout_set.get("note", "") or "",
                    }
                    for index, workout_set in enumerate(exercise.get("sets", []))
                ],
            }
            for exercise in normalized.get("exercises", [])
        ],
    }
    return json.dumps(content, ensure_ascii=False, sort_keys=True)
