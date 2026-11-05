from app.core import clock
from datetime import date as date_type
import json
from pathlib import Path
from threading import Lock

from sqlalchemy import exists, func, select

from app.db import session as db_session
from app.db.storage import require_ready, storage_session
from app.db.storage import CommitUnknownError, IdempotencyConflict, WorkoutDetailRequired, recover_committed
from app.db.models import WorkoutRecordModel


_fallback_lock = Lock()


def list_workouts(limit: int = 30, user_id: str = "demo-user", start_date: date_type | None = None, end_date: date_type | None = None) -> list[dict]:
    if not db_session.DB_READY or db_session.SessionLocal is None:
        require_ready()
    try:
        with storage_session() as session:
            statement = select(WorkoutRecordModel).where(WorkoutRecordModel.user_id == user_id)
            if start_date:
                statement = statement.where(WorkoutRecordModel.workout_date >= start_date)
            if end_date:
                statement = statement.where(WorkoutRecordModel.workout_date <= end_date)
            statement = statement.order_by(WorkoutRecordModel.workout_date.desc(), WorkoutRecordModel.id.desc()).limit(limit)
            return [_serialize(row) for row in session.scalars(statement).all()]
    except Exception:
        raise




def recovery_event_summary(user_id: str = "demo-user", *, days: int = 7,
                           today: date_type | None = None) -> dict:
    """Count training events, not rows or sets, for the recovery page.

    One workout session is one event. A legacy workout record is one event only
    when no session or session-exercise of the same owner points at it; mirrors
    therefore never count twice. Recent counts use the local window and exclude
    future dates, while the last date comes from all valid history.
    """
    from datetime import timedelta

    from app.db.workout_session_models import WorkoutExerciseModel, WorkoutSessionModel

    owner = str(user_id or "demo-user")
    today = today or clock.today()
    start = today - timedelta(days=max(1, int(days)) - 1)
    session_mirror = exists(
        select(1).where(
            WorkoutSessionModel.user_id == owner,
            WorkoutSessionModel.source_record_id == WorkoutRecordModel.id,
        )
    )
    exercise_mirror = exists(
        select(1)
        .select_from(WorkoutExerciseModel)
        .join(WorkoutSessionModel, WorkoutExerciseModel.session_id == WorkoutSessionModel.id)
        .where(
            WorkoutSessionModel.user_id == owner,
            WorkoutExerciseModel.source_record_id == WorkoutRecordModel.id,
        )
    )
    with storage_session() as session:
        sessions = session.scalar(select(func.count(WorkoutSessionModel.id)).where(
            WorkoutSessionModel.user_id == owner,
            WorkoutSessionModel.session_date >= start,
            WorkoutSessionModel.session_date <= today,
        )) or 0
        legacy_only = session.scalar(select(func.count(WorkoutRecordModel.id)).where(
            WorkoutRecordModel.user_id == owner,
            WorkoutRecordModel.workout_date >= start,
            WorkoutRecordModel.workout_date <= today,
            ~session_mirror,
            ~exercise_mirror,
        )) or 0
        last_session = session.scalar(select(func.max(WorkoutSessionModel.session_date)).where(
            WorkoutSessionModel.user_id == owner,
            WorkoutSessionModel.session_date <= today,
        ))
        last_record = session.scalar(select(func.max(WorkoutRecordModel.workout_date)).where(
            WorkoutRecordModel.user_id == owner,
            WorkoutRecordModel.workout_date <= today,
            ~session_mirror,
            ~exercise_mirror,
        ))
    dates = [value for value in (last_session, last_record) if value is not None]
    return {
        "recent_events": int(sessions + legacy_only),
        "recent_sessions": int(sessions),
        "recent_legacy_only": int(legacy_only),
        "last_workout_date": max(dates) if dates else None,
        "window_days": max(1, int(days)),
    }


def query_workout_read_rows(user_id: str = "demo-user", *, start_date: date_type | None = None,
                            end_date: date_type | None = None, exercises: list[str] | None = None,
                            limit: int = 30) -> tuple[list[dict], bool]:
    """Bounded per-action read across legacy records and training sessions.

    A record-backed session action is represented by its source record and is
    skipped here, so one business training record is counted once even though it
    exists in both representations.  Independent session actions (including
    actions added to a mirrored session) keep their own per-set facts.

    Owner/date/action predicates run before each source's limit; each source
    fetches one extra row, then the two bounded lists are merged, stably sorted
    and truncated together so a global top-N is not assembled from two already
    truncated lists.
    """
    owner = str(user_id or "demo-user")
    limit = max(1, min(int(limit), 100))
    wanted_exercises = [str(item).strip() for item in exercises or [] if str(item).strip()]
    from app.db.workout_session_models import (
        WorkoutExerciseModel, WorkoutSessionModel, WorkoutSetModel,
    )

    with storage_session() as session:
        record_statement = select(WorkoutRecordModel).where(
            WorkoutRecordModel.user_id == owner)
        if start_date is not None:
            record_statement = record_statement.where(
                WorkoutRecordModel.workout_date >= start_date)
        if end_date is not None:
            record_statement = record_statement.where(
                WorkoutRecordModel.workout_date <= end_date)
        if wanted_exercises:
            record_statement = record_statement.where(
                WorkoutRecordModel.exercise.in_(wanted_exercises))
        record_statement = record_statement.order_by(
            WorkoutRecordModel.workout_date.desc(), WorkoutRecordModel.id.desc()
        ).limit(limit + 1)
        records = list(session.scalars(record_statement).all())

        exercise_statement = select(WorkoutExerciseModel, WorkoutSessionModel).join(
            WorkoutSessionModel,
            WorkoutExerciseModel.session_id == WorkoutSessionModel.id,
        ).where(
            WorkoutSessionModel.user_id == owner,
            # A session action that mirrors a record is the same business row;
            # the record is the canonical read representation.  Do not fetch it
            # here, so mirror rows cannot consume the bounded session window.
            WorkoutExerciseModel.source_record_id.is_(None),
        )
        if start_date is not None:
            exercise_statement = exercise_statement.where(
                WorkoutSessionModel.session_date >= start_date)
        if end_date is not None:
            exercise_statement = exercise_statement.where(
                WorkoutSessionModel.session_date <= end_date)
        if wanted_exercises:
            exercise_statement = exercise_statement.where(
                WorkoutExerciseModel.exercise_name.in_(wanted_exercises))
        exercise_statement = exercise_statement.order_by(
            WorkoutSessionModel.session_date.desc(), WorkoutExerciseModel.id.desc()
        ).limit(limit + 1)
        session_actions = list(session.execute(exercise_statement).all())

        sets_by_exercise: dict[int, list] = {}
        exercise_ids = [exercise.id for exercise, _ in session_actions]
        if exercise_ids:
            set_rows = session.scalars(select(WorkoutSetModel).where(
                WorkoutSetModel.exercise_id.in_(exercise_ids)
            ).order_by(
                WorkoutSetModel.exercise_id, WorkoutSetModel.set_index, WorkoutSetModel.id
            )).all()
            for item in set_rows:
                sets_by_exercise.setdefault(item.exercise_id, []).append(item)

    rows: list[dict] = []
    for row in records:
        item = _serialize(row)
        item["source"] = "record"
        item["source_id"] = f"record:{row.id}"
        item["workout_date"] = row.workout_date.isoformat()
        rows.append(item)

    for exercise, session_row in session_actions:
        details = [{
            "set_index": item.set_index,
            "weight_kg": item.weight_kg,
            "reps": item.reps,
            "distance_km": item.distance_km,
            "duration_minutes": item.duration_minutes,
            "difficulty": getattr(item, "difficulty", "未记录") or "未记录",
            "completed": bool(item.completed),
            "note": item.note,
        } for item in sets_by_exercise.get(exercise.id, [])]
        first = details[0] if details else {}
        uniform = bool(details) and all(
            (item.get("weight_kg"), item.get("reps"),
             item.get("distance_km"), item.get("duration_minutes"))
            == (first.get("weight_kg"), first.get("reps"),
                first.get("distance_km"), first.get("duration_minutes"))
            for item in details)
        rows.append({
            "id": exercise.id,
            "user_id": session_row.user_id,
            "date": session_row.session_date.isoformat(),
            "workout_date": session_row.session_date.isoformat(),
            "exercise": exercise.exercise_name,
            "sets": len(details),
            "reps": first.get("reps", 0),
            "weight_kg": first.get("weight_kg", 0.0),
            "difficulty": getattr(exercise, "difficulty", "未记录") or "未记录",
            "completed": all(item.get("completed", True) for item in details) if details else True,
            "note": exercise.note,
            "set_details": details,
            "summary_mode": "uniform" if uniform else "first_set",
            "source": "session",
            "source_id": f"session:{exercise.id}",
            "session_id": session_row.id,
            "session_title": session_row.title,
            "session_note": session_row.note,
        })

    # Stable global order: newest date first; records precede independent session
    # actions on the same date so record-only reads keep their old order.
    rows.sort(key=lambda item: (
        str(item.get("date") or ""),
        1 if item.get("source") == "record" else 0,
        int(item.get("id") or 0),
    ), reverse=True)
    truncated = len(rows) > limit
    return rows[:limit], truncated


def query_workout_sources(user_id: str = "demo-user", *, start_date: date_type,
                          end_date: date_type, exercises: list[str] | None = None,
                          limit: int = 30) -> tuple[list[dict], bool]:
    """Bounded source rows for a structured workout-record query.

    Compatibility wrapper over ``query_workout_read_rows``; callers that only
    need navigation fields still receive them, now including independent
    session actions and their per-set facts.
    """
    return query_workout_read_rows(
        user_id, start_date=start_date, end_date=end_date,
        exercises=exercises, limit=limit)


def _normalized_workout_params(exercise: str, sets: int, reps: int, weight_kg: float = 0.0,
                               note: str = "", difficulty: str = "未记录", completed: bool = True,
                               date: date_type | None = None,
                               set_details: list[dict] | None = None) -> dict:
    return {
        "exercise": str(exercise).strip(),
        "sets": int(sets),
        "reps": int(reps),
        "weight_kg": round(float(weight_kg or 0.0), 4),
        "note": str(note or "").strip(),
        "difficulty": str(difficulty or "未记录"),
        "completed": bool(completed),
        "date": (date or clock.today()).isoformat(),
        "set_details": (json.dumps(set_details, ensure_ascii=False, sort_keys=True, default=str)
                        if set_details is not None else None),
    }


def _parsed_details(raw) -> list[dict] | None:
    if not raw:
        return None
    try:
        details = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(details, list):
        return None
    parsed = [item for item in details if isinstance(item, dict)]
    return parsed or None


def _project_summary_from_details(row, details: list[dict] | None) -> None:
    """R012: top-level weight/reps are the current first set's summary.

    The record's explicit per-set facts stay authoritative; the legacy
    top-level fields become a compatibility projection of set_index 0.
    """
    if not details:
        return
    first = details[0] if isinstance(details[0], dict) else {}
    row.sets = len(details)
    row.weight_kg = float(first.get("weight_kg", row.weight_kg) or 0.0)
    row.reps = int(first.get("reps", row.reps) or 1)


def _row_workout_params(row: WorkoutRecordModel) -> dict:
    try:
        details = json.loads(row.set_details) if row.set_details else None
    except (TypeError, json.JSONDecodeError):
        details = None
    return _normalized_workout_params(
        row.exercise, row.sets, row.reps, row.weight_kg, row.note,
        getattr(row, "difficulty", "未记录"), bool(getattr(row, "completed", True)),
        row.workout_date, details,
    )


def assert_workout_key_matches(row: WorkoutRecordModel, exercise: str, sets: int, reps: int,
                               weight_kg: float, note: str, difficulty: str, completed: bool,
                               date: date_type | None, set_details: list[dict] | None,
                               idempotency_key: str) -> None:
    """A reused key must describe the same business operation.

    An omitted explicit date or set detail is not a parameter difference: the
    stored first-dispatch value stays authoritative for relative dates.
    """
    existing = _row_workout_params(row)
    incoming = _normalized_workout_params(exercise, sets, reps, weight_kg, note, difficulty,
                                          completed, date, set_details)
    if set_details:
        # R012: the stored top-level summary is projected from the first set,
        # so the incoming payload is compared with the same projection.
        first = set_details[0] if isinstance(set_details[0], dict) else {}
        incoming["sets"] = len(set_details)
        incoming["weight_kg"] = round(float(first.get("weight_kg", weight_kg) or 0.0), 4)
        incoming["reps"] = int(first.get("reps", reps) or 1)
    for field in ("exercise", "sets", "reps", "weight_kg", "note", "difficulty", "completed"):
        if existing[field] != incoming[field]:
            raise IdempotencyConflict(
                f"idempotency key {idempotency_key} reused with different {field}")
    if date is not None and existing["date"] != incoming["date"]:
        raise IdempotencyConflict(
            f"idempotency key {idempotency_key} reused with different date")
    if set_details is not None and existing["set_details"] != incoming["set_details"]:
        raise IdempotencyConflict(
            f"idempotency key {idempotency_key} reused with different set_details")


def find_workout_by_key(user_id: str, idempotency_key: str) -> dict | None:
    """Read-only recovery lookup; never writes and never retries."""
    if not idempotency_key:
        return None
    require_ready()
    from app.db.workout_session_models import WorkoutSessionModel
    with storage_session() as session:
        row = session.scalar(select(WorkoutRecordModel).where(
            WorkoutRecordModel.user_id == user_id,
            WorkoutRecordModel.idempotency_key == idempotency_key))
        if row is None:
            return None
        linked = session.scalar(select(WorkoutSessionModel).where(
            WorkoutSessionModel.source_record_id == row.id,
            WorkoutSessionModel.user_id == user_id))
        return {**_serialize(row), "session_id": linked.id if linked else None}


def find_workout_by_key_in_session(session, user_id: str, idempotency_key: str) -> dict | None:
    """C2-1: session-scoped read-only lookup; never opens/closes/commits."""
    if not idempotency_key:
        return None
    from app.db.workout_session_models import WorkoutSessionModel
    row = session.scalar(select(WorkoutRecordModel).where(
        WorkoutRecordModel.user_id == user_id,
        WorkoutRecordModel.idempotency_key == idempotency_key))
    if row is None:
        return None
    linked = session.scalar(select(WorkoutSessionModel).where(
        WorkoutSessionModel.source_record_id == row.id,
        WorkoutSessionModel.user_id == user_id))
    return {**_serialize(row), "session_id": linked.id if linked else None}


def create_workout_in_session(session, user_id: str, exercise: str, sets: int, reps: int, weight_kg: float = 0.0,
                              note: str = "", difficulty: str = "未记录", completed: bool = True,
                              date: date_type | None = None, set_details: list[dict] | None = None,
                              idempotency_key: str | None = None) -> dict:
    """C2-1: create a record and its mirror inside the caller's session.

    The primitive only flushes: it never opens another session, commits or
    rolls back. The public wrappers keep owning transaction/recovery logic.
    """
    from app.db.workout_session_models import WorkoutExerciseModel, WorkoutSetModel
    if idempotency_key and session.get_bind().dialect.name == "postgresql":
        from sqlalchemy import text
        session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:scope, 0))"),
                        {"scope": "workout:" + user_id + ":" + idempotency_key})
    if idempotency_key:
        existing = session.scalar(select(WorkoutRecordModel).where(
            WorkoutRecordModel.user_id == user_id,
            WorkoutRecordModel.idempotency_key == idempotency_key))
        if existing is not None:
            assert_workout_key_matches(existing, exercise, sets, reps, weight_kg, note, difficulty,
                                       completed, date, set_details, idempotency_key)
            session_id = _sync_workout_session(session, existing)
            session.flush()
            exercise_row = session.scalar(select(WorkoutExerciseModel).where(
                WorkoutExerciseModel.source_record_id == existing.id))
            set_ids = []
            if exercise_row is not None:
                set_ids = session.scalars(select(WorkoutSetModel.id).where(
                    WorkoutSetModel.exercise_id == exercise_row.id).order_by(WorkoutSetModel.set_index)).all()
            return {**_serialize(existing), "session_id": session_id, "created": False,
                    "record_id": existing.id, "exercise_id": exercise_row.id if exercise_row is not None else None,
                    "set_ids": [int(item) for item in set_ids]}
    row = WorkoutRecordModel(workout_date=date or clock.today(), user_id=user_id, exercise=exercise, sets=sets,
                             reps=reps, weight_kg=weight_kg, difficulty=difficulty, completed=completed, note=note,
                             set_details=json.dumps(set_details, ensure_ascii=False) if set_details is not None else None,
                             idempotency_key=idempotency_key)
    _project_summary_from_details(row, set_details)
    session.add(row)
    session.flush()
    session_id = _sync_workout_session(session, row)
    session.flush()
    exercise_row = session.scalar(select(WorkoutExerciseModel).where(
        WorkoutExerciseModel.source_record_id == row.id))
    set_ids = []
    if exercise_row is not None:
        set_ids = session.scalars(select(WorkoutSetModel.id).where(
            WorkoutSetModel.exercise_id == exercise_row.id).order_by(WorkoutSetModel.set_index)).all()
    return {**_serialize(row), "session_id": session_id, "created": True, "record_id": row.id,
            "exercise_id": exercise_row.id if exercise_row is not None else None,
            "set_ids": [int(item) for item in set_ids]}


def create_workout(user_id: str, exercise: str, sets: int, reps: int, weight_kg: float = 0.0, note: str = "", difficulty: str = "未记录", completed: bool = True, date: date_type | None = None, set_details: list[dict] | None = None, idempotency_key: str | None = None) -> dict:
    from app.db.workout_session_models import WorkoutSessionModel
    def lookup():
        recovered = find_workout_by_key(user_id, idempotency_key or "")
        if recovered is None or recovered.get("session_id") is None:
            return None
        return recovered
    try:
        with storage_session() as session:
            if idempotency_key:
                # Serialize concurrent requests with the same key on PostgreSQL.
                if db_session.engine is not None and db_session.engine.dialect.name == "postgresql":
                    from sqlalchemy import text
                    session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:scope, 0))"), {"scope": "workout:" + user_id + ":" + idempotency_key})
                existing = session.scalar(select(WorkoutRecordModel).where(WorkoutRecordModel.user_id == user_id, WorkoutRecordModel.idempotency_key == idempotency_key))
                if existing is not None:
                    assert_workout_key_matches(existing, exercise, sets, reps, weight_kg, note,
                                               difficulty, completed, date, set_details, idempotency_key)
                    session_id = _sync_workout_session(session, existing)
                    session.commit()
                    return {**_serialize(existing), "session_id": session_id}
            row = WorkoutRecordModel(workout_date=date or clock.today(), user_id=user_id, exercise=exercise, sets=sets, reps=reps, weight_kg=weight_kg, difficulty=difficulty, completed=completed, note=note, set_details=json.dumps(set_details, ensure_ascii=False) if set_details is not None else None, idempotency_key=idempotency_key)
            _project_summary_from_details(row, set_details)
            session.add(row)
            session.flush()
            session_id = _sync_workout_session(session, row)
            session.commit()
            session.refresh(row)
            return {**_serialize(row), "session_id": session_id}
    except CommitUnknownError as exc:
        if not idempotency_key:
            raise
        return recover_committed(lookup, exc)


def _sync_workout_session(session, record):
    """Atomically expose an Agent record in the same API used by the frontend."""
    from app.db.workout_session_models import WorkoutSessionModel
    from app.db.workout_session_repository import _replace_exercises, mirror_sets_from_record, sync_mirror_session_from_record
    from app.db.storage import WorkoutAssociationError
    existing = session.scalar(select(WorkoutSessionModel).where(WorkoutSessionModel.source_record_id == record.id))
    if existing:
        # R012: an existing mirror must follow record edits instead of keeping
        # the values captured at creation time.
        try:
            sync_mirror_session_from_record(session, existing, record)
        except WorkoutAssociationError:
            # Idempotent replay of an already-created record: the mirror was
            # intentionally reshaped and has no unique adoption evidence, so
            # keep the existing session untouched instead of guessing.
            return existing.id
        return existing.id
    row = WorkoutSessionModel(user_id=record.user_id, source_record_id=record.id,
        session_date=record.workout_date, title=record.exercise, note=record.note or "")
    session.add(row)
    session.flush()
    _replace_exercises(session, row.id, [{"exercise_name": record.exercise,
        "difficulty": getattr(record, "difficulty", "未记录") or "未记录",
        "note": record.note or "", "sets": mirror_sets_from_record(record),
        "source_record_id": record.id}])
    return row.id


def update_workout(user_id: str, record_id: int, record: dict) -> dict | None:
    require_ready()
    with storage_session() as session:
        from app.db.workout_session_models import WorkoutSessionModel
        from app.db.workout_session_repository import _record_relation_session, _resolve_bound_exercise, sync_mirror_session_from_record
        row = session.scalar(select(WorkoutRecordModel).where(WorkoutRecordModel.id == record_id, WorkoutRecordModel.user_id == user_id))
        if row is None:
            return None
        # R012: complete relation preflight before any ORM mutation/flush; a
        # damaged representation raises 409 with zero business DML.
        session_row = _record_relation_session(session, row, user_id)
        previous_exercise = row.exercise
        bound = None
        if session_row is not None:
            # Resolve the source-action identity before any rename so legacy
            # adoption can only use the pre-edit name.
            bound = _resolve_bound_exercise(session, session_row, row, previous_name=previous_exercise)
        requested_date = record.get("date")
        if requested_date:
            row.workout_date = date_type.fromisoformat(str(requested_date))
        incoming_details = record.get("set_details")
        current_details = _parsed_details(row.set_details)
        incoming_sets = int(record["sets"])
        incoming_reps = int(record["reps"])
        incoming_weight = float(record["weight_kg"])

        def apply_metadata() -> None:
            row.exercise = str(record["exercise"])
            if "difficulty" in record:
                row.difficulty = str(record["difficulty"])
            if "completed" in record:
                row.completed = bool(record["completed"])
            if "note" in record:
                row.note = str(record["note"])

        if incoming_details is not None:
            # Explicit per-set facts are authoritative; the top-level summary
            # is projected from the first set instead of flattening the groups.
            apply_metadata()
            row.set_details = json.dumps(incoming_details, ensure_ascii=False)
            _project_summary_from_details(row, incoming_details)
        elif current_details:
            metadata_only = (incoming_sets == int(row.sets) and incoming_reps == int(row.reps)
                             and abs(incoming_weight - float(row.weight_kg)) < 1e-9)
            uniform = all(
                (detail.get("weight_kg"), detail.get("reps"))
                == (current_details[0].get("weight_kg"), current_details[0].get("reps"))
                for detail in current_details
            )
            if not metadata_only and not uniform:
                # R012: a heterogeneous record needs explicit per-set facts;
                # a top-level-only edit must not silently flatten them.
                raise WorkoutDetailRequired("该记录包含不同的逐组重量/次数；修改顶层数值前请提供 set_details。")
            if metadata_only:
                for detail in current_details:
                    if "difficulty" in record:
                        detail["difficulty"] = str(record["difficulty"])
                    if "completed" in record:
                        detail["completed"] = bool(record["completed"])
                row.set_details = json.dumps(current_details, ensure_ascii=False)
                apply_metadata()
            else:
                # Uniform details: an explicit top-level edit keeps the uniform
                # record semantics and regenerates the projection.
                apply_metadata()
                row.sets = incoming_sets
                row.reps = incoming_reps
                row.weight_kg = incoming_weight
                row.set_details = None
        else:
            apply_metadata()
            row.sets = incoming_sets
            row.reps = incoming_reps
            row.weight_kg = incoming_weight
        session.flush()
        if session_row is not None:
            # R012: update the bound source action in the same transaction;
            # a record-linked session without actions gets the bound one.
            sync_mirror_session_from_record(session, session_row, row, exercise=bound)
        session.commit()
        session.refresh(row)
        return _serialize(row)


def delete_workout(user_id: str, record_id: int) -> bool:
    require_ready()
    with storage_session() as session:
        from app.db.workout_session_models import WorkoutSessionModel
        from app.db.workout_session_repository import _delete_exercise_row, _ordered_exercises, _record_relation_session, _resolve_bound_exercise
        row = session.scalar(select(WorkoutRecordModel).where(WorkoutRecordModel.id == record_id, WorkoutRecordModel.user_id == user_id))
        if row is None:
            return False
        # R012: complete relation preflight before any delete; a damaged
        # representation raises 409 with zero business DML.
        session_row = _record_relation_session(session, row, user_id)
        if session_row is not None:
            bound = _resolve_bound_exercise(session, session_row, row, previous_name=row.exercise)
            # Delete the record first so the SQL order stays record -> actions
            # -> sets; the bound action is removed while unrelated actions and
            # the session are preserved (detached when they remain).
            session.delete(row)
            session.flush()
            if bound is not None:
                _delete_exercise_row(session, bound)
                session.flush()
            if _ordered_exercises(session, session_row.id):
                session_row.source_record_id = None
            else:
                session.delete(session_row)
        else:
            session.delete(row)
        session.commit()
        return True




def _default_set_details(item: dict) -> list[dict]:
    raw_count = item.get("sets", 1)
    try:
        count = max(0, int(raw_count))
    except (TypeError, ValueError):
        count = 1
    raw_details = item.get("set_details")
    details = raw_details if isinstance(raw_details, list) else []
    result = []
    for index in range(count):
        detail = details[index] if index < len(details) and isinstance(details[index], dict) else {}
        result.append({"weight_kg": float(detail.get("weight_kg", item.get("weight_kg", 0)) or 0), "reps": int(detail.get("reps", item.get("reps", 1)) or 1), "difficulty": str(detail.get("difficulty", item.get("difficulty", "未记录")) or "未记录"), "completed": bool(detail.get("completed", item.get("completed", True))), "note": str(detail.get("note", item.get("note", "")) or "")})
    return result

def _normalize_workout(item: dict) -> dict:
    normalized = {**item}
    normalized["difficulty"] = normalized.get("difficulty", "未记录") or "未记录"
    normalized["completed"] = bool(normalized.get("completed", True))
    normalized["set_details"] = _default_set_details(normalized)
    return normalized

def _parse_set_details(raw: str | None, item: dict) -> list[dict]:
    try:
        parsed = json.loads(raw) if raw else None
    except (TypeError, json.JSONDecodeError):
        parsed = None
    item_with_details = {**item, "set_details": parsed}
    return _default_set_details(item_with_details)

def _serialize(row: WorkoutRecordModel) -> dict:
    base = {"id": row.id, "date": row.workout_date.isoformat(), "user_id": row.user_id, "exercise": row.exercise, "sets": row.sets, "reps": row.reps, "weight_kg": row.weight_kg, "difficulty": getattr(row, "difficulty", "未记录") or "未记录", "completed": bool(getattr(row, "completed", True)), "note": row.note, "idempotency_key": getattr(row, "idempotency_key", None)}
    details = _parse_set_details(getattr(row, "set_details", None), base)
    base["set_details"] = details
    # R012: read consumers must know whether the top-level weight/reps is a
    # uniform value or only the first set's compatibility summary.
    uniform = True
    if details:
        first = (details[0].get("weight_kg"), details[0].get("reps"))
        uniform = all((detail.get("weight_kg"), detail.get("reps")) == first for detail in details)
    base["summary_mode"] = "uniform" if uniform else "first_set"
    return base




