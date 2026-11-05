"""Manual offline import. Defaults to dry-run; never modifies input JSON."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import date
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.models import WorkoutRecordModel
from app.db.workout_session_models import WorkoutSessionModel
from app.db.workout_session_repository import _replace_exercises, _serialize, _session_signature


def insert_batch(session, kind, records, source_hash):
    inserted = 0
    for index, item in enumerate(records):
        if kind == "workouts":
            key = f"offline-{source_hash[:32]}-{index}"
            if session.scalar(select(WorkoutRecordModel).where(WorkoutRecordModel.user_id == item["user_id"], WorkoutRecordModel.idempotency_key == key)):
                continue
            fields = {k: item[k] for k in ("user_id", "exercise", "sets", "reps", "weight_kg", "note", "difficulty", "completed") if k in item}
            session.add(WorkoutRecordModel(**fields, workout_date=date.fromisoformat(item["date"][:10]), idempotency_key=key, set_details=json.dumps(item.get("set_details"), ensure_ascii=False) if item.get("set_details") is not None else None))
        else:
            candidates = session.scalars(select(WorkoutSessionModel).where(WorkoutSessionModel.user_id == item["user_id"], WorkoutSessionModel.session_date == date.fromisoformat(item["session_date"][:10]))).all()
            if any(_session_signature(_serialize(session, row)) == _session_signature(item) for row in candidates):
                continue
            row = WorkoutSessionModel(user_id=item["user_id"], session_date=date.fromisoformat(item["session_date"][:10]), title=item["title"], note=item.get("note", ""))
            session.add(row)
            session.flush()
            _replace_exercises(session, row.id, item["exercises"])
        inserted += 1
    session.flush()
    return inserted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["workouts", "workout_sessions"])
    parser.add_argument("source", type=Path)
    parser.add_argument("--apply", action="store_true", help="Commit imports after checking existing SQL records")
    args = parser.parse_args()
    raw = args.source.read_bytes()
    records = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError("Expected JSON record list")
    fingerprint = hashlib.sha256(raw).hexdigest()
    print(json.dumps({"count": len(records), "sha256": fingerprint, "apply": args.apply}))
    if not args.apply:
        return
    engine = create_engine(settings.database_url)
    try:
        with Session(engine) as session, session.begin():
            count = insert_batch(session, args.kind, records, fingerprint)
        print(json.dumps({"inserted": count}))
    finally:
        engine.dispose()

if __name__ == "__main__":
    main()
