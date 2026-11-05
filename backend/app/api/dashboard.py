from app.core import clock
"""Database-only read model for the FitZen dashboard."""
from collections import defaultdict
from datetime import datetime, time, timedelta
import json
import logging
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select, text
from app.core.config import settings
from app.db import session as db
from app.db.models import UserProfileModel, WorkoutRecordModel, NutritionLogModel, TrainingPlanModel
from app.db.workout_session_models import WorkoutSessionModel, WorkoutExerciseModel, WorkoutSetModel
from app.db.food_log_listing import (
    _serialize_group,
    query_food_catalog,
    food_range_candidates,
    food_record_datetime,
    food_record_in_range,
)

router = APIRouter(tags=["dashboard"])
logger = logging.getLogger(__name__)


def serialize(row):
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


def aggregate(workouts, sessions, meals, start, days):
    daily = {(start + timedelta(days=i)).isoformat(): dict(
        date=(start + timedelta(days=i)).isoformat(), training_records=0,
        completed_sets=0, training_volume=0, cardio_distance_km=0, cardio_duration_minutes=0, meal_count=0,
        calories=None, protein_g=None, carbs_g=None, fat_g=None, tdee=None, difference=None)
        for i in range(days)}
    best = {}
    def add_sets(day, name, sets):
        for item in sets:
            if not item.get("completed", True):
                continue
            if item.get("duration_minutes") is not None:
                day["cardio_distance_km"] += float(item.get("distance_km") or 0)
                day["cardio_duration_minutes"] += float(item["duration_minutes"])
                continue
            weight, reps = float(item.get("weight_kg") or 0), int(item.get("reps") or 0)
            day["completed_sets"] += 1
            day["training_volume"] += weight * reps
            best[name] = max(best.get(name, 0), weight)
    for record in workouts:
        day = daily.get(str(record["date"])[:10])
        if day is None:
            continue
        day["training_records"] += 1
        details = record.get("set_details")
        if details is None:
            details = [record] * record["sets"]
        add_sets(day, record["exercise"], details)
    for record in sessions:
        day = daily.get(str(record["session_date"])[:10])
        if day is None:
            continue
        day["training_records"] += 1
        for exercise in record["exercises"]:
            add_sets(day, exercise["exercise_name"], exercise["sets"])
    grouped = defaultdict(list)
    for meal in meals:
        grouped[meal["date_key"]].append(meal)
    for key, day in daily.items():
        rows = grouped[key]
        day["meal_count"] = len(rows)
        day["training_volume"] = round(day["training_volume"], 1)
        for nutrient in ("calories", "protein_g", "carbs_g", "fat_g"):
            values = [row.get(nutrient) for row in rows]
            if values and all(value is not None for value in values):
                day[nutrient] = round(sum(float(value) for value in values), 1)
    return list(daily.values()), [{"exercise": k, "weight_kg": v} for k, v in sorted(best.items())]


def build_dashboard(session, user_id, days, today=None):
    zone = ZoneInfo(settings.app_timezone)
    today = today or clock.now().date()
    start = today - timedelta(days=days-1)
    profile_row = session.get(UserProfileModel, user_id)
    profile = serialize(profile_row) if profile_row else {"user_id": user_id}
    sessions = []
    session_map = {}
    for row in session.scalars(select(WorkoutSessionModel).where(
            WorkoutSessionModel.user_id == user_id, WorkoutSessionModel.session_date >= start,
            WorkoutSessionModel.session_date <= today).order_by(WorkoutSessionModel.session_date.desc(), WorkoutSessionModel.id.desc())):
        record = {**serialize(row), "session_date": row.session_date.isoformat(), "exercises": []}
        sessions.append(record)
        session_map[row.id] = record

    # Agent-created workout records are mirrored into workout_sessions so the
    # session editor can manage them. They are one business record with two
    # storage projections; use the session projection as the canonical read
    # model and do not count/render its source workout a second time.
    linked_workout_ids = {
        int(record["source_record_id"])
        for record in sessions
        if record.get("source_record_id") is not None
    }
    workouts = []
    for row in session.scalars(select(WorkoutRecordModel).where(
            WorkoutRecordModel.user_id == user_id, WorkoutRecordModel.workout_date >= start,
            WorkoutRecordModel.workout_date <= today).order_by(WorkoutRecordModel.workout_date.desc(), WorkoutRecordModel.id.desc())):
        if row.id in linked_workout_ids:
            continue
        record = serialize(row)
        record["date"] = row.workout_date.isoformat()
        record["set_details"] = json.loads(row.set_details) if row.set_details else None
        workouts.append(record)
    exercise_map = {}
    if session_map:
        for row in session.scalars(select(WorkoutExerciseModel).where(
                WorkoutExerciseModel.session_id.in_(session_map)).order_by(WorkoutExerciseModel.order_index, WorkoutExerciseModel.id)):
            record = {**serialize(row), "sets": []}
            exercise_map[row.id] = record
            session_map[row.session_id]["exercises"].append(record)
    if exercise_map:
        for row in session.scalars(select(WorkoutSetModel).where(
                WorkoutSetModel.exercise_id.in_(exercise_map)).order_by(WorkoutSetModel.set_index, WorkoutSetModel.id)):
            exercise_map[row.exercise_id]["sets"].append(serialize(row))
    meals = []
    for row in session.scalars(select(NutritionLogModel).where(
            NutritionLogModel.user_id == user_id, NutritionLogModel.log_date >= start,
            NutritionLogModel.log_date <= today)):
        meals.append({**serialize(row), "date": row.log_date.isoformat(),
                      "date_key": row.log_date.isoformat(), "source": "manual"})
    dialect_name = session.get_bind().dialect.name
    food_start = datetime.combine(start, time.min, zone)
    food_end = datetime.combine(today+timedelta(days=1), time.min, zone)
    food_time_filter = food_range_candidates(
        "record_time", dialect_name, start_parameter="start", end_parameter="end")
    food_rows = session.execute(text(
        f"SELECT * FROM user_food_log WHERE user_id=:user_id AND {food_time_filter} ORDER BY record_time,id"),
        {"user_id": user_id, "start": food_start, "end": food_end}).mappings().all()
    groups = defaultdict(list)
    for row in food_rows:
        if not food_record_in_range(row["record_time"], food_start, food_end):
            continue
        groups[row["meal_group_id"]].append(row)
    for rows in groups.values():
        record = _serialize_group(rows)
        record["date_key"] = food_record_datetime(rows[0]["record_time"]).astimezone(zone).date().isoformat()
        meals.append(record)
    meals.sort(key=lambda row: (row["date_key"], str(row["date"])), reverse=True)
    catalog_cutoff = clock.utc_now() - timedelta(days=90)
    catalog_fields = ("name", "kcal", "protein", "carbs", "fat", "fiber", "usage_count")
    catalog = [{key: item[key] for key in catalog_fields} for item in
               query_food_catalog(session, user_id, catalog_cutoff, 12, by_category=False)]
    plan_row = session.scalar(select(TrainingPlanModel).where(TrainingPlanModel.user_id == user_id))
    plan = None if plan_row is None else {**serialize(plan_row), "content": json.loads(plan_row.plan_json)}
    daily, best = aggregate(workouts, sessions, meals, start, days)
    return dict(source="postgresql", timezone=settings.app_timezone, user_id=user_id,
                start_date=start.isoformat(), end_date=today.isoformat(), days=days,
                fetched_at=clock.now().isoformat(), profile=profile, workouts=workouts,
                sessions=sessions, nutrition=meals, food_catalog=catalog, plan=plan,
                daily=daily, today=daily[-1], exercise_bests=best)


@router.get("/dashboard")
def read_dashboard(user_id: str = "demo-user", days: int = Query(365, ge=1, le=366)):
    try:
        if db.SessionLocal is None:
            raise RuntimeError("database not configured")
        with db.SessionLocal() as session:
            return build_dashboard(session, user_id, days)
    except Exception as exc:
        logger.error("Dashboard database read failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="数据库暂时无法读取，请稍后重试。") from exc
