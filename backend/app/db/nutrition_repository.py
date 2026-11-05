from app.core import clock
from datetime import date, timedelta
from contextlib import contextmanager
from contextvars import ContextVar

from sqlalchemy import func, select

from app.db import session as db_session
from app.db.storage import require_ready, storage_session
from app.db.models import NutritionLogModel
from app.db.profile_repository import get_profile




def list_nutrition(user_id: str = "demo-user", limit: int = 30, start_date: date | None = None, end_date: date | None = None) -> list[dict]:
    require_ready()
    with storage_session() as session:
        statement = select(NutritionLogModel).where(NutritionLogModel.user_id == user_id)
        if start_date:
            statement = statement.where(NutritionLogModel.log_date >= start_date)
        if end_date:
            statement = statement.where(NutritionLogModel.log_date <= end_date)
        rows = session.scalars(statement.order_by(NutritionLogModel.log_date.desc(), NutritionLogModel.id.desc()).limit(limit)).all()
        return [_serialize(row) for row in rows]


def create_nutrition(user_id: str, meal_type: str, description: str, calories: float | None = None, protein_g: float | None = None, carbs_g: float | None = None, fat_g: float | None = None, log_date: date | None = None) -> dict:
    # This legacy endpoint has no idempotency key: uncertain commits must not
    # be guessed successful or automatically retried by content similarity.
    with storage_session() as session:
        row = NutritionLogModel(log_date=log_date or clock.today(), meal_type=meal_type, description=description, calories=calories, protein_g=protein_g, carbs_g=carbs_g, fat_g=fat_g, user_id=user_id)
        session.add(row)
        session.commit()
        session.refresh(row)
        return _serialize(row)


def update_nutrition(user_id: str, record_id: int, record: dict) -> dict | None:
    require_ready()
    with storage_session() as session:
        row = session.scalar(select(NutritionLogModel).where(NutritionLogModel.id == record_id, NutritionLogModel.user_id == user_id))
        if row is None:
            return None
        for field in ("meal_type", "description", "calories", "protein_g", "carbs_g", "fat_g"):
            if field in record and record[field] is not None:
                setattr(row, field, record[field])
        if record.get("log_date"):
            row.log_date = date.fromisoformat(str(record["log_date"]))
        session.commit()
        session.refresh(row)
        return _serialize(row)


def delete_nutrition(user_id: str, record_id: int) -> bool:
    require_ready()
    with storage_session() as session:
        row = session.scalar(select(NutritionLogModel).where(NutritionLogModel.id == record_id, NutritionLogModel.user_id == user_id))
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True


_target_profile: ContextVar[tuple[str, dict] | None] = ContextVar("nutrition_target_profile", default=None)


@contextmanager
def nutrition_profile_context(user_id: str, profile: dict):
    """Reuse a trusted profile only during this context read, never across writes."""
    token = _target_profile.set((user_id, profile))
    try:
        yield
    finally:
        _target_profile.reset(token)


def nutrition_summary(user_id: str = "demo-user", days: int = 1) -> dict:
    require_ready()
    cutoff = clock.today() - timedelta(days=days - 1)
    fields = ("calories", "protein_g", "carbs_g", "fat_g")
    with storage_session() as session:
        row = session.execute(select(
            func.count(NutritionLogModel.id),
            *(func.coalesce(func.sum(getattr(NutritionLogModel, field)), 0) for field in fields),
        ).where(NutritionLogModel.user_id == user_id, NutritionLogModel.log_date >= cutoff)).one()
    result = {"user_id": user_id, "days": days, "entries": row[0],
              **{field: round(value, 1) for field, value in zip(fields, row[1:])}}
    result.update(_nutrition_targets(user_id))
    return result


def _nutrition_targets(user_id: str) -> dict:
    scoped = _target_profile.get()
    profile = scoped[1] if scoped is not None and scoped[0] == user_id else get_profile(user_id)
    weight = profile.get("body_weight_kg")
    if not weight:
        return {"target_ready": False, "calories_target": None, "protein_target_g": None, "carbs_target_g": None, "fat_target_g": None}
    goal = str(profile.get("goal", "")).lower()
    calorie_factor = 25 if any(term in goal for term in ("减脂", "减重", "瘦身")) else 33 if any(term in goal for term in ("增肌", "增重")) else 30
    calories = round(float(weight) * calorie_factor)
    protein = round(float(weight) * (1.8 if "增肌" in goal else 1.6), 1)
    fat = round(float(weight) * 0.8, 1)
    carbs = round(max((calories - protein * 4 - fat * 9) / 4, 0), 1)
    return {"target_ready": True, "calories_target": calories, "protein_target_g": protein, "carbs_target_g": carbs, "fat_target_g": fat}


def _serialize(row: NutritionLogModel) -> dict:
    return {"id": row.id, "date": row.log_date.isoformat(), "user_id": row.user_id, "meal_type": row.meal_type, "description": row.description, "calories": row.calories, "protein_g": row.protein_g, "carbs_g": row.carbs_g, "fat_g": row.fat_g, "source": "manual_nutrition_log"}

def create_nutrition_in_session(session, user_id: str, record: dict) -> dict:
    """C2-2 session primitive: add one manual nutrition row and flush only.

    The caller owns the transaction/connection; this never commits, rolls
    back or closes the session. Returns the real positive id of this
    transaction.
    """
    log_date = record.get("log_date")
    if isinstance(log_date, str):
        log_date = date.fromisoformat(log_date)
    row = NutritionLogModel(log_date=log_date, meal_type=record.get("meal_type"),
                            description=record.get("description"), calories=record.get("calories"),
                            protein_g=record.get("protein_g"), carbs_g=record.get("carbs_g"),
                            fat_g=record.get("fat_g"), user_id=user_id)
    session.add(row)
    session.flush()
    return {"created": True, "record_id": int(row.id), "record": _serialize(row)}
