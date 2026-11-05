import json
from pathlib import Path

from sqlalchemy import select

from app.db import session as db_session
from app.db.storage import require_ready, storage_session
from app.db.models import UserProfileModel


DEFAULT_PROFILE = {"user_id": "demo-user", "display_name": "我的训练", "goal": "保持健康", "experience": "初学者", "available_days": 3, "body_weight_kg": None, "target_weight_kg": None, "height_cm": None, "body_fat_pct": None}


def get_profile(user_id: str = "demo-user") -> dict:
    require_ready()
    with storage_session() as session:
        row = session.scalars(select(UserProfileModel).where(UserProfileModel.user_id == user_id)).first()
        if row:
            return _serialize(row)
    return {**DEFAULT_PROFILE, "user_id": user_id}


def get_stored_profile(user_id: str = "demo-user") -> dict | None:
    """Return only a real stored row; never project system defaults as user data."""
    require_ready()
    with storage_session() as session:
        row = session.scalars(select(UserProfileModel).where(UserProfileModel.user_id == user_id)).first()
        return _serialize(row) if row else None


def save_profile(user_id: str, values: dict) -> dict:
    item = {**get_profile(user_id), **values, "user_id": user_id}
    require_ready()
    with storage_session() as session:
        row = session.get(UserProfileModel, user_id)
        if row is None:
            row = UserProfileModel(user_id=user_id)
            session.add(row)
        for key in ("display_name", "goal", "experience", "available_days", "body_weight_kg", "target_weight_kg", "height_cm", "body_fat_pct"):
            setattr(row, key, item.get(key))
        session.commit()
        session.refresh(row)
        return _serialize(row)


def _serialize(row: UserProfileModel) -> dict:
    return {"user_id": row.user_id, "display_name": row.display_name, "goal": row.goal, "experience": row.experience, "available_days": row.available_days, "body_weight_kg": row.body_weight_kg, "target_weight_kg": row.target_weight_kg, "height_cm": row.height_cm, "body_fat_pct": row.body_fat_pct}




