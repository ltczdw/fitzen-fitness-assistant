from __future__ import annotations
from app.core import clock

import json
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import text

from app.core.config import settings
from app.db import session as db_session
from app.db.storage import require_ready, storage_session
from app.db.food_log_listing import food_cutoff_filter, food_record_at_or_after
from app.db.food_repository import ensure_food_schema


def _summary_result(user_id: str, days: int, entries: int, kcal, protein, carb, fat, fiber) -> dict[str, Any]:
    return {"user_id": user_id, "days": days, "entries": int(entries),
            "calories": round(float(kcal), 1), "protein_g": round(float(protein), 1),
            "carbs_g": round(float(carb), 1), "fat_g": round(float(fat), 1),
            "fiber_g": round(float(fiber), 1), "source": "user_food_log"}


def food_nutrition_summary(user_id: str = "demo-user", days: int = 7) -> dict[str, Any]:
    """Aggregate precise food logs; the 7-day planning worker consumes this summary."""
    days = max(1, int(days))
    local_today = clock.now().date()
    start_local = datetime.combine(local_today - timedelta(days=days - 1), time.min, tzinfo=ZoneInfo(settings.app_timezone))
    cutoff = start_local.astimezone(timezone.utc)
    require_ready()
    with storage_session() as session:
        dialect_name = session.get_bind().dialect.name
        if dialect_name == "sqlite":
            rows = session.execute(text(
                f"""SELECT record_time, total_kcal, total_protein, total_carb, total_fat, total_fiber
                    FROM user_food_log
                    WHERE user_id=:user_id AND {food_cutoff_filter('record_time', dialect_name)}"""),
                {"user_id": user_id, "cutoff": cutoff}).mappings().all()
            included = [row for row in rows if food_record_at_or_after(row["record_time"], cutoff)]
            return _summary_result(
                user_id, days, len(included),
                sum(float(row["total_kcal"] or 0) for row in included),
                sum(float(row["total_protein"] or 0) for row in included),
                sum(float(row["total_carb"] or 0) for row in included),
                sum(float(row["total_fat"] or 0) for row in included),
                sum(float(row["total_fiber"] or 0) for row in included))
        row = session.execute(text(f"""SELECT COUNT(*) AS entries, COALESCE(SUM(total_kcal),0) AS kcal, COALESCE(SUM(total_protein),0) AS protein_g, COALESCE(SUM(total_carb),0) AS carb_g, COALESCE(SUM(total_fat),0) AS fat_g, COALESCE(SUM(total_fiber),0) AS fiber_g FROM user_food_log WHERE user_id=:user_id AND {food_cutoff_filter('record_time', dialect_name)}"""), {"user_id": user_id, "cutoff": cutoff}).mappings().one()
        return _summary_result(user_id, days, row["entries"], row["kcal"], row["protein_g"], row["carb_g"], row["fat_g"], row["fiber_g"])


