from __future__ import annotations

from typing import Any

from app.db.food_summary import food_nutrition_summary
from app.db.nutrition_repository import nutrition_summary


def synced_nutrition_summary(user_id: str = "demo-user", days: int = 7) -> dict[str, Any]:
    """Return one nutrition view for both legacy logs and precise food logs."""
    days = max(1, int(days))
    legacy = nutrition_summary(user_id, days)
    food = food_nutrition_summary(user_id, days)
    totals = {
        "calories": round(float(legacy.get("calories", 0)) + float(food.get("calories", 0)), 1),
        "protein_g": round(float(legacy.get("protein_g", 0)) + float(food.get("protein_g", 0)), 1),
        "carbs_g": round(float(legacy.get("carbs_g", 0)) + float(food.get("carbs_g", 0)), 1),
        "fat_g": round(float(legacy.get("fat_g", 0)) + float(food.get("fat_g", 0)), 1),
        "fiber_g": round(float(food.get("fiber_g", 0)), 1),
    }
    daily_average = {key: round(value / days, 1) for key, value in totals.items()}
    result = {
        **legacy,
        "days": days,
        "entries": int(legacy.get("entries", 0)) + int(food.get("entries", 0)),
        **totals,
        "daily_average": daily_average,
        "food_log": food,
        "sync_source": "nutrition_logs + user_food_log",
    }
    targets = {
        "calories": result.get("calories_target"),
        "protein_g": result.get("protein_target_g"),
        "carbs_g": result.get("carbs_target_g"),
        "fat_g": result.get("fat_target_g"),
    }
    result["target_gap_daily"] = {key: round(float(targets[key]) - daily_average[key], 1) if targets[key] is not None else None for key in targets}
    return result

