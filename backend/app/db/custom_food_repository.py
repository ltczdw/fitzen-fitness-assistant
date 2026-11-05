from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db import session as db_session
from app.db.food_repository import ensure_food_schema


def _clean_name(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _row_to_food(row: Any) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "name": str(row["food_name"]),
        "grams": 100,
        "kcal": float(row["energy_kcal"]),
        "protein": float(row["protein_g"]),
        "carbs": float(row["carb_g"]),
        "fat": float(row["fat_g"]),
        "fiber": float(row["fiber_g"]),
        "usage_count": int(row["use_count"] or 0),
        "last_used_at": row["last_used_at"].isoformat() if isinstance(row["last_used_at"], datetime) else (str(row["last_used_at"]) if row["last_used_at"] else None),
    }


def list_custom_foods(user_id: str = "demo-user", search: str = "", limit: int = 30) -> list[dict[str, Any]]:
    if not (db_session.DB_READY and db_session.SessionLocal is not None and ensure_food_schema()):
        return []
    user_id = str(user_id or "demo-user").strip() or "demo-user"
    search = str(search or "").strip()
    limit = max(1, min(int(limit), 200))
    try:
        with db_session.SessionLocal() as session:
            rows = session.execute(
                text(
                    """
                    SELECT id, food_name, energy_kcal, protein_g, carb_g, fat_g, fiber_g,
                           use_count, last_used_at
                    FROM user_custom_food
                    WHERE user_id=:user_id AND is_active=TRUE
                      AND (:search='' OR food_name ILIKE :pattern)
                    ORDER BY use_count DESC, last_used_at DESC NULLS LAST,
                             updated_at DESC, food_name ASC
                    LIMIT :limit
                    """
                ),
                {"user_id": user_id, "search": search, "pattern": f"%{search}%", "limit": limit},
            ).mappings().all()
            return [_row_to_food(row) for row in rows]
    except Exception:
        return []


def upsert_custom_food(
    user_id: str,
    food_name: str,
    *,
    kcal: float,
    protein_g: float,
    carbs_g: float,
    fat_g: float,
    fiber_g: float = 0,
) -> dict[str, Any]:
    if not (db_session.DB_READY and db_session.SessionLocal is not None and ensure_food_schema()):
        raise RuntimeError("数据库暂时不可用，无法保存自定义食物")
    name = _clean_name(food_name)
    if not name:
        raise ValueError("自定义食物名称不能为空")
    values = {
        "user_id": str(user_id or "demo-user").strip() or "demo-user",
        "food_name": name,
        "kcal": max(0.0, float(kcal)),
        "protein_g": max(0.0, float(protein_g)),
        "carbs_g": max(0.0, float(carbs_g)),
        "fat_g": max(0.0, float(fat_g)),
        "fiber_g": max(0.0, float(fiber_g)),
    }
    with db_session.SessionLocal() as session:
        row = session.execute(
            text(
                """
                INSERT INTO user_custom_food
                    (user_id, food_name, energy_kcal, protein_g, carb_g, fat_g, fiber_g, updated_at, is_active)
                VALUES
                    (:user_id, :food_name, :kcal, :protein_g, :carbs_g, :fat_g, :fiber_g, CURRENT_TIMESTAMP, TRUE)
                ON CONFLICT (user_id, food_name) DO UPDATE SET
                    energy_kcal=EXCLUDED.energy_kcal,
                    protein_g=EXCLUDED.protein_g,
                    carb_g=EXCLUDED.carb_g,
                    fat_g=EXCLUDED.fat_g,
                    fiber_g=EXCLUDED.fiber_g,
                    updated_at=CURRENT_TIMESTAMP,
                    is_active=TRUE
                RETURNING id, food_name, energy_kcal, protein_g, carb_g, fat_g, fiber_g, use_count, last_used_at
                """
            ),
            values,
        ).mappings().one()
        session.commit()
        return _row_to_food(row)


def touch_custom_food(user_id: str, food_name: str) -> None:
    if not (db_session.DB_READY and db_session.SessionLocal is not None and ensure_food_schema()):
        return
    name = _clean_name(food_name)
    if not name:
        return
    try:
        with db_session.SessionLocal() as session:
            session.execute(
                text(
                    """
                    UPDATE user_custom_food
                    SET use_count=use_count+1, last_used_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
                    WHERE user_id=:user_id AND food_name=:food_name AND is_active=TRUE
                    """
                ),
                {"user_id": str(user_id or "demo-user").strip() or "demo-user", "food_name": name},
            )
            session.commit()
    except Exception:
        return

def delete_custom_food(food_id: int, user_id: str = "demo-user") -> bool:
    """Hide a library entry without changing saved nutrition snapshots."""
    if not (db_session.DB_READY and db_session.SessionLocal is not None and ensure_food_schema()):
        raise RuntimeError("数据库暂时不可用，无法删除食物")
    try:
        with db_session.SessionLocal() as session:
            row = session.execute(
                text("UPDATE user_custom_food SET is_active=FALSE WHERE id=:food_id AND user_id=:user_id RETURNING id"),
                {"food_id": food_id, "user_id": str(user_id or "demo-user").strip() or "demo-user"},
            ).first()
            session.commit()
            return row is not None
    except SQLAlchemyError as exc:
        raise RuntimeError("删除食物失败，请稍后重试") from exc
