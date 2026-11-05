from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import bindparam, text

from app.db import session as db_session
from app.db.storage import BusinessValidationError, require_ready, storage_session
from app.db.food_repository import CALCULATION_VERSION, ensure_food_schema, resolve_food


def _number(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def update_food_meal(user_id: str, meal_group_id: str, meal_type: str | None = None, items: list[dict[str, Any]] | None = None, record_time: datetime | None = None) -> dict | None:
    require_ready()
    with storage_session() as session:
        rows = session.execute(text("""SELECT l.id, l.food_id, l.food_name, l.match_method, l.weight_g, l.meal_type, l.record_time,
            COALESCE(f.energy_kcal, CASE WHEN l.weight_g>0 THEN l.total_kcal*100/l.weight_g ELSE 0 END) AS energy_kcal,
            COALESCE(f.protein_g, CASE WHEN l.weight_g>0 THEN l.total_protein*100/l.weight_g ELSE 0 END) AS protein_g,
            COALESCE(f.carb_g, CASE WHEN l.weight_g>0 THEN l.total_carb*100/l.weight_g ELSE 0 END) AS carb_g,
            COALESCE(f.fat_g, CASE WHEN l.weight_g>0 THEN l.total_fat*100/l.weight_g ELSE 0 END) AS fat_g,
            COALESCE(f.fiber_g, CASE WHEN l.weight_g>0 THEN l.total_fiber*100/l.weight_g ELSE 0 END) AS fiber_g
            FROM user_food_log l LEFT JOIN food_nutrition f ON f.id=l.food_id
            WHERE l.user_id=:user_id AND l.meal_group_id=:group_id ORDER BY l.id"""), {"user_id": user_id, "group_id": meal_group_id}).mappings().all()
        if not rows:
            return None
        existing = {int(row["id"]): row for row in rows}
        requested_items = items if items is not None else [{"id": int(row["id"]), "weight_g": row["weight_g"]} for row in rows]
        kept_ids: set[int] = set()
        updates: list[tuple[Any, Decimal, Any | None, str | None]] = []
        additions: list[tuple[Any | None, str, Decimal, dict[str, Any] | None, str]] = []
        matches = {}

        def resolve_once(name):
            if name not in matches:
                matches[name] = resolve_food(name)
            return matches[name]

        for item in requested_items:
            raw_id = item.get("id")
            try:
                weight = Decimal(str(item.get("weight_g", 0)))
            except (ArithmeticError, TypeError, ValueError) as exc:
                raise BusinessValidationError("食物重量格式无效") from exc
            if weight <= 0 or weight > 100000:
                raise BusinessValidationError("食物重量必须在 0 到 100000 克之间")
            if raw_id is not None:
                row_id = int(raw_id)
                if row_id in kept_ids:
                    raise BusinessValidationError("同一食物明细不能重复提交")
                row = existing.get(row_id)
                if row is None:
                    raise BusinessValidationError("存在不属于该餐次的食物明细")
                kept_ids.add(row_id)
                requested_name = str(item.get("food_name") or "").strip() or None
                same_name = requested_name is not None and requested_name == str(row["food_name"] or "")
                match = resolve_once(requested_name) if requested_name and not same_name else None
                if requested_name and match is None:
                    # Custom and AI rows have no food_nutrition foreign key.
                    # Keeping their name while changing weight reuses the
                    # stored per-100g values instead of requiring local lookup.
                    if not same_name:
                        raise BusinessValidationError(f"食物库暂时没有匹配“{requested_name}”，请换一个名称")
                updates.append((row, weight, match, requested_name))
                continue
            food_name = str(item.get("food_name") or "").strip()
            if not food_name:
                raise BusinessValidationError("新增食物名称不能为空")
            supplied = item.get("nutrition_per_100g")
            source = str(item.get("source") or "local")
            if supplied:
                additions.append((None, food_name, weight, supplied, source))
                continue
            match = resolve_once(food_name)
            if match is None:
                raise BusinessValidationError(f"食物库暂时没有匹配“{food_name}”，可改用自定义或 AI 营养数据")
            additions.append((match, food_name, weight, None, "local"))

        selected_meal_type = meal_type if meal_type is not None else rows[0]["meal_type"]
        deleted_ids = list(set(existing) - kept_ids) if items is not None else []
        if deleted_ids:
            session.execute(text("DELETE FROM user_food_log WHERE id IN :ids AND user_id=:user_id AND meal_group_id=:group_id")
                            .bindparams(bindparam("ids", expanding=True)),
                            {"ids": deleted_ids, "user_id": user_id, "group_id": meal_group_id})
        update_params = []
        for row, weight, match, requested_name in updates:
            factor = weight / Decimal("100")
            if match is None:
                energy_kcal, protein_g, carb_g, fat_g, fiber_g = row["energy_kcal"], row["protein_g"], row["carb_g"], row["fat_g"], row["fiber_g"]
                # 未提交名称时保留原始名；显式提交（含自定义同名）时保存本次输入。
                food_params = {"food_id": None, "raw_food_name": requested_name, "food_name": None, "match_method": None, "match_confidence": None}
            else:
                energy_kcal, protein_g, carb_g, fat_g, fiber_g = match.energy_kcal, match.protein_g, match.carb_g, match.fat_g, match.fiber_g
                # raw_food_name 保存用户输入，food_name 保存食物库规范名。
                food_params = {"food_id": match.food_id, "raw_food_name": requested_name, "food_name": match.food_name, "match_method": match.match_method, "match_confidence": match.confidence}
            update_params.append({"meal_type": selected_meal_type, "weight": _number(weight), "kcal": _number(factor * Decimal(str(energy_kcal))), "protein": _number(factor * Decimal(str(protein_g))), "carb": _number(factor * Decimal(str(carb_g))), "fat": _number(factor * Decimal(str(fat_g))), "fiber": _number(factor * Decimal(str(fiber_g))), "id": int(row["id"]), "user_id": user_id, "group_id": meal_group_id, **food_params})
        if update_params:
            session.execute(text("""UPDATE user_food_log SET meal_type=:meal_type, food_id=COALESCE(:food_id, food_id), raw_food_name=COALESCE(:raw_food_name, raw_food_name), food_name=COALESCE(:food_name, food_name), weight_g=:weight, total_kcal=:kcal, total_protein=:protein, total_carb=:carb, total_fat=:fat, total_fiber=:fiber, is_estimated=FALSE, weight_source='user_input', match_method=COALESCE(:match_method, match_method), match_confidence=COALESCE(:match_confidence, match_confidence) WHERE id=:id AND user_id=:user_id AND meal_group_id=:group_id"""), update_params)
        insert_params = []
        for index, (match, raw_name, weight, supplied, source) in enumerate(additions):
            factor = weight / Decimal("100")
            if supplied:
                food_id = None
                saved_name = raw_name
                energy_kcal = supplied.get("kcal", 0)
                protein_g = supplied.get("protein_g", 0)
                carb_g = supplied.get("carbs_g", 0)
                fat_g = supplied.get("fat_g", 0)
                fiber_g = supplied.get("fiber_g", 0)
                estimated = source == "ai"
                match_method = "ai_estimate" if estimated else "user_custom"
                match_confidence = 0.65 if estimated else 1.0
            else:
                food_id = match.food_id
                saved_name = match.food_name
                energy_kcal, protein_g, carb_g, fat_g, fiber_g = match.energy_kcal, match.protein_g, match.carb_g, match.fat_g, match.fiber_g
                estimated = False
                match_method = match.match_method
                match_confidence = match.confidence
            insert_params.append({"user_id": user_id, "meal_group_id": meal_group_id, "record_time": rows[0]["record_time"], "meal_type": selected_meal_type, "food_id": food_id, "raw_food_name": raw_name, "food_name": saved_name, "weight": _number(weight), "kcal": _number(factor * Decimal(str(energy_kcal))), "protein": _number(factor * Decimal(str(protein_g))), "carb": _number(factor * Decimal(str(carb_g))), "fat": _number(factor * Decimal(str(fat_g))), "fiber": _number(factor * Decimal(str(fiber_g))), "estimated": estimated, "match_method": match_method, "match_confidence": match_confidence, "idempotency_key": f"meal-edit-{str(meal_group_id)[:60]}-{uuid4().hex}-{index}", "calculation_version": CALCULATION_VERSION})
        if insert_params:
            session.execute(text("""INSERT INTO user_food_log (user_id, meal_group_id, record_time, meal_type, food_id, raw_food_name, food_name, weight_g, total_kcal, total_protein, total_carb, total_fat, total_fiber, is_estimated, weight_source, match_method, match_confidence, idempotency_key, calculation_version) VALUES (:user_id, :meal_group_id, :record_time, :meal_type, :food_id, :raw_food_name, :food_name, :weight, :kcal, :protein, :carb, :fat, :fiber, :estimated, 'user_input', :match_method, :match_confidence, :idempotency_key, :calculation_version)"""), insert_params)
        if record_time is not None:
            session.execute(text("UPDATE user_food_log SET record_time=:record_time WHERE user_id=:user_id AND meal_group_id=:group_id"), {"record_time": record_time, "user_id": user_id, "group_id": meal_group_id})
        session.commit()
        return {"updated": len(updates) + len(additions), "deleted": len(set(existing) - kept_ids) if items is not None else 0, "meal_group_id": meal_group_id, "meal_type": selected_meal_type}


def delete_food_meal(user_id: str, meal_group_id: str) -> dict | None:
    require_ready()
    with storage_session() as session:
        result = session.execute(text("DELETE FROM user_food_log WHERE user_id=:user_id AND meal_group_id=:group_id"), {"user_id": user_id, "group_id": meal_group_id})
        session.commit()
        if result.rowcount:
            return {"deleted": result.rowcount, "meal_group_id": meal_group_id}
        return None
