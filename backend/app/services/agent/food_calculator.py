from __future__ import annotations
from app.core import clock

import hashlib
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from threading import Lock
from typing import Any, Iterable

from app.db.food_repository import SEED_PORTIONS, log_food_items, portion_weight, resolve_food
from app.services.agent.food_parser import FoodInput


def _number(value: Decimal | float | int) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _idempotency_key(user_id: str, message: str, record_time: datetime) -> str:
    raw = f"{user_id}|{record_time.date().isoformat()}|{message.strip()}"
    return "food-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class PendingFoodRecord:
    entries: tuple[FoodInput, ...]
    raw_message: str
    idempotency_key: str | None = None
    meal_group_id: str | None = None
    created_at: float = 0.0


_PENDING_TTL_SECONDS = 15 * 60
_PENDING_LOCK = Lock()
_pending_food: dict[tuple[str, str], PendingFoodRecord] = {}


def remember_pending_food(
    user_id: str,
    conversation_id: str,
    entries: Iterable[FoodInput],
    *,
    raw_message: str,
    idempotency_key: str | None = None,
    meal_group_id: str | None = None,
) -> None:
    record = PendingFoodRecord(
        entries=tuple(entries),
        raw_message=raw_message,
        idempotency_key=idempotency_key,
        meal_group_id=meal_group_id,
        created_at=time.monotonic(),
    )
    with _PENDING_LOCK:
        _pending_food[(user_id, conversation_id)] = record


def get_pending_food(user_id: str, conversation_id: str) -> PendingFoodRecord | None:
    key = (user_id, conversation_id)
    with _PENDING_LOCK:
        record = _pending_food.get(key)
        if record is None:
            return None
        if time.monotonic() - record.created_at > _PENDING_TTL_SECONDS:
            _pending_food.pop(key, None)
            return None
        return record


def clear_pending_food(user_id: str, conversation_id: str, *,
                       idempotency_key: str | None = None) -> bool:
    """Consume the exact pending draft after a real successful save.

    A different/newer pending draft is never removed. Failure, unresolved or
    replay paths simply do not call this helper.
    """
    key = (user_id, conversation_id)
    with _PENDING_LOCK:
        record = _pending_food.get(key)
        if record is None:
            return False
        if idempotency_key is not None and record.idempotency_key != idempotency_key:
            return False
        _pending_food.pop(key, None)
        return True


def format_food_resolution_message(unresolved: Iterable[dict[str, Any]]) -> str:
    """Describe exactly which food input is missing or needs correction."""
    labels: list[str] = []
    has_no_entity = False
    for raw_item in unresolved:
        item = raw_item if isinstance(raw_item, dict) else {}
        reason = str(item.get("reason") or "")
        name = str(item.get("name") or "该项饮食")
        if reason == "no_food_entity":
            has_no_entity = True
            continue
        if reason == "food_not_found":
            label = f"{name}：可识别的食物名称"
        elif reason == "weight_not_found":
            label = f"{name}：重量（克）或可换算的份量（如1碗、2个）"
        elif reason == "weight_invalid":
            label = f"{name}：有效重量（大于0克）"
        else:
            label = f"{name}：食物名称和重量（克）或份量"
        if label not in labels:
            labels.append(label)
    if has_no_entity and not labels:
        return "缺少：食物名称，以及重量（克）或可识别的份量（如1碗、2个）。"
    if has_no_entity:
        labels.append("食物名称，以及重量（克）或可识别的份量")
    return "需要补充或修正：" + "；".join(dict.fromkeys(labels)) + "。" if labels else "缺少：食物名称，以及重量（克）或可识别的份量（如1碗、2个）。"


def _build_food_items(entries: Iterable[FoodInput], nutrition_overrides: dict | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for entry in entries:
        supplied = (nutrition_overrides or {}).get(entry.raw_text)
        if supplied is not None:
            fields = ("kcal", "protein_g", "carbs_g", "fat_g", "fiber_g")
            valid = isinstance(supplied, dict) and all(
                isinstance(supplied.get(k), (int, float)) and not isinstance(supplied.get(k), bool)
                and math.isfinite(supplied[k]) and 0 <= supplied[k] <= (1200 if k == "kcal" else 100)
                for k in fields)
            valid = valid and sum(supplied[k] for k in fields[1:]) <= 110
            weight = entry.weight_g
            if not valid or not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight) or not 0 < weight <= 100000:
                unresolved.append({"name": entry.requested_name, "reason": "weight_invalid"})
                continue
            factor = Decimal(str(weight)) / Decimal(100)
            items.append({"food_id": None, "raw_food_name": entry.raw_text,
                "food_name": entry.requested_name, "weight_g": _number(weight),
                "energy_kcal": _number(factor * Decimal(str(supplied["kcal"]))),
                "protein_g": _number(factor * Decimal(str(supplied["protein_g"]))),
                "carb_g": _number(factor * Decimal(str(supplied["carbs_g"]))),
                "fat_g": _number(factor * Decimal(str(supplied["fat_g"]))),
                "fiber_g": _number(factor * Decimal(str(supplied["fiber_g"]))),
                "is_estimated": True, "weight_source": "user_input",
                "match_method": "ai_estimate", "match_confidence": .65})
            continue
        match = resolve_food(entry.requested_name)
        if match is None:
            unresolved.append({"name": entry.requested_name, "reason": "food_not_found"})
            continue
        weight_g = entry.weight_g
        weight_source = "user_input"
        is_estimated = False
        if weight_g is None:
            per_portion = portion_weight(match.food_id, entry.portion_name or entry.unit)
            if per_portion is None:
                per_portion = float(SEED_PORTIONS.get(match.food_name, {}).get(entry.portion_name or entry.unit, 0)) or None
            if per_portion is None:
                unresolved.append({"name": entry.requested_name, "reason": "weight_not_found", "unit": entry.unit})
                continue
            weight_g = per_portion * entry.quantity
            weight_source = "portion_estimate"
            is_estimated = True
        weight = Decimal(str(weight_g))
        if weight <= 0 or weight > Decimal("100000"):
            unresolved.append({"name": entry.requested_name, "reason": "weight_invalid", "unit": entry.unit})
            continue
        factor = weight / Decimal("100")
        items.append({"food_id": match.food_id, "raw_food_name": entry.raw_text, "food_name": match.food_name, "weight_g": _number(weight), "energy_kcal": _number(factor * Decimal(str(match.energy_kcal))), "protein_g": _number(factor * Decimal(str(match.protein_g))), "carb_g": _number(factor * Decimal(str(match.carb_g))), "fat_g": _number(factor * Decimal(str(match.fat_g))), "fiber_g": _number(factor * Decimal(str(match.fiber_g))), "is_estimated": is_estimated, "weight_source": weight_source, "match_method": match.match_method, "match_confidence": _number(match.confidence)})
    return items, unresolved


def _totals(items: list[dict[str, Any]]) -> dict[str, float]:
    return {"kcal": _number(sum(Decimal(str(item["energy_kcal"])) for item in items)), "protein_g": _number(sum(Decimal(str(item["protein_g"])) for item in items)), "carb_g": _number(sum(Decimal(str(item["carb_g"])) for item in items)), "fat_g": _number(sum(Decimal(str(item["fat_g"])) for item in items)), "fiber_g": _number(sum(Decimal(str(item["fiber_g"])) for item in items))}


def calculate_manual_nutrition(
    user_id: str,
    foods: Iterable[dict[str, Any]],
    *,
    idempotency_key: str,
    meal_group_id: str,
    record_time: datetime | None = None,
    meal_type: str | None = None,
) -> dict[str, Any]:
    timestamp = record_time or clock.utc_now()
    items: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for raw in foods:
        name = str(raw.get("food_name") or "").strip()
        weight = Decimal(str(raw.get("weight_g") or 0))
        if not name:
            unresolved.append({"name": "该项饮食", "reason": "food_not_found"})
            continue
        if weight <= 0 or weight > Decimal("100000"):
            unresolved.append({"name": name, "reason": "weight_invalid"})
            continue
        supplied = raw.get("nutrition_per_100g")
        source = str(raw.get("source") or "custom")
        if supplied:
            factor = weight / Decimal("100")
            items.append({
                "food_id": None,
                "raw_food_name": name,
                "food_name": name,
                "weight_g": _number(weight),
                "energy_kcal": _number(factor * Decimal(str(supplied.get("kcal", 0)))),
                "protein_g": _number(factor * Decimal(str(supplied.get("protein_g", 0)))),
                "carb_g": _number(factor * Decimal(str(supplied.get("carbs_g", 0)))),
                "fat_g": _number(factor * Decimal(str(supplied.get("fat_g", 0)))),
                "fiber_g": _number(factor * Decimal(str(supplied.get("fiber_g", 0)))),
                "is_estimated": source == "ai",
                "weight_source": "user_input",
                "match_method": "ai_estimate" if source == "ai" else "user_custom",
                "match_confidence": 0.65 if source == "ai" else 1.0,
            })
            continue
        match = resolve_food(name)
        if match is None:
            unresolved.append({"name": name, "reason": "food_not_found"})
            continue
        factor = weight / Decimal("100")
        items.append({
            "food_id": match.food_id,
            "raw_food_name": name,
            "food_name": match.food_name,
            "weight_g": _number(weight),
            "energy_kcal": _number(factor * Decimal(str(match.energy_kcal))),
            "protein_g": _number(factor * Decimal(str(match.protein_g))),
            "carb_g": _number(factor * Decimal(str(match.carb_g))),
            "fat_g": _number(factor * Decimal(str(match.fat_g))),
            "fiber_g": _number(factor * Decimal(str(match.fiber_g))),
            "is_estimated": False,
            "weight_source": "user_input",
            "match_method": match.match_method,
            "match_confidence": _number(match.confidence),
        })
    if unresolved:
        return {"status": "needs_confirmation", "items": items, "unresolved": unresolved, "message": format_food_resolution_message(unresolved), "calculation_version": "food-calc-v1"}
    if not items:
        missing = [{"reason": "no_food_entity"}]
        return {"status": "needs_confirmation", "items": [], "unresolved": missing, "message": format_food_resolution_message(missing), "calculation_version": "food-calc-v1"}
    totals = _totals(items)
    ids = log_food_items(user_id, items, totals, meal_group_id, idempotency_key, timestamp, meal_type)
    return {"status": "calculated", "items": items, "totals": totals, "log_ids": ids, "meal_group_id": meal_group_id, "idempotency_key": idempotency_key, "meal_type": meal_type, "calculation_version": "food-calc-v1"}


def calculate_nutrition(
    user_id: str,
    entries: Iterable[FoodInput],
    *,
    raw_message: str,
    idempotency_key: str | None = None,
    meal_group_id: str | None = None,
    record_time: datetime | None = None,
    meal_type: str | None = None,
    nutrition_overrides: dict | None = None,
) -> dict[str, Any]:
    timestamp = record_time or clock.utc_now()
    group_id = meal_group_id or f"meal-{timestamp.strftime('%Y%m%d%H%M%S')}"
    key = idempotency_key or _idempotency_key(user_id, raw_message, timestamp)
    entries = tuple(entries)
    resolved_meal_type = meal_type or next((entry.meal_type for entry in entries if entry.meal_type), None)
    items, unresolved = _build_food_items(entries, nutrition_overrides)
    if unresolved:
        return {"status": "needs_confirmation", "items": items, "unresolved": unresolved, "message": format_food_resolution_message(unresolved), "calculation_version": "food-calc-v1"}
    if not items:
        unresolved = [{"reason": "no_food_entity"}]
        return {"status": "needs_confirmation", "items": [], "unresolved": unresolved, "message": format_food_resolution_message(unresolved), "calculation_version": "food-calc-v1"}
    totals = _totals(items)
    ids = log_food_items(user_id, items, totals, group_id, key, timestamp, resolved_meal_type)
    return {"status": "calculated", "items": items, "totals": totals, "log_ids": ids, "meal_group_id": group_id, "idempotency_key": key, "meal_type": resolved_meal_type, "calculation_version": "food-calc-v1"}


def calculate_nutrition_batch(
    user_id: str,
    groups: Iterable[dict[str, Any]],
    *,
    raw_message: str,
    idempotency_key: str | None = None,
    meal_group_id: str | None = None,
    record_time: datetime | None = None,
    nutrition_overrides: dict | None = None,
) -> dict[str, Any]:
    """Calculate several meal groups atomically from the caller's perspective.

    Every group is validated before any row is written. This prevents a message
    containing two meals from saving lunch while dinner is still unresolved.
    """
    timestamp = record_time or clock.utc_now()
    normalized_groups = [(str(group.get("meal_type") or "其他"), tuple(FoodInput(**entry) if isinstance(entry, dict) else entry for entry in group.get("entries", []))) for group in groups]
    normalized_groups = [(meal_type, entries) for meal_type, entries in normalized_groups if entries]
    if not normalized_groups:
        unresolved = [{"reason": "no_food_entity"}]
        return {"status": "needs_confirmation", "items": [], "unresolved": unresolved, "message": format_food_resolution_message(unresolved), "calculation_version": "food-calc-v1"}

    prepared: list[tuple[str, tuple[FoodInput, ...], list[dict[str, Any]], dict[str, float]]] = []
    unresolved: list[dict[str, Any]] = []
    all_items: list[dict[str, Any]] = []
    for meal_type, entries in normalized_groups:
        items, group_unresolved = _build_food_items(entries, nutrition_overrides)
        prepared.append((meal_type, entries, items, _totals(items)))
        all_items.extend(items)
        unresolved.extend([{**item, "meal_type": meal_type} for item in group_unresolved])
    if unresolved:
        return {"status": "needs_confirmation", "items": all_items, "unresolved": unresolved, "message": format_food_resolution_message(unresolved), "calculation_version": "food-calc-v1"}

    base_key = idempotency_key or _idempotency_key(user_id, raw_message, timestamp)
    base_group = meal_group_id or f"meal-{timestamp.strftime('%Y%m%d%H%M%S')}"
    calculations: list[dict[str, Any]] = []
    all_ids: list[int] = []
    for index, (meal_type, _entries, items, totals) in enumerate(prepared):
        group_id = f"{base_group}-{index + 1}"
        key = f"{base_key}:{index + 1}"
        ids = log_food_items(user_id, items, totals, group_id, key, timestamp, meal_type)
        all_ids.extend(ids)
        calculations.append({"status": "calculated", "items": items, "totals": totals, "log_ids": ids, "meal_group_id": group_id, "idempotency_key": key, "meal_type": meal_type, "calculation_version": "food-calc-v1"})
    return {"status": "calculated", "items": all_items, "totals": _totals(all_items), "log_ids": all_ids, "calculations": calculations, "meal_group_ids": [item["meal_group_id"] for item in calculations], "idempotency_key": base_key, "meal_type": "多餐次", "calculation_version": "food-calc-v1"}
