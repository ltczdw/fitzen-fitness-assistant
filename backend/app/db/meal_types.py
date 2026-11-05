"""Finite meal-type protocol values and their stored labels.

The existing API accepts both the UI's Chinese labels and the English protocol
values used by model output. Known protocol values are normalized to the labels
already written by the food writer; unknown/custom labels are preserved verbatim
so user-defined meal names still match exactly.
"""
from __future__ import annotations

MEAL_TYPE_STORAGE = {
    "breakfast": "早餐",
    "lunch": "午餐",
    "dinner": "晚餐",
    "snack": "加餐",
    "早餐": "早餐",
    "午餐": "午餐",
    "晚餐": "晚餐",
    "加餐": "加餐",
}


def storage_meal_type(value: str) -> str:
    text = str(value or "").strip()
    return MEAL_TYPE_STORAGE.get(text.casefold(), text)


def storage_meal_types(values) -> list[str]:
    result: list[str] = []
    for value in values or []:
        normalized = storage_meal_type(value)
        if normalized and normalized not in result:
            result.append(normalized)
    return result
