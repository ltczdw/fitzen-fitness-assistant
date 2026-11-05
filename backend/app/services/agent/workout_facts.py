"""Shared rendering and identity for saved workout facts.

The legacy record table and the session tables expose different shapes.  Read
callers must render one line per action and keep source identity stable across
both shapes, so the formatting/identity rules live here instead of being
re-derived in each adapter.
"""
from __future__ import annotations

from typing import Any, Mapping


def display_number(value: Any) -> str:
    if value is None or value == "":
        return "未记录"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def workout_source_key(record: Mapping[str, Any], index: int) -> str:
    """Request-local selection key; not a cross-snapshot business identity.

    Legacy records keep ``workout:{id}:{index}``. Session actions use a
    distinct prefix so a session exercise id can never collide with a record
    id. Missing ids stay row-addressed and are never turned into a fabricated
    business id. Durable dedupe must use the repository ``source_id``
    (``record:{id}`` / ``session:{id}``), never this index-bearing key.
    """
    identity = record.get("id")
    if record.get("source") == "session":
        if identity is not None:
            return f"workout:session:{identity}:{index}"
        return f"workout:session:row:{index}"
    if identity is not None:
        return f"workout:{identity}:{index}"
    return f"workout:row:{index}"


def _summary_body(record: Mapping[str, Any]) -> str:
    return (
        display_number(record.get("weight_kg")) + " kg，"
        + display_number(record.get("sets")) + " 组，每组 "
        + display_number(record.get("reps")) + " 次"
    )


def _cardio_segment(detail: Mapping[str, Any]) -> str:
    duration = detail.get("duration_minutes")
    distance = detail.get("distance_km")
    if duration is not None and distance is not None:
        return display_number(duration) + " 分钟 / " + display_number(distance) + " 公里"
    if duration is not None:
        return display_number(duration) + " 分钟"
    return display_number(distance) + " 公里"


def _same_set(first: Mapping[str, Any], detail: Mapping[str, Any]) -> bool:
    return (
        detail.get("weight_kg") == first.get("weight_kg")
        and detail.get("reps") == first.get("reps")
        and detail.get("distance_km") == first.get("distance_km")
        and detail.get("duration_minutes") == first.get("duration_minutes")
        and bool(detail.get("completed", True)) == bool(first.get("completed", True))
    )


def workout_fact_line(record: Mapping[str, Any]) -> str:
    """One user-facing fact line; per-set values are never replaced by a summary."""
    exercise = str(record.get("exercise") or "未命名动作").strip()
    date = str(record.get("date") or record.get("workout_date") or "").strip()
    details = record.get("set_details")
    normalized = [item for item in details or [] if isinstance(item, dict)] \
        if isinstance(details, list) else []
    if not normalized:
        body = _summary_body(record)
    else:
        first = normalized[0]
        cardio = any(
            item.get("distance_km") is not None or item.get("duration_minutes") is not None
            for item in normalized)
        if not cardio and all(_same_set(first, item) for item in normalized):
            body = (
                display_number(first.get("weight_kg")) + " kg，"
                + str(len(normalized)) + " 组，每组 "
                + display_number(first.get("reps")) + " 次"
            )
        else:
            parts: list[str] = []
            for index, item in enumerate(normalized, start=1):
                if item.get("distance_km") is not None or item.get("duration_minutes") is not None:
                    segment = _cardio_segment(item)
                else:
                    segment = (
                        display_number(item.get("weight_kg")) + " kg × "
                        + display_number(item.get("reps")) + " 次"
                    )
                if item.get("completed", True) is False:
                    segment += "（未完成）"
                parts.append("第" + str(index) + "组 " + segment)
            body = str(len(normalized)) + " 组；" + "；".join(parts)
    return (date + " " if date else "") + exercise + "：" + body


__all__ = ["display_number", "workout_fact_line", "workout_source_key"]
