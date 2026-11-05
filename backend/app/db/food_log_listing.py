from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import bindparam, text

from app.db import session as db_session
from app.db.storage import require_ready, storage_session
from app.db.food_repository import ensure_food_schema


def list_food_logs(user_id: str = "demo-user", limit: int = 30, start_date: date | None = None, end_date: date | None = None,
                   *, start_at: datetime | None = None, end_at: datetime | None = None) -> list[dict[str, Any]]:
    """List bounded meal groups for one owner.

    ``start_at``/``end_at`` are exact UTC instants for callers that need a
    local-calendar window. The older date parameters keep their existing UTC
    midnight semantics for compatibility.
    """
    limit = max(1, min(int(limit), 100))
    require_ready()
    with storage_session() as session:
        filters = ["l.user_id=:user_id"]
        params: dict[str, Any] = {"user_id": user_id}
        if start_at is not None:
            filters.append("l.record_time >= :start_at")
            params["start_at"] = start_at
        elif start_date:
            filters.append("l.record_time >= :start_time")
            params["start_time"] = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
        if end_at is not None:
            filters.append("l.record_time < :end_at")
            params["end_at"] = end_at
        elif end_date:
            filters.append("l.record_time < :end_time")
            params["end_time"] = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc)
        params["group_limit"] = limit
        rows = session.execute(text(f"""WITH recent_groups AS (
            SELECT meal_group_id, MAX(record_time) AS latest
            FROM user_food_log l
            WHERE {' AND '.join(filters)}
            GROUP BY meal_group_id
            ORDER BY latest DESC
            LIMIT :group_limit
        )
        SELECT l.id, l.meal_group_id, l.record_time, l.meal_type, l.raw_food_name, l.food_name, l.food_id, l.match_method, l.weight_g, l.total_kcal, l.total_protein, l.total_carb, l.total_fat, l.total_fiber
        FROM user_food_log l JOIN recent_groups g ON g.meal_group_id=l.meal_group_id
        WHERE {' AND '.join(filters)}
        ORDER BY l.record_time DESC, l.id DESC"""), params).mappings().all()
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row["meal_group_id"])].append(row)
        ordered = sorted(groups.values(), key=lambda group: group[0]["record_time"], reverse=True)[:limit]
        return [_serialize_group(group) for group in ordered]


def _serialize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    timestamp = first["record_time"]
    return {"id": str(first["meal_group_id"]), "date": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp), "meal_type": first.get("meal_type") or _meal_type(timestamp), "description": "、".join(str(row["raw_food_name"]) for row in rows), "calories": round(sum(float(row["total_kcal"]) for row in rows), 1), "protein_g": round(sum(float(row["total_protein"]) for row in rows), 1), "carbs_g": round(sum(float(row["total_carb"]) for row in rows), 1), "fat_g": round(sum(float(row["total_fat"]) for row in rows), 1), "source": "ai_food_log", "items": [{"id": int(row["id"]), "food_name": row["food_name"], "raw_food_name": row["raw_food_name"], "weight_g": float(row["weight_g"]), "calories": round(float(row["total_kcal"]), 2), "protein_g": round(float(row["total_protein"]), 2), "carbs_g": round(float(row["total_carb"]), 2), "fat_g": round(float(row["total_fat"]), 2), "fiber_g": round(float(row["total_fiber"]), 2), "source": "custom" if row.get("food_id") is None and row.get("match_method") == "user_custom" else "ai" if row.get("food_id") is None and row.get("match_method") == "ai_estimate" else "local", "match_method": row.get("match_method")} for row in rows]}


def query_food_log_sources(user_id: str, *, start_at: datetime, end_at: datetime,
                           meal_types: list[str] | None = None,
                           food_names: list[str] | None = None,
                           group_limit: int = 30) -> tuple[list[dict[str, Any]], bool]:
    """Bounded source rows for a structured saved-food query.

    Owner/date/meal/name predicates run in SQL before the group LIMIT. Only the
    columns needed to render date/meal/name/grams and application-side nutrition
    totals are selected. ``truncated`` is true when at least one more matching
    group exists beyond ``group_limit``.
    """
    user_id = str(user_id or "demo-user").strip() or "demo-user"
    group_limit = max(1, min(int(group_limit), 100))
    row_limit = group_limit + 1
    from app.db.meal_types import storage_meal_types
    meal_types = storage_meal_types(meal_types or [])
    require_ready()
    with storage_session() as session:
        dialect = session.get_bind().dialect.name
        filters = ["l.user_id=:user_id"]
        params: dict[str, Any] = {"user_id": user_id, "row_limit": row_limit}
        if dialect == "sqlite":
            session.connection().connection.driver_connection.create_function(
                "fitzen_food_epoch_us", 1, _food_epoch_us, deterministic=True)
            filters.append("fitzen_food_epoch_us(l.record_time) >= :start_us")
            filters.append("fitzen_food_epoch_us(l.record_time) < :end_us")
            params["start_us"] = _food_epoch_us(start_at)
            params["end_us"] = _food_epoch_us(end_at)
            order = "fitzen_food_epoch_us(l.record_time) DESC, l.id DESC"
        else:
            filters.append("l.record_time >= :start_at")
            filters.append("l.record_time < :end_at")
            params["start_at"] = start_at
            params["end_at"] = end_at
            order = "l.record_time DESC, l.id DESC"
        if meal_types:
            filters.append("l.meal_type IN :meal_types")
            params["meal_types"] = list(meal_types)
        if food_names:
            filters.append("(l.raw_food_name IN :food_names OR l.food_name IN :food_names)")
            params["food_names"] = list(food_names)
        query = text(f"""SELECT l.id, l.meal_group_id, l.record_time, l.meal_type,
               l.raw_food_name, l.food_name, l.weight_g,
               l.total_kcal, l.total_protein, l.total_carb, l.total_fat, l.total_fiber
        FROM user_food_log l
        WHERE {' AND '.join(filters)}
        ORDER BY {order}
        LIMIT :row_limit""")
        if meal_types:
            query = query.bindparams(bindparam("meal_types", expanding=True))
        if food_names:
            query = query.bindparams(bindparam("food_names", expanding=True))
        rows = [dict(row) for row in session.execute(query, params).mappings().all()]
        truncated = len(rows) > group_limit
        rows = rows[:group_limit]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row["meal_group_id"])].append(row)
        return list(groups.values()), truncated


def food_record_datetime(value: datetime | str) -> datetime:
    """Raw SQL returns datetime on PostgreSQL and ISO text on SQLite.

    Food timestamps are stored in UTC; do not interpret a naive stored value
    using the machine's local timezone. Invalid data still fails the read.
    """
    from datetime import timezone
    timestamp = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    return timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)


def food_cutoff_filter(column: str, dialect_name: str, *, parameter: str = "cutoff") -> str:
    """Candidate SQL predicate for ``column >= parameter``.

    PostgreSQL compares ``TIMESTAMPTZ`` values directly. SQLite can contain the
    same instant as ISO text using ``T`` or a space and an explicit UTC offset,
    so a textual comparison is not chronological. ``julianday`` parses those
    forms; the one-second margin absorbs its floating-point rounding. Callers must
    still confirm each candidate with ``food_record_at_or_after`` so the final
    boundary is exact rather than truncated to a coarser SQLite date function.
    """
    if dialect_name == "sqlite":
        return f"julianday({column}) >= julianday(:{parameter}) - 1.0/86400"
    return f"{column} >= :{parameter}"


def food_range_candidates(column: str, dialect_name: str, *,
                          start_parameter: str = "start",
                          end_parameter: str = "end") -> str:
    """Candidate SQL predicate for ``start <= column < end``.

    The SQLite range is deliberately widened by one second. Production callers
    apply ``food_record_in_range`` to the returned rows for the exact decision.
    """
    if dialect_name == "sqlite":
        return (f"julianday({column}) >= julianday(:{start_parameter}) - 1.0/86400 "
                f"AND julianday({column}) < julianday(:{end_parameter}) + 1.0/86400")
    return f"{column} >= :{start_parameter} AND {column} < :{end_parameter}"


def food_record_at_or_after(value: datetime | str, cutoff: datetime) -> bool:
    return food_record_datetime(value) >= cutoff


def food_record_in_range(value: datetime | str, start: datetime, end: datetime) -> bool:
    timestamp = food_record_datetime(value)
    return start <= timestamp < end


def _food_epoch_us(value: datetime | str | None) -> int | None:
    if value is None:
        return None
    delta = food_record_datetime(value).astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def query_food_catalog(session, user_id: str, cutoff: datetime, limit: int,
                       query: str = "", *, by_category: bool = True) -> list[dict[str, Any]]:
    """Aggregate and limit in SQL; SQLite timestamps retain microsecond precision.

    The connection-local function handles historical mixed ISO encodings without
    rewriting stored values or rounding the cutoff through julianday/datetime.
    """
    dialect = session.get_bind().dialect.name
    params = {"user_id": user_id, "cutoff": cutoff, "limit": limit}
    predicate = food_cutoff_filter("l.record_time", dialect)
    last_used = "l.record_time"
    if dialect == "sqlite":
        session.connection().connection.driver_connection.create_function(
            "fitzen_food_epoch_us", 1, _food_epoch_us, deterministic=True)
        params["cutoff_us"] = _food_epoch_us(cutoff)
        predicate += " AND fitzen_food_epoch_us(l.record_time) >= :cutoff_us"
        last_used = "fitzen_food_epoch_us(l.record_time)"
    where = "f.is_active=TRUE"
    if query:
        where += " AND (lower(replace(f.food_name,' ','')) LIKE :query OR lower(f.category) LIKE :query)"
        params["query"] = f"%{query}%"
    secondary_order = "f.category, f.food_name" if by_category else "f.food_name"
    identity_fields = "f.id AS id, f.category, f.preparation_state," if by_category else ""
    rows = session.execute(text(f"""SELECT {identity_fields} f.food_name AS name,
            f.energy_kcal AS kcal, f.protein_g AS protein,
            f.carb_g AS carbs, f.fat_g AS fat, f.fiber_g AS fiber,
            COUNT(l.id) AS usage_count
        FROM food_nutrition f
        LEFT JOIN user_food_log l ON l.food_id=f.id AND l.user_id=:user_id AND {predicate}
        WHERE {where}
        GROUP BY f.id
        ORDER BY COUNT(l.id) DESC, MAX({last_used}) DESC NULLS LAST, {secondary_order}
        LIMIT :limit"""), params).mappings().all()
    return [dict(row) for row in rows]


def food_catalog_from_rows(rows, cutoff: datetime, limit: int) -> list[dict[str, Any]]:
    """Aggregate catalog usage exactly from SQLite candidate rows.

    Each row must expose the catalog fields plus ``record_time``. This keeps
    the cutoff decision at the stored microsecond precision and supports the
    ISO text forms SQLite may contain.
    """
    catalog: dict[int, dict[str, Any]] = {}
    for row in rows:
        food_id = int(row["id"])
        item = catalog.get(food_id)
        if item is None:
            item = {
                "id": food_id,
                "name": row["name"],
                "category": row.get("category"),
                "preparation_state": row.get("preparation_state"),
                "kcal": row["kcal"],
                "protein": row["protein"],
                "carbs": row["carbs"],
                "fat": row["fat"],
                "fiber": row["fiber"],
                "usage_count": 0,
                "_last_used": None,
            }
            catalog[food_id] = item
        record_time = row.get("record_time")
        if record_time is None:
            continue
        timestamp = food_record_datetime(record_time)
        if timestamp < cutoff:
            continue
        item["usage_count"] += 1
        if item["_last_used"] is None or timestamp > item["_last_used"]:
            item["_last_used"] = timestamp
    ordered = list(catalog.values())
    ordered.sort(key=lambda item: (item["category"] or "", item["name"]))
    ordered.sort(key=lambda item: item["_last_used"] or datetime.min.replace(tzinfo=timezone.utc),
                 reverse=True)
    ordered.sort(key=lambda item: item["usage_count"], reverse=True)
    for item in ordered:
        item.pop("_last_used", None)
    return ordered[:limit]


def _meal_type(timestamp: datetime) -> str:
    timestamp = food_record_datetime(timestamp)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(ZoneInfo("Asia/Shanghai"))
    hour = timestamp.hour
    if hour < 10:
        return "早餐"
    if hour < 12:
        return "早上加餐"
    if hour < 14:
        return "午餐"
    if hour < 17:
        return "下午加餐"
    if hour < 21:
        return "晚餐"
    return "晚上加餐"


