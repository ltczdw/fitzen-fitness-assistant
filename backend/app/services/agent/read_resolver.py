"""Application-owned reads for structured semantic read specifications.

The model may choose a business resource and the fields it needs; it never
supplies account identity, SQL, or stored values. This module binds the owner,
reads the real repository and renders the stored values. Missing data, missing
fields and read failures stay distinct.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from app.services.agent.read_contracts import (
    PROFILE_FIELD_LABELS, PROFILE_FIELD_NAMES, validate_read_spec,
)

# Backwards-compatible name for callers that only need the rendering registry.
PROFILE_FIELDS: dict[str, tuple[str, str]] = PROFILE_FIELD_LABELS
PLAN_FIELDS = {"plan_name", "goal", "days_per_week", "sessions"}

RECOVERY_UNKNOWNS = "睡眠时长：未记录；静息心率：未记录（未知）。"


def _field_value(spec: Any, name: str, default=None):
    if isinstance(spec, dict):
        return spec.get(name, default)
    return getattr(spec, name, default)


def _number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _as_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _date_label(start: date, end: date) -> str:
    return start.isoformat() if start == end else f"{start.isoformat()} 至 {end.isoformat()}"


def profile_answer(user_id: str, spec: Any) -> dict[str, Any]:
    from app.db import profile_repository

    requested = [str(item) for item in (_field_value(spec, "fields", []) or [])]
    unknown = [item for item in requested if item not in PROFILE_FIELD_NAMES]
    if unknown:
        return {"status": "unavailable", "model_used": False,
                "answer": "这次档案读取包含无法识别的字段，未读取或展示档案内容。",
                "error": "read_spec_unknown_profile_field"}
    fields = requested or list(PROFILE_FIELD_NAMES)
    try:
        profile = profile_repository.get_stored_profile(user_id)
    except Exception:
        return {"status": "unavailable", "model_used": False,
                "answer": "读取档案暂时失败，无法确认当前值；本次没有使用默认值代替。",
                "error": "profile_read_failed"}
    if profile is None:
        return {"status": "completed", "model_used": False, "authoritative": True,
                "answer": "我没有找到你的档案记录；档案里还没有保存这些信息。",
                "facts": {"resource": "profile", "available": False, "values": {}}}
    values = {name: profile.get(name) for name in fields}
    present = {name: value for name, value in values.items() if value not in (None, "")}
    missing = [name for name in fields if name not in present]
    if not present:
        return {"status": "completed", "model_used": False, "authoritative": True,
                "answer": "档案中未填写：" + "、".join(PROFILE_FIELDS[name][0] for name in missing) + "。",
                "facts": {"resource": "profile", "available": True, "values": {}}}
    lines = []
    for name in fields:
        if name not in present:
            continue
        label, unit = PROFILE_FIELDS[name]
        lines.append(f"- {label}：{_number(present[name])}{unit}")
    answer = "档案中的记录：\n" + "\n".join(lines)
    if missing:
        answer += "\n\n档案中未填写：" + "、".join(PROFILE_FIELDS[name][0] for name in missing) + "。"
    return {"status": "completed", "model_used": False, "authoritative": True,
            "answer": answer, "facts": {"resource": "profile", "available": True, "values": present}}


def plan_answer(user_id: str, spec: Any) -> dict[str, Any]:
    from app.db import plan_repository

    try:
        plan = plan_repository.get_stored_plan(user_id)
    except Exception:
        return {"status": "unavailable", "model_used": False,
                "answer": "读取已保存计划暂时失败，无法确认当前计划；本次没有生成或修改计划。",
                "error": "plan_read_failed"}
    sessions = (plan or {}).get("sessions") if isinstance(plan, dict) else None
    if not plan or not isinstance(sessions, list) or not sessions:
        return {"status": "completed", "model_used": False, "authoritative": True,
                "answer": "没有找到已保存的训练计划；本次没有生成或修改计划。",
                "facts": {"resource": "training_plan", "available": False}}
    name = str(plan.get("plan_name") or "已保存计划")
    days_per_week = plan.get("days_per_week")
    header = f"已保存的计划：{name}"
    if isinstance(days_per_week, int) and not isinstance(days_per_week, bool):
        header += f"（每周 {days_per_week} 天）"
    lines = [header]
    for session in sessions:
        if not isinstance(session, dict):
            continue
        day = str(session.get("day") or "").strip()
        title = str(session.get("name") or "").strip()
        items = [str(item).strip() for item in session.get("items") or [] if str(item).strip()]
        label = "：".join(part for part in (day, title) if part) or "训练日"
        lines.append(f"- {label}" + ("；" + "、".join(items) if items else ""))
    return {"status": "completed", "model_used": False, "authoritative": True,
            "answer": "\n".join(lines),
            "facts": {"resource": "training_plan", "available": True, "plan": plan}}


def workout_answer(user_id: str, spec: Any) -> dict[str, Any]:
    from app.db import repositories
    from app.services.agent.workout_facts import workout_fact_line

    start = _as_date(_field_value(spec, "date_start"))
    end = _as_date(_field_value(spec, "date_end"))
    exercises = [str(item).strip() for item in (_field_value(spec, "exercises", []) or [])
                 if str(item).strip()]
    if start is None or end is None:
        return {"status": "needs_clarification", "model_used": False,
                "answer": "这次读取已保存训练缺少日期范围，暂时无法确认；请补充范围后重试。",
                "error": "read_spec_date_missing"}
    if end < start:
        return {"status": "unavailable", "model_used": False,
                "answer": "这次读取已保存训练的日期范围无效，未输出任何记录。",
                "error": "read_spec_date_invalid"}
    try:
        rows, truncated = repositories.query_workout_read_rows(
            user_id, start_date=start, end_date=end, exercises=exercises or None, limit=30)
    except Exception:
        return {"status": "unavailable", "model_used": False,
                "answer": "读取已保存训练暂时失败，无法确认当前值；本次没有使用旧对话原话代替。",
                "error": "workout_read_failed"}
    if not rows:
        return {"status": "completed", "model_used": False, "authoritative": True,
                "answer": f"{_date_label(start, end)} 没有已保存的训练记录。",
                "facts": {"resource": "workout_records", "rows": [], "truncated": truncated,
                          "date_start": start.isoformat(), "date_end": end.isoformat()}}
    answer = "已保存的训练记录：\n- " + "\n- ".join(workout_fact_line(row) for row in rows)
    if truncated:
        answer += "\n\n匹配记录超过单次显示上限，以上仅为部分结果。"
    return {"status": "completed", "model_used": False, "authoritative": True,
            "answer": answer,
            "facts": {"resource": "workout_records", "rows": rows, "truncated": truncated,
                      "date_start": start.isoformat(), "date_end": end.isoformat()}}


def _set_count(rows: list[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        details = row.get("set_details")
        if isinstance(details, list) and details:
            total += len([item for item in details if isinstance(item, dict)])
        else:
            value = row.get("sets")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                total += value
    return total


def advisory_read_facts(user_id: str, spec: Any) -> dict[str, Any]:
    """One shared read for legacy and React advice based on saved workouts.

    The application binds the owner and date range, reads real rows and renders
    the same prefix for both execution paths. Unknown sleep/resting-heart-rate
    data stays explicitly unknown; this function never invents measurements.
    """
    resolved = workout_answer(user_id, spec)
    facts = resolved.get("facts") if isinstance(resolved.get("facts"), dict) else {}
    rows = facts.get("rows") if isinstance(facts.get("rows"), list) else []
    start = _as_date(facts.get("date_start")) or _as_date(_field_value(spec, "date_start"))
    end = _as_date(facts.get("date_end")) or _as_date(_field_value(spec, "date_end"))
    if resolved.get("status") != "completed":
        return {
            "status": resolved.get("status", "unavailable"),
            "error": resolved.get("error"),
            "answer": resolved.get("answer") or "",
            "facts": facts,
            "prefix": "",
            "unknown_metrics": {
                "sleep_hours": None, "resting_heart_rate": None,
                "sleep_recorded": False, "resting_heart_rate_recorded": False,
            },
        }
    if not rows:
        label = _date_label(start, end) if start and end else "本次范围"
        prefix = f"{label} 没有已保存的训练记录。"
    else:
        from app.services.agent.workout_facts import workout_fact_line

        label = _date_label(start, end) if start and end else "本次范围"
        prefix = (
            f"{label} 已保存的训练记录（{len(rows)} 项动作，共 {_set_count(rows)} 组）：\n- "
            + "\n- ".join(workout_fact_line(row) for row in rows)
        )
        if facts.get("truncated"):
            prefix += "\n\n匹配记录超过单次显示上限，以上仅为部分结果。"
    return {
        "status": "completed",
        "answer": resolved.get("answer") or "",
        "facts": facts,
        "prefix": prefix,
        "unknowns": RECOVERY_UNKNOWNS,
        "unknown_metrics": {
            "sleep_hours": None, "resting_heart_rate": None,
            "sleep_recorded": False, "resting_heart_rate_recorded": False,
        },
    }


def read_spec_answer(user_id: str, spec: Any) -> dict[str, Any]:
    try:
        checked = validate_read_spec(spec)
    except Exception:
        return {"status": "unavailable", "model_used": False,
                "answer": "这次读取请求的字段合同不一致，未读取任何数据；请按已登记的字段重试。",
                "error": "read_spec_schema_invalid"}
    resource = str(_field_value(checked, "resource") or "")
    if resource == "profile":
        return profile_answer(user_id, checked)
    if resource == "training_plan":
        return plan_answer(user_id, checked)
    if resource == "workout_records":
        return workout_answer(user_id, checked)
    return {"status": "unavailable", "model_used": False,
            "answer": "这次读取请求包含无法识别的资源，未读取任何数据。",
            "error": "read_spec_unknown_resource"}
