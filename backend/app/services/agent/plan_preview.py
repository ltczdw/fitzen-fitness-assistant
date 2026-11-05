"""Application-owned state for a previously rendered training preview.

The assistant prose remains the user-facing artifact. This module stores only a
compact, verifiable structure beside that answer (inside the existing
``MemoryTurn.messages`` JSON) so a later "shorten the second day" request can
read the real previous preview instead of guessing from Wiki history.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.services.agent.training_timing import TimedTrainingPlan
from app.services.agent.training_structure import UntimedTrainingPlan

NO_PREVIEW_COPY = (
    "上一轮没有成功生成可修改的训练预览，所以这次没有可缩减或保存的版本；"
    "本次没有生成、修改或保存任何计划。"
)
UNAVAILABLE_PREVIEW_COPY = (
    "无法读取上一轮预览，不能确认或修改它；本次没有生成、修改或保存任何计划。请稍后重试。"
)
NO_PREVIEW_DETAIL = (
    "上一轮没有成功生成可修改的训练预览，所以这次没有可缩减或保存的版本；"
    "本次没有生成、修改或保存任何计划。"
)


class PreviewDay(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=100)
    minutes: int | None = Field(default=None,ge=1, le=1440)
    exercises: list[str] = Field(default_factory=list, max_length=40)


class PlanPreviewState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    preview_id: str
    conversation_id: str
    user_id: str
    request_id: str
    message_id: str
    created_at: str
    base_preview_id: str | None = None
    expected_sessions: int
    weekdays: list[str] = Field(default_factory=list)
    session_minutes: int | None = None
    show_timing: bool = True
    days: list[PreviewDay] = Field(default_factory=list)
    answer_digest: str


def _digest(value: str) -> str:
    # Same canonical string digest as compact.digest(str).
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


_WEEKDAY_CHARS = {"一": "周一", "二": "周二", "三": "周三", "四": "周四",
                  "五": "周五", "六": "周六", "日": "周日", "天": "周日"}
_WEEKDAY_PATTERN = re.compile(r"(?:周|星期|礼拜)([一二三四五六日天])")
_WEEKDAY_SEPARATORS = "、,，和与及跟"
_RANGE_MARKERS = "至到~～—－-"
_RELATIVE_DAY_MARKERS = ("今天", "明天", "后天", "大后天", "昨天", "前天")


def find_weekday_list(source: str) -> list[str] | None:
    """Extract one finite weekday list from source text.

    This is a closed-vocabulary format parser, not intent classification. It
    rejects ranges, repeated weekdays and mixed relative-day wording; it never
    fills a missing weekday.
    """
    text = str(source or "")
    matches = list(_WEEKDAY_PATTERN.finditer(text))
    if not matches:
        return None
    prefix = text[:matches[0].start()]
    if any(marker in prefix[-8:] for marker in _RELATIVE_DAY_MARKERS):
        return None
    values: list[str] = []
    previous_end = None
    for match in matches:
        if match.end() < len(text) and text[match.end()] in "次回":
            # "每周三次" is a frequency, not the weekday Wednesday.
            return None
        if previous_end is not None:
            gap = text[previous_end:match.start()]
            if any(marker in gap for marker in _RANGE_MARKERS):
                return None
            if any(char not in _WEEKDAY_SEPARATORS for char in gap if not char.isspace()):
                return None
        label = _WEEKDAY_CHARS[match.group(1)]
        if label in values:
            return None
        values.append(label)
        previous_end = match.end()
    return values or None


def _exercise_names(plan: TimedTrainingPlan | UntimedTrainingPlan) -> list[list[str]]:
    return [[exercise.name for exercise in session.main.exercises] for session in plan.sessions]


def build_preview_state(*, user_id: str, conversation_id: str, request_id: str,
                        message_id: str, created_at: str, plan: TimedTrainingPlan | UntimedTrainingPlan,
                        answer: str, source: str, expected_sessions: int | None = None,
                        session_minutes: int | None = None,
                        weekdays: list[str] | None = None,
                        base_preview_id: str | None = None,show_timing: bool = True) -> dict[str, Any]:
    """Build a compact state from a validated timed plan and its final answer."""
    resolved_weekdays = list(weekdays) if weekdays is not None else (find_weekday_list(source) or [])
    days = []
    names = _exercise_names(plan)
    for index, session in enumerate(plan.sessions):
        total = getattr(session,"total_seconds",None)
        minutes = max(1, round(total / 60)) if total is not None else None
        label = session.label
        days.append(PreviewDay(label=label, minutes=minutes,
                               exercises=names[index] if index < len(names) else []))
    preview_id = hashlib.sha256(
        f"{user_id}|{conversation_id}|{request_id}|{message_id}|{len(plan.sessions)}".encode("utf-8")
    ).hexdigest()[:32]
    state = PlanPreviewState(
        preview_id=preview_id,
        conversation_id=str(conversation_id or ""),
        user_id=str(user_id or ""),
        request_id=str(request_id or ""),
        message_id=str(message_id or ""),
        created_at=str(created_at or ""),
        base_preview_id=base_preview_id or None,
        expected_sessions=expected_sessions or len(plan.sessions),
        weekdays=resolved_weekdays,
        session_minutes=session_minutes,
        show_timing=show_timing,
        days=days,
        answer_digest=_digest(answer),
    )
    return state.model_dump(mode="json")


def removed_exercises(before: dict[str, Any] | None,
                      after: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Align two previews by day label and report only removed exercise names."""
    before_days = {day.get("label"): set(day.get("exercises") or [])
                   for day in (before or {}).get("days") or [] if isinstance(day, dict)}
    after_days = {day.get("label"): set(day.get("exercises") or [])
                  for day in (after or {}).get("days") or [] if isinstance(day, dict)}
    removed = []
    for label in before_days.keys() & after_days.keys():
        gone = sorted(before_days[label] - after_days[label])
        if gone:
            removed.append({"label": label, "removed": gone})
    return removed


def preview_context(state: dict[str, Any] | None, *, content: str = "") -> dict[str, Any]:
    """Bounded semantic/worker payload; never copies the whole prose by default."""
    if not state:
        return {}
    payload = {
        "preview_id": state.get("preview_id"),
        "message_id": state.get("message_id"),
        "weekdays": list(state.get("weekdays") or []),
        "expected_sessions": state.get("expected_sessions"),
        "session_minutes": state.get("session_minutes"),
        "days": list(state.get("days") or []),
    }
    if content:
        payload["content"] = content[:6000]
        if len(content) > 6000:
            payload["content_truncated"] = True
    return payload


def _assistant_content(messages: list[Any]) -> str:
    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _load_preview_states(user_id: str, conversation_id: str, *, limit: int = 20) -> dict[str, Any]:
    from sqlalchemy import select

    from app.db.memory_models import MemoryTurn
    from app.services.memory import compact

    try:
        with compact.session() as session:
            rows = list(session.scalars(select(MemoryTurn).where(
                MemoryTurn.conversation_id == str(conversation_id or ""),
                MemoryTurn.status == "completed",
                MemoryTurn.hidden_at.is_(None),
                MemoryTurn.deleted_at.is_(None),
            ).order_by(MemoryTurn.completed_at.desc(), MemoryTurn.turn_id.desc()).limit(max(1, min(limit, 50)))))
    except Exception:
        return {"status": "unavailable", "states": []}
    states = []
    for row in rows:
        for message in reversed(row.messages or []):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            artifact = (message.get("app_meta") or {}).get("plan_preview")
            if not isinstance(artifact, dict):
                continue
            try:
                state = PlanPreviewState.model_validate(artifact)
            except Exception:
                return ({"status": "ok", "states": states} if states
                        else {"status": "unavailable", "states": []})
            if state.user_id != str(user_id or "") or state.conversation_id != str(conversation_id or ""):
                # Another owner's artifact is invisible, not a read failure.
                continue
            if state.answer_digest != _digest(str(message.get("content") or "")):
                return ({"status": "ok", "states": states} if states
                        else {"status": "unavailable", "states": []})
            payload = state.model_dump(mode="json")
            payload["content"] = str(message.get("content") or "")
            states.append(payload)
    return {"status": "ok" if states else "none", "states": states}


def recent_plan_previews(user_id: str, conversation_id: str, *, limit: int = 20) -> dict[str, Any]:
    """All verified previews for this owner/conversation, newest first."""
    return _load_preview_states(user_id, conversation_id, limit=limit)


def latest_plan_preview(user_id: str, conversation_id: str, *, limit: int = 20) -> dict[str, Any]:
    """Return the newest verified preview for this owner/conversation.

    Three states are distinct: ``ok`` (verified), ``none`` (no preview), and
    ``unavailable`` (read failure or artifact mismatch). A mismatched artifact
    is never reported as "no preview", because that would invite a false claim.
    """
    loaded = _load_preview_states(user_id, conversation_id, limit=limit)
    if loaded["status"] != "ok":
        return {"status": loaded["status"]}
    return {"status": "ok", "state": loaded["states"][0]}


__all__ = [
    "NO_PREVIEW_COPY",
    "NO_PREVIEW_DETAIL",
    "PlanPreviewState",
    "PreviewDay",
    "UNAVAILABLE_PREVIEW_COPY",
    "build_preview_state",
    "find_weekday_list",
    "latest_plan_preview",
    "preview_context",
    "recent_plan_previews",
    "removed_exercises",
]
