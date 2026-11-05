"""Bridge validated agent task outlines to durable semantic-task budgets.

Only the original user source span (plus a digest of that exact span) forms the
goal identity. Model-generated ids, tool_call identifiers, re-planning labels
and task_type edits never create a new allowance.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from typing import Any

from app.services.llm.budget import model_prepare, model_task

PLAN_TASK_TYPE = "training_plan"
_LABELS = {
    "training_plan": "训练计划",
    "training_qa": "训练问答",
    "nutrition_advice": "营养建议",
    "nutrition_record": "营养记录",
    "workout_record": "训练记录",
    "safety_triage": "安全评估",
}


def _valid_span(source: str, start: Any, end: Any) -> tuple[int, int] | None:
    if type(start) is not int or type(end) is not int:
        return None
    if not (0 <= start < end <= len(source)):
        return None
    return start, end


def task_goal(task: dict[str, Any], source_text: str) -> str:
    """Stable frozen-goal identity from source span, never a free/form label."""
    source = str(source_text or "")
    payload = task.get("payload") if isinstance(task, dict) else None
    payload = payload if isinstance(payload, dict) else {}
    span = _valid_span(source, payload.get("source_start"), payload.get("source_end"))
    if span is None:
        span = _valid_span(source, task.get("source_start"), task.get("source_end"))
    text = str(task.get("text") or "") if isinstance(task, dict) else ""
    if span is None and text:
        position = source.find(text)
        if position >= 0:
            span = (position, position + len(text))
    if span is None:
        # Unspanned tasks share the whole-request identity. This can only make
        # the allowance stricter; it never creates a fresh per-name budget.
        span = (0, len(source)) if source else (0, 0)
    start, end = span
    digest = sha256(source[start:end].encode("utf-8")).hexdigest()
    return f"source:{start}:{end}:{digest}"


def task_label(task_type: Any) -> str:
    value = str(task_type or "")
    return _LABELS.get(value, "未完成目标")


@contextmanager
def task_model_budget(task: dict[str, Any], source_text: str):
    """Charge task-internal chat/embedding/rerank/retries to one goal."""
    goal = task_goal(task or {}, source_text)
    with model_task(goal, task_type=str((task or {}).get("task_type") or "")) as budget:
        yield budget


@contextmanager
def prepare_model_budget():
    with model_prepare() as budget:
        yield budget


__all__ = ["PLAN_TASK_TYPE", "prepare_model_budget", "task_goal", "task_label",
           "task_model_budget"]
