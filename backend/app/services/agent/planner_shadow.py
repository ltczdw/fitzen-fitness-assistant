"""Diagnostic comparison helpers for the opt-in, non-authoritative Planner."""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

from app.services.agent.task_models import RECORD_TASK_TYPES, TaskType

WRITE_TYPES = RECORD_TASK_TYPES
WRITE_ACTIONS = {"record", "confirm"}


def public_tasks(tasks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [{key: task.get(key) for key in (
        "task_type", "action", "text", "is_negated", "is_hypothetical", "is_third_party"
    ) if key in task} for task in (tasks or [])]


def find_differences(message: str, rule_tasks: list[dict], llm_tasks: list[dict] | None) -> dict[str, list[dict]]:
    """Return per-dimension examples; all strings belong to the offline sample or caller."""
    model = public_tasks(llm_tasks)
    rules = public_tasks(rule_tasks)
    result: dict[str, list[dict]] = {}

    def add(dimension: str, **detail: Any) -> None:
        result.setdefault(dimension, []).append(detail)

    if len(rules) != len(model):
        add("task_count", rule_count=len(rules), llm_count=len(model))
    def locate(task: dict[str, Any]) -> tuple[int, int] | None:
        span = task.get("text")
        if not isinstance(span, str) or not span.strip():
            return None
        start = message.find(span.strip())
        return (start, start + len(span.strip())) if start >= 0 else None

    rule_spans = [locate(task) for task in rules]
    model_spans = [locate(task) for task in model]
    for index, (planned, span) in enumerate(zip(model, model_spans)):
        if span is None:
            add("text_not_contiguous_original", index=index, llm_text=planned.get("text"))
            continue
        overlapping = []
        for rule_index, (rule, rule_span) in enumerate(zip(rules, rule_spans)):
            if rule_span is None:
                continue
            overlap = max(0, min(span[1], rule_span[1]) - max(span[0], rule_span[0]))
            if overlap:
                union = max(span[1], rule_span[1]) - min(span[0], rule_span[0])
                overlapping.append((overlap / union, overlap, rule_index, rule))
        best_rule = max(overlapping, default=None, key=lambda item: (item[0], item[1]))
        rule = best_rule[3] if best_rule else None
        if rule is not None:
            if rule.get("task_type") != planned.get("task_type"):
                add("task_type", index=index, rule=rule.get("task_type"), llm=planned.get("task_type"),
                    rule_text=rule.get("text"), llm_text=planned.get("text"))
            if rule.get("action") != planned.get("action"):
                add("action", index=index, rule=rule.get("action"), llm=planned.get("action"),
                    rule_text=rule.get("text"), llm_text=planned.get("text"))
            if rule.get("text") != planned.get("text"):
                add("task_text_span", index=index, rule_text=rule.get("text"), llm_text=planned.get("text"),
                    llm_is_contiguous_substring=True)
        if planned.get("task_type") in WRITE_TYPES and planned.get("action") in WRITE_ACTIONS:
            explicitly_unsafe = any(planned.get(flag) is True for flag in (
                "is_negated", "is_hypothetical", "is_third_party"
            ))
            if explicitly_unsafe:
                add("llm_record_with_unsafe_flag", index=index, llm_task=planned)
            if rule is None or rule.get("task_type") not in WRITE_TYPES or rule.get("action") not in WRITE_ACTIONS:
                add("llm_record_rule_non_record", index=index, rule_task=rule, llm_task=planned)
    rule_types = {t.get("task_type") for t in rules}
    model_types = {t.get("task_type") for t in model}
    compound = {TaskType.NUTRITION_RECORD.value, TaskType.NUTRITION_ADVICE.value}
    if compound <= rule_types and not compound <= model_types:
        add("missed_record_or_advice", rule_types=sorted(rule_types), llm_types=sorted(model_types))
    food_records = [t for t in model if t.get("task_type") == TaskType.NUTRITION_RECORD.value and t.get("action") in WRITE_ACTIONS]
    rule_food_count = sum(t.get("task_type") == TaskType.NUTRITION_RECORD.value for t in rules)
    command_count = len(re.findall(r"补记|记录|记下|登记|记一下|记下来", message))
    meal_cues = set(re.findall(r"早餐|早饭|午餐|中餐|中午|晚餐|晚饭|加餐|夜宵", message))
    if len(food_records) > 1 and rule_food_count <= 1 and command_count <= 1 and len(meal_cues) <= 1:
        add("one_meal_split_into_multiple_food_tasks", rule_count=sum(t.get("task_type") == TaskType.NUTRITION_RECORD.value for t in rules), llm_count=len(food_records))
    keys = [(t.get("task_type"), t.get("action"), t.get("text")) for t in model]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    for duplicate in duplicates:
        add("duplicate_task", task=list(duplicate))
    return result


def rule_plan(message: str) -> list[dict[str, Any]]:
    from app.services.agent.intent import classify_intent_rule
    from app.services.agent.task_planner import plan_tasks_rule
    return plan_tasks_rule(message, classify_intent_rule(message))


def request_fingerprint(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]
