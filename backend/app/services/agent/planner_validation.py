"""Pure deterministic V1-V5 validation for the live Planner request boundary."""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.services.agent.task_models import (
    RECORD_TASK_TYPES,
    TaskAction,
    TaskType,
    answer_identity,
    is_record_write,
    task_scope_value,
)
from app.services.security.input_guard import inspect_input
from app.services.security.write_guard import guard_planner_tasks


SPAN_COVERAGE_MIN = 0.7
_TASK_REQUIRED_KEYS = {
    "task_type", "action", "text", "is_negated", "is_hypothetical", "is_third_party"
}
_TASK_OPTIONAL_KEYS = {
    "goal_role", "quoted", "output_format", "history_requirement",
    "record_query", "read_spec", "delivery_type", "preview_ref",
    "plan_result", "result_scope", "plan_goal_id",
    "plan_request_quote", "shared_scope_quote", "plan_target_quote",
    "merged_task_ids",
}
_TASK_SCHEMA_KEYS = _TASK_REQUIRED_KEYS | _TASK_OPTIONAL_KEYS
_TASK_ENTITY_KEYS = {"foods", "workouts"}
_WRITE_TYPES = RECORD_TASK_TYPES
_WRITE_ACTIONS = {TaskAction.RECORD.value, TaskAction.CONFIRM.value}
_PLAN_TYPES = {TaskType.TRAINING_PLAN.value, TaskType.NUTRITION_ADVICE.value}
_NORMALIZE_REMOVALS = (
    "然后", "接着", "随后", "之后", "同时", "并且", "另外", "顺便", "以及", "还要",
    "再帮我", "再", "并", "吗", "呢", "吧", "啊", "呀", "啦", "嘛",
)
_MEAL_CUES = ("早餐", "早饭", "午餐", "午饭", "中餐", "中午", "晚餐", "晚饭", "晚上", "夜宵", "加餐")
_MERGE_SEPARATORS = set("。！？；;!?，,、\n\r\t ")


@dataclass(frozen=True)
class PendingRecordContext:
    """Trusted app-side pending item; never populated from Planner output."""

    record_id: str
    task_type: str
    session_id: str
    expires_at: datetime


@dataclass
class PlannerValidationResult:
    tasks: list[dict[str, Any]]
    planner_backend: str
    degraded_reason: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    coverage: float | None = None
    fallback_required: bool = False


def _parse_json(raw_plan: Any) -> Any:
    if isinstance(raw_plan, str):
        return json.loads(raw_plan)
    return raw_plan


def _validate_schema(parsed: Any) -> tuple[bool, str | None]:
    if not isinstance(parsed, dict) or set(parsed) != {"tasks"}:
        return False, "top_level_shape"
    tasks = parsed.get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 20:
        return False, "task_count"
    task_types = {item.value for item in TaskType}
    actions = {item.value for item in TaskAction}
    for item in tasks:
        if (not isinstance(item, dict)
                or not _TASK_REQUIRED_KEYS.issubset(item)
                or not set(item).issubset(_TASK_SCHEMA_KEYS | _TASK_ENTITY_KEYS)):
            return False, "task_shape"
        for key in _TASK_ENTITY_KEYS & set(item):
            values = item.get(key)
            if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
                return False, f"{key}_shape"
        task_type = item.get("task_type")
        action = item.get("action")
        if not isinstance(task_type, str) or not isinstance(action, str):
            return False, "enum"
        if task_type not in task_types or action not in actions:
            return False, "enum"
        text = item.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            return False, "text"
        if any(type(item.get(flag)) is not bool for flag in (
            "is_negated", "is_hypothetical", "is_third_party"
        )):
            return False, "safety_flags"
        if "goal_role" in item and item.get("goal_role") not in {"result", "constraint", "background"}:
            return False, "goal_role"
        if "quoted" in item and type(item.get("quoted")) is not bool:
            return False, "quoted"
        if "plan_result" in item and item.get("plan_result") is not None \
                and type(item.get("plan_result")) is not bool:
            return False, "plan_result"
        if "result_scope" in item and item.get("result_scope") not in {
                None, "independent", "shared_leader", "shared_member"}:
            return False, "result_scope"
        for key in ("plan_goal_id", "plan_request_quote", "shared_scope_quote",
                    "plan_target_quote"):
            if key in item and not isinstance(item.get(key), str):
                return False, key
    return True, None


def _pending_is_valid(
    pending: PendingRecordContext | Mapping[str, Any] | None,
    task_type: str,
    current_session_id: str | None,
    now: datetime,
) -> bool:
    if pending is None:
        return False
    values = pending if isinstance(pending, Mapping) else {
        "record_id": pending.record_id,
        "task_type": pending.task_type,
        "session_id": pending.session_id,
        "expires_at": pending.expires_at,
    }
    expiry = values.get("expires_at")
    if isinstance(expiry, str):
        try:
            expiry = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        except ValueError:
            return False
    if not isinstance(expiry, datetime):
        return False
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return bool(
        values.get("record_id")
        and values.get("task_type") == task_type
        and current_session_id
        and values.get("session_id") == current_session_id
        and expiry > current
    )


def _source_span(source: str, text: str, offset: int = 0) -> tuple[int, int] | None:
    normalized = text.strip()
    if not normalized:
        return None
    start = source.find(normalized, offset)
    if start < 0:
        return None
    return start, start + len(normalized)


def _normalize_positions(text: str, offset: int = 0) -> list[int]:
    """Return source indexes after removing punctuation, spaces and fillers."""
    removals = sorted(_NORMALIZE_REMOVALS, key=len, reverse=True)
    skipped: set[int] = set()
    index = 0
    while index < len(text):
        match = next((token for token in removals if text.startswith(token, index)), None)
        if match:
            skipped.update(range(index, index + len(match)))
            index += len(match)
        else:
            index += 1
    result: list[int] = []
    for index, char in enumerate(text):
        if index in skipped or char.isspace() or unicodedata.category(char)[0] in {"P", "Z"}:
            continue
        result.append(offset + index)
    return result


def measure_span_coverage(source: str, tasks: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """Measure the union of task text positions over normalized source chars."""
    source_positions = _normalize_positions(source)
    if not source_positions:
        return {"coverage": 0.0, "covered_chars": 0, "source_chars": 0}
    spans: list[tuple[int, int]] = []
    offsets: dict[str, int] = {}
    for task in tasks:
        text = task.get("text")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        span = _source_span(source, text, offsets.get(text, 0))
        if span:
            spans.append(span)
            offsets[text] = span[1]
    covered = sum(any(start <= position < end for start, end in spans) for position in source_positions)
    return {
        "coverage": covered / len(source_positions),
        "covered_chars": covered,
        "source_chars": len(source_positions),
    }


def _trace_task_corrections(
    source: str,
    items: list[dict[str, Any]],
    *,
    pending_record: PendingRecordContext | Mapping[str, Any] | None,
    current_session_id: str | None,
    now: datetime,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    offsets: dict[str, int] = {}
    for index, task in enumerate(items, start=1):
        task_id = str(task.get("id") or f"task-{index}")
        task["id"] = task_id
        task_type = str(task.get("task_type") or "")
        action = str(task.get("action") or TaskAction.QUESTION.value)
        original_write_type = task_type in _WRITE_TYPES
        original_write_action = action in _WRITE_ACTIONS
        text = str(task.get("text") or "").strip()
        task["text"] = text
        task.setdefault("is_negated", False)
        task.setdefault("is_hypothetical", False)
        task.setdefault("is_third_party", False)
        task.setdefault("execution_status", "pending")
        task.setdefault("side_effect", False)
        original_unsafe_write = original_write_type and original_write_action and any(
            task.get(flag) is True for flag in ("is_negated", "is_hypothetical", "is_third_party")
        )
        if task_type in _WRITE_TYPES and action not in _WRITE_ACTIONS:
            # The domain label stays: a record domain with a non-write action is
            # a read of that domain, not an advice/knowledge task. Only the
            # action is normalized so no downstream gate treats it as a write.
            trace.append({"code": "v1_record_action_downgraded", "task_id": task_id,
                          "from": {"task_type": task_type, "action": action},
                          "to": {"task_type": task_type, "action": TaskAction.QUESTION.value}})
            action = TaskAction.QUESTION.value
        elif task_type not in _WRITE_TYPES and action in _WRITE_ACTIONS:
            trace.append({"code": "v1_nonwrite_action_downgraded", "task_id": task_id,
                          "from": action, "to": TaskAction.QUESTION.value})
            action = TaskAction.QUESTION.value
        if action == TaskAction.CONFIRM.value and task_type in _WRITE_TYPES and not _pending_is_valid(
            pending_record, task_type, current_session_id, now
        ):
            trace.append({"code": "v1_confirm_unbound_downgraded", "task_id": task_id,
                          "task_type": task_type, "to": {"task_type": task_type, "action": "question"}})
            action = TaskAction.QUESTION.value
        task["_v1_write_candidate"] = original_unsafe_write or (
            task_type in _WRITE_TYPES and action in _WRITE_ACTIONS
        )
        task["task_type"] = task_type
        task["action"] = action
        span = _source_span(source, text, offsets.get(text, 0))
        if span is not None:
            offsets[text] = span[1]
        else:
            # Duplicate model spans without another source occurrence keep the
            # first identity and are removed by V2; never create another write.
            span = _source_span(source, text)
        if span is not None:
            task["source_start"], task["source_end"] = span
        else:
            task["source_start"], task["source_end"] = -1, -1
        task.setdefault("status", "pending")
        if task_type in _PLAN_TYPES:
            task["database_refresh_required"] = True
    return trace


def _merge_equivalent_answer_targets(
    source: str, items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge adjacent equivalent answer targets with one contiguous source span.

    Identity is structured (task/action/purpose/delivery/role/flags/scopes), not
    a keyword match. Two different questions that merely share a domain stay
    separate unless their source spans are separated only by punctuation/space.
    """
    trace: list[dict[str, Any]] = []
    result: list[dict[str, Any]] = []
    for task in items:
        if not result:
            result.append(task)
            continue
        previous = result[-1]
        if not _mergeable(previous) or not _mergeable(task):
            result.append(task)
            continue
        if answer_identity(previous) != answer_identity(task):
            result.append(task)
            continue
        if str(previous.get("text") or "") == str(task.get("text") or ""):
            # Identical wording at two source occurrences is two legitimate
            # targets; only distinct adjacent fragments are merged.
            result.append(task)
            continue
        if (str(task_scope_value(previous, "output_format") or "default")
                != str(task_scope_value(task, "output_format") or "default")):
            result.append(task)
            continue
        if (str(task_scope_value(previous, "reply_mode") or "grounded")
                != str(task_scope_value(task, "reply_mode") or "grounded")):
            result.append(task)
            continue
        start = previous.get("source_start")
        end = previous.get("source_end")
        next_start = task.get("source_start")
        next_end = task.get("source_end")
        if not all(type(value) is int for value in (start, end, next_start, next_end)):
            result.append(task)
            continue
        if next_start < end:
            result.append(task)
            continue
        gap = source[end:next_start]
        if any(char not in _MERGE_SEPARATORS for char in gap):
            result.append(task)
            continue
        merged_ids = list(previous.get("merged_task_ids") or [previous.get("id")])
        merged_ids.extend(task.get("merged_task_ids") or [task.get("id")])
        merged_ids = list(dict.fromkeys(str(item) for item in merged_ids if item))
        merged_text = source[start:next_end]
        payload = dict(previous.get("payload") or {})
        payload["merged_task_ids"] = merged_ids
        merged = {
            **previous,
            "text": merged_text,
            "source_start": start,
            "source_end": next_end,
            "context_ids": list(dict.fromkeys([*(previous.get("context_ids") or []),
                                               *(task.get("context_ids") or [])])),
            "fact_fields": list(dict.fromkeys([*(previous.get("fact_fields") or []),
                                               *(task.get("fact_fields") or [])])),
            "critical_fact_fields": list(dict.fromkeys(
                [*(previous.get("critical_fact_fields") or []),
                 *(task.get("critical_fact_fields") or [])])),
            "depends_on": list(dict.fromkeys([*(previous.get("depends_on") or []),
                                              *(task.get("depends_on") or [])])),
            "payload": payload,
            "merged_task_ids": merged_ids,
        }
        trace.append({"code": "v2_equivalent_answer_targets_merged",
                      "task_ids": merged_ids,
                      "span": [start, next_end]})
        result[-1] = merged
    return result, trace


def _mergeable(task: dict[str, Any]) -> bool:
    return (str(task_scope_value(task, "goal_role") or "result") == "result"
            and str(task_scope_value(task, "action") or "") in {"question", "answer"}
            and str(task_scope_value(task, "task_type") or "") not in {
                TaskType.TRAINING_PLAN.value, TaskType.NUTRITION_ADVICE.value}
            and str(task_scope_value(task, "delivery_type") or "") == "advice"
            and not task_scope_value(task, "record_query")
            and not task_scope_value(task, "read_spec")
            and not task_scope_value(task, "plan_contract")
            and not task_scope_value(task, "side_effect"))


def _dedupe_exact_texts(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: set[tuple[str, int, int]] = set()
    result: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for task in items:
        text = task["text"]
        identity = (text, task.get("source_start", -1), task.get("source_end", -1))
        if identity in seen:
            trace.append({"code": "v2_duplicate_text_removed", "task_id": task["id"], "text": text})
            continue
        seen.add(identity)
        result.append(task)
    return result, trace


def _meal_cues(text: str) -> set[str]:
    return {cue for cue in _MEAL_CUES if cue in text}


def _attach_dependencies(items: list[dict[str, Any]]) -> None:
    by_id = {item["id"]: item for item in items}
    for task in items:
        if task.get("task_type") not in _PLAN_TYPES:
            continue
        wanted = TaskType.NUTRITION_RECORD.value if task["task_type"] == TaskType.NUTRITION_ADVICE.value else TaskType.WORKOUT_RECORD.value
        target_meals = _meal_cues(task["text"])
        dependencies: list[str] = []
        for candidate in items:
            if candidate.get("task_type") != wanted or candidate["id"] == task["id"]:
                continue
            if not is_record_write(candidate):
                # Only a committed write can satisfy a plan's record dependency;
                # a read of the same domain provides no write result to wait on.
                continue
            if candidate.get("status") == "rejected_unauthorized":
                continue
            record_meals = _meal_cues(candidate["text"])
            if target_meals and record_meals and target_meals.isdisjoint(record_meals):
                continue
            dependencies.append(candidate["id"])
        # E007：规则规划器基于原文显式结果引用（“根据今天的蛋白质摄入…”）预挂的
        # 依赖边不得被本函数清空——跨业务必要依赖只在原文明确要求时存在，一旦
        # 标注就必须保留；这里仅过滤指向非写任务或不存在任务的悬空引用。
        preserved = [
            dependency_id for dependency_id in task.get("depends_on") or []
            if dependency_id in by_id
            and is_record_write(by_id[dependency_id])
            and by_id[dependency_id].get("status") != "rejected_unauthorized"
        ]
        task["depends_on"] = list(dict.fromkeys(preserved + dependencies))


def _sort_tasks(items: list[dict[str, Any]]) -> None:
    indexed = list(enumerate(items))
    indexed.sort(key=lambda pair: (
        0 if pair[1].get("task_type") in _WRITE_TYPES and pair[1].get("action") in _WRITE_ACTIONS else 1,
        pair[1].get("source_start", -1) if pair[1].get("source_start", -1) >= 0 else 10**9,
        pair[0],
    ))
    items[:] = [item for _, item in indexed]


def _apply_v5(items: list[dict[str, Any]]) -> None:
    _attach_dependencies(items)
    for task in items:
        if task.get("task_type") not in _PLAN_TYPES:
            continue
        dependency_ids = task.get("depends_on") or []
        by_id = {item["id"]: item for item in items}
        blockers = [by_id[item_id] for item_id in dependency_ids if item_id in by_id and (
            by_id[item_id].get("status") not in {"completed", "success", "succeeded"}
        )]
        if blockers:
            task["status"] = "blocked"
            task["execution_status"] = "blocked"
            task["blocked_reason"] = "record_dependency_not_ready"
        elif task.get("blocked_reason") == "record_dependency_not_ready":
            task["status"] = "pending"
            task["execution_status"] = "pending"
            task.pop("blocked_reason", None)
            task.pop("blocked_by", None)


def resolve_dependency_outcomes(
    items: Sequence[Mapping[str, Any]], outcomes: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Mark advice/plan blocked unless each attached record completed."""
    result = [dict(item) for item in items]
    by_id = {item.get("id"): item for item in result}
    for task in result:
        if task.get("task_type") not in _PLAN_TYPES:
            continue
        blocked_by: list[str] = []
        for dependency_id in task.get("depends_on") or []:
            state = outcomes.get(dependency_id)
            dependency = by_id.get(dependency_id, {})
            if state not in {"completed", "success", "succeeded"} or dependency.get("status") in {
                "needs_confirmation", "rejected_unauthorized", "failed", "blocked"
            }:
                blocked_by.append(str(dependency_id))
        if blocked_by:
            task["status"] = "blocked"
            task["execution_status"] = "blocked"
            task["blocked_reason"] = "record_dependency_not_ready"
            task["blocked_by"] = blocked_by
        elif task.get("blocked_reason") == "record_dependency_not_ready":
            task["status"] = "pending"
            task["execution_status"] = "pending"
            task.pop("blocked_reason", None)
            task.pop("blocked_by", None)
    return result


def validate_rule_plan(
    source_text: str,
    rule_tasks: Sequence[Mapping[str, Any]],
    *,
    pending_record: PendingRecordContext | Mapping[str, Any] | None = None,
    current_session_id: str | None = None,
    now: datetime | None = None,
    semantic_available: bool = False,
    urgent: bool | None = None,
    safety_assessed: bool | None = None,
) -> PlannerValidationResult:
    """Apply V3-V5 to rule tasks; without model semantics writes stay read-only."""
    source = inspect_input(source_text).sanitized_text
    current = now or datetime.now(timezone.utc)
    items = [dict(task) for task in rule_tasks if isinstance(task, Mapping)]
    trace = _trace_task_corrections(source, items, pending_record=pending_record,
                                    current_session_id=current_session_id, now=current)
    items, merge_trace = _merge_equivalent_answer_targets(source, items)
    trace.extend(merge_trace)
    deduped, duplicate_trace = _dedupe_exact_texts(items)
    trace.extend(duplicate_trace)
    for task in deduped:
        task.setdefault("execution_status", "pending")
    trace.extend(guard_planner_tasks(source, deduped, now=current, pending_record=pending_record,
                                     current_session_id=current_session_id, semantic_available=semantic_available,
                                     urgent=urgent, safety_assessed=safety_assessed))
    _sort_tasks(deduped)
    _apply_v5(deduped)
    return PlannerValidationResult(deduped, "rule", trace=trace,
                                   coverage=float(measure_span_coverage(source, deduped)["coverage"]))


def validate_planner_output(
    source_text: str,
    raw_plan: Any,
    *,
    rule_tasks: Sequence[Mapping[str, Any]] = (),
    pending_record: PendingRecordContext | Mapping[str, Any] | None = None,
    current_session_id: str | None = None,
    now: datetime | None = None,
    urgent: bool | None = None,
    safety_assessed: bool | None = None,
) -> PlannerValidationResult:
    """Run V1-V5 on an LLM plan; return rule fallback tasks on V1/V2 failure."""
    inspected = inspect_input(source_text)
    source = inspected.sanitized_text
    current = now or datetime.now(timezone.utc)
    try:
        parsed = _parse_json(raw_plan)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    valid, _schema_error = _validate_schema(parsed)
    if not valid:
        fallback = validate_rule_plan(source, rule_tasks, pending_record=pending_record,
                                      current_session_id=current_session_id, now=current)
        fallback.degraded_reason = "planner_invalid_schema"
        fallback.fallback_required = True
        fallback.trace.insert(0, {"code": "v1_planner_invalid_schema", "detail": _schema_error or "json_parse"})
        return fallback

    items = [dict(task) for task in parsed["tasks"]]
    trace = _trace_task_corrections(source, items, pending_record=pending_record,
                                    current_session_id=current_session_id, now=current)
    items, merge_trace = _merge_equivalent_answer_targets(source, items)
    trace.extend(merge_trace)
    for task in items:
        span = _source_span(source, task["text"])
        if span is None:
            fallback = validate_rule_plan(source, rule_tasks, pending_record=pending_record,
                                          current_session_id=current_session_id, now=current)
            fallback.degraded_reason = "planner_bad_span"
            fallback.fallback_required = True
            fallback.trace.insert(0, {"code": "v2_planner_bad_span", "task_id": task["id"], "text": task["text"]})
            return fallback
    items, duplicate_trace = _dedupe_exact_texts(items)
    trace.extend(duplicate_trace)
    coverage_detail = measure_span_coverage(source, items)
    coverage = float(coverage_detail["coverage"])
    if coverage < SPAN_COVERAGE_MIN:
        fallback = validate_rule_plan(source, rule_tasks, pending_record=pending_record,
                                      current_session_id=current_session_id, now=current)
        fallback.degraded_reason = "planner_low_coverage"
        fallback.fallback_required = True
        fallback.trace.insert(0, {"code": "v2_planner_low_coverage", **coverage_detail,
                                  "threshold": SPAN_COVERAGE_MIN})
        return fallback
    trace.extend(guard_planner_tasks(source, items, now=current, pending_record=pending_record,
                                     current_session_id=current_session_id, semantic_available=True,
                                     urgent=urgent, safety_assessed=safety_assessed))
    _sort_tasks(items)
    _apply_v5(items)
    return PlannerValidationResult(items, "llm", trace=trace, coverage=coverage)
