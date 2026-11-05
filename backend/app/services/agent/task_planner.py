from __future__ import annotations

from typing import Any

from app.services.agent.intent import (
    IntentResult,
)
from app.services.agent.task_models import (
    TASK_TYPE_TO_INTENT, TASK_TYPE_TO_WORKER, TaskSpec, TaskType, answer_identity,
    is_preview_task, is_record_write, task_scope_value, task_worker,
)
from app.services.security.write_guard import authorize_write_span


def _semantic_answer_payload(semantic: dict[str, Any], understanding: dict[str, Any],
                             task: dict[str, Any]) -> dict[str, Any]:
    """Build the per-target answer scope shared by parser and rule paths."""
    if not semantic:
        return {}
    inherit_critical = not any(semantic.get(flag) is True for flag in (
        "quoted", "is_hypothetical", "is_third_party"))
    payload: dict[str, Any] = {"semantic_available": True, "semantic": {
        "text": str(semantic.get("text") or task.get("text") or ""),
        "source_task_type": str(semantic.get("task_type") or task.get("task_type") or ""),
        "context_ids": list(semantic.get("context_ids") or []),
        "fact_fields": list(semantic.get("fact_fields") or []),
        "critical_fact_fields": list(semantic.get("critical_fact_fields") or [])
        + (list(understanding.get("critical_fact_fields") or []) if inherit_critical else []),
        "critical_history_ids": (list(understanding.get("critical_history_ids") or [])
                                 if inherit_critical else []),
        "goal_role": semantic.get("goal_role", "result"),
        "quoted": bool(semantic.get("quoted", False)),
        "is_hypothetical": bool(semantic.get("is_hypothetical", False)),
        "is_third_party": bool(semantic.get("is_third_party", False)),
        "reply_purpose": semantic.get("reply_purpose", "general"),
        "output_format": semantic.get("output_format", "default"),
        "history_requirement": semantic.get("history_requirement"),
        "record_query": semantic.get("record_query"),
        "workout_correction": semantic.get("workout_correction"),
        "read_spec": semantic.get("read_spec"),
        "result_reference": semantic.get("result_reference"),
        "plan_result": semantic.get("plan_result"),
        "delivery_type": semantic.get("delivery_type"),
        "preview_ref": semantic.get("preview_ref", ""),
        "context_states": list(understanding.get("context_states") or []),
    }}
    workout_sources = list(semantic.get("workout_sources") or [])
    if workout_sources:
        payload["semantic"]["workout_sources"] = workout_sources
    merged_ids = list((task.get("payload") or {}).get("merged_task_ids") or [])
    if merged_ids:
        payload["merged_task_ids"] = merged_ids
    # The source span is part of the task identity. Keep it beside the semantic
    # scope so both planner paths (validated LLM outline and rule plan) can
    # prove that a non-deliverable goal is bound to the exact user text.
    for key in ("source_start", "source_end"):
        value = task.get(key)
        if type(value) is not int:
            value = ((task.get("payload") or {}).get(key)
                     if isinstance(task.get("payload"), dict) else None)
        if type(value) is int:
            payload[key] = value
    status_like = semantic.get("reply_purpose") in {
        "execution_status", "fact_recall", "memory_update"}
    if (not is_preview_task(semantic)
            and (semantic.get("reply_mode") == "conversation" or status_like)
            and action_is_read(task)):
        payload["dialogue"] = {
            "source_task_type": str(semantic.get("task_type") or task.get("task_type") or ""),
            "reply_mode": "conversation",
            "reply_purpose": semantic.get("reply_purpose", "general"),
            "goal_role": semantic.get("goal_role", "result"),
            "quoted": bool(semantic.get("quoted", False)),
            "output_format": semantic.get("output_format", "default"),
            "history_requirement": semantic.get("history_requirement"),
            "record_query": semantic.get("record_query"),
            "read_spec": semantic.get("read_spec"),
            "context_states": list(understanding.get("context_states") or []),
            "dialogue_ids": semantic.get("context_ids", []),
            "fact_fields": semantic.get("fact_fields", []),
            "critical_history_ids": (list(understanding.get("critical_history_ids") or [])
                                     if inherit_critical else []),
            "critical_fact_fields": (list(understanding.get("critical_fact_fields") or [])
                                     if inherit_critical else []),
        }
    return payload


def _semantic_by_span(understanding: dict[str, Any], message: str) -> dict[tuple[int, int], dict[str, Any]]:
    """Map each semantic task to its ordered original-text occurrence.

    Text alone is not a task identity: the same clause may be a legitimate
    second target. The ordered source span keeps each occurrence's selectors
    attached to its own worker without rejecting valid repeated text.
    """
    mapping: dict[tuple[int, int], dict[str, Any]] = {}
    offset = 0
    for item in understanding.get("tasks") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start = message.find(text, offset)
        if start < 0:
            start = message.find(text)
        if start < 0:
            continue
        end = start + len(text)
        mapping.setdefault((start, end), item)
        offset = end
    return mapping


def semantic_scope_for_merged_span(source: str, merged_text: str,
                                   understanding: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the semantic scope of a merged planner span.

    The planner may merge two adjacent equivalent answer targets. The semantic
    understanding still contains the original two spans, so an exact-span
    lookup would lose delivery/reply/context fields. Collect every semantic
    target inside the merged span, require the same structured identity, and
    synthesize one scope over the merged text.
    """
    if not isinstance(understanding, dict) or not merged_text:
        return {}
    start = source.find(merged_text)
    if start < 0:
        return {}
    end = start + len(merged_text)
    candidates = []
    offset = 0
    for item in understanding.get("tasks") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        if not text:
            continue
        position = source.find(text, offset)
        if position < 0:
            position = source.find(text)
        if position < 0:
            continue
        item_end = position + len(text)
        offset = item_end
        if start <= position and item_end <= end:
            candidates.append(item)
    if len(candidates) < 2:
        return {}
    if len({answer_identity(item) for item in candidates}) != 1:
        return {}
    merged = dict(candidates[0])
    merged["text"] = merged_text
    merged["source_task_type"] = str(candidates[0].get("task_type") or "")
    for key in ("context_ids", "fact_fields", "critical_fact_fields"):
        values = []
        for item in candidates:
            values.extend(item.get(key) or [])
        merged[key] = list(dict.fromkeys(str(value) for value in values))
    workout_sources = []
    for item in candidates:
        workout_sources.extend(item.get("workout_sources") or [])
    if workout_sources:
        merged["workout_sources"] = workout_sources
    return merged


def _tasks_from_structured_input(parsed_input: dict[str, Any] | None, message: str) -> list[TaskSpec]:
    """Convert planner LLM JSON into executable task instances."""
    if not isinstance(parsed_input, dict):
        return []
    tasks: list[TaskSpec] = []
    from app.services.agent.semantics import get_understanding
    understanding = get_understanding(message) or {}
    semantic_tasks = understanding.get("tasks") or []
    semantic_spans = {}
    semantic_offsets: dict[str, int] = {}
    for semantic in semantic_tasks:
        source_text = str(semantic.get("text") or "").strip()
        position = message.find(source_text, semantic_offsets.get(source_text, 0)) if source_text else -1
        if position >= 0:
            end = position + len(source_text)
            semantic_offsets[source_text] = end
            semantic_spans[(position, end)] = semantic
    offsets: dict[str, int] = {}
    for index, item in enumerate(parsed_input.get("tasks") or [], start=1):
        if not isinstance(item, dict):
            continue
        task_type = str(item.get("task_type") or "")
        if task_type == "chitchat":
            task_type = "training_qa"
        worker = TASK_TYPE_TO_WORKER.get(task_type)
        text = str(item.get("text") or "").strip()
        if not worker or not text:
            continue
        # The model may paraphrase a task span. Never pass generated text to a
        # worker as if it were user input; use the original request instead.
        start = message.find(text, offsets.get(text, 0))
        if parsed_input.get("authorization_validated") is True and "source_start" in item:
            position, end = item.get("source_start"), item.get("source_end")
            if (type(position) is not int or type(end) is not int or position < 0
                    or end <= position or message[position:end] != text):
                return []
            start = position
        if start < 0:
            # A paraphrased span cannot prove that its entities came from the
            # user. Drop the whole structured plan and use the explicit
            # unavailable-model response instead of passing generated values to a write.
            return []
        offsets[text] = start + len(text)
        payload = {
            "source_start": start, "source_end": start + len(text),
            "is_negated": bool(item.get("is_negated", False)),
            "is_hypothetical": bool(item.get("is_hypothetical", False)),
            "is_third_party": bool(item.get("is_third_party", False)),
            "semantic_available": True,
            # Only a gate-validated plan may carry a pre-computed authorization;
            # raw model output can never bless its own write tasks.
            "validated_write_authorization": (
                item.get("write_authorization")
                if parsed_input.get("authorization_validated") is True else None),
        }
        if item.get("merged_task_ids"):
            payload["merged_task_ids"] = list(item["merged_task_ids"])
        # Preserve "the model supplied no entities" distinctly from an explicit
        # empty list so consumers can run the bounded extraction fallback.
        for key in ("foods", "workouts"):
            if key in item:
                payload[key] = item[key]
        # Only request-scoped, source-validated semantics may select this path.
        # Never accept execution metadata from raw parser output.
        semantic = semantic_spans.get((start, start + len(text)), {})
        if not semantic and item.get("merged_task_ids"):
            semantic = semantic_scope_for_merged_span(message, text, understanding)
        plan_contract = {key: item.get(key) for key in (
            "plan_result", "result_scope", "plan_goal_id", "plan_request_quote",
            "shared_scope_quote", "plan_target_quote") if key in item}
        if semantic.get("plan_target_quote"):
            plan_contract.setdefault("plan_target_quote", semantic["plan_target_quote"])
        if plan_contract:
            payload["plan_contract"] = plan_contract
        if semantic:
            payload.update(_semantic_answer_payload(semantic, understanding, item))
        action = str(item.get("action") or "answer")
        # ``task_type`` is the domain and ``action`` is the operation. Never
        # translate a record domain into an advice/QA domain: that would drop a
        # business-record read (record_query/fact_recall) and silently turn it
        # into a memory or knowledge answer.
        worker = task_worker(task_type, action) or worker
        if task_type == "nutrition_record" and "foods" in payload:
            payload["foods"] = [food for food in payload["foods"] if not isinstance(food, dict) or not food.get("is_negated")]
        tasks.append(TaskSpec(
            f"task-{index}", task_type, worker, action, text,
            payload=payload,
            side_effect=is_record_write({"task_type": task_type, "action": action}),
        ))
    return tasks


def _attach_task_dependencies(tasks: list[TaskSpec], *, semantic_available: bool = False) -> list[dict[str, Any]]:
    completed_side_effects: list[TaskSpec] = []
    for task in tasks:
        _apply_task_write_authorization(task, semantic_available=semantic_available)
        record_candidate = is_record_write(task)
        if not record_candidate and completed_side_effects and (
            task.task_type in {"nutrition_advice", "training_plan"} or task.payload.get("dialogue")
        ):
            predecessors = completed_side_effects
            if task.payload.get("dialogue"):
                predecessors = [t for t in predecessors if t.payload.get("source_start", 0) < task.payload.get("source_start", 0)]
            task.depends_on = [t.id for t in predecessors]
        if record_candidate:
            completed_side_effects.append(task)
    return [task.to_dict() for task in tasks]


def _apply_task_write_authorization(task: TaskSpec, *, semantic_available: bool = False) -> None:
    if not is_record_write(task):
        if task.action in {"record", "confirm"}:
            task.action = "question"
            task.side_effect = False
        return
    authorization = task.payload.get("validated_write_authorization")
    if authorization not in {"authorized", "needs_confirmation", "rejected_unauthorized"}:
        authorization = task_write_authorization(
            task.text, {"semantic_available": semantic_available, **task.payload, "task_type": task.task_type})
    task.payload["write_authorization"] = authorization
    task.payload.pop("validated_write_authorization", None)
    task.side_effect = authorization == "authorized"


def action_is_read(task: dict) -> bool:
    return task.get("action", "answer") in {"answer", "question"} and task.get("task_type") != "safety_triage"


def _task_for_clause(clause: str, index: int) -> list[TaskSpec]:
    """Legacy hook kept for callers that patch it; it never classifies text.

    Without model semantics the rule path is read-only, so a clause always maps
    to a non-writing question task.
    """
    return [TaskSpec(f"task-{index}", "training_qa", "training_qa", "question", clause)]


def task_write_authorization(task_text: str, flags: dict[str, Any] | None = None) -> str:
    authorization, _ = authorize_write_span(task_text, flags)
    return authorization


def intent_from_planned_tasks(tasks: list[dict[str, Any]], safety_fallback: IntentResult) -> IntentResult:
    if not tasks:
        return safety_fallback
    task_type = str(tasks[0].get("task_type") or "training_qa")
    task_enum = next((kind for kind in TaskType if kind.value == task_type), TaskType.TRAINING_QA)
    name = TASK_TYPE_TO_INTENT[task_enum]
    action = str(tasks[0].get("action") or "question")
    if action not in {"record", "question", "confirm", "answer"}:
        action = "question"
    if safety_fallback.safety_level == "urgent":
        return safety_fallback
    level = "caution" if task_enum == TaskType.SAFETY_TRIAGE and safety_fallback.safety_level == "normal" else safety_fallback.safety_level
    return IntentResult(name, 0.0, level, action)


def _outline_covers_message(message: str, structured_tasks: list[TaskSpec]) -> bool:
    """Measure exact original spans; do not reclassify model clauses with regex."""
    from app.services.agent.planner_validation import measure_span_coverage, SPAN_COVERAGE_MIN
    return measure_span_coverage(message, [task.to_dict() for task in structured_tasks])["coverage"] >= SPAN_COVERAGE_MIN


def plan_tasks_from_llm(
    message: str,
    parsed_input: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Convert a validated planner-LLM outline and entities into executable tasks."""
    if not isinstance(parsed_input, dict) or not parsed_input.get("tasks"):
        return None
    structured_tasks = _tasks_from_structured_input(parsed_input, message)
    if not structured_tasks:
        return None
    # An incomplete outline is treated as an invalid model plan; the caller will
    # stop explicitly instead of mixing model output and local semantic guesses.
    # A single write task may span the remainder of the request when a leading
    # command governs it ("补记昨天的训练：..."); the authorization gate still
    # requires the command to be inside the task span.
    if not _outline_covers_message(message, structured_tasks):
        single_ledger_write = (
            len(structured_tasks) == 1
            and is_record_write(structured_tasks[0])
            and structured_tasks[0].payload.get("source_start", 0) > 0
        )
        if not single_ledger_write:
            return None
    return _attach_task_dependencies(structured_tasks, semantic_available=True)


def _nonwriting_fallback(message: str) -> list[dict[str, Any]]:
    return [{"id": "task-1", "task_type": "training_qa", "worker": "training_qa",
             "action": "question", "text": message, "side_effect": False,
             "depends_on": [], "payload": {}, "status": "pending"}]


def plan_tasks_rule(message: str, intent_result=None) -> list[dict[str, Any]]:
    """Read-only fallback when no validated model semantics exist.

    The 2026-10-18 contract forbids local keyword/regex task classification.
    With a request-scoped model decision the validated tasks are reused; with
    no model decision this returns one non-writing question task so the request
    is answered without guessing a record type or authorizing a write.
    """
    from app.services.agent.intent import is_greeting
    from app.services.agent.semantics import get_understanding

    if is_greeting(message):
        return []
    understanding = get_understanding(message)
    if understanding and not understanding.get("__semantic_unavailable__"):
        tasks = [dict(task) for task in understanding.get("validated_tasks") or []]
        semantic_by_span = _semantic_by_span(understanding, message)
        unique_by_text = {}
        for item in understanding.get("tasks") or []:
            if isinstance(item, dict):
                unique_by_text.setdefault(str(item.get("text") or "").strip(), []).append(item)
        for task in tasks:
            span = (task.get("source_start"), task.get("source_end"))
            semantic = semantic_by_span.get(span)
            if semantic is None:
                candidates = unique_by_text.get(str(task.get("text") or "").strip()) or []
                semantic = candidates[0] if len(candidates) == 1 else None
            if semantic is None and task.get("merged_task_ids"):
                semantic = semantic_scope_for_merged_span(
                    message, str(task.get("text") or ""), understanding)
            if not semantic:
                continue
            payload = dict(task.get("payload") or {})
            addition = _semantic_answer_payload(semantic, understanding, task)
            # The semantic binding is not optional metadata: role, source span
            # and source selectors must stay attached on both planner paths so
            # the dispatch/finalize filter cannot silently lose a constraint.
            payload.update(addition)
            contract = {key: semantic.get(key) for key in (
                "plan_result", "result_scope", "plan_goal_id", "plan_request_quote",
                "shared_scope_quote", "plan_target_quote") if semantic.get(key) not in (None, "")}
            if contract:
                payload["plan_contract"] = contract
            if payload:
                task["payload"] = payload
        return tasks
    return _nonwriting_fallback(message)


def validated_semantic_tasks(message: str) -> list[dict[str, Any]] | None:
    """Validated request tasks enriched with their per-target answer payloads.

    React read/write callers must keep the same validated task identity as the
    legacy planner; only the payload (semantic/dialogue selectors) is attached so
    a source-bound conversation answer can be selected without reclassifying text.
    Returns ``None`` when this request has no usable semantic understanding.
    """
    from app.services.agent.semantics import get_understanding

    if get_understanding(message) is None:
        return None
    return plan_tasks_rule(message)


def validated_rule_candidates(message: str) -> list[dict[str, Any]]:
    """Rule tasks after the same V3-V5 gate; never authorizes a write.

    Used whenever no validated semantic understanding exists (offline model,
    legacy fake adapter, or an explicit configured-call failure). Write
    candidates are kept but marked ``semantic_intent_unavailable``.
    """
    from app.services.agent.planner_validation import validate_rule_plan
    return list(validate_rule_plan(message, plan_tasks_rule(message)).tasks)


def plan_tasks(
    message: str,
    intent_result: IntentResult | dict[str, Any] | None = None,
    parsed_input: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compatibility entry point: use validated model tasks, otherwise a guarded rule plan."""
    planned = plan_tasks_from_llm(message, parsed_input)
    if planned is not None:
        return planned
    return plan_tasks_rule(message, intent_result)
