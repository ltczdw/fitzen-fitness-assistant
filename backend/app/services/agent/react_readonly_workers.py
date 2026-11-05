"""阶段B单次只读回答 worker：answer_training / advise_nutrition / plan_training。

三种回答都只调用一次无工具的 ``complete_message``，不嵌套 legacy 的
``complete_with_tools`` 循环，也不进入营养 record/confirm 或计划 save 分支。
输入使用当前应用已完成的只读观察/缓存（经现有 context_budget 提示词路径）与必要
RAG 检索证据；证据质量不足时返回明确“不足”受限观察，不调用回答模型、不伪成功。
结构化引用只由实际合格检索资料生成（citations_from_hits），模型不能自造来源。
"""
from __future__ import annotations

from typing import Any
from app.services.agent.answer_style import REPLY_STYLE

from app.services.agent.react_models import DomainObservation, ToolStatus
from app.services.agent.read_contracts import PROFILE_FIELD_NAMES
from app.services.agent.task_models import RECORD_TASK_TYPES, TaskType
from app.services.llm.client import LLMContextBudgetError


def _budget_failure(exc, query):
    reason = str(getattr(exc, "reason", "") or "")
    summary = ("所选记忆单元缺少必要依据，无法在不丢失条件/否定的情况下回答；请明确要继续的来源。"
               if reason == "missing_dependency" else
               "本次输入与必要事实超过了上下文预算，请缩小问题范围后重试。")
    return DomainObservation(status=ToolStatus.UNAVAILABLE,
                             error=("semantic_unit_dependency_missing"
                                    if reason == "missing_dependency" else "context_budget_exceeded"),
                             summary=summary, result={"query": query})


_QA_SYSTEM = (
    "你是健身科普答疑Agent。基于召回知识回答动作、恢复、睡眠和补剂问题；"
    "涉及明显肿胀、剧烈疼痛、麻木、变形或活动受限时，建议停止训练并寻求专业帮助，"
    "不进行医疗诊断。"
    + REPLY_STYLE
)

_NUTRITION_SYSTEM = (
    "你是饮食营养方案助手。用户专属数据只能使用应用已读取的既有事实；回答中不要出现SQL、表名、"
    "工具名、字段名、记录编号或其他内部实现术语。知识原则只能使用"
    "召回内容。不要伪造测量数据，不做医疗诊断。估算值必须明确标注。内容保持简洁、可执行。"
    + REPLY_STYLE
)

_PLAN_SYSTEM = (
    "你是训练方案助手。用户专属数据只能使用应用已读取的既有事实；回答中不要出现SQL、表名、"
    "工具名、字段名、记录编号或其他内部实现术语。知识原则只能使用"
    "召回内容。不要伪造测量数据，不做医疗诊断。估算值必须明确标注。内容保持简洁、可执行。"
    + REPLY_STYLE
)


def _observation_state(context: dict[str, Any]) -> dict[str, Any]:
    """已完成只读观察 → context_budget 的 task_context（数据，非指令）。"""
    task_context: list[Any] = []
    for item in (context or {}).get("observations") or []:
        if not isinstance(item, dict) or item.get("status") != "completed":
            continue
        result = item.get("result")
        if isinstance(result, dict):
            task_context.append(result)
    state: dict[str, Any] = {}
    if task_context:
        state["task_context"] = task_context
    return state


async def _answer_state(context, task_type):
    """Add applicable durable constraints independently of optional observations."""
    state = _observation_state(context)
    memory = _request_memory_snapshot()
    scope = _dialogue_scope(context)
    result_reference = scope.get('result_reference')
    if result_reference and memory.get('recent_result_dialogue'):
        from app.services.agent.result_reference import select_result_sources
        memory = select_result_sources(memory, scope.get('context_ids') or [])
    if not memory.get('versioned_view') and not result_reference and not scope.get('context_ids'):
        return state
    from app.schemas.chat import ChatRequest
    from app.services.agent.dialogue_context import restore_worker_scope, DialogueContextUnavailable
    from app.services.memory.wiki_context import WikiReadError
    request = ChatRequest(user_id=context.get('user_id', 'demo-user'),
        conversation_id=context.get('conversation_id') or memory.get('wiki_conversation_id') or 'demo-conversation',
        request_id=context.get('request_id') or 'source-check',
        message=context.get('source_text') or context.get('task_text') or '来源校验')
    try:
        checked = await restore_worker_scope(request, memory, scope, task_type)
    except (DialogueContextUnavailable, WikiReadError):
        return None
    state = {**state, 'persistent_constraints': checked.get('persistent_constraints', {}),
             'current_states': checked.get('current_states', {})}
    if result_reference:
        from app.services.agent.dialogue_context import project_history
        from app.services.agent.result_reference import result_sources
        selected = [m for m in project_history(checked) if m['message_id'] in scope.get('context_ids', [])]
        if not any(m['message_id'] == result_reference['message_id'] for m in selected):
            return None
        target = next((item for item in result_sources(checked)
                       if item['message_id'] == result_reference['message_id']), {})
        goal_sources = list(dict.fromkeys([*target.get('source_user_ids', []), target.get('user_message_id')]))
        state['task_context'] = [*(state.get('task_context') or []), {
            'referenced_result': selected,
            'goal_source_ids': [mid for mid in goal_sources if mid],
            'reference_purpose': result_reference.get('purpose', 'revise'),
            'source_rule': ('仅把所选回答作为背景，执行当前新目标及其领域，不重做旧任务；'
                            if result_reference.get('purpose') == 'context' else '重做原目标；')
                           + '助手旧回答不是用户已执行/已食用事实，也不授予保存权限，当前纠正优先。'}]
    elif scope.get('context_ids'):
        from app.services.agent.dialogue_context import scoped_worker_memory
        selected=scoped_worker_memory(checked,scope,task_type)
        for key in ('short_memory','retrieved_history','wiki_units','wiki_as_of','wiki_dependencies'):
            if key in selected:
                state[key]=selected[key]
    return state


_PROFILE_FIELDS = PROFILE_FIELD_NAMES


def _completed_profile(context: dict[str, Any]) -> dict[str, Any]:
    """已完成 read_profile 观察 → 可信 profile（只投影档案字段）。

    观察没有工具名，只有领域结果；仅当结果带档案规划字段时才视为 profile
    投影，避免把训练/营养结果误当个人档案。非 completed 观察一律不采用。
    """
    profile: dict[str, Any] = {}
    for item in (context or {}).get("observations") or []:
        if not isinstance(item, dict) or item.get("status") != "completed":
            continue
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        candidate = result.get("profile") if isinstance(result.get("profile"), dict) else result
        if not any(name in candidate for name in ("goal", "experience", "available_days")):
            continue
        for name in _PROFILE_FIELDS:
            value = candidate.get(name)
            if value not in (None, ""):
                profile[name] = value
    return profile


def _plan_information_state(context: dict[str, Any]) -> dict[str, Any]:
    """只读方案来源门状态：应用注入的可信 context + 已完成 profile 投影。

    模型工具参数（``args.query``）永远不进入个人事实来源；只有原始请求原文
    与应用已完成的只读观察可以补资料，默认档案值仍由 user_sources 过滤。
    """
    context = context or {}
    source_text = str(context.get("source_text") or "")
    task_text = str(context.get("task_text") or source_text)
    state: dict[str, Any] = {
        "raw_message": source_text,
        "task_text": task_text,
        "task_id": str(context.get("task_id") or ""),
        "task_context": list(context.get("task_context") or []),
        "plan_contract": dict(context.get("plan_contract") or {}),
        "plan_siblings": list(context.get("plan_siblings") or []),
        "semantic_scope": dict(context.get("semantic_scope") or {}),
    }
    profile = dict(context.get("profile") or {})
    profile.update(_completed_profile(context))
    if profile:
        state["profile"] = profile
    for name in ("long_memory", "short_memory", "information_sources"):
        value = context.get(name)
        if isinstance(value, dict):
            state[name] = dict(value)
        elif isinstance(value, list):
            state[name] = list(value)
    return state


def _insufficient(task_type: str, query: str, *, evidence_count: int | None = None,
                  quality=None, filtered: bool = False) -> DomainObservation:
    from app.services.rag.quality import insufficient_evidence_reply, assess_rag_evidence

    result: dict[str, Any] = {"query": query}
    if quality is None and evidence_count in (None, 0):
        quality = assess_rag_evidence(query, [], task_type=task_type)
    if quality is not None:
        result["retrieval_quality"] = quality.as_dict()
    result["evidence_filtered"] = filtered
    if evidence_count is not None:
        result["evidence_count"] = evidence_count
    return DomainObservation(
        # A completed retrieval with insufficient evidence is a final refusal,
        # not a transient outage. Cover the read goal so unchanged retrieval
        # cannot loop until its explanation is replaced by a generic limit.
        status=ToolStatus.REJECTED,
        error="insufficient_evidence",
        summary=f"证据不足：{insufficient_evidence_reply(task_type, quality, filtered=filtered)}",
        missing_fields=["knowledge_evidence"],
        result=result,
    )


def _retrieval_failure(query: str) -> DomainObservation:
    return DomainObservation(
        status=ToolStatus.UNAVAILABLE, error="knowledge_retrieval_failed",
        summary="知识资料检索暂时失败，本次无法核实回答依据，请稍后重试。",
        result={"query": query})


def _dialogue_scope(context: dict[str, Any]) -> dict[str, Any]:
    """Merge the per-target semantic flags with its read-only dialogue purpose."""
    semantic = (context or {}).get("semantic_scope")
    dialogue = (context or {}).get("dialogue_scope")
    scope = dict(semantic) if isinstance(semantic, dict) else {}
    if isinstance(dialogue, dict):
        scope.update(dialogue)
    for key in ("quoted", "is_hypothetical", "is_third_party", "goal_role"):
        if isinstance(semantic, dict) and key in semantic \
                and not (isinstance(dialogue, dict) and key in dialogue):
            scope[key] = semantic[key]
    return scope


def _request_memory_snapshot() -> dict[str, Any]:
    """The current request's already-loaded memory, when the caller owns one."""
    try:
        from app.services.agent.execution_state import _request_memory
        value = _request_memory.get()
    except Exception:
        value = None
    return dict(value) if isinstance(value, dict) else {}


def _display_number(value: Any) -> str:
    from app.services.agent.workout_facts import display_number

    return display_number(value)


def _workout_fact_line(record: dict[str, Any]) -> str:
    from app.services.agent.workout_facts import workout_fact_line

    return workout_fact_line(record)


def _nutrition_fact_line(summary: dict[str, Any]) -> str:
    days = summary.get("days")
    entries = summary.get("entries")
    prefix = ("近" + str(days) + "天" if isinstance(days, int) and not isinstance(days, bool)
              else "已保存营养记录")
    return (
        prefix + "汇总：记录 " + _display_number(entries) + " 条；热量 "
        + _display_number(summary.get("calories")) + " kcal；蛋白质 "
        + _display_number(summary.get("protein_g")) + " g；碳水 "
        + _display_number(summary.get("carbs_g")) + " g；脂肪 "
        + _display_number(summary.get("fat_g")) + " g；膳食纤维 "
        + _display_number(summary.get("fiber_g")) + " g。")


def _positive_number(value: Any) -> bool:
    try:
        return float(value or 0) > 0
    except (TypeError, ValueError):
        return False


def _allowed_data_kinds(scope: dict[str, Any]) -> set[str]:
    """Candidate source kinds from the current target's structured task type.

    Without an explicit source type both explicit data kinds may be offered to
    the source-bound selector; they are never automatically rendered together.
    """
    task_type = str(scope.get("source_task_type") or "")
    if task_type == TaskType.WORKOUT_RECORD.value:
        return {"workouts"}
    if task_type == TaskType.NUTRITION_RECORD.value:
        return {"nutrition"}
    return {"workouts", "nutrition"}


def _observation_kind(item: dict[str, Any]) -> str | None:
    """Application-owned read identity, with a shape fallback for old callers."""
    source = item.get("_source")
    if isinstance(source, dict):
        if source.get("permission") != "read":
            return None
        tool = str(source.get("tool_name") or "")
        if tool == "read_workout_history":
            return "workouts"
        if tool == "read_nutrition_summary":
            return "nutrition"
        return None
    result = item.get("result")
    if isinstance(result, dict):
        if isinstance(result.get("workouts"), list):
            return "workouts"
        if any(key in result for key in ("calories", "protein_g", "carbs_g", "fat_g")):
            return "nutrition"
    return None


def _workout_sources(records: list[Any]) -> dict[str, str]:
    from app.services.agent.workout_facts import workout_source_key

    sources: dict[str, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        sources[workout_source_key(record, index)] = _workout_fact_line(record)
    return sources


def _current_data_sources(
    context: dict[str, Any], scope: dict[str, Any] | None = None,
) -> tuple[dict[str, str], bool, bool]:
    """Latest completed read per source kind, with later failure/empty winning.

    Source identity comes from the app's tool/permission annotation when
    available. Old callers without it use the explicit result shape. A later
    failed read of the same kind clears earlier values; a later empty readable
    result also clears them. Profile defaults are never fact sources.
    """
    allowed = _allowed_data_kinds(scope or {})
    latest_status: dict[str, str] = {}
    latest_entries: dict[str, dict[str, str]] = {}
    attempted = False
    for item in (context or {}).get("observations") or []:
        if not isinstance(item, dict):
            continue
        kind = _observation_kind(item)
        if kind not in allowed:
            continue
        attempted = True
        status = str(item.get("status") or "")
        latest_status[kind] = status
        if status != "completed":
            latest_entries[kind] = {}
            continue
        result = item.get("result")
        if not isinstance(result, dict):
            latest_entries[kind] = {}
            continue
        if kind == "workouts" and isinstance(result.get("workouts"), list):
            latest_entries[kind] = _workout_sources(result["workouts"])
        elif kind == "nutrition":
            entries = result.get("entries")
            if (isinstance(entries, int) and not isinstance(entries, bool) and entries > 0) \
                    or _positive_number(result.get("calories")):
                line = _nutrition_fact_line(result)
                latest_entries[kind] = {"nutrition:summary:0": line}
            else:
                latest_entries[kind] = {}
        else:
            latest_entries[kind] = {}
    sources: dict[str, str] = {}
    for kind, status in latest_status.items():
        if status == "completed":
            sources.update(latest_entries.get(kind, {}))
    failed = any(status != "completed" for status in latest_status.values())
    return sources, attempted, failed


def _current_data_truncated(
    context: dict[str, Any], scope: dict[str, Any] | None = None,
) -> bool:
    """Whether the latest completed read for an allowed source was truncated."""
    allowed = _allowed_data_kinds(scope or {})
    truncated = False
    for item in (context or {}).get("observations") or []:
        if not isinstance(item, dict):
            continue
        if _observation_kind(item) not in allowed:
            continue
        if str(item.get("status") or "") != "completed":
            continue
        result = item.get("result")
        if isinstance(result, dict) and result.get("truncated") is True:
            truncated = True
    return truncated


class SourceSelection(list):
    """Selected source lines plus the structured completeness of the selection.

    A list subclass keeps the existing selector contract (callers still receive
    a list of lines) while exposing whether the model returned every candidate.
    """

    def __init__(self, lines, *, complete: bool, candidate_count: int,
                 selected_count: int):
        super().__init__(lines)
        self.complete = complete
        self.candidate_count = candidate_count
        self.selected_count = selected_count


async def _select_current_sources(
    context: dict[str, Any], task_text: str, sources: dict[str, str],
) -> list[str] | None:
    """Use the existing source-selection protocol; never fall back to all rows."""
    if not sources:
        return []
    from app.schemas.chat import ChatRequest
    from app.services.agent.agent import agent
    from app.services.agent.dialogue_context import select_source_excerpts

    llm = agent.graph.llm
    try:
        configured = bool(llm.configured()) if callable(getattr(llm, "configured", None)) else False
    except Exception:
        configured = False
    if not configured:
        return None
    request = ChatRequest(
        user_id=str((context or {}).get("user_id") or "demo-user"),
        conversation_id=str((context or {}).get("conversation_id") or ""),
        request_id=str((context or {}).get("request_id") or ""),
        message=task_text,
    )
    try:
        excerpts = await select_source_excerpts(
            llm, request, sources, name="select_current_record_sources")
    except Exception:
        return None
    selected: list[str] = []
    seen_ids: set[str] = set()
    for excerpt in excerpts:
        line = sources.get(excerpt.source_id)
        if line and excerpt.source_id not in seen_ids:
            seen_ids.add(excerpt.source_id)
            selected.append(line)
    complete = len(seen_ids) == len(sources)
    if selected and not complete:
        selected.append("当前只列出了部分候选记录，请缩小查询范围核对。")
    return SourceSelection(selected, complete=complete,
                           candidate_count=len(sources), selected_count=len(seen_ids))


async def _dialogue_answer(
    context: dict[str, Any], scope: dict[str, Any], task_text: str,
    memory: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    from app.schemas.chat import ChatRequest
    from app.services.agent.dialogue_context import conversation_answer
    from app.services.agent.agent import agent

    request = ChatRequest(
        user_id=str((context or {}).get("user_id") or "demo-user"),
        conversation_id=str((context or {}).get("conversation_id") or ""),
        request_id=str((context or {}).get("request_id") or ""),
        message=task_text,
    )
    return await conversation_answer(
        agent.graph.llm, request,
        _request_memory_snapshot() if memory is None else memory, scope)


def _source_bound_observation(
    *, summary: str, task_text: str, authoritative: bool, quoted: bool = False,
    error: str | None = None, status: ToolStatus = ToolStatus.COMPLETED,
    state_update_requested: bool = False,
    complete: bool | None = None,
    selection_complete: bool | None = None,
    candidate_count: int | None = None,
    selected_count: int | None = None,
    truncated: bool | None = None,
) -> DomainObservation:
    result: dict[str, Any] = {
        "query": task_text,
        "source_bound": True,
        "authoritative": authoritative,
    }
    if quoted:
        result["quoted"] = True
    if state_update_requested:
        result["state_update_requested"] = True
    for key, value in (
        ("complete", complete), ("selection_complete", selection_complete),
        ("candidate_count", candidate_count), ("selected_count", selected_count),
        ("truncated", truncated),
    ):
        if value is not None:
            result[key] = value
    return DomainObservation(
        status=status,
        error=error,
        summary=summary,
        result=result,
    )


async def _source_bound_dialogue_answer(
    context: dict[str, Any], task_text: str,
) -> DomainObservation | None:
    """Answer a validated conversation/fact-recall goal without knowledge RAG.

    Returns ``None`` only when this target genuinely needs new knowledge; the
    caller then runs the ordinary grounded worker. Completed SQL reads are
    rendered by the application, so a model cannot invent or replace them.
    """
    scope = _dialogue_scope(context)
    if not scope:
        return None
    purpose = str(scope.get("reply_purpose") or "")
    conversation_read = (
        scope.get("reply_mode") == "conversation" and purpose == "general")
    quoted = conversation_read and (
        scope.get("quoted") is True or scope.get("is_third_party") is True)
    if purpose == "fact_recall":
        sources, attempted, failed = _current_data_sources(context, scope)
        source_task_type = str(scope.get("source_task_type") or "")
        if (source_task_type == TaskType.NUTRITION_RECORD.value and sources
                and all(key.startswith("nutrition:summary:") for key in sources)):
            # The summary tool is aggregate-only; a saved-item query must still
            # read the owner/date-filtered user_food_log rows below.
            sources, attempted = {}, False
        if source_task_type in RECORD_TASK_TYPES and not attempted:
            if not isinstance(scope.get("record_query"), dict):
                # No structured date/meal scope and no observed read: ask for
                # the range instead of letting the dialogue/LLM path answer
                # "no record" while rows exist.
                return _source_bound_observation(
                    summary="这次读取已保存记录缺少日期或餐次范围，暂时无法确认；请补充范围后重试。",
                    task_text=task_text, authoritative=False,
                    status=ToolStatus.NEEDS_CLARIFICATION,
                    error="saved_record_query_scope_missing")
            # A business-record recall with no observed read is a current SQL
            # read, not a memory recollection. The shared dialogue path owns the
            # bounded owner/date query and source rendering; failed or unselected
            # reads stay unavailable rather than becoming a false empty answer.
            current = await _dialogue_answer(context, scope, task_text)
            if current and current.get("status") in {"unavailable", "context_budget"}:
                return _source_bound_observation(
                    summary=str(current.get("answer") or "读取已保存记录暂时失败，无法确认当前值。"),
                    task_text=task_text, authoritative=False, status=ToolStatus.UNAVAILABLE,
                    error=str(current.get("error") or "saved_record_query_unavailable"),
                    state_update_requested=bool(current.get("state_update_requested")))
            if current and current.get("answer"):
                return _source_bound_observation(
                    summary=str(current["answer"]), task_text=task_text,
                    authoritative=bool(current.get("authoritative", True)),
                    state_update_requested=bool(current.get("state_update_requested")))
            return _source_bound_observation(
                summary="没有可用的已保存记录读取结果，无法确认当前值；本次没有使用旧对话原话代替。",
                task_text=task_text, authoritative=False, status=ToolStatus.UNAVAILABLE,
                error="saved_record_query_unavailable")
        if failed:
            return _source_bound_observation(
                summary="读取已保存记录暂时失败，无法确认当前值；本次没有使用较早读取或旧对话原话代替。",
                task_text=task_text, authoritative=False, status=ToolStatus.UNAVAILABLE,
                error="saved_record_read_failed")
        if sources:
            query = scope.get("record_query")
            exact = isinstance(query, dict) and any(
                query.get(key) for key in ("exercises", "meal_types", "food_names"))
            truncated = _current_data_truncated(context, scope)
            if exact:
                complete = not truncated
                prefix = "已保存的记录：" if complete else "匹配到的已保存记录："
                return _source_bound_observation(
                    summary=prefix + "\n- " + "\n- ".join(sources.values()),
                    task_text=task_text, authoritative=True,
                    complete=complete, selection_complete=True,
                    candidate_count=len(sources), selected_count=len(sources),
                    truncated=truncated)
            selected = await _select_current_sources(context, task_text, sources)
            if selected is None:
                return _source_bound_observation(
                    summary="当前无法可靠选择已保存记录来源，未输出未经来源确认的数据。",
                    task_text=task_text, authoritative=False, status=ToolStatus.UNAVAILABLE,
                    error="saved_record_selection_unavailable")
            if selected:
                selection_complete = getattr(selected, "complete", None)
                complete = bool(selection_complete) and not truncated \
                    if selection_complete is not None else None
                candidate_count = getattr(selected, "candidate_count", None)
                selected_count = getattr(selected, "selected_count", None)
                return _source_bound_observation(
                    summary="\n".join(selected), task_text=task_text, authoritative=True,
                    complete=complete, selection_complete=selection_complete,
                    candidate_count=candidate_count, selected_count=selected_count,
                    truncated=truncated)
            if not isinstance(scope.get("record_query"), dict):
                # Legacy unstructured selection explicitly defines an empty
                # selection as no relevant source, not a bounded SQL query.
                return _source_bound_observation(
                    summary="没有找到与你查询目标对应的已保存记录。",
                    task_text=task_text, authoritative=True)
            return _source_bound_observation(
                summary="有候选记录，但本次未能确认哪些符合查询；请缩小范围后重试。",
                task_text=task_text, authoritative=False, status=ToolStatus.UNAVAILABLE,
                error="saved_record_selection_unavailable")
        if attempted:
            return _source_bound_observation(
                summary="没有找到与你查询条件对应的已保存记录。",
                task_text=task_text, authoritative=True)
        fields = list(dict.fromkeys(scope.get("fact_fields") or []))
        if fields:
            memory = _request_memory_snapshot()
            memory["short_memory"] = []
            active_scope = {
                **scope,
                "dialogue_ids": [],
                "context_ids": [],
                "critical_history_ids": [],
                "fact_fields": fields,
                "critical_fact_fields": [],
            }
            result = await _dialogue_answer(context, active_scope, task_text, memory)
            if result and result.get("answer") and result.get("status") not in {
                    "unavailable", "context_budget"}:
                return _source_bound_observation(
                    summary=str(result["answer"]), task_text=task_text, authoritative=True)
        return _source_bound_observation(
            summary="没有可用的已保存记录读取结果，无法确认当前值；本次没有使用旧对话原话代替。",
            task_text=task_text, authoritative=False, status=ToolStatus.UNAVAILABLE,
            error="saved_record_query_unavailable")
    if conversation_read:
        result = await _dialogue_answer(context, scope, task_text)
        if result and result.get("answer") and result.get("status") not in {
                "unavailable", "context_budget"}:
            return _source_bound_observation(
                summary=str(result["answer"]), task_text=task_text,
                authoritative=quoted, quoted=quoted,
                state_update_requested=bool(result.get("state_update_requested")))
        if quoted:
            summary = str((result or {}).get("answer") or "未能完成引用内容复述；本次没有保存任何内容。")
            return _source_bound_observation(
                summary=summary, task_text=task_text, authoritative=False, quoted=True,
                status=ToolStatus.UNAVAILABLE, error="quoted_answer_unavailable")
    return None


async def _single_pass_answer(
    args: Any, context: dict[str, Any], task_type: str, system_prompt: str,
) -> DomainObservation:
    from app.schemas.citation import citations_from_hits
    from app.services.agent.agent import agent
    from app.services.context.budget import ContextBudgetExceeded, context_budget
    from app.services.rag.evidence_conflict import (
        analyze_evidence_conflicts, answer_system_rules, controlled_evidence_answer,
        filter_hits_for_answer, finalize_evidence_answer,
    )
    from app.services.rag.online import search as rag_search
    from app.services.rag.quality import assess_rag_evidence

    query = (args or {}).get("query", "") if isinstance(args, dict) else ""
    query = str(query or "")
    task_text = str((context or {}).get("task_text") or query
                    or (context or {}).get("source_text") or "")
    source_bound = await _source_bound_dialogue_answer(context, task_text)
    if source_bound is not None:
        return source_bound

    semantic_scope = (context or {}).get("semantic_scope") or {}
    read_spec = semantic_scope.get("read_spec") if isinstance(semantic_scope, dict) else None
    read_facts = None
    read_prefix = ""
    read_unknowns = ""
    if isinstance(read_spec, dict) and str(read_spec.get("resource") or "") == "workout_records":
        from app.services.agent.read_resolver import advisory_read_facts
        resolved = advisory_read_facts(
            str((context or {}).get("user_id") or "demo-user"), read_spec)
        if resolved.get("status") != "completed":
            return DomainObservation(
                status=ToolStatus.UNAVAILABLE,
                error=str(resolved.get("error") or "workout_read_failed"),
                summary=str(resolved.get("answer") or "读取已保存训练暂时失败。"),
                result={"query": task_text},
            )
        read_facts = resolved.get("facts")
        read_prefix = str(resolved.get("prefix") or "")
        read_unknowns = str(resolved.get("unknowns") or "")

    # A short regeneration request is not a new retrieval topic. Restore its
    # selected current-conversation goal before retrieval, without treating an
    # old assistant answer as new user facts or relying on tool-written prose.
    answer_state = None
    if semantic_scope.get('result_reference'):
        answer_state = await _answer_state(context, task_type)
        if answer_state is None:
            return DomainObservation(status=ToolStatus.UNAVAILABLE, error='memory_source_unavailable',
                                     summary='所选回答来源已变更，无法可靠重做，请重新选择来源。')
        originals = [m['content'] for block in answer_state.get('task_context', [])
                     for m in block.get('referenced_result', []) if m.get('role') == 'user'
                     and (not block.get('goal_source_ids') or m['message_id'] in block['goal_source_ids'])]
        if semantic_scope['result_reference'].get('purpose', 'revise') == 'revise':
            query = '\n'.join([*originals, str(context.get('source_text') or task_text)])
        else:
            # Retrieve for the new task, not for the old training/diet command.
            # The verified background remains available in task_context.
            query = task_text

    try:
        hits = list(await rag_search(agent.graph.rag, query, top_k=5) or [])
    except Exception:
        return _retrieval_failure(query)

    if not hits:
        return _insufficient(task_type, query)

    # 非空 hit 不是充分知识证据：沿用既有 RAG 质量门，不足时不调用回答模型。
    quality = assess_rag_evidence(query, hits, task_type=task_type)
    if not quality.answerable:
        return _insufficient(task_type, query, evidence_count=len(hits), quality=quality)

    high_risk = task_type in {"safety_triage", "symptom_triage"}
    analysis = await analyze_evidence_conflicts(
        agent.graph.llm, query, hits, task_type=task_type, high_risk=high_risk)
    filtered = filter_hits_for_answer(hits, analysis)
    hits = list(filtered.hits)
    if filtered.applied:
        filtered_quality = assess_rag_evidence(query, hits, task_type=task_type)
        if not filtered_quality.answerable:
            return _insufficient(task_type, query, evidence_count=len(hits), quality=filtered_quality, filtered=True)
    controlled = controlled_evidence_answer(analysis, high_risk=high_risk)
    if controlled is not None:
        content = controlled
        if read_prefix:
            content = read_prefix + "\n\n" + content
        if read_unknowns:
            content = content + "\n\n" + read_unknowns
        return DomainObservation(
            status=ToolStatus.COMPLETED,
            summary=content,
            result={
                "query": query,
                "evidence_count": len(hits),
                "citations": citations_from_hits(hits),
            },
        )

    try:
        if answer_state is None:
            answer_state = await _answer_state(context, task_type)
        if answer_state is None:
            return DomainObservation(status=ToolStatus.UNAVAILABLE, error='memory_source_unavailable',
                                     summary='所需当前条件或持久约束已变更，请重试。')
        if read_facts is not None:
            answer_state["task_context"] = [
                *(answer_state.get("task_context") or []),
                {"read_spec_facts": read_facts.get("rows") or [],
                 "read_spec_prefix": read_prefix,
                 "read_spec_truncated": bool(read_facts.get("truncated")),
                 "read_spec_note": ("没有找到与本次日期/动作条件对应的已保存训练记录。"
                                    if not read_facts.get("rows") else "以下为应用读取的已保存训练记录。")},
            ]
        messages = context_budget.assemble(
            system=system_prompt + answer_system_rules(analysis),
            query=query,
            state=answer_state,
            hits=hits,
        )
    except ContextBudgetExceeded as exc:
        return _budget_failure(exc, query)

    try:
        answer = await agent.graph.llm.complete_message(messages)
    except LLMContextBudgetError as exc:
        return _budget_failure(exc, query)
    content = answer.get("content") if isinstance(answer, dict) else answer
    content = str(content or "").strip()
    if not content:
        return DomainObservation(
            status=ToolStatus.UNAVAILABLE,
            error="answer_model_empty",
            summary="回答模型未返回有效内容，请稍后重试；无需重复补充已有信息。",
            result={"query": query},
        )
    if read_prefix and read_prefix not in content:
        content = read_prefix + "\n\n" + content
    if read_unknowns and read_unknowns not in content:
        content = content + "\n\n" + read_unknowns
    content = finalize_evidence_answer(content, analysis, high_risk=high_risk)

    return DomainObservation(
        status=ToolStatus.COMPLETED,
        summary=content,
        result={
            "query": query,
            "evidence_count": len(hits),
            "citations": citations_from_hits(hits),
        },
    )


async def answer_training(args: Any, context: dict[str, Any]) -> DomainObservation:
    scope = _dialogue_scope(context)
    if (scope.get('source_task_type') == TaskType.SAFETY_TRIAGE.value
            or (context or {}).get('safety_level') == 'medical'):
        return await _single_pass_answer(args, context, TaskType.SAFETY_TRIAGE.value,
            _QA_SYSTEM + '本次为非急症症状相关安全问答，只提供一般信息及寻求专业评估的指引；'
                         '不得诊断、开具治疗或用药方案，不以训练计划替代症状评估。')
    return await _single_pass_answer(args, context, "training_qa", _QA_SYSTEM)


async def advise_nutrition(args: Any, context: dict[str, Any]) -> DomainObservation:
    return await _single_pass_answer(args, context, "nutrition_advice", _NUTRITION_SYSTEM)


async def _rag_hits(query: str) -> list[Any] | None:
    from app.services.agent.agent import agent
    from app.services.rag.online import search as rag_search

    try:
        return list(await rag_search(agent.graph.rag, query, top_k=5) or [])
    except Exception:
        return None


async def _delegated_plan_preview(context: dict[str, Any], source_text: str, *, reference_state=None) -> DomainObservation:
    """Reuse the application's timed-plan/preview worker from the ReAct catalog.

    The legacy plan worker already owns information grounding, timed rendering,
    preview state and the removal diff. ReAct calls the same compiled worker
    read-only, then returns its answer and ``plan_preview`` for the turn commit.
    """
    from app.services.agent.agent import agent

    worker = getattr(getattr(agent, "graph", None), "workers", {}).get("plan_adjust")
    if worker is None:
        return DomainObservation(
            status=ToolStatus.UNAVAILABLE, error="plan_worker_unavailable",
            summary="训练计划预览暂时不可用，请稍后重试。", result={"query": source_text})

    state = _plan_information_state(context)
    if reference_state:
        state['task_context'] = [*(state.get('task_context') or []), *(reference_state.get('task_context') or [])]
    memory = _request_memory_snapshot()
    user_id = str(context.get("user_id") or "demo-user")
    profile = dict(state.get("profile") or {})
    if not profile:
        try:
            from app.db import profile_repository
            profile = dict(profile_repository.get_profile(user_id) or {})
        except Exception:
            profile = {}
    if profile:
        state["profile"] = profile
    for name in ("long_memory", "short_memory", "information_sources",
                 "persistent_constraints", "current_states"):
        value = memory.get(name)
        if isinstance(value, (dict, list)):
            state.setdefault(name, value)
    state.update({
        "user_id": user_id,
        "conversation_id": str(context.get("conversation_id") or ""),
        "request_id": str(context.get("request_id") or ""),
        "current_plan_preview": (memory.get("current_plan_preview")
                                 or context.get("current_plan_preview") or {}),
    })

    try:
        raw = await worker.ainvoke(state)
    except Exception as exc:
        from app.services.context.budget import ContextBudgetExceeded
        from app.services.llm.client import LLMContextBudgetError
        if isinstance(exc, (ContextBudgetExceeded, LLMContextBudgetError)):
            return DomainObservation(
                status=ToolStatus.REJECTED, error="context_input_budget_exhausted",
                summary="当前修改所需的完整预览与约束超出上下文预算，未生成修改后的计划；请缩小修改范围后重试。",
                result={"query": source_text})
        return DomainObservation(
            status=ToolStatus.UNAVAILABLE, error="plan_worker_failed",
            summary="训练计划预览暂时不可用，请稍后重试。", result={"query": source_text})
    result = raw.get("result") if isinstance(raw, dict) else None
    if not isinstance(result, dict):
        return DomainObservation(
            status=ToolStatus.UNAVAILABLE, error="plan_worker_empty",
            summary="训练计划预览暂时不可用，请稍后重试。", result={"query": source_text})
    status = str(result.get("status") or "")
    answer = str(result.get("answer") or "")
    payload: dict[str, Any] = {"query": source_text,
                               "citations": list(result.get("citations") or [])}
    if isinstance(result.get("retrieval_quality"), dict):
        payload["retrieval_quality"] = result["retrieval_quality"]
    if result.get("evidence_filtered") is True:
        payload["evidence_filtered"] = True
    if isinstance(result.get("plan_preview"), dict):
        payload["plan_preview"] = result["plan_preview"]
    if status == "needs_confirmation":
        return DomainObservation(
            status=ToolStatus.NEEDS_CLARIFICATION,
            missing_fields=[str(item) for item in result.get("missing_fields") or []],
            summary=answer, result={**payload, "clarification": result.get("clarification")})
    if status in {"complete", "completed"}:
        return DomainObservation(status=ToolStatus.COMPLETED, summary=answer, result=payload)
    if status == "insufficient_evidence":
        # A grounded evidence refusal is a final domain outcome, not a
        # transient tool outage to retry until a supervisor limit hides it.
        return DomainObservation(status=ToolStatus.REJECTED,
                                 error="insufficient_evidence", summary=answer, result=payload)
    return DomainObservation(status=ToolStatus.UNAVAILABLE, error=status or "plan_unavailable",
                             summary=answer or "训练计划预览暂时无法生成，请稍后重试。",
                             result=payload)


async def plan_training(args: Any, context: dict[str, Any]) -> DomainObservation:
    """训练计划只读预览：先走来源/缺项门，再在 RAG/生成前澄清，最后交上下文。

    个人事实只能来自用户原始请求原文（``context.source_text``）；模型工具参数
    （``args.query``）永远不能成为个人信息来源。只有已完成的可信观察用于补资料，
    缺项在检索/生成之前澄清，通过的 planning_information 作为数据进入回答上下文。
    """
    from app.schemas.citation import citations_from_hits
    from app.services.agent.agent import agent
    from app.services.agent.task_information import check_plan_information
    from app.services.context.budget import ContextBudgetExceeded, context_budget
    from app.services.rag.evidence_conflict import (
        analyze_evidence_conflicts, answer_system_rules, controlled_evidence_answer,
        filter_hits_for_answer, finalize_evidence_answer,
    )
    from app.services.rag.quality import assess_rag_evidence

    source_text = str((context or {}).get("source_text") or "")
    query = (args or {}).get("query", "") if isinstance(args, dict) else ""
    query = str(query or "")

    scope = dict((context or {}).get("semantic_scope") or {})
    delivery_type = str(scope.get("delivery_type") or "")
    contract = (context or {}).get("plan_contract") or {}
    plan_result = (scope.get("plan_result") is True
                   or (isinstance(contract, dict) and contract.get("plan_result") is True))
    if delivery_type in {"plan", "preview"} or plan_result:
        # A concrete plan/preview result is application-owned (timed rendering and
        # preview state), so ReAct reuses the same compiled plan worker instead of
        # answering with generic advice.
        reference_state = None
        if (scope.get('result_reference') or {}).get('purpose') == 'context':
            reference_state = await _answer_state(context, 'training_plan')
            if reference_state is None:
                return DomainObservation(status=ToolStatus.UNAVAILABLE, error='memory_source_unavailable',
                                         summary='所参考回答的原文暂时无法确认，未生成新的训练计划。')
        return await _delegated_plan_preview(context, source_text, reference_state=reference_state)

    state = _plan_information_state(context)
    information, clarification = await check_plan_information(agent.graph.llm, state)
    if clarification:
        if clarification.category == "service_unavailable":
            # 信息模型/来源识别失败不是用户缺项：立即受限返回并保留真实失败
            # 分类；不得先检索或生成，也不把服务失败改写成证据不足。
            return DomainObservation(
                status=ToolStatus.UNAVAILABLE,
                error="information_service_unavailable",
                summary=clarification.question,
                result={"query": source_text, "clarification": clarification.model_dump()},
            )
        return DomainObservation(
            status=ToolStatus.NEEDS_CLARIFICATION,
            missing_fields=[field.name for field in clarification.fields],
            summary=clarification.question,
            result={"query": source_text, "clarification": clarification.model_dump()},
        )

    # 通过来源/缺项门后，planning_information 作为数据进入回答上下文（非指令）。
    hits = await _rag_hits(query or source_text)
    if hits is None:
        return _retrieval_failure(query or source_text)
    if not hits:
        return _insufficient("training_plan", query or source_text)
    quality = assess_rag_evidence(query or source_text, hits, task_type="training_plan")
    if not quality.answerable:
        return _insufficient("training_plan", query or source_text, evidence_count=len(hits), quality=quality)

    analysis = await analyze_evidence_conflicts(
        agent.graph.llm, query or source_text, hits, task_type="training_plan")
    filtered = filter_hits_for_answer(hits, analysis)
    hits = list(filtered.hits)
    if filtered.applied:
        filtered_quality = assess_rag_evidence(
            query or source_text, hits, task_type="training_plan")
        if not filtered_quality.answerable:
            return _insufficient("training_plan", query or source_text,
                                 evidence_count=len(hits), quality=filtered_quality, filtered=True)
    controlled = controlled_evidence_answer(analysis)
    if controlled is not None:
        return DomainObservation(
            status=ToolStatus.COMPLETED,
            summary=controlled,
            result={
                "query": source_text,
                "evidence_count": len(hits),
                "citations": citations_from_hits(hits),
            },
        )

    answer_state = await _answer_state(context, 'training_plan')
    if answer_state is None:
        return DomainObservation(status=ToolStatus.UNAVAILABLE, error='memory_source_unavailable',
                                 summary='所需当前条件或持久约束已变更，请重试。')
    answer_state["task_context"] = [
        *(answer_state.get("task_context") or []),
        {"planning_information": information},
    ]
    try:
        messages = context_budget.assemble(
            system=_PLAN_SYSTEM + answer_system_rules(analysis),
            query=query or source_text,
            state=answer_state,
            hits=hits,
        )
    except ContextBudgetExceeded as exc:
        return _budget_failure(exc, source_text)

    try:
        answer = await agent.graph.llm.complete_message(messages)
    except LLMContextBudgetError as exc:
        return _budget_failure(exc, source_text)
    content = answer.get("content") if isinstance(answer, dict) else answer
    content = str(content or "").strip()
    if not content:
        return DomainObservation(
            status=ToolStatus.UNAVAILABLE,
            error="answer_model_empty",
            summary="回答模型未返回有效内容，请稍后重试；无需重复补充已有信息。",
            result={"query": source_text},
        )
    content = finalize_evidence_answer(content, analysis)
    return DomainObservation(
        status=ToolStatus.COMPLETED,
        summary=content,
        result={
            "query": source_text,
            "evidence_count": len(hits),
            "citations": citations_from_hits(hits),
        },
    )


def candidate_record_query(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """The structured record query attached to one semantic target."""
    if not isinstance(candidate, dict):
        return None
    query = candidate.get("record_query")
    if isinstance(query, dict):
        return query
    payload = candidate.get("payload")
    if not isinstance(payload, dict):
        return None
    for key in ("dialogue", "semantic"):
        scope = payload.get(key)
        if isinstance(scope, dict) and isinstance(scope.get("record_query"), dict):
            return scope["record_query"]
    return None


def _query_date(value: Any, fallback):
    from datetime import date

    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value)
        except ValueError:
            return fallback
    return fallback


def saved_workout_observation(candidate: dict[str, Any], context: dict[str, Any]) -> DomainObservation:
    """Deterministic saved-workout answer owned by the application.

    A model that probed an auxiliary read (or chose no read at all) must not
    leave a structured workout goal unhandled.  This renders the goal's own
    date/action scope from SQL, so the target can complete without treating an
    auxiliary probe as proof and without another paid answer-model turn.
    """
    from app.core import clock
    from app.db import repositories
    from app.services.agent.workout_facts import workout_fact_line

    task_text = str((candidate or {}).get("text") or "")
    query = candidate_record_query(candidate)
    if not isinstance(query, dict) or not query.get("date_start") or not query.get("date_end"):
        return DomainObservation(
            status=ToolStatus.NEEDS_CLARIFICATION,
            error="saved_record_query_scope_missing",
            summary="这次读取已保存记录缺少日期范围，暂时无法确认；请补充范围后重试。",
            missing_fields=["date_range"],
            result={"query": task_text},
        )
    today = clock.today()
    start_date = _query_date(query.get("date_start"), today)
    end_date = _query_date(query.get("date_end"), today)
    raw_exercises = query.get("exercises")
    exercises = [str(item).strip() for item in raw_exercises
                 if str(item).strip()] if isinstance(raw_exercises, list) else []
    try:
        rows, truncated = repositories.query_workout_read_rows(
            str((context or {}).get("user_id") or "demo-user"),
            start_date=start_date, end_date=end_date,
            exercises=exercises or None, limit=30,
        )
    except Exception:
        return DomainObservation(
            status=ToolStatus.BLOCKED_CAPABILITY,
            error="saved_record_read_failed",
            summary="读取已保存记录暂时失败，无法确认当前值；本次没有使用旧对话原话代替。",
            result={"query": task_text},
        )
    if not rows:
        summary = "没有找到与你查询条件对应的已保存记录。"
    else:
        # The repository has already removed linked mirrors. Equal text can
        # represent independent records and must remain visible separately.
        lines = [workout_fact_line(row) for row in rows]
        summary = "已保存的记录：\n- " + "\n- ".join(lines)
        if truncated:
            summary += "\n\n匹配记录超过单次显示上限，以上仅为部分结果。"
    return DomainObservation(
        status=ToolStatus.COMPLETED,
        summary=summary,
        result={
            "query": task_text,
            "source_bound": True,
            "authoritative": True,
            "workouts": rows,
            "truncated": truncated,
            "complete": not truncated,
            "candidate_count": len(rows),
            "selected_count": len(rows),
        },
    )


ANSWER_WORKERS = {
    "answer_training": answer_training,
    "advise_nutrition": advise_nutrition,
    "plan_training": plan_training,
}


__all__ = ["ANSWER_WORKERS", "answer_training", "advise_nutrition", "plan_training",
           "candidate_record_query", "saved_workout_observation"]
