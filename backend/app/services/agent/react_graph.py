"""阶段A ReAct 独立循环图：prepare → main_agent ⇄ agent_tools → completion_check → finalize。

只实现协议与循环骨架，不替换 legacy 图，不接入 HTTP 默认路径，不开启 react 写入。
依赖注入使用 ``ReactDependencies``（模型接口 + 假执行器），身份/授权不进模型可填
参数；生产写 adapter 本轮显式关闭（blocked_capability），由 GLM 注入的假执行器可
模拟成功/失败观察，但不得称为真实保存验收。
"""

from __future__ import annotations

import contextvars
import copy
import inspect
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping

from langgraph.graph import END, START, StateGraph

from app.services.agent.react_models import (
    DomainObservation,
    ReactState,
    SourceScope,
    ToolCall,
    ToolStatus,
    is_int,
)
from app.services.agent.react_prompts import (
    RECOVERY_OBSERVATION_NOTICE,
    SUPERVISOR_SYSTEM,
    UNHANDLED_GOALS_HINT,
)
from app.services.agent.task_models import (
    RECORD_TASK_TYPES,
    TaskType,
    is_record_read,
    is_record_write,
    serves_read_goal,
)
from app.services.agent.react_tools import (
    ADVICE_TASK_TYPES,
    TOOL_CATALOG,
    bind_domain_task,
    tool_definitions,
    validate_source_ref,
    validate_tool_batch,
)
from app.services.security.safety import URGENT_SAFETY_ANSWER

logger = logging.getLogger(__name__)

# 首批逻辑上限（需经对照评测再定；缓存命中也消耗逻辑轮次）。
MAX_SUPERVISOR_TURNS = 4
MAX_TOOL_STEPS = 16
MAX_TASKS = 6
MAX_TOOL_STEPS_PER_TASK = 4
MAX_NO_PROGRESS = 2
# 主管物理模型尝试上限（含网关重试）；与逻辑轮次分开计数，仍受原 request/day/自动 stage 更严限制。
MAX_SUPERVISOR_PHYSICAL_ATTEMPTS = 4
SUPERVISOR_STAGE = "react_supervisor"

# 写任务的领域类型（用于覆盖/授权判断）；操作仍由 action/side_effect 判定。
_WRITE_TASK_TYPE_VALUES = RECORD_TASK_TYPES
_TERMINAL_WRITE_AUTH = frozenset({"rejected_unauthorized", "needs_confirmation"})

# 视为“确定性处理该调用”的观察状态（用于覆盖/无进展清零）。
# 注意：blocked_dependency 不在其内——它是“等待依赖”，不等于目标已处理完成。
_DEFINITE_STATUSES = frozenset({
    ToolStatus.COMPLETED,
    ToolStatus.PARTIAL,
    ToolStatus.REJECTED,
    ToolStatus.NEEDS_CLARIFICATION,
    ToolStatus.BLOCKED_CAPABILITY,
})

# 面向用户的只读目标类型（来自当前工具目录：读权限且带应用 task_type）。
# 辅助读（profile/history/summary/current_plan）没有task_type，不进入目标答案。
# Dispatch and completed-answer delivery share the same capability relation;
# compatible goals retain their domain labels even when a tool serves more
# than one type (for example non-urgent symptom education).
_READ_GOAL_TASK_TYPES = frozenset(
    goal.value for goal in TaskType if any(
        spec.permission == 'read' and serves_read_goal(spec.task_type, goal)
        for spec in TOOL_CATALOG.values() if spec.task_type))


@dataclass
class ReactDependencies:
    """依赖注入对象；模型接口与假执行器由 GLM 注入，不放入模型可填参数。

    ``llm`` 需提供 ``complete_message(messages, temperature, tools, tool_choice)``
    单步接口（生产复用 ``LLMClient.complete_message``，不嵌套 complete_with_tools）。
    ``tool_executor`` 为假执行器：callable ``(name, arguments, context) -> ...`` 或
    映射 ``{name: callable(arguments, context)}``（也可兼容三参数 ``(name, args,
    context)`` handler），可同步可异步，返回 ``DomainObservation`` 或 dict。
    生产不注入时，写工具返回 blocked_capability，只读/建议工具返回 unavailable。
    """

    llm: Any
    tool_executor: Any = None
    # 每请求冻结的只读目录（阶段B注入）；None 时回落阶段A原全目录。不能全局改
    # TOOL_CATALOG 以免并发串模式。
    tool_catalog: Any = None
    user_id: str = ""
    conversation_id: str = ""
    request_id: str = ""
    # 只读路径的预算/截止时间停止信号（默认无效果，保持阶段A接口不变）。
    policy_stop_check: Any = None
    # 写运行时可选持久化 hooks（默认 None：阶段A/B 行为与接口完全不变）。
    # hooks 可为同步/异步：model_intent(state) / model_result(assistant, state) /
    # tool_intent(call, state) / tool_result(call, observation, state)。
    persistence: Any = None
    # 写运行时：批处理后先做完成检查再决定是否发起下一次主管模型调用。
    stop_after_batch: bool = False
    max_supervisor_turns: int = MAX_SUPERVISOR_TURNS
    max_tool_steps: int = MAX_TOOL_STEPS
    max_tool_steps_per_task: int = MAX_TOOL_STEPS_PER_TASK
    max_tasks: int = MAX_TASKS
    max_no_progress: int = MAX_NO_PROGRESS


class PersistenceUnavailable(RuntimeError):
    """写运行时持久化 hook 显式失败（fail closed，不伪装模型不可用）。"""

    code = "write_persistence_failed"


async def _persist_hook(dependencies: "ReactDependencies", name: str, *args: Any) -> None:
    persistence = dependencies.persistence
    if persistence is None:
        return
    hook = getattr(persistence, name, None)
    if hook is None:
        return
    try:
        value = hook(*args)
        if inspect.isawaitable(value):
            await value
    except PersistenceUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - 持久化失败必须显式 fail closed
        raise PersistenceUnavailable(f"persistence_{name}_failed") from exc


def _active_catalog(dependencies: ReactDependencies) -> Any:
    """本请求冻结目录：注入的只读目录优先，否则阶段A原全目录。"""
    return dependencies.tool_catalog or TOOL_CATALOG


# ---------------------------------------------------------------------------
# 意图候选（覆盖输入，只读）
# ---------------------------------------------------------------------------
def _validated_read_source_hints(candidates: list[dict[str, Any]],
                                 source_text: str) -> list[dict[str, Any]]:
    """Application-owned source refs for validated read goals.

    The model should copy these spans instead of counting Unicode characters.
    They are navigation data only, never write authorization.
    """
    positions = _candidate_full_positions(candidates, source_text)
    hints=[]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        task_type = str(candidate.get("task_type") or "")
        action = str(candidate.get("action") or "answer")
        if (not task_type or candidate.get("side_effect") is True
                or is_record_write(candidate)
                or action not in {"answer", "question"}):
            continue
        span = positions.get(candidate.get("id"))
        if not isinstance(span, tuple) or len(span) != 2:
            continue
        start, end = span
        if not (is_int(start) and is_int(end) and 0 <= start < end <= len(source_text)):
            continue
        hints.append({
            "goal_id": str(candidate.get("id") or ""),
            "task_type": task_type,
            "source_ref": {"start": start, "end": end,
                           "exact_text": source_text[start:end]},
        })
    return hints


def _validated_write_source_hints(candidates: list[dict[str, Any]],
                                  source_text: str) -> list[dict[str, Any]]:
    """Application-owned source refs for authorized write goals.

    These spans are navigation data for the model to copy. They never authorize
    a write: the normal source-ref, candidate and authorization gates still run
    before any domain dispatch.
    """
    positions = _candidate_full_positions(candidates, source_text)
    hints = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        task_type = str(candidate.get("task_type") or "")
        action = str(candidate.get("action") or "")
        if (not is_record_write(candidate)
                or action not in {"record", "confirm"}
                or candidate.get("write_authorization") != "authorized"
                or candidate.get("side_effect") is not True):
            continue
        span = positions.get(candidate.get("id"))
        if not isinstance(span, tuple) or len(span) != 2:
            continue
        start, end = span
        if not (is_int(start) and is_int(end) and 0 <= start < end <= len(source_text)):
            continue
        hints.append({
            "goal_id": str(candidate.get("id") or ""),
            "task_type": task_type,
            "source_ref": {"start": start, "end": end,
                           "exact_text": source_text[start:end]},
        })
    return hints


def _rule_candidates(source_text: str) -> tuple[list[dict[str, Any]], str | None]:
    """Validated model goals when available, otherwise deterministic rule goals.

    A missing semantics adapter is a supported offline mode, so the rule plan
    (already passed through the same V3-V5 gate) is authoritative. A configured
    semantic call that failed is handled upstream as an explicit unavailable
    reply; ``plan_tasks_rule`` then yields a read-only fallback.
    """
    from app.services.agent.semantics import semantic_candidates

    try:
        candidates, error = semantic_candidates(source_text)
        if error is None:
            from app.services.agent.task_planner import validated_semantic_tasks
            enriched = validated_semantic_tasks(source_text)
            return (enriched if enriched is not None else candidates), None
        from app.services.agent.task_planner import validated_rule_candidates
        return validated_rule_candidates(source_text), None
    except Exception:
        logger.exception("react_rule_candidates_failed")
        # A candidate-generation crash is never the same thing as an empty
        # request. Keep an explicit failure marker even when the semantic cache
        # itself reported no error, so prepare cannot claim a complete run.
        return [], "rule_candidates_unavailable"


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def prepare_node(state: ReactState, dependencies: ReactDependencies) -> dict[str, Any]:
    """冻结 sanitized 原文、保存安全等级、构建 SourceScope 与初始消息、清零计数。"""
    raw = str(state.get("raw_message") or "")
    from app.services.security.input_guard import inspect_input
    from app.services.agent.semantics import safety_for

    guard = inspect_input(raw)
    source_text = guard.sanitized_text
    supplied = state.get("intent_candidates")
    if isinstance(supplied, list):
        candidates, candidate_error = copy.deepcopy(supplied), None
    else:
        candidates, candidate_error = _rule_candidates(source_text)
    too_many_tasks = len(candidates) > dependencies.max_tasks
    if too_many_tasks:
        candidates = candidates[:dependencies.max_tasks]
        candidate_error = "task_count_limit"
    safety = safety_for(raw)
    if safety.get("status") == "assessed":
        safety_level = str(safety.get("level") or "normal")
    else:
        # No valid model judgment is unknown, never a lexical emergency.
        safety_level = "unknown"
    source_scope = SourceScope(text=source_text, intent_candidates=candidates)
    out: dict[str, Any] = {
        "user_id": str(state.get("user_id") or dependencies.user_id),
        "conversation_id": str(state.get("conversation_id") or dependencies.conversation_id),
        "request_id": str(state.get("request_id") or dependencies.request_id),
        "safety_level": safety_level,
        "source_text": source_text,
        "source_scope": source_scope,
        "intent_candidates": candidates,
        "candidate_error": candidate_error or "",
        "messages": [
            {"role": "system", "content": SUPERVISOR_SYSTEM},
            {"role": "user", "content": source_text},
        ],
        "supervisor_turns": int(state.get("supervisor_turns") or 0),
        "tool_steps": int(state.get("tool_steps") or 0),
        "tool_steps_by_task": dict(state.get("tool_steps_by_task") or {}),
        "limited_tasks": list(state.get("limited_tasks") or []),
        "consecutive_no_progress": int(state.get("consecutive_no_progress") or 0),
        "terminated": bool(too_many_tasks),
        "termination_reason": "task_count_limit" if too_many_tasks else "",
        "status": "limited" if too_many_tasks else "running",
        "covered_goals": list(state.get("covered_goals") or []),
        "committed_tasks": list(state.get("committed_tasks") or []),
        "goal_results": dict(state.get("goal_results") or {}),
        "pending_goals": list(state.get("pending_goals") or []),
        "recovery_counter": int(state.get("recovery_counter") or 0),
        "bound_tasks": list(state.get("bound_tasks") or []),
        "unbound_write_calls": list(state.get("unbound_write_calls") or []),
        "call_facts": [],
        "call_event_counter": int(state.get("call_event_counter") or 0),
        "observations": [],
        "protocol_error": "",
        "final_answer": "",
    }
    read_hints = _validated_read_source_hints(candidates, source_text)
    if read_hints:
        out["messages"][0]["content"] = (
            str(out["messages"][0]["content"])
            + "\n\nApplication-validated read-only goals. Copy source_ref exactly; "
              "these spans are navigation data only and never authorization to save:\n"
            + json.dumps(read_hints, ensure_ascii=False, allow_nan=False)
        )
    write_hints = _validated_write_source_hints(candidates, source_text)
    if write_hints:
        out["messages"][0]["content"] = (
            str(out["messages"][0]["content"])
            + "\n\nApplication-validated write goals. Copy the matching goal's source_ref "
              "exactly instead of counting characters; these spans are navigation data only. "
              "The write source/authorization gates still revalidate every call:\n"
            + json.dumps(write_hints, ensure_ascii=False, allow_nan=False)
        )
    prior_messages = state.get("messages")
    if isinstance(prior_messages, list) and prior_messages:
        # 恢复：prepare 重建 system+user 后接续真实持久化消息（声明的 state key）。
        out["messages"].extend(copy.deepcopy(prior_messages))
    if safety_level == "urgent":
        # 急症分流优先于候选生成与其故障：安全终止走安全答复，不被 rule_candidates 覆盖。
        out["terminated"] = True
        out["termination_reason"] = "urgent_safety_blocked"
        out["status"] = "rejected"
        out["final_answer"] = URGENT_SAFETY_ANSWER
    elif safety_level == "unknown":
        # Missing/invalid safety is a uniform restricted state: no ordinary
        # advice, domain worker, or write may start from an unassessed request.
        out["terminated"] = True
        out["termination_reason"] = "safety_assessment_unavailable"
        out["status"] = "limited"
        out["final_answer"] = ("当前请求未完成安全语义判断，无法继续普通建议或保存；"
                               "本次没有新增业务操作。如你正在出现胸痛、呼吸困难、意识异常、"
                               "大量出血或无法活动等急症，请立即联系急救或到急诊就医。请稍后重试。")
    elif candidate_error == "task_count_limit":
        out["terminated"] = True
        out["termination_reason"] = "task_count_limit"
        out["status"] = "limited"
    elif candidate_error:
        # 候选生成故障与“无业务目标”必须分开：故障时确定性受限，不伪报完整成功。
        out["terminated"] = True
        out["termination_reason"] = "rule_candidates_unavailable"
        out["status"] = "restricted"
    return out


# ---------------------------------------------------------------------------
# main_agent
# ---------------------------------------------------------------------------
async def supervisor_step(state: ReactState, dependencies: ReactDependencies) -> dict[str, Any]:
    """构建消息，使用注入的单步模型接口，返回 assistant 消息。

    复用 LLMClient.complete_message（生产），不嵌套 complete_with_tools；用稳定
    stage 名包裹，缓存命中与网关重试都计入同一请求的主管额度。公开入口本层即检查
    急症，急症 state 零模型调用，返回确定性安全结果而非包装成模型失败。
    """
    if _state_is_urgent(state):
        return {"role": "assistant", "content": URGENT_SAFETY_ANSWER}
    messages = list(state.get("messages") or [])
    from app.services.llm.budget import model_stage
    from app.services.agent.task_budget import prepare_model_budget

    await _persist_hook(dependencies, "model_intent", state)
    # Shared understanding/safety/supervision is one fixed server-preparation
    # identity per root request. It is never re-bought per user task.
    with prepare_model_budget():
        with model_stage(SUPERVISOR_STAGE, max_calls=MAX_SUPERVISOR_PHYSICAL_ATTEMPTS):
            assistant = await dependencies.llm.complete_message(
                messages,
                temperature=0.0,
                tools=tool_definitions(_active_catalog(dependencies)),
                tool_choice="auto",
            )
    await _persist_hook(dependencies, "model_result", assistant, state)
    return assistant


def is_budget_policy_stop(exc: BaseException) -> bool:
    """Deprecated re-export; the shared implementation lives in the budget module."""
    from app.services.llm.budget import is_budget_policy_stop as shared
    return shared(exc)


def _policy_stop_requested(dependencies: ReactDependencies) -> bool:
    """统一停止入口：读取已注入的 policy-stop/截止时间信号（未注入时恒 False）。

    信号由只读运行时复用现有 request scope 提供；默认不改变阶段A目录/接口与旧
    模型预算行为。停止只影响此后不再发起模型/handler 调用，不回滚已读结果。
    """
    checker = dependencies.policy_stop_check
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        logger.warning("react_policy_stop_check_failed", exc_info=True)
        return False


def _last_pending_assistant(messages: list) -> dict | None:
    """最近一条带 tool_calls 的 assistant（其后的消息只有 tool/user，未开始新一轮主管）。

    写运行时恢复：历史保留原完整 batch + 已持久化 tool 结果；未派发调用由独立 dispatch
    cursor（persistence）决定，不由第二条 assistant 复制原调用续接。"""
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            calls = message.get("tool_calls")
            return message if isinstance(calls, list) and calls else None
    return None


async def _main_agent_node(state: ReactState, dependencies: ReactDependencies) -> dict[str, Any]:
    if state.get("terminated"):
        return {}
    # 急症输入在主管决策之前确定性安全终止（防御：prepare 已终止，直接节点调用也同等生效）。
    if _state_is_urgent(state):
        return {
            "terminated": True,
            "termination_reason": "urgent_safety_blocked",
            "status": "rejected",
            "final_answer": URGENT_SAFETY_ANSWER,
        }
    # 已有预算/截止时间停止信号：不再发起任何后续模型调用，明确受限终止。
    if _policy_stop_requested(dependencies):
        return {
            "terminated": True,
            "termination_reason": "policy_stop",
            "status": "limited",
        }
    # 每次实际模型调用之前检查逻辑轮次上限：第 4 次模型的 tool_calls 仍会被处理，
    # 之后不得有第 5 次模型调用。恢复时同样以真实持久计数为准：已达/超过上限时
    # 连可信恢复批也不得再派发（零新领域写）。
    if int(state.get("supervisor_turns") or 0) >= dependencies.max_supervisor_turns:
        return {
            "terminated": True,
            "termination_reason": "supervisor_turn_limit",
            "status": "limited",
        }
    # 写运行时恢复：仅真实可信的写恢复（已有持久 model_result 的未派发 native 调用）
    # 直接复用，不重发模型。判据来自真实持久化对象按已提交事件得出的判定，不使用公有
    # state 标记；同一恢复至多复用一次（派发/处理后不得重复复用）。普通只读与阶段 A/B
    # 路径（无真实写持久化）没有该判定，必须继续一轮主管模型调用。
    prior = state.get("messages") or []
    if prior:
        persistence = dependencies.persistence
        trusted_recovery = bool(
            persistence is not None
            and getattr(persistence, "resume_pending_native", False)
            and not getattr(persistence, "pending_native_reused", False))
        if trusted_recovery and _last_pending_assistant(prior) is not None:
            # 原 batch 的最后一条 assistant 仍有未完成 cursor（其后仅有已持久化 tool
            # 结果）：直接复用进入批处理，不重发模型、不复制第二条 assistant。
            try:
                persistence.pending_native_reused = True
            except Exception:  # noqa: BLE001 - 标记失败不改变本次复用判定
                logger.warning("react_pending_native_reuse_mark_failed", exc_info=True)
            return {}
    try:
        assistant = await supervisor_step(state, dependencies)
    except PersistenceUnavailable as exc:
        logger.warning("react_write_persistence_failed: %s", exc)
        return {
            "terminated": True,
            "termination_reason": "write_persistence_failed",
            "status": "limited",
            "protocol_error": str(exc),
        }
    except Exception as exc:
        from app.services.llm import budget as _budget_mod
        from app.services.agent.failure_details import exception_failure
        failure = exception_failure(exc)
        failure_code = failure['code'] if failure else ''

        # 真实分类取自预算层异常本身：它可能被直接抛出，也可能是外层网关错误在
        # __cause__ 上携带的真实原因（外层错误自身不携带预算 code）。图内不导入
        # Gateway/客户端类型，保持阶段 A/B 的无 Gateway 依赖边界。
        cause = exc.__cause__ if isinstance(exc.__cause__, BaseException) else None
        budget_error = None
        if isinstance(exc, _budget_mod.ModelBudgetError):
            budget_error = exc
        elif isinstance(cause, _budget_mod.ModelBudgetError):
            budget_error = cause
        category = str(getattr(getattr(exc, "category", None), "name", None) or "")
        write_runtime = dependencies.persistence is not None
        policy = budget_error is not None or category == "POLICY"
        scope_missing = (budget_error is not None
                         and str(getattr(budget_error, "code", "") or "")
                         == "model_budget_scope_required")
        if policy:
            if write_runtime and not scope_missing:
                # 真实写运行时的计量/额度拒绝：明确受限终止，不再真实重试。
                logger.warning("react_supervisor_budget_stopped: %s", type(exc).__name__)
                return {
                    "terminated": True,
                    "termination_reason": "model_budget_exhausted",
                    "status": "limited",
                    "protocol_error": f"model_budget:{type(exc).__name__}",
                    "model_failure_code": failure_code,
                }
            # 普通只读/阶段 A/B（含缺预算 scope）：保持旧 generic 受限形状，零重试。
            logger.warning("react_supervisor_policy_stopped: %s", type(exc).__name__)
            return {
                "terminated": True,
                "termination_reason": "model_unavailable",
                "status": "limited",
                "protocol_error": f"model_unavailable:{type(exc).__name__}",
                "model_failure_code": failure_code,
            }
        logger.warning("react_supervisor_call_failed: %s", type(exc).__name__)
        # 物理尝试上限（仅真实写运行时）：仅对真实可重试的上游错误在同一身份再重试
        # 一轮（无领域派发），总物理尝试不超过 supervisor_physical_attempts；普通只读
        # 路径不新增重试，其余失败保持旧语义。
        retryable_upstream = (write_runtime and category == "UPSTREAM"
                              and bool(getattr(exc, "retryable", False)))
        if not retryable_upstream:
            return {
                "terminated": True,
                "termination_reason": "model_unavailable",
                "status": "limited",
                "protocol_error": f"model_unavailable:{type(exc).__name__}",
                "model_failure_code": failure_code,
            }
        try:
            assistant = await supervisor_step(state, dependencies)
        except Exception as exc2:
            logger.warning("react_supervisor_retry_failed: %s", type(exc2).__name__)
            return {
                "terminated": True,
                "termination_reason": "model_unavailable",
                "status": "limited",
                "protocol_error": f"model_unavailable:{type(exc2).__name__}",
                "model_failure_code": (exception_failure(exc2) or {}).get("code", ""),
            }
    if not isinstance(assistant, dict):
        return {
            "terminated": True,
            "termination_reason": "model_unavailable",
            "status": "restricted",
            "protocol_error": "assistant_not_object",
        }
    messages = list(state.get("messages") or [])
    content = assistant.get("content") if isinstance(assistant.get("content"), str) else ""
    assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
    tool_calls = assistant.get("tool_calls")
    if tool_calls is not None:
        assistant_message["tool_calls"] = tool_calls
    messages.append(assistant_message)
    return {
        "messages": messages,
        "supervisor_turns": int(state.get("supervisor_turns") or 0) + 1,
    }


def _route_main_agent(state: ReactState, dependencies: ReactDependencies | None = None) -> str:
    if state.get("terminated"):
        return "finalize"
    messages = state.get("messages") or []
    if not messages:
        return "finalize"
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        # 写运行时恢复：最近 assistant 仍是未完成 batch（其后仅有已持久化 tool 结果），
        # 直接进入批处理续接独立 cursor；阶段A/B 保持原 finalize 行为。
        if dependencies is not None and dependencies.persistence is not None \
                and _last_pending_assistant(messages) is not None:
            return "agent_tools"
        return "finalize"
    tool_calls = last.get("tool_calls")
    if tool_calls is None:
        return "completion_check"
    if isinstance(tool_calls, list) and not tool_calls:
        return "completion_check"
    # 非空 list（或非法容器）：交给 agent_tools 做整批容器/schema 校验。
    return "agent_tools"


# ---------------------------------------------------------------------------
# agent_tools / 单调用执行
# ---------------------------------------------------------------------------
def _execution_context(state: ReactState, dependencies: ReactDependencies, call=None) -> dict[str, Any]:
    return {
        "user_id": str(state.get("user_id") or dependencies.user_id),
        "conversation_id": str(state.get("conversation_id") or dependencies.conversation_id),
        "request_id": str(state.get("request_id") or dependencies.request_id),
        "source_text": state.get("source_text") or "",
        "safety_level": state.get("safety_level") or "unknown",
        "source_scope": state.get("source_scope"),
        "observations": list(state.get("observations") or []),
        "covered_goals": list(state.get("covered_goals") or []),
        "committed_tasks": list(state.get("committed_tasks") or []),
        "tool_call_id": str(getattr(call, "id", "") or ""),
    }


def _resolve_executor(executor: Any, name: str) -> tuple[Any, bool]:
    if isinstance(executor, Mapping):
        return executor.get(name), True
    return executor, False


def _invoke_mapping_handler(handler: Any, name: str, args: dict[str, Any], context: dict[str, Any]) -> Any:
    """按签名调用 mapping handler：两参 (args, context) 优先，三参 (name, args, context) 兼容。

    不能“捕获 TypeError 再换参数重试”，以免执行器内部错误或已发生的副作用被二次执行。
    """
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return handler(args, context)
    positional = [
        p for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if any(p.kind == p.VAR_POSITIONAL for p in signature.parameters.values()):
        return handler(args, context)
    required = sum(1 for p in positional if p.default is p.empty)
    if len(positional) >= 3 and required >= 3:
        return handler(name, args, context)
    return handler(args, context)


async def _invoke_executor(executor: Any, call, context: dict[str, Any]) -> DomainObservation | None:
    handler, is_mapping = _resolve_executor(executor, call.name)
    if handler is None:
        return None
    try:
        if is_mapping:
            result = _invoke_mapping_handler(handler, call.name, call.args, context)
        else:
            result = handler(call.name, call.args, context)
        if inspect.isawaitable(result):
            result = await result
        # 统一返回边界校验：dict 与直接 DomainObservation 都重新归一化，断开可变引用，
        # 处理构造后被修改的 result/回执/状态，不因 isinstance 直接返回原对象。
        return DomainObservation.normalize(result)
    except Exception as exc:
        from app.services.agent.failure_details import exception_failure
        detail = exception_failure(exc)
        if detail:
            return DomainObservation(status=ToolStatus.UNAVAILABLE, error=detail['code'], summary=detail['message'])
        return DomainObservation(status=ToolStatus.UNAVAILABLE, error=f"executor_error:{type(exc).__name__}")


def _find_unconsumed(source_text: str, text: str, consumed: list[tuple[int, int]]) -> int:
    start = 0
    while True:
        index = source_text.find(text, start)
        if index < 0:
            return -1
        if not any(cs < index + len(text) and index < ce for cs, ce in consumed):
            return index
        start = index + 1


# 写运行时激活标记（仅在 invoke_react 的调用上下文内为 True）：控制写运行时的
# 事实区间定位（命令前缀剥离 + 空白不敏感映射），阶段A/B 默认行为不变。
_WRITE_RUNTIME_ACTIVE = contextvars.ContextVar("react_write_runtime_active", default=False)


def _locate_candidate(source_text: str, candidate: dict[str, Any], consumed: list[tuple[int, int]],
                      ws_fallback: bool) -> tuple[int, int] | None:
    """定位候选的完整真实区间（未剥离命令前缀），保持原有非法显式区间拒绝。"""
    text = candidate.get("text")
    start = candidate.get("source_start")
    end = candidate.get("source_end")
    if (start is None and end is None) or (
        is_int(start) and is_int(end) and start == -1 and end == -1
    ):
        if not isinstance(text, str) or not text:
            return None
        index = _find_unconsumed(source_text, text, consumed)
        if index >= 0:
            return (index, index + len(text))
        if ws_fallback:
            return _find_unconsumed_ws(source_text, text, consumed)
        return None
    if is_int(start) and is_int(end) and 0 <= start <= end <= len(source_text) \
            and isinstance(text, str) and source_text[start:end] == text:
        return (start, end)
    return None


def _candidate_positions(
    candidates: list[dict[str, Any]], source_text: str, *, ws_fallback: bool = False
) -> dict[Any, tuple[int, int]]:
    """给每个候选分配来源区间身份（写运行时按事实区间，供匹配使用）。"""
    positions: dict[Any, tuple[int, int]] = {}
    consumed: list[tuple[int, int]] = []
    for candidate in candidates:
        span = _locate_candidate(source_text, candidate, consumed, ws_fallback)
        if span is None:
            continue
        if ws_fallback:
            span = _strip_command_prefix(source_text, span) or span
        positions[candidate.get("id")] = span
        consumed.append(span)
    return positions


def _candidate_full_positions(
    candidates: list[dict[str, Any]], source_text: str, *, ws_fallback: bool = False
) -> dict[Any, tuple[int, int]]:
    """候选的完整真实区间（含命令/元数据前缀），与匹配同一有序消费顺序。"""
    positions: dict[Any, tuple[int, int]] = {}
    consumed: list[tuple[int, int]] = []
    for candidate in candidates:
        span = _locate_candidate(source_text, candidate, consumed, ws_fallback)
        if span is None:
            continue
        positions[candidate.get("id")] = span
        consumed.append(span)
    return positions


# 写运行时：规划器候选可能包含写命令前缀（“帮我记录今天午餐：”）。这里只剥掉
# 有限明确的写命令词并推进到命令/目标分隔的冒号之后——绝不把 昨天/晚餐/今天 这类
# 业务事实当作命令词剥掉；候选映射到冒号后完整真实事实区间，绝不按模型 ref 倒推。
def _write_child_candidates(candidates: list[dict[str, Any]],
                            parent_id: str) -> list[dict[str, Any]]:
    return [
        candidate for candidate in candidates
        if isinstance(candidate, dict)
        and str(candidate.get("parent_goal_id") or "") == parent_id
    ]


def _write_parent_positions(
    parent: dict[str, Any], source_text: str,
    candidates: list[dict[str, Any]],
) -> tuple[int, int] | None:
    # Coverage is over this task's fact region, not its command suffix
    # ("请分别保存这两项" contains 两, which is not a workout quantity).
    position = _candidate_positions(candidates, source_text, ws_fallback=True).get(
        parent.get("id"))
    if position is not None:
        return position
    start, end = parent.get("source_start"), parent.get("source_end")
    if (is_int(start) and is_int(end) and 0 <= start < end <= len(source_text)
            and isinstance(parent.get("text"), str)
            and source_text[start:end] == parent.get("text")):
        return start, end
    return _candidate_full_positions(candidates, source_text, ws_fallback=True).get(
        parent.get("id"))


def _materialize_write_subgoal(call, spec, state: ReactState) -> dict[str, Any] | None:
    """Bind one action ref inside a broader authorized workout task.

    A semantic task may legitimately contain several independent actions. The
    model may cite each action's own contiguous fact span; that narrower ref is
    an action subgoal under the same authorized parent, never a new authorization
    and never a fragment that can borrow a sibling's fields. The adapter still
    extracts exactly one record from the cited span and the write bridge still
    authorizes the parent task.
    """
    if not _WRITE_RUNTIME_ACTIVE.get() or spec.permission != "write" \
            or spec.task_type != TaskType.WORKOUT_RECORD.value:
        return None
    args = call.args if isinstance(call.args, dict) else {}
    ref = args.get("source_ref")
    if not isinstance(ref, dict):
        return None
    start, end = ref.get("start"), ref.get("end")
    source_text = str(state.get("source_text") or "")
    if not (is_int(start) and is_int(end) and 0 <= start < end <= len(source_text)):
        return None
    if end - start < 2:
        return None
    candidates = state.get("intent_candidates")
    if not isinstance(candidates, list):
        return None
    full_positions = _candidate_full_positions(candidates, source_text, ws_fallback=True)
    fact_positions = _candidate_positions(candidates, source_text, ws_fallback=True)
    parents: list[tuple[dict[str, Any], tuple[int, int]]] = []
    overlapping: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("task_type") != spec.task_type:
            continue
        position = full_positions.get(candidate.get("id"))
        if position is None:
            continue
        if position[0] < end and start < position[1]:
            overlapping.append(candidate)
        fact_position = fact_positions.get(candidate.get("id"))
        covers_fact = (
            isinstance(fact_position, tuple) and len(fact_position) == 2
            and start <= fact_position[0] and fact_position[1] <= end)
        if (position[0] <= start and end <= position[1]
                and (position[0], position[1]) != (start, end)
                and not covers_fact
                and candidate.get("parent_goal_id") is None):
            parents.append((candidate, position))
    if len(parents) != 1 or len(overlapping) != 1:
        return None
    parent, _position = parents[0]
    allowed = (
        is_record_write(parent)
        and parent.get("action") in {"record", "confirm"}
        and parent.get("write_authorization") == "authorized"
        and parent.get("side_effect") is True
    )
    if not allowed:
        return None
    parent_id = str(parent.get("id") or "")
    if not parent_id:
        return None
    child_id = f"{parent_id}::write::{start}-{end}"
    for candidate in candidates:
        if isinstance(candidate, dict) and str(candidate.get("id") or "") == child_id:
            return candidate
    child = copy.deepcopy(parent)
    child.update({
        "id": child_id,
        "text": source_text[start:end],
        "source_start": start,
        "source_end": end,
        "parent_goal_id": parent_id,
        "parent_goal_text": str(parent.get("text") or ""),
    })
    payload = child.get("payload")
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["source_start"] = start
        payload["source_end"] = end
        child["payload"] = payload
    child.pop("_source_span", None)
    candidates.append(child)
    return child


def _prefer_write_ancestors(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A full-parent ref must not also mark its action children handled."""
    matched_ids = {str(candidate.get("id") or "") for candidate in matches}
    parents = {str(candidate.get("id") or ""): str(candidate.get("parent_goal_id") or "")
               for candidate in matches}
    result = []
    for candidate in matches:
        current = parents.get(str(candidate.get("id") or ""), "")
        skip = False
        while current:
            if current in matched_ids:
                skip = True
                break
            current = parents.get(current, "")
        if not skip:
            result.append(candidate)
    return result


def _write_parent_handled(parent: dict[str, Any], state: ReactState) -> bool:
    """Whether every numeric fact in a split authorized task is accounted for.

    Children are action-level calls under the same parent. A child is handled
    only by a trusted commit or an explicit per-action clarification. Numeric
    token coverage is technical syntax, not action-name classification: it
    prevents a single shorthand from silently completing a multi-action parent
    while a later action's numbers were never cited.
    """
    candidates = state.get("intent_candidates") or []
    parent_id = str(parent.get("id") or "")
    payload = parent.get("payload") if isinstance(parent.get("payload"), dict) else {}
    semantic = payload.get("semantic") if isinstance(payload.get("semantic"), dict) else {}
    workout_units = (payload.get("workout_sources") or semantic.get("workout_sources")
                     or parent.get("workout_sources"))
    if isinstance(workout_units, list) and workout_units:
        from app.services.agent.workout_extractor import normalize_workout_text

        expected = [normalize_workout_text(str(item.get("exercise") or ""))
                    for item in workout_units if isinstance(item, dict)]
        expected = [item for item in expected if item]
        goal_results = state.get("goal_results") or {}
        observations = [goal_results.get(parent_id)]
        observations.extend(goal_results.get(str(child.get("id") or ""))
                            for child in _write_child_candidates(candidates, parent_id))
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            status = str(observation.get("status") or "")
            if status not in {ToolStatus.COMPLETED.value,
                              ToolStatus.NEEDS_CLARIFICATION.value}:
                continue
            result = observation.get("result")
            unit_results = result.get("workout_units") if isinstance(result, dict) else None
            if isinstance(unit_results, list) and expected:
                actual = [normalize_workout_text(str(item.get("exercise") or ""))
                          for item in unit_results if isinstance(item, dict)]
                actual = [item for item in actual if item]
                if actual == expected:
                    return True
            elif len(expected) == 1 and status == ToolStatus.COMPLETED.value \
                    and observation.get("commit_state") == "committed":
                return True
        return False
    if parent_id in {str(item) for item in state.get("covered_goals") or []}:
        return True
    children = _write_child_candidates(candidates, parent_id)
    if not children:
        return False
    goal_results = state.get("goal_results") or {}
    committed = {str(item) for item in state.get("committed_tasks") or []}
    handled_spans: list[tuple[int, int]] = []
    for child in children:
        child_id = str(child.get("id") or "")
        observation = goal_results.get(child_id)
        if not isinstance(observation, dict):
            return False
        status = str(observation.get("status") or "")
        if child_id not in committed and status != ToolStatus.NEEDS_CLARIFICATION.value:
            return False
        start, end = child.get("source_start"), child.get("source_end")
        if not (is_int(start) and is_int(end) and 0 <= start < end):
            return False
        handled_spans.append((start, end))
    source_text = str(state.get("source_text") or "")
    position = _write_parent_positions(parent, source_text, candidates)
    if position is None:
        return False
    parent_start, parent_end = position
    from app.services.agent.workout_syntax import numeric_token_spans

    for rel_start, rel_end in numeric_token_spans(source_text[parent_start:parent_end]):
        absolute_start = parent_start + rel_start
        absolute_end = parent_start + rel_end
        if not any(start <= absolute_start and absolute_end <= end
                   for start, end in handled_spans):
            return False
    return True


def _strip_command_prefix(source_text: str, span: tuple[int, int]):
    """Use the model's exact fact span, never a command-prefix expression."""
    from app.services.agent.semantics import get_understanding
    understanding = get_understanding(source_text) or {}
    begin, end = span
    text = source_text[begin:end]
    for task in understanding.get("tasks") or []:
        fact = task.get("fact_text")
        if task.get("text") == text and isinstance(fact, str) and fact.strip():
            offset = text.find(fact)
            if offset >= 0 and text.find(fact, offset+1) < 0:
                return begin+offset, begin+offset+len(fact)
    return None


def _find_unconsumed_ws(source_text: str, text: str, consumed: list[tuple[int, int]]):
    """空白不敏感定位：把候选文本（规划器已去空白）映射回真实 source 坐标。"""
    stripped = re.sub(r"\s+", "", text)
    if not stripped:
        return None
    index_map: list[int] = []
    for position, char in enumerate(source_text):
        if not char.isspace():
            index_map.append(position)
    compact = "".join(char for char in source_text if not char.isspace())
    search_from = 0
    while True:
        found = compact.find(stripped, search_from)
        if found < 0:
            return None
        begin = index_map[found]
        end = index_map[found + len(stripped) - 1] + 1
        if not any(cs < end and begin < ce for cs, ce in consumed):
            return (begin, end)
        search_from = found + 1


def _read_ref_is_valid(source_text: str, source_ref: Any) -> bool:
    """只读引用先做真实类型/边界/exact_text 校验（非空，不接受伪造文本）。"""
    if not isinstance(source_ref, dict):
        return False
    start = source_ref.get("start")
    end = source_ref.get("end")
    exact = source_ref.get("exact_text")
    if not is_int(start) or not is_int(end) or not isinstance(exact, str):
        return False
    if not (0 <= start < end <= len(source_text)):
        return False
    return source_text[start:end] == exact


def _read_goal_matches(
    call, spec, state: ReactState
) -> tuple[list[dict[str, Any]], str | None]:
    """只读目标身份匹配：合法局部/完整 ref 归属唯一同类型目标，歧义不猜。

    返回 ``(候选, 问题)``。问题为 ``invalid_source_ref``（类型/边界/exact_text
    不真实）、``ambiguous_goal``（ref 跨多个独立同类型目标）或
    ``no_matching_goal``（合法 ref 但没有可归属目标）。无 ref 时保留既有合同：
    按类型取首个候选，辅助读（task_type 为空）不在这里产生目标身份。
    """
    candidates = state.get("intent_candidates") or []
    source_text = state.get("source_text") or ""
    write_runtime = bool(_WRITE_RUNTIME_ACTIVE.get())
    positions = _candidate_positions(candidates, source_text, ws_fallback=write_runtime)
    full_positions = _candidate_full_positions(candidates, source_text, ws_fallback=write_runtime)
    args = call.args if isinstance(call.args, dict) else {}
    source_ref = args.get("source_ref")
    if source_ref is None:
        # Prefer the exact goal type before the broader answer-domain match.
        # An advice call must bind the advice goal, not an earlier record goal
        # that merely shares the same answer domain (which would also hide the
        # advice goal's record dependencies).
        exact = [candidate for candidate in candidates
                 if candidate.get("task_type") == spec.task_type]
        compatible = exact or [
            candidate for candidate in candidates
            if serves_read_goal(spec.task_type, candidate.get("task_type"))]
        for candidate in compatible:
            enriched = dict(candidate)
            enriched["_source_span"] = full_positions.get(candidate.get("id"))
            return [enriched], None
        return [], None
    if not _read_ref_is_valid(source_text, source_ref):
        return [], "invalid_source_ref"
    start, end = source_ref["start"], source_ref["end"]
    overlapping: list[dict[str, Any]] = []
    related: list[dict[str, Any]] = []
    for candidate in candidates:
        if not serves_read_goal(spec.task_type, candidate.get("task_type")):
            continue
        pos = full_positions.get(candidate.get("id")) or positions.get(candidate.get("id"))
        if pos is None:
            continue
        # 任意部分擦过都视为同一引用触及了该目标；真正可归属必须是完整包含关系。
        if not (pos[0] < end and start < pos[1]):
            continue
        overlapping.append(candidate)
        if (pos[0] <= start and end <= pos[1]) or (start <= pos[0] and pos[1] <= end):
            enriched = dict(candidate)
            enriched["_source_span"] = full_positions.get(candidate.get("id"))
            related.append(enriched)
    if len(overlapping) > 1 or len(related) > 1:
        return [], "ambiguous_goal"
    if len(related) == 1:
        return related, None
    return [], "no_matching_goal"


def _matching_candidates(call, spec, state: ReactState) -> list[dict[str, Any]]:
    """返回本次调用对应的意图候选：写工具按区间充分对应绑定，读取按目标身份匹配。

    写：ref 必须完整覆盖候选事实位置（事实 ⊆ ref），保持严格授权范围；读取：
    合法真实局部/完整 ref 必须归属唯一同类型目标，不能用写工具的完整覆盖规则
    误拒局部读取，也不能让一个 ref 覆盖多个独立目标。无 ref 的读取保留原合同。
    """
    if spec.permission != "write" and spec.task_type:
        matched, _issue = _read_goal_matches(call, spec, state)
        return matched
    candidates = state.get("intent_candidates") or []
    source_text = state.get("source_text") or ""
    write_runtime = bool(_WRITE_RUNTIME_ACTIVE.get())
    positions = _candidate_positions(candidates, source_text, ws_fallback=write_runtime)
    full_positions = _candidate_full_positions(candidates, source_text, ws_fallback=write_runtime)
    args = call.args if isinstance(call.args, dict) else {}
    source_ref = args.get("source_ref")
    start = end = None
    if isinstance(source_ref, dict) and is_int(source_ref.get("start")) and is_int(source_ref.get("end")):
        start = source_ref["start"]
        end = source_ref["end"]
    matched: list[dict[str, Any]] = []
    if call.name == "correct_workout":
        for candidate in candidates:
            semantic = (candidate.get("payload") or {}).get("semantic") or {}
            correction = semantic.get("workout_correction") or candidate.get("workout_correction")
            pos = full_positions.get(candidate.get("id"))
            if (correction and pos is not None and start is not None and end is not None
                    and pos[0] <= start < end <= pos[1]
                    and correction.get("exercise_quote") in source_text[start:end]):
                matched.append({**candidate, "_source_span": pos})
        return matched if len(matched) == 1 else []
    for candidate in candidates:
        if candidate.get("task_type") != spec.task_type:
            continue
        if start is not None and end is not None:
            pos = positions.get(candidate.get("id"))
            if pos is None:
                continue
            # 完整事实对应：ref 必须完整覆盖候选事实位置（事实 ⊆ ref），缺事实的 ref 不能
            # 证明该目标完成；移除“大于一半”比例授权，80%/90% 等比例同样不合格。
            if not (start <= pos[0] and pos[1] <= end):
                continue
        enriched = dict(candidate)
        enriched["_source_span"] = full_positions.get(candidate.get("id"))
        matched.append(enriched)
        if start is None:
            break
    if matched:
        return _prefer_write_ancestors(matched)
    child = _materialize_write_subgoal(call, spec, state)
    if child is not None:
        enriched = dict(child)
        enriched["_source_span"] = (start, end)
        return [enriched]
    return []


def _matched_ids(call, spec, state: ReactState) -> list[str]:
    return [str(c.get("id")) for c in _matching_candidates(call, spec, state) if c.get("id")]


def _candidate_write_blocker(call, state: ReactState) -> DomainObservation | None:
    """候选携带的模型写授权/拒绝必须先于派发生效（阶段A/B/持久写）。

    2026-10-18 合同把“记录”判定交给模型，否定/假设/第三方 flags 与
    ``semantic_intent_unavailable`` 保存在应用候选上。模型随后选择写工具并只引用
    事实片段时（ref 可能只是候选 span 的子集），授权仍必须按候选判定：拒绝候选
    零派发，需确认候选澄清。持久化路径与无持久化路径使用同一候选门，不能因 bridge
    自带 span 校验而复活一次已被语义拒绝/非写决定的调用。
    """
    source_ref = call.args.get("source_ref") if isinstance(call.args, dict) else None
    if not isinstance(source_ref, dict):
        return None
    start = source_ref.get("start")
    end = source_ref.get("end")
    if not is_int(start) or not is_int(end):
        return None
    candidates = state.get("intent_candidates") or []
    positions = _candidate_positions(candidates, state.get("source_text") or "")
    overlapping = []
    for candidate in candidates:
        span = positions.get(candidate.get("id"))
        if span is None or not (span[0] < end and start < span[1]):
            continue
        overlapping.append(candidate)
    if not overlapping:
        return None
    # Protocol-control synthetic candidates from phases A/B may not carry a
    # semantic decision at all; leave their legacy structural path unchanged.
    if not any(
            candidate.get("write_authorization") is not None
            or candidate.get("side_effect") is not None
            or candidate.get("semantic_available") is not None
            for candidate in overlapping):
        return None
    # A rejected or confirmation-requiring overlapping candidate must never be
    # covered by an authorized sibling. Only when every overlapping candidate is
    # itself an authorized record/confirm write may the call proceed.
    rejected = None
    confirmation: DomainObservation | None = None
    all_authorized = True
    for candidate in overlapping:
        authorization = candidate.get("write_authorization")
        reason = str(candidate.get("write_rejection_reason") or "")
        allowed = (is_record_write(candidate)
                   and candidate.get("action") in {"record", "confirm"}
                   and authorization == "authorized"
                   and candidate.get("side_effect") is True)
        if not allowed:
            all_authorized = False
        if authorization == "rejected_unauthorized" and rejected is None:
            rejected = DomainObservation(
                status=ToolStatus.REJECTED, error=reason or "write_authorization_rejected")
        if authorization == "needs_confirmation" and confirmation is None:
            confirmation = DomainObservation(
                status=ToolStatus.NEEDS_CLARIFICATION,
                missing_fields=["write_authorization"],
                error=reason or "write_authorization_required",
            )
    if rejected is not None:
        return rejected
    if confirmation is not None:
        return confirmation
    if all_authorized:
        return None
    # Semantic question/nonwrite, missing authorization or a flagged write that
    # no longer carries side effects must never be revived by a later tool call.
    return DomainObservation(status=ToolStatus.REJECTED, error="semantic_write_not_authorized")


def _is_write_commit(observation: DomainObservation, spec) -> bool:
    """只有写领域任务、明确 committed、合法假回执证明且完整完成才算可信提交。"""
    return (
        spec is not None
        and spec.permission == "write"
        and observation.status == ToolStatus.COMPLETED
        and observation.commit_state == "committed"
        and bool(observation.receipt_refs)
    )


def _record_observation(
    matched: list[str],
    observation: DomainObservation,
    spec,
    covered: list[str],
    committed: list[str],
    goal_results: dict[str, dict[str, Any]],
) -> None:
    """把确定性观察覆盖到的候选标记为已处理；仅可信写提交进入 committed_tasks。"""
    for cid in matched:
        if cid not in covered:
            covered.append(cid)
        goal_results[cid] = observation.to_dict()
        if _is_write_commit(observation, spec) and cid not in committed:
            committed.append(cid)


def _dependency_blocker(call, spec, state: ReactState) -> DomainObservation | None:
    """建议/计划依赖本轮尚未可信提交的记录时，返回 blocked_dependency。"""
    if spec.task_type not in ADVICE_TASK_TYPES:
        return None
    candidates = _matching_candidates(call, spec, state)
    if not candidates:
        return None
    deps = candidates[0].get("depends_on") or []
    if not deps:
        return None
    committed = set(state.get("committed_tasks") or [])
    missing = [d for d in deps if d not in committed]
    if missing:
        return DomainObservation(
            status=ToolStatus.BLOCKED_DEPENDENCY,
            error="record_dependency_not_ready",
            missing_fields=["record_dependency"],
        )
    return None


def _state_is_urgent(state: ReactState) -> bool:
    level = state.get("safety_level")
    if level is None:
        source_text = state.get("source_text")
        if source_text:
            from app.services.security.input_guard import inspect_input

            level = inspect_input(source_text).level
    return level == "urgent"


def _all_covered(matched_ids: list[str], state: ReactState) -> bool:
    covered = set(state.get("covered_goals") or [])
    return bool(matched_ids) and all(cid in covered for cid in matched_ids)


def _reused_observation(matched_ids: list[str], state: ReactState) -> DomainObservation:
    """已处理的内存目标再次 call：复用可信观察，不再派发执行器。"""
    goal_results = state.get("goal_results") or {}
    for cid in matched_ids:
        previous = goal_results.get(cid)
        if isinstance(previous, dict):
            return DomainObservation.from_dict(previous)
    return DomainObservation(status=ToolStatus.COMPLETED, error="goal_already_handled")


def _ref_effectively_binds(call, spec, state: ReactState) -> bool:
    """写调用的 source_ref 是否“有效绑定”至少一个候选（充分覆盖，而非单字符等片段）。

    充分对应由 ``_matching_candidates`` 保证（重叠超过候选一半），故这里只再要求
    ref 长度至少 2 且确实匹配到候选。
    """
    args = call.args if isinstance(call.args, dict) else {}
    source_ref = args.get("source_ref")
    if not isinstance(source_ref, dict):
        return False
    start = source_ref.get("start")
    end = source_ref.get("end")
    if not is_int(start) or not is_int(end):
        return False
    if end - start < 2:
        return False
    return bool(_matching_candidates(call, spec, state))


def _goal_for_ids(state: ReactState, matched_ids: list[str]) -> dict[str, Any] | None:
    wanted = {str(cid) for cid in matched_ids if cid}
    if not wanted:
        return None
    for candidate in state.get("intent_candidates") or []:
        if isinstance(candidate, dict) and str(candidate.get("id")) in wanted:
            return candidate
    return None


def _bounded_task_observation(observation, task):
    """A swallowed model-budget error still cannot become a completed answer.

    Real committed/unknown writes keep their receipt proof; read/advice output
    produced after the task's last allowed call is not reported complete.
    """
    if task is None or not getattr(task, "limited", False):
        return observation
    if observation.commit_state in ("committed", "unknown") or observation.receipt_refs:
        return observation
    return DomainObservation(
        status=ToolStatus.UNAVAILABLE,
        error=getattr(task, "exhausted_code", None) or "model_task_budget_exhausted",
        summary="该任务已达到模型调用上限，未完成；其他独立任务继续执行。",
    )


async def _run_call_with_task(call, state: ReactState, dependencies: ReactDependencies,
                              matched_ids: list[str]):
    """Run one dispatch inside its matched semantic goal's durable budget."""
    from app.services.agent.task_budget import task_model_budget

    candidate = _goal_for_ids(state, matched_ids)
    if candidate is None:
        observation, facts = await _run_call(call, state, dependencies)
        return observation, facts, None
    with task_model_budget(candidate, state.get("source_text") or "") as task:
        observation, facts = await _run_call(call, state, dependencies)
    return _bounded_task_observation(observation, task), facts, task


async def _run_call(call, state: ReactState, dependencies: ReactDependencies) -> tuple[DomainObservation, dict[str, Any]]:
    """普通批调用与 pending 恢复共用的内部执行过程。

    返回 ``(规范观察, 本地事实)``。本地事实含 ``dispatched``（是否真正进入 handler，
    含进入后异常/非法返回）与 ``gate_reason``（未派发的应用分支原因，None 表示非门禁）。
    两者均由实际执行分支确定，不据 permission/匹配/返回 status/领域 error 估算。
    """
    spec = _active_catalog(dependencies).get(call.name)
    facts: dict[str, Any] = {"dispatched": False, "gate_reason": None}
    if spec is None:
        facts["gate_reason"] = "unknown_tool"
        return DomainObservation(status=ToolStatus.UNAVAILABLE, error="unknown_tool"), facts
    # 急症门禁：所有工具（写/读/建议/问答）在派发之前确定性安全终止；直接节点调用也同等生效。
    if _state_is_urgent(state):
        facts["gate_reason"] = "urgent_safety_gate"
        return DomainObservation(status=ToolStatus.REJECTED, error="urgent_safety_gate"), facts
    source_scope = state.get("source_scope")
    if source_scope is None and state.get("source_text"):
        source_scope = SourceScope(text=state["source_text"], intent_candidates=state.get("intent_candidates") or [])
    context = _execution_context(state, dependencies, call)

    if spec.permission == "write":
        # 先验证 source_ref 原始类型/边界/exact_text 与授权，再匹配应用目标，最后考虑复用。
        source_ref = call.args.get("source_ref") if isinstance(call.args, dict) else None
        # 写运行时（persistence 注入）：结构/exact_text 在本门校验；写授权窗口由真实
        # 闭桥在 C1 任务 span（含命令的完整句子）上按冻结策略执行，避免把“记录”命令
        # 与冒号后目标切开导致合法记录被误判。
        auth_task_type = None if dependencies.persistence is not None else spec.task_type
        validation = validate_source_ref(source_scope, source_ref, auth_task_type)
        if not validation.ok:
            facts["gate_reason"] = "invalid_source_ref"
            return DomainObservation(status=ToolStatus.REJECTED, error=validation.reason), facts
        if validation.authorization == "rejected_unauthorized":
            facts["gate_reason"] = "rejected_unauthorized"
            return DomainObservation(status=ToolStatus.REJECTED, error=validation.reason), facts
        if validation.authorization == "needs_confirmation":
            facts["gate_reason"] = "write_authorization"
            return DomainObservation(
                status=ToolStatus.NEEDS_CLARIFICATION,
                missing_fields=["write_authorization"],
                error=validation.reason,
            ), facts
        # 写工具调用不能借结构校验或持久化 bridge 绕过候选拒绝/需确认。持久化路径
        # 的闭桥只证明 span，不重新判断自然语言意图，因此候选门必须两条路径都生效。
        blocker = _candidate_write_blocker(call, state)
        if blocker is not None:
            facts["gate_reason"] = blocker.error or "write_authorization"
            return blocker, facts
        # authorized：匹配应用目标；无有效匹配目标则零派发，不创建新写绑定。
        matched_candidates = _matching_candidates(call, spec, state)
        matched_ids = [str(item.get("id")) for item in matched_candidates if item.get("id")]
        if not matched_ids:
            # 授权验证通过仍必须有明确、有效且充分对应的应用目标；匹配为空/非法/歧义时
            # 在绑定/adapter/执行器之前确定性澄清，派发 0、不追加 bound_tasks、不覆盖/提交。
            facts["gate_reason"] = "no_matching_goal"
            return DomainObservation(
                status=ToolStatus.NEEDS_CLARIFICATION,
                missing_fields=["goal_binding"],
                error="no_matching_goal",
            ), facts
        if _all_covered(matched_ids, state):
            # 复用已有目标结果：零派发、无门禁原因、不生成新保存身份。
            return _reused_observation(matched_ids, state), facts
        # 本次真实匹配的授权候选区间（含命令/元数据前缀）显式传给应用 adapter：
        # 餐次/日期等应用元数据只能来自该候选区间，不借整句其它候选的事实。
        candidate_span = matched_candidates[0].get("_source_span")
        if isinstance(candidate_span, tuple) and len(candidate_span) == 2:
            context["candidate_span"] = {"start": candidate_span[0], "end": candidate_span[1],
                                         "goal_id": str(matched_candidates[0].get("id") or "")}
        payload = matched_candidates[0].get("payload")
        payload = payload if isinstance(payload, dict) else {}
        semantic = payload.get("semantic")
        semantic = semantic if isinstance(semantic, dict) else {}
        correction = semantic.get("workout_correction") or matched_candidates[0].get("workout_correction")
        if (call.name == "correct_workout") != bool(correction) and spec.task_type == "workout_record":
            return DomainObservation(status=ToolStatus.REJECTED,
                                     error="workout_operation_mismatch"), facts
        if correction:
            context["workout_correction"] = dict(correction)
        units = (payload.get("workout_sources") or semantic.get("workout_sources")
                 or matched_candidates[0].get("workout_sources"))
        if isinstance(units, list) and units:
            context["workout_sources"] = [dict(item) for item in units if isinstance(item, dict)]
            source_text = str(state.get("source_text") or "")
            source_span = candidate_span if isinstance(candidate_span, tuple) else (start, end)
            parent_id = str(matched_candidates[0].get("parent_goal_id") or "")
            if parent_id:
                parent_span = _candidate_full_positions(
                    state.get("intent_candidates") or [], source_text,
                    ws_fallback=True).get(parent_id)
                if parent_span is not None:
                    source_span = parent_span
            context["workout_source_span"] = source_span
        task_spec = bind_domain_task(
            call, source_scope, request_id=dependencies.request_id,
            authorization=validation.authorization,
        )
        if dependencies.tool_executor is None:
            facts["gate_reason"] = "write_adapter_blocked"
            return DomainObservation(
                status=ToolStatus.BLOCKED_CAPABILITY,
                task_id=task_spec.id,
                result={"task_spec": task_spec.to_dict()},
                error="production_write_adapter_blocked",
                summary="真实写 adapter 在参数冻结/操作预留/保存派发前显式阻断。",
            ), facts
        observation = await _invoke_executor(dependencies.tool_executor, call, context)
        if observation is None:
            facts["gate_reason"] = "write_executor_missing"
            return DomainObservation(
                status=ToolStatus.BLOCKED_CAPABILITY,
                task_id=task_spec.id,
                result={"task_spec": task_spec.to_dict()},
                error="write_executor_missing",
            ), facts
        # 真正进入 handler（含进入后异常/非法返回）：dispatched=true，观察受限不改变触达事实。
        facts["dispatched"] = True
        # 应用绑定为最终身份：task_id、task_spec 由应用生成，执行器返回的同名字段不能覆写。
        observation.task_id = task_spec.id
        merged_result = {**(observation.result or {}), "task_spec": task_spec.to_dict()}
        observation.result = merged_result
        return observation, facts

    # 只读 / 建议 / 问答工具
    blocker = _dependency_blocker(call, spec, state)
    if blocker is not None:
        facts["gate_reason"] = "record_dependency_not_ready"
        return blocker, facts
    _matched: list[dict[str, Any]] = []
    if spec.task_type:
        # 面向目标的读取：带 ref 时先核真实类型/边界/exact_text 和目标身份，
        # 错误/空绑定/跨目标歧义在 worker 前拒绝或澄清，零派发。无 ref 的既有
        # 正常读取保持原合同；四个辅助读 task_type 为空，不受此门限制。
        _matched, read_issue = _read_goal_matches(call, spec, state)
        if read_issue is not None:
            source_ref = call.args.get("source_ref") if isinstance(call.args, dict) else None
            if read_issue == "invalid_source_ref":
                validation = validate_source_ref(source_scope, source_ref, None)
                facts["gate_reason"] = "invalid_read_source_ref"
                error = validation.reason if not validation.ok else "empty_read_source_ref"
                return DomainObservation(status=ToolStatus.REJECTED, error=error), facts
            facts["gate_reason"] = "read_goal_binding"
            return DomainObservation(
                status=ToolStatus.NEEDS_CLARIFICATION,
                missing_fields=["goal_binding"],
                error=read_issue,
            ), facts
    if _matched:
        # The read worker's per-target scope comes from the exact matched
        # candidate, never from the whole parent request. Plan information can
        # therefore distinguish this target's own equipment/time from shared
        # conditions without seeing a sibling target as this target.
        candidate = _matched[0]
        payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
        contract = payload.get("plan_contract") if isinstance(payload.get("plan_contract"), dict) else {
            key: candidate.get(key) for key in (
                "plan_result", "result_scope", "plan_goal_id", "plan_request_quote",
                "shared_scope_quote", "plan_target_quote") if candidate.get(key) is not None}
        current_id = str(candidate.get("id") or "")
        siblings = []
        for sibling in state.get("intent_candidates") or []:
            if not isinstance(sibling, dict) or str(sibling.get("id") or "") == current_id:
                continue
            if sibling.get("task_type") != "training_plan":
                continue
            sibling_payload = sibling.get("payload") if isinstance(sibling.get("payload"), dict) else {}
            sibling_contract = (sibling_payload.get("plan_contract")
                                if isinstance(sibling_payload.get("plan_contract"), dict) else {})
            siblings.append({
                "task_id": str(sibling.get("id") or ""),
                "text": str(sibling.get("text") or ""),
                "plan_target_quote": str(sibling_contract.get("plan_target_quote")
                                         or sibling.get("plan_target_quote") or ""),
            })
        context["task_id"] = current_id
        context["task_text"] = str(candidate.get("text") or "")
        context["plan_contract"] = dict(contract)
        context["plan_siblings"] = siblings
        context["semantic_scope"] = (payload.get("semantic")
                                     if isinstance(payload.get("semantic"), dict) else {})
        context["dialogue_scope"] = (payload.get("dialogue")
                                     if isinstance(payload.get("dialogue"), dict) else {})
    matched_ids = _matched_ids(call, spec, state)
    if _all_covered(matched_ids, state):
        return _reused_observation(matched_ids, state), facts
    if dependencies.tool_executor is None:
        facts["gate_reason"] = "read_adapter_not_wired"
        return DomainObservation(status=ToolStatus.UNAVAILABLE, error="read_adapter_not_wired"), facts
    observation = await _invoke_executor(dependencies.tool_executor, call, context)
    if observation is None:
        facts["gate_reason"] = "read_executor_missing"
        return DomainObservation(status=ToolStatus.UNAVAILABLE, error="read_executor_missing"), facts
    facts["dispatched"] = True
    return observation, facts


async def _execute_call(call, state: ReactState, dependencies: ReactDependencies) -> DomainObservation:
    """兼容单调用入口：返回规范观察（普通/恢复共享 ``_run_call`` 内部过程）。"""
    observation, _facts = await _run_call(call, state, dependencies)
    return observation


def _deps_satisfied(goal_ids: list[str], state: ReactState, committed: list[str]) -> bool:
    candidates = {c.get("id"): c for c in state.get("intent_candidates") or []}
    committed_set = set(committed)
    for gid in goal_ids:
        candidate = candidates.get(gid)
        if candidate is None:
            continue
        deps = candidate.get("depends_on") or []
        if any(d not in committed_set for d in deps):
            return False
    return True


def _pending_has_goal(pending: list[dict[str, Any]], goal_ids: list[str]) -> bool:
    """pending 是否已存在同一应用目标（按 goal_ids 集合去重）。"""
    if not goal_ids:
        return False
    target = set(goal_ids)
    return any(set(item.get("goal_ids") or []) == target for item in pending)


async def _auto_retry_pending(
    pending: list[dict[str, Any]],
    state: ReactState,
    dependencies: ReactDependencies,
    observations: list[dict[str, Any]],
    covered: list[str],
    committed: list[str],
    goal_results: dict[str, dict[str, Any]],
    tool_steps: int,
    tool_steps_by_task: dict[str, int],
    limited_tasks: list[str],
    recovery_events: list[dict[str, Any]],
    recovery_counter: int,
    call_event_counter: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str], dict[str, dict[str, Any]], int, dict[str, int], list[str], int, int]:
    """统一 pending（同批/跨批）中依赖已提交的目标自动恢复执行。

    恢复与普通调用共用 tool_steps：派发前检查余量并占一步，剩余额度 0 时零派发。
    恢复成功（确定性状态）覆盖目标并清除 pending；受限结果（unavailable/commit_unknown
    等）保留失败观察并反馈主管，但不覆盖、不解除依赖、不转为 complete，同时结束本次
    尝试避免无界重试。事件含原 tool_call_id 与跨批稳定的 recovery_index；恢复与普通
    调用共享 ``_run_call`` 内部过程并占用请求内连续事件序号（``call_event_counter``）。
    """
    if not pending:
        return (pending, observations, covered, committed, goal_results, tool_steps,
                tool_steps_by_task, limited_tasks, recovery_counter, call_event_counter)
    remaining: list[dict[str, Any]] = []
    for item in pending:
        goal_ids = item.get("goal_ids") or []
        if not _deps_satisfied(goal_ids, state, committed):
            remaining.append(item)
            continue
        task_key = str(goal_ids[0]) if goal_ids else "__aux__"
        task_step_limit = (dependencies.max_tool_steps_per_task
                           if len(state.get("intent_candidates") or []) > 1
                           else dependencies.max_tool_steps)
        if tool_steps_by_task.get(task_key, 0) >= task_step_limit:
            if task_key not in limited_tasks:
                limited_tasks.append(task_key)
            remaining.append(item)  # 单任务预算耗尽：恢复零派发
            continue
        if tool_steps >= dependencies.max_tool_steps:
            remaining.append(item)  # 全局预算耗尽：恢复零派发
            continue
        call = ToolCall(id=item.get("tool_call_id", "retry"), name=item["name"], args=item["args"])
        try:
            await _persist_hook(dependencies, "tool_intent", call, state)
            matched = _matched_ids(call, _active_catalog(dependencies)[item["name"]], state)
            observation, _exec_facts, task_budget = await _run_call_with_task(
                call, state, dependencies, matched)
            if task_budget is not None and getattr(task_budget, "limited", False) \
                    and task_key not in limited_tasks:
                limited_tasks.append(task_key)
            await _persist_hook(dependencies, "tool_result", call, observation, state)
        except PersistenceUnavailable as exc:
            return (remaining, observations, covered, committed, goal_results, tool_steps,
                    tool_steps_by_task, limited_tasks, recovery_counter, call_event_counter)
        tool_steps += 1
        tool_steps_by_task[task_key] = tool_steps_by_task.get(task_key, 0) + 1
        call_event_counter += 1
        spec = _active_catalog(dependencies)[item["name"]]
        matched = _matched_ids(call, spec, state)
        if observation.status == ToolStatus.BLOCKED_DEPENDENCY:
            remaining.append(item)
            continue
        # 恢复已实际执行（非 blocked）：从 pending 移除，记录事件与观察
        recovery_counter += 1
        recovery_events.append({
            "tool_call_id": item.get("tool_call_id", ""),
            "goal_ids": goal_ids,
            "recovery_index": recovery_counter,
            "observation": observation.to_dict(),
        })
        observations.append(observation.to_dict())
        # 仅确定性状态覆盖目标；受限结果不覆盖、不解除依赖，目标保持受限
        if observation.status in _DEFINITE_STATUSES:
            _record_observation(matched, observation, spec, covered, committed, goal_results)
    return remaining, observations, covered, committed, goal_results, tool_steps, tool_steps_by_task, limited_tasks, recovery_counter, call_event_counter


def _limited_tool_message(call_id: str, reason: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(
            {"status": ToolStatus.UNAVAILABLE.value, "error": reason},
            ensure_ascii=False,
        ),
    }


async def execute_react_batch(state: ReactState, dependencies: ReactDependencies) -> dict[str, Any]:
    """整批先校验，合法调用按原顺序执行并追加带原 tool_call_id 的 tool 消息。

    协议损坏整批零派发并确定性终止；合法协议中的单项业务错误生成匹配 tool 消息，
    其他独立调用继续。批内每项使用即时更新的观察/提交/覆盖状态；写提交后自动恢复
    历史 blocked 依赖目标；恢复与普通调用共用 tool_steps；预算耗尽为剩余未执行调用
    补齐受限结果，末轮无悬空协议。
    """
    messages = list(state.get("messages") or [])
    last = messages[-1] if messages else {}
    batch_source = last
    if not (isinstance(last, dict) and last.get("role") == "assistant") \
            and dependencies.persistence is not None:
        # 写运行时恢复：历史保留原完整 batch + 已持久化 tool 结果，独立 cursor 决定
        # 剩余派发；批处理来源是最近一条带 tool_calls 的 assistant（原 call ID/参数）。
        batch_source = _last_pending_assistant(messages) or last
    validation = validate_tool_batch(batch_source, _active_catalog(dependencies))
    if not validation.ok:
        return {
            "protocol_error": validation.error or "protocol_error",
            "status": "restricted",
            "terminated": True,
            "termination_reason": "protocol_error",
        }
    assistant_index = -1
    for index, message in enumerate(messages):
        if message is batch_source:
            assistant_index = index
            break
    existing_observations: dict[str, dict] = {}
    if assistant_index >= 0:
        for message in messages[assistant_index + 1:]:
            if not (isinstance(message, dict) and message.get("role") == "tool"):
                continue
            call_id = str(message.get("tool_call_id") or "")
            if not call_id or call_id in existing_observations:
                continue
            try:
                parsed = json.loads(str(message.get("content") or ""))
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                existing_observations[call_id] = parsed
    attempted_unknown = (set(getattr(dependencies.persistence, "restored_attempted", set()) or set())
                         if dependencies.persistence is not None else set())

    observations = list(state.get("observations") or [])
    bound_tasks = list(state.get("bound_tasks") or [])
    unbound_write_calls = list(state.get("unbound_write_calls") or [])
    # call_facts 只记录本批普通调用事件（非跨批累积）：事件身份跨批连续由 call_event_counter 保证。
    call_facts: list[dict[str, Any]] = []
    call_event_counter = int(state.get("call_event_counter") or 0)
    covered = list(state.get("covered_goals") or [])
    committed = list(state.get("committed_tasks") or [])
    goal_results = dict(state.get("goal_results") or {})
    pending = list(state.get("pending_goals") or [])
    recovery_counter = int(state.get("recovery_counter") or 0)
    tool_steps = int(state.get("tool_steps") or 0)
    tool_steps_by_task = dict(state.get("tool_steps_by_task") or {})
    limited_tasks = list(state.get("limited_tasks") or [])
    no_progress = int(state.get("consecutive_no_progress") or 0)
    terminated = bool(state.get("terminated"))
    termination_reason = state.get("termination_reason") or ""
    status = state.get("status") or "running"
    recovery_events: list[dict[str, Any]] = []

    def live_state() -> dict[str, Any]:
        snapshot = dict(state)
        snapshot["observations"] = observations
        snapshot["committed_tasks"] = committed
        snapshot["covered_goals"] = covered
        snapshot["goal_results"] = goal_results
        snapshot["tool_steps_by_task"] = tool_steps_by_task
        snapshot["limited_tasks"] = limited_tasks
        # 真实累计无进展随每次 hook 持久（恢复读到的即实际值，不假 count）。
        snapshot["consecutive_no_progress"] = no_progress
        return snapshot

    def _limit_messages(calls_slice, reason: str) -> None:
        # 已有真实结果的协议重建不追加 limited/重复 tool 消息。
        for item in calls_slice:
            if item.id in existing_observations:
                continue
            messages.append(_limited_tool_message(item.id, reason))

    for index, call in enumerate(validation.calls):
        if _policy_stop_requested(dependencies):
            _limit_messages(validation.calls[index:], "policy_stop")
            terminated = True
            termination_reason = "policy_stop"
            status = "limited"
            break

        spec = _active_catalog(dependencies)[call.name]
        matched_for_step = _matched_ids(call, spec, live_state())
        task_key = matched_for_step[0] if matched_for_step else "__aux__"
        task_step_limit = (dependencies.max_tool_steps_per_task
                           if len(state.get("intent_candidates") or []) > 1
                           else dependencies.max_tool_steps)
        # Auxiliary calls without a matched application goal consume only the
        # request-wide tool budget; the per-task cap must not mask the global
        # tool_step_limit termination for them.
        if matched_for_step and tool_steps_by_task.get(task_key, 0) >= task_step_limit:
            _limit_messages([call], "task_step_limit")
            if task_key not in limited_tasks:
                limited_tasks.append(task_key)
            continue

        if tool_steps >= dependencies.max_tool_steps:
            _limit_messages(validation.calls[index:], "tool_step_limit")
            terminated = True
            termination_reason = "tool_step_limit"
            status = "limited"
            break

        # 尝试在危险派发之前计数：即使派发或持久化被中断，本次尝试也计入本轮
        # 逻辑步数（应用侧 tool_intent 同样先于派发持久化真实计数）。
        persisted = existing_observations.get(call.id)
        if persisted is not None:
            # 协议重建已有真实结果：不占额度/步数、不重新持久、不追加重复消息，也绝不
            # 用历史草稿覆盖当前应用目标真相（覆盖仅补缺，不覆写已核验的 committed）。
            observation = DomainObservation.from_dict(persisted)
            exec_facts = {"dispatched": False, "gate_reason": "restored_result_reused"}
        else:
            # 新 attempt：计数/停止门照旧（4/16/2 不放宽）。
            tool_steps += 1
            tool_steps_by_task[task_key] = tool_steps_by_task.get(task_key, 0) + 1
            if spec.permission != "write" and call.id in attempted_unknown:
                # 已尝试但结果未知的付费只读/建议：绝不重复调用，给出保守真实结果，
                # 由下方唯一 tool_result hook 持久关闭该 cursor（不重发 intent）。
                observation = DomainObservation(status=ToolStatus.UNAVAILABLE,
                                                error="read_result_unconfirmed_not_retried")
                exec_facts = {"dispatched": False, "gate_reason": "attempted_result_unknown"}
            else:
                try:
                    await _persist_hook(dependencies, "tool_intent", call, live_state())
                except PersistenceUnavailable as exc:
                    return {
                        "protocol_error": str(exc),
                        "status": "limited",
                        "terminated": True,
                        "termination_reason": "write_persistence_failed",
                    }
                # 持久 intent 之后、危险派发（adapter/原桥 claim）之前重查真实停止信号：
                # 窗口内出现的真实取消/预算停止/期限到期必须阻断本次与剩余调用（零派发）。
                if _policy_stop_requested(dependencies):
                    _limit_messages(validation.calls[index:], "policy_stop")
                    terminated = True
                    termination_reason = "policy_stop"
                    status = "limited"
                    break
                try:
                    observation, exec_facts, task_budget = await _run_call_with_task(
                        call, live_state(), dependencies, matched_for_step)
                except PersistenceUnavailable as exc:
                    return {
                        "protocol_error": str(exc),
                        "status": "limited",
                        "terminated": True,
                        "termination_reason": "write_persistence_failed",
                    }
                if task_budget is not None and getattr(task_budget, "limited", False) \
                        and task_key not in limited_tasks:
                    limited_tasks.append(task_key)
        should_cover = False
        if persisted is None:
            # 先按真实观察推进无进展计数（与实际结果同一状态），再持久唯一 tool_result：
            # 普通中断/恢复读到的即真实累计值；原阈值/计数逻辑不变。
            if observation.status == ToolStatus.BLOCKED_DEPENDENCY:
                no_progress += 1
            elif observation.status in _DEFINITE_STATUSES:
                should_cover = True
                if observation.status == ToolStatus.NEEDS_CLARIFICATION and spec.permission == "write":
                    should_cover = _ref_effectively_binds(call, spec, live_state())
                if spec.permission != "write" and exec_facts.get("gate_reason"):
                    # 读取门禁（坏 ref/无绑定/歧义/依赖等待）没有真实覆盖，不得清零无进展。
                    should_cover = False
                if exec_facts.get("gate_reason") == "invalid_read_source_ref":
                    # A copyable-ref error gets the next bounded supervisor turn
                    # (and the app-provided correct refs); the tool step is still
                    # counted, and all turn/step/task budgets remain unchanged.
                    pass
                else:
                    no_progress = 0 if should_cover else no_progress + 1
            else:
                no_progress += 1
            try:
                await _persist_hook(dependencies, "tool_result", call, observation, live_state())
            except PersistenceUnavailable as exc:
                return {
                    "protocol_error": str(exc),
                    "status": "limited",
                    "terminated": True,
                    "termination_reason": "write_persistence_failed",
                }
        call_event_counter += 1
        event_id = f"{dependencies.request_id}:ev{call_event_counter}:{call.id}"
        matched = _matched_ids(call, spec, live_state())

        if persisted is not None:
            # 协议覆盖仅补缺：历史 pending/needs_clarification 不得回退最新真实
            # committed；goal_results 已有最新证明时不覆盖。
            cover_ok = observation.status in _DEFINITE_STATUSES
            if cover_ok and observation.status == ToolStatus.NEEDS_CLARIFICATION \
                    and spec.permission == "write":
                cover_ok = _ref_effectively_binds(call, spec, live_state())
            if cover_ok:
                for cid in matched:
                    if cid not in covered:
                        covered.append(cid)
                    if cid not in goal_results:
                        goal_results[cid] = observation.to_dict()
        elif observation.status == ToolStatus.BLOCKED_DEPENDENCY:
            # 同批立即进入统一 pending（保存原 tool_call_id），按应用目标去重
            if not _pending_has_goal(pending, matched):
                pending.append({
                    "name": call.name,
                    "args": call.args,
                    "goal_ids": matched,
                    "tool_call_id": call.id,
                })
        elif observation.status in _DEFINITE_STATUSES:
            if should_cover:
                _record_observation(matched, observation, spec, covered, committed, goal_results)

        obs_dict = observation.to_dict()
        if persisted is None:
            # 历史中已有该调用结果的（恢复复用）不再追加第二条 tool 消息，保持协议配对。
            tool_message = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(obs_dict, ensure_ascii=False, allow_nan=False),
            }
            messages.append(tool_message)
        canonical_observation = {
            **obs_dict,
            **({"_source": {
                "tool_name": call.name,
                "permission": spec.permission,
                "task_type": getattr(spec, "task_type", None),
            }} if spec.permission == "read" else {}),
        }
        observations.append(canonical_observation)
        # bound_tasks 只从应用本地生成的绑定来源（写工具合并的应用 task_spec）追加，
        # 不扫描领域 result 里的 task_spec 认领（只读伪造 task_spec 不得进入）。
        if spec.permission == "write" and observation.result and isinstance(observation.result.get("task_spec"), dict):
            bound_tasks.append(observation.result["task_spec"])
        # 应用调用事实：unbound_write_calls 只由真实 no_matching_goal 门禁分支记录，
        # 不据领域 error 字符串或"写 + matched 空"估算（坏 ref 拒绝/复用不得记无绑定身份）。
        if exec_facts["gate_reason"] == "no_matching_goal":
            unbound_write_calls.append(call.id)
        # 请求内应用调用事实关联（阶段A内存结构，非持久账本）：工具目录 permission、匹配
        # 目标、应用绑定、派发/门禁均由本地执行路径产生，不据领域 task_id/result.task_spec/error
        # 推断；observation 为本次规范观察的深复制快照，与 observations 同源可追溯。
        call_facts.append({
            "event_id": event_id,
            "tool_call_id": call.id,
            "tool_name": call.name,
            "permission": spec.permission,
            "goal_ids": list(matched),
            "bound_task_id": observation.task_id if spec.permission == "write" else None,
            "gate_reason": exec_facts["gate_reason"],
            "dispatched": exec_facts["dispatched"],
            "observation": copy.deepcopy(canonical_observation),
        })

        # 停止信号在本次调用期间出现（如回答/RAG 触发策略停止）：未派发的剩余调用
        # 一律返回有限受限结果并终止本批，不再进入任何模型或领域 handler。
        if _policy_stop_requested(dependencies):
            _limit_messages(validation.calls[index + 1:], "policy_stop")
            terminated = True
            termination_reason = "policy_stop"
            status = "limited"
            break

        # 可信写提交后，恢复可执行的统一 pending（同批/跨批）依赖目标。
        if _is_write_commit(observation, spec):
            pending, observations, covered, committed, goal_results, tool_steps, tool_steps_by_task, limited_tasks, recovery_counter, call_event_counter = await _auto_retry_pending(
                pending, live_state(), dependencies, observations, covered, committed, goal_results,
                tool_steps, tool_steps_by_task, limited_tasks, recovery_events, recovery_counter, call_event_counter,
            )

        if no_progress >= dependencies.max_no_progress:
            _limit_messages(validation.calls[index + 1:], "no_progress")
            terminated = True
            termination_reason = "no_progress"
            status = "limited"
            break

    # 批次全部 tool 结果配对完毕后，追加应用恢复事件摘要（仅为数据，不代表用户授权）。
    if recovery_events:
        messages.append({
            "role": "user",
            "content": RECOVERY_OBSERVATION_NOTICE
                       + json.dumps(recovery_events, ensure_ascii=False, allow_nan=False),
        })

    # A structured saved-workout goal must not depend on the model choosing the
    # one goal-bearing tool.  After the batch, render any still-unhandled
    # workout-record read from its own date/action scope.  This is an
    # application-owned read, not an auxiliary probe becoming proof.
    if not terminated and dependencies.persistence is None:
        from app.services.agent.react_readonly_workers import (
            candidate_record_query, saved_workout_observation,
        )

        fallback_context = {
            "user_id": str(state.get("user_id") or dependencies.user_id),
        }
        covered_now = set(covered)
        for candidate in state.get("intent_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            cid = str(candidate.get("id") or "")
            if not cid or cid in covered_now:
                continue
            if candidate.get("task_type") != TaskType.WORKOUT_RECORD.value:
                continue
            if not is_record_read(candidate):
                continue
            query = candidate_record_query(candidate)
            if not isinstance(query, dict) or not query.get("date_start") or not query.get("date_end"):
                # No structured date scope: the ordinary goal-bearing worker
                # owns the clarification, not this fallback.
                continue
            try:
                observation = saved_workout_observation(candidate, fallback_context)
            except Exception:  # noqa: BLE001 - fallback failure stays a failed read
                logger.warning("react_saved_workout_fallback_failed", exc_info=True)
                continue
            if observation.status not in _DEFINITE_STATUSES:
                continue
            covered.append(cid)
            covered_now.add(cid)
            goal_results[cid] = observation.to_dict()
            observations.append({
                **observation.to_dict(),
                "_source": {"tool_name": "read_workout_history",
                            "permission": "read", "task_type": "workout_record"},
            })
            no_progress = 0

    if limited_tasks and not terminated:
        snapshot = {**state, "covered_goals": covered, "goal_results": goal_results,
                    "committed_tasks": committed}
        outstanding = _unhandled_goals(snapshot)
        if outstanding and all(str(cid) in limited_tasks for cid in outstanding):
            terminated = True
            termination_reason = "task_step_limit"
            status = "limited"

    return {
        "messages": messages,
        "observations": observations,
        "intent_candidates": list(state.get("intent_candidates") or []),
        "bound_tasks": bound_tasks,
        "unbound_write_calls": unbound_write_calls,
        "call_facts": call_facts,
        "call_event_counter": call_event_counter,
        "covered_goals": covered,
        "committed_tasks": committed,
        "goal_results": goal_results,
        "pending_goals": pending,
        "recovery_counter": recovery_counter,
        "tool_steps": tool_steps,
        "tool_steps_by_task": tool_steps_by_task,
        "limited_tasks": limited_tasks,
        "consecutive_no_progress": no_progress,
        "terminated": terminated,
        "termination_reason": termination_reason,
        "status": status,
    }


# ---------------------------------------------------------------------------
# completion_check / finalize
# ---------------------------------------------------------------------------
def _goal_key(candidate: dict[str, Any]) -> str | None:
    cid = candidate.get("id")
    return str(cid) if cid else None


def _unhandled_goals(state: ReactState) -> list[dict[str, Any]]:
    candidates = state.get("intent_candidates") or []
    covered = set(state.get("covered_goals") or [])
    outstanding: list[dict[str, Any]] = []
    for candidate in candidates:
        cid = _goal_key(candidate)
        if cid and cid in covered:
            continue
        task_type = candidate.get("task_type")
        # A model-declared condition/background is not an independently
        # deliverable goal. It still remains on the task for negation/write
        # authorization; only completion tracking excludes it. A real write
        # (side_effect) is never excluded this way.
        if (candidate.get("side_effect") is not True
                and candidate.get("goal_role") in {"constraint", "background"}):
            continue
        if is_record_write(candidate) and candidate.get("write_authorization") in _TERMINAL_WRITE_AUTH:
            continue  # 已被安全门拒绝或需用户确认，本轮无需再处理
        if is_record_write(candidate):
            children = _write_child_candidates(candidates, str(candidate.get("id") or ""))
            if children:
                if _write_parent_handled(candidate, state):
                    continue
                # Unhandled children stay independently actionable. Only when
                # every action was already handled but some numeric fact was
                # never cited does the parent itself remain outstanding.
                if any(str(child.get("id") or "") not in covered for child in children):
                    continue
        outstanding.append(candidate)
    return outstanding


def _action_goals(goals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """写运行时统一动作目标过滤（写 side_effect + 建议/计划/问答类型）。

    completion_check 与 final_status 共用同一规则，避免“检查通过、终态受限”两套标准；
    只读 A/B 默认路径不调用（保持原行为）。训练问答是真实用户目标，不能因不是
    “建议/计划”而被过滤成虚假 completed；只有明确 chitchat 尾随寒暄不计入。"""
    from app.services.agent.task_models import TaskType as _TaskType
    action_types = _WRITE_TASK_TYPE_VALUES | ADVICE_TASK_TYPES | {
        _TaskType.TRAINING_QA.value, _TaskType.SAFETY_TRIAGE.value}
    return [goal for goal in goals
            if bool(goal.get("side_effect")) or goal.get("task_type") in action_types]


def completion_check(state: ReactState, dependencies: ReactDependencies | None = None) -> dict[str, Any]:
    """未处理目标、无进展、预算与终止判断；不把内存循环称为持久恢复。"""
    if state.get("terminated"):
        return {}
    if state.get("candidate_error"):
        return {"terminated": True, "termination_reason": "rule_candidates_unavailable", "status": "restricted"}
    unhandled = _unhandled_goals(state)
    if dependencies is not None and dependencies.persistence is not None:
        # 写运行时：批处理结束后只追未完成的写/建议目标；尾随闲聊目标（如“谢谢”）
        # 不再发起额外主管模型调用（一次付费 native 调用的固定行为）。
        unhandled = _action_goals(unhandled)
    if not unhandled:
        return {"terminated": True, "termination_reason": "goals_complete", "status": "complete"}
    turns = int(state.get("supervisor_turns") or 0)
    tool_steps = int(state.get("tool_steps") or 0)
    no_progress = int(state.get("consecutive_no_progress") or 0)
    max_turns = dependencies.max_supervisor_turns if dependencies else MAX_SUPERVISOR_TURNS
    max_steps = dependencies.max_tool_steps if dependencies else MAX_TOOL_STEPS
    max_np = dependencies.max_no_progress if dependencies else MAX_NO_PROGRESS
    if turns >= max_turns:
        return {"terminated": True, "termination_reason": "supervisor_turn_limit", "status": "limited"}
    if tool_steps >= max_steps:
        return {"terminated": True, "termination_reason": "tool_step_limit", "status": "limited"}
    if no_progress >= max_np:
        return {"terminated": True, "termination_reason": "no_progress", "status": "limited"}
    return {"status": "running"}


def _completion_check_node(state: ReactState, dependencies: ReactDependencies) -> dict[str, Any]:
    decision = completion_check(state, dependencies)
    if not decision.get("terminated"):
        goals = _unhandled_goals(state)
        if goals:
            rendered = "；".join(
                f"{g.get('task_type')}:{str(g.get('text') or '')[:40]}" for g in goals[:5]
            )
            messages = list(state.get("messages") or [])
            messages.append({"role": "user", "content": UNHANDLED_GOALS_HINT.format(goals=rendered)})
            decision["messages"] = messages
    return decision


def _route_completion(state: ReactState) -> str:
    return "finalize" if state.get("terminated") else "main_agent"


def _observation_user_answer(observation: Mapping[str, Any]) -> str:
    """单个已完成观察的面向用户答复（真实summary优先，其次规范result文本）。"""
    summary = observation.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary
    result = observation.get("result")
    if isinstance(result, Mapping):
        for key in ("answer", "summary", "text"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def completed_read_goals(
    candidates: Any,
    goal_results: Any,
    covered: Any,
) -> list[dict[str, Any]]:
    """按原请求顺序返回真实已完成、已覆盖的只读目标及其用户答复。

    身份只取应用候选id：执行顺序、goal_results插入顺序、工具call_id和相同
    summary文本都不能代替目标身份。辅助读没有task_type，不会出现在这里；
    未覆盖、非completed、带失败的观察一律不进入。
    """
    if not isinstance(goal_results, Mapping):
        return []
    covered_ids = {str(cid) for cid in covered or []}
    goals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates or []:
        if not isinstance(candidate, Mapping):
            continue
        cid = str(candidate.get("id") or "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        task_type = str(candidate.get("task_type") or "")
        if task_type not in _READ_GOAL_TASK_TYPES:
            continue
        if is_record_write(candidate):
            continue
        if (candidate.get("side_effect") is not True
                and candidate.get("goal_role") in {"constraint", "background"}):
            continue
        if cid not in covered_ids:
            continue
        observation = goal_results.get(cid)
        if not isinstance(observation, Mapping):
            continue
        if observation.get("status") != ToolStatus.COMPLETED.value:
            continue
        if observation.get("error"):
            continue
        goals.append({
            "id": cid,
            "task_type": task_type,
            "answer": _observation_user_answer(observation),
            # A source-bound worker answer is already rendered from current
            # application reads. It must not be replaced by a later supervisor
            # sentence that still repeats an earlier tool failure.
            "authoritative": bool(
                isinstance(observation.get("result"), Mapping)
                and observation["result"].get("authoritative") is True),
        })
    return goals


def _normalized_answer_text(value: str) -> str:
    """空白不敏感的确定性比较文本；不改变返回给用户的原文。"""
    return "".join(str(value or "").split())


def completed_delivery_goals(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Deliver read answers and application-confirmed write receipts in request order.

    A supervisor sentence or an unconfirmed observation cannot establish a save.
    Commit identities come from the existing write completion gate.
    """
    candidates = state.get("intent_candidates") or []
    results = state.get("goal_results") or {}
    reads = {goal["id"]: goal for goal in completed_read_goals(
        candidates, results, state.get("covered_goals") or [])}
    committed = set(state.get("committed_tasks") or [])
    covered = set(state.get("covered_goals") or [])
    delivered = []
    seen = set()
    positions = {candidate.get("id"): index for index, candidate in enumerate(candidates)}
    by_id = {candidate.get("id"): candidate for candidate in candidates}
    def delivery_order(candidate):
        # Narrow write refs create action children after initial planning. Keep
        # those receipts in their original parent's slot, before later goals.
        root = candidate
        visited = set()
        while root.get("parent_goal_id") in by_id and root.get("id") not in visited:
            visited.add(root.get("id"))
            root = by_id[root["parent_goal_id"]]
        index = positions[candidate.get("id")]
        start = candidate.get("source_start")
        return (positions[root.get("id")], start if is_int(start) else index, index)
    for candidate in sorted(candidates, key=delivery_order):
        cid = candidate.get("id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        if cid in reads:
            delivered.append(reads[cid])
            continue
        observation = results.get(cid) or {}
        if (is_record_write(candidate) and cid in committed and cid in covered
                and observation.get("status") == "completed"
                and observation.get("commit_state") == "committed"
                and observation.get("receipt_refs") and not observation.get("error")):
            # Summary is generated by the adapter from canonical saved fields.
            # Old persisted proofs may lack it; still acknowledge their receipt.
            answer = _observation_user_answer(observation) or "记录已保存，保存回执已保留。"
            delivered.append({"id": cid, "task_type": candidate.get("task_type"),
                              "answer": answer, "authoritative": True})
    return delivered


def _contains_answers_in_order(text: str, answers: list[str]) -> bool:
    """最终文字是否已按请求顺序逐条包含目标答案（重复答案需重复出现）。"""
    haystack = _normalized_answer_text(text)
    cursor = 0
    for answer in answers:
        needle = _normalized_answer_text(answer)
        if not needle:
            continue
        index = haystack.find(needle, cursor)
        if index < 0:
            return False
        cursor = index + len(needle)
    return True


def answer_from_completed_read_goals(
    goals: list[dict[str, Any]], final_text: str = ""
) -> str | None:
    """共享确定性交付规则：多个目标收齐，单个目标保持原答案。

    ``None`` 表示没有只读目标，调用者沿用旧的无目标行为。已有合法最终总结
    （按顺序包含全部目标答案）原样保留，避免重复拼接；否则按请求顺序用空行
    连接各目标答案，两个相同类型/相同文字的独立目标也各交付一次。
    """
    if not goals:
        return None
    answers = [str(goal.get("answer") or "") for goal in goals
               if str(goal.get("answer") or "").strip()]
    text = final_text if isinstance(final_text, str) else ""
    if any(goal.get("authoritative") is True for goal in goals):
        # A completed source-bound worker is the current delivery. A supervisor
        # tail can contain a stale failure sentence from before that observation;
        # joining the goal answers preserves request order without forwarding it.
        return "\n\n".join(answers) if answers else ""
    if text.strip():
        # 单目标保持旧行为：最终assistant文字优先，不被观察摘要替换。
        if len(goals) == 1:
            return text
        if not answers or _contains_answers_in_order(text, answers):
            return text
    if answers:
        return "\n\n".join(answers)
    return ""


def _final_assistant_text(messages: Any) -> str:
    """最后一条真实最终assistant文字；工具调用轮的文字不是最终回答。"""
    for message in reversed(messages or []):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return ""
        return content
    return ""


def _extract_final_answer(state: ReactState) -> str:
    goals = completed_delivery_goals(state)
    if goals:
        # 有真实完成目标时，目标答案优先：最后assistant若只是最后一答或工具
        # 调用说明，不得覆盖其它已完成请求答案；辅助观察也不得被兜底打印。
        answer = answer_from_completed_read_goals(
            goals, _final_assistant_text(state.get("messages") or []))
        if answer is not None:
            return answer
    for message in reversed(state.get("messages") or []):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    # 无主管最终文字（工具调用即终态，例如只读建议单次回答）：使用最后一条已完成
    # 回答观察的真实 summary/规范 result 答复，不伪造、不拼接模型文字。
    for observation in reversed(state.get("observations") or []):
        if not isinstance(observation, dict) or observation.get("status") != "completed":
            continue
        summary = observation.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary
        result = observation.get("result")
        if isinstance(result, dict):
            for key in ("answer", "summary", "text"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    return ""


def _has_confirmed_observation(state: ReactState) -> bool:
    return any(
        isinstance(o, dict) and o.get("status") in {"completed", "partial"}
        for o in state.get("observations") or []
    )


def _has_unconfirmed_write(state: ReactState) -> bool:
    """是否存在 completed 但无可信提交证明的写目标（提交尚未确认）。

    只查 goal_results 中已覆盖的写目标；工具种类由候选 task_type 判定，不据领域
    result.task_spec 形状或裸 task_id 等值扫描 observations（只读伪 ID 不得获得写身份）。
    """
    committed = set(state.get("committed_tasks") or [])
    candidates = {c.get("id"): c for c in state.get("intent_candidates") or []}
    goal_results = state.get("goal_results") or {}
    for cid, obs in goal_results.items():
        if not isinstance(obs, dict):
            continue
        candidate = candidates.get(cid) or {}
        if is_record_write(candidate) and obs.get("status") == "completed" and cid not in committed:
            return True
    return False


def _has_orphaned_write_observation(state: ReactState) -> bool:
    """completed 且带提交声明（commit_state unknown/committed 或回执）的观察，若在
    goal_results 写目标与 call_facts 写事实里都找不到对应，即应用关联丢失的异常已有
    写状态，不得默认 complete。只据 commit_state/回执识别写观察，不据 task_id/task_spec/error。
    """
    goal_results = state.get("goal_results") or {}
    candidates = {c.get("id"): c for c in state.get("intent_candidates") or []}
    write_result_values = [
        obs for cid, obs in goal_results.items()
        if isinstance(obs, dict) and is_record_write(candidates.get(cid) or {})
    ]
    write_fact_obs = [
        f.get("observation") for f in (state.get("call_facts") or [])
        if isinstance(f, dict) and f.get("permission") == "write"
    ]
    for obs in state.get("observations") or []:
        if not isinstance(obs, dict) or obs.get("status") != "completed":
            continue
        if obs.get("commit_state") not in ("unknown", "committed") and not obs.get("receipt_refs"):
            continue
        if obs in write_fact_obs or obs in write_result_values:
            continue
        return True
    return False


def _final_status(state: ReactState, dependencies: ReactDependencies | None = None) -> str:
    reason = state.get("termination_reason") or ""
    if reason == "urgent_safety_blocked":
        return "rejected"
    if reason in {"protocol_error", "model_unavailable", "rule_candidates_unavailable",
                  "safety_assessment_unavailable"}:
        return "restricted"
    if reason in {"supervisor_turn_limit", "tool_step_limit", "task_step_limit",
                  "task_count_limit", "no_progress", "policy_stop"}:
        return "limited"
    # 未处理目标仍存在（已确定性终止）→ 不 complete。写运行时与 completion_check
    # 共用同一动作目标规则：尾随闲聊目标不把真实完成的写/建议投影成 limited。
    outstanding = _unhandled_goals(state)
    if dependencies is not None and dependencies.persistence is not None:
        outstanding = _action_goals(outstanding)
    if outstanding:
        return "limited"
    # 无有效目标绑定的写调用：即使零派发也明确受限，不据空集合遍历默认完成。
    if state.get("unbound_write_calls"):
        return "needs_clarification"
    # 已有写观察但应用关联（goal_results 写目标 / call_facts 写事实）丢失，或 facts 缺失/
    # 空/仅无关读事实：不得默认 complete，明确受限。正常聊天/纯读/合法提交/复用不受影响。
    if _has_orphaned_write_observation(state):
        return "restricted"
    # 提交证明与终态共用同一判定：写观察/写目标 completed 但无可信提交证明 → 不 complete。
    if _has_unconfirmed_write(state):
        return "blocked"
    # 终态基于每个应用目标当前有效结果，不汇总历史 status 字符串。
    goal_results = state.get("goal_results") or {}
    candidates = {c.get("id"): c for c in state.get("intent_candidates") or []}
    has_partial = False
    has_rejected = False
    has_clarification = False
    has_blocked = False
    for cid, obs in goal_results.items():
        if not isinstance(obs, dict):
            continue
        status = obs.get("status")
        if is_record_write(candidates.get(cid) or {}):
            if status == "partial":
                has_partial = True
            elif status == "rejected":
                has_rejected = True
            elif status == "needs_clarification":
                has_clarification = True
            elif status in {"blocked_capability", "unavailable", "commit_unknown"}:
                has_blocked = True
        else:
            if status == "partial":
                has_partial = True
            elif status == "rejected":
                has_rejected = True
            elif status == "needs_clarification":
                has_clarification = True
            elif status in {"blocked_capability", "unavailable", "commit_unknown", "blocked_dependency"}:
                has_blocked = True
    if has_blocked:
        return "blocked"
    if has_clarification:
        return "needs_clarification"
    if has_rejected:
        return "rejected"
    if has_partial:
        return "partial"
    return "complete"


def _partial_notice(state: ReactState) -> str:
    return "部分目标已确认，其余未完全完成，请核对后重试。"


def _notice_for_state(state: ReactState, status: str, reason: str) -> str:
    """确定性终态文案，按可信观察与终止原因区分，不原样转发模型文字。"""
    if reason == "urgent_safety_blocked":
        return URGENT_SAFETY_ANSWER
    if reason == "task_count_limit":
        return "本次请求包含的任务过多，已停止处理。请拆分为多次请求；本次未保存任何数据。"
    if reason == "task_step_limit":
        from app.services.agent.task_budget import task_label
        candidates = {str(c.get("id")): c for c in state.get("intent_candidates") or []
                      if isinstance(c, dict)}
        labels = []
        for cid in state.get("limited_tasks") or []:
            candidate = candidates.get(str(cid))
            if candidate is not None:
                labels.append(task_label(candidate.get("task_type")))
        detail = ("未完成目标：" + "、".join(dict.fromkeys(labels)) + "。") if labels else ""
        return detail + "部分任务达到单任务步数上限或模型调用上限，未完成部分没有自动重试；已完成的结果已保留。"
    if reason == "protocol_error":
        if _has_confirmed_observation(state):
            return "后续批次的工具协议异常，本批未执行；此前已确认的操作结果已保留。"
        return "本次请求的工具协议异常，未执行任何领域操作。"
    if reason == "model_unavailable":
        if _has_confirmed_observation(state):
            return "模型暂时不可用，后续处理未继续；此前已确认的操作结果已保留。"
        return "模型暂时不可用，本次未完成处理。"
    if reason == "rule_candidates_unavailable":
        return "目标识别暂时不可用，本次未完成处理。"
    if reason == "safety_assessment_unavailable":
        return ("当前请求未完成安全语义判断，无法继续普通建议或保存；本次没有新增业务操作。"
                "如你正在出现胸痛、呼吸困难、意识异常、大量出血或无法活动等急症，"
                "请立即联系急救或到急诊就医。请稍后重试。")
    if reason == "policy_stop":
        return "本次模型预算或执行策略要求停止，后续步骤已停止；此前已读取的结果已保留，未执行的部分没有自动重试。"
    if status == "blocked" and _has_unconfirmed_write(state):
        return "部分记录的提交状态尚未确认，未作为保存成功处理；请稍后核对记录是否已保存。"
    if status == "blocked" and any(
            isinstance(item, Mapping)
            and item.get("error") in {
                "saved_record_query_unavailable",
                "saved_record_read_failed",
                "saved_record_selection_unavailable",
            }
            for item in state.get("observations") or []):
        return "读取或选择已保存记录暂时失败，无法确认当前值；本次没有使用历史对话原话代替。"
    if reason == "no_progress":
        return "连续处理未取得进展，已停止重复尝试；未完成的目标没有自动重跑。"
    if reason == "supervisor_turn_limit":
        return "已达到本轮规划轮数上限，未完成的目标没有自动重跑。"
    if reason == "tool_step_limit":
        return "已达到本轮工具执行步数上限，未完成的目标没有自动重跑。"
    return {
        "restricted": "当前处理状态无法核实，未确认完成，请先核对记录。",
        "limited": "已达到本轮执行上限，未完成的目标不会自动重跑，请核对后重试。",
        "blocked": "部分记录或建议因写权限未开启或依赖未满足而未完成，请核对后重试。",
        "needs_clarification": "部分目标需要补充或确认信息后才能继续。",
        "rejected": "部分内容因安全或授权原因未处理。",
    }.get(status, "本轮处理未完全完成。")


def _clarification_notice(state: ReactState) -> str:
    """Per-action clarifications already produced by the domain adapter."""
    parts: list[str] = []
    for cid, observation in (state.get("goal_results") or {}).items():
        if not isinstance(observation, dict):
            continue
        if observation.get("status") != ToolStatus.NEEDS_CLARIFICATION.value:
            continue
        summary = observation.get("summary")
        if isinstance(summary, str) and summary.strip():
            parts.append(summary.strip())
            continue
        result = observation.get("result")
        exercise = ""
        fields = observation.get("missing_fields") or []
        if isinstance(result, dict):
            exercise = str(result.get("exercise") or "").strip()
            fields = result.get("missing_fields") or fields
        if exercise or fields:
            from app.services.agent.workout_extractor import format_missing_workout_fields
            labels = format_missing_workout_fields(fields)
            parts.append(f"{exercise or '该训练动作'}还缺{labels or '可确认的训练信息'}，请补充后再保存。")
    return "；".join(dict.fromkeys(parts))


def finalize_node(state: ReactState, dependencies: ReactDependencies) -> dict[str, Any]:
    reason = state.get("termination_reason") or ""
    status = _final_status(state, dependencies)
    if status == "partial":
        # partial 必须确定性说明哪些已确认、哪些未完成，不原样转发模型“全部成功”。
        answer = _partial_notice(state)
    elif status == "complete":
        answer = _extract_final_answer(state)
    elif status == "needs_clarification":
        notice = _notice_for_state(state, status, reason)
        detail = _clarification_notice(state)
        answer = (detail + "\n\n" + notice) if detail else notice
    else:
        answer = _notice_for_state(state, status, reason)
    if status in {"limited", "blocked", "restricted"} and reason not in {
            "urgent_safety_blocked", "safety_assessment_unavailable"}:
        from app.services.agent.failure_details import read_failure_details
        details = read_failure_details(state)
        if details:
            answer = "\n".join(d["message"] + d["remedy"] for d in details) + "\n\n" + answer
    if status == "rejected" and reason != "urgent_safety_blocked":
        candidates = state.get("intent_candidates") or []
        results = state.get("goal_results") or {}
        # Only canonical read-goal evidence refusals may replace the generic
        # safety/authorization notice. Mixed writes and other rejections keep
        # their existing priority and can never be presented as successful.
        readonly = bool(candidates) and all(
            item.get("task_type") in _READ_GOAL_TASK_TYPES and not is_record_write(item)
            for item in candidates)
        rejected = [results.get(item.get("id"), {}) for item in candidates
                    if results.get(item.get("id"), {}).get("status") == "rejected"]
        if readonly and rejected and all(obs.get("error") in {"insufficient_evidence", "context_input_budget_exhausted"} for obs in rejected):
            answer = "\n\n".join(_observation_user_answer(obs) for obs in rejected if _observation_user_answer(obs)) or answer
    deliver_completed = status in {"limited", "partial"} or (
        status == "blocked"
        and not _has_unconfirmed_write(state)
        and not _has_orphaned_write_observation(state)
        and not state.get("unbound_write_calls")
    )
    if deliver_completed:
        # A budget/step/partial/blocked stop is not a reason to discard answers
        # already produced by completed read workers. Deliver them in request
        # order, then state honestly that the rest did not finish.  An
        # unconfirmed write or safety rejection still replaces every ordinary
        # answer; read answers must not imply that write succeeded.
        goals = completed_read_goals(
            state.get("intent_candidates") or [],
            state.get("goal_results") or {},
            state.get("covered_goals") or [],
        )
        delivered = answer_from_completed_read_goals(goals, "")
        if delivered:
            answer = delivered + ("\n\n" + answer if answer else "")
    return {"final_answer": answer, "status": status}


# ---------------------------------------------------------------------------
# 图构造
# ---------------------------------------------------------------------------
def build_react_graph(dependencies: ReactDependencies):
    """返回可独立调用的图；不替换 legacy 图。依赖注入对象绑定到节点闭包。"""

    async def prepare(state: ReactState) -> dict[str, Any]:
        return prepare_node(state, dependencies)

    async def main_agent(state: ReactState) -> dict[str, Any]:
        return await _main_agent_node(state, dependencies)

    async def agent_tools(state: ReactState) -> dict[str, Any]:
        return await execute_react_batch(state, dependencies)

    def completion_check(state: ReactState) -> dict[str, Any]:
        return _completion_check_node(state, dependencies)

    def finalize(state: ReactState) -> dict[str, Any]:
        return finalize_node(state, dependencies)

    def route_after_tools(state: ReactState) -> str:
        # 写运行时：批处理后先做完成检查再决定是否发起下一次主管模型调用；
        # 阶段A/B 默认（False）保持原 agent_tools -> main_agent 边。
        if dependencies.stop_after_batch:
            return "completion_check"
        # 只读：仅当本轮主管是纯工具调用轮（没有产出任何文字）且目标已全部覆盖时，
        # 才确定性终止（该批观察本身即最终回答，不再发起额外“总结”主管调用）；
        # 主管在本轮已写出文字时，观察之后必须继续一轮主管以完成该回复。
        catalog = dependencies.tool_catalog
        readonly_only = bool(catalog) and all(
            getattr(spec, "permission", None) == "read" for spec in catalog.values())
        if readonly_only:
            last_text = ""
            for message in reversed(state.get("messages") or []):
                if isinstance(message, dict) and message.get("role") == "assistant":
                    content = message.get("content")
                    last_text = content if isinstance(content, str) else ""
                    break
            if not last_text.strip() and not _unhandled_goals(state):
                return "completion_check"
        return "main_agent"

    def route_main_agent(state: ReactState) -> str:
        return _route_main_agent(state, dependencies)

    builder = StateGraph(ReactState)
    builder.add_node("prepare", prepare)
    builder.add_node("main_agent", main_agent)
    builder.add_node("agent_tools", agent_tools)
    builder.add_node("completion_check", completion_check)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "main_agent")
    builder.add_conditional_edges(
        "main_agent", route_main_agent,
        {"agent_tools": "agent_tools", "completion_check": "completion_check", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "agent_tools", route_after_tools,
        {"completion_check": "completion_check", "main_agent": "main_agent"},
    )
    builder.add_conditional_edges(
        "completion_check", _route_completion,
        {"main_agent": "main_agent", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)
    return builder.compile()


async def invoke_react(dependencies: ReactDependencies, raw_message: str, **initial: Any) -> dict[str, Any]:
    """便捷入口：以 ``raw_message`` 构建初始状态并运行整张图，返回最终状态。"""
    from app.services.agent.semantics import understand, semantic_candidates
    if not isinstance(initial.get("intent_candidates"), list):
        await understand(dependencies.llm, raw_message)
        candidates, error = semantic_candidates(raw_message)
        if error is None:
            from app.services.agent.task_planner import validated_semantic_tasks
            try:
                enriched = validated_semantic_tasks(raw_message)
            except Exception:
                # Candidate enrichment is part of prepare's candidate gate, not
                # an optional convenience. Let prepare retry and record the
                # explicit candidate-generation failure instead of raising out
                # of the graph or silently using an empty plan.
                logger.exception("react_semantic_candidate_enrichment_failed")
            else:
                initial["intent_candidates"] = (enriched if enriched is not None
                                                else candidates)
    graph = build_react_graph(dependencies)
    state: dict[str, Any] = {"raw_message": raw_message, **initial}
    token = _WRITE_RUNTIME_ACTIVE.set(dependencies.persistence is not None)
    try:
        return await graph.ainvoke(state)
    finally:
        _WRITE_RUNTIME_ACTIVE.reset(token)


__all__ = [
    "MAX_NO_PROGRESS",
    "MAX_SUPERVISOR_PHYSICAL_ATTEMPTS",
    "MAX_SUPERVISOR_TURNS",
    "MAX_TOOL_STEPS",
    "ReactDependencies",
    "SUPERVISOR_STAGE",
    "build_react_graph",
    "completion_check",
    "execute_react_batch",
    "invoke_react",
    "prepare_node",
    "supervisor_step",
]
