"""阶段B只读运行时：模式选择、只读目录、只读循环执行与响应映射。

模式选择复用共享模型任务与来源验证；含写/确认/混合整请求先选 legacy。只读循环复用阶段A原生 tool_calls 图（注入只读目录与
只读 executor），主管用 ``agent.graph.llm.complete_message``，读取走真实 repo。
"""
from __future__ import annotations

import copy
import inspect
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from app.core import clock
from app.core.config import settings
from app.services.agent.react_models import DomainObservation, ToolStatus
from app.services.agent.react_tools import SOURCE_REF_SCHEMA, ToolSpec
from app.services.agent.task_models import (
    TaskType, is_plan_read_task, is_pure_read_spec_task,
)

logger = logging.getLogger(__name__)

# 阶段B只读目录：恰7个 read 工具。回答类工具只接受 query（单次无工具回答），
# 读取类工具按 strict schema（days 严格整数 1..365 / 空参数）。
_QUERY_PARAMS = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "source_ref": SOURCE_REF_SCHEMA},
    "additionalProperties": False,
}
_DAYS_PARAMS = {"type": "object", "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 365}}, "additionalProperties": False}
_EMPTY_PARAMS = {"type": "object", "properties": {}, "additionalProperties": False}

_READONLY_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("answer_training", "回答一般训练/健身问题，或按日期/动作查询已保存的逐条训练记录（只读，来源绑定）。", "read", _QUERY_PARAMS, "training_qa", "training_qa"),
    ToolSpec("advise_nutrition", "回答饮食/营养问题或查询已保存的逐条饮食记录（只读，不进入记录或计算保存路径）。", "read", _QUERY_PARAMS, "nutrition_advice", "nutrition"),
    ToolSpec("plan_training", "给出训练计划预览（只读，不保存计划）。", "read", _QUERY_PARAMS, "training_plan", "plan_adjust"),
    ToolSpec("read_profile", "读取用户档案（身份由应用注入）。", "read", _EMPTY_PARAMS),
    ToolSpec("read_workout_history", "辅助读取近期训练上下文（日期/条数受限）；不能单独完成已保存记录查询。", "read", _DAYS_PARAMS),
    ToolSpec("read_nutrition_summary", "读取营养聚合摘要（不是逐条食物记录；逐条记录请用 advise_nutrition）。", "read", _DAYS_PARAMS),
    ToolSpec("read_current_plan", "读取当前训练计划（不能转为写计划）。", "read", _EMPTY_PARAMS),
)


def readonly_catalog() -> dict[str, ToolSpec]:
    """每请求冻结的只读目录：恰7个 read 工具，模型可见与执行校验共用同一目录。"""
    return {spec.name: spec for spec in _READONLY_TOOLS}


def _readonly_executor() -> dict[str, Any]:
    from app.services.agent import react_adapters, react_readonly_workers
    return {
        "read_profile": react_adapters.read_profile,
        "read_workout_history": react_adapters.read_workout_history,
        "read_nutrition_summary": react_adapters.read_nutrition_summary,
        "read_current_plan": react_adapters.read_current_plan,
        "answer_training": react_readonly_workers.answer_training,
        "advise_nutrition": react_readonly_workers.advise_nutrition,
        "plan_training": react_readonly_workers.plan_training,
    }


def _is_write_task(task: dict[str, Any]) -> bool:
    """A domain label is not an action; only a real side effect is a write."""
    return bool(task.get("side_effect")) or task.get("action") in {"record", "confirm"}


def _has_write_intent(task_specs: list[dict[str, Any]]) -> bool:
    return any(isinstance(task, dict) and _is_write_task(task) for task in task_specs or [])


def _read_spec_value(task: dict[str, Any]) -> dict[str, Any] | None:
    return _scope_value(task, "read_spec")


def _scope_value(task: dict[str, Any], name: str):
    if name in task:
        return task.get(name)
    payload = task.get("payload")
    if isinstance(payload, dict):
        for key in ("semantic", "dialogue"):
            scope = payload.get(key)
            if isinstance(scope, dict) and name in scope:
                return scope.get(name)
    return None


def _read_task_probe(task: dict[str, Any]) -> dict[str, Any]:
    probe = dict(task)
    for name in ("task_type", "action", "reply_purpose", "goal_role", "plan_result", "read_spec"):
        value = _scope_value(task, name)
        if value is not None:
            probe[name] = value
    return probe


def _has_pure_read_spec(task_specs: list[dict[str, Any]]) -> bool:
    for task in task_specs or []:
        if not isinstance(task, dict):
            continue
        if is_pure_read_spec_task(_read_task_probe(task)):
            return True
    return False



def _has_conversation_only_task(task_specs: list[dict[str, Any]]) -> bool:
    """A conversation target has no read tool that can complete it safely.

    Only a deliverable (``goal_role=result``) chitchat target counts. A
    constraint/background sibling must not drag an independent read/advice goal
    into legacy. If the whole request has no result target, the existing legacy
    execution-boundary reply still owns it.
    """
    result_tasks = [
        task for task in task_specs or []
        if isinstance(task, dict)
        and str(_scope_value(task, "goal_role") or "result") == "result"
    ]
    if not result_tasks:
        return True
    for task in result_tasks:
        if str(_scope_value(task, "task_type") or task.get("task_type") or "") != TaskType.CHITCHAT.value:
            continue
        if isinstance(task.get("record_query"), dict):
            continue
        if isinstance(task.get("read_spec"), dict):
            continue
        payload = task.get("payload")
        if isinstance(payload, dict):
            scopes = [payload.get("dialogue"), payload.get("semantic")]
            if any(isinstance(scope, dict) and isinstance(scope.get("record_query"), dict)
                   for scope in scopes):
                continue
            if any(isinstance(scope, dict) and isinstance(scope.get("read_spec"), dict)
                   for scope in scopes):
                continue
        return True
    return False


def select_mode(request, task_specs) -> tuple[str, str]:
    """返回 ``(effective_mode, reason)``。

    - 不带 mode 或 legacy → legacy
    - react_readonly + 开关开启 + compact_v1 + 纯读且每个目标都有只读载体 → react_readonly
    - 应用直读资源（档案/已存计划）→ legacy，由共享 read_resolver 确定性渲染
    - 其余（开关关闭 / 非 compact_v1 / 含写或确认 / 有效 pending 确认 / 纯对话目标）→ legacy，记录原因
    """
    requested = getattr(request, "mode", None) or "legacy"
    if requested != "react_readonly":
        return "legacy", "requested_legacy"
    if _has_write_intent(task_specs):
        return "legacy", "contains_write_or_confirmation"
    if _has_pure_read_spec(task_specs):
        return "legacy", "read_spec_legacy"
    if any(is_plan_read_task(_read_task_probe(task))
           for task in task_specs or [] if isinstance(task, dict)):
        return "legacy", "plan_read_legacy"
    # Preview modification is supported by the readonly plan_training tool,
    # which delegates to the same application-owned timed-plan worker and
    # returns plan_preview for the final turn commit.
    if not settings.react_readonly_enabled:
        return "legacy", "react_readonly_disabled"
    if settings.memory_read_backend != "compact_v1":
        return "legacy", "non_compact_memory_backend"
    if _has_conversation_only_task(task_specs):
        return "legacy", "conversation_requires_legacy"
    return "react_readonly", "pure_read"


def resolve_auto_mode(agent, request, semantics) -> tuple[str, str]:
    """Resolve a request that omitted ``mode`` into one concrete execution mode.

    ``auto`` never overrides an explicit mode. A validated write is upgraded to
    ``react_write`` only when the write runtime gates are ready; otherwise the
    legacy write path keeps working instead of returning a hard ``execution_limited``.
    A pending draft stays legacy so record-clarification continuation is preserved.
    Pure reads follow the same eligibility as an explicit ``react_readonly`` request.
    """
    from app.services.agent.food_calculator import get_pending_food

    if get_pending_food(request.user_id, request.conversation_id) is not None:
        return "legacy", "pending_draft"
    tasks = [task for task in semantics.get("tasks") or [] if isinstance(task, dict)]
    if any(_is_write_task(task) for task in tasks):
        reason = agent._write_gate_reason()
        if reason is not None:
            return "legacy", reason
        from app.db import session as db

        if not getattr(db, "DB_READY", False):
            # The legacy path will report a real storage failure; auto must not
            # turn a missing database into a write-runtime limited response.
            return "legacy", "database_unavailable"
        return "react_write", "auto_write"
    probe = request.model_copy(update={"mode": "react_readonly"})
    mode, reason = select_mode(probe, tasks)
    if mode == "react_readonly":
        return "react_readonly", reason
    return "legacy", reason


@dataclass(frozen=True)
class ReadonlyPreparation:
    """One request's pre-route judgement, owned by the shared semantics boundary."""

    mode: str
    reason: str
    tasks: tuple[dict[str, Any], ...] = ()
    intent: Any = None
    needs_clarification: bool = False


async def prepare_readonly(agent, request, memory: dict | None = None) -> ReadonlyPreparation:
    """Decide one new ``react_readonly`` request before any legacy parse or plan freeze.

    Deterministic eligibility comes first (explicit legacy/disabled/non-compact/
    valid pending): those paths never buy a new semantic call. Only an eligible
    first request runs the existing shared RequestUnderstanding once inside its
    request budget, with the real application memory context. A validated
    record/confirm/side-effect candidate turns the whole request into legacy; a
    validated non-write selection stays readonly and reuses the same tasks.
    """
    requested = getattr(request, "mode", None) or "legacy"
    if requested != "react_readonly":
        return ReadonlyPreparation("legacy", "requested_legacy")
    if not settings.react_readonly_enabled:
        return ReadonlyPreparation("legacy", "react_readonly_disabled")
    if settings.memory_read_backend != "compact_v1":
        return ReadonlyPreparation("legacy", "non_compact_memory_backend")
    from app.services.agent.food_calculator import get_pending_food
    if get_pending_food(request.user_id, request.conversation_id) is not None:
        # An unexpired draft keeps legacy pending binding/TTL/ownership; never
        # buy a semantic judgement that could bypass the bound confirmation.
        return ReadonlyPreparation("legacy", "contains_write_or_confirmation")

    from app.services.agent.semantics import (
        get_understanding, safety_for, semantic_attempt_failed, semantic_candidates, understand,
    )
    from app.services.agent.task_planner import validated_semantic_tasks
    tasks, error = semantic_candidates(request.message)
    if error is not None:
        from app.services.llm.budget import ModelBudgetError
        try:
            await understand(agent.graph.llm, request.message, context=memory)
        except ModelBudgetError:
            # A real budget/policy stop is an explicit limited outcome; it must
            # never fall back to text guessing or a second paid attempt.
            logger.warning("readonly_semantic_budget_stopped")
            return ReadonlyPreparation("limited", "semantic_budget_stopped")
        except Exception as exc:
            from app.services.llm.client import LLMErrorCategory, LLMGatewayError
            if isinstance(exc, LLMGatewayError) and exc.category == LLMErrorCategory.POLICY:
                logger.warning("readonly_semantic_policy_stopped")
                return ReadonlyPreparation("limited", "semantic_budget_stopped")
            logger.warning("readonly_semantic_failed", exc_info=True)
            return ReadonlyPreparation("limited", "semantic_understanding_unavailable")
        tasks, error = semantic_candidates(request.message)
    if error is not None:
        if semantic_attempt_failed(request.message):
            # A configured model was tried and failed. The real legacy parser
            # owns the explicit unavailable decision; a controlled validated
            # write plan may still run legacy once, while a read-only fallback is
            # never reported as pure_read (see execution.py).
            return ReadonlyPreparation("defer", "semantic_understanding_unavailable")
        # No source/schema-validated RequestUnderstanding exists: a readonly
        # request may not be confirmed as pure_read from rule candidates, and it
        # may not continue to a legacy parser that could guess a write.
        logger.warning("readonly_semantic_unavailable")
        return ReadonlyPreparation("limited", "semantic_understanding_unavailable")
    enriched = validated_semantic_tasks(request.message)
    if enriched is not None:
        tasks = enriched
    safety = safety_for(request.message)
    if safety.get("status") != "assessed":
        return ReadonlyPreparation("limited", "safety_assessment_unavailable")
    if safety.get("level") == "urgent":
        return ReadonlyPreparation("limited", "safety_urgent")
    from app.services.agent.intent import IntentResult
    from app.services.agent.task_planner import intent_from_planned_tasks
    mode, reason = select_mode(request, tasks)
    understanding = get_understanding(request.message) or {}
    intent = intent_from_planned_tasks(tasks, IntentResult("training_qa", 0, action="question"))
    return ReadonlyPreparation(mode, reason, tuple(tasks), intent,
                               needs_clarification=bool(understanding.get("needs_clarification")))


def _cache_key(name: str, args: Any) -> str:
    """许可读取工具 + 规范参数：python/json 往返保证同参数稳定同一 key。"""
    try:
        payload = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        payload = repr(args)
    return f"{name}|{payload}"


def _cached_executor(base: dict[str, Any]):
    """当前请求局部的读取结果缓存：成功结果存快照，取出/存入都断开可变引用。

    同请求同工具同参数只真实执行一次；失败/拒绝不缓存；下一个请求重建缓存并重读
    SQL 新值。缓存只是只读事实，不改变目录、权限或授权。
    """

    cache: dict[str, dict[str, Any]] = {}

    async def execute(name: str, args: Any, context: dict[str, Any]) -> Any:
        key = _cache_key(name, args)
        if key in cache:
            return copy.deepcopy(cache[key])
        handler = base.get(name)
        if handler is None:
            return None
        value = handler(args, context)
        if inspect.isawaitable(value):
            value = await value
        observation = DomainObservation.normalize(value)
        data = observation.to_dict()
        if observation.status == ToolStatus.COMPLETED:
            cache[key] = copy.deepcopy(data)
        return copy.deepcopy(data)

    return execute


def _budget_stop_check() -> bool:
    """现有 request scope 的 policy-stop/取消/截止时间信号（不新建 scope/账本）。"""
    from app.services.llm.budget import active_scope

    scope = active_scope.get()
    if scope is None:
        return False
    if getattr(scope, "policy_stops", None):
        return True
    try:
        return scope.remaining() <= 0
    except Exception:
        return False


def _readonly_candidates(task_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """只读候选：仅保留非写任务，供主管选工具与目标匹配（不冻结完整计划）。"""
    candidates: list[dict[str, Any]] = []
    for task in task_specs or []:
        if not isinstance(task, dict):
            continue
        if _is_write_task(task):
            continue
        candidates.append(dict(task))
    return candidates


def _collect_citations(state: dict[str, Any]) -> list[dict[str, Any]]:
    """汇总实际完成的回答观察携带的结构化引用（只收编已完成、去重并规范化）。"""
    from app.schemas.citation import normalize_citations

    collected: list[dict[str, Any]] = []
    for observation in state.get("observations") or []:
        if not isinstance(observation, dict) or not (
                observation.get("status") == "completed" or
                (observation.get("status") == "rejected" and observation.get("error") == "insufficient_evidence")):
            continue
        result = observation.get("result")
        if isinstance(result, dict) and isinstance(result.get("citations"), list):
            collected.extend(result["citations"])
    return normalize_citations(collected)


def _map_readonly_result(state: dict[str, Any]) -> dict[str, Any]:
    """把 react 只读循环终态映射为 legacy ``graph.run`` 兼容结构。"""
    final_answer = str(state.get("final_answer") or "")
    status = state.get("status") or "complete"
    facts = state.get("call_facts") or []
    # 主管是否真实调用模型：本批产生过普通调用事件即说明图已运行模型选工具/回答。
    model_used = bool(facts)
    clarifications = [item["result"]["clarification"] for item in state.get("observations", [])
                      if isinstance(item, dict) and isinstance(item.get("result"), dict) and item["result"].get("clarification")]
    if clarifications:
        final_answer = "\n\n".join(item["question"] for item in clarifications)
    state_update_requested = any(
        isinstance(item, dict) and isinstance(item.get("result"), dict)
        and item["result"].get("state_update_requested") is True
        for item in state.get("observations", []))
    plan_preview = None
    for observation in state.get("observations") or []:
        if not isinstance(observation, dict) or observation.get("status") != "completed":
            continue
        result = observation.get("result")
        if isinstance(result, dict) and isinstance(result.get("plan_preview"), dict):
            plan_preview = result["plan_preview"]
    from app.services.agent.failure_details import read_failure_details
    return {
        "final_answer": final_answer,
        "worker_results": [{"worker": "react_readonly", "status": status, "answer": final_answer}],
        "citations": _collect_citations(state),
        "model_used": model_used,
        "safety_level": state.get("safety_level", "normal"),
        "current_task": {"task_type": "training_qa"},
        "react_status": status,
        "react_termination_reason": state.get("termination_reason", ""),
        "react_supervisor_turns": int(state.get("supervisor_turns") or 0),
        "react_tool_steps": int(state.get("tool_steps") or 0),
        "failure_details": read_failure_details(state),
        "clarifications": clarifications,
        "state_update_requested": state_update_requested,
        "plan_preview": plan_preview,
    }


async def run_readonly(agent, request, task_specs) -> dict[str, Any]:
    """执行阶段B只读循环，返回 legacy ``graph.run`` 兼容结构。"""
    from app.services.agent.react_graph import ReactDependencies, invoke_react

    catalog = readonly_catalog()
    deps = ReactDependencies(
        llm=agent.graph.llm,
        tool_executor=_cached_executor(_readonly_executor()),
        tool_catalog=catalog,
        user_id=request.user_id,
        conversation_id=request.conversation_id,
        request_id=request.request_id,
        policy_stop_check=_budget_stop_check,
    )
    candidates = _readonly_candidates(task_specs)
    from app.services.agent.semantics import safety_for
    safety = safety_for(request.message)
    state = await invoke_react(
        deps,
        request.message,
        intent_candidates=candidates,
        safety_level=str(safety.get("level") or "unknown") if safety.get("status") == "assessed" else "unknown",
    )
    return _map_readonly_result(state)


__all__ = ["ReadonlyPreparation", "prepare_readonly", "readonly_catalog", "resolve_auto_mode",
           "run_readonly", "select_mode"]
