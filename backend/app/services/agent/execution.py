from __future__ import annotations

from app.services.rag.blocking import run_retrieval

import asyncio
import logging
from app.core import clock
from app.core.config import settings
from app.services.agent.operation_store import OperationStoreUnavailable, current_operations
from app.services.memory.compact import current_turn
from app.services.mcp.executor import ToolSchemaError
from app.services.rag.quality import assess_rag_evidence, insufficient_evidence_reply

from dataclasses import asdict

from langsmith import tracing_context

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.citation import citations_from_hits, normalize_citations

from app.services.agent.intent import IntentResult, is_greeting
from app.services.agent.workout_extractor import InvalidWorkoutCandidate, invalid_structured_workout_candidates
from app.services.context.budget import ContextBudgetExceeded
from app.services.memory.redis_memory import memory_service
from app.services.observability import trace_span
from app.services.security.input_guard import inspect_input
from app.services.security.output_guard import apply_output_guardrail


from app.services.agent.execution_state import _request_memory, _resumed_frozen_mode, PlannerDecision
from app.services.telemetry import active_metrics, mark_degraded, mark_safety_refusal
from app.services.agent.events import emit
logger = logging.getLogger(__name__)


def _chat_model_used() -> bool:
    """Whether this request actually produced a metered chat attempt."""
    metrics = active_metrics.get()
    return bool(metrics and any(
        call.get("capability") == "chat" for call in metrics.calls))


_PLANNER_FAILURE_REASONS = {
    "planner_llm_unavailable_or_invalid", "planner_task_mapping_failed",
    "planner_invalid_schema", "planner_bad_span", "planner_low_coverage",
}


def _task_scope_value(task: dict, name: str):
    """Read a validated task field from either the task or its answer scope."""
    if not isinstance(task, dict):
        return None
    if name in task:
        return task.get(name)
    payload = task.get("payload")
    if not isinstance(payload, dict):
        return None
    for key in ("semantic", "dialogue"):
        scope = payload.get(key)
        if isinstance(scope, dict) and name in scope:
            return scope.get(name)
    return None


def _single_nutrition_totals_query(task_specs) -> tuple[dict, dict] | None:
    """One date-range-only saved-diet query, answered by application totals.

    The React supervisor may otherwise ask the model to narrate a summary and
    add target gaps or a follow-up question. A date-only fact recall has no
    ambiguity for source selection: every SQL row in that range belongs to the
    requested total. Return the task and its query only for that closed shape.
    """
    result_tasks = [
        task for task in task_specs or []
        if isinstance(task, dict)
        and str(_task_scope_value(task, "goal_role") or "result") == "result"
    ]
    if len(result_tasks) != 1:
        return None
    task = result_tasks[0]
    if str(_task_scope_value(task, "task_type") or task.get("task_type") or "") != "nutrition_record":
        return None
    if str(_task_scope_value(task, "reply_purpose") or "") != "fact_recall":
        return None
    query = _task_scope_value(task, "record_query")
    if not isinstance(query, dict):
        return None
    if any(query.get(key) for key in ("meal_types", "food_names", "exercises")):
        return None
    if not query.get("date_start") or not query.get("date_end"):
        return None
    return task, query


def _planner_unavailable_for_write(degraded_reason: str | None, semantics: dict) -> bool:
    """A parser failure after a validated write intent must not become user blame.

    Read-only requests keep the existing rule-plan fallback; only an already
    understood record/confirm task stops with an explicit unavailable reply.
    """
    if degraded_reason not in _PLANNER_FAILURE_REASONS:
        return False
    return any(
        task.get("action") in {"record", "confirm"}
        for task in semantics.get("tasks") or [] if isinstance(task, dict)
    )


class ChatExecutionMixin:
    async def _build_nutrition_advice_messages(
        self,
        request: ChatRequest,
        memory: dict,
        record_result: dict,
        *,
        hits: list | None = None,
        context: dict | None = None,
        evidence_analysis=None,
    ):
        """Build advice context after the record transaction has completed."""
        if context is None:
            context = await self.graph._load_context_node({"user_id": request.user_id})
        if hits is None:
            from app.services.rag.online import search
            hits = await search(self.graph.rag, "营养饮食 " + request.message, top_k=5, intent_result=IntentResult("nutrition", 1.0, action="question"))
        worker_state = {
            "raw_message": request.message,
            **context,
            **memory,
            "newly_recorded_meal": self._last_worker_result(record_result).get("calculation", {}),
        }
        return (
            self.graph._planning_prompt(
                "饮食营养方案", worker_state, hits,
                evidence_analysis=evidence_analysis),
            hits,
            context,
        )


    async def _complete_nutrition_advice(self, request: ChatRequest, memory: dict, record_result: dict,
                                         task: dict | None = None) -> tuple[str, bool, list[dict]]:
        from app.services.agent.task_budget import task_model_budget
        from app.services.rag.evidence_conflict import (
            controlled_evidence_answer, filter_hits_for_answer,
            finalize_evidence_answer,
        )
        budget = None
        analysis = None
        try:
            with task_model_budget(task or {"task_type": "nutrition_advice",
                                            "text": request.message}, request.message) as budget:
                messages, hits, context = await self._build_nutrition_advice_messages(request, memory, record_result)
                quality = assess_rag_evidence(request.message, hits, task_type="nutrition_advice")
                if not quality.answerable:
                    return insufficient_evidence_reply("nutrition_advice", quality), False, []
                from app.services.rag.evidence_conflict import analyze_evidence_conflicts
                analysis = await analyze_evidence_conflicts(
                    self.graph.llm, request.message, hits, task_type="nutrition_advice")
                filtered = filter_hits_for_answer(hits, analysis)
                hits = list(filtered.hits)
                if filtered.applied:
                    filtered_quality = assess_rag_evidence(
                        request.message, hits, task_type="nutrition_advice")
                    if not filtered_quality.answerable:
                        return insufficient_evidence_reply("nutrition_advice", filtered_quality, filtered=True), False, []
                controlled = controlled_evidence_answer(analysis)
                if controlled is not None:
                    return controlled, False, citations_from_hits(hits)
                if analysis.status != "not_applicable":
                    messages, _, _ = await self._build_nutrition_advice_messages(
                        request, memory, record_result, hits=hits, context=context,
                        evidence_analysis=analysis)
                answer = await self.graph.llm.complete(messages)
            if budget is not None and getattr(budget, "limited", False):
                return "这部分饮食建议已达到模型调用上限，未完成；已完成的记录保留。", False, []
            if answer:
                return finalize_evidence_answer(str(answer), analysis), True, citations_from_hits(hits)
            mark_degraded("answer_model_fallback", failure=True)
            return finalize_evidence_answer(
                self.graph._nutrition_fallback({
                    **context, "nutrition_summary": context.get("nutrition_summary", {})}),
                analysis), False, citations_from_hits(hits)
        except ContextBudgetExceeded:
            return "当前问题和必要的安全信息超过了上下文预算，请缩短本次输入后再试。", False, []
        except Exception as exc:
            mark_degraded("answer_model_fallback", failure=True)
            from app.services.llm.budget import is_task_budget_stop
            if is_task_budget_stop(exc) or (
                    budget is not None and getattr(budget, "limited", False)):
                return "这部分饮食建议已达到模型调用上限，未完成；已完成的记录保留。", False, []
            fallback = self.graph._nutrition_fallback({"nutrition_summary": {}})
            return finalize_evidence_answer(fallback, analysis), False, []


    @staticmethod
    def _plan_has_write(tasks) -> bool:
        """A failed ledger may only block plans that contain side effects."""
        return any(isinstance(task, dict) and bool(task.get("side_effect")) for task in tasks)

    async def _stable_plan(self, request: ChatRequest, task_specs) -> list[dict]:
        """Persist the first confirmed plan and reuse it for every retry (P1-2).

        The plan is stored before any write is dispatched. A retry of the same
        request therefore keeps the original task order, authorization and
        stable operation positions even if the model would answer differently.

        If the ledger cannot be read or persisted, a plan containing writes
        fails closed (``OperationStoreUnavailable``): no save tool may run
        against a plan that a retry could not recover. Read-only plans keep
        working because they carry no replay risk.
        """
        tasks = list(task_specs)
        store = current_operations.get()
        if store is None:
            return tasks
        try:
            stored = await asyncio.to_thread(store.stored_plan)
            if stored is not None:
                logger.info("request_plan_reused", extra={"request_id": request.request_id,
                                                          "tasks": len(stored)})
                return stored
            # R011-H: invalid structured candidates are rejected before the plan
            # is persisted, so a non-finite value never reaches the JSON ledger
            # and is not misreported as a storage outage.
            candidate_problems = invalid_structured_workout_candidates(tasks)
            if candidate_problems:
                raise InvalidWorkoutCandidate(candidate_problems)
            await asyncio.to_thread(store.save_plan, tasks, time_base=clock.utc_now())
        except OperationStoreUnavailable:
            if self._plan_has_write(tasks):
                raise
            logger.exception("operation_ledger_unavailable_readonly",
                             extra={"request_id": request.request_id})
        return tasks

    @staticmethod
    def _write_gate_reason() -> str | None:
        """Application-owned zero-call gates before buying any semantics."""
        if _resumed_frozen_mode.get() == "react_write" and not settings.react_write_enabled:
            return "react_write_disabled_frozen"
        if not settings.react_write_enabled:
            return "react_write_disabled"
        if settings.memory_read_backend != "compact_v1":
            return "non_compact_memory_backend"
        return None

    def _write_unavailable_response(self, request: ChatRequest, guard,
                                    safety_level: str, reason: str) -> ChatResponse:
        response = self._early_stop_response(
            request, self._merge_safety(safety_level, guard.level), reason,
            "本次写入请求未进入执行：写入运行时当前不可用；未执行模型或保存。")
        return response

    async def _respond_readonly(self, safe_request: ChatRequest, guard, decision, mode_metrics: dict,
                                safety_level: str = "unknown") -> ChatResponse:
        """阶段B只读循环：绕过 legacy planner/稳定计划冻结，复用输出守卫与记忆提交。

        只读循环失败不切换 legacy 重跑，明确受限；含写/确认整请求不会进入本分支。
        """
        from app.services.agent.react_runtime import run_readonly
        try:
            result = await run_readonly(self, safe_request, list(decision.task_specs))
        except Exception:
            mark_degraded("react_readonly_failed", failure=True)
            answer = "本次只读处理未能完成，请稍后重试；涉及保存的数据请先在记录页核对。"
            response = ChatResponse(answer=answer, intent="error", citations=[],
                                    safety_level=self._merge_safety(safety_level, guard.level), model_used=False,
                                    metrics={**mode_metrics, "react_status": "error"})
            await self._persist_turn(safe_request, answer)
            return response
        final_answer = self._record_answer(result)
        citations = result.get("citations", [])
        model_used = bool(result.get("model_used", False))
        guarded = apply_output_guardrail(final_answer)
        if result.get("clarifications"):
            from app.services.agent import planning_questions
            memory = _request_memory.get() or {}
            try:
                await planning_questions.remember(
                    safe_request, result, list(decision.task_specs),
                    memory.get("information_sources", []))
            except Exception:
                logger.warning("planning_question_persistence_failed")
                mark_degraded("planning_question_persistence", failure=True)
                result["clarifications"] = []
                guarded = apply_output_guardrail(
                    "待补充的计划暂时无法保留，请稍后重新说明训练安排；本次没有新增保存。")
        if result.get("state_update_requested"):
            mode_metrics["state_update_requested"] = True
        response = ChatResponse(
            answer=guarded.text,
            clarifications=result.get("clarifications", []),
            intent=self._response_intent(result),
            citations=citations,
            safety_level=self._merge_safety(safety_level, self._merge_safety(guard.level, result.get("safety_level", "normal"))),
            model_used=model_used,
            metrics={**mode_metrics, "react_status": result.get("react_status", ""),
                     **{key: result[key] for key in ("react_termination_reason", "react_supervisor_turns", "react_tool_steps", "failure_details") if key in result},
                     **({"plan_preview": result["plan_preview"]}
                        if isinstance(result.get("plan_preview"), dict) else {})},
        )
        await self._persist_turn(safe_request, response.answer)
        return response

    async def _record_mode_metadata(self, request: ChatRequest, effective_mode: str, mode_reason: str) -> None:
        """首次请求：把已判定的 effective_mode/reason 合并进现有记忆 JSON 元数据。

        requested_mode 已由 begin_turn 原子冻结；本方法只做 best-effort 补充，失败不
        影响业务响应（无新表/列/账本）。
        """
        tid = current_turn.get()
        if not tid or (getattr(request, "mode", None) or "legacy") != "react_readonly":
            return
        try:
            from app.services.memory.compact import attach_effective_mode
            await asyncio.to_thread(attach_effective_mode, tid, effective_mode, mode_reason)
        except Exception:
            logger.warning("mode_metadata_persist_failed", exc_info=True)

    async def _respond_write(self, request: ChatRequest, guard, safety_level: str = "unknown",
                             *, mode_reason: str = "explicit_write",
                             declared_mode: str | None = None) -> ChatResponse:
        """C2-3 写入入口：资格与预算门先于一切 legacy 拆分/计划冻结/领域派发。

        开关关闭/非 compact_v1/失计量/运行时未就绪都返回明确 execution_limited，
        零模型、零桥、零 C1 业务写，且不回落 legacy。``mode_reason`` 区分客户端显式
        react_write 与服务端 auto 解析出的写入；``declared_mode`` 保留原始声明。
        """
        from app.services.agent import react_write_runtime as write_runtime

        mode_metrics = {"requested_mode": "react_write"}
        if declared_mode is not None:
            mode_metrics["declared_mode"] = declared_mode
        if (_request_memory.get() or {}).get("_record_reply_safety_level") == "medical":
            safety_level = self._merge_safety("medical", safety_level)

        reason = self._write_gate_reason()
        if reason is not None:
            return self._write_unavailable_response(request, guard, safety_level, reason)
        mode_metrics.update({"execution_mode": "react_write", "mode_reason": mode_reason})
        try:
            result = await write_runtime.run_write(self, request)
        except write_runtime.WriteRuntimeUnavailable:
            return self._early_stop_response(
                request, self._merge_safety(safety_level, guard.level),
                "write_runtime_unavailable",
                "本次写入运行时尚未就绪，没有新增保存。")
        metrics = {
            **mode_metrics,
            "mode_reason": "privacy_invalidated" if result.get("mode_reason") == "privacy_invalidated" else mode_metrics["mode_reason"],
            "react_write": str(result.get("react_write_status") or ""),
            "react_status": str(result.get("react_status") or ""),
            "receipt_refs": list(result.get("receipt_refs") or []),
            "pending_confirmations": list(result.get("pending_confirmations") or []),
        }
        # 真实冻结 write ID 贯穿 typed HTTP：常规 payload 以嵌套 response_ids 携带原
        # controller 身份；仅在 request_id 与本请求一致（非客户端注入/别名）时投影为
        # top-level；top-level 缺失才用嵌套，保持 terminal 形状不变。
        nested = result.get("response_ids") if isinstance(result.get("response_ids"), dict) else {}
        nested_trusted = (
            all(isinstance(nested.get(key), str) and nested.get(key)
                for key in ("request_id", "message_id", "run_id"))
            and nested.get("request_id") == str(request.request_id))

        def _frozen_id(key: str):
            top = result.get(key)
            if isinstance(top, str) and top:
                return top
            if nested_trusted:
                return str(nested[key])
            return None

        return ChatResponse(
            answer=str(result.get("final_answer") or ""),
            intent=str(result.get("intent") or "execution_limited"),
            citations=list(result.get("citations") or []),
            safety_level=self._merge_safety(safety_level, str(result.get("safety_level") or guard.level)),
            model_used=bool(result.get("model_used", False)),
            metrics=metrics,
            replayed=bool(result.get("replayed", False)),
            request_id=_frozen_id("request_id"),
            message_id=_frozen_id("message_id"),
            run_id=_frozen_id("run_id"),
            clarifications=result.get("clarifications", []),
        )

    async def _restore_persisted_semantics(self, request: ChatRequest) -> None:
        """Reuse the original request's model decision after a real resume.

        The early safety gate must run before the write runtime, but a retried
        react_write request already has its validated RequestUnderstanding in
        the C1 controller snapshot. Restoring it is source/digest-bound inside
        ``_restore_source_understanding``; a new request or a different text
        never receives another request's judgment, and no provider call is
        repeated just because the process restarted.
        """
        if (getattr(request, "mode", None) or "legacy") != "react_write":
            return
        try:
            from app.db import session as db
            from app.services.agent import execution_ledger_store as ledger_api
            from app.services.agent.react_write_runtime import WriteRuntimeSession
            factory = getattr(db, "SessionLocal", None)
            if factory is None:
                return
            snapshot = await asyncio.to_thread(
                ledger_api.ExecutionLedgerStore(factory).load_request,
                str(request.user_id), str(request.request_id))
            if snapshot:
                WriteRuntimeSession._restore_source_understanding(snapshot, request.message)
        except Exception:
            # Restore is an optimization for idempotent resume, never a source
            # of authorization. On failure the normal model path runs and the
            # write gates still fail closed if safety is unavailable.
            logger.warning("resume_semantic_restore_failed", exc_info=True)

    @staticmethod
    def _early_stop_response(request: ChatRequest, safety_level: str, reason: str,
                             default_answer: str) -> ChatResponse:
        """Build a limited response without claiming zero when C1 shows progress."""
        progress = None
        if (getattr(request, "mode", None) or "legacy") == "react_write":
            try:
                from app.services.agent.react_write_runtime import unfinished_progress
                progress = unfinished_progress(request)
            except Exception:
                progress = None
        ids = (progress or {}).get("response_ids") or {}
        trusted_ids = all(isinstance(ids.get(key), str) and ids.get(key)
                          for key in ("request_id", "message_id", "run_id"))
        metrics = {"requested_mode": getattr(request, "mode", None) or "legacy",
                   "execution_mode": "limited", "mode_reason": reason,
                   "react_status": "limited",
                   "receipt_refs": list((progress or {}).get("receipt_refs") or [])}
        if progress is not None:
            metrics["react_write"] = "limited"
            if trusted_ids:
                metrics["preserve_frozen_ids"] = True
        return ChatResponse(
            answer=str((progress or {}).get("answer") or default_answer),
            intent="execution_limited", citations=[], safety_level=safety_level,
            model_used=False,
            request_id=ids.get("request_id") if trusted_ids else None,
            message_id=ids.get("message_id") if trusted_ids else None,
            run_id=ids.get("run_id") if trusted_ids else None,
            metrics=metrics)

    @classmethod
    def _semantic_limit_response(cls, request: ChatRequest, safety_level: str,
                                 error: Exception | None = None) -> ChatResponse:
        from app.services.llm.budget import ModelBudgetError
        from app.services.llm.client import LLMGatewayError, LLMErrorCategory

        # Read typed error metadata only; provider messages may contain private data.
        pending, seen, code, upstream = [error], set(), None, None
        while pending and len(seen) < 16:
            item = pending.pop(0)
            if not isinstance(item, BaseException) or id(item) in seen:
                continue
            seen.add(id(item))
            if isinstance(item, (ModelBudgetError, LLMGatewayError)):
                code = code or getattr(item, "code", None)
            if (isinstance(item, LLMGatewayError) and item.retryable
                    and item.category in (LLMErrorCategory.UPSTREAM, LLMErrorCategory.TRANSPORT)):
                upstream = item
            pending.extend([item.__cause__, item.__context__])
            if isinstance(item, LLMGatewayError):
                pending.append(item.retry_failure)
        if code in ("model_request_deadline", "model_budget_expired"):
            reason, answer = "deadline", "本次处理已超时，未能完成请求。请稍后重试。"
        elif code == "model_request_cancelled":
            reason, answer = "cancelled", "本次请求已取消，未能完成处理。"
        elif isinstance(code, str) and (code in (
                "model_prepare_token_budget_exhausted", "model_task_token_budget_exhausted")
                or code.startswith("model_token_budget_exhausted:")):
            reason = "token_budget"
            answer = ("模型服务暂时不可用，本次 Token 预算不足以继续重试。请稍后重试。"
                      if upstream is not None else
                      "本次请求的 Token 预算不足，无法继续处理。请缩小请求范围或稍后重试。")
        elif isinstance(code, str) and (code in (
                "model_prepare_budget_exhausted", "model_task_budget_exhausted")
                or code.startswith("model_budget_exhausted:")):
            reason = "call_budget"
            answer = ("模型服务暂时不可用，本次调用次数预算不足以继续重试。请稍后重试。"
                      if upstream is not None else
                      "本次请求的模型调用次数已达到限制，未能完成处理。请稍后重试。")
        else:
            reason, answer = "policy", "本次请求受模型调用策略限制，未能完成处理。请稍后重试。"
        from app.services.agent.failure_details import exception_failure
        failure_detail = exception_failure(error) if error is not None else None
        if reason == "policy" and failure_detail:
            answer = failure_detail["message"] + failure_detail["remedy"]
        elif failure_detail and failure_detail.get("preceding_failure"):
            answer = failure_detail["preceding_failure"]["message"] + "；" + answer
        response = cls._early_stop_response(request, safety_level, "semantic_budget_stopped", answer)
        if failure_detail:
            response.metrics["failure_details"] = [failure_detail]
        response.metrics.update({"model_stop_reason": reason, "retry_after_provider_failure": upstream is not None})
        if upstream is not None and upstream.status_code in (408, 429, 500, 502, 503, 504):
            response.metrics["upstream_status"] = upstream.status_code
        return response

    @classmethod
    def _safety_unavailable_response(cls, request: ChatRequest, safety_level: str,
                                     memory: dict | None = None) -> ChatResponse:
        from app.services.agent.semantics import understanding_unavailable_reply

        code = None
        if isinstance(memory, dict):
            code = memory.get("semantic_failure_code")
        if code == "context_input_budget_exhausted":
            context_detail = memory.get("semantic_failure_detail")
            answer = "本次必要输入超出上下文预算，无法完整提交给AI。请缩小需引用的计划或历史范围后重试；本次未执行保存或生成计划。"
            if context_detail and (context_detail["code"] == "semantic_unit_dependency_missing"
                    or context_detail.get("phase") == "structured_correction"
                    or context_detail.get("component") == "tools"):
                answer = context_detail["message"] + context_detail["remedy"]
            # Compose the default first: an existing C1 receipt/progress answer
            # has priority and must never be overwritten by a context diagnostic.
            response = cls._early_stop_response(request, safety_level, code, answer)
            response.metrics["semantic_failure_code"] = code
            if context_detail:
                response.metrics["failure_details"] = [context_detail]
            return response
        # Reuse the shared, allow-listed failure classification instead of
        # claiming every failure was a structured-output error. Only the
        # category wording is copied; provider bodies, prompts and user text
        # never enter the reply.
        failure_detail = memory.get("semantic_failure_detail") if isinstance(memory, dict) else None
        if code == 'semantic_result_reference_invalid':
            # Source validation failure is not an emergency judgment. Stop all
            # execution without relabelling an unrelated source fault as safety.
            answer = ('无法确认你要参考或修改的那份回答。请明确所指计划，或重新提供相关内容；'
                      '本次未执行保存，也未生成或修改计划。')
            if failure_detail:
                answer = failure_detail['message'] + failure_detail['remedy'] + '本次未执行保存或生成计划。'
            response = cls._early_stop_response(request, safety_level, code, answer)
            response.metrics['semantic_failure_code'] = code
            if failure_detail:
                response.metrics['failure_details'] = [failure_detail]
            return response
        detail = understanding_unavailable_reply(failure_detail=failure_detail)
        requested_mode = getattr(request, "mode", None)
        if not requested_mode and isinstance(memory, dict):
            requested_mode = memory.get("_requested_mode")
        zero_write = ("本次未保存任何记录，也没有保存任何数据，"
                      "也无法生成或修改训练计划/预览。"
                      if requested_mode == "react_write"
                      else "本次没有保存任何数据，也无法生成或修改训练计划/预览。")
        response = cls._early_stop_response(
            request, safety_level, "safety_assessment_unavailable",
            "本次请求未完成安全语义判断：" + detail + " " + zero_write
            + "如你正在出现胸痛、呼吸困难、意识异常、大量出血或无法活动等急症，"
            "请立即联系急救或到急诊就医。")
        if code:
            response.metrics["semantic_failure_code"] = str(code)[:160]
        if failure_detail:
            response.metrics["failure_details"] = [failure_detail]
        return response

    async def _understand_or_limit(self, request: ChatRequest, memory: dict,
                                   safety_level: str) -> ChatResponse | None:
        """Run the shared semantic gate; a policy/budget stop stays explicit."""
        from app.services.agent.semantics import understand
        from app.services.llm.budget import ModelBudgetError
        from app.services.llm.client import LLMErrorCategory, LLMGatewayError
        from app.services.memory.wiki_context import WikiReadError
        try:
            await self._restore_persisted_semantics(request)
            await understand(self.graph.llm, request.message, context=memory)
        except WikiReadError as exc:
            answer = ('已保存的必要约束存在同版本冲突，请先确认应采用的值；本次未新增记录。'
                      if str(exc) == 'persistent_constraint_conflict' else
                      '本次依赖的历史来源没有被明确选择，无法继续该目标。请明确要继续的计划或对话范围；本次未新增记录。'
                      if str(exc) == 'wiki_address_required' else
                      '本次所选记忆语义单元缺少必要依赖或必要内容超出读取预算，无法在不丢失条件/否定的情况下继续；'
                      '请缩小范围或明确要继续的来源；本次未新增记录。'
                      if str(exc).startswith('wiki_semantic_unit_') or str(exc).startswith('wiki_cached_unit_') else
                      '本次需要的历史原文暂时无法完整读取，可能已变更或超出本次读取预算。请明确要继续的计划或对话范围；本次未新增记录。')
            return ChatResponse(
                answer=answer,
                intent="clarification", model_used=bool(memory.get('wiki_locator_attempted') or memory.get('wiki_semantic_attempted')), safety_level=safety_level,
                metrics={"memory_context_status":"unavailable", "memory_context_reason":str(exc)})
        except ModelBudgetError as exc:
            return self._semantic_limit_response(request, safety_level, exc)
        except LLMGatewayError as exc:
            if exc.category == LLMErrorCategory.POLICY:
                return self._semantic_limit_response(request, safety_level, exc)
            raise
        return None

    async def _respond_impl(self, request: ChatRequest, parsed_input: dict | None = None) -> ChatResponse:
        guard = inspect_input(request.message)
        safe_request = request.model_copy(update={"message": guard.sanitized_text})
        safety_level = "unknown"
        if is_greeting(safe_request.message):
            answer = "你好！我是 FitZen 健身助手。你可以问我训练、饮食记录、营养分析或恢复问题。"
            await self._persist_turn(safe_request, answer)
            return ChatResponse(answer=answer, intent="training_qa", citations=[], safety_level=self._merge_safety(safety_level, guard.level), model_used=False)
        memory = _request_memory.get()
        if memory is None:
            memory = await memory_service.load(safe_request.user_id, safe_request.conversation_id, safe_request.message)
            if current_turn.get(): _request_memory.set(memory)
        requested_mode = getattr(safe_request, "mode", None) or "legacy"
        declared_mode = requested_mode
        if requested_mode == "react_write":
            reason = self._write_gate_reason()
            if reason is not None:
                return self._write_unavailable_response(safe_request, guard, safety_level, reason)
        from app.services.agent import record_clarifications as clarifications
        from app.services.llm.budget import ModelBudgetError
        from app.services.llm.client import LLMErrorCategory, LLMGatewayError
        pending_record_snapshot = None
        if requested_mode in {"auto", "react_readonly", "react_write"}:
            try:
                pending_record_snapshot = await asyncio.to_thread(
                    clarifications.load_pending, safe_request)
            except Exception:
                logger.warning("record_clarification_unavailable")
                return clarifications.unavailable_response()
            if pending_record_snapshot:
                reason = self._write_gate_reason()
                if reason is not None:
                    return self._write_unavailable_response(
                        safe_request, guard, safety_level, reason)
            if pending_record_snapshot:
                # A draft reply has its own typed, source-bound intent AND
                # current safety contract. Asking the general task extractor
                # for fresh workout facts from "yes" creates foreign quotes.
                try:
                    adopted = await clarifications.continue_request(
                        self.graph.llm, safe_request, pending=pending_record_snapshot)
                except ModelBudgetError as exc:
                    return self._semantic_limit_response(safe_request, safety_level, exc)
                except LLMGatewayError as exc:
                    if exc.category == LLMErrorCategory.POLICY:
                        return self._semantic_limit_response(safe_request, safety_level, exc)
                    raise
                except clarifications.DraftError:
                    logger.warning("record_clarification_unavailable")
                    return clarifications.unavailable_response()
                pending_record_snapshot = None
                if adopted and adopted.get("safety_status") == "unavailable":
                    return self._safety_unavailable_response(safe_request, safety_level, memory)
                if adopted and "answer" in adopted:
                    return ChatResponse(
                        answer=adopted["answer"], intent=adopted.get("intent", "clarification"),
                        model_used=_chat_model_used(),
                        safety_level=adopted.get("safety_level", "unknown"),
                        clarifications=adopted.get("clarifications", []),
                        metrics={"requested_mode": declared_mode,
                                 "execution_mode": "react_write",
                                 "mode_reason": "record_continuation",
                                 "safety_status": adopted.get("safety_status", "assessed")})
                if adopted:
                    # _transition already froze a command from the original
                    # field sources and the validated reply. The raw compact
                    # user message stays unchanged, including on C1 resume.
                    safe_request = safe_request.model_copy(update={
                        "message": adopted["source"], "mode": "react_write"})
                    parsed_input = adopted["parsed_input"]
                    requested_mode = request.mode = "react_write"
                    if isinstance(memory, dict) and adopted.get("reply_safety_level") in {"normal", "medical"}:
                        memory["_record_reply_safety_level"] = adopted["reply_safety_level"]
                if isinstance(memory, dict):
                    memory.pop("pending_record_draft", None)
        from app.services.agent import planning_questions
        planning_checked = False
        if (declared_mode in {"auto", "react_readonly"} and not parsed_input
                and not (_resumed_frozen_mode.get() == "react_readonly"
                         and not settings.react_readonly_enabled)):
            # A bounded current supplement has a goal only through its live
            # server-side question. Bind that source before asking the general
            # task extractor whether the reply alone is an independent goal.
            planning_checked = True
            try:
                adopted = await planning_questions.continue_request(self.graph.llm, safe_request)
            except ModelBudgetError as exc:
                return self._semantic_limit_response(safe_request, safety_level, exc)
            except LLMGatewayError as exc:
                if exc.category == LLMErrorCategory.POLICY:
                    return self._semantic_limit_response(safe_request, safety_level, exc)
                raise
            except planning_questions.ContinuationError:
                return ChatResponse(answer="待补充的计划已变化或不可用，请重新说明训练安排；已保存的记录不会重复执行。", intent="clarification")
            if adopted and "answer" in adopted:
                return ChatResponse(
                    answer=adopted["answer"], intent=adopted.get("intent", "clarification"),
                    model_used=_chat_model_used(), safety_level=adopted.get("safety_level", "unknown"),
                    clarifications=adopted.get("clarifications", []),
                    metrics={"requested_mode": declared_mode, "execution_mode": "react_readonly",
                             "mode_reason": "planning_continuation",
                             "safety_status": adopted.get("safety_status", "unavailable")})
            if adopted:
                safe_request = safe_request.model_copy(update={"message": adopted["source"]})
                parsed_input = adopted["parsed_input"]
                memory = {**memory, "information_sources": adopted.get("information_sources", []),
                          "_planning_reply_safety_level": adopted.get("reply_safety_level", "unknown")}
                _request_memory.set(memory)
        if isinstance(memory, dict):
            # Bounded record_query correction applies to the legacy live path;
            # ReactReadonly may satisfy the goal through an observed read tool.
            memory["_requested_mode"] = requested_mode
            # A previous preview is application-owned state, not Wiki history.
            # Load it before semantics so a preview_ref can be validated and a
            # required-history target does not buy a Wiki read.
            try:
                from app.services.agent.plan_preview import recent_plan_previews
                loaded = await asyncio.to_thread(
                    recent_plan_previews, safe_request.user_id, safe_request.conversation_id)
                if loaded.get("status") == "ok" and loaded.get("states"):
                    memory["current_plan_preview"] = {
                        "status": "ok", "state": loaded["states"][0],
                        "available": loaded["states"],
                    }
                else:
                    memory["current_plan_preview"] = {"status": loaded.get("status") or "unavailable"}
            except Exception:
                memory["current_plan_preview"] = {"status": "unavailable"}
        # Eligibility was checked before loading record continuation context.
        if requested_mode == "react_readonly" and _resumed_frozen_mode.get() == "react_readonly" and not settings.react_readonly_enabled:
            answer = ("该请求此前已按只读（react_readonly）模式冻结，但功能开关当前关闭，无法继续执行；"
                      "未切换到 legacy，也未运行新的模型或写操作。")
            await self._persist_turn(safe_request, answer)
            return ChatResponse(answer=answer, intent="execution_limited", citations=[],
                                safety_level=self._merge_safety(safety_level, guard.level), model_used=False,
                                metrics={"requested_mode": "react_readonly", "execution_mode": "limited",
                                         "mode_reason": "react_readonly_disabled_frozen", "react_status": "limited"})
        # Emergency/medical meaning comes from the same request-level structured
        # understanding as tasks. The technical guard only redacts identifiers.
        # This runs after draft continuation so the safety judgment is bound to
        # the exact source that will execute.
        # Emergency safety must precede every mode gate, including react_write.
        from app.services.agent.semantics import safety_for
        limited = await self._understand_or_limit(safe_request, memory, safety_level)
        if limited is not None:
            return limited
        safety = safety_for(safe_request.message)
        safety_level = str(safety.get("level") or "unknown") if safety.get("status") == "assessed" else "unknown"
        if memory.get("_planning_reply_safety_level") == "medical":
            safety_level = self._merge_safety(safety_level, "medical")
        if safety_level == "urgent":
            return ChatResponse(
                answer="检测到可能存在需要及时处理的危险信号。请立即停止训练；如有胸痛、呼吸困难、意识异常、明显变形或持续加重，请尽快联系急救或到急诊就医。",
                intent="symptom_triage", citations=[], safety_level="urgent",
                model_used=_chat_model_used(), metrics={"safety_status": "assessed"},
            )
        if safety.get("status") != "assessed":
            return self._safety_unavailable_response(safe_request, safety_level, memory)
        # Preserve the ordinary auto/direct-write route. Only the source-bound
        # record continuation must run before selecting its execution mode.
        from app.services.agent.semantics import get_understanding
        from app.services.agent.task_models import is_record_write
        understanding = get_understanding(safe_request.message) or {}
        record_flow = pending_record_snapshot or clarifications.needs_record_draft(
            understanding)
        has_workout_correction = any(
            task.get("workout_correction") and is_record_write(task)
            for task in understanding.get("validated_tasks") or [])
        if (requested_mode == "react_readonly" or has_workout_correction) and any(
                task.get("task_type") == "workout_record" and is_record_write(task)
                for task in understanding.get("validated_tasks") or []):
            # The product's readonly default used to send ordinary records to
            # legacy while clarification records used ReAct. The validated
            # workout write now has one writer; its own gates still decide
            # authorization, source validity, budget and storage readiness.
            requested_mode = request.mode = "react_write"
            safe_request = safe_request.model_copy(update={"mode": "react_write"})
        if requested_mode == "auto" and not record_flow:
            from app.services.agent.react_runtime import resolve_auto_mode
            resolved, auto_reason = resolve_auto_mode(
                self, safe_request, get_understanding(safe_request.message) or {})
            requested_mode = request.mode = resolved
            safe_request = safe_request.model_copy(update={"mode": resolved})
            if isinstance(memory, dict):
                memory["auto_mode_reason"] = auto_reason
        if requested_mode == "react_write" and not record_flow:
            return await self._respond_write(
                safe_request, guard, safety_level,
                mode_reason="auto_write" if declared_mode == "auto" else "explicit_write",
                declared_mode=declared_mode if declared_mode != "react_write" else None)
        try:
            continuation = (await clarifications.continue_request(
                self.graph.llm, safe_request, pending=pending_record_snapshot)
                if pending_record_snapshot or requested_mode == "legacy" else None)
        except clarifications.DraftError:
            logger.warning("record_clarification_unavailable")
            return clarifications.unavailable_response()
        record_continuation = bool(continuation)
        from app.services.agent import planning_questions
        if not continuation and requested_mode != "react_write" and not planning_checked:
            try:
                continuation = await planning_questions.continue_request(self.graph.llm, safe_request)
            except planning_questions.ContinuationError:
                return ChatResponse(answer="待补充的计划已变化或不可用，请重新说明训练安排；已保存的记录不会重复执行。", intent="clarification")
        if continuation:
            if "answer" in continuation:
                return ChatResponse(answer=continuation["answer"], intent=continuation.get("intent", "clarification"), model_used=True,
                                    clarifications=continuation.get("clarifications", []),
                                    metrics=({"requested_mode": declared_mode,
                                              "execution_mode": "react_write",
                                              "mode_reason": "record_continuation"}
                                             if declared_mode != "legacy" else {}))
            # This command was frozen and adopted through the server-side draft
            # transaction. Raw user history remains unchanged in MemoryTurn.
            safe_request = safe_request.model_copy(update={"message": continuation["source"]})
            parsed_input = continuation["parsed_input"]
            memory = {**memory, "information_sources": continuation.get("information_sources", [])}
            memory.pop("pending_record_draft", None)
            _request_memory.set(memory)
        from app.services.agent.semantics import safety_for
        limited = await self._understand_or_limit(safe_request, memory, safety_level)
        if limited is not None:
            return limited
        safety = safety_for(safe_request.message)
        safety_level = str(safety.get("level") or "unknown") if safety.get("status") == "assessed" else "unknown"
        if memory.get("_planning_reply_safety_level") == "medical":
            safety_level = self._merge_safety(safety_level, "medical")
        if safety_level == "urgent":
            return ChatResponse(
                answer="检测到可能存在需要及时处理的危险信号。请立即停止训练；如有胸痛、呼吸困难、意识异常、明显变形或持续加重，请尽快联系急救或到急诊就医。",
                intent="symptom_triage",
                citations=[],
                safety_level="urgent",
                model_used=_chat_model_used(),
                metrics={"safety_status": "assessed"},
            )
        if safety.get("status") != "assessed":
            return self._safety_unavailable_response(safe_request, safety_level, memory)
        from app.services.agent.semantics import get_understanding
        understanding = get_understanding(safe_request.message) or {}
        if declared_mode != "legacy" and (
                record_continuation or clarifications.needs_record_draft(understanding)):
            requested_mode = request.mode = "react_write"
            safe_request = safe_request.model_copy(update={"mode": "react_write"})
        if requested_mode == "auto":
            from app.services.agent.react_runtime import resolve_auto_mode
            resolved, auto_reason = resolve_auto_mode(self, safe_request, understanding)
            requested_mode = request.mode = resolved
            safe_request = safe_request.model_copy(update={"mode": resolved})
            if isinstance(memory, dict):
                memory["auto_mode_reason"] = auto_reason
        if requested_mode == "react_write":
            return await self._respond_write(
                safe_request, guard, safety_level,
                mode_reason="auto_write" if declared_mode == "auto" else "explicit_write",
                declared_mode="auto" if declared_mode == "auto" else None)
        pre_mode = pre_reason = None
        prechecked = None
        semantic_deferred = False
        if requested_mode == "react_readonly" and parsed_input is None and not continuation:
            # A genuinely new readonly request buys the shared understanding once
            # inside its existing request budget. A validated write/confirm turns
            # the whole request legacy before any readonly execution; a pure read
            # reuses the same validated tasks without the legacy parse path.
            from app.services.agent.react_runtime import prepare_readonly
            preparation = await prepare_readonly(self, safe_request, memory)
            if preparation.mode == "limited":
                from app.services.agent.semantics import understanding_unavailable_reply
                answer = ("当前请求的模型预算已用完，未能完成语义判断；本次未保存任何数据。请稍后重试。"
                          if preparation.reason == "semantic_budget_stopped"
                          else understanding_unavailable_reply())
                return ChatResponse(
                    answer=answer,
                    intent="execution_limited",
                    citations=[],
                    safety_level=self._merge_safety(safety_level, guard.level),
                    model_used=False,
                    metrics={"requested_mode": requested_mode,
                             "execution_mode": "limited",
                             "mode_reason": preparation.reason,
                             "react_status": "limited"},
                )
            pre_mode, pre_reason = preparation.mode, preparation.reason
            semantic_deferred = preparation.mode == "defer"
            if semantic_deferred:
                # The configured understanding failed. The real legacy parser
                # owns the explicit unavailable decision; if it returns a
                # validated write plan the request still runs legacy. A readonly
                # fallback is never reported as pure_read.
                pre_mode = pre_reason = None
            elif preparation.mode == "react_readonly":
                prechecked = PlannerDecision(
                    intent=preparation.intent,
                    parsed_input={},
                    task_specs=tuple(preparation.tasks),
                    needs_clarification=preparation.needs_clarification,
                    degraded_reason=None,
                )
        from app.services.agent import dialogue_context
        from app.services.agent.semantics import get_understanding
        from app.services.agent.task_models import (
            is_capability_task, is_plan_read_task, is_plan_save_capability_task,
            is_preview_task, read_spec_of, task_scope_value,
        )
        semantics = get_understanding(safe_request.message) or {}
        preview_tasks = [task for task in semantics.get("tasks") or []
                         if isinstance(task, dict) and is_preview_task(task)]
        capability_plan_tasks = [
            task for task in semantics.get("tasks") or []
            if isinstance(task, dict) and is_plan_save_capability_task(task)
        ]
        if capability_plan_tasks and all(
                is_plan_save_capability_task(task) or is_capability_task(task)
                for task in semantics.get("tasks") or [] if isinstance(task, dict)):
            # The chat has no plan-write tool. A save/confirm request must be
            # answered from the application capability contract before any
            # planner, Wiki read or prepare budget is spent. Only a request
            # made entirely of capability goals takes this shortcut; a mixed
            # request keeps its independent record/read goals visible.
            from app.services.agent.capability_catalog import capability_copy
            answer = capability_copy("plan_save")
            await self._persist_turn(safe_request, answer)
            return ChatResponse(
                answer=answer, intent="conversation", citations=[],
                safety_level=self._merge_safety(safety_level, guard.level),
                model_used=False,
                metrics={"requested_mode": requested_mode,
                         "execution_mode": "legacy",
                         "mode_reason": "capability_catalog",
                         "capability_kind": "plan_save",
                         "execution_status": "not_executed",
                         "task_outcomes": []},
            )
        modification_tasks = [
            task for task in preview_tasks
            if not (task.get("plan_result") is True and not task.get("preview_ref"))
        ]
        if modification_tasks:
            preview = memory.get("current_plan_preview") or {"status": "unavailable"}
            preview_status = str(preview.get("status") or "unavailable")
            if preview_status in {"none", "unavailable"}:
                from app.services.agent.plan_preview import (
                    NO_PREVIEW_COPY, UNAVAILABLE_PREVIEW_COPY,
                )
                from app.services.agent.capability_catalog import capability_copy
                answer = (NO_PREVIEW_COPY if preview_status == "none"
                          else UNAVAILABLE_PREVIEW_COPY)
                if any(task_scope_value(task, "delivery_type") == "capability"
                       for task in semantics.get("tasks") or [] if isinstance(task, dict)):
                    answer += "\n\n" + capability_copy("plan_save")
                await self._persist_turn(safe_request, answer)
                return ChatResponse(
                    answer=answer, intent="clarification", citations=[],
                    safety_level=self._merge_safety(safety_level, guard.level),
                    model_used=False,
                    metrics={"requested_mode": requested_mode,
                             "execution_mode": "legacy",
                             "mode_reason": "preview_state_" + preview_status,
                             "plan_preview_status": preview_status,
                             "task_outcomes": []},
                )
        decision = prechecked if prechecked is not None else await self._route(safe_request, memory, parsed_input)
        if prechecked is None:
            # The legacy parser may have replaced the router's provisional
            # understanding with its own validated source-bound plan (for
            # example a record-domain conversation answer). Downstream
            # dialogue/worker selection must see that updated decision.
            from app.services.agent.semantics import get_understanding
            semantics = get_understanding(safe_request.message) or semantics
        from app.services.agent.react_graph import MAX_TASKS
        if len(decision.task_specs or ()) > MAX_TASKS:
            return ChatResponse(
                answer=f"本次请求包含的任务过多（最多 {MAX_TASKS} 个），请拆分为多次请求；本次未保存任何数据。",
                intent="execution_limited", citations=[],
                safety_level=self._merge_safety(safety_level, guard.level), model_used=False,
                metrics={"requested_mode": requested_mode, "execution_mode": "limited",
                         "mode_reason": "task_count_limit", "react_status": "limited"},
            )
        if _planner_unavailable_for_write(decision.degraded_reason, semantics):
            # The parser/extractor failed after the user's write intent was
            # already established. Falling back to a read-only rule task made
            # the app say "无法确认记录意图", which misattributes a service
            # failure to the user. Stop explicitly with zero write instead.
            answer = "AI记录解析暂时不可用，本次未保存。请重试，或使用手动表单添加记录。"
            return ChatResponse(
                answer=answer, intent="execution_limited", citations=[],
                safety_level=self._merge_safety(safety_level, guard.level), model_used=False,
                metrics={"requested_mode": requested_mode, "execution_mode": "limited",
                         "mode_reason": "planner_unavailable_for_write",
                         "react_status": "limited"},
            )
        try:
            memory = await dialogue_context.restore(safe_request, memory, semantics)
        except dialogue_context.DialogueContextUnavailable:
            preview_state = memory.get("current_plan_preview") or {}
            if preview_tasks and isinstance(preview_state, dict) \
                    and preview_state.get("status") == "ok":
                # A preview edit is sourced from the application-owned preview
                # state, not Wiki history. A failed history restore must not
                # turn it into "the quoted conversation is unavailable".
                logger.info("preview_edit_history_unavailable")
            else:
                return ChatResponse(answer="刚才引用的对话已不可用，请补充你指的是哪一项；本次未新增记录。",
                                    intent="clarification", model_used=False)
        if current_turn.get():
            _request_memory.set(memory)
        if decision.needs_clarification:
            from app.services.agent.semantics import get_understanding, understanding_unavailable_reply
            semantics = get_understanding(safe_request.message) or {}
            question = semantics.get("clarification")
            if semantics.get("needs_clarification"):
                try:
                    question = await clarifications.create_question(self.graph.llm, safe_request, semantics) or question
                except clarifications.DraftError:
                    logger.warning("record_clarification_proposal_invalid")
                    return clarifications.unavailable_response()
            answer = question or understanding_unavailable_reply()
            if question and semantics.get("needs_clarification"):
                return await clarifications.question_response(safe_request, answer)
            from app.services.agent.task_information import question as information_question
            clarification = information_question("request", [], category="ambiguous_reference" if semantics else "service_unavailable", text=answer)
            limited_metrics = ({"requested_mode": requested_mode, "execution_mode": "limited",
                                "mode_reason": "semantic_understanding_unavailable",
                                "react_status": "limited"} if semantic_deferred else {})
            return ChatResponse(answer=answer, intent="clarification", model_used=bool(semantics),
                                clarifications=[clarification], metrics=limited_metrics)
        if requested_mode == "legacy" and clarifications.missing_record_entities(decision.parsed_input):
            try:
                question = await clarifications.create_question(self.graph.llm, safe_request, semantics)
            except clarifications.DraftError:
                return clarifications.unavailable_response()
            if question:
                return await clarifications.question_response(safe_request, question)
        if requested_mode == "legacy" and not preview_tasks:
            conversation = await dialogue_context.conversation_answer(self.graph.llm, safe_request, memory, semantics)
            if conversation:
                guarded = apply_output_guardrail(conversation["answer"])
                await self._persist_turn(safe_request, guarded.text)
                conversation_metrics = {}
                if conversation.get("memory_update_requested"):
                    conversation_metrics["memory_update_requested"] = True
                if conversation.get("state_update_requested"):
                    conversation_metrics["state_update_requested"] = True
                return ChatResponse(answer=guarded.text, intent="conversation", safety_level=self._merge_safety(safety_level, guard.level),
                                    model_used=conversation["model_used"], metrics=conversation_metrics)
        intent_result = decision.intent
        parsed_input = decision.parsed_input
        if pre_mode is None:
            # 阶段B模式选择：react_readonly + 开关开启 + 纯读 → 只读循环（绕过 legacy planner/稳定计划）。
            from app.services.agent.react_runtime import select_mode
            effective_mode, mode_reason = select_mode(safe_request, decision.task_specs)
            if semantic_deferred and effective_mode == "react_readonly":
                # A failed configured understanding must never be reported as a
                # successful pure read, even if a fallback plan looks read-only.
                from app.services.agent.semantics import understanding_unavailable_reply
                return ChatResponse(
                    answer=understanding_unavailable_reply(),
                    intent="execution_limited",
                    citations=[],
                    safety_level=self._merge_safety(safety_level, guard.level),
                    model_used=False,
                    metrics={"requested_mode": requested_mode,
                             "execution_mode": "limited",
                             "mode_reason": "semantic_understanding_unavailable",
                             "react_status": "limited"},
                )
        else:
            effective_mode, mode_reason = pre_mode, pre_reason
        if effective_mode == "legacy" and (
                mode_reason in {"read_spec_legacy", "plan_read_legacy"}
                or any(read_spec_of(task) or is_plan_read_task(task)
                       for task in semantics.get("tasks") or [])):
            conversation = await dialogue_context.conversation_answer(
                self.graph.llm, safe_request, memory, semantics)
            if conversation:
                guarded = apply_output_guardrail(conversation["answer"])
                await self._persist_turn(safe_request, guarded.text)
                return ChatResponse(
                    answer=guarded.text, intent="conversation",
                    safety_level=self._merge_safety(safety_level, guard.level),
                    model_used=conversation["model_used"],
                    metrics={"requested_mode": requested_mode,
                             "execution_mode": "legacy", "mode_reason": mode_reason})
        mode_metrics = {"requested_mode": requested_mode,
                        "execution_mode": effective_mode, "mode_reason": mode_reason}
        if declared_mode == "auto":
            mode_metrics["declared_mode"] = "auto"
        if effective_mode == "react_readonly":
            totals = _single_nutrition_totals_query(decision.task_specs)
            if totals is not None:
                task, query = totals
                from app.services.agent.dialogue_context import saved_record_answer
                result = await saved_record_answer(
                    None, safe_request, "nutrition_record", query, select_all=True)
                answer = apply_output_guardrail(str(result.get("answer") or ""))
                await self._persist_turn(safe_request, answer.text)
                return ChatResponse(
                    answer=answer.text,
                    intent="nutrition_record",
                    citations=[],
                    safety_level=self._merge_safety(safety_level, guard.level),
                    model_used=False,
                    metrics={**mode_metrics,
                             "react_status": "deterministic_nutrition_totals",
                             "task_outcomes": [{
                                 "task_id": str(task.get("id") or ""),
                                 "task_type": "nutrition_record",
                                 "status": "complete" if result.get("status") != "unavailable" else "unavailable",
                             }]},
                )
            return await self._respond_readonly(safe_request, guard, decision, mode_metrics, safety_level)
        try:
            planned_tasks = await self._stable_plan(safe_request, decision.task_specs)
        except OperationStoreUnavailable:
            # A write cannot be made retry-safe without the ledger. Refuse the
            # whole request before any save tool runs instead of double-writing
            # on the next attempt.
            mark_degraded("operation_ledger_unavailable", failure=True)
            answer = "操作账本暂时不可用；为避免重复或遗漏保存，本次未执行任何记录。请稍后重试。"
            await self._persist_turn(safe_request, answer)
            return ChatResponse(answer=answer, intent="error", citations=[],
                                safety_level=self._merge_safety(safety_level, guard.level), model_used=False)
        except InvalidWorkoutCandidate:
            # R011-H: an explicit input rejection, not a storage outage. The bad
            # candidate never reaches the plan ledger or a save dispatch.
            answer = ("训练数据包含无效数值，无法保存。请补充有效的动作、重量（kg）、组数和次数后重试；"
                      "本次未保存。")
            await self._persist_turn(safe_request, answer)
            return ChatResponse(answer=answer, intent="clarification", citations=[],
                                safety_level=self._merge_safety(safety_level, guard.level), model_used=False)
        emit("plan", {"tasks": [{k: t.get(k) for k in ("id", "task_type", "depends_on")} for t in planned_tasks]})
        with trace_span("fitness.chat", inputs={"message_chars": len(safe_request.message)},
                        metadata={"conversation": self._trace_identity(safe_request.conversation_id)},
                        tags=["sync"]) as span:
            # LangGraph state includes private profile/history, so only the
            # privacy-safe parent span is exported to LangSmith.
            with tracing_context(enabled=False):
                try:
                    result = await self.graph.run(safe_request.user_id, safe_request.conversation_id,
                                                  safe_request.message, memory, asdict(intent_result), parsed_input or {},
                                                  safe_request.request_id, planned_tasks=planned_tasks)
                    for item in result.get("worker_results", []):
                        status = item.get("status", "complete")
                        if status in {"complete", "completed", "needs_confirmation", "needs_clarification"}:
                            continue
                        if status == "rejected_unauthorized":
                            # A correct safety refusal is a normal, explicable
                            # outcome; count it, never mark the service degraded
                            # or open an automatic failure case.
                            mark_safety_refusal(status)
                            continue
                        mark_degraded("task_" + str(status), failure=True)
                    answer = self._record_answer(result)
                    citations = result.get("citations", [])
                    model_used = bool(result.get("model_used", False))
                    if self._is_compound_nutrition_request(safe_request.message, intent_result, planned_tasks) and not self._has_planned_tasks(result):
                        worker_result = self._last_worker_result(result)
                        if worker_result.get("status") != "complete":
                            final_answer = answer
                        else:
                            advice_task = next(
                                (task for task in planned_tasks
                                 if isinstance(task, dict)
                                 and str(task.get("task_type") or "") == "nutrition_advice"),
                                None)
                            advice, advice_model_used, advice_citations = await self._complete_nutrition_advice(
                                safe_request, memory, result, task=advice_task)
                            final_answer = "\n\n".join(part for part in (answer, advice) if part)
                            citations = normalize_citations([*citations, *advice_citations])
                            model_used = model_used or advice_model_used
                    else:
                        final_answer = answer
                    if capability_plan_tasks:
                        from app.services.agent.capability_catalog import capability_copy
                        boundary = capability_copy("plan_save")
                        if boundary not in final_answer:
                            final_answer = f"{final_answer}\n\n{boundary}"
                except ContextBudgetExceeded:
                    mark_degraded("context_budget", failure=True)
                    answer = "当前问题和必要的安全信息超过了上下文预算，请缩短本次输入后再试。"
                    await self._persist_turn(safe_request, answer)
                    span.end({"answer_chars": len(answer), "intent": "context_budget", "model_used": False})
                    return ChatResponse(answer=answer, intent="context_budget", citations=[],
                                        safety_level=self._merge_safety(safety_level, guard.level), model_used=False)
                except ToolSchemaError:
                    # Schema rejection precedes this tool's dispatch. Earlier
                    # independent tasks may already have committed, so never
                    # claim that the entire request performed zero writes.
                    mark_degraded("tool_schema_rejected", failure=True)
                    return ChatResponse(answer="这次处理遇到工具参数错误，未能完成全部任务。已保存的记录会保留，请先核对记录后再重试。",
                                        intent="error", safety_level=self._merge_safety(safety_level, guard.level), model_used=False,
                                        metrics={**mode_metrics, "error_code": "tool_schema_rejected"})
            guarded = apply_output_guardrail(final_answer)
            # Only pending read tasks are retained. Completed writes keep their
            # original execution ledger and are never adopted by this workflow.
            try:
                await planning_questions.remember(safe_request, result, planned_tasks, memory.get("information_sources", []))
            except Exception:
                logger.warning("planning_question_persistence_failed")
                mark_degraded("planning_question_persistence", failure=True)
                if result.get("clarifications"):
                    guarded = apply_output_guardrail(guarded.text + "\n待补充状态暂未保存，请稍后重新发送计划问题；已完成的记录仍保留。")
            response = ChatResponse(
                answer=guarded.text,
                intent=self._response_intent(result),
                citations=citations,
                safety_level=self._merge_safety(safety_level, self._merge_safety(guard.level, result.get("safety_level", "normal"))),
                model_used=model_used,
                metrics={**mode_metrics, "task_outcomes": result.get("task_outcomes", []),
                         **({"plan_preview": result["plan_preview"]} if result.get("plan_preview") else {}),
                         **({"semantic_failure_code": memory.get("semantic_failure_code")}
                            if memory.get("semantic_failure_code") else {})},
                clarifications=result.get("clarifications", []),
            )
            await self._persist_turn(safe_request, response.answer)
            span.end({"answer_chars": len(response.answer), "intent": response.intent,
                      "model_used": response.model_used, "memory_backend": memory["memory_backend"]})
            return response


