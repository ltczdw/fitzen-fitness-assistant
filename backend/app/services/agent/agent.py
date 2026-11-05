from __future__ import annotations

from app.services.rag.blocking import run_retrieval

import json
import logging
import asyncio
from datetime import timedelta
from contextvars import ContextVar
from contextlib import asynccontextmanager
from app.core import clock
from app.core.config import settings
from app.services.memory.compact import current_turn, begin_turn, finish_turn, fail_turn, unfinished_frozen_mode
from app.services.rag.retrieval import embedding_cache, retrieval_traces
from app.services.rag.quality import assess_rag_evidence, insufficient_evidence_reply

import hashlib
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass

from langsmith import tracing_context

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.citation import citations_from_hits, normalize_citations

from app.services.agent.multi_agent_graph_fixed import FitnessMultiAgentGraph
from app.services.agent.intent import IntentResult, classify_intent_rule, is_greeting
from app.services.agent.planner_shadow import find_differences, request_fingerprint, rule_plan
from app.services.agent.task_planner import intent_from_planned_tasks, plan_tasks_from_llm, plan_tasks_rule
from app.services.agent.planner_validation import validate_planner_output, validate_rule_plan
from app.services.agent.food_calculator import get_pending_food
from app.services.agent.workout_extractor import preprocess_workout
from app.services.context.budget import ContextBudgetExceeded
from app.services.llm.client import LLMGatewayError
from app.services.memory.redis_memory import memory_service
from app.services.observability import trace_span
from app.services.security.input_guard import inspect_input
from app.services.security.output_guard import apply_output_guardrail


from app.services.agent.execution_state import (
    _memory_error, _request_route, _request_memory, _resumed_frozen_mode, PlannerDecision, RequestIdConflict,
    conversation_lock,
)
from app.services.agent.operation_store import OperationStore, current_operations
from app.services.telemetry import mark_degraded
from app.services.agent.events import emit
logger = logging.getLogger(__name__)

from app.services.agent.semantics import request_understanding, get_understanding
from app.services.agent.routing import PlannerRoutingMixin
from app.services.agent.execution import ChatExecutionMixin
from app.services.agent.runtime import execute_request
from app.services.agent.transport import stream_response

class FitnessAgent(PlannerRoutingMixin, ChatExecutionMixin):
    def __init__(self) -> None:
        self.graph = FitnessMultiAgentGraph()


    @staticmethod
    def _is_compound_nutrition_request(message: str, intent_result, planned_tasks=()) -> bool:
        if intent_result.name != "nutrition" or intent_result.action not in {"record", "confirm"}:
            return False
        task_types = [task.get("task_type") for task in planned_tasks]
        return "nutrition_record" in task_types and "nutrition_advice" in task_types


    @staticmethod
    def _last_worker_result(result: dict) -> dict:
        results = result.get("worker_results") or []
        return results[-1] if results else {}


    @staticmethod
    def _record_answer(result: dict) -> str:
        if result.get("final_answer"):
            return str(result["final_answer"])
        worker_result = FitnessAgent._last_worker_result(result)
        return str(worker_result.get("answer") or result.get("final_answer") or "")


    @staticmethod
    def _has_planned_tasks(result: dict) -> bool:
        return any(item.get("task_id") for item in result.get("worker_results", []))


    @staticmethod
    def _response_intent(result: dict) -> str:
        current = result.get("current_task", "training_qa")
        if isinstance(current, dict):
            return str(current.get("task_type") or current.get("worker") or "training_qa")
        return str(current or "training_qa")


    @asynccontextmanager
    async def _request_scope(self, request):
        if settings.memory_read_backend != "compact_v1":
            semantic_token = request_understanding.set({})
            route_token = _request_route.set(None)
            cache_token = embedding_cache.set({})
            traces_token = retrieval_traces.set([])
            memory_token = _request_memory.set(None)
            operations_token = current_operations.set(None)
            resumed_token = _resumed_frozen_mode.set(None)
            try:
                yield None
            finally:
                request_understanding.reset(semantic_token)
                _request_route.reset(route_token)
                embedding_cache.reset(cache_token)
                retrieval_traces.reset(traces_token)
                _request_memory.reset(memory_token)
                current_operations.reset(operations_token)
                _resumed_frozen_mode.reset(resumed_token)
            return
        lock = conversation_lock(asyncio.get_running_loop(), request.conversation_id)
        async with lock:
            resumed_mode = None
            try:
                # begin_turn 之前读取已有未完成请求的冻结模式；跨线程只回传普通值，
                # contextvar 由本主协程设置/复原。
                resumed_mode = await asyncio.to_thread(unfinished_frozen_mode, request)
                tid, cached = await asyncio.to_thread(begin_turn, request)
            except ValueError as exc:
                if str(exc) in {"request_id_payload_conflict", "request_was_deleted"}:
                    raise RequestIdConflict(str(exc)) from exc
                raise
            tokens = [current_turn.set(tid), _request_route.set(None), embedding_cache.set({}), retrieval_traces.set([]), _request_memory.set(None), current_operations.set(OperationStore(request.request_id)), _resumed_frozen_mode.set(resumed_mode), request_understanding.set({})]
            try:
                yield cached
            except BaseException:
                try:
                    await asyncio.shield(asyncio.to_thread(fail_turn, tid))
                except Exception:
                    logger.exception("memory_fail_turn_failed")
                raise
            finally:
                current_turn.reset(tokens[0]); _request_route.reset(tokens[1])
                embedding_cache.reset(tokens[2]); retrieval_traces.reset(tokens[3]); _request_memory.reset(tokens[4])
                current_operations.reset(tokens[5]); _resumed_frozen_mode.reset(tokens[6]); request_understanding.reset(tokens[7])


    @staticmethod
    def _merge_safety(input_level: str, graph_level: str) -> str:
        order = {"normal": 0, "sensitive": 1, "phi": 2, "caution": 2, "medical": 2, "urgent": 3}
        return input_level if order.get(input_level, 0) >= order.get(graph_level, 0) else graph_level


    async def _persist_turn(self, request: ChatRequest, answer: str) -> None:
        if not answer:
            return
        if settings.memory_read_backend == "compact_v1":
            # Compact memory has one atomic final commit in the runtime, which
            # stores the complete response together with completion. Persisting
            # an answer here would create a completed-but-unreplayable turn.
            return
        try:
            await memory_service.append_turn(request.user_id, request.conversation_id, request.message, answer)
            await memory_service.remember_explicit(request.user_id, request.message)
        except Exception:
            _memory_error.set("memory_persistence_failed")
            logger.exception("memory_persistence_failed", extra={"request_id": request.request_id})


    @staticmethod
    def _committable_semantics(values):
        """Only source-bound, assessed, non-urgent semantics may commit facts.

        The final memory commit is a business write. An urgent safety judgment
        (or a missing/foreign one) must preempt memory_requests even though the
        turn itself is still stored for replay. The check consumes the existing
        structured SafetyAssessment; it does not classify wording.
        """
        from app.services.security.safety import SafetyAssessment

        kept = []
        for source, value in values:
            if not isinstance(source, str) or not isinstance(value, dict):
                continue
            raw = value.get("safety")
            if not isinstance(raw, dict):
                continue
            try:
                assessment = SafetyAssessment.model_validate(raw)
            except Exception:
                continue
            if assessment.level == "urgent" or not assessment.bound_to(source):
                continue
            kept.append((source, value))
        return kept

    async def _finish_memory(self, tid, answer, response):
        try:
            from app.services.agent.semantics import request_understanding
            semantic_values = self._committable_semantics(
                list((request_understanding.get() or {}).items()))
            return await asyncio.to_thread(finish_turn, tid, answer, response, semantic_values=semantic_values)
        except Exception:
            _memory_error.set("memory_completion_failed")
            logger.exception("memory_completion_failed", extra={"turn_id": tid})


    @staticmethod
    def _trace_identity(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


    @staticmethod
    def _parse_workout(message: str) -> dict | None:
        return preprocess_workout(message).record


    async def respond(self, request: ChatRequest, parsed_input: dict | None = None) -> ChatResponse:
        return await execute_request(self, request, parsed_input)

    async def respond_stream(self, request: ChatRequest) -> AsyncIterator[dict[str, object]]:
        async for event in self._respond_stream_impl(request):
            if event["event"] in {"meta", "done"} and "citations" in event["data"]:
                event = {**event, "data": {**event["data"], "citations": normalize_citations(event["data"]["citations"])}}
            yield event

    async def _respond_stream_impl(self, request: ChatRequest):
        async for event in stream_response(self, request):
            yield event


# 模块级单例：JSON/SSE 路由与 ReAct 只读 worker/adapter 共用同一 FitnessAgent 实例，
# 使测试注入的 graph.llm/rag 与真实执行链引用一致（阶段B接口约定）。
agent = FitnessAgent()
