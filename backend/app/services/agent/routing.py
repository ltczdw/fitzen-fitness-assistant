from __future__ import annotations


import logging
import asyncio
from datetime import timedelta
from app.core import clock
from app.core.config import settings




from app.services.agent.intent import classify_intent_rule
from app.services.agent.planner_shadow import find_differences, request_fingerprint, rule_plan
from app.services.agent.task_planner import intent_from_planned_tasks, plan_tasks_from_llm, plan_tasks_rule
from app.services.agent.planner_validation import validate_planner_output, validate_rule_plan
from app.services.agent.food_calculator import get_pending_food
from app.services.security.input_guard import inspect_input


from app.services.agent.execution_state import _request_route, PlannerDecision
from app.services.telemetry import mark_degraded
logger = logging.getLogger(__name__)

class PlannerRoutingMixin:
    async def _run_planner_shadow(self, message: str) -> None:
        if not settings.planner_shadow_enabled:
            return
        safe_message = inspect_input(message).sanitized_text
        try:
            rule_future = asyncio.to_thread(rule_plan, safe_message)
            llm_future = self.graph.plan_shadow(safe_message)
            rule_result, llm_result = await asyncio.gather(rule_future, llm_future, return_exceptions=True)
            if isinstance(rule_result, BaseException):
                logger.exception("planner_shadow_rule_failed", exc_info=rule_result)
                return
            if isinstance(llm_result, BaseException):
                logger.exception("planner_shadow_call_failed", exc_info=llm_result)
                shadow_tasks, status = None, "llm_error"
            else:
                shadow_tasks, status = llm_result
            # Failed/unavailable Planner calls are status counts, never empty-plan diffs.
            differences = find_differences(safe_message, rule_result, shadow_tasks) if status == "ok" else {}
            logger.info("planner_shadow_comparison", extra={
                "planner_shadow": {
                    "query_hash": request_fingerprint(safe_message),
                    "llm_status": status,
                    "rule_task_count": len(rule_result),
                    "llm_task_count": len(shadow_tasks or []),
                    "difference_counts": {key: len(value) for key, value in differences.items()},
                }
            })
        except Exception:
            # Shadow evaluation must never change the response path.
            logger.exception("planner_shadow_unavailable")


    async def _parse_input(self, message: str) -> dict | None:
        parser = getattr(self.graph, "parse_input", None)
        if parser is None:
            return None
        try:
            return await parser(message)
        except Exception:
            return None


    async def _route(self, request, memory, parsed_input_override=None):
        cached = _request_route.get()
        if cached is not None:
            return cached
        safety_fallback = classify_intent_rule(request.message)
        rule_tasks = plan_tasks_rule(request.message, safety_fallback)
        pending_food = get_pending_food(request.user_id, request.conversation_id)
        pending_record = None
        if pending_food is not None:
            # get_pending_food has already checked the real monotonic TTL and the
            # exact user/conversation key. The graph worker checks it again before
            # writing, so this short binding window cannot authorize a stale save.
            pending_record = {
                "record_id": pending_food.idempotency_key or "pending-food",
                "task_type": "nutrition_record",
                "session_id": request.conversation_id,
                "expires_at": (clock.utc_now() + timedelta(seconds=30)).isoformat(),
            }

        parsed_input = parsed_input_override
        if parsed_input is None:
            parsed_input = await self._parse_input(request.message)
        semantic_unavailable = False
        if parsed_input is None:
            from app.services.agent.semantics import semantic_attempt_failed
            # A configured semantic call that actually failed must stop with an
            # explicit unavailable reply. An offline/unconfigured model is a
            # supported rule-plan mode, so its validated tasks stay runnable.
            semantic_unavailable = semantic_attempt_failed(request.message)
            if semantic_unavailable:
                rule_tasks = plan_tasks_rule(request.message, safety_fallback)
            validation = validate_rule_plan(
                request.message, rule_tasks, pending_record=pending_record,
                current_session_id=request.conversation_id,
            )
            planned_tasks = validation.tasks
            parsed_input = {}
            intent = safety_fallback
            degraded_reason = "planner_llm_unavailable_or_invalid"
            self._log_planner_validation(validation, fallback_required=True, reason=degraded_reason)
        else:
            validation = validate_planner_output(
                request.message, parsed_input, rule_tasks=rule_tasks,
                pending_record=pending_record,
                current_session_id=request.conversation_id,
            )
            if validation.fallback_required:
                planned_tasks = validation.tasks
                parsed_input = {}
                degraded_reason = validation.degraded_reason or "planner_validation_fallback"
                intent = intent_from_planned_tasks(planned_tasks, safety_fallback)
            else:
                # Keep only validated source spans and the parser's structured
                # entities. Planner metadata is not forwarded to workers. The
                # marker lets plan_tasks_from_llm trust these gate-computed
                # authorizations; raw model output never carries it.
                parsed_input = {"authorization_validated": True, "tasks": [
                    {key: item[key] for key in (
                        "task_type", "action", "text", "is_negated", "is_hypothetical",
                        "is_third_party", "foods", "workouts", "write_window",
                        "write_authorization",
                        "source_start", "source_end", "merged_task_ids",
                    ) if key in item}
                    for item in validation.tasks
                ]}
                planned_tasks = plan_tasks_from_llm(request.message, parsed_input)
                if planned_tasks is None:
                    validation = validate_rule_plan(
                        request.message, rule_tasks, pending_record=pending_record,
                        current_session_id=request.conversation_id,
                    )
                    planned_tasks = validation.tasks
                    parsed_input = {}
                    degraded_reason = "planner_task_mapping_failed"
                    intent = safety_fallback
                else:
                    intent = intent_from_planned_tasks(planned_tasks, safety_fallback)
                    degraded_reason = None
            self._log_planner_validation(
                validation,
                fallback_required=bool(validation.fallback_required or degraded_reason),
                reason=degraded_reason,
            )

        from app.services.agent.semantics import get_understanding
        semantics = get_understanding(request.message)
        semantic_question = bool(semantics and semantics.get("needs_clarification"))
        decision = PlannerDecision(
            intent=intent,
            parsed_input=parsed_input or {},
            task_specs=tuple(planned_tasks),
            needs_clarification=semantic_unavailable or semantic_question,
            degraded_reason=degraded_reason,
        )
        if degraded_reason:
            mark_degraded(degraded_reason)
        _request_route.set(decision)
        return decision


    @staticmethod
    def _log_planner_validation(validation, *, fallback_required: bool, reason: str | None = None) -> None:
        event_counts: dict[str, int] = {}
        for item in validation.trace:
            code = item.get("code")
            if code:
                event_counts[str(code)] = event_counts.get(str(code), 0) + 1
        logger.info("planner_validation", extra={"planner_validation": {
            "backend": validation.planner_backend,
            "coverage": validation.coverage,
            "fallback_required": fallback_required,
            "degraded_reason": reason or validation.degraded_reason,
            "event_counts": event_counts,
            "task_count": len(validation.tasks),
        }})


