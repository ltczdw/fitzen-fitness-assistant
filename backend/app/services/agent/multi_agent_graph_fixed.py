from __future__ import annotations

from typing import Any

from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph as BaseFitnessMultiAgentGraph, _coerce_intent


def _tasks_for_message(message: str, intent_result=None) -> list[str]:
    # 使用主管图统一拆分，确保 PlannerLLM 的任务计划贯穿到任务路由。
    return _base_tasks(message, intent_result)


def _base_tasks(message: str, intent_result=None) -> list[str]:
    from app.services.agent.multi_agent_graph import _tasks_for_message as base_tasks

    return base_tasks(message, intent_result)


class FitnessMultiAgentGraph(BaseFitnessMultiAgentGraph):
    """Compatibility wrapper adding explicit nutrition-planning language routing."""

    @staticmethod
    def _preprocess_node(state: dict[str, Any]) -> dict[str, Any]:
        result = BaseFitnessMultiAgentGraph._preprocess_node(state)
        intent = _coerce_intent(state["raw_message"], state.get("intent_result"))
        # The base graph already keeps the bounded short-memory window.
        result["safety_level"] = intent.safety_level
        result["intent_result"] = {
            "name": intent.name,
            "confidence": intent.confidence,
            "safety_level": intent.safety_level,
            "action": intent.action,
        }
        return result
