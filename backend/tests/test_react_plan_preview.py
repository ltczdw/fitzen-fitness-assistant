"""ReAct plan_training delegates concrete plans to the application plan worker."""
import asyncio

import pytest

from app.services.agent import react_readonly_workers as workers


class _Worker:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def ainvoke(self, state):
        self.calls.append(state)
        return {"result": self.result}


PREVIEW = {"preview_id": "p1", "expected_sessions": 2,
           "days": [{"label": "周二", "minutes": 30, "exercises": ["深蹲"]}]}
CONTEXT = {
    "source_text": "给周二和周六各30分钟的哑铃全身训练",
    "task_text": "给周二和周六各30分钟的哑铃全身训练",
    "user_id": "demo-user",
    "conversation_id": "conv-plan",
    "request_id": "req-plan",
    "semantic_scope": {"task_type": "training_plan", "delivery_type": "plan"},
    "plan_contract": {"plan_result": True},
}


def _install(monkeypatch, result):
    from app.services.agent.agent import agent
    worker = _Worker(result)
    monkeypatch.setitem(agent.graph.workers, "plan_adjust", worker)
    return worker


def test_plan_result_delegates_and_keeps_preview(monkeypatch):
    worker = _install(monkeypatch, {
        "status": "complete", "answer": "周二/周六各30分钟的安排。",
        "citations": [{"title": "训练学", "source": "s1"}], "plan_preview": PREVIEW,
        "model_used": True,
    })
    observation = asyncio.run(workers.plan_training(
        {"query": CONTEXT["task_text"]}, CONTEXT))
    assert observation.status.value == "completed"
    assert observation.summary == "周二/周六各30分钟的安排。"
    assert observation.result["plan_preview"] == PREVIEW
    assert observation.result["citations"]
    assert worker.calls and worker.calls[0]["request_id"] == "req-plan"
    assert worker.calls[0]["semantic_scope"]["delivery_type"] == "plan"


def test_plan_preview_edit_also_delegates(monkeypatch):
    context = {**CONTEXT, "semantic_scope": {
        "task_type": "training_plan", "delivery_type": "preview",
        "preview_ref": "p1", "plan_result": False}}
    worker = _install(monkeypatch, {
        "status": "complete", "answer": "周六改为25分钟。",
        "citations": [], "plan_preview": {**PREVIEW, "preview_id": "p2"}})
    observation = asyncio.run(workers.plan_training({"query": "周六25分钟"}, context))
    assert observation.status.value == "completed"
    assert observation.result["plan_preview"]["preview_id"] == "p2"
    assert worker.calls[0]["current_plan_preview"] == {}


def test_plan_clarification_maps_to_observation(monkeypatch):
    clarification = {"task_id": "", "task_type": "training_plan",
                     "category": "missing_information", "fields": [],
                     "question": "还需要补充：训练目标。"}
    _install(monkeypatch, {"status": "needs_confirmation",
                           "answer": "还需要补充：训练目标。",
                           "citations": [], "model_used": False,
                           "clarification": clarification})
    observation = asyncio.run(workers.plan_training({"query": "给我计划"}, CONTEXT))
    assert observation.status.value == "needs_clarification"
    assert observation.summary == "还需要补充：训练目标。"
    assert observation.result["clarification"] == clarification


def test_readonly_result_surfaces_plan_preview():
    from app.services.agent.react_runtime import _map_readonly_result

    mapped = _map_readonly_result({
        "final_answer": "周二/周六各30分钟。",
        "status": "complete",
        "observations": [{"status": "completed",
                          "result": {"plan_preview": PREVIEW}}],
        "call_facts": [{"name": "plan_training"}],
        "safety_level": "normal",
    })
    assert mapped["plan_preview"] == PREVIEW


@pytest.mark.parametrize("delivery", ["plan", "preview"])
def test_select_mode_allows_react_for_concrete_plan(monkeypatch, delivery):
    from app.core.config import settings
    from app.schemas.chat import ChatRequest
    from app.services.agent.react_runtime import select_mode

    monkeypatch.setattr(settings, "react_readonly_enabled", True)
    monkeypatch.setattr(settings, "memory_read_backend", "compact_v1")
    task = {"task_type": "training_plan", "action": "question",
            "delivery_type": delivery, "goal_role": "result"}
    request = ChatRequest(message="给一份计划", mode="react_readonly")
    assert select_mode(request, [task]) == ("react_readonly", "pure_read")
