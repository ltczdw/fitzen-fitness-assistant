"""Write scope and negation scope under the 2026-10-18 semantic contract.

Migrated from the pre-semantic text contract:
- question / negation / hypothesis / third-party intent is expressed by the
  model's action or structured flags, never re-derived from wording;
- no model result means a read-only fallback with
  ``semantic_intent_unavailable`` and zero writes;
- original source boundaries and entities still apply.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.schemas.chat import ChatRequest
from app.services.agent import execution
from app.services.agent.agent import FitnessAgent
from app.services.agent.food_parser import FoodInput
from app.services.agent.planner_validation import validate_planner_output, validate_rule_plan
from app.services.agent.routing import PlannerRoutingMixin
from app.services.security.write_guard import guard_planner_tasks


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _workout_task(message: str, text: str, action: str = "record", **flags) -> dict:
    return {
        "task_type": "workout_record", "action": action, "text": text,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        **flags,
    }


def _validate(message, raw, **kwargs):
    """Seed the source-bound model safety result these direct gate cases use."""
    from app.services.agent.semantics import remember_safety
    remember_safety(message, {"level": "normal", "evidence": "", "categories": [],
                              "reason": "scripted normal"})
    return validate_planner_output(message, raw, **kwargs)


# ---------------------------------------------------------------------------
# Question semantics are model-owned
# ---------------------------------------------------------------------------

def test_question_action_rejects_the_write():
    message = "记录卧推60kg，4组8次？"
    result = _validate(message, {"tasks": [_workout_task(message, message, action="question")]})
    # The record-domain label is kept for a non-write action; the action is
    # question, so the shared gate must not authorize or execute a save.
    record = next(task for task in result.tasks if task["task_type"] == "workout_record")
    assert record["action"] == "question"
    assert record["status"] != "authorized"
    assert record["side_effect"] is False


def test_no_model_fallback_keeps_write_candidates_read_only():
    message = "记录卧推60kg，4组8次？"
    rule = [{"task_type": "workout_record", "action": "record", "text": message}]
    result = validate_rule_plan(message, rule)
    record = next(task for task in result.tasks if task["task_type"] == "workout_record")
    assert record["status"] == "needs_confirmation"
    assert record["write_rejection_reason"] == "semantic_intent_unavailable"
    assert record["side_effect"] is False


@pytest.mark.parametrize("message,flag,reason", [
    ("记录卧推60kg，4组8次，不要保存", "is_negated", "planner_flag_negated"),
    ("记录卧推60kg，4组8次，他昨天练的", "is_third_party", "planner_flag_third_party"),
])
def test_trailing_negation_and_third_party_flags_are_applied_by_the_shared_gate(message, flag, reason):
    result = _validate(message, {"tasks": [_workout_task(message, message, **{flag: True})]})
    record = next(task for task in result.tasks if task["task_type"] == "workout_record")
    assert record["status"] == "rejected_unauthorized"
    assert record["write_rejection_reason"] == reason


def test_old_tail_window_is_not_reclassified_and_model_question_still_prevents_write():
    message = "记录卧推60kg，4组8次？"
    record = _validate(message, {"tasks": [_workout_task(message, "记录卧推60kg，4组8次")]})
    assert record.tasks[0]["status"] == "authorized"
    assert "tail_window" not in record.tasks[0]
    question = _validate(message, {"tasks": [
        _workout_task(message, "记录卧推60kg，4组8次", action="question")]})
    assert not question.tasks[0]["side_effect"]


@pytest.mark.parametrize("message", [
    "记录他昨天的卧推60kg，4组8次",
    "记录她昨天的卧推60kg，4组8次",
    "记录他的卧推60kg，4组8次",
])
def test_third_party_record_commands_are_rejected(message):
    result = _validate(message, {"tasks": [_workout_task(message, message, is_third_party=True)]})
    record = next(task for task in result.tasks if task["task_type"] == "workout_record")
    assert record["status"] == "rejected_unauthorized"
    assert record["write_rejection_reason"] == "planner_flag_third_party"


def test_other_actions_is_not_mistaken_for_third_party():
    message = "记录其他动作，卧推60kg，4组8次"
    result = _validate(message, {"tasks": [_workout_task(message, message)]})
    assert result.tasks[0]["status"] == "authorized"
    task = {"id": "write", "task_type": "workout_record", "action": "record", "text": message}
    guard_planner_tasks(message, [task], now=_now(), semantic_available=True)
    assert task["payload"]["write_authorization"] == "authorized"


def test_follow_up_question_and_backfill_keep_the_record_authorized():
    message = "记录卧推60kg，4组8次，然后解释一下为什么要热身？"
    parsed = {"tasks": [
        _workout_task(message, "记录卧推60kg，4组8次"),
        {**_workout_task(message, "然后解释一下为什么要热身？", action="question"),
         "task_type": "training_qa"},
    ]}
    result = _validate(message, parsed)
    assert result.planner_backend == "llm"
    record = next(task for task in result.tasks if task["task_type"] == "workout_record")
    question = next(task for task in result.tasks if task["task_type"] == "training_qa")
    assert record["status"] == "authorized"
    assert "tail_window" not in record
    assert question["status"] == "pending"


# ---------------------------------------------------------------------------
# Every planning path keeps questions read-only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["llm", "shadow", "planner_unavailable", "invalid_llm", "task_mapping_failed"])
def test_question_scope_is_rejected_on_every_planning_path(mode, monkeypatch):
    from app.services.agent import routing

    message = "记录卧推60kg，4组8次？"
    monkeypatch.setattr(routing, "get_pending_food", lambda *_: None)
    monkeypatch.setattr(settings, "planner_shadow_enabled", mode == "shadow")
    llm_task = _workout_task(message, "记录卧推60kg，4组8次", action="question")
    if mode == "shadow":
        graph = SimpleNamespace()
    elif mode == "llm":
        graph = SimpleNamespace(parse_input=AsyncMock(return_value={"tasks": [llm_task]}))
    elif mode == "invalid_llm":
        graph = SimpleNamespace(parse_input=AsyncMock(return_value={"tasks": [{"broken": True}]}))
    else:
        graph = SimpleNamespace(parse_input=AsyncMock(return_value=None))
    if mode == "task_mapping_failed":
        graph.parse_input = AsyncMock(return_value={"tasks": [llm_task]})
        monkeypatch.setattr(routing, "plan_tasks_from_llm", lambda *_: None)

    class Agent(PlannerRoutingMixin):
        def __init__(self):
            self.graph = graph

    request = SimpleNamespace(message=message, user_id="demo-user", conversation_id="scope-fixture")
    decision = asyncio.run(Agent()._route(request, {}))
    for task in decision.task_specs:
        assert task["payload"].get("write_authorization") != "authorized", mode
        assert task["side_effect"] is False, mode


# ---------------------------------------------------------------------------
# End-to-end routing -> validation -> dispatch -> worker with mocked tools
# ---------------------------------------------------------------------------

class _RecordingAgent:
    """Build a real agent with every external boundary replaced."""

    def __init__(self, monkeypatch, *, workouts=(), food_by_message=None, calculations=None,
                 parsed_input=None, shadow=False):
        self.calls: list[tuple[str, dict]] = []
        self.agent = FitnessAgent()
        monkeypatch.setattr(settings, "memory_read_backend", "legacy")
        monkeypatch.setattr(settings, "planner_shadow_enabled", shadow)
        monkeypatch.setattr(execution.memory_service, "load", AsyncMock(return_value={"memory_backend": "fixture"}))
        monkeypatch.setattr(self.agent, "_run_planner_shadow", AsyncMock())
        monkeypatch.setattr(self.agent.graph, "parse_input", AsyncMock(return_value=parsed_input))
        from app.services.agent.semantics import RequestUnderstanding
        tasks = (parsed_input or {}).get("tasks") or []
        core = [{key: task[key] for key in (
            "task_type", "action", "text", "is_negated", "is_hypothetical", "is_third_party")
            if key in task} for task in tasks if isinstance(task, dict)]
        async def semantic(messages, schema, *, name=None, **kwargs):
            if name != "return_request_understanding":
                raise AssertionError(name)
            return RequestUnderstanding.model_validate({
                "tasks": core or [{"task_type": "chitchat", "action": "answer",
                                   "text": "语义夹具", "is_negated": False,
                                   "is_hypothetical": False, "is_third_party": False}],
                "safety": {"level": "normal", "evidence": "", "categories": [],
                           "reason": "fixture"},
            })
        monkeypatch.setattr(self.agent.graph.llm, "configured", lambda: True)
        monkeypatch.setattr(self.agent.graph.llm, "complete_structured", semantic)
        monkeypatch.setattr(self.agent.graph.rag, "search", lambda *args, **kwargs: [])
        monkeypatch.setattr(self.agent.graph, "_run_authorized_model_tool", AsyncMock(return_value=None))
        if workouts:
            monkeypatch.setattr(
                self.agent.graph.workout_extraction, "run_many",
                AsyncMock(return_value=[{"record": record} for record in workouts]),
            )
        if food_by_message is not None:
            from app.services.agent import multi_agent_graph as graph_module
            async def fake_extract(_llm, message):
                return tuple(food_by_message(message))
            monkeypatch.setattr(graph_module, "extract_foods_with_llm", fake_extract)
        self.calculations = calculations if calculations is not None else [{
            "status": "calculated",
            "items": [{"food_name": "米饭", "is_estimated": False}],
            "totals": {"kcal": 116.0, "protein_g": 2.6, "carb_g": 25.9, "fat_g": 0.3, "fiber_g": 0.3},
            "log_ids": [1], "meal_group_id": "meal", "idempotency_key": "food-key",
        }]
        monkeypatch.setattr(self.agent.graph, "_call_mcp", self._call_mcp)

    async def _call_mcp(self, name, arguments, **kwargs):
        self.calls.append((name, arguments))
        if name == "get_user_profile":
            return {}
        if name == "get_training_history":
            return []
        if name == "get_nutrition_summary":
            return {}
        if name == "save_workout_record":
            return {"saved": True, "record": {**arguments["record"], "reps_inferred": False}}
        if name in {"calculate_nutrition", "calculate_nutrition_batch"}:
            return dict(self.calculations[0])
        raise AssertionError(f"unexpected tool: {name}")

    def named(self, name):
        return [arguments for call_name, arguments in self.calls if call_name == name]

    async def respond(self, message, request_id=None):
        kwargs = {"request_id": request_id} if request_id else {}
        request = ChatRequest(message=message, conversation_id="scope-e2e", **kwargs)
        from app.services.agent.semantics import RequestUnderstanding
        async def semantic(messages, schema, *, name=None, **kwargs):
            if name != "return_request_understanding":
                raise AssertionError(name)
            return RequestUnderstanding.model_validate({
                "tasks": [{"task_type": "training_qa", "action": "question", "text": message,
                           "is_negated": False, "is_hypothetical": False,
                           "is_third_party": False}],
                "safety": {"level": "normal", "evidence": "", "categories": [],
                           "reason": "fixture"},
            })
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(self.agent, "_persist_turn", AsyncMock())
            patch.setattr(self.agent.graph.llm, "configured", lambda: True)
            patch.setattr(self.agent.graph.llm, "complete_structured", semantic)
            return await self.agent.respond(request)


def test_question_record_is_never_saved_end_to_end(monkeypatch):
    message = "记录卧推60kg，4组8次？"
    recorder = _RecordingAgent(
        monkeypatch,
        workouts=[{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}],
        parsed_input={"tasks": [_workout_task(message, "记录卧推60kg，4组8次", action="question")]},
    )
    asyncio.run(recorder.respond(message))
    assert not recorder.named("save_workout_record")


def test_model_span_without_command_text_never_saves_when_model_marks_question(monkeypatch):
    # The old case relied on re-scanning a text tail; the current contract uses
    # the model's action. This keeps the same "no save" outcome.
    message = "记录卧推60kg，4组8次？"
    recorder = _RecordingAgent(
        monkeypatch,
        workouts=[{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}],
        parsed_input={"tasks": [_workout_task(message, "记录卧推60kg，4组8次", action="question")]},
    )
    asyncio.run(recorder.respond(message))
    assert not recorder.named("save_workout_record")


@pytest.mark.parametrize("message", [
    "记录他昨天的卧推60kg，4组8次",
    "记录她昨天的卧推60kg，4组8次",
])
def test_third_party_record_is_never_saved_end_to_end(message, monkeypatch):
    recorder = _RecordingAgent(
        monkeypatch,
        workouts=[{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}],
        parsed_input={"tasks": [_workout_task(message, message, is_third_party=True)]},
    )
    response = asyncio.run(recorder.respond(message))
    assert not recorder.named("save_workout_record")
    assert "未保存" in response.answer


def _follow_up_plan(message):
    return {"tasks": [
        _workout_task(message, "记录卧推60kg，4组8次"),
        {**_workout_task(message, "然后解释一下为什么要热身？", action="question"),
         "task_type": "training_qa"},
    ]}


def test_authorized_record_with_follow_up_question_saves_once(monkeypatch):
    message = "记录卧推60kg，4组8次，然后解释一下为什么要热身？"
    recorder = _RecordingAgent(
        monkeypatch,
        workouts=[{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}],
        parsed_input=_follow_up_plan(message),
    )
    response = asyncio.run(recorder.respond(message))
    saves = recorder.named("save_workout_record")
    assert len(saves) == 1
    record = saves[0]["record"]
    assert (record["exercise"], record["weight_kg"], record["sets"], record["reps"]) == ("卧推", 60, 4, 8)
    assert "已记录" in response.answer
    # The following question is answered as its own task, not swallowed.
    assert response.answer.count("已记录") == 1
    assert len(response.answer.split("\n\n")) >= 2


# ---------------------------------------------------------------------------
# Negation/hypothesis scope is per task, not whole message
# ---------------------------------------------------------------------------

def test_negated_food_clause_does_not_swallow_the_rice_record(monkeypatch):
    message = "不要记录鸡蛋，然后记录100克米饭"

    def food(_message):
        return [FoodInput(raw_text="100g米饭", requested_name="米饭", quantity=100, unit="g",
                          weight_g=100, portion_name=None)]

    parsed = {"tasks": [
        {**_workout_task(message, "不要记录鸡蛋", is_negated=True), "task_type": "nutrition_record"},
        {**_workout_task(message, "记录100克米饭"), "task_type": "nutrition_record",
         "foods": [{"name": "米饭", "quantity": 100, "unit": "克", "weight_g": 100,
                    "meal_type": None, "record_date": None}]},
    ]}
    recorder = _RecordingAgent(monkeypatch, food_by_message=food, parsed_input=parsed)
    response = asyncio.run(recorder.respond(message))

    calculations = recorder.named("calculate_nutrition")
    assert len(calculations) == 1, recorder.calls
    entry = calculations[0]["entries"][0]
    assert entry["requested_name"] == "米饭"
    assert not any("鸡蛋" in str(call) for call in recorder.named("save_workout_record"))
    # Both tasks produced an explicable outcome: the egg rejection and the
    # rice record confirmation.
    assert "未保存" in response.answer
    assert "已记录" in response.answer


def test_record_with_later_hypothetical_question_keeps_the_write(monkeypatch):
    message = "记录卧推60kg，4组8次，然后如果受伤了怎么办"
    parsed = {"tasks": [
        _workout_task(message, "记录卧推60kg，4组8次"),
        {**_workout_task(message, "然后如果受伤了怎么办", action="question", is_hypothetical=True),
         "task_type": "safety_triage"},
    ]}
    recorder = _RecordingAgent(
        monkeypatch,
        workouts=[{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}],
        parsed_input=parsed,
    )
    response = asyncio.run(recorder.respond(message))

    saves = recorder.named("save_workout_record")
    assert len(saves) == 1
    assert saves[0]["record"]["exercise"] == "卧推"
    assert "已记录" in response.answer
    # The hypothetical safety question is handled as its own task instead of
    # cancelling the write.
    assert len(response.answer.split("\n\n")) >= 2
