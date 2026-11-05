"""E005/E006: object-bound commands, source domains and real SQL effects.

Migrated to the 2026-10-18 semantic contract. The model decides task type,
action and safety flags; the application still enforces original-span source
coverage, entity provenance and idempotent execution.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from app.services.agent.multi_agent_graph import (
    _food_inputs_from_structured_payload,
    validate_workout_records_against_source,
)
from app.services.agent.planner_validation import validate_planner_output
from app.services.agent.task_planner import plan_tasks_from_llm
from evals.v2.harness import GraphScenario, freeze
from evals.v2.safety import deny_network


@pytest.fixture(autouse=True)
def offline():
    with deny_network():
        yield


def _seed_safety(message):
    from app.services.agent.semantics import remember_safety
    remember_safety(message, {"level": "normal", "evidence": "", "categories": [], "reason": "scripted normal"})


def _validate(message, raw, **kwargs):
    _seed_safety(message)
    return validate_planner_output(message, raw, **kwargs)


def model_task(message, kind, action="record", **extra):
    return {"task_type": kind, "action": action, "text": message,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            **extra}


@pytest.mark.parametrize("message", [
    "早餐吃了两个鸡蛋、一杯豆浆和一根香蕉，帮我记一下。",
    "午餐吃了米饭100克，记录一下",
    "记录晚餐米饭100克，帮我记下",
])
def test_tail_command_binds_to_one_food_task(message):
    result = _validate(message, {"tasks": [model_task(message, "nutrition_record")]})
    assert [t["task_type"] for t in result.tasks] == ["nutrition_record"]
    assert result.tasks[0]["write_authorization"] == "authorized"
    assert result.tasks[0]["text"] != "帮我记一下"


def test_two_independent_equal_records_are_not_deduplicated_by_subject():
    message = "记录早餐米饭100克，然后记录晚餐米饭100克"
    planned = plan_tasks_from_llm(message, {"tasks": [
        model_task("记录早餐米饭100克", "nutrition_record"),
        model_task("记录晚餐米饭100克", "nutrition_record"),
    ]})
    assert len(planned) == 2 and all(t["side_effect"] for t in planned)
    assert planned[0]["id"] != planned[1]["id"]


@pytest.mark.parametrize("command", ["帮我记一下", "保存", "记录一下"])
def test_bare_command_cannot_authorize_invented_model_entities(command):
    payload = {"foods": [{"name": "米饭", "quantity": 1, "unit": "碗"}]}
    assert _food_inputs_from_structured_payload(payload, command) == ()


@pytest.mark.parametrize("message", [
    "今天体重68.4公斤，帮我记下",
    "记录身体重量70kg",
    "请保存称重68公斤",
    "跑步后体重68公斤，请记录",
    "跑步后体重六十八点四公斤，请记录",
    "跑步后称重68公斤，比昨天重1公斤，请记录",
    "跑步后体重六十八公斤，比昨天重一点五公斤，请记录",
])
def test_model_question_label_keeps_body_weight_read_only(message):
    # Body weight is excluded by the model's task type, not by a text pattern.
    result = _validate(message, {"tasks": [model_task(message, "training_qa", action="question")]})
    assert not result.fallback_required
    assert result.tasks[0]["action"] == "question"
    assert not result.tasks[0]["side_effect"]


@pytest.mark.parametrize("message", [
    "如果我吃了两个鸡蛋帮我记录",
    "要是昨天吃了炸鸡，今天是不是该少吃点？",
])
def test_hypothetical_semantics_keep_advice_without_write(message):
    result = _validate(message, {"tasks": [
        model_task(message, "nutrition_advice", action="question", is_hypothetical=True)]})
    assert [t["task_type"] for t in result.tasks] == ["nutrition_advice"]
    assert all(not t.get("side_effect", False) for t in result.tasks)
    assert all(t.get("write_authorization") != "authorized" for t in result.tasks)


def case(message, identity, **conditions):
    return {"id": identity, "input": {"message": message, "now": "2026-10-15T12:00:00+08:00",
            "timezone": "Asia/Shanghai"}, "conditions": conditions}


def counts(factory):
    names = ("user_food_log", "workout_records", "workout_sessions", "workout_session_exercises", "workout_session_sets")
    with factory() as s:
        return {name: s.execute(text("SELECT COUNT(*) FROM " + name)).scalar_one() for name in names}


def test_food_tail_has_one_real_write_and_completed_replay_has_none(compact_db, monkeypatch):
    message = "早餐吃了米饭100克，帮我记一下。"
    _seed_safety(message)
    fixture = case(message, "food-tail-sql", food=True, grams=100)
    freeze(monkeypatch, fixture)
    scenario = GraphScenario(monkeypatch, fixture,
                             planner={"tasks": [model_task(message, "nutrition_record")]})
    first = asyncio.run(scenario.respond())
    saved = counts(compact_db)
    assert saved["user_food_log"] == 1
    assert sum(saved[name] for name in saved if name != "user_food_log") == 0
    assert scenario.calls.count("calculate_nutrition") + scenario.calls.count("calculate_nutrition_batch") == 1
    dispatches = list(scenario.calls)
    second = asyncio.run(scenario.respond())
    assert second.replayed and second.answer == first.answer
    assert scenario.calls == dispatches and counts(compact_db) == saved


def test_identical_independent_commands_keep_distinct_source_positions():
    span = "记录早餐米饭100克"
    message = span + "，然后" + span
    raw = {"tasks": [model_task(span, "nutrition_record"), model_task(span, "nutrition_record")]}
    tasks = _validate(message, raw).tasks
    assert len(tasks) == 2 and all(t["side_effect"] for t in tasks)
    assert tasks[0]["source_start"] == 0
    assert tasks[1]["source_start"] == message.rfind(span)
    denied_message = span + "，然后不要" + span
    denied_raw = {"tasks": [
        model_task(span, "nutrition_record"),
        model_task("不要" + span, "nutrition_record", is_negated=True),
    ]}
    tasks = _validate(denied_message, denied_raw).tasks
    assert sum(t["side_effect"] for t in tasks) == 1


def test_identical_independent_commands_write_twice_and_replay_is_stable(compact_db, monkeypatch):
    message = "记录早餐米饭100克，然后记录早餐米饭100克"
    _seed_safety(message)
    fixture = case(message, "two-identical-sql", food=True, grams=100)
    freeze(monkeypatch, fixture)
    span = "记录早餐米饭100克"
    planner = {"tasks": [model_task(span, "nutrition_record"), model_task(span, "nutrition_record")]}
    scenario = GraphScenario(monkeypatch, fixture, planner=planner)
    asyncio.run(scenario.respond())
    saved = counts(compact_db)
    assert saved["user_food_log"] == 2
    assert scenario.calls.count("calculate_nutrition") + scenario.calls.count("calculate_nutrition_batch") == 2
    with compact_db() as s:
        keys = list(s.execute(text("SELECT idempotency_key FROM user_food_log")).scalars())
    assert len(set(keys)) == 2
    dispatched = list(scenario.calls)
    replay = asyncio.run(scenario.respond())
    assert replay.replayed and scenario.calls == dispatched and counts(compact_db) == saved


def test_body_weight_forged_workout_entities_have_zero_workout_sql(compact_db, monkeypatch):
    message = "今天体重68.4公斤，帮我记下"
    fixture = case(message, "body-weight-sql")
    freeze(monkeypatch, fixture)
    raw = {"tasks": [model_task(message, "workout_record", workouts=[
        {"exercise": "卧推", "weight_kg": 68.4, "sets": 4, "reps": 8}])]}
    scenario = GraphScenario(monkeypatch, fixture, planner=raw)
    asyncio.run(scenario.respond())
    assert all(value == 0 for value in counts(compact_db).values())
    assert "save_workout_record" not in scenario.calls
    kept, rejected = validate_workout_records_against_source(
        [{"exercise": "卧推", "weight_kg": 68.4, "sets": 4, "reps": 8}], message)
    assert kept == [] and rejected


def test_hypothetical_advice_dispatches_read_worker_with_zero_business_writes(compact_db, monkeypatch):
    message = "要是昨天吃了炸鸡，今天是不是该少吃点？"
    fixture = case(message, "hypothetical-advice-sql")
    freeze(monkeypatch, fixture)
    planner = {"tasks": [model_task(message, "nutrition_advice", action="question", is_hypothetical=True)]}
    scenario = GraphScenario(monkeypatch, fixture, planner=planner)
    advice = AsyncMock(return_value={"result": {"worker": "nutrition", "status": "complete",
                                                "answer": "按正常饮食安排。", "citations": []}})
    monkeypatch.setitem(scenario.agent.graph.workers, "nutrition", type("Worker", (), {"ainvoke": advice})())
    response = asyncio.run(scenario.respond())
    assert response.answer == "按正常饮食安排。"
    assert advice.await_count == 1 and advice.call_args.args[0]["task_action"] == "question"
    assert all(value == 0 for value in counts(compact_db).values())
    assert not any(name in scenario.calls for name in ("calculate_nutrition", "calculate_nutrition_batch", "save_workout_record"))


def test_body_weight_worker_saves_nothing_without_a_valid_exercise(monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    graph = FitnessMultiAgentGraph()
    extract = AsyncMock(return_value=[])
    save = AsyncMock(side_effect=AssertionError("body weight must not be saved as a workout"))
    monkeypatch.setattr(graph.workout_extraction, "run_many", extract)
    monkeypatch.setattr(graph.tools, "save_workout_record", save)
    asyncio.run(graph.workers["workout_record"].ainvoke({"user_id": "demo-user",
        "raw_message": "记录体重68公斤", "task_text": "记录体重68公斤", "task_action": "record",
        "task_payload": {"write_authorization": "authorized", "workouts": []}}))
    extract.assert_awaited()
    save.assert_not_called()
