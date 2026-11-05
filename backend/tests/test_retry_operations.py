"""P1-2: retries must reuse the first execution plan, keys and parameters."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from app.core import clock
from app.db import repositories
from app.db.food_repository import log_food_items
from app.db.storage import CommitUnknownError, IdempotencyConflict, StorageUnavailableError
from app.services.agent.agent import FitnessAgent
from app.services.agent.food_parser import FoodInput
from app.services.agent.operation_store import (OperationStore, OperationStoreUnavailable,
                                                food_operation_identity, workout_operation_identity)
from app.services.agent import execution
from app.schemas.chat import ChatRequest


class RetryScenario:
    """A real agent whose only replacements are model and tool boundaries."""

    def __init__(self, monkeypatch, factory, *, parsed_input=None, workout_records=None,
                 food_entries=None, fail=None, fail_once=False):
        self.factory = factory
        self.fail = dict(fail or {})
        self.fail_once = fail_once
        self.saves: list[dict] = []
        self.tool_calls: list[str] = []
        self.agent = FitnessAgent()
        monkeypatch.setattr(self.agent.graph, "parse_input", AsyncMock(return_value=parsed_input))
        async def default_semantic(messages, schema, *, name=None, **kwargs):
            if name != "return_request_understanding":
                raise AssertionError(name)
            payload = json.loads(messages[-1]["content"])
            source = str(payload.get("query") or "")
            return schema.model_validate({
                "tasks": [{"task_type": "training_qa", "action": "question", "text": source,
                           "is_negated": False, "is_hypothetical": False,
                           "is_third_party": False}],
                "safety": {"level": "normal", "evidence": "", "categories": [],
                           "reason": "scripted normal"},
            })
        monkeypatch.setattr(self.agent.graph.llm, "configured", lambda: True)
        monkeypatch.setattr(self.agent.graph.llm, "complete_structured", default_semantic)
        monkeypatch.setattr(self.agent, "_run_planner_shadow", AsyncMock())
        monkeypatch.setattr(execution.memory_service, "load",
                            AsyncMock(return_value={"memory_backend": "fixture"}))
        monkeypatch.setattr(self.agent.graph.rag, "search", lambda *args, **kwargs: [])
        monkeypatch.setattr(self.agent.graph, "_run_authorized_model_tool", AsyncMock(return_value=None))
        self.workout_records = workout_records
        if workout_records is not None:
            async def extract(_message):
                return [{"record": record} for record in self.workout_records(_message)]
            monkeypatch.setattr(self.agent.graph.workout_extraction, "run_many", extract)
        if food_entries is not None:
            from app.services.agent import multi_agent_graph as graph_module
            async def fake_food(_llm, _message):
                return tuple(food_entries)
            monkeypatch.setattr(graph_module, "extract_foods_with_llm", fake_food)
        monkeypatch.setattr(self.agent.graph, "_call_mcp", self._call_mcp)

    def restore_provider_semantics(self, monkeypatch):
        """Restore the real shared LLM boundary after fixture defaults.

        The HTTP tests below simulate the provider transport, so the shared
        semantic entry must reach that transport instead of the default
        training_qa stub installed for legacy unit scenarios.
        """
        llm = self.agent.graph.llm
        llm_type = type(llm)
        monkeypatch.setattr(llm, "configured",
                            llm_type.configured.__get__(llm, llm_type))
        monkeypatch.setattr(llm, "complete_structured",
                            llm_type.complete_structured.__get__(llm, llm_type))

    async def _call_mcp(self, name, arguments, **kwargs):
        self.tool_calls.append(name)
        if name == "get_user_profile":
            return {}
        if name == "get_training_history":
            return []
        if name == "get_nutrition_summary":
            return {}
        if name == "save_workout_record":
            record = arguments["record"]
            if self.fail.get(record.get("exercise")):
                if self.fail_once:
                    self.fail.pop(record.get("exercise"))
                raise StorageUnavailableError("fixture save unavailable")
            saved = self.agent.graph.tools.save_workout_record(**arguments)
            self.saves.append(saved["record"])
            return saved
        if name == "calculate_nutrition":
            calculation = self.agent.graph.tools.calculate_nutrition(**arguments)
            if self.fail.get("nutrition"):
                raise CommitUnknownError("fixture commit outcome unknown")
            return calculation
        raise AssertionError(f"unexpected tool: {name}")

    async def respond(self, request):
        return await self.agent.respond(request)


# ---------------------------------------------------------------------------
# Repository-level conflict rules
# ---------------------------------------------------------------------------

def test_same_workout_key_same_parameters_returns_original(compact_db):
    first = repositories.create_workout("demo-user", "卧推", 4, 8, 60, idempotency_key="key-1")
    again = repositories.create_workout("demo-user", "卧推", 4, 8, 60, idempotency_key="key-1")
    assert again["id"] == first["id"]
    with compact_db() as session:
        count = session.execute(text("SELECT COUNT(*) FROM workout_records")).scalar()
    assert count == 1


@pytest.mark.parametrize("kwargs", [
    {"exercise": "卧推", "sets": 4, "reps": 10, "weight_kg": 60},
    {"exercise": "卧推", "sets": 4, "reps": 8, "weight_kg": 65},
    {"exercise": "深蹲", "sets": 4, "reps": 8, "weight_kg": 60},
])
def test_same_workout_key_different_parameters_conflicts(compact_db, kwargs):
    repositories.create_workout("demo-user", "卧推", 4, 8, 60, idempotency_key="key-conflict")
    with pytest.raises(IdempotencyConflict) as error:
        repositories.create_workout("demo-user", idempotency_key="key-conflict", **kwargs)
    assert "key-conflict" in str(error.value)
    with compact_db() as session:
        assert session.execute(text("SELECT COUNT(*) FROM workout_records")).scalar() == 1


def _food_item(name: str, weight: float) -> dict:
    return {
        "food_id": None, "raw_food_name": name, "food_name": name, "weight_g": weight,
        "energy_kcal": 100.0, "protein_g": 3.0, "carb_g": 20.0, "fat_g": 1.0, "fiber_g": 0.5,
        "is_estimated": False, "weight_source": "user_input", "match_method": "fallback_exact",
        "match_confidence": 1.0,
    }


def test_same_food_key_same_items_replays_and_conflicts_on_change(compact_db):
    totals = {"kcal": 100.0, "protein_g": 3.0, "carb_g": 20.0, "fat_g": 1.0, "fiber_g": 0.5}
    first = log_food_items("demo-user", [_food_item("米饭", 100)], totals, "meal-1", "food-key", meal_type="午餐")
    again = log_food_items("demo-user", [_food_item("米饭", 100)], totals, "meal-1", "food-key", meal_type="午餐")
    assert again == first
    with pytest.raises(IdempotencyConflict):
        log_food_items("demo-user", [_food_item("米饭", 250)], totals, "meal-1", "food-key", meal_type="午餐")
    with compact_db() as session:
        assert session.execute(text("SELECT COUNT(*) FROM user_food_log")).scalar() == 1


# ---------------------------------------------------------------------------
# Plan reuse
# ---------------------------------------------------------------------------

def test_stored_plan_is_never_overwritten(compact_db):
    from app.services.memory import compact
    compact.begin_turn(ChatRequest(message="计划", conversation_id="plan",
                                  request_id="request-plan"))
    store = OperationStore("request-plan")
    store.save_plan([{"id": "task-1", "text": "深蹲"}], time_base=datetime(2026, 9, 1, tzinfo=timezone.utc))
    store.save_plan([{"id": "task-1", "text": "卧推"}])
    reused = OperationStore("request-plan")
    assert reused.stored_plan() == [{"id": "task-1", "text": "深蹲"}]
    assert reused.time_base() == datetime(2026, 9, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Ledger failures must stop save dispatch
# ---------------------------------------------------------------------------

def test_ledger_write_failure_is_never_swallowed(compact_db, monkeypatch):
    from app.services.memory import compact
    request = ChatRequest(message="计划", conversation_id="ledger-unit", request_id="ledger-unit-1")
    compact.begin_turn(request)
    plan = [{"id": "task-1", "text": "记录卧推", "side_effect": True}]

    def broken(*_args, **_kwargs):
        raise StorageUnavailableError("ledger down")

    monkeypatch.setattr(compact, "save_request_state", broken)
    with pytest.raises(OperationStoreUnavailable):
        OperationStore(request.request_id).save_plan(plan)

    # A rejected save (turn gone) is also a ledger failure, not a silent no-op.
    monkeypatch.setattr(compact, "save_request_state", lambda *_args, **_kwargs: False)
    with pytest.raises(OperationStoreUnavailable):
        OperationStore(request.request_id).save_plan(plan)


def test_identical_content_keeps_positional_operation_identity():
    common = {"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}
    first = {**common, "idempotency_key": "workout-task-req-1-task-1-0"}
    second = {**common, "idempotency_key": "workout-task-req-1-task-2-0"}
    later = {**common, "idempotency_key": "workout-task-req-1-task-1-1"}
    first_id, first_fp = workout_operation_identity(first)
    second_id, second_fp = workout_operation_identity(second)
    later_id, later_fp = workout_operation_identity(later)
    assert first_fp == second_fp == later_fp  # identical content
    assert len({first_id, second_id, later_id}) == 3  # distinct positions stay distinct
    assert workout_operation_identity(dict(first))[0] == first_id  # stable across a retry

    entries = [{"requested_name": "米饭", "weight_g": 100, "quantity": 100, "unit": "g",
                "meal_type": "午餐", "record_date": None}]
    key_a = food_operation_identity(entries, key="food-task-a")[0]
    key_b = food_operation_identity(entries, key="food-task-b")[0]
    assert key_a != key_b
    assert key_a == food_operation_identity(list(entries), key="food-task-a")[0]


def test_same_position_with_different_parameters_conflicts(compact_db):
    from app.services.memory import compact
    request = ChatRequest(message="计划", conversation_id="ledger-conflict",
                          request_id="ledger-conflict-1")
    compact.begin_turn(request)
    store = OperationStore(request.request_id)
    store.save_plan([{"id": "task-1", "text": "卧推", "side_effect": True}])
    reserve = dict(kind="workout", key="workout-task-ledger-conflict-1-task-1-0")
    store.reserve_operation("workout:op:fixed", fingerprint="fp-a", params={"a": 1}, **reserve)
    with pytest.raises(IdempotencyConflict):
        store.reserve_operation("workout:op:fixed", fingerprint="fp-b", params={"a": 2}, **reserve)
    same = store.reserve_operation("workout:op:fixed", fingerprint="fp-a", params={"a": 1}, **reserve)
    assert same["fingerprint"] == "fp-a"


# ---------------------------------------------------------------------------
# End-to-end partial write retries
# ---------------------------------------------------------------------------

def record_plan(message, task_type="workout_record"):
    """Validated semantic record decision with exact user spans.

    The 2026-10-18 contract forbids deriving writes from keywords, so every
    retry test supplies the model's task decision explicitly. The application
    still validates spans, authorization, plan freezing and operation identity.
    """
    return {"tasks": [
        {"task_type": task_type, "action": "record", "text": span,
         "is_negated": False, "is_hypothetical": False, "is_third_party": False}
        for span in message.split("，然后")
    ]}


def _workouts(message):
    if "深蹲" in message or "杠铃深蹲" in message:
        return [{"exercise": "深蹲", "weight_kg": 40, "sets": 3, "reps": 8}]
    return [{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}]


def test_partial_workout_retry_reuses_plan_when_the_model_reverses_order(compact_db, monkeypatch):
    request = ChatRequest(
        message="记录深蹲40kg3组8次，然后记录卧推60kg，4组8次",
        conversation_id="retry-reversed", request_id="retry-reversed-1",
    )
    first = RetryScenario(monkeypatch, compact_db, parsed_input=record_plan(request.message),
                          workout_records=_workouts, fail={"卧推": True})
    with pytest.raises(StorageUnavailableError):
        asyncio.run(first.respond(request))
    assert [record["exercise"] for record in first.saves] == ["深蹲"]

    # The retry sees a different (reversed) model plan; it must be ignored.
    reversed_input = {"tasks": [
        {"task_type": "workout_record", "action": "record", "text": "记录卧推60kg，4组8次",
         "is_negated": False, "is_hypothetical": False, "is_third_party": False},
        {"task_type": "workout_record", "action": "record", "text": "记录深蹲40kg3组8次",
         "is_negated": False, "is_hypothetical": False, "is_third_party": False},
    ]}
    second = RetryScenario(monkeypatch, compact_db, parsed_input=reversed_input, workout_records=_workouts)
    response = asyncio.run(second.respond(request))

    exercises = [record["exercise"] for record in second.saves]
    assert exercises == ["卧推"], exercises
    with compact_db() as session:
        rows = session.execute(text(
            "SELECT exercise, sets, reps, weight_kg FROM workout_records ORDER BY exercise")).all()
    assert [(row[0], row[1], row[2], row[3]) for row in rows] == [("卧推", 4, 8, 60.0), ("深蹲", 3, 8, 40.0)]
    assert "已记录" in response.answer


def test_multi_action_task_retry_completes_without_duplicates(compact_db, monkeypatch):
    message = "帮我补记昨天卧推40公斤3组8次，深蹲60公斤4组6次"
    request = ChatRequest(message=message, conversation_id="retry-multi", request_id="retry-multi-1")

    def two_actions(_message):
        return [
            {"exercise": "卧推", "weight_kg": 40, "sets": 3, "reps": 8},
            {"exercise": "深蹲", "weight_kg": 60, "sets": 4, "reps": 6},
        ]

    first = RetryScenario(monkeypatch, compact_db, parsed_input=record_plan(request.message),
                          workout_records=two_actions, fail={"深蹲": True})
    with pytest.raises(StorageUnavailableError):
        asyncio.run(first.respond(request))
    assert [record["exercise"] for record in first.saves] == ["卧推"]

    second = RetryScenario(monkeypatch, compact_db, parsed_input=record_plan(request.message),
                           workout_records=two_actions)
    response = asyncio.run(second.respond(request))
    assert [record["exercise"] for record in second.saves] == ["深蹲"]
    with compact_db() as session:
        rows = session.execute(text(
            "SELECT exercise, weight_kg FROM workout_records ORDER BY exercise")).all()
    assert [(row[0], row[1]) for row in rows] == [("卧推", 40.0), ("深蹲", 60.0)]
    assert "已记录" in response.answer


def test_worker_reports_an_explicit_conflict_instead_of_rebinding_a_key(compact_db, monkeypatch):
    # A previous action already owns this key with different parameters.
    repositories.create_workout("demo-user", "深蹲", 3, 8, 40,
                                idempotency_key="workout-task-conflict-1-task-1-0")
    request = ChatRequest(message="记录卧推60kg，4组8次", conversation_id="retry-conflict",
                          request_id="conflict-1")
    scenario = RetryScenario(
        monkeypatch, compact_db, parsed_input=record_plan(request.message),
        workout_records=lambda _message: [{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}],
    )
    response = asyncio.run(scenario.respond(request))
    assert "冲突" in response.answer
    assert not scenario.saves  # nothing was returned as a successful save
    with compact_db() as session:
        rows = session.execute(text(
            "SELECT exercise, weight_kg FROM workout_records ORDER BY exercise")).all()
    assert [(row[0], row[1]) for row in rows] == [("深蹲", 40.0)]


def test_concurrent_same_request_id_never_runs_two_plans(compact_db, monkeypatch):
    request = ChatRequest(
        message="记录深蹲40kg3组8次，然后记录卧推60kg，4组8次",
        conversation_id="retry-concurrent", request_id="retry-concurrent-1",
    )
    scenario = RetryScenario(monkeypatch, compact_db, parsed_input=record_plan(request.message),
                             workout_records=_workouts, fail={"卧推": True}, fail_once=True)

    async def run():
        return await asyncio.gather(
            scenario.respond(request), scenario.respond(request), return_exceptions=True,
        )

    results = asyncio.run(run())
    failures = [item for item in results if isinstance(item, BaseException)]
    responses = [item for item in results if not isinstance(item, BaseException)]
    assert len(failures) == 1 and len(responses) == 1
    with compact_db() as session:
        rows = session.execute(text(
            "SELECT exercise, sets, reps, weight_kg FROM workout_records ORDER BY exercise")).all()
    assert [(row[0], row[1], row[2], row[3]) for row in rows] == [
        ("卧推", 4, 8, 60.0), ("深蹲", 3, 8, 40.0)]
    # The successful request is either the retry of the failed one or its
    # replay; in both cases exactly one plan ran.
    assert "已记录" in responses[0].answer


def test_food_retry_across_utc_midnight_keeps_one_key(compact_db, monkeypatch):
    request = ChatRequest(message="记录午餐100克米饭", conversation_id="retry-food", request_id="retry-food-1")
    entries = [FoodInput(raw_text="100g米饭", requested_name="米饭", quantity=100, unit="g",
                         weight_g=100, portion_name=None, meal_type="午餐")]

    day_one = datetime(2026, 9, 1, 23, 59, tzinfo=timezone.utc)
    monkeypatch.setattr(clock, "utc_now", lambda: day_one)
    first = RetryScenario(monkeypatch, compact_db, parsed_input=record_plan(request.message, "nutrition_record"),
                          food_entries=entries, fail={"nutrition": True})
    with pytest.raises(CommitUnknownError):
        asyncio.run(first.respond(request))
    with compact_db() as session:
        rows_after_first = session.execute(text("SELECT COUNT(*) FROM user_food_log")).scalar()
    assert rows_after_first == 1

    day_two = day_one + timedelta(minutes=2)
    monkeypatch.setattr(clock, "utc_now", lambda: day_two)
    second = RetryScenario(monkeypatch, compact_db, parsed_input=record_plan(request.message, "nutrition_record"),
                           food_entries=entries)
    response = asyncio.run(second.respond(request))
    with compact_db() as session:
        rows_after_second = session.execute(text("SELECT COUNT(*) FROM user_food_log")).scalar()
        keys = [row[0] for row in session.execute(text("SELECT idempotency_key FROM user_food_log")).all()]
    assert rows_after_second == 1
    assert keys and keys[0].startswith("food-task-")
    assert "20260901" not in keys[0]  # the key is a digest, not a raw date; stability comes from reuse
    assert "已记录" in response.answer
    # The retry re-reads the existing meal instead of inserting a new one.
    assert second.tool_calls.count("calculate_nutrition") == 1


# ---------------------------------------------------------------------------
# Ledger failure before dispatch: the save tool must not run
# ---------------------------------------------------------------------------

def test_plan_ledger_failure_blocks_workout_save(compact_db, monkeypatch):
    from app.services.memory import compact
    request = ChatRequest(message="记录卧推60kg，4组8次", conversation_id="ledger-plan-down",
                          request_id="ledger-plan-down-1")
    scenario = RetryScenario(monkeypatch, compact_db, parsed_input=record_plan(request.message),
                             workout_records=_workouts)

    def broken(*_args, **_kwargs):
        raise StorageUnavailableError("ledger down")

    monkeypatch.setattr(compact, "save_request_state", broken)
    response = asyncio.run(scenario.respond(request))
    assert "未执行" in response.answer
    assert "save_workout_record" not in scenario.tool_calls
    with compact_db() as session:
        assert session.execute(text("SELECT COUNT(*) FROM workout_records")).scalar() == 0


def test_ledger_failure_before_food_dispatch_blocks_calculation(compact_db, monkeypatch):
    from app.services.memory import compact
    request = ChatRequest(message="记录午餐100克米饭", conversation_id="ledger-food",
                          request_id="ledger-food-1")
    entries = [FoodInput(raw_text="100g米饭", requested_name="米饭", quantity=100, unit="g",
                         weight_g=100, portion_name=None, meal_type="午餐")]
    scenario = RetryScenario(monkeypatch, compact_db, parsed_input=record_plan(request.message, "nutrition_record"),
                             food_entries=entries)
    original = compact.save_request_state
    calls = {"count": 0}

    def flaky(request_id, state):
        calls["count"] += 1
        if calls["count"] > 1:  # plan save succeeds, task-parameter save fails
            raise StorageUnavailableError("ledger down")
        return original(request_id, state)

    monkeypatch.setattr(compact, "save_request_state", flaky)
    response = asyncio.run(scenario.respond(request))
    assert "未执行" in response.answer
    assert "calculate_nutrition" not in scenario.tool_calls
    assert "calculate_nutrition_batch" not in scenario.tool_calls
    with compact_db() as session:
        assert session.execute(text("SELECT COUNT(*) FROM user_food_log")).scalar() == 0


def test_reservation_ledger_failure_blocks_workout_save(compact_db, monkeypatch):
    from app.services.memory import compact
    request = ChatRequest(message="记录卧推60kg，4组8次", conversation_id="ledger-reserve",
                          request_id="ledger-reserve-1")
    scenario = RetryScenario(monkeypatch, compact_db, parsed_input=record_plan(request.message),
                             workout_records=_workouts)
    original = compact.save_request_state
    calls = {"count": 0}

    def flaky(request_id, state):
        calls["count"] += 1
        if calls["count"] > 2:  # plan and task parameters persist, the reservation does not
            raise StorageUnavailableError("ledger down")
        return original(request_id, state)

    monkeypatch.setattr(compact, "save_request_state", flaky)
    response = asyncio.run(scenario.respond(request))
    assert "未执行" in response.answer
    assert "save_workout_record" not in scenario.tool_calls
    with compact_db() as session:
        assert session.execute(text("SELECT COUNT(*) FROM workout_records")).scalar() == 0


def test_read_only_request_survives_ledger_failure(compact_db, monkeypatch):
    from app.services.memory import compact
    request = ChatRequest(message="卧推时肩胛骨应该收紧吗", conversation_id="ledger-readonly",
                          request_id="ledger-readonly-1")
    scenario = RetryScenario(monkeypatch, compact_db)

    def broken(*_args, **_kwargs):
        raise StorageUnavailableError("ledger down")

    monkeypatch.setattr(compact, "save_request_state", broken)
    response = asyncio.run(scenario.respond(request))
    assert "未执行" not in response.answer
    assert "账本" not in response.answer
    assert response.answer


# ---------------------------------------------------------------------------
# Two intentionally identical operations in one request
# ---------------------------------------------------------------------------

def test_two_identical_workout_tasks_are_not_merged(compact_db, monkeypatch):
    # Distinct spans keep two planned positions; the extracted record is the
    # same, so only the position-bearing operation id keeps them separate.
    request = ChatRequest(
        message="记录卧推60kg，4组8次，然后补记卧推60kg 4组8次",
        conversation_id="retry-identical", request_id="retry-identical-1",
    )
    records = lambda _message: [{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}]
    scenario = RetryScenario(monkeypatch, compact_db, parsed_input=record_plan(request.message),
                             workout_records=records)
    response = asyncio.run(scenario.respond(request))
    assert len(scenario.saves) == 2
    assert len({record["id"] for record in scenario.saves}) == 2
    with compact_db() as session:
        assert session.execute(text("SELECT COUNT(*) FROM workout_records")).scalar() == 2
    assert "已记录" in response.answer


def test_two_identical_workout_tasks_retry_completes_each_once(compact_db, monkeypatch):
    request = ChatRequest(
        message="记录卧推60kg，4组8次，然后补记卧推60kg 4组8次",
        conversation_id="retry-identical-partial", request_id="retry-identical-partial-1",
    )
    records = lambda _message: [{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}]
    first = RetryScenario(monkeypatch, compact_db, parsed_input=record_plan(request.message),
                          workout_records=records)
    original = first.agent.graph._call_mcp
    calls = {"saves": 0}

    async def fail_second_save(name, arguments, **kwargs):
        if name == "save_workout_record":
            calls["saves"] += 1
            if calls["saves"] == 2:
                raise StorageUnavailableError("second save unavailable")
        return await original(name, arguments, **kwargs)

    monkeypatch.setattr(first.agent.graph, "_call_mcp", fail_second_save)
    with pytest.raises(StorageUnavailableError):
        asyncio.run(first.respond(request))
    assert [record["exercise"] for record in first.saves] == ["卧推"]
    with compact_db() as session:
        assert session.execute(text("SELECT COUNT(*) FROM workout_records")).scalar() == 1

    second = RetryScenario(monkeypatch, compact_db, parsed_input=record_plan(request.message),
                           workout_records=records)
    response = asyncio.run(second.respond(request))
    assert len(second.saves) == 1  # only the missing second operation is dispatched
    with compact_db() as session:
        rows = session.execute(text("SELECT exercise FROM workout_records ORDER BY id")).all()
    assert [row[0] for row in rows] == ["卧推", "卧推"]
    assert "已记录" in response.answer


def test_two_identical_food_tasks_are_not_merged(compact_db, monkeypatch):
    request = ChatRequest(
        message="记录午餐100克米饭，然后补记午餐100克米饭",
        conversation_id="retry-food-identical", request_id="retry-food-identical-1",
    )
    entries = [FoodInput(raw_text="100g米饭", requested_name="米饭", quantity=100, unit="g",
                         weight_g=100, portion_name=None, meal_type="午餐")]
    scenario = RetryScenario(monkeypatch, compact_db, parsed_input=record_plan(request.message, "nutrition_record"),
                             food_entries=entries)
    response = asyncio.run(scenario.respond(request))
    assert scenario.tool_calls.count("calculate_nutrition") == 2
    with compact_db() as session:
        assert session.execute(text("SELECT COUNT(*) FROM user_food_log")).scalar() == 2
    assert "已记录" in response.answer
