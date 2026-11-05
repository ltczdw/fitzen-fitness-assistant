"""SIM30 R4: six targeted product behaviours, not just function units."""
from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sim30_r2"
R4_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sim30_r4"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _r4_fixture(name: str) -> dict:
    return json.loads((R4_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _nutrition_planned_tasks():
    fixture = _fixture("nutrition_advice")
    raw = json.loads(fixture["items"][0]["arguments"])
    from app.services.agent import semantics, task_planner

    checked = semantics.validate_understanding(fixture["source_message"], raw, context={})
    token = semantics.request_understanding.set({})
    try:
        semantics.remember_understanding(fixture["source_message"], checked)
        return fixture, checked, task_planner.plan_tasks_rule(fixture["source_message"])
    finally:
        semantics.request_understanding.reset(token)


def test_nutrition_advice_keeps_two_tasks_and_scopes_each_prompt():
    fixture, checked, tasks = _nutrition_planned_tasks()
    assert len(checked["tasks"]) == 2
    assert len(tasks) == 2
    assert "蛋白粉" in tasks[0]["text"] and "日常食物" not in tasks[0]["text"]
    assert "日常食物" in tasks[1]["text"] and "蛋白粉" not in tasks[1]["text"]

    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph

    graph = object.__new__(FitnessMultiAgentGraph)
    state = {
        "raw_message": fixture["source_message"],
        "task_text": tasks[0]["text"],
        "task_type": "nutrition_advice",
        "task_context": [], "shared_conditions": [], "current_plan_preview": {},
        "profile": {}, "short_memory": [], "retrieved_history": [], "long_memory": {},
        "semantic_memory": [], "memory_summary": "", "training_history": [],
        "nutrition_summary": {}, "persistent_constraints": {}, "current_states": {},
        "source_availability": {}, "message_importance": {}, "memory_backend": "x",
        "intent_result": {},
    }
    messages = graph._planning_prompt("饮食营养方案", state, [])
    prompt = "\n".join(item["content"] for item in messages)
    assert "日常食物" not in prompt
    assert "只回答当前 task_text" in prompt


def _seed_oats(factory):
    from app.core import clock
    from sqlalchemy import text

    with factory.begin() as connection:
        connection.execute(text("""
            INSERT INTO user_food_log
                (user_id, meal_group_id, record_time, meal_type, raw_food_name, food_name,
                 weight_g, total_kcal, total_protein, total_carb, total_fat, total_fiber,
                 idempotency_key)
            VALUES ('demo-user', 'group-oats', :record_time, '早餐', '燕麦', '燕麦',
                    45, 169.65, 5.67, 28.0, 3.5, 4.0, 'sim30-r4-oats')
        """), {"record_time": clock.utc_now().isoformat()})


def test_nutrition_summary_adds_totals_from_sql(compact_db):
    from app.core import clock
    from app.services.agent.dialogue_context import saved_record_answer

    _seed_oats(compact_db)
    request = SimpleNamespace(user_id="demo-user", message="告诉我总热量和蛋白质")
    query = {"date_start": clock.today().isoformat(), "date_end": clock.today().isoformat(),
             "meal_types": ["早餐"], "food_names": [], "exercises": []}
    result = asyncio.run(saved_record_answer(None, request, "nutrition_record", query))
    answer = result["answer"]
    assert "合计：" in answer
    assert "169.7 kcal" in answer
    assert "5.7 g" in answer
    assert "燕麦 45克" in answer
    assert "nutrition_totals" in result


def test_nutrition_summary_totals_pg(monkeypatch, tmp_path):
    from app.core import clock
    from app.services.agent.dialogue_context import saved_record_answer
    from sqlalchemy import text
    from tests.react_phase_c1_support import isolated

    with isolated(monkeypatch, tmp_path, "postgres") as engine:
        with engine.begin() as connection:
            connection.execute(text("""
                INSERT INTO user_food_log
                    (user_id, meal_group_id, record_time, meal_type, raw_food_name, food_name,
                     weight_g, total_kcal, total_protein, total_carb, total_fat, total_fiber,
                     idempotency_key)
                VALUES ('demo-user', 'pg-group-oats', CAST(:record_time AS TIMESTAMPTZ), '早餐',
                        '燕麦', '燕麦', 45, 169.65, 5.67, 28.0, 3.5, 4.0, 'sim30-r4-pg-oats')
            """), {"record_time": clock.utc_now()})
        request = SimpleNamespace(user_id="demo-user", message="告诉我总热量和蛋白质")
        query = {"date_start": clock.today().isoformat(), "date_end": clock.today().isoformat(),
                 "meal_types": ["早餐"], "food_names": [], "exercises": []}
        result = asyncio.run(saved_record_answer(None, request, "nutrition_record", query))
        assert "合计：" in result["answer"]
        assert "169.7 kcal" in result["answer"]
        assert "5.7 g" in result["answer"]



def test_nutrition_summary_empty_does_not_pad_zero(compact_db):
    from app.core import clock
    from app.services.agent.dialogue_context import saved_record_answer

    _seed_oats(compact_db)
    request = SimpleNamespace(user_id="demo-user", message="告诉我总热量和蛋白质")
    missing = clock.today() - timedelta(days=2)
    query = {"date_start": missing.isoformat(), "date_end": missing.isoformat(),
             "meal_types": ["早餐"], "food_names": [], "exercises": []}
    result = asyncio.run(saved_record_answer(None, request, "nutrition_record", query))
    assert "没有找到" in result["answer"]
    assert "合计" not in result["answer"]
    assert "0 kcal" not in result["answer"]


def test_mixed_constraint_execution_status_is_normalized_not_rejected():
    from app.services.agent import semantics
    from app.services.agent.dialogue_context import status_only_request

    fixture = _r4_fixture("mixed_read_recovery")
    checked = semantics.validate_understanding(
        fixture["source_message"], fixture["arguments"], context={})
    tasks = checked["tasks"]
    assert [task["task_type"] for task in tasks] == [
        "workout_record", "training_qa", "chitchat"]
    constraint = tasks[2]
    assert constraint["goal_role"] == "constraint"
    assert constraint["is_negated"] is True
    assert constraint["reply_purpose"] == "general"
    assert status_only_request(checked) is False
    assert len(checked["validated_tasks"]) == 3


def test_real_result_status_keeps_execution_status():
    from app.services.agent import semantics
    from app.services.agent.dialogue_context import status_only_request

    source = "这次训练到底保存成功了吗？"
    raw = {"tasks": [{
        "task_type": "chitchat", "action": "question", "text": source,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        "goal_role": "result", "quoted": False, "history_requirement": "none",
        "reply_mode": "conversation", "reply_purpose": "execution_status",
        "delivery_type": "capability",
    }], "reply_mode": "conversation", "reply_purpose": "execution_status",
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}}
    checked = semantics.validate_understanding(source, raw, context={})
    assert checked["tasks"][0]["reply_purpose"] == "execution_status"
    assert status_only_request(checked) is True


class _SemanticLLM:
    def __init__(self, raw):
        self.raw = raw

    def configured(self):
        return True

    async def complete_structured(self, messages, schema, *, name, temperature, validator):
        value = schema.model_validate(self.raw)
        validator(value)
        return value


def test_plan_save_capability_skips_wiki_prepare(monkeypatch):
    from app.services.agent import semantics
    from app.services.agent.semantics import RequestUnderstanding

    fixture = _fixture("plan_save_boundary")
    raw = json.loads(fixture["items"][1]["arguments"])
    ids = list(raw["tasks"][0].get("context_ids") or [])
    context = {
        "current_plan_preview": {"status": "none"},
        "short_memory": [{"role": "user", "message_id": item, "content": "x"} for item in ids],
    }

    async def forbidden(*args, **kwargs):
        raise AssertionError("wiki prepare must not run for a plan-save capability request")

    from app.services.memory import wiki_context
    monkeypatch.setattr(wiki_context, "prepare", forbidden)
    monkeypatch.setattr(wiki_context, "prepare_cached", forbidden)
    llm = _SemanticLLM(raw)
    token = semantics.request_understanding.set({})
    try:
        result = asyncio.run(semantics.understand(llm, fixture["source_message"], context=context))
        assert result is not None
        assert semantics.semantic_attempt_failed(fixture["source_message"]) is False
    finally:
        semantics.request_understanding.reset(token)
    assert RequestUnderstanding.model_validate(raw)


def _preseed(source: str, checked: dict, safety: dict):
    from app.services.agent import semantics
    semantics.remember_understanding(source, checked)
    semantics.remember_safety(source, safety)


def test_plan_save_capability_answers_before_route(compact_db, monkeypatch):
    from unittest.mock import AsyncMock

    from app.schemas.chat import ChatRequest
    from app.services.agent import semantics
    from app.services.agent.agent import FitnessAgent

    fixture = _fixture("plan_save_boundary")
    raw = json.loads(fixture["items"][1]["arguments"])
    ids = list(raw["tasks"][0].get("context_ids") or [])
    context = {
        "current_plan_preview": {"status": "none"},
        "short_memory": [{"role": "user", "message_id": item, "content": "x"} for item in ids],
    }
    checked = semantics.validate_understanding(
        fixture["source_message"], raw, context=context, absorb_background_plans=True)
    agent = FitnessAgent()
    agent._route = AsyncMock(side_effect=AssertionError("planner/plan worker must not run"))
    token = semantics.request_understanding.set({})
    try:
        _preseed(fixture["source_message"], checked, raw["safety"])
        response = asyncio.run(agent._respond_impl(ChatRequest(
            user_id="demo-user", conversation_id="r4-plan-save",
            request_id="r4-plan-save-1", message=fixture["source_message"], mode="legacy")))
    finally:
        semantics.request_understanding.reset(token)
    from app.services.agent.capability_catalog import capability_copy
    assert response.answer == capability_copy("plan_save")
    assert "当前对话没有保存训练计划的工具" in response.answer
    assert response.metrics["mode_reason"] == "capability_catalog"
    agent._route.assert_not_awaited()


def test_preview_without_state_returns_deterministic_copies(compact_db):
    from unittest.mock import AsyncMock

    from app.schemas.chat import ChatRequest
    from app.services.agent import semantics
    from app.services.agent.agent import FitnessAgent

    fixture = _fixture("plan_save_boundary")
    raw = json.loads(fixture["items"][0]["arguments"])
    context = {"current_plan_preview": {"status": "none"}}
    checked = semantics.validate_understanding(
        fixture["source_message"], raw, context=context, absorb_background_plans=True)
    agent = FitnessAgent()
    agent._route = AsyncMock(side_effect=AssertionError("plan worker must not run"))
    token = semantics.request_understanding.set({})
    try:
        _preseed(fixture["source_message"], checked, raw["safety"])
        response = asyncio.run(agent._respond_impl(ChatRequest(
            user_id="demo-user", conversation_id="r4-preview-none",
            request_id="r4-preview-none-1", message=fixture["source_message"], mode="legacy")))
    finally:
        semantics.request_understanding.reset(token)
    from app.services.agent.capability_catalog import capability_copy
    from app.services.agent.plan_preview import NO_PREVIEW_COPY
    assert response.answer.startswith(NO_PREVIEW_COPY)
    assert capability_copy("plan_save") in response.answer
    assert response.metrics["plan_preview_status"] == "none"
    agent._route.assert_not_awaited()


def test_semantic_failure_code_is_exposed_with_explicit_preview_failure():
    from app.schemas.chat import ChatRequest
    from app.services.agent.agent import FitnessAgent

    response = FitnessAgent._safety_unavailable_response(
        ChatRequest(user_id="demo-user", conversation_id="r4-bad-json",
                    request_id="r4-bad-json-1", message="给两天预览", mode="legacy"),
        "unknown", {"semantic_failure_code": "structured_invalid_json"})
    assert response.metrics["semantic_failure_code"] == "structured_invalid_json"
    assert response.answer.startswith("本次请求未完成安全语义判断")
    assert "无法生成或修改训练计划/预览" in response.answer
    assert "没有保存任何数据" in response.answer
    assert "胸痛" in response.answer
