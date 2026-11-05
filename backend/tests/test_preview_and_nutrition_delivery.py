"""Tests for preview and nutrition delivery."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "plan_dialogue"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _plan_raw(minutes: int = 30) -> str:
    return json.dumps({
        "sessions": [
            {
                "label": "周二 全身A",
                "warmup": {"seconds": 300, "activity": "动态热身"},
                "main": {"seconds": minutes * 60 - 600,
                         "exercises": [{"name": "哑铃深蹲", "sets": 3, "reps": 10},
                                       {"name": "哑铃划船", "sets": 3, "reps": 10}]},
                "breaks": {"seconds": 240},
                "cooldown": {"seconds": 60, "activity": "拉伸"},
                "time_consistency": "consistent",
            },
            {
                "label": "周六 全身B",
                "warmup": {"seconds": 300, "activity": "动态热身"},
                "main": {"seconds": minutes * 60 - 600,
                         "exercises": [{"name": "哑铃卧推", "sets": 3, "reps": 10},
                                       {"name": "哑铃硬拉", "sets": 3, "reps": 8}]},
                "breaks": {"seconds": 240},
                "cooldown": {"seconds": 60, "activity": "拉伸"},
                "time_consistency": "consistent",
            },
        ]
    }, ensure_ascii=False)


def _preview_understanding(source: str) -> dict:
    return {
        "tasks": [{
            "task_type": "training_plan", "action": "question", "text": source,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            "plan_result": True, "result_scope": "independent",
            "plan_request_quote": "给这两天的完整预览", "goal_role": "result",
            "quoted": False, "history_requirement": "none", "fact_text": None,
            "reply_mode": "grounded", "reply_purpose": "general",
            "output_format": "default", "delivery_type": "plan",
        }],
        "reply_mode": "grounded", "reply_purpose": "general",
        "output_format": "default", "relation": "new",
        "current_is_critical": False, "needs_clarification": False,
        "safety": {"level": "normal", "evidence": "无伤病", "categories": ["negated"],
                   "reason": "用户明确否认伤病，无当前症状或医疗风险。"},
    }


def _plan_message(raw: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{
        "type": "function",
        "function": {"name": "return_timed_training_plan", "arguments": raw},
    }]}


def _hits():
    from app.services.rag.retrieval import RetrievedChunk
    return [RetrievedChunk("训练", "循序渐进，规律训练。", .95, "official",
                           {"rerank_score": .95, "rerank_status": "model"}, "synthetic")]


async def _plan_information(_llm, state):
    text = str(state.get("task_text") or state.get("raw_message") or "")
    minutes = 25 if "25分钟" in text else 30
    quote = f"每次最多{minutes}分钟"
    return {
        "scope": "session",
        "known": {
            "goal": "基础力量",
            "experience": "新手",
            "equipment": "哑铃",
            "session_time": quote,
            "available_days": "周二和周六",
        },
        "session_time_seconds": minutes * 60,
        "available_days_count": 2,
        "phase_limits": [],
    }, None


def _preview_request(message: str, conversation: str, request_id: str):
    from app.schemas.chat import ChatRequest
    return ChatRequest(user_id="demo-user", conversation_id=conversation,
                       request_id=request_id, message=message, mode="legacy")


def _planned_tasks(source: str, checked: dict):
    from app.services.agent import semantics, task_planner

    token = semantics.request_understanding.set({source: checked})
    try:
        return task_planner.plan_tasks_rule(source)
    finally:
        semantics.request_understanding.reset(token)


def _decision(tasks):
    from app.services.agent.execution_state import PlannerDecision
    from app.services.agent.intent import IntentResult
    return PlannerDecision(IntentResult("plan_adjust", 1.0, action="question"), {}, tuple(tasks))


def test_plan_preview_artifact_flows_to_metrics_app_meta_and_followup(compact_db, monkeypatch):
    from app.core import clock
    from app.services.agent import semantics
    from app.services.agent.agent import FitnessAgent
    from app.services.agent.plan_preview import latest_plan_preview
    from app.services.memory import compact

    preview = _fixture("plan_preview")
    source = preview["source_message"]
    raw = _preview_understanding(source)
    checked = semantics.validate_understanding(
        source, raw, context={"current_plan_preview": {"status": "none"}},
        absorb_background_plans=True)
    sem_token = semantics.request_understanding.set({})
    try:
        semantics.remember_understanding(source, checked)
        semantics.remember_safety(source, raw["safety"])
        tasks = _planned_tasks(source, checked)
        agent = FitnessAgent()
        agent._route = AsyncMock(return_value=_decision(tasks))
        monkeypatch.setattr("app.services.agent.task_information.check_plan_information",
                            AsyncMock(side_effect=_plan_information))
        monkeypatch.setattr(agent.graph, "_readonly_retrieval", AsyncMock(return_value=_hits()))
        monkeypatch.setattr(agent.graph.llm, "complete_message",
                            AsyncMock(return_value=_plan_message(_plan_raw(30))))

        first = _preview_request(source, "r5-preview", "r5-preview-1")
        first_turn, _ = compact.begin_turn(first)
        turn_token = compact.current_turn.set(first_turn)
        try:
            response = asyncio.run(agent._respond_impl(first))
        finally:
            compact.current_turn.reset(turn_token)
        compact.finish_turn(first_turn, response.answer, response.model_dump())

        artifact = (response.metrics or {}).get("plan_preview")
        assert isinstance(artifact, dict) and artifact.get("days")
        assert artifact["expected_sessions"] == 2
        assert [day["minutes"] for day in artifact["days"]] == [30, 30]
        from app.db.memory_models import MemoryTurn
        with compact.session() as session:
            stored_turn = session.get(MemoryTurn, first_turn)
            stored_meta = (stored_turn.messages[-1].get("app_meta") or {}).get("plan_preview")
        assert stored_meta["preview_id"] == artifact["preview_id"]
        loaded = latest_plan_preview("demo-user", "r5-preview")
        assert loaded["status"] == "ok"
        assert loaded["state"]["preview_id"] == artifact["preview_id"]

        follow = _fixture("plan_followup")
        follow_source = follow["source_message"]
        follow_raw = json.loads(follow["items"][0]["arguments"])
        follow_checked = semantics.validate_understanding(
            follow_source, follow_raw,
            context={"current_plan_preview": {"status": "ok", "state": loaded["state"]}},
            absorb_background_plans=True)
        semantics.remember_understanding(follow_source, follow_checked)
        semantics.remember_safety(follow_source, follow_raw["safety"])
        follow_tasks = _planned_tasks(follow_source, follow_checked)
        agent._route = AsyncMock(return_value=_decision(follow_tasks))
        from app.services.agent.dialogue_context import DialogueAnswer
        monkeypatch.setattr(agent.graph.llm, "configured", lambda: True)
        monkeypatch.setattr(agent.graph.llm, "complete_structured",
                            AsyncMock(return_value=DialogueAnswer(
                                answer="需要按上一轮预览生成训练安排。",
                                requires_grounding=True)))
        monkeypatch.setattr(agent.graph.llm, "complete_message",
                            AsyncMock(return_value=_plan_message(_plan_raw(25))))

        second = _preview_request(follow_source, "r5-preview", "r5-preview-2")
        second_turn, _ = compact.begin_turn(second)
        turn_token = compact.current_turn.set(second_turn)
        try:
            follow_response = asyncio.run(agent._respond_impl(second))
        finally:
            compact.current_turn.reset(turn_token)
        compact.finish_turn(second_turn, follow_response.answer, follow_response.model_dump())

        assert "没有成功生成" not in follow_response.answer
        assert "25分钟" in follow_response.answer
        follow_artifact = (follow_response.metrics or {}).get("plan_preview")
        assert isinstance(follow_artifact, dict)
        assert follow_artifact["session_minutes"] == 25
        assert follow_artifact["base_preview_id"] == artifact["preview_id"]
        assert latest_plan_preview("demo-user", "r5-preview")["state"]["preview_id"] == follow_artifact["preview_id"]
    finally:
        semantics.request_understanding.reset(sem_token)


def test_nutrition_advice_workers_keep_own_question_and_finalize_labels(monkeypatch):
    from app.services.agent import semantics, task_planner
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph

    fixture = _fixture("nutrition_advice")
    source = fixture["source_message"]
    raw = json.loads(fixture["items"][0]["arguments"])
    checked = semantics.validate_understanding(source, raw, context={})
    token = semantics.request_understanding.set({source: checked})
    try:
        tasks = task_planner.plan_tasks_rule(source)
    finally:
        semantics.request_understanding.reset(token)
    assert len(tasks) == 2
    first, second = tasks[0]["text"], tasks[1]["text"]

    graph = FitnessMultiAgentGraph()
    base = {
        "raw_message": source, "task_type": "nutrition_advice",
        "task_context": [], "shared_conditions": [], "current_plan_preview": {},
        "profile": {}, "short_memory": [], "retrieved_history": [], "long_memory": {},
        "semantic_memory": [], "memory_summary": "", "training_history": [],
        "nutrition_summary": {}, "persistent_constraints": {}, "current_states": {},
        "source_availability": {}, "message_importance": {}, "memory_backend": "x",
        "intent_result": {},
    }
    prompts = []
    for task_text in (first, second):
        messages = graph._planning_prompt(
            "饮食营养方案", {**base, "task_text": task_text}, [])
        prompts.append("\n".join(item["content"] for item in messages))
    assert second not in prompts[0]
    assert first not in prompts[1]
    assert source not in prompts[0] and source not in prompts[1]

    planned = [dict(task, status="pending") for task in tasks]
    results = [
        {"task_id": tasks[0]["id"], "task_type": "nutrition_advice",
         "worker": "nutrition", "status": "complete", "answer": "蛋白粉不是必需品。"},
        {"task_id": tasks[1]["id"], "task_type": "nutrition_advice",
         "worker": "nutrition", "status": "complete", "answer": "一半蔬菜、四分之一蛋白质。"},
    ]
    finalized = graph._finalize_node({
        "planned_tasks": planned, "worker_results": results,
        "tasks": [], "raw_message": source,
    })
    assert finalized["task_outcomes"] == [
        {"task_id": tasks[0]["id"], "task_type": "nutrition_advice", "status": "complete"},
        {"task_id": tasks[1]["id"], "task_type": "nutrition_advice", "status": "complete"},
    ]
    assert finalized["final_answer"].count("【第1问】") == 1
    assert finalized["final_answer"].count("【第2问】") == 1
    assert finalized["final_answer"].index("【第1问】") < finalized["final_answer"].index("【第2问】")


def _seed_oats(factory):
    from app.core import clock
    from sqlalchemy import text

    with factory.begin() as connection:
        connection.execute(text("""
            INSERT INTO user_food_log
                (user_id, meal_group_id, record_time, meal_type, raw_food_name, food_name,
                 weight_g, total_kcal, total_protein, total_carb, total_fat, total_fiber,
                 idempotency_key)
            VALUES ('demo-user', 'r5-group-oats', :record_time, '早餐', '燕麦', '燕麦',
                    45, 169.65, 5.67, 28.0, 3.5, 4.0, 'sim30-r5-oats')
        """), {"record_time": clock.utc_now().isoformat()})


def _nutrition_summary_understanding(source: str):
    from app.core import clock
    from app.services.agent import semantics

    raw = {
        "tasks": [{
            "task_type": "nutrition_record", "action": "question", "text": source,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            "goal_role": "result", "quoted": False, "history_requirement": "none",
            "fact_text": None, "reply_mode": "conversation",
            "reply_purpose": "fact_recall", "output_format": "default",
            "context_ids": [], "fact_fields": [], "critical_fact_fields": [],
            "record_query": {"date_start": clock.today().isoformat(),
                             "date_end": clock.today().isoformat()},
            "delivery_type": "read",
        }],
        "reply_mode": "conversation", "reply_purpose": "fact_recall",
        "output_format": "default", "relation": "new",
        "current_is_critical": False, "needs_clarification": False,
        "safety": {"level": "normal", "evidence": "", "categories": [],
                   "reason": "只读汇总，无医疗风险"},
    }
    return semantics.validate_understanding(source, raw, context={})


def test_nutrition_summary_execution_uses_deterministic_app_totals(compact_db, monkeypatch):
    from app.core.config import settings
    from app.schemas.chat import ChatRequest
    from app.services.agent import semantics
    from app.services.agent.agent import FitnessAgent

    monkeypatch.setattr(settings, "react_readonly_enabled", True)
    _seed_oats(compact_db)
    source = ("只根据我今天已经保存的饮食，告诉我总热量和蛋白质分别是多少；"
              "没有登记的餐不要估算或补记。")
    checked = _nutrition_summary_understanding(source)
    token = semantics.request_understanding.set({})
    try:
        semantics.remember_understanding(source, checked)
        semantics.remember_safety(source, checked["safety"])
        agent = FitnessAgent()
        agent._route = AsyncMock(side_effect=AssertionError("deterministic totals must not route"))
        response = asyncio.run(agent._respond_impl(ChatRequest(
            user_id="demo-user", conversation_id="r5-nutrition-summary",
            request_id="r5-nutrition-summary-1", message=source,
            mode="react_readonly")))
    finally:
        semantics.request_understanding.reset(token)

    answer = response.answer
    assert answer.startswith("合计：热量 169.7 kcal，蛋白质 5.7 g")
    assert "燕麦 45克" in answer
    assert "还差" not in answer and "目标" not in answer
    assert "补录" not in answer
    assert response.metrics["react_status"] == "deterministic_nutrition_totals"
    assert response.model_used is False
    agent._route.assert_not_awaited()


def test_nutrition_summary_deterministic_pg(monkeypatch, tmp_path):
    from app.core import clock
    from app.core.config import settings
    from app.schemas.chat import ChatRequest
    from app.services.agent import semantics
    from app.services.agent.agent import FitnessAgent
    from sqlalchemy import text
    from tests.react_phase_c1_support import isolated

    with isolated(monkeypatch, tmp_path, "postgres"):
        with monkeypatch.context() as scoped:
            scoped.setattr(settings, "react_readonly_enabled", True)
            from app.db import session as db
            with db.SessionLocal.begin() as connection:
                connection.execute(text("""
                    INSERT INTO user_food_log
                        (user_id, meal_group_id, record_time, meal_type, raw_food_name, food_name,
                         weight_g, total_kcal, total_protein, total_carb, total_fat, total_fiber,
                         idempotency_key)
                    VALUES ('demo-user', 'r5-pg-group-oats', CAST(:record_time AS TIMESTAMPTZ),
                            '早餐', '燕麦', '燕麦', 45, 169.65, 5.67, 28.0, 3.5, 4.0,
                            'sim30-r5-pg-oats')
                """), {"record_time": clock.utc_now()})
            source = ("只根据我今天已经保存的饮食，告诉我总热量和蛋白质分别是多少；"
                      "没有登记的餐不要估算或补记。")
            checked = _nutrition_summary_understanding(source)
            token = semantics.request_understanding.set({})
            try:
                semantics.remember_understanding(source, checked)
                semantics.remember_safety(source, checked["safety"])
                agent = FitnessAgent()
                agent._route = AsyncMock(side_effect=AssertionError(
                    "deterministic totals must not route"))
                response = asyncio.run(agent._respond_impl(ChatRequest(
                    user_id="demo-user", conversation_id="r5-pg-nutrition-summary",
                    request_id="r5-pg-nutrition-summary-1", message=source,
                    mode="react_readonly")))
            finally:
                semantics.request_understanding.reset(token)
            assert response.answer.startswith("合计：热量 169.7 kcal，蛋白质 5.7 g")
            assert "还差" not in response.answer and "补录" not in response.answer
            assert response.metrics["react_status"] == "deterministic_nutrition_totals"
            agent._route.assert_not_awaited()


def test_nutrition_summary_empty_does_not_pad_zero(compact_db):
    from datetime import timedelta

    from app.core import clock
    from app.services.agent.dialogue_context import saved_record_answer
    from types import SimpleNamespace

    _seed_oats(compact_db)
    request = SimpleNamespace(user_id="demo-user", message="告诉我总热量和蛋白质")
    missing = clock.today() - timedelta(days=2)
    query = {"date_start": missing.isoformat(), "date_end": missing.isoformat()}
    result = asyncio.run(saved_record_answer(
        None, request, "nutrition_record", query, select_all=True))
    assert "没有找到" in result["answer"]
    assert "合计" not in result["answer"]
    assert "0 kcal" not in result["answer"]
