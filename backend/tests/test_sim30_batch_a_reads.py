"""SIM30 batch A: profile/plan reads, exact record rendering, recovery events.

All scenarios use synthetic rows in pytest's temporary SQLite database. No
model, embedding, rerank or real user data is used.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from app.core import clock


def _set(weight: float, reps: int, **extra):
    return {"weight_kg": weight, "reps": reps, **extra}


def _session(title: str, exercises: list[dict], user_id: str = "demo-user", when=None):
    from app.db.workout_session_repository import create_session

    return create_session({
        "user_id": user_id,
        "session_date": when or clock.today(),
        "title": title,
        "note": "",
        "exercises": exercises,
    })


def _request(message: str):
    from app.schemas.chat import ChatRequest

    return ChatRequest(user_id="demo-user", conversation_id="sim30-a",
                       request_id="sim30-a", message=message, mode="react_readonly")


def test_profile_read_spec_renders_stored_values_only(compact_db):
    from app.db import profile_repository
    from app.services.agent.read_resolver import read_spec_answer

    profile_repository.save_profile("demo-user", {
        "height_cm": 175, "body_weight_kg": 70, "target_weight_kg": 68,
        "available_days": 3, "body_fat_pct": None,
    })
    result = read_spec_answer("demo-user", {
        "resource": "profile",
        "fields": ["height_cm", "body_weight_kg", "target_weight_kg", "available_days"],
    })
    assert result["authoritative"] is True
    assert result["model_used"] is False
    assert "175cm" in result["answer"] and "70kg" in result["answer"]
    assert "68kg" in result["answer"] and "3天" in result["answer"]

    missing = read_spec_answer("demo-user", {"resource": "profile", "fields": ["body_fat_pct"]})
    assert "未填写" in missing["answer"] and "体脂率" in missing["answer"]

    other = read_spec_answer("other-user", {"resource": "profile", "fields": ["height_cm"]})
    assert "没有找到你的档案记录" in other["answer"]
    assert "保持健康" not in other["answer"]


def test_saved_plan_read_spec_renders_stored_plan(compact_db):
    from app.db import plan_repository
    from app.services.agent.read_resolver import read_spec_answer

    plan_repository.save_plan("demo-user", {
        "plan_name": "一周三练基础计划", "goal": "提升基础力量", "days_per_week": 3,
        "sessions": [
            {"day": "周二", "name": "全身A", "items": ["高位下拉 3组×10次"], "status": "待开始"},
            {"day": "周六", "name": "全身C", "items": ["哑铃划船 2组×10次"], "status": "待开始"},
        ],
    })
    result = read_spec_answer("demo-user", {"resource": "training_plan"})
    assert result["authoritative"] is True
    assert "一周三练基础计划" in result["answer"]
    assert "周二" in result["answer"] and "高位下拉" in result["answer"]
    assert "周六" in result["answer"] and "哑铃划船" in result["answer"]

    other = read_spec_answer("other-user", {"resource": "training_plan"})
    assert "没有找到已保存的训练计划" in other["answer"]


def test_plan_read_fallback_is_not_background_or_new_plan(compact_db):
    from app.services.agent import semantics, dialogue_context
    from app.db import plan_repository

    source = "读取我已经保存的一周三练基础计划，告诉我安排在哪几天、各是什么训练；不要生成新计划，也不要修改。"
    raw = {
        "tasks": [{
            "task_type": "training_plan", "action": "question", "text": source,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            "plan_result": False, "reply_purpose": "fact_recall", "goal_role": "result",
            "reply_mode": "conversation",
        }],
        "reply_mode": "conversation", "reply_purpose": "fact_recall",
    }
    checked = semantics.validate_understanding(source, raw, absorb_background_plans=True)
    task = checked["tasks"][0]
    assert task["task_type"] == "training_plan"
    assert task["plan_result"] is False

    plan_repository.save_plan("demo-user", {
        "plan_name": "已存计划", "days_per_week": 2,
        "sessions": [{"day": "周三", "name": "全身", "items": ["深蹲 3组×8次"]}],
    })
    result = asyncio.run(dialogue_context.conversation_answer(
        None, _request(source), {}, checked))
    assert result and "已存计划" in result["answer"]
    assert "周三" in result["answer"] and "深蹲" in result["answer"]
    assert result["model_used"] is False


def test_profile_read_spec_validates_and_routes_to_legacy(compact_db, monkeypatch):
    from app.services.agent import semantics, react_runtime
    from app.services.agent.task_models import TaskSpec
    from app.core.config import settings

    source = "我档案里的身高和当前体重是多少？只读档案。"
    raw = {
        "tasks": [{
            "task_type": "chitchat", "action": "question", "text": source,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            "reply_purpose": "fact_recall", "goal_role": "result", "reply_mode": "conversation",
            "read_spec": {"resource": "profile", "fields": ["height_cm", "body_weight_kg"]},
        }],
        "reply_mode": "conversation", "reply_purpose": "fact_recall",
    }
    checked = semantics.validate_understanding(source, raw)
    assert checked["tasks"][0]["read_spec"]["resource"] == "profile"

    spec = TaskSpec("task-1", "chitchat", "chitchat", "question", source, payload={
        "semantic": {**checked["tasks"][0], "read_spec": {"resource": "profile",
                                                          "fields": ["height_cm"]}},
    })
    monkeypatch.setattr(settings, "react_readonly_enabled", True)
    monkeypatch.setattr(settings, "memory_read_backend", "compact_v1")
    assert react_runtime.select_mode(_request(source), [spec.to_dict()]) == (
        "legacy", "read_spec_legacy")

    bad = {**raw, "tasks": [{**raw["tasks"][0],
                              "read_spec": {"resource": "profile", "fields": ["password"]}}]}
    with pytest.raises(semantics.SemanticValidationError):
        semantics.validate_understanding(source, bad)


def test_exact_record_query_renders_sql_rows_without_a_second_selector(compact_db, monkeypatch):
    from app.db.repositories import create_workout
    from app.services.agent import dialogue_context

    today = clock.today()
    create_workout("demo-user", "高位下拉", 3, 10, 35.0, date=today,
                   set_details=[_set(35, 10)] * 3)
    create_workout("demo-user", "坐姿推胸", 2, 12, 22.0, date=today,
                   set_details=[_set(22, 12)] * 2)

    async def forbidden(*args, **kwargs):
        raise AssertionError("selector must not run for exact SQL conditions")

    monkeypatch.setattr(dialogue_context, "select_source_excerpts", forbidden)
    result = asyncio.run(dialogue_context.saved_record_answer(
        None, _request("只查今天高位下拉"), "workout_record",
        {"date_start": today.isoformat(), "date_end": today.isoformat(),
         "exercises": ["高位下拉"], "meal_types": [], "food_names": []}))
    assert "35" in result["answer"] and "高位下拉" in result["answer"]
    assert "坐姿推胸" not in result["answer"]
    assert result["authoritative"] is True
    assert result["complete"] is True
    assert result["selection_complete"] is True
    assert result["answer"].startswith("已保存的记录：")


def test_recovery_counts_training_events_and_dedupes_mirrors(compact_db):
    from app.db.models import WorkoutRecordModel
    from app.db.repositories import create_workout, recovery_event_summary
    from app.db.storage import storage_session

    today = clock.today()
    _session("单会话八组", [{"exercise_name": "高位下拉", "sets": [_set(35, 10)] * 8}])
    assert recovery_event_summary("demo-user", today=today)["recent_events"] == 1

    _session("同日第二会话", [{"exercise_name": "划船", "sets": [_set(8, 10)] * 3}])
    assert recovery_event_summary("demo-user", today=today)["recent_events"] == 2

    create_workout("demo-user", "深蹲", 3, 10, 60.0, date=today,
                   set_details=[_set(60, 10)] * 3)
    assert recovery_event_summary("demo-user", today=today)["recent_events"] == 3

    with storage_session() as session:
        session.add(WorkoutRecordModel(
            user_id="demo-user", workout_date=today, exercise="腿举",
            sets=3, reps=12, weight_kg=80.0))
        session.commit()
    assert recovery_event_summary("demo-user", today=today)["recent_events"] == 4

    _session("未来会话", [{"exercise_name": "肩推", "sets": [_set(10, 10)]}],
             when=today + timedelta(days=2))
    summary = recovery_event_summary("demo-user", today=today)
    assert summary["recent_events"] == 4
    assert summary["last_workout_date"] == today

    old = today - timedelta(days=40)
    _session("旧会话", [{"exercise_name": "硬拉", "sets": [_set(50, 5)]}],
             user_id="old-user", when=old)
    old_summary = recovery_event_summary("old-user", today=today)
    assert old_summary["recent_events"] == 0
    assert old_summary["last_workout_date"] == old


def test_advice_read_spec_injects_sql_facts(compact_db, monkeypatch):
    from unittest.mock import AsyncMock

    from app.db.repositories import create_workout
    from app.services.agent import react_readonly_workers
    from app.services.context import budget as context_budget_module
    from app.services.rag import online as rag_online
    from app.services.rag import quality as rag_quality
    from app.services.agent.agent import agent

    today = clock.today()
    create_workout("demo-user", "高位下拉", 3, 10, 35.0, date=today,
                   set_details=[_set(35, 10)] * 3)

    async def fake_search(*args, **kwargs):
        from types import SimpleNamespace
        return [SimpleNamespace(title="恢复建议依据", source="test", chunk_id="1", text="恢复建议依据",
                                score=1.0, metadata={})]

    monkeypatch.setattr(rag_online, "search", fake_search)
    monkeypatch.setattr(rag_quality, "assess_rag_evidence",
                        lambda *args, **kwargs: type("Q", (), {"answerable": True})())

    async def fake_state(context, task_type):
        return {"task_context": []}

    monkeypatch.setattr(react_readonly_workers, "_answer_state", fake_state)
    captured = {}

    def fake_assemble(*, system, query, state, hits):
        captured["state"] = state
        return [{"role": "user", "content": "question"}]

    monkeypatch.setattr(context_budget_module.context_budget, "assemble", fake_assemble)
    monkeypatch.setattr(agent.graph.llm, "complete_message",
                        AsyncMock(return_value={"content": "今天已练高位下拉，建议充分休息。"}))

    spec = {"resource": "workout_records", "date_start": today.isoformat(),
            "date_end": today.isoformat(), "exercises": []}
    result = asyncio.run(react_readonly_workers._single_pass_answer(
        {"query": "按今天训练给恢复建议"}, {"user_id": "demo-user", "task_text": "按今天训练给恢复建议",
                                          "semantic_scope": {"read_spec": spec}},
        "training_qa", "system"))
    facts = [item for item in captured["state"]["task_context"]
             if isinstance(item, dict) and "read_spec_facts" in item]
    assert facts and facts[0]["read_spec_facts"]
    assert facts[0]["read_spec_facts"][0]["exercise"] == "高位下拉"
    assert result.status.value == "completed"


def test_advice_read_spec_routes_to_readonly_loop(compact_db, monkeypatch):
    from app.services.agent import react_runtime
    from app.services.agent.task_models import TaskSpec
    from app.core.config import settings

    monkeypatch.setattr(settings, "react_readonly_enabled", True)
    monkeypatch.setattr(settings, "memory_read_backend", "compact_v1")
    source = "按我今天训练给恢复建议"
    spec = TaskSpec("task-1", "training_qa", "training_qa", "question", source, payload={
        "semantic": {"task_type": "training_qa", "action": "question", "reply_purpose": "general",
                     "reply_mode": "grounded", "read_spec": {
                         "resource": "workout_records", "date_start": "2026-10-30",
                         "date_end": "2026-10-30", "exercises": []}},
    })
    assert react_runtime.select_mode(_request(source), [spec.to_dict()]) == (
        "react_readonly", "pure_read")


def test_profile_read_spec_executes_end_to_end(compact_db, monkeypatch):
    from unittest.mock import AsyncMock

    from app.core.config import settings
    from app.db import profile_repository
    from app.schemas.chat import ChatRequest
    from app.services.agent import semantics
    from app.services.agent.agent import FitnessAgent
    from app.services.agent.execution_state import PlannerDecision
    from app.services.agent.intent import IntentResult
    from app.services.agent.task_models import TaskSpec

    source = "我档案里的身高、当前体重、目标体重和每周可训练天数分别是多少？只读档案。"
    profile_repository.save_profile("demo-user", {
        "height_cm": 175, "body_weight_kg": 70, "target_weight_kg": 68, "available_days": 3})
    read_spec = {"resource": "profile",
                 "fields": ["height_cm", "body_weight_kg", "target_weight_kg", "available_days"]}
    raw = {"tasks": [{
        "task_type": "chitchat", "action": "question", "text": source,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        "reply_purpose": "fact_recall", "goal_role": "result", "reply_mode": "conversation",
        "history_requirement": "none",
        "read_spec": read_spec,
    }], "reply_mode": "conversation", "reply_purpose": "fact_recall",
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}}
    checked = semantics.validate_understanding(source, raw)
    spec = TaskSpec("task-1", "chitchat", "chitchat", "question", source,
                    payload={"semantic": checked["tasks"][0]}).to_dict()
    agent = FitnessAgent()
    agent._route = AsyncMock(return_value=PlannerDecision(
        IntentResult("chitchat", 1.0), {}, (spec,)))
    monkeypatch.setattr(settings, "react_readonly_enabled", True)
    monkeypatch.setattr(settings, "memory_read_backend", "compact_v1")
    token = semantics.request_understanding.set({})
    try:
        semantics.remember_understanding(source, checked)
        semantics.remember_safety(source, raw["safety"])
        response = asyncio.run(agent._respond_impl(
            ChatRequest(message=source, mode="react_readonly")))
    finally:
        semantics.request_understanding.reset(token)
    assert "175cm" in response.answer and "70kg" in response.answer
    assert "68kg" in response.answer and "3天" in response.answer
    assert response.model_used is False
    assert response.metrics["execution_mode"] == "legacy"
    assert response.metrics["mode_reason"] == "read_spec_legacy"


def test_plan_read_fallback_executes_end_to_end(compact_db, monkeypatch):
    from unittest.mock import AsyncMock

    from app.core.config import settings
    from app.db import plan_repository
    from app.schemas.chat import ChatRequest
    from app.services.agent import semantics
    from app.services.agent.agent import FitnessAgent
    from app.services.agent.execution_state import PlannerDecision
    from app.services.agent.intent import IntentResult
    from app.services.agent.task_models import TaskSpec

    source = "读取我已经保存的一周三练基础计划，告诉我安排在哪几天、各是什么训练；不要生成新计划。"
    plan_repository.save_plan("demo-user", {
        "plan_name": "一周三练基础计划", "days_per_week": 3,
        "sessions": [{"day": "周二", "name": "全身A", "items": ["高位下拉 3组×10次"]},
                     {"day": "周四", "name": "全身B", "items": ["徒手深蹲 2组×10次"]}],
    })
    raw = {"tasks": [{
        "task_type": "training_plan", "action": "question", "text": source,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        "plan_result": False, "reply_purpose": "fact_recall", "goal_role": "result",
        "reply_mode": "conversation", "history_requirement": "none",
    }], "reply_mode": "conversation", "reply_purpose": "fact_recall",
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}}
    checked = semantics.validate_understanding(source, raw)
    spec = TaskSpec("task-1", "training_plan", "plan_adjust", "question", source,
                    payload={"semantic": checked["tasks"][0]}).to_dict()
    agent = FitnessAgent()
    agent._route = AsyncMock(return_value=PlannerDecision(
        IntentResult("training_plan", 1.0), {}, (spec,)))
    monkeypatch.setattr(settings, "react_readonly_enabled", True)
    monkeypatch.setattr(settings, "memory_read_backend", "compact_v1")
    token = semantics.request_understanding.set({})
    try:
        semantics.remember_understanding(source, checked)
        semantics.remember_safety(source, raw["safety"])
        response = asyncio.run(agent._respond_impl(
            ChatRequest(message=source, mode="react_readonly")))
    finally:
        semantics.request_understanding.reset(token)
    assert "一周三练基础计划" in response.answer
    assert "周二" in response.answer and "周四" in response.answer
    assert response.metrics["mode_reason"] == "plan_read_legacy"


def test_saved_plan_model_output_is_accepted_as_a_read():
    import json
    from pathlib import Path
    from app.services.agent import semantics
    from app.services.agent.task_models import is_plan_read_task

    path = (Path(__file__).resolve().parents[2] / "evidence/user-simulation-20261030-full/"
            "live/synthetic-model-responses.jsonl")
    if not path.exists():
        pytest.skip("saved SIM30 model response is not present")
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[1])
    arguments = json.loads(record["messages"][0]["tool_calls"][0]["function"]["arguments"])
    source = arguments["tasks"][0]["text"]
    checked = semantics.validate_understanding(source, arguments, absorb_background_plans=True)
    task = checked["tasks"][0]
    assert task["task_type"] == "training_plan" and task["plan_result"] is False
    assert is_plan_read_task(task)


def test_recovery_summary_postgres(monkeypatch, tmp_path):
    from tests.react_phase_c1_support import isolated
    from app.db.models import WorkoutRecordModel
    from app.db.repositories import create_workout, recovery_event_summary
    from app.db.storage import storage_session
    from app.db import profile_repository, plan_repository
    from app.services.agent.read_resolver import read_spec_answer

    with isolated(monkeypatch, tmp_path, "postgres"):
        today = clock.today()
        _session("PG会话", [{"exercise_name": "高位下拉", "sets": [_set(35, 10)] * 8}])
        create_workout("demo-user", "深蹲", 3, 10, 60.0, date=today,
                       set_details=[_set(60, 10)] * 3)
        with storage_session() as session:
            session.add(WorkoutRecordModel(
                user_id="demo-user", workout_date=today, exercise="腿举",
                sets=3, reps=12, weight_kg=80.0))
            session.commit()
        summary = recovery_event_summary("demo-user", today=today)
        assert summary["recent_sessions"] == 2
        assert summary["recent_legacy_only"] == 1
        assert summary["recent_events"] == 3
        assert summary["last_workout_date"] == today

        profile_repository.save_profile("demo-user", {
            "height_cm": 175, "body_weight_kg": 70, "target_weight_kg": 68,
            "available_days": 3})
        plan_repository.save_plan("demo-user", {
            "plan_name": "PG计划", "days_per_week": 2,
            "sessions": [{"day": "周二", "name": "全身", "items": ["高位下拉 3组×10次"]}]})
        assert "175cm" in read_spec_answer("demo-user", {
            "resource": "profile", "fields": ["height_cm"]})["answer"]
        assert "PG计划" in read_spec_answer("demo-user", {
            "resource": "training_plan"})["answer"]


def test_recovery_api_reports_session_events(compact_db):
    from app.api.recovery import get_recovery

    _session("API恢复", [{"exercise_name": "高位下拉", "sets": [_set(35, 10)] * 8}])
    result = get_recovery("demo-user")
    assert result["recent_workouts"] == 1
    assert result["last_workout_date"] == clock.today().isoformat()
    assert result["sleep_hours"] is None and result["sleep_label"] == "未记录"


def test_recovery_does_not_truncate_at_thirty(compact_db):
    from app.db.repositories import recovery_event_summary

    today = clock.today()
    for index in range(35):
        _session(f"会话{index}", [{"exercise_name": "深蹲", "sets": [_set(20, 10)]}])
    summary = recovery_event_summary("demo-user", today=today)
    assert summary["recent_events"] == 35
    assert summary["recent_sessions"] == 35

