"""Tests for workout read consistency."""
from __future__ import annotations

from datetime import timedelta

from app.core import clock


def _set(weight: float, reps: int, **extra):
    return {"weight_kg": weight, "reps": reps, **extra}


def _session(title: str, exercises: list[dict], user_id: str = "demo-user",
             when=None):
    from app.db.workout_session_repository import create_session

    return create_session({
        "user_id": user_id,
        "session_date": when or clock.today(),
        "title": title,
        "note": "",
        "exercises": exercises,
    })


def test_session_only_training_is_read_by_agent_and_keeps_each_set(compact_db):
    from app.db.repositories import query_workout_read_rows

    today = clock.today()
    _session("今天训练", [
        {"exercise_name": "高位下拉", "sets": [
            _set(35, 10), _set(36, 9), _set(37, 8)]},
        {"exercise_name": "坐姿推胸", "sets": [_set(22, 12), _set(22, 12)]},
    ])
    rows, truncated = query_workout_read_rows(
        "demo-user", start_date=today, end_date=today, limit=30)
    assert truncated is False
    assert {row["exercise"] for row in rows} == {"高位下拉", "坐姿推胸"}
    pulldown = next(row for row in rows if row["exercise"] == "高位下拉")
    assert pulldown["source"] == "session"
    assert pulldown["sets"] == 3
    assert [item["weight_kg"] for item in pulldown["set_details"]] == [35, 36, 37]
    assert [item["reps"] for item in pulldown["set_details"]] == [10, 9, 8]
    assert pulldown["summary_mode"] == "first_set"

    filtered, _ = query_workout_read_rows(
        "demo-user", start_date=today, end_date=today,
        exercises=["高位下拉"], limit=30)
    assert [row["exercise"] for row in filtered] == ["高位下拉"]

    other, _ = query_workout_read_rows(
        "another-user", start_date=today, end_date=today, limit=30)
    assert other == []


def test_mirrored_agent_record_is_counted_once(compact_db):
    from app.db.repositories import create_workout, query_workout_read_rows

    today = clock.today()
    created = create_workout(
        "demo-user", "哑铃划船", 3, 10, 8.0, date=today,
        set_details=[_set(8, 10), _set(8, 10), _set(8, 10)],
    )
    assert created
    rows, _ = query_workout_read_rows(
        "demo-user", start_date=today, end_date=today, limit=30)
    matches = [row for row in rows if row["exercise"] == "哑铃划船"]
    assert len(matches) == 1
    assert matches[0]["source"] == "record"
    assert matches[0]["source_id"] == f"record:{matches[0]['id']}"


def test_independent_multi_action_session_is_not_collapsed(compact_db):
    from app.db.repositories import query_workout_read_rows

    today = clock.today()
    _session("全身", [
        {"exercise_name": "深蹲", "sets": [_set(50, 10)]},
        {"exercise_name": "卧推", "sets": [_set(40, 10)]},
        {"exercise_name": "划船", "sets": [_set(30, 12)]},
    ])
    rows, _ = query_workout_read_rows(
        "demo-user", start_date=today, end_date=today, limit=30)
    assert {row["exercise"] for row in rows} == {"深蹲", "卧推", "划船"}


def test_unlinked_lookalike_session_action_is_not_merged(compact_db):
    from app.db.repositories import create_workout, query_workout_read_rows

    today = clock.today()
    create_workout("demo-user", "高位下拉", 3, 10, 35.0, date=today)
    _session("表单", [{"exercise_name": "高位下拉", "sets": [_set(35, 10)]}])
    rows, _ = query_workout_read_rows(
        "demo-user", start_date=today, end_date=today, limit=30)
    matches = [row for row in rows if row["exercise"] == "高位下拉"]
    # No explicit source_record_id relation: never dedupe by name or by the
    # whole-session shape.  Both representations remain visible; the residual
    # duplicate risk is documented rather than guessed away.
    assert len(matches) == 2
    assert {row["source"] for row in matches} == {"record", "session"}


def test_saved_workout_summary_includes_record_and_session_actions(compact_db):
    from app.db.repositories import create_workout
    from app.services.agent.react_readonly_workers import saved_workout_observation

    today = clock.today()
    create_workout("demo-user", "坐姿推胸", 2, 12, 22.0, date=today)
    _session("今天训练", [
        {"exercise_name": "高位下拉", "sets": [_set(35, 10), _set(35, 10)]},
    ])
    candidate = {
        "id": "goal-summary", "task_type": "workout_record", "action": "question",
        "text": "我今天已经登记了哪两项训练？分别有几组、每组多少次？",
        "payload": {"dialogue": {"record_query": {
            "date_start": today.isoformat(), "date_end": today.isoformat(),
        }}},
    }
    observation = saved_workout_observation(candidate, {"user_id": "demo-user"})
    assert observation.status == "completed"
    assert "高位下拉" in observation.summary
    assert "坐姿推胸" in observation.summary
    assert len(observation.result["workouts"]) == 2


def test_saved_workout_read_failure_is_a_real_blocked_state(compact_db, monkeypatch):
    from app.db import repositories
    from app.services.agent.react_readonly_workers import saved_workout_observation

    today = clock.today()

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic read failure")

    monkeypatch.setattr(repositories, "query_workout_read_rows", fail)
    candidate = {
        "id": "goal-read", "task_type": "workout_record", "action": "question",
        "text": "查今天的训练",
        "payload": {"dialogue": {"record_query": {
            "date_start": today.isoformat(), "date_end": today.isoformat()}}},
    }
    observation = saved_workout_observation(candidate, {"user_id": "demo-user"})
    assert observation.status == "blocked_capability"
    assert observation.error == "saved_record_read_failed"
    assert "失败" in observation.summary


def test_cardio_session_facts_are_preserved(compact_db):
    from app.db.repositories import query_workout_read_rows

    today = clock.today()
    _session("有氧", [{"exercise_name": "跑步", "sets": [
        {"weight_kg": 0, "reps": 0, "distance_km": 5.0, "duration_minutes": 30},
    ]}])
    rows, _ = query_workout_read_rows(
        "demo-user", start_date=today, end_date=today, limit=30)
    cardio = next(row for row in rows if row["exercise"] == "跑步")
    detail = cardio["set_details"][0]
    assert detail["distance_km"] == 5.0
    assert detail["duration_minutes"] == 30


def test_incomplete_set_is_not_hidden_by_uniform_summary(compact_db):
    from app.services.agent.react_readonly_workers import saved_workout_observation

    today = clock.today()
    _session("今天训练", [{"exercise_name": "高位下拉", "sets": [
        _set(35, 10), _set(35, 10, completed=False)]}])
    candidate = {
        "id": "goal-read", "task_type": "workout_record", "action": "question",
        "text": "查今天高位下拉",
        "payload": {"dialogue": {"record_query": {
            "date_start": today.isoformat(), "date_end": today.isoformat(),
            "exercises": ["高位下拉"]}}},
    }
    observation = saved_workout_observation(candidate, {"user_id": "demo-user"})
    assert observation.status == "completed"
    assert "未完成" in observation.summary


def test_global_limit_reports_truncation(compact_db):
    from app.db.repositories import query_workout_read_rows

    today = clock.today()
    _session("一", [{"exercise_name": "深蹲", "sets": [_set(50, 10)]}])
    _session("二", [{"exercise_name": "卧推", "sets": [_set(40, 10)]}])
    rows, truncated = query_workout_read_rows(
        "demo-user", start_date=today, end_date=today, limit=1)
    assert len(rows) == 1
    assert truncated is True


def test_saved_workout_fallback_completes_and_finalize_delivers(compact_db):
    from app.services.agent.react_graph import finalize_node
    from app.services.agent.react_readonly_workers import saved_workout_observation

    today = clock.today()
    _session("今天训练", [
        {"exercise_name": "高位下拉", "sets": [_set(35, 10), _set(35, 10)]},
        {"exercise_name": "坐姿推胸", "sets": [_set(22, 12)]},
    ])
    candidate = {
        "id": "goal-read",
        "task_type": "workout_record",
        "action": "question",
        "text": "只查今天高位下拉，推胸不要列。",
        "goal_role": "result",
        "side_effect": False,
        "payload": {"dialogue": {"record_query": {
            "date_start": today.isoformat(), "date_end": today.isoformat(),
            "exercises": ["高位下拉"],
        }}},
    }
    observation = saved_workout_observation(candidate, {"user_id": "demo-user"})
    assert observation.status == "completed"
    assert "高位下拉" in observation.summary
    assert "坐姿推胸" not in observation.summary
    assert observation.result["authoritative"] is True

    state = {
        "intent_candidates": [candidate],
        "covered_goals": ["goal-read"],
        "goal_results": {"goal-read": observation.to_dict()},
        "observations": [],
        "messages": [],
        "termination_reason": "",
        "limited_tasks": [],
    }
    result = finalize_node(state, None)
    assert result["status"] == "complete"
    assert "高位下拉" in result["final_answer"]
    assert "坐姿推胸" not in result["final_answer"]


def test_conversation_only_request_uses_legacy_carrier(monkeypatch):
    from app.core.config import settings
    from app.schemas.chat import ChatRequest
    from app.services.agent.react_runtime import select_mode

    monkeypatch.setattr(settings, "react_readonly_enabled", True)
    monkeypatch.setattr(settings, "memory_read_backend", "compact_v1")
    request = ChatRequest(message="我当前有效的长期训练时间偏好是什么？", mode="react_readonly")
    conversation = [{
        "task_type": "chitchat", "action": "question", "text": "我当前有效的长期训练时间偏好是什么？",
        "payload": {"dialogue": {"reply_purpose": "fact_recall", "record_query": None}},
    }]
    assert select_mode(request, conversation) == ("legacy", "conversation_requires_legacy")
    record_read = [{
        "task_type": "workout_record", "action": "question", "text": "查今天高位下拉",
        "payload": {"dialogue": {"reply_purpose": "fact_recall", "record_query": {
            "date_start": clock.today().isoformat(),
            "date_end": clock.today().isoformat()}}},
    }]
    assert select_mode(request, record_read) == ("react_readonly", "pure_read")


def test_finalize_delivers_completed_read_when_another_goal_is_blocked():
    from app.services.agent.react_graph import finalize_node

    state = {
        "intent_candidates": [
            {"id": "read", "task_type": "training_qa", "action": "question",
             "text": "查下拉", "goal_role": "result", "side_effect": False},
            {"id": "advice", "task_type": "training_qa", "action": "question",
             "text": "给建议", "goal_role": "result", "side_effect": False},
        ],
        "covered_goals": ["read", "advice"],
        "goal_results": {
            "read": {"status": "completed", "summary": "已保存的记录：高位下拉 35 kg，2 组，每组 10 次"},
            "advice": {"status": "unavailable", "error": "insufficient_evidence"},
        },
        "observations": [],
        "messages": [],
        "termination_reason": "",
        "limited_tasks": [],
    }
    result = finalize_node(state, None)
    assert result["status"] == "blocked"
    assert "高位下拉" in result["final_answer"]
    assert "未完成" in result["final_answer"] or "未完全" in result["final_answer"]


def test_unified_read_covers_previous_day_only(compact_db):
    from app.db.repositories import query_workout_read_rows

    today = clock.today()
    _session("昨天", [{"exercise_name": "深蹲", "sets": [_set(50, 10)]}],
             when=today - timedelta(days=1))
    rows, _ = query_workout_read_rows(
        "demo-user", start_date=today, end_date=today, limit=30)
    assert rows == []


def test_readonly_loop_completes_workout_goal_after_auxiliary_probe(compact_db, monkeypatch):
    """The real failing shape: model probes read_workout_history, goal still must complete."""
    import asyncio

    from app.services.agent import react_adapters
    from app.services.agent.react_graph import ReactDependencies, invoke_react
    from app.services.agent.react_runtime import readonly_catalog

    today = clock.today()
    _session("今天训练", [
        {"exercise_name": "高位下拉", "sets": [_set(35, 10), _set(35, 10)]},
        {"exercise_name": "坐姿推胸", "sets": [_set(22, 12)]},
    ])
    candidate = {
        "id": "goal-read",
        "task_type": "workout_record",
        "action": "question",
        "text": "只查今天高位下拉，推胸不要列。",
        "goal_role": "result",
        "side_effect": False,
        "payload": {"dialogue": {"record_query": {
            "date_start": today.isoformat(), "date_end": today.isoformat(),
            "exercises": ["高位下拉"],
        }}},
    }

    class OneProbeLLM:
        def __init__(self):
            self.calls = 0

        async def complete_message(self, messages, **kwargs):
            self.calls += 1
            return {"role": "assistant", "content": "", "tool_calls": [{
                "id": "probe-1", "type": "function",
                "function": {"name": "read_workout_history", "arguments": "{\"days\": 1}"},
            }]}

        def configured(self):
            return True

    monkeypatch.setattr(
        "app.services.agent.semantics.safety_for",
        lambda source: {"status": "assessed", "level": "normal"})
    llm = OneProbeLLM()
    deps = ReactDependencies(
        llm=llm,
        tool_executor={"read_workout_history": react_adapters.read_workout_history},
        tool_catalog=readonly_catalog(),
        user_id="demo-user",
        conversation_id="read-repair",
        request_id="read-repair-goal",
    )
    state = asyncio.run(invoke_react(
        deps, candidate["text"], intent_candidates=[candidate], safety_level="normal"))
    assert state["status"] == "complete"
    assert llm.calls == 1
    assert "高位下拉" in state["final_answer"]
    assert "坐姿推胸" not in state["final_answer"]

