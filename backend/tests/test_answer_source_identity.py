"""Tests for answer source identity."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core import clock


def test_distinct_identical_workouts_are_both_rendered(compact_db):
    from app.db.repositories import create_workout
    from app.db.workout_session_repository import create_session
    from app.services.agent.react_readonly_workers import saved_workout_observation

    today = clock.today()
    create_workout("demo-user", "高位下拉", 3, 10, 35, date=today)
    create_session({"user_id": "demo-user", "session_date": today, "title": "训练",
                    "note": "", "exercises": [{"exercise_name": "高位下拉",
                    "sets": [{"weight_kg": 35, "reps": 10} for _ in range(3)]}]})
    goal = {"id": "read", "task_type": "workout_record", "action": "question",
            "text": "列出今天的高位下拉", "record_query": {
                "date_start": today.isoformat(), "date_end": today.isoformat(),
                "exercises": ["高位下拉"]}}
    result = saved_workout_observation(goal, {"user_id": "demo-user"})
    assert len(result.result["workouts"]) == 2
    assert result.summary.count("高位下拉") == 2


@pytest.mark.parametrize("kind", ["workout_record", "nutrition_record"])
def test_selection_preserves_distinct_sources_with_same_text(monkeypatch, kind):
    from app.schemas.chat import ChatRequest
    from app.services.agent import dialogue_context

    sources = {"source:1": "同一条显示文字", "source:2": "同一条显示文字"}
    monkeypatch.setattr(dialogue_context, "_saved_record_sources", lambda *a, **kw: (sources, True, None, False))
    monkeypatch.setattr(dialogue_context, "select_source_excerpts", AsyncMock(return_value=[
        SimpleNamespace(source_id="source:1"), SimpleNamespace(source_id="source:1"),
        SimpleNamespace(source_id="source:2")]))
    result = asyncio.run(dialogue_context.saved_record_answer(
        object(), ChatRequest(message="列出记录"), kind,
        {"date_start": "2026-10-30", "date_end": "2026-10-30"}))
    assert result["answer"].count("同一条显示文字") == 2


def test_readonly_source_selection_preserves_distinct_ids(monkeypatch):
    from app.services.agent import dialogue_context, react_readonly_workers
    from app.services.agent.agent import agent

    monkeypatch.setattr(agent.graph.llm, "configured", lambda: True)
    monkeypatch.setattr(dialogue_context, "select_source_excerpts", AsyncMock(return_value=[
        SimpleNamespace(source_id="one"), SimpleNamespace(source_id="one"),
        SimpleNamespace(source_id="two")]))
    result = asyncio.run(react_readonly_workers._select_current_sources(
        {"user_id": "demo-user", "conversation_id": "test-conversation", "request_id": "test-request"},
        "列出训练", {"one": "相同", "two": "相同"}))
    assert result == ["相同", "相同"]


@pytest.mark.parametrize("count,chosen", [(2, 1), (25, 24)])
def test_partial_source_selection_is_not_presented_as_complete(monkeypatch, count, chosen):
    from app.schemas.chat import ChatRequest
    from app.services.agent import dialogue_context, react_readonly_workers
    from app.services.agent.agent import agent

    sources = {f"source:{i}": f"训练记录 {i}" for i in range(count)}
    excerpts = [SimpleNamespace(source_id=f"source:{i}") for i in range(chosen)]
    monkeypatch.setattr(dialogue_context, "_saved_record_sources", lambda *a, **kw: (sources, True, None, False))
    monkeypatch.setattr(dialogue_context, "select_source_excerpts", AsyncMock(return_value=excerpts))
    monkeypatch.setattr(agent.graph.llm, "configured", lambda: True)
    answer = asyncio.run(dialogue_context.saved_record_answer(
        object(), ChatRequest(message="列出记录"), "workout_record",
        {"date_start": "2026-10-30", "date_end": "2026-10-30"}))
    readonly = asyncio.run(react_readonly_workers._select_current_sources(
        {"user_id": "demo-user", "conversation_id": "test-conversation", "request_id": "test-request"},
        "列出训练", sources))
    assert "只列出了部分候选记录" in answer["answer"]
    assert answer["complete"] is False
    assert answer["selection_complete"] is False
    assert answer["candidate_count"] == count
    assert answer["selected_count"] == chosen
    assert answer["answer"].startswith("匹配到的已保存记录：")
    assert "只列出了部分候选记录" in readonly[-1]
    assert readonly.complete is False
    assert readonly.candidate_count == count
    assert readonly.selected_count == chosen


def test_empty_selection_with_candidates_is_not_claimed_empty(monkeypatch):
    from app.schemas.chat import ChatRequest
    from app.services.agent import dialogue_context, react_readonly_workers

    monkeypatch.setattr(dialogue_context, "_saved_record_sources", lambda *a, **kw:
                        ({"source:1": "高位下拉 35kg"}, True, None, False))
    monkeypatch.setattr(dialogue_context, "select_source_excerpts", AsyncMock(return_value=[]))
    result = asyncio.run(dialogue_context.saved_record_answer(
        object(), ChatRequest(message="列出训练"), "workout_record",
        {"date_start": "2026-10-30", "date_end": "2026-10-30"}))
    assert result["status"] == "unavailable"
    assert "未能确认" in result["answer"]
    assert result["error"] == "saved_record_selection_unavailable"
    monkeypatch.setattr(react_readonly_workers, "_dialogue_scope", lambda _: {
        "reply_purpose": "fact_recall", "source_task_type": "workout_record",
        "record_query": {"date_start": "2026-10-30", "date_end": "2026-10-30"}})
    monkeypatch.setattr(react_readonly_workers, "_current_data_sources", lambda *a:
                        ({"source:1": "高位下拉 35kg"}, True, False))
    monkeypatch.setattr(react_readonly_workers, "_select_current_sources", AsyncMock(return_value=[]))
    observation = asyncio.run(react_readonly_workers._source_bound_dialogue_answer({}, "列出训练"))
    assert observation.status == "unavailable"
    assert observation.error == "saved_record_selection_unavailable"


def test_constraint_not_dispatched_or_delivered_but_write_and_missing_role_remain():
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph

    message = "查询训练；只查询不保存"
    text = "只查询不保存"
    start = message.index(text)
    constraint = {"id": "constraint", "task_type": "training_qa", "action": "answer",
                  "text": text, "side_effect": False, "payload": {
                      "semantic_available": True, "source_start": start, "source_end": len(message),
                      "semantic": {"text": text, "goal_role": "constraint"}}}
    answer = {"id": "answer", "task_type": "training_qa", "action": "question", "text": "查询训练", "payload": {}}
    write = {**constraint, "id": "write", "action": "record", "side_effect": False}
    legacy = {"id": "legacy", "task_type": "training_qa", "action": "answer", "text": "你好", "payload": {}}
    dependent = {**answer, "id": "dependent", "depends_on": ["constraint"]}
    state = {"raw_message": message, "planned_tasks": [answer, constraint, write, legacy, dependent]}
    tasks = FitnessMultiAgentGraph._preprocess_node(state)["tasks"]
    assert [t["id"] for t in tasks] == ["answer", "write", "legacy", "dependent"]
    assert tasks[-1]["execution_status"] == "blocked"
    assert len(state["planned_tasks"]) == 5
    output = FitnessMultiAgentGraph._finalize_node({**state, "tasks": [], "worker_results": [
        {"task_id": "answer", "task_type": "training_qa", "status": "complete", "answer": "查询结果"},
        {"task_id": "constraint", "task_type": "training_qa", "status": "complete", "answer": "请说明需求"}]})
    assert output["final_answer"] == "查询结果"
    assert [x["task_id"] for x in output["task_outcomes"]] == ["answer"]


def test_real_planner_preserves_bound_constraint_for_dispatch_filter():
    from app.services.agent import semantics
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.services.agent.task_planner import plan_tasks_from_llm

    message = "查今天的训练；给恢复建议；只查询，不新增记录"
    parts = [
        {"task_type": "workout_record", "action": "question", "text": "查今天的训练", "goal_role": "result"},
        {"task_type": "training_qa", "action": "question", "text": "给恢复建议", "goal_role": "result"},
        {"task_type": "chitchat", "action": "answer", "text": "只查询，不新增记录", "goal_role": "constraint", "is_negated": True},
    ]
    token = semantics.request_understanding.set({})
    semantics.remember_understanding(message, {"tasks": parts})
    try:
        planned = plan_tasks_from_llm(message, {"tasks": parts})
        state = {"raw_message": message, "planned_tasks": planned}
        tasks = FitnessMultiAgentGraph._preprocess_node(state)["tasks"]
        assert [task["text"] for task in tasks] == ["查今天的训练", "给恢复建议"]
        assert len(state["planned_tasks"]) == 3
    finally:
        semantics.request_understanding.reset(token)


def test_rule_planner_binds_constraint_source_span_before_dispatch_filter():
    """The rule path must carry the same role/span contract as the parser path."""
    from app.services.agent import semantics
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.services.agent.task_planner import plan_tasks_rule

    message = "查今天的训练；给恢复建议；只查询，不新增记录"
    parts = [
        {"task_type": "workout_record", "action": "question", "text": "查今天的训练",
         "goal_role": "result", "is_negated": False, "is_hypothetical": False,
         "is_third_party": False},
        {"task_type": "training_qa", "action": "question", "text": "给恢复建议",
         "goal_role": "result", "is_negated": False, "is_hypothetical": False,
         "is_third_party": False},
        {"task_type": "chitchat", "action": "answer", "text": "只查询，不新增记录",
         "goal_role": "constraint", "is_negated": True, "is_hypothetical": False,
         "is_third_party": False},
    ]
    offset = 0
    validated = []
    for index, part in enumerate(parts, start=1):
        start = message.index(part["text"], offset)
        end = start + len(part["text"])
        offset = end
        validated.append({
            **part,
            "id": f"task-{index}",
            "worker": "training_qa" if part["task_type"] != "workout_record" else "workout_record",
            "source_start": start,
            "source_end": end,
            "side_effect": False,
        })
    token = semantics.request_understanding.set({})
    semantics.remember_understanding(
        message, {"tasks": parts, "validated_tasks": validated})
    try:
        planned = plan_tasks_rule(message)
        constraint = next(task for task in planned if task["id"] == "task-3")
        assert constraint["payload"]["semantic_available"] is True
        assert constraint["payload"]["semantic"]["goal_role"] == "constraint"
        assert constraint["payload"]["source_start"] == message.index("只查询，不新增记录")
        state = {"raw_message": message, "planned_tasks": planned}
        tasks = FitnessMultiAgentGraph._preprocess_node(state)["tasks"]
        assert [task["text"] for task in tasks] == ["查今天的训练", "给恢复建议"]
    finally:
        semantics.request_understanding.reset(token)


def test_pure_constraint_conversation_does_not_call_answer_model():
    from app.schemas.chat import ChatRequest
    from app.services.agent import dialogue_context

    llm = SimpleNamespace(complete_structured=AsyncMock(side_effect=AssertionError(
        "a pure constraint must not reach the answer model")))
    result = asyncio.run(dialogue_context.conversation_answer(
        llm, ChatRequest(message="只查询，不保存"), {},
        {"reply_mode": "conversation", "reply_purpose": "general",
         "tasks": [{"goal_role": "constraint", "action": "answer",
                    "task_type": "chitchat", "text": "只查询，不保存"}]}))
    assert result["model_used"] is False
    assert result["status"] == "complete"
    assert "没有需要单独回答的目标" in result["answer"]
    assert llm.complete_structured.called is False


def test_truncated_workout_observation_is_structured_as_incomplete(monkeypatch):
    from app.db import repositories
    from app.services.agent.react_readonly_workers import saved_workout_observation

    today = clock.today()
    rows = [{
        "id": 1, "source": "record", "source_id": "record:1", "date": today.isoformat(),
        "exercise": "高位下拉", "sets": 3, "reps": 10, "weight_kg": 35,
        "set_details": [{"weight_kg": 35, "reps": 10}] * 3,
    }]
    monkeypatch.setattr(repositories, "query_workout_read_rows",
                        lambda *a, **kw: (rows, True))
    goal = {"id": "read", "task_type": "workout_record", "action": "question",
            "text": "列出今天的高位下拉", "record_query": {
                "date_start": today.isoformat(), "date_end": today.isoformat(),
                "exercises": ["高位下拉"]}}
    result = saved_workout_observation(goal, {"user_id": "demo-user"})
    assert result.result["truncated"] is True
    assert result.result["complete"] is False
    assert "部分结果" in result.summary


def test_exact_read_from_truncated_observation_is_not_complete(monkeypatch):
    from app.services.agent import react_readonly_workers

    today = clock.today()
    rows = [{
        "id": 1, "source": "record", "source_id": "record:1", "date": today.isoformat(),
        "exercise": "高位下拉", "sets": 3, "reps": 10, "weight_kg": 35,
        "set_details": [{"weight_kg": 35, "reps": 10}] * 3,
    }]
    monkeypatch.setattr(react_readonly_workers, "_dialogue_scope", lambda _: {
        "reply_purpose": "fact_recall", "source_task_type": "workout_record",
        "record_query": {"date_start": today.isoformat(), "date_end": today.isoformat(),
                         "exercises": ["高位下拉"]}})
    observation = asyncio.run(react_readonly_workers._source_bound_dialogue_answer(
        {"observations": [{"status": "completed",
                           "_source": {"permission": "read", "tool_name": "read_workout_history"},
                           "result": {"workouts": rows, "truncated": True}}]},
        "查今天的高位下拉"))
    assert observation.result["truncated"] is True
    assert observation.result["complete"] is False
    assert observation.summary.startswith("匹配到的已保存记录：")


def test_finalizer_keeps_structured_completeness_outcome():
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph

    task = {"id": "read", "task_type": "training_qa", "action": "question",
            "text": "查今天高位下拉", "payload": {}}
    output = FitnessMultiAgentGraph._finalize_node({
        "raw_message": "查今天高位下拉", "planned_tasks": [task],
        "tasks": [], "worker_results": [
            {"task_id": "read", "task_type": "training_qa", "status": "complete",
             "answer": "匹配到的已保存记录：高位下拉", "complete": False}]})
    assert output["task_outcomes"][0]["complete"] is False
