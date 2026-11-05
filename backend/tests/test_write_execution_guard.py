"""Write execution boundary: model semantics decide, workers enforce.

Migrated from the pre-2026-10-18 text-classification contract. Negation,
hypothesis, third-party and question semantics are now expressed by the
validated model task/action/flags; no test here asks the authorization module
to infer them from wording.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
from app.services.agent.planner_validation import validate_planner_output
from app.services.agent.task_planner import plan_tasks_from_llm, task_write_authorization


def semantic_task(message, kind, action="record", **flags):
    return {"task_type": kind, "action": action, "text": message,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            **flags}


@pytest.mark.parametrize("message,kind,flags", [
    ("如果我吃了两个鸡蛋帮我记录", "nutrition_record", {"is_hypothetical": True}),
    ("朋友练了深蹲帮我记录", "workout_record", {"is_third_party": True}),
    ("不要记录鸡蛋", "nutrition_record", {"is_negated": True}),
    ("怎么记录两个鸡蛋", "nutrition_record", {"action": "question"}),
    ("不要保存但补记两个鸡蛋", "nutrition_record", {"is_negated": True}),
])
def test_unsafe_semantics_are_blocked(message, kind, flags):
    result = validate_planner_output(message, {"tasks": [semantic_task(message, kind, **flags)]})
    assert result.tasks[0].get("write_authorization") != "authorized"
    assert not result.tasks[0].get("side_effect", False)


@pytest.mark.parametrize("message,span", [("记录两个鸡蛋", "记录两个鸡蛋"), ("帮我记录，我做了卧推", "我做了卧推")])
def test_explicit_model_record_is_preserved(message, span):
    planned = plan_tasks_from_llm(message, {"tasks": [semantic_task(span, "nutrition_record")]})
    assert planned is not None
    assert planned[0]["payload"]["write_authorization"] == "authorized"
    assert planned[0]["side_effect"] is True


def test_model_span_must_come_from_the_original_request():
    message = "补记昨天的训练：哑铃卧推20kg 4组每组10次、深蹲40kg 3组每组12次"
    invented = semantic_task("模型编造的卧推100kg", "workout_record")
    assert plan_tasks_from_llm(message, {"tasks": [invented]}) is None


def test_authorization_is_scoped_to_each_structured_task():
    message = "记录卧推；不要记录深蹲"
    planned = plan_tasks_from_llm(message, {"tasks": [
        semantic_task("记录卧推", "workout_record"),
        semantic_task("不要记录深蹲", "workout_record", is_negated=True),
    ]})
    assert planned is not None
    assert [task["payload"]["write_authorization"] for task in planned] == ["authorized", "rejected_unauthorized"]
    assert [task["side_effect"] for task in planned] == [True, False]


@pytest.mark.parametrize("message,kind,flags", [
    ("补充蛋白质，记得提醒我喝水", "nutrition_advice", {"action": "question"}),
    ("别记录我今天的训练", "workout_record", {"is_negated": True}),
])
def test_non_write_or_denied_semantics_never_authorize(message, kind, flags):
    result = validate_planner_output(message, {"tasks": [semantic_task(message, kind, **flags)]})
    assert result.tasks[0].get("write_authorization") != "authorized"
    assert not result.tasks[0].get("side_effect", False)


@pytest.mark.parametrize("flag", ["is_negated", "is_hypothetical", "is_third_party"])
def test_model_safety_flags_reject_write_task(flag):
    assert task_write_authorization("记录我完成的卧推", {flag: True}) == "rejected_unauthorized"


def test_worker_rejects_model_authorized_hypothetical(monkeypatch):
    graph = FitnessMultiAgentGraph()
    extract = AsyncMock()
    monkeypatch.setattr(graph.workout_extraction, "run_many", extract)
    result = asyncio.run(graph.workers["workout_record"].ainvoke({"user_id": "fixture", "raw_message": "如果我做了卧推请记录", "task_text": "记录卧推", "task_action": "record", "task_payload": {"write_authorization": "rejected_unauthorized"}}))
    assert result["result"]["status"] == "rejected_unauthorized"
    extract.assert_not_called()
