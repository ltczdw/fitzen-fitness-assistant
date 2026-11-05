"""One contract for task domain/operation across planner, gate and answer routing.

``task_type`` names the domain; ``action`` names the operation. Only a write
action on a record domain has a side effect; a read action keeps the domain
label so the saved-record read path can find it. These tests enumerate the
enums, so a new type or action cannot silently inherit an old branch.
"""
from datetime import datetime, timezone

import pytest

from app.services.agent.dialogue_context import business_record_task
from app.services.agent.planner_validation import _trace_task_corrections
from app.services.agent.task_models import (
    RECORD_TASK_TYPES,
    TASK_TYPE_TO_ANSWER_DOMAIN,
    TaskAction,
    TaskType,
    answer_domain,
    is_record_read,
    is_record_write,
    serves_read_goal,
)
from app.services.agent.task_planner import plan_tasks


MESSAGE = "今天早餐我吃了45克燕麦"
READ_ACTIONS = (TaskAction.QUESTION.value, TaskAction.ANSWER.value)
WRITE_ACTIONS = (TaskAction.RECORD.value, TaskAction.CONFIRM.value)
NON_RECORD_TYPES = [TaskType.NUTRITION_ADVICE.value, TaskType.TRAINING_QA.value,
                    TaskType.TRAINING_PLAN.value, TaskType.SAFETY_TRIAGE.value,
                    TaskType.CHITCHAT.value]
QUERY = {"date_start": "2026-10-30", "date_end": "2026-10-30",
         "meal_types": ["breakfast"], "food_names": [], "exercises": []}


def _raw(kind, action, text=MESSAGE):
    return {"tasks": [{"task_type": kind, "action": action, "text": text,
                       "is_negated": False, "is_hypothetical": False,
                       "is_third_party": False}]}


def test_every_task_type_declares_its_answer_domain():
    assert set(TASK_TYPE_TO_ANSWER_DOMAIN) == set(TaskType)


@pytest.mark.parametrize("action", [item.value for item in TaskAction])
@pytest.mark.parametrize("kind", [item.value for item in TaskType
                                  if item is not TaskType.CHITCHAT])
def test_planner_never_relabels_a_domain(kind, action):
    planned = plan_tasks(MESSAGE, parsed_input=_raw(kind, action))
    assert planned, (kind, action)
    assert planned[0]["task_type"] == kind
    if action in READ_ACTIONS:
        assert planned[0]["side_effect"] is False


@pytest.mark.parametrize("kind", sorted(RECORD_TASK_TYPES))
@pytest.mark.parametrize("action", [item.value for item in TaskAction])
def test_planning_gate_normalizes_action_without_relabelling(kind, action):
    text = "怎么记录今天的早餐"
    items = [{"id": "task-1", "task_type": kind, "action": action, "text": text}]
    _trace_task_corrections(
        text, items, pending_record=None, current_session_id="session",
        now=datetime(2026, 10, 30, tzinfo=timezone.utc))
    assert items[0]["task_type"] == kind
    if action in READ_ACTIONS:
        assert items[0]["action"] in READ_ACTIONS
    else:
        # An unbound confirm becomes a read; a plain record stays a write
        # candidate and is decided later by the write authorization gate.
        assert items[0]["action"] in {action, "question"}


@pytest.mark.parametrize("kind", sorted(RECORD_TASK_TYPES))
@pytest.mark.parametrize("action", [item.value for item in TaskAction])
def test_business_record_read_recognition_follows_action(kind, action):
    task = {"task_type": kind, "action": action, "reply_purpose": "fact_recall",
            "payload": {"record_query": QUERY}}
    found = business_record_task({"tasks": [task], "reply_purpose": "fact_recall"})
    if action in READ_ACTIONS:
        assert found is not None
        assert found["task_type"] == kind
        assert found["record_query"] == QUERY
    else:
        assert found is None


@pytest.mark.parametrize("kind", NON_RECORD_TYPES)
@pytest.mark.parametrize("action", [item.value for item in TaskAction])
def test_non_record_types_are_never_record_reads(kind, action):
    assert is_record_read({"task_type": kind, "action": action}) is False
    assert is_record_write({"task_type": kind, "action": action}) is False
    task = {"task_type": kind, "action": action, "reply_purpose": "fact_recall",
            "record_query": QUERY}
    assert business_record_task({"tasks": [task], "reply_purpose": "fact_recall"}) is None


@pytest.mark.parametrize("kind", sorted(RECORD_TASK_TYPES))
def test_record_read_without_a_query_is_history_recall_not_sql(kind):
    task = {"task_type": kind, "action": "question", "reply_purpose": "fact_recall",
            "context_ids": ["history-1"]}
    assert business_record_task({"tasks": [task], "reply_purpose": "fact_recall"}) is None


@pytest.mark.parametrize("tool_type,goal_type,expected", [
    ("nutrition_advice", "nutrition_record", True),
    ("nutrition_advice", "nutrition_advice", True),
    ("training_qa", "workout_record", True),
    ("training_qa", "training_qa", True),
    ("training_plan", "training_plan", True),
    ("training_qa", "training_plan", False),
    ("training_plan", "training_qa", False),
    ("nutrition_advice", "training_qa", False),
    ("training_qa", "nutrition_record", False),
])
def test_read_tool_serves_goals_of_its_own_answer_domain(tool_type, goal_type, expected):
    assert serves_read_goal(tool_type, goal_type) is expected


def test_answer_domain_is_none_for_unknown_types():
    assert answer_domain("not_a_task_type") is None
    assert answer_domain(None) is None
