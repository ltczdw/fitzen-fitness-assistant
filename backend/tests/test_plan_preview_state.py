"""Tests for plan preview state."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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


def _state(answer: str, *, preview_id: str = "p1", user: str = "demo-user",
           conversation: str = "sim30-preview", minutes: int = 30) -> dict:
    from app.services.agent.plan_preview import build_preview_state
    from app.services.agent.training_timing import parse_timed_plan

    raw = _plan_raw(minutes)
    plan = parse_timed_plan(raw, limit_seconds=minutes * 60, expected_sessions=2)
    state = build_preview_state(
        user_id=user, conversation_id=conversation, request_id="req-" + preview_id,
        message_id=preview_id + ":assistant", created_at="2026-10-30T00:00:00+00:00",
        plan=plan, answer=answer,
        source="本周只能周二和周六练，每次从热身到结束最多30分钟，器械只有哑铃。",
        expected_sessions=2, session_minutes=minutes,
    )
    state["preview_id"] = preview_id
    return state


def _insert_preview(conversation: str, answer: str, state: dict, *,
                    when: datetime | None = None, user: str = "demo-user") -> str:
    from app.db.memory_models import MemoryTurn
    from app.services.memory import compact

    turn_id = state["preview_id"] + "-turn"
    messages = [
        {"role": "user", "content": "给这两天预览", "message_id": turn_id + ":user", "turn_id": turn_id},
        {"role": "assistant", "content": answer, "message_id": turn_id + ":assistant",
         "turn_id": turn_id, "app_meta": {"plan_preview": state}},
    ]
    with compact.session() as session, session.begin():
        session.add(MemoryTurn(
            turn_id=turn_id, request_id="req-" + turn_id, conversation_id=conversation,
            messages=messages, status="completed",
            completed_at=when or datetime.now(timezone.utc), content_hash="", memory_epoch=0,
        ))
    return turn_id


def test_replay_plan_preview_malformed_arguments():
    from app.services.agent.semantics import RequestUnderstanding
    from app.services.llm.structured import StructuredOutputError, parse_tool_result

    fixture = _fixture("plan_preview")
    for item in fixture["items"]:
        message = {"tool_calls": [{"type": "function", "function": {
            "name": item["tool"], "arguments": item["arguments"]}}]}
        with pytest.raises(StructuredOutputError) as error:
            parse_tool_result(message, RequestUnderstanding, item["tool"])
        assert error.value.code == "structured_invalid_json"


def test_replay_plan_followup_valid_task_accepted():
    from app.services.agent import semantics
    from app.services.agent.task_models import is_preview_task

    fixture = _fixture("plan_followup")
    raw = json.loads(fixture["items"][0]["arguments"])
    context = {"current_plan_preview": {"status": "ok", "state": {
        "preview_id": "p1", "message_id": "turn-1:assistant",
        "weekdays": ["周二", "周六"], "expected_sessions": 2,
        "session_minutes": 30, "days": [],
    }}}
    checked = semantics.validate_understanding(
        fixture["source_message"], raw, context=context, absorb_background_plans=True)
    task = checked["tasks"][0]
    assert task["delivery_type"] == "preview"
    assert is_preview_task(task)
    assert checked["needs_clarification"] is False
    assert semantics._history_required(raw, context) is False


def test_preview_two_days_30_minutes():
    from app.services.agent.plan_preview import build_preview_state, find_weekday_list
    from app.services.agent.training_timing import parse_timed_plan, render_timed_plan

    source = "本周只能周二和周六练，每次从热身到结束最多30分钟，器械只有哑铃。"
    assert find_weekday_list(source) == ["周二", "周六"]
    raw = _plan_raw()
    rendered = render_timed_plan(raw, limit_seconds=1800, expected_sessions=2,
                                expected_weekdays=["周二", "周六"])
    assert "周二" in rendered and "周六" in rendered
    plan = parse_timed_plan(raw, limit_seconds=1800, expected_sessions=2)
    state = build_preview_state(
        user_id="demo-user", conversation_id="c", request_id="r", message_id="t:assistant",
        created_at="now", plan=plan, answer=rendered, source=source,
        expected_sessions=2, session_minutes=30)
    assert state["expected_sessions"] == 2
    assert state["weekdays"] == ["周二", "周六"]
    assert [day["minutes"] for day in state["days"]] == [30, 30]


def test_preview_followup_25_minutes_reads_state(compact_db):
    from app.services.agent.plan_preview import latest_plan_preview, removed_exercises

    first_answer = "第一版两天预览"
    first = _state(first_answer, preview_id="p1")
    _insert_preview("sim30-preview", first_answer, first,
                    when=datetime.now(timezone.utc) - timedelta(minutes=2))
    loaded = latest_plan_preview("demo-user", "sim30-preview")
    assert loaded["status"] == "ok"
    assert loaded["state"]["preview_id"] == "p1"

    second_answer = "第二版：第二天25分钟"
    second = _state(second_answer, preview_id="p2", minutes=25)
    second["base_preview_id"] = "p1"
    second["days"][0]["exercises"] = ["哑铃深蹲"]
    removed = removed_exercises(first, second)
    assert removed and removed[0]["label"] == "周二 全身A"
    assert "哑铃划船" in removed[0]["removed"]


def test_preview_followup_without_preview(compact_db):
    from app.services.agent.plan_preview import NO_PREVIEW_COPY, latest_plan_preview

    assert latest_plan_preview("demo-user", "no-preview-conversation") == {"status": "none"}
    assert "上一轮没有成功生成" in NO_PREVIEW_COPY
    assert "本次没有生成、修改或保存任何计划" in NO_PREVIEW_COPY


def test_preview_followup_without_preview_execution_response(compact_db, monkeypatch):
    from unittest.mock import AsyncMock

    from app.schemas.chat import ChatRequest
    from app.services.agent import semantics
    from app.services.agent.agent import FitnessAgent
    from app.services.agent.execution_state import PlannerDecision
    from app.services.agent.intent import IntentResult
    from app.services.agent.task_models import TaskSpec

    source = "沿用刚才的两天预览，把第二天缩到25分钟，减少一个动作。"
    raw = {"tasks": [{
        "task_type": "training_plan", "action": "answer", "text": source,
        "is_negated": True, "is_hypothetical": False, "is_third_party": False,
        "plan_result": False, "delivery_type": "preview", "goal_role": "result",
        "history_requirement": "required", "reply_purpose": "general",
        "reply_mode": "conversation", "context_ids": [], "fact_fields": [],
        "critical_fact_fields": [], "preview_ref": "",
    }], "reply_mode": "conversation", "reply_purpose": "general",
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}}
    checked = semantics.validate_understanding(
        source, raw, context={"current_plan_preview": {"status": "none"}},
        absorb_background_plans=True)
    spec = TaskSpec("task-1", "training_plan", "plan_adjust", "answer", source,
                    payload={"semantic": checked["tasks"][0]}).to_dict()
    agent = FitnessAgent()
    agent._route = AsyncMock(return_value=PlannerDecision(
        IntentResult("training_plan", 1.0), {}, (spec,)))
    token = semantics.request_understanding.set({})
    try:
        semantics.remember_understanding(source, checked)
        semantics.remember_safety(source, raw["safety"])
        response = asyncio.run(agent._respond_impl(
            ChatRequest(user_id="demo-user", conversation_id="no-preview-conversation",
                        request_id="no-preview-request", message=source,
                        mode="react_readonly")))
    finally:
        semantics.request_understanding.reset(token)
    assert "上一轮没有成功生成" in response.answer
    assert "本次没有生成、修改或保存任何计划" in response.answer
    assert response.metrics["plan_preview_status"] == "none"


def test_new_preview_without_previous_state_is_not_blocked():
    from app.services.agent import semantics

    source = "本周只能周二和周六练，每次30分钟，给这两天的完整预览。"
    raw = {"tasks": [{
        "task_type": "training_plan", "action": "question", "text": source,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        "plan_result": True, "result_scope": "independent",
        "plan_request_quote": "给这两天的完整预览", "delivery_type": "preview",
        "goal_role": "result", "history_requirement": "none",
        "reply_purpose": "general", "reply_mode": "grounded",
    }], "reply_mode": "grounded", "reply_purpose": "general",
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}}
    context = {"current_plan_preview": {"status": "none"}}
    checked = semantics.validate_understanding(
        source, raw, context=context, absorb_background_plans=True)
    assert checked["tasks"][0]["delivery_type"] == "preview"
    assert semantics._history_required(raw, context) is False


def test_preview_defaults_to_latest_when_ref_missing(compact_db):
    from app.services.agent.plan_preview import latest_plan_preview

    older = _state("旧预览", preview_id="old")
    newer = _state("新预览", preview_id="new")
    now = datetime.now(timezone.utc)
    _insert_preview("sim30-preview", "旧预览", older, when=now - timedelta(minutes=5))
    _insert_preview("sim30-preview", "新预览", newer, when=now)
    loaded = latest_plan_preview("demo-user", "sim30-preview")
    assert loaded["status"] == "ok"
    assert loaded["state"]["preview_id"] == "new"


def test_preview_explicit_ref_wins():
    from app.services.agent import semantics

    fixture = _fixture("plan_followup")
    raw = json.loads(fixture["items"][0]["arguments"])
    raw["tasks"][0]["preview_ref"] = "old"
    older = _state("旧预览", preview_id="old")
    newer = _state("新预览", preview_id="new")
    context = {"current_plan_preview": {"status": "ok", "state": newer,
                                        "available": [newer, older]}}
    checked = semantics.validate_understanding(
        fixture["source_message"], raw, context=context, absorb_background_plans=True)
    assert checked["tasks"][0]["preview_ref"] == "old"
    assert semantics._history_required(raw, context) is False


def test_preview_unknown_ref_rejected():
    from app.services.agent import semantics

    fixture = _fixture("plan_followup")
    raw = json.loads(fixture["items"][0]["arguments"])
    raw["tasks"][0]["preview_ref"] = "not-provided"
    context = {"current_plan_preview": {"status": "ok", "state": _state("新预览", preview_id="new"),
                                        "available": [_state("新预览", preview_id="new")]}}
    with pytest.raises(semantics.SemanticValidationError) as error:
        semantics.validate_understanding(
            fixture["source_message"], raw, context=context, absorb_background_plans=True)
    assert error.value.code == "semantic_preview_ref_invalid"


def test_preview_state_unavailable_not_reported_as_none(compact_db):
    from app.services.agent.plan_preview import latest_plan_preview

    state = _state("回答内容", preview_id="bad")
    _insert_preview("sim30-preview", "被改写的回答", state)
    assert latest_plan_preview("demo-user", "sim30-preview") == {"status": "unavailable"}


def test_preview_partial_generation_not_saved():
    from app.services.agent.training_timing import render_timed_plan

    raw = json.dumps({"sessions": [json.loads(_plan_raw())["sessions"][0]]}, ensure_ascii=False)
    with pytest.raises(ValueError) as error:
        render_timed_plan(raw, limit_seconds=1800, expected_sessions=2)
    assert str(error.value) == "session_count_mismatch"


def test_preview_day_labels_from_user_source():
    from app.services.agent.plan_preview import find_weekday_list

    source = "本周只能周二和周六练，每次30分钟"
    assert find_weekday_list(source) == ["周二", "周六"]
    assert find_weekday_list("每周三次，每次30分钟") is None


def test_preview_weekday_repeat_or_range_rejected():
    from app.services.agent.plan_preview import find_weekday_list

    assert find_weekday_list("周二和周二练") is None
    assert find_weekday_list("周一至周五练") is None
    assert find_weekday_list("明天和周六练") is None


def test_preview_cross_user_not_loaded(compact_db):
    from app.services.agent.plan_preview import latest_plan_preview

    state = _state("别人的预览", preview_id="other", user="other-user")
    _insert_preview("sim30-preview", "别人的预览", state)
    assert latest_plan_preview("demo-user", "sim30-preview") == {"status": "none"}


def test_preview_finish_turn_persists_app_meta(compact_db):
    from app.schemas.chat import ChatRequest
    from app.services.agent.plan_preview import latest_plan_preview
    from app.services.memory import compact

    request = ChatRequest(user_id="demo-user", conversation_id="finish-preview",
                          request_id="finish-preview-1", message="给两天预览")
    turn_id, _ = compact.begin_turn(request)
    answer = "两天预览答案"
    state = _state(answer, preview_id="finish", conversation="finish-preview")
    compact.finish_turn(turn_id, answer, {"answer": answer, "metrics": {"plan_preview": state}})
    loaded = latest_plan_preview("demo-user", "finish-preview")
    assert loaded["status"] == "ok"
    assert loaded["state"]["preview_id"] == "finish"


def test_preview_pg_persistence(monkeypatch, tmp_path):
    from tests.react_phase_c1_support import isolated
    from app.db.memory_models import MemoryTurn
    from app.services.agent.plan_preview import latest_plan_preview
    from app.services.memory import compact

    with isolated(monkeypatch, tmp_path, "postgres"):
        answer = "PG预览答案"
        state = _state(answer, preview_id="pg", conversation="pg-conversation")
        with compact.session() as session, session.begin():
            session.add(MemoryTurn(
                turn_id="pg-turn", request_id="req-pg", conversation_id="pg-conversation",
                messages=[
                    {"role": "user", "content": "给预览", "message_id": "pg-turn:user", "turn_id": "pg-turn"},
                    {"role": "assistant", "content": answer, "message_id": "pg-turn:assistant",
                     "turn_id": "pg-turn", "app_meta": {"plan_preview": state}},
                ], status="completed", completed_at=datetime.now(timezone.utc),
                content_hash="", memory_epoch=0))
        loaded = latest_plan_preview("demo-user", "pg-conversation")
        assert loaded["status"] == "ok"
