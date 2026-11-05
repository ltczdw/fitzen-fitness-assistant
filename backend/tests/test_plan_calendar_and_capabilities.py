"""Tests for plan calendar and capabilities."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.services.agent.training_timing import (
    expected_session_count, weekday_list, weekly_sessions,
)


@pytest.mark.parametrize("quote, days", [
    ("周二和周六", [2, 6]),
    ("周二、周四、周六", [2, 4, 6]),
    ("星期一与星期五", [1, 5]),
    ("周日", [7]),
    ("周二和周二", None),
    ("周二到周六", None),
    ("明天和周六", None),
])
def test_weekday_list_is_a_closed_format(quote, days):
    assert weekday_list(quote) == days


@pytest.mark.parametrize("quote, count", [
    ("周二和周六", 2), ("周二、周四、周六", 3), ("每周两次", 2), ("2天", 2),
])
def test_weekly_sessions_accepts_grounded_weekdays_or_counts(quote, count):
    assert weekly_sessions(quote) == count


def test_expected_sessions_uses_grounded_count_even_for_session_scope():
    assert expected_session_count("session", 2) == 2
    assert expected_session_count("weekly", 2) == 2
    assert expected_session_count("session", None) == 1
    assert expected_session_count("weekly", None) is None


def test_saved_two_day_plan_information_passes_validation():
    from app.services.agent import task_information

    source = ("我成年、无伤病，是力量训练新手。本周只能周二和周六练，每次从热身到结束最多30分钟，"
              "器械只有哑铃。给这两天的完整预览，组间休息也算进30分钟；不要保存或覆盖已存计划。")
    info = {
        "scope": "session",
        "facts": [
            {"field": "goal", "source_id": "profile:goal", "quote": "提升基础力量"},
            {"field": "experience", "source_id": "current", "quote": "是力量训练新手"},
            {"field": "equipment", "source_id": "current", "quote": "器械只有哑铃"},
            {"field": "session_time", "source_id": "current",
             "quote": "每次从热身到结束最多30分钟", "quantity_quote": "30分钟", "seconds": 1800},
            {"field": "available_days", "source_id": "current",
             "quote": "本周只能周二和周六练", "quantity_quote": "周二和周六", "count": 2},
        ],
        "phase_limits": [],
    }
    llm = type("LLM", (), {"complete_structured": AsyncMock(return_value=info)})()
    details, clarification = __import__("asyncio").run(task_information.check_plan_information(
        llm, {"raw_message": source, "task_text": source,
              "profile": {"goal": "提升基础力量"}, "semantic_scope": {}}))
    assert clarification is None
    assert details["available_days_count"] == 2
    assert details["session_time_seconds"] == 1800
    assert expected_session_count(details["scope"], details["available_days_count"]) == 2


def test_general_advice_is_normalized_away_from_complete_plan():
    from app.services.agent import semantics

    source = "这次出差临时住酒店，当前能用的器械只有哑铃。按这个条件给一个通用力量动作建议，不要改长期限制。"
    raw = {
        "tasks": [{
            "task_type": "training_plan", "action": "question", "text": source,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            "plan_result": True, "result_scope": "independent",
            "delivery_type": "advice", "reply_purpose": "general", "goal_role": "result",
            "reply_mode": "grounded",
        }],
        "reply_mode": "grounded", "reply_purpose": "general",
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"},
    }
    checked = semantics.validate_understanding(source, raw)
    task = checked["tasks"][0]
    assert task["task_type"] == "training_qa"
    assert task["delivery_type"] == "advice"
    assert task["plan_result"] is None


def test_capability_copy_answers_the_asked_operation():
    from app.services.agent.dialogue_context import execution_status_text

    device = execution_status_text("我可以记录", "capability", "device_sync")
    assert "不能直接同步 Apple Watch" in device
    assert "没有已连接或已导入的数据" in device

    plan_save = execution_status_text("可以保存", "capability", "plan_save")
    assert "没有保存训练计划的工具" in plan_save
    assert "当前版本没有可确认的计划保存页面或入口" in plan_save
    assert "请在计划页面保存" not in plan_save

    general = execution_status_text("可以", "capability", None)
    assert "不能直接同步设备" in general
    assert "批量清空" in general


def test_saved_plan_information_offline_replay():
    path = (Path(__file__).resolve().parents[2] / "evidence/user-simulation-20261030-full/"
            "live/synthetic-model-responses.jsonl")
    if not path.exists():
        pytest.skip("saved SIM30 model response is not present")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    record = next(item for item in records if item["call"] == 14)
    info = json.loads(record["messages"][0]["tool_calls"][0]["function"]["arguments"])
    source = ("我成年、无伤病，是力量训练新手。本周只能周二和周六练，每次从热身到结束最多30分钟，"
              "器械只有哑铃。给这两天的完整预览，组间休息也算进30分钟；不要保存或覆盖已存计划。")
    llm = type("LLM", (), {"complete_structured": AsyncMock(return_value=info)})()
    from app.services.agent import task_information
    details, clarification = __import__("asyncio").run(task_information.check_plan_information(
        llm, {"raw_message": source, "task_text": source,
              "profile": {"goal": "提升基础力量"}, "semantic_scope": {}}))
    assert clarification is None
    assert details["available_days_count"] == 2
