"""Tests for adjacent task boundaries."""
from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "plan_dialogue"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _raw_task(task_type: str, text: str, **extra) -> dict:
    task = {
        "task_type": task_type, "action": "question", "text": text,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        "goal_role": "result", "quoted": False, "reply_mode": "grounded",
        "reply_purpose": "general", "delivery_type": "advice",
    }
    task.update(extra)
    return task


def test_replay_nutrition_two_tasks_stay_separate():
    from app.services.agent import semantics, task_planner

    fixture = _fixture("nutrition_advice")
    raw = json.loads(fixture["items"][0]["arguments"])
    source = fixture["source_message"]
    checked = semantics.validate_understanding(source, raw, context={})
    assert len(checked["tasks"]) == 2
    assert len(checked["validated_tasks"]) == 2
    first, second = checked["validated_tasks"]
    assert "蛋白粉" in first["text"] and "日常食物" not in first["text"]
    assert "日常食物" in second["text"] and "蛋白粉" not in second["text"]
    assert all("merged_task_ids" not in task for task in checked["validated_tasks"])

    token = semantics.request_understanding.set({})
    try:
        semantics.remember_understanding(source, checked)
        tasks = task_planner.plan_tasks_rule(source)
    finally:
        semantics.request_understanding.reset(token)
    assert len(tasks) == 2
    assert [task["text"] for task in tasks] == [first["text"], second["text"]]
    assert all(task["payload"]["semantic"]["delivery_type"] == "advice" for task in tasks)


def test_each_task_keeps_its_own_question():
    from app.services.agent import semantics

    fixture = _fixture("nutrition_advice")
    source = fixture["source_message"]
    raw = json.loads(fixture["items"][0]["arguments"])
    checked = semantics.validate_understanding(source, raw, context={})
    tasks = checked["validated_tasks"]
    assert "日常食物" in tasks[1]["text"]
    assert "蛋白粉" in tasks[0]["text"]
    assert tasks[0]["text"] != tasks[1]["text"]


def test_non_contiguous_two_nutrition_questions_not_merged():
    from app.services.agent import semantics

    source = "蛋白粉一定要喝吗？帮我记录今天卧推60kg3组10次。日常食物怎么搭配？"
    raw = {"tasks": [
        _raw_task("nutrition_advice", "蛋白粉一定要喝吗？"),
        _raw_task("workout_record", "帮我记录今天卧推60kg3组10次。", action="record"),
        _raw_task("nutrition_advice", "日常食物怎么搭配？"),
    ], "reply_mode": "grounded", "reply_purpose": "general",
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}}
    checked = semantics.validate_understanding(source, raw, context={})
    assert len(checked["validated_tasks"]) == 3
    assert all("merged_task_ids" not in task for task in checked["validated_tasks"])


def test_record_plus_advice_not_merged():
    from app.services.agent import semantics

    source = "记录今天早餐燕麦45g。营养上怎么搭配更好？"
    raw = {"tasks": [
        _raw_task("nutrition_record", "记录今天早餐燕麦45g。", action="record"),
        _raw_task("nutrition_advice", "营养上怎么搭配更好？"),
    ], "reply_mode": "grounded", "reply_purpose": "general",
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}}
    checked = semantics.validate_understanding(source, raw, context={})
    assert len(checked["validated_tasks"]) == 2
    assert all("merged_task_ids" not in task for task in checked["validated_tasks"])


def test_two_plans_not_merged():
    from app.services.agent import semantics

    source = "给我一个周二计划。再给我一个周六计划。"
    raw = {"tasks": [
        _raw_task("training_plan", "给我一个周二计划。", delivery_type="plan",
                  plan_result=True, result_scope="independent", plan_request_quote="给我一个周二计划"),
        _raw_task("training_plan", "再给我一个周六计划。", delivery_type="plan",
                  plan_result=True, result_scope="independent", plan_request_quote="再给我一个周六计划"),
    ], "reply_mode": "grounded", "reply_purpose": "general",
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}}
    checked = semantics.validate_understanding(source, raw, context={})
    assert len(checked["validated_tasks"]) == 2
    assert all("merged_task_ids" not in task for task in checked["validated_tasks"])


def test_merge_requires_contiguous_source():
    from app.services.agent import semantics

    source = "蛋白粉一定要喝吗？另外请先记住我乳糖不耐受。日常食物怎么搭配？"
    raw = {"tasks": [
        _raw_task("nutrition_advice", "蛋白粉一定要喝吗？"),
        _raw_task("chitchat", "另外请先记住我乳糖不耐受。", reply_purpose="memory_update",
                  reply_mode="conversation"),
        _raw_task("nutrition_advice", "日常食物怎么搭配？"),
    ], "reply_mode": "grounded", "reply_purpose": "general",
        "memory_requests": [{"action": "remember", "quote": "我乳糖不耐受",
                             "key": "lactose_intolerance", "confidence": 0.9}],
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}}
    checked = semantics.validate_understanding(source, raw, context={})
    assert len(checked["validated_tasks"]) == 3
    assert all("merged_task_ids" not in task for task in checked["validated_tasks"])


def test_planner_outline_merge_preserves_semantic_scope():
    from app.services.agent import semantics, task_planner

    fixture = _fixture("nutrition_advice")
    source = fixture["source_message"]
    raw = json.loads(fixture["items"][0]["arguments"])
    checked = semantics.validate_understanding(source, raw, context={})
    parsed = {"authorization_validated": True, "tasks": [{
        "task_type": "nutrition_advice", "action": "question", "text": source,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        "source_start": 0, "source_end": len(source),
        "merged_task_ids": ["task-1", "task-2"],
    }]}
    token = semantics.request_understanding.set({})
    try:
        semantics.remember_understanding(source, checked)
        tasks = task_planner.plan_tasks_from_llm(source, parsed)
    finally:
        semantics.request_understanding.reset(token)
    assert tasks and len(tasks) == 1
    payload = tasks[0]["payload"]
    assert payload["merged_task_ids"] == ["task-1", "task-2"]
    assert payload["semantic"]["delivery_type"] == "advice"
    assert payload["semantic"]["text"] == source


def test_merge_no_keyword_or_regex():
    from app.services.agent import semantics

    source = "蛋白粉一定要喝吗？日常食物怎么搭配？"
    raw = {"tasks": [
        _raw_task("nutrition_advice", "蛋白粉一定要喝吗？", delivery_type="advice"),
        _raw_task("nutrition_advice", "日常食物怎么搭配？", delivery_type="read"),
    ], "reply_mode": "grounded", "reply_purpose": "general",
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}}
    checked = semantics.validate_understanding(source, raw, context={})
    assert len(checked["validated_tasks"]) == 2
    assert all("merged_task_ids" not in task for task in checked["validated_tasks"])
