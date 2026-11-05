from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.services.agent.planner_validation import (
    PendingRecordContext,
    SPAN_COVERAGE_MIN,
    measure_span_coverage,
    resolve_dependency_outcomes,
    validate_planner_output,
    validate_rule_plan,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "planner_shadow_cases.json"
CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]
BY_ID = {case["id"]: case for case in CASES}


def _rule_fallback(case_id: str) -> list[dict]:
    return BY_ID[case_id]["rule"]


def _seed_safety(source: str, level: str = "normal", evidence: str = "") -> None:
    """The model's request-local safety result for these direct gate cases."""
    from app.services.agent.semantics import remember_safety
    remember_safety(source, {"level": level, "evidence": evidence or source if level != "normal" else "",
                             "categories": [], "reason": "scripted " + level})


def _result(case_id: str):
    case = BY_ID[case_id]
    raw = {"tasks": case["llm"]} if case["llm_status"] == "ok" else "{"  # B20 raw body was not retained.
    _seed_safety(case["message"])
    return validate_planner_output(case["message"], raw, rule_tasks=case["rule"])


def _find_task(result, task_type: str, *, text_contains: str | None = None):
    return next(task for task in result.tasks if task["task_type"] == task_type and (
        text_contains is None or text_contains in task["text"]
    ))


def test_fixture_captures_all_40_real_shadow_cases_and_coverage_numbers():
    assert len(CASES) == 40
    assert [case["id"] for case in CASES] == [f"B{i:02d}" for i in range(1, 21)] + [f"C{i:02d}" for i in range(1, 21)]
    assert BY_ID["B20"]["llm_status"] == "invalid_json"
    assert BY_ID["B20"]["llm"] == []
    measured = {
        case["id"]: measure_span_coverage(case["message"], case["llm"])["coverage"]
        for case in CASES if case["llm_status"] == "ok"
    }
    assert measured["C08"] == pytest.approx(0.462, abs=0.001)
    assert measured["C13"] == pytest.approx(0.462, abs=0.001)
    assert measured["C15"] == pytest.approx(0.333, abs=0.001)
    assert measured["C16"] == pytest.approx(0.300, abs=0.001)
    assert all(measure_span_coverage(case["message"], case["rule"])["coverage"] == 1.0 for case in CASES)
    assert SPAN_COVERAGE_MIN == 0.7


def test_v1_v5_run_purely_over_every_frozen_fixture_case():
    for case in CASES:
        result = _result(case["id"])
        assert result.planner_backend in {"llm", "rule"}, case["id"]
        assert result.tasks, case["id"]
        assert all("source_start" in task and "source_end" in task for task in result.tasks), case["id"]
        assert not any("_v1_write_candidate" in task for task in result.tasks), case["id"]


@pytest.mark.parametrize("case_id", ["B17", "B19", "C09"])
def test_v1_downgrades_contradictory_record_combinations(case_id: str):
    case = BY_ID[case_id]
    result = _result(case_id)
    assert result.planner_backend == "llm"
    assert result.degraded_reason is None
    task = result.tasks[0]
    # The domain label is preserved; only the action becomes a read, so the
    # record read can still be answered from real records.
    assert task["task_type"] == case["llm"][0]["task_type"]
    assert task["action"] == "question"
    assert task["side_effect"] is False
    assert any(event["code"] == "v1_record_action_downgraded" for event in result.trace)


def test_v1_confirm_requires_current_session_matching_unexpired_pending_record():
    source = "确认保存刚才的鸡蛋"
    _seed_safety(source)
    raw = {"tasks": [{
        "task_type": "nutrition_record", "action": "confirm", "text": source,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
    }]}
    now = datetime(2026, 9, 28, 12, tzinfo=timezone.utc)
    rule = [{"id": "pending-food", "task_type": "nutrition_record", "action": "confirm", "text": source}]
    mismatched = PendingRecordContext("food-7", "workout_record", "session-A", now + timedelta(minutes=1))
    denied = validate_planner_output(source, raw, rule_tasks=rule, pending_record=mismatched,
                                     current_session_id="session-A", now=now)
    assert denied.tasks[0]["task_type"] == "nutrition_record"
    assert denied.tasks[0]["action"] == "question"
    assert denied.tasks[0]["side_effect"] is False
    assert any(event["code"] == "v1_confirm_unbound_downgraded" for event in denied.trace)

    matching = PendingRecordContext("food-7", "nutrition_record", "session-A", now + timedelta(minutes=1))
    allowed = validate_planner_output(source, raw, rule_tasks=rule, pending_record=matching,
                                      current_session_id="session-A", now=now)
    assert allowed.tasks[0]["action"] == "confirm"
    assert allowed.tasks[0]["status"] == "authorized"

    expired = PendingRecordContext("food-7", "nutrition_record", "session-A", now - timedelta(seconds=1))
    expired_result = validate_planner_output(source, raw, rule_tasks=rule, pending_record=expired,
                                             current_session_id="session-A", now=now)
    assert expired_result.tasks[0]["action"] == "question"


def test_v1_invalid_json_or_enum_falls_back_with_reason():
    case = BY_ID["B20"]
    invalid_json = validate_planner_output(case["message"], "{", rule_tasks=case["rule"])
    assert invalid_json.planner_backend == "rule"
    assert invalid_json.fallback_required is True
    assert invalid_json.degraded_reason == "planner_invalid_schema"

    source = "记录两个鸡蛋"
    invalid_enum = {"tasks": [{
        "task_type": "food_log", "action": "record", "text": source,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
    }]}
    result = validate_planner_output(source, invalid_enum, rule_tasks=[{
        "task_type": "nutrition_advice", "action": "question", "text": source,
    }])
    assert result.planner_backend == "rule"
    assert result.degraded_reason == "planner_invalid_schema"

    invalid_type = {"tasks": [{
        "task_type": [], "action": "record", "text": source,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
    }]}
    result = validate_planner_output(source, invalid_type, rule_tasks=[{
        "task_type": "nutrition_advice", "action": "question", "text": source,
    }])
    assert result.planner_backend == "rule"
    assert result.degraded_reason == "planner_invalid_schema"


def test_v2_rejects_rewritten_text_and_low_coverage_plans():
    source = "帮我记录今天深蹲，顺便分析饮食"
    rule = [{"task_type": "training_qa", "action": "question", "text": source}]
    rewritten = {"tasks": [{
        "task_type": "workout_record", "action": "record", "text": "保存我的深蹲记录",
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
    }]}
    bad_span = validate_planner_output(source, rewritten, rule_tasks=rule)
    assert bad_span.planner_backend == "rule"
    assert bad_span.degraded_reason == "planner_bad_span"

    for case_id in ("C08", "C13", "C15", "C16"):
        result = _result(case_id)
        assert result.planner_backend == "rule", case_id
        assert result.degraded_reason == "planner_low_coverage", case_id
        assert result.trace[0]["coverage"] < SPAN_COVERAGE_MIN


def test_v2_exact_duplicate_text_is_removed_and_traced():
    source = "记录两个鸡蛋"
    item = {
        "task_type": "nutrition_record", "action": "record", "text": source,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
    }
    result = validate_planner_output(source, {"tasks": [item, dict(item)]}, rule_tasks=[item])
    assert result.planner_backend == "llm"
    assert len(result.tasks) == 1
    assert any(event["code"] == "v2_duplicate_text_removed" for event in result.trace)


def test_v3_required_fixture_cases_and_c14_are_task_scoped():
    for case_id in ("B16", "B18", "C06"):
        first = _result(case_id).tasks[0]
        assert first["status"] == "rejected_unauthorized", case_id

    case = BY_ID["C14"]
    result = _result("C14")
    egg, rice = result.tasks[0], result.tasks[1]
    advice = _find_task(result, "nutrition_advice")
    assert egg["status"] == "rejected_unauthorized"
    assert rice["status"] == "authorized"
    assert advice["id"] in {task["id"] for task in result.tasks}
    assert advice["depends_on"] == [rice["id"]]
    assert advice["status"] == "blocked"  # retained, but waits for its record dependency.
    assert case["message"].startswith("不要记录鸡蛋")


def test_v3_does_not_override_false_model_flags_from_source_words():
    for case_id in ("B18", "C06"):
        case = BY_ID[case_id]
        task = dict(case["llm"][0], is_negated=False, is_hypothetical=False, is_third_party=False)
        _seed_safety(case["message"])
        result = validate_planner_output(case["message"], {"tasks": [task]}, rule_tasks=case["rule"])
        assert result.tasks[0]["status"] == "authorized"
        assert result.tasks[0]["write_rejection_reason"] is None


@pytest.mark.parametrize(
    ("source", "flag", "reason"),
    [
        ("如果我吃了鸡蛋就记录", "is_hypothetical", "planner_flag_hypothetical"),
        ("朋友练了深蹲，记录他的训练", "is_third_party", "planner_flag_third_party"),
        ("不要记录鸡蛋", "is_negated", "planner_flag_negated"),
    ],
)
def test_v3_uses_structured_semantic_flags(source: str, flag: str, reason: str):
    task = {
        "task_type": "workout_record" if "深蹲" in source else "nutrition_record",
        "action": "record", "text": source,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        flag: True,
    }
    _seed_safety(source)
    result = validate_planner_output(source, {"tasks": [task]})
    assert result.tasks[0]["status"] == "rejected_unauthorized"
    assert result.tasks[0]["write_rejection_reason"] == reason


def test_c05_preserves_workout_record_and_hypothetical_safety_question():
    result = _result("C05")
    assert result.planner_backend == "llm"
    workout = _find_task(result, "workout_record")
    safety = _find_task(result, "safety_triage")
    assert workout["status"] == "authorized"
    assert safety["action"] == "question"
    assert "如果受伤怎么办" in safety["text"]


def test_b09_model_semantics_authorize_second_task_without_text_windows():
    result = _result("B09")
    second = next(task for task in result.tasks if "今天深蹲" in task["text"])
    assert second["status"] == "authorized"
    assert "forward_window" not in second


def test_rule_fallback_cannot_authorize_c19_without_model_semantics():
    case = BY_ID["C19"]
    result = validate_rule_plan(case["message"], case["rule"])
    rejected = next(task for task in result.tasks if task["task_type"] == "workout_record")
    assert result.planner_backend == "rule"
    assert rejected["status"] == "needs_confirmation"
    assert rejected["write_rejection_reason"] == "semantic_intent_unavailable"


def test_v4_reorders_advice_before_record_and_v5_blocks_until_record_completes():
    source = "记录鸡蛋，再安排晚餐"
    record = {
        "task_type": "nutrition_record", "action": "record", "text": "记录鸡蛋",
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
    }
    advice = {
        "task_type": "nutrition_advice", "action": "question", "text": "再安排晚餐",
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
    }
    _seed_safety(source)
    result = validate_planner_output(source, {"tasks": [advice, record]}, rule_tasks=[record, advice])
    assert [task["task_type"] for task in result.tasks] == ["nutrition_record", "nutrition_advice"]
    write, suggestion = result.tasks
    assert suggestion["depends_on"] == [write["id"]]
    assert suggestion["status"] == "blocked"
    failed = resolve_dependency_outcomes(result.tasks, {write["id"]: "failed"})
    assert failed[1]["status"] == "blocked"
    succeeded = resolve_dependency_outcomes(result.tasks, {write["id"]: "completed"})
    assert succeeded[1]["status"] == "pending"


def test_v3_urgent_gate_blocks_write_for_the_whole_message():
    source = "记录卧推，出现胸痛和呼吸困难"
    _seed_safety(source, "urgent", source)
    raw = {"tasks": [
        {"task_type": "workout_record", "action": "record", "text": "记录卧推",
         "is_negated": False, "is_hypothetical": False, "is_third_party": False},
        {"task_type": "safety_triage", "action": "question", "text": "出现胸痛和呼吸困难",
         "is_negated": False, "is_hypothetical": False, "is_third_party": False},
    ]}
    result = validate_planner_output(source, raw, rule_tasks=[
        {"task_type": "training_qa", "action": "question", "text": source},
    ])
    assert result.tasks[0]["status"] == "rejected_unauthorized"
    assert result.tasks[0]["write_rejection_reason"] == "urgent_safety_gate"
