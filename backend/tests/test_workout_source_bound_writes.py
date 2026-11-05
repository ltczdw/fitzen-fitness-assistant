"""Tests for workout source bound writes."""
from __future__ import annotations

import json
from uuid import uuid4

import pytest

from app.services.agent.workout_extractor import (
    preprocess_workout,
    validate_workout_candidate,
    validate_workout_records_against_source,
)
from app.services.agent.workout_syntax import source_values
from tests.react_phase_c2_3_support import (
    C3HTTPHarness,
    C3SQLFault,
    all_business_rows,
    c3_env,
    mock_provider_transport,
    workout_rows,
)


SOURCE_TWO = (
    "今天我实际完成了哑铃侧平举：单只4公斤，2组12次；"
    "还有哑铃划船：单只9公斤，2组8次。请分别保存这两项。"
)
FACT_TWO = "哑铃侧平举：单只4公斤，2组12次；还有哑铃划船：单只9公斤，2组8次"
REF_TWO_FIRST = "今天我实际完成了哑铃侧平举：单只4公斤，2组12次；"
REF_TWO_SECOND = "还有哑铃划船：单只9公斤，2组8次。"


def semantic_response(tasks, *, safety=None):
    rendered = []
    for task in tasks:
        item = {
            "task_type": task["task_type"],
            "action": task["action"],
            "text": task["text"],
            "fact_text": task.get("fact_text"),
            "is_negated": task.get("is_negated", False),
            "is_hypothetical": task.get("is_hypothetical", False),
            "is_third_party": task.get("is_third_party", False),
            "goal_role": task.get("goal_role", "result"),
            "history_requirement": task.get("history_requirement", "none"),
            "reply_mode": task.get("reply_mode", "grounded"),
            "reply_purpose": task.get("reply_purpose", "general"),
        }
        if task.get("workout_sources"):
            item["workout_sources"] = list(task["workout_sources"])
        rendered.append(item)
    value = {
        "tasks": rendered,
        "safety": safety or {"level": "normal", "evidence": "", "categories": [],
                             "reason": "scripted normal"},
        "needs_clarification": False,
    }
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
        "id": "semantic-result", "type": "function", "function": {
            "name": "return_request_understanding",
            "arguments": json.dumps(value, ensure_ascii=False),
        },
    }]}}], "usage": {"prompt_tokens": 6, "completion_tokens": 4}}


def native(value):
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(
        value, ensure_ascii=False)}}],
        "usage": {"prompt_tokens": 6, "completion_tokens": 4}}


def native_tools(calls):
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        }}
        for call_id, name, args in calls
    ]}}], "usage": {"prompt_tokens": 6, "completion_tokens": 4}}


def ref_of(source, fragment):
    start = source.index(fragment)
    return {"start": start, "end": start + len(fragment), "exact_text": fragment}


def record(exercise, weight, sets, reps):
    return {"exercise": exercise, "weight_kg": weight, "sets": sets, "reps": reps,
            "inferred_fields": [], "record_date": None}


def saved_rows(engine):
    return [(row["exercise"], row["weight_kg"], row["sets"], row["reps"])
            for row in workout_rows(engine)]


# ---------------------------------------------------------------------------
# Numeric contract: technical syntax only; intent/action semantics stay model.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("notation,expected", [
    ("10*12*3", (10.0, 3, 12)),
    ("10 x 12 x 3", (10.0, 3, 12)),
    ("10X12X3", (10.0, 3, 12)),
    ("10×12×3", (10.0, 3, 12)),
    ("10＊12＊3", (10.0, 3, 12)),
    ("10 * 12 * 3", (10.0, 3, 12)),
    ("10kg×3组×12次", (10.0, 3, 12)),
    ("10kg×12次×3组", (10.0, 3, 12)),
    ("负重引体向上5kg×8×3", (5.0, 3, 8)),
    ("引体向上自重*12*3", (0.0, 3, 12)),
    ("徒手×12×3", (0.0, 3, 12)),
])
def test_three_factor_notation_roles(notation, expected):
    values = source_values(notation)
    assert values["weight_kg"] == [expected[0]], f"{notation}: {values}"
    assert values["reps"] == [expected[2]], f"{notation}: {values}"
    assert values["sets"] == [expected[1]], f"{notation}: {values}"


@pytest.mark.parametrize("source,expected", [
    ("哑铃卧推10*12*3", (10.0, 3, 12)),
    ("哑铃卧推10 x 12 x 3", (10.0, 3, 12)),
    ("哑铃卧推10X12X3", (10.0, 3, 12)),
    ("哑铃卧推10×12×3", (10.0, 3, 12)),
    ("哑铃卧推10＊12＊3", (10.0, 3, 12)),
    ("哑铃卧推10 * 12 * 3", (10.0, 3, 12)),
    ("负重引体向上5kg×8×3", (5.0, 3, 8)),
    ("徒手深蹲×12×3", (0.0, 3, 12)),
])
def test_three_factor_rule_parser_and_source_evidence(source, expected):
    parsed = preprocess_workout(source)
    assert parsed.status == "valid", (source, parsed)
    assert (parsed.record["weight_kg"], parsed.record["sets"], parsed.record["reps"]) == expected, (
        source, parsed.record)
    candidate = {"exercise": parsed.record["exercise"],
                 "weight_kg": parsed.record["weight_kg"],
                 "sets": parsed.record["sets"], "reps": parsed.record["reps"]}
    kept, dropped = validate_workout_records_against_source([candidate], "记录" + source)
    assert kept == [candidate] and not dropped, (source, kept, dropped)


def test_unitless_triple_reversed_counts_are_not_saved():
    candidate = {"exercise": "哑铃卧推", "weight_kg": 10, "sets": 12, "reps": 3}
    kept, dropped = validate_workout_records_against_source(
        [candidate], "记录哑铃卧推10*12*3")
    assert not kept and dropped
    assert "sets" in dropped[0]["fields"] or "reps" in dropped[0]["fields"]


def test_single_dumbbell_weight_is_not_doubled_by_source_evidence():
    source = "哑铃侧平举：单只4公斤，2组12次"
    correct = {"exercise": "哑铃侧平举", "weight_kg": 4, "sets": 2, "reps": 12}
    assert validate_workout_records_against_source([correct], source) == ([correct], [])
    doubled = {**correct, "weight_kg": 8}
    kept, dropped = validate_workout_records_against_source([doubled], source)
    assert not kept and dropped and "weight_kg" in dropped[0]["fields"]


def test_bare_bodyweight_expression_is_a_complete_triple():
    source = "徒手×12×3"
    parsed = validate_workout_candidate(
        {"exercise": "徒手", "weight_kg": None, "sets": 3, "reps": 12},
        source, {"normalized_text": source, "inferred_fields": [], "number_mentions": []})
    assert parsed.status == "valid", parsed
    assert (parsed.record["weight_kg"], parsed.record["sets"], parsed.record["reps"]) == (
        0.0, 3, 12)
    candidate = {"exercise": "徒手", "weight_kg": 0, "sets": 3, "reps": 12}
    assert validate_workout_records_against_source([candidate], source) == ([candidate], [])


def test_bodyweight_marker_is_not_borrowed_by_sibling_action():
    source = "记录直臂下压自重*10*4，哑铃弯举*10*4"
    candidate = {"exercise": "哑铃弯举", "weight_kg": 0, "sets": 10, "reps": 4}
    kept, dropped = validate_workout_records_against_source([candidate], source)
    assert not kept and dropped and "weight_kg" in dropped[0]["fields"]

    parsed = validate_workout_candidate(
        {"exercise": "直臂下压", "weight_kg": None, "sets": 4, "reps": 10},
        source, {"normalized_text": source, "inferred_fields": [], "number_mentions": []})
    assert parsed.status == "valid" and parsed.record["weight_kg"] == 0.0


# ---------------------------------------------------------------------------
# Real main chain: ChatRequest -> native supervisor/tools -> bridge -> rows.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["sqlite"])
def test_one_semantic_task_with_two_actions_saves_both_once(c3_env, monkeypatch, backend):
    first = ref_of(SOURCE_TWO, REF_TWO_FIRST)
    second = ref_of(SOURCE_TWO, REF_TWO_SECOND)
    responses = [
        semantic_response([{"task_type": "workout_record", "action": "record",
                            "text": SOURCE_TWO, "fact_text": FACT_TWO}]),
        native_tools([
            ("call-bench", "record_workout", {"source_ref": first}),
            ("call-row", "record_workout", {"source_ref": second}),
        ]),
        native({"records": [record("哑铃侧平举", 4, 2, 12)]}),
        native({"records": [record("哑铃划船", 9, 2, 8)]}),
    ]
    seen = mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    rid = str(uuid4())

    status, payload = harness.post_json(SOURCE_TWO, request_id=rid)
    assert status == 200, payload
    assert saved_rows(c3_env) == [
        ("哑铃侧平举", 4.0, 2, 12),
        ("哑铃划船", 9.0, 2, 8),
    ], (saved_rows(c3_env), payload)
    assert payload["metrics"].get("react_write") == "completed", payload
    assert len(payload["metrics"].get("receipt_refs") or []) >= 2
    assert len(seen) == 4, len(seen)

    before = all_business_rows(c3_env)
    replay_status, replay = harness.post_json(SOURCE_TWO, request_id=rid)
    assert replay_status == 200
    assert replay["metrics"].get("receipt_refs") == payload["metrics"].get("receipt_refs")
    assert all_business_rows(c3_env) == before
    assert len(seen) == 4, "replay must not call the provider again"


@pytest.mark.parametrize("backend", ["sqlite"])
def test_same_action_name_different_weights_are_distinct_rows(c3_env, monkeypatch, backend):
    source = "哑铃弯举：单只5公斤，2组10次；哑铃弯举：单只6公斤，2组8次。都保存。"
    fact = "哑铃弯举：单只5公斤，2组10次；哑铃弯举：单只6公斤，2组8次"
    first = ref_of(source, "哑铃弯举：单只5公斤，2组10次；")
    second = ref_of(source, "哑铃弯举：单只6公斤，2组8次。")
    responses = [
        semantic_response([{"task_type": "workout_record", "action": "record",
                            "text": source, "fact_text": fact}]),
        native_tools([
            ("call-first", "record_workout", {"source_ref": first}),
            ("call-second", "record_workout", {"source_ref": second}),
        ]),
        native({"records": [record("哑铃弯举", 5, 2, 10)]}),
        native({"records": [record("哑铃弯举", 6, 2, 8)]}),
    ]
    seen = mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)

    status, payload = harness.post_json(source, request_id=str(uuid4()))
    assert status == 200, payload
    assert len(payload["metrics"].get("receipt_refs") or []) >= 2, payload
    assert saved_rows(c3_env) == [
        ("哑铃弯举", 5.0, 2, 10),
        ("哑铃弯举", 6.0, 2, 8),
    ], (saved_rows(c3_env), payload)
    assert len(seen) == 4


@pytest.mark.parametrize("backend", ["sqlite"])
def test_resume_after_first_action_does_not_duplicate_or_lose_second(
        c3_env, monkeypatch, backend):
    first = ref_of(SOURCE_TWO, REF_TWO_FIRST)
    second = ref_of(SOURCE_TWO, REF_TWO_SECOND)
    responses = [
        semantic_response([{"task_type": "workout_record", "action": "record",
                            "text": SOURCE_TWO, "fact_text": FACT_TWO}]),
        native_tools([
            ("call-first", "record_workout", {"source_ref": first}),
            ("call-second", "record_workout", {"source_ref": second}),
        ]),
        native({"records": [record("哑铃侧平举", 4, 2, 12)]}),
        native({"records": [record("哑铃划船", 9, 2, 8)]}),
    ]
    seen = mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    rid = str(uuid4())

    def _second_intent(statement, parameters):
        text = str(parameters)
        return "call-second" in text and "tool_intent" in text

    fault = C3SQLFault(c3_env, table="agent_execution_events", action="insert",
                       occurrence=1, match=_second_intent)
    fault.install()
    try:
        status, payload = harness.post_json(SOURCE_TWO, request_id=rid)
    finally:
        fault.remove()
    assert status == 200, payload
    assert fault.raised is True
    assert saved_rows(c3_env) == [("哑铃侧平举", 4.0, 2, 12)], saved_rows(c3_env)

    resume_status, resumed = harness.post_json(SOURCE_TWO, request_id=rid)
    assert resume_status == 200, resumed
    assert saved_rows(c3_env) == [
        ("哑铃侧平举", 4.0, 2, 12),
        ("哑铃划船", 9.0, 2, 8),
    ], saved_rows(c3_env)
    assert resumed["metrics"].get("react_write") == "completed", resumed
    assert len(resumed["metrics"].get("receipt_refs") or []) >= 2
    assert len(seen) == 4

    before = all_business_rows(c3_env)
    replay_status, replay = harness.post_json(SOURCE_TWO, request_id=rid)
    assert replay_status == 200
    assert replay["metrics"].get("receipt_refs") == resumed["metrics"].get("receipt_refs")
    assert all_business_rows(c3_env) == before
    assert len(seen) == 4


@pytest.mark.parametrize("backend", ["sqlite"])
def test_full_goal_resume_replays_both_without_duplicates(c3_env, monkeypatch, backend):
    units = [
        {"exercise": "哑铃侧平举", "quote": "哑铃侧平举：单只4公斤，2组12次"},
        {"exercise": "哑铃划船", "quote": "哑铃划船：单只9公斤，2组8次"},
    ]
    responses = [
        semantic_response([{"task_type": "workout_record", "action": "record",
                            "text": SOURCE_TWO, "fact_text": FACT_TWO,
                            "workout_sources": units}]),
        native_tools([("save-full-goal", "record_workout",
                       {"source_ref": ref_of(SOURCE_TWO, SOURCE_TWO)})]),
        native({"records": [record("哑铃侧平举", 4, 2, 12),
                            record("哑铃划船", 9, 2, 8)]}),
        native({"records": [record("哑铃侧平举", 4, 2, 12),
                            record("哑铃划船", 9, 2, 8)]}),
    ]
    seen = mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    rid = str(uuid4())

    def _tool_result(statement, parameters):
        text = str(parameters)
        return "tool_result" in text and rid in text

    fault = C3SQLFault(c3_env, table="agent_execution_events", action="insert",
                       occurrence=1, match=_tool_result)
    fault.install()
    try:
        status, payload = harness.post_json(SOURCE_TWO, request_id=rid)
    finally:
        fault.remove()
    assert status == 200, payload
    assert fault.raised is True
    assert saved_rows(c3_env) == [
        ("哑铃侧平举", 4.0, 2, 12),
        ("哑铃划船", 9.0, 2, 8),
    ], saved_rows(c3_env)

    resume_status, resumed = harness.post_json(SOURCE_TWO, request_id=rid)
    assert resume_status == 200, resumed
    assert resumed["metrics"].get("react_write") == "completed", resumed
    assert len(resumed["metrics"].get("receipt_refs") or []) >= 2
    assert saved_rows(c3_env) == [
        ("哑铃侧平举", 4.0, 2, 12),
        ("哑铃划船", 9.0, 2, 8),
    ], saved_rows(c3_env)
    # The persisted C1 operations let the resume recover the terminal without
    # buying another extraction call; no action is lost or duplicated.
    assert len(seen) == 3

    before = all_business_rows(c3_env)
    replay_status, replay = harness.post_json(SOURCE_TWO, request_id=rid)
    assert replay_status == 200
    assert replay["metrics"].get("receipt_refs") == resumed["metrics"].get("receipt_refs")
    assert all_business_rows(c3_env) == before
    assert len(seen) == 3


@pytest.mark.parametrize("backend", ["sqlite"])
def test_two_single_action_requests_each_save_once(c3_env, monkeypatch, backend):
    first_source = "帮我记录哑铃卧推10kg3组10次。"
    second_source = "帮我记录哑铃划船9kg2组8次。"
    first_fact = "哑铃卧推10kg3组10次"
    second_fact = "哑铃划船9kg2组8次"
    responses = [
        semantic_response([{"task_type": "workout_record", "action": "record",
                            "text": first_source, "fact_text": first_fact}]),
        native_tools([("call-first", "record_workout",
                       {"source_ref": ref_of(first_source, first_fact)})]),
        native({"records": [record("哑铃卧推", 10, 3, 10)]}),
        semantic_response([{"task_type": "workout_record", "action": "record",
                            "text": second_source, "fact_text": second_fact}]),
        native_tools([("call-second", "record_workout",
                       {"source_ref": ref_of(second_source, second_fact)})]),
        native({"records": [record("哑铃划船", 9, 2, 8)]}),
    ]
    seen = mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)

    first_status, first_payload = harness.post_json(first_source, request_id=str(uuid4()))
    second_status, second_payload = harness.post_json(second_source, request_id=str(uuid4()))
    assert first_status == second_status == 200
    assert first_payload["metrics"].get("react_write") == "completed"
    assert second_payload["metrics"].get("react_write") == "completed"
    assert saved_rows(c3_env) == [
        ("哑铃卧推", 10.0, 3, 10),
        ("哑铃划船", 9.0, 2, 8),
    ], saved_rows(c3_env)
    assert len(seen) == 6


@pytest.mark.parametrize("source,fact", [
    ("记录哑铃卧推10*12*3", "哑铃卧推10*12*3"),
    ("记录哑铃弯举10kg×3组×12次", "哑铃弯举10kg×3组×12次"),
])
@pytest.mark.parametrize("backend", ["sqlite"])
def test_source_notation_saves_weight_reps_sets_in_order(c3_env, monkeypatch, backend,
                                                        source, fact):
    responses = [
        semantic_response([{"task_type": "workout_record", "action": "record",
                            "text": source, "fact_text": fact}]),
        native_tools([("call-notation", "record_workout",
                       {"source_ref": ref_of(source, fact)})]),
        native({"records": [record("哑铃卧推" if "卧推" in fact else "哑铃弯举", 10, 3, 12)]}),
    ]
    seen = mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)

    status, payload = harness.post_json(source, request_id=str(uuid4()))
    assert status == 200, payload
    assert payload["metrics"].get("react_write") == "completed", payload
    rows = saved_rows(c3_env)
    assert len(rows) == 1 and rows[0][1:] == (10.0, 3, 12), rows
    assert len(seen) == 3


@pytest.mark.parametrize("backend", ["sqlite"])
def test_swapped_unitless_counts_are_not_saved(c3_env, monkeypatch, backend):
    source = "记录哑铃卧推10*12*3"
    fact = "哑铃卧推10*12*3"
    responses = [
        semantic_response([{"task_type": "workout_record", "action": "record",
                            "text": source, "fact_text": fact}]),
        native_tools([("call-swapped", "record_workout",
                       {"source_ref": ref_of(source, fact)})]),
        native({"records": [record("哑铃卧推", 10, 12, 3)]}),
    ]
    seen = mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)

    status, payload = harness.post_json(source, request_id=str(uuid4()))
    assert status == 200, payload
    assert all_business_rows(c3_env) == []
    assert payload["metrics"].get("react_write") != "completed"
    answer = str(payload.get("answer") or "")
    assert "组数" in answer or "次数" in answer, answer


@pytest.mark.parametrize("backend", ["sqlite"])
def test_separate_semantic_tasks_still_save_both(c3_env, monkeypatch, backend):
    source = "帮我记录哑铃卧推10kg3组10次。再记录哑铃划船9kg2组8次。"
    first_text = "帮我记录哑铃卧推10kg3组10次。"
    second_text = "再记录哑铃划船9kg2组8次。"
    responses = [
        semantic_response([
            {"task_type": "workout_record", "action": "record", "text": first_text,
             "fact_text": "哑铃卧推10kg3组10次"},
            {"task_type": "workout_record", "action": "record", "text": second_text,
             "fact_text": "哑铃划船9kg2组8次"},
        ]),
        native_tools([
            ("call-first", "record_workout", {"source_ref": ref_of(source, first_text)}),
            ("call-second", "record_workout", {"source_ref": ref_of(source, second_text)}),
        ]),
        native({"records": [record("哑铃卧推", 10, 3, 10)]}),
        native({"records": [record("哑铃划船", 9, 2, 8)]}),
    ]
    seen = mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)

    status, payload = harness.post_json(source, request_id=str(uuid4()))
    assert status == 200, payload
    assert payload["metrics"].get("react_write") == "completed", payload
    assert saved_rows(c3_env) == [
        ("哑铃卧推", 10.0, 3, 10),
        ("哑铃划船", 9.0, 2, 8),
    ], saved_rows(c3_env)
    assert len(seen) == 4


@pytest.mark.parametrize("backend", ["sqlite"])
def test_complete_action_saves_and_missing_action_asks_specific_field(c3_env, monkeypatch, backend):
    source = "今天完成哑铃卧推10公斤3组10次；还有哑铃划船2组8次，重量忘了。请保存。"
    fact = "哑铃卧推10公斤3组10次；还有哑铃划船2组8次，重量忘了"
    first = ref_of(source, "今天完成哑铃卧推10公斤3组10次；")
    second = ref_of(source, "还有哑铃划船2组8次，重量忘了。")
    responses = [
        semantic_response([{"task_type": "workout_record", "action": "record",
                            "text": source, "fact_text": fact}]),
        native_tools([
            ("call-complete", "record_workout", {"source_ref": first}),
            ("call-missing", "record_workout", {"source_ref": second}),
        ]),
        native({"records": [record("哑铃卧推", 10, 3, 10)]}),
        native({"records": [record("哑铃划船", None, 2, 8)]}),
    ]
    seen = mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)

    status, payload = harness.post_json(source, request_id=str(uuid4()))
    assert status == 200, payload
    assert saved_rows(c3_env) == [("哑铃卧推", 10.0, 3, 10)], saved_rows(c3_env)
    answer = str(payload.get("answer") or "")
    assert "哑铃划船" in answer and "重量" in answer, answer
    assert payload["metrics"].get("react_write") != "completed", payload


@pytest.mark.parametrize("backend", ["sqlite"])
def test_missing_weight_cannot_borrow_sibling_action_value(c3_env, monkeypatch, backend):
    source = "今天完成哑铃卧推10公斤3组10次；还有哑铃划船2组8次。请保存。"
    fact = "哑铃卧推10公斤3组10次；还有哑铃划船2组8次"
    first = ref_of(source, "今天完成哑铃卧推10公斤3组10次；")
    second = ref_of(source, "还有哑铃划船2组8次。")
    responses = [
        semantic_response([{"task_type": "workout_record", "action": "record",
                            "text": source, "fact_text": fact}]),
        native_tools([
            ("call-complete", "record_workout", {"source_ref": first}),
            ("call-borrowed", "record_workout", {"source_ref": second}),
        ]),
        native({"records": [record("哑铃卧推", 10, 3, 10)]}),
        # The extractor wrongly borrows the sibling's 10kg. Source evidence must
        # drop it and clarify this action instead of saving a fabricated weight.
        native({"records": [record("哑铃划船", 10, 2, 8)]}),
    ]
    seen = mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)

    status, payload = harness.post_json(source, request_id=str(uuid4()))
    assert status == 200, payload
    assert saved_rows(c3_env) == [("哑铃卧推", 10.0, 3, 10)], saved_rows(c3_env)
    answer = str(payload.get("answer") or "")
    assert "哑铃划船" in answer and "重量" in answer, answer


@pytest.mark.parametrize("flags", [
    {"is_negated": True},
    {"is_hypothetical": True},
    {"is_third_party": True},
])
@pytest.mark.parametrize("backend", ["sqlite"])
def test_non_authorized_flags_never_save_multi_action(c3_env, monkeypatch, backend, flags):
    first = ref_of(SOURCE_TWO, REF_TWO_FIRST)
    second = ref_of(SOURCE_TWO, REF_TWO_SECOND)
    task = {"task_type": "workout_record", "action": "record",
            "text": SOURCE_TWO, "fact_text": FACT_TWO, **flags}
    responses = [
        semantic_response([task]),
        native_tools([
            ("call-first", "record_workout", {"source_ref": first}),
            ("call-second", "record_workout", {"source_ref": second}),
        ]),
        native({"choices": [{"message": {"role": "assistant", "content": "未保存"}}]}),
    ]
    seen = mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)

    status, payload = harness.post_json(SOURCE_TWO, request_id=str(uuid4()))
    assert status == 200, payload
    assert all_business_rows(c3_env) == [], all_business_rows(c3_env)
    assert not payload["metrics"].get("receipt_refs")
    assert len(seen) <= 3


@pytest.mark.parametrize("backend", ["sqlite"])
def test_record_domain_question_is_not_write_authorization(c3_env, monkeypatch, backend):
    source = "哑铃卧推10公斤3组10次，我只是问该怎么记录。"
    fact = "哑铃卧推10公斤3组10次"
    responses = [
        semantic_response([{"task_type": "workout_record", "action": "question",
                            "text": source, "fact_text": fact,
                            "reply_purpose": "general"}]),
        native_tools([("call-question", "record_workout",
                       {"source_ref": ref_of(source, fact)})]),
        native({"choices": [{"message": {"role": "assistant",
                                         "content": "这是只读问题，未保存。"}}]}),
    ]
    seen = mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)

    status, payload = harness.post_json(source, request_id=str(uuid4()))
    assert status == 200, payload
    assert all_business_rows(c3_env) == []
    assert not payload["metrics"].get("receipt_refs")
    assert len(seen) <= 3


@pytest.mark.parametrize("backend", ["sqlite"])
def test_bad_reference_is_rejected_without_write(c3_env, monkeypatch, backend):
    source = "帮我记录卧推60kg4组8次。"
    fact = "卧推60kg4组8次"
    bad = ref_of(source, fact)
    bad["exact_text"] = fact + "（模型改写）"
    responses = [
        semantic_response([{"task_type": "workout_record", "action": "record",
                            "text": source, "fact_text": fact}]),
        native_tools([("call-bad", "record_workout", {"source_ref": bad})]),
        native({"choices": [{"message": {"role": "assistant", "content": "引用不合法，未保存。"}}]}),
    ]
    seen = mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)

    status, payload = harness.post_json(source, request_id=str(uuid4()))
    assert status == 200, payload
    assert all_business_rows(c3_env) == []
    assert not payload["metrics"].get("receipt_refs")
