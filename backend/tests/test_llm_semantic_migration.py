"""Primary regression for the 2026-10-27 semantic-migration batch.

These cases exercise the real application paths that now depend on a
source-bound model decision: RequestUnderstanding/safety, plan-goal structure,
planning-information proof, the four-phase time contract, output guarding,
read-goal completion, and the JSON/SSE replay boundary. All model output is a
provider-boundary mock; no production keyword list decides a semantic case.
"""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.services.agent import semantics
from app.services.agent.planner_validation import validate_planner_output
from app.services.agent.semantics import (
    RequestUnderstanding,
    SemanticValidationError,
    get_understanding,
    remember_understanding,
    safety_for,
    understand,
    validate_understanding,
)
from app.services.agent.task_information import PlanInformation, check_plan_information
from app.services.agent.training_timing import render_timed_plan
from app.services.security.output_guard import apply_output_guardrail
from app.services.security.write_guard import guard_planner_tasks
from evals.v2.safety import deny_network


@pytest.fixture(autouse=True)
def request_scope():
    token = semantics.request_understanding.set({})
    with deny_network():
        yield
    semantics.request_understanding.reset(token)


class ScriptedModel:
    """Provider-boundary mock returning exact structured semantics."""

    def __init__(self, value, *, fail=None):
        self.value = value
        self.fail = fail
        self.calls = 0

    def configured(self):
        return True

    async def complete_structured(self, messages, schema, *, name=None,
                                  temperature=0.0, validator=None, **_kwargs):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        value = schema.model_validate(deepcopy(self.value))
        if validator is not None:
            validator(value)
        return value


def _safety(level="normal", evidence="", categories=None, reason="scripted"):
    return {"level": level, "evidence": evidence,
            "categories": list(categories or []), "reason": reason}


def _task(kind, text, action="question", **extra):
    item = {"task_type": kind, "action": action, "text": text,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False}
    item.update(extra)
    return item


_UNSET = object()


def _understanding(tasks, safety=_UNSET, **extra):
    return {"tasks": [{**task, "history_requirement": task.get("history_requirement", "none")}
                      for task in tasks],
            "safety": _safety() if safety is _UNSET else safety, **extra}


def _info(*, facts=(), phase_limits=(), scope="session", scope_quote=""):
    return PlanInformation.model_validate({
        "scope": scope, "scope_quote": scope_quote,
        "facts": list(facts), "phase_limits": list(phase_limits),
    })


def _fact(field, quote, *, source="current", quantity="", seconds=None, count=None):
    value = {"field": field, "source_id": source, "quote": quote,
             "quantity_quote": quantity}
    if seconds is not None:
        value["seconds"] = seconds
    if count is not None:
        value["count"] = count
    return value


def _phase(phase, relation, context, quantity, seconds, qualifiers=()):
    return {"phase": phase, "relation": relation, "seconds": seconds,
            "source_id": "current", "context_quote": context,
            "quantity_quote": quantity, "qualifiers": list(qualifiers)}


# ---------------------------------------------------------------------------
# Model-owned plan structure and conditions
# ---------------------------------------------------------------------------


def test_model_plan_structure_accepts_new_wording_and_independent_goals():
    source = "劳烦拟一份哑铃增肌课表，另拟一份弹力带力量课表。"
    tasks = [
        _task("training_plan", "劳烦拟一份哑铃增肌课表", plan_result=True,
              result_scope="independent", plan_request_quote="拟一份哑铃增肌课表"),
        _task("training_plan", "另拟一份弹力带力量课表。", plan_result=True,
              result_scope="independent", plan_request_quote="拟一份弹力带力量课表"),
    ]
    value = validate_understanding(source, _understanding(tasks))
    assert [item["text"] for item in value["validated_tasks"]] == [item["text"] for item in tasks]
    assert all(not item["side_effect"] for item in value["validated_tasks"])


def test_model_plan_structure_accepts_one_shared_three_result_directive():
    source = ("我需要三份各20分钟的独立训练方案：一份只用哑铃，"
              "一份只用弹力带，一份只用自重。均只给建议。")
    tasks = [
        _task("training_plan", "我需要三份各20分钟的独立训练方案：一份只用哑铃",
              plan_result=True, result_scope="shared_leader", plan_goal_id="shared-1",
              shared_scope_quote="三份各20分钟的独立训练方案"),
        _task("training_plan", "一份只用弹力带", plan_result=True,
              result_scope="shared_member", plan_goal_id="shared-1"),
        _task("training_plan", "一份只用自重。均只给建议。", plan_result=True,
              result_scope="shared_member", plan_goal_id="shared-1"),
    ]
    value = validate_understanding(source, _understanding(tasks))
    assert len(value["validated_tasks"]) == 3
    assert [task["source_start"] for task in value["validated_tasks"]] == sorted(
        task["source_start"] for task in value["validated_tasks"])


def test_conditions_are_attached_to_a_goal_and_never_become_phantom_plans():
    source = ("增肌新手、只有哑铃；每周两练，每次30分钟。"
              "先回顾刚记的重量，再给周计划。这次不保存。")
    bad = [
        _task("training_qa", "增肌新手、只有哑铃；每周两练，每次30分钟。", action="answer"),
        _task("training_plan", "先回顾刚记的重量", plan_result=False),
        _task("training_plan", "再给周计划。这次不保存。", plan_result=True,
              result_scope="independent", plan_request_quote="给周计划"),
    ]
    with pytest.raises(SemanticValidationError) as error:
        validate_understanding(source, _understanding(bad))
    assert error.value.code == "semantic_condition_split_as_plan"

    # Correct model shape: the shared conditions stay inside the adjacent
    # read-only span, and only the final directive is a plan task.
    good = [
        _task("training_qa", "增肌新手、只有哑铃；每周两练，每次30分钟。先回顾刚记的重量",
              action="answer"),
        _task("training_plan", "再给周计划。这次不保存。", plan_result=True,
              result_scope="independent", plan_request_quote="给周计划"),
    ]
    value = validate_understanding(source, _understanding(good))
    assert [item["task_type"] for item in value["validated_tasks"]] == [
        "training_qa", "training_plan"]
    assert all(not item["side_effect"] for item in value["validated_tasks"])


def test_foreign_or_rewritten_plan_span_is_rejected_before_workers():
    source = "只回顾已经记录的重量"
    raw = _understanding([
        _task("training_plan", "模型编造的三份增肌计划", plan_result=True,
              result_scope="independent", plan_request_quote="编造的三份增肌计划")])
    with pytest.raises(SemanticValidationError):
        validate_understanding(source, raw)


# ---------------------------------------------------------------------------
# Planning information: complete context and quantity proof
# ---------------------------------------------------------------------------


def test_planning_information_accepts_english_bounds_with_complete_context():
    source = ("Beginner strength training with dumbbells only;"
              "Make one 18 minute session;"
              "Warm-up exactly 3 minutes;cool-down at least 2 minutes;")
    info = _info(
        facts=[
            _fact("goal", "strength training"),
            _fact("experience", "Beginner"),
            _fact("equipment", "dumbbells only"),
            _fact("session_time", "Make one 18 minute session",
                  quantity="18 minute", seconds=1080),
        ],
        phase_limits=[
            _phase("warmup", "exact", "Warm-up exactly 3 minutes", "3 minutes", 180),
            _phase("cooldown", "at_least", "cool-down at least 2 minutes", "2 minutes", 120),
        ],
    )
    details, clarification = asyncio.run(check_plan_information(
        ScriptedModel(info), {"raw_message": source, "task_text": source}))
    assert clarification is None
    assert details["session_time_seconds"] == 1080
    assert details["phase_limits"] == [
        {"phase": "warmup", "relation": "exact", "seconds": 180,
         "quote": "Warm-up exactly 3 minutes", "quantity_quote": "3 minutes"},
        {"phase": "cooldown", "relation": "at_least", "seconds": 120,
         "quote": "cool-down at least 2 minutes", "quantity_quote": "2 minutes"},
    ]
    answer = render_timed_plan(_timed_json(1, warmup=180, main=600, breaks=180, cooldown=120),
                               limit_seconds=1080, expected_sessions=1,
                               phase_limits=details["phase_limits"])
    assert "| 收尾／放松 | 2分钟 |" in answer


@pytest.mark.parametrize("change", ["short_context", "quantity_mismatch", "quantity_outside",
                                    "foreign_source"])
def test_planning_phase_limits_close_on_bad_model_proof(change):
    source = ("增肌新手，无伤病，只有哑铃，总共20分钟。"
              "热身固定3分钟；放松不得少于两分钟。")
    limit = _phase("warmup", "exact", "热身固定3分钟", "3分钟", 180)
    if change == "short_context":
        limit["context_quote"] = "3分钟"
    elif change == "quantity_mismatch":
        limit["seconds"] = 181
    elif change == "quantity_outside":
        limit["quantity_quote"] = "5分钟"
    elif change == "foreign_source":
        limit["source_id"] = "history:0"
    info = _info(
        facts=[_fact("goal", "增肌新手"), _fact("experience", "新手"),
               _fact("equipment", "只有哑铃"), _fact("session_time", "总共20分钟",
                                                    quantity="20分钟", seconds=1200)],
        phase_limits=[limit])
    details, clarification = asyncio.run(check_plan_information(
        ScriptedModel(info), {"raw_message": source, "task_text": source}))
    assert clarification is not None
    assert clarification.category == "service_unavailable"


@pytest.mark.parametrize("qualifier,context,quantity,seconds", [
    ("negated", "热身不要求固定3分钟", "3分钟", 180),
    ("approximate", "热身大约3分钟", "3分钟", 180),
    ("range", "热身两三分钟", "两三分钟", 150),
])
def test_qualified_phase_limit_is_soft_not_a_plan_failure(qualifier, context, quantity, seconds):
    """Model-declared soft wording never becomes a hard budget or total failure."""
    source = f"增肌新手，无伤病，只有哑铃，总共20分钟。{context}。"
    info = _info(
        facts=[_fact("goal", "增肌新手"), _fact("experience", "新手"),
               _fact("equipment", "只有哑铃"), _fact("session_time", "总共20分钟",
                                                    quantity="20分钟", seconds=1200)],
        phase_limits=[_phase("warmup", "exact", context, quantity, seconds,
                             qualifiers=[qualifier])])
    details, clarification = asyncio.run(check_plan_information(
        ScriptedModel(info), {"raw_message": source, "task_text": source}))
    assert clarification is None
    assert details["phase_limits"] == []
    assert details["session_time_seconds"] == 1200
    assert context in source and quantity in context


def test_original_1200_second_payload_is_two_models_of_the_same_bound_limit():
    source = "新手哑铃增肌，每周两次，每次20分钟。"
    info = _info(
        scope="weekly", scope_quote="每周两次",
        facts=[
            _fact("goal", "增肌"), _fact("experience", "新手"),
            _fact("equipment", "哑铃"), _fact("session_time", "每次20分钟",
                                              quantity="20分钟", seconds=1200),
            _fact("available_days", "每周两次", quantity="两次", count=2),
        ])
    details, clarification = asyncio.run(check_plan_information(
        ScriptedModel(info), {"raw_message": source, "task_text": source}))
    assert clarification is None
    assert details["session_time_seconds"] == 1200
    assert details["available_days_count"] == 2
    answer = render_timed_plan(_timed_json(2), limit_seconds=1200, expected_sessions=2)
    assert answer.count("| 收尾／放松 | 2分钟 |") == 2
    assert "累计40分钟" in answer


# ---------------------------------------------------------------------------
# Four-phase time math and model-consistency field
# ---------------------------------------------------------------------------


def _timed_json(count=1, *, warmup=120, main=660, breaks=300, cooldown=120,
                time_consistency="consistent"):
    session = {
        "warmup": {"seconds": warmup, "activity": "关节活动和轻负荷热身"},
        "main": {"seconds": main, "exercises": [
            {"name": "哑铃深蹲", "sets": 2, "reps": 10},
            {"name": "哑铃划船", "sets": 2, "reps": 10}]},
        "breaks": {"seconds": breaks},
        "cooldown": {"seconds": cooldown, "activity": "轻缓活动与放松"},
    }
    if time_consistency is not None:
        session["time_consistency"] = time_consistency
    payload = {"sessions": [{"label": f"训练{index + 1}", **deepcopy(session)}
                            for index in range(count)]}
    return json.dumps(payload, ensure_ascii=False)


def test_four_phase_sum_passes_at_the_limit_and_rejects_one_second_over():
    assert render_timed_plan(_timed_json(), limit_seconds=1200, expected_sessions=1)
    with pytest.raises(ValueError, match="session_time_exceeded"):
        render_timed_plan(_timed_json(cooldown=121), limit_seconds=1200, expected_sessions=1)


def test_timed_exercise_hold_seconds_cannot_exceed_the_main_phase():
    payload = json.loads(_timed_json(main=600))
    payload["sessions"][0]["main"]["exercises"] = [
        {"name": "平板支撑", "sets": 3, "hold_seconds": 300}]
    with pytest.raises(ValueError, match="timed_exercises_exceed_main_phase"):
        render_timed_plan(json.dumps(payload), limit_seconds=1200, expected_sessions=1)


def test_model_consistency_field_is_required_and_prose_is_model_owned():
    with pytest.raises(ValueError):
        render_timed_plan(_timed_json(time_consistency=None), limit_seconds=1200,
                          expected_sessions=1)
    with pytest.raises(ValueError):
        render_timed_plan(_timed_json(time_consistency="inconsistent"), limit_seconds=1200,
                          expected_sessions=1)
    # The model explicitly marked the structured result consistent. The
    # application no longer re-reads negative/extra prose with a numeric scan.
    payload = json.loads(_timed_json())
    payload["sessions"][0]["cooldown"]["activity"] = "不要持续保持10分钟，只做轻缓拉伸"
    answer = render_timed_plan(json.dumps(payload), limit_seconds=1200, expected_sessions=1)
    assert "不要持续保持10分钟" in answer


def test_source_bound_phase_limits_are_enforced_after_generation():
    limits = [{"phase": "warmup", "relation": "exact", "seconds": 180},
              {"phase": "cooldown", "relation": "at_least", "seconds": 120}]
    assert render_timed_plan(_timed_json(warmup=180, main=720, breaks=180, cooldown=120),
                             limit_seconds=1200, expected_sessions=1, phase_limits=limits)
    with pytest.raises(ValueError, match="phase_limit_violation"):
        render_timed_plan(_timed_json(warmup=181, main=719, breaks=180, cooldown=120),
                          limit_seconds=1200, expected_sessions=1, phase_limits=limits)


# ---------------------------------------------------------------------------
# Output guard: semantic text is untouched; technical leaks are not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "我不能保证训练效果。",
    "不要忍痛训练。",
    "请勿自行停药。",
    "这里引用“绝对不会”是为了说明这个承诺不可信。",
])
def test_output_guard_keeps_negation_and_quotation_unchanged(text):
    observed = apply_output_guardrail(text)
    assert observed.text == text
    assert observed.triggered_rules == ()


def test_output_guard_still_redacts_credentials_pii_and_markup():
    observed = apply_output_guardrail(
        "api_key=synthetic-not-real 13800138000 user@example.com <script>alert(1)</script>")
    assert "synthetic-not-real" not in observed.text
    assert "13800138000" not in observed.text
    assert "user@example.com" not in observed.text
    assert "<script>" not in observed.text


# ---------------------------------------------------------------------------
# Safety: source-bound model judgment, reuse and request isolation
# ---------------------------------------------------------------------------


def test_urgent_judgment_is_source_bound_and_reused_within_the_request():
    source = "刚举起来就觉得胸前被压着，怎么都吸不够空气。"
    model = ScriptedModel(_understanding(
        [_task("safety_triage", source)],
        safety=_safety("urgent", source, ["current_symptom"], "current emergency")))
    first = asyncio.run(understand(model, source))
    assert first and model.calls == 1
    assert safety_for(source)["level"] == "urgent"
    assert asyncio.run(understand(model, source)) == first
    assert model.calls == 1
    assert semantics.semantic_candidates(source)[1] is None


def test_missing_or_cross_request_safety_is_unknown_and_cannot_authorize_write():
    source = "请把今天的训练存下来。"
    model = ScriptedModel(_understanding(
        [_task("workout_record", source, action="record")], safety=None))
    value = asyncio.run(understand(model, source))
    # A structured understanding without safety is not a usable partial result:
    # the whole attempt fails closed and is reused as unavailable.
    assert value is None
    assert get_understanding(source) is None
    assert semantics.semantic_attempt_failed(source)
    assert safety_for(source)["status"] == "unavailable"
    assert safety_for(source)["level"] == "unknown"

    checked = validate_planner_output(source, {"tasks": [
        _task("workout_record", source, action="record")]})
    assert checked.tasks[0]["write_authorization"] == "needs_confirmation"
    assert checked.tasks[0]["write_rejection_reason"] == "safety_assessment_unavailable"
    assert checked.tasks[0]["side_effect"] is False

    # A second request gets a new scope: no cache or safety judgment leaks in.
    token = semantics.request_understanding.set({})
    try:
        assert get_understanding(source) is None
        assert safety_for(source)["status"] == "unavailable"
    finally:
        semantics.request_understanding.reset(token)


def test_bad_safety_evidence_fails_closed_without_caching_normal():
    source = "请把今天的训练存下来。"
    model = ScriptedModel(_understanding(
        [_task("workout_record", source, action="record")],
        safety=_safety("urgent", "完全不在原文里的证据", ["current_symptom"], "bad")))
    assert asyncio.run(understand(model, source)) is None
    assert get_understanding(source) is None
    assert safety_for(source)["status"] == "unavailable"
    assert semantics.semantic_attempt_failed(source)


def test_negated_history_and_mixed_judgments_follow_the_model_structure():
    negated = "今天没有胸痛，也没有呼吸困难，只问一般热身原则。"
    normal = ScriptedModel(_understanding(
        [_task("training_qa", negated)], safety=_safety("normal")))
    assert asyncio.run(understand(normal, negated))
    assert safety_for(negated)["level"] == "normal"

    history = "去年那次胸口不适已经恢复；现在没有症状，想了解恢复训练的通用原则。"
    medical = ScriptedModel(_understanding(
        [_task("training_qa", history)],
        safety=_safety("medical", "去年那次胸口不适已经恢复", ["medical_history"], "resolved")))
    assert asyncio.run(understand(medical, history))
    assert safety_for(history)["level"] == "medical"

    mixed = "去年的腿伤好了，但今天训练时胸口突然痛，呼吸也费劲。"
    urgent = ScriptedModel(_understanding(
        [_task("safety_triage", mixed)],
        safety=_safety("urgent", "今天训练时胸口突然痛，呼吸也费劲",
                       ["current_symptom"], "current overrides history")))
    assert asyncio.run(understand(urgent, mixed))
    assert safety_for(mixed)["level"] == "urgent"
    tasks = [{"task_type": "workout_record", "action": "record", "text": mixed,
              "is_negated": False, "is_hypothetical": False, "is_third_party": False}]
    guard_planner_tasks(mixed, tasks, now=datetime.now(timezone.utc), semantic_available=True)
    assert tasks[0]["write_authorization"] == "rejected_unauthorized"
    assert tasks[0]["side_effect"] is False


def test_medical_history_does_not_become_a_current_emergency():
    source = "去年那次胸口不适已经恢复；现在没有症状，想了解恢复训练的通用原则。"
    model = ScriptedModel(_understanding(
        [_task("training_qa", source)],
        safety=_safety("medical", "去年那次胸口不适已经恢复", ["medical_history"])))
    assert asyncio.run(understand(model, source))
    tasks = [{"task_type": "workout_record", "action": "record", "text": source,
              "is_negated": False, "is_hypothetical": False, "is_third_party": False}]
    guard_planner_tasks(source, tasks, now=datetime.now(timezone.utc), semantic_available=True)
    assert tasks[0]["write_authorization"] == "authorized"


def test_rule_fallback_keeps_the_established_read_only_reason():
    source = "记录卧推60kg4组8次"
    rule = [{"task_type": "workout_record", "action": "record", "text": source}]
    result = validate_planner_output(source, "{", rule_tasks=rule)
    assert result.fallback_required
    assert all(not item.get("side_effect", False) for item in result.tasks)
    record = next(item for item in result.tasks if item["task_type"] == "workout_record")
    assert record["write_rejection_reason"] == "semantic_intent_unavailable"


# ---------------------------------------------------------------------------
# Three read goals: bad evidence cannot fake completion
# ---------------------------------------------------------------------------

THREE_QUESTIONS = ("请分别回答三个问题：成年人力量训练后怎样安排睡眠恢复？"
                   "早餐用鸡蛋和豆制品怎样补蛋白质？新手热身时如何循序渐进？"
                   "三问都只讲通用原则，不保存任何训练、饮食或长期记忆。")
THREE_ANSWERS = ["睡眠恢复答案：规律睡眠并给同一肌群留出间隔。",
                 "早餐蛋白答案：鸡蛋加豆制品，分散到各餐。",
                 "热身答案：从低强度开始，逐步增加活动范围。"]


def _three_candidates():
    parts = [("sleep", "training_qa", "成年人力量训练后怎样安排睡眠恢复？"),
             ("protein", "nutrition_advice", "早餐用鸡蛋和豆制品怎样补蛋白质？"),
             ("warmup", "training_qa", "新手热身时如何循序渐进？")]
    result = []
    offset = 0
    for cid, kind, text_value in parts:
        start = THREE_QUESTIONS.index(text_value, offset)
        offset = start + len(text_value)
        result.append({"id": cid, "task_type": kind, "action": "question",
                       "text": text_value, "source_start": start,
                       "source_end": offset, "depends_on": []})
    return result


def test_bad_first_ref_cannot_mark_the_three_question_goal_complete():
    from app.services.agent.react_graph import (
        ReactDependencies, _read_goal_matches, completion_check,
    )
    candidates = _three_candidates()
    spec = SimpleNamespace(task_type="training_qa")
    bad_ref = {"start": 10, "end": 26,
               "exact_text": "成年人力量训练后怎样安排睡眠恢复？"}
    matched, problem = _read_goal_matches(
        SimpleNamespace(args={"source_ref": bad_ref}), spec,
        {"intent_candidates": candidates, "source_text": THREE_QUESTIONS})
    assert matched == [] and problem == "invalid_source_ref"

    state = {"intent_candidates": candidates, "covered_goals": ["protein", "warmup"],
             "goal_results": {"protein": {"status": "completed", "summary": THREE_ANSWERS[1]},
                              "warmup": {"status": "completed", "summary": THREE_ANSWERS[2]}},
             "supervisor_turns": 1, "tool_steps": 2, "consecutive_no_progress": 0}
    decision = completion_check(state, ReactDependencies(llm=object(), persistence=object()))
    assert decision.get("status") != "complete"
    assert decision.get("termination_reason") != "goals_complete"


def test_valid_refs_deliver_all_three_answers_once_in_source_order():
    from tests.react_phase_a_support import FakeLLM, RecordingExecutor, assistant, ref_of, tool_call
    from app.services.agent.react_graph import ReactDependencies, invoke_react

    candidates = _three_candidates()
    first_answer = ref_of(THREE_QUESTIONS, "成年人力量训练后怎样安排睡眠恢复？")
    bad_first = {"start": first_answer["start"], "end": first_answer["end"] - 1,
                 "exact_text": first_answer["exact_text"]}

    def result_for(_nth, name, args, _ctx):
        exact = str((args.get("source_ref") or {}).get("exact_text") or "")
        index = next(index for index, part in enumerate(
            ["成年人力量训练后怎样安排睡眠恢复？", "早餐用鸡蛋和豆制品怎样补蛋白质？",
             "新手热身时如何循序渐进？"]) if part == exact)
        return {"status": "completed", "summary": THREE_ANSWERS[index]}

    executor = RecordingExecutor({
        "answer_training": result_for, "advise_nutrition": result_for})
    script = [
        assistant("", tool_calls=[
            tool_call("bad-first", "answer_training",
                      {"query": "睡眠", "source_ref": bad_first}),
            tool_call("protein", "advise_nutrition",
                      {"query": "蛋白质",
                       "source_ref": ref_of(THREE_QUESTIONS, "早餐用鸡蛋和豆制品怎样补蛋白质？")}),
            tool_call("warmup", "answer_training",
                      {"query": "热身",
                       "source_ref": ref_of(THREE_QUESTIONS, "新手热身时如何循序渐进？")}),
        ]),
        assistant("", tool_calls=[
            tool_call("sleep", "answer_training",
                      {"query": "睡眠", "source_ref": first_answer}),
        ]),
    ]
    semantics.remember_safety(THREE_QUESTIONS, {
        "level": "normal", "evidence": "", "categories": [], "reason": "scripted normal"})
    dependencies = ReactDependencies(
        llm=FakeLLM(script=script, semantic=False), tool_executor=executor,
        user_id="semantic-migration", conversation_id="three-questions",
        request_id="three-questions-run", stop_after_batch=False)
    state = asyncio.run(invoke_react(
        dependencies, THREE_QUESTIONS, intent_candidates=candidates))

    assert state["status"] == "complete", state
    positions = [state["final_answer"].find(answer) for answer in THREE_ANSWERS]
    assert all(position >= 0 for position in positions), state["final_answer"]
    assert positions == sorted(positions)
    for answer in THREE_ANSWERS:
        assert state["final_answer"].count(answer) == 1
    assert executor.names.count("advise_nutrition") == 1
    assert executor.names.count("answer_training") == 2


# ---------------------------------------------------------------------------
# Real HTTP boundaries: urgent model judgment and strict replay
# ---------------------------------------------------------------------------


def _http_call(app, path, body):
    import httpx

    async def run():
        with deny_network() as audit:
            async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                    base_url="http://api.test") as client:
                reply = await client.post(path, json=body)
            assert audit["blocked_connections"] == audit["blocked_dns"] == 0
            return reply
    return asyncio.run(run())


@pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
def test_http_unknown_safety_blocks_arbitrary_wording_both_transports(compact_db, monkeypatch, path):
    from app.main import app
    from app.api import chat
    from test_retry_operations import RetryScenario

    source = "请记录今天的抬臂训练。"
    scenario = RetryScenario(monkeypatch, compact_db)
    raw = RequestUnderstanding.model_validate(_understanding(
        [_task("workout_record", source, action="record")], safety=None))
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", lambda: True)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured",
                        AsyncMock(return_value=raw))
    monkeypatch.setattr(chat, "agent", scenario.agent)
    reply = _http_call(app, path, {"message": source, "mode": "legacy",
                                   "request_id": uuid4().hex})
    assert reply.status_code == 200
    if path.endswith("/stream"):
        events = [json.loads(block.split("data: ", 1)[1])
                  for block in reply.text.strip().split("\n\n")
                  if block.startswith("event: ")]
        result = events[-1]
        answer = "".join(str(event.get("content") or "") for event in events)
    else:
        result = reply.json()
        answer = result["answer"]
    assert result["intent"] == "execution_limited"
    assert "安全语义判断" in answer
    assert not scenario.saves
    with compact_db() as session:
        for table in ("workout_records", "user_food_log", "nutrition_logs",
                      "training_plans", "user_facts"):
            assert session.execute(text("SELECT COUNT(*) FROM " + table)).scalar_one() == 0


@pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
def test_http_urgent_model_judgment_blocks_business_writes(compact_db, monkeypatch, path):
    from app.main import app
    from app.api import chat
    from test_retry_operations import RetryScenario

    source = "刚举起来就觉得胸前被压着，怎么都吸不够空气。"
    scenario = RetryScenario(monkeypatch, compact_db)
    raw = RequestUnderstanding.model_validate(_understanding(
        [_task("safety_triage", source)],
        safety=_safety("urgent", source, ["current_symptom"], "scripted current emergency")))
    model = AsyncMock(return_value=raw)
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", lambda: True)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", model)
    monkeypatch.setattr(chat, "agent", scenario.agent)
    reply = _http_call(app, path, {"message": source, "mode": "legacy",
                                   "request_id": uuid4().hex})
    assert reply.status_code == 200
    if path.endswith("/stream"):
        events = [json.loads(block.split("data: ", 1)[1])
                  for block in reply.text.strip().split("\n\n")
                  if block.startswith("event: ")]
        results = [event for event in events if "safety_level" in event]
    else:
        results = [reply.json()]
    assert results
    for result in results:
        assert result["safety_level"] == "urgent"
        assert result["intent"] == "symptom_triage"
        assert not (result.get("metrics") or {}).get("receipt_refs")
    assert model.await_count == 1
    assert not scenario.saves
    with compact_db() as session:
        for table in ("workout_records", "user_food_log", "nutrition_logs",
                      "training_plans", "user_facts"):
            assert session.execute(text("SELECT COUNT(*) FROM " + table)).scalar_one() == 0


def test_partial_commit_survives_cold_semantic_budget_stop(tmp_path, monkeypatch):
    from tests.react_phase_c1_support import isolated
    from tests.react_phase_c2_2_support import ensure_catalog
    from tests.react_phase_c2_3_support import C3HTTPHarness, business_keys
    from test_sim26_multi_read_delivery import (
        MIXED_SOURCE, _count_read_handlers, _install_provider, _install_rag,
        _mixed_program, _mixed_understanding, _phase_fault,
    )
    from app.services.llm.budget import ModelBudgetError

    with isolated(monkeypatch, tmp_path / "partial-commit", "sqlite") as engine:
        ensure_catalog(engine)
        _install_rag(monkeypatch)
        seen, supervisor_calls = [], []
        _install_provider(
            monkeypatch, understanding=_mixed_understanding(),
            supervisor=_mixed_program(), answers=["训练后恢复建议", "蛋白质分配建议"],
            seen=seen, supervisor_calls=supervisor_calls)
        harness = C3HTTPHarness(monkeypatch, engine)
        handler_counts = _count_read_handlers(monkeypatch)
        before = business_keys(engine)
        rid = "semantic-partial-" + uuid4().hex[:8]
        fault = _phase_fault(engine, rid, "final_reply")
        with fault:
            first_status, first = harness.post_json(MIXED_SOURCE, request_id=rid)
        assert fault.raised is True and first_status == 200
        after_first = business_keys(engine)
        assert len(after_first["nutrition_logs"]) == len(before["nutrition_logs"]) + 1
        assert after_first["user_food_log"] == before["user_food_log"]
        first_receipts = list(first.get("metrics", {}).get("receipt_refs") or [])
        assert first_receipts
        first_ids = (first["request_id"], first["message_id"], first["run_id"])

        # Simulate an old controller snapshot without the new safety metadata.
        with engine.begin() as connection:
            row = connection.execute(text(
                "SELECT parameters FROM agent_execution_tasks "
                "WHERE user_id='demo-user' AND request_id=:r AND task_type='react_controller'"),
                {"r": rid}).scalar_one()
            parameters = json.loads(row) if isinstance(row, str) else dict(row or {})
            parameters.pop("semantic_understanding", None)
            parameters.pop("task_outline", None)
            connection.execute(text(
                "UPDATE agent_execution_tasks SET parameters=:p "
                "WHERE user_id='demo-user' AND request_id=:r AND task_type='react_controller'"),
                {"p": json.dumps(parameters, ensure_ascii=False), "r": rid})

        from app.services.agent.agent import agent as app_agent
        async def budget_stop(*args, **kwargs):
            raise ModelBudgetError("model_budget_exhausted:request")
        monkeypatch.setattr(app_agent.graph.llm, "configured", lambda: True)
        monkeypatch.setattr(app_agent.graph.llm, "complete_structured", budget_stop)
        calls_before = len(seen)
        workers_before = handler_counts["total"]

        status, resumed = harness.post_json(MIXED_SOURCE, request_id=rid)
        assert status == 200
        assert resumed["intent"] == "execution_limited"
        assert resumed["metrics"]["receipt_refs"] == first_receipts
        assert "此前已提交" in resumed["answer"]
        assert (resumed["request_id"], resumed["message_id"], resumed["run_id"]) == first_ids
        assert len(seen) == calls_before
        assert handler_counts["total"] == workers_before
        assert business_keys(engine) == after_first


def test_json_sse_replay_is_strict_with_no_extra_provider_worker_or_rows(tmp_path, monkeypatch):
    from tests.react_phase_c1_support import isolated
    from tests.react_phase_c2_2_support import ensure_catalog
    from tests.react_phase_c2_3_support import C3HTTPHarness, business_keys
    from test_sim26_multi_read_delivery import (
        SAVED_TWO_READ_CASES, _count_read_handlers, _install_provider, _install_rag,
        _nine_table_counts, _saved_batch, _task, _understanding,
    )

    case = SAVED_TWO_READ_CASES[0]
    with isolated(monkeypatch, tmp_path / "semantic-replay", "sqlite") as engine:
        ensure_catalog(engine)
        rag_calls = _install_rag(monkeypatch)
        seen, supervisor_calls = [], []
        _install_provider(
            monkeypatch,
            understanding=_understanding(*[
                _task(kind, text_value) for kind, text_value in case["tasks"]]),
            supervisor=_saved_batch(case),
            answers=case["answers"], seen=seen, supervisor_calls=supervisor_calls)
        harness = C3HTTPHarness(monkeypatch, engine)
        handler_counts = _count_read_handlers(monkeypatch)
        before = business_keys(engine)
        expected = case["answers"][0] + "\n\n" + case["answers"][1]
        rid = "semantic-replay-" + uuid4().hex[:8]

        status, first = harness.post_json(case["source"], request_id=rid)
        assert status == 200
        assert first["answer"] == expected
        assert first["metrics"]["react_write"] == "completed"
        ids = (first["request_id"], first["message_id"], first["run_id"])
        assert ids[0] == rid and all(ids)
        assert len(seen) == 6 and len(supervisor_calls) == 1
        assert handler_counts["total"] == 2 and len(rag_calls) == 2
        assert business_keys(engine) == before
        nine_before = _nine_table_counts(engine)

        sse_status, events, _ = harness.stream(case["source"], request_id=rid)
        sse_answer = "".join(str(frame.get("content") or "") for frame in events)
        assert sse_status == 200 and events
        assert sse_answer == first["answer"]
        assert events[-1]["intent"] == first["intent"]
        assert (events[-1]["request_id"], events[-1]["message_id"],
                events[-1]["run_id"]) == ids
        assert len(seen) == 6 and len(supervisor_calls) == 1
        assert handler_counts["total"] == 2 and len(rag_calls) == 2
        assert business_keys(engine) == before
        assert _nine_table_counts(engine) == nine_before
