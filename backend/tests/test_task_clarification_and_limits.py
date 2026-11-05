"""Tests for task clarification and limits."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.agent.react_graph import ReactDependencies, finalize_node
from app.services.agent.react_write_runtime import WriteRuntimeSession
from app.services.agent.semantics import SemanticValidationError, validate_understanding
from app.services.agent.task_information import PlanInformation, check_plan_information
from app.services.agent.task_planner import plan_tasks_rule


PREFIX_SOURCE = (
    "增肌新手、无伤病、只有哑铃；每周两练，一次最多30分钟。"
    "先复述刚存的卧推与深蹲重量，然后给周计划。这次不保存任何东西。"
)
CANCEL_SOURCE = "刚才那份豆腐晚餐取消，不要把草稿写成已吃记录。"
TWO_PLAN_SOURCE = (
    "我是成年增肌新手、无伤病。请给我两份独立的今天训练计划："
    "一份哑铃力量计划（只有哑铃）；另一份弹力带全身计划（只有弹力带）。"
    "每份从热身到放松总共最多20分钟，休息转换也计入。两份都只给建议，不保存。"
)
TEN_MIN_SOURCE = (
    "我是成年减脂新手，无伤病，只有弹力带。今天从热身到放松最多10分钟，"
    "请排完整一节，休息也算在内。不要把10分钟全用来拉伸，放松一两分钟就够。"
    "这次只看方案，不保存。"
)


def _flags(**extra):
    return {
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        **extra,
    }


def _understanding(tasks, *, relation="new", safety="normal"):
    return {
        "tasks": [{**task, "history_requirement": task.get("history_requirement", "none")}
                  for task in tasks],
        "safety": {"level": safety, "evidence": "", "categories": [], "reason": "scripted"},
        "memory_requests": [], "references": [], "critical_history_ids": [],
        "dialogue_ids": [], "relation": relation, "reply_mode": "grounded",
        "reply_purpose": "general", "critical_fact_fields": [],
        "current_is_critical": False, "needs_clarification": False,
    }


def _recap_task():
    return {
        "task_type": "training_qa", "action": "answer",
        "text": "先复述刚存的卧推与深蹲重量", **_flags(),
        "fact_text": None, "reply_mode": "conversation", "reply_purpose": "general",
        "context_ids": [], "fact_fields": [],
    }


def _conditional_plan_task():
    return {
        "task_type": "training_plan", "action": "question",
        "text": "增肌新手、无伤病、只有哑铃；每周两练，一次最多30分钟。",
        **_flags(plan_result=False, result_scope=None, plan_goal_id="",
                 plan_request_quote="", shared_scope_quote=""),
        "fact_text": None, "reply_mode": "grounded", "reply_purpose": "general",
        "context_ids": [], "fact_fields": [],
    }


def _real_plan_task():
    return {
        "task_type": "training_plan", "action": "question",
        "text": "然后给周计划。这次不保存任何东西。",
        **_flags(plan_result=True, result_scope="independent", plan_goal_id="",
                 plan_request_quote="然后给周计划", shared_scope_quote=""),
        "fact_text": None, "reply_mode": "grounded", "reply_purpose": "general",
        "context_ids": [], "fact_fields": [],
    }


def test_condition_only_plan_span_is_absorbed_and_does_not_sink_the_request():
    raw = _understanding([_recap_task(), _conditional_plan_task(), _real_plan_task()])

    value = validate_understanding(PREFIX_SOURCE, raw, absorb_background_plans=True)

    planned = value["validated_tasks"]
    assert [task["task_type"] for task in planned] == ["training_qa", "training_plan"]
    # The background span is merged into the adjacent recap span, preserving all
    # source text and never dispatching a second plan worker.
    assert planned[0]["text"] == (
        "增肌新手、无伤病、只有哑铃；每周两练，一次最多30分钟。先复述刚存的卧推与深蹲重量"
    )
    assert planned[-1]["text"] == "然后给周计划。这次不保存任何东西。"

    # The legacy rule entry point reuses the same validated tasks.
    from app.services.agent.semantics import remember_understanding
    remember_understanding(PREFIX_SOURCE, value)
    reused = plan_tasks_rule(PREFIX_SOURCE)
    assert [task["task_type"] for task in reused] == ["training_qa", "training_plan"]


def test_default_strict_validation_still_rejects_the_condition_split():
    raw = _understanding([_recap_task(), _conditional_plan_task(), _real_plan_task()])

    with pytest.raises(SemanticValidationError) as caught:
        validate_understanding(PREFIX_SOURCE, raw)
    assert caught.value.code == "semantic_condition_split_as_plan"


class _ScriptedUnderstanding:
    configured = lambda self: True

    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    async def complete_structured(self, messages, schema, *, name=None, temperature=0,
                                  validator=None, **kwargs):
        self.calls += 1
        value = schema.model_validate(self.raw)
        if validator is not None:
            validator(value)
        return value


def test_production_understanding_absorbs_saved_split_without_a_second_call():
    from app.services.agent import semantics

    raw = _understanding([_recap_task(), _conditional_plan_task(), _real_plan_task()])
    client = _ScriptedUnderstanding(raw)
    token = semantics.request_understanding.set({})
    try:
        value = asyncio.run(semantics.understand(client, PREFIX_SOURCE, context={}))
    finally:
        semantics.request_understanding.reset(token)

    assert value is not None
    assert client.calls == 1
    assert [task["task_type"] for task in value["validated_tasks"]] == ["training_qa", "training_plan"]


def test_production_understanding_accepts_saved_cancel_for_draft_continuation():
    from app.services.agent import semantics

    raw = _understanding([{
        "task_type": "nutrition_record", "action": "confirm",
        "text": CANCEL_SOURCE, **_flags(),
        "fact_text": None, "reply_mode": "conversation",
        "reply_purpose": "execution_status", "context_ids": [], "fact_fields": [],
    }], relation="cancel")
    client = _ScriptedUnderstanding(raw)
    token = semantics.request_understanding.set({})
    try:
        value = asyncio.run(semantics.understand(client, CANCEL_SOURCE, context={}))
    finally:
        semantics.request_understanding.reset(token)

    assert value is not None
    assert client.calls == 1
    assert value["relation"] == "cancel"
    assert value["tasks"][0]["action"] == "answer"


def test_background_only_plan_is_rejected_instead_of_silently_answering():
    raw = _understanding([{
        "task_type": "training_plan", "action": "question",
        "text": "只有哑铃，每周两练。", **_flags(
            plan_result=False, result_scope=None, plan_goal_id="",
            plan_request_quote="", shared_scope_quote=""),
        "fact_text": None, "reply_mode": "grounded", "reply_purpose": "general",
        "context_ids": [], "fact_fields": [],
    }])

    with pytest.raises(SemanticValidationError) as caught:
        validate_understanding("只有哑铃，每周两练。", raw, absorb_background_plans=True)
    assert caught.value.code == "semantic_background_only"


def test_cancel_draft_relation_normalizes_conflicting_confirm_action():
    raw = _understanding([{
        "task_type": "nutrition_record", "action": "confirm",
        "text": CANCEL_SOURCE, **_flags(),
        "fact_text": None, "reply_mode": "conversation",
        "reply_purpose": "execution_status", "context_ids": [], "fact_fields": [],
    }], relation="cancel")

    value = validate_understanding(CANCEL_SOURCE, raw)

    task = value["tasks"][0]
    assert value["relation"] == "cancel"
    assert task["action"] == "answer"
    assert task["reply_mode"] == "conversation"
    assert all(not planned["side_effect"] for planned in value["validated_tasks"])


def test_a_real_confirmation_is_not_normalized_away():
    raw = _understanding([{
        "task_type": "nutrition_record", "action": "confirm",
        "text": "确认保存刚才的豆腐", **_flags(),
        "fact_text": None, "reply_mode": "grounded", "reply_purpose": "general",
        "context_ids": [], "fact_fields": [],
    }], relation="confirmation")
    raw["safety"] = {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}

    value = validate_understanding("确认保存刚才的豆腐", raw)

    assert value["tasks"][0]["action"] == "confirm"


def _plan_information(raw):
    payload = {key: value for key, value in raw.items() if key != "_source"}
    llm = Mock(complete_structured=AsyncMock(return_value=PlanInformation.model_validate(payload)))
    return asyncio.run(check_plan_information(llm, {"raw_message": raw["_source"], "task_text": raw["_source"]}))


def _facts(source, seconds):
    return [
        {"field": "goal", "source_id": "current", "quote": "成年增肌新手" if "增肌" in source else "成年减脂新手"},
        {"field": "experience", "source_id": "current", "quote": "成年增肌新手" if "增肌" in source else "成年减脂新手"},
        {"field": "equipment", "source_id": "current", "quote": "只有弹力带"},
        {"field": "session_time", "source_id": "current",
         "quote": "每份从热身到放松总共最多20分钟，休息转换也计入。" if "两份" in source
         else "今天从热身到放松最多10分钟，请排完整一节，休息也算在内。",
         "quantity_quote": "20分钟" if "两份" in source else "10分钟", "seconds": seconds},
    ]


def test_total_limit_misread_as_qualified_phase_is_soft_not_a_service_failure():
    raw = {
        "_source": TWO_PLAN_SOURCE,
        "scope": "session",
        "facts": _facts(TWO_PLAN_SOURCE, 1200),
        "phase_limits": [{
            "phase": "main", "relation": "at_most", "seconds": 1200,
            "source_id": "current",
            "context_quote": "每份从热身到放松总共最多20分钟，休息转换也计入。",
            "quantity_quote": "20分钟", "qualifiers": ["incomplete"],
        }],
    }
    details, clarification = _plan_information(raw)

    assert clarification is None
    assert details["phase_limits"] == []
    assert details["session_time_seconds"] == 1200


def test_range_phase_wording_keeps_the_ten_minute_plan_deliverable():
    raw = {
        "_source": TEN_MIN_SOURCE,
        "scope": "session",
        "facts": _facts(TEN_MIN_SOURCE, 600),
        "phase_limits": [{
            "phase": "cooldown", "relation": "at_most", "seconds": 120,
            "source_id": "current",
            "context_quote": "不要把10分钟全用来拉伸，放松一两分钟就够。",
            "quantity_quote": "一两分钟", "qualifiers": ["range"],
        }],
    }
    details, clarification = _plan_information(raw)

    assert clarification is None
    assert details["phase_limits"] == []
    assert details["session_time_seconds"] == 600


def test_ordinary_phase_budget_is_still_hard():
    source = "我是成年新手，只有弹力带，今天总共20分钟，主体最多10分钟，只给建议不保存。"
    raw = {
        "_source": source,
        "scope": "session",
        "facts": [
            {"field": "goal", "source_id": "current", "quote": "成年新手"},
            {"field": "experience", "source_id": "current", "quote": "成年新手"},
            {"field": "equipment", "source_id": "current", "quote": "只有弹力带"},
            {"field": "session_time", "source_id": "current", "quote": "今天总共20分钟",
             "quantity_quote": "20分钟", "seconds": 1200},
        ],
        "phase_limits": [{
            "phase": "main", "relation": "at_most", "seconds": 600,
            "source_id": "current", "context_quote": "主体最多10分钟",
            "quantity_quote": "10分钟",
        }],
    }
    details, clarification = _plan_information(raw)

    assert clarification is None
    assert details["phase_limits"] == [{
        "phase": "main", "relation": "at_most", "seconds": 600,
        "quote": "主体最多10分钟", "quantity_quote": "10分钟",
    }]


def test_phase_context_ending_at_sentence_punctuation_is_complete():
    source = "成年增肌新手，无伤病，只有哑铃，总共20分钟。热身最多3分钟。请给完整安排，不保存。"
    raw = {
        "_source": source,
        "scope": "session",
        "facts": [
            {"field": "goal", "source_id": "current", "quote": "成年增肌新手"},
            {"field": "experience", "source_id": "current", "quote": "成年增肌新手"},
            {"field": "equipment", "source_id": "current", "quote": "只有哑铃"},
            {"field": "session_time", "source_id": "current", "quote": "总共20分钟",
             "quantity_quote": "20分钟", "seconds": 1200},
        ],
        "phase_limits": [{
            "phase": "warmup", "relation": "at_most", "seconds": 180,
            "source_id": "current", "context_quote": "热身最多3分钟。",
            "quantity_quote": "3分钟",
        }],
    }
    details, clarification = _plan_information(raw)

    assert clarification is None
    assert details["phase_limits"][0]["relation"] == "at_most"


def test_phase_context_fragment_still_rejects_mid_sentence_cut():
    source = "成年增肌新手，只有哑铃，总共20分钟，热身最多3分钟然后再做主体，只给建议不保存。"
    raw = {
        "_source": source,
        "scope": "session",
        "facts": [
            {"field": "goal", "source_id": "current", "quote": "成年增肌新手"},
            {"field": "experience", "source_id": "current", "quote": "成年增肌新手"},
            {"field": "equipment", "source_id": "current", "quote": "只有哑铃"},
            {"field": "session_time", "source_id": "current", "quote": "总共20分钟",
             "quantity_quote": "20分钟", "seconds": 1200},
        ],
        "phase_limits": [{
            "phase": "warmup", "relation": "at_most", "seconds": 180,
            "source_id": "current", "context_quote": "热身最多3分钟",
            "quantity_quote": "3分钟",
        }],
    }
    _, clarification = _plan_information(raw)

    assert clarification is not None
    assert clarification.category == "service_unavailable"


def _limited_state():
    return {
        "intent_candidates": [
            {"id": "g1", "task_type": "training_qa", "action": "question", "text": "睡眠恢复"},
            {"id": "g2", "task_type": "nutrition_advice", "action": "question", "text": "早餐蛋白"},
            {"id": "g3", "task_type": "training_qa", "action": "question", "text": "热身安排"},
        ],
        "goal_results": {
            "g1": {"status": "completed", "summary": "睡眠恢复答案。"},
            "g2": {"status": "completed", "summary": "早餐蛋白答案。"},
        },
        "covered_goals": ["g1", "g2"],
        "observations": [], "messages": [],
        "termination_reason": "tool_step_limit", "safety_level": "normal",
        "candidate_error": "", "unbound_write_calls": [], "call_facts": [],
    }


def _runtime_without_io():
    runtime = WriteRuntimeSession.__new__(WriteRuntimeSession)
    runtime.termination_reason = "tool_step_limit"
    runtime.receipts = []
    runtime.committed_observations = []
    runtime.resume_goal_results = {}
    runtime.resume_covered = []
    runtime.frozen_parameters = {}
    return runtime


def test_read_limited_final_answer_keeps_both_completed_answers():
    final = finalize_node(_limited_state(), ReactDependencies(llm=object()))

    assert final["status"] == "limited"
    assert final["final_answer"].startswith("睡眠恢复答案。\n\n早餐蛋白答案。\n\n")
    assert "工具执行步数上限" in final["final_answer"]
    assert final["final_answer"].count("睡眠恢复答案。") == 1
    assert final["final_answer"].count("早餐蛋白答案。") == 1


def test_write_limited_final_answer_keeps_both_completed_answers():
    answer, receipts = _runtime_without_io()._limited_answer(_limited_state())

    assert receipts == []
    assert answer.startswith("睡眠恢复答案。\n\n早餐蛋白答案。\n\n")
    assert "部分目标未完成" in answer
    assert "本次没有已确认保存的记录" in answer


def _tool_message(arguments, name="return_plan_information"):
    return {"tool_calls": [{"type": "function", "function": {
        "name": name, "arguments": arguments}}]}


def test_structured_output_repairs_only_redundant_trailing_delimiters():
    from app.services.agent.task_information import PlanInformation
    from app.services.llm.structured import StructuredOutputError, parse_tool_result

    good = '{"scope":"session","facts":[],"phase_limits":[]}'
    parsed = parse_tool_result(_tool_message(good), PlanInformation, "return_plan_information")
    assert parsed.scope == "session"

    repaired = parse_tool_result(_tool_message(good + "]}"), PlanInformation,
                                 "return_plan_information")
    assert repaired.scope == "session"

    for malformed in ('{"scope"::"session","facts":[],"phase_limits":[]}',
                      '{"scope":"session"',
                      good + ' extra'):
        with pytest.raises(StructuredOutputError) as caught:
            parse_tool_result(_tool_message(malformed), PlanInformation,
                              "return_plan_information")
        assert caught.value.code in {"structured_invalid_json", "structured_invalid_arguments"}


def test_planner_failure_guard_only_stops_understood_writes():
    from app.services.agent.execution import _planner_unavailable_for_write

    write = {"tasks": [{"task_type": "workout_record", "action": "record"}]}
    read = {"tasks": [{"task_type": "training_qa", "action": "question"}]}

    assert _planner_unavailable_for_write("planner_llm_unavailable_or_invalid", write) is True
    assert _planner_unavailable_for_write("planner_task_mapping_failed", write) is True
    assert _planner_unavailable_for_write("planner_llm_unavailable_or_invalid", read) is False
    assert _planner_unavailable_for_write(None, write) is False


def test_understood_write_with_parser_failure_gets_service_unavailable_copy():
    from unittest.mock import AsyncMock

    from app.services.agent import execution, semantics
    from app.services.agent.agent import FitnessAgent
    from app.services.agent.execution_state import PlannerDecision
    from app.services.agent.intent import IntentResult
    from app.schemas.chat import ChatRequest

    source = "深蹲也记上：72kg，4×6，都是今天练的。"
    understanding = {
        "tasks": [{
            "task_type": "workout_record", "action": "record", "text": source,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        }],
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"},
    }
    safety = {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}
    agent = FitnessAgent()
    agent._route = AsyncMock(return_value=PlannerDecision(
        IntentResult("workout_record", 1.0), {}, (),
        degraded_reason="planner_llm_unavailable_or_invalid"))
    token = semantics.request_understanding.set({})
    try:
        semantics.remember_understanding(source, understanding)
        semantics.remember_safety(source, safety)
        response = asyncio.run(agent._respond_impl(ChatRequest(message=source)))
    finally:
        semantics.request_understanding.reset(token)

    assert response.intent == "execution_limited"
    assert "AI记录解析暂时不可用，本次未保存" in response.answer
    assert response.metrics["mode_reason"] == "planner_unavailable_for_write"


def test_json_syntax_error_gets_a_syntax_specific_correction():
    from app.services.agent.task_information import PlanInformation
    from app.services.llm.structured import complete_structured

    class Client:
        def __init__(self):
            self.calls = []

        async def complete_message(self, messages, temperature=0, tools=None, tool_choice=None):
            self.calls.append(messages)
            arguments = ('{"scope"::"session","facts":[],"phase_limits":[]}'
                         if len(self.calls) == 1
                         else '{"scope":"session","facts":[],"phase_limits":[]}')
            return {"tool_calls": [{"type": "function", "function": {
                "name": "return_plan_information", "arguments": arguments}}]}

    client = Client()
    value = asyncio.run(complete_structured(
        client, [{"role": "system", "content": "sys"}], PlanInformation,
        name="return_plan_information"))

    assert value.scope == "session"
    assert len(client.calls) == 2
    assert "JSON语法无效" in client.calls[1][0]["content"]


def test_per_task_step_limit_isolates_the_exhausted_task():
    from tests.react_supervisor_fixtures import (
        FakeLLM, RecordingExecutor, assistant, batch_state, ref_of, tool_call,
    )
    from app.services.agent.react_graph import ReactDependencies, execute_react_batch

    source = "第一问怎么安排？第二问怎么安排？"
    candidates = [
        {"id": "t1", "task_type": "nutrition_advice", "action": "question",
         "text": "第一问怎么安排？", "depends_on": []},
        {"id": "t2", "task_type": "nutrition_advice", "action": "question",
         "text": "第二问怎么安排？", "depends_on": []},
    ]
    calls = [
        tool_call("c1", "advise_nutrition", {
            "query": "a", "source_ref": ref_of(source, "第一问怎么安排？")}),
        tool_call("c2", "advise_nutrition", {
            "query": "b", "source_ref": ref_of(source, "第一问怎么安排？")}),
        tool_call("c3", "advise_nutrition", {
            "query": "c", "source_ref": ref_of(source, "第二问怎么安排？")}),
    ]
    executor = RecordingExecutor({"advise_nutrition": {
        "status": "completed", "summary": "答案"}})
    state = batch_state(source, assistant("", tool_calls=calls), candidates)
    deps = ReactDependencies(llm=FakeLLM(script=[], semantic=False),
                             tool_executor=executor, max_tool_steps_per_task=1)

    result = asyncio.run(execute_react_batch(state, deps))

    assert executor.names == ["advise_nutrition", "advise_nutrition"]
    assert result["tool_steps_by_task"] == {"t1": 1, "t2": 1}
    assert "t1" in result["limited_tasks"]
    assert any(
        "task_step_limit" in str(message.get("content") or "")
        for message in result["messages"] if message.get("role") == "tool"
    )


def test_task_count_cap_stops_before_dispatch():
    from app.services.agent import semantics
    from app.services.agent.react_graph import ReactDependencies, prepare_node

    source = "第一问怎么安排？第二问怎么安排？第三问怎么安排？"
    candidates = [
        {"id": f"t{index}", "task_type": "nutrition_advice", "action": "question",
         "text": text, "depends_on": []}
        for index, text in enumerate(
            ["第一问怎么安排？", "第二问怎么安排？", "第三问怎么安排？"], start=1)
    ]
    token = semantics.request_understanding.set({})
    try:
        semantics.remember_safety(source, {
            "level": "normal", "evidence": "", "categories": [], "reason": "scripted"})
        out = prepare_node(
            {"raw_message": source, "intent_candidates": candidates},
            ReactDependencies(llm=object(), max_tasks=2))
    finally:
        semantics.request_understanding.reset(token)

    assert out["terminated"] is True
    assert out["termination_reason"] == "task_count_limit"
    assert len(out["intent_candidates"]) == 2


def test_legacy_task_count_cap_stops_before_route_execution():
    from unittest.mock import AsyncMock

    from app.services.agent import semantics
    from app.services.agent.agent import FitnessAgent
    from app.services.agent.execution_state import PlannerDecision
    from app.services.agent.intent import IntentResult
    from app.schemas.chat import ChatRequest

    source = "记录七条训练"
    understanding = {"tasks": [{"task_type": "training_qa", "action": "question", "text": source}],
                     "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}}
    specs = tuple({"id": f"t{index}", "task_type": "training_qa", "depends_on": []}
                  for index in range(7))
    agent = FitnessAgent()
    agent._route = AsyncMock(return_value=PlannerDecision(
        IntentResult("training_qa", 1.0), {}, specs))
    token = semantics.request_understanding.set({})
    try:
        semantics.remember_understanding(source, understanding)
        semantics.remember_safety(source, understanding["safety"])
        response = asyncio.run(agent._respond_impl(ChatRequest(message=source)))
    finally:
        semantics.request_understanding.reset(token)

    assert response.intent == "execution_limited"
    assert "最多 6 个" in response.answer
    assert response.metrics["mode_reason"] == "task_count_limit"


def test_saved_two_plan_json_with_extra_bracket_parses_and_validates():
    import json

    from app.services.agent.semantics import RequestUnderstanding, validate_understanding
    from app.services.llm.structured import parse_tool_result

    leader = {
        "task_type": "training_plan", "action": "question",
        "text": "我是成年增肌新手、无伤病。请给我两份独立的今天训练计划：一份哑铃力量计划（只有哑铃）",
        **_flags(plan_result=True, result_scope="shared_leader", plan_goal_id="today_two_plans",
                 shared_scope_quote="请给我两份独立的今天训练计划"),
        "fact_text": None, "reply_mode": "grounded", "reply_purpose": "general",
        "context_ids": [], "fact_fields": [],
    }
    member = {
        "task_type": "training_plan", "action": "question",
        "text": "另一份弹力带全身计划（只有弹力带）。每份从热身到放松总共最多20分钟，休息转换也计入。两份都只给建议，不保存。",
        **_flags(plan_result=True, result_scope="shared_member", plan_goal_id="today_two_plans"),
        "fact_text": None, "reply_mode": "grounded", "reply_purpose": "general",
        "context_ids": [], "fact_fields": [],
    }
    raw = json.dumps(_understanding([leader, member]), ensure_ascii=False) + "]}"
    message = _tool_message(raw, "return_request_understanding")

    parsed = parse_tool_result(message, RequestUnderstanding, "return_request_understanding")
    value = validate_understanding(TWO_PLAN_SOURCE, parsed, absorb_background_plans=True)

    assert [task["task_type"] for task in value["validated_tasks"]] == ["training_plan", "training_plan"]
