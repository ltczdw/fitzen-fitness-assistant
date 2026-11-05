"""SIM28-POST-01..03 common-cause regressions.

All provider/model boundaries are mocked. SQLite is the only database used by
these tests; no real model, embedding, rerank, personal data or service call.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from app.schemas.chat import ChatRequest
from app.services.agent import dialogue_context as dialogue
from app.services.agent import semantics
from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
from app.services.agent.semantics import (
    RequestUnderstanding,
    SemanticValidationError,
    remember_understanding,
    request_understanding,
    validate_understanding,
)
from app.services.agent.task_information import (
    PlanInformation,
    check_plan_information,
)
from app.services.agent.task_planner import plan_tasks_from_llm, plan_tasks_rule


# ---------------------------------------------------------------------------
# Shared fixtures / source text
# ---------------------------------------------------------------------------

NEGATIVE_SOURCE = (
    "我只是假设晚餐吃170克牛肉，用这个例子讨论一下。"
    "不要保存；这样的假设会被你记为我的真实饮食吗？"
)
NEGATIVE_ADVICE = "我只是假设晚餐吃170克牛肉，用这个例子讨论一下。"
NEGATIVE_STATUS = "不要保存；这样的假设会被你记为我的真实饮食吗？"

QUOTED_SOURCE = (
    "这是同事发来的训练：卧推85公斤，5组，每组5次。"
    "他的训练不是我的，请不要登记到我名下。"
)

TWO_PLAN_SOURCE = (
    "我成年，刚开始增肌，没有伤病。要两份互相独立的今天训练安排："
    "第一份器械只能用哑铃；第二份只能用弹力带。"
    "每份各自从热身、主体、组间休息转场到放松合计不超过20分钟。"
    "请把两份分别写完整，都不保存。"
)
PLAN_LEADER = "我成年，刚开始增肌，没有伤病。要两份互相独立的今天训练安排：第一份器械只能用哑铃"
PLAN_MEMBER = "第二份只能用弹力带。每份各自从热身、主体、组间休息转场到放松合计不超过20分钟。请把两份分别写完整，都不保存。"
PLAN_CONTRACT = {
    "plan_result": True,
    "result_scope": "shared_leader",
    "plan_goal_id": "today_two_plans",
    "plan_request_quote": "要两份互相独立的今天训练安排",
    "shared_scope_quote": "要两份互相独立的今天训练安排",
}
PLAN_CONTRACT_MEMBER = {**PLAN_CONTRACT, "result_scope": "shared_member"}


def _normal_safety():
    return {"level": "normal", "evidence": "", "categories": [], "reason": "scripted normal"}


def _task(kind, text, *, action="question", **extra):
    return {
        "task_type": kind,
        "action": action,
        "text": text,
        "is_negated": False,
        "is_hypothetical": False,
        "is_third_party": False,
        **extra,
    }


def _understanding(tasks, *, purpose="general", **extra):
    return {
        "tasks": tasks,
        "safety": _normal_safety(),
        "reply_mode": "grounded",
        "reply_purpose": purpose,
        "critical_fact_fields": [],
        "critical_history_ids": [],
        "dialogue_ids": [],
        "current_is_critical": False,
        "needs_clarification": False,
        "memory_requests": [],
        "references": [],
        **extra,
    }


def _remember(source, value):
    token = request_understanding.set({})
    remember_understanding(source, value)
    return token


# ---------------------------------------------------------------------------
# SIM28-POST-01: answer context is selected per target
# ---------------------------------------------------------------------------

PRIVATE = "PEACH_ALLERGY_PRIVATE"
COUNT = "EXISTING_RECORD_COUNT_PRIVATE"


def _memory_with_unrelated_personal_context():
    return {
        "profile": {"goal": "增肌", "note": PRIVATE},
        "long_memory": {
            "allergy": {
                "text": "我对桃子过敏",
                "source_quote": "我对桃子过敏",
                "source_message_id": "u-private",
            },
            "equipment": {
                "text": "我只有哑铃",
                "source_quote": "我只有哑铃",
                "source_message_id": "u-private-2",
            },
        },
        "short_memory": [
            {"message_id": "u-private", "role": "user", "content": "我对桃子过敏"},
            {"message_id": "u-private-2", "role": "user", "content": "我只有哑铃"},
        ],
        "retrieved_history": [{"turn_id": "old", "content": PRIVATE}],
        "semantic_memory": [{"text": PRIVATE, "category": "constraint", "speaker": "user"}],
        "memory_summary": PRIVATE,
        "training_history": [{"exercise": "卧推", "count": 2}],
        "nutrition_summary": {"entries": 2, "note": COUNT},
    }


def test_grounded_advice_receives_only_its_selected_personal_sources():
    graph = FitnessMultiAgentGraph()
    worker = SimpleNamespace(ainvoke=AsyncMock(return_value={
        "result": {"status": "complete", "answer": "示例建议", "model_used": False}}))
    graph.workers["nutrition"] = worker
    task = {
        "id": "t-advice",
        "task_type": "nutrition_advice",
        "action": "question",
        "text": NEGATIVE_ADVICE,
        "payload": {"semantic": {
            "context_ids": [], "fact_fields": [], "critical_fact_fields": [],
            "critical_history_ids": [],
            "is_hypothetical": True, "is_third_party": False, "quoted": False,
        }},
    }
    state = {
        "user_id": "synthetic", "raw_message": NEGATIVE_SOURCE,
        "current_task": task, "request_id": "sim28-post-context",
        **_memory_with_unrelated_personal_context(),
    }
    asyncio.run(graph._run_worker("nutrition", state))
    sent = worker.ainvoke.await_args.args[0]
    assert sent["long_memory"] == {}
    assert sent["profile"] == {}
    assert sent["short_memory"] == []
    assert sent["retrieved_history"] == []
    assert sent["semantic_memory"] == []
    assert sent["memory_summary"] == ""
    assert sent["training_history"] == []
    assert sent["nutrition_summary"] == {}

    messages = graph._planning_prompt("饮食营养方案", sent, [])
    wire = json.dumps(messages, ensure_ascii=False)
    assert PRIVATE not in wire
    assert COUNT not in wire
    assert "桃子" not in wire


def test_selected_critical_allergy_is_kept_for_a_relevant_advice_target():
    graph = FitnessMultiAgentGraph()
    worker = SimpleNamespace(ainvoke=AsyncMock(return_value={
        "result": {"status": "complete", "answer": "无奶建议", "model_used": False}}))
    graph.workers["nutrition"] = worker
    source = "我是成年人，对牛奶过敏，做饭只有微波炉。给一个不含奶的晚餐建议，不保存。"
    memory = {
        "profile": {"goal": "增肌", "note": PRIVATE},
        "long_memory": {
            "allergy": {"text": "我对牛奶过敏", "source_quote": "对牛奶过敏",
                        "source_message_id": "u-milk"},
            "equipment": {"text": "我只有哑铃", "source_quote": "我只有哑铃",
                          "source_message_id": "u-equipment"},
        },
        "short_memory": [{"message_id": "u-milk", "role": "user", "content": "对牛奶过敏"}],
        "nutrition_summary": {"entries": 2}, "training_history": [{"x": 1}],
    }
    task = {
        "id": "t-milk", "task_type": "nutrition_advice", "action": "question",
        "text": source,
        "payload": {"semantic": {
            "context_ids": [], "fact_fields": [],
            "critical_fact_fields": ["allergy"],
            "critical_history_ids": ["u-milk"],
            "is_hypothetical": False, "is_third_party": False, "quoted": False,
        }},
    }
    state = {"user_id": "synthetic", "raw_message": source, "current_task": task,
             "request_id": "sim28-post-milk", **memory}
    asyncio.run(graph._run_worker("nutrition", state))
    sent = worker.ainvoke.await_args.args[0]
    assert set(sent["long_memory"]) == {"allergy"}
    assert sent["long_memory"]["allergy"]["source_quote"] == "对牛奶过敏"
    assert sent["profile"] == {}
    assert sent["nutrition_summary"] == {}
    assert sent["training_history"] == []
    messages = graph._planning_prompt("饮食营养方案", sent, [])
    wire = json.dumps(messages, ensure_ascii=False)
    assert "牛奶" in wire
    assert PRIVATE not in wire


def test_status_answer_is_application_copy_without_personal_counts():
    memory = _memory_with_unrelated_personal_context()
    seen = []

    async def structured(messages, schema, **kwargs):
        seen.append(messages)
        return schema.model_validate({
            "answer": "你记录里已经有两条，而且你对桃子过敏。",
            "requires_grounding": False,
            "status_kind": "save_permission",
        })

    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(side_effect=structured))
    result = asyncio.run(dialogue.conversation_answer(
        llm, ChatRequest(message=NEGATIVE_STATUS), memory,
        {"reply_mode": "conversation", "reply_purpose": "execution_status",
         "goal_role": "result", "quoted": False,
         "dialogue_ids": [], "fact_fields": [], "critical_fact_fields": []}))
    assert result["answer"] == "不会保存。"
    wire = json.dumps(seen, ensure_ascii=False)
    assert PRIVATE not in wire and COUNT not in wire and "桃子" not in wire


def test_mixed_advice_and_status_use_separate_scopes_and_both_answers_survive():
    graph = FitnessMultiAgentGraph()
    captured = {}

    async def advice_worker(state):
        captured["advice"] = state
        return {"result": {"status": "complete", "answer": "牛肉例子建议", "model_used": False}}

    graph.workers["nutrition"] = SimpleNamespace(ainvoke=advice_worker)
    seen = []

    async def structured(messages, schema, **kwargs):
        seen.append(messages)
        return schema.model_validate({
            "answer": "你会保存吗？", "requires_grounding": False,
            "status_kind": "save_permission"})

    graph.llm.complete_structured = AsyncMock(side_effect=structured)
    graph.llm.configured = lambda: True
    status_task = {
        "id": "t-status", "task_type": "chitchat", "action": "question",
        "text": NEGATIVE_STATUS, "payload": {"dialogue": {
            "reply_mode": "conversation", "reply_purpose": "execution_status",
            "goal_role": "result", "quoted": False,
            "dialogue_ids": [], "fact_fields": [], "critical_fact_fields": [],
        }},
    }
    base = {"user_id": "synthetic", "conversation_id": "sim28-post-mixed",
            "raw_message": NEGATIVE_SOURCE,
            "request_id": "sim28-post-mixed", **_memory_with_unrelated_personal_context()}
    first = asyncio.run(graph._run_worker("nutrition", {
        **base, "current_task": {**_task("nutrition_advice", NEGATIVE_ADVICE),
                                 "id": "t-advice", "payload": {"semantic": {
                                     "fact_fields": [], "critical_fact_fields": [],
                                     "critical_history_ids": [], "quoted": False,
                                     "is_hypothetical": True}}}}))
    second = asyncio.run(graph._run_worker("training_qa", {**base, "current_task": status_task}))
    assert captured["advice"]["long_memory"] == {}
    assert first["worker_results"][-1]["answer"] == "牛肉例子建议"
    assert second["worker_results"][-1]["answer"] == "不会保存。"
    wire = json.dumps(seen, ensure_ascii=False)
    assert PRIVATE not in wire and COUNT not in wire and "桃子" not in wire
    combined = FitnessMultiAgentGraph._finalize_node({"worker_results": [
        first["worker_results"][-1], second["worker_results"][-1]]})
    assert "牛肉例子建议" in combined["final_answer"]
    assert "不会保存" in combined["final_answer"]


# ---------------------------------------------------------------------------
# SIM28-POST-02: quoted third-party read is not a write contract failure
# ---------------------------------------------------------------------------

def _quoted_payload(*, purpose="general", action="answer", kind="workout_record"):
    return _understanding([_task(
        kind, QUOTED_SOURCE, action=action, is_negated=True, is_third_party=True,
        goal_role="result", quoted=True, reply_mode="conversation",
        reply_purpose=purpose, context_ids=[], fact_fields=[])], purpose=purpose)


def test_saved_third_person_quoted_payload_is_accepted_as_read_only():
    value = validate_understanding(QUOTED_SOURCE, _quoted_payload())
    assert value["reply_purpose"] == "general"
    validated = value["validated_tasks"]
    assert len(validated) == 1
    assert validated[0]["task_type"] == "workout_record"
    assert validated[0]["action"] in {"answer", "question"}
    assert validated[0]["quoted"] is True
    assert validated[0]["is_third_party"] is True
    assert not validated[0]["side_effect"]


def test_saved_third_person_payload_with_status_purpose_is_normalized_not_rejected():
    value = validate_understanding(QUOTED_SOURCE, _quoted_payload(purpose="execution_status"))
    assert value["reply_purpose"] == "general"
    assert dialogue.status_only_request(value) is False
    assert value["validated_tasks"][0]["task_type"] == "workout_record"


@pytest.mark.parametrize("kind", ["workout_record", "nutrition_record"])
@pytest.mark.parametrize("flag", ["is_negated", "is_hypothetical", "is_third_party"])
def test_quoted_read_of_any_record_subject_stays_read_only(kind, flag):
    task = _task(kind, QUOTED_SOURCE, action="answer", goal_role="result",
                 quoted=True, reply_mode="conversation", reply_purpose="general",
                 context_ids=[], fact_fields=[])
    task[flag] = True
    value = validate_understanding(QUOTED_SOURCE, _understanding([task]))
    assert value["validated_tasks"][0]["task_type"] == kind
    assert value["validated_tasks"][0]["action"] in {"answer", "question"}
    assert not value["validated_tasks"][0]["side_effect"]


def test_quoted_save_command_is_transcribed_without_write_authorization():
    source = "请转述这句话：‘帮我保存卧推85公斤5组5次’，但不要真的保存。"
    task = _task("workout_record", source, action="answer", goal_role="result",
                 quoted=True, reply_mode="conversation", reply_purpose="general")
    value = validate_understanding(source, _understanding([task]))
    assert value["validated_tasks"][0]["task_type"] == "workout_record"
    assert value["validated_tasks"][0]["action"] in {"answer", "question"}
    assert not value["validated_tasks"][0]["side_effect"]
    token = _remember(source, value)
    try:
        parsed = {"tasks": [{key: item[key] for key in (
            "task_type", "action", "text", "is_negated", "is_hypothetical", "is_third_party")}
            for item in value["tasks"]]}
        planned = plan_tasks_from_llm(source, parsed)
    finally:
        request_understanding.reset(token)
    assert planned is not None and len(planned) == 1
    assert planned[0]["side_effect"] is False
    # The read keeps its record domain and is dispatched through the read-only
    # dialogue payload, never through the record write branch.
    assert planned[0]["task_type"] == "workout_record"
    assert planned[0]["payload"].get("dialogue", {}).get("quoted") is True


def test_quoted_record_with_explicit_write_action_remains_fail_closed():
    task = _task("workout_record", QUOTED_SOURCE, action="record", is_third_party=True,
                 goal_role="result", quoted=True, reply_mode="conversation",
                 reply_purpose="general", context_ids=[], fact_fields=[])
    with pytest.raises(SemanticValidationError) as caught:
        validate_understanding(QUOTED_SOURCE, _understanding([task]))
    assert caught.value.code == "semantic_quoted_write_invalid"


def test_real_first_person_save_next_to_quoted_other_does_not_cross_write():
    source = (
        "请记录我今天的卧推85公斤，5组，每组5次。"
        "这是同事发来的深蹲100公斤5组5次，不是我的，别登记到我名下。"
    )
    mine = "请记录我今天的卧推85公斤，5组，每组5次。"
    other = "这是同事发来的深蹲100公斤5组5次，不是我的，别登记到我名下。"
    raw = _understanding([
        _task("workout_record", mine, action="record", goal_role="result",
              quoted=False, reply_mode="grounded", fact_text="卧推85公斤，5组，每组5次"),
        _task("workout_record", other, action="answer", is_third_party=True,
              is_negated=True, goal_role="result", quoted=True,
              reply_mode="conversation", reply_purpose="general"),
    ])
    value = validate_understanding(source, raw)
    token = _remember(source, value)
    try:
        parsed = {"tasks": [{key: task[key] for key in (
            "task_type", "action", "text", "is_negated", "is_hypothetical", "is_third_party")}
            for task in value["tasks"]]}
        planned = plan_tasks_from_llm(source, parsed)
    finally:
        request_understanding.reset(token)
    assert planned is not None and len(planned) == 2
    write = next(item for item in planned if item["text"] == mine)
    quoted = next(item for item in planned if item["text"] == other)
    assert write["side_effect"] is True and write["action"] == "record"
    assert write["worker"] == "workout_record"
    assert quoted["side_effect"] is False and quoted["action"] in {"answer", "question"}
    assert quoted["worker"] != "workout_record"
    assert "semantic" in quoted["payload"]


def test_quoted_third_person_end_to_end_zero_write_sqlite(compact_db, monkeypatch):
    from test_retry_operations import RetryScenario

    scenario = RetryScenario(monkeypatch, compact_db)

    async def structured(messages, schema, *, name=None, **kwargs):
        if name == "return_request_understanding":
            return schema.model_validate(_quoted_payload())
        if name == "return_dialogue_answer":
            return schema.model_validate({
                "answer": "这是同事的训练，不是你的；我不会登记到你名下。",
                "requires_grounding": False,
            })
        raise AssertionError(name)

    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", structured)
    response = asyncio.run(scenario.respond(ChatRequest(message=QUOTED_SOURCE)))
    assert "同事" in response.answer and "不会登记" in response.answer
    assert scenario.saves == []
    assert all(call != "save_workout_record" for call in scenario.tool_calls)
    with compact_db() as session:
        assert session.execute(text("SELECT COUNT(*) FROM workout_records")).scalar() == 0


# ---------------------------------------------------------------------------
# SIM28-POST-03: plan information extraction is scoped to its target
# ---------------------------------------------------------------------------

def _plan_state(task_text, contract, *, sibling=None):
    state = {
        "raw_message": TWO_PLAN_SOURCE,
        "task_text": task_text,
        "task_id": "t-plan",
        "plan_contract": contract,
        "semantic_scope": {"fact_fields": [], "critical_fact_fields": [],
                           "context_ids": [], "critical_history_ids": []},
    }
    if sibling is not None:
        state["plan_siblings"] = [sibling]
    return state


def _plan_info_payload(*, equipment=(), session_time=None, include_common=True):
    facts = []
    if include_common:
        facts.extend([
            {"field": "goal", "source_id": "current", "quote": "我成年，刚开始增肌，没有伤病"},
            {"field": "experience", "source_id": "current", "quote": "我成年，刚开始增肌，没有伤病"},
        ])
    facts.extend({"field": "equipment", "source_id": "current", "quote": quote}
                 for quote in equipment)
    if session_time:
        quote, seconds = session_time
        facts.append({"field": "session_time", "source_id": "current",
                      "quote": quote, "quantity_quote": "20分钟", "seconds": seconds})
    return {"scope": "session", "facts": facts, "phase_limits": []}


def _plan_llm(payload, seen=None):
    async def structured(messages, schema, **kwargs):
        if seen is not None:
            seen.append(json.loads(messages[-1]["content"]))
        if isinstance(payload, Exception):
            raise payload
        return PlanInformation.model_validate(payload) if schema is PlanInformation else payload
    return SimpleNamespace(complete_structured=AsyncMock(side_effect=structured))


def test_saved_two_plan_payload_does_not_reject_sibling_equipment_for_leader():
    saved = _plan_info_payload(
        equipment=("第一份器械只能用哑铃", "第二份只能用弹力带"),
        session_time=("每份各自从热身、主体、组间休息转场到放松合计不超过20分钟", 1200),
    )
    seen = []
    details, clarification = asyncio.run(check_plan_information(
        _plan_llm(saved, seen), _plan_state(PLAN_LEADER, PLAN_CONTRACT)))
    assert clarification is None
    assert details["known"]["equipment"] == "第一份器械只能用哑铃"
    assert details["known"]["goal"] == "我成年，刚开始增肌，没有伤病"
    assert details["session_time_seconds"] == 1200
    assert seen[0]["sources"]["current"] == PLAN_LEADER
    assert "第二份只能用弹力带" not in seen[0]["sources"]["current"]


def test_saved_two_plan_payload_keeps_member_own_equipment():
    saved = _plan_info_payload(
        equipment=("第一份器械只能用哑铃", "第二份只能用弹力带"),
        session_time=("每份各自从热身、主体、组间休息转场到放松合计不超过20分钟", 1200),
    )
    member = PLAN_MEMBER
    details, clarification = asyncio.run(check_plan_information(
        _plan_llm(saved), _plan_state(member, PLAN_CONTRACT_MEMBER)))
    assert clarification is None
    assert details["known"]["equipment"] == "第二份只能用弹力带"
    assert details["session_time_seconds"] == 1200


def test_target_quote_removes_sibling_equipment_from_shared_source():
    state = _plan_state(PLAN_LEADER, {**PLAN_CONTRACT, "plan_target_quote": "第一份器械只能用哑铃"},
                        sibling={"text": PLAN_MEMBER,
                                 "plan_target_quote": "第二份只能用弹力带"})
    seen = []
    saved = _plan_info_payload(equipment=("第一份器械只能用哑铃",),
                               session_time=("每份各自从热身、主体、组间休息转场到放松合计不超过20分钟", 1200))
    details, clarification = asyncio.run(check_plan_information(_plan_llm(saved, seen), state))
    assert clarification is None
    assert details["known"]["equipment"] == "第一份器械只能用哑铃"
    shared = seen[0]["sources"].get("shared", "")
    # Sibling-specific equipment is excluded from this target's shared source;
    # shared conditions (20 minutes) remain available.
    assert "第二份只能用弹力带" not in shared
    assert "20分钟" in shared


def test_independent_targets_keep_different_durations():
    leader = "第一份哑铃增肌新手计划合计20分钟。"
    member = "第二份弹力带增肌新手计划合计30分钟。"
    source = leader + member
    shared_contract = {"plan_result": True, "result_scope": "independent",
                       "plan_request_quote": "第一份哑铃增肌新手计划", "plan_target_quote": "第一份哑铃增肌新手计划合计20分钟"}
    member_contract = {"plan_result": True, "result_scope": "independent",
                       "plan_request_quote": "第二份弹力带增肌新手计划", "plan_target_quote": "第二份弹力带增肌新手计划合计30分钟"}
    leader_payload = {"scope": "session", "facts": [
        {"field": "goal", "source_id": "current", "quote": "增肌新手"},
        {"field": "experience", "source_id": "current", "quote": "增肌新手"},
        {"field": "equipment", "source_id": "current", "quote": "哑铃"},
        {"field": "session_time", "source_id": "current", "quote": leader,
         "quantity_quote": "20分钟", "seconds": 1200},
    ]}
    member_payload = {"scope": "session", "facts": [
        {"field": "goal", "source_id": "current", "quote": "增肌新手"},
        {"field": "experience", "source_id": "current", "quote": "增肌新手"},
        {"field": "equipment", "source_id": "current", "quote": "弹力带"},
        {"field": "session_time", "source_id": "current", "quote": member,
         "quantity_quote": "30分钟", "seconds": 1800},
    ]}
    leader_state = {"raw_message": source, "task_text": leader, "task_id": "a",
                    "plan_contract": shared_contract,
                    "plan_siblings": [{"text": member, "plan_target_quote": member_contract["plan_target_quote"]}]}
    member_state = {"raw_message": source, "task_text": member, "task_id": "b",
                    "plan_contract": member_contract,
                    "plan_siblings": [{"text": leader, "plan_target_quote": shared_contract["plan_target_quote"]}]}
    left, left_clarification = asyncio.run(check_plan_information(_plan_llm(leader_payload), leader_state))
    right, right_clarification = asyncio.run(check_plan_information(_plan_llm(member_payload), member_state))
    assert left_clarification is None and right_clarification is None
    assert left["known"]["equipment"] == "哑铃" and left["session_time_seconds"] == 1200
    assert right["known"]["equipment"] == "弹力带" and right["session_time_seconds"] == 1800


def test_plan_target_quote_reaches_worker_contract():
    raw = _understanding([
        _task("training_plan", PLAN_LEADER, plan_result=True,
              result_scope="shared_leader", plan_goal_id="today_two_plans",
              plan_request_quote="要两份互相独立的今天训练安排",
              shared_scope_quote="要两份互相独立的今天训练安排",
              plan_target_quote="第一份器械只能用哑铃"),
        _task("training_plan", PLAN_MEMBER, plan_result=True,
              result_scope="shared_member", plan_goal_id="today_two_plans",
              plan_target_quote="第二份只能用弹力带"),
    ])
    value = validate_understanding(TWO_PLAN_SOURCE, raw)
    token = _remember(TWO_PLAN_SOURCE, value)
    try:
        tasks = plan_tasks_rule(TWO_PLAN_SOURCE)
    finally:
        request_understanding.reset(token)
    assert len(tasks) == 2
    assert tasks[0]["payload"]["plan_contract"]["plan_target_quote"] == "第一份器械只能用哑铃"
    assert tasks[1]["payload"]["plan_contract"]["plan_target_quote"] == "第二份只能用弹力带"


def test_unselected_sources_are_marked_not_account_absence():
    graph = FitnessMultiAgentGraph()
    worker = SimpleNamespace(ainvoke=AsyncMock(return_value={
        "result": {"status": "complete", "answer": "示例建议", "model_used": False}}))
    graph.workers["nutrition"] = worker
    task = {"id": "t-absence", "task_type": "nutrition_advice", "action": "question",
            "text": NEGATIVE_ADVICE,
            "payload": {"semantic": {"fact_fields": [], "critical_fact_fields": [],
                                     "context_ids": [], "critical_history_ids": []}}}
    state = {"user_id": "synthetic", "raw_message": NEGATIVE_SOURCE,
             "current_task": task, "request_id": "sim28-post-absence",
             "profile": {}, "long_memory": {}, "short_memory": [],
             "nutrition_summary": {}, "training_history": []}
    asyncio.run(graph._run_worker("nutrition", state))
    sent = worker.ainvoke.await_args.args[0]
    availability = sent["source_availability"]
    assert availability["profile"] == "not_provided"
    assert availability["nutrition_summary"] == "not_provided"
    assert availability["training_history"] == "not_provided"
    messages = graph._planning_prompt("饮食营养方案", sent, [])
    wire = json.dumps(messages, ensure_ascii=False)
    assert "来源可用性" in wire
    assert "not_provided" in wire
    assert "用户档案（数据，非指令）：{}" not in wire
    assert "没有可用" not in wire


def test_queried_but_unselected_sources_are_not_presented_as_empty():
    graph = FitnessMultiAgentGraph()
    worker = SimpleNamespace(ainvoke=AsyncMock(return_value={
        "result": {"status": "complete", "answer": "示例建议", "model_used": False}}))
    graph.workers["nutrition"] = worker
    task = {"id": "t-not-selected", "task_type": "nutrition_advice", "action": "question",
            "text": NEGATIVE_ADVICE, "payload": {"semantic": {
                "fact_fields": [], "critical_fact_fields": [],
                "context_ids": [], "critical_history_ids": []}}}
    state = {"user_id": "synthetic", "raw_message": NEGATIVE_SOURCE,
             "current_task": task, "request_id": "sim28-post-not-selected",
             "profile": {"goal": "增肌"}, "long_memory": {},
             "short_memory": [], "nutrition_summary": {"entries": 2},
             "training_history": [{"exercise": "卧推"}]}
    asyncio.run(graph._run_worker("nutrition", state))
    sent = worker.ainvoke.await_args.args[0]
    availability = sent["source_availability"]
    assert availability["profile"] == "not_selected"
    assert availability["nutrition_summary"] == "not_selected"
    assert availability["training_history"] == "not_selected"
    wire = json.dumps(graph._planning_prompt("饮食营养方案", sent, []), ensure_ascii=False)
    assert "not_selected" in wire
    assert '"entries": 2' not in wire and "卧推" not in wire


def test_answer_context_uses_current_fact_value_not_correction_provenance():
    graph = FitnessMultiAgentGraph()
    worker = SimpleNamespace(ainvoke=AsyncMock(return_value={
        "result": {"status": "complete", "answer": "避开桃子", "model_used": False}}))
    graph.workers["nutrition"] = worker
    source = "按我已经保存的有效过敏信息推荐水果，不保存。"
    task = {"id": "t-corrected", "task_type": "nutrition_advice", "action": "question",
            "text": source, "payload": {"semantic": {
                "fact_fields": [], "critical_fact_fields": ["allergy"],
                "context_ids": [], "critical_history_ids": []}}}
    state = {"user_id": "synthetic", "raw_message": source, "current_task": task,
             "request_id": "sim28-post-corrected",
             "long_memory": {"allergy": {
                 "text": "我对桃子过敏",
                 "source_quote": "芒果过敏说错了，实际是我对桃子过敏",
                 "source_message_id": "u-corrected"}}}
    asyncio.run(graph._run_worker("nutrition", state))
    sent = worker.ainvoke.await_args.args[0]
    assert sent["long_memory"]["allergy"]["source_quote"].startswith("芒果")
    wire = json.dumps(graph._planning_prompt("饮食营养方案", sent, []), ensure_ascii=False)
    assert "我对桃子过敏" in wire
    assert "芒果过敏说错了" not in wire


def test_empty_fact_recall_still_explains_no_matching_record():
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock())
    value = {"reply_mode": "conversation", "reply_purpose": "fact_recall",
             "tasks": [{"reply_purpose": "fact_recall", "context_ids": [], "fact_fields": []}]}
    result = asyncio.run(dialogue.conversation_answer(
        llm, ChatRequest(message="我昨天的体重记录是多少？"),
        {"long_memory": {}, "short_memory": []}, value))
    assert result["answer"] == "我没有找到与你这个问题对应的实际记录，无法给出具体信息。"
    llm.complete_structured.assert_not_awaited()


def test_plan_retrieval_keeps_verified_shared_conditions_without_sibling_equipment():
    graph = FitnessMultiAgentGraph()
    information = {
        "scope": "session",
        "known": {"goal": "增肌", "experience": "新手",
                  "equipment": "哑铃"},
        "session_time_seconds": None,
        "available_days_count": None,
        "phase_limits": [],
    }
    retrieval = AsyncMock(return_value=[])
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr("app.services.agent.task_information.check_plan_information",
                            AsyncMock(return_value=(information, None)))
        monkeypatch.setattr(graph, "_readonly_retrieval", retrieval)
        monkeypatch.setattr("app.services.agent.multi_agent_graph.assess_rag_evidence",
                            lambda *args, **kwargs: SimpleNamespace(
                                answerable=True, as_dict=lambda: {"answerable": True}))
        from tests.plan_fixture_support import plan_message
        answer = AsyncMock(return_value=plan_message("哑铃划船"))
        monkeypatch.setattr(graph.llm, "complete_message", answer)
        state = {"user_id": "synthetic",
                 "raw_message": "甲弹力带12分钟。乙哑铃18分钟，增肌新手。",
                 "task_text": "乙哑铃18分钟，增肌新手。",
                 "task_id": "t-plan-query", "task_type": "training_plan",
                 "task_action": "question", "plan_target_count": 2,
                 "plan_contract": {"result_scope": "shared_member",
                                   "plan_goal_id": "limits"},
                 "plan_siblings": [{"text": "甲弹力带12分钟。",
                                    "plan_target_quote": "甲弹力带12分钟。"}],
                 "profile": {}, "long_memory": {}, "short_memory": [],
                 "task_context": []}
        result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]
    finally:
        monkeypatch.undo()
    assert result["status"] == "complete"
    query = retrieval.await_args.args[0]
    assert "乙哑铃18分钟" in query
    assert "增肌" in query and "新手" in query and "哑铃" in query
    assert "弹力带" not in query


def test_plan_retrieval_insufficient_evidence_stays_insufficient():
    graph = FitnessMultiAgentGraph()
    information = {
        "scope": "session",
        "known": {"goal": "增肌", "experience": "新手",
                  "equipment": "弹力带"},
        "session_time_seconds": None, "available_days_count": None,
        "phase_limits": [],
    }
    retrieval = AsyncMock(return_value=[object()])
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr("app.services.agent.task_information.check_plan_information",
                            AsyncMock(return_value=(information, None)))
        monkeypatch.setattr(graph, "_readonly_retrieval", retrieval)
        monkeypatch.setattr("app.services.agent.multi_agent_graph.assess_rag_evidence",
                            lambda *args, **kwargs: SimpleNamespace(
                                answerable=False, reason="no_candidates", as_dict=lambda: {"answerable": False}))
        answer = AsyncMock(return_value=("不应生成", True))
        monkeypatch.setattr(graph, "_complete_or_fallback_with_mcp", answer)
        state = {"user_id": "synthetic", "raw_message": "乙哑铃18分钟，增肌新手。",
                 "task_text": "乙哑铃18分钟，增肌新手。", "task_id": "t-plan-poor",
                 "task_type": "training_plan", "task_action": "question",
                 "plan_target_count": 1, "plan_contract": {},
                 "profile": {}, "long_memory": {}, "short_memory": [],
                 "task_context": []}
        result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]
    finally:
        monkeypatch.undo()
    assert result["status"] == "insufficient_evidence"
    answer.assert_not_awaited()


def test_projected_shared_adjacency_is_not_a_real_source_quote():
    task = "当前只用弹力带，请给一节训练。"
    parent = task + "我是新手。目标增肌。兄弟仅用哑铃。每次20分钟。"
    state = {"raw_message": parent, "task_text": task,
             "plan_siblings": [{"text": "兄弟仅用哑铃。",
                                "plan_target_quote": "兄弟仅用哑铃。"}]}
    fabricated = {"scope": "session", "facts": [
        {"field": "goal", "source_id": "shared", "quote": "目标增肌。每次20分钟。"},
        {"field": "experience", "source_id": "shared", "quote": "我是新手。"},
        {"field": "equipment", "source_id": "current", "quote": "只用弹力带"},
        {"field": "session_time", "source_id": "shared", "quote": "每次20分钟。",
         "quantity_quote": "20分钟", "seconds": 1200},
    ]}
    details, clarification = asyncio.run(check_plan_information(_plan_llm(fabricated), state))
    assert clarification is not None and not details

    valid = {"scope": "session", "facts": [
        {"field": "goal", "source_id": "shared", "quote": "目标增肌。"},
        {"field": "experience", "source_id": "shared", "quote": "我是新手。"},
        {"field": "equipment", "source_id": "current", "quote": "只用弹力带"},
        {"field": "session_time", "source_id": "shared", "quote": "每次20分钟。",
         "quantity_quote": "20分钟", "seconds": 1200},
    ]}
    details, clarification = asyncio.run(check_plan_information(_plan_llm(valid), state))
    assert clarification is None
    assert details["known"]["goal"] == "目标增肌。"
    assert details["session_time_seconds"] == 1200


def test_projected_phase_context_cannot_cross_a_removed_sibling():
    task = "当前只用弹力带，请给一节训练。"
    parent = task + "我是新手。目标增肌。兄弟仅用哑铃。热身3分钟。每次20分钟。"
    state = {"raw_message": parent, "task_text": task,
             "plan_siblings": [{"text": "兄弟仅用哑铃。",
                                "plan_target_quote": "兄弟仅用哑铃。"}]}
    raw = {"scope": "session", "facts": [
        {"field": "goal", "source_id": "shared", "quote": "目标增肌。"},
        {"field": "experience", "source_id": "shared", "quote": "我是新手。"},
        {"field": "equipment", "source_id": "current", "quote": "只用弹力带"},
        {"field": "session_time", "source_id": "shared", "quote": "每次20分钟。",
         "quantity_quote": "20分钟", "seconds": 1200},
    ], "phase_limits": [{
        "phase": "warmup", "relation": "exact", "seconds": 180,
        "source_id": "shared", "context_quote": "目标增肌。热身3分钟。",
        "quantity_quote": "3分钟",
    }]}
    details, clarification = asyncio.run(check_plan_information(_plan_llm(raw), state))
    assert clarification is not None and not details


def test_repeated_task_text_binds_selectors_by_source_occurrence():
    clause = "复述已保存的信息。"
    source = clause + clause
    raw = {"tasks": [
        {"task_type": "chitchat", "action": "question", "text": clause,
         "is_negated": False, "is_hypothetical": False, "is_third_party": False,
         "reply_mode": "conversation", "reply_purpose": "fact_recall",
         "fact_fields": [key], "goal_role": "result", "quoted": False}
        for key in ["first", "second"]
    ], "reply_mode": "conversation", "reply_purpose": "fact_recall",
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "mock"}}
    facts = {key: {"text": value, "source_quote": value,
                   "source_message_id": key + ":user"}
             for key, value in [("first", "第一个来源"), ("second", "第二个来源")]}
    value = validate_understanding(source, raw, context={"long_memory": facts})
    token = _remember(source, value)
    try:
        planned = plan_tasks_rule(source)
    finally:
        request_understanding.reset(token)
    assert len(planned) == 2
    assert [task["payload"]["semantic"]["fact_fields"] for task in planned] == [
        ["first"], ["second"]]


def test_repeated_text_without_span_does_not_borrow_ambiguous_selectors():
    from app.services.agent.dialogue_context import task_answer_scope

    clause = "复述已保存的信息。"
    source = clause + clause
    understanding = {"tasks": [
        {"text": clause, "fact_fields": ["first"]},
        {"text": clause, "fact_fields": ["second"]},
    ]}
    ambiguous = task_answer_scope({"text": clause, "payload": {}}, understanding,
                                  source=source)
    assert not ambiguous.get("fact_fields")
    misplaced = task_answer_scope({"text": clause, "source_start": 1,
                                   "source_end": len(clause) + 1, "payload": {}},
                                  understanding, source=source)
    assert not misplaced.get("fact_fields")
    first = task_answer_scope({"text": clause, "source_start": 0,
                               "source_end": len(clause), "payload": {}},
                              understanding, source=source)
    second = task_answer_scope({"text": clause, "source_start": len(clause),
                                "source_end": len(source), "payload": {}},
                               understanding, source=source)
    assert first["fact_fields"] == ["first"]
    assert second["fact_fields"] == ["second"]


def test_true_same_field_conflict_asks_for_clarification():
    source = "请安排训练：用哑铃或弹力带都行，30分钟。"
    state = {"raw_message": source, "task_text": source, "task_id": "conflict"}
    payload = {"scope": "session", "facts": [
        {"field": "equipment", "source_id": "current", "quote": "用哑铃"},
        {"field": "equipment", "source_id": "current", "quote": "弹力带都行"},
    ]}
    _, clarification = asyncio.run(check_plan_information(_plan_llm(payload), state))
    assert clarification is not None
    assert clarification.category == "invalid_parameter"


def test_unbound_or_foreign_plan_fact_is_not_used():
    state = {"raw_message": "安排明天训练，30分钟。", "task_text": "安排明天训练，30分钟。",
             "task_id": "foreign"}
    payload = {"scope": "session", "facts": [
        {"field": "equipment", "source_id": "foreign", "quote": "健身房"},
    ]}
    _, clarification = asyncio.run(check_plan_information(_plan_llm(payload), state))
    assert clarification is not None
    assert clarification.category == "service_unavailable"


def test_plan_information_does_not_read_unselected_long_memory_or_history():
    seen = []
    state = {
        "raw_message": "安排明天训练。",
        "task_text": "安排明天训练。",
        "task_id": "scope",
        "long_memory": {"peach": {"text": PRIVATE, "source_quote": PRIVATE}},
        "short_memory": [{"message_id": "u", "role": "user", "content": PRIVATE}],
        "semantic_scope": {"fact_fields": [], "critical_fact_fields": [],
                           "context_ids": [], "critical_history_ids": []},
    }
    _, clarification = asyncio.run(check_plan_information(
        _plan_llm({"scope": "session", "facts": []}, seen), state))
    assert clarification is not None
    assert PRIVATE not in json.dumps(seen, ensure_ascii=False)


def test_plan_phase_limits_still_use_verbatim_source_and_total_boundary():
    source = "安排明天训练，总18分钟，热身固定3分钟，收尾固定2分钟，器械只有哑铃，新手增肌。"
    state = {"raw_message": source, "task_text": source, "task_id": "phase"}
    payload = {"scope": "session", "facts": [
        {"field": "goal", "source_id": "current", "quote": "新手增肌"},
        {"field": "experience", "source_id": "current", "quote": "新手增肌"},
        {"field": "equipment", "source_id": "current", "quote": "器械只有哑铃"},
        {"field": "session_time", "source_id": "current", "quote": "总18分钟",
         "quantity_quote": "18分钟", "seconds": 1080},
    ], "phase_limits": [
        {"phase": "warmup", "relation": "exact", "seconds": 180,
         "source_id": "current", "context_quote": "热身固定3分钟", "quantity_quote": "3分钟"},
        {"phase": "cooldown", "relation": "exact", "seconds": 120,
         "source_id": "current", "context_quote": "收尾固定2分钟", "quantity_quote": "2分钟"},
    ]}
    details, clarification = asyncio.run(check_plan_information(_plan_llm(payload), state))
    assert clarification is None
    assert details["session_time_seconds"] == 1080
    assert [(item["phase"], item["seconds"]) for item in details["phase_limits"]] == [
        ("warmup", 180), ("cooldown", 120)]


# ---------------------------------------------------------------------------
# React read-only caller shares the same per-target contract
# ---------------------------------------------------------------------------

def test_react_plan_information_uses_matched_target_and_shared_scope():
    from app.services.agent import react_readonly_workers

    context = {
        "source_text": TWO_PLAN_SOURCE,
        "task_text": PLAN_LEADER,
        "task_id": "react-plan",
        "plan_contract": {**PLAN_CONTRACT, "plan_target_quote": "第一份器械只能用哑铃"},
        "plan_siblings": [{"text": PLAN_MEMBER,
                           "plan_target_quote": "第二份只能用弹力带"}],
        "semantic_scope": {"fact_fields": [], "critical_fact_fields": [],
                           "context_ids": [], "critical_history_ids": []},
    }
    state = react_readonly_workers._plan_information_state(context)
    assert state["task_text"] == PLAN_LEADER
    assert state["plan_contract"]["plan_target_quote"] == "第一份器械只能用哑铃"
    seen = []
    saved = _plan_info_payload(
        equipment=("第一份器械只能用哑铃",),
        session_time=("每份各自从热身、主体、组间休息转场到放松合计不超过20分钟", 1200),
    )
    details, clarification = asyncio.run(check_plan_information(_plan_llm(saved, seen), state))
    assert clarification is None
    assert details["known"]["equipment"] == "第一份器械只能用哑铃"
    assert seen[0]["sources"]["current"] == PLAN_LEADER
    assert "第二份只能用弹力带" not in seen[0]["sources"].get("shared", "")


def test_grounded_worker_rejects_hidden_selected_source(compact_db):
    from app.core import clock
    from app.db.memory_models import MemoryTurn
    from app.services.memory import compact
    from test_ai_common_causes import save_memory

    save_memory("我对桃子过敏", "allergy")
    facts = compact.current_facts()
    field, fact = next(iter(facts.items()))
    request = ChatRequest(message="给一个饮食建议", conversation_id="scope-hidden")
    turn, _ = compact.begin_turn(request)
    token = compact.current_turn.set(turn)
    try:
        with compact_db() as session, session.begin():
            session.get(MemoryTurn, fact["source_turn_id"]).hidden_at = clock.utc_now()
        graph = FitnessMultiAgentGraph()
        worker = SimpleNamespace(ainvoke=AsyncMock(return_value={
            "result": {"status": "complete", "answer": "不应生成", "model_used": False}}))
        graph.workers["nutrition"] = worker
        task = {"id": "hidden", "task_type": "nutrition_advice", "action": "question",
                "text": "给一个饮食建议", "payload": {"semantic": {
                    "fact_fields": [], "critical_fact_fields": [field],
                    "context_ids": [], "critical_history_ids": []}}}
        result = asyncio.run(graph._run_worker("nutrition", {
            "user_id": "demo-user", "conversation_id": "scope-hidden",
            "request_id": "scope-hidden", "raw_message": "给一个饮食建议",
            "long_memory": facts, "current_task": task}))
        assert result["worker_results"][-1]["status"] == "unavailable"
        worker.ainvoke.assert_not_awaited()
    finally:
        compact.current_turn.reset(token)


def test_react_readonly_candidates_include_normalized_quoted_read():
    from app.services.agent import react_runtime
    from app.services.agent.react_graph import _validated_read_source_hints

    value = validate_understanding(QUOTED_SOURCE, _quoted_payload())
    token = _remember(QUOTED_SOURCE, value)
    try:
        parsed = {"tasks": [{key: task[key] for key in (
            "task_type", "action", "text", "is_negated", "is_hypothetical", "is_third_party")}
            for task in value["tasks"]]}
        planned = plan_tasks_from_llm(QUOTED_SOURCE, parsed)
    finally:
        request_understanding.reset(token)
    candidates = react_runtime._readonly_candidates(planned)
    assert len(candidates) == 1
    assert candidates[0]["task_type"] == "workout_record"
    assert candidates[0]["side_effect"] is False
    hints = _validated_read_source_hints(candidates, QUOTED_SOURCE)
    assert hints and hints[0]["source_ref"]["exact_text"] == QUOTED_SOURCE


# ---------------------------------------------------------------------------
# SIM28-POST-03: the final timed plan uses the existing two-attempt structured
# correction for schema and numeric/target contract failures.
# ---------------------------------------------------------------------------

CALL16_TARGET = "方案乙只用哑铃，热身到收尾合计最多18分钟。"


def _timed_plan_payload(*, warmup=240, main=600, breaks=180, cooldown=60,
                        label="哑铃全身增肌单次训练"):
    return {"sessions": [{
        "label": label,
        "warmup": {"seconds": warmup, "activity": "原地踏步与肩部活动"},
        "main": {"seconds": main, "exercises": [
            {"name": "哑铃高脚杯深蹲", "sets": 3, "reps": 10, "hold_seconds": None},
            {"name": "哑铃俯身划船", "sets": 3, "reps": 10, "hold_seconds": None},
        ]},
        "breaks": {"seconds": breaks},
        "cooldown": {"seconds": cooldown, "activity": "放松与拉伸"},
        "time_consistency": "consistent",
    }]}


def _saved_call16_arguments():
    """The exact saved shape: ``main.breaks`` with no session ``breaks``."""
    payload = _timed_plan_payload()
    session = payload["sessions"][0]
    session["main"]["breaks"] = {"seconds": 180}
    del session["breaks"]
    return json.dumps(payload, ensure_ascii=False)


def _timed_tool(arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "type": "function", "function": {
            "name": "return_timed_training_plan", "arguments": arguments}}]}


def _timed_information(seconds, *, phase_limits=None, equipment="方案乙只用哑铃"):
    return {
        "scope": "session",
        "known": {"goal": "增肌", "experience": "新手", "equipment": equipment,
                  "session_time": f"热身到收尾合计最多{seconds // 60}分钟"},
        "session_time_seconds": seconds,
        "available_days_count": None,
        "phase_limits": phase_limits or [],
    }


def _wire_timed_plan_worker(monkeypatch, responses, *, information=None,
                            source=CALL16_TARGET):
    graph = FitnessMultiAgentGraph()
    monkeypatch.setattr("app.services.agent.task_information.check_plan_information",
                        AsyncMock(return_value=(information or _timed_information(1080), None)))
    monkeypatch.setattr(graph, "_readonly_retrieval", AsyncMock(return_value=[]))
    monkeypatch.setattr("app.services.agent.multi_agent_graph.assess_rag_evidence",
                        lambda *args, **kwargs: SimpleNamespace(
                            answerable=True, as_dict=lambda: {"answerable": True}))
    calls = []
    iterator = None if callable(responses) else iter(responses)

    async def complete_message(messages, **kwargs):
        calls.append(messages)
        item = await responses(messages) if callable(responses) else next(iterator)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            item = json.dumps(item, ensure_ascii=False)
        return _timed_tool(item)

    monkeypatch.setattr(graph.llm, "complete_message", complete_message)
    state = {
        "user_id": "synthetic",
        "raw_message": source,
        "task_text": source,
        "task_id": "timed-target",
        "plan_contract": {
            "plan_result": True, "result_scope": "independent",
            "plan_request_quote": source, "plan_target_quote": source,
        },
        "profile": {},
    }
    return graph, state, calls


def _correction_text(calls):
    return "\n".join(str(item.get("content") or "") for item in calls[-1])


def test_saved_call16_nested_breaks_gets_one_bounded_schema_correction(monkeypatch):
    valid = json.dumps(_timed_plan_payload(), ensure_ascii=False)
    graph, state, calls = _wire_timed_plan_worker(
        monkeypatch, [_saved_call16_arguments(), valid])
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]

    assert result["status"] == "complete" and result["model_used"] is True
    assert len(calls) == 2
    assert "| 组间休息与转换（穿插进行） | 3分钟 |" in result["answer"]
    assert "合计：18分钟，不超过18分钟。" in result["answer"]
    # The second call is the existing bounded correction, not an application
    # relocation of ``main.breaks``: the first raw result never completed.
    first_prompt = "\n".join(str(item.get("content") or "") for item in calls[0])
    first_prompt = first_prompt.replace('\\"', '"')
    assert '"time_limit_seconds": 1080' in first_prompt
    correction = _correction_text(calls)
    assert "extra_forbidden" in correction and "missing" in correction


def test_saved_call16_shape_twice_fails_honestly_after_two_calls(monkeypatch):
    graph, state, calls = _wire_timed_plan_worker(
        monkeypatch, [_saved_call16_arguments(), _saved_call16_arguments()])
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]

    assert result["status"] == "error" and result["model_used"] is True
    assert "时间分配未通过校验" in result["answer"]
    assert "time_consistency" not in result["answer"]
    assert len(calls) == 2


def test_valid_timed_plan_uses_only_one_call(monkeypatch):
    valid = json.dumps(_timed_plan_payload(), ensure_ascii=False)
    graph, state, calls = _wire_timed_plan_worker(monkeypatch, [valid])
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]

    assert result["status"] == "complete" and result["model_used"] is True
    assert len(calls) == 1


def test_duplicate_plan_field_keeps_the_renderer_as_the_correction_oracle(monkeypatch):
    valid_payload = _timed_plan_payload()
    duplicate = ('{"sessions":[],"sessions":'
                 + json.dumps(valid_payload["sessions"], ensure_ascii=False) + "}")
    valid = json.dumps(valid_payload, ensure_ascii=False)
    graph, state, calls = _wire_timed_plan_worker(monkeypatch, [duplicate, valid])
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]

    assert result["status"] == "complete"
    assert len(calls) == 2
    assert "duplicate_plan_field" in _correction_text(calls)


def test_schema_valid_over_limit_plan_gets_bounded_numeric_correction(monkeypatch):
    over = json.dumps(_timed_plan_payload(warmup=240, main=720, breaks=180,
                                          cooldown=60), ensure_ascii=False)
    valid = json.dumps(_timed_plan_payload(), ensure_ascii=False)
    graph, state, calls = _wire_timed_plan_worker(monkeypatch, [over, valid])
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]

    assert result["status"] == "complete"
    assert len(calls) == 2
    correction = _correction_text(calls)
    assert "session_time_exceeded" in correction
    assert "1200" not in correction


def test_phase_limit_mismatch_gets_bounded_correction(monkeypatch):
    phase_limits = [{"phase": "warmup", "relation": "exact", "seconds": 300}]
    wrong_phase = json.dumps(_timed_plan_payload(warmup=240, main=600, breaks=180,
                                                 cooldown=60), ensure_ascii=False)
    valid = json.dumps(_timed_plan_payload(warmup=300, main=540, breaks=180,
                                           cooldown=60), ensure_ascii=False)
    graph, state, calls = _wire_timed_plan_worker(
        monkeypatch, [wrong_phase, valid],
        information=_timed_information(1080, phase_limits=phase_limits))
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]

    assert result["status"] == "complete"
    assert len(calls) == 2
    assert "phase_limit_violation" in _correction_text(calls)
    assert "| 准备／热身 | 5分钟 |" in result["answer"]


def test_distinct_target_limits_do_not_share_each_others_budget(monkeypatch):
    seen_prompts = []

    async def respond(messages):
        text = "\n".join(str(item.get("content") or "") for item in messages)
        text = text.replace('\\"', '"')
        seen_prompts.append(text)
        if '"time_limit_seconds": 720' in text:
            count = sum(1 for prompt in seen_prompts if '"time_limit_seconds": 720' in prompt)
            if count == 1:
                return _timed_plan_payload(warmup=240, main=480, breaks=180,
                                           cooldown=60)  # 960, over 720
            return _timed_plan_payload(warmup=180, main=420, breaks=60,
                                       cooldown=60)  # exactly 720
        assert '"time_limit_seconds": 1080' in text
        return _timed_plan_payload()

    short_graph, short_state, short_calls = _wire_timed_plan_worker(
        monkeypatch, respond,
        information=_timed_information(720, equipment="方案甲只用弹力带"))
    short = asyncio.run(short_graph.workers["plan_adjust"].ainvoke(short_state))["result"]
    long_graph, long_state, long_calls = _wire_timed_plan_worker(
        monkeypatch, respond, information=_timed_information(1080))
    long = asyncio.run(long_graph.workers["plan_adjust"].ainvoke(long_state))["result"]

    assert short["status"] == "complete" and "不超过12分钟" in short["answer"]
    assert long["status"] == "complete" and "不超过18分钟" in long["answer"]
    assert len(short_calls) == 2 and len(long_calls) == 1
    short_prompts = ["\n".join(str(item.get("content") or "") for item in call).replace('\\"', '"')
                     for call in short_calls]
    long_prompts = ["\n".join(str(item.get("content") or "") for item in call).replace('\\"', '"')
                    for call in long_calls]
    assert all('"time_limit_seconds": 720' in prompt for prompt in short_prompts)
    assert all('"time_limit_seconds": 1080' in prompt for prompt in long_prompts)


def test_second_correction_is_not_sent_after_a_task_budget_stop(monkeypatch):
    from app.services.llm.budget import ModelTaskBudgetError

    attempts = []
    provider_responses = []

    async def respond(messages):
        attempts.append(messages)
        if len(attempts) == 1:
            provider_responses.append(messages)
            return _saved_call16_arguments()
        raise ModelTaskBudgetError("model_task_budget_exhausted:task_plan")

    graph, state, calls = _wire_timed_plan_worker(monkeypatch, respond)
    with pytest.raises(ModelTaskBudgetError):
        asyncio.run(graph.workers["plan_adjust"].ainvoke(state))
    assert len(calls) == 2 and len(attempts) == 2
    # The second attempt stopped at reservation; only the first provider result
    # exists, so nothing was sent after the budget stop.
    assert len(provider_responses) == 1

