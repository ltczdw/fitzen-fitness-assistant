"""SIM28 common-cause regressions reproduced from the saved uniform12 output.

All model/provider boundaries are mocked; temporary SQLite is used for the C1
rejection-replay case.  No paid call, personal database or real service.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.services.agent import dialogue_context as dialogue
from app.services.agent import semantics
from app.services.agent.react_graph import (
    ReactDependencies,
    _unhandled_goals,
    prepare_node,
)
from app.services.agent.semantics import (
    RequestUnderstanding,
    SemanticValidationError,
    remember_safety,
    request_understanding,
    validate_understanding,
)
from app.services.llm.structured import (
    _previous_goal_summary,
    complete_structured,
)


# ---------------------------------------------------------------------------
# SIM28-01: malformed JSON tail still gives a sanitized task skeleton + syntax
# ---------------------------------------------------------------------------

SOURCE_01 = (
    "请把我今天做的哑铃划船写进日志：24公斤，3组，每组12次。"
    "然后说明普通休息日怎么安排。这次不保存任何记录。"
)
TASK_01 = "请把我今天做的哑铃划船写进日志：24公斤，3组，每组12次。"
TASK_01B = "然后说明普通休息日怎么安排。这次不保存任何记录。"


def _tool_message(arguments, call_id="sim28-correction"):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": "return_request_understanding", "arguments": arguments},
    }]}


def _task(kind, text, *, action="question", history_requirement="none", **extra):
    return {"task_type": kind, "action": action, "text": text,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            "history_requirement": history_requirement, **extra}


def _understanding(tasks, *, safety=None, **extra):
    return {"tasks": [{**task, "history_requirement": task.get("history_requirement", "none")}
                      for task in tasks], "safety": safety or {
        "level": "normal", "evidence": "", "categories": [], "reason": "scripted normal"},
        **extra}


def _malformed_tasks_only(tasks, *, private="PRIVATE-VALUE"):
    payload = json.dumps({"tasks": tasks}, ensure_ascii=False)
    # Exact saved failure shape: the task array is complete, then a duplicated
    # colon and a private value begin after the next field name.
    return payload[:-1] + ', "safety":: "' + private + '"} '


def test_saved_invalid_json_tail_recovers_only_task_context():
    tasks = [
        _task("workout_record", TASK_01, action="record"),
        _task("training_qa", TASK_01B, is_negated=True),
    ]
    original = [{"role": "user", "content": json.dumps(
        {"query": SOURCE_01}, ensure_ascii=False)}]
    summary = _previous_goal_summary(_tool_message(_malformed_tasks_only(tasks)), original)
    assert summary is not None
    recovered = json.loads(summary)["tasks"]
    assert [item["text"] for item in recovered] == [TASK_01, TASK_01B]
    assert [item["task_type"] for item in recovered] == ["workout_record", "training_qa"]


def test_saved_invalid_json_correction_has_redacted_field_feedback_and_context():
    tasks = [
        _task("workout_record", TASK_01, action="record"),
        _task("training_qa", TASK_01B, is_negated=True),
    ]

    class RepairClient:
        def __init__(self):
            self.prompts = []

        async def complete_message(self, messages, **kwargs):
            self.prompts.append(json.loads(json.dumps(messages, ensure_ascii=False)))
            if len(self.prompts) == 1:
                return _tool_message(_malformed_tasks_only(tasks))
            return _tool_message(json.dumps(_understanding(tasks), ensure_ascii=False))

    client = RepairClient()
    seen = []

    def validator(value):
        seen.append([item.text for item in value.tasks])

    result = asyncio.run(complete_structured(
        client,
        [{"role": "system", "content": "semantic result only"},
         {"role": "user", "content": json.dumps({"query": SOURCE_01}, ensure_ascii=False)}],
        RequestUnderstanding, name="return_request_understanding", validator=validator,
    ))
    correction = "\n".join(str(item.get("content") or "") for item in client.prompts[1]
                           if item.get("role") == "system")
    assert len(client.prompts) == 2
    assert [item.text for item in result.tasks] == [TASK_01, TASK_01B]
    assert "UNTRUSTED_PREVIOUS" in correction
    assert TASK_01 in correction and TASK_01B in correction
    assert '"field": "safety"' in correction
    assert '"syntax"' in correction
    assert "PRIVATE-VALUE" not in correction
    assert "never copy or invent a safety value" in correction
    assert seen == [[TASK_01, TASK_01B]]


def test_previous_goal_summary_drops_an_invented_previous_task():
    tasks = [_task("workout_record", TASK_01, action="record"),
             _task("workout_record", "帮我把这个训练保存下来", action="record")]
    original = [{"role": "user", "content": SOURCE_01}]
    summary = _previous_goal_summary(_tool_message(_malformed_tasks_only(tasks)), original)
    recovered = json.loads(summary)["tasks"]
    assert [item["text"] for item in recovered] == [TASK_01]


def test_recovered_task_context_keeps_negated_and_third_party_source_text():
    source = "我朋友说他没有练过；我本人也没有胸痛。请说明新手怎么开始。"
    tasks = [
        _task("chitchat", "我朋友说他没有练过", is_third_party=True),
        _task("training_qa", "我本人也没有胸痛", is_negated=True),
        _task("training_qa", "请说明新手怎么开始"),
    ]
    summary = _previous_goal_summary(
        _tool_message(_malformed_tasks_only(tasks)),
        [{"role": "user", "content": source}])
    recovered = json.loads(summary)["tasks"]
    assert [item["text"] for item in recovered] == [
        "我朋友说他没有练过", "我本人也没有胸痛", "请说明新手怎么开始"]



def test_valid_structured_result_is_not_repaired_or_called_twice():
    class ValidClient:
        def __init__(self):
            self.calls = 0

        async def complete_message(self, messages, **kwargs):
            self.calls += 1
            return _tool_message(json.dumps(_understanding(
                [_task("training_qa", TASK_01B)]), ensure_ascii=False))

    client = ValidClient()
    value = asyncio.run(complete_structured(
        client,
        [{"role": "user", "content": json.dumps({"query": SOURCE_01}, ensure_ascii=False)}],
        RequestUnderstanding, name="return_request_understanding",
    ))
    assert client.calls == 1
    assert [item.text for item in value.tasks] == [TASK_01B]


# ---------------------------------------------------------------------------
# SIM28-02: write goals get the same copyable frozen source_ref as read goals
# ---------------------------------------------------------------------------

SOURCE_02 = "请把我今天做的哑铃划船写进日志：24公斤，3组，每组12次。"


def _prepare(candidates):
    token = request_understanding.set({})
    try:
        remember_safety(SOURCE_02, {"level": "normal", "evidence": "",
                                    "categories": [], "reason": "scripted normal"})
        return prepare_node({"raw_message": SOURCE_02, "intent_candidates": candidates},
                            ReactDependencies(llm=object(), user_id="synthetic"))
    finally:
        request_understanding.reset(token)


def test_authorized_write_goal_exposes_exact_source_ref_navigation_hint():
    candidate = {"id": "w1", "task_type": "workout_record", "action": "record",
                 "text": SOURCE_02, "source_start": 0, "source_end": len(SOURCE_02),
                 "side_effect": True, "write_authorization": "authorized"}
    system = _prepare([candidate])["messages"][0]["content"]
    assert "Application-validated write goals" in system
    assert '"goal_id": "w1"' in system
    assert '"start": 0' in system
    assert f'"end": {len(SOURCE_02)}' in system
    assert SOURCE_02 in system


def test_mixed_read_and_write_goals_receive_separate_navigation_sections():
    source = "先说一个训练恢复原则。请把今天的哑铃划船写进日志：24公斤，3组，每组12次。"
    read_text = "先说一个训练恢复原则。"
    write_text = "请把今天的哑铃划船写进日志：24公斤，3组，每组12次。"
    read = {"id": "r1", "task_type": "training_qa", "action": "question",
            "text": read_text, "source_start": 0, "source_end": len(read_text),
            "side_effect": False}
    write = {"id": "w1", "task_type": "workout_record", "action": "record",
             "text": write_text, "source_start": source.index(write_text),
             "source_end": source.index(write_text) + len(write_text),
             "side_effect": True, "write_authorization": "authorized"}
    token = request_understanding.set({})
    try:
        remember_safety(source, {"level": "normal", "evidence": "",
                                 "categories": [], "reason": "scripted normal"})
        system = prepare_node({"raw_message": source, "intent_candidates": [read, write]},
                              ReactDependencies(llm=object(), user_id="synthetic"))["messages"][0]["content"]
    finally:
        request_understanding.reset(token)
    assert "Application-validated read-only goals" in system
    assert "Application-validated write goals" in system
    assert '"goal_id": "r1"' in system and '"goal_id": "w1"' in system
    assert read_text in system and write_text in system



def test_rejected_write_goal_is_not_given_a_dispatch_hint():
    candidate = {"id": "w1", "task_type": "workout_record", "action": "record",
                 "text": SOURCE_02, "source_start": 0, "source_end": len(SOURCE_02),
                 "side_effect": False, "write_authorization": "rejected_unauthorized"}
    system = _prepare([candidate])["messages"][0]["content"]
    assert "Application-validated write goals" not in system
    assert '"goal_id": "w1"' not in system


# ---------------------------------------------------------------------------
# SIM28-04: one shared leader is not a complete multi-result plan contract
# ---------------------------------------------------------------------------

SOURCE_04 = ("我是成年增肌新手，没有伤病。今天想比较两份独立计划：一份只用哑铃，"
             "另一份只用弹力带。每份从热身到放松合计最多20分钟，休息转换也包含在内。"
             "请分别给出，两份都不保存。")


def test_single_shared_leader_without_member_is_rejected():
    raw = _understanding([_task(
        "training_plan", SOURCE_04, plan_result=True, result_scope="shared_leader",
        plan_goal_id="dumbbell-vs-band",
        shared_scope_quote="今天想比较两份独立计划")])
    with pytest.raises(SemanticValidationError) as caught:
        validate_understanding(SOURCE_04, raw)
    assert caught.value.code == "semantic_plan_scope_incomplete"


def test_shared_leader_and_member_keep_two_results():
    split = SOURCE_04.index("另一份只用弹力带")
    leader_text, member_text = SOURCE_04[:split], SOURCE_04[split:]
    raw = _understanding([
        _task("training_plan", leader_text, plan_result=True,
              result_scope="shared_leader", plan_goal_id="dumbbell-vs-band",
              shared_scope_quote="今天想比较两份独立计划"),
        _task("training_plan", member_text, plan_result=True,
              result_scope="shared_member", plan_goal_id="dumbbell-vs-band"),
    ])
    value = validate_understanding(SOURCE_04, raw)
    assert [task["task_type"] for task in value["validated_tasks"]] == [
        "training_plan", "training_plan"]
    assert [task["result_scope"] for task in value["validated_tasks"]] == [
        "shared_leader", "shared_member"]
    assert {task["plan_goal_id"] for task in value["validated_tasks"]} == {"dumbbell-vs-band"}


def test_one_weekly_plan_with_multiple_sessions_is_still_one_task():
    source = "我是新手，只有哑铃，每周两练，每次20分钟，请给我一份周计划。"
    raw = _understanding([_task(
        "training_plan", source, plan_result=True, result_scope="independent",
        plan_request_quote="请给我一份周计划")])
    value = validate_understanding(source, raw)
    assert [task["task_type"] for task in value["validated_tasks"]] == ["training_plan"]


# ---------------------------------------------------------------------------
# SIM28-05: quoted negation is not a current execution-status read
# ---------------------------------------------------------------------------

SOURCE_05 = "请转述这句否定说明：‘我没有删除任何记录，所以不会声称已经删除成功。’这次只是转述文字，不执行数据操作。"


def test_quoted_negation_is_normalized_out_of_execution_status_copy():
    quoted = _task("chitchat", SOURCE_05, action="answer",
                   is_negated=True, reply_mode="conversation",
                   reply_purpose="execution_status", goal_role="result", quoted=True)
    value = validate_understanding(SOURCE_05, _understanding([quoted]))
    assert value["reply_purpose"] == "general"
    assert value["tasks"][0]["reply_purpose"] == "general"
    assert dialogue.status_only_request(value) is False


def test_real_execution_status_question_keeps_bounded_status_path():
    source = "这次会不会保存我的训练？"
    raw = _understanding([_task(
        "chitchat", source, action="question", reply_mode="conversation",
        reply_purpose="execution_status", goal_role="result", quoted=False)],
        reply_purpose="execution_status")
    value = validate_understanding(source, raw)
    assert dialogue.status_only_request(value) is True
    assert dialogue.execution_status_text("", "save_permission") == "不会保存。"


def test_transcription_path_returns_model_answer_instead_of_fixed_delete_copy():
    class AnswerClient:
        def configured(self):
            return True

        async def complete_structured(self, messages, schema, **kwargs):
            return schema.model_validate({
                "answer": "转述如下：我没有删除任何记录，所以不会声称已经删除成功。",
                "requires_grounding": False,
                "status_kind": "delete_or_clear",
            })

    request = type("Request", (), {
        "message": SOURCE_05, "request_id": "sim28-quoted",
        "conversation_id": "sim28-quoted", "user_id": "synthetic",
    })()
    understanding = {
        "reply_mode": "conversation", "reply_purpose": "general",
        "tasks": [{"task_type": "chitchat", "action": "answer",
                   "reply_mode": "conversation", "reply_purpose": "general",
                   "goal_role": "result", "quoted": True,
                   "context_ids": [], "fact_fields": []}],
    }
    result = asyncio.run(dialogue.conversation_answer(
        AnswerClient(), request, {"long_memory": {}, "short_memory": []}, understanding))
    assert result is not None
    assert result["answer"].startswith("转述如下")
    assert "我不能删除或清空" not in result["answer"]


# ---------------------------------------------------------------------------
# SIM28-06: conditions/background do not become outstanding deliverables
# ---------------------------------------------------------------------------

def _candidate(cid, task_type, text, *, role="result", side_effect=False, **extra):
    return {"id": cid, "task_type": task_type, "action": extra.pop("action", "question"),
            "text": text, "side_effect": side_effect, "goal_role": role, **extra}


def test_constraint_and_background_are_not_chased_after_real_goal_completes():
    state = {"intent_candidates": [
        _candidate("q1", "training_qa", "一般成年人每周应安排多少有氧运动？"),
        _candidate("c1", "chitchat", "这次不登记任何记录", role="constraint",
                   action="answer", is_negated=True),
        _candidate("b1", "chitchat", "谢谢", role="background", action="answer"),
    ], "covered_goals": ["q1"]}
    assert _unhandled_goals(state) == []
    from app.services.agent.react_graph import completion_check
    decision = completion_check(state, ReactDependencies(llm=object(), user_id="synthetic"))
    assert decision["terminated"] is True
    assert decision["termination_reason"] == "goals_complete"
    assert decision["status"] == "complete"



def test_unanswered_result_and_negated_question_remain_outstanding():
    state = {"intent_candidates": [
        _candidate("q1", "training_qa", "一般成年人每周应安排多少有氧运动？"),
        _candidate("q2", "training_qa", "我没有删除记录，对吗？", is_negated=True),
        _candidate("c1", "chitchat", "这次不登记任何记录", role="constraint",
                   action="answer", is_negated=True),
    ], "covered_goals": ["q1"]}
    outstanding = _unhandled_goals(state)
    assert [item["id"] for item in outstanding] == ["q2"]


def test_constraint_label_cannot_bypass_a_real_write_side_effect():
    state = {"intent_candidates": [
        _candidate("w1", "workout_record", SOURCE_02, role="constraint",
                   side_effect=True, action="record"),
    ], "covered_goals": []}
    assert [item["id"] for item in _unhandled_goals(state)] == ["w1"]


# ---------------------------------------------------------------------------
# SIM28-03: a trusted pre-dispatch rejection is a frozen terminal and replays
# ---------------------------------------------------------------------------

@pytest.fixture
def sim28_db(request, tmp_path, monkeypatch):
    from tests.react_phase_c1_support import isolated
    from tests.react_phase_c2_2_support import ensure_catalog

    backend = getattr(request, "param", "sqlite")
    with isolated(monkeypatch, tmp_path / ("sim28-db-" + backend), backend) as engine:
        ensure_catalog(engine)
        yield engine


@pytest.mark.parametrize("sim28_db", ["sqlite", "postgres"], indirect=True)
def test_rejected_before_dispatch_freezes_identity_and_replays_zero_write(sim28_db, monkeypatch):
    from tests.react_phase_c2_3_support import (
        C3HTTPHarness,
        ledger_snapshot,
        mock_provider_transport,
        workout_rows,
    )

    source = SOURCE_02
    understanding = _understanding([
        _task("workout_record", source, action="record",
              fact_text="24公斤，3组，每组12次")])
    semantic_response = {"choices": [{"message": {
        "role": "assistant", "content": "", "tool_calls": [{
            "id": "sim28-semantic", "type": "function", "function": {
                "name": "return_request_understanding",
                "arguments": json.dumps(understanding, ensure_ascii=False)}}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
    # Exact saved bad ref from call 112: it drops the leading command text.
    bad_ref = {"start": 3, "end": 29,
               "exact_text": "今天做的哑铃划船写进日志：24公斤，3组，每组12次。"}
    supervisor_response = {"choices": [{"message": {
        "role": "assistant", "content": "", "tool_calls": [{
            "id": "sim28-write", "type": "function", "function": {
                "name": "record_workout",
                "arguments": json.dumps({"source_ref": bad_ref}, ensure_ascii=False)}}]}}],
        "usage": {"prompt_tokens": 6, "completion_tokens": 4}}

    def extra(request):
        raise AssertionError("unexpected provider call")

    seen = mock_provider_transport(monkeypatch, [semantic_response, supervisor_response, extra])
    harness = C3HTTPHarness(monkeypatch, sim28_db)
    rid = "sim28-native-rejected"
    status, first = harness.post_json(source, request_id=rid)
    assert status == 200
    assert len(seen) == 2
    assert first["metrics"]["react_write"] == "limited"
    assert first["answer"] == "部分内容因安全或授权原因未处理。"
    first_ids = (first["request_id"], first["message_id"], first["run_id"])
    snapshot = ledger_snapshot(sim28_db, rid)
    assert snapshot["request"]["status"] == "limited"
    assert snapshot["operations"] == []
    journals = [event for event in snapshot["events"]
                if str(event.get("phase")) == "final_reply"]
    assert len(journals) == 1
    journal = journals[0]["observation"]
    if isinstance(journal, str):
        journal = json.loads(journal)
    assert journal["status"] == "limited"
    assert journal["terminal_kind"] == "rejected_before_dispatch"
    assert journal["response_hash"]
    assert not snapshot["events"] or all(
        not str(event.get("event_key", "")).startswith(
            ("c2-1-claim:v1:", "c2-2-claim:v1:", "c2-1-commit:v1:", "c2-2-commit:v1:"))
        for event in snapshot["events"])
    before_rows = workout_rows(sim28_db, "demo-user")

    status_replay, replay = harness.post_json(source, request_id=rid)
    assert status_replay == 200
    assert len(seen) == 2
    assert replay["replayed"] is True
    assert replay["answer"] == first["answer"]
    assert (replay["request_id"], replay["message_id"], replay["run_id"]) == first_ids
    assert workout_rows(sim28_db, "demo-user") == before_rows
    assert ledger_snapshot(sim28_db, rid)["operations"] == []


def test_unknown_or_claimed_terminal_is_not_replayed_as_pre_dispatch(sim28_db, monkeypatch):
    from tests.react_phase_c2_3_support import ledger_snapshot
    from sqlalchemy.orm import sessionmaker
    from app.services.agent import execution_ledger_store as ledger_api
    from app.services.agent.react_write_runtime import WriteRuntimeSession

    source = "请把今天的训练存下来。"
    factory = sessionmaker(bind=sim28_db, expire_on_commit=False)
    store = ledger_api.ExecutionLedgerStore(factory)
    identity = ledger_api.RequestIdentity(
        user_id="demo-user", request_id="sim28-unknown",
        conversation_id="sim28-unknown", source=source,
        source_digest=__import__("hashlib").sha256(source.encode("utf-8")).hexdigest(),
        requested_mode="react_write", effective_mode="react_write",
        schema_version=2, tool_version="react-tools-v2",
        time_base="2026-10-28T00:00:00+00:00",
        budget_scope_id="scope:sim28-unknown", request_kind="tasks")
    store.ensure_request(identity)
    request = type("Request", (), {
        "user_id": "demo-user", "request_id": "sim28-unknown",
        "conversation_id": "sim28-unknown", "message": source,
    })()
    runtime = WriteRuntimeSession(object(), request)
    runtime.store, runtime.factory = store, factory
    # No write fact is present: this is not the pre-dispatch rejection proof.
    assert runtime._rejected_before_dispatch({"call_facts": []}) is False
    # A dispatched write cannot use the shortcut either.
    assert runtime._rejected_before_dispatch({"call_facts": [{
        "permission": "write", "dispatched": True,
        "gate_reason": "invalid_source_ref"}]}) is False
    # Missing/None dispatch is not an explicit False proof.
    assert runtime._rejected_before_dispatch({"call_facts": [{
        "permission": "write", "gate_reason": "invalid_source_ref"}]}) is False
    assert runtime._rejected_before_dispatch({"call_facts": [{
        "permission": "write", "dispatched": None,
        "gate_reason": "invalid_source_ref"}]}) is False
    # Every write must carry its own valid gate reason; one bare/missing write
    # is an incomplete proof even when a sibling has a real gate reason.
    assert runtime._rejected_before_dispatch({"call_facts": [
        {"permission": "write", "dispatched": False,
         "gate_reason": "invalid_source_ref"},
        {"permission": "write", "dispatched": False},
    ]}) is False
    assert runtime._rejected_before_dispatch({"call_facts": [
        {"permission": "write", "dispatched": False,
         "gate_reason": "invalid_source_ref"},
        {"permission": "write", "gate_reason": "invalid_source_ref"},
    ]}) is False
    # Reused-result sentinels are not fresh gate proofs.
    assert runtime._rejected_before_dispatch({"call_facts": [{
        "permission": "write", "dispatched": False,
        "gate_reason": "restored_result_reused"}]}) is False
    # Complete positive proof: every write explicitly not dispatched and each
    # has its own application gate reason.
    assert runtime._rejected_before_dispatch({"call_facts": [
        {"permission": "write", "dispatched": False,
         "gate_reason": "invalid_source_ref"},
        {"permission": "write", "dispatched": False,
         "gate_reason": "no_matching_goal"},
    ]}) is True
    ledger = ledger_snapshot(sim28_db, "sim28-unknown")
    assert ledger["request"]["status"] in {"pending", "running"}


# ---------------------------------------------------------------------------
# SIM28-04b: plan worker generation consumes only its current target contract
# ---------------------------------------------------------------------------

PLAN_PARENT = (
    "我是成年增肌新手，没有伤病。今天想比较两份独立计划：一份只用哑铃，"
    "另一份只用弹力带。每份从热身到放松合计最多20分钟，休息转换也包含在内。"
    "请分别给出，两份都不保存。"
)
PLAN_LEADER = "我是成年增肌新手，没有伤病。今天想比较两份独立计划：一份只用哑铃"
PLAN_MEMBER = "另一份只用弹力带。每份从热身到放松合计最多20分钟，休息转换也包含在内。请分别给出，两份都不保存。"


def _plan_worker_state(task_id, text, *, scope, sibling_text=None, expected_sessions=1,
                       equipment_quote="一份只用哑铃"):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    state = {
        "task_id": task_id,
        "task_text": text,
        "raw_message": PLAN_PARENT,
        "task_payload": {},
        "plan_contract": {
            "plan_result": True,
            "result_scope": scope,
            "plan_goal_id": "compare-two",
            "plan_request_quote": "请分别给出",
            "shared_scope_quote": "今天想比较两份独立计划",
        },
        "task_context": ([{"task_id": "task-old", "task_type": "training_plan",
                            "task_text": sibling_text, "answer": "兄弟方案"}]
                         if sibling_text else []),
        "profile": {}, "short_memory": [], "long_memory": {},
        "plan_target_count": 2,
    }
    target = FitnessMultiAgentGraph._plan_worker_state(state)
    target["task_context"].append({"planning_information": {
        "scope": "session", "expected_sessions": expected_sessions,
        "facts": [
            {"field": "goal", "quote": "成年增肌新手"},
            {"field": "experience", "quote": "没有伤病"},
            {"field": "equipment", "quote": equipment_quote},
            {"field": "session_time", "quote": "每份从热身到放松合计最多20分钟",
             "quantity_quote": "20分钟", "seconds": 1200},
        ],
    }})
    return target


def test_member_plan_prompt_excludes_parent_and_sibling_generation_target():
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    state = _plan_worker_state("task-2", PLAN_MEMBER, scope="shared_member",
                               sibling_text=PLAN_LEADER,
                               equipment_quote="另一份只用弹力带")
    messages = FitnessMultiAgentGraph()._planning_prompt(
        "训练方案", state, [], timed_plan=True,
        plan_target_only=bool(state.get("plan_target_only")))
    text = "\n".join(str(m.get("content") or "") for m in messages)
    assert PLAN_PARENT not in text
    assert PLAN_LEADER not in text
    assert "task-old" not in text
    assert PLAN_MEMBER in text
    assert '"result_scope": "shared_member"' in text
    assert '"expected_sessions": 1' in text
    assert "current_plan_target" in text
    assert "父请求中其他已处理的方案或问题不是本任务指令" in text


def test_independent_same_type_plan_keeps_its_own_target_and_shared_conditions():
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    state = _plan_worker_state("task-2", "另一份只用弹力带", scope="independent",
                               sibling_text="一份只用哑铃",
                               equipment_quote="另一份只用弹力带")
    messages = FitnessMultiAgentGraph()._planning_prompt(
        "训练方案", state, [], timed_plan=True,
        plan_target_only=bool(state.get("plan_target_only")))
    text = "\n".join(str(m.get("content") or "") for m in messages)
    assert '"result_scope": "independent"' in text
    assert "另一份只用弹力带" in text
    assert "一份只用哑铃" not in text
    assert "兄弟方案" not in text
    assert "每份从热身到放松合计最多20分钟" in text
    assert '"expected_sessions": 1' in text


def test_weekly_plan_prompt_keeps_multiple_sessions_for_one_target():
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    state = _plan_worker_state("task-1", "请给我一份每周两练的哑铃周计划",
                               scope="independent", expected_sessions=2)
    messages = FitnessMultiAgentGraph()._planning_prompt(
        "训练方案", state, [], timed_plan=True,
        plan_target_only=bool(state.get("plan_target_only")))
    text = "\n".join(str(m.get("content") or "") for m in messages)
    assert '"expected_sessions": 2' in text
    assert "session 数量必须等于" in text


def test_weekly_plan_renderer_accepts_two_sessions_and_rejects_two_for_one_target():
    from app.services.agent.training_timing import render_timed_plan

    def session(label):
        return {
            "label": label,
            "warmup": {"seconds": 180, "activity": "关节活动"},
            "main": {"seconds": 780, "exercises": [
                {"name": "哑铃深蹲", "sets": 3, "reps": 10, "hold_seconds": None},
                {"name": "哑铃划船", "sets": 3, "reps": 10, "hold_seconds": None},
            ]},
            "breaks": {"seconds": 180},
            "cooldown": {"seconds": 60, "activity": "放松"},
            "time_consistency": "consistent",
        }

    raw = json.dumps({"sessions": [session("第一次"), session("第二次")]}, ensure_ascii=False)
    rendered = render_timed_plan(raw, limit_seconds=1200, expected_sessions=2)
    assert "第一次" in rendered and "第二次" in rendered
    with pytest.raises(ValueError, match="session_count_mismatch"):
        render_timed_plan(raw, limit_seconds=1200, expected_sessions=1)


def test_real_plan_worker_uses_member_target_not_parent_or_sibling(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph

    graph = FitnessMultiAgentGraph()
    information = {
        "scope": "session",
        "known": {"goal": "增肌", "experience": "新手",
                  "equipment": "另一份只用弹力带",
                  "session_time": "每份从热身到放松合计最多20分钟"},
        "session_time_seconds": 1200,
        "available_days_count": None,
        "phase_limits": [],
    }
    monkeypatch.setattr("app.services.agent.task_information.check_plan_information",
                        AsyncMock(return_value=(information, None)))
    monkeypatch.setattr(graph, "_readonly_retrieval", AsyncMock(return_value=[]))
    monkeypatch.setattr("app.services.agent.multi_agent_graph.assess_rag_evidence",
                        lambda *args, **kwargs: SimpleNamespace(
                            answerable=True, as_dict=lambda: {"answerable": True}))
    one_session = {
        "sessions": [{
            "label": "弹力带单次方案",
            "warmup": {"seconds": 180, "activity": "关节活动"},
            "main": {"seconds": 780, "exercises": [
                {"name": "弹力带深蹲", "sets": 3, "reps": 12, "hold_seconds": None},
                {"name": "弹力带划船", "sets": 3, "reps": 12, "hold_seconds": None},
            ]},
            "breaks": {"seconds": 180},
            "cooldown": {"seconds": 60, "activity": "放松"},
            "time_consistency": "consistent",
        }]
    }
    seen = []

    async def complete_message(messages, **kwargs):
        seen.append(messages)
        return {"role": "assistant", "content": "", "tool_calls": [{
            "type": "function", "function": {"name": "return_timed_training_plan",
                                             "arguments": json.dumps(one_session, ensure_ascii=False)}}]}

    monkeypatch.setattr(graph.llm, "complete_message", complete_message)
    state = _plan_worker_state("task-2", PLAN_MEMBER, scope="shared_member",
                               sibling_text=PLAN_LEADER,
                               equipment_quote="另一份只用弹力带")
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]
    assert result["status"] == "complete"
    assert "弹力带单次方案" in result["answer"]
    sent = "\n".join(str(m.get("content") or "") for m in seen[0])
    assert PLAN_PARENT not in sent
    assert PLAN_LEADER not in sent
    assert "task-old" not in sent
    assert "另一份只用弹力带" in sent
    assert '"expected_sessions": 1' in sent




