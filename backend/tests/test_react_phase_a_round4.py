"""ReAct 阶段A第四轮复验：R4-01…R4-06 有限边界补测（round4 任务 §4）。

绑定 DeepSeek 211634 实现。只覆盖第四轮合同现有不变量；每组设合法对照；
失败按原 RA 归属登记。标注「协议控制」的合成候选直构 state 用例均实际注入
被测节点（execute_react_batch），并另核真实来源校验分支，不经 prepare 重生成。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.services.agent.react_graph import (
    ReactDependencies,
    execute_react_batch,
    finalize_node,
    invoke_react,
)
from app.services.agent.react_models import DomainObservation, ToolStatus
from app.services.agent.react_prompts import RECOVERY_OBSERVATION_NOTICE

from tests.react_phase_a_support import (
    RecordingExecutor,
    SemanticModelMixin,
    assistant,
    batch_state,
    committed_receipt,
    ref_of,
    tool_call,
)

SRC_RECORD = "帮我记录午餐吃了牛肉面和可乐"
SRC_DEPS = "早餐吃了燕麦粥,帮我记录。然后根据早餐给我营养建议"
SRC_TWO_CLAUSE = "帮我记录早餐吃了燕麦,午餐吃了米饭"
SRC_NO_VERB = "早上吃了燕麦粥,中午吃了米饭"
SRC_TWO_RECORDS = "早餐吃了燕麦粥,帮我记录。晚餐吃了鸡胸肉,也帮我记录"


def _deps(llm, executor=None, **options):
    return ReactDependencies(llm=llm, tool_executor=executor, user_id="app-user-1",
                             conversation_id="conv-1", request_id="req-4", **options)


def _run(deps, message):
    return asyncio.run(invoke_react(deps, message))


class FakeLLMPolicy(SemanticModelMixin):
    def __init__(self, policy, semantic=True):
        self.policy = policy
        self.semantic = semantic
        self.calls = []
        self.responses = []
        self.semantic_calls = []

    async def complete_message(self, messages, temperature=0.0, tools=None, tool_choice=None):
        self.calls.append([dict(m) for m in messages])
        response = dict(self.policy(len(self.calls), messages))
        self.responses.append(response)
        return response


# ---------------------------------------------------------------------------
# R4-01 / RA-07：候选位置重定位与充分对应
# ---------------------------------------------------------------------------
def _two_same_text_candidates(first_span):
    return [
        {"id": "goal-a", "task_type": "workout_record", "action": "record",
         "text": "跑步5公里", "source_start": first_span[0], "source_end": first_span[1],
         "depends_on": []},
        {"id": "goal-b", "task_type": "workout_record", "action": "record",
         "text": "跑步5公里", "depends_on": []},
    ]


SRC_MIXED = "早上跑步5公里,晚上也跑步5公里"


def test_r4_01_minus_one_minus_one_relocates_and_binds_second_position():
    """协议控制：合法整数 (-1,-1) 未定位标记 → 重新定位第二出现位置，精确绑定 goal-b；
    无写授权来源下重定位目标经澄清覆盖但不保存，重复调用零新增派发。"""
    state = batch_state(SRC_MIXED, assistant(tool_calls=[
        tool_call("t1", "record_workout",
                  {"source_ref": ref_of(SRC_MIXED, "跑步5公里", 1)})]),
        candidates=_two_same_text_candidates((-1, -1)))
    executor = RecordingExecutor({"record_workout": committed_receipt("rcpt-fixed")})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert result["covered_goals"] == ["goal-b"], (
        f"R4-01: (-1,-1)应重定位且第二次出现调用精确绑定goal-b: {result['covered_goals']}")
    assert "goal-a" not in result["covered_goals"], f"R4-01: goal-a被吞并: {result['covered_goals']}"
    # 重复调用零新增派发
    dispatches_after_first = len(executor.calls)
    state.update(result)
    state["messages"].append(assistant(tool_calls=[
        tool_call("t2", "record_workout",
                  {"source_ref": ref_of(SRC_MIXED, "跑步5公里", 1)})]))
    asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert len(executor.calls) == dispatches_after_first, (
        f"R4-01: 已覆盖目标重复调用应零新增派发: {executor.names}")


def test_r4_01_control_unrelocated_second_needs_authorization_and_saves_nothing():
    """对照（2026-10-18 语义合同迁移）：重定位到真实区间不等于写授权。

    ``(-1,-1)`` 仍会重定位，但应用候选若没有模型写决定（``needs_confirmation`` /
    ``semantic_intent_unavailable``），写调用必须澄清、零派发、零提交；不能用原文
    动词重新授权，也不能因重定位成功就保存。
    """
    state = batch_state(SRC_NO_VERB, assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": ref_of(SRC_NO_VERB, "早上吃了燕麦粥", 0)})]),
        candidates=[{"id": "goal-a", "task_type": "nutrition_record", "action": "record",
                     "text": "早上吃了燕麦粥", "source_start": -1, "source_end": -1,
                     "depends_on": [], "write_authorization": "needs_confirmation",
                     "write_rejection_reason": "semantic_intent_unavailable"}])
    executor = RecordingExecutor({"record_nutrition": committed_receipt()})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert executor.calls == [], f"R4-01对照: 无写授权不得保存: {executor.names}"
    assert result["committed_tasks"] == []
    observation = result["observations"][0]
    assert observation["status"] == "needs_clarification" and \
        observation.get("error") == "semantic_intent_unavailable", (
            f"R4-01对照: 无模型写决定应澄清而非保存: {observation}")


@pytest.mark.parametrize("bad_span", [(None, 3), (2, None), (-1.0, -1.0)],
                         ids=["none-start", "none-end", "float-neg"])
def test_r4_01_loose_none_or_float_spans_must_not_get_position(bad_span):
    """T7校准（round5裁定）：夹具只含被测坏位置 goal-a（移除合法兄弟 goal-b）。

    合同：单边缺失/None、非严格整数 (-1.0,-1.0) 不回退 find、不得得到合法位置。
    断言：整体零覆盖、零提交、goal-a 无有效绑定、执行器零派发。原整体空断言
    会误伤合法兄弟候选（其按未消费位置获得合法位置属正确行为，见
    test_r4_01_minus_one_minus_one_relocates 与 round2 mixed 重定位对照）。
    原样首跑证据：round5 asoriginal 日志（covered=['goal-b']，兄弟被正确定位）。"""
    state = batch_state(SRC_MIXED, assistant(tool_calls=[
        tool_call("t1", "record_workout",
                  {"source_ref": ref_of(SRC_MIXED, "跑步5公里", 0)})]),
        candidates=[{"id": "goal-a", "task_type": "workout_record", "action": "record",
                     "text": "跑步5公里", "source_start": bad_span[0], "source_end": bad_span[1],
                     "depends_on": []}])
    executor = RecordingExecutor({"record_workout": committed_receipt("rcpt-fixed")})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert result["covered_goals"] == [], (
        f"R4-01/T7: 坏显式区间 {bad_span} 得到合法位置并被覆盖: {result['covered_goals']}")
    assert result["committed_tasks"] == [], f"T7: 坏位置不得提交: {result['committed_tasks']}"
    assert executor.calls == [], (
        f"T7: 坏位置候选无有效绑定, 执行器应零派发: {executor.names}, "
        f"observations={[(o['status'], o.get('error')) for o in result['observations']]}")
    assert result["bound_tasks"] == [], f"T7: 坏位置不得创建应用绑定: {result['bound_tasks']}"


@pytest.mark.parametrize("ref_end,covered_ratio", [(15, "4/6"), (16, "5/6")],
                         ids=["overlap-4of6", "overlap-5of6"])
def test_r4_01_more_than_half_overlap_still_not_full_second_goal(ref_end, covered_ratio):
    """协议控制：ref 覆盖 goal-a 全部、goal-b 仅 4/6 或 5/6 且缺完整食物事实
    → goal-b 不得 covered/committed（检查“大于一半”仍误认，50%阈值非语义合同）。"""
    candidates = [
        {"id": "goal-a", "task_type": "nutrition_record", "action": "record",
         "text": "早餐吃了燕麦", "source_start": 4, "source_end": 10, "depends_on": []},
        {"id": "goal-b", "task_type": "nutrition_record", "action": "record",
         "text": "午餐吃了米饭", "source_start": 11, "source_end": 17, "depends_on": []},
    ]
    ref = {"start": 4, "end": ref_end, "exact_text": SRC_TWO_CLAUSE[4:ref_end]}
    state = batch_state(SRC_TWO_CLAUSE, assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": ref})]),
        candidates=candidates)
    executor = RecordingExecutor({"record_nutrition": committed_receipt("rcpt-fixed")})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert result["covered_goals"] == ["goal-a"], (
        f"R4-01: goal-b 仅{covered_ratio}重叠仍被覆盖: {result['covered_goals']}, "
        f"committed={result['committed_tasks']}, executor={executor.names}")


def test_r4_01_control_two_full_clauses_both_committed():
    """对照：两个 ref 分别完整覆盖两目标 → 各自绑定、两次可信提交。"""
    candidates = [
        {"id": "goal-a", "task_type": "nutrition_record", "action": "record",
         "text": "早餐吃了燕麦", "source_start": 4, "source_end": 10, "depends_on": []},
        {"id": "goal-b", "task_type": "nutrition_record", "action": "record",
         "text": "午餐吃了米饭", "source_start": 11, "source_end": 17, "depends_on": []},
    ]
    state = batch_state(SRC_TWO_CLAUSE, assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": {"start": 4, "end": 10, "exact_text": "早餐吃了燕麦"}}),
        tool_call("t2", "record_nutrition",
                  {"source_ref": {"start": 11, "end": 17, "exact_text": "午餐吃了米饭"}})]),
        candidates=candidates)
    executor = RecordingExecutor({"record_nutrition": committed_receipt("rcpt-fixed")})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert result["committed_tasks"] == ["goal-a", "goal-b"], (
        f"R4-01对照: 完整分句应各自提交: {result['committed_tasks']}")


# ---------------------------------------------------------------------------
# R4-02 / RA-08：返回边界与可变引用（经真实执行器、有绑定状态）
# ---------------------------------------------------------------------------
_BOUND_CANDIDATES = [{
    "id": "task-1", "task_type": "nutrition_record", "action": "record",
    "text": SRC_RECORD, "source_start": 0, "source_end": len(SRC_RECORD), "depends_on": [],
}]


def test_r4_02_mixed_receipt_via_executor_with_binding_no_commit_finite_json():
    """混合坏回执经真实执行器返回（有绑定）：整份失效、零提交、有限JSON、前项保留。"""
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "read_profile", {}),
        tool_call("t2", "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}})]),
        candidates=_BOUND_CANDIDATES)
    executor = RecordingExecutor({
        "read_profile": {"status": "completed", "result": {"profile": "ok"}},
        "record_nutrition": {"status": "completed", "commit_state": "committed",
                             "receipt_refs": ["r-ok", None]},
    })
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert result["committed_tasks"] == [], (
        f"R4-02: 混合回执经执行器返回仍构成提交: {result['committed_tasks']}")
    assert len(result["observations"]) == 2 and result["observations"][0]["status"] == "completed", (
        f"R4-02: 批内前项应保留: {result['observations']}")
    for message in result["messages"]:
        if message.get("role") == "tool":
            json.loads(message["content"], parse_constant=lambda v: (_ for _ in ()).throw(
                ValueError(f"非有限JSON: {v}")))


def test_r4_02_shared_result_mutation_no_backpropagation():
    """dict/direct 共享嵌套 result 在**首次返回之后**被执行器突变：既有观察/
    goal_results/消息/后续context不被反向修改（以真实状态输出为主证据）。"""
    shared = {"profile": {"note": "clean"}}
    observation = DomainObservation(status=ToolStatus.COMPLETED, result=shared)
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "read_profile", {}),
        tool_call("t2", "read_profile", {})]))
    executor = RecordingExecutor()

    def executor_result(nth, name, arguments, context):
        if nth >= 1:
            shared["profile"]["note"] = "mutated-later"  # 首次返回之后的突变
        return observation if nth == 0 else {"status": "completed", "result": {"n": nth}}

    executor.results["read_profile"] = executor_result
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    first_stored = result["observations"][0]
    assert "mutated" not in json.dumps(first_stored, ensure_ascii=False), (
        f"R4-02: 共享result突变反向污染已存储观察: {first_stored}")
    assert "mutated" not in result["messages"][3]["content"], (
        "R4-02: 已序列化tool消息被反向污染")


def test_r4_02_mutated_non_result_metadata_must_converge_not_crash():
    """协议控制：构造后突变非result元数据（summary=NaN、missing_fields含set）
    → 整份观察协议不批崩溃、不污染JSON、受影响项不伪成功、独立前项保留。"""
    clean = DomainObservation(status=ToolStatus.COMPLETED, result={"profile": "ok"})
    dirty = DomainObservation(status=ToolStatus.COMPLETED, result={"n": 1})
    dirty.summary = float("nan")  # 返回前突变
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "read_profile", {}),
        tool_call("t2", "read_profile", {})]))

    class TwoMetaExecutor:
        def __init__(self):
            self.calls = []
            self.names = []

        def __call__(self, name, arguments, context):
            self.calls.append({"name": name})
            self.names.append(name)
            return clean if len(self.calls) == 1 else dirty

    try:
        result = asyncio.run(execute_react_batch(state, _deps(
            FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), TwoMetaExecutor())))
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"R4-02: summary=NaN突变导致整批崩溃: {exc!r}")
    assert len(result["observations"]) == 2, f"R4-02: 前项应保留: {result['observations']}"
    for message in result["messages"]:
        if message.get("role") == "tool":
            json.loads(message["content"], parse_constant=lambda v: (_ for _ in ()).throw(
                ValueError(f"非有限JSON: {v}")))


def test_r4_02_mutated_missing_fields_with_set_must_converge_not_crash():
    dirty = DomainObservation(status=ToolStatus.COMPLETED, result={"n": 1})
    dirty.missing_fields = [{1, 2}]  # set 元素，返回前突变
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "read_profile", {})]))
    executor = RecordingExecutor()
    executor.results["read_profile"] = dirty
    try:
        result = asyncio.run(execute_react_batch(state, _deps(
            FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
        _ = result
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"R4-02: missing_fields含set突变导致整批崩溃: {exc!r}")


# ---------------------------------------------------------------------------
# R4-03 / RA-14：只读伪造身份的下游边界（完整图）
# ---------------------------------------------------------------------------
def test_r4_03_read_forged_task_spec_full_graph_not_write_unconfirmed():
    """完整图对照：有真实只读目标时，伪造写task_spec不得把纯读目标变成“写提交未知”。"""
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "answer_training", {})]}
    ) if n == 1 else {"role": "assistant", "content": "循序渐进即可。"})
    forged = {"id": "forged-write-id", "task_type": "nutrition_record", "worker": "nutrition",
              "action": "record", "text": "FORGED", "payload": {}, "side_effect": True,
              "status": "pending"}
    executor = RecordingExecutor({"answer_training": {
        "status": "completed", "result": {"task_spec": forged}}})
    result = _run(_deps(llm, executor), "怎么练卧推")
    assert result["bound_tasks"] == [], f"R4-03: 伪造绑定进入bound_tasks: {result['bound_tasks']}"
    assert result["committed_tasks"] == [], f"R4-03: 伪绑定进入提交集合: {result['committed_tasks']}"
    assert result["status"] == "complete", (
        f"R4-03: 纯读目标被伪task_spec变成写提交未知: {result['status']!r}, "
        f"answer={result['final_answer']!r}")


def test_r4_03_control_normal_read_full_graph_completes():
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "answer_training", {})]}
    ) if n == 1 else {"role": "assistant", "content": "循序渐进即可。"})
    executor = RecordingExecutor({"answer_training": {"status": "completed"}})
    result = _run(_deps(llm, executor), "怎么练卧推")
    assert result["status"] == "complete" and result["final_answer"] == "循序渐进即可。"


# ---------------------------------------------------------------------------
# R4-04 / RA-04：无绑定派发与当前终态
# ---------------------------------------------------------------------------
def test_r4_04_authorized_ref_without_matching_goal_zero_dispatch():
    """合法ref/写授权但没有有效匹配应用目标：新写执行器零派发，明确拒绝/澄清。"""
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}})]),
        candidates=[])  # 应用侧无任何目标绑定
    executor = RecordingExecutor({"record_nutrition": committed_receipt()})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert executor.calls == [], (
        f"R4-04: 无有效匹配目标的授权写仍派发执行器: {executor.names}, "
        f"observations={[(o['status'], o.get('error')) for o in result['observations']]}")


@pytest.mark.parametrize("commit_payload", [
    {"status": "completed", "commit_state": "unknown"},
    {"status": "completed", "commit_state": "committed", "receipt_refs": ["r1"]},
], ids=["unbound-unknown", "unbound-committed"])
def test_r4_04_unbound_write_observations_cannot_default_complete(commit_payload):
    """协议控制：应用明确分类为写、无有效目标绑定——unknown 与 committed+回执
    都不能靠空 goal_results 默认 complete。"""
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}})]),
        candidates=[])
    executor = RecordingExecutor({"record_nutrition": commit_payload})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "全部保存成功"}), executor)))
    state.update(result)
    final = finalize_node(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "全部保存成功"})))
    assert final["status"] != "complete", (
        f"R4-04: 无绑定写观察 {commit_payload.get('commit_state')} 靠空goal_results默认complete: "
        f"{final['status']!r}")


def test_r4_04_current_valid_result_overrides_history_status():
    """协议控制：有效目标当前可信完成，历史观察曾受限/未知 → 以当前有效结果裁终态。"""
    history = [{"status": "unavailable", "error": "earlier-attempt"}]
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}})]),
        candidates=_BOUND_CANDIDATES,
        covered_goals=["task-1"],
        goal_results={"task-1": committed_receipt("rcpt-current")},
        observations=history,
    )
    executor = RecordingExecutor()
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    state.update(result)
    final = finalize_node(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"})))
    assert final["status"] == "complete", (
        f"R4-04: 当前可信完成被历史受限观察拖累: {final['status']!r}")
    assert executor.calls == [], f"R4-04前置: 已覆盖目标应复用零派发: {executor.names}"


def test_r4_04_control_current_unknown_still_blocked():
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}})]),
        candidates=_BOUND_CANDIDATES,
        covered_goals=["task-1"],
        goal_results={"task-1": {"status": "completed", "commit_state": "unknown"}},
    )
    executor = RecordingExecutor()
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    state.update(result)
    final = finalize_node(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"})))
    assert final["status"] != "complete", (
        f"R4-04对照: 当前仍未知的写目标不得complete: {final['status']!r}")


# ---------------------------------------------------------------------------
# R4-05 / RA-15：恢复身份、预算与终态
# ---------------------------------------------------------------------------
def _cross_batch_recovery_deps():
    """跨批两个不同建议目标的独立恢复（合成 advice 候选带真实来源显式区间）：
    turn1 advise-c1（deps task-1 缺）→ turn2 record t2（提交 task-1 → 恢复 c1）
    → turn3 advise-c2（deps task-2 缺）→ turn4 record t4（提交 task-2 → 恢复 c2）。"""
    candidates = [
        {"id": "task-1", "task_type": "nutrition_record", "action": "record",
         "text": "早餐吃了燕麦粥", "source_start": 0, "source_end": 7, "depends_on": []},
        {"id": "task-2", "task_type": "nutrition_record", "action": "record",
         "text": "晚餐吃了鸡胸肉", "source_start": 13, "source_end": 20, "depends_on": []},
        {"id": "goal-c1", "task_type": "nutrition_advice", "action": "question",
         "text": "帮我记录", "source_start": 8, "source_end": 12, "depends_on": ["task-1"]},
        {"id": "goal-c2", "task_type": "nutrition_advice", "action": "question",
         "text": "也帮我记录", "source_start": 21, "source_end": 26, "depends_on": ["task-2"]},
    ]

    def advise_ref(start, end, exact):
        return tool_call("advise", "advise_nutrition",
                         {"source_ref": {"start": start, "end": end, "exact_text": exact}})

    def policy(nth, messages):
        scripted = {
            1: [tool_call("t1", "advise_nutrition",
                          {"source_ref": {"start": 8, "end": 12, "exact_text": "帮我记录"}})],
            2: [tool_call("t2", "record_nutrition",
                          {"source_ref": ref_of(SRC_TWO_RECORDS, "早餐吃了燕麦粥")})],
            3: [tool_call("t3", "advise_nutrition",
                          {"source_ref": {"start": 21, "end": 26, "exact_text": "也帮我记录"}})],
            4: [tool_call("t4", "record_nutrition",
                          {"source_ref": ref_of(SRC_TWO_RECORDS, "晚餐吃了鸡胸肉")})],
        }
        calls = scripted.get(nth)
        if calls:
            return {"role": "assistant", "content": "", "tool_calls": calls}
        return {"role": "assistant", "content": "都完成了"}

    llm = FakeLLMPolicy(policy)
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-fixed"),
        "advise_nutrition": {"status": "completed", "result": {"advice": "ok"}},
    })
    return llm, executor, candidates


def _advance(state, deps, calls):
    state["messages"].append(assistant(tool_calls=calls))
    result = asyncio.run(execute_react_batch(state, deps))
    state.update(result)
    return result


def test_r4_05_cross_batch_recovery_identity_stable_and_distinguishable():
    """跨批两次独立恢复：原call ID分别正确、recovery_index跨批累计可区分。"""
    llm, executor, candidates = _cross_batch_recovery_deps()
    deps = _deps(llm, executor, max_tool_steps=16)
    state = batch_state(SRC_TWO_RECORDS, assistant(tool_calls=[
        tool_call("t1", "advise_nutrition",
                  {"source_ref": {"start": 8, "end": 12, "exact_text": "帮我记录"}})]), candidates=candidates)
    state.update(asyncio.run(execute_react_batch(state, deps)))  # 执行turn1批次（advise阻塞入pending）
    _advance(state, deps, [tool_call("t2", "record_nutrition",
                                     {"source_ref": ref_of(SRC_TWO_RECORDS, "早餐吃了燕麦粥")})])
    _advance(state, deps, [tool_call("t3", "advise_nutrition",
                             {"source_ref": {"start": 21, "end": 26, "exact_text": "也帮我记录"}})])
    _advance(state, deps, [tool_call("t4", "record_nutrition",
                                     {"source_ref": ref_of(SRC_TWO_RECORDS, "晚餐吃了鸡胸肉")})])
    events = []
    for message in state["messages"]:
        if message.get("role") == "user" and str(message.get("content", "")).startswith(
                RECOVERY_OBSERVATION_NOTICE):
            events.extend(json.loads(message["content"][len(RECOVERY_OBSERVATION_NOTICE):]))
    assert len(events) == 2, f"R4-05前置: 应有两次恢复事件: {events}"
    assert [e["tool_call_id"] for e in events] == ["t1", "t3"], (
        f"R4-05: 恢复事件原call ID应分别为t1/t3: {[(e['tool_call_id']) for e in events]}")
    assert [e["recovery_index"] for e in events] == [1, 2], (
        f"R4-05: recovery_index应跨批累计可区分: {[e['recovery_index'] for e in events]}")
    assert [e["goal_ids"] for e in events] == [["goal-c1"], ["goal-c2"]], (
        f"R4-05: 事件应用目标应可区分: {events}")


def test_r4_05_restricted_recovery_not_covered_visible_and_bounded():
    """恢复返回unavailable：不covered、失败事件可见、pending移除不无界重试、不complete。"""
    llm, executor, candidates = _cross_batch_recovery_deps()
    executor.results["advise_nutrition"] = {"status": "unavailable",
                                            "error": "recovery-attempt-failed"}
    deps = _deps(llm, executor, max_tool_steps=16)
    state = batch_state(SRC_TWO_RECORDS, assistant(tool_calls=[
        tool_call("t1", "advise_nutrition",
                  {"source_ref": {"start": 8, "end": 12, "exact_text": "帮我记录"}})]), candidates=candidates)
    state.update(asyncio.run(execute_react_batch(state, deps)))  # 执行turn1批次（advise阻塞入pending）
    _advance(state, deps, [tool_call("t2", "record_nutrition",
                                     {"source_ref": ref_of(SRC_TWO_RECORDS, "早餐吃了燕麦粥")})])
    assert "goal-c1" not in state["covered_goals"], (
        f"R4-05: 受限恢复不得覆盖目标: {state['covered_goals']}")
    assert state["pending_goals"] == [], (
        f"R4-05: 受限恢复后应结束本次尝试（pending移除，防无界重试）: {state['pending_goals']}")
    events = []
    for message in state["messages"]:
        if message.get("role") == "user" and str(message.get("content", "")).startswith(
                RECOVERY_OBSERVATION_NOTICE):
            events.extend(json.loads(message["content"][len(RECOVERY_OBSERVATION_NOTICE):]))
    assert any(e["observation"]["status"] == "unavailable" for e in events), (
        f"R4-05: 失败恢复事件应可见: {events}")
    assert state["status"] != "complete", f"R4-05: 受限恢复不得complete: {state['status']!r}"
    # R4-05校准（round5裁定）：调用真实finalize验证非complete/非空受限答复
    state.update(finalize_node(state, deps))
    assert state["status"] != "complete", (
        f"R4-05: finalize终态不得complete: {state['status']!r}")
    assert state["final_answer"].strip(), (
        f"R4-05: 受限答复为空: {state['final_answer']!r}")
    assert "全部保存成功" not in state["final_answer"], (
        f"R4-05: 受限答复含肯定保存宣称: {state['final_answer']!r}")
    # 随后另一合法写提交：旧失败建议不得再次自动派发（恢复去重/有界）
    advise_before = len([c for c in executor.calls if c["name"] == "advise_nutrition"])
    _advance(state, deps, [tool_call("t5", "record_nutrition",
                                     {"source_ref": ref_of(SRC_TWO_RECORDS, "晚餐吃了鸡胸肉")})])
    advise_after = len([c for c in executor.calls if c["name"] == "advise_nutrition"])
    assert advise_after == advise_before == 1, (
        f"R4-05: 后续写提交不得再次自动派发旧失败建议: before={advise_before}, "
        f"after={advise_after}, names={executor.names}")


def test_r4_05_control_success_recovery_then_reselect_no_redispatch():
    """成功恢复后主管再次选同目标：复用可信观察零重派。"""
    llm, executor, candidates = _cross_batch_recovery_deps()
    deps = _deps(llm, executor, max_tool_steps=16)
    state = batch_state(SRC_TWO_RECORDS, assistant(tool_calls=[
        tool_call("t1", "advise_nutrition",
                  {"source_ref": {"start": 8, "end": 12, "exact_text": "帮我记录"}})]), candidates=candidates)
    state.update(asyncio.run(execute_react_batch(state, deps)))  # 执行turn1批次（advise阻塞入pending）
    _advance(state, deps, [tool_call("t2", "record_nutrition",
                                     {"source_ref": ref_of(SRC_TWO_RECORDS, "早餐吃了燕麦粥")})])
    # 先证恰好一次真实建议恢复（同身份 goal-c1），再证重选复用
    advise_recoveries = [c for c in executor.calls if c["name"] == "advise_nutrition"]
    assert len(advise_recoveries) == 1, (
        f"R4-05: 重选前应恰好一次成功建议恢复: {executor.names}")
    assert advise_recoveries[0]["context"]["committed_tasks"] == ["task-1"], (
        f"R4-05: 恢复context未看见本批提交: {advise_recoveries[0]['context']}")
    dispatches_after_recovery = len(executor.calls)
    _advance(state, deps, [tool_call("t3", "advise_nutrition",
                             {"source_ref": {"start": 8, "end": 12, "exact_text": "帮我记录"}})])
    assert len(executor.calls) == dispatches_after_recovery, (
        f"R4-05: 同目标重选应复用零新增派发: {executor.names}")
    assert "goal-c1" in state["covered_goals"], (
        f"R4-05: 重选后目标身份应仍为goal-c1: {state['covered_goals']}")


# ---------------------------------------------------------------------------
# R4-06 / RA-12：安全控制保持（原集R3急症控制复跑即覆盖，此处补直接执行旁路确认）
# ---------------------------------------------------------------------------
def test_r4_06_urgent_direct_execute_call_all_tools_rejected():
    """直接 _execute_call 旁路：急症下写/读/建议全部零派发（确定性拒绝）。"""
    from app.services.agent.react_graph import _execute_call
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt(),
        "read_profile": {"status": "completed"},
        "advise_nutrition": {"status": "completed"},
    })
    deps = _deps(FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)
    state = {"safety_level": "urgent", "source_text": SRC_URGENT,
             "source_scope": None, "intent_candidates": []}
    from app.services.agent.react_models import ToolCall
    for name, args in [("record_nutrition", {"source_ref": ref_of(SRC_URGENT, SRC_URGENT)}),
                       ("read_profile", {}), ("advise_nutrition", {})]:
        observation = asyncio.run(_execute_call(
            ToolCall(id="x", name=name, args=args), state, deps))
        assert observation.status == ToolStatus.REJECTED, (
            f"R4-06: 急症下 {name} 未确定性拒绝: {observation}")
    assert executor.calls == [], f"R4-06: 急症下执行器被派发: {executor.names}"


SRC_URGENT = "胸痛得厉害,帮我记录今天心率120"
