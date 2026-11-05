"""Tests for react dispatch safety."""

from __future__ import annotations

import asyncio
import json

import pytest

from app.services.agent.react_graph import (
    ReactDependencies,
    execute_react_batch,
    invoke_react,
)
from app.services.agent.react_models import DomainObservation, ToolStatus

from tests.react_supervisor_fixtures import (
    RecordingExecutor,
    SemanticModelMixin,
    assistant,
    batch_state,
    committed_receipt,
    ref_of,
    tool_call,
)

SRC_RECORD = "帮我记录午餐吃了牛肉面和可乐"

# T8（round5裁定）：七个写观察返回边界用例的合法目标前置——完整文本、严格区间、
# 稳定应用ID、与 source_scope 一致；经真实 source_ref/写授权校验到达执行器。
_T8_BOUND_CANDIDATES = [{
    "id": "task-1", "task_type": "nutrition_record", "action": "record",
    "text": SRC_RECORD, "source_start": 0, "source_end": len(SRC_RECORD), "depends_on": [],
}]
SRC_DEPS = "帮我记录早餐吃了燕麦粥。然后帮我记录早餐吃了鸡蛋。然后根据早餐给我营养建议"
SRC_URGENT = "胸痛得厉害,帮我记录今天心率120"
SRC_VISIBILITY = "帮我记录早餐吃了燕麦粥。然后帮我记录早餐吃了鸡蛋。然后根据早餐给我营养建议。卧推怎么练？"
SRC_MIXED = "早上跑步5公里,晚上也跑步5公里"


def _deps(llm, executor=None, **options):
    return ReactDependencies(llm=llm, tool_executor=executor, user_id="app-user-1",
                             conversation_id="conv-1", request_id="req-2", **options)


def _run(deps, message):
    return asyncio.run(invoke_react(deps, message))


class FakeLLMPolicy(SemanticModelMixin):
    """R2 专用最小策略模型（与 support.FakeLLM 同约定，避免循环依赖）。"""

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
# R2-01 安全门真正前置：急症输入下模型调用与普通执行器派发均为 0
# ---------------------------------------------------------------------------
def test_urgent_input_stops_supervisor_and_all_normal_dispatch():
    llm_calls_seen = []

    def policy(nth, messages):
        llm_calls_seen.append(nth)
        if nth == 1:
            return {"role": "assistant", "content": "", "tool_calls": [
                tool_call("t1", "advise_nutrition", {}),
                tool_call("t2", "record_nutrition",
                          {"source_ref": ref_of(SRC_URGENT, "帮我记录今天心率120")}),
                tool_call("t3", "read_profile", {}),
            ]}
        return {"role": "assistant", "content": "处理完成"}

    llm = FakeLLMPolicy(policy)
    executor = RecordingExecutor({
        "advise_nutrition": {"status": "completed", "result": {"advice": "x"}},
        "record_nutrition": committed_receipt(),
        "read_profile": {"status": "completed"},
    })
    result = _run(_deps(llm, executor), SRC_URGENT)
    assert llm_calls_seen == [], (
        f"R2-01: 急症输入应先于任何模型调用终止, 实际模型调用 {llm_calls_seen} 次; "
        f"executor={executor.names}, observations="
        f"{[(o['status'], o.get('error')) for o in result['observations']]}")
    assert executor.calls == [], (
        f"R2-01: 急症输入不得派发普通记录/建议/读取执行器: {executor.names}")


def test_direct_node_urgent_read_tool_must_not_dispatch():
    """协议控制：直接节点调用也不能绕过急症门（读工具同样不得派发）。"""
    executor = RecordingExecutor({"read_profile": {"status": "completed"}})
    state = batch_state(
        SRC_URGENT,
        assistant(tool_calls=[tool_call("t1", "read_profile", {})]),
        safety_level="urgent",
    )
    result = asyncio.run(execute_react_batch(state, _deps(FakeLLMPolicy(
        lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert executor.calls == [], (
        f"R2-01: 直接节点急症状态下读取工具仍被派发: {executor.names}, "
        f"observation={result['observations'][0] if result['observations'] else None}")


def test_control_non_urgent_normal_flow_dispatches():
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt(),
    })
    result = _run(_deps(FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "record_nutrition",
                      {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]}
    ) if n == 1 else {"role": "assistant", "content": "完成"}), executor), SRC_RECORD)
    assert executor.names == ["record_nutrition"], f"R2-01对照: 非急症正常派发: {executor.names}"
    assert result["status"] == "complete"


# ---------------------------------------------------------------------------
# R2-02 嵌套类型与整批原子拒绝
# ---------------------------------------------------------------------------
def _float_ref_record_call(call_id):
    return tool_call(call_id, "record_nutrition",
                     {"source_ref": {"start": 1.0, "end": 4.0,
                                     "exact_text": SRC_RECORD[1:4]}})


@pytest.mark.parametrize("order", ["read-bad-read", "bad-read-read"],
                         ids=["legal-around-bad", "bad-first"])
def test_nested_float_source_ref_batch_must_be_atomic(order):
    calls = [tool_call("t1", "read_profile", {}), _float_ref_record_call("t2"),
             tool_call("t3", "read_profile", {})]
    if order == "bad-read-read":
        calls = [calls[1], calls[0], calls[2]]
    executor = RecordingExecutor({"read_profile": {"status": "completed"}})
    state = batch_state(SRC_RECORD, assistant(tool_calls=calls))
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert executor.calls == [], (
        f"R2-02: 含嵌套float source_ref的批次必须整批零执行, 实际派发 {executor.names}, "
        f"observations={[(o['status'], o.get('error')) for o in result['observations']]}")
    assert result["status"] == "restricted", (
        f"R2-02: 整批原子拒绝应为restricted, 实际 {result['status']!r}")


# ---------------------------------------------------------------------------
# R2-03 精确绑定与复用前校验
# ---------------------------------------------------------------------------
def test_bad_ref_on_covered_goal_must_be_rejected_not_reused():
    """协议控制：已处理目标后提交坏 exact_text 的 ref，必须明确拒绝而非返回旧回执。"""
    bad_ref = {"start": 0, "end": 4, "exact_text": "XX编造XX"}
    state = batch_state(
        SRC_RECORD,
        assistant(tool_calls=[tool_call("t1", "record_nutrition", {"source_ref": bad_ref})]),
        candidates=[{"id": "task-1", "task_type": "nutrition_record", "action": "record",
                     "text": SRC_RECORD, "source_start": 0, "source_end": len(SRC_RECORD),
                     "depends_on": []}],
        covered_goals=["task-1"],
        goal_results={"task-1": committed_receipt()},
    )
    executor = RecordingExecutor()
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    observation = result["observations"][0]
    assert observation["status"] == "rejected" and observation.get("error") == "exact_text_mismatch", (
        f"R2-03: 已覆盖目标的坏ref被复用旧观察而非拒绝: {observation}")


def test_single_char_ref_must_not_cover_whole_goal():
    source = SRC_RECORD
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "record_nutrition", {"source_ref": {"start": 2, "end": 3, "exact_text": "记"}})]}
    ) if n == 1 else {"role": "assistant", "content": "完成"})
    executor = RecordingExecutor({"record_nutrition": committed_receipt()})
    result = _run(_deps(llm, executor), source)
    assert "task-1" not in result["covered_goals"], (
        f"R2-03: 单字符ref不能随意覆盖整个目标: covered={result['covered_goals']}, "
        f"observations={[(o['status'], o.get('error')) for o in result['observations']]}")
    assert executor.calls == [], f"R2-03: 单字符ref不得派发: {executor.names}"


def test_mixed_explicit_and_implicit_positions_must_not_swallow_second():
    """协议控制：显式区间+同文本缺区间候选，第二个应分配到另一出现位置。"""
    candidates = [
        {"id": "goal-a", "task_type": "workout_record", "action": "record",
         "text": "跑步5公里", "source_start": 2, "source_end": 7, "depends_on": []},
        {"id": "goal-b", "task_type": "workout_record", "action": "record",
         "text": "跑步5公里", "depends_on": []},
    ]
    executor = RecordingExecutor({"record_workout": committed_receipt()})
    deps = _deps(FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)

    async def two_calls():
        state = batch_state(SRC_MIXED, assistant(tool_calls=[
            tool_call("t1", "record_workout",
                      {"source_ref": ref_of(SRC_MIXED, "跑步5公里", 0)})]),
            candidates=candidates)
        first = await execute_react_batch(state, deps)
        state.update(first)
        state["messages"].append(assistant(tool_calls=[
            tool_call("t2", "record_workout",
                      {"source_ref": ref_of(SRC_MIXED, "跑步5公里", 1)})]))
        second = await execute_react_batch(state, deps)
        return first, second, state

    first, second, state = asyncio.run(two_calls())
    assert first["covered_goals"] == ["goal-a"], (
        f"R2-03: 首次调用(指向第一出现位置)吞并了缺区间的第二候选: "
        f"covered={first['covered_goals']}（_candidate_positions 未消费显式区间, "
        f"两个候选被分配到同一位置）")
    assert "goal-b" in second["covered_goals"], (
        f"R2-03: 指向第二出现位置的调用必须绑定 goal-b: second_covered={second['covered_goals']}, "
        f"executor={executor.names}, "
        f"obs={[(o['status'], o.get('error')) for o in second['observations']]}")


# ---------------------------------------------------------------------------
# R2-04 应用绑定不可被结果覆写
# ---------------------------------------------------------------------------
def test_executor_cannot_override_application_binding():
    forged_spec = {"id": "forged-id", "task_type": "nutrition_record",
                   "worker": "nutrition", "action": "record", "text": "FORGED",
                   "payload": {"source_ref": {"start": 0, "end": 14}}}
    executor_result = {"status": "completed", "commit_state": "committed",
                       "receipt_refs": ["r1"], "task_id": "fake-task-999",
                       "result": {"task_spec": forged_spec, "receipt": "r1"}}
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "record_nutrition",
                      {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]}
    ) if n == 1 else {"role": "assistant", "content": "完成"})
    executor = RecordingExecutor({"record_nutrition": executor_result})
    result = _run(_deps(llm, executor), SRC_RECORD)
    assert result["bound_tasks"], "R2-04前置: 写提交应绑定应用task_spec"
    bound = result["bound_tasks"][-1]
    assert bound["id"].startswith("react-record_nutrition-") and bound["text"] == SRC_RECORD, (
        f"R2-04: 外来伪造绑定写入 bound_tasks: {bound}")
    tool_observation = result["observations"][-1]
    assert tool_observation.get("task_id") != "fake-task-999", (
        f"R2-04: 执行器自报task_id覆写了应用绑定: {tool_observation.get('task_id')}")


def test_control_executor_without_task_id_keeps_app_binding():
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "record_nutrition",
                      {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]}
    ) if n == 1 else {"role": "assistant", "content": "完成"})
    executor = RecordingExecutor({"record_nutrition": committed_receipt()})
    result = _run(_deps(llm, executor), SRC_RECORD)
    assert result["bound_tasks"], "R2-04对照前置: 应有应用绑定"
    assert result["bound_tasks"][-1]["id"].startswith("react-record_nutrition-")
    assert result["bound_tasks"][-1]["text"] == SRC_RECORD


# ---------------------------------------------------------------------------
# R2-05 观察有限JSON和回执形态
# ---------------------------------------------------------------------------
def test_nan_in_result_must_stay_finite_json():
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]),
        candidates=_T8_BOUND_CANDIDATES)
    executor = RecordingExecutor({"record_nutrition": {
        "status": "completed", "result": {"value": float("nan")}}})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert len(executor.calls) == 1, (
        f"T8前置: 合法目标前置后执行器应确实派发一次: {executor.names}")

    def _reject_constant(value):
        raise ValueError(f"非有限JSON常量: {value}")

    content = result["messages"][-1]["content"]
    try:
        json.loads(content, parse_constant=_reject_constant)
    except ValueError as exc:
        raise AssertionError(f"R2-05: tool结果含非有限JSON常量（NaN/Infinity 未收敛）: {content!r} ({exc})")


def test_unserializable_result_must_not_crash_graph():
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]),
        candidates=_T8_BOUND_CANDIDATES)
    executor = RecordingExecutor()
    executor.results["record_nutrition"] = {"status": "completed", "result": {"x": object()}}
    try:
        result = asyncio.run(execute_react_batch(state, _deps(
            FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"R2-05: 不可序列化result导致整批崩溃而非收敛错误: {exc!r}")
    assert len(executor.calls) == 1, (
        f"T8前置: 合法目标前置后执行器应确实派发一次: {executor.names}")
    observation = result["observations"][0]
    assert observation["status"] in {"unavailable", "commit_unknown"}, (
        f"R2-05: 不可序列化result应收敛为受限观察: {observation}")


def test_direct_observation_non_object_result_must_converge():
    """T4修订（round4裁定）：垃圾领域result不得进入观察/tool/恢复消息。

    原载荷 DomainObservation(status=COMPLETED, result="not-an-object")（无
    commit_state/receipt_refs，不与R3非对象result+committed+回执载荷混写）。
    语义断言：归一化错误保留、result为空或仅有应用核对正确的task_spec、
    受限状态、零可信提交。原样运行证据：round4 asoriginal 日志。
    """
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]),
        candidates=_T8_BOUND_CANDIDATES)
    executor = RecordingExecutor()
    executor.results["record_nutrition"] = DomainObservation(
        status=ToolStatus.COMPLETED, result="not-an-object")
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert len(executor.calls) == 1, (
        f"T8前置: 合法目标前置后执行器应确实派发一次: {executor.names}")
    observation = result["observations"][0]
    # 1) 垃圾值不得进入任何协议字段
    assert "not-an-object" not in json.dumps(observation, ensure_ascii=False), (
        f"T4: 垃圾result进入观察: {observation}")
    # 2) 归一化错误保留
    assert observation.get("error") == "result_not_object", (
        f"T4: 归一化错误丢失: {observation}")
    # 3) 受限状态：COMPLETED 因非法 result 降级，不得保持完成
    assert observation["status"] != "completed", (
        f"T4: 非法result未降级状态: {observation['status']!r}")
    # 4) result 为空或仅有应用核对正确的 task_spec（id/来源/类型与应用绑定一致）
    observed_result = observation.get("result")
    if observed_result:
        spec_dict = observed_result.get("task_spec")
        assert isinstance(spec_dict, dict) and spec_dict.get("text") == SRC_RECORD, (
            f"T4: result含非应用核对内容: {observed_result}")
    # 5) 零可信提交（原载荷无 commit_state/receipt_refs）
    assert result["committed_tasks"] == [], (
        f"T4: 零提交证明被违反: {result['committed_tasks']}")
    # T4加强（round5裁定）：完整应用身份与所有协议输出
    spec_dict = (observed_result or {}).get("task_spec") or {}
    assert spec_dict.get("id") == observation.get("task_id") and spec_dict.get("id"), (
        f"T4加强: 观察task_id与应用task_spec不一致: {observation.get('task_id')} vs {spec_dict.get('id')}")
    assert spec_dict.get("task_type") == "nutrition_record" and spec_dict.get("text") == SRC_RECORD, (
        f"T4加强: 应用task_spec类型/来源不正确: {spec_dict}")
    for message in result["messages"]:
        if message.get("role") in {"tool", "user"}:
            assert "not-an-object" not in str(message.get("content")), (
                f"T4加强: 垃圾值进入协议消息: {message}")
    assert result["pending_goals"] == [] and not result.get("recovery_counter"), (
        "T4加强: 本用例无恢复路径, 不得声称覆盖恢复输出")


@pytest.mark.parametrize("bad_receipts", [[""], [None], [{"doc": 1}]],
                         ids=["empty-string", "null", "dict"])
def test_degenerate_receipt_refs_must_not_count_as_commit_proof(bad_receipts):
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "record_nutrition",
                      {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]}
    ) if n == 1 else {"role": "assistant", "content": "完成"})
    executor = RecordingExecutor({"record_nutrition": {
        "status": "completed", "commit_state": "committed", "receipt_refs": bad_receipts}})
    result = _run(_deps(llm, executor), SRC_RECORD)
    assert result["committed_tasks"] == [], (
        f"R2-05: receipt_refs={bad_receipts!r}不能作为提交证明: committed={result['committed_tasks']}")


# ---------------------------------------------------------------------------
# R2-06 自动恢复的预算、可见性与最终状态
# ---------------------------------------------------------------------------
def _recovery_deps_script(max_tool_steps=16):
    """turn1: advise(阻塞) → turn2: [record, record]（提交触发自动恢复）。"""
    def policy(nth, messages):
        if nth == 1:
            return {"role": "assistant", "content": "", "tool_calls": [
                tool_call("t1", "advise_nutrition", {})]}
        if nth == 2:
            return {"role": "assistant", "content": "", "tool_calls": [
                tool_call("t2", "record_nutrition",
                          {"source_ref": ref_of(SRC_DEPS, "帮我记录早餐吃了燕麦粥")}),
                tool_call("t3", "record_nutrition",
                          {"source_ref": ref_of(SRC_DEPS, "帮我记录早餐吃了鸡蛋")})]}
        return {"role": "assistant", "content": "都完成了"}
    llm = FakeLLMPolicy(policy)
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-fixed"),
        "advise_nutrition": {"status": "completed", "result": {"advice": "蛋白充足"}},
    })
    return _deps(llm, executor, max_tool_steps=max_tool_steps), llm, executor


def test_auto_recovery_must_respect_tool_step_budget():
    """上一轮blocked、本轮记录恰好耗尽剩余工具额度：不得额外执行建议。"""
    deps, llm, executor = _recovery_deps_script(max_tool_steps=3)
    result = _run(deps, SRC_DEPS)
    assert "advise_nutrition" not in executor.names, (
        f"R2-06: 预算耗尽后自动恢复仍额外执行了建议: executor={executor.names}, "
        f"tool_steps={result['tool_steps']}/max=3")


def test_auto_recovery_runs_once_and_observation_must_reach_supervisor():
    deps, llm, executor = _recovery_deps_script()
    result = _run(deps, SRC_VISIBILITY)
    assert executor.names.count("advise_nutrition") == 1, (
        f"R2-06: 恢复的建议应只执行一次: {executor.names}")
    recovered = [o for o in result["observations"]
                 if o.get("result", {}).get("advice") == "蛋白充足"]
    assert recovered, "R2-06前置: 恢复观察应进入 observations"
    # 下一次主管调用（task-4 qa 目标未覆盖，必然还有模型调用）的输入必须包含恢复观察。
    later_inputs = llm.calls[2:]
    assert later_inputs, f"R2-06前置: task-4未处理应还有主管调用, turns={result.get('supervisor_turns')}"
    serialized = json.dumps(later_inputs[0], ensure_ascii=False)
    assert "蛋白充足" in serialized, (
        "R2-06: 自动恢复的新观察未进入下一次主管实际上下文（messages 中无该观察）")


def test_recovery_pairing_and_no_faked_tool_messages():
    deps, llm, executor = _recovery_deps_script()
    result = _run(deps, SRC_DEPS)
    assistant_call_ids = [c.get("id") for m in result["messages"]
                          if m.get("role") == "assistant"
                          for c in (m.get("tool_calls") or []) if isinstance(c, dict)]
    tool_ids = [m.get("tool_call_id") for m in result["messages"] if m.get("role") == "tool"]
    assert sorted(assistant_call_ids) == sorted(tool_ids), (
        f"R2-06: 恢复不得伪造模型工具调用或添加无对应call的tool消息: "
        f"calls={assistant_call_ids}, tools={tool_ids}")


def test_same_batch_block_then_commit_final_status_not_blocked():
    """同批阻塞→记录→（下一轮重试执行）全部完成：不得因历史blocked误报未完成。"""
    def policy(nth, messages):
        if nth == 1:
            return {"role": "assistant", "content": "", "tool_calls": [
                tool_call("t1", "advise_nutrition", {}),
                tool_call("t2", "record_nutrition",
                          {"source_ref": ref_of(SRC_DEPS, "帮我记录早餐吃了燕麦粥")}),
                tool_call("t3", "record_nutrition",
                          {"source_ref": ref_of(SRC_DEPS, "帮我记录早餐吃了鸡蛋")})]}
        if nth == 2:
            return {"role": "assistant", "content": "", "tool_calls": [
                tool_call("t4", "advise_nutrition", {})]}
        return {"role": "assistant", "content": "全部完成"}
    llm = FakeLLMPolicy(policy)
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-fixed"),
        "advise_nutrition": {"status": "completed", "result": {"advice": "蛋白充足"}},
    })
    result = _run(_deps(llm, executor), SRC_DEPS)
    assert "advise_nutrition" in executor.names, (
        f"R2-06前置: 依赖提交后的建议重试应执行: {executor.names}")
    assert result["status"] == "complete", (
        f"R2-06: 同批阻塞→记录→重试完成, 历史blocked使最终状态误报: {result['status']!r}, "
        f"reason={result['termination_reason']!r}, answer={result['final_answer']!r}")


def test_cross_batch_block_then_commit_final_status_not_blocked():
    deps, llm, executor = _recovery_deps_script()
    result = _run(deps, SRC_DEPS)
    assert result["status"] == "complete", (
        f"R2-06: 跨批阻塞→记录→恢复完成, 历史blocked使最终状态误报: {result['status']!r}")


# ---------------------------------------------------------------------------
# R2-07 独立写任务的提交未知呈现
# ---------------------------------------------------------------------------
def test_completed_unknown_commit_must_not_claim_success():
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "record_nutrition",
                      {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]}
    ) if n == 1 else {"role": "assistant", "content": "全部保存成功"})
    executor = RecordingExecutor({"record_nutrition": {
        "status": "completed", "commit_state": "unknown"}})
    result = _run(_deps(llm, executor), SRC_RECORD)
    assert result["status"] != "complete" or "全部保存成功" not in result["final_answer"], (
        f"R2-07: completed+unknown的独立写任务被判成功并转发模型宣称: "
        f"status={result['status']!r}, answer={result['final_answer']!r}")


def test_control_read_only_completed_completes_normally():
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "answer_training", {})]}
    ) if n == 1 else {"role": "assistant", "content": "训练建议是循序渐进。"})
    executor = RecordingExecutor({"answer_training": {"status": "completed"}})
    result = _run(_deps(llm, executor), "怎么练卧推")
    assert result["status"] == "complete", f"R2-07对照: 只读完成应正常complete: {result['status']!r}"
    assert result["final_answer"] == "训练建议是循序渐进。"


def test_control_legal_committed_receipt_completes():
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "record_nutrition",
                      {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]}
    ) if n == 1 else {"role": "assistant", "content": "已记录。"})
    executor = RecordingExecutor({"record_nutrition": committed_receipt("rcpt-legal")})
    result = _run(_deps(llm, executor), SRC_RECORD)
    assert result["status"] == "complete", (
        f"R2-07对照: 合法committed回执应complete: {result['status']!r}")
    assert result["committed_tasks"] == ["task-1"]
