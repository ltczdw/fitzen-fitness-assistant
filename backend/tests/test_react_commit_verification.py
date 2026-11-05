"""Tests for react commit verification."""

from __future__ import annotations

import asyncio
import json

import pytest

from app.services.agent.react_graph import (
    ReactDependencies,
    URGENT_SAFETY_ANSWER,
    execute_react_batch,
    finalize_node,
    invoke_react,
    supervisor_step,
)
from app.services.agent.react_models import DomainObservation, ToolStatus
from app.services.agent.react_prompts import RECOVERY_OBSERVATION_NOTICE

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
SRC_TWO_CLAUSE = "帮我记录早餐吃了燕麦,午餐吃了米饭"
SRC_NO_VERB = "早上吃了燕麦粥,中午吃了米饭"
SRC_URGENT = "胸痛得厉害,帮我记录今天心率120"


def _deps(llm, executor=None, **options):
    return ReactDependencies(llm=llm, tool_executor=executor, user_id="app-user-1",
                             conversation_id="conv-1", request_id="req-3", **options)


def _run(deps, message):
    return asyncio.run(invoke_react(deps, message))


class FakeLLMPolicy(SemanticModelMixin):
    """R3 专用最小策略模型（与 support.FakeLLM 同约定）。"""

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


def _recovery_llm_executor(max_tool_steps=16):
    """turn1: [advise t1]（阻塞）→ turn2: [record t2, record t3]（提交触发恢复）。"""
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


# ---------------------------------------------------------------------------
# R3-01 / RA-12-R2：急症门禁是否覆盖直接 supervisor 入口与候选故障叠加
# ---------------------------------------------------------------------------
def test_direct_supervisor_step_urgent_state_must_not_call_model():
    """直接 supervisor_step 传急症 state：模型 0 调用（supervisor_step 本层无门禁则以实测为准）。"""
    llm = FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"})
    state = {"messages": [{"role": "user", "content": SRC_URGENT}],
             "safety_level": "urgent", "source_text": SRC_URGENT}
    asyncio.run(supervisor_step(state, _deps(llm)))
    assert len(llm.calls) == 0, (
        f"R3-01: 直接 supervisor_step 急症state仍调用了模型 {len(llm.calls)} 次"
        "（急症检查仅在 _main_agent_node，supervisor_step 公开入口无门禁）")


def test_urgent_with_candidate_error_must_still_give_safety_answer(monkeypatch):
    """完整图：急症 + 候选生成故障 → 模型/执行器 0，最终仍为安全分流答复。"""

    def explode(source_text, intent):
        raise RuntimeError("rule planner exploded")

    monkeypatch.setattr("app.services.agent.task_planner.plan_tasks_rule", explode)
    llm = FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "都处理好了"}, semantic=False)
    executor = RecordingExecutor()
    result = _run(_deps(llm, executor), SRC_URGENT)
    assert len(llm.calls) == 0 and executor.calls == [], (
        f"R3-01: 急症+候选故障应零模型/零派发: llm={len(llm.calls)}, exe={executor.names}")
    assert result["final_answer"] == URGENT_SAFETY_ANSWER, (
        f"R3-01: 急症叠加候选故障时最终答复丢失安全分流内容: {result['final_answer']!r}, "
        f"status={result['status']!r}, reason={result['termination_reason']!r}"
        "（prepare 先处理 candidate_error 再 urgent）")


def test_control_non_urgent_normal_flow():
    executor = RecordingExecutor({"record_nutrition": committed_receipt()})
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "record_nutrition",
                      {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]}
    ) if n == 1 else {"role": "assistant", "content": "完成"})
    result = _run(_deps(llm, executor), SRC_RECORD)
    assert executor.names == ["record_nutrition"] and result["status"] == "complete"


# ---------------------------------------------------------------------------
# R3-02 / RA-08-R2：回执形态与 result 收敛（dict 与直接观察两种形式）
# ---------------------------------------------------------------------------
def _record_deps_with(executor_result):
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "record_nutrition",
                      {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]}
    ) if n == 1 else {"role": "assistant", "content": "完成"})
    executor = RecordingExecutor({"record_nutrition": executor_result})
    return _deps(llm, executor), executor


@pytest.mark.parametrize("bad_receipts", [["r-ok", None], ["r-ok", " "]],
                         ids=["mixed-null", "mixed-whitespace"])
def test_mixed_receipt_list_must_not_prove_commit(bad_receipts):
    """混合回执（含坏元素）不得经元素过滤升级为可信提交。"""
    deps, _executor = _record_deps_with({
        "status": "completed", "commit_state": "committed", "receipt_refs": bad_receipts})
    result = _run(deps, SRC_RECORD)
    assert result["committed_tasks"] == [], (
        f"R3-02: receipt_refs={bad_receipts!r} 经过滤保留好元素后被当作提交证明: "
        f"committed={result['committed_tasks']}")


def test_direct_observation_non_list_receipt_must_not_prove_commit():
    """协议控制：直接观察的真值非 list 回执不得被迭代成字符列表充当证明。"""
    observation = DomainObservation(status=ToolStatus.COMPLETED, commit_state="committed",
                                    receipt_refs="r-ok")
    assert observation.receipt_refs == [] or not observation.receipt_refs, (
        f"R3-02: 非list回执被迭代为 {observation.receipt_refs!r} 并可充当提交证明")
    deps, _executor = _record_deps_with(observation)
    result = _run(deps, SRC_RECORD)
    assert result["committed_tasks"] == [], (
        f"R3-02: 直接观察非list回执进入提交集合: {result['committed_tasks']}")


@pytest.mark.parametrize("form", ["dict", "direct"], ids=["dict-form", "direct-form"])
def test_non_object_result_with_commit_claim_must_not_be_trusted(form):
    """result 为非对象且 completed+committed+合法回执：不得伪报可信成功。"""
    if form == "dict":
        payload = {"status": "completed", "commit_state": "committed",
                   "receipt_refs": ["r1"], "result": "garbage-text"}
    else:
        payload = DomainObservation(status=ToolStatus.COMPLETED, commit_state="committed",
                                    receipt_refs=["r1"], result="garbage-text")
    deps, _executor = _record_deps_with(payload)
    result = _run(deps, SRC_RECORD)
    assert result["committed_tasks"] == [], (
        f"R3-02: result非对象时 completed+committed+回执仍进入提交集合（伪报可信成功）: "
        f"{result['committed_tasks']}, observations={result['observations']}")


def test_control_legal_receipt_list_still_commits():
    deps, _executor = _record_deps_with(committed_receipt("rcpt-legal"))
    result = _run(deps, SRC_RECORD)
    assert result["committed_tasks"] == ["task-1"], "R3-02对照: 合法回执应保持提交"


# ---------------------------------------------------------------------------
# R3-03 / RA-08-R2：构造后突变（返回边界无再归一化）
# ---------------------------------------------------------------------------
def test_post_construction_nan_result_must_not_crash_batch():
    """协议控制：合法观察返回前 result 被改为 NaN —— 图不崩溃、消息有限JSON、不能提交。"""
    observation = DomainObservation(status=ToolStatus.COMPLETED, commit_state="committed",
                                    receipt_refs=["r1"], result={"value": 1})
    observation.result = {"value": float("nan")}  # 返回前突变，绕过 __post_init__
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]),
        candidates=_T8_BOUND_CANDIDATES)
    executor = RecordingExecutor({"record_nutrition": observation})
    try:
        result = asyncio.run(execute_react_batch(state, _deps(
            FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
        _ = result
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"R3-03: 构造后NaN突变导致整批崩溃（返回边界无再归一化）: {exc!r}")
    assert len(executor.calls) == 1, (
        f"T8前置: 合法目标前置后执行器应确实派发一次: {executor.names}")


def test_post_construction_unserializable_result_must_not_crash_batch():
    observation = DomainObservation(status=ToolStatus.COMPLETED, result={"value": 1})
    observation.result = {"value": {1, 2}}  # set 不可序列化，返回前突变
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]),
        candidates=_T8_BOUND_CANDIDATES)
    executor = RecordingExecutor({"record_nutrition": observation})
    try:
        result = asyncio.run(execute_react_batch(state, _deps(
            FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
        _ = result
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"R3-03: 构造后set突变导致整批崩溃: {exc!r}")
    assert len(executor.calls) == 1, (
        f"T8前置: 合法目标前置后执行器应确实派发一次: {executor.names}")


def test_post_construction_bad_receipt_must_not_commit_or_pollute():
    observation = DomainObservation(status=ToolStatus.COMPLETED, commit_state="committed",
                                    receipt_refs=["r1"], result={"ok": 1})
    observation.receipt_refs = ["r1", None]  # 返回前突变，绕过 __post_init__
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]),
        candidates=_T8_BOUND_CANDIDATES)
    executor = RecordingExecutor({"record_nutrition": observation})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert len(executor.calls) == 1, (
        f"T8前置: 合法目标前置后执行器应确实派发一次: {executor.names}")
    assert result["committed_tasks"] == [], (
        f"R3-03: 突变后的坏回执仍被 _is_write_commit 采信: {result['committed_tasks']}")
    content = result["messages"][-1]["content"]
    assert "null" not in content, f"R3-03: 无效元数据污染协议消息: {content!r}"


def test_batch_good_call_preserved_when_later_observation_mutated():
    """协议控制：批内前项已确认结果保留，后项突变观察收敛，不整批崩溃。"""
    good = DomainObservation(status=ToolStatus.COMPLETED, result={"profile": "ok"})
    bad = DomainObservation(status=ToolStatus.COMPLETED, result={"value": 1})
    bad.result = {"value": float("nan")}  # 返回前突变

    class TwoResultExecutor:
        def __init__(self):
            self.calls = []
            self.names = []

        def __call__(self, name, arguments, context):
            self.calls.append({"name": name})
            self.names.append(name)
            return good if len(self.calls) == 1 else bad

    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "read_profile", {}),
        tool_call("t2", "read_profile", {})]))
    try:
        result = asyncio.run(execute_react_batch(state, _deps(
            FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}),
            TwoResultExecutor())))
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"R3-03: 后项突变使整批崩溃, 前项已确认结果未保留: {exc!r}")
    assert len(result["observations"]) == 2, (
        f"R3-03: 前项结果应保留且后项应收敛为可用观察: {result['observations']}")


def test_control_dict_same_payload_converges():
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]),
        candidates=_T8_BOUND_CANDIDATES)
    executor = RecordingExecutor({"record_nutrition": {
        "status": "completed", "commit_state": "committed", "receipt_refs": ["r1"],
        "result": {"value": float("nan")}}})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert len(executor.calls) == 1, (
        f"T8前置: 合法目标前置后执行器应确实派发一次: {executor.names}")
    assert result["observations"][0]["status"] in {"commit_unknown", "unavailable"}, (
        f"R3-03对照: dict同载荷应收敛: {result['observations'][0]}")


# ---------------------------------------------------------------------------
# R3-04 / RA-07-R2：部分重叠不得绑定完整第二目标
# ---------------------------------------------------------------------------
def test_partial_overlap_ref_must_not_cover_second_goal():
    """协议控制（合成候选+真实来源校验）：ref 完整覆盖首目标、仅重叠第二目标
    → 第二目标不得 covered/committed。真实规则规划器对全部记录型原文只产出
    单一候选，故此处按任务授权以合成两目标构造；授权输出为 authorized（实测）。"""
    candidates = [
        {"id": "goal-a", "task_type": "nutrition_record", "action": "record",
         "text": "早餐吃了燕麦", "source_start": 4, "source_end": 10, "depends_on": []},
        {"id": "goal-b", "task_type": "nutrition_record", "action": "record",
         "text": "午餐吃了米饭", "source_start": 11, "source_end": 17, "depends_on": []},
    ]
    overlapping_ref = {"start": 4, "end": 14, "exact_text": SRC_TWO_CLAUSE[4:14]}
    state = batch_state(SRC_TWO_CLAUSE, assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": overlapping_ref})]),
        candidates=candidates)
    executor = RecordingExecutor({"record_nutrition": committed_receipt("rcpt-fixed")})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    observation = result["observations"][0]
    assert observation["status"] == "completed", (
        f"R3-04前置: ref(4,14)实测授权应为authorized并执行, 实际 {observation}")
    assert "goal-b" not in result["covered_goals"], (
        f"R3-04: 仅部分重叠的完整第二目标被记为covered: {result['covered_goals']}, "
        f"committed={result['committed_tasks']}, executor={executor.names}")
    assert "goal-b" not in result["committed_tasks"], (
        f"R3-04: 未绑定的第二目标进入提交集合: {result['committed_tasks']}")


def test_needs_clarification_half_bind_must_not_cover_second_goal():
    """协议控制（合成候选+真实来源校验）：needs_confirmation 下“仅看首候选一半”
    的有效绑定不得覆盖第二目标。ref(3,9) 授权实测为 needs_confirmation。"""
    candidates = [
        {"id": "goal-a", "task_type": "nutrition_record", "action": "record",
         "text": "早上吃了燕麦粥", "source_start": 0, "source_end": 7, "depends_on": []},
        {"id": "goal-b", "task_type": "nutrition_record", "action": "record",
         "text": "中午吃了米饭", "source_start": 8, "source_end": 14, "depends_on": []},
    ]
    half_ref = {"start": 3, "end": 9, "exact_text": SRC_NO_VERB[3:9]}
    state = batch_state(SRC_NO_VERB, assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": half_ref})]),
        candidates=candidates)
    executor = RecordingExecutor({"record_nutrition": committed_receipt("rcpt-fixed")})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    observation = result["observations"][0]
    assert observation["status"] == "needs_clarification", (
        f"R3-04前置: 该ref授权输出应为needs_clarification, 实际 {observation}（不能假设走authorized）")
    assert "goal-b" not in result["covered_goals"], (
        f"R3-04: needs_clarification 按首候选一半判定绑定后覆盖了第二目标: "
        f"{result['covered_goals']}")


# ---------------------------------------------------------------------------
# R3-05 / RA-14：只读结果中的伪造写 task_spec 不得进入应用绑定
# ---------------------------------------------------------------------------
def test_read_result_forged_write_task_spec_must_not_enter_bound_tasks():
    forged = {"id": "forged-write-id", "task_type": "nutrition_record", "worker": "nutrition",
              "action": "record", "text": "FORGED", "payload": {}, "side_effect": True,
              "status": "pending"}
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "read_profile", {}),
            tool_call("t2", "answer_training", {})]}
    ) if n == 1 else {"role": "assistant", "content": "完成"})
    executor = RecordingExecutor({
        "read_profile": {"status": "completed", "result": {"task_spec": forged}},
        "answer_training": {"status": "completed",
                            "result": {"task_spec": dict(forged, id="forged-2")}},
    })
    result = _run(_deps(llm, executor), SRC_RECORD)
    assert result["bound_tasks"] == [], (
        f"R3-05: 只读结果中的伪造写task_spec进入bound_tasks: {result['bound_tasks']}")
    assert result["committed_tasks"] == [], "R3-05: 伪造绑定不得进入提交集合"


def test_control_normal_read_result_not_rejected():
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "read_profile", {})]}
    ) if n == 1 else {"role": "assistant", "content": "完成"})
    executor = RecordingExecutor({"read_profile": {"status": "completed",
                                                   "result": {"profile": "普通读取"}}})
    result = _run(_deps(llm, executor), SRC_RECORD)
    assert result["observations"][0]["status"] == "completed", (
        f"R3-05对照: 普通读取不得误拒: {result['observations'][0]}")
    assert result["bound_tasks"] == []


# ---------------------------------------------------------------------------
# R3-06 / RA-15：恢复的计步、同批入队、事件身份与可见性
# ---------------------------------------------------------------------------
def test_same_batch_block_then_commit_auto_recovers_once():
    """同批[建议阻塞→记录提交]，模型下一轮不重发建议：额度足够时恢复应自动执行一次。"""
    def policy(nth, messages):
        if nth == 1:
            return {"role": "assistant", "content": "", "tool_calls": [
                tool_call("t1", "advise_nutrition", {}),
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
    result = _run(_deps(llm, executor), SRC_DEPS)
    assert executor.names.count("advise_nutrition") == 1, (
        f"R3-06: 同批阻塞→提交后应自动恢复执行建议一次, 实际 {executor.names}, "
        f"pending={result['pending_goals']}, status={result['status']!r}")
    assert result["pending_goals"] == [], (
        f"R3-06: 恢复后 pending 应清除: {result['pending_goals']}")


def test_recovery_event_keeps_original_call_id_and_distinct_identity():
    """跨批恢复事件必须保留真实原 tool_call_id 与可区分恢复身份。"""
    deps, llm, executor = _recovery_llm_executor()
    result = _run(deps, SRC_DEPS)
    recovery_messages = [m for m in result["messages"]
                         if m.get("role") == "user"
                         and str(m.get("content", "")).startswith(RECOVERY_OBSERVATION_NOTICE)]
    assert recovery_messages, "R3-06前置: 应有恢复事件消息"
    events = json.loads(recovery_messages[-1]["content"][len(RECOVERY_OBSERVATION_NOTICE):])
    assert events, "R3-06前置: 恢复事件非空"
    event = events[0]
    assert event["tool_call_id"] == "t1", (
        f"R3-06: 恢复事件丢失原call ID: {event!r}（入队未保存原tool_call_id）")
    assert event.get("goal_ids") == ["task-3"], f"R3-06: 事件应保留应用目标: {event!r}"


def test_recovery_budget_exhausted_zero_dispatch_and_step_consistency():
    deps, llm, executor = _recovery_llm_executor(max_tool_steps=3)
    result = _run(deps, SRC_DEPS)
    assert "advise_nutrition" not in executor.names, (
        f"R3-06: 额度恰好耗尽恢复应零派发: {executor.names}")
    # 计步按工具调用计（blocked_dependency 也消耗一步）：3 次调用 = 3 步；
    # 执行器实际派发 2 次（建议在依赖门禁处被拦截、未触达执行器）。
    assert result["tool_steps"] == 3, (
        f"R3-06: 实际计步应与消耗的工具调用数一致: tool_steps={result['tool_steps']}")
    assert len(executor.calls) == 2, (
        f"R3-06: 执行器派发数应与触达执行器的调用一致: {executor.names}")


def test_recovery_visibility_and_message_position_controls():
    deps, llm, executor = _recovery_llm_executor()
    result = _run(deps, SRC_DEPS)
    # 恢复观察进入下一次实际模型输入（SRC_DEPS 无多余目标时恢复后即完成，
    # 用恢复消息本身验证可见性：它在 messages 中且位于本批全部 tool 结果之后）。
    recovery_index = next(
        (i for i, m in enumerate(result["messages"])
         if m.get("role") == "user"
         and str(m.get("content", "")).startswith(RECOVERY_OBSERVATION_NOTICE)), None)
    assert recovery_index is not None, "R3-06前置: 恢复消息应存在"
    last_tool_index = max(i for i, m in enumerate(result["messages"])
                          if m.get("role") == "tool")
    assert recovery_index > last_tool_index, (
        f"R3-06: 恢复事件必须在当前批所有tool结果之后: index={recovery_index}, last_tool={last_tool_index}")
    no_assistant_between = all(m.get("role") != "assistant"
                               for m in result["messages"][last_tool_index + 1:recovery_index])
    assert no_assistant_between, "R3-06: 恢复事件不得插入新assistant call"


# ---------------------------------------------------------------------------
# R3-07 / RA-15、RA-04-R2：恢复结果为受限状态时不得 complete
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("recovered_status", [{"status": "unavailable"},
                                              {"status": "commit_unknown"}],
                         ids=["unavailable", "commit-unknown"])
def test_recovered_limited_result_must_not_complete(recovered_status):
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
        return {"role": "assistant", "content": "全部完成"}
    llm = FakeLLMPolicy(policy)
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-fixed"),
        "advise_nutrition": dict(recovered_status),
    })
    result = _run(_deps(llm, executor), SRC_DEPS)
    assert result["status"] != "complete", (
        f"R3-07: 恢复返回 {recovered_status['status']} 后目标被covered且判complete: "
        f"{result['status']!r}, answer={result['final_answer']!r}")
    assert "全部完成" not in result["final_answer"], (
        f"R3-07: 受限恢复结果照转模型全部完成宣称: {result['final_answer']!r}")


def test_control_successful_recovery_completes_and_history_blocked_not_dragged():
    deps, _llm, executor = _recovery_llm_executor()
    result = _run(deps, SRC_DEPS)
    assert "advise_nutrition" in executor.names
    assert result["status"] == "complete", (
        f"R3-07对照: 成功恢复应complete且历史blocked不拖累: {result['status']!r}")


# ---------------------------------------------------------------------------
# R3-08 / RA-04-R2：写终态两条断言（状态与答复）与无绑定观察
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_commit", [
    {"status": "completed", "commit_state": "unknown"},
    {"status": "completed", "commit_state": "pending"},
    {"status": "completed"},
    {"status": "completed", "commit_state": "committed", "receipt_refs": [""]},
], ids=["unknown", "pending", "no-receipt", "illegal-receipt"])
def test_unconfirmed_write_both_status_and_answer_claims_hold(bad_commit):
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "record_nutrition",
                      {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]}
    ) if n == 1 else {"role": "assistant", "content": "全部保存成功"})
    executor = RecordingExecutor({"record_nutrition": bad_commit})
    result = _run(_deps(llm, executor), SRC_RECORD)
    assert result["status"] != "complete", (
        f"R3-08: {bad_commit} 状态仍判complete: {result['status']!r}")
    assert "全部保存成功" not in result["final_answer"], (
        f"R3-08: 答复转发了模型的全量成功宣称: {result['final_answer']!r}")
    assert "尚未确认" in result["final_answer"], (
        f"R3-08: 答复未明确说明提交尚未确认: {result['final_answer']!r}")


def test_mixed_committed_and_unconfirmed_writes():
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "record_nutrition",
                      {"source_ref": ref_of(SRC_DEPS, "帮我记录早餐吃了燕麦粥")}),
            tool_call("t2", "record_nutrition",
                      {"source_ref": ref_of(SRC_DEPS, "帮我记录早餐吃了鸡蛋")}),
            tool_call("t3", "advise_nutrition", {})]}
    ) if n == 1 else {"role": "assistant", "content": "全部保存成功"})

    def record_result(nth, name, arguments, context):
        # nth 为 0 基：第一次记录（t1 → task-1）返回可信回执，第二次（t2 → task-2）返回未知提交。
        return committed_receipt("rcpt-first") if nth == 0 else {
            "status": "completed", "commit_state": "unknown"}

    executor = RecordingExecutor({"record_nutrition": record_result})
    result = _run(_deps(llm, executor), SRC_DEPS)
    assert result["committed_tasks"] == ["task-1"], (
        f"R3-08: 已提交前项回执应保留、未确认项不得进入: {result['committed_tasks']}")
    assert result["status"] != "complete", (
        f"R3-08: 混合提交+未确认写不得判complete: {result['status']!r}")
    assert "advise_nutrition" not in executor.names, (
        f"R3-08: 未确认项不得解除建议依赖: {executor.names}")


def test_unbound_write_observation_cannot_prove_complete():
    """协议控制：无有效目标绑定的写观察不得成为完整成功证明。"""
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]),
        candidates=[])
    executor = RecordingExecutor({"record_nutrition": {
        "status": "completed", "commit_state": "unknown"}})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "全部保存成功"}),
        executor)))
    state.update(result)
    final = finalize_node(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "全部保存成功"})))
    # T6修订（round4裁定）：否定文案不得按子串判成功。语义断言：
    # status不为complete AND 答复非空 AND 不转发模型肯定句 AND 明确受限原因。
    # 原样运行证据（round4 asoriginal 日志）：实际答复为确定性免责文案
    # “部分记录的提交状态尚未确认，未作为保存成功处理；…”，属受限说明而非肯定成功。
    assert final["status"] != "complete", (
        f"T6: 无绑定写观察（completed+unknown）终态仍判complete: {final['status']!r}")
    assert final["final_answer"].strip(), (
        f"T6: 答复为空不能作为充分证据: {final['final_answer']!r}")
    assert "全部保存成功" not in final["final_answer"], (
        f"T6: 答复转发了模型肯定句: {final['final_answer']!r}")
    # round5分支语义：零派发/无有效目标绑定分支——应用事实 unbound_write_calls，
    # 原因说明为需要确认/补足目标绑定；不假称“提交尚未确认”（本分支无已执行观察）。
    assert executor.calls == [], (
        f"T6: 无绑定写应零派发: {executor.names}")
    assert state.get("unbound_write_calls") == ["t1"], (
        f"T6: 应用未记录无绑定写调用事实: {state.get('unbound_write_calls')}")
    assert ("确认" in final["final_answer"] or "补充" in final["final_answer"]), (
        f"T6: 答复未说明需要确认/补足目标绑定: {final['final_answer']!r}")
    assert "尚未确认" not in final["final_answer"], (
        f"T6: 零派发分支不得假称提交尚未确认: {final['final_answer']!r}")


def test_control_read_only_completed_completes():
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "answer_training", {})]}
    ) if n == 1 else {"role": "assistant", "content": "循序渐进即可。"})
    executor = RecordingExecutor({"answer_training": {"status": "completed"}})
    result = _run(_deps(llm, executor), "怎么练卧推")
    assert result["status"] == "complete" and result["final_answer"] == "循序渐进即可。"
