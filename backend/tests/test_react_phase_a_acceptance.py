"""阶段A独立验收：图级（invoke_react / build_react_graph 真实状态推进）。

对应任务 docs/react-phase-a-glm-acceptance-20261004.md：
- A01 轮次/终止、A02 同批依赖状态、A05 协议旁路（falsy tool_calls 路由）、
  A06 来源授权范围、A07 覆盖与重复位置；
- 原合同控制 1（动态决策）、5（真实adapter关闭）、6（依赖/覆盖可信性）、
  8（输入与覆盖故障）、9（响应与legacy）。

所有模型为 FakeLLM（记录入参/出参）、执行器为 RecordingExecutor（记录调用顺序
与可信context），无真实凭证、无真实网络。期望一律来自验收合同；失败断言原样
保留。标注「协议控制」的用例直接构造 state，仅用于精确边界覆盖，不代替真实
prepare/rule 候选证据。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.services.agent.react_graph import (
    ReactDependencies,
    execute_react_batch,
    invoke_react,
    prepare_node,
)
from app.services.agent.react_prompts import (
    RECOVERY_OBSERVATION_NOTICE as RECOVERY_NOTICE_PREFIX,
)

from tests.react_phase_a_support import (
    FakeLLM,
    RecordingExecutor,
    assistant,
    batch_state,
    committed_receipt,
    ref_of,
    tool_call,
)

SRC_RECORD = "帮我记录午餐吃了牛肉面和可乐"
SRC_DEPS = "帮我记录早餐吃了燕麦粥。然后帮我记录早餐吃了鸡蛋。然后根据早餐给我营养建议"


def _deps(llm, executor=None, **options):
    return ReactDependencies(llm=llm, tool_executor=executor, user_id="app-user-1",
                             conversation_id="conv-1", request_id="req-1", **options)


def _run(deps, message):
    return asyncio.run(invoke_react(deps, message))


def _tool_message_contents(result):
    return [json.loads(m["content"]) for m in result["messages"]
            if m.get("role") == "tool"]


def _obs_status(result, index=-1):
    return result["observations"][index]["status"]


# ---------------------------------------------------------------------------
# A01 轮次/终止：max_supervisor_turns=4 时不得有第5次模型调用
# ---------------------------------------------------------------------------
def test_a01_supervisor_turn_limit_caps_model_calls():
    always_tools = assistant("继续", tool_calls=[tool_call("t", "read_profile", {})])
    llm = FakeLLM(on_exhausted=always_tools)
    executor = RecordingExecutor({"read_profile": {"status": "completed"}})
    deps = _deps(llm, executor, max_supervisor_turns=4)
    result = _run(deps, "你好")
    assert len(llm.calls) <= 4, (
        f"A01: max_supervisor_turns=4 但发生了 {len(llm.calls)} 次模型调用 "
        f"（main_agent 不检查轮次, 上限只在 completion_check, 而持续 tool_calls "
        f"永远路由回 agent_tools 不经过 completion_check）")


def test_a01_turn_limit_termination_reason_is_supervisor_limit():
    always_tools = assistant("继续", tool_calls=[tool_call("t", "read_profile", {})])
    llm = FakeLLM(on_exhausted=always_tools)
    executor = RecordingExecutor({"read_profile": {"status": "completed"}})
    deps = _deps(llm, executor, max_supervisor_turns=4)
    result = _run(deps, "你好")
    assert result["termination_reason"] == "supervisor_turn_limit", (
        f"A01: 期望轮次上限终止, 实际 {result['termination_reason']!r} "
        f"（模型被调用 {len(llm.calls)} 次后由工具步数上限收尾）")


def test_a01_control_no_recursion_crash_and_graph_returns():
    """T2修订（复验合同§2）：原断言编码了RA-01旧bug行为（tool_step_limit收尾）。

    轮次门禁修复后：max_supervisor_turns=4 且持续 tool_calls，应先到主管轮次上限。
    保留 status=limited、图正常返回、无递归崩溃；补模型调用<=4与call-result配对。
    不把上限改成17，也不允许任意终止原因掩盖轮次失败。
    """
    always_tools = assistant("继续", tool_calls=[tool_call("t", "read_profile", {})])
    llm = FakeLLM(on_exhausted=always_tools)
    executor = RecordingExecutor({"read_profile": {"status": "completed"}})
    deps = _deps(llm, executor, max_supervisor_turns=4)
    result = _run(deps, "你好")
    assert result["status"] == "limited"
    assert result["termination_reason"] == "supervisor_turn_limit", (
        f"T2: 轮次上限终止原因应为supervisor_turn_limit, 实际 "
        f"{result['termination_reason']!r}（模型调用 {len(llm.calls)} 次）")
    assert len(llm.calls) <= 4, f"T2: 模型调用 {len(llm.calls)} 次超过 max_supervisor_turns=4"
    assistant_call_ids = [c.get("id") for m in result["messages"]
                          if m.get("role") == "assistant"
                          for c in (m.get("tool_calls") or []) if isinstance(c, dict)]
    tool_ids = [m.get("tool_call_id") for m in result["messages"] if m.get("role") == "tool"]
    assert sorted(assistant_call_ids) == sorted(tool_ids), (
        f"T2: call-result配对不完整: calls={assistant_call_ids}, tools={tool_ids}")


def test_a01_control_last_round_remaining_calls_get_paired_limited_results():
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "read_profile", {})]),
        assistant(tool_calls=[tool_call("t2a", "read_profile", {}),
                              tool_call("t2b", "read_profile", {})]),
    ])
    executor = RecordingExecutor({"read_profile": {"status": "completed"}})
    deps = _deps(llm, executor, max_tool_steps=2)
    result = _run(deps, "你好")
    tool_ids = [m.get("tool_call_id") for m in result["messages"] if m.get("role") == "tool"]
    assert sorted(tool_ids) == ["t1", "t2a", "t2b"], (
        f"A01对照: 末轮/剩余调用必须全部有配对tool消息: {tool_ids}")
    assert len(executor.calls) == 2, "A01对照: 只有预算内的2次真实执行"
    assert result["termination_reason"] == "tool_step_limit"


# ---------------------------------------------------------------------------
# A02 同批依赖状态：同批 record→advise 必须看到新提交状态
# ---------------------------------------------------------------------------
def test_a02_same_batch_advise_must_see_fresh_commit_state():
    record_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了燕麦粥")
    bare_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了鸡蛋")
    llm = FakeLLM(script=[assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": record_ref}),
        tool_call("t2", "record_nutrition", {"source_ref": bare_ref}),
        tool_call("t3", "advise_nutrition", {}),
    ]), assistant("按记录给建议")])
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-fixed"),
        "advise_nutrition": {"status": "completed", "result": {"advice": "蛋白充足"}},
    })
    result = _run(_deps(llm, executor), SRC_DEPS)
    assert executor.names.count("advise_nutrition") == 1, (
        f"A02: 同批 record 提交后 advise 应依赖新状态执行, 实际执行序列 {executor.names}, "
        f"observations={[(o['status'], o.get('error')) for o in result['observations']]}")


def test_a02_control_reverse_order_advise_first_is_blocked():
    """T5修订（round4裁定）：同批可信提交后自动恢复方向保留。

    模型批次 [t1 advise, t2 record, t3 record]：首条观察必须仍为
    blocked_dependency；两次记录可信提交后建议自动恢复执行恰好一次，
    最终执行序列 [record, record, advise]。同时证明：可信应用绑定/回执、
    执行器 context 看见本批最新提交、pending 清除、历史 blocked 不拖累终态、
    tool_steps==4 与执行器派发 3 次两口径、t1/t2/t3 各恰一个 tool 结果且 t1
    仍为 blocked、恢复事件引用 t1 且位于全部 tool 结果之后、无伪造消息、
    source_text 不变。
    """
    record_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了燕麦粥")
    bare_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了鸡蛋")
    llm = FakeLLM(script=[assistant(tool_calls=[
        tool_call("t1", "advise_nutrition", {}),
        tool_call("t2", "record_nutrition", {"source_ref": record_ref}),
        tool_call("t3", "record_nutrition", {"source_ref": bare_ref}),
    ]), assistant("好的")])
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-fixed"),
        "advise_nutrition": {"status": "completed", "result": {"advice": "x"}},
    })
    result = _run(_deps(llm, executor), SRC_DEPS)
    # 首条观察仍为 blocked_dependency（先阻塞语义保留）
    first_observation = result["observations"][0]
    assert first_observation["status"] == "blocked_dependency", (
        f"T5: 首条观察应为blocked_dependency: {first_observation}")
    # 最终执行序列：两次可信记录 + 恰好一次恢复的建议
    assert executor.names == ["record_nutrition", "record_nutrition", "advise_nutrition"], (
        f"T5: 最终执行序列应为[record, record, advise]: {executor.names}")
    # 两次写均有可信应用绑定、committed 与合法回执
    assert result["committed_tasks"] == ["task-1", "task-2"], (
        f"T5: 两次写应可信提交: {result['committed_tasks']}")
    assert len(result["bound_tasks"]) == 2 and all(
        b.get("payload", {}).get("write_authorization") == "authorized"
        for b in result["bound_tasks"]), f"T5: 应用绑定缺失: {result['bound_tasks']}"
    write_observations = [o for o in result["observations"] if o.get("receipt_refs")]
    assert len(write_observations) == 2 and all(
        o.get("commit_state") == "committed" for o in write_observations), (
        f"T5: 可信回执缺失: {write_observations}")
    # 建议恢复调用看见本批最新提交状态
    advise_context = next(c["context"] for c in executor.calls if c["name"] == "advise_nutrition")
    assert set(advise_context["committed_tasks"]) == {"task-1", "task-2"}, (
        f"T5: 恢复context未看见本批最新提交: {advise_context['committed_tasks']}")
    # 成功后 pending 清除、当前建议目标成功、历史 blocked 不拖累终态
    assert result["pending_goals"] == [], f"T5: pending未清除: {result['pending_goals']}"
    assert result["covered_goals"] == ["task-1", "task-2", "task-3"], (
        f"T5: 目标覆盖不完整: {result['covered_goals']}")
    assert result["status"] == "complete", f"T5: 终态被历史blocked拖累: {result['status']!r}"
    # 两口径不混用：tool_steps==4（三次普通含首个阻塞 + 一次恢复），执行器派发 3 次
    assert result["tool_steps"] == 4, (
        f"T5: tool_steps应为4（3普通+1恢复）: {result['tool_steps']}")
    assert len(executor.calls) == 3, f"T5: 执行器派发应为3次: {executor.names}"
    # t1/t2/t3 各有且仅有一个 tool 结果；t1 原结果仍是 blocked
    tool_results = {}
    for message in result["messages"]:
        if message.get("role") == "tool":
            tool_results.setdefault(message.get("tool_call_id"), []).append(message)
    assert sorted(tool_results) == ["t1", "t2", "t3"], f"T5: tool结果配对异常: {sorted(tool_results)}"
    assert all(len(v) == 1 for v in tool_results.values()), "T5: 存在重复tool结果"
    assert json.loads(tool_results["t1"][0]["content"])["status"] == "blocked_dependency", (
        f"T5: t1原tool结果应保持blocked: {tool_results['t1'][0]['content']}")
    # 恢复事件引用原 t1，位于该批全部 tool 结果之后，无伪造 assistant tool_calls
    recovery_messages = [m for m in result["messages"]
                         if m.get("role") == "user"
                         and str(m.get("content", "")).startswith(RECOVERY_NOTICE_PREFIX)]
    assert recovery_messages, "T5: 恢复事件消息缺失"
    events = json.loads(recovery_messages[-1]["content"][len(RECOVERY_NOTICE_PREFIX):])
    assert any(e.get("tool_call_id") == "t1" for e in events), f"T5: 恢复事件未引用t1: {events}"
    last_tool_index = max(i for i, m in enumerate(result["messages"]) if m.get("role") == "tool")
    recovery_index = result["messages"].index(recovery_messages[-1])
    assert recovery_index > last_tool_index, "T5: 恢复事件必须在全部tool结果之后"
    assert all(m.get("role") != "assistant"
               for m in result["messages"][last_tool_index + 1:recovery_index]), (
        "T5: 恢复事件前不得插入新assistant消息")
    # source_text 不被恢复数据改变
    assert result["source_text"] == SRC_DEPS, "T5: source_text被恢复数据改变"


def test_a02_t5_control_unknown_commit_keeps_advice_blocked():
    """T5对照：记录无可信提交证明时建议零派发（不能因自动恢复放宽提交语义）。"""
    record_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了燕麦粥")
    bare_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了鸡蛋")
    llm = FakeLLM(script=[assistant(tool_calls=[
        tool_call("t1", "advise_nutrition", {}),
        tool_call("t2", "record_nutrition", {"source_ref": record_ref}),
        tool_call("t3", "record_nutrition", {"source_ref": bare_ref}),
    ]), assistant("好的")])
    executor = RecordingExecutor({
        "record_nutrition": {"status": "completed", "commit_state": "unknown"},
        "advise_nutrition": {"status": "completed", "result": {"advice": "x"}},
    })
    result = _run(_deps(llm, executor), SRC_DEPS)
    assert "advise_nutrition" not in executor.names, (
        f"T5对照: 未知提交不得触发建议恢复派发: {executor.names}")


def test_a02_t5_control_budget_three_steps_zero_recovery_dispatch():
    """T5对照：预算只够三步（三次普通调用）时恢复零派发，不放宽预算凑通过。"""
    record_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了燕麦粥")
    bare_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了鸡蛋")
    llm = FakeLLM(script=[assistant(tool_calls=[
        tool_call("t1", "advise_nutrition", {}),
        tool_call("t2", "record_nutrition", {"source_ref": record_ref}),
        tool_call("t3", "record_nutrition", {"source_ref": bare_ref}),
    ]), assistant("好的")])
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-fixed"),
        "advise_nutrition": {"status": "completed", "result": {"advice": "x"}},
    })
    result = _run(_deps(llm, executor, max_tool_steps=3), SRC_DEPS)
    assert "advise_nutrition" not in executor.names, (
        f"T5对照: 预算耗尽恢复应零派发: {executor.names}")
    assert result["tool_steps"] == 3, f"T5对照: 计步应与三次普通调用一致: {result['tool_steps']}"


def test_a02_control_advise_after_commits_across_turns_passes():
    record_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了燕麦粥")
    bare_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了鸡蛋")
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "record_nutrition", {"source_ref": record_ref}),
                              tool_call("t2", "record_nutrition", {"source_ref": bare_ref})]),
        assistant(tool_calls=[tool_call("t3", "advise_nutrition", {})]),
        assistant("已给建议"),
    ])
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-fixed"),
        "advise_nutrition": {"status": "completed", "result": {"advice": "蛋白充足"}},
    })
    result = _run(_deps(llm, executor), SRC_DEPS)
    assert executor.names == ["record_nutrition", "record_nutrition", "advise_nutrition"], (
        f"A02对照: 跨轮先记录后建议应放行: {executor.names}")
    assert result["status"] == "complete"


def test_a02_control_missing_or_unknown_commit_never_releases_dependency():
    """记录返回 completed 但 commit_state=unknown：跨轮建议仍须阻塞（另见 A04）。"""
    record_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了燕麦粥")
    bare_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了鸡蛋")
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "record_nutrition", {"source_ref": record_ref}),
                              tool_call("t2", "record_nutrition", {"source_ref": bare_ref})]),
        assistant(tool_calls=[tool_call("t3", "advise_nutrition", {})]),
        assistant("结束"),
    ])
    executor = RecordingExecutor({
        "record_nutrition": {"status": "completed", "commit_state": "unknown"},
        "advise_nutrition": {"status": "completed", "result": {"advice": "x"}},
    })
    result = _run(_deps(llm, executor), SRC_DEPS)
    assert "advise_nutrition" not in executor.names, (
        f"A02对照: missing/unknown提交不得放行建议依赖: {executor.names}, "
        f"committed_tasks={result['committed_tasks']}")


# ---------------------------------------------------------------------------
# A04 提交语义（跨轮）：completed 无提交证明仍标记 committed 并放行建议
# ---------------------------------------------------------------------------
def test_a04_completed_unknown_commit_releases_advice_defect():
    record_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了燕麦粥")
    bare_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了鸡蛋")
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "record_nutrition", {"source_ref": record_ref}),
                              tool_call("t2", "record_nutrition", {"source_ref": bare_ref})]),
        assistant(tool_calls=[tool_call("t3", "advise_nutrition", {})]),
        assistant("结束"),
    ])
    executor = RecordingExecutor({
        "record_nutrition": {"status": "completed", "commit_state": "unknown"},
        "advise_nutrition": {"status": "completed", "result": {"advice": "x"}},
    })
    result = _run(_deps(llm, executor), SRC_DEPS)
    assert "advise_nutrition" not in executor.names, (
        f"A04: completed但commit_state=unknown的写观察不应放行依赖建议, "
        f"实际执行 {executor.names}")


def test_a04_completed_unknown_commit_marks_committed_tasks_defect():
    record_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了燕麦粥")
    bare_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了鸡蛋")
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "record_nutrition", {"source_ref": record_ref})]),
        assistant("结束"),
    ])
    executor = RecordingExecutor({
        "record_nutrition": {"status": "completed", "commit_state": "unknown"},
    })
    result = _run(_deps(llm, executor), SRC_DEPS)
    assert result["committed_tasks"] == [], (
        f"A04: 无提交证明的completed写观察混入committed_tasks: {result['committed_tasks']}")



def test_a04_control_final_status_complete_for_legal_committed_run():
    record_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了燕麦粥")
    bare_ref = ref_of(SRC_DEPS, "帮我记录早餐吃了鸡蛋")
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "record_nutrition", {"source_ref": record_ref}),
                              tool_call("t2", "record_nutrition", {"source_ref": bare_ref})]),
        assistant(tool_calls=[tool_call("t3", "advise_nutrition", {})]),
        assistant("完成"),
    ])
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-fixed"),
        "advise_nutrition": {"status": "completed", "result": {"advice": "x"}},
    })
    result = _run(_deps(llm, executor), SRC_DEPS)
    assert result["status"] == "complete"
    assert result["committed_tasks"]


# ---------------------------------------------------------------------------
# A05 协议旁路：falsy tool_calls 不能当作无工具完成
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("falsy", [{}, "", 0, False],
                         ids=["empty-dict", "empty-string", "zero", "false"])
def test_a05_falsy_tool_calls_must_be_protocol_rejected(falsy):
    llm = FakeLLM(on_exhausted=assistant("好的", tool_calls=falsy))
    executor = RecordingExecutor()
    result = _run(_deps(llm, executor), "你好")
    assert result["status"] == "restricted", (
        f"A05: tool_calls={falsy!r} 被当作无工具完成: status={result['status']!r}, "
        f"reason={result.get('termination_reason')!r}, protocol_error="
        f"{result.get('protocol_error')!r}（路由按真值分流, 容器校验从未运行）")
    assert result.get("protocol_error"), "A05: 畸形 tool_calls 必须产生协议错误标记"
    assert executor.calls == []


def test_a05_control_empty_list_and_missing_are_legal_completion():
    for response in [assistant("好的", tool_calls=[]), assistant("好的")]:
        # 正常合成请求仍须提供有来源绑定的 safety；“你好”没有业务目标，
        # 只验证容器/路由的合法完成。
        llm = FakeLLM(on_exhausted=response)
        result = _run(_deps(llm, RecordingExecutor()), "你好")
        assert result["status"] == "complete", (
            f"A05对照: 合法无工具完成被误判: {response.get('tool_calls')!r} -> {result['status']!r}")
        assert result["protocol_error"] == ""
        assert len(llm.calls) == 1


def test_a05_control_malformed_openai_arguments_json_is_restricted_end_to_end():
    llm = FakeLLM(on_exhausted=assistant(tool_calls=[
        {"id": "t1", "type": "function",
         "function": {"name": "read_profile", "arguments": "{bad"}}]))
    executor = RecordingExecutor()
    result = _run(_deps(llm, executor), "你好")
    assert result["status"] == "restricted"
    assert result["termination_reason"] == "protocol_error"
    assert executor.calls == []


# ---------------------------------------------------------------------------
# A06 来源授权范围：独立子句不得借授权/不得拒绝合法记录
# ---------------------------------------------------------------------------
def test_a06_clause_without_command_must_not_borrow_authorization():
    source = "帮我记录早餐吃了鸡蛋。午餐吃了沙拉。"
    llm = FakeLLM(script=[assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": ref_of(source, "午餐吃了沙拉")})]),
        assistant("结束")])
    executor = RecordingExecutor({"record_nutrition": committed_receipt()})
    result = _run(_deps(llm, executor), source)
    assert executor.calls == [], (
        f"A06: 无写命令子句借到授权并被派发: {executor.names}, "
        f"observations={[(o['status'], o.get('error')) for o in result['observations']]}")


def test_a06_independent_negation_clause_must_not_reject_legal_record():
    source = "帮我记录早餐吃了鸡蛋。体重数据不要记录。"
    llm = FakeLLM(script=[assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": ref_of(source, "帮我记录早餐吃了鸡蛋")})]),
        assistant("结束")])
    executor = RecordingExecutor({"record_nutrition": committed_receipt()})
    result = _run(_deps(llm, executor), source)
    assert executor.names == ["record_nutrition"], (
        f"A06: 独立否定子句拒绝了另一合法记录: "
        f"observations={[(o['status'], o.get('error')) for o in result['observations']]}, "
        f"executor={executor.names}")


@pytest.mark.parametrize("source,needle", [
    ("今天吃了蛋糕,不要保存", "今天吃了蛋糕"),
    ("不要记录今天跑步5公里", "今天跑步5公里"),
    ("如果明天下雨我就在家练俯卧撑30分钟", "在家练俯卧撑30分钟"),
    ("朋友昨天练了卧推5组5次", "练了卧推5组5次"),
], ids=["tail-negation", "truncated-negation", "hypothetical", "third-party"])
def test_a06_unsafe_spans_still_rejected_with_zero_dispatch(source, needle):
    llm = FakeLLM(script=[assistant(tool_calls=[
        tool_call("t1", "record_nutrition", {"source_ref": ref_of(source, needle)})]),
        assistant("结束")])
    executor = RecordingExecutor()
    result = _run(_deps(llm, executor), source)
    observation = result["observations"][0]
    assert observation["status"] == "rejected", (
        f"A06对照: {source} 应拒绝, 实际 {observation}")
    assert executor.calls == [], f"A06对照: 拒绝前不得派发: {executor.names}"
    assert result["bound_tasks"] == [], "A06对照: 拒绝的写请求不得绑定 task_spec"


# ---------------------------------------------------------------------------
# A07 覆盖与重复位置
# ---------------------------------------------------------------------------
def test_a07_control_second_goal_stays_unhandled_after_partial_claim():
    source = "帮我记录早餐吃了鸡蛋。帮我记录午餐吃了牛肉面。"
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "record_nutrition",
                                        {"source_ref": ref_of(source, "帮我记录早餐吃了鸡蛋")})]),
        assistant("已全部保存成功"),
        assistant("真的完成了"),
        assistant("完成"),
    ])
    executor = RecordingExecutor({"record_nutrition": committed_receipt()})
    result = _run(_deps(llm, executor), source)
    uncovered = [c["id"] for c in result["intent_candidates"]
                 if c["id"] not in result["covered_goals"]]
    assert uncovered, f"A07对照: 未处理目标必须保持未覆盖: covered={result['covered_goals']}"
    assert result["status"] == "limited"
    assert result["final_answer"] != "已全部保存成功"


def test_a07_control_advice_without_source_ref_covers_only_first_candidate():
    source = "今天午餐吃什么好？晚餐吃什么好？"
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "advise_nutrition", {})]),
        assistant("完成"),
    ])
    executor = RecordingExecutor({"advise_nutrition": {"status": "completed"}})
    result = _run(_deps(llm, executor), source)
    assert result["covered_goals"] == ["task-1"], (
        f"A07对照: 无source_ref建议按类型取首候选: {result['covered_goals']}")
    assert "task-2" not in result["covered_goals"]


def test_a07_repeat_unbound_advice_dispatches_duplicate_and_never_serves_second():
    """合同A07「建议未绑定原文不能随意覆盖目标」：重复无ref调用重复派发同一目标。"""
    source = "今天午餐吃什么好？晚餐吃什么好？"
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "advise_nutrition", {})]),
        assistant(tool_calls=[tool_call("t2", "advise_nutrition", {})]),
        assistant("完成"),
    ])
    executor = RecordingExecutor({"advise_nutrition": {"status": "completed"}})
    result = _run(_deps(llm, executor), source)
    assert len(executor.calls) <= 1, (
        f"A07: 未绑定原文的建议被重复派发 {len(executor.calls)} 次且第二个目标"
        f"仍未服务: covered={result['covered_goals']}, names={executor.names}")


def test_a07_identical_text_second_position_never_binds_and_redispatches():
    """协议控制用例（非真实prepare候选）：相同文字两个出现位置, find首匹配回退。"""
    source = "早上跑步5公里,晚上也跑步5公里"
    candidates = [
        {"id": "goal-a", "task_type": "workout_record", "action": "record",
         "text": "跑步5公里", "source_start": -1, "source_end": -1, "depends_on": []},
        {"id": "goal-b", "task_type": "workout_record", "action": "record",
         "text": "跑步5公里", "source_start": -1, "source_end": -1, "depends_on": []},
    ]
    second_ref = ref_of(source, "跑步5公里", occurrence=1)
    executor = RecordingExecutor({"record_workout": committed_receipt()})
    deps = _deps(FakeLLM(), executor)

    async def two_batches():
        state = batch_state(source, assistant(tool_calls=[
            tool_call("t1", "record_workout", {"source_ref": second_ref})]),
            candidates=candidates)
        first = await execute_react_batch(state, deps)
        state.update(first)
        state["messages"].append(assistant(tool_calls=[
            tool_call("t2", "record_workout", {"source_ref": second_ref})]))
        second = await execute_react_batch(state, deps)
        return first, second, state

    first, second, state = asyncio.run(two_batches())
    assert first["covered_goals"] or second["covered_goals"], (
        "A07: 指向第二个出现位置的调用必须绑定到对应目标（find首匹配回退使覆盖永远为空: "
        f"covered={state['covered_goals']}, executor={executor.names}）")
    assert len(executor.calls) <= 1, (
        f"A07: 相同文字第二位置被重复派发: {executor.names}, "
        f"covered={state['covered_goals']}")


def test_a07_blocked_advice_must_not_be_permanently_swallowed_by_later_commit():
    source = SRC_DEPS
    record_ref = ref_of(source, "帮我记录早餐吃了燕麦粥")
    bare_ref = ref_of(source, "帮我记录早餐吃了鸡蛋")
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "advise_nutrition", {})]),
        assistant(tool_calls=[tool_call("t2", "record_nutrition", {"source_ref": record_ref}),
                              tool_call("t3", "record_nutrition", {"source_ref": bare_ref})]),
        assistant("都完成了"),
    ])
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-fixed"),
        "advise_nutrition": {"status": "completed", "result": {"advice": "x"}},
    })
    result = _run(_deps(llm, executor), source)
    advice_id = "task-3"
    assert advice_id in result["covered_goals"], (
        f"A07机制证据: blocked_dependency被当作已处理: {result['covered_goals']}")
    assert "advise_nutrition" in executor.names, (
        f"A07: 先阻塞后记录提交不能永久吞掉仍可执行的建议——提交完成后循环直接"
        f"以goals_complete终止(最终status={result['status']!r}), 建议从未被重新"
        f"提供或执行: executor={executor.names}")


# ---------------------------------------------------------------------------
# 合同控制1：动态决策（第二个响应检查上一个tool结果后选择不同工具）
# ---------------------------------------------------------------------------
def _nutrition_observation(value):
    return {"read_nutrition_summary": {"status": "completed",
                                       "result": {"protein_avg_g": value}}}


def _dynamic_policy(nth_call, messages):
    if nth_call == 1:
        return {"role": "assistant", "content": "",
                "tool_calls": [tool_call("t1", "read_nutrition_summary", {"days": 7})]}
    if nth_call == 2:
        last = messages[-1]
        observation = json.loads(last["content"]) if last.get("role") == "tool" else {}
        protein = (observation.get("result") or {}).get("protein_avg_g")
        if protein == 60:
            nxt = "advise_nutrition"
        elif protein == 180:
            nxt = "read_workout_history"
        else:
            nxt = "read_profile"
        return {"role": "assistant", "content": "", "tool_calls": [tool_call("t2", nxt, {})]}
    return {"role": "assistant", "content": "根据结果处理完毕"}


def test_contract1_dynamic_decision_selects_second_tool_from_observation():
    source = "你好"
    sequences = {}
    for label, protein in (("low", 60), ("high", 180)):
        llm = FakeLLM(policy=_dynamic_policy)
        executor = RecordingExecutor(_nutrition_observation(protein) | {
            "advise_nutrition": {"status": "completed", "result": {"advice": "补蛋白"}},
            "read_workout_history": {"status": "completed", "result": {"days": 30}},
        })
        result = _run(_deps(llm, executor), source)
        sequences[label] = llm.tool_call_sequence
        assert llm.calls[1]["messages"][-1]["role"] == "tool", (
            "合同1: 第二次模型调用的输入必须包含上一个tool结果")
        assert json.loads(llm.calls[1]["messages"][-1]["content"])["result"]["protein_avg_g"] == protein
    assert sequences["low"][1] == ["advise_nutrition"], f"合同1: low蛋白应转建议: {sequences}"
    assert sequences["high"][1] == ["read_workout_history"], f"合同1: high蛋白应转历史: {sequences}"
    assert sequences["low"] != sequences["high"]


# ---------------------------------------------------------------------------
# 合同控制5：真实写 adapter 关闭（不注入 executor）
# ---------------------------------------------------------------------------
def test_contract5_production_write_adapter_blocked_and_read_unavailable():
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "record_nutrition",
                                        {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]),
        assistant(tool_calls=[tool_call("t2", "read_profile", {})]),
        assistant("都处理好了"),
    ])
    result = _run(_deps(llm, None), SRC_RECORD)
    record_observation = result["observations"][0]
    assert record_observation["status"] == "blocked_capability", (
        f"合同5: 生产写adapter必须显式阻断: {record_observation}")
    assert record_observation["error"] == "production_write_adapter_blocked"
    assert result["observations"][1]["status"] == "unavailable"
    assert result["bound_tasks"], "合同5: 写候选应绑定应用分配的 task_spec"
    assert result["bound_tasks"][0]["side_effect"] is True
    assert result["status"] == "blocked"
    assert result["final_answer"] != "都处理好了"
    assert "保存" not in result["final_answer"] or "未完成" in result["final_answer"]


def test_contract5_react_modules_do_not_import_worker_gateway_or_mcp():
    module_dir = Path(__file__).resolve().parents[1] / "app" / "services" / "agent"
    forbidden = ["app.services.worker", "app.services.mcp",
                 "from app.services.llm.client", "import httpx", "import requests"]
    for module in ["react_graph.py", "react_models.py", "react_tools.py", "react_prompts.py"]:
        text = (module_dir / module).read_text(encoding="utf-8")
        hit = [token for token in forbidden if token in text]
        assert not hit, f"合同5: {module} 引入了 {hit}"


# ---------------------------------------------------------------------------
# 合同控制6：无进展停止 / 状态呈现 / 恶意文字不可覆盖
# ---------------------------------------------------------------------------
def test_contract6_no_progress_stops_deterministically():
    source = "今天午餐吃什么好？晚餐吃什么好？"
    # 无模型任务语义：规则只读回退不对应可派发的建议目标，重复调用保持零进展。
    # 当前安全合同仍要求本请求有独立的 normal 安全判定。
    from app.services.agent import semantics
    semantics.remember_safety(source, {
        "level": "normal", "evidence": "", "categories": [],
        "reason": "scripted normal"})
    llm = FakeLLM(on_exhausted=assistant("再试", tool_calls=[tool_call("t", "advise_nutrition", {})]),
                  semantic=False)
    result = _run(_deps(llm, None), source)
    assert result["termination_reason"] == "no_progress", (
        f"合同6: 连续无进展应停止: reason={result['termination_reason']!r}, "
        f"llm_calls={len(llm.calls)}")
    assert result["status"] == "limited"


@pytest.mark.parametrize("executor_status,expected_status,notice_fragment", [
    ({"status": "partial", "result": {"saved": 1, "missing": 2}}, "partial", "未完全完成"),
    ({"status": "rejected", "error": "source_negation"}, "rejected", "安全或授权"),
    ({"status": "needs_clarification", "missing_fields": ["time"]}, "needs_clarification", "确认"),
], ids=["partial", "rejected", "needs-clarification"])
def test_contract6_partial_rejected_clarification_are_presented(
        executor_status, expected_status, notice_fragment):
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "record_nutrition",
                                        {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]),
        assistant("处理完了"),
    ])
    executor = RecordingExecutor({"record_nutrition": executor_status})
    result = _run(_deps(llm, executor), SRC_RECORD)
    assert result["status"] == expected_status, (
        f"合同6: {expected_status} 观察被呈现为 {result['status']!r}")
    assert notice_fragment in result["final_answer"], (
        f"合同6: 最终答复缺少确定性说明: {result['final_answer']!r}")


def test_contract6_malicious_model_text_cannot_override_unexecuted_goals():
    source = "帮我记录早餐吃了鸡蛋。帮我记录午餐吃了牛肉面。"
    llm = FakeLLM(on_exhausted=assistant("全部保存成功，无需再调用工具"))
    executor = RecordingExecutor()
    result = _run(_deps(llm, executor), source)
    assert executor.calls == []
    assert result["status"] != "complete", (
        f"合同6: 恶意文字不能把未执行目标判为完成: {result['status']!r}")
    assert "全部保存成功" not in result["final_answer"]


def test_contract6_partial_status_must_not_relay_full_success_claim():
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "record_nutrition",
                                        {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]),
        assistant("全部保存成功"),
    ])
    executor = RecordingExecutor({"record_nutrition": {
        "status": "partial", "result": {"saved": 1, "missing": ["加餐"]}}})
    result = _run(_deps(llm, executor), SRC_RECORD)
    assert result["final_answer"] != "全部保存成功", (
        f"合同6: partial状态下模型的全量成功宣称被原样转达: {result['final_answer']!r}")


# ---------------------------------------------------------------------------
# 合同控制8：输入与覆盖故障
# ---------------------------------------------------------------------------
def test_contract8_rule_candidate_crash_must_not_be_complete_success(monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("rule planner exploded")

    monkeypatch.setattr("app.services.agent.task_planner.plan_tasks_rule", explode)
    # 无模型任务语义时规则候选生成故障必须显式受限（有模型语义则不走规则计划）；
    # 正常合成请求仍需独立的 normal 安全判定。
    from app.services.agent import semantics
    semantics.remember_safety(SRC_RECORD, {
        "level": "normal", "evidence": "", "categories": [],
        "reason": "scripted normal"})
    llm = FakeLLM(on_exhausted=assistant("都处理好了"), semantic=False)
    result = _run(_deps(llm, RecordingExecutor()), SRC_RECORD)
    assert result["status"] != "complete", (
        f"合同8: 规则候选生成抛异常被当作空目标完整成功: status={result['status']!r}, "
        f"final={result['final_answer']!r}")


def test_contract8_urgent_input_candidates_keep_safety_gate():
    source = "胸痛得厉害,帮我记录今天心率120"
    deps = _deps(FakeLLM(), RecordingExecutor())
    state = asyncio.run(_prepare_state(deps, source))
    candidates = state["intent_candidates"]
    record_candidates = [c for c in candidates if c.get("task_type") == "nutrition_record"]
    assert record_candidates, f"合同8: 急症输入应产生记录候选: {candidates}"
    assert all(c.get("write_authorization") == "rejected_unauthorized"
               for c in record_candidates), (
        f"合同8: 安全规则源码未变——急症写候选应被拒绝: "
        f"{[c.get('write_authorization') for c in record_candidates]}")
    assert any(c.get("task_type") == "safety_triage" for c in candidates), (
        f"合同8: 急症输入应有安全分流候选: {candidates}")


async def _prepare_state(deps, source):
    # 与 invoke_react 相同的请求前置：先取得本轮结构化语义，再进入 prepare。
    from app.services.agent.semantics import understand

    await understand(deps.llm, source)
    return prepare_node({"raw_message": source}, deps)


def test_contract8_urgent_input_must_not_dispatch_normal_record_tool():
    source = "胸痛得厉害,帮我记录今天心率120"
    llm = FakeLLM(script=[assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": ref_of(source, "帮我记录今天心率120")})]),
        assistant("结束")])
    executor = RecordingExecutor({"record_nutrition": committed_receipt()})
    result = _run(_deps(llm, executor), source)
    assert executor.calls == [], (
        f"合同8: 急症输入不应触发普通记录fake派发: {executor.names}, "
        f"observations={[(o['status'], o.get('error')) for o in result['observations']]}")


def test_contract8_control_triage_goal_cannot_be_covered_by_qa_tool():
    source = "胸痛得厉害,帮我记录今天心率120"
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "answer_training", {})]),
        assistant("结束"),
    ])
    executor = RecordingExecutor({"answer_training": {"status": "completed"}})
    result = _run(_deps(llm, executor), source)
    triage = [c for c in result["intent_candidates"] if c.get("task_type") == "safety_triage"]
    assert triage
    assert all(c["id"] not in result["covered_goals"] for c in triage), (
        f"合同8对照: 安全分流目标不得被普通问答工具覆盖: {result['covered_goals']}")


# ---------------------------------------------------------------------------
# 合同控制9：响应解释准确 + legacy 未切换
# ---------------------------------------------------------------------------
def test_contract9_legacy_graph_and_http_do_not_import_react_modules():
    app_dir = Path(__file__).resolve().parents[1] / "app"
    targets = [app_dir / "main.py"]
    targets.extend((app_dir / "api").glob("*.py"))
    targets.extend([
        app_dir / "services" / "agent" / "multi_agent_graph.py",
        app_dir / "services" / "agent" / "workout_graph.py",
        app_dir / "services" / "agent" / "agent.py",
        app_dir / "services" / "agent" / "task_planner.py",
    ])
    react_tokens = ("react_graph", "react_tools", "react_models", "react_prompts",
                    "build_react_graph", "invoke_react")
    for path in targets:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        hit = [token for token in react_tokens if token in text]
        assert not hit, f"合同9: legacy 文件 {path.name} 引用了 react 模块: {hit}"


def test_contract9_later_batch_protocol_failure_must_not_claim_zero_operations():
    llm = FakeLLM(script=[
        assistant(tool_calls=[tool_call("t1", "record_nutrition",
                                        {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)})]),
        assistant(tool_calls=[tool_call("t2", "read_profile", {}),
                              tool_call("t2", "read_profile", {})]),  # 重复ID
        assistant("结束"),
    ])
    executor = RecordingExecutor({"record_nutrition": committed_receipt()})
    result = _run(_deps(llm, executor), SRC_RECORD)
    assert executor.names == ["record_nutrition"], "前置: 第一批记录已实际执行"
    assert result["status"] == "restricted"
    assert "未执行任何领域操作" not in result["final_answer"], (
        f"合同9: 后批协议故障被解释为整请求零操作: {result['final_answer']!r}")


def test_contract9_model_failure_explanation_must_name_model_not_protocol():
    class ExplodingLLM:
        async def complete_message(self, messages, temperature=0.0, tools=None, tool_choice=None):
            raise ConnectionError("mock transport down")

    llm = FakeLLM()
    llm.complete_message = ExplodingLLM().complete_message
    result = _run(_deps(llm, RecordingExecutor()), SRC_RECORD)
    assert result["status"] == "restricted"
    assert "协议" not in result["final_answer"], (
        f"合同9: 模型故障被解释成协议异常: {result['final_answer']!r}")
