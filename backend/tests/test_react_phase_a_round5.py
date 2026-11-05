"""ReAct 阶段A第五轮复验：R5-01…R5-04 有限补证（round5 任务 §3）。

绑定 DeepSeek 001252 最终实现。只落实第五轮修复合同的既有边界与此前空验证；
标注「协议控制」的用例实际注入被测节点，不经 prepare 重生成候选。期望来自
第五轮修复合同 §3；失败按原 RA 归属登记，原样保留。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.services.agent.react_graph import (
    ReactDependencies,
    _execute_call,
    execute_react_batch,
    finalize_node,
    invoke_react,
)
from app.services.agent.react_models import DomainObservation, ToolCall, ToolStatus

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
SRC_TWO_RECORDS = "早餐吃了燕麦粥,帮我记录。晚餐吃了鸡胸肉,也帮我记录"
SRC_URGENT = "胸痛得厉害,帮我记录今天心率120"
SRC_MIXED = "早上跑步5公里,晚上也跑步5公里"
SRC_TWO_CLAUSE = "帮我记录早餐吃了燕麦,午餐吃了米饭"

BOUND_CANDIDATES = [{
    "id": "task-1", "task_type": "nutrition_record", "action": "record",
    "text": SRC_RECORD, "source_start": 0, "source_end": len(SRC_RECORD), "depends_on": [],
}]


def _deps(llm, executor=None, **options):
    return ReactDependencies(llm=llm, tool_executor=executor, user_id="app-user-1",
                             conversation_id="conv-1", request_id="req-5", **options)


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
# R5-01 / RA-08：整份非法观察必须受限，不能仅清理字段
# ---------------------------------------------------------------------------
ILLEGAL_METADATA_PAYLOADS = [
    {"summary": float("nan")},
    {"missing_fields": [{1, 2}]},
    {"task_id": {}},
    {"context_version": True},
    {"error": {}},
]


def _write_call_state(candidates):
    return batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}})]),
        candidates=candidates)


def _run_bound_write(executor):
    state = _write_call_state(BOUND_CANDIDATES)
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    return result


@pytest.mark.parametrize("illegal", ILLEGAL_METADATA_PAYLOADS,
                         ids=["summary-nan", "missing-fields-set", "task-id-dict",
                              "context-version-bool", "error-dict"])
def test_r5_01_illegal_metadata_with_commit_claim_must_be_restricted(illegal):
    """合同RA-08-R4：非法元数据的整份观察必须受限（非completed）、不保留提交证明，
    不得进入 committed_tasks。dict 返回经真实执行器与返回边界 normalize。"""
    payload = {"status": "completed", "commit_state": "committed",
               "receipt_refs": ["r1"], "result": {"receipt": "r1"}}
    payload.update(illegal)
    executor = RecordingExecutor({"record_nutrition": payload})
    result = _run_bound_write(executor)
    assert len(executor.calls) == 1, f"R5-01前置: 应真实派发一次: {executor.names}"
    observation = result["observations"][0]
    assert observation["status"] != "completed", (
        f"R5-01: 非法元数据 {illegal} 未降级状态: {observation}")
    # 第六轮加强：commit_state/receipt_refs 清证明、安全错误原因
    assert not observation.get("commit_state") and not observation.get("receipt_refs"), (
        f"R5-01: 非法元数据未清提交证明字段: {observation}")
    assert observation.get("error") in {"invalid_summary", "invalid_missing_fields",
                                        "invalid_task_id", "invalid_context_version",
                                        "invalid_error", "result_not_object",
                                        "result_not_serializable"}, (
        f"R5-01: 错误原因不安全/缺失: {observation.get('error')!r}")
    assert result["committed_tasks"] == [], (
        f"R5-01: 非法元数据 {illegal} 未清提交证明: {result['committed_tasks']}")
    for message in result["messages"]:
        if message.get("role") == "tool":
            json.loads(message["content"], parse_constant=lambda v: (_ for _ in ()).throw(
                ValueError(f"非有限JSON: {v}")))


def test_r5_01_illegal_metadata_direct_mutated_form_restricted():
    """协议控制：构造后突变 direct 返回（summary=NaN）经返回边界 normalize 后同样受限。"""
    observation = DomainObservation(status=ToolStatus.COMPLETED, commit_state="committed",
                                    receipt_refs=["r1"], result={"receipt": "r1"})
    observation.summary = float("nan")  # 返回前突变
    executor = RecordingExecutor({"record_nutrition": observation})
    result = _run_bound_write(executor)
    observation_out = result["observations"][0]
    assert observation_out["status"] != "completed", (
        f"R5-01: direct突变summary=NaN未降级: {observation_out}")
    assert not observation_out.get("commit_state") and not observation_out.get("receipt_refs"), (
        f"R5-01: direct突变未清提交证明字段: {observation_out}")
    assert observation_out.get("error") in {"invalid_summary", "result_not_object",
                                            "result_not_serializable"}, (
        f"R5-01: direct突变错误原因不安全: {observation_out.get('error')!r}")
    assert result["committed_tasks"] == [], (
        f"R5-01: direct突变summary=NaN未清提交证明: {result['committed_tasks']}")


def test_r5_01_illegal_metadata_batch_prior_item_kept_and_pairing_intact():
    """同批合法前项保留、独立后项在原预算内继续、每个原call恰一tool结果。"""
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "read_profile", {}),
        tool_call("t2", "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}})]),
        candidates=BOUND_CANDIDATES)
    executor = RecordingExecutor({
        "read_profile": {"status": "completed", "result": {"profile": "ok"}},
        "record_nutrition": {"status": "completed", "commit_state": "committed",
                             "receipt_refs": ["r1"], "summary": float("nan")},
    })
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert len(result["observations"]) == 2, f"R5-01: 前后项都应完成: {result['observations']}"
    assert result["observations"][0]["status"] == "completed", "R5-01: 合法前项应保留"
    tool_ids = [m.get("tool_call_id") for m in result["messages"] if m.get("role") == "tool"]
    assert sorted(tool_ids) == ["t1", "t2"], f"R5-01: 配对不完整: {tool_ids}"


def _recovery_state_deps(advice_result):
    """turn1 advice-c1（deps task-1 缺，blocked）→ turn2 record 提交 → 自动恢复返回
    携带 summary=NaN 的建议观察。合成 advice 候选（真实来源显式区间）。"""
    candidates = [
        {"id": "task-1", "task_type": "nutrition_record", "action": "record",
         "text": "早餐吃了燕麦粥", "source_start": 0, "source_end": 7, "depends_on": []},
        {"id": "goal-c1", "task_type": "nutrition_advice", "action": "question",
         "text": "帮我记录", "source_start": 8, "source_end": 12, "depends_on": ["task-1"]},
    ]

    def policy(nth, messages):
        if nth == 1:
            return {"role": "assistant", "content": "", "tool_calls": [
                tool_call("t1", "advise_nutrition",
                          {"source_ref": {"start": 8, "end": 12, "exact_text": "帮我记录"}})]}
        if nth == 2:
            return {"role": "assistant", "content": "", "tool_calls": [
                tool_call("t2", "record_nutrition",
                          {"source_ref": ref_of(SRC_TWO_RECORDS, "早餐吃了燕麦粥")})]}
        return {"role": "assistant", "content": "都完成了"}

    llm = FakeLLMPolicy(policy)
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-fixed"),
        "advise_nutrition": advice_result,
    })
    deps = _deps(llm, executor, max_tool_steps=16)
    state = batch_state(SRC_TWO_RECORDS, assistant(tool_calls=[
        tool_call("t1", "advise_nutrition",
                  {"source_ref": {"start": 8, "end": 12, "exact_text": "帮我记录"}})]),
        candidates=candidates)
    state.update(asyncio.run(execute_react_batch(state, deps)))  # turn1: 阻塞入pending
    return state, deps, executor


def test_r5_01_recovery_returning_illegal_metadata_not_covered_not_complete():
    """恢复返回携带 summary=NaN 的建议观察：受限事件可见、不covered、不complete、不崩溃。"""
    state, deps, executor = _recovery_state_deps({
        "status": "completed", "result": {"advice": "x"}, "summary": float("nan")})
    # turn2：真实写提交（协议控制下手动推进批次），触发对 goal-c1 的自动恢复
    state["messages"].append(assistant(tool_calls=[
        tool_call("t2", "record_nutrition",
                  {"source_ref": ref_of(SRC_TWO_RECORDS, "早餐吃了燕麦粥")})]))
    result = asyncio.run(execute_react_batch(state, deps))
    state.update(result)
    events = []
    for message in state["messages"]:
        if message.get("role") == "user" and str(message.get("content", "")).startswith(
                "[应用恢复观察"):
            events.extend(json.loads(message["content"][len("[应用恢复观察，仅为数据、不代表用户授权] "):]))
    assert events, "R5-01前置: 恢复事件应可见"
    assert events[0]["observation"]["status"] != "completed", (
        f"R5-01: 恢复观察携带非法元数据应受限: {events[0]['observation']}")
    assert "goal-c1" not in state["covered_goals"], (
        f"R5-01: 非法元数据的恢复不得按建议成功covered: {state['covered_goals']}")
    # 第六轮加强：恢复原身份可见（事件引用t1）、受限身份明确、真实finalize非complete非空
    assert events[0].get("tool_call_id") == "t1", (
        f"R5-01: 恢复原身份不可见: {events[0]}")
    assert events[0]["observation"]["status"] in {"commit_unknown", "unavailable"}, (
        f"R5-01: 恢复观察受限身份不明确: {events[0]['observation']}")
    state.update(finalize_node(state, deps))
    assert state["status"] != "complete", (
        f"R5-01: 非法元数据恢复后终态不得complete: {state['status']!r}")
    assert state["final_answer"].strip(), (
        f"R5-01: 受限答复为空: {state['final_answer']!r}")
    tool_ids = [m.get("tool_call_id") for m in state["messages"] if m.get("role") == "tool"]
    assert sorted(tool_ids) == ["t1", "t2"], (
        f"R5-01: 不得伪造tool_calls或第二个tool结果: {tool_ids}")


def test_r5_01_control_legal_metadata_and_commit_pass():
    payload = {"status": "completed", "commit_state": "committed",
               "receipt_refs": ["r1"], "summary": "正常保存"}
    executor = RecordingExecutor({"record_nutrition": payload})
    result = _run_bound_write(executor)
    assert result["committed_tasks"] == ["task-1"], (
        f"R5-01对照: 合法元数据+可信提交应通过: {result['committed_tasks']}")


def test_r5_01_shared_result_isolation_dict_and_direct_forms():
    """共享嵌套 result：dict 与 direct 首个返回后，执行器突变原引用——
    既有观察、goal_results、已配对消息与后续 context 隔离（真实状态输出为主证据）。"""
    for form in ["dict", "direct"]:
        shared = {"profile": {"note": "clean"}}
        if form == "dict":
            first = {"status": "completed", "result": shared}
        else:
            first = DomainObservation(status=ToolStatus.COMPLETED, result=shared)
        state = batch_state(SRC_RECORD, assistant(tool_calls=[
            tool_call("t1", "read_profile", {}),
            tool_call("t2", "read_profile", {})]))
        executor = RecordingExecutor()

        def executor_result(nth, name, arguments, context):
            if nth >= 1:
                shared["profile"]["note"] = "mutated-later"
            return first if nth == 0 else {"status": "completed", "result": {"n": nth}}

        executor.results["read_profile"] = executor_result
        result = asyncio.run(execute_react_batch(state, _deps(
            FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
        first_stored = result["observations"][0]
        assert "mutated" not in json.dumps(first_stored, ensure_ascii=False), (
            f"R5-01({form}): 共享result突变反向污染已存观察: {first_stored}")
        goal_results = result.get("goal_results") or {}
        assert "mutated" not in json.dumps(goal_results, ensure_ascii=False), (
            f"R5-01({form}): goal_results被反向污染")
        assert "mutated" not in result["messages"][3]["content"], (
            f"R5-01({form}): 已配对消息被反向污染")


# ---------------------------------------------------------------------------
# R5-02 / RA-07：严格位置与完整事实兼容
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_span", [(True, 7), ("0", 7)],
                         ids=["bool-span", "string-span"])
def test_r5_02_bool_and_string_spans_get_no_position(bad_span):
    """协议控制：bool/字符串显式区间不重定位、不得到合法位置、零覆盖零派发。"""
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
        f"R5-02: bool/字符串区间 {bad_span} 得到位置并被覆盖: {result['covered_goals']}")
    assert executor.calls == [], f"R5-02: 坏区间候选零派发: {executor.names}"


def test_r5_02_complete_fact_ref_full_containment_binding():
    """协议控制：完整事实对应实现为“候选区间完整包含于ref”——
    完整早餐 ref(0,10) 只绑定早餐目标；ref 覆盖两目标全部区间时两者均提交（合法对照）。"""
    candidates = [
        {"id": "goal-a", "task_type": "nutrition_record", "action": "record",
         "text": "早餐吃了燕麦", "source_start": 4, "source_end": 10, "depends_on": []},
        {"id": "goal-b", "task_type": "nutrition_record", "action": "record",
         "text": "午餐吃了米饭", "source_start": 11, "source_end": 17, "depends_on": []},
    ]
    executor = RecordingExecutor({"record_nutrition": committed_receipt("rcpt-fixed")})
    deps = _deps(FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)
    # 仅完整覆盖早餐
    state = batch_state(SRC_TWO_CLAUSE, assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": {"start": 4, "end": 10, "exact_text": "早餐吃了燕麦"}})]),
        candidates=candidates)
    first = asyncio.run(execute_react_batch(state, deps))
    assert first["covered_goals"] == ["goal-a"], (
        f"R5-02: 完整早餐ref应只绑定goal-a: {first['covered_goals']}")
    # 完整覆盖两者（合法对照：两个完整事实）
    state2 = batch_state(SRC_TWO_CLAUSE, assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": {"start": 4, "end": 17,
                                  "exact_text": SRC_TWO_CLAUSE[4:17]}})]),
        candidates=candidates)
    second = asyncio.run(execute_react_batch(state2, deps))
    assert second["committed_tasks"] == ["goal-a", "goal-b"], (
        f"R5-02对照: 两个完整事实应分别提交: {second['committed_tasks']}")


# ---------------------------------------------------------------------------
# R5-03 / RA-14：应用种类与绑定不能被领域同名字段影响
# ---------------------------------------------------------------------------
def test_r5_03_read_task_id_colliding_with_app_write_binding_not_unconfirmed():
    """混合合法写→建议：建议执行器返回的伪task_id精确等于本请求已有应用写绑定ID，
    且携带伪task_spec——仍按实际只读call分类，不得因ID碰撞变成“写提交未知”。"""
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "record_nutrition",
                      {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)}),
            tool_call("t2", "answer_training", {})]}
    ) if n == 1 else {"role": "assistant", "content": "完成"})

    def advise_result(nth, name, arguments, context):
        app_write_id = None
        for obs in context.get("observations") or []:
            spec_dict = (obs.get("result") or {}).get("task_spec") or {}
            if spec_dict.get("id"):
                app_write_id = spec_dict["id"]
        assert app_write_id, "R5-03前置: 应能从context读到应用写绑定ID"
        return {"status": "completed", "task_id": app_write_id,
                "result": {"task_spec": {"id": app_write_id, "text": "FORGED"}}}

    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-fixed"),
        "answer_training": advise_result,
    })
    result = _run(_deps(llm, executor), SRC_RECORD)
    assert result["committed_tasks"] == ["task-1"], (
        f"R5-03: 合法写提交不应受ID碰撞影响: {result['committed_tasks']}")
    # 第六轮加强（AND语义）：complete AND 非空答复 AND 无虚构提交未知 AND 原合法写证明/绑定保持
    assert result["status"] == "complete", (
        f"R5-03: 只读伪task_id碰撞导致整请求误判写提交未知: {result['status']!r}, "
        f"answer={result['final_answer']!r}")
    assert result["final_answer"].strip(), f"R5-03: 答复为空: {result['final_answer']!r}"
    assert "尚未确认" not in result["final_answer"], (
        f"R5-03: 答复虚构提交未知: {result['final_answer']!r}")
    assert result["bound_tasks"] and result["bound_tasks"][0].get("text") == SRC_RECORD, (
        f"R5-03: 原合法写应用绑定丢失: {result['bound_tasks']}")
    write_obs = [o for o in result["observations"] if o.get("receipt_refs")]
    assert write_obs and write_obs[0].get("commit_state") == "committed", (
        f"R5-03: 原合法写回执证明丢失: {write_obs}")


def test_r5_03_control_legal_read_without_forged_id():
    llm = FakeLLMPolicy(lambda n, m: (
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call("t1", "record_nutrition",
                      {"source_ref": ref_of(SRC_RECORD, SRC_RECORD)}),
            tool_call("t2", "answer_training", {})]}
    ) if n == 1 else {"role": "assistant", "content": "完成"})
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-fixed"),
        "answer_training": {"status": "completed", "result": {"answer": "正常"}},
    })
    result = _run(_deps(llm, executor), SRC_RECORD)
    assert result["status"] == "complete", f"R5-03对照: 合法只读应complete: {result['status']!r}"


def test_r5_03_domain_no_matching_goal_error_on_bound_write_not_app_unbound():
    """已有效绑定的合法写返回 error="no_matching_goal" 同名字段：不能因领域错误字符串
    碰巧同名就给已有绑定call追加应用“无绑定”身份（unbound_write_calls）。"""
    state = _write_call_state(BOUND_CANDIDATES)
    executor = RecordingExecutor({"record_nutrition": {
        "status": "unavailable", "error": "no_matching_goal"}})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert result.get("unbound_write_calls") == [], (
        f"R5-03: 领域错误字符串碰巧同名被记为应用无绑定身份: {result.get('unbound_write_calls')}")
    assert result["observations"][0]["error"] == "no_matching_goal", (
        "R5-03: 领域错误应按正常受限观察处理（保留领域错误）")
    state.update(result)
    state.update(finalize_node(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}))))
    assert state["status"] != "complete", (
        f"R5-03: 受限观察终态不得complete: {state['status']!r}")


# ---------------------------------------------------------------------------
# R5-04 / RA-04：新调用门禁与异常已有状态分开
# ---------------------------------------------------------------------------
def test_r5_04_no_matching_goal_gate_full_app_facts():
    """新写门禁全套应用事实：零派发、零有效绑定、不covered/committed、
    原call有配对澄清结果、finalize非complete/非空正确原因、unbound_write_calls
    与实际call身份一致。"""
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}})]),
        candidates=[])
    executor = RecordingExecutor({"record_nutrition": committed_receipt()})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    assert executor.calls == [], f"R5-04: 应零派发: {executor.names}"
    assert result["bound_tasks"] == [], f"R5-04: 不得创建应用绑定: {result['bound_tasks']}"
    assert result["covered_goals"] == [] and result["committed_tasks"] == []
    tool_ids = [m.get("tool_call_id") for m in result["messages"] if m.get("role") == "tool"]
    assert tool_ids == ["t1"], f"R5-04: 原call应有配对澄清结果: {tool_ids}"
    observation = result["observations"][0]
    assert observation["status"] == "needs_clarification" and \
        observation.get("error") == "no_matching_goal", f"R5-04: {observation}"
    assert result.get("unbound_write_calls") == ["t1"], (
        f"R5-04: unbound_write_calls应与实际call身份一致: {result.get('unbound_write_calls')}")
    state.update(result)
    state.update(finalize_node(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}))))
    assert state["status"] != "complete" and state["final_answer"].strip(), (
        f"R5-04: finalize应非complete且非空: {state['status']!r}, {state['final_answer']!r}")


def _abnormal_existing_state(observation):
    """协议控制（内存状态防御）：真实工具目录/批校验产生的原写调用观察，
    但finalize入口前应用绑定/目标/调用事实丢失（bound_tasks/goal_results/
    unbound_write_calls/intent_candidates为空）。不得手填unbound_write_calls。"""
    state = _write_call_state(BOUND_CANDIDATES)
    executor = RecordingExecutor({"record_nutrition": observation})
    result = asyncio.run(execute_react_batch(state, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}), executor)))
    abnormal = {
        "messages": list(result["messages"]),
        "observations": list(result["observations"]),
        "bound_tasks": [],
        "goal_results": {},
        "committed_tasks": [],
        "covered_goals": [],
        "intent_candidates": [],
        "unbound_write_calls": [],
        "pending_goals": [],
        "recovery_counter": 0,
        "tool_steps": result["tool_steps"],
        "consecutive_no_progress": 0,
        "terminated": False,
        "termination_reason": "",
        "status": "running",
        "protocol_error": "",
        "final_answer": "",
        "safety_level": "normal",
        "source_text": SRC_RECORD,
        "source_scope": None,
        "user_id": "app-user-1",
        "conversation_id": "conv-1",
        "request_id": "req-5",
    }
    return abnormal


@pytest.mark.parametrize("payload", [
    {"status": "completed", "commit_state": "unknown"},
    {"status": "completed", "commit_state": "committed", "receipt_refs": ["r1"]},
], ids=["abnormal-unknown", "abnormal-committed"])
def test_r5_04_abnormal_existing_write_state_cannot_complete(payload):
    """异常已有写观察（缺少可信应用调用事实/绑定）单独构造：unknown 与
    committed+合法回执均不能完整成功；缺少可信应用调用事实时明确受限。"""
    abnormal = _abnormal_existing_state(payload)
    final = finalize_node(abnormal, _deps(
        FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "全部保存成功"})))
    assert final["status"] != "complete", (
        f"R5-04: 异常已有写观察 {payload.get('commit_state')} 缺少应用调用事实仍判complete: "
        f"{final['status']!r}")
    assert final["final_answer"].strip() and "全部保存成功" not in final["final_answer"], (
        f"R5-04: 答复空或转发肯定宣称: {final['final_answer']!r}")


def _t9_fixture(call_id, payload):
    """T9独立mock夹具：同request/user/source/候选/写工具，原call ID指定，
    执行器恰实际触达一次；经真实execute_react_batch产生规范观察、应用绑定与调用事实。"""
    candidates = [{
        "id": "task-1", "task_type": "nutrition_record", "action": "record",
        "text": SRC_RECORD, "source_start": 0, "source_end": len(SRC_RECORD),
        "depends_on": [],
    }]
    state = batch_state(SRC_RECORD, assistant(tool_calls=[
        tool_call(call_id, "record_nutrition",
                  {"source_ref": {"start": 0, "end": len(SRC_RECORD),
                                  "exact_text": SRC_RECORD}})]),
        candidates=candidates)
    executor = RecordingExecutor({"record_nutrition": payload})
    result = asyncio.run(execute_react_batch(state, ReactDependencies(
        llm=FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"}),
        tool_executor=executor, user_id="app-user-1",
        conversation_id="conv-1", request_id="req-t9")))
    assert len(executor.calls) == 1, f"T9前置: 夹具{call_id}应恰触达一次: {executor.names}"
    return result, executor


def _t9_combine(hist_result, current_result):
    """T9显式终态单元夹具：当前结果字段原样取当前夹具；observations/call_facts
    合并两份真实生成结果；消息仅保留两组真实assistant/tool片段（历史在前）；
    不调用执行节点处理组合状态。"""
    real_messages = [m for m in hist_result["messages"] if m.get("role") in {"assistant", "tool"}]
    real_messages += [m for m in current_result["messages"] if m.get("role") in {"assistant", "tool"}]
    return {
        "messages": real_messages,
        "observations": list(hist_result["observations"]) + list(current_result["observations"]),
        "call_facts": list(hist_result["call_facts"]) + list(current_result["call_facts"]),
        "bound_tasks": list(current_result["bound_tasks"]),
        "goal_results": dict(current_result["goal_results"]),
        "committed_tasks": list(current_result["committed_tasks"]),
        "covered_goals": list(current_result["covered_goals"]),
        "unbound_write_calls": list(current_result["unbound_write_calls"]),
        "pending_goals": [],
        "recovery_counter": 0,
        "tool_steps": current_result["tool_steps"],
        "consecutive_no_progress": 0,
        "terminated": False,
        "termination_reason": "",
        "status": "running",
        "protocol_error": "",
        "final_answer": "",
        "safety_level": "normal",
        "source_text": SRC_RECORD,
        "source_scope": None,
        "user_id": "app-user-1",
        "conversation_id": "conv-1",
        "request_id": "req-t9",
        "intent_candidates": [{
            "id": "task-1", "task_type": "nutrition_record", "action": "record",
            "text": SRC_RECORD, "source_start": 0, "source_end": len(SRC_RECORD),
            "depends_on": [],
        }],
    }


def _t9_finalize(combined):
    """追加同一mock最终assistant候选答复“全部保存成功”（无tool_calls）后finalize；
    该答复是mock模型输入，不是手填成功状态或提交证明。"""
    combined["messages"].append({"role": "assistant", "content": "全部保存成功"})
    return finalize_node(combined, ReactDependencies(
        llm=FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "全部保存成功"}),
        tool_executor=RecordingExecutor(), user_id="app-user-1",
        conversation_id="conv-1", request_id="req-t9"))


def test_r5_04_control_history_unknown_not_dragging_current_trusted_completion():
    """T9快照组合终态控制（合成终态单元，非两轮活跃图恢复/未知写重放）：
    历史completed+commit_state=unknown（原call ID h1）、当前committed回执（原call ID c1）
    → 终态取舍应为当前可信完成：complete且非空，不被历史拖累。
    应用绑定ID/来源/候选goal由两夹具真实路径产生并断言相同；
    当前结果字段取当前夹具，禁止被历史覆盖。"""
    hist_result, _hist_executor = _t9_fixture(
        "h1", {"status": "completed", "commit_state": "unknown"})
    current_result, _cur_executor = _t9_fixture("c1", committed_receipt("rcpt-current"))
    # 两夹具应用身份一致（同request/user/source/候选/工具 → 同应用绑定ID）
    hist_binding = hist_result["bound_tasks"][0]
    current_binding = current_result["bound_tasks"][0]
    assert hist_binding["id"] == current_binding["id"], (
        f"T9: 同身份应用绑定ID应相同: {hist_binding['id']} vs {current_binding['id']}")
    assert hist_binding["text"] == current_binding["text"] == SRC_RECORD, (
        f"T9: 绑定来源应相同: {hist_binding['text']!r} vs {current_binding['text']!r}")
    assert hist_result["covered_goals"] == current_result["covered_goals"] == ["task-1"], (
        f"T9: 候选goal应相同: {hist_result['covered_goals']} vs {current_result['covered_goals']}")
    # 组合：当前结果/提交集合取当前，历史数据仅作为观测历史合入
    combined = _t9_combine(hist_result, current_result)
    assert combined["goal_results"]["task-1"]["commit_state"] == "committed", (
        f"T9: 当前结果字段被历史覆盖: {combined['goal_results']}")
    assert combined["committed_tasks"] == ["task-1"], (
        f"T9: 提交集合被历史改写: {combined['committed_tasks']}")
    assert len(combined["call_facts"]) == 2 and combined["call_facts"][0][
        "tool_call_id"] == "h1" and combined["call_facts"][1]["tool_call_id"] == "c1", (
        f"T9: 两组调用事实应可追溯: {combined['call_facts']}")
    final = _t9_finalize(combined)
    assert final["status"] == "complete", (
        f"T9: 历史unknown拖累了当前可信完成: {final['status']!r}")
    assert final["final_answer"].strip(), f"T9: 答复为空: {final['final_answer']!r}"


def test_r5_04_control_current_unknown_not_covered_by_history_success():
    """T9快照组合终态控制（合成终态单元）：历史committed回执（h1）、当前
    completed+commit_state=unknown（c1）→ 终态取舍为当前结果：非complete、非空、
    不转发肯定保存宣称——历史证明不能代替当前结果。当前结果字段禁止被历史覆盖。"""
    hist_result, _hist_executor = _t9_fixture("h1", committed_receipt("rcpt-history"))
    current_result, _cur_executor = _t9_fixture(
        "c1", {"status": "completed", "commit_state": "unknown"})
    hist_binding = hist_result["bound_tasks"][0]
    current_binding = current_result["bound_tasks"][0]
    assert hist_binding["id"] == current_binding["id"], (
        f"T9: 同身份应用绑定ID应相同: {hist_binding['id']} vs {current_binding['id']}")
    assert hist_binding["text"] == current_binding["text"] == SRC_RECORD
    assert hist_result["covered_goals"] == current_result["covered_goals"] == ["task-1"]
    combined = _t9_combine(hist_result, current_result)
    assert combined["goal_results"]["task-1"]["commit_state"] == "unknown", (
        f"T9: 当前结果字段被历史覆盖: {combined['goal_results']}")
    assert combined["committed_tasks"] == [], (
        f"T9: 历史成功ID被追加到当前提交集合: {combined['committed_tasks']}")
    final = _t9_finalize(combined)
    assert final["status"] != "complete", (
        f"T9: 当前unknown被历史成功覆盖: {final['status']!r}")
    assert final["final_answer"].strip() and "全部保存成功" not in final["final_answer"], (
        f"T9: 答复为空或转发肯定保存宣称: {final['final_answer']!r}")


def test_r4_06_urgent_preserved_quick_check():
    """急症控制保持确认（完整图）：零模型、零普通派发、确定性安全答复。"""
    llm = FakeLLMPolicy(lambda n, m: {"role": "assistant", "content": "x"})
    executor = RecordingExecutor({"record_nutrition": committed_receipt()})
    result = _run(_deps(llm, executor), SRC_URGENT)
    assert len(llm.calls) == 0 and executor.calls == [], (
        f"R4-06: 急症下模型/派发应为零: {len(llm.calls)}, {executor.names}")
    assert result["final_answer"].strip() and result["status"] == "rejected"
