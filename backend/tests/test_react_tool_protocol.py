"""Tests for react tool protocol."""

from __future__ import annotations

import json

import pytest

from app.services.agent.react_models import (
    DomainObservation,
    SourceScope,
    ToolStatus,
    parse_arguments,
    validate_tool_calls_container,
)
from app.services.agent.react_tools import (
    TOOL_CATALOG,
    tool_definitions,
    validate_source_ref,
    validate_tool_batch,
)
from app.services.agent.react_graph import ReactDependencies, execute_react_batch

from tests.react_supervisor_fixtures import (
    RecordingExecutor,
    assistant,
    batch_state,
    ref_of,
    run_async,
    tool_call,
)

RECORD_SOURCE = "帮我记录午餐吃了牛肉面和可乐"


def _read_profile_state(**overrides):
    return batch_state(
        "你好",
        assistant(tool_calls=[tool_call("t1", "read_profile", {})]),
        **overrides,
    )


def _record_state(executor_result, **overrides):
    candidates = [{
        "id": "task-1", "task_type": "nutrition_record", "action": "record",
        "text": RECORD_SOURCE, "source_start": 0, "source_end": len(RECORD_SOURCE),
        "depends_on": [],
    }]
    message = assistant(tool_calls=[tool_call(
        "t1", "record_nutrition", {"source_ref": ref_of(RECORD_SOURCE, RECORD_SOURCE)})])
    executor = RecordingExecutor({"record_nutrition": executor_result})
    state = batch_state(RECORD_SOURCE, message, candidates=candidates, **overrides)
    return state, executor


def _deps(executor):
    return ReactDependencies(llm=object(), tool_executor=executor, user_id="app-user-1",
                             conversation_id="conv-1", request_id="req-1")


# ---------------------------------------------------------------------------
# A03 注入接口：文档声明的映射 handler(args, context) 不能被 TypeError 变 unavailable
# ---------------------------------------------------------------------------
def test_mapping_executor_two_arg_sync_handler_is_invoked():
    calls = []

    def handler(args, context):
        calls.append((args, context))
        return {"status": "completed", "result": {"via": "mapping-2arg-sync"}}

    state = _read_profile_state()
    result = run_async(execute_react_batch(state, _deps({"read_profile": handler})))
    assert calls, "A03: 文档声明的映射 handler(args, context) 应被调用"
    observation = json.loads(result["messages"][-1]["content"])
    assert observation["status"] == ToolStatus.COMPLETED.value, (
        f"A03: 合法映射2参数handler被降级为 {observation}, "
        f"error={observation.get('error')!r}（实现按3参数调用handler）")


def test_mapping_executor_two_arg_async_handler_is_invoked():
    calls = []

    async def handler(args, context):
        calls.append((args, context))
        return {"status": "completed", "result": {"via": "mapping-2arg-async"}}

    state = _read_profile_state()
    result = run_async(execute_react_batch(state, _deps({"read_profile": handler})))
    assert calls, "A03: 异步映射 handler(args, context) 应被调用"
    observation = json.loads(result["messages"][-1]["content"])
    assert observation["status"] == ToolStatus.COMPLETED.value, (
        f"A03: 合法异步映射2参数handler被降级为 {observation}")


def test_callable_executor_three_arg_sync_and_async_controls():
    sync_calls, async_calls = [], []

    def sync_handler(name, args, context):
        sync_calls.append(name)
        return {"status": "completed", "result": {"via": "callable-sync"}}

    async def async_handler(name, args, context):
        async_calls.append(name)
        return {"status": "completed", "result": {"via": "callable-async"}}

    result = run_async(execute_react_batch(_read_profile_state(), _deps(sync_handler)))
    observation = json.loads(result["messages"][-1]["content"])
    assert observation["status"] == ToolStatus.COMPLETED.value
    result = run_async(execute_react_batch(_read_profile_state(), _deps(async_handler)))
    observation = json.loads(result["messages"][-1]["content"])
    assert observation["status"] == ToolStatus.COMPLETED.value
    assert sync_calls == ["read_profile"] and async_calls == ["read_profile"]


def test_mapping_executor_three_arg_handler_documented_reality_control():
    """实际实现只支持 (name, args, context) 3参数映射 handler——与文档字符串冲突。"""

    def handler(name, args, context):
        return {"status": "completed", "result": {"via": "mapping-3arg"}}

    state = _read_profile_state()
    result = run_async(execute_react_batch(state, _deps({"read_profile": handler})))
    observation = json.loads(result["messages"][-1]["content"])
    assert observation["status"] == ToolStatus.COMPLETED.value


# ---------------------------------------------------------------------------
# A04 提交语义：completed/无提交证明不得标记 committed；只读完成不是写提交
# ---------------------------------------------------------------------------
def test_completed_with_commit_state_unknown_must_not_mark_committed():
    state, executor = _record_state({
        "status": "completed", "commit_state": "unknown", "result": {"note": "无提交证明"},
    })
    result = run_async(execute_react_batch(state, _deps(executor)))
    assert "task-1" not in result["committed_tasks"], (
        f"A04: status=completed但commit_state=unknown被标记committed: "
        f"{result['committed_tasks']}")


def test_completed_without_commit_proof_must_not_mark_committed():
    state, executor = _record_state({"status": "completed", "result": {"note": "无commit_state字段"}})
    result = run_async(execute_react_batch(state, _deps(executor)))
    assert "task-1" not in result["committed_tasks"], (
        f"A04: status=completed但无提交证明被标记committed: {result['committed_tasks']}")


def test_completed_with_pending_commit_must_not_mark_committed():
    state, executor = _record_state({"status": "completed", "commit_state": "pending"})
    result = run_async(execute_react_batch(state, _deps(executor)))
    assert "task-1" not in result["committed_tasks"], (
        f"A04: commit_state=pending被标记committed: {result['committed_tasks']}")


def test_legal_committed_with_receipt_control_passes():
    """合法对照：committed + 假回执允许标记（协议模拟，不称为真实保存）。"""
    state, executor = _record_state({
        "status": "completed", "commit_state": "committed",
        "result": {"receipt": "rcpt-1"}, "receipt_refs": ["rcpt-1"],
    })
    result = run_async(execute_react_batch(state, _deps(executor)))
    assert result["committed_tasks"] == ["task-1"]
    observation = json.loads(result["messages"][-1]["content"])
    assert observation["status"] == ToolStatus.COMPLETED.value
    assert observation["receipt_refs"] == ["rcpt-1"]


def test_read_only_completed_is_not_a_write_commit():
    qa_candidates = [{
        "id": "task-q1", "task_type": "training_qa", "action": "question",
        "text": "你好", "source_start": 0, "source_end": 2, "depends_on": [],
    }]
    state = batch_state(
        "你好",
        assistant(tool_calls=[tool_call("t1", "answer_training", {})]),
        candidates=qa_candidates,
    )
    executor = RecordingExecutor({"answer_training": {"status": "completed"}})
    result = run_async(execute_react_batch(state, _deps(executor)))
    assert "task-q1" not in result["committed_tasks"], (
        f"A04: 只读completed混入committed_tasks(写提交语义): {result['committed_tasks']}")


# ---------------------------------------------------------------------------
# A05 容器协议：畸形 tool_calls 必须拒绝；[] 与缺失是合法完成
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_tool_calls", [{}, "", 0, False],
                         ids=["empty-dict", "empty-string", "zero", "false"])
def test_falsy_non_list_tool_calls_are_protocol_rejected(bad_tool_calls):
    message = assistant("x", tool_calls=bad_tool_calls)
    _normalized, error = validate_tool_calls_container(message)
    assert error is not None, (
        f"A05: tool_calls={bad_tool_calls!r} 未被容器协议拒绝")
    batch = validate_tool_batch(message)
    assert not batch.ok


def test_empty_list_and_missing_tool_calls_are_legal_controls():
    _normalized, error = validate_tool_calls_container(assistant("done", tool_calls=[]))
    assert error is None, "A05: 合法 [] 对照被误拒"
    _normalized, error = validate_tool_calls_container(assistant("done"))
    assert error is None, "A05: 真正缺失 tool_calls 对照被误拒"
    _normalized, error = validate_tool_calls_container(assistant("done", tool_calls=None))
    assert error is None


@pytest.mark.parametrize("message,expected_error", [
    (assistant(tool_calls=[tool_call("t1", "read_profile", "{}"),
                           tool_call("t1", "read_profile", "{}")]),
     "tool_call_id_duplicate"),
    (assistant(tool_calls=[{"id": "", "type": "function",
                            "function": {"name": "read_profile", "arguments": "{}"}}]),
     "tool_call_id_missing"),
    (assistant(tool_calls=[{"type": "function",
                            "function": {"name": "read_profile", "arguments": "{}"}}]),
     "tool_call_id_missing"),
    (assistant(tool_calls=[{"id": "t1", "type": "function",
                            "function": {"name": "", "arguments": "{}"}}]),
     "tool_call_name_missing"),
    (assistant(tool_calls=[{"id": "t1", "type": "function", "function": "read_profile"}]),
     "tool_call_name_missing"),
    (assistant(tool_calls=[tool_call("t1", "read_profile", "{bad json")]),
     "arguments_invalid_json"),
    (assistant(tool_calls=[tool_call("t1", "read_profile", "null")]),
     "arguments_not_object"),
    (assistant(tool_calls=[tool_call("t1", "read_profile", '{"days": NaN}')]),
     "arguments_invalid_json"),
    (assistant(tool_calls=[tool_call("t1", "read_nutrition_summary", '{"days": 1e999}')]),
     "arguments_non_finite"),
    (assistant(tool_calls=[tool_call("t1", "read_profile", '["array"]')]),
     "arguments_not_object"),
], ids=["dup-id", "empty-id", "missing-id", "empty-name", "function-not-dict",
        "bad-json", "null-args", "nan-constant", "overflow-1e999", "array-args"])
def test_malformed_openai_shapes_rejected(message, expected_error):
    _normalized, error = validate_tool_calls_container(message)
    assert error == expected_error, f"A05: 期望 {expected_error}, 实际 {error!r}"
    batch = validate_tool_batch(message)
    assert not batch.ok


def test_infinity_and_dict_nan_arguments_rejected():
    batch = validate_tool_batch(assistant(tool_calls=[
        tool_call("t1", "read_nutrition_summary", '{"days": Infinity}')]))
    assert not batch.ok, "A05: Infinity 常量未拒绝"
    raw = {"days": float("nan")}
    parsed = parse_arguments(raw)
    assert parsed == "arguments_non_finite", f"A05: dict形态NaN未拒绝: {parsed!r}"
    parsed = parse_arguments({"days": float("inf")})
    assert parsed == "arguments_non_finite", f"A05: dict形态Infinity未拒绝: {parsed!r}"


# ---------------------------------------------------------------------------
# 合同控制2：JSONSchema integer 不得按库默认把 1.0 / bool / 字符串当整数
# ---------------------------------------------------------------------------
def test_float_1_0_must_not_pass_as_integer():
    """合同：不能靠库默认推断严格Python原始类型（jsonschema 2020-12 视 1.0 为 integer）。"""
    batch = validate_tool_batch(assistant(tool_calls=[
        tool_call("t1", "read_nutrition_summary", '{"days": 1.0}')]))
    assert not batch.ok, (
        f"合同2: days=1.0 通过schema校验: ok={batch.ok}, calls={batch.calls}, "
        f"error={batch.error!r}, detail={batch.detail!r}（执行器将收到float 1.0）")
    assert isinstance(batch.ok, bool)


def test_library_jsonschema_accepts_float_one_documentation_only():
    """T3修订（复验合同§2）：jsonschema Draft2020-12 按 JSON 规则把 1.0 视为 integer
    ——这是库行为事实记录（对库 schema 直接断言），不是应用层合同的合法依据。"""
    from jsonschema import Draft202012Validator
    assert Draft202012Validator({"type": "integer", "minimum": 0}).is_valid(1.0)
    assert not Draft202012Validator({"type": "integer"}).is_valid(True)
    assert not Draft202012Validator({"type": "integer"}).is_valid("3")


def test_nested_float_source_ref_must_reject_at_batch_layer():
    """T3修订（复验合同§2）：应用合同要求嵌套整数严格校验、整批异常零执行。

    validate_tool_batch 必须拒绝 float start/end 的嵌套 source_ref。190748 只补了
    顶层 integer（_strict_top_level_integers），嵌套是否放行以实际结果为准。
    """
    batch = validate_tool_batch(assistant(tool_calls=[
        tool_call("t1", "record_nutrition",
                  {"source_ref": {"start": 1.0, "end": 4.0, "exact_text": RECORD_SOURCE[1:4]}})]))
    assert not batch.ok, (
        f"合同2(T3/R2-02): 嵌套 float source_ref 通过整批校验: ok={batch.ok}, "
        f"calls={batch.calls}, error={batch.error!r}（190748 仅覆盖顶层 integer）")


def test_source_ref_float_bounds_rejected_by_second_gate():
    """合同4/2 对照（保留）：直接调用 validate_source_ref 的二层拒绝，传原始
    source_ref，不依赖 batch.calls。不得改二层拒绝期望。"""
    validation = validate_source_ref(
        SourceScope(text=RECORD_SOURCE),
        {"start": 1.0, "end": 4.0, "exact_text": RECORD_SOURCE[1:4]},
        "nutrition_record")
    assert not validation.ok, "合同4/2: source_ref浮点边界必须被严格int校验拒绝"
    assert validation.reason == "start_not_integer"


@pytest.mark.parametrize("args", [{"days": True}, {"days": "3"}, {"days": 1.5}],
                         ids=["bool-as-int", "string-as-int", "fractional-float"])
def test_bool_string_fractional_integers_rejected(args):
    batch = validate_tool_batch(assistant(tool_calls=[
        tool_call("t1", "read_nutrition_summary", args)]))
    assert not batch.ok, f"合同2: {args} 应被schema拒绝, 实际 ok={batch.ok}"


def test_legal_read_call_around_bad_calls_zero_dispatch_graph_level():
    """合法调用夹在坏调用前后：整批拒绝、零执行（图级在验收文件再验一次）。"""
    message = assistant(tool_calls=[
        tool_call("t1", "read_profile", {}),
        tool_call("t2", "delete_everything", {}),
        tool_call("t3", "read_profile", {}),
    ])
    executor = RecordingExecutor()
    state = _read_profile_state()
    state["messages"][-1] = message
    result = run_async(execute_react_batch(state, _deps(executor)))
    assert executor.calls == [], f"合同2: 整批坏调用仍有派发: {executor.names}"
    assert result["status"] == "restricted"
    assert result["termination_reason"] == "protocol_error"


# ---------------------------------------------------------------------------
# 合同控制3：身份不进模型参数、目录与模型定义一致且不可被参数修改
# ---------------------------------------------------------------------------
def test_extra_identity_or_authorization_fields_rejected():
    for name, args in [
        ("read_profile", {"user_id": "attacker"}),
        ("read_profile", {"request_id": "r2"}),
        ("record_nutrition", {"source_ref": ref_of(RECORD_SOURCE, RECORD_SOURCE),
                              "authorized": True}),
        ("advise_nutrition", {"query": "q", "user_id": "attacker"}),
    ]:
        batch = validate_tool_batch(assistant(tool_calls=[tool_call("t1", name, args)]))
        assert not batch.ok, f"合同3: {name} 附加字段 {set(args) - {'source_ref', 'query'}} 未拒绝"


def test_catalog_matches_model_definitions_and_is_param_frozen():
    definitions = tool_definitions()
    assert [d["function"]["name"] for d in definitions] == list(TOOL_CATALOG.keys())
    for definition in definitions:
        spec = TOOL_CATALOG[definition["function"]["name"]]
        assert definition["function"]["parameters"] == spec.parameters
        assert definition["function"]["description"] == spec.description
        assert definition["type"] == "function"
    snapshot = {name: (spec.parameters, spec.permission, spec.task_type)
                for name, spec in TOOL_CATALOG.items()}
    validate_tool_batch(assistant(tool_calls=[tool_call(
        "t1", "read_profile", {"user_id": "x"})]))
    validate_tool_batch(assistant(tool_calls=[tool_call(
        "t2", "record_nutrition", {"source_ref": {"start": 0, "end": 1, "exact_text": "帮"},
                                   "user_id": "x"})]))
    after = {name: (spec.parameters, spec.permission, spec.task_type)
             for name, spec in TOOL_CATALOG.items()}
    assert snapshot == after, "合同3: 校验过程修改了工具目录"
    write_tools = {name for name, spec in TOOL_CATALOG.items() if spec.permission == "write"}
    assert write_tools == {"record_workout", "record_nutrition", "correct_workout"}
    read_names = {name for name, spec in TOOL_CATALOG.items() if spec.permission == "read"}
    assert {"read_profile", "read_workout_history", "read_nutrition_summary",
            "read_current_plan", "advise_nutrition", "plan_training",
            "answer_training"} <= read_names


def test_trusted_context_identity_comes_from_application():
    executor = RecordingExecutor({"read_profile": {"status": "completed"}})
    state = _read_profile_state()
    del state["user_id"]  # 状态缺省时必须回落到应用注入身份
    run_async(execute_react_batch(state, _deps(executor)))
    assert executor.calls, "合同3: 执行器未被调用"
    context = executor.calls[0]["context"]
    assert context["user_id"] == "app-user-1"
    assert context["request_id"] == "req-1"
    assert context["conversation_id"] == "conv-1"


# ---------------------------------------------------------------------------
# 合同控制4：source_ref 结构、边界、中文区间与历史/观察文本
# ---------------------------------------------------------------------------
def _scope(text: str) -> SourceScope:
    return SourceScope(text=text)


def test_chinese_char_interval_and_tail_command_control():
    source = "今天跑步5公里30分钟,帮我记一下"
    validation = validate_source_ref(_scope(source), ref_of(source, "今天跑步5公里30分钟"),
                                     "workout_record")
    assert validation.ok and validation.authorization == "authorized", (
        f"合同4: 尾置命令授权对照失败: {validation}")


def test_structural_rejections():
    source = RECORD_SOURCE
    good = ref_of(source, RECORD_SOURCE)
    cases = [
        ({**good, "start": -1}, "source_ref_out_of_bounds"),
        ({**good, "end": len(source) + 1}, "source_ref_out_of_bounds"),
        ({**good, "start": 5, "end": 2}, "source_ref_out_of_bounds"),
        ({**good, "exact_text": "帮我记录午餐吃了牛肉面和面"}, "exact_text_mismatch"),
        ({**good, "start": True}, "start_not_integer"),
        ({**good, "end": "14"}, "end_not_integer"),
        ({**good, "exact_text": 123}, "exact_text_not_string"),
        ("not-a-dict", "source_ref_missing"),
        (None, "source_ref_missing"),
        ({**good, "start": 1.0}, "start_not_integer"),
    ]
    for source_ref, expected in cases:
        validation = validate_source_ref(_scope(source), source_ref, "nutrition_record")
        assert not validation.ok, f"合同4: {source_ref!r} 应拒绝"
        assert validation.reason == expected, (
            f"合同4: {source_ref!r} 期望 {expected}, 实际 {validation.reason!r}")


def test_history_or_observation_text_cannot_become_source():
    source = RECORD_SOURCE
    for foreign in ["工具观察：已保存成功", "历史记录：昨天卧推5组", "全部保存成功"]:
        validation = validate_source_ref(
            _scope(source),
            {"start": 0, "end": len(foreign), "exact_text": foreign},
            "nutrition_record")
        assert not validation.ok, f"合同4: 观察/历史文本 {foreign!r} 不能成为来源"
        assert validation.reason in {"exact_text_mismatch", "source_ref_out_of_bounds"}


def test_write_authorization_follows_semantic_flags_not_wording():
    """合同4（语义迁移）：写授权由模型 task/action/flags 决定，不再按原文命令词判断。

    模型选择 record 工具即表达写意图，跨句无命令词片段仍 authorized；
    否定/假设/第三人称 flags 拒绝同一片段；无模型决定绝不授权写入。
    """
    source = "帮我记录两组训练。第一组卧推50公斤"
    span = ref_of(source, "第一组卧推50公斤")
    validation = validate_source_ref(_scope(source), span, "workout_record")
    assert validation.ok and validation.authorization == "authorized", (
        f"合同4: 模型已选record工具的无命令词片段应authorized: {validation}")
    from app.services.security.write_guard import authorize_write_span
    for flag, reason in (("is_negated", "planner_flag_negated"),
                         ("is_hypothetical", "planner_flag_hypothetical"),
                         ("is_third_party", "planner_flag_third_party")):
        authorization, got = authorize_write_span("第一组卧推50公斤", {flag: True, "semantic_available": True})
        assert authorization == "rejected_unauthorized" and got == reason, (
            f"合同4: {flag} 应拒绝授权: {(authorization, got)}")
    authorization, reason = authorize_write_span("第一组卧推50公斤", {"semantic_available": False})
    assert authorization == "needs_confirmation", (
        f"合同4: 无模型决定应needs_confirmation: {(authorization, reason)}")


def test_read_tool_span_is_structural_only():
    source = "今天训练感觉如何"
    validation = validate_source_ref(_scope(source), ref_of(source, "训练感觉"), None)
    assert validation.ok and validation.authorization is None


# ---------------------------------------------------------------------------
# A08 观察归一化：list/dict 状态必须确定性收敛，不允许崩溃或污染协议
# ---------------------------------------------------------------------------
def test_dict_observation_list_status_must_converge_not_crash():
    try:
        observation = DomainObservation.from_dict({"status": ["completed"]})
    except TypeError as exc:
        raise AssertionError(
            f"A08: dict观察status为list抛出TypeError而非收敛unavailable: {exc!r} "
            "(react_models DomainObservation.from_dict 对 unhashable raw_status 崩溃)")
    assert observation.status is ToolStatus.UNAVAILABLE


def test_dict_observation_dict_status_must_converge_not_crash():
    try:
        observation = DomainObservation.from_dict({"status": {"value": "completed"}})
    except TypeError as exc:
        raise AssertionError(f"A08: dict观察status为dict抛出TypeError: {exc!r}")
    assert observation.status is ToolStatus.UNAVAILABLE


def test_dict_observation_list_commit_state_must_converge_not_crash():
    try:
        observation = DomainObservation.from_dict(
            {"status": "completed", "commit_state": ["committed"]})
    except TypeError as exc:
        raise AssertionError(f"A08: dict观察commit_state为list抛出TypeError: {exc!r}")
    assert observation.commit_state is None, (
        f"A08: 非法commit_state必须收敛为None: {observation.commit_state!r}")


def test_unknown_status_and_non_object_result_controls():
    observation = DomainObservation.from_dict({"status": "totally-unknown-value"})
    assert observation.status is ToolStatus.UNAVAILABLE
    for bad in [None, "text", 123, True, ["list"]]:
        observation = DomainObservation.from_dict(bad)
        assert observation.status is ToolStatus.UNAVAILABLE
        assert observation.error == "observation_not_object"
    observation = DomainObservation.from_dict({"status": True})
    assert observation.status is ToolStatus.UNAVAILABLE
    observation = DomainObservation.from_dict({"status": "completed", "result": "not-object"})
    assert observation.result is None


def test_batch_with_list_status_observation_must_not_crash_graph():
    executor_result = {"status": ["completed"], "result": {"note": "bad adapter"}}
    state, executor = _record_state(executor_result)
    try:
        result = run_async(execute_react_batch(state, _deps(executor)))
    except TypeError as exc:
        raise AssertionError(
            f"A08: 执行器返回坏status导致 execute_react_batch 抛出TypeError: {exc!r} "
            "(from_dict 在执行器异常捕获之外转换, 图级会整请求崩溃)")
    observation = json.loads(result["messages"][-1]["content"])
    assert observation["status"] == ToolStatus.UNAVAILABLE.value
    assert observation["status"] != "committed"


def test_direct_domain_observation_with_raw_string_status_must_not_crash():
    state, executor = _record_state(DomainObservation(status="completed"))
    try:
        result = run_async(execute_react_batch(state, _deps(executor)))
    except AttributeError as exc:
        raise AssertionError(
            f"A08: 直接返回 status=字符串 的 DomainObservation 导致 to_dict 崩溃: {exc!r}")
    observation = json.loads(result["messages"][-1]["content"])
    assert observation["status"] in {s.value for s in ToolStatus}


def test_direct_domain_observation_bad_commit_state_must_be_normalized():
    state, executor = _record_state(
        DomainObservation(status=ToolStatus.COMPLETED, commit_state=["committed"]))
    result = run_async(execute_react_batch(state, _deps(executor)))
    observation = json.loads(result["messages"][-1]["content"])
    assert observation.get("commit_state") in {None, "committed", "pending", "unknown"}, (
        f"A08: 直接DomainObservation的非法commit_state未归一化进入协议消息: "
        f"{observation.get('commit_state')!r}")


def test_executor_exception_becomes_deterministic_unavailable():
    def broken(name, args, context):
        raise RuntimeError("adapter exploded")

    state = _read_profile_state()
    result = run_async(execute_react_batch(state, _deps(broken)))
    observation = json.loads(result["messages"][-1]["content"])
    assert observation["status"] == ToolStatus.UNAVAILABLE.value
    assert observation["error"] == "executor_error:RuntimeError"
    assert "task-1" not in result["committed_tasks"]
