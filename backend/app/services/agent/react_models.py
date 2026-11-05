"""阶段A ReAct 协议与状态模型（纯数据 + 纯校验，零副作用）。

本模块只定义状态、来源引用、调用批次、领域观察与消息协议校验，不导入
领域 worker、LLM 网关、数据库或任何网络客户端，可被 GLM 以 mock 模型与
假执行器独立验收。它被 react_tools / react_graph 复用，自身不反向依赖二者。

关键约定：
- 来源引用 ``source_ref={start,end,exact_text}`` 针对冻结后的 sanitized 原文，
  使用 Python Unicode 字符下标、左闭右开；``start/end`` 必须是真正的 int
  （拒绝 bool/字符串），``exact_text`` 必须与 ``source[start:end]`` 完全一致。
- 领域观察状态为稳定枚举；映射层不得把 "保存成功" 之类字符串升级为 committed。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypedDict


# ---------------------------------------------------------------------------
# 结果与提交状态枚举
# ---------------------------------------------------------------------------
class ToolStatus(StrEnum):
    """领域工具观察的稳定状态。"""

    COMPLETED = "completed"
    PARTIAL = "partial"
    NEEDS_CLARIFICATION = "needs_clarification"
    REJECTED = "rejected"
    BLOCKED_DEPENDENCY = "blocked_dependency"
    UNAVAILABLE = "unavailable"
    COMMIT_UNKNOWN = "commit_unknown"
    # 阶段A专有：真实写 adapter 在参数冻结/操作预留/保存派发之前显式阻断。
    BLOCKED_CAPABILITY = "blocked_capability"


class CommitState(StrEnum):
    COMMITTED = "committed"
    PENDING = "pending"
    UNKNOWN = "unknown"


_ALLOWED_STATUS = {s.value for s in ToolStatus}
_ALLOWED_COMMIT = {c.value for c in CommitState}


# ---------------------------------------------------------------------------
# 基础类型守卫
# ---------------------------------------------------------------------------
def is_int(value: Any) -> bool:
    """真正的 int（拒绝 bool 与字符串），用于 source_ref 边界等严格校验。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _reject_constant(value: str) -> Any:
    # JSON 的 NaN/Infinity/-Infinity 字面量一律拒绝。
    raise ValueError(f"非法 JSON 常量: {value}")


def _reject_non_finite(obj: Any) -> None:
    """递归拒绝非有限浮点（NaN/Inf/-Inf），防止 1e999 等溢出混入参数。"""
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError("参数包含非有限数值")
    if isinstance(obj, dict):
        for value in obj.values():
            _reject_non_finite(value)
    elif isinstance(obj, list):
        for value in obj:
            _reject_non_finite(value)


def parse_arguments(raw: Any) -> dict[str, Any] | str:
    """把函数参数解析为严格合法对象；返回 ``str`` 表示协议损坏原因。

    ``raw`` 为字符串时按严格 JSON 解析（拒绝 NaN/Infinity 与数值溢出），结果
    必须是对象；``raw`` 已是 dict 时递归拒绝非有限值。布尔、数组、标量一律
    拒绝，杜绝隐式数值转换（``"1"``/``1.0`` 不会被悄悄当成 int）。
    """
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw, parse_constant=_reject_constant)
        except (json.JSONDecodeError, TypeError, ValueError):
            return "arguments_invalid_json"
        if not isinstance(parsed, dict):
            return "arguments_not_object"
        try:
            _reject_non_finite(parsed)
        except ValueError:
            return "arguments_non_finite"
        return parsed
    if isinstance(raw, dict):
        try:
            _reject_non_finite(raw)
        except ValueError:
            return "arguments_non_finite"
        return raw
    return "arguments_not_object"


def _deep_copy_json(value: Any) -> Any:
    """深复制 JSON 兼容值以断开可变引用；不可序列化值原样返回，交由 __post_init__ 收敛。

    json 往返会保留 NaN/Infinity（``json.dumps`` 默认 allow_nan=True），随后由
    ``__post_init__`` 的 ``allow_nan=False`` 校验收敛为受限观察，不会把脏值带进协议。
    """
    if value is None:
        return None
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return value


def _demote_invalid_result(observation: "DomainObservation") -> None:
    """非法 result（非对象/不可序列化）使观察收敛，清除可信提交证明，绝不伪报成功。"""
    if observation.status == ToolStatus.COMPLETED:
        observation.status = ToolStatus.COMMIT_UNKNOWN
        observation.commit_state = None
        observation.receipt_refs = []
    elif observation.status == ToolStatus.PARTIAL:
        observation.status = ToolStatus.UNAVAILABLE


# ---------------------------------------------------------------------------
# 来源引用
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceRef:
    """记录工具接受的唯一参数：冻结 sanitized 原文的字符区间（左闭右开）。"""

    start: int
    end: int
    exact_text: str


@dataclass
class SourceScope:
    """请求开始时冻结的来源与意图候选（覆盖输入，只读）。

    ``text`` 是 sanitized 原文；``intent_candidates`` 是原规则识别出的意图候选，
    只用于覆盖与回执检查，不预生成实体参数，不决定执行顺序。
    """

    text: str
    intent_candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SourceRefValidation:
    """``validate_source_ref`` 的返回值。

    ``ok`` 只表示 source_ref 结构合法（类型/边界/exact_text 匹配）；写工具的
    授权门禁另在 ``authorization`` 字段（authorized / needs_confirmation /
    rejected_unauthorized / None）。
    """

    ok: bool
    reason: str | None = None
    authorization: str | None = None
    span_text: str | None = None


# ---------------------------------------------------------------------------
# 调用批次
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    """校验通过后的内部归一化调用 ``{id, name, args}``。"""

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class BatchValidation:
    """``validate_tool_batch`` 的返回值：整批协议校验结果。"""

    ok: bool
    calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None
    detail: str | None = None


# ---------------------------------------------------------------------------
# 领域观察
# ---------------------------------------------------------------------------
@dataclass
class DomainObservation:
    """一次工具执行的领域观察，消息/结果合同的最小必要字段集。

    ``status`` / ``commit_state`` 在构造后即被归一化：直接以字符串、list、dict
    或未知值构造也不会崩溃，一律收敛为稳定枚举（未知 status → UNAVAILABLE，
    未知 commit_state → None）。绝不臆造 committed。
    """

    status: Any
    task_id: str | None = None
    commit_state: Any = None
    result: Any = None
    receipt_refs: Any = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    context_version: int | None = None
    summary: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ToolStatus):
            self.status = (
                ToolStatus(self.status)
                if isinstance(self.status, str) and self.status in _ALLOWED_STATUS
                else ToolStatus.UNAVAILABLE
            )
        if self.commit_state is not None and not (
            isinstance(self.commit_state, str) and self.commit_state in _ALLOWED_COMMIT
        ):
            self.commit_state = None
        # result 仅允许对象或空，递归 JSON 值必须有限、可序列化；非法结果收敛为
        # 稳定错误观察，不把脏结果带入协议或冒充可信成功。
        if self.result is not None:
            if not isinstance(self.result, dict):
                self.result = None
                self.error = self.error or "result_not_object"
                _demote_invalid_result(self)
            else:
                try:
                    json.dumps(self.result, allow_nan=False, ensure_ascii=False)
                except (TypeError, ValueError):
                    self.result = None
                    self.error = self.error or "result_not_serializable"
                    _demote_invalid_result(self)
        # receipt_refs 只有“非空 list 且所有元素均为非空/非纯空白字符串”才有效；
        # 任何坏元素使整份提交证明失效，不迭代非 list、不字符串化其他容器。
        if self.receipt_refs:
            if not isinstance(self.receipt_refs, list) or not all(
                isinstance(r, str) and r.strip() for r in self.receipt_refs
            ):
                self.receipt_refs = []
        # 整份观察元数据边界：先识别非法性并整份降级、清提交证明，再清理输出。
        # 不能在 from_dict 提前把非法值变 None 后让构造器误以为缺省合法；也不能忽略 {} 等假值非法字段。
        metadata_error = None
        if self.task_id is not None and not isinstance(self.task_id, str):
            self.task_id = None
            metadata_error = metadata_error or "invalid_task_id"
        if self.missing_fields is not None and (
            not isinstance(self.missing_fields, list)
            or not all(isinstance(m, str) for m in self.missing_fields)
        ):
            self.missing_fields = []
            metadata_error = metadata_error or "invalid_missing_fields"
        if self.context_version is not None and not is_int(self.context_version):
            self.context_version = None
            metadata_error = metadata_error or "invalid_context_version"
        if self.summary is not None and not isinstance(self.summary, str):
            self.summary = None
            metadata_error = metadata_error or "invalid_summary"
        if self.error is not None and not isinstance(self.error, str):
            self.error = None
            metadata_error = metadata_error or "invalid_error"
        if metadata_error:
            # 非法元数据与非法 result 同等对待：整份降级并清除可信提交证明，绝不伪报成功；
            # 非法 error 字段本身不得进入错误消息，改用稳定安全错误码。
            if not self.error:
                self.error = metadata_error
            _demote_invalid_result(self)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"status": self.status.value}
        if self.task_id is not None:
            data["task_id"] = self.task_id
        if self.commit_state is not None:
            data["commit_state"] = self.commit_state
        if self.result is not None:
            data["result"] = _deep_copy_json(self.result)
        if self.receipt_refs:
            data["receipt_refs"] = list(self.receipt_refs)
        if self.missing_fields:
            data["missing_fields"] = list(self.missing_fields)
        if self.context_version is not None:
            data["context_version"] = self.context_version
        if self.summary is not None:
            data["summary"] = self.summary
        if self.error is not None:
            data["error"] = self.error
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "DomainObservation":
        """外部观察归一化：未知状态收敛为 unavailable，绝不臆造 committed。

        映射层不得把 "保存成功" 之类的字符串升级为 committed：只有显式出现
        合法 ``status`` / ``commit_state`` 枚举值才会被采用。result 原始值保留，
        交由 ``__post_init__`` 统一区分“缺失/None 的空结果”与“明确非对象/不可序列化
        的协议错误”。
        """
        if not isinstance(data, dict):
            return cls(status=ToolStatus.UNAVAILABLE, error="observation_not_object")
        raw_status = data.get("status")
        status = (
            ToolStatus(raw_status)
            if isinstance(raw_status, str) and raw_status in _ALLOWED_STATUS
            else ToolStatus.UNAVAILABLE
        )
        raw_commit = data.get("commit_state")
        commit_state = (
            raw_commit
            if isinstance(raw_commit, str) and raw_commit in _ALLOWED_COMMIT
            else None
        )
        # 元数据原值保留，交由 __post_init__ 统一区分“缺失/None 的缺省合法”与
        # “明确提供但类型非法的协议错误”；不在 from_dict 提前把非法值变 None 洗白非法性。
        return cls(
            status=status,
            task_id=data.get("task_id"),
            commit_state=commit_state,
            result=data.get("result"),
            receipt_refs=data.get("receipt_refs"),
            missing_fields=data.get("missing_fields"),
            context_version=data.get("context_version"),
            summary=data.get("summary"),
            error=data.get("error"),
        )

    @classmethod
    def normalize(cls, raw: Any) -> "DomainObservation":
        """统一返回边界校验：dict 与直接 DomainObservation 都重新归一化，断开可变引用。

        直接 DomainObservation 实例不得因 isinstance 直接返回原对象——构造后突变
        （result 改 NaN/set、receipt_refs 改坏值）必须在进入状态前重新收敛。
        """
        if isinstance(raw, DomainObservation):
            return cls(
                status=raw.status,
                task_id=raw.task_id,
                commit_state=raw.commit_state,
                result=_deep_copy_json(raw.result),
                receipt_refs=(list(raw.receipt_refs) if isinstance(raw.receipt_refs, list) else raw.receipt_refs),
                missing_fields=(list(raw.missing_fields) if isinstance(raw.missing_fields, list) else raw.missing_fields),
                context_version=raw.context_version,
                summary=raw.summary,
                error=raw.error,
            )
        return cls.from_dict(raw)


# ---------------------------------------------------------------------------
# 消息协议校验（容器层；目录/schema 校验在 react_tools.validate_tool_batch）
# ---------------------------------------------------------------------------
def validate_tool_calls_container(message: Any) -> tuple[list[dict[str, Any]], str | None]:
    """协议级容器校验：tool_calls 数组、call 对象、ID 非空且批内唯一、arguments 对象。

    返回 ``(normalized_calls, error)``。error 非空时调用方必须整批零派发。本函数
    不做目录/schema 校验。无 tool_calls（None）返回 ``([], None)``，表示模型希望
    结束而非协议损坏。

    同时兼容 OpenAI 形态（``function.name`` / ``function.arguments`` 字符串）与已
    归一化形态（``name`` / ``args`` 对象），但都做同样的严格解析与类型检查。
    """
    if not isinstance(message, dict):
        return [], "assistant_message_not_object"
    tool_calls = message.get("tool_calls")
    if tool_calls is None:
        return [], None
    if not isinstance(tool_calls, list):
        return [], "tool_calls_not_array"
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for call in tool_calls:
        if not isinstance(call, dict):
            return [], "tool_call_not_object"
        function = call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            raw_args = function.get("arguments")
        else:
            name = call.get("name")
            raw_args = call.get("args")
        if not isinstance(name, str) or not name.strip():
            return [], "tool_call_name_missing"
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            return [], "tool_call_id_missing"
        if call_id in seen_ids:
            return [], "tool_call_id_duplicate"
        args = parse_arguments(raw_args)
        if isinstance(args, str):
            return [], args
        seen_ids.add(call_id)
        normalized.append({"id": call_id, "name": name.strip(), "args": args})
    return normalized, None


# ---------------------------------------------------------------------------
# 图状态
# ---------------------------------------------------------------------------
class ReactState(TypedDict, total=False):
    """独立 ReAct 图状态。messages 为 OpenAI 格式 dict 列表，由节点全量维护。"""

    user_id: str
    conversation_id: str
    request_id: str
    raw_message: str
    safety_level: str
    source_text: str
    source_scope: SourceScope
    intent_candidates: list[dict[str, Any]]
    candidate_error: str
    messages: list[dict[str, Any]]
    supervisor_turns: int
    tool_steps: int
    tool_steps_by_task: dict[str, int]
    limited_tasks: list[str]
    consecutive_no_progress: int
    terminated: bool
    termination_reason: str
    status: str
    covered_goals: list[str]
    committed_tasks: list[str]
    goal_results: dict[str, dict[str, Any]]
    pending_goals: list[dict[str, Any]]
    recovery_counter: int
    bound_tasks: list[dict[str, Any]]
    unbound_write_calls: list[str]
    call_facts: list[dict[str, Any]]
    call_event_counter: int
    observations: list[dict[str, Any]]
    protocol_error: str
    model_failure_code: str
    final_answer: str


__all__ = [
    "BatchValidation",
    "CommitState",
    "DomainObservation",
    "ReactState",
    "SourceRef",
    "SourceRefValidation",
    "SourceScope",
    "ToolCall",
    "ToolStatus",
    "is_int",
    "parse_arguments",
    "validate_tool_calls_container",
]
