"""阶段A 静态工具目录、严格 schema、source_ref 校验与领域 adapter 协议。

目录在请求开始时冻结，模型选择与执行共享同一版本；未知工具拒绝；身份、授权、
request_id、task_id、幂等键与时间基准不进入模型可填 schema。真实写 adapter 本轮
不在此模块内接入任何 worker，仅保留绑定与阻断语义（blocked_capability）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from app.services.agent.task_models import RECORD_TASK_TYPES, TaskSpec, TaskType
from app.services.agent.react_models import (
    BatchValidation,
    SourceRefValidation,
    SourceScope,
    ToolCall,
    is_int,
    validate_tool_calls_container,
)
from app.services.security.write_guard import authorize_write_span


# 写工具名 -> 领域任务类型（决定 source_ref 是否需做完整写授权窗口校验）。
_WRITE_TASK_TYPES: dict[str, str] = {
    "record_workout": TaskType.WORKOUT_RECORD.value,
    "correct_workout": TaskType.WORKOUT_RECORD.value,
    "record_nutrition": TaskType.NUTRITION_RECORD.value,
}
WRITE_TOOL_NAMES = frozenset(_WRITE_TASK_TYPES)
_WRITE_TASK_TYPE_VALUES = RECORD_TASK_TYPES

# 建议/计划类工具（需要本轮记录作为依赖时须等提交完成）。
ADVICE_TASK_TYPES = frozenset({TaskType.NUTRITION_ADVICE.value, TaskType.TRAINING_PLAN.value})

SOURCE_REF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start": {"type": "integer", "minimum": 0},
        "end": {"type": "integer", "minimum": 0},
        "exact_text": {"type": "string"},
    },
    "required": ["start", "end", "exact_text"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    permission: str  # "read" | "write"
    parameters: dict[str, Any]
    task_type: str | None = None
    worker: str | None = None


def _source_ref_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"source_ref": SOURCE_REF_SCHEMA},
        "required": ["source_ref"],
        "additionalProperties": False,
    }


def _query_parameters(*, with_source_ref: bool) -> dict[str, Any]:
    props: dict[str, Any] = {"query": {"type": "string"}}
    if with_source_ref:
        props["source_ref"] = SOURCE_REF_SCHEMA
    return {"type": "object", "properties": props, "additionalProperties": False}


def _empty_parameters() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _days_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 365}},
        "additionalProperties": False,
    }


# 固定目录：顺序稳定，模型侧 tools 定义与执行侧共享同一份。
_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "correct_workout", "更正已保存训练动作名，仅用于workout_correction，引用本轮更正原文；保留其它字段，不能新增记录。",
        "write", _source_ref_parameters(), TaskType.WORKOUT_RECORD.value, "workout_record",
    ),
    ToolSpec(
        "record_workout", "记录一次训练（只引用用户原文区间，不提交实体数组）。",
        "write", _source_ref_parameters(), TaskType.WORKOUT_RECORD.value, "workout_record",
    ),
    ToolSpec(
        "record_nutrition", "记录饮食（只引用用户原文区间，不提交食物实体数组）。",
        "write", _source_ref_parameters(), TaskType.NUTRITION_RECORD.value, "nutrition",
    ),
    ToolSpec(
        "advise_nutrition", "给出饮食/营养建议（依赖本轮记录时须等提交完成）。",
        "read", _query_parameters(with_source_ref=True), TaskType.NUTRITION_ADVICE.value, "nutrition",
    ),
    ToolSpec(
        "plan_training", "给出训练计划建议（不新增计划保存权限）。",
        "read", _query_parameters(with_source_ref=True), TaskType.TRAINING_PLAN.value, "plan_adjust",
    ),
    ToolSpec(
        "answer_training", "回答一般训练/健身问题。",
        "read", _query_parameters(with_source_ref=True), TaskType.TRAINING_QA.value, "training_qa",
    ),
    ToolSpec(
        "read_profile", "读取用户档案（身份由应用注入）。",
        "read", _empty_parameters(),
    ),
    ToolSpec(
        "read_workout_history", "读取训练历史（日期/条数受限）。",
        "read", _days_parameters(),
    ),
    ToolSpec(
        "read_nutrition_summary", "读取营养摘要（范围/版本受限）。",
        "read", _days_parameters(),
    ),
    ToolSpec(
        "read_current_plan", "读取当前训练计划（不能转为写计划）。",
        "read", _empty_parameters(),
    ),
)

TOOL_CATALOG: Mapping[str, ToolSpec] = {spec.name: spec for spec in _TOOLS}


def tool_definitions(registry: Mapping[str, ToolSpec] = TOOL_CATALOG) -> list[dict[str, Any]]:
    """OpenAI 兼容的模型侧工具定义；顺序稳定以利前缀缓存。"""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in registry.values()
    ]


def _schema_error(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    try:
        error = best_match(Draft202012Validator(schema).iter_errors(args))
    except Exception:
        return "schema_definition_invalid"
    if error is None:
        return None
    where = f" (字段 {error.json_path})" if error.json_path != "$" else ""
    return f"{error.message}{where}"


def _strict_integers_recursive(schema: dict[str, Any], value: Any, path: str = "") -> str | None:
    """递归检查 schema 中 integer 字段必须是严格 Python int（拒绝 bool/字符串/浮点）。

    覆盖嵌套对象（如 source_ref.start/end）与数组；在任何执行器调用前完成，保证
    整批原子性。jsonschema Draft2020-12 视 1.0 为 integer，故这里用 ``is_int`` 补
    严格 Python 原始类型二层校验。
    """
    if not isinstance(schema, dict):
        return None
    stype = schema.get("type")
    if stype == "integer":
        if not is_int(value):
            return f"字段 {path or '$'} 必须是整数"
        return None
    if stype == "object":
        properties = schema.get("properties")
        if isinstance(properties, dict) and isinstance(value, dict):
            for key, subschema in properties.items():
                if key in value:
                    child_path = f"{path}.{key}" if path else key
                    error = _strict_integers_recursive(subschema, value[key], child_path)
                    if error:
                        return error
        return None
    if stype == "array":
        items = schema.get("items")
        if isinstance(value, list):
            for item in value:
                error = _strict_integers_recursive(items, item, path)
                if error:
                    return error
        return None
    return None


def validate_tool_batch(
    message: Any,
    registry: Mapping[str, ToolSpec] = TOOL_CATALOG,
) -> BatchValidation:
    """整批结构与 schema 校验；验证完成前零工具执行。

    先做容器校验（tool_calls 数组、call 对象、ID 非空且批内唯一、arguments
    严格合法对象 JSON），再逐项校验 name 在目录且参数符合 schema。任何一项失败
    都整批拒绝（``ok=False``），调用方必须零派发；schema 的 additionalProperties:
    false 拒绝未知身份/授权等字段，integer 类型递归拒绝 bool/字符串/浮点。
    """
    normalized, error = validate_tool_calls_container(message)
    if error is not None:
        return BatchValidation(ok=False, error=error)
    calls: list[ToolCall] = []
    for item in normalized:
        name = item["name"]
        spec = registry.get(name)
        if spec is None:
            return BatchValidation(ok=False, error="unknown_tool", detail=name)
        schema_error = _schema_error(spec.parameters, item["args"])
        if schema_error is not None:
            return BatchValidation(ok=False, error="schema_invalid", detail=f"{name}: {schema_error}")
        int_error = _strict_integers_recursive(spec.parameters, item["args"])
        if int_error is not None:
            return BatchValidation(ok=False, error="schema_invalid", detail=f"{name}: {int_error}")
        calls.append(ToolCall(id=item["id"], name=name, args=item["args"]))
    return BatchValidation(ok=True, calls=calls)


def validate_source_ref(
    source_scope: SourceScope | None,
    source_ref: Any,
    task_type: str | None,
) -> SourceRefValidation:
    """校验来源类型、边界和原文一致性；写意图由模型选择工具表达。

    不再用原文正则重新判断命令、否定或问句。历史、工具观察和模型文字
    不属于当前 source_scope，不能通过这里冒充用户原文。
    """
    if not isinstance(source_scope, SourceScope):
        return SourceRefValidation(ok=False, reason="source_scope_missing")
    text = source_scope.text
    if not isinstance(source_ref, dict):
        return SourceRefValidation(ok=False, reason="source_ref_missing")
    start = source_ref.get("start")
    end = source_ref.get("end")
    exact = source_ref.get("exact_text")
    if not is_int(start):
        return SourceRefValidation(ok=False, reason="start_not_integer")
    if not is_int(end):
        return SourceRefValidation(ok=False, reason="end_not_integer")
    if not isinstance(exact, str):
        return SourceRefValidation(ok=False, reason="exact_text_not_string")
    if start < 0 or end < start or end > len(text):
        return SourceRefValidation(ok=False, reason="source_ref_out_of_bounds")
    if text[start:end] != exact:
        return SourceRefValidation(ok=False, reason="exact_text_mismatch")
    if task_type in _WRITE_TASK_TYPE_VALUES:
        if not exact.strip():
            return SourceRefValidation(ok=False, reason="empty_write_source")
        # Selecting a write tool is the model's semantic decision; the app only
        # proves the cited span comes from the user's original text. Original
        # wording outside the span is not re-classified here.
        authorization, reason = authorize_write_span(exact, semantic_available=True)
        return SourceRefValidation(
            ok=True,
            authorization=authorization,
            span_text=exact,
            reason=reason if authorization != "authorized" else None,
        )
    return SourceRefValidation(ok=True, authorization=None, span_text=exact)


def _candidate_task_id(request_id: str, name: str, start: int, end: int) -> str:
    digest = hashlib.sha256(f"{request_id}|{name}|{start}|{end}".encode("utf-8")).hexdigest()[:16]
    return f"react-{name}-{start}-{end}-{digest}"


def bind_domain_task(
    call: ToolCall,
    source_scope: SourceScope,
    *,
    task_id: str | None = None,
    request_id: str = "",
    authorization: str | None = None,
) -> TaskSpec:
    """生成合法 TaskSpec 候选；身份由应用分配，不冒充已经持久冻结的账本。

    本函数只构造候选对象，不冻结参数、不预留操作、不保存。候选 id 由应用侧
    request_id 与来源区间确定，模型不可填。写工具 candidate 的 action 固定为
    record，建议/问答工具的 action 为 answer；侧效应仅写工具为 True。
    """
    spec = TOOL_CATALOG.get(call.name)
    if spec is None:
        raise ValueError(f"未知工具: {call.name}")
    if not isinstance(source_scope, SourceScope):
        raise ValueError("bind_domain_task 需要 SourceScope")
    source_ref = call.args.get("source_ref") if isinstance(call.args, dict) else None
    if not isinstance(source_ref, dict) or not is_int(source_ref.get("start")) or not is_int(source_ref.get("end")):
        raise ValueError("写/建议工具缺少合法 source_ref")
    start = source_ref["start"]
    end = source_ref["end"]
    text = source_scope.text[start:end]
    if task_id is None:
        task_id = _candidate_task_id(request_id or "", call.name, start, end)
    return TaskSpec(
        id=task_id,
        task_type=spec.task_type or "",
        worker=spec.worker or "",
        action="record" if spec.permission == "write" else "answer",
        text=text,
        payload={
            "source_ref": {"start": start, "end": end, "exact_text": source_ref.get("exact_text", text)},
            "write_authorization": authorization or "authorized",
        },
        side_effect=spec.permission == "write",
        status="pending",
    )


__all__ = [
    "ADVICE_TASK_TYPES",
    "SOURCE_REF_SCHEMA",
    "TOOL_CATALOG",
    "ToolSpec",
    "WRITE_TOOL_NAMES",
    "bind_domain_task",
    "tool_definitions",
    "validate_source_ref",
    "validate_tool_batch",
]
