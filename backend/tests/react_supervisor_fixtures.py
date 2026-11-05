"""Tests for react supervisor fixtures."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable


def _goal(task_type: str, action: str, text: str, *, negated: bool = False,
          hypothetical: bool = False, third_party: bool = False) -> dict[str, Any]:
    """One scripted model task: the model's structured semantic decision."""
    return {
        "task_type": task_type,
        "action": action,
        "text": text,
        "is_negated": negated,
        "is_hypothetical": hypothetical,
        "is_third_party": third_party,
    }


# 各验收用例 source 对应的结构化模型判定（mock 模型输出，不是应用侧文本规则）。
# 每个 task.text 都是该 source 的连续原文，保证来源/覆盖校验真实生效。
# Scripted, source-bound safety decisions for the synthetic fixture requests.
# These are model outputs, not application-side keyword classification: each
# entry states the judgment the fake semantic model returns for that exact
# source.  Every unlisted request is an ordinary, non-medical request.
PHASE_A_SAFETY: dict[str, dict[str, Any]] = {
    "胸痛得厉害,帮我记录今天心率120": {
        "level": "urgent",
        "evidence": "胸痛得厉害",
        "categories": ["current_symptom"],
        "reason": "scripted current symptom",
    },
}


PHASE_A_GOALS: dict[str, list[dict[str, Any]]] = {
    "你好": [_goal("chitchat", "answer", "你好")],
    "怎么练卧推": [_goal("training_qa", "question", "怎么练卧推")],
    "帮我记录午餐吃了牛肉面和可乐": [
        _goal("nutrition_record", "record", "帮我记录午餐吃了牛肉面和可乐"),
    ],
    "帮我记录早餐吃了燕麦粥。然后帮我记录早餐吃了鸡蛋。然后根据早餐给我营养建议": [
        _goal("nutrition_record", "record", "帮我记录早餐吃了燕麦粥"),
        _goal("nutrition_record", "record", "帮我记录早餐吃了鸡蛋"),
        _goal("nutrition_advice", "question", "然后根据早餐给我营养建议"),
    ],
    "帮我记录早餐吃了燕麦粥。然后帮我记录早餐吃了鸡蛋。然后根据早餐给我营养建议。卧推怎么练？": [
        _goal("nutrition_record", "record", "帮我记录早餐吃了燕麦粥"),
        _goal("nutrition_record", "record", "帮我记录早餐吃了鸡蛋"),
        _goal("nutrition_advice", "question", "然后根据早餐给我营养建议"),
        _goal("training_qa", "question", "卧推怎么练"),
    ],
    "帮我记录早餐吃了鸡蛋。午餐吃了沙拉。": [
        _goal("nutrition_record", "record", "帮我记录早餐吃了鸡蛋"),
        _goal("nutrition_advice", "question", "午餐吃了沙拉"),
    ],
    "帮我记录早餐吃了鸡蛋。体重数据不要记录。": [
        _goal("nutrition_record", "record", "帮我记录早餐吃了鸡蛋"),
        _goal("workout_record", "record", "体重数据不要记录", negated=True),
    ],
    "今天吃了蛋糕,不要保存": [
        _goal("nutrition_record", "record", "今天吃了蛋糕,不要保存", negated=True),
    ],
    "不要记录今天跑步5公里": [
        _goal("workout_record", "record", "不要记录今天跑步5公里", negated=True),
    ],
    "如果明天下雨我就在家练俯卧撑30分钟": [
        _goal("workout_record", "record", "如果明天下雨我就在家练俯卧撑30分钟", hypothetical=True),
    ],
    "朋友昨天练了卧推5组5次": [
        _goal("workout_record", "record", "朋友昨天练了卧推5组5次", third_party=True),
    ],
    "今天午餐吃什么好？晚餐吃什么好？": [
        _goal("nutrition_advice", "question", "今天午餐吃什么好？"),
        _goal("nutrition_advice", "question", "晚餐吃什么好？"),
    ],
    "胸痛得厉害,帮我记录今天心率120": [
        _goal("safety_triage", "question", "胸痛得厉害"),
        _goal("nutrition_record", "record", "帮我记录今天心率120"),
    ],
}


def phase_a_understanding(source: str) -> dict[str, Any]:
    """Structured model understanding for one test source (scripted decision)."""
    tasks = PHASE_A_GOALS.get(source)
    if tasks is None:
        # Unscripted chat input: the model has no business goal for it.
        tasks = [_goal("chitchat", "answer", source)]
    safety = PHASE_A_SAFETY.get(source, {
        "level": "normal", "evidence": "", "categories": [],
        "reason": "scripted normal",
    })
    return {"tasks": [dict(task) for task in tasks], "safety": dict(safety)}


def _semantic_query(messages: list[dict[str, Any]]) -> str:
    """Read the request query from ``dialogue_context.semantic_messages`` payload."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(str(message.get("content") or ""))
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("query"), str):
            return payload["query"]
    return ""


class SemanticModelMixin:
    """Current structured-understanding protocol for scripted fake models.

    ``semantic=False`` models a deployment without a semantic model: ``configured()``
    is False, so the request keeps the read-only rule fallback.
    """

    semantic: bool = True

    def configured(self) -> bool:
        return bool(self.semantic)

    async def complete_structured(self, messages, schema, *, name=None,
                                  temperature=0.0, validator=None, **_kwargs):
        source = _semantic_query(messages)
        self.semantic_calls.append({
            "messages": [dict(message) for message in messages],
            "name": name,
            "temperature": temperature,
        })
        value = schema.model_validate(phase_a_understanding(source))
        if validator is not None:
            validator(value)
        return value


class FakeLLM(SemanticModelMixin):
    """脚本化/策略化单步模型：complete_message(messages, temperature, tools, tool_choice)。

    ``script`` 为按次回放的 assistant 响应列表；``policy(nth_call, messages)`` 可
    根据上一个 tool 结果动态选择下一响应（动态决策验收必须走 policy）。
    每次调用都记录完整入参与返回值，作为原始证据。``semantic`` 控制是否提供
    当前结构化语义协议（见 ``SemanticModelMixin``）。
    """

    def __init__(
        self,
        script: list[dict[str, Any]] | None = None,
        policy: Callable[[int, list[dict[str, Any]]], dict[str, Any]] | None = None,
        on_exhausted: dict[str, Any] | None = None,
        semantic: bool = True,
    ) -> None:
        self.script = list(script or [])
        self.policy = policy
        self.on_exhausted = on_exhausted or {"role": "assistant", "content": "（脚本耗尽）"}
        self.semantic = semantic
        self.calls: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []
        self.semantic_calls: list[dict[str, Any]] = []

    async def complete_message(self, messages, temperature=0.0, tools=None, tool_choice=None):
        self.calls.append({
            "messages": [dict(m) for m in messages],
            "temperature": temperature,
            "tools": tools,
            "tool_choice": tool_choice,
        })
        index = len(self.calls) - 1
        if self.policy is not None:
            response = dict(self.policy(len(self.calls), messages))
        elif index < len(self.script):
            response = dict(self.script[index])
        else:
            response = dict(self.on_exhausted)
        self.responses.append(response)
        return response

    @property
    def tool_call_sequence(self) -> list[list[str]]:
        """每次响应里的工具名顺序（模型侧证据）。"""
        sequence: list[list[str]] = []
        for response in self.responses:
            names = []
            for call in response.get("tool_calls") or []:
                if isinstance(call, dict):
                    function = call.get("function")
                    if isinstance(function, dict):
                        names.append(str(function.get("name")))
                    else:
                        names.append(str(call.get("name")))
            sequence.append(names)
        return sequence


class RecordingExecutor:
    """callable(name, args, context) 形式的假执行器，记录全部调用与顺序。

    ``results`` 按工具名给出返回值（DomainObservation 兼容 dict / dataclass /
    可调用 name->返回值）；未命中时返回 unavailable 占位。
    """

    def __init__(self, results: dict[str, Any] | None = None) -> None:
        self.results = dict(results or {})
        self.calls: list[dict[str, Any]] = []

    def __call__(self, name: str, arguments: dict[str, Any], context: dict[str, Any]):
        self.calls.append({
            "name": name,
            "args": json.loads(json.dumps(arguments, ensure_ascii=False, default=str)),
            "context": json.loads(json.dumps(context, ensure_ascii=False, default=str)),
        })
        result = self.results.get(name)
        if callable(result):
            result = result(len(self.calls) - 1, name, arguments, context)
        if result is None:
            return {"status": "unavailable", "error": "no_scripted_result"}
        return result

    @property
    def names(self) -> list[str]:
        return [c["name"] for c in self.calls]


def tool_call(call_id: str, name: str, args: dict[str, Any] | str) -> dict[str, Any]:
    """OpenAI 原生形态的 tool_call（function.arguments 为 JSON 字符串）。"""
    raw = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else args
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": raw},
    }


def assistant(content: str = "", tool_calls: Any = None, **extra: Any) -> dict[str, Any]:
    """模型侧 assistant 响应；tool_calls 传 None 表示不携带该字段。"""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None or "tool_calls" in extra:
        message["tool_calls"] = tool_calls
    message.update(extra)
    return message


def ref_of(source_text: str, needle: str, occurrence: int = 0) -> dict[str, Any]:
    """按中文原文构造 source_ref（Python 字符下标，左闭右开）。"""
    start = -1
    for _ in range(occurrence + 1):
        start = source_text.find(needle, start + 1)
        assert start >= 0, f"fixture 原文缺少片段: {needle!r}"
    return {"start": start, "end": start + len(needle), "exact_text": needle}


def batch_state(
    source_text: str,
    assistant_message: dict[str, Any],
    candidates: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """构造 execute_react_batch 可直接消费的协议控制状态（绕过 prepare）。

    仅用于容器/依赖/预算边界的精确覆盖；图级验收必须走 invoke_react 真实路径。
    """
    from app.services.agent.react_models import SourceScope

    state: dict[str, Any] = {
        "user_id": "app-user-1",
        "conversation_id": "conv-1",
        "request_id": "req-1",
        "raw_message": source_text,
        "source_text": source_text,
        "source_scope": SourceScope(text=source_text, intent_candidates=candidates or []),
        "intent_candidates": candidates or [],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": source_text},
            assistant_message,
        ],
        "supervisor_turns": 1,
        "tool_steps": 0,
        "consecutive_no_progress": 0,
        "terminated": False,
        "termination_reason": "",
        "status": "running",
        "covered_goals": [],
        "committed_tasks": [],
        "bound_tasks": [],
        "observations": [],
        "protocol_error": "",
        "final_answer": "",
    }
    state.update(overrides)
    return state


def committed_receipt(receipt_id: str = "rcpt-1") -> dict[str, Any]:
    """可信假提交回执（协议模拟，非真实保存证据）。"""
    return {
        "status": "completed",
        "commit_state": "committed",
        "result": {"receipt": receipt_id},
        "receipt_refs": [receipt_id],
    }


def run_async(awaitable: Any) -> Any:
    return asyncio.run(awaitable)
