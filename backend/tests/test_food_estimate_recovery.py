"""Controlled fallback/retry for the React nutrition estimate."""
import asyncio
import json
from types import SimpleNamespace

from app.services.agent.react_write_adapters import (
    _fallback_food_entries, _normalized_food_name, _source_unit,
)
from app.services.agent.react_write_runtime import WriteRuntimeSession, _parse_food_estimate


class _LLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete_message(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        value = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(value, BaseException):
            raise value
        return value


def _session(responses):
    llm = _LLM(responses)
    runtime = SimpleNamespace(agent=SimpleNamespace(graph=SimpleNamespace(llm=llm)),
                              policy_stopped=False, budget_stop_reason="")
    return runtime, llm


def _assistant(content):
    return {"role": "assistant", "content": content}


def test_parse_plain_fenced_numeric_and_prose():
    payload = {"kcal": 377, "protein_g": 12.6, "carbs_g": 66.3,
               "fat_g": 7.3, "fiber_g": 5.3}
    assert _parse_food_estimate(_assistant(json.dumps(payload)))["kcal"] == 377
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    assert _parse_food_estimate(_assistant(fenced))["kcal"] == 377
    numeric = {**payload, "kcal": "377", "protein_g": "12.6"}
    assert _parse_food_estimate(_assistant(json.dumps(numeric)))["protein_g"] == 12.6
    prose = "估算如下：" + json.dumps(payload) + " 供参考"
    assert _parse_food_estimate(_assistant(prose))["kcal"] == 377


def test_parse_rejects_missing_or_invalid_kcal():
    assert _parse_food_estimate(_assistant("not json")) is None
    assert _parse_food_estimate(_assistant(json.dumps({"protein_g": 1}))) is None
    assert _parse_food_estimate(_assistant(json.dumps({"kcal": -1}))) is None
    assert _parse_food_estimate(_assistant(json.dumps({"kcal": True}))) is None
    assert _parse_food_estimate(_assistant("")) is None


def test_estimate_retries_once_on_parse_failure():
    runtime, llm = _session([
        _assistant("```json\n{\"kcal\": \"bad\"}\n```"),
        _assistant(json.dumps({"kcal": 377, "protein_g": 12.6, "carbs_g": 66.3,
                               "fat_g": 7.3, "fiber_g": 5.3})),
    ])
    result = asyncio.run(WriteRuntimeSession.estimate_food(
        runtime, {"entry": {"requested_name": "燕麦", "weight_g": 45}}))
    assert result["kcal"] == 377
    assert len(llm.calls) == 2
    assert llm.calls[0]["temperature"] == 0.0
    assert "上一次输出无法解析" in llm.calls[1]["messages"][-1]["content"]


def test_estimate_stops_after_two_invalid_outputs():
    runtime, llm = _session([_assistant("bad"), _assistant("still bad")])
    assert asyncio.run(WriteRuntimeSession.estimate_food(
        runtime, {"entry": {"requested_name": "未知食物", "weight_g": 100}})) is None
    assert len(llm.calls) == 2


def test_deterministic_food_fallback_entries():
    first = _fallback_food_entries("45克燕麦")
    assert first and first[0]["name"] == "燕麦" and first[0]["weight_g"] == 45
    second = _fallback_food_entries("燕麦45g")
    assert second and second[0]["name"] == "燕麦" and second[0]["weight_g"] == 45
    assert _fallback_food_entries("7-11饭团") == []


def test_unit_uses_the_literal_source_token():
    assert _source_unit("g", "今天早餐吃了45克燕麦") == "克"
    assert _source_unit("克", "燕麦45g") == "g"
    assert _source_unit("kg", "吃了1千克米饭") == "千克"
    assert _source_unit("ml", "喝了300毫升牛奶") == "毫升"
    assert _source_unit("g", "燕麦45") == "g"


def test_food_name_quantity_normalization():
    assert _normalized_food_name("45克燕麦") == "燕麦"
    assert _normalized_food_name("燕麦45g") == "燕麦"
    assert _normalized_food_name("300ml牛奶") == "牛奶"
    assert _normalized_food_name("7-11饭团") == "7-11饭团"
    assert _normalized_food_name("鸡蛋") == "鸡蛋"
