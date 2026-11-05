"""Omitted mode (auto) saves a nutrition record whose parser unit differs from source.

The parser emits ``g`` while the user wrote ``克``. The adapter must use the
literal source token, otherwise the write bridge refuses with
``unit_evidence_missing``. The AI estimate stays in the same request budget.
"""
import json

import pytest

from tests.react_phase_c2_3_support import C3HTTPHarness, c3_env, mock_provider_transport

SOURCE = "今天早餐实际吃了45克燕麦，请保存成今天早餐的饮食记录。"
FRAGMENT = "今天早餐实际吃了45克燕麦"


def native(value):
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(value, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 4}}


def semantic():
    value = {"tasks": [{
        "task_type": "nutrition_record", "action": "record", "text": SOURCE,
        "fact_text": FRAGMENT, "is_negated": False, "is_hypothetical": False,
        "is_third_party": False, "history_requirement": "none", "goal_role": "result",
        "reply_mode": "grounded", "reply_purpose": "general", "output_format": "default"}],
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "synthetic"}}
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
        "id": "semantic", "type": "function", "function": {
            "name": "return_request_understanding",
            "arguments": json.dumps(value, ensure_ascii=False)}}]}}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 4}}


def tool():
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
        "id": "tool", "type": "function", "function": {"name": "record_nutrition", "arguments": json.dumps(
            {"source_ref": {"start": 0, "end": len(SOURCE), "exact_text": SOURCE}}, ensure_ascii=False)}}]}}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 4}}


def parser():
    return native({"foods": [{"source_text": FRAGMENT, "name": "燕麦", "quantity": 1,
                              "unit": "g", "weight_g": 45, "meal_type": "早餐",
                              "record_date": "今天", "is_negated": False}]})


def estimate():
    return native({"kcal": 377, "protein_g": 12.6, "carbs_g": 66.3,
                   "fat_g": 7.3, "fiber_g": 5.3, "confidence": 0.6})


@pytest.mark.parametrize("backend", ["sqlite"])
def test_omitted_mode_auto_saves_estimated_breakfast(c3_env, monkeypatch, backend):
    seen = mock_provider_transport(monkeypatch, [semantic(), tool(), parser(), estimate()])
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    monkeypatch.setattr(harness.settings, "chat_default_mode", "auto", raising=False)

    status, payload = harness.post_json(SOURCE, mode=None, request_id="auto-food-1")

    assert status == 200, payload
    metrics = payload["metrics"]
    assert metrics["declared_mode"] == "auto", metrics
    assert metrics["execution_mode"] == "react_write", metrics
    assert metrics["react_write"] == "completed", metrics
    assert metrics["receipt_refs"], metrics
    assert "AI估算" in payload["answer"]
    assert len(seen) == 4
