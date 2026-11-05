"""Omitted mode (auto) upgrades a real write to react_write end to end.

Uses the real ASGI JSON entry over an isolated database with a provider-boundary
mock: shared semantics, supervisor tool call and workout extraction are scripted,
while the write authorization, bridge, transaction and receipts stay real.
"""
import json

import pytest
from sqlalchemy import text

from tests.react_phase_c2_3_support import C3HTTPHarness, c3_env, mock_provider_transport

SOURCE = "帮我记一下今天卧推 60kg 4组8次，谢谢"
FRAGMENT = "卧推 60kg 4组8次"


def native(value):
    return {"choices": [{"message": {"role": "assistant",
                                     "content": json.dumps(value, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 4}}


def semantic(source, fragment):
    value = {
        "tasks": [{
            "task_type": "workout_record", "action": "record", "text": source,
            "fact_text": fragment, "is_negated": False, "is_hypothetical": False,
            "is_third_party": False, "history_requirement": "none", "goal_role": "result",
        }],
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "synthetic"},
    }
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
        "id": "semantic-result", "type": "function", "function": {
            "name": "return_request_understanding",
            "arguments": json.dumps(value, ensure_ascii=False)}}]}}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 4}}


def tool(source, fragment, name):
    start = source.index(fragment)
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
        "id": "native-" + name, "type": "function", "function": {"name": name, "arguments": json.dumps(
            {"source_ref": {"start": start, "end": start + len(fragment), "exact_text": fragment}},
            ensure_ascii=False)}}]}}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 4}}


@pytest.mark.parametrize("backend", ["sqlite"])
def test_omitted_mode_auto_saves_workout(c3_env, monkeypatch, backend):
    seen = mock_provider_transport(monkeypatch, [
        semantic(SOURCE, FRAGMENT),
        tool(SOURCE, FRAGMENT, "record_workout"),
        native({"records": [{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8,
                             "inferred_fields": [], "record_date": None}]}),
    ])
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    monkeypatch.setattr(harness.settings, "chat_default_mode", "auto", raising=False)

    status, payload = harness.post_json(SOURCE, mode=None, request_id="auto-write-1")

    assert status == 200, payload
    print("SIM_WRITE=" + json.dumps({
        "status": status, "answer": payload.get("answer"),
        "declared_mode": payload["metrics"].get("declared_mode"),
        "execution_mode": payload["metrics"].get("execution_mode"),
        "react_write": payload["metrics"].get("react_write"),
        "receipts": payload["metrics"].get("receipt_refs"),
        "provider_calls": len(seen)}, ensure_ascii=False))
    metrics = payload["metrics"]
    assert metrics["declared_mode"] == "auto", metrics
    assert metrics["execution_mode"] == "react_write", metrics
    assert metrics["mode_reason"] == "auto_write", metrics
    assert metrics["react_write"] == "completed", metrics
    assert len(seen) == 3
    with c3_env.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM workout_session_sets")).scalar_one() == 4
