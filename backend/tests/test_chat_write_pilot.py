"""Current semantic record policy over real HTTP/SSE and isolated SQLite."""
import json
from uuid import uuid4

import pytest
from sqlalchemy import text

from tests.react_phase_c2_3_support import (
    C3HTTPHarness, all_business_rows, c3_env, food_rows, mock_provider_transport, set_write_flag,
)


def native(value):
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(value, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 4}}


def semantic(source, fragment, kind, **extra):
    value = {"tasks": [{"task_type": kind, "action": "record", "text": source,
                        "fact_text": fragment, "is_negated": False,
                        "is_hypothetical": False, "is_third_party": False,
                        "reply_mode": "grounded", "reply_purpose": "general",
                        "history_requirement": "none"}],
             "safety": {"level": "normal", "evidence": "", "categories": [],
                        "reason": "scripted normal"},
             **extra}
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
        "id": "semantic-result", "type": "function", "function": {
            "name": "return_request_understanding", "arguments": json.dumps(value, ensure_ascii=False)}}]}}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 4}}


def tool(source, fragment, name):
    start = source.index(fragment)
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
        "id": "native-"+name, "type": "function", "function": {"name": name,
        "arguments": json.dumps({"source_ref": {"start": start, "end": start+len(fragment), "exact_text": fragment}}, ensure_ascii=False)}}]}}],
        "usage": {"prompt_tokens": 6, "completion_tokens": 4}}


@pytest.mark.parametrize("backend", ["sqlite"])
@pytest.mark.parametrize("transport", ["json", "sse"])
def test_model_semantics_direct_estimate_save_disabled_gate_and_replay(c3_env, monkeypatch, backend, transport):
    workout = "不要忘了收进今天训练：卧推 60kg 4组8次。谢谢"
    workout_fact = "卧推 60kg 4组8次"
    food = "帮我记上今天午餐：300克神秘盖饭。谢谢"
    food_fact = "300克神秘盖饭"
    estimate = {"food_name": "神秘盖饭", "preparation_state": "unknown", "kcal": 180,
                "protein_g": 7, "carbs_g": 25, "fat_g": 6, "fiber_g": 1, "notes": "模拟估算", "confidence": .5}
    responses = [semantic(workout, workout_fact, "workout_record"), tool(workout, workout_fact, "record_workout"),
                 native({"records": [{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8,
                                      "inferred_fields": [], "record_date": None}]}),
                 semantic(food, food_fact, "nutrition_record"), tool(food, food_fact, "record_nutrition"),
                 native({"foods": [{"source_text": food_fact, "name": "神秘盖饭", "quantity": 1, "unit": "克",
                                    "weight_g": 300, "meal_type": "午餐", "record_date": None, "is_negated": False}]}),
                 native(estimate)]
    seen = mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=False)

    def send(message, rid):
        if transport == "json":
            status, payload = harness.post_json(message, request_id=rid)
        else:
            status, events, _ = harness.stream(message, request_id=rid)
            payload = {**events[-1], "answer": "".join(str(item.get("content") or "") for item in events)}
        assert status == 200
        return payload

    def state():
        with c3_env.connect() as conn:
            counters = {name: conn.execute(text("SELECT COUNT(*) FROM "+name)).scalar_one() for name in (
                "model_call_ledger", "workout_sessions", "workout_session_exercises", "workout_session_sets")}
        return {"rows": all_business_rows(c3_env), "counters": counters, "providers": len(seen)}

    initial = state()
    blocked = send(workout, str(uuid4()))
    assert blocked["metrics"]["mode_reason"] == "react_write_disabled"
    assert state() == initial
    set_write_flag(monkeypatch, True)
    saved_id = str(uuid4())
    saved = send(workout, saved_id)
    assert saved["metrics"]["react_write"] == "completed", saved
    assert saved["metrics"]["receipt_refs"]
    assert state()["counters"]["workout_session_sets"] == 4
    assert len(seen) == 3

    food_id = str(uuid4())
    estimated = send(food, food_id)
    assert estimated["metrics"]["react_write"] == "completed", estimated
    assert estimated["metrics"]["receipt_refs"]
    assert estimated["metrics"]["pending_confirmations"] == []
    assert "AI估算" in estimated["answer"]
    rows = food_rows(c3_env)
    assert len(rows) == 1 and rows[0]["weight_g"] == 300
    assert rows[0]["is_estimated"] and rows[0]["match_method"] == "ai_estimate"
    committed = state()
    assert committed["providers"] == committed["counters"]["model_call_ledger"] == 7
    replay = send(food, food_id)
    assert replay["metrics"]["receipt_refs"] == estimated["metrics"]["receipt_refs"]
    assert state() == committed
    set_write_flag(monkeypatch, False)
    disabled = send(workout, str(uuid4()))
    assert disabled["metrics"]["mode_reason"] == "react_write_disabled"
    assert state() == committed
    prior = send(workout, saved_id)
    assert prior["metrics"]["receipt_refs"] == saved["metrics"]["receipt_refs"]
    assert state() == committed


@pytest.mark.parametrize("backend", ["sqlite"])
def test_concrete_model_question_stops_before_supervisor_and_save(c3_env, monkeypatch, backend):
    source = "神秘盖饭，帮我记到午餐"
    proposal = semantic(source, "神秘盖饭", "nutrition_record")
    function = proposal["choices"][0]["message"]["tool_calls"][0]["function"]
    function.update(name="return_record_proposals", arguments=json.dumps({"records": [{
        "kind": "nutrition", "source_quote": source, "name": "神秘盖饭",
        "meal_type": "午餐", "meal_quote": "午餐", "weight_g": None,
        "quantity": None, "unit": None,
    }]}, ensure_ascii=False))
    seen = mock_provider_transport(monkeypatch, [semantic(source, "神秘盖饭", "nutrition_record",
        needs_clarification=True, clarification="这份神秘盖饭大约多少克？"), proposal])
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    status, payload = harness.post_json(source, request_id=str(uuid4()))
    assert status == 200 and "多少克" in payload["answer"]
    assert not payload["metrics"].get("receipt_refs")
    assert all_business_rows(c3_env) == []
    assert payload["metrics"]["execution_mode"] == "react_write"
    assert payload["metrics"]["react_write"] == "needs_clarification"
    # Understanding + source-bound draft only: no supervisor or save call.
    assert len(seen) == 2


@pytest.mark.parametrize("backend", ["sqlite"])
def test_semantic_source_forgery_cannot_dispatch_or_save(c3_env, monkeypatch, backend):
    source = "午餐一个鸡蛋"
    invalid = semantic("模型编造的鸡胸肉300克", "鸡胸肉300克", "nutrition_record")
    # Initial response and the one bounded correction both forge the source.
    seen = mock_provider_transport(monkeypatch, [invalid, invalid])
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    status, payload = harness.post_json(source, request_id=str(uuid4()))
    assert status == 200
    assert not payload["metrics"].get("receipt_refs")
    assert "未保存" in payload["answer"]
    assert all_business_rows(c3_env) == []
    assert len(seen) == 2
