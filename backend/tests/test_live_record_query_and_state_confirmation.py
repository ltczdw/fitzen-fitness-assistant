"""Live-path regressions for USER-LIVE29-07/P2 and USER-LIVE29-03.

The tests use the real ASGI/graph/gateway/SQL path over an isolated SQLite
database. Provider calls are scripted MockTransport responses; no real model,
network, personal database or paid call is used.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import text

from app.core import clock
from app.db.food_repository import log_food_items
from app.db.memory_models import MemoryState, MemoryTurn
from app.db.repositories import create_workout
from app.schemas.chat import ChatRequest
from app.services.agent.food_calculator import get_pending_food, remember_pending_food
from app.services.agent.food_parser import FoodInput
from app.services.memory import compact, lifecycle as lc
from tests.react_phase_c1_support import isolated
from tests.react_phase_c2_2_support import ensure_catalog
from tests.react_phase_c2_3_support import (
    API_OWNER,
    C3HTTPHarness,
    business_keys,
    food_rows,
    mock_provider_transport,
)


SOURCE = "我今天早餐已经保存的食物和克数分别是什么？不要补做法，不要估量，也不要再次保存。"
CLEAR = "解除本会话刚才的临时器械限制。"
SET = "本会话临时只有哑铃可用，先按这个条件。"


@pytest.fixture
def live_db(tmp_path, monkeypatch):
    with isolated(monkeypatch, tmp_path / "live-record-state", "sqlite") as engine:
        ensure_catalog(engine)
        yield engine


def _safety() -> dict:
    return {"level": "normal", "evidence": "", "categories": [], "reason": "scripted"}


def _semantic(source: str, *, task_type: str = "nutrition_record",
              action: str = "question", reply_purpose: str = "fact_recall",
              goal_role: str = "result", context_states=None,
              record_query: dict | None = None) -> dict:
    if record_query is None and task_type == "nutrition_record" and reply_purpose == "fact_recall":
        today = clock.today().isoformat()
        record_query = {"date_start": today, "date_end": today,
                        "meal_types": ["早餐"], "food_names": [], "exercises": []}
    task = {
        "task_type": task_type, "action": action, "text": source,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        "goal_role": goal_role, "quoted": False,
        "history_requirement": "none", "fact_text": None,
        "reply_mode": "conversation", "reply_purpose": reply_purpose,
        "context_ids": [], "fact_fields": [], "critical_fact_fields": [],
        "record_query": record_query,
    }
    return {
        "tasks": [task], "safety": _safety(),
        "reply_mode": "conversation", "reply_purpose": reply_purpose,
        "context_states": list(context_states or []),
        "memory_requests": [], "references": [], "dialogue_ids": [],
        "critical_history_ids": [], "critical_fact_fields": [],
        "current_is_critical": False, "needs_clarification": False,
    }


def _tool_message(name: str, arguments: dict) -> dict:
    return {
        "choices": [{
            "message": {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call-" + name, "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }]},
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _wire(understanding: dict, *, advice: str = "可以，先按当前条件安排。",
          selected_text: str = "燕麦", prompts: list[list[dict]] | None = None,
          semantics: list[dict] | None = None,
          semantic_calls: list[dict] | None = None):
    seen = []
    sequence = list(semantics) if semantics is not None else None

    def handler(request):
        body = json.loads(request.content)
        seen.append(str(request.url))
        messages = body.get("messages") or []
        tools = {(item.get("function") or {}).get("name") for item in body.get("tools") or []}
        chosen = ((body.get("tool_choice") or {}).get("function") or {}).get("name") \
            if isinstance(body.get("tool_choice"), dict) else None
        if chosen == "return_request_understanding":
            value = sequence.pop(0) if sequence else understanding
            if semantic_calls is not None:
                semantic_calls.append(value)
            return httpx.Response(200, json=_tool_message("return_request_understanding", value))
        if chosen == "select_current_saved_record_sources":
            payload = json.loads(messages[-1]["content"])
            sources = payload.get("sources") or {}
            evidence = []
            for source_id, quote in sources.items():
                if selected_text in str(quote):
                    evidence.append({"source_id": source_id, "quote": quote})
                    break
            return httpx.Response(200, json=_tool_message(chosen, {"evidence": evidence}))
        if chosen == "return_dialogue_answer":
            if prompts is not None:
                prompts.append(messages)
            return httpx.Response(200, json=_tool_message(chosen, {
                "answer": advice, "requires_grounding": False}))
        if any(item.get("role") == "tool" for item in messages):
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "已完成"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1}})
        if "advise_nutrition" in tools:
            return httpx.Response(200, json=_tool_message("advise_nutrition", {
                "query": understanding["tasks"][0]["text"],
                "source_ref": {"start": 0, "end": len(understanding["tasks"][0]["text"]),
                               "exact_text": understanding["tasks"][0]["text"]}}))
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": advice},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1}})

    return mock_provider_transport, handler, seen


def _seed_food(*, user: str = API_OWNER, when: datetime | None = None,
               meal: str = "早餐", name: str = "燕麦", grams: float = 45.0,
               key: str = "oats") -> None:
    log_food_items(
        user,
        [{
            "food_id": None, "raw_food_name": name, "food_name": name,
            "weight_g": grams, "energy_kcal": grams * 3.8, "protein_g": grams * 0.13,
            "carb_g": grams * 0.66, "fat_g": grams * 0.07, "fiber_g": grams * 0.10,
            "is_estimated": False, "weight_source": "user_input",
            "match_method": "exact", "match_confidence": 1.0,
        }],
        {}, key, key, record_time=when or clock.now(), meal_type=meal)


def _no_fixed_clock(monkeypatch):
    fixed = datetime(2026, 10, 29, 1, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(clock, "utc_now", lambda: fixed)


def _ask(harness, source: str, *, mode: str, transport: str, rid: str):
    if transport == "json":
        status, payload = harness.post_json(source, mode=mode, request_id=rid)
        return status, payload["answer"], payload
    status, events, _ = harness.stream(source, mode=mode, request_id=rid)
    answer = "".join(str(frame.get("content") or "") for frame in events)
    return status, answer, events[-1]


@pytest.mark.parametrize("transport", ["json", "sse"])
def test_legacy_food_query_returns_saved_user_food_log(live_db, monkeypatch, transport):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    _seed_food()
    understanding = _semantic(SOURCE)
    factory, handler, seen = _wire(understanding)
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id="live-food-legacy")
    rid = "live-food-legacy-" + transport + "-" + uuid4().hex[:8]
    status, answer, payload = _ask(harness, SOURCE, mode="legacy", transport=transport, rid=rid)
    assert status == 200
    assert "燕麦" in answer and "45" in answer
    assert "没有找到" not in answer
    assert food_rows(live_db)[0]["weight_g"] == 45
    # SIM30-A: a fully pushed-down date/meal query is rendered from the SQL
    # rows directly; no second source-selection model call remains.
    assert len(seen) == 1, seen  # semantic decision only
    replay = harness.post_json(SOURCE, mode="legacy", request_id=rid)
    assert replay[0] == 200 and replay[1]["replayed"] is True
    assert replay[1]["answer"] == answer
    assert len(seen) == 1


@pytest.mark.parametrize("transport", ["json", "sse"])
def test_react_readonly_food_query_returns_saved_user_food_log(live_db, monkeypatch, transport):
    from app.core.config import settings
    monkeypatch.setattr(settings, "react_readonly_enabled", True)
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    _seed_food()
    understanding = _semantic(SOURCE)
    factory, handler, seen = _wire(understanding)
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id="live-food-react")
    before = business_keys(live_db)
    rid = "live-food-react-" + transport + "-" + uuid4().hex[:8]
    status, answer, payload = _ask(harness, SOURCE, mode="react_readonly", transport=transport, rid=rid)
    assert status == 200
    assert "燕麦" in answer and "45" in answer
    assert payload["metrics"]["requested_mode"] == "react_readonly"
    assert payload["metrics"]["execution_mode"] == "react_readonly"
    assert payload["metrics"]["mode_reason"] == "pure_read"
    assert business_keys(live_db) == before
    replay = harness.post_json(SOURCE, mode="react_readonly", request_id=rid)
    assert replay[0] == 200 and replay[1]["replayed"] is True
    assert replay[1]["answer"] == answer
    assert business_keys(live_db) == before


def test_food_query_excludes_other_user_and_other_meal(live_db, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    _seed_food()
    _seed_food(when=clock.now() - timedelta(days=1), meal="午餐", name="米饭", grams=100.0, key="rice")
    _seed_food(user="other-user", name="燕麦", grams=999.0, key="other-oats")
    understanding = _semantic(SOURCE)
    factory, handler, seen = _wire(understanding, selected_text="燕麦 45克")
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id="live-food-exclusion")
    status, answer, payload = _ask(harness, SOURCE, mode="legacy", transport="json",
                                   rid="live-food-exclusion-" + uuid4().hex[:8])
    assert status == 200
    assert "燕麦" in answer and "45" in answer
    assert "米饭" not in answer and "999" not in answer
    assert food_rows(live_db, owner="other-user")[0]["weight_g"] == 999


def test_food_name_filter_runs_in_sql_before_source_choice(live_db, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    _seed_food(meal="早餐", name="燕麦", grams=45.0, key="oats-named")
    _seed_food(when=clock.now() + timedelta(microseconds=1), meal="早餐",
               name="米饭", grams=100.0, key="rice-named")
    today = clock.today().isoformat()
    understanding = _semantic(SOURCE, record_query={
        "date_start": today, "date_end": today, "meal_types": ["早餐"],
        "food_names": ["燕麦"], "exercises": []})
    factory, handler, seen = _wire(understanding, selected_text="燕麦 45克")
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id="live-food-name")
    status, answer, payload = _ask(
        harness, SOURCE, mode="legacy", transport="json",
        rid="live-food-name-" + uuid4().hex[:8])
    assert status == 200
    assert "燕麦" in answer and "45" in answer
    assert "米饭" not in answer


def test_workout_query_uses_structured_action_and_date(live_db, monkeypatch):
    from app.services.agent import dialogue_context as dc
    _no_fixed_clock(monkeypatch)
    today = clock.today()
    create_workout(API_OWNER, "卧推", 3, 8, 60, date=today)
    create_workout(API_OWNER, "深蹲", 4, 8, 80, date=today)
    query = {"date_start": today.isoformat(), "date_end": today.isoformat(),
             "meal_types": [], "food_names": [], "exercises": ["卧推"]}
    sources, attempted, error, truncated = dc._saved_record_sources(
        API_OWNER, "workout_record", query)
    assert attempted and error is None and truncated is False
    values = list(sources.values())
    assert len(values) == 1 and "卧推" in values[0] and "深蹲" not in values[0]


def test_missing_record_query_is_corrected_once_with_the_gateway_mock(live_db, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    _seed_food()
    missing = _semantic(SOURCE)
    missing["tasks"][0].pop("record_query", None)
    fixed = _semantic(SOURCE)
    calls: list[dict] = []
    factory, handler, seen = _wire(fixed, semantics=[missing, fixed],
                                   semantic_calls=calls)
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id="live-food-correction")
    status, answer, payload = _ask(
        harness, SOURCE, mode="legacy", transport="json",
        rid="live-food-correction-" + uuid4().hex[:8])
    assert status == 200
    assert "燕麦" in answer and "45" in answer
    assert len(calls) == 2


def test_missing_record_query_correction_failure_is_unavailable(live_db, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    _seed_food()
    missing = _semantic(SOURCE)
    missing["tasks"][0].pop("record_query", None)
    calls: list[dict] = []
    factory, handler, seen = _wire(missing, semantics=[missing] * 5,
                                   semantic_calls=calls)
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id="live-food-no-correction")
    status, answer, payload = _ask(
        harness, SOURCE, mode="legacy", transport="json",
        rid="live-food-no-correction-" + uuid4().hex[:8])
    assert status == 200
    assert "燕麦" not in answer
    assert "没有找到" not in answer
    assert "无法" in answer or "未完成安全语义" in answer


def test_urgent_record_without_query_scope_is_not_forced_into_correction(live_db, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    urgent = _semantic(SOURCE, task_type="safety_triage", action="question",
                       reply_purpose="general")
    urgent["tasks"][0].pop("record_query", None)
    urgent["tasks"][0]["reply_mode"] = "grounded"
    urgent["reply_purpose"] = "general"
    urgent["reply_mode"] = "grounded"
    urgent["safety"] = {"level": "urgent", "evidence": SOURCE,
                        "categories": [], "reason": "scripted urgent"}
    calls: list[dict] = []
    factory, handler, seen = _wire(urgent, semantics=[urgent], semantic_calls=calls)
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id="live-food-urgent")
    status, answer, payload = _ask(
        harness, SOURCE, mode="legacy", transport="json",
        rid="live-food-urgent-" + uuid4().hex[:8])
    assert status == 200
    assert len(calls) == 1
    assert "急" in answer or "就医" in answer


def test_record_query_long_range_is_bounded_by_sql_rows(live_db, monkeypatch):
    from app.services.agent import dialogue_context as dc
    _no_fixed_clock(monkeypatch)
    _seed_food(when=clock.now() - timedelta(days=40), key="long-range-a")
    _seed_food(when=clock.now() - timedelta(days=39), key="long-range-b")
    today = clock.today()
    query = {"date_start": (today - timedelta(days=60)).isoformat(),
             "date_end": today.isoformat(), "meal_types": ["早餐"],
             "food_names": [], "exercises": []}
    sources, attempted, error, truncated = dc._saved_record_sources(
        API_OWNER, "nutrition_record", query, limit=1)
    assert attempted and error is None
    assert len(sources) == 1 and "燕麦" in next(iter(sources.values()))
    assert truncated is True  # the long date range is allowed; SQL row cap still applies


def test_english_breakfast_protocol_maps_to_stored_chinese_label(live_db, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    _seed_food(meal="早餐", name="燕麦", grams=45.0, key="english-breakfast")
    today = clock.today().isoformat()
    understanding = _semantic(SOURCE, record_query={
        "date_start": today, "date_end": today, "meal_types": ["breakfast"],
        "food_names": [], "exercises": []})
    factory, handler, seen = _wire(understanding)
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id="live-food-english")
    status, answer, payload = _ask(
        harness, SOURCE, mode="legacy", transport="json",
        rid="live-food-english-" + uuid4().hex[:8])
    assert status == 200
    assert "燕麦" in answer and "45" in answer
    assert "早餐" in answer


def test_custom_meal_label_is_preserved_not_rewritten(live_db, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    _seed_food(meal="训练前加餐", name="香蕉", grams=100.0, key="custom-label")
    today = clock.today().isoformat()
    understanding = _semantic(SOURCE, record_query={
        "date_start": today, "date_end": today, "meal_types": ["训练前加餐"],
        "food_names": ["香蕉"], "exercises": []})
    factory, handler, seen = _wire(understanding, selected_text="香蕉 100克")
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id="live-food-custom")
    status, answer, payload = _ask(
        harness, SOURCE, mode="legacy", transport="json",
        rid="live-food-custom-" + uuid4().hex[:8])
    assert status == 200
    assert "香蕉" in answer and "100" in answer
    assert "燕麦" not in answer


def test_real_food_write_same_conversation_query_and_cold_paths(live_db, monkeypatch):
    """Actual save chain, then the same conversation, then cold React/legacy."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    monkeypatch.setattr(settings, "react_readonly_enabled", True)
    monkeypatch.setattr(settings, "app_timezone", "Asia/Shanghai")
    instant = datetime(2026, 10, 29, 16, 30, tzinfo=timezone.utc)  # 2026-10-30 local
    monkeypatch.setattr(clock, "utc_now", lambda: instant)
    ensure_catalog(live_db)
    save_source = "今天早餐实际吃了45克燕麦，请保存成今天早餐的饮食记录。"
    span = "今天早餐实际吃了45克燕麦"
    save_understanding = _semantic(save_source, task_type="nutrition_record",
                                   action="record", reply_purpose="general")
    save_understanding["tasks"][0]["reply_mode"] = "grounded"
    save_understanding["tasks"][0]["fact_text"] = span
    save_understanding["reply_mode"] = "grounded"
    parser = {"tasks": [{"task_type": "nutrition_record", "action": "record",
                         "text": save_source, "is_negated": False,
                         "is_hypothetical": False, "is_third_party": False}]}
    foods = [{"source_text": span, "name": "燕麦", "quantity": 1.0,
              "unit": "克", "weight_g": 45.0, "meal_type": "早餐",
              "record_date": None, "is_negated": False}]

    def save_handler(request):
        body = json.loads(request.content)
        chosen = ((body.get("tool_choice") or {}).get("function") or {}).get("name") \
            if isinstance(body.get("tool_choice"), dict) else None
        tools = {(item.get("function") or {}).get("name") for item in body.get("tools") or []}
        system = " ".join(str(item.get("content") or "") for item in body.get("messages") or []
                          if item.get("role") == "system")
        if chosen == "return_request_understanding":
            return httpx.Response(200, json=_tool_message("return_request_understanding",
                                                          save_understanding))
        if not tools and "饮食实体提取器" in system:
            return httpx.Response(200, json={"choices": [{
                "message": {"role": "assistant", "content": json.dumps({"foods": foods}, ensure_ascii=False)},
                "finish_reason": "stop"}], "usage": {"prompt_tokens": 6, "completion_tokens": 4}})
        if not tools:
            return httpx.Response(200, json={"choices": [{
                "message": {"role": "assistant", "content": json.dumps(parser, ensure_ascii=False)},
                "finish_reason": "stop"}], "usage": {"prompt_tokens": 6, "completion_tokens": 4}})
        return httpx.Response(200, json={"choices": [{
            "message": {"role": "assistant", "content": "已记录早餐燕麦45克。"},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}})

    mock_provider_transport(monkeypatch, [save_handler])
    cid = "live-food-chain"
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id=cid)
    save_rid = "live-food-chain-save-" + uuid4().hex[:8]
    save_status, saved = harness.post_json(save_source, mode="legacy", request_id=save_rid)
    assert save_status == 200, saved
    assert saved["intent"] == "nutrition_record", saved
    rows = food_rows(live_db)
    assert len(rows) == 1 and rows[0]["weight_g"] == 45
    with live_db.connect() as connection:
        assert connection.execute(text(
            "SELECT meal_type FROM user_food_log WHERE user_id=:owner"),
            {"owner": API_OWNER}).scalar() == "早餐"
    assert get_pending_food(API_OWNER, cid) is None
    before_query = business_keys(live_db)
    today = clock.today().isoformat()
    query_understanding = _semantic(SOURCE, record_query={
        "date_start": today, "date_end": today, "meal_types": ["breakfast"],
        "food_names": [], "exercises": []})
    factory, handler, seen = _wire(query_understanding)
    factory(monkeypatch, [handler])
    query_rid = "live-food-chain-query-" + uuid4().hex[:8]
    status, answer, payload = _ask(
        harness, SOURCE, mode="react_readonly", transport="json", rid=query_rid)
    assert status == 200
    assert "燕麦" in answer and "45" in answer
    assert payload["metrics"]["execution_mode"] == "react_readonly"
    assert payload["metrics"]["mode_reason"] == "pure_read"
    assert business_keys(live_db) == before_query
    calls_after_query = len(seen)
    replay = harness.post_json(SOURCE, mode="react_readonly", request_id=query_rid)
    assert replay[0] == 200 and replay[1]["replayed"] is True
    assert replay[1]["answer"] == answer and len(seen) == calls_after_query
    assert business_keys(live_db) == before_query

    for mode, cid_cold in (("react_readonly", "live-food-chain-cool-r"),
                           ("legacy", "live-food-chain-cool-l")):
        cold_understanding = _semantic(SOURCE, record_query={
            "date_start": today, "date_end": today, "meal_types": ["breakfast"],
            "food_names": [], "exercises": []})
        cold_factory, cold_handler, _ = _wire(cold_understanding)
        cold_factory(monkeypatch, [cold_handler])
        cold_harness = C3HTTPHarness(monkeypatch, live_db, conversation_id=cid_cold)
        cold_status, cold_answer, cold_payload = _ask(
            cold_harness, SOURCE, mode=mode, transport="json",
            rid=cid_cold + "-" + uuid4().hex[:8])
        assert cold_status == 200
        assert "燕麦" in cold_answer and "45" in cold_answer
        assert business_keys(live_db) == before_query


def test_valid_pending_keeps_legacy_but_preserves_business_read_scope(live_db, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "react_readonly_enabled", True)
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    _seed_food(meal="早餐", name="燕麦", grams=45.0, key="pending-query")
    cid = "live-food-pending-query"
    remember_pending_food(
        API_OWNER, cid,
        (FoodInput(raw_text="燕麦45克", requested_name="燕麦", quantity=1.0,
                   unit="克", weight_g=45.0, portion_name=None),),
        raw_message="记录燕麦45克", idempotency_key="pending-query-key",
        meal_group_id="pending-query-group")
    assert get_pending_food(API_OWNER, cid) is not None
    before = business_keys(live_db)
    today = clock.today().isoformat()
    understanding = _semantic(SOURCE, record_query={
        "date_start": today, "date_end": today, "meal_types": ["breakfast"],
        "food_names": [], "exercises": []})
    factory, handler, _ = _wire(understanding)
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id=cid)
    status, answer, payload = _ask(
        harness, SOURCE, mode="react_readonly", transport="json",
        rid="live-food-pending-" + uuid4().hex[:8])
    assert status == 200
    assert "燕麦" in answer and "45" in answer
    assert payload["metrics"]["execution_mode"] == "legacy"
    assert payload["metrics"]["mode_reason"] == "contains_write_or_confirmation"
    assert business_keys(live_db) == before
    assert get_pending_food(API_OWNER, cid) is not None


def test_food_query_truncation_is_not_reported_as_complete_or_empty(live_db, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    for index in range(31):
        _seed_food(when=clock.now() + timedelta(microseconds=index),
                   meal="早餐", name="燕麦", grams=45.0, key=f"many-oats-{index}")
    understanding = _semantic(SOURCE)
    factory, handler, seen = _wire(understanding)
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id="live-food-truncated")
    status, answer, payload = _ask(
        harness, SOURCE, mode="legacy", transport="json",
        rid="live-food-truncated-" + uuid4().hex[:8])
    assert status == 200
    assert "仅为部分结果" in answer
    assert "没有找到" not in answer
    assert "燕麦" in answer


def test_food_query_successful_empty_and_sql_failure_are_distinct(live_db, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    understanding = _semantic(SOURCE)
    factory, handler, seen = _wire(understanding)
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id="live-food-empty")
    status, answer, payload = _ask(harness, SOURCE, mode="legacy", transport="json",
                                   rid="live-food-empty-" + uuid4().hex[:8])
    assert status == 200
    assert "没有找到" in answer

    from app.services.agent import dialogue_context
    original = dialogue_context._saved_record_sources

    def broken(*args, **kwargs):
        return {}, True, "saved_record_read_failed", False
    monkeypatch.setattr(dialogue_context, "_saved_record_sources", broken)
    try:
        status2, answer2, payload2 = _ask(
            harness, SOURCE, mode="legacy", transport="json",
            rid="live-food-failed-" + uuid4().hex[:8])
    finally:
        monkeypatch.setattr(dialogue_context, "_saved_record_sources", original)
    assert status2 == 200
    assert "读取已保存记录暂时失败" in answer2
    assert "没有找到" not in answer2


def _seed_state(cid: str) -> None:
    semantic = {**_semantic(SET, task_type="chitchat", action="answer",
                            reply_purpose="general", goal_role="result",
                            context_states=[{
                                "key": "available_equipment", "quote": "哑铃",
                                "action": "set", "scope": "conversation",
                                "domain": "training"}]),
                "safety_assessed": True}
    tid, _ = compact.begin_turn(ChatRequest(
        message=SET, conversation_id=cid, request_id="seed-" + cid))
    compact.finish_turn(tid, "已记录。", {"answer": "已记录。"},
                        semantic_values=[(SET, semantic)])


def test_clear_state_is_confirmed_after_commit_and_replayed_once(live_db, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    cid = "live-state-clear"
    _seed_state(cid)
    assert "context:training:available_equipment" in lc.snapshot(cid)["current_states"]
    understanding = _semantic(CLEAR, task_type="chitchat", action="answer",
                              reply_purpose="general", goal_role="result",
                              context_states=[{
                                  "key": "available_equipment", "quote": CLEAR,
                                  "action": "clear", "scope": "conversation",
                                  "domain": "training"}])
    prompts: list[list[dict]] = []
    factory, handler, seen = _wire(understanding, advice="可以，按当前可用器械安排。",
                                   prompts=prompts)
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id=cid)
    rid = "live-state-clear-" + uuid4().hex[:8]
    status, answer, payload = _ask(harness, CLEAR, mode="legacy", transport="json", rid=rid)
    assert status == 200
    assert prompts
    system_prompt = str(prompts[0][0].get("content") or "")
    assert "服务端会话临时条件" in system_prompt
    assert "本路径没有任何保存/修改工具" not in system_prompt
    assert "可以，按当前可用器械安排。" in answer
    assert "本会话已解除临时条件。" in answer
    assert "没有保存或修改工具" not in answer
    assert "不会作为一条记录留存" not in answer
    assert not lc.snapshot(cid)["current_states"]
    replay = harness.post_json(CLEAR, mode="legacy", request_id=rid)
    assert replay[0] == 200 and replay[1]["replayed"] is True
    assert replay[1]["answer"] == answer
    assert len(seen) <= 2


def test_clear_state_confirmation_survives_sse_final_response(live_db, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    cid = "live-state-clear-sse"
    _seed_state(cid)
    understanding = _semantic(CLEAR, task_type="chitchat", action="answer",
                              reply_purpose="general", goal_role="result",
                              context_states=[{
                                  "key": "available_equipment", "quote": CLEAR,
                                  "action": "clear", "scope": "conversation",
                                  "domain": "training"}])
    factory, handler, seen = _wire(understanding, advice="可以，按当前可用器械安排。")
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id=cid)
    status, events, _ = harness.stream(CLEAR, mode="legacy",
                                       request_id="live-state-clear-sse-" + uuid4().hex[:8])
    answer = "".join(str(frame.get("content") or "") for frame in events)
    assert status == 200
    assert "可以，按当前可用器械安排。" in answer
    assert "本会话已解除临时条件。" in answer
    assert not lc.snapshot(cid)["current_states"]


def test_state_update_without_commit_does_not_claim_success(live_db, monkeypatch):
    _no_fixed_clock(monkeypatch)
    cid = "live-state-uncommitted"
    source = CLEAR
    semantic = _semantic(source, task_type="chitchat", action="answer",
                         reply_purpose="general", goal_role="result",
                         context_states=[{
                             "key": "available_equipment", "quote": source,
                             "action": "clear", "scope": "conversation",
                             "domain": "training"}])
    response = {"answer": "正在处理。", "metrics": {"state_update_requested": True}}
    tid, _ = compact.begin_turn(ChatRequest(
        message=source, conversation_id=cid, request_id="uncommitted-" + cid))
    committed = compact.finish_turn(tid, response["answer"], response,
                                    semantic_values=[(source, semantic)])
    assert "本次会话临时条件未更新或未确认。" in committed["answer"]
    assert "已解除" not in committed["answer"]
    assert lc.snapshot(cid)["current_states"] == {}


def test_reference_state_does_not_append_confirmation_or_renew_ttl(live_db, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    cid = "live-state-reference"
    _seed_state(cid)
    before = lc.snapshot(cid)["current_states"]["context:training:available_equipment"]
    understanding = _semantic("按刚才的器械条件给建议。", task_type="chitchat",
                              action="answer", reply_purpose="general",
                              goal_role="result", context_states=[{
                                  "key": "available_equipment", "quote": "刚才的器械条件",
                                  "action": "reference", "scope": "conversation",
                                  "domain": "training"}])
    factory, handler, seen = _wire(understanding, advice="按刚才的条件继续。")
    factory(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, live_db, conversation_id=cid)
    status, answer, payload = _ask(
        harness, "按刚才的器械条件给建议。", mode="legacy", transport="json",
        rid="live-state-reference-" + uuid4().hex[:8])
    assert status == 200
    assert answer == "按刚才的条件继续。"
    after = lc.snapshot(cid)["current_states"]["context:training:available_equipment"]
    assert after["expires_at"] == before["expires_at"]
    assert after["state_id"] == before["state_id"]


def test_mixed_advice_keeps_body_and_appends_state_confirmation(live_db, monkeypatch):
    _no_fixed_clock(monkeypatch)
    cid = "live-state-mixed"
    _seed_state(cid)
    source = "解除本会话刚才的临时器械限制，然后给我一段独立练习建议。"
    semantic = {**_semantic(source, task_type="chitchat", action="answer",
                            reply_purpose="general", goal_role="result",
                            context_states=[{
                                "key": "available_equipment", "quote": "解除本会话刚才的临时器械限制",
                                "action": "clear", "scope": "conversation",
                                "domain": "training"}]),
                "safety_assessed": True}
    semantic["tasks"].append({
        "task_type": "nutrition_advice", "action": "question", "text": "然后给我一段独立练习建议。",
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        "goal_role": "result", "quoted": False, "history_requirement": "none",
        "fact_text": None, "reply_mode": "grounded", "reply_purpose": "general",
        "context_ids": [], "fact_fields": [], "critical_fact_fields": [],
    })
    response = {"answer": "建议正文：先做低强度全身练习。", "intent": "training_qa",
                "metrics": {}}
    tid, _ = compact.begin_turn(ChatRequest(
        message=source, conversation_id=cid, request_id="mixed-" + cid))
    committed = compact.finish_turn(tid, response["answer"], response,
                                    semantic_values=[(source, semantic)])
    assert committed["answer"].startswith("建议正文：先做低强度全身练习。")
    assert "本会话已解除临时条件。" in committed["answer"]
    assert not lc.snapshot(cid)["current_states"]


def test_late_old_clear_does_not_claim_success(live_db):
    cid = "live-state-late-clear"
    seed_tid, _ = compact.begin_turn(ChatRequest(
        message="种子状态", conversation_id=cid, request_id="late-clear-seed"))
    compact.finish_turn(seed_tid, "种子", {"answer": "种子"}, semantic_values=[])
    with compact.session() as session, session.begin():
        session.add(MemoryState(
            state_id="newer-equipment", owner_id=API_OWNER, conversation_id=cid,
            scope="conversation", logical_key="training:available_equipment",
            kind="transient", domain="training", value="哑铃", version=9,
            status="active", valid_from=compact.now(),
            expires_at=compact.now() + timedelta(hours=1),
            source_turn_id=seed_tid, source_message_id=seed_tid + ":user",
            memory_epoch=0,
        ))
    source = CLEAR
    semantic = {**_semantic(source, task_type="chitchat", action="answer",
                            reply_purpose="general", goal_role="result",
                            context_states=[{
                                "key": "available_equipment", "quote": source,
                                "action": "clear", "scope": "conversation",
                                "domain": "training"}]),
                "safety_assessed": True}
    tid, _ = compact.begin_turn(ChatRequest(
        message=source, conversation_id=cid, request_id="late-clear"))
    # Simulate a delayed old completion whose source version is behind v9.
    with compact.session() as session, session.begin():
        row = session.get(MemoryTurn, tid)
        first = dict(row.messages[0])
        first["app_meta"] = {**(first.get("app_meta") or {}), "_memory_version": 5}
        row.messages = [first, *row.messages[1:]]
    committed = compact.finish_turn(
        tid, "建议正文。", {"answer": "建议正文。", "metrics": {}},
        semantic_values=[(source, semantic)])
    assert "建议正文。" in committed["answer"]
    assert "未更新：已有更新版本" in committed["answer"]
    assert "已解除" not in committed["answer"]
    current = lc.snapshot(cid)["current_states"]["context:training:available_equipment"]
    assert current["text"] == "哑铃"


def test_record_read_without_query_fails_closed_in_both_modes(live_db, monkeypatch):
    """A current record read with no structured query must not answer from memory."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "react_readonly_enabled", True)
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0,0", raising=False)
    _no_fixed_clock(monkeypatch)
    _seed_food()
    for mode in ("legacy", "react_readonly"):
        understanding = _semantic(SOURCE)
        understanding["tasks"][0]["record_query"] = None
        factory, handler, seen = _wire(understanding)
        factory(monkeypatch, [handler])
        harness = C3HTTPHarness(monkeypatch, live_db, conversation_id="no-query-" + uuid4().hex[:8])
        status, answer, payload = _ask(
            harness, SOURCE, mode=mode, transport="json", rid="no-query-" + uuid4().hex[:8])
        assert status == 200
        assert "燕麦" not in answer and "没有找到" not in answer, (mode, answer)
        if mode == "legacy":
            assert "未通过校验" in answer, (mode, answer)
        else:
            assert "补充" in answer or "范围" in answer, (mode, answer)
