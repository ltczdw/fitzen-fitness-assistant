"""USER-RETRY-01/02: source-bound fact delivery and quoted restatement.

All provider calls are mocked at the provider boundary and all SQL uses an
isolated temporary SQLite database. The tests exercise the real ASGI JSON/SSE
entry, React runtimes, read adapters, completion/finalize projection and the
replay cache; no real provider, resident service or personal data is used.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from sqlalchemy import text

from tests.react_phase_c2_3_support import (
    C3HTTPHarness,
    all_business_rows,
    c3_env,
    mock_provider_transport,
    workout_rows,
)

SAVE_SOURCE = "请记录我今天实际做的哑铃划船：24公斤，3组，每组12次。"
READ_SOURCE = "请查我今天已经保存的哑铃划船，列出重量、组数和每组次数。只查询，不重新保存。"
QUOTED_SOURCE = (
    "同事发来一句话：“我今天卧推85公斤，5组5次。”请复述他的训练，"
    "这是他的记录，不是我的，不要记到我名下。"
)


def _usage():
    return {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}


def _tool_message(name, arguments, *, call_id="call-test"):
    return {
        "choices": [{
            "message": {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": name,
                                 "arguments": json.dumps(arguments, ensure_ascii=False)},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": _usage(),
    }


def _native(value):
    return {
        "choices": [{"message": {"role": "assistant",
                                 "content": json.dumps(value, ensure_ascii=False)},
                     "finish_reason": "stop"}],
        "usage": _usage(),
    }


def _semantic(task, **extra):
    task = {**task, "history_requirement": task.get("history_requirement", "none")}
    value = {"tasks": [task], "reply_mode": task.get("reply_mode", "grounded"),
             "reply_purpose": task.get("reply_purpose", "general"),
             "relation": "new", "dialogue_ids": task.get("context_ids", []),
             "critical_history_ids": [], "critical_fact_fields": [],
             "current_is_critical": False, "references": [],
             "memory_requests": [], "needs_clarification": False,
             "safety": {"level": "normal", "evidence": "", "categories": [],
                        "reason": "scripted normal"}, **extra}
    return _tool_message("return_request_understanding", value, call_id="semantic-result")


def _save_task():
    return {
        "task_type": "workout_record", "action": "record", "text": SAVE_SOURCE,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        "goal_role": "result", "quoted": False, "fact_text": "哑铃划船：24公斤，3组，每组12次",
        "reply_mode": "grounded", "reply_purpose": "general",
        "context_ids": [], "fact_fields": [], "critical_fact_fields": [],
    }


def _read_task():
    return {
        "task_type": "workout_record", "action": "question", "text": READ_SOURCE,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        "goal_role": "result", "quoted": False, "fact_text": None,
        "reply_mode": "conversation", "reply_purpose": "fact_recall",
        "context_ids": [], "fact_fields": [], "critical_fact_fields": [],
    }


def _quoted_task():
    return {
        "task_type": "workout_record", "action": "answer", "text": QUOTED_SOURCE,
        "is_negated": True, "is_hypothetical": False, "is_third_party": True,
        "goal_role": "result", "quoted": True, "fact_text": None,
        "reply_mode": "conversation", "reply_purpose": "general",
        "context_ids": [], "fact_fields": [], "critical_fact_fields": [],
    }


def _source_ref(source, text=None):
    text = source if text is None else text
    start = source.index(text)
    return {"start": start, "end": start + len(text), "exact_text": text}


def _provider_handler(state):
    def handle(request):
        body = json.loads(request.content.decode("utf-8"))
        tools = {item.get("function", {}).get("name") for item in body.get("tools") or []}
        messages = body.get("messages") or []
        if "return_wiki_read_plan" in tools:
            # These requests use current text or business SQL, not old chat.
            return httpx.Response(200, json=_tool_message("return_wiki_read_plan", {"page_ids": [], "done": True}))
        if "return_request_understanding" in tools:
            payload = json.loads(messages[-1]["content"])
            source = str(payload.get("query") or "")
            if source == SAVE_SOURCE:
                return httpx.Response(200, json=_semantic(_save_task()))
            if source == READ_SOURCE:
                return httpx.Response(200, json=_semantic(_read_task()))
            if source == QUOTED_SOURCE:
                return httpx.Response(200, json=_semantic(_quoted_task()))
            raise AssertionError("unexpected semantic source: " + source)
        if "return_dialogue_answer" in tools:
            return httpx.Response(200, json=_tool_message(
                "return_dialogue_answer",
                {"answer": "同事今天卧推85公斤，5组5次；这是他的记录，不会保存到你名下。",
                 "requires_grounding": False,
                 "status_kind": None},
                call_id="dialogue-result"))
        if "select_current_record_sources" in tools:
            payload = json.loads(messages[-1]["content"])
            sources = payload.get("sources") or {}
            evidence = [{"source_id": source_id, "quote": text}
                        for source_id, text in sources.items()
                        if str(source_id).startswith("workout:")]
            return httpx.Response(200, json=_tool_message(
                "select_current_record_sources", {"evidence": evidence},
                call_id="record-source-selection"))
        if not tools:
            # The workout extractor uses the plain JSON completion protocol.
            return httpx.Response(200, json=_native({"records": [{
                "exercise": "哑铃划船", "weight_kg": 24, "sets": 3, "reps": 12,
                "inferred_fields": [], "record_date": None,
            }]}))
        source = next((str(item.get("content") or "") for item in messages
                       if item.get("role") == "user"), "")
        if source == SAVE_SOURCE:
            if tools & {"record_workout", "record_nutrition"}:
                return httpx.Response(200, json=_tool_message(
                    "record_workout", {"source_ref": _source_ref(SAVE_SOURCE)},
                    call_id="save-workout"))
            return httpx.Response(200, json=_native({"records": [{
                "exercise": "哑铃划船", "weight_kg": 24, "sets": 3, "reps": 12,
                "inferred_fields": [], "record_date": None,
            }]}))
        if source == READ_SOURCE:
            state["read_supervisor"] += 1
            if state["read_supervisor"] == 1:
                return httpx.Response(200, json=_tool_message(
                    "read_workout_history", {"days": 1}, call_id="read-history"))
            return httpx.Response(200, json=_tool_message(
                "answer_training",
                {"query": READ_SOURCE, "source_ref": _source_ref(READ_SOURCE)},
                call_id="answer-read"))
        if source == QUOTED_SOURCE:
            return httpx.Response(200, json=_tool_message(
                "answer_training",
                {"query": QUOTED_SOURCE, "source_ref": _source_ref(QUOTED_SOURCE)},
                call_id="answer-quoted"))
        raise AssertionError("unexpected supervisor source: " + source)

    return handle


def _business_snapshot(engine):
    with engine.connect() as conn:
        counters = {
            name: conn.execute(text("SELECT COUNT(*) FROM " + name)).scalar_one()
            for name in ("workout_records", "workout_sessions",
                         "workout_session_exercises", "workout_session_sets",
                         "user_food_log", "nutrition_logs")
        }
    return {"rows": all_business_rows(engine), "counters": counters}


@pytest.mark.parametrize("backend", ["sqlite"])
def test_saved_workout_fact_recall_delivers_real_values_and_replays(
    c3_env, monkeypatch, backend,
):
    from app.core.config import settings

    monkeypatch.setattr(settings, "react_readonly_enabled", True, raising=False)
    monkeypatch.setattr(settings, "react_write_enabled", True, raising=False)

    async def no_rag(*args, **kwargs):
        raise AssertionError("knowledge RAG must not run for a saved-record fact query")

    import app.services.rag.online as rag_online
    monkeypatch.setattr(rag_online, "search", no_rag, raising=False)

    state = {"read_supervisor": 0}
    seen = mock_provider_transport(monkeypatch, [_provider_handler(state)])
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)

    save_status, saved = harness.post_json(SAVE_SOURCE, mode="react_write",
                                           request_id="user-retry-save-01")
    assert save_status == 200, saved
    assert saved["metrics"]["react_write"] == "completed", saved
    assert workout_rows(c3_env), saved
    rows = workout_rows(c3_env)
    assert len(rows) == 1
    assert (rows[0]["exercise"], rows[0]["weight_kg"], rows[0]["sets"], rows[0]["reps"]) == (
        "哑铃划船", 24.0, 3, 12)
    before_query = _business_snapshot(c3_env)
    calls_before_query = len(seen)

    read_status, payload = harness.post_json(READ_SOURCE, mode="react_readonly",
                                             request_id="user-retry-read-01")
    assert read_status == 200, payload
    assert payload["metrics"]["execution_mode"] == "react_readonly", payload
    assert payload["metrics"]["react_status"] == "complete", payload
    assert "哑铃划船" in payload["answer"]
    assert "24" in payload["answer"] and "3" in payload["answer"] and "12" in payload["answer"]
    assert "执行上限" not in payload["answer"]
    assert "证据不足" not in payload["answer"]
    assert payload["metrics"]["model_policy_stops"] == []
    assert _business_snapshot(c3_env) == before_query
    assert len(seen) > calls_before_query

    ids = (payload["request_id"], payload["message_id"], payload["run_id"])
    calls_after_read = len(seen)

    replay_status, replay = harness.post_json(READ_SOURCE, mode="react_readonly",
                                              request_id="user-retry-read-01")
    assert replay_status == 200 and replay["replayed"] is True
    assert replay["intent"] == payload["intent"]
    assert replay["answer"] == payload["answer"]
    assert (replay["request_id"], replay["message_id"], replay["run_id"]) == ids
    assert len(seen) == calls_after_read
    assert _business_snapshot(c3_env) == before_query

    sse_status, events, _ = harness.stream(READ_SOURCE, mode="react_readonly",
                                           request_id="user-retry-read-01")
    assert sse_status == 200 and events
    final = events[-1]
    assert "".join(str(frame.get("content") or "") for frame in events) == payload["answer"]
    assert (final["request_id"], final["message_id"], final["run_id"]) == ids
    assert len(seen) == calls_after_read
    assert _business_snapshot(c3_env) == before_query


@pytest.mark.parametrize("backend", ["sqlite"])
def test_quoted_third_party_restatement_skips_rag_and_stays_zero_write(
    c3_env, monkeypatch, backend,
):
    from app.core.config import settings

    monkeypatch.setattr(settings, "react_write_enabled", True, raising=False)

    async def no_rag(*args, **kwargs):
        raise AssertionError("quoted restatement must not request knowledge RAG")

    import app.services.rag.online as rag_online
    monkeypatch.setattr(rag_online, "search", no_rag, raising=False)

    state = {"read_supervisor": 0}
    seen = mock_provider_transport(monkeypatch, [_provider_handler(state)])
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    before = _business_snapshot(c3_env)

    status, payload = harness.post_json(QUOTED_SOURCE, mode="react_write",
                                        request_id="user-retry-quoted-02")
    assert status == 200, payload
    assert payload["metrics"]["react_write"] == "completed", payload
    assert "85" in payload["answer"] and "5" in payload["answer"]
    assert "同事" in payload["answer"]
    assert "不会保存" in payload["answer"]
    for stale in ("证据不足", "总负荷", "2125", "执行上限"):
        assert stale not in payload["answer"]
    assert _business_snapshot(c3_env) == before
    assert payload["metrics"]["receipt_refs"] == []

    # A real first-person record still saves after the quoted request.
    save_status, saved = harness.post_json(SAVE_SOURCE, mode="react_write",
                                           request_id="user-retry-save-after-quote")
    assert save_status == 200 and saved["metrics"]["react_write"] == "completed", saved
    assert len(workout_rows(c3_env)) == 1
    assert saved["metrics"]["receipt_refs"]


def _install_selection(monkeypatch, choose):
    """Mock the source-selection protocol; choose(sources) -> evidence rows."""
    from app.services.agent.agent import agent

    async def structured(messages, schema, *, name=None, validator=None, **kwargs):
        assert name == "select_current_record_sources"
        payload = json.loads(messages[-1]["content"])
        value = schema.model_validate({"evidence": choose(dict(payload["sources"]))})
        if validator is not None:
            validator(value)
        return value

    monkeypatch.setattr(agent.graph.llm, "configured", lambda: True)
    monkeypatch.setattr(agent.graph.llm, "complete_structured", structured)


def _select_ids(*source_ids):
    def choose(sources):
        return [{"source_id": source_id, "quote": sources[source_id]}
                for source_id in source_ids if source_id in sources]
    return choose


def _worker_context(**overrides):
    context = {
        "user_id": "demo-user",
        "conversation_id": "user-retry-worker",
        "request_id": "user-retry-worker",
        "source_text": READ_SOURCE,
        "task_text": READ_SOURCE,
        "observations": [],
        "semantic_scope": {
            "reply_mode": "conversation", "reply_purpose": "fact_recall",
            "goal_role": "result", "quoted": False, "is_third_party": False,
            "context_ids": [], "fact_fields": [], "critical_fact_fields": [],
        },
        "dialogue_scope": {
            "reply_mode": "conversation", "reply_purpose": "fact_recall",
            "goal_role": "result", "quoted": False, "dialogue_ids": [],
            "fact_fields": [], "critical_fact_fields": [],
        },
    }
    context.update(overrides)
    return context


def test_empty_and_failed_reads_do_not_reuse_old_user_words():
    from app.services.agent import react_readonly_workers
    from app.services.agent.execution_state import _request_memory

    token = _request_memory.set({
        "short_memory": [{
            "message_id": "old-user-message",
            "role": "user",
            "content": "请记录我今天实际做的哑铃划船：24公斤，3组，每组12次。",
        }],
        "long_memory": {},
    })
    try:
        empty = asyncio.run(react_readonly_workers.answer_training(
            {"query": READ_SOURCE},
            _worker_context(observations=[{"status": "completed", "result": {"workouts": []}}])))
        assert empty.status == "completed"
        assert empty.result["authoritative"] is True
        assert "没有找到" in empty.summary
        assert "24" not in empty.summary

        failed = asyncio.run(react_readonly_workers.answer_training(
            {"query": READ_SOURCE},
            _worker_context(observations=[{
                "status": "unavailable", "error": "storage_unavailable",
                "_source": {"tool_name": "read_workout_history", "permission": "read",
                            "task_type": None},
            }])))
        assert failed.status == "unavailable"
        assert failed.error == "saved_record_read_failed"
        assert "24" not in failed.summary and "3组" not in failed.summary
        assert "旧对话" in failed.summary
    finally:
        _request_memory.reset(token)


def test_foreign_user_record_is_not_read_as_current_user(compact_db):
    from app.db import repositories
    from app.services.agent.react_adapters import read_workout_history

    repositories.create_workout("other-user", "哑铃划船", 3, 12, 24)
    observation = read_workout_history({"days": 1}, {"user_id": "demo-user"})
    assert observation["status"] == "completed"
    assert observation["result"]["workouts"] == []


def test_updated_or_deleted_record_is_not_reported_from_old_value(compact_db):
    from app.db import repositories
    from app.services.agent.react_adapters import read_workout_history

    record = repositories.create_workout("demo-user", "深蹲", 3, 10, 50)
    assert read_workout_history({"days": 1}, {"user_id": "demo-user"})["result"]["workouts"]
    repositories.update_workout("demo-user", record["id"], {
        **{key: value for key, value in record.items()
           if key not in {"id", "user_id", "set_details"}},
        "weight_kg": 55, "sets": 3, "reps": 10,
    })
    updated = read_workout_history({"days": 1}, {"user_id": "demo-user"})["result"]["workouts"]
    assert len(updated) == 1 and updated[0]["weight_kg"] == 55
    assert updated[0]["weight_kg"] != 50
    repositories.delete_workout("demo-user", record["id"])
    assert read_workout_history({"days": 1}, {"user_id": "demo-user"})["result"]["workouts"] == []


def test_active_fact_recall_uses_verified_fact_not_old_history(monkeypatch):
    from app.services.agent import react_readonly_workers
    from app.services.agent.agent import agent
    from app.services.agent.dialogue_context import RecallSelection
    from app.services.agent.execution_state import _request_memory

    memory = {
        "short_memory": [{
            "message_id": "old-user-message", "role": "user",
            "content": "请记录我今天实际做的哑铃划船：24公斤，3组，每组12次。",
        }],
        "long_memory": {"preferred_equipment": {
            "text": "我更喜欢用哑铃训练",
            "source_quote": "我更喜欢用哑铃训练",
            "source_message_id": "fact-source",
        }},
    }
    token = _request_memory.set(memory)
    seen = []

    async def structured(messages, schema, *, name=None, validator=None, **kwargs):
        seen.append(json.dumps(messages, ensure_ascii=False))
        value = schema.model_validate({"evidence": [{
            "source_id": "fact:preferred_equipment",
            "quote": "我更喜欢用哑铃训练",
        }]})
        if validator is not None:
            validator(value)
        return value

    monkeypatch.setattr(agent.graph.llm, "configured", lambda: True)
    monkeypatch.setattr(agent.graph.llm, "complete_structured", structured)
    try:
        context = _worker_context(
            source_text="我以前让你记住喜欢什么训练器械？",
            task_text="我以前让你记住喜欢什么训练器械？",
            semantic_scope={
                "reply_mode": "conversation", "reply_purpose": "fact_recall",
                "goal_role": "result", "quoted": False, "is_third_party": False,
                "context_ids": ["old-user-message"],
                "fact_fields": ["preferred_equipment"],
                "critical_fact_fields": [],
            },
            dialogue_scope={
                "reply_mode": "conversation", "reply_purpose": "fact_recall",
                "goal_role": "result", "quoted": False,
                "dialogue_ids": ["old-user-message"],
                "fact_fields": ["preferred_equipment"],
                "critical_fact_fields": [],
            },
        )
        result = asyncio.run(react_readonly_workers.answer_training(
            {"query": context["task_text"]}, context))
    finally:
        _request_memory.reset(token)
    assert result.status == "completed"
    assert "哑铃" in result.summary
    assert "24" not in result.summary
    assert "old-user-message" not in "".join(seen)


def test_nutrition_summary_fact_recall_uses_completed_read_not_rag(monkeypatch):
    from app.services.agent import react_readonly_workers

    _install_selection(monkeypatch, _select_ids("nutrition:summary:0"))

    async def no_rag(*args, **kwargs):
        raise AssertionError("nutrition summary recall must not request RAG")

    import app.services.rag.online as rag_online
    monkeypatch.setattr(rag_online, "search", no_rag, raising=False)
    context = _worker_context(
        source_text="请查我最近7天的营养摄入汇总。只查询，不保存。",
        task_text="请查我最近7天的营养摄入汇总。只查询，不保存。",
        observations=[{"status": "completed", "result": {
            "days": 7, "entries": 2, "calories": 1234.5, "protein_g": 78.2,
            "carbs_g": 140.3, "fat_g": 42.1, "fiber_g": 12.4,
        }}])
    result = asyncio.run(react_readonly_workers.advise_nutrition(
        {"query": context["task_text"]}, context))
    assert result.status == "completed"
    assert result.result["authoritative"] is True
    for value in ("1234.5", "78.2", "140.3", "42.1", "12.4"):
        assert value in result.summary


def test_grounded_advice_still_enters_the_evidence_gate(monkeypatch):
    from app.services.agent import react_readonly_workers

    calls = []

    async def empty_rag(*args, **kwargs):
        calls.append(True)
        return []

    import app.services.rag.online as rag_online
    monkeypatch.setattr(rag_online, "search", empty_rag, raising=False)
    context = _worker_context(
        source_text="成年人每周做几次力量训练比较合适？",
        task_text="成年人每周做几次力量训练比较合适？",
        semantic_scope={"reply_purpose": "general", "goal_role": "result",
                        "quoted": False, "is_third_party": False,
                        "context_ids": [], "fact_fields": []},
        dialogue_scope={},
    )
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert calls == [True]
    assert result.status == "rejected"
    assert result.error == "insufficient_evidence"


def test_conversation_requires_grounding_returns_to_existing_evidence_gate(monkeypatch):
    from app.services.agent import react_readonly_workers
    from app.services.agent.agent import agent

    structured_calls = []
    rag_calls = []

    async def structured(messages, schema, *, name=None, **kwargs):
        structured_calls.append(name)
        return schema.model_validate({
            "answer": "这需要新的训练知识。",
            "requires_grounding": True,
            "status_kind": None,
        })

    async def empty_rag(*args, **kwargs):
        rag_calls.append(True)
        return []

    import app.services.rag.online as rag_online
    monkeypatch.setattr(rag_online, "search", empty_rag, raising=False)
    monkeypatch.setattr(agent.graph.llm, "configured", lambda: True)
    monkeypatch.setattr(agent.graph.llm, "complete_structured", structured)
    context = _worker_context(
        source_text="接下来该怎么练？",
        task_text="接下来该怎么练？",
        semantic_scope={"reply_mode": "conversation", "reply_purpose": "general",
                        "goal_role": "result", "quoted": False,
                        "is_third_party": False, "fact_fields": []},
        dialogue_scope={"reply_mode": "conversation", "reply_purpose": "general",
                        "goal_role": "result", "quoted": False,
                        "dialogue_ids": [], "fact_fields": []},
    )
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert structured_calls == ["return_dialogue_answer"]
    assert rag_calls == [True]
    assert result.status == "rejected"
    assert result.error == "insufficient_evidence"


def test_read_failure_notice_is_not_a_write_permission_claim():
    from app.services.agent.react_graph import _notice_for_state

    notice = _notice_for_state(
        {"observations": [{"status": "unavailable",
                           "error": "saved_record_query_unavailable"}]},
        "blocked", "")
    assert "读取或选择已保存记录暂时失败" in notice
    assert "写权限" not in notice


def _record(record_id, weight, name="哑铃划船"):
    return {"id": record_id, "exercise": name, "weight_kg": weight,
            "sets": 3, "reps": 12, "date": "2026-10-29"}


def test_read_observation_carries_application_source_identity():
    from app.services.agent.react_graph import ReactDependencies, execute_react_batch
    from tests.react_phase_a_support import (
        RecordingExecutor, assistant, batch_state, tool_call,
    )

    state = batch_state(
        READ_SOURCE,
        assistant(tool_calls=[tool_call("read-identity", "read_workout_history",
                                        {"days": 1})]),
        candidates=[{"id": "goal-read", "task_type": "training_qa",
                     "action": "question", "text": READ_SOURCE,
                     "source_start": 0, "source_end": len(READ_SOURCE),
                     "side_effect": False}],
    )
    result = asyncio.run(execute_react_batch(state, ReactDependencies(
        llm=object(),
        tool_executor=RecordingExecutor({
            "read_workout_history": {
                "status": "completed",
                "result": {"workouts": [_record(7, 24)]},
            },
        }),
        user_id="synthetic-owner", conversation_id="synthetic-conversation",
        request_id="synthetic-request",
    )))
    observation = result["observations"][0]
    assert observation["_source"] == {
        "tool_name": "read_workout_history",
        "permission": "read",
        "task_type": None,
    }


def test_unselected_profile_is_not_mixed_into_workout_fact_answer(monkeypatch):
    from app.services.agent import react_readonly_workers

    _install_selection(monkeypatch, _select_ids("workout:7:0"))
    context = _worker_context(
        observations=[
            {"status": "completed", "result": {"goal": "减脂", "body_weight_kg": 77.777}},
            {"status": "completed", "result": {"workouts": [_record(7, 24)]}},
        ])
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert result.status == "completed"
    assert "24" in result.summary and "哑铃划船" in result.summary
    assert "77.777" not in result.summary
    assert "个人档案" not in result.summary


def test_profile_alone_does_not_complete_workout_fact_recall():
    from app.services.agent import react_readonly_workers

    context = _worker_context(observations=[
        {"status": "completed", "result": {"goal": "减脂", "body_weight_kg": 77.777}},
    ])
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert result.status != "completed"
    assert "77.777" not in result.summary


def test_latest_failed_read_does_not_turn_old_value_into_current_fact():
    from app.services.agent import react_readonly_workers

    source = {"tool_name": "read_workout_history", "permission": "read",
              "task_type": None}
    context = _worker_context(observations=[
        {"status": "completed", "result": {"workouts": [_record(7, 24)]},
         "_source": source},
        {"status": "unavailable", "error": "synthetic_current_workout_read_failed",
         "result": {"workouts": []}, "_source": source},
    ])
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert result.status != "completed"
    assert result.error == "saved_record_read_failed"
    assert "24" not in result.summary


def test_unrelated_source_failure_does_not_erase_completed_workout(monkeypatch):
    from app.services.agent import react_readonly_workers

    _install_selection(monkeypatch, _select_ids("workout:7:0"))
    context = _worker_context(
        observations=[
            {"status": "completed", "result": {"workouts": [_record(7, 24)]},
             "_source": {"tool_name": "read_workout_history", "permission": "read",
                         "task_type": None}},
            {"status": "unavailable", "error": "nutrition_read_failed",
             "result": {"workouts": []},
             "_source": {"tool_name": "read_nutrition_summary", "permission": "read",
                         "task_type": None}},
        ],
        semantic_scope={
            "reply_mode": "conversation", "reply_purpose": "fact_recall",
            "source_task_type": "workout_record", "goal_role": "result",
            "quoted": False, "is_third_party": False,
            "context_ids": [], "fact_fields": [], "critical_fact_fields": [],
        })
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert result.status == "completed"
    assert "24" in result.summary and "哑铃划船" in result.summary


def test_later_workout_read_replaces_earlier_and_empty_clears_it(monkeypatch):
    from app.services.agent import react_readonly_workers

    _install_selection(monkeypatch, _select_ids("workout:7:0"))
    later = _worker_context(observations=[
        {"status": "completed", "result": {"workouts": [_record(7, 24)]}},
        {"status": "completed", "result": {"workouts": [_record(7, 30)]}},
    ])
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": later["task_text"]}, later))
    assert result.status == "completed"
    assert "30" in result.summary and "24" not in result.summary

    cleared = _worker_context(observations=[
        {"status": "completed", "result": {"workouts": [_record(7, 24)]}},
        {"status": "completed", "result": {"workouts": []}},
    ])
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": cleared["task_text"]}, cleared))
    assert result.status == "completed"
    assert "24" not in result.summary and "没有找到" in result.summary


def test_source_task_type_excludes_unselected_nutrition_read(monkeypatch):
    from app.services.agent import react_readonly_workers

    _install_selection(monkeypatch, _select_ids("workout:7:0"))
    context = _worker_context(
        observations=[
            {"status": "completed", "result": {
                "days": 7, "entries": 2, "calories": 1234.5, "protein_g": 78.2,
                "carbs_g": 140.3, "fat_g": 42.1, "fiber_g": 12.4,
            }},
            {"status": "completed", "result": {"workouts": [_record(7, 24)]}},
        ],
        semantic_scope={
            "reply_mode": "conversation", "reply_purpose": "fact_recall",
            "source_task_type": "workout_record", "goal_role": "result",
            "quoted": False, "is_third_party": False,
            "context_ids": [], "fact_fields": [], "critical_fact_fields": [],
        })
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert result.status == "completed"
    assert "24" in result.summary and "哑铃划船" in result.summary
    assert "1234.5" not in result.summary and "热量" not in result.summary


def test_structured_selection_narrows_multiple_actions(monkeypatch):
    from app.services.agent import react_readonly_workers

    _install_selection(monkeypatch, _select_ids("workout:7:0"))
    context = _worker_context(observations=[{
        "status": "completed",
        "result": {"workouts": [_record(7, 24), _record(8, 30, "卧推")]},
    }])
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert result.status == "completed"
    assert "哑铃划船" in result.summary and "24" in result.summary
    assert "卧推" not in result.summary and "30" not in result.summary


def test_negative_mention_is_not_a_record_selector(monkeypatch):
    from app.services.agent import react_readonly_workers

    _install_selection(monkeypatch, _select_ids("workout:7:0"))
    context = _worker_context(
        source_text="只查哑铃划船，不要列出卧推。",
        task_text="只查哑铃划船，不要列出卧推。",
        observations=[{
            "status": "completed",
            "result": {"workouts": [_record(7, 24), _record(8, 90, "卧推")]},
        }])
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert result.status == "completed"
    assert "24" in result.summary
    assert "90" not in result.summary and "卧推" not in result.summary


def test_alias_is_resolved_by_structured_source_selection(monkeypatch):
    from app.services.agent import react_readonly_workers

    _install_selection(monkeypatch, _select_ids("workout:7:0"))
    context = _worker_context(
        source_text="只查划船类动作。",
        task_text="只查划船类动作。",
        observations=[{
            "status": "completed",
            "result": {"workouts": [_record(7, 24)]},
        }])
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert result.status == "completed"
    assert "哑铃划船" in result.summary and "24" in result.summary


def test_generic_list_uses_structured_multi_source_selection(monkeypatch):
    from app.services.agent import react_readonly_workers

    _install_selection(monkeypatch, _select_ids("workout:7:0", "workout:8:1"))
    context = _worker_context(
        source_text="列出今天所有训练。",
        task_text="列出今天所有训练。",
        observations=[{
            "status": "completed",
            "result": {"workouts": [_record(7, 24), _record(8, 30, "卧推")]},
        }])
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert result.status == "completed"
    assert "哑铃划船" in result.summary and "24" in result.summary
    assert "卧推" in result.summary and "30" in result.summary


def test_empty_source_selection_is_a_real_no_match(monkeypatch):
    from app.services.agent import react_readonly_workers

    _install_selection(monkeypatch, _select_ids())
    context = _worker_context(observations=[{
        "status": "completed", "result": {"workouts": [_record(7, 24)]},
    }])
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert result.status == "completed"
    assert "没有找到" in result.summary and "24" not in result.summary


def test_forged_source_id_and_quote_are_rejected(monkeypatch):
    from app.services.agent import react_readonly_workers

    context = _worker_context(observations=[{
        "status": "completed", "result": {"workouts": [_record(7, 24)]},
    }])
    _install_selection(monkeypatch, lambda sources: [{
        "source_id": "workout:999:0", "quote": "伪造来源",
    }])
    forged_id = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert forged_id.status == "unavailable"
    assert forged_id.error == "saved_record_selection_unavailable"
    assert "24" not in forged_id.summary

    _install_selection(monkeypatch, lambda sources: [{
        "source_id": "workout:7:0", "quote": "不存在的原文",
    }])
    forged_quote = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert forged_quote.status == "unavailable"
    assert forged_quote.error == "saved_record_selection_unavailable"
    assert "24" not in forged_quote.summary


def test_grounded_third_party_question_still_reaches_knowledge_gate(monkeypatch):
    from app.services.agent import react_readonly_workers

    rag_calls = []

    async def empty_rag(*args, **kwargs):
        rag_calls.append(True)
        return []

    import app.services.rag.online as rag_online
    monkeypatch.setattr(rag_online, "search", empty_rag, raising=False)
    context = _worker_context(
        source_text="同事想了解一般力量训练如何安排恢复，请讲通用原则。",
        task_text="同事想了解一般力量训练如何安排恢复，请讲通用原则。",
        semantic_scope={"reply_purpose": "general", "reply_mode": "grounded",
                        "goal_role": "result", "quoted": False,
                        "is_third_party": True, "fact_fields": []},
        dialogue_scope={"reply_purpose": "general", "reply_mode": "grounded",
                        "goal_role": "result", "quoted": False,
                        "fact_fields": []},
    )
    result = asyncio.run(react_readonly_workers.answer_training(
        {"query": context["task_text"]}, context))
    assert rag_calls == [True]
    assert result.status == "rejected"
    assert result.error == "insufficient_evidence"


def test_authoritative_current_observation_is_not_overridden_by_stale_failure_tail():
    from app.services.agent.react_graph import (
        answer_from_completed_read_goals,
        completed_read_goals,
    )

    candidates = [{"id": "goal-read", "task_type": "training_qa", "action": "question",
                   "text": READ_SOURCE, "goal_role": "result", "side_effect": False}]
    goal_results = {"goal-read": {
        "status": "completed", "summary": "2026-10-28 哑铃划船：24 kg，3 组，每组 12 次",
        "result": {"authoritative": True},
    }}
    goals = completed_read_goals(candidates, goal_results, ["goal-read"])
    assert goals[0]["authoritative"] is True
    answer = answer_from_completed_read_goals(
        goals, "工具返回证据不足，未能产出内容；总负荷约2125公斤。")
    assert answer == "2026-10-28 哑铃划船：24 kg，3 组，每组 12 次"
    assert "证据不足" not in answer and "2125" not in answer

    # A normal single read goal keeps the existing final-assistant contract.
    goal_results["goal-read"]["result"] = {}
    goals = completed_read_goals(candidates, goal_results, ["goal-read"])
    assert answer_from_completed_read_goals(goals, "普通最终答复") == "普通最终答复"


def test_mixed_fact_and_grounded_goal_keeps_request_order():
    from app.services.agent.react_graph import (
        answer_from_completed_read_goals,
        completed_read_goals,
    )

    candidates = [
        {"id": "fact", "task_type": "training_qa", "action": "question",
         "text": "查今天训练", "goal_role": "result", "side_effect": False},
        {"id": "advice", "task_type": "training_qa", "action": "question",
         "text": "再给建议", "goal_role": "result", "side_effect": False},
    ]
    goal_results = {
        "fact": {"status": "completed", "summary": "事实答案",
                 "result": {"authoritative": True}},
        "advice": {"status": "completed", "summary": "建议答案", "result": {}},
    }
    goals = completed_read_goals(candidates, goal_results, ["fact", "advice"])
    assert answer_from_completed_read_goals(goals, "主管失败尾注") == "事实答案\n\n建议答案"
