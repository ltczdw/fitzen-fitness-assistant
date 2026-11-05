import asyncio
import json
from copy import deepcopy
from datetime import date, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import select

from app.core import clock
from app.db.memory_models import MemoryTurn
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agent import planning_questions as lifecycle
from app.services.agent.task_information import (
    PlanInformation, attach_record_clarification, check_plan_information,
    food_fields, pending_result, question, user_sources,
)


def info(fields=(), scope="session", scope_quote=""):
    return PlanInformation(scope=scope, scope_quote=scope_quote, facts=[
        {"field": name, "source_id": source, "quote": value} for name, source, value in fields])


@pytest.mark.parametrize("record", ["记录一个鸡蛋和一碗饭", "记录哑铃卧推10kg4组10次"])
def test_both_record_types_require_same_personal_information_before_retrieval(monkeypatch, record):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    graph = FitnessMultiAgentGraph()
    monkeypatch.setattr(graph.llm, "complete_structured", AsyncMock(return_value=info()))
    retrieval = AsyncMock(side_effect=AssertionError("missing personal input must precede retrieval"))
    answer = AsyncMock(side_effect=AssertionError("missing input must precede answer"))
    monkeypatch.setattr(graph, "_readonly_retrieval", retrieval)
    monkeypatch.setattr(graph, "_complete_or_fallback_with_mcp", answer)
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke({"user_id": "demo-user", "raw_message": record + "，再安排明天训练",
        "task_text": "安排明天训练", "task_id": "plan", "task_context": [{"status": "complete", "answer": "已记录"}]}))["result"]
    assert result["status"] == "needs_confirmation"
    assert [field["name"] for field in result["clarification"]["fields"]] == ["goal", "experience"]
    assert "每周" not in result["answer"]
    retrieval.assert_not_awaited()
    answer.assert_not_awaited()


def test_known_fields_are_not_asked_again_and_assistant_is_not_a_fact():
    state = {"raw_message": "明天只有30分钟", "profile": {"goal": "增肌", "experience": "训练一年"},
             "short_memory": [{"role": "assistant", "content": "假设你有哑铃"}]}
    assert "假设你有哑铃" not in user_sources(state).values()
    llm = Mock(complete_structured=AsyncMock(return_value=info([
        ("goal", "profile:goal", "增肌"), ("experience", "profile:experience", "训练一年"),
        ("session_time", "current", "30分钟"),
    ])))
    details, missing = asyncio.run(check_plan_information(llm, state))
    assert missing is None and details["known"]["equipment"]=="健身房"
    assert details["defaulted_fields"]==["equipment"] and details["timing_requested"] is True


def test_profile_defaults_are_not_silently_treated_as_confirmed():
    sources = user_sources({"profile": {"goal": "保持健康", "experience": "初学者", "available_days": 3}})
    assert all(not key.startswith("profile:") for key in sources)


@pytest.mark.parametrize("scope, quote, required", [("session", "", ["goal", "experience"]),
    ("weekly", "一周训练计划", ["goal", "experience", "available_days"]),
    ("generic", "通用示例", [])])
def test_requirements_depend_on_requested_scope(scope, quote, required):
    llm = Mock(complete_structured=AsyncMock(return_value=info(scope=scope, scope_quote=quote)))
    _, missing = asyncio.run(check_plan_information(llm, {"raw_message": quote or "安排明天训练"}))
    assert ([field.name for field in missing.fields] if missing else []) == required


def test_unbound_model_fact_and_unavailable_model_are_not_reported_as_user_omissions():
    llm = Mock(complete_structured=AsyncMock(return_value=info([("equipment", "current", "健身房")])) )
    _, missing = asyncio.run(check_plan_information(llm, {"raw_message": "安排明天训练"}))
    assert missing.category == "service_unavailable" and missing.fields == []
    llm.complete_structured.side_effect = RuntimeError("offline")
    _, missing = asyncio.run(check_plan_information(llm, {"raw_message": "安排明天训练"}))
    assert missing.category == "service_unavailable" and missing.fields == []


def test_evidence_failure_does_not_ask_for_personal_fields(monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    graph = FitnessMultiAgentGraph()
    text = "减脂，初学者，徒手，30分钟"
    values = [(name, "current", value) for name, value in zip(("goal", "experience", "equipment", "session_time"), text.split("，"))]
    monkeypatch.setattr(graph.llm, "complete_structured", AsyncMock(return_value=info(values)))
    monkeypatch.setattr(graph, "_readonly_retrieval", AsyncMock(return_value=[]))
    answer = AsyncMock()
    monkeypatch.setattr(graph, "_complete_or_fallback_with_mcp", answer)
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke({"raw_message": text, "task_text": "安排明天训练"}))["result"]
    assert result["status"] == "insufficient_evidence"
    assert "补充训练目标" not in result["answer"]
    assert "clarification" not in result
    answer.assert_not_awaited()


def test_record_fields_are_derived_from_real_validator_errors():
    result = attach_record_clarification({"status": "needs_confirmation", "task_type": "workout_record", "task_id": "w",
        "missing_fields": ["sets", "reps"], "answer": "还缺组数和每组次数。"})
    assert [field["name"] for field in result["clarification"]["fields"]] == ["sets", "reps"]
    fields = food_fields([{"name": "米饭", "reason": "weight_not_found"}])
    assert [(field.name, field.entity) for field in fields] == [("food_amount", "米饭")]
    assert food_fields([{"reason": "unknown_service_failure"}]) == []


def test_json_and_sse_share_structured_missing_fields(monkeypatch):
    from app.services.agent import transport
    value = question("training_plan", ["equipment"], task_id="plan")
    response = ChatResponse(answer=value.question, intent="clarification", clarifications=[value])
    monkeypatch.setattr(transport, "execute_request", AsyncMock(return_value=response))
    async def run():
        return [event async for event in transport.stream_response(object(), ChatRequest(message="安排训练"))]
    events = asyncio.run(run())
    for kind in ("meta", "done"):
        event = next(item for item in events if item["event"] == kind)
        assert event["data"]["clarifications"] == response.model_dump()["clarifications"]


def begin(request):
    tid, _ = __import__("app.services.memory.compact", fromlist=["begin_turn"]).begin_turn(request)
    return tid


def complete(factory, tid):
    with factory() as s, s.begin():
        row = s.get(MemoryTurn, tid)
        row.status = "completed"
        row.completed_at = clock.utc_now()


def draft_fixture(factory):
    from app.services.memory import compact
    request = ChatRequest(message="记录午餐鸡蛋，再安排明天训练", conversation_id="planning")
    tid = begin(request)
    with factory() as s, s.begin():
        row = s.get(MemoryTurn, tid)
        row.operations = {"version": 1, "operations": {"saved": {"status": "committed"}}, "tasks": {}}
    tasks = [{"id": "saved", "task_type": "nutrition_record", "action": "record", "text": "记录午餐鸡蛋", "side_effect": True},
        {"id": "plan", "task_type": "training_plan", "action": "question", "text": "安排明天训练", "depends_on": ["saved"]}]
    clarification = question("training_plan", ["goal", "experience"], task_id="plan")
    result = {"worker_results": [{"task_id": "saved", "task_type": "nutrition_record", "status": "complete"},
        {"task_id": "plan", "task_type": "training_plan", **pending_result(clarification)}]}
    token = compact.current_turn.set(tid)
    try:
        asyncio.run(lifecycle.remember(request, result, tasks))
    finally:
        compact.current_turn.reset(token)
    complete(factory, tid)
    return request, tid


@pytest.mark.parametrize("action", ["update", "cancel"])
def test_planning_reply_never_replays_completed_writes(compact_db, action):
    from app.services.memory import compact
    original, tid = draft_fixture(compact_db)
    reply = ChatRequest(message="徒手，30分钟" if action == "update" else "取消计划", conversation_id=original.conversation_id)
    current = begin(reply)
    def decide(messages, *args, **kwargs):
        payload = json.loads(messages[-1]["content"])
        return lifecycle.PlanningReply(draft_id=payload["draft_id"], action=action, quote=reply.message)
    llm = Mock(complete_structured=AsyncMock(side_effect=decide))
    token = compact.current_turn.set(current)
    try:
        result = asyncio.run(lifecycle.continue_request(llm, reply))
        if action == "update":
            assert [task["task_type"] for task in result["parsed_input"]["tasks"]] == ["training_plan"]
            assert all(task["action"] == "question" for task in result["parsed_input"]["tasks"])
            assert "记录午餐" not in result["source"]
            replay = asyncio.run(lifecycle.continue_request(llm, reply))
            assert replay == result
            llm.complete_structured.assert_awaited_once()
        else:
            assert "取消" in result["answer"]
        with compact_db() as s:
            assert s.get(MemoryTurn, tid).operations["operations"] == {"saved": {"status": "committed"}}
    finally:
        compact.current_turn.reset(token)


@pytest.mark.parametrize("change", ["conversation", "expired", "hidden", "deleted", "epoch"])
def test_planning_question_is_bound_to_visible_live_conversation(compact_db, change):
    original, tid = draft_fixture(compact_db)
    reply = ChatRequest(message="徒手30分钟", conversation_id="other" if change == "conversation" else original.conversation_id)
    current = begin(reply)
    with compact_db() as s, s.begin():
        row = s.get(MemoryTurn, tid)
        if change == "expired":
            state = deepcopy(row.operations)
            state["tasks"][lifecycle.KEY]["created_at"] = (clock.utc_now() - timedelta(hours=1)).isoformat()
            row.operations = state
        elif change == "hidden": row.hidden_at = clock.utc_now()
        elif change == "deleted": row.deleted_at = clock.utc_now()
        elif change == "epoch": row.memory_epoch -= 1
    assert lifecycle._load(reply) is None


def test_plan_supplement_keeps_original_relative_date_across_midnight(compact_db, monkeypatch):
    monkeypatch.setattr(clock, "today", lambda: date(2026, 10, 19))
    original, _ = draft_fixture(compact_db)
    monkeypatch.setattr(clock, "today", lambda: date(2026, 10, 20))
    reply = ChatRequest(message="徒手30分钟", conversation_id=original.conversation_id)
    begin(reply)
    draft = lifecycle._load(reply)
    result = lifecycle._transition(reply, draft, lifecycle.PlanningReply(
        draft_id=draft["id"], action="update", quote=reply.message))
    assert "原计划请求日期：2026-10-19" in result["source"]
    assert "2026-10-20" not in result["source"]


@pytest.mark.parametrize("action", ["update", "cancel"])
def test_http_followup_only_resumes_plan_and_preserves_real_saved_record(compact_db, monkeypatch, action):
    import httpx
    from sqlalchemy import text
    from app.main import app
    from app.api import chat
    from app.services.agent.execution_state import PlannerDecision
    from app.services.agent.intent import IntentResult
    from app.services.rag.retrieval import RetrievedChunk
    from test_retry_operations import RetryScenario

    scenario = RetryScenario(monkeypatch, compact_db, workout_records=lambda message: [
        {"exercise": "卧推", "weight_kg": 40, "sets": 3, "reps": 8}])
    agent = scenario.agent
    monkeypatch.setattr(chat, "agent", agent)
    initial = ChatRequest(message="记录卧推40kg3组8次，再安排明天训练", conversation_id="http-plan")
    supplement = "减脂，初学者，徒手，30分钟"
    reply = ChatRequest(message=supplement if action == "update" else "取消计划", conversation_id=initial.conversation_id)
    first_specs = [
        {"id": "saved", "task_type": "workout_record", "worker": "workout_record", "action": "record", "text": "记录卧推40kg3组8次",
         "side_effect": True, "depends_on": [], "payload": {"write_authorization": "authorized"}},
        {"id": "plan", "task_type": "training_plan", "worker": "plan_adjust", "action": "question", "text": "安排明天训练", "depends_on": ["saved"]},
    ]
    async def route(request, memory, parsed_input=None):
        if request.request_id == initial.request_id:
            specs = deepcopy(first_specs)
        else:
            assert parsed_input and all(task["task_type"] == "training_plan" for task in parsed_input["tasks"])
            specs = [{**task, "id": "continued-plan", "worker": "plan_adjust", "depends_on": [], "side_effect": False} for task in parsed_input["tasks"]]
        return PlannerDecision(IntentResult("plan_adjust", 1.0), {}, tuple(specs))
    monkeypatch.setattr(agent, "_route", route)
    def structured(messages, schema, **kwargs):
        payload = json.loads(messages[-1]["content"])
        if kwargs.get("name") == "return_request_understanding":
            source = str(payload.get("query") or "")
            return schema.model_validate({
                "tasks": [{"task_type": "training_qa", "action": "question", "text": source,
                           "is_negated": False, "is_hypothetical": False,
                           "is_third_party": False}],
                "safety": {"level": "normal", "evidence": "", "categories": [],
                           "reason": "scripted normal"},
            })
        if schema is lifecycle.PlanningReply:
            return lifecycle.PlanningReply(draft_id=payload["draft_id"], action=action, quote=reply.message)
        if supplement not in payload["sources"]["current"]:
            return info()
        return info([(name, "current", value) for name, value in zip(("goal", "experience", "equipment", "session_time"), supplement.split("，"))])
    monkeypatch.setattr(agent.graph.llm, "complete_structured", AsyncMock(side_effect=structured))
    hit = RetrievedChunk("训练安排原则", "每周逐步增加训练量，并为肌群保留恢复。", .9, "official", {"rerank_score": .9, "rerank_status": "model"}, "fixture-1")
    monkeypatch.setattr(agent.graph, "_readonly_retrieval", AsyncMock(return_value=[hit]))
    timed_answer = json.dumps({"sessions": [{
        "label": "本次训练", "warmup": {"seconds": 120, "activity": "轻缓热身"},
        "main": {"seconds": 1080, "exercises": [{"name": "徒手深蹲", "sets": 2, "reps": 8}, {"name": "俯卧撑", "sets": 2, "reps": 8}]},
        "breaks": {"seconds": 480},
        "cooldown": {"seconds": 120, "activity": "轻缓活动放松"},
        "time_consistency": "consistent"}]}, ensure_ascii=False)
    monkeypatch.setattr(agent.graph.llm, "complete_message", AsyncMock(return_value={"tool_calls": [{"type": "function", "function": {"name": "return_timed_training_plan", "arguments": timed_answer}}]}))
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/chat", json=initial.model_dump())
            assert response.status_code == 200
            first = response.json()
            assert "已记录" in first["answer"]
            assert first["clarifications"][0]["task_id"] == "plan"
            assert len(scenario.saves) == 1
            streamed = await client.post("/api/chat/stream", json=reply.model_dump())
            assert streamed.status_code == 200
            frames = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ")]
            answer = "".join(frame.get("content", "") for frame in frames)
            expected = "已根据确认" if action == "update" else "已取消"
            assert expected in answer
            assert frames[-1].get("clarifications") == []
            calls = agent.graph.llm.complete_structured.await_count
            replay = await client.post("/api/chat", json=reply.model_dump())
            assert replay.status_code == 200 and expected in replay.json()["answer"]
            assert agent.graph.llm.complete_structured.await_count == calls
            assert len(scenario.saves) == 1
        with compact_db() as session:
            assert session.execute(text("SELECT COUNT(*) FROM workout_records")).scalar() == 1
    asyncio.run(run())


def test_readonly_plan_cannot_use_model_tool_arguments_as_personal_facts(monkeypatch):
    from app.services.agent.agent import agent
    from app.services.agent import react_readonly_workers, react_runtime
    def check(messages, *args, **kwargs):
        payload = json.loads(messages[-1]["content"])
        assert payload["sources"]["current"] == "安排明天训练"
        assert "哑铃" not in payload["sources"]["current"]
        return info()
    monkeypatch.setattr(agent.graph.llm, "complete_structured", AsyncMock(side_effect=check))
    result = asyncio.run(react_readonly_workers.plan_training(
        {"query": "用户目标减脂，初学者，有哑铃，30分钟"}, {"source_text": "安排明天训练"})).to_dict()
    assert result["status"] == "needs_clarification"
    assert result["missing_fields"] == ["goal", "experience"]
    mapped = react_runtime._map_readonly_result({"final_answer": "模型错误地声称已生成计划", "observations": [result]})
    assert mapped["final_answer"] == result["summary"]
    assert mapped["clarifications"][0]["fields"]


def test_food_clarification_metadata_is_safe_for_later_task_context():
    item = attach_record_clarification({"status": "needs_confirmation", "task_type": "nutrition_record", "answer": "米饭还缺份量。",
                                       "missing_fields": food_fields([{"name": "米饭", "reason": "weight_not_found"}])})
    encoded = json.dumps(item, ensure_ascii=False)
    assert "米饭" in encoded and item["missing_fields"] == ["food_amount"]
