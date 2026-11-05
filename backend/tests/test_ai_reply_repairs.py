"""AI-SIM-01..06: real final transactions and read-only source boundaries."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from app.core import clock
from app.db.memory_models import MemoryTurn, UserFact
from app.schemas.chat import ChatRequest
from app.services.agent import semantics
from app.services.agent.dialogue_context import conversation_answer
from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
from app.services.memory import compact
from evals.v2.safety import deny_network


@pytest.fixture(autouse=True)
def isolated_semantics():
    token = semantics.request_understanding.set({})
    with deny_network():
        yield
    semantics.request_understanding.reset(token)


def outline(source, *, purpose="general", kind="training_qa", action="question", **extra):
    return {"tasks": [{"task_type": kind, "action": action, "text": source,
                       "is_negated": False, "is_hypothetical": False, "is_third_party": False}],
            "reply_purpose": purpose,
            # Provider-bound structured safety result; memory commits and the
            # live entry require a real source-bound judgment, never a default.
            "safety": {"level": "normal", "evidence": "",
                       "categories": [], "reason": "scripted normal"},
            **extra}


def memory_outline(source):
    return outline(source, purpose="memory_update", kind="safety_triage", action="answer",
        memory_requests=[{"action": "remember", "quote": "我对花生过敏", "key": "allergy_peanut", "confidence": .99}])


@pytest.mark.parametrize("purpose,kind", [("execution_status", "nutrition_record"), ("fact_recall", "safety_triage")])
def test_read_purpose_does_not_require_new_health_evidence(purpose, kind):
    source = "我对哪种食物过敏？" if purpose == "fact_recall" else "不要保存，只回答是否会保存。"
    raw = outline(source, purpose=purpose, kind=kind)
    # Current schema carries the read purpose on the deliverable task as well
    # as the envelope. This fixture predates that per-task projection.
    raw["tasks"][0].update(goal_role="result", quoted=False,
                           history_requirement="none",
                           reply_mode="conversation", reply_purpose=purpose)
    value = semantics.validate_understanding(source, raw)
    assert value["reply_mode"] == "conversation"
    assert not any(t["side_effect"] for t in value["validated_tasks"])
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(return_value={"answer": "不会保存。", "requires_grounding": False}))
    result = asyncio.run(conversation_answer(llm, ChatRequest(message=source), {}, value))
    assert result["answer"]
    assert result["model_used"] is (purpose != "fact_recall")


@pytest.mark.parametrize("purpose", ["execution_status", "fact_recall", "memory_update"])
def test_read_purpose_cannot_hide_write(purpose):
    source = "帮我记录卧推60kg4组8次。"
    with pytest.raises(semantics.SemanticValidationError):
        semantics.validate_understanding(source, outline(source, purpose=purpose, kind="workout_record", action="record"))


def test_health_question_still_requires_grounding():
    source = "肩膀疼痛能继续训练吗？"
    value = semantics.validate_understanding(source, outline(source, kind="safety_triage", reply_mode="conversation"))
    assert value["reply_mode"] == "grounded"


def test_fact_identity_does_not_become_a_history_message_or_write_source():
    source = "我对哪种食物过敏？"
    facts = {"semantic:own:allergy": {"text": "我对花生过敏", "source_message_id": "own-source:user"}}
    raw = outline(source, purpose="fact_recall", references=[{"source_text": source, "query": "用户已存过敏事实", "evidence_ids": list(facts)}])
    value = semantics.validate_understanding(source, raw, context={"long_memory": facts})
    ref = value["references"][0]
    assert ref["evidence_ids"] == [] and ref["fact_fields"] == list(facts)
    assert not value["validated_tasks"][0]["side_effect"]
    raw["references"][0]["fact_fields"] = ["foreign-fact"]
    with pytest.raises(semantics.SemanticValidationError):
        semantics.validate_understanding(source, raw, context={"long_memory": facts})


def test_self_contained_question_keeps_source_and_rejects_unbound_rewrite():
    source = "我昨天的体重是多少？"
    raw = outline(source, purpose="fact_recall", references=[{"source_text": source, "query": "用户昨日体重", "evidence_ids": []}])
    with pytest.raises(semantics.SemanticValidationError) as error:
        semantics.validate_understanding(source, raw)
    assert "references=[]" in error.value.validation_errors[0]["hint"]
    raw["references"] = []
    assert semantics.validate_understanding(source, raw)["reply_mode"] == "conversation"


def commit_explicit():
    source = "请长期记住：我对花生过敏。"
    value = semantics.validate_understanding(source, memory_outline(source))
    request = ChatRequest(message=source, conversation_id="repair-memory")
    tid, _ = compact.begin_turn(request)
    response = {"answer": "已收到你的记忆请求。", "intent": "conversation", "metrics": {"memory_update_requested": True}}
    return tid, source, value, response


def test_memory_ack_matches_committed_fact_history_and_replay(compact_db):
    tid, source, value, response = commit_explicit()
    result = compact.finish_turn(tid, response["answer"], response, semantic_values=[(source, value)])
    assert result["answer"] == "已记住：我对花生过敏。"
    assert result["metrics"]["memory_update_status"] == "committed"
    with compact_db() as session:
        row = session.get(MemoryTurn, tid)
        assert row.response == result and row.messages[-1]["content"] == result["answer"]
        assert len(list(session.scalars(select(UserFact)))) == 1
    _, cached = compact.begin_turn(ChatRequest(message=source, conversation_id="repair-memory", request_id=row.request_id))
    assert cached["answer"] == result["answer"]


def test_memory_only_reply_cannot_swallow_an_independent_plan():
    source = "请长期记住：我对花生过敏。再帮我安排一周训练计划。"
    raw = memory_outline(source)
    raw["tasks"] = [outline("请长期记住：我对花生过敏。", kind="safety_triage", action="answer")["tasks"][0],
                    outline("再帮我安排一周训练计划。", kind="training_plan")["tasks"][0]]
    with pytest.raises(semantics.SemanticValidationError) as error:
        semantics.validate_understanding(source, raw)
    assert any(item["type"] == "semantic_memory_only_mixed_tasks" for item in error.value.validation_errors)


def test_committed_memory_ack_preserves_other_task_answer(compact_db):
    tid, source, value, response = commit_explicit()
    response = {"answer": "请补充本次训练时长。", "intent": "clarification", "metrics": {}}
    result = compact.finish_turn(tid, response["answer"], response, semantic_values=[(source, value)])
    assert result["answer"] == "请补充本次训练时长。\n\n已记住：我对花生过敏。"
    with compact_db() as session:
        row = session.get(MemoryTurn, tid)
        assert row.messages[-1]["content"] == result["answer"] == row.response["answer"]


def test_final_commit_failure_cannot_return_success_or_leave_fact(compact_db, monkeypatch):
    from app.services.memory import wiki
    tid, source, value, response = commit_explicit()
    monkeypatch.setattr(wiki, "materialize", lambda *_: (_ for _ in ()).throw(RuntimeError("injected final failure")))
    with pytest.raises(RuntimeError):
        compact.finish_turn(tid, response["answer"], response, semantic_values=[(source, value)])
    with compact_db() as session:
        row = session.get(MemoryTurn, tid)
        assert row.status != "completed" and row.response is None
        assert list(session.scalars(select(UserFact))) == []
        assert not any(m.get("role") == "assistant" for m in row.messages)


@pytest.mark.parametrize("mutation", ["hidden", "deleted", "epoch"])
def test_deleted_or_stale_source_cannot_be_recalled(compact_db, mutation):
    tid, source, value, response = commit_explicit()
    compact.finish_turn(tid, response["answer"], response, semantic_values=[(source, value)])
    assert compact.current_facts()
    with compact_db() as session, session.begin():
        row = session.get(MemoryTurn, tid)
        if mutation == "epoch": row.memory_epoch += 1
        elif mutation == "deleted": row.deleted_at = clock.utc_now()
        else: row.hidden_at = clock.utc_now()
    assert compact.current_facts() == {}


@pytest.mark.parametrize("transport", ["json", "sse"])
def test_http_ack_is_final_committed_answer_and_failure_is_not_success(compact_db, monkeypatch, transport):
    import httpx
    from app.api import chat
    from app.main import app
    from test_retry_operations import RetryScenario
    from app.services.memory import wiki
    scenario = RetryScenario(monkeypatch, compact_db)
    source = "请长期记住：我对花生过敏。"
    async def parse(message):
        value = semantics.validate_understanding(message, memory_outline(message))
        semantics.remember_understanding(message, value)
        return {"tasks": [{k: item[k] for k in ("task_type", "action", "text", "is_negated", "is_hypothetical", "is_third_party")}
                          for item in value["tasks"]]}
    monkeypatch.setattr(scenario.agent.graph, "parse_input", parse)
    monkeypatch.setattr(chat, "agent", scenario.agent)
    async def send(rid):
        import json
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://repair") as client:
            reply = await client.post("/api/chat/stream" if transport == "sse" else "/api/chat", json={"message": source, "request_id": rid, "conversation_id": "repair-api"})
            assert reply.status_code == 200
            if transport == "json": return reply.json()
            frames = [json.loads(line[6:]) for line in reply.text.splitlines() if line.startswith("data: ")]
            return {**frames[-1], "answer": "".join(f.get("content", "") for f in frames)}
    result = asyncio.run(send("repair-success"))
    assert result["answer"] == "已记住：我对花生过敏。"
    replay = asyncio.run(send("repair-success"))
    assert replay["answer"] == result["answer"] and replay["replayed"]
    monkeypatch.setattr(wiki, "materialize", lambda *_: (_ for _ in ()).throw(RuntimeError("commit failure")))
    failed = asyncio.run(send("repair-failed"))
    assert failed["answer"].startswith("这次记忆保存结果尚未确认")
    assert failed["memory_status"] == "degraded" and failed["intent"] == "error"
    assert failed["metrics"]["memory_update_status"] == "unconfirmed"
    with compact_db() as session:
        row = session.scalar(select(MemoryTurn).where(MemoryTurn.request_id == "repair-failed"))
        assert row.status != "completed" and row.response is None


def test_only_identical_insufficient_evidence_answers_are_merged():
    result = FitnessMultiAgentGraph._finalize_node({"worker_results": [
        {"worker": "nutrition", "status": "insufficient_evidence", "answer": "资料不足。"},
        {"worker": "nutrition", "status": "insufficient_evidence", "answer": "资料不足。"},
        {"worker": "workout_record", "status": "complete", "answer": "已记录卧推。"},
        {"worker": "workout_record", "status": "complete", "answer": "已记录卧推。"}]})
    assert result["final_answer"].count("资料不足。") == 1
    assert result["final_answer"].count("已记录卧推。") == 2


def test_router_plan_question_defers_to_complete_required_field_check():
    source = "帮我做一份适合我的增肌周计划。"
    raw = outline(source, kind="training_plan", needs_clarification=True, clarification="你有器械吗？")
    value = semantics.validate_understanding(source, raw)
    assert not value["needs_clarification"]
    from app.services.agent.task_information import PlanInformation, check_plan_information
    llm = SimpleNamespace(complete_structured=AsyncMock(return_value=PlanInformation(scope="weekly", scope_quote="周计划")))
    _, missing = asyncio.run(check_plan_information(llm, {"raw_message": source}))
    assert {f.name for f in missing.fields} == {"goal", "experience", "available_days"}


@pytest.mark.parametrize("proof,expected", [("workout_db_v1", "workout_record"), ("nutrition_db_v1", "nutrition_record")])
@pytest.mark.parametrize("restored", [False, True])
def test_react_completion_intent_comes_from_actual_proof(proof, expected, restored):
    from app.services.agent.react_write_runtime import WriteRuntimeSession
    runtime = WriteRuntimeSession.__new__(WriteRuntimeSession)
    runtime.ai_estimate_committed = False
    runtime.termination_reason = ""
    runtime.receipts = ["real-test-receipt"]
    runtime.committed_observations = [{"status": "completed", "commit_state": "committed", "result": {"proof_type": proof}}]
    runtime.resume_goal_results = {}
    if restored:
        runtime.resume_goal_results["goal"] = runtime.committed_observations.pop()
    runtime.response_ids = {}
    # Actual committed-payload implementation; unrelated response identity stub.
    runtime._response_ids = lambda: {}
    result = runtime._committed_payload({"final_answer": "已完成记录。"})
    assert result["intent"] == expected and result["current_task"]["task_type"] == expected


def test_referenced_fact_cannot_be_silently_dropped_by_budget():
    from app.services.context.budget import ContextBudgetManager, ContextBudgetExceeded
    source = "只复述已记住的信息。"
    fact = {"own-fact": {"text": "不能丢失的用户事实" * 500, "source_message_id": "own-source"}}
    value = semantics.validate_understanding(source, outline(source, purpose="fact_recall",
        references=[{"source_text": source, "query": "用户已存信息", "fact_fields": ["own-fact"]}]), context={"long_memory": fact})
    semantics.remember_understanding(source, value)
    with pytest.raises(ContextBudgetExceeded):
        ContextBudgetManager(window=400, output=70, margin=0).assemble(system="只读", query=source, state={"long_memory": fact}, hits=[])


def test_fact_deleted_after_load_is_rechecked_before_use(compact_db):
    from app.services.agent import dialogue_context
    tid, source, value, response = commit_explicit()
    compact.finish_turn(tid, response["answer"], response, semantic_values=[(source, value)])
    facts = compact.current_facts()
    recall = ChatRequest(message="只复述我的过敏史。", conversation_id="another")
    live, _ = compact.begin_turn(recall)
    understanding = semantics.validate_understanding(recall.message,
        outline(recall.message, purpose="fact_recall", critical_fact_fields=list(facts)), context={"long_memory": facts})
    with compact_db() as session, session.begin():
        session.get(MemoryTurn, tid).deleted_at = clock.utc_now()
    token = compact.current_turn.set(live)
    try:
        with pytest.raises(dialogue_context.DialogueContextUnavailable):
            asyncio.run(dialogue_context.restore(recall, {"long_memory": facts}, understanding))
    finally:
        compact.current_turn.reset(token)
