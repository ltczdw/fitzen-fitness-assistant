"""Repair source/reference errors within the same two-attempt contract."""
import asyncio
import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import text

from app.schemas.chat import ChatRequest
from app.services.agent import semantics
from app.services.llm.structured import StructuredOutputError
from evals.v2.safety import deny_network
from test_dialogue_contract import provider, wire
from test_dialogue_continuity import outline
from test_retry_operations import RetryScenario


@pytest.fixture(autouse=True)
def isolated_semantics():
    token = semantics.request_understanding.set({})
    with deny_network(): yield
    semantics.request_understanding.reset(token)


@pytest.mark.parametrize("bad_field,bad_value,code", [
    ("critical_fact_fields", ["PRIVATE-NEW-CONSTRAINT"], "semantic_critical_fact_invalid"),
    ("critical_history_ids", ["PRIVATE-UNKNOWN-MESSAGE"], "semantic_context_evidence_invalid"),
    ("dialogue_ids", ["PRIVATE-FOREIGN-DIALOGUE"], "semantic_dialogue_context_invalid"),
    ("references", [{"source_text": "PRIVATE-WRONG-SOURCE", "query": "PRIVATE-QUERY"}], "semantic_reference_evidence_invalid"),
    ("references", [{"source_text": "分析当前情况", "query": "PRIVATE-UNSUPPORTED-REWRITE"}], "semantic_reference_without_evidence"),
    ("memory_requests", [{"action": "remember", "quote": "PRIVATE-INVENTED-FACT", "key": "invented", "confidence": 1.0}], "semantic_memory_evidence_invalid"),
])
def test_source_constraints_get_specific_bounded_correction(monkeypatch, caplog, bad_field, bad_value, code):
    source = "分析当前情况"
    good = outline(source, current_is_critical=True)
    bad = {**good, bad_field: bad_value}
    client, seen = provider(monkeypatch, lambda body, number: wire(bad if number == 1 else good))
    result = asyncio.run(semantics.understand(client, source, context={}))
    assert result and result["current_is_critical"] and not result["validated_tasks"][0]["side_effect"]
    assert len(seen) == 2
    repair = seen[1]["messages"][0]["content"]
    assert code in repair and bad_field in repair
    assert "PRIVATE-" not in repair and "PRIVATE-" not in caplog.text
    assert seen[0]["messages"][1:] == seen[1]["messages"][1:]


def test_current_constraints_are_not_required_to_exist_as_saved_facts():
    source = "这几天不适，先分析原因"
    result = semantics.validate_understanding(source, outline(source, current_is_critical=True))
    assert result["current_is_critical"] and result["critical_fact_fields"] == []
    assert result["memory_requests"] == []


def test_multiple_bad_references_are_reported_together_without_extra_rounds(monkeypatch):
    source = "分析当前情况"
    good = outline(source, current_is_critical=True)
    bad = {**good, "critical_fact_fields": ["PRIVATE-FACT"],
           "critical_history_ids": ["PRIVATE-HISTORY"], "dialogue_ids": ["PRIVATE-DIALOGUE"]}
    client, seen = provider(monkeypatch, lambda body, number: wire(bad if number == 1 else good))
    assert asyncio.run(semantics.understand(client, source))
    assert len(seen) == 2
    repair = seen[1]["messages"][0]["content"]
    for field in ("critical_fact_fields", "critical_history_ids", "dialogue_ids"):
        assert field in repair
    assert "PRIVATE-" not in repair


def test_schema_and_source_corrections_share_one_physical_allowance(monkeypatch):
    source = "分析当前情况"
    schema_bad = outline(source, relation="unsupported")
    source_bad = outline(source, critical_fact_fields=["PRIVATE-INVENTED-FIELD"])
    client, seen = provider(monkeypatch, lambda body, number: wire(schema_bad if number == 1 else source_bad))
    assert asyncio.run(semantics.understand(client, source)) is None
    assert len(seen) == 2 and semantics.get_understanding(source) is None


def test_corrected_source_is_validated_again_and_invalid_result_never_cached(monkeypatch):
    source = "分析当前情况"
    bad = outline(source, critical_fact_fields=["PRIVATE-INVENTED-FIELD"])
    client, seen = provider(monkeypatch, lambda *_: wire(bad))
    assert asyncio.run(semantics.understand(client, source)) is None
    assert len(seen) == 2 and semantics.get_understanding(source) is None


@pytest.mark.parametrize("field", ["source", "memory_target", "fact_span"])
def test_other_application_validation_errors_also_receive_specific_repair(monkeypatch, field):
    source = "分析当前情况"
    good = outline(source)
    bad = copy.deepcopy(good)
    if field == "source": bad["tasks"][0]["text"] = "PRIVATE-INVENTED-EXECUTION"
    elif field == "fact_span": bad["tasks"][0]["fact_text"] = "PRIVATE-FOREIGN-FACT"
    else: bad["memory_requests"] = [{"action": "replace", "quote": source, "key": "item", "target_fields": ["PRIVATE-TARGET"], "confidence": 1.0}]
    client, seen = provider(monkeypatch, lambda body, number: wire(bad if number == 1 else good))
    assert asyncio.run(semantics.understand(client, source))
    assert len(seen) == 2 and 'semantic_' in seen[1]["messages"][0]["content"]


@pytest.mark.parametrize("transport", ["json", "sse"])
@pytest.mark.parametrize("repaired", [True, False])
def test_complete_first_question_and_repair_reach_answer_without_any_business_write(compact_db, monkeypatch, transport, repaired):
    from app.api import chat
    from app.main import app
    from app.core.config import settings
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.services.rag.retrieval import RetrievedChunk
    api_client = httpx.AsyncClient
    scenario = RetryScenario(monkeypatch, compact_db)
    agent = scenario.agent
    source = "推举时肩前不适是什么原因"
    good = {**outline(source, current_is_critical=True), "tasks": [{
        "task_type": "safety_triage", "action": "question", "text": source,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False}]}
    bad = {**good, "critical_fact_fields": ["PRIVATE-CURRENT-SYMPTOM"]}
    answer = "已将本次问题交给安全问答流程。"
    def respond(body, number):
        if isinstance(body.get("tool_choice"), dict):
            return wire(bad if number == 1 or not repaired else good)
        assert repaired and not body.get("tools")
        return {"content": answer}
    client, seen = provider(monkeypatch, respond)
    monkeypatch.setattr(agent.graph, "llm", client)
    monkeypatch.setattr(agent.graph.input_parser, "llm", client)
    monkeypatch.setattr(agent.graph, "parse_input", FitnessMultiAgentGraph.parse_input.__get__(agent.graph))
    monkeypatch.setattr(settings, "input_parser_enabled", True)
    monkeypatch.setattr(chat, "agent", agent)
    monkeypatch.setattr(agent.graph.mcp, "complete_with_tools", AsyncMock(return_value=None))
    hit = RetrievedChunk("测试依据", source + "。根据情况关注动作与恢复。", .99, "fixture",
                         {"rerank_status": "model", "rerank_score": .99}, "source-repair-hit")
    search = lambda *args, **kwargs: [hit]
    monkeypatch.setattr(agent.graph.rag, "search", search)
    request = ChatRequest(message=source, conversation_id="source-repair")
    async def run():
        async with api_client(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
            response = await api.post("/api/chat" + ("/stream" if transport == "sse" else ""), json=request.model_dump())
            assert response.status_code == 200
            if transport == "json": payload = response.json()
            else:
                frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                payload = {**frames[-1], "answer": "".join(frame.get("content", "") for frame in frames)}
            if repaired:
                assert payload["answer"] == answer and payload["intent"] == "safety_triage"
            else:
                assert payload["intent"] == "execution_limited"
                assert "安全语义判断" in payload["answer"] and "手动表单" not in payload["answer"]
            before = len(seen)
            replay = await api.post("/api/chat", json=request.model_dump())
            assert replay.status_code == 200 and replay.json()["replayed"]
            assert replay.json()["answer"] == payload["answer"] and len(seen) == before
    asyncio.run(run())
    assert len(seen) == (3 if repaired else 2) and not scenario.saves
    with compact_db() as session:
        for table in ("workout_records", "user_food_log", "nutrition_logs", "training_plans", "user_facts"):
            assert session.execute(text("SELECT COUNT(*) FROM " + table)).scalar() == 0
