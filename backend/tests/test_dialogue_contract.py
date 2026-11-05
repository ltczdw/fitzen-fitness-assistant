"""Dialogue relations and shared repair, using mock HTTP and isolated SQL."""
import asyncio
import copy
import json
from enum import StrEnum
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.schemas.chat import ChatRequest
from app.services.agent import semantics
from app.services.llm.structured import StructuredOutputError, parse_tool_result
from app.services.memory import compact
from evals.v2.safety import deny_network
from test_dialogue_continuity import history, outline, task
from test_retry_operations import RetryScenario
from test_structured_memory_summary import mock_http


@pytest.fixture(autouse=True)
def isolated_context_and_network():
    token = semantics.request_understanding.set({})
    with deny_network():
        yield
    semantics.request_understanding.reset(token)


def wire(value, name="return_request_understanding"):
    return {"content": None, "tool_calls": [{"id": "typed-result", "type": "function",
        "function": {"name": name, "arguments": json.dumps(value, ensure_ascii=False)}}]}


def provider(monkeypatch, respond):
    seen = []
    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        message = respond(body, len(seen))
        return httpx.Response(200, json={"choices": [{"message": message}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5}})
    return mock_http(monkeypatch, handle), seen


@pytest.mark.parametrize("relation", ["new", "follow_up", "answer", "clarification", "correction", "confirmation", "cancel"])
def test_all_documented_relations_are_descriptive_and_never_grant_write_permission(relation):
    source = "选择第二项"
    schema = semantics.RequestUnderstanding.model_json_schema()["properties"]["relation"]
    assert relation in schema["enum"] and relation in semantics.PROMPT
    result = semantics.validate_understanding(source, outline(source, relation=relation))
    assert result["relation"] == relation
    assert result["validated_tasks"][0]["action"] == "question"
    assert not result["validated_tasks"][0]["side_effect"]


@pytest.mark.parametrize("field,invalid,valid,location", [
    ("relation", "responding_to_previous_question", "answer", ["relation"]),
    ("reply_mode", "ordinary_chat", "grounded", ["reply_mode"]),
    ("action", "save_everything", "question", ["tasks", 0, "action"]),
])
def test_shared_repair_identifies_actual_field_and_options_without_repeating_bad_values(monkeypatch, field, invalid, valid, location):
    source = "第二项适合什么情况"
    bad = outline(source)
    if field == "action": bad["tasks"][0][field] = invalid
    else: bad[field] = invalid
    corrected = copy.deepcopy(bad)
    if field == "action": corrected["tasks"][0][field] = valid
    else: corrected[field] = valid
    client, seen = provider(monkeypatch, lambda body, number: wire(bad if number == 1 else corrected))
    result = asyncio.run(semantics.understand(client, source))
    assert result is not None and not result["validated_tasks"][0]["side_effect"]
    assert len(seen) == 2
    repair = seen[1]["messages"][0]["content"]
    assert json.dumps(location, ensure_ascii=False) in repair
    assert 'allowed_values' in repair and valid in repair and invalid not in repair
    assert seen[0]["messages"][1:] == seen[1]["messages"][1:]


def test_repeated_invalid_relation_fails_closed_after_two_posts_with_neutral_copy(monkeypatch, caplog):
    from app.services.telemetry import active_metrics, RequestMetrics
    raw = outline("第二项", relation="PRIVATE-INVALID-RELATION")
    client, seen = provider(monkeypatch, lambda *_: wire(raw))
    token = active_metrics.set(RequestMetrics())
    try:
        assert asyncio.run(semantics.understand(client, "第二项")) is None
        assert semantics.get_understanding("第二项") is None
        answer = semantics.understanding_unavailable_reply()
        assert "未通过校验" in answer
        assert "未保存" not in answer and "手动表单" not in answer
        assert "未保存" in semantics.understanding_unavailable_reply(record_mode=True)
    finally:
        active_metrics.reset(token)
    assert len(seen) == 2 and 'PRIVATE-INVALID-RELATION' not in caplog.text
    assert 'PRIVATE-INVALID-RELATION' not in seen[1]["messages"][0]["content"]


class Format(StrEnum):
    TABLE = "table"
    LIST = "list"


class NestedItem(BaseModel):
    format: Format


class NestedOutput(BaseModel):
    items: list[NestedItem]
    enabled: bool = Field(strict=True)


def test_generic_nested_enum_options_follow_schema_refs_and_array_paths():
    with pytest.raises(StructuredOutputError) as caught:
        parse_tool_result(wire({"items": [{"format": "PRIVATE-VALUE"}], "enabled": "yes"}, "nested"), NestedOutput, "nested")
    issues = caught.value.validation_errors
    assert {"loc": ["items", 0, "format"], "type": "enum", "allowed_values": ["table", "list"]} in issues
    assert {"loc": ["enabled"], "type": "bool_type"} in issues
    assert "PRIVATE-VALUE" not in json.dumps(issues)


@pytest.mark.parametrize("violation", ["source", "reference", "assistant_fact"])
def test_legal_answer_relation_does_not_relax_source_or_history_validation(violation):
    source = "第二项"
    raw = outline(source, relation="answer")
    if violation == "source": raw["tasks"][0]["text"] = "从未说过的训练记录"
    elif violation == "reference": raw["references"] = [{"source_text": source, "query": "补全问题", "context_ids": ["outside"]}]
    else: raw["critical_history_ids"] = ["assistant"]
    with pytest.raises(ValueError):
        semantics.validate_understanding(source, raw, context={"short_memory": [
            {"message_id": "assistant", "role": "assistant", "content": "建议方案"}]})


CASES = [
    ("卧推时肩膀不适是什么原因", "不适在什么位置、哪个阶段？", "前侧，推起时，最近几天", "answer", "safety_triage", "卧推前侧不适推起时最近几天的安全注意事项"),
    ("帮我分析饮食安排", "你更关心什么目标？", "控制总热量，晚餐清淡", "answer", "nutrition_advice", "控制总热量且晚餐清淡的饮食安排"),
    ("解释两个动作的区别", "你想先了解哪一个？", "第二个的注意事项", "follow_up", "training_qa", "上文第二个动作的注意事项"),
    ("比较家用器械", "你有哑铃吗？", "不是哑铃，是弹力带", "correction", "training_qa", "弹力带在居家训练中的使用原则"),
    ("比较两种训练方式", "你指的是第二种居家训练吗？", "对，第二种", "confirmation", "training_qa", "上文第二种居家训练方式的注意事项"),
    ("比较训练方式", "先了解居家训练吗？", "换个话题，解释蛋白质的作用", "new", "nutrition_advice", "蛋白质在营养中的作用"),
    ("比较训练方式", "继续介绍第二种吗？", "先取消刚才的问题", "cancel", "training_qa", "取消刚才的问题"),
]


@pytest.mark.parametrize("transport", ["json", "sse"])
@pytest.mark.parametrize("original,assistant,source,relation,kind,resolved", CASES,
                         ids=["symptom", "nutrition", "followup", "correction", "confirmation", "topic", "cancel"])
def test_real_http_continuation_keeps_context_and_routes_readonly_without_record_writes(
        compact_db, monkeypatch, transport, original, assistant, source, relation, kind, resolved):
    from app.api import chat
    from app.main import app
    from app.core.config import settings
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.services.rag.retrieval import RetrievedChunk
    api_client = httpx.AsyncClient
    tid, messages = history(compact_db, cid="contract", user=original, assistant=assistant)
    scenario = RetryScenario(monkeypatch, compact_db)
    agent = scenario.agent
    answer = "已结合前文和本轮补充继续处理。" if relation != "cancel" else "已结束刚才的问题。"
    raw = {"tasks": [task(source, kind)], "relation": relation,
           "dialogue_ids": [tid + ":user", tid + ":assistant"],
           "safety": {"level": "normal", "evidence": "", "categories": [],
                      "reason": "scripted normal"},
           "reply_mode": "conversation" if relation == "cancel" else "grounded"}
    if relation != "cancel":
        raw["references"] = [{"source_text": source, "query": resolved,
            "evidence_ids": [tid + ":user"], "context_ids": [tid + ":assistant"]}]
    def respond(body, number):
        choice = body.get("tool_choice")
        name = choice.get("function", {}).get("name") if isinstance(choice, dict) else None
        if name == "return_request_understanding": return wire(raw)
        if name == "return_dialogue_answer": return wire({"answer": answer, "requires_grounding": False}, name)
        assert not body.get("tools"), "No record or other tool may be invoked"
        return {"content": answer}
    client, seen = provider(monkeypatch, respond)
    monkeypatch.setattr(agent.graph, "llm", client)
    monkeypatch.setattr(agent.graph.input_parser, "llm", client)
    monkeypatch.setattr(agent.graph, "parse_input", FitnessMultiAgentGraph.parse_input.__get__(agent.graph))
    monkeypatch.setattr(settings, "input_parser_enabled", True)
    monkeypatch.setattr(chat, "agent", agent)
    from app.services.agent import execution
    monkeypatch.setattr(execution.memory_service, "load", AsyncMock(return_value={"short_memory": messages, "memory_backend": "fixture"}))
    monkeypatch.setattr(agent.graph.mcp, "complete_with_tools", AsyncMock(return_value=None))
    queries = []
    def search(query, **kwargs):
        queries.append(query)
        return [RetrievedChunk("测试依据", resolved + "。按个体情况逐步调整，注意动作与恢复。", .99,
            "fixture", {"rerank_status": "model", "rerank_score": .99}, "contract-hit")]
    monkeypatch.setattr(agent.graph.rag, "search", search)
    request = ChatRequest(message=source, conversation_id="contract")
    async def run():
        async with api_client(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
            response = await api.post("/api/chat" + ("/stream" if transport == "sse" else ""), json=request.model_dump())
            assert response.status_code == 200
            if transport == "json": payload = response.json()
            else:
                frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                assert "".join(frame.get("content", "") for frame in frames) == answer
                payload = {**frames[-1], "answer": "".join(frame.get("content", "") for frame in frames)}
            assert payload["answer"] == answer and payload["clarifications"] == []
            assert payload["intent"] == ("conversation" if relation == "cancel" else kind)
            count = len(seen)
            replay = await api.post("/api/chat", json=request.model_dump())
            assert replay.status_code == 200 and replay.json()["replayed"]
            assert replay.json()["answer"] == answer and len(seen) == count
    asyncio.run(run())
    prefix = "营养饮食 " if kind == "nutrition_advice" else ""
    assert queries == ([] if relation == "cancel" else [prefix + resolved])
    assert len(seen) == 2  # one understanding and one answer, no repair for canonical labels
    assert any(m["role"] == "assistant" and assistant in m["content"] for m in seen[0]["messages"])
    assert any(original in m["content"] for m in seen[0]["messages"])
    assert not scenario.saves and not any(name.startswith("save_") for name in scenario.tool_calls)
    with compact_db() as session:
        for table in ("workout_records", "nutrition_logs", "user_food_log", "training_plans"):
            assert session.execute(text("SELECT COUNT(*) FROM " + table)).scalar() == 0


@pytest.mark.parametrize("transport", ["json", "sse"])
def test_failed_understanding_never_claims_a_record_failure_or_dispatches_workers(compact_db, monkeypatch, transport):
    from app.api import chat
    from app.main import app
    from app.core.config import settings
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.services.agent import execution
    api_client = httpx.AsyncClient
    _, messages = history(compact_db, cid="failed-contract", user="两个选项分别是什么", assistant="你更想了解哪一项？")
    scenario = RetryScenario(monkeypatch, compact_db)
    source = "第二项"
    client, seen = provider(monkeypatch, lambda *_: wire(outline(source, relation="unknown-dialogue-relation")))
    monkeypatch.setattr(scenario.agent.graph, "llm", client)
    monkeypatch.setattr(scenario.agent.graph.input_parser, "llm", client)
    monkeypatch.setattr(scenario.agent.graph, "parse_input", FitnessMultiAgentGraph.parse_input.__get__(scenario.agent.graph))
    monkeypatch.setattr(settings, "input_parser_enabled", True)
    monkeypatch.setattr(chat, "agent", scenario.agent)
    monkeypatch.setattr(execution.memory_service, "load", AsyncMock(return_value={"short_memory": messages}))
    workers = AsyncMock(side_effect=AssertionError("Invalid understanding must not dispatch a worker"))
    monkeypatch.setattr(scenario.agent.graph, "run", workers)
    request = ChatRequest(message=source, conversation_id="failed-contract")
    async def run():
        async with api_client(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
            response = await api.post("/api/chat" + ("/stream" if transport == "sse" else ""), json=request.model_dump())
            assert response.status_code == 200
            if transport == "json": payload = response.json()
            else:
                frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                payload = {**frames[-1], "answer": "".join(frame.get("content", "") for frame in frames)}
            assert payload["intent"] == "execution_limited"
            assert "安全语义判断" in payload["answer"]
            assert "未保存" not in payload["answer"] and "手动表单" not in payload["answer"]
            assert not payload.get("clarifications")
            assert "semantic_model_protocol" in payload["metrics"]["failure_reasons"]
    asyncio.run(run())
    assert len(seen) == 2 and not workers.called and not scenario.saves
