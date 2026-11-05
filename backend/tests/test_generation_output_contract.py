"""Visible answers, truncation accounting and honest worker status (no API)."""
import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import select, func

from app.core.config import settings
from app.db.model_budget_models import ModelCallLedger, ModelResultCache
from app.services.llm.budget import model_scope
from app.services.llm.client import LLMClient, LLMGatewayError, LLMErrorCategory
from evals.v2.safety import deny_network
from test_model_budget import budget_store


@pytest.fixture(autouse=True)
def offline():
    with deny_network():
        yield


def wire_model(monkeypatch, handler):
    monkeypatch.setattr(settings, "llm_base_url", "https://api.deepseek.com/v1")
    monkeypatch.setattr(settings, "llm_api_key", "synthetic-" + uuid4().hex)
    monkeypatch.setattr(settings, "llm_model", "deepseek-flash")
    monkeypatch.setattr(settings, "model_http_max_attempts", 2)
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    return LLMClient()


def result(content="visible answer", finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 23, "total_tokens": 40}}


@pytest.mark.parametrize("mode", ["async", "sync", "stream"])
def test_plain_answers_use_visible_output_mode_without_increasing_allowance(monkeypatch, budget_store, mode):
    seen = []
    def handle(request):
        payload = json.loads(request.content)
        seen.append(payload)
        # Reproduce a provider which spends its whole allowance on reasoning
        # when no mode is specified, leaving no final answer.
        if payload.get("thinking") != {"type": "disabled"}:
            return httpx.Response(200, json=result("", "length"))
        if not payload["stream"]:
            return httpx.Response(200, json=result())
        frames = [{"choices": [{"delta": {"content": "visible answer"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": result()["usage"]}]
        body = "".join("data: " + json.dumps(f) + "\n\n" for f in frames) + "data: [DONE]\n\n"
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})
    client = wire_model(monkeypatch, handle)
    messages = [{"role": "user", "content": "synthetic general question"}]
    with model_scope("visible-" + mode, store=budget_store):
        if mode == "sync":
            answer = client.complete_sync(messages)
        elif mode == "async":
            answer = asyncio.run(client.complete(messages))
        else:
            async def consume():
                return "".join([part async for part in client.stream(messages)])
            answer = asyncio.run(consume())
    assert answer == "visible answer" and len(seen) == 1
    assert seen[0]["max_tokens"] == settings.llm_max_output_tokens


@pytest.mark.parametrize("content", ["", "an incomplete answer"])
@pytest.mark.parametrize("mode", ["text", "message"])
def test_truncated_json_is_not_a_success_or_cached_or_retried(monkeypatch, budget_store, content, mode):
    seen = []
    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=result(content, "length"))
    client = wire_model(monkeypatch, handle)
    async def run():
        with model_scope("truncated-json", store=budget_store):
            call = client.complete if mode == "text" else client.complete_message
            with pytest.raises(LLMGatewayError) as error:
                await call([{"role": "user", "content": "synthetic request"}])
            assert error.value.category == LLMErrorCategory.PROTOCOL and not error.value.retryable
    asyncio.run(run())
    assert len(seen) == 1
    with budget_store.factory() as s:
        ledger = s.scalars(select(ModelCallLedger)).all()
        assert len(ledger) == 1 and ledger[0].status == "error" and ledger[0].usage["total_tokens"] == 40
        assert s.scalar(select(func.count()).select_from(ModelResultCache)) == 0


def test_truncated_stream_drains_usage_then_fails_without_retry(monkeypatch, budget_store):
    seen, emitted = [], []
    frames = [{"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
        {"choices": [], "usage": result()["usage"]}]
    def handle(request):
        seen.append(request)
        data = "".join("data: " + json.dumps(f) + "\n\n" for f in frames) + "data: [DONE]\n\n"
        return httpx.Response(200, content=data.encode())
    client = wire_model(monkeypatch, handle)
    async def run():
        with model_scope("truncated-stream", store=budget_store):
            with pytest.raises(LLMGatewayError, match="stream_response_truncated"):
                async for part in client.stream([{"role": "user", "content": "synthetic"}]):
                    emitted.append(part)
    asyncio.run(run())
    assert emitted == ["partial"] and len(seen) == 1
    with budget_store.factory() as s:
        ledger = s.scalar(select(ModelCallLedger))
        assert ledger.status == "error" and ledger.usage["total_tokens"] == 40


@pytest.mark.parametrize("worker,kind", [("plan_adjust", "training_plan"), ("nutrition", "nutrition_advice"), ("training_qa", "training_qa")])
def test_generation_failure_keeps_existing_fallback_but_does_not_complete_task(monkeypatch, worker, kind):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.services.agent.task_information import PlanInformation
    from app.services.rag.retrieval import RetrievedChunk
    graph = FitnessMultiAgentGraph()
    source = "给通用训练示例"
    monkeypatch.setattr(graph.llm, "complete_structured", AsyncMock(return_value=PlanInformation(scope="generic", scope_quote="通用训练示例")))
    monkeypatch.setattr(graph.llm, "complete", AsyncMock(side_effect=LLMGatewayError("synthetic truncated", LLMErrorCategory.PROTOCOL)))
    monkeypatch.setattr(graph.mcp, "complete_with_tools", AsyncMock(return_value=None))
    hits = [RetrievedChunk("训练和营养", "规律训练与均衡饮食。", .95, "official", {"rerank_score": .95, "rerank_status": "model"}, "synthetic")]
    monkeypatch.setattr(graph, "_readonly_retrieval", AsyncMock(return_value=hits))
    state = {"user_id": "demo-user", "raw_message": source, "task_text": source, "task_type": kind, "task_action": "question"}
    if worker == "plan_adjust":
        failure = AsyncMock(side_effect=LLMGatewayError("synthetic truncated", LLMErrorCategory.PROTOCOL))
        monkeypatch.setattr(graph.llm, "complete_message", failure)
        with pytest.raises(LLMGatewayError) as error:
            asyncio.run(graph.workers[worker].ainvoke(state))
        assert error.value.category == LLMErrorCategory.PROTOCOL
        assert failure.await_count == 1
        return
    value = asyncio.run(graph.workers[worker].ainvoke(state))["result"]
    assert value["status"] == "error" and value["model_used"] is False and value["answer"]
    final = graph._finalize_node({"worker_results": [{**value, "task_id": "one", "task_type": kind}]})
    assert final["task_outcomes"][0]["status"] == "error" and final["final_answer"] == value["answer"]


@pytest.mark.parametrize("base", ["https://other.invalid", "https://api.deepseek.com.other.invalid"])
def test_plain_provider_mode_does_not_leak_to_other_hosts(monkeypatch, base):
    monkeypatch.setattr(settings, "llm_base_url", base)
    assert "thinking" not in LLMClient._payload([{"role": "user", "content": "synthetic"}], 0, False)


@pytest.mark.parametrize('action,side_effect', [('question',False),('question',True),('record',False),('record',True)])
def test_only_readonly_goal_gateway_failure_can_become_a_local_error(monkeypatch, action, side_effect):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    graph = FitnessMultiAgentGraph()
    error = LLMGatewayError('private-provider-body', LLMErrorCategory.UPSTREAM, status_code=503)
    monkeypatch.setitem(graph.workers, 'plan_adjust', SimpleNamespace(ainvoke=AsyncMock(side_effect=error)))
    prior = {'task_id':'saved','task_type':'workout_record','status':'complete',
             'answer':'已记录。','commit_state':'committed','receipt_refs':['synthetic-proof']}
    state = {'user_id':'demo-user','raw_message':'synthetic',
        'current_task':{'id':'plan','task_type':'training_plan','action':action,
                        'text':'synthetic','side_effect':side_effect},
        'worker_results':[prior],'completed_tasks':['saved']}
    if action != 'question' or side_effect:
        with pytest.raises(LLMGatewayError):
            asyncio.run(graph._run_worker('plan_adjust', state))
        return
    result = asyncio.run(graph._run_worker('plan_adjust', state))
    assert result['worker_results'][0] == prior
    failed = result['worker_results'][1]
    assert failed['status'] == 'error' and failed['failure_details'][0]['code'] == 'model_http_503'
    assert 'HTTP 503' in failed['answer'] and 'private-provider-body' not in failed['answer']
    assert result['completed_tasks'] == ['saved']


def test_readonly_goal_policy_stop_is_not_relabelled_as_a_gateway_failure(monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    graph = FitnessMultiAgentGraph()
    error = LLMGatewayError('private-policy', LLMErrorCategory.POLICY, code='model_request_cancelled')
    monkeypatch.setitem(graph.workers, 'plan_adjust', SimpleNamespace(ainvoke=AsyncMock(side_effect=error)))
    state = {'user_id':'demo-user','raw_message':'synthetic',
             'current_task':{'id':'plan','task_type':'training_plan','action':'question','text':'synthetic'}}
    with pytest.raises(LLMGatewayError) as caught:
        asyncio.run(graph._run_worker('plan_adjust', state))
    assert caught.value.code == 'model_request_cancelled'


@pytest.mark.parametrize("stream", [False, True])
def test_write_then_generation_error_is_honest_and_replay_never_repeats_write(compact_db, monkeypatch, stream):
    from app.api import chat
    from app.main import app
    from app.schemas.chat import ChatRequest
    from app.services.agent.execution_state import PlannerDecision
    from app.services.agent.intent import IntentResult
    from app.services.agent.task_information import PlanInformation
    from app.services.rag.retrieval import RetrievedChunk
    from test_retry_operations import RetryScenario
    scenario = RetryScenario(monkeypatch, compact_db, workout_records=lambda message: [
        {"exercise": "卧推", "weight_kg": 40, "sets": 3, "reps": 8}])
    agent = scenario.agent
    monkeypatch.setattr(chat, "agent", agent)
    write, plan = "记录卧推40kg3组8次", "再给减脂初学者徒手30分钟单次训练安排"
    source = write + "，" + plan
    specs = [dict(id="saved", task_type="workout_record", worker="workout_record", action="record", text=write,
        side_effect=True, depends_on=[], payload={"write_authorization": "authorized"}),
        dict(id="plan", task_type="training_plan", worker="plan_adjust", action="question", text=plan, depends_on=["saved"])]
    monkeypatch.setattr(agent, "_route", AsyncMock(return_value=PlannerDecision(IntentResult("plan_adjust", 1), {}, tuple(specs))))
    facts = [("goal", "减脂"), ("experience", "初学者"), ("equipment", "徒手"), ("session_time", "30分钟")]
    info = PlanInformation(facts=[dict(field=k, source_id="current", quote=v) for k, v in facts])
    async def structured(messages, schema, *, name=None, **kwargs):
        if name == "return_request_understanding":
            payload = json.loads(messages[-1]["content"])
            source_value = str(payload.get("query") or "")
            return schema.model_validate({
                "tasks": [{"task_type": "training_qa", "action": "question",
                           "text": source_value, "is_negated": False,
                           "is_hypothetical": False, "is_third_party": False}],
                "safety": {"level": "normal", "evidence": "", "categories": [],
                           "reason": "scripted normal"},
            })
        return info
    monkeypatch.setattr(agent.graph.llm, "complete_structured", AsyncMock(side_effect=structured))
    failed = AsyncMock(side_effect=LLMGatewayError("synthetic truncated", LLMErrorCategory.PROTOCOL))
    monkeypatch.setattr(agent.graph.llm, "complete_message", failed)
    monkeypatch.setattr(agent.graph.mcp, "complete_with_tools", AsyncMock(return_value=None))
    hits = [RetrievedChunk("恢复训练", "逐步增加训练量。", .95, "official", {"rerank_score": .95, "rerank_status": "model"}, "synthetic")]
    monkeypatch.setattr(agent.graph, "_readonly_retrieval", AsyncMock(return_value=hits))
    request = ChatRequest(message=source, conversation_id="partial-generation")
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
            async def send(streaming):
                response = await api.post("/api/chat/stream" if streaming else "/api/chat", json=request.model_dump())
                assert response.status_code == 200
                if not streaming:
                    return response.json()
                frames = [json.loads(s[6:]) for s in response.text.splitlines() if s.startswith("data: ")]
                return {**frames[-1], "answer": "".join(v.get("content", "") for v in frames)}
            first = await send(stream)
            assert [(v["task_id"], v["status"]) for v in first["metrics"]["task_outcomes"]] == [("saved", "complete"), ("plan", "error")]
            assert "已记录" in first["answer"] and len(scenario.saves) == 1
            count = failed.await_count
            replay = await send(not stream)
            assert replay["answer"] == first["answer"] and replay["replayed"] is True
            assert failed.await_count == count and len(scenario.saves) == 1
        with compact_db() as s:
            from sqlalchemy import text
            assert s.scalar(text("SELECT COUNT(*) FROM workout_records")) == 1
            assert s.scalar(text("SELECT COUNT(*) FROM workout_session_sets")) == 3
    asyncio.run(run())
