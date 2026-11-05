"""Provider outage versus local budgets: mocked HTTP and isolated durable ledgers."""
import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import select

from app.core.config import settings
from app.db.model_budget_models import ModelCallLedger, ModelQuota
from app.schemas.chat import ChatRequest
from app.services.agent.execution import ChatExecutionMixin
from app.services.llm.budget import ModelBudgetError, model_prepare
from app.services.llm.client import LLMClient, LLMErrorCategory, LLMGatewayError
from tests.test_task_model_budget import make_store, request_scope


def response(error):
    return ChatExecutionMixin._semantic_limit_response(
        ChatRequest(message="给我具体的食物", mode="react_readonly"), "normal", error)


@pytest.mark.parametrize("gateway", ["text", "tools", "stream"])
def test_503_then_token_refusal_keeps_unknown_reservation(tmp_path, monkeypatch, gateway):
    store, engine = make_store(tmp_path)
    monkeypatch.setattr(settings, "llm_api_key", "mock-not-real")
    monkeypatch.setattr(settings, "llm_base_url", f"https://provider-budget-{gateway}.invalid/v1")
    monkeypatch.setattr(settings, "llm_model", "mock")
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0")
    seen = []
    original = httpx.AsyncClient

    def handle(request):
        seen.append(request)
        return httpx.Response(503)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs:
                        original(transport=httpx.MockTransport(handle), **kwargs))

    async def run():
        client = LLMClient()
        client.admission.token = AsyncMock()
        messages = [{"role": "user", "content": "x" * 40000}]
        with request_scope(store), model_prepare():
            with pytest.raises(LLMGatewayError) as caught:
                if gateway == "text":
                    await client.complete(messages)
                elif gateway == "tools":
                    await client.complete_message(messages, tools=[], tool_choice="auto")
                else:
                    assert [item async for item in client.stream(messages)] == []
        return caught.value

    try:
        error = asyncio.run(run())
        assert error.category == LLMErrorCategory.POLICY
        assert error.code == "model_prepare_token_budget_exhausted"
        # Existing budget classifiers retain the actual budget exception cause.
        assert isinstance(error.__cause__, ModelBudgetError)
        assert error.retry_failure.status_code == 503
        assert len(seen) == 1
        result = response(error)
        assert "暂时不可用" in result.answer and "Token" in result.answer
        assert "次数" not in result.answer and "没有新增保存" not in result.answer
        assert result.metrics["upstream_status"] == 503
        with store.factory() as session:
            rows = list(session.scalars(select(ModelCallLedger)))
            assert len(rows) == 1 and rows[0].status == "error"
            assert rows[0].error_code == "HTTPStatusError:http_503"
            assert not rows[0].usage
            assert rows[0].reserved_tokens > 32000
            quotas = list(session.scalars(select(ModelQuota).where(
                ModelQuota.quota_id.in_(rows[0].quota_ids))))
            assert len(quotas) == len(rows[0].quota_ids)
            assert all(row.calls == 1 and row.tokens == rows[0].reserved_tokens for row in quotas)
    finally:
        engine.dispose()


@pytest.mark.parametrize("code,reason,word", [
    ("model_prepare_token_budget_exhausted", "token_budget", "Token"),
    ("model_task_token_budget_exhausted", "token_budget", "Token"),
    ("model_token_budget_exhausted:daily", "token_budget", "Token"),
    ("model_prepare_budget_exhausted", "call_budget", "次数"),
    ("model_task_budget_exhausted", "call_budget", "次数"),
    ("model_budget_exhausted:request", "call_budget", "次数"),
    ("model_request_deadline", "deadline", "超时"),
    ("model_request_cancelled", "cancelled", "取消"),
    ("model_budget_storage_unavailable", "policy", "策略"),
])
def test_typed_stop_reasons_do_not_invent_provider_outage(code, reason, word):
    result = response(ModelBudgetError(code))
    assert result.metrics["model_stop_reason"] == reason
    assert result.metrics["retry_after_provider_failure"] is False
    assert word in result.answer and "暂时不可用" not in result.answer
    assert "没有新增保存" not in result.answer


def test_provider_message_is_not_exposed_and_completed_receipts_are_preserved(monkeypatch):
    from app.services.agent import react_write_runtime
    error = LLMGatewayError("PRIVATE PROVIDER CONTENT", LLMErrorCategory.POLICY,
                            code="model_prepare_token_budget_exhausted")
    error.retry_failure = LLMGatewayError("PRIVATE UPSTREAM CONTENT", LLMErrorCategory.UPSTREAM,
                                         retryable=True, status_code=503)
    monkeypatch.setattr(react_write_runtime, "unfinished_progress", lambda request: {
        "answer": "目标甲已保存，目标乙未完成。", "receipt_refs": ["receipt-a"],
        "response_ids": {"request_id": "frozen-request", "message_id": "frozen-message", "run_id": "frozen-run"}})
    result = ChatExecutionMixin._semantic_limit_response(
        ChatRequest(message="synthetic", mode="react_write"), "normal", error)
    assert result.answer == "目标甲已保存，目标乙未完成。"
    assert result.metrics["receipt_refs"] == ["receipt-a"]
    assert result.request_id == "frozen-request"
    assert "PRIVATE" not in result.model_dump_json()


@pytest.mark.parametrize("transport", ["json", "sse"])
def test_api_followup_explains_retry_stop_and_replay_makes_no_calls(tmp_path, monkeypatch, transport):
    from tests.react_phase_b_codex_support import environment, sse_body
    error = LLMGatewayError("local stop", LLMErrorCategory.POLICY,
                            code="model_prepare_token_budget_exhausted")
    error.retry_failure = LLMGatewayError("temporary", LLMErrorCategory.UPSTREAM,
                                         retryable=True, status_code=503)
    with environment(tmp_path, monkeypatch, "provider-stop-" + transport) as env:
        env.mock_llm(semantic_error=error)
        before = env.snapshot()
        def call():
            if transport == "json":
                return env.post("followup-stop", message="给我具体的食物")
            return sse_body(env.stream("followup-stop", message="给我具体的食物"))
        result = call()
        assert "暂时不可用" in result["answer"] and "Token" in result["answer"]
        assert "没有新增保存" not in result["answer"]
        assert result["metrics"]["model_stop_reason"] == "token_budget"
        calls = len(env.llm.semantic_calls)
        assert calls == 1
        replay = call()
        assert replay["answer"] == result["answer"]
        assert len(env.llm.semantic_calls) == calls
        assert env.snapshot() == before


@pytest.mark.parametrize("gateway", ["text", "tools", "stream"])
def test_503_retry_succeeds_when_original_budget_is_sufficient(tmp_path, monkeypatch, gateway):
    import json
    store, engine = make_store(tmp_path)
    monkeypatch.setattr(settings, "llm_api_key", "mock-not-real")
    monkeypatch.setattr(settings, "llm_base_url", f"https://provider-recovery-{gateway}.invalid/v1")
    monkeypatch.setattr(settings, "llm_model", "mock")
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0")
    seen = []
    original = httpx.AsyncClient
    answer = "具体食物示例：鸡蛋、燕麦、豆腐。"
    def handle(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(503)
        if gateway == "stream":
            payload = {"choices": [{"delta": {"content": answer}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 4, "completion_tokens": 2}}
            return httpx.Response(200, text="data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n",
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"content": answer}}],
                                        "usage": {"prompt_tokens": 4, "completion_tokens": 2}})
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs:
                        original(transport=httpx.MockTransport(handle), **kwargs))
    async def run():
        client = LLMClient()
        client.admission.token = AsyncMock()
        messages = [{"role": "user", "content": "给我具体的食物"}]
        with request_scope(store), model_prepare():
            if gateway == "text":
                return await client.complete(messages)
            if gateway == "tools":
                return (await client.complete_message(messages, tools=[], tool_choice="auto"))["content"]
            return "".join([item async for item in client.stream(messages)])
    try:
        assert asyncio.run(run()) == answer
        assert len(seen) == 2
        with store.factory() as session:
            rows = list(session.scalars(select(ModelCallLedger).order_by(ModelCallLedger.started_at)))
            assert [row.status for row in rows] == ["error", "success"]
            assert not rows[0].usage and rows[1].usage == {"prompt_tokens": 4, "completion_tokens": 2}
            first = rows[0].reserved_tokens
            quotas = list(session.scalars(select(ModelQuota).where(
                ModelQuota.quota_id.in_(rows[1].quota_ids))))
            assert quotas and all(row.calls == 2 and row.tokens == first + 6 for row in quotas)
    finally:
        engine.dispose()
