"""One bounded retry for retryable provider failures in shared semantics."""
import asyncio

import pytest

from app.services.agent import semantics
from app.services.llm.client import LLMErrorCategory, LLMGatewayError
from app.services.llm.structured import StructuredOutputError


def _run(monkeypatch, fake):
    calls = []

    async def wrapper(llm, messages, validator):
        calls.append(1)
        return await fake(calls)

    monkeypatch.setattr(semantics, "_complete_understanding", wrapper)
    return calls


def test_retryable_provider_error_buys_one_more_attempt(monkeypatch):
    async def fake(calls):
        if len(calls) == 1:
            raise LLMGatewayError("temporary", LLMErrorCategory.UPSTREAM,
                                  retryable=True, status_code=503)
        return "ok"

    calls = _run(monkeypatch, fake)
    assert asyncio.run(semantics._complete_understanding_retry(None, [], None)) == "ok"
    assert len(calls) == 2


def test_non_retryable_policy_and_schema_errors_are_not_retried(monkeypatch):
    cases = [
        LLMGatewayError("hard", LLMErrorCategory.UPSTREAM, retryable=False, status_code=400),
        LLMGatewayError("budget", LLMErrorCategory.POLICY, retryable=True),
        StructuredOutputError("structured_invalid_json", []),
    ]
    for exc in cases:
        async def fake(_calls, _exc=exc):
            raise _exc

        calls = _run(monkeypatch, fake)
        with pytest.raises(type(exc)):
            asyncio.run(semantics._complete_understanding_retry(None, [], None))
        assert len(calls) == 1, type(exc).__name__


def test_retry_is_capped_at_two_attempts(monkeypatch):
    async def fake(_calls):
        raise LLMGatewayError("down", LLMErrorCategory.TRANSPORT,
                              retryable=True, status_code=502)

    calls = _run(monkeypatch, fake)
    with pytest.raises(LLMGatewayError):
        asyncio.run(semantics._complete_understanding_retry(None, [], None))
    assert len(calls) == 2


@pytest.mark.parametrize("wrapped", [False, True])
def test_semantic_retry_keeps_provider_reason_when_next_call_is_budget_stopped(monkeypatch, wrapped):
    from app.services.llm.budget import ModelBudgetError
    from tests.test_provider_budget_feedback import response
    upstream = LLMGatewayError("temporary", LLMErrorCategory.UPSTREAM,
                               retryable=True, status_code=503)
    stopped = (LLMGatewayError("stop", LLMErrorCategory.POLICY,
                              code="model_prepare_token_budget_exhausted") if wrapped else
               ModelBudgetError("model_prepare_token_budget_exhausted"))
    async def fake(calls):
        raise upstream if len(calls) == 1 else stopped
    calls = _run(monkeypatch, fake)
    with pytest.raises(type(stopped)) as caught:
        asyncio.run(semantics._complete_understanding_retry(None, [], None))
    assert len(calls) == 2
    result = response(caught.value)
    assert result.metrics["upstream_status"] == 503
    assert "暂时不可用" in result.answer and "Token" in result.answer
