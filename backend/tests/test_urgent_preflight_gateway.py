"""Urgent DB-down preflight: real gateway metering, budget fail-closed, isolation."""
from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api import chat
from app.core.config import settings
from app.db.model_budget_models import ModelCallLedger
from app.main import app
from app.services.agent.agent import FitnessAgent
from app.services.llm.budget import ModelBudgetUnavailable, store_override
from app.services.llm.client import LLMClient

URGENT = "胸痛并呼吸困难，帮我查训练建议"
NORMAL = "现在没有胸痛也没有呼吸困难"


def _raw_understanding(query: str, level: str, *, include_safety: bool = True) -> dict:
    task_type = "safety_triage" if level == "urgent" else "training_qa"
    raw = {
        "tasks": [{
            "task_type": task_type,
            "action": "question",
            "text": query,
            "is_negated": False,
            "is_hypothetical": False,
            "is_third_party": False,
        }],
    }
    if include_safety:
        raw["safety"] = {
            "level": level,
            "evidence": query if level == "urgent" else "",
            "categories": ["current_symptom"] if level == "urgent" else [],
            "reason": "scripted gateway result",
        }
    return raw


def _gateway(monkeypatch, responder):
    """Install a real LLMClient HTTP gateway over MockTransport."""
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        result = responder(len(calls) - 1, payload)
        if isinstance(result, httpx.Response):
            return result
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "structured-1",
                "type": "function",
                "function": {
                    "name": "return_request_understanding",
                    "arguments": json.dumps(result, ensure_ascii=False),
                },
            }],
        }
        return httpx.Response(200, json={
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "total_tokens": 15},
        })

    from app.services.llm import client as gateway_module
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        gateway_module.httpx, "AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(handler), **kwargs))
    return calls


def _agent_with_real_gateway(monkeypatch) -> FitnessAgent:
    for name, value in (
            ("llm_api_key", "mock-not-real"),
            ("llm_base_url", "https://urgent-preflight.invalid/v1"),
            ("llm_model", "mock-structured"),
            ("llm_retry_delays", "0,0,0,0,0")):
        monkeypatch.setattr(settings, name, value)
    agent = FitnessAgent()
    agent.graph.llm = LLMClient()
    monkeypatch.setattr(chat, "agent", agent)
    return agent


def _post(client, path, message, user, conversation, request_id=None):
    body = {"message": message, "user_id": user,
            "conversation_id": conversation,
            "request_id": request_id or "urgent-" + uuid4().hex}
    if not path.endswith("/stream"):
        response = client.post(path, json=body)
        return {"status_code": response.status_code, **response.json()}
    with client.stream("POST", path, json=body) as response:
        return _sse(response)


def _sse(response):
    events, current = [], None
    for line in response.iter_lines():
        if line.startswith("event:"):
            current = line.partition(":")[2].strip()
        elif line.startswith("data:"):
            events.append((current, json.loads(line.partition(":")[2])))
    done = [data for event, data in events if event == "done"]
    assert len(done) == 1, [event for event, _ in events]
    answer = "".join(str(data.get("content") or "")
                     for event, data in events if event == "token")
    return {"status_code": response.status_code, **done[0], "answer": answer}


@pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
def test_urgent_preflight_uses_real_gateway_and_meters(monkeypatch, path):
    _agent_with_real_gateway(monkeypatch)
    calls = _gateway(monkeypatch, lambda _index, _payload: _raw_understanding(
        URGENT, "urgent"))
    store = store_override.get()
    client = TestClient(app)
    result = _post(client, path, URGENT, "preflight-user", "preflight-conversation")
    client.close()

    assert result["status_code"] == 200
    assert result["safety_level"] == "urgent"
    assert result["intent"] == "symptom_triage"
    assert result["model_used"] is True
    assert result["metrics"]["model_calls"] >= 1
    assert result["metrics"]["total_tokens"] == 15
    assert calls, "the real structured gateway must be reached"
    with store.factory() as session:
        rows = list(session.scalars(select(ModelCallLedger)))
    assert len(rows) == 1
    assert rows[0].status == "success"
    assert rows[0].usage["total_tokens"] == 15


def test_urgent_preflight_cached_understanding_reports_not_model_used(monkeypatch):
    _agent_with_real_gateway(monkeypatch)
    from app.services.agent import semantics

    async def cached(_llm, source):
        semantics.remember_safety(source, {
            "level": "urgent", "evidence": source,
            "categories": ["current_symptom"],
            "reason": "cached request-local assessment",
        })
        return {"tasks": [], "safety": semantics.safety_for(source)}

    monkeypatch.setattr(semantics, "understand", cached)
    client = TestClient(app)
    result = _post(client, "/api/chat", URGENT, "cache-user", "cache-conversation")
    client.close()

    assert result["status_code"] == 200
    assert result["safety_level"] == "urgent"
    assert result["model_used"] is False
    assert result["metrics"]["model_calls"] == 0


def test_urgent_preflight_does_not_reuse_same_text_safety_across_requests(monkeypatch):
    _agent_with_real_gateway(monkeypatch)
    levels = ["urgent", "normal"]
    calls = _gateway(
        monkeypatch,
        lambda index, _payload: _raw_understanding(URGENT, levels[index]))
    store = store_override.get()
    client = TestClient(app)
    first = _post(client, "/api/chat", URGENT, "user-a", "conversation-a")
    second = _post(client, "/api/chat", URGENT, "user-b", "conversation-b")
    client.close()

    assert first["status_code"] == 200 and first["safety_level"] == "urgent"
    assert second["status_code"] == 503
    assert second["code"] == "storage_unavailable"
    assert len(calls) == 2, "same text must not reuse another request's safety"
    with store.factory() as session:
        rows = list(session.scalars(select(ModelCallLedger)))
    assert len(rows) == 2
    assert all(row.status == "success" for row in rows)


def test_urgent_preflight_budget_unavailable_fails_closed_before_provider(monkeypatch):
    _agent_with_real_gateway(monkeypatch)
    calls = _gateway(monkeypatch, lambda _index, _payload: _raw_understanding(
        URGENT, "urgent"))
    store = store_override.get()

    def unavailable(*_args, **_kwargs):
        raise ModelBudgetUnavailable("model_budget_storage_unavailable")

    monkeypatch.setattr(store, "reserve", unavailable)
    client = TestClient(app)
    result = _post(client, "/api/chat", URGENT, "budget-user", "budget-conversation")
    client.close()

    assert result["status_code"] == 503
    assert result["code"] == "storage_unavailable"
    assert calls == [], "a failed budget store must not reach the provider"
    with store.factory() as session:
        assert list(session.scalars(select(ModelCallLedger))) == []


@pytest.mark.parametrize("case", ["normal", "negated", "missing-safety", "provider-error"])
def test_urgent_preflight_never_invents_urgent(monkeypatch, case):
    _agent_with_real_gateway(monkeypatch)

    def responder(_index, _payload):
        if case == "normal":
            return _raw_understanding(NORMAL, "normal")
        if case == "negated":
            return _raw_understanding("现在没有胸痛，也没有呼吸困难", "normal")
        if case == "missing-safety":
            return _raw_understanding(URGENT, "urgent", include_safety=False)
        return httpx.Response(500, json={"error": "synthetic upstream failure"})

    calls = _gateway(monkeypatch, responder)
    client = TestClient(app)
    result = _post(client, "/api/chat", URGENT, "control-user", "control-conversation")
    client.close()

    assert result["status_code"] == 503
    assert result["code"] == "storage_unavailable"
    assert calls, "the preflight must attempt the model boundary"
