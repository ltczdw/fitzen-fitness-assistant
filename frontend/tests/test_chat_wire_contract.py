"""Real HTTP schema/serializer/transport; write runtime and storage gate are mocked.

Run from backend with scripts/run_tests_safely.py. No main-app startup, database,
provider, memory or diagnostics writes are performed. The fixture only changes
in-memory test settings; it never changes production configuration.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api import chat as chat_api
from app.core.config import settings
from app.schemas.chat import ChatResponse
from app.services.agent.agent import FitnessAgent
from app.services.agent import react_write_runtime, transport
from app.services.security.input_guard import inspect_input


def test_actual_chat_wire(monkeypatch):
    import app.services.agent.agent as agent_module

    op = "op2_" + "a" * 32
    digest = "b" * 64
    selector = {"pending_request_id": "synthetic-original", "pending_op_id": op,
                "pending_digest": digest}
    receipt = "nutrition:c2-2:" + op + ":g0:17"
    result = {"final_answer": "模拟结果", "intent": "nutrition_record",
              "react_write_status": "pending", "react_status": "pending",
              "mode_reason": "protected_pending", "receipt_refs": [],
              "pending_confirmations": [{**selector, "expires_at": "2099-01-01T00:00:00+00:00"}],
              "response_ids": {"request_id": "synthetic-original",
                               "message_id": "synthetic-original:assistant", "run_id": "synthetic-run"}}
    runtime = AsyncMock(side_effect=lambda *_: dict(result))
    monkeypatch.setattr(react_write_runtime, "run_write", runtime)
    monkeypatch.setattr(settings, "react_write_enabled", True)
    monkeypatch.setattr(settings, "memory_read_backend", "compact_v1")
    monkeypatch.setattr(chat_api, "require_ready", lambda: None)

    # The real FitnessAgent supplies _merge_safety/_write_gate_reason and the
    # real wire serializer. Only execution/storage/runtime boundaries are
    # replaced; no fake mixin or bare object stands in for the agent.
    agent = FitnessAgent()
    async def respond(agent_instance, request, parsed_input=None):
        return await agent_instance._respond_write(request, inspect_input(request.message))
    monkeypatch.setattr(agent_module, "execute_request", respond)
    monkeypatch.setattr(transport, "execute_request", respond)
    monkeypatch.setattr(chat_api, "agent", agent)
    app = FastAPI()
    app.include_router(chat_api.router, prefix="/api")
    client = TestClient(app)
    request = {"message": "模拟午餐", "mode": "react_write", "conversation_id": "synthetic-chat",
               "request_id": "synthetic-original"}
    pending = client.post("/api/chat", json=request)
    assert pending.status_code == 200
    pending = pending.json()
    assert pending["metrics"]["pending_confirmations"][0] == {**selector, "expires_at": "2099-01-01T00:00:00+00:00"}
    streamed = client.post("/api/chat/stream", json=request)
    assert streamed.status_code == 200
    events = [json.loads(block.split("data: ", 1)[1]) for block in streamed.text.strip().split("\n\n")]
    assert events[1] == {k: v for k, v in pending.items() if k != "answer"}
    assert events[-1] == events[1]
    confirmation = {**request, "request_id": "synthetic-confirmation", "message": "确认保存", "confirmation": selector}
    result.update(react_write_status="completed", react_status="completed",
                  receipt_refs=[receipt], pending_confirmations=[], response_ids={
                      "request_id": "synthetic-confirmation", "message_id": "synthetic-confirmation:assistant", "run_id": "synthetic-confirm-run"})
    completed = client.post("/api/chat", json=confirmation)
    assert completed.status_code == 200
    assert completed.json()["metrics"]["receipt_refs"] == [receipt]
    completed_stream = client.post("/api/chat/stream", json=confirmation)
    assert completed_stream.status_code == 200
    before = runtime.await_count
    for bad in ({**selector, "expires_at": "2099-01-01"}, {**selector, "pending_digest": "bad"}):
        assert client.post("/api/chat", json={**confirmation, "confirmation": bad}).status_code == 422
    assert client.post("/api/chat", json={**confirmation, "mode": "legacy"}).status_code == 422
    assert runtime.await_count == before
    monkeypatch.setattr(settings, "react_write_enabled", False)
    disabled = client.post("/api/chat", json=request)
    assert disabled.status_code == 200 and runtime.await_count == before
    assert disabled.json()["metrics"]["execution_mode"] == "limited"
    assert disabled.json()["metrics"]["mode_reason"] == "react_write_disabled"
    monkeypatch.setattr(settings, "react_write_enabled", True)
    result.update(react_write_status="limited", react_status="limited", mode_reason="privacy_invalidated",
                  receipt_refs=[], pending_confirmations=[])
    privacy_confirmation = client.post("/api/chat", json=confirmation).json()
    assert privacy_confirmation["metrics"]["mode_reason"] == "privacy_invalidated"
    privacy_stream = client.post("/api/chat/stream", json=confirmation)
    privacy_events = [json.loads(block.split("data: ", 1)[1]) for block in privacy_stream.text.strip().split("\n\n")]
    assert privacy_events[1]["metrics"]["mode_reason"] == "privacy_invalidated"
    assert privacy_events[-1]["metrics"]["mode_reason"] == "privacy_invalidated"
    terminal_privacy = ChatResponse.model_validate(react_write_runtime._privacy_refused_payload()).model_dump()
    fixtures = {"pending": pending, "pending_sse": streamed.text, "completed": completed.json(),
                "completed_sse": completed_stream.text, "disabled": disabled.json(),
                "confirmation_privacy_limited": privacy_confirmation, "confirmation_privacy_sse": privacy_stream.text,
                "terminal_privacy": terminal_privacy}
    destination = Path(__file__).with_name("chat-wire-contract.generated.json")
    destination.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_privacy_reason_preserves_other_mode_reasons(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from app.schemas.chat import ChatRequest
    monkeypatch.setattr(settings, "react_write_enabled", True)
    monkeypatch.setattr(settings, "memory_read_backend", "compact_v1")
    monkeypatch.setattr(react_write_runtime, "unfinished_progress", lambda _request: None)
    agent = FitnessAgent()
    for reason in ("privacy_invalidated", "protected_pending", "lease_busy"):
        monkeypatch.setattr(react_write_runtime, "run_write", AsyncMock(return_value={
            "final_answer": "模拟结果", "react_write_status": "limited",
            "mode_reason": reason, "receipt_refs": [], "pending_confirmations": []}))
        response = asyncio.run(agent._respond_write(
            ChatRequest(message="模拟", mode="react_write"), SimpleNamespace(level="normal")))
        assert response.metrics["mode_reason"] == ("privacy_invalidated" if reason == "privacy_invalidated" else "explicit_write")
