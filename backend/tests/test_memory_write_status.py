import asyncio
from unittest.mock import AsyncMock
from app.services.agent import agent as module
from app.schemas.chat import ChatRequest, ChatResponse


def test_memory_completion_failure_preserves_success(monkeypatch):
    agent = module.FitnessAgent()
    monkeypatch.setattr(module, "begin_turn", lambda request: ("turn-test", None))
    monkeypatch.setattr(module, "finish_turn", lambda *args: (_ for _ in ()).throw(RuntimeError("database disconnected")))
    monkeypatch.setattr(agent, "_respond_impl", AsyncMock(return_value=ChatResponse(answer="saved", intent="workout_record")))
    response = asyncio.run(agent.respond(ChatRequest(message="record")))
    assert response.answer == "saved"
    assert response.memory_status == "degraded"


def test_legacy_append_failure_preserves_response_state(monkeypatch):
    agent = module.FitnessAgent()
    monkeypatch.setattr(module.settings, "memory_read_backend", "legacy")
    monkeypatch.setattr(module.memory_service, "append_turn", AsyncMock(side_effect=RuntimeError("memory down")))
    async def check():
        module._memory_error.set(None)
        await agent._persist_turn(ChatRequest(message="record"), "saved")
        assert module._memory_error.get() == "memory_persistence_failed"
    asyncio.run(check())


def test_answer_only_persistence_is_disabled_for_compact_memory(monkeypatch):
    """P1-3: compact memory commits once, with the complete response."""
    agent = module.FitnessAgent()
    monkeypatch.setattr(module.settings, "memory_read_backend", "compact_v1")
    append = AsyncMock()
    monkeypatch.setattr(module.memory_service, "append_turn", append)
    async def check():
        module._memory_error.set(None)
        await agent._persist_turn(ChatRequest(message="record"), "saved")
        assert module._memory_error.get() is None
    asyncio.run(check())
    append.assert_not_awaited()
