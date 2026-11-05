import asyncio
from unittest.mock import AsyncMock
import httpx
from app.main import app
from app.db import session as db
from app.services.agent import agent as agent_module
from app.services.memory.redis_memory import memory_service


def test_urgent_bypasses_all_memory_and_database(monkeypatch):
    def forbidden(*a, **k): raise AssertionError("Memory/DB called")
    monkeypatch.setattr(agent_module, "begin_turn", forbidden)
    monkeypatch.setattr(memory_service, "load", forbidden)
    monkeypatch.setattr(memory_service, "initialize", forbidden)
    source = "胸痛并且呼吸困难"
    from app.services.agent.semantics import RequestUnderstanding
    model = RequestUnderstanding.model_validate({
        "tasks": [{"task_type": "safety_triage", "action": "question", "text": source,
                   "is_negated": False, "is_hypothetical": False, "is_third_party": False}],
        "safety": {"level": "urgent", "evidence": source,
                   "categories": ["current_symptom"], "reason": "scripted current emergency"},
    })
    async def complete(messages, schema, *, validator=None, **kwargs):
        assert schema is RequestUnderstanding
        if validator is not None:
            validator(model)
        return model
    monkeypatch.setattr(agent_module.agent.graph.llm, "configured", lambda: True)
    monkeypatch.setattr(agent_module.agent.graph.llm, "complete_structured", complete)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
            for route in ("/api/chat", "/api/chat/stream"):
                reply = await client.post(route,json={"message":source})
                assert reply.status_code == 200
                assert "urgent" in reply.text
    asyncio.run(check())

    # Missing model safety is never inferred from the dangerous wording: the
    # request falls through to the existing storage fail-closed gate.
    monkeypatch.setattr(agent_module.agent.graph.llm, "configured", lambda: False)
    async def fail_closed():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
            for route in ("/api/chat", "/api/chat/stream"):
                reply = await client.post(route,json={"message":source})
                assert reply.status_code == 503
                assert "urgent" not in reply.text.lower()
    asyncio.run(fail_closed())


def test_normal_request_before_stream_headers_returns_503():
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
            for route in ("/api/chat", "/api/chat/stream"):
                reply=await client.post(route,json={"message":"记录两个鸡蛋"})
                assert reply.status_code == 503
                assert reply.json()["code"] == "storage_unavailable"
    asyncio.run(check())


def test_health_does_not_confuse_connection_with_initialization(monkeypatch):
    from app.api import health as health_module
    monkeypatch.setattr(health_module,"_database_probe",lambda:True)
    monkeypatch.setattr(memory_service,"initialize",AsyncMock(return_value=False))
    monkeypatch.setattr(db,"init_db",lambda:(_ for _ in ()).throw(AssertionError("DDL in health")))
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
            result=await client.get("/api/health")
            assert result.status_code == 503
            body=result.json()
            assert body["components"]["db"] == {"initialized":False,"connected":True,"ready":False}
            assert body["components"]["llm"]["probe"] == "configuration_only"
    asyncio.run(check())


def test_startup_refuses_uninitialized_database(monkeypatch):
    from app import main
    from app.db.storage import StorageUnavailableError
    import pytest
    monkeypatch.setattr(main,"init_db",lambda:False)
    async def check():
        with pytest.raises(StorageUnavailableError):
            async with main.lifespan(app):
                raise AssertionError("Startup must fail")
    asyncio.run(check())
