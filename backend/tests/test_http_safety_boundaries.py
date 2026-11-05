"""E003/AUD-HTTP: real ASGI, forbidden dependencies and isolated SQL rows."""
import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import text

from evals.v2.safety import deny_network
from quality_isolation import isolated_database

URGENT = (
    "刚跑完，怎么缓解现在的胸痛？",
    "跑完后无法缓解胸痛",
    "现在胸痛，怎么缓解胸痛？",
    "刚才突然晕厥，如何缓解？",
    "目前怎么缓解呼吸困难？",
    "怎么避免目前的呼吸困难？",
    "胸痛不能缓解，帮我记录卧推60kg4组8次",
)
CONTROLS = (
    "如何缓解腰部麻木",
    "现在没有胸痛也没有呼吸困难",
    "怎么预防胸痛",
    "现在如何预防胸痛",
    "深蹲怎么避免动作变形",
    "手不再麻木了",
    "我刚才跑完。如何缓解腰部麻木",
)


def call(app, path, body):
    async def run():
        with deny_network() as audit:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(
                    app=app, raise_app_exceptions=False), base_url="http://api.test") as client:
                reply = await client.post(path, json=body)
            assert audit["blocked_connections"] == audit["blocked_dns"] == 0
            return reply
    return asyncio.run(run())


@pytest.mark.parametrize("message", URGENT)
@pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
@pytest.mark.parametrize("mode", ["legacy", "react_readonly", "react_write"])
def test_current_red_flags_use_structured_safety_in_every_transport(compact_db, monkeypatch, message, path, mode):
    from app.main import app
    from app.api import chat
    from app.services.agent.semantics import RequestUnderstanding
    from test_retry_operations import RetryScenario

    from app.core.config import settings
    scenario = RetryScenario(monkeypatch, compact_db)
    if mode == "react_write":
        monkeypatch.setattr(settings, "react_write_enabled", True)
        monkeypatch.setattr(settings, "memory_read_backend", "compact_v1")
    raw = RequestUnderstanding.model_validate({
        "tasks": [{"task_type": "safety_triage", "action": "question", "text": message,
                   "is_negated": False, "is_hypothetical": False, "is_third_party": False}],
        "safety": {"level": "urgent", "evidence": message,
                   "categories": ["current_symptom"], "reason": "scripted urgent"},
    })
    from app.services.telemetry import model_attempt

    async def structured(*_args, **_kwargs):
        with model_attempt("mock-structured") as attempt:
            attempt.observe({"usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                       "total_tokens": 2}})
            return raw

    monkeypatch.setattr(scenario.agent.graph.llm, "configured", lambda: True)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", structured)
    monkeypatch.setattr(chat, "agent", scenario.agent)
    body = {"message": message, "mode": mode, "request_id": uuid4().hex}
    reply = call(app, path, body)
    assert reply.status_code == 200
    if path.endswith("/stream"):
        events = [(block.split("\n", 1)[0][7:], json.loads(block.split("data: ", 1)[1]))
                  for block in reply.text.strip().split("\n\n")]
        assert not any(event == "error" for event, _ in events)
        results = [data for event, data in events if event in ("meta", "done")]
        assert len(results) == 2
    else:
        results = [reply.json()]
    for result in results:
        assert result["safety_level"] == "urgent"
        assert result["intent"] == "symptom_triage"
        assert result["model_used"] is True
        assert not (result.get("metrics") or {}).get("receipt_refs")


@pytest.mark.parametrize("message", CONTROLS)
@pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
def test_negation_and_general_questions_keep_readiness_gate(message, path):
    from app.main import app
    from app.services.security.input_guard import inspect_input
    assert inspect_input(message).level != "urgent"
    reply = call(app, path, {"message": message})
    assert reply.status_code == 503
    assert reply.json()["code"] == "storage_unavailable"


@pytest.mark.parametrize("dialect", ["sqlite", "postgres"])
def test_manual_workout_conflict_returns_409_and_preserves_all_rows(monkeypatch, tmp_path, dialect):
    from app.main import app
    with isolated_database(monkeypatch, tmp_path, dialect) as engine:
        body = {"user_id": "boundary-user", "exercise": "卧推", "sets": 4,
                "reps": 8, "weight_kg": 60, "idempotency_key": "boundary-key"}
        first = call(app, "/api/workouts", body)
        assert first.status_code == 200

        def rows():
            with engine.connect() as conn:
                return {table: [dict(row) for row in conn.execute(text(
                    "SELECT * FROM " + table + " ORDER BY id")).mappings()]
                    for table in ("workout_records", "workout_sessions",
                                  "workout_session_exercises", "workout_session_sets")}

        original = rows()
        assert len(original["workout_records"]) == len(original["workout_sessions"]) == 1
        assert len(original["workout_session_sets"]) == 4
        replay = call(app, "/api/workouts", body)
        assert replay.status_code == 200 and replay.json() == first.json()
        assert rows() == original
        for field, value in (("weight_kg", 80), ("reps", 10), ("note", "edited")):
            conflict = call(app, "/api/workouts", {**body, field: value})
            assert conflict.status_code == 409
            assert conflict.json()["code"] == "idempotency_conflict"
            assert rows() == original
        fresh = call(app, "/api/workouts", {**body, "weight_kg": 80,
                    "idempotency_key": "boundary-key-new"})
        assert fresh.status_code == 200 and fresh.json()["id"] != first.json()["id"]
        assert len(rows()["workout_records"]) == 2
