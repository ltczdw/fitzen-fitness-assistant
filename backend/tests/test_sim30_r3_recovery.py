"""SIM30 R3 item 3: recovery advice reads the same real saved sessions."""
from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from app.core import clock


def _set(weight: float, reps: int):
    return {"weight_kg": weight, "reps": reps}


def _session(title: str, exercise: str, sets: list[dict], user_id: str = "demo-user"):
    from app.db.workout_session_repository import create_session

    return create_session({
        "user_id": user_id, "session_date": clock.today(), "title": title,
        "note": "", "exercises": [{"exercise_name": exercise, "sets": sets}],
    })


def _spec(day=None, user_id: str = "demo-user"):
    day = (day or clock.today()).isoformat()
    return {"resource": "workout_records", "date_start": day, "date_end": day,
            "exercises": []}


def _three_sessions(user_id: str = "demo-user"):
    _session("下拉", "高位下拉", [_set(35, 10)] * 3, user_id=user_id)
    _session("推胸", "坐姿推胸", [_set(22, 12)] * 2, user_id=user_id)
    _session("划船", "哑铃划船", [_set(8, 10)] * 3, user_id=user_id)


def test_recovery_facts_three_sessions_eight_sets(compact_db):
    from app.services.agent.read_resolver import advisory_read_facts

    _three_sessions()
    result = advisory_read_facts("demo-user", _spec())
    assert result["status"] == "completed"
    assert "3 项动作，共 8 组" in result["prefix"]
    for value in ("35 kg", "22 kg", "8 kg", "高位下拉", "坐姿推胸", "哑铃划船"):
        assert value in result["prefix"]
    assert result["unknown_metrics"]["sleep_hours"] is None
    assert result["unknown_metrics"]["resting_heart_rate"] is None
    assert "睡眠时长：未记录" in result["unknowns"]
    assert "静息心率：未记录" in result["unknowns"]


def test_recovery_no_records_says_none(compact_db):
    from app.services.agent.read_resolver import advisory_read_facts

    result = advisory_read_facts("demo-user", _spec())
    assert result["status"] == "completed"
    assert "没有已保存的训练记录" in result["prefix"]
    assert "35 kg" not in result["prefix"]


def test_recovery_cross_user(compact_db):
    from app.services.agent.read_resolver import advisory_read_facts

    _three_sessions("other-user")
    demo = advisory_read_facts("demo-user", _spec())
    other = advisory_read_facts("other-user", _spec())
    assert "没有已保存的训练记录" in demo["prefix"]
    assert "3 项动作，共 8 组" in other["prefix"]


def test_recovery_legacy_worker_injects_facts(compact_db):
    from app.services.agent.multi_agent_graph import _read_spec_facts_for_task

    _three_sessions()
    facts = _read_spec_facts_for_task("demo-user", {"read_spec": _spec()}, "training_qa")
    assert facts and facts["status"] == "completed"
    assert "3 项动作，共 8 组" in facts["prefix"]


def test_recovery_react_worker_uses_same_resolver(compact_db, monkeypatch):
    from unittest.mock import AsyncMock

    from app.services.agent import react_readonly_workers
    from app.services.agent.agent import agent
    from app.services.context import budget as context_budget_module
    from app.services.rag import online as rag_online
    from app.services.rag import quality as rag_quality

    _three_sessions()

    async def fake_search(*args, **kwargs):
        from types import SimpleNamespace
        return [SimpleNamespace(title="恢复依据", source="test", chunk_id="1",
                                text="恢复依据", score=1.0, metadata={})]

    monkeypatch.setattr(rag_online, "search", fake_search)
    monkeypatch.setattr(rag_quality, "assess_rag_evidence",
                        lambda *args, **kwargs: type("Q", (), {"answerable": True})())

    async def fake_state(context, task_type):
        return {"task_context": []}

    monkeypatch.setattr(react_readonly_workers, "_answer_state", fake_state)
    captured = {}

    def fake_assemble(*, system, query, state, hits):
        captured["state"] = state
        return [{"role": "user", "content": "question"}]

    monkeypatch.setattr(context_budget_module.context_budget, "assemble", fake_assemble)
    monkeypatch.setattr(agent.graph.llm, "complete_message",
                        AsyncMock(return_value={"content": "建议充分休息。"}))

    result = asyncio.run(react_readonly_workers._single_pass_answer(
        {"query": "按今天训练给恢复建议"},
        {"user_id": "demo-user", "task_text": "按今天训练给恢复建议",
         "semantic_scope": {"read_spec": _spec()}},
        "training_qa", "system"))
    assert result.status.value == "completed"
    assert result.summary.startswith("2026-") or "已保存的训练记录" in result.summary
    assert "3 项动作，共 8 组" in result.summary
    assert "睡眠时长：未记录" in result.summary
    assert "静息心率：未记录" in result.summary
    contexts = captured["state"]["task_context"]
    assert any(item.get("read_spec_prefix") for item in contexts if isinstance(item, dict))


def test_recovery_constraint_sibling_does_not_force_legacy():
    from app.services.agent.react_runtime import _has_conversation_only_task

    tasks = [
        {"task_type": "training_qa", "action": "question", "goal_role": "result",
         "payload": {"semantic": {"read_spec": _spec(), "delivery_type": "advice"}}},
        {"task_type": "chitchat", "action": "answer", "goal_role": "constraint"},
    ]
    assert _has_conversation_only_task(tasks) is False


def test_constraint_only_request_still_legacy():
    from app.services.agent.react_runtime import _has_conversation_only_task

    tasks = [{"task_type": "chitchat", "action": "answer", "goal_role": "constraint"}]
    assert _has_conversation_only_task(tasks) is True


def test_recovery_pg(monkeypatch, tmp_path):
    from tests.react_phase_c1_support import isolated
    from app.services.agent.read_resolver import advisory_read_facts

    with isolated(monkeypatch, tmp_path, "postgres"):
        _three_sessions()
        result = advisory_read_facts("demo-user", _spec())
        assert "3 项动作，共 8 组" in result["prefix"]
