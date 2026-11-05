"""P1-3: completion state and the final response are one atomic commit.

The reproduced defect completed a turn with an answer string and only wrote the
full response later, so an interruption in between left a "completed" turn that
could not be replayed and whose cached answer could drift from the history.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from app.db.memory_models import MemoryTurn
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agent.agent import FitnessAgent
from app.services.memory import compact
from test_retry_operations import RetryScenario, _workouts


def _stored_turn(compact_db, request_id):
    with compact_db() as session:
        return session.scalar(select(MemoryTurn).where(MemoryTurn.request_id == request_id))


def _responses_are_consistent(response, turn_assistant_message):
    return (response.answer == turn_assistant_message["content"]
            and response.message_id == turn_assistant_message["message_id"])


def test_finish_turn_commits_history_completion_and_response_together(compact_db):
    request = ChatRequest(message="记录卧推", conversation_id="commit", request_id="commit-1")
    tid, cached = compact.begin_turn(request)
    assert cached is None
    response = {"answer": "已记录卧推。", "intent": "workout_record", "run_id": "run-1",
                "message_id": tid + ":assistant", "request_id": request.request_id}
    compact.finish_turn(tid, "已记录卧推。", response)
    with compact_db() as session:
        row = session.get(MemoryTurn, tid)
        assert row.status == "completed"
        assert row.response == response
        assert [message["role"] for message in row.messages] == ["user", "assistant"]
        assert row.messages[1]["message_id"] == tid + ":assistant"
        assert row.completed_at is not None


def test_finish_turn_rolls_back_everything_when_a_later_step_fails(compact_db, monkeypatch):
    request = ChatRequest(message="记录深蹲", conversation_id="rollback", request_id="rollback-1")
    tid, _ = compact.begin_turn(request)

    def boom(session, turn):
        raise RuntimeError("fixture failure after status write")

    with monkeypatch.context() as patch:
        patch.setattr("app.services.memory.wiki.materialize", boom)
        with pytest.raises(RuntimeError):
            compact.finish_turn(tid, "已记录。", {"answer": "已记录。", "intent": "workout_record",
                                                 "run_id": "run-x", "message_id": tid + ":assistant"})
    with compact_db() as session:
        row = session.get(MemoryTurn, tid)
        assert row.status != "completed"
        assert row.response is None
        assert [message["role"] for message in row.messages] == ["user"]
    # The turn can still be executed and completed afterwards.
    compact.finish_turn(tid, "已记录。", {"answer": "已记录。", "intent": "workout_record",
                                          "run_id": "run-x", "message_id": tid + ":assistant"})
    with compact_db() as session:
        assert session.get(MemoryTurn, tid).status == "completed"


def _legacy_split_commit(compact_db, request, answer):
    """Reproduce the historical shape: completed + answer + response=None."""
    tid, _ = compact.begin_turn(request)
    with compact_db() as session, session.begin():
        row = session.get(MemoryTurn, tid)
        row.messages = [*row.messages, {"role": "assistant", "content": answer,
                                        "message_id": tid + ":assistant", "turn_id": tid}]
        row.status = "completed"
        row.response = None
    return tid


def test_completed_turn_without_response_rebuilds_from_diagnostics(compact_db):
    from app.db import quality_repository as repo
    request = ChatRequest(message="如何热身", conversation_id="replay", request_id="replay-1")
    tid = _legacy_split_commit(compact_db, request, "第一次的回答")
    repo.record_run(request, ChatResponse(answer="第一次的回答", intent="training_qa",
                                          run_id="run-original", message_id=tid + ":assistant"),
                    {}, status="completed")

    again_tid, cached = compact.begin_turn(request)
    assert again_tid == tid
    assert cached["run_id"] == "run-original"
    assert cached["answer"] == "第一次的回答"


def test_completed_turn_without_response_or_diagnostics_reports_without_rerunning(compact_db):
    request = ChatRequest(message="如何热身", conversation_id="replay", request_id="replay-2")
    _legacy_split_commit(compact_db, request, "第一次的回答")

    agent = FitnessAgent()
    agent._run_planner_shadow = AsyncMock()
    agent._respond_impl = AsyncMock(side_effect=AssertionError("business must not rerun"))
    response = asyncio.run(agent.respond(request))
    agent._respond_impl.assert_not_awaited()
    assert response.intent == "replay_unavailable"
    assert "没有成功保存" in response.answer


def test_completed_but_cancelled_diagnostics_are_not_used_as_a_success_answer(compact_db):
    from app.db import quality_repository as repo
    request = ChatRequest(message="记录卧推", conversation_id="replay", request_id="replay-3")
    tid = _legacy_split_commit(compact_db, request, "第一次的回答")
    repo.record_run(request, ChatResponse(answer="被取消的中途回答", intent="error",
                                          run_id="run-cancelled", message_id=tid + ":assistant"),
                    {}, status="cancelled")

    _, cached = compact.begin_turn(request)
    assert cached["intent"] == "replay_unavailable"
    assert cached["run_id"] is None


def _authorized_workout_scenario(monkeypatch, factory, request):
    # The current application requires a model-authorized task. This fixture
    # tests final-commit recovery, not the removed rule-only write fallback.
    parsed = {"tasks": [{"task_type": "workout_record", "action": "record", "text": request.message,
                         "is_negated": False, "is_hypothetical": False, "is_third_party": False,
                         "workouts": _workouts(request.message)}]}
    return RetryScenario(monkeypatch, factory, parsed_input=parsed, workout_records=_workouts)


def test_crash_before_final_commit_retries_without_repeating_writes(compact_db, monkeypatch):
    request = ChatRequest(message="记录深蹲40kg3组8次", conversation_id="crash", request_id="crash-1")
    with monkeypatch.context() as patch:
        patch.setattr("app.services.agent.agent.finish_turn",
                      lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("commit interrupted")))
        first = _authorized_workout_scenario(monkeypatch, compact_db, request)
        response = asyncio.run(first.respond(request))
    assert response.memory_status == "degraded"
    stored = _stored_turn(compact_db, request.request_id)
    assert stored.status != "completed" and stored.response is None

    second = _authorized_workout_scenario(monkeypatch, compact_db, request)
    replayed = asyncio.run(second.respond(request))
    assert replayed.replayed is False
    assert "已记录" in replayed.answer
    with compact_db() as session:
        from sqlalchemy import text
        count = session.execute(text("SELECT COUNT(*) FROM workout_records")).scalar()
    assert count == 1
    assert [record["exercise"] for record in second.saves] == []
    stored = _stored_turn(compact_db, request.request_id)
    assert stored.status == "completed" and stored.response["answer"] == replayed.answer


def test_cancellation_never_marks_a_completed_response(compact_db):
    request = ChatRequest(message="记录卧推", conversation_id="cancel", request_id="cancel-1")
    agent = FitnessAgent()
    agent._run_planner_shadow = AsyncMock()

    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    agent._respond_impl = cancelled
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(agent.respond(request))
    stored = _stored_turn(compact_db, request.request_id)
    assert stored.status == "failed" and stored.response is None


def test_diagnostics_failure_does_not_block_commit_or_replay(compact_db, monkeypatch):
    request = ChatRequest(message="记录深蹲40kg3组8次", conversation_id="diagnostics", request_id="diag-1")
    monkeypatch.setattr("app.db.quality_repository.record_run",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("diagnostics down")))
    first = _authorized_workout_scenario(monkeypatch, compact_db, request)
    response = asyncio.run(first.respond(request))
    assert response.diagnostics_status == "degraded"
    assert "已记录" in response.answer

    stored = _stored_turn(compact_db, request.request_id)
    assert stored.status == "completed"
    second = _authorized_workout_scenario(monkeypatch, compact_db, request)
    replay = asyncio.run(second.respond(request))
    assert replay.replayed is True
    assert replay.answer == response.answer
    assert replay.run_id == response.run_id
    assert replay.message_id == stored.messages[1]["message_id"]
    assert _responses_are_consistent(replay, stored.messages[1])


def test_answer_only_append_cannot_split_commit(compact_db):
    memory = compact.CompactMemory(SimpleNamespace())
    with pytest.raises(RuntimeError):
        asyncio.run(memory.append_turn("demo-user", "conversation", "问题", "回答"))
