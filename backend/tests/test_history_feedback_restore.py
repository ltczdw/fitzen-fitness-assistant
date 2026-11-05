"""P2-4: history must restore run ids, citations and feedback state."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from sqlalchemy import event, select

from app.api.quality import FeedbackInput
from app.db import quality_repository as repo
from app.db.memory_models import MemoryTurn
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.memory import compact
from app.services.memory.compact import CompactMemory


def _memory():
    return CompactMemory(SimpleNamespace())


def _complete_rollout(factory, request, answer, run_id, citations=None):
    tid, _ = compact.begin_turn(request)
    response = {"answer": answer, "intent": "training_qa", "citations": citations or [],
                "run_id": run_id, "message_id": tid + ":assistant",
                "request_id": request.request_id}
    repo.record_run(request, ChatResponse.model_validate(response), {}, status="completed")
    compact.finish_turn(tid, answer, response)
    return tid, response


def _history(factory, conversation_id):
    return asyncio.run(_memory().load_history("demo-user", conversation_id))["messages"]


def test_history_restores_identifiers_citations_and_feedback(compact_db):
    first = ChatRequest(message="第一个问题", conversation_id="history", request_id="history-1")
    tid_one, response_one = _complete_rollout(
        compact_db, first, "第一个回答", "run-one",
        citations=[{"title": "训练指南", "source": "knowledge", "chunk_id": "c1"}])
    repo.feedback(FeedbackInput(run_id="run-one", message_id=response_one["message_id"],
                                rating="down", note=""))
    second = ChatRequest(message="第二个问题", conversation_id="history", request_id="history-2")
    _complete_rollout(compact_db, second, "第二个回答", "run-two")

    messages = _history(compact_db, "history")
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    first_answer, second_answer = messages[1], messages[3]
    assert first_answer["run_id"] == "run-one"
    assert first_answer["message_id"] == tid_one + ":assistant"
    assert first_answer["citations"] == [{"title": "训练指南", "source": "knowledge", "chunk_id": "c1"}]
    assert first_answer["feedback"] == "down"
    assert second_answer["run_id"] == "run-two"
    assert "feedback" not in second_answer


def test_legacy_split_turn_restores_identity_from_diagnostics(compact_db):
    request = ChatRequest(message="旧问题", conversation_id="legacy-history", request_id="legacy-1")
    tid, _ = compact.begin_turn(request)
    with compact_db() as session, session.begin():
        row = session.get(MemoryTurn, tid)
        row.messages = [*row.messages, {"role": "assistant", "content": "旧回答",
                                        "message_id": tid + ":assistant", "turn_id": tid}]
        row.status = "completed"
        row.response = None
    repo.record_run(request, ChatResponse(answer="旧回答", intent="training_qa", run_id="run-legacy",
                                          message_id=tid + ":assistant"), {}, status="completed")

    answer = _history(compact_db, "legacy-history")[1]
    assert answer["run_id"] == "run-legacy"
    assert answer["message_id"] == tid + ":assistant"
    assert answer["citations"] == []

    # A cancelled/failed diagnostic attempt must not donate its run id.
    failed = ChatRequest(message="失败问题", conversation_id="failed-history", request_id="failed-1")
    failed_tid, _ = compact.begin_turn(failed)
    with compact_db() as session, session.begin():
        row = session.get(MemoryTurn, failed_tid)
        row.messages = [*row.messages, {"role": "assistant", "content": "失败回答",
                                        "message_id": failed_tid + ":assistant", "turn_id": failed_tid}]
        row.status = "completed"
        row.response = None
    repo.record_run(failed, ChatResponse(answer="失败回答", intent="error", run_id="run-cancelled",
                                         message_id=failed_tid + ":assistant"), {}, status="cancelled")
    failed_answer = _history(compact_db, "failed-history")[1]
    assert "run_id" not in failed_answer
    assert "feedback" not in failed_answer


def test_unrecoverable_turn_gets_no_synthetic_identifier(compact_db):
    request = ChatRequest(message="没有诊断", conversation_id="no-diagnostics", request_id="no-diag-1")
    tid, _ = compact.begin_turn(request)
    with compact_db() as session, session.begin():
        row = session.get(MemoryTurn, tid)
        row.messages = [*row.messages, {"role": "assistant", "content": "回答",
                                        "message_id": tid + ":assistant", "turn_id": tid}]
        row.status = "completed"
        row.response = None
    answer = _history(compact_db, "no-diagnostics")[1]
    assert "run_id" not in answer
    assert "message_id" in answer and answer["message_id"] == tid + ":assistant"


def test_response_without_completed_diagnostics_keeps_citations_but_no_feedback_id(compact_db):
    request = ChatRequest(message="诊断失败", conversation_id="no-run", request_id="no-run-1")
    tid, _ = compact.begin_turn(request)
    response = {"answer": "回答", "intent": "training_qa",
                "citations": [{"title": "指南", "source": "kb"}],
                "run_id": "run-without-diagnostics", "message_id": tid + ":assistant",
                "request_id": request.request_id}
    compact.finish_turn(tid, "回答", response)  # no ChatRun was recorded
    answer = _history(compact_db, "no-run")[1]
    assert "run_id" not in answer
    assert answer["message_id"] == tid + ":assistant"
    assert answer["citations"] == [{"title": "指南", "source": "kb"}]
    assert "feedback" not in answer


def test_deleted_conversation_never_reappears_in_history(compact_db):
    request = ChatRequest(message="将被删除", conversation_id="deleted-history", request_id="deleted-1")
    _, response = _complete_rollout(compact_db, request, "回答", "run-deleted")
    repo.feedback(FeedbackInput(run_id="run-deleted", message_id=response["message_id"],
                                rating="down", note=""))
    assert compact.delete_memory("deleted-history") == 1
    assert _history(compact_db, "deleted-history") == []


def test_history_uses_a_fixed_number_of_batched_queries(compact_db):
    for index in range(4):
        _complete_rollout(compact_db,
                          ChatRequest(message=f"问题{index}", conversation_id="batch-history",
                                      request_id=f"batch-{index}"),
                          f"回答{index}", f"run-batch-{index}")
    engine = compact_db.kw["bind"]
    statements: list[str] = []

    def before(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", before)
    try:
        one_turn = _history(compact_db, "batch-history")[:2]
        first_count = len(statements)
        statements.clear()
        all_turns = _history(compact_db, "batch-history")
    finally:
        event.remove(engine, "before_cursor_execute", before)
    assert len(one_turn) == 2 and len(all_turns) == 8
    # Three batched queries (turns, diagnostics, feedback) serve one turn or
    # four; nothing is per message.
    assert first_count == len(statements) == 3
