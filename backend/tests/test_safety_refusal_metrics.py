"""P2-2: a correct safety refusal is not a degraded request or a failure case."""
from __future__ import annotations

import asyncio

from sqlalchemy import func, select

from app.db import quality_repository as repo
from app.db.quality_models import FailureCase
from app.schemas.chat import ChatRequest
from app.services.telemetry import RequestMetrics, active_metrics, mark_degraded, mark_safety_refusal
from test_retry_operations import RetryScenario, _workouts


def test_request_metrics_separate_refusals_from_failures():
    metrics = RequestMetrics()
    token = active_metrics.set(metrics)
    try:
        mark_safety_refusal("rejected_unauthorized")
        summary = metrics.summary()
        assert summary["safety_refusals"] == ["rejected_unauthorized"]
        assert summary["degraded"] is False
        assert summary["failure_reasons"] == []
        mark_degraded("task_error", failure=True)
        summary = metrics.summary()
        assert summary["degraded"] is True
        assert summary["failure_reasons"] == ["task_error"]
        assert summary["safety_refusals"] == ["rejected_unauthorized"]
    finally:
        active_metrics.reset(token)


def _failure_case_count(factory) -> int:
    with factory() as session:
        return session.scalar(select(func.count()).select_from(FailureCase))


def _model_plan(message: str, *, negated: bool = False) -> dict:
    return {"tasks": [{
        "task_type": "workout_record", "action": "record", "text": message,
        "is_negated": negated, "is_hypothetical": False, "is_third_party": False,
    }]}


def test_correct_refusal_stays_out_of_degradation_and_failure_cases(compact_db, monkeypatch):
    message = "别把卧推记录一下"
    scenario = RetryScenario(monkeypatch, compact_db, workout_records=_workouts,
                             parsed_input=_model_plan(message, negated=True))
    request = ChatRequest(message=message, conversation_id="refusal", request_id="refusal-1")
    response = asyncio.run(scenario.respond(request))

    assert "未保存" in response.answer
    assert response.metrics["degraded"] is False
    assert response.metrics["failure_reasons"] == []
    assert response.metrics["safety_refusals"] == ["rejected_unauthorized"]
    assert not scenario.saves
    assert _failure_case_count(compact_db) == 0
    assert repo.summary()["degraded_rate"] == 0.0


def test_real_tool_failure_is_still_recorded(compact_db, monkeypatch):
    class ExplodingScenario(RetryScenario):
        async def _call_mcp(self, name, arguments, **kwargs):
            if name == "save_workout_record":
                raise RuntimeError("tool exploded")
            return await super()._call_mcp(name, arguments, **kwargs)

    message = "记录深蹲40kg3组8次"
    scenario = ExplodingScenario(monkeypatch, compact_db, workout_records=_workouts,
                                 parsed_input=_model_plan(message))
    request = ChatRequest(message=message, conversation_id="tool-failure",
                          request_id="tool-failure-1")
    response = asyncio.run(scenario.respond(request))

    assert response.metrics["degraded"] is True
    assert "task_error" in response.metrics["failure_reasons"]
    assert response.metrics["safety_refusals"] == []
    assert _failure_case_count(compact_db) == 1


def test_preview_script_selects_only_refusal_only_cases(compact_db):
    from app.schemas.chat import ChatResponse
    from scripts.audit_safety_refusals import pending_refusal_cases

    refused = ChatRequest(message="别记录", conversation_id="preview", request_id="preview-1")
    repo.record_run(refused, ChatResponse(answer="未保存", intent="training_qa", run_id="run-refused",
                                          message_id="preview-1:assistant"),
                    {"failure_reasons": ["task_rejected_unauthorized"], "degraded": True})
    genuine = ChatRequest(message="真实故障", conversation_id="preview", request_id="preview-2")
    repo.record_run(genuine, ChatResponse(answer="失败", intent="error", run_id="run-genuine",
                                          message_id="preview-2:assistant"),
                    {"failure_reasons": ["task_rejected_unauthorized", "task_error"], "degraded": True})

    cases, total = pending_refusal_cases(10)
    assert total == 1
    assert [case["run_id"] for case in cases] == ["run-refused"]
    # The genuine mixed failure case is left untouched for human review.
    assert repo.list_cases()["total"] == 2


def test_needs_confirmation_is_not_degraded(compact_db, monkeypatch):
    message = "记录深蹲"
    scenario = RetryScenario(monkeypatch, compact_db, workout_records=lambda _message: [None],
                             parsed_input=_model_plan(message))
    request = ChatRequest(message=message, conversation_id="confirm", request_id="confirm-1")
    response = asyncio.run(scenario.respond(request))
    assert "缺少" in response.answer
    assert response.metrics["degraded"] is False
    assert response.metrics["failure_reasons"] == []
