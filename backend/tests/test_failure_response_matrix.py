"""Fault injection across semantics/supervision/tools and HTTP/SSE error exits."""
import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from app.services.llm.client import LLMGatewayError, LLMErrorCategory
from app.services.llm.budget import ModelBudgetError, ModelBudgetUnavailable
from app.services.llm.structured import StructuredOutputError
from app.services.agent.failure_details import exception_failure, transport_failure
from evals.v2.safety import deny_network
from tests.react_phase_a_support import tool_call
from tests.react_phase_b_codex_support import environment, chunks, semantic_outline, semantic_task, sse_body

SOURCE = "请给我饮食建议"
SECRET = "SECRET_PROVIDER_BODY_TOKEN"


def fault(kind):
    if kind.startswith("http_"):
        return LLMGatewayError(SECRET, LLMErrorCategory.UPSTREAM, status_code=int(kind[5:]))
    if kind == "timeout":
        return LLMGatewayError(SECRET, LLMErrorCategory.TRANSPORT, code="model_timeout")
    if kind == "network": return LLMGatewayError(SECRET, LLMErrorCategory.TRANSPORT)
    if kind == "config": return LLMGatewayError(SECRET, LLMErrorCategory.CONFIG)
    if kind == "circuit": return LLMGatewayError(SECRET, LLMErrorCategory.UPSTREAM, code="model_circuit_open")
    if kind == "json": return StructuredOutputError("structured_invalid_json")
    if kind == "reference": return StructuredOutputError("semantic_result_reference_invalid")
    if kind == "tokens": return ModelBudgetError("model_prepare_token_budget_exhausted")
    if kind == "calls": return ModelBudgetError("model_budget_exhausted:request")
    if kind == "storage": return ModelBudgetUnavailable("model_budget_storage_unavailable")
    if kind == "cancel": return ModelBudgetError("model_request_cancelled")
    if kind == "deadline": return ModelBudgetError("model_request_deadline")
    raise AssertionError(kind)


CASES = [
    ("http_400", "model_request_rejected"), ("http_401", "model_auth_failed"),
    ("http_403", "model_auth_failed"), ("http_402", "model_payment_required"),
    ("http_413", "model_request_rejected"), ("http_422", "model_request_rejected"),
    ("http_429", "model_rate_limited"), ("http_500", "model_upstream_failed"),
    ("http_502", "model_upstream_failed"), ("http_503", "model_http_503"),
    ("http_504", "model_timeout"), ("timeout", "model_timeout"),
    ("network", "model_transport_failed"), ("config", "model_not_configured"),
    ("circuit", "model_circuit_open"), ("json", "model_protocol_invalid"),
    ("reference", "semantic_result_reference_invalid"),
    ("tokens", "model_token_budget_exhausted"), ("calls", "model_call_budget_exhausted"),
    ("storage", "model_budget_unavailable"), ("cancel", "model_request_cancelled"),
    ("deadline", "model_request_deadline"),
]


@pytest.mark.parametrize("kind,expected", CASES)
def test_typed_error_classification_never_exposes_raw_values(kind, expected):
    detail = exception_failure(fault(kind))
    assert detail["code"] == expected
    assert detail["message"] and detail["remedy"]
    assert SECRET not in str(detail)
    assert exception_failure(RuntimeError(SECRET)) is None


@pytest.mark.parametrize("transport", ["json", "sse"])
@pytest.mark.parametrize("kind,expected", CASES)
def test_semantic_failure_has_accurate_public_reason_and_replay(tmp_path, monkeypatch, transport, kind, expected):
    with deny_network() as audit, environment(tmp_path, monkeypatch, "semantic-failure") as env:
        env.mock_llm(semantic_error=fault(kind))
        before = env.snapshot()
        result = (env.post("fault-semantic", message=SOURCE) if transport == "json" else
                  sse_body(env.stream("fault-semantic", message=SOURCE)))
        details = result["metrics"].get("failure_details")
        assert details and details[0]["code"] == expected, result
        # Token/call stops already have more specific shared-budget wording.
        if kind not in {"tokens", "calls", "cancel", "deadline"}:
            assert details[0]["message"] in result["answer"], result
        assert result["metrics"]["react_status"] != "complete"
        assert SECRET not in str(result)
        calls = (len(env.llm.calls), len(env.llm.semantic_calls))
        replay = env.post("fault-semantic", message=SOURCE)
        assert replay["answer"] == result["answer"]
        assert replay["metrics"]["failure_details"] == details
        assert (len(env.llm.calls), len(env.llm.semantic_calls)) == calls
        assert env.snapshot() == before
    assert audit == {"blocked_connections": 0, "blocked_dns": 0}


@pytest.mark.parametrize("transport", ["json", "sse"])
@pytest.mark.parametrize("boundary", ["supervisor", "worker"])
@pytest.mark.parametrize("kind,expected", [CASES[i] for i in (1, 6, 9, 11, 12, 13, 15, 19)])
def test_model_failure_provenance_survives_execution_boundary(tmp_path, monkeypatch, transport, boundary, kind, expected):
    with deny_network() as audit, environment(tmp_path, monkeypatch, "worker-failure") as env:
        before = env.snapshot()
        env.mock_rag(chunks(SOURCE))
        llm = env.mock_llm(policy=lambda index, messages: {"role": "assistant", "content": "", "tool_calls": [
            tool_call("diet-" + str(index), "advise_nutrition", {"query": SOURCE})]},
            semantic=lambda text: semantic_outline(text, [dict(semantic_task("nutrition_advice", "question", text), history_requirement="none")]))
        original = llm.complete_message
        async def fail(messages, **kwargs):
            if bool(kwargs.get("tools")) == (boundary == "supervisor"):
                raise fault(kind)
            return await original(messages, **kwargs)
        monkeypatch.setattr(llm, "complete_message", fail)
        result = (env.post("fault-worker", message=SOURCE) if transport == "json" else
                  sse_body(env.stream("fault-worker", message=SOURCE)))
        details = result["metrics"].get("failure_details")
        assert details and details[0]["code"] == expected, result
        assert details[0]["message"] in result["answer"]
        assert result["metrics"]["react_status"] != "complete"
        assert SECRET not in str(result)
        calls = (len(llm.calls), len(llm.semantic_calls))
        replay = env.post("fault-worker", message=SOURCE)
        assert replay["answer"] == result["answer"]
        assert (len(llm.calls), len(llm.semantic_calls)) == calls
        assert env.snapshot() == before
    assert audit == {"blocked_connections": 0, "blocked_dns": 0}


@pytest.mark.parametrize("kind,code,status,retryable", [
    ("database", "storage_unavailable", 503, True),
    ("commit", "commit_unknown", 503, False),
    ("conflict", "request_id_conflict", 409, False),
    ("idempotency", "idempotency_conflict", 409, False),
    ("unknown", "execution_failed", 500, False),
])
def test_http_and_stream_error_envelopes_are_consistent(tmp_path, monkeypatch, kind, code, status, retryable):
    from app.db.storage import StorageUnavailableError, CommitUnknownError, IdempotencyConflict
    from app.services.agent.execution_state import RequestIdConflict
    from app.services.agent import transport as stream_module
    types = {"database": StorageUnavailableError, "commit": CommitUnknownError, "conflict": RequestIdConflict,
             "idempotency": IdempotencyConflict, "unknown": RuntimeError}
    with deny_network() as audit, environment(tmp_path, monkeypatch, "transport-failure") as env:
        before = env.snapshot()
        async def fail(*args, **kwargs):
            exc = types[kind](SECRET)
            exc._request_model_used = True
            raise exc
        monkeypatch.setattr(env.agent, "respond", fail)
        monkeypatch.setattr(stream_module, "execute_request", fail)
        async def call():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=env.app, raise_app_exceptions=False), base_url="http://test") as client:
                return await client.post("/api/chat", json={"message": SOURCE, "request_id": "http-fault"})
        response = asyncio.run(call())
        assert response.status_code == status
        body = response.json()
        assert body["code"] == code and body["retryable"] is retryable
        assert body["model_used"] is True
        events = env.stream("sse-fault", message=SOURCE)
        errors = [e["data"] for e in events if e["event"] == "error"]
        done = [e["data"] for e in events if e["event"] == "done"]
        assert len(errors) == len(done) == 1
        assert errors[0]["code"] == code and errors[0]["http_status"] == status
        assert errors[0]["detail"] == body["detail"]
        assert errors[0]["retryable"] is retryable
        assert done[0]["status"] == "failed" and done[0]["model_used"] is True
        assert SECRET not in str(events) and SECRET not in str(body)
        assert not any(e["event"] == "token" for e in events)
        if kind == "commit": assert "不要直接重复保存" in body["detail"]
        assert env.snapshot() == before
    assert audit == {"blocked_connections": 0, "blocked_dns": 0}


@pytest.mark.parametrize("transport", ["json", "sse"])
@pytest.mark.parametrize("configured", [False, True])
def test_real_disabled_configuration_reports_no_paid_attempt(tmp_path, monkeypatch, transport, configured):
    with environment(tmp_path, monkeypatch, "configuration-disabled") as env:
        env.mock_llm(configured=configured, semantic=False)
        result = (env.post("disabled-model", message=SOURCE) if transport == "json" else
                  sse_body(env.stream("disabled-model", message=SOURCE)))
        assert ("结构化识别接口" if configured else "AI服务尚未配置") in result["answer"], result
        assert result["metrics"]["failure_details"][0]["code"] == ("model_adapter_unavailable" if configured else "model_not_configured")
        assert result["model_used"] is False
        assert not env.llm.calls and not env.llm.semantic_calls


@pytest.mark.parametrize("kind,code", [("tokens", "model_token_budget_exhausted"), ("storage", "model_budget_unavailable")])
@pytest.mark.parametrize("transport", ["json", "sse"])
def test_root_policy_stop_has_same_typed_cause_and_model_accounting(tmp_path, monkeypatch, kind, code, transport):
    from app.services.telemetry import model_attempt
    with environment(tmp_path, monkeypatch, "root-policy") as env:
        before = env.snapshot()
        async def fail(*args, **kwargs):
            with model_attempt("synthetic-failure") as attempt:
                attempt.observe({"usage": {"total_tokens": 2}})
                raise fault(kind)
        monkeypatch.setattr(env.agent, "_respond_impl", fail)
        result = (env.post("root-failure", message=SOURCE) if transport == "json" else
                  sse_body(env.stream("root-failure", message=SOURCE)))
        assert result["metrics"]["failure_details"][0]["code"] == code
        assert result["model_used"] is True
        assert result["metrics"]["react_status"] == "limited"
        assert "已完成的记录保留" in result["answer"]
        assert env.snapshot() == before


@pytest.mark.parametrize("kind", ["unknown", "commit"])
@pytest.mark.parametrize("transport", ["json", "sse"])
def test_real_runtime_keeps_paid_attempt_flag_on_terminal_failure(tmp_path, monkeypatch, kind, transport):
    from app.db.storage import CommitUnknownError
    from app.services.telemetry import model_attempt
    with environment(tmp_path, monkeypatch, "real-terminal-failure") as env:
        before = env.snapshot()
        async def fail(*args, **kwargs):
            with model_attempt("synthetic-paid-attempt") as attempt:
                attempt.observe({"usage": {"total_tokens": 2}})
            raise (CommitUnknownError(SECRET) if kind == "commit" else RuntimeError(SECRET))
        monkeypatch.setattr(env.agent, "_respond_impl", fail)
        if transport == "sse":
            events = env.stream("terminal-failure", message=SOURCE)
            result = next(e["data"] for e in events if e["event"] == "done")
            assert result["status"] == "failed"
        else:
            async def call():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=env.app, raise_app_exceptions=False), base_url="http://test") as client:
                    return await client.post("/api/chat", json={"message": SOURCE, "request_id": "terminal-failure"})
            response = asyncio.run(call())
            assert response.status_code == (503 if kind == "commit" else 500)
            result = response.json()
        assert result["model_used"] is True
        assert SECRET not in str(result)
        assert env.snapshot() == before


@pytest.mark.parametrize("assessed", [False, True])
def test_planner_failure_retains_only_source_bound_urgent_judgment(tmp_path, monkeypatch, assessed):
    from app.services.agent import react_graph
    from app.services.agent.react_graph import ReactDependencies, invoke_react, URGENT_SAFETY_ANSWER
    from app.services.agent import task_planner
    source = "胸痛并呼吸困难，请帮我处理"
    with environment(tmp_path, monkeypatch, "urgent-planner-failure") as env:
        llm = env.mock_llm(configured=assessed, semantic=lambda text: semantic_outline(text,
            [dict(semantic_task("safety_triage", "question", text), history_requirement="none")],
            safety={"level": "urgent", "evidence": text, "categories": ["current_symptom"]}))
        def fail(*args, **kwargs): raise RuntimeError("candidate planner fault")
        monkeypatch.setattr(task_planner, "validated_semantic_tasks", fail)
        monkeypatch.setattr(react_graph, "_rule_candidates", lambda text: ([], "synthetic_candidate_failure"))
        executor = AsyncMock(side_effect=AssertionError("urgent/unknown safety must not dispatch"))
        result = asyncio.run(invoke_react(ReactDependencies(llm=llm, tool_executor=executor), source))
        assert executor.await_count == 0 and not llm.calls
        if assessed:
            assert result["final_answer"] == URGENT_SAFETY_ANSWER
        else:
            assert result["termination_reason"] == "safety_assessment_unavailable"
            assert "未完成安全语义判断" in result["final_answer"]


def test_ordinary_process_restart_reads_dispatch_intent_without_marking_success(tmp_path, monkeypatch):
    # Ordinary component verification; does not claim the historical formal
    # runner's authorization/binding gate has passed.
    import json, subprocess, sys
    from tests.react_phase_c1_support import LedgerHarness
    with environment(tmp_path, monkeypatch, "ordinary-recovery") as env:
        ledger = LedgerHarness(env.engine)
        ledger.freeze()
        ledger.reserve()
        operation = ledger.operation()
        ledger.call("mark_dispatch_intent", operation["op_id"])
        before = env.snapshot()
        packet = {"url": env.engine.url.render_as_string(hide_password=False), "ids": list(ledger.ids)}
        script = """
import json,sys
from sqlalchemy.orm import sessionmaker
from tests.quality_isolation import checked_engine
from app.services.agent.execution_ledger_store import ExecutionLedgerStore
packet=json.load(sys.stdin)
assert packet['url'].startswith('sqlite:///')
engine=checked_engine(packet['url'])
try:
 state=ExecutionLedgerStore(sessionmaker(bind=engine,expire_on_commit=False)).load_recovery_state(*packet['ids'])
 print(json.dumps({'replay_allowed':state['replay_allowed'],'pending':[
  {'op_id':row['op_id'],'status':row['status']} for row in state['pending_verification']]}))
finally:engine.dispose()
"""
        child = subprocess.run([sys.executable, "-c", script], input=json.dumps(packet), capture_output=True,
                               text=True, encoding="utf-8", cwd=str(__import__('pathlib').Path(__file__).resolve().parents[1]), timeout=20)
        assert child.returncode == 0, "isolated ordinary recovery child failed"
        result = json.loads(child.stdout)
        assert result["replay_allowed"] is False
        assert result["pending"] == [{"op_id": operation["op_id"], "status": "dispatch_intent"}]
        assert env.snapshot() == before


@pytest.mark.parametrize("status", ["limited", "rejected", "restricted", "partial"])
@pytest.mark.parametrize("failed_turns", [1, 12])
def test_worker_failure_cannot_replace_valid_result_even_outside_recent_window(tmp_path, monkeypatch, status, failed_turns):
    from app.core import clock
    from datetime import timedelta
    from app.schemas.chat import ChatRequest
    from app.services.memory import compact
    from app.services.memory.recent_window import load_window
    from app.services.agent.result_reference import result_sources
    with environment(tmp_path, monkeypatch, "failed-result-filter") as env:
        monkeypatch.setattr(__import__('app.core.config', fromlist=['settings']).settings, "memory_short_window_turns", 10)
        base = clock.utc_now()
        good, _ = compact.begin_turn(ChatRequest(message=SOURCE, conversation_id="demo-conversation"))
        compact.finish_turn(good, "VALID_DIET_RESULT", {"answer": "VALID_DIET_RESULT", "intent": "nutrition_advice",
                                                      "metrics": {"execution_mode": "react_readonly", "react_status": "complete"}})
        failed_ids = []
        for n in range(failed_turns):
            env.mp.setattr(clock, "utc_now", lambda n=n: base + timedelta(seconds=n + 1))
            tid, _ = compact.begin_turn(ChatRequest(message=SOURCE, conversation_id="demo-conversation"))
            compact.finish_turn(tid, "FAILED_WORKER_REPLY", {"answer": "FAILED_WORKER_REPLY", "intent": "nutrition_advice",
                "metrics": {"execution_mode": "react_readonly", "react_status": status}})
            failed_ids.append(tid)
        sources = result_sources({"recent_result_dialogue": compact.latest_result_dialogue("demo-conversation", 0)})
        assert sources[0]["message_id"] == good + ":assistant"
        memory = load_window("demo-conversation", 0)
        assert result_sources(memory)[0]["message_id"] == good + ":assistant"
        assert any(m["content"] == "FAILED_WORKER_REPLY" for m in memory["short_memory"])
        assert all(s["message_id"] not in {tid + ":assistant" for tid in failed_ids} for s in result_sources(memory))


def test_successful_goal_does_not_keep_previous_supervisor_failure_in_metrics():
    from app.services.agent.failure_details import read_failure_details
    state = {"status": "complete", "model_failure_code": "model_http_503",
             "intent_candidates": [{"id": "goal", "task_type": "nutrition_advice", "action": "question", "text": SOURCE}],
             "covered_goals": ["goal"], "goal_results": {"goal": {"status": "completed", "error": ""}}}
    assert read_failure_details(state) == []
