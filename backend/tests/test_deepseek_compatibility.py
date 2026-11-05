"""Provider compatibility, with all inference requests mocked."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import httpx
import pytest

from app.core.config import Settings, settings
from app.services.llm.client import LLMClient, LLMGatewayError, LLMErrorCategory
from app.services.agent.semantics import request_understanding, understand, understanding_unavailable_reply
from app.services.telemetry import RequestMetrics, active_metrics


@pytest.mark.parametrize("base,model,expected", [
    ("https://api.deepseek.com", "deepseek-v4-flash", "deepseek-flash"),
    ("https://api.deepseek.com/v1", "deepseek-v4-flash", "deepseek-flash"),
    ("https://api.deepseek.com", "deepseek-v4-pro", "deepseek-v4-pro"),
    ("https://other.invalid/v1", "deepseek-v4-flash", "deepseek-v4-flash"),
    ("https://api.deepseek.com.other.invalid", "deepseek-v4-flash", "deepseek-v4-flash"),
])
def test_flash_alias_only_changes_for_the_official_host(base, model, expected):
    assert Settings(_env_file=None, llm_base_url=base, llm_model=model).llm_model == expected


@pytest.mark.parametrize("choice", [{"type": "function", "function": {"name": "return_request_understanding"}}, "required"])
def test_forced_deepseek_tools_disable_incompatible_default_thinking(monkeypatch, choice):
    monkeypatch.setattr(settings, "llm_base_url", "https://api.deepseek.com/v1")
    tool = {"type": "function", "function": {"name": "return_request_understanding", "parameters": {"type": "object"}}}
    payload = LLMClient._payload([{"role": "user", "content": "synthetic"}], 0, False, tools=[tool], tool_choice=choice)
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["tool_choice"] == choice and payload["tools"] == [tool]


@pytest.mark.parametrize("base,tools,choice", [
    ("https://api.deepseek.com", [{"type": "function"}], "auto"),
    ("https://other.invalid", [{"type": "function"}], "required"),
    ("https://api.deepseek.com.other.invalid", [{"type": "function"}], "required"),
])
def test_auto_tools_or_other_provider_calls_do_not_receive_the_extension(monkeypatch, base, tools, choice):
    monkeypatch.setattr(settings, "llm_base_url", base)
    payload = LLMClient._payload([{"role": "user", "content": "synthetic"}], 0, False, tools=tools, tool_choice=choice)
    assert "thinking" not in payload


@pytest.mark.parametrize("category,status,expected", [
    (LLMErrorCategory.UPSTREAM, 400, "HTTP 400"),
    (LLMErrorCategory.UPSTREAM, 401, "鉴权失败"),
    (LLMErrorCategory.UPSTREAM, 403, "鉴权失败"),
    (LLMErrorCategory.UPSTREAM, 429, "请求受限"),
    (LLMErrorCategory.TRANSPORT, None, "连接AI服务失败"),
    (LLMErrorCategory.PROTOCOL, None, "未通过校验"),
])
def test_semantic_failures_keep_the_real_category_without_logging_private_text(caplog, category, status, expected):
    secret = "DO-NOT-LOG-PRIVATE-PROVIDER-BODY"
    model = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(
        side_effect=LLMGatewayError(secret, category, status_code=status)))
    metrics = RequestMetrics()
    token = active_metrics.set(metrics)
    cache = request_understanding.set({})
    try:
        assert asyncio.run(understand(model, "synthetic private query")) is None
        assert expected in understanding_unavailable_reply()
        assert "手动表单" not in understanding_unavailable_reply()
        assert "未保存" not in understanding_unavailable_reply()
        assert "未保存" in understanding_unavailable_reply(record_mode=True)
        from app.schemas.chat import ChatRequest
        from app.services.agent.agent import FitnessAgent
        response = FitnessAgent._safety_unavailable_response(
            ChatRequest(message="synthetic private query", mode="react_write"), "unknown")
        assert expected in response.answer
        assert response.safety_level == "unknown"
        assert response.intent == "execution_limited"
        assert response.metrics["execution_mode"] == "limited"
        assert not response.metrics.get("receipt_refs")
        assert secret not in response.answer and "synthetic private query" not in response.answer
        assert metrics.failure_reasons
        assert secret not in caplog.text and "synthetic private query" not in caplog.text
    finally:
        request_understanding.reset(cache)
        active_metrics.reset(token)


@pytest.mark.parametrize("backend", ["sqlite"])
@pytest.mark.parametrize("transport", ["json", "sse"])
def test_real_http_boundary_accepts_canonical_flash_and_forced_tool_mode(c3_env, monkeypatch, backend, transport):
    from uuid import uuid4
    from sqlalchemy import text
    from tests.react_phase_c2_3_support import C3HTTPHarness, mock_provider_transport
    from test_chat_write_pilot import semantic, native, tool
    source = "帮我记录今天卧推60kg4组8次"
    fragment = "卧推60kg4组8次"
    responses = [semantic(source, fragment, "workout_record"), tool(source, fragment, "record_workout"),
        native({"records": [{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8, "inferred_fields": [], "record_date": None}]})]
    def guard(response):
        def handle(request):
            body = json.loads(request.content)
            if body["model"] != "deepseek-flash":
                return httpx.Response(400, json={"error": {"code": "model_not_found"}})
            if isinstance(body.get("tool_choice"), dict) and body.get("thinking") != {"type": "disabled"}:
                return httpx.Response(400, json={"error": {"code": "incompatible_tool_choice"}})
            return httpx.Response(200, json=response)
        return handle
    seen = mock_provider_transport(monkeypatch, [guard(r) for r in responses])
    config = Settings(_env_file=None, llm_base_url="https://api.deepseek.com", llm_model="deepseek-v4-flash")
    monkeypatch.setattr(settings, "llm_base_url", config.llm_base_url)
    monkeypatch.setattr(settings, "llm_model", config.llm_model)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    rid = str(uuid4())
    if transport == "json":
        status, payload = harness.post_json(source, request_id=rid)
    else:
        status, events, _ = harness.stream(source, request_id=rid)
        payload = events[-1]
    assert status == 200 and payload["metrics"]["react_write"] == "completed", payload
    assert payload["metrics"]["receipt_refs"] and len(seen) == 3
    with c3_env.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM workout_session_sets")).scalar_one() == 4
    replay_status, replay = harness.post_json(source, request_id=rid)
    assert replay_status == 200 and replay["metrics"]["receipt_refs"] == payload["metrics"]["receipt_refs"]
    assert len(seen) == 3


@pytest.mark.parametrize("backend", ["sqlite"])
def test_http_400_is_visible_in_the_real_response_without_any_business_write(c3_env, monkeypatch, backend):
    from tests.react_phase_c2_3_support import C3HTTPHarness, mock_provider_transport, all_business_rows
    seen = mock_provider_transport(monkeypatch, [lambda _: httpx.Response(400, json={"error": {"message": "private-provider-body"}})])
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    status, payload = harness.post_json("帮我记录今天卧推60kg4组8次")
    assert status == 200 and "拒绝了请求参数" in payload["answer"]
    assert payload['metrics']['failure_details'][0]['code'] == 'model_request_rejected'
    assert "private-provider-body" not in json.dumps(payload)
    assert "semantic_model_http_400" in payload["metrics"]["failure_reasons"]
    assert not payload["metrics"].get("receipt_refs") and all_business_rows(c3_env) == []
    assert len(seen) == 1


_FAILURE_KINDS = [
    ("http_400", 400, "semantic_model_http_400", "拒绝了请求参数"),
    ("http_401", 401, "semantic_model_http_401", "鉴权失败"),
    ("http_403", 403, "semantic_model_http_403", "鉴权失败"),
    ("http_429", 429, "semantic_model_http_429", "请求受限"),
    ("transport", None, "semantic_model_transport", "连接AI服务失败"),
    ("protocol", None, "semantic_model_protocol", "校验"),
    ("missing_safety", None, "semantic_model_protocol", "校验"),
]


def _missing_safety_wire():
    value = {"tasks": [{"task_type": "workout_record", "action": "record",
                        "text": "帮我记录今天卧推60kg4组8次",
                        "is_negated": False, "is_hypothetical": False,
                        "is_third_party": False, "history_requirement": "none"}]}
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
        "id": "missing-safety", "type": "function", "function": {
            "name": "return_request_understanding",
            "arguments": json.dumps(value, ensure_ascii=False)}}]}}],
        "usage": {"prompt_tokens": 6, "completion_tokens": 4}}


@pytest.mark.parametrize("kind,http_status,reason,expected", _FAILURE_KINDS,
                         ids=[item[0] for item in _FAILURE_KINDS])
@pytest.mark.parametrize("transport", ["json", "sse"])
@pytest.mark.parametrize("backend", ["sqlite"])
def test_http_failure_categories_are_safe_and_fail_closed(c3_env, monkeypatch, kind,
                                                           http_status, reason, expected,
                                                           transport, backend):
    from tests.react_phase_c2_3_support import C3HTTPHarness, mock_provider_transport, all_business_rows

    def handler(_request):
        if kind == "transport":
            raise httpx.ConnectError("private-provider-body")
        if kind == "protocol":
            return httpx.Response(200, json={"choices": [{"message": {
                "role": "assistant", "content": "no structured tool call"}}]})
        if kind == "missing_safety":
            return httpx.Response(200, json=_missing_safety_wire())
        return httpx.Response(http_status, json={"error": {"message": "private-provider-body"}})

    mock_provider_transport(monkeypatch, [handler])
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    source = "帮我记录今天卧推60kg4组8次"
    if transport == "json":
        status, payload = harness.post_json(source, request_id="failure-" + kind)
        answer = payload["answer"]
    else:
        status, events, _ = harness.stream(source, request_id="failure-" + kind)
        assert status == 200 and events
        payload = events[-1]
        answer = "".join(str(frame.get("content") or "") for frame in events)
    # A retryable transport/protocol/429 failure may legitimately exhaust the
    # real per-request budget before the semantic layer classifies it. Both
    # outcomes are safe: the real category when available, otherwise the
    # explicit budget-limited copy. Neither may claim a structured-output error.
    details = payload['metrics'].get('failure_details')
    assert details
    expected_code = {'http_400':'model_request_rejected','http_401':'model_auth_failed',
        'http_403':'model_auth_failed','http_429':'model_rate_limited',
        'transport':'model_transport_failed','protocol':'model_protocol_invalid',
        'missing_safety':'model_protocol_invalid'}[kind]
    if details[0]['code'] == expected_code:
        assert expected in answer
    else:
        assert details[0]['code'] in {'model_token_budget_exhausted','model_call_budget_exhausted'}
        assert 'Token' in answer if details[0]['code'] == 'model_token_budget_exhausted' else '调用' in answer
    assert "结构化输出失败" not in answer
    assert "private-provider-body" not in json.dumps(payload)
    assert not payload["metrics"].get("receipt_refs")
    assert all_business_rows(c3_env) == []
    if kind == "missing_safety":
        assert payload["metrics"]["semantic_failure_code"] == "semantic_safety_missing"
