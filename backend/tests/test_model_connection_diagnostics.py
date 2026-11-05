"""Tests for model connection diagnostics."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from app.services.llm import diagnostics
from app.services.llm.client import LLMGatewayError, LLMErrorCategory, _observed_post, _record_retry


class _Response:
    status_code = 200
    http_version = "HTTP/1.1"


class _Client:
    def __init__(self, *, error=None, response=None):
        self.error, self.response = error, response

    async def post(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        return self.response


def _run(client):
    return asyncio.run(_observed_post(client, "https://mock.invalid/v1/chat", {}, {}, 0))


def test_connect_failure_records_phase_and_no_request_content():
    diagnostics.begin()
    with pytest.raises(httpx.ConnectError):
        _run(_Client(error=httpx.ConnectError("connection refused")))
    events = diagnostics.snapshot()
    assert [item["phase"] for item in events] == ["send", "connect"]
    assert events[1]["error_type"] == "ConnectError"
    assert "duration_ms" in events[1]
    assert all(set(item) <= diagnostics._ALLOWED for item in events)
    assert not any(key in item for item in events for key in ("url", "headers", "payload", "body"))


def test_post_send_disconnect_is_a_response_phase_error():
    diagnostics.begin()
    with pytest.raises(httpx.RemoteProtocolError):
        _run(_Client(error=httpx.RemoteProtocolError("peer closed connection")))
    events = diagnostics.snapshot()
    assert events[-1]["phase"] == "response"
    assert events[-1]["error_type"] == "RemoteProtocolError"


def test_success_records_status_version_and_latency():
    diagnostics.begin()
    response = _run(_Client(response=_Response()))
    assert response is not None
    events = diagnostics.snapshot()
    assert [item["phase"] for item in events] == ["send", "response"]
    assert events[1]["http_status"] == 200
    assert events[1]["http_version"] == "HTTP/1.1"
    assert events[1]["response_received"] is True


@pytest.mark.parametrize("exc, phase", [
    (httpx.ReadTimeout("timeout"), "timeout"),
    (httpx.ReadError("read failed"), "response"),
    (httpx.HTTPStatusError("429", request=httpx.Request("POST", "https://mock.invalid"),
                           response=httpx.Response(429)), "http_status"),
])
def test_exception_phase_mapping(exc, phase):
    assert diagnostics.phase_for(exc) == phase


def test_retry_decision_is_recorded_without_request_content():
    diagnostics.begin()
    _record_retry(0, LLMGatewayError("LLM 网络错误", LLMErrorCategory.TRANSPORT, retryable=True), True)
    _record_retry(1, LLMGatewayError("HTTP 429", LLMErrorCategory.UPSTREAM, retryable=False, status_code=429), False)
    events = diagnostics.snapshot()
    assert events[0]["retry_decision"] == "retry" and events[0]["category"] == "transport"
    assert events[1]["retry_decision"] == "stop" and events[1]["status_code"] == 429
    assert all("message" not in item for item in events)


class _StreamResponse:
    status_code = 200
    http_version = "HTTP/1.1"

    def __init__(self, *, partial: bool):
        self.partial = partial

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        yield 'data: {"choices": [{"delta": {"content": "a"}}]}'
        if self.partial:
            raise httpx.RemoteProtocolError("peer closed mid-stream")


class _StreamClient:
    def __init__(self, *, partial: bool = True, error=None):
        self.partial, self.error = partial, error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        return _StreamResponse(partial=self.partial)


class _Paid:
    class _Scope:
        @staticmethod
        def remaining():
            return 30.0

    scope = _Scope()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def observe(self, data):
        return None


def _stream(monkeypatch, client):
    from app.services.llm import client as llm_client

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", lambda *a, **kw: client)
    monkeypatch.setattr(llm_client, "async_paid_attempt", lambda *a, **kw: _Paid())
    test_client = llm_client.LLMClient()
    monkeypatch.setattr(test_client, "_endpoint", lambda: ("https://mock.invalid/v1/chat", {}))
    monkeypatch.setattr(test_client, "_payload", lambda *a, **kw: {})
    return test_client


def test_partial_stream_records_response_phase_and_no_retry_at_once_layer(monkeypatch):
    diagnostics.begin()
    client = _stream(monkeypatch, _StreamClient(partial=True))

    async def consume():
        return [item async for item in client._stream_once([], 0, 0)]

    with pytest.raises(httpx.RemoteProtocolError):
        asyncio.run(consume())
    events = diagnostics.snapshot()
    assert [item["phase"] for item in events] == ["send", "response", "response"]
    assert events[-1]["streamed"] is True
    assert events[-1]["usage_available"] is False


def test_timeout_before_stream_response_records_timeout(monkeypatch):
    diagnostics.begin()
    client = _stream(monkeypatch, _StreamClient(error=httpx.ReadTimeout("timeout")))

    async def consume():
        return [item async for item in client._stream_once([], 0, 0)]

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(consume())
    events = diagnostics.snapshot()
    assert [item["phase"] for item in events] == ["send", "timeout"]
