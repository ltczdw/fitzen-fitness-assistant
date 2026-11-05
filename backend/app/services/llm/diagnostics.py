"""Safe, request-local provider attempt diagnostics.

Only metadata is recorded: phase, attempt number, HTTP status/version, latency,
whether a response/usage arrived, retry decision and exception class. URLs,
headers, keys, request bodies and user text are never stored here.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import httpx

_EVENTS: ContextVar[list[dict[str, Any]] | None] = ContextVar("llm_attempt_events", default=None)
_ALLOWED = {
    "attempt", "phase", "http_status", "http_version", "duration_ms",
    "response_received", "usage_available", "error_type", "retry_decision",
    "category", "streamed", "status_code",
}


def begin() -> None:
    _EVENTS.set([])


def ensure() -> None:
    if _EVENTS.get() is None:
        _EVENTS.set([])


def record(**fields: Any) -> None:
    events = _EVENTS.get()
    if events is None:
        return
    safe = {key: value for key, value in fields.items()
            if key in _ALLOWED and isinstance(value, (bool, int, float, str))}
    events.append(safe)


def snapshot() -> list[dict[str, Any]]:
    return [dict(item) for item in (_EVENTS.get() or [])]


def reset() -> None:
    _EVENTS.set(None)


def phase_for(exc: BaseException) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        return "connect"
    if isinstance(exc, (httpx.RemoteProtocolError, httpx.ReadError)):
        return "response"
    if isinstance(exc, httpx.HTTPStatusError):
        return "http_status"
    return "unknown"


def error_fields(exc: BaseException) -> dict[str, Any]:
    fields: dict[str, Any] = {"phase": phase_for(exc), "error_type": type(exc).__name__}
    if isinstance(exc, httpx.HTTPStatusError):
        fields["status_code"] = exc.response.status_code
    return fields
