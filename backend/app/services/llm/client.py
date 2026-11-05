"""OpenAI-compatible model gateway.

The gateway keeps the existing ``complete`` API while adding sync/async
calls, upstream SSE streaming, bounded retries, admission limits and a
Sentinel-style circuit breaker.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from contextvars import ContextVar
from app.services.llm.limits import Admission, sync_loop
import json
import re
import threading
from urllib.parse import urlsplit
import time
from datetime import timedelta
from dataclasses import dataclass
from enum import Enum
from collections.abc import Awaitable, Callable
from typing import Any, AsyncIterator

import httpx

from app.core import clock
from app.core.config import settings
from app.db.storage import StorageUnavailableError, CommitUnknownError
from app.services.context.budget import ContextBudgetExceeded, context_budget
from app.services.observability import trace_span
from app.services.telemetry import model_attempt, mark_degraded
from app.services.llm.budget import (ModelBudgetError, async_paid_attempt, cache_key, cached_async, cache_async, active_scope)
from app.services.llm.retry_policy import pause_seconds, TRANSIENT_STATUS
from app.services.llm import diagnostics as attempt_diagnostics


class LLMErrorCategory(str, Enum):
    """对外稳定的四级异常分类。"""

    CONFIG = "config"
    TRANSPORT = "transport"  # network / timeout
    UPSTREAM = "upstream"
    POLICY = "policy"
    PROTOCOL = "protocol"  # malformed JSON / SSE / provider response


class LLMGatewayError(RuntimeError):
    """可分类、可观测、带重试标记的大模型网关异常。"""

    def __init__(
        self,
        message: str,
        category: LLMErrorCategory,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.status_code = status_code
        self.retry_failure: LLMGatewayError | None = None
        if code is not None:
            self.code = code


class LLMContextBudgetError(LLMGatewayError):
    """Local input rejection, never a malformed provider response."""
    code = "context_input_budget_exhausted"

    def __init__(self, error: ContextBudgetExceeded) -> None:
        super().__init__(self.code, LLMErrorCategory.PROTOCOL, retryable=False)
        self.used, self.budget, self.reason = error.used, error.budget, error.reason
        self.detail = dict(error.detail)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class SentinelCircuitBreaker:
    """轻量 Sentinel 风格熔断器：失败阈值触发 OPEN，冷却后半开探测。"""

    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 30.0) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_seconds = max(1.0, recovery_seconds)
        self.failure_count = 0
        self.opened_at: float | None = None
        self.state = CircuitState.CLOSED
        self._half_open_probe = False
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if self.opened_at is None or time.monotonic() - self.opened_at < self.recovery_seconds:
                    return False
                self.state = CircuitState.HALF_OPEN
                self._half_open_probe = True
                return True
            if self._half_open_probe:
                return False
            self._half_open_probe = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self.failure_count = 0
            self.opened_at = None
            self.state = CircuitState.CLOSED
            self._half_open_probe = False

    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self._half_open_probe = False
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "recovery_seconds": self.recovery_seconds,
            }


@dataclass(frozen=True)
class _RetryPlan:
    delays: tuple[float, ...]

    @property
    def max_attempts(self) -> int:
        return min(settings.model_http_max_attempts, len(self.delays) + 1)


async def _observed_post(client, url: str, headers: dict[str, str], payload: dict[str, Any],
                         attempt: int):
    """One physical POST with metadata-only diagnostics; never records bodies."""
    attempt_diagnostics.ensure()
    number = attempt + 1
    started = time.monotonic()
    attempt_diagnostics.record(attempt=number, phase="send")
    try:
        response = await client.post(url, headers=headers, json=payload)
    except Exception as exc:
        attempt_diagnostics.record(attempt=number, **attempt_diagnostics.error_fields(exc),
                                   duration_ms=round((time.monotonic() - started) * 1000, 1))
        raise
    attempt_diagnostics.record(
        attempt=number, phase="response", http_status=response.status_code,
        http_version=str(response.http_version or ""), response_received=True,
        duration_ms=round((time.monotonic() - started) * 1000, 1))
    return response


def _record_retry(attempt: int, error: LLMGatewayError, retry: bool) -> None:
    attempt_diagnostics.record(
        attempt=attempt + 1, phase="retry",
        retry_decision="retry" if retry else "stop",
        category=error.category.value,
        status_code=error.status_code or 0)


def _retry_plan() -> _RetryPlan:
    values: list[float] = []
    for item in str(settings.llm_retry_delays).split(","):
        try:
            value = float(item.strip())
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            values.append(value)
    return _RetryPlan(tuple(values))


logger = logging.getLogger(__name__)

MAX_AGENT_STEPS = 6


class ToolLoopLimitReached(RuntimeError):
    """Carry completed tool results without restarting an exhausted task."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(str(result.get("content") or "Agent step limit reached"))
        self.result = result

_shared_runtimes = {}
_runtime_lock = threading.Lock()
_call_deadline = ContextVar("llm_call_deadline", default=None)


class LLMClient:
    """OpenAI-compatible API client with sync, async and upstream SSE modes."""

    def __init__(self) -> None:
        key = (settings.llm_base_url, settings.llm_model,
               hashlib.sha256(settings.llm_api_key.encode()).hexdigest(),
               settings.llm_max_concurrency, settings.llm_requests_per_second,
               settings.llm_rate_burst, settings.llm_circuit_failure_threshold,
               settings.llm_circuit_recovery_seconds)
        with _runtime_lock:
            if key not in _shared_runtimes:
                _shared_runtimes[key] = (
                    SentinelCircuitBreaker(settings.llm_circuit_failure_threshold, settings.llm_circuit_recovery_seconds),
                    Admission(settings.llm_max_concurrency, settings.llm_requests_per_second, settings.llm_rate_burst))
            self.breaker, self.admission = _shared_runtimes[key]

    async def _bounded(self, operation):
        budget = active_scope.get()
        if budget is not None:
            budget.check()
        if _call_deadline.get() is not None:
            return await operation()
        if self.breaker.snapshot()["state"] == "open" and not self._cooldown_elapsed():
            raise LLMGatewayError("LLM circuit OPEN", LLMErrorCategory.UPSTREAM, retryable=True, code="model_circuit_open")
        seconds = min(settings.llm_total_deadline_seconds, budget.remaining() if budget else settings.llm_total_deadline_seconds)
        token = _call_deadline.set(time.monotonic() + seconds)
        try:
            async with asyncio.timeout(seconds):
                async with self.admission.slot():
                    return await operation()
        except ModelBudgetError as exc:
            raise self._classify_exception(exc) from exc
        except TimeoutError as exc:
            self.breaker.record_failure()
            raise LLMGatewayError("LLM total deadline exceeded", LLMErrorCategory.TRANSPORT, retryable=False, code="model_timeout") from exc
        finally:
            _call_deadline.reset(token)

    def _cooldown_elapsed(self):
        with self.breaker._lock:
            return self.breaker.opened_at is not None and time.monotonic() - self.breaker.opened_at >= self.breaker.recovery_seconds

    def configured(self) -> bool:
        return bool(settings.llm_base_url and settings.llm_api_key and settings.llm_model)

    def status(self) -> dict[str, Any]:
        return {"configured": self.configured(), "circuit": self.breaker.snapshot()}

    def _endpoint(self) -> tuple[str, dict[str, str]]:
        if not self.configured():
            raise LLMGatewayError("LLM 未配置", LLMErrorCategory.CONFIG)
        return (
            settings.llm_base_url.rstrip("/") + "/chat/completions",
            {"Authorization": f"Bearer {settings.llm_api_key}"},
        )

    @staticmethod
    def _classify_exception(exc: Exception) -> LLMGatewayError:
        if isinstance(exc, LLMGatewayError):
            return exc
        if isinstance(exc, ModelBudgetError):
            return LLMGatewayError(exc.code, LLMErrorCategory.POLICY, retryable=False, code=exc.code)
        if isinstance(exc, ContextBudgetExceeded):
            return LLMContextBudgetError(exc)
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            return LLMGatewayError(str(exc) or "LLM 请求超时", LLMErrorCategory.TRANSPORT, retryable=True, code="model_timeout")
        if isinstance(exc, httpx.TransportError):
            return LLMGatewayError(str(exc) or "LLM 网络错误", LLMErrorCategory.TRANSPORT, retryable=True)
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            retryable = status in TRANSIENT_STATUS
            return LLMGatewayError(
                f"LLM 上游 HTTP {status}",
                LLMErrorCategory.UPSTREAM,
                retryable=retryable,
                status_code=status,
            )
        if isinstance(exc, (KeyError, TypeError, ValueError, json.JSONDecodeError)):
            return LLMGatewayError(str(exc) or "LLM 响应格式错误", LLMErrorCategory.PROTOCOL)
        return LLMGatewayError(str(exc) or "LLM 未知错误", LLMErrorCategory.UPSTREAM, retryable=False)

    @staticmethod
    def _payload(
        messages: list[dict[str, Any]],
        temperature: float,
        stream: bool,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "max_tokens": settings.llm_max_output_tokens,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        # Plain answers must fit the application's visible-output allowance;
        # DeepSeek's default thinking can consume it before emitting content.
        # Forced tools also require non-thinking mode. Auto tool conversations
        # retain their existing reasoning/history protocol; other providers
        # must not receive this provider-specific extension.
        if (urlsplit(settings.llm_base_url).hostname == "api.deepseek.com"
                and (not tools or isinstance(tool_choice, dict) or tool_choice == "required")):
            payload["thinking"] = {"type": "disabled"}
        with trace_span("context.budget", inputs={"message_count": len(messages)},
                        metadata={"model": settings.llm_model}, tags=["budget"]) as span:
            from app.services.telemetry import record_context_preflight
            try:
                prepared = context_budget.prepare_request(payload)
            except ContextBudgetExceeded as error:
                report = {"request_input_tokens":error.used, "input_budget":error.budget,
                          "output_reserved":payload["max_tokens"], "reason":error.reason,
                          "counter_mode":error.detail.get("counter_mode", "unknown")}
                record_context_preflight("rejected", report)
                span.end(report)
                raise
            span.end(prepared.report)
            record_context_preflight("accepted", prepared.report)
        return prepared

    @staticmethod
    def _content(data: Any) -> str:
        try:
            content = LLMClient._assistant_message(data)["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMGatewayError("LLM 响应缺少 choices.message.content", LLMErrorCategory.PROTOCOL) from exc
        if content is None:
            return ""
        if not isinstance(content, str):
            raise LLMGatewayError("LLM 响应 content 不是字符串", LLMErrorCategory.PROTOCOL)
        return content

    async def _complete_once(self, messages: list[dict[str, str]], temperature: float,
                             attempt: int = 0) -> str:
        url, headers = self._endpoint()
        payload = self._payload(messages, temperature, False)
        identifier = cache_key('chat', settings.llm_model, payload, hashlib.sha256((url + headers['Authorization']).encode()).hexdigest())
        cached = await cached_async(identifier)
        if cached is not None:
            return self._content(cached['data'])
        async with async_paid_attempt('chat', settings.llm_model, payload, settings.llm_max_output_tokens) as paid:
            with model_attempt(settings.llm_model, streaming=False) as attempt_counter:
                async with httpx.AsyncClient(timeout=min(settings.llm_timeout_seconds, paid.scope.remaining())) as client:
                    response = await _observed_post(client, url, headers, payload, attempt)
                    response.raise_for_status()
                    data = response.json()
                    attempt_diagnostics.record(
                        attempt=attempt + 1, phase="usage",
                        usage_available=isinstance(data.get("usage"), dict))
                    paid.observe(data); attempt_counter.observe(data)
                    content = self._content(data)
                    await cache_async(identifier, 'chat', {'data': data})
                    return content

    @staticmethod
    def _assistant_message(data: Any) -> dict[str, Any]:
        try:
            choice = data["choices"][0]
            if not isinstance(choice, dict):
                raise TypeError("Invalid LLM choice")
            if choice.get("finish_reason") == "length":
                raise LLMGatewayError("structured_response_truncated", LLMErrorCategory.PROTOCOL)
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMGatewayError("LLM 响应缺少 choices.message", LLMErrorCategory.PROTOCOL) from exc
        if not isinstance(message, dict):
            raise LLMGatewayError("LLM 响应 message 不是对象", LLMErrorCategory.PROTOCOL)
        return message

    async def _complete_message_once(self, messages, temperature, tools, tool_choice, attempt: int = 0):
        url, headers = self._endpoint()
        payload = self._payload(messages, temperature, False, tools=tools, tool_choice=tool_choice)
        identifier = cache_key('chat', settings.llm_model, payload, hashlib.sha256((url + headers['Authorization']).encode()).hexdigest())
        cached = await cached_async(identifier)
        if cached is not None:
            return self._assistant_message(cached['data'])
        async with async_paid_attempt('chat', settings.llm_model, payload, settings.llm_max_output_tokens) as paid:
            with model_attempt(settings.llm_model, streaming=False) as attempt_counter:
                async with httpx.AsyncClient(timeout=min(settings.llm_timeout_seconds, paid.scope.remaining())) as client:
                    response = await _observed_post(client, url, headers, payload, attempt)
                    response.raise_for_status()
                    data = response.json()
                    attempt_diagnostics.record(
                        attempt=attempt + 1, phase="usage",
                        usage_available=isinstance(data.get("usage"), dict))
                    paid.observe(data); attempt_counter.observe(data)
                    message = self._assistant_message(data)
                    await cache_async(identifier, 'chat', {'data': data})
                    return message

    async def _complete_message_impl(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not self.configured():
            raise LLMGatewayError("LLM 未配置", LLMErrorCategory.CONFIG)
        plan = _retry_plan()
        if not self.breaker.allow_request():
            raise LLMGatewayError("LLM 熔断器处于 OPEN 状态", LLMErrorCategory.UPSTREAM, retryable=True, code="model_circuit_open")
        previous_failure = None
        for attempt in range(plan.max_attempts):
            try:
                await self.admission.token()
                result = await self._complete_message_once(messages, temperature, tools, tool_choice, attempt)
                self.breaker.record_success()
                return result
            except Exception as raw_exc:
                error = self._classify_exception(raw_exc)
                if error.category == LLMErrorCategory.POLICY:
                    error.retry_failure = previous_failure
                if not isinstance(error, LLMContextBudgetError):
                    self.breaker.record_failure()
                will_retry = bool(error.retryable and attempt < plan.max_attempts - 1)
                _record_retry(attempt, error, will_retry)
                if not error.retryable or attempt >= plan.max_attempts - 1:
                    raise error from raw_exc
                previous_failure = error
                mark_degraded("model_retry")
                remaining = max(0, (_call_deadline.get() or (time.monotonic() + settings.llm_total_deadline_seconds)) - time.monotonic())
                after = raw_exc.response.headers.get("Retry-After") if isinstance(raw_exc, httpx.HTTPStatusError) else None
                await asyncio.sleep(pause_seconds(plan.delays[attempt], remaining, after))
        raise LLMGatewayError("LLM 工具调用未返回结果", LLMErrorCategory.PROTOCOL)

    async def _complete_with_tools_impl(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tool_executor: Callable[[str, dict[str, Any]], Awaitable[Any]],
        temperature: float = 0.2,
        max_tool_rounds: int = MAX_AGENT_STEPS,
    ) -> dict[str, Any] | None:
        """Run an OpenAI-compatible MCP tool-calling loop.

        ``tools`` must come from MCP ``tools/list``.  The model chooses a
        function, this gateway executes it through the supplied MCP callback,
        and the tool result is sent back until the model returns final text.
        """
        if not self.configured() or not tools:
            return None
        history: list[dict[str, Any]] = [dict(message) for message in messages]
        executed = 0
        tool_results: list[dict[str, Any]] = []
        step_limit = min(MAX_AGENT_STEPS, max(1, max_tool_rounds))
        for round_index in range(step_limit):
            try:
                assistant = await self._complete_message(history, temperature, tools, "auto")
            except asyncio.CancelledError:
                deadline = _call_deadline.get()
                if deadline is None or time.monotonic() < deadline or not tool_results:
                    raise
                logger.warning("llm_tool_loop_deadline_after_completed_tools", extra={"tool_count": executed})
                return {"content": "", "tool_results": tool_results, "error": "completion_after_tools_failed"}
            except Exception:
                if not tool_results:
                    raise
                logger.exception("llm_completion_after_tools_failed")
                return {"content": "", "tool_results": tool_results, "error": "completion_after_tools_failed"}
            tool_calls = assistant.get("tool_calls") or []
            assistant_content = assistant.get("content") if isinstance(assistant.get("content"), str) else ""
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content,
            }
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            history.append(assistant_message)
            if not tool_calls:
                content = assistant.get("content")
                return {
                    "content": content if isinstance(content, str) else "",
                    "messages": history,
                    "tool_calls": executed,
                    "tool_results": tool_results,
                }

            # Like the graph agent, stop before executing calls emitted at
            # the limit: no remaining model step can consume their results.
            if round_index + 1 >= step_limit:
                break

            for call_index, call in enumerate(tool_calls):
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                name = str(function.get("name") or "").strip()
                raw_arguments = function.get("arguments", {})
                if isinstance(raw_arguments, str):
                    try:
                        arguments = json.loads(raw_arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                else:
                    arguments = raw_arguments
                if not isinstance(arguments, dict):
                    arguments = {}
                tool_call_id = str(call.get("id") or f"mcp-call-{round_index}-{call_index}")
                try:
                    result = await tool_executor(name, arguments)
                    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
                    tool_results.append({"name": name, "result": result})
                except asyncio.CancelledError:
                    raise
                except (StorageUnavailableError, CommitUnknownError):
                    raise
                except Exception as exc:
                    content = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    tool_results.append({"name": name, "error": str(exc)})
                history.append({"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": content})
                executed += 1
        return {
            "content": f"当前任务已达到 {step_limit} 轮执行上限，尚未全部完成。已执行的操作不会自动重跑；涉及保存的数据请先查看记录。",
            "messages": history,
            "tool_calls": executed,
            "tool_results": tool_results,
            "error": "max_tool_rounds",
        }

    async def _complete_impl(self, messages: list[dict[str, str]], temperature: float = 0.3) -> str | None:
        if not self.configured():
            return None
        plan = _retry_plan()
        # 熔断只阻断新的逻辑请求；同一请求的退避重试不被自己的失败计数提前截断。
        if not self.breaker.allow_request():
            raise LLMGatewayError("LLM 熔断器处于 OPEN 状态", LLMErrorCategory.UPSTREAM, retryable=True, code="model_circuit_open")
        previous_failure = None
        for attempt in range(plan.max_attempts):
            try:
                await self.admission.token()
                result = await self._complete_once(messages, temperature, attempt)
                self.breaker.record_success()
                return result
            except Exception as raw_exc:
                error = self._classify_exception(raw_exc)
                if error.category == LLMErrorCategory.POLICY:
                    error.retry_failure = previous_failure
                if not isinstance(error, LLMContextBudgetError):
                    self.breaker.record_failure()
                will_retry = bool(error.retryable and attempt < plan.max_attempts - 1)
                _record_retry(attempt, error, will_retry)
                if not error.retryable or attempt >= plan.max_attempts - 1:
                    raise error from raw_exc
                previous_failure = error
                mark_degraded("model_retry")
                remaining = max(0, (_call_deadline.get() or (time.monotonic() + settings.llm_total_deadline_seconds)) - time.monotonic())
                after = raw_exc.response.headers.get("Retry-After") if isinstance(raw_exc, httpx.HTTPStatusError) else None
                await asyncio.sleep(pause_seconds(plan.delays[attempt], remaining, after))
        return None

    def complete_sync(self, messages, temperature: float = 0.3):
        future = asyncio.run_coroutine_threadsafe(self.complete(messages, temperature), sync_loop())
        try:
            return future.result(timeout=settings.llm_total_deadline_seconds + 1)
        except TimeoutError as exc:
            future.cancel()
            raise LLMGatewayError("LLM total deadline exceeded", LLMErrorCategory.TRANSPORT, retryable=False, code="model_timeout") from exc

    async def _stream_once(self, messages: list[dict[str, str]], temperature: float,
                           attempt: int = 0) -> AsyncIterator[str]:
        url, headers = self._endpoint()
        payload = self._payload(messages, temperature, True)
        attempt_diagnostics.ensure()
        number = attempt + 1
        started = time.monotonic()
        attempt_diagnostics.record(attempt=number, phase="send", streamed=True)
        async with async_paid_attempt("chat", settings.llm_model, payload, settings.llm_max_output_tokens) as paid:
            with model_attempt(settings.llm_model, streaming=True) as attempt_counter:
                truncated = False
                usage_seen = False
                timeout = httpx.Timeout(settings.llm_stream_timeout_seconds)
                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        async with client.stream("POST", url, headers=headers, json=payload) as response:
                            attempt_diagnostics.record(
                                attempt=number, phase="response", http_status=response.status_code,
                                http_version=str(response.http_version or ""),
                                response_received=True,
                                duration_ms=round((time.monotonic() - started) * 1000, 1),
                                streamed=True)
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if not line or not line.startswith("data:"):
                                    continue
                                data_text = line[5:].strip()
                                if data_text == "[DONE]":
                                    if truncated:
                                        raise LLMGatewayError("stream_response_truncated", LLMErrorCategory.PROTOCOL)
                                    return
                                try:
                                    data = json.loads(data_text)
                                except json.JSONDecodeError as exc:
                                    raise LLMGatewayError("LLM SSE 数据格式错误", LLMErrorCategory.PROTOCOL) from exc
                                if not usage_seen and isinstance(data.get("usage"), dict):
                                    usage_seen = True
                                    attempt_diagnostics.record(
                                        attempt=number, phase="usage", usage_available=True, streamed=True)
                                paid.observe(data); attempt_counter.observe(data)
                                choice = (data.get("choices") or [{}])[0]
                                # Drain through the usage trailer before failing so
                                # a truncated stream is still fully accounted for.
                                truncated = truncated or choice.get("finish_reason") == "length"
                                delta = choice.get("delta") or {}
                                content = delta.get("content") or (choice.get("message") or {}).get("content")
                                if content:
                                    yield str(content)
                            if truncated:
                                raise LLMGatewayError("stream_response_truncated", LLMErrorCategory.PROTOCOL)
                except Exception as exc:
                    attempt_diagnostics.record(
                        attempt=number, **attempt_diagnostics.error_fields(exc),
                        duration_ms=round((time.monotonic() - started) * 1000, 1),
                        streamed=True, usage_available=usage_seen)
                    raise
    async def _stream_impl(self, messages: list[dict[str, str]], temperature: float = 0.3) -> AsyncIterator[str]:
        """转发 OpenAI-compatible SSE token；未产出内容时才允许重试。"""
        if not self.configured():
            return
        plan = _retry_plan()
        if not self.breaker.allow_request():
            raise LLMGatewayError("LLM 熔断器处于 OPEN 状态", LLMErrorCategory.UPSTREAM, retryable=True, code="model_circuit_open")
        previous_failure = None
        for attempt in range(plan.max_attempts):
            emitted = False
            try:
                await self.admission.token()
                async for content in self._stream_once(messages, temperature, attempt):
                    emitted = True
                    yield content
                self.breaker.record_success()
                return
            except Exception as raw_exc:
                error = self._classify_exception(raw_exc)
                if error.category == LLMErrorCategory.POLICY:
                    error.retry_failure = previous_failure
                if not isinstance(error, LLMContextBudgetError):
                    self.breaker.record_failure()
                will_retry = bool(not emitted and error.retryable and attempt < plan.max_attempts - 1)
                _record_retry(attempt, error, will_retry)
                if emitted or not error.retryable or attempt >= plan.max_attempts - 1:
                    raise error from raw_exc
                previous_failure = error
                mark_degraded("model_retry")
                remaining = max(0, (_call_deadline.get() or (time.monotonic() + settings.llm_total_deadline_seconds)) - time.monotonic())
                after = raw_exc.response.headers.get("Retry-After") if isinstance(raw_exc, httpx.HTTPStatusError) else None
                await asyncio.sleep(pause_seconds(plan.delays[attempt], remaining, after))

    async def complete(self, messages, temperature: float = 0.3):
        return await self._bounded(lambda: self._complete_impl(messages, temperature))

    async def complete_structured(self, messages, schema, *, name="structured_result", temperature=0.0, validator=None,
                                  allow_json_retry=False):
        from app.services.llm.structured import complete_structured
        return await complete_structured(self, messages, schema, name=name, temperature=temperature,
                                         validator=validator, allow_json_retry=allow_json_retry)

    async def complete_message(self, messages, temperature=0.0, tools=None, tool_choice=None):
        return await self._complete_message(messages, temperature, tools, tool_choice)

    async def _complete_message(self, messages, temperature, tools, tool_choice):
        return await self._bounded(lambda: self._complete_message_impl(messages, temperature, tools, tool_choice))

    async def complete_with_tools(self, messages, **kwargs):
        return await self._bounded(lambda: self._complete_with_tools_impl(messages, **kwargs))

    async def stream(self, messages, temperature: float = 0.3):
        if self.breaker.snapshot()["state"] == "open" and not self._cooldown_elapsed():
            raise LLMGatewayError("LLM circuit OPEN", LLMErrorCategory.UPSTREAM, retryable=True, code="model_circuit_open")
        budget = active_scope.get()
        if budget is not None:
            budget.check()
        seconds = min(settings.llm_total_deadline_seconds, budget.remaining() if budget else settings.llm_total_deadline_seconds)
        token = _call_deadline.set(time.monotonic() + seconds)
        try:
            async with asyncio.timeout(seconds):
                async with self.admission.slot():
                    async for content in self._stream_impl(messages, temperature):
                        yield content
        except TimeoutError as exc:
            self.breaker.record_failure()
            raise LLMGatewayError("LLM total deadline exceeded", LLMErrorCategory.TRANSPORT, retryable=False, code="model_timeout") from exc
        finally:
            _call_deadline.reset(token)

    async def extract_workout_json(
        self,
        message: str,
        preprocessed: dict[str, Any],
        enhanced: bool = False,
    ) -> dict[str, Any] | None:
        """Use the model only for ambiguous semantics; Python validates the result."""
        if not self.configured():
            return None
        today = clock.today()
        retry_instruction = (
            "这是一次增强容错重试。请重新检查数字语序、中文数字、被省略的量词和动作同义词；仍然不能确定的字段必须填 null，禁止猜测。"
            if enhanced
            else "先结合代码预处理结果判断；能合理确定就抽取，不要因为量词省略而返回信息不足。"
        )
        system = """你是健身训练记录实体抽取器，不是聊天助手。
任务：从用户自然语言中抽取四个训练实体：exercise、weight_kg、sets、reps。
容忍口语化表达、语序调换、空格和标点差异、量词省略、公斤/kg同义词以及动作名称变体。
“每组8”在训练记录上下文中默认是 reps=8；“4组，每组8的10kg哑铃卧推”和“每组8的10kg哑铃卧推，我做了4组”都应抽取为 sets=4、reps=8、weight_kg=10。
只使用原文或代码预处理可以合理推出的信息，禁止编造训练数据。日期字段 record_date 使用合法 ISO 日期；原文有今天/昨天/前天/明天/后天或周几时按下面应用日期映射填写；原文无日期表达时填写今天。
只输出一个合法 JSON 对象，不要 Markdown，不要解释。字段必须是：
{"exercise": string|null, "weight_kg": number|null, "sets": integer|null, "reps": integer|null, "inferred_fields": string[], "record_date": string|null}
无法确定的字段填 null。"""
        system += (
            f"\n应用时区：{settings.app_timezone}。今天={today.isoformat()}；"
            f"昨天={(today - timedelta(days=1)).isoformat()}；"
            f"前天={(today - timedelta(days=2)).isoformat()}；"
            f"明天={(today + timedelta(days=1)).isoformat()}；"
            f"后天={(today + timedelta(days=2)).isoformat()}。"
        )
        examples = """示例：
输入：记录卧推60kg，4组，每组8次
输出：{"exercise":"卧推","weight_kg":60,"sets":4,"reps":8,"inferred_fields":[],"record_date":"今天对应的 ISO 日期"}
输入：记录今天的训练，我做了4组，每组8的10kg哑铃卧推
输出：{"exercise":"哑铃卧推","weight_kg":10,"sets":4,"reps":8,"inferred_fields":["reps"],"record_date":"今天对应的 ISO 日期"}
输入：我做了哑铃卧推，4组，10kg
输出：{"exercise":"哑铃卧推","weight_kg":10,"sets":4,"reps":null,"inferred_fields":[],"record_date":"今天对应的 ISO 日期"}"""
        user = f"{examples}\n{retry_instruction}\n原始输入：{message}\n代码预处理结果：{json.dumps(preprocessed, ensure_ascii=False)}"
        raw = await self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        if not raw:
            return None
        text = raw.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1)
        else:
            object_match = re.search(r"\{.*\}", text, re.DOTALL)
            if object_match:
                text = object_match.group(0)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
