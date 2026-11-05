"""Request-local metrics. No prompts, responses, headers or credentials exported."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
from uuid import uuid4

active_metrics = ContextVar("active_metrics", default=None)


@dataclass
class RequestMetrics:
    run_id: str = field(default_factory=lambda: str(uuid4()))
    started: float = field(default_factory=time.monotonic)
    calls: list = field(default_factory=list)
    degraded_reasons: set = field(default_factory=set)
    failure_reasons: set = field(default_factory=set)
    safety_refusals: set = field(default_factory=set)
    context_preflights: list = field(default_factory=list)

    def summary(self):
        known = [c for c in self.calls if (
            c["input_tokens"] is not None and c["output_tokens"] is not None
            if c["capability"] == "chat" else
            c["input_tokens"] is not None or c["total_tokens"] is not None)]
        return {
            "latency_ms": round((time.monotonic() - self.started) * 1000, 2),
            "model_calls": len(self.calls),
            "input_tokens": sum(c["input_tokens"] or 0 for c in self.calls),
            "output_tokens": sum(c["output_tokens"] or 0 for c in self.calls),
            "total_tokens": sum(c["total_tokens"] or 0 for c in self.calls),
            "cached_tokens": sum(c["cached_tokens"] or 0 for c in self.calls),
            "calls_by_capability": {kind: sum(c["capability"] == kind for c in self.calls) for kind in ("chat", "embedding", "rerank")},
            "usage_missing_calls": len(self.calls) - len(known),
            "usage_complete": len(known) == len(self.calls),
            "degraded": bool(self.degraded_reasons),
            "degraded_reasons": sorted(self.degraded_reasons),
            "failure_reasons": sorted(self.failure_reasons),
            # Counted separately from service failures: a correct safety
            # refusal is a user-visible outcome, not degraded service.
            "safety_refusals": sorted(self.safety_refusals),
            "calls": list(self.calls),
            "context_preflights": list(self.context_preflights),
        }


def record_context_preflight(status: str, report: dict):
    """Estimates are separate from metered usage; export no input text or IDs."""
    current = active_metrics.get()
    if current is not None:
        allowed = {"window_tokens", "input_budget", "output_reserved", "request_before_tokens",
                   "request_input_tokens", "counter_mode", "protocol_allowance", "reason"}
        current.context_preflights.append({"status":status, **{k:v for k,v in report.items() if k in allowed},
            "dropped_blocks":len(report.get("dropped_blocks") or []),
            "dropped_recent_turns":len(report.get("dropped_recent_turns") or [])})


def mark_degraded(reason, *, failure=False):
    current = active_metrics.get()
    if current:
        current.degraded_reasons.add(reason)
        if failure:
            current.failure_reasons.add(reason)


def mark_safety_refusal(reason):
    """Record a deterministic refusal without implying service degradation."""
    current = active_metrics.get()
    if current:
        current.safety_refusals.add(str(reason))


class ModelAttempt:
    def __init__(self, model, streaming=False, capability="chat"):
        self.started = time.monotonic()
        self.row = {"capability": capability, "model": model, "streaming": streaming, "input_tokens": None,
                    "output_tokens": None, "total_tokens": None, "cached_tokens": None, "status": "success"}

    def observe(self, data):
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            return
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
        details = details if isinstance(details, dict) else {}
        values = {
            "total_tokens": usage.get("total_tokens"),
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
            "cached_tokens": details.get("cached_tokens", usage.get("prompt_cache_hit_tokens")),
        }
        for key, value in values.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                # OpenAI-compatible stream usage snapshots are cumulative;
                # repeated final frames must not be added together.
                self.row[key] = max(self.row[key] or 0, value)
        if self.row["input_tokens"] is not None and self.row["output_tokens"] is not None:
            self.row["total_tokens"] = max(self.row["total_tokens"] or 0, self.row["input_tokens"] + self.row["output_tokens"])


@contextmanager
def model_attempt(model, streaming=False, capability="chat"):
    attempt = ModelAttempt(model, streaming, capability)
    try:
        yield attempt
    except BaseException as exc:
        attempt.row["status"] = "cancelled" if type(exc).__name__ in {"CancelledError", "GeneratorExit"} else "error"
        attempt.row["error_type"] = type(exc).__name__
        raise
    finally:
        attempt.row["latency_ms"] = round((time.monotonic() - attempt.started) * 1000, 2)
        current = active_metrics.get()
        if current is not None:
            current.calls.append(attempt.row)
