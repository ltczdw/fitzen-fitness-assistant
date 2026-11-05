"""Fail-closed CNY guard for the explicitly authorized Flash acceptance run.

Not the application's daily quota. Reservations are never refunded, including
timeouts and retries. Prices checked against official documentation 2026-10-21:
https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
"""
from __future__ import annotations

import json
import os
import threading
from decimal import Decimal
from pathlib import Path


class CostStop(RuntimeError):
    pass


class CnyGuard:
    LIMIT = Decimal("1.75")
    MAX_CALLS = 24
    INPUT_RATE = Decimal("2") / 1_000_000
    OUTPUT_RATE = Decimal("8") / 1_000_000

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lease = self.path.with_suffix(".lock")
        self.lease_fd = os.open(self.lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        self.state = {"limit_cny": str(self.LIMIT), "reserved_cny": "0",
                      "usage_peak_upper_cny": "0", "halted": False, "calls": []}
        try:
            if self.path.exists():
                self.state = json.loads(self.path.read_text(encoding="utf-8"))
                if self.state["limit_cny"] != str(self.LIMIT):
                    raise CostStop("ledger_limit_mismatch")
                if any(c["status"] == "reserved" for c in self.state["calls"]):
                    self.state["halted"] = True
            self._save()
        except BaseException:
            self.close()
            raise

    def close(self):
        with self.lock:
            if getattr(self, "lease_fd", None) is not None:
                os.close(self.lease_fd)
                self.lease_fd = None
                self.lease.unlink()

    def _require_open(self):
        if getattr(self, "lease_fd", None) is None:
            raise CostStop("meter_closed")

    def _save(self):
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.state, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def reserve(self, url, body: bytes, case="unspecified"):
        with self.lock:
            self._require_open()
            if self.state["halted"]:
                raise CostStop("meter_halted")
            try:
                from urllib.parse import urlsplit
                endpoint = urlsplit(str(url))
                if (endpoint.scheme != "https" or endpoint.hostname != "api.deepseek.com"
                        or endpoint.port not in (None, 443) or endpoint.username or endpoint.password
                        or endpoint.query or endpoint.fragment
                        or endpoint.path not in ("/chat/completions", "/v1/chat/completions")):
                    raise CostStop("endpoint_not_allowed")
                payload = json.loads(body)
                if payload.get("model") != "deepseek-flash" or payload.get("stream"):
                    raise CostStop("model_or_stream_not_allowed")
                output = payload.get("max_tokens")
                if type(output) is not int or not 1 <= output <= 4096:
                    raise CostStop("missing_or_excess_output_cap")
                if any(t.get("type") != "function" for t in payload.get("tools", [])):
                    raise CostStop("billable_server_tool_not_allowed")
                messages = payload.get("messages")
                if not isinstance(messages, list) or not messages or len(messages) > 64:
                    raise CostStop("messages_not_bounded")
                if any(m.get("role") not in ("system", "user", "assistant", "tool")
                       or not isinstance(m.get("content", ""), (str, type(None))) for m in messages):
                    raise CostStop("only_text_messages_allowed")
                # UTF-8 byte count is a conservative byte-BPE token bound.
                # Include the entire wire schema/tools and another 8192 tokens
                # for protocol framing. Validate against actual returned usage.
                input_bound = len(body) + 8192
                if input_bound > 100_000:
                    raise CostStop("input_too_large")
                reserved = input_bound * self.INPUT_RATE + output * self.OUTPUT_RATE
                total = Decimal(self.state["reserved_cny"]) + reserved
                if total > self.LIMIT or len(self.state["calls"]) >= self.MAX_CALLS:
                    raise CostStop("next_call_would_exceed_budget")
            except BaseException:
                self.state["halted"] = True
                self._save()
                raise
            entry = {"number": len(self.state["calls"]) + 1, "case": case,
                     "input_bound": input_bound, "output_bound": output,
                     "reservation_cny": str(reserved), "status": "reserved"}
            self.state["calls"].append(entry)
            self.state["reserved_cny"] = str(total)
            self._save()  # Durable before network dispatch; no refund ever.
            return entry["number"]

    def finish(self, number, data=None, *, error=None):
        with self.lock:
            self._require_open()
            entry = self.state["calls"][number - 1]
            if entry["status"] != "reserved":
                raise CostStop("duplicate_settlement")
            usage = data.get("usage") if isinstance(data, dict) else None
            valid = (isinstance(usage, dict)
                     and all(type(usage.get(k)) is int and usage[k] >= 0
                             for k in ("prompt_tokens", "completion_tokens", "total_tokens")))
            if valid:
                valid = (usage["prompt_tokens"] <= entry["input_bound"]
                         and usage["completion_tokens"] <= entry["output_bound"]
                         and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"])
            if error or not valid:
                entry["status"] = "unknown_cost_reserved"
                self.state["halted"] = True
                self._save()
                raise CostStop("upstream_error_or_unmetered_usage")
            entry["status"] = "metered"
            entry["usage"] = {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
            details = usage.get("completion_tokens_details") or {}
            if type(details.get("reasoning_tokens")) is int:
                entry["usage"]["reasoning_tokens"] = details["reasoning_tokens"]
            # Charge cache hits as cache misses and all calls at peak pricing.
            cost = usage["prompt_tokens"] * self.INPUT_RATE + usage["completion_tokens"] * self.OUTPUT_RATE
            entry["usage_peak_upper_cny"] = str(cost)
            self.state["usage_peak_upper_cny"] = str(Decimal(self.state["usage_peak_upper_cny"]) + cost)
            self._save()


class Cny4Guard(CnyGuard):
    """Separate user-authorized CNY4 batch; preserve all per-call safeguards."""

    LIMIT = Decimal("3.50")
    MAX_CALLS = 50
