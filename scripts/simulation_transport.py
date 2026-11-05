"""Audited synthetic live traffic with the user's fixed call ceiling removed.

Application request/stage/token budgets still apply. Provider failures, missing
usage and duplicate retries halt the run; only approved endpoints can leave it.
"""
import hashlib
import json

from run_plan_knowledge_acceptance import AcceptanceStop, save


class RequestAudit:
    def __init__(self, endpoints, output, *, max_output_tokens):
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("invalid configured output allowance")
        self.endpoints, self.output = endpoints, output
        self.max_output_tokens = max_output_tokens
        self.seen = set()
        self.state = {"currency_limit_cancelled_by_user": True, "currency_cost": "unknown",
            "physical_call_limit": None, "fixed_call_limit_cancelled_by_user": True,
            "chat_output_limit": max_output_tokens, "halted": False, "calls": []}
        save(output, self.state)

    def stop(self, code):
        self.state.update(halted=True, stop_code=code)
        save(self.output, self.state)
        raise AcceptanceStop(code)

    def reserve(self, request):
        if self.state["halted"]:
            raise AcceptanceStop("run_halted")
        identity = str(request.url)
        if request.method != "POST" or identity not in self.endpoints:
            self.stop("unapproved_endpoint")
        capability, model = self.endpoints[identity]
        if len(request.content) > 262144:
            self.stop("request_size_limit")
        payload = json.loads(request.content)
        if payload.get("model") != model:
            self.stop("unapproved_model")
        if capability == "chat" and (payload.get("stream") or type(payload.get("max_tokens")) is not int
                or not 1 <= payload["max_tokens"] <= self.max_output_tokens):
            self.stop("chat_output_limit")
        if capability == "embedding" and (not isinstance(payload.get("input"), list)
                or not 1 <= len(payload["input"]) <= 20 or payload.get("dimensions") != 1024):
            self.stop("embedding_batch_limit")
        signature = hashlib.sha256(identity.encode() + request.content).hexdigest()
        if signature in self.seen:
            self.stop("physical_retry_blocked")
        self.seen.add(signature)
        record = {"number": len(self.state["calls"]) + 1, "capability": capability,
            "request_bytes": len(request.content), "status": "dispatched_cost_unknown"}
        self.state["calls"].append(record)
        save(self.output, self.state)
        return record

    def finish(self, record, response):
        record["http_status"] = response.status_code
        if response.status_code != 200:
            self.stop("provider_http_error")
        data = response.json()
        usage = data.get("usage") if isinstance(data, dict) else None
        safe = {k: v for k, v in (usage or {}).items() if "token" in k and type(v) is int and v >= 0} if isinstance(usage, dict) else {}
        if not safe:
            self.stop("provider_usage_missing")
        record.update(status="response_received", usage=safe)
        choices = data.get("choices", [])
        record["finish_reasons"] = [c["finish_reason"] for c in choices if c.get("finish_reason") in
            {"stop", "length", "tool_calls", "content_filter"}]
        save(self.output, self.state)
        print(json.dumps({"physical_call": record["number"], "capability": record["capability"],
            "http_status": response.status_code, "usage": safe}), flush=True)
