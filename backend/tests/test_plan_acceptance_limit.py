"""Physical request accounting for the explicitly authorized limited live run."""
from pathlib import Path
import sys
import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_plan_knowledge_acceptance as live
from evals.v2.safety import deny_network


@pytest.fixture(autouse=True)
def offline():
    with deny_network():
        yield


def guard(tmp_path):
    return live.RequestLimit({"https://embedding.example.test/embeddings": ("embedding", "fixture")}, tmp_path / "requests.json")


def request(**extra):
    return httpx.Request("POST", "https://embedding.example.test/embeddings", json={
        "model": "fixture", "input": ["public text"], "dimensions": 1024, **extra})


def test_accounting_precedes_send_and_preserves_unknown_cost(tmp_path):
    limited = guard(tmp_path)
    record = limited.reserve(request())
    assert len(limited.state["calls"]) == 1 and record["status"] == "dispatched_cost_unknown"
    limited.finish(record, httpx.Response(200, json={"usage": {"total_tokens": 12}}))
    assert record["usage"] == {"total_tokens": 12} and limited.state["currency_cost"] == "unknown"


def test_physical_retry_never_dispatches_twice(tmp_path):
    limited = guard(tmp_path)
    limited.reserve(request())
    with pytest.raises(live.AcceptanceStop, match="retry_blocked"):
        limited.reserve(request())
    assert len(limited.state["calls"]) == 1 and limited.state["halted"]


@pytest.mark.parametrize("wire", [
    httpx.Request("POST", "https://unapproved.example.test/embeddings", json={}),
    httpx.Request("GET", "https://embedding.example.test/embeddings"),
    request(model="other"), request(input=["x"] * 21), request(dimensions=768),
])
def test_invalid_requests_stop_before_physical_dispatch(tmp_path, wire):
    limited = guard(tmp_path)
    with pytest.raises(live.AcceptanceStop):
        limited.reserve(wire)
    assert limited.state["halted"] and limited.state["calls"] == []


@pytest.mark.parametrize("status,body", [(400, {}), (200, {}), (200, {"usage": {"total_tokens": -1}})])
def test_bad_or_unmetered_response_stops_run(tmp_path, status, body):
    limited = guard(tmp_path)
    record = limited.reserve(request())
    with pytest.raises(live.AcceptanceStop):
        limited.finish(record, httpx.Response(status, json=body))
    with pytest.raises(live.AcceptanceStop, match="run_halted"):
        limited.reserve(request(input=["another text"]))
    assert len(limited.state["calls"]) == 1


def test_count_limit_cannot_be_renewed_in_same_run(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "MAX_CALLS", 1)
    limited = guard(tmp_path)
    limited.reserve(request())
    with pytest.raises(live.AcceptanceStop, match="physical_call_limit"):
        limited.reserve(request(input=["another text"]))
    assert len(limited.state["calls"]) == 1
