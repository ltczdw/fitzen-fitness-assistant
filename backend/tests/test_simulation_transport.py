from pathlib import Path
import sys
import httpx
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from simulation_transport import RequestAudit, AcceptanceStop


def audit(tmp_path):
    return RequestAudit({"https://synthetic.example/chat": ("chat", "test")}, tmp_path / "ledger.json", max_output_tokens=4096)


def wire(value=0, **extra):
    return httpx.Request("POST", "https://synthetic.example/chat", json={
        "model": "test", "max_tokens": 4096, "messages": [{"role": "user", "content": str(value)}], **extra})


def test_fixed_100_ceiling_removed_and_configured_output_unchanged(tmp_path):
    guard = audit(tmp_path)
    for i in range(102):
        row = guard.reserve(wire(i))
        guard.finish(row, httpx.Response(200, json={"usage": {"total_tokens": 1}, "choices": [{"finish_reason": "stop"}]}))
    assert len(guard.state["calls"]) == 102 and guard.state["physical_call_limit"] is None
    assert all(r["usage"] and r["finish_reasons"] == ["stop"] for r in guard.state["calls"])


@pytest.mark.parametrize("extra", [{"model": "other"}, {"max_tokens": 4097}, {"max_tokens": True}, {"stream": True}])
def test_normal_output_allowance_not_unbounded_or_unmetered(tmp_path, extra):
    guard = audit(tmp_path)
    with pytest.raises(AcceptanceStop):
        guard.reserve(wire(**extra))
    assert not guard.state["calls"]


def test_retry_and_missing_usage_still_stop(tmp_path):
    guard = audit(tmp_path)
    row = guard.reserve(wire())
    with pytest.raises(AcceptanceStop, match="usage_missing"):
        guard.finish(row, httpx.Response(200, json={}))
    with pytest.raises(AcceptanceStop, match="run_halted"):
        guard.reserve(wire(1))
