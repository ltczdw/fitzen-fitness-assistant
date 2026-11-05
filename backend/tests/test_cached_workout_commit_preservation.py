"""Tests for cached workout commit preservation."""
from copy import deepcopy
import json
import pytest
from test_chat_http_write_guard import chat_db
from test_r011_acceptance_semantics import AcceptanceHarness, record, seed_cache


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan_weight", "inf_weight"])
@pytest.mark.parametrize("completed", [False, True], ids=["unfinished", "saved"])
def test_nonfinite_cache_no_new_write_preserves_saved_result(chat_db, monkeypatch, record_property, value, completed):
    from app.services.agent.operation_store import OperationStore
    message = "记录卧推60kg，4组8次"
    h = AcceptanceHarness(monkeypatch, chat_db, property_writer=record_property)
    rid, cid, store = seed_cache(h, message, [record(date="2026-10-03")], source=message, completed=(0,) if completed else ())
    before = deepcopy(store.load())
    original = OperationStore.task_params
    def damaged(s, task):
        p = deepcopy(original(s, task)); p["records"][0]["weight_kg"] = value
        return p
    monkeypatch.setattr(OperationStore, "task_params", damaged)
    h.trace["freezes"].clear(); h.trace["reservations"].clear()
    response = h.run(message, request_id=rid, conversation_id=cid)
    after = OperationStore(rid).load()
    record_property("ledger_before", json.dumps(before, ensure_ascii=False, default=str))
    record_property("ledger_after", json.dumps(after, ensure_ascii=False, default=str))
    assert h.trace["source_checks"], "cached record did not reach actual source/type verification"
    assert h.trace["freezes"] == h.trace["reservations"] == h.payloads == []
    assert after == before and len(h.rows()) == int(completed)
    if completed:
        result = h.trace["workers"][-1]["result"]["worker_results"][-1]
        saved = next(iter(before["operations"].values()))["result"]
        assert result.get("records") == [saved]
    else:
        h.assert_no_write(response)
        assert "缓存" in response["answer"] and "账本" not in response["answer"]
