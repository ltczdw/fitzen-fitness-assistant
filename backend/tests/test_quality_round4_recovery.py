"""Fourth-round bounded synthetic cache faults through real HTTP/SQL recovery."""
from copy import deepcopy
import json

import pytest
from test_chat_http_write_guard import chat_db, plan, workout_plan
from test_r011_acceptance_semantics import AcceptanceHarness, make_harness, record, seed_cache


def resumed(h, message, rid, cid, before, record_property):
    from app.services.agent.operation_store import OperationStore
    try:
        return h.run(message, request_id=rid, conversation_id=cid)
    finally:
        record_property("ledger_before", json.dumps(before, ensure_ascii=False, default=str))
        record_property("ledger_after", json.dumps(OperationStore(rid).load(), ensure_ascii=False, default=str))


def target_result(h):
    return next(w["result"]["worker_results"][-1] for w in h.trace["workers"] if w["task"]["id"] == "t1")


@pytest.mark.parametrize("shape,value", [("empty_list", []), ("null", None), ("empty_object", {}), ("empty_string", "")])
@pytest.mark.parametrize("completed", [False, True], ids=["pending", "saved"])
def test_falsey_frozen_records_never_restart_extraction(chat_db, monkeypatch, record_property, shape, value, completed):
    from app.services.agent.operation_store import OperationStore
    message = "记录卧推60kg，4组8次"
    h = AcceptanceHarness(monkeypatch, chat_db, property_writer=record_property)
    rid, cid, store = seed_cache(h, message, [record(date="2026-10-03")], source=message, completed=(0,) if completed else ())
    before, rows_before = deepcopy(store.load()), deepcopy(h.rows())
    original_params = OperationStore.task_params
    original_extract = h.agent.graph.workout_extraction.run_many
    hits, extractions = [], []
    def damaged(s, task):
        p = deepcopy(original_params(s, task)); p["records"] = deepcopy(value)
        hits.append(task)
        return p
    async def extracted(source):
        extractions.append(source)
        return await original_extract(source)
    monkeypatch.setattr(OperationStore, "task_params", damaged)
    monkeypatch.setattr(h.agent.graph.workout_extraction, "run_many", extracted)
    h.trace["reservations"].clear(); h.trace["freezes"].clear(); h.frozen.clear()
    record_property("cache_fault", shape)
    record_property("previously_saved", completed)
    response = resumed(h, message, rid, cid, before, record_property)
    record_property("fault_hits", json.dumps(hits))
    record_property("fresh_extractions", len(extractions))
    assert hits, "fault accessor was not exercised"
    assert extractions == [], {"extraction_count": len(extractions), "response": response}
    assert h.payloads == h.trace["reservations"] == h.frozen == []
    assert h.rows() == rows_before and OperationStore(rid).load() == before
    result = target_result(h)
    if completed:
        saved = [o["result"]["id"] for o in before["operations"].values() if o["status"] == "saved"]
        assert [r["id"] for r in result.get("records", [])] == saved or any(s in result["answer"] for s in ("无法核验", "恢复受限"))
    assert any(s in result["answer"] for s in ("缓存", "恢复", "损坏")), result


def test_actual_missing_task_params_is_fresh_extraction_control(chat_db, monkeypatch, record_property):
    from app.services.agent.operation_store import OperationStore
    message = "记录卧推60kg，4组8次"
    h = AcceptanceHarness(monkeypatch, chat_db, workout_plan(message), property_writer=record_property)
    original, observed = OperationStore.task_params, []
    def checked(s, task):
        value = original(s, task); observed.append(value)
        return value
    monkeypatch.setattr(OperationStore, "task_params", checked)
    h.run(message)
    record_property("initial_task_params_absent", observed == [None])
    assert observed == [None]
    assert len(h.payloads) == len(h.rows()) == len(h.frozen) == 1


def test_bounded_long_ascii_suffix_refuses_without_http500(chat_db, monkeypatch, record_property):
    from app.services.agent.operation_store import OperationStore
    message = "记录卧推60kg，4组8次"
    h = AcceptanceHarness(monkeypatch, chat_db, property_writer=record_property)
    rid, cid, store = seed_cache(h, message, [record(date="2026-10-03")], source=message)
    before = deepcopy(store.load()); original = OperationStore.task_params
    hits = []
    def damaged(s, task):
        p = deepcopy(original(s, task)); p.pop("positions", None)
        p["operation_keys"] = [f"workout-task-{rid}-t1-" + "9" * 5000]
        hits.append(task)
        return p
    monkeypatch.setattr(OperationStore, "task_params", damaged)
    h.trace["reservations"].clear(); h.frozen.clear()
    record_property("ascii_suffix_length", 5000)
    record_property("fault_boundary", "synthetic frozen accessor; no integer-limit override")
    try:
        response = resumed(h, message, rid, cid, before, record_property)
    finally:
        record_property("fault_hits", json.dumps(hits))
    h.assert_no_write(response)
    assert hits and h.trace["reservations"] == [] and h.frozen == []
    assert OperationStore(rid).load() == before


def test_completed_results_use_original_position_order(chat_db, monkeypatch, record_property):
    from app.services.agent.operation_store import OperationStore
    message = "记录卧推60kg4组8次，深蹲80kg3组5次，划船30kg2组10次"
    values = [record(date="2026-10-03"), record(exercise="深蹲", weight_kg=80, sets=3, reps=5, date="2026-10-03"),
              record(exercise="划船", weight_kg=30, sets=2, reps=10, date="2026-10-03")]
    h = AcceptanceHarness(monkeypatch, chat_db, property_writer=record_property)
    rid, cid, store = seed_cache(h, message, values, source=message, completed=(2, 0))
    before = deepcopy(store.load()); original = OperationStore.task_params
    keys = [o["key"] for o in before["operations"].values()]
    record_property("ledger_inserted_position_order", "2,0")
    assert keys == [f"workout-task-{rid}-t1-2", f"workout-task-{rid}-t1-0"], "fixture insertion order was not retained"
    def damaged(s, task):
        p = deepcopy(original(s, task)); p.update(records=values[:1], positions=None)
        return p
    monkeypatch.setattr(OperationStore, "task_params", damaged)
    h.trace["reservations"].clear(); h.frozen.clear()
    resumed(h, message, rid, cid, before, record_property)
    result = target_result(h)
    record_property("target_task_result", json.dumps(result, ensure_ascii=False, default=str))
    expected = [before["operations"][op]["result"] for op in before["operations"]][::-1]
    assert h.payloads == h.trace["reservations"] == h.frozen == [] and len(h.rows()) == 2
    assert OperationStore(rid).load() == before
    assert [r["id"] for r in result.get("records", [])] == [r["id"] for r in expected], result


@pytest.mark.parametrize("bad_saved", [None, {"exercise": "卧推", "weight_kg": "damaged", "sets": 4, "reps": 8}], ids=["null_saved_item", "bad_saved_fact"])
def test_bad_completed_item_preserved_and_pending_items_not_compacted(chat_db, monkeypatch, record_property, bad_saved):
    from app.services.agent.operation_store import OperationStore
    message = "记录卧推60kg4组8次，深蹲80kg3组5次"
    values = [record(date="2026-10-03"), record(exercise="深蹲", weight_kg=80, sets=3, reps=5, date="2026-10-03"),
              record(exercise="划船", weight_kg=999, sets=2, reps=10, date="2026-10-03")]
    h = AcceptanceHarness(monkeypatch, chat_db, property_writer=record_property)
    rid, cid, store = seed_cache(h, message, values, source=message, completed=(0,))
    before = deepcopy(store.load()); original = OperationStore.task_params
    saved = next(o["result"] for o in before["operations"].values() if o["status"] == "saved")
    def damaged(s, task):
        p = deepcopy(original(s, task)); p["records"][0] = deepcopy(bad_saved)
        return p
    monkeypatch.setattr(OperationStore, "task_params", damaged)
    h.trace["reservations"].clear(); h.frozen.clear()
    resumed(h, message, rid, cid, before, record_property)
    after = OperationStore(rid).load(); result = target_result(h)
    record_property("target_task_result", json.dumps(result, ensure_ascii=False, default=str))
    assert [r["id"] for r in result["records"]][0] == saved["id"]
    assert len(h.payloads) == 1 and h.payloads[0]["exercise"] == "深蹲"
    assert h.payloads[0]["idempotency_key"] == f"workout-task-{rid}-t1-1"
    assert len(h.rows()) == 2 and h.frozen == []
    assert all(after[k] == before[k] for k in ("plan", "tasks", "time_base"))
    for op, value in before["operations"].items(): assert after["operations"][op] == value


@pytest.mark.parametrize("message", ["记录卧推共2组，第1组60kg8次，第2组60kg8次", "记录卧推60kg，2组8次"])
def test_uniform_source_may_omit_detail_array(chat_db, monkeypatch, record_property, message):
    candidate = record(sets=2)
    h = make_harness(monkeypatch, chat_db, message, [candidate], record_property)
    h.expect_no_workout_freeze = False
    h.run(message); h.assert_structured_path(message, [candidate])
    assert len(h.payloads) == len(h.rows()) == 1
    assert (h.rows()[0][1], h.rows()[0][2], h.rows()[0][3]) == (60, 2, 8)
