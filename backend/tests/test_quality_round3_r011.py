"""Third-round HTTP evidence for planner, recovery boundaries and malformed cache.

Faults below affect synthetic accessor results, never external user input or real data.
"""
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
from uuid import uuid4

import pytest
from test_chat_http_write_guard import chat_db
from test_r011_acceptance_semantics import AcceptanceHarness, make_harness, record, seed_cache


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "negative_inf"])
def test_nonfinite_candidate_rejected_before_executable_plan(chat_db, monkeypatch, record_property, value):
    message = "记录卧推60kg，4组8次"
    h = make_harness(monkeypatch, chat_db, message, [record(weight_kg=value)], record_property)
    response = h.run(message)
    assert h.trace["routes"][0]["degraded_reason"] is None, h.trace
    assert h.trace["plans"] == h.trace["freezes"] == h.trace["reservations"] == [], h.trace
    h.assert_no_write(response)
    assert "账本" not in response["answer"] and "存储" not in response["answer"], response
    assert any(s in response["answer"] for s in ("非法", "有效", "数值", "数字", "参数")), response


def test_explicit_heterogeneous_source_cannot_omit_details(chat_db, monkeypatch, record_property):
    message = "记录卧推共2组，第1组60kg8次，第2组62.5kg6次"
    h = make_harness(monkeypatch, chat_db, message, [record(sets=2)], record_property)
    response = h.run(message)
    h.assert_structured_path(message, [record(sets=2)])
    assert h.trace["reservations"] == [], h.trace
    h.assert_no_write(response)


@pytest.mark.parametrize("missing", ["positions", "operation_keys"])
def test_only_one_missing_identity_field_keeps_original_position(chat_db, monkeypatch, record_property, missing):
    from app.services.agent.operation_store import OperationStore
    message = "记录卧推60kg，4组8次"
    h = AcceptanceHarness(monkeypatch, chat_db, property_writer=record_property)
    rid, cid, store = seed_cache(h, message, [record(date="2026-10-03")], source=message)
    original = OperationStore.task_params
    def params(s, task):
        p = deepcopy(original(s, task))
        p.update(positions=[2], operation_keys=[f"workout-task-{rid}-t1-2"])
        del p[missing]
        return p
    monkeypatch.setattr(OperationStore, "task_params", params)
    before = deepcopy(store.load())
    h.trace["reservations"].clear()
    h.run(message, request_id=rid, conversation_id=cid)
    assert len(h.payloads) == len(h.rows()) == 1
    assert h.payloads[0]["idempotency_key"] == f"workout-task-{rid}-t1-2"
    after = OperationStore(rid).load()
    assert after["plan"] == before["plan"] and after["time_base"] == before["time_base"]
    assert after["tasks"] == before["tasks"]


@pytest.mark.parametrize("fault", ["suffix_superscript", "suffix_text", "record_null", "record_number", "record_text", "record_bad_list"])
def test_malformed_cache_object_or_key_refuses_without_http500(chat_db, monkeypatch, record_property, fault):
    from app.services.agent.operation_store import OperationStore
    message = "记录卧推60kg，4组8次"
    h = AcceptanceHarness(monkeypatch, chat_db, property_writer=record_property)
    rid, cid, store = seed_cache(h, message, [record(date="2026-10-03")], source=message)
    before = deepcopy(store.load())
    original = OperationStore.task_params
    def corrupted(s, task):
        p = deepcopy(original(s, task))
        if fault.startswith("suffix"):
            p["operation_keys"] = [f"workout-task-{rid}-t1-" + ("²" if fault == "suffix_superscript" else "bad")]
            p.pop("positions", None)
        else:
            p["records"] = [{"record_null": None, "record_number": 7, "record_text": "broken", "record_bad_list": ["bad"]}[fault]]
        return p
    monkeypatch.setattr(OperationStore, "task_params", corrupted)
    record_property("fault_boundary", "synthetic cached accessor; HTTP and real SQL worker")
    record_property("cache_fault", fault)
    h.trace["reservations"].clear(); h.frozen.clear()
    response = h.run(message, request_id=rid, conversation_id=cid)
    h.assert_no_write(response)
    assert h.trace["reservations"] == []
    assert OperationStore(rid).load() == before


def seed_two_tasks(h):
    from app.schemas.chat import ChatRequest
    from app.services.memory import compact
    from app.services.agent.operation_store import OperationStore, workout_operation_identity
    rid, cid = str(uuid4()), str(uuid4())
    message = "记录卧推60kg4组8次，深蹲80kg3组5次，然后记录划船30kg2组10次"
    tid, _ = compact.begin_turn(ChatRequest(message=message, request_id=rid, conversation_id=cid, user_id="quality-user"))
    store = OperationStore(rid)
    tasks = [{"id": task, "task_type": "workout_record", "worker": "workout_record", "text": source,
              "action": "record", "side_effect": True, "depends_on": [], "payload": {"write_authorization": "authorized"}}
             for task, source in (("t1", "记录卧推60kg4组8次，深蹲80kg3组5次"), ("t1-extra", "记录划船30kg2组10次"))]
    store.save_plan(tasks, time_base=datetime(2026, 10, 3, 3, tzinfo=timezone.utc))
    values = {"t1": [record(date="2026-10-03"), record(exercise="深蹲", weight_kg=80, sets=3, reps=5, date="2026-10-03")],
              "t1-extra": [record(exercise="划船", weight_kg=30, sets=2, reps=10, date="2026-10-03")]}
    saved = {}
    for task in tasks:
        task_id = task["id"]
        store.save_task_params(task_id, {"kind": "workout", "records": values[task_id], "source_text": task["text"], "llm_attempts": 0})
        keyed = {**values[task_id][0], "idempotency_key": f"workout-task-{rid}-{task_id}-0"}
        op_id, fingerprint = workout_operation_identity(keyed)
        store.reserve_operation(op_id, kind="workout", key=keyed["idempotency_key"], fingerprint=fingerprint, params=keyed)
        saved[task_id] = h.agent.graph.tools.save_workout_record(user_id="quality-user", record=keyed)["record"]
        store.mark_operation(op_id, "saved", result=saved[task_id])
    compact.fail_turn(tid)
    return message, rid, cid, store, saved


@pytest.mark.parametrize("fault", ["prefix_boundary", "ledger_read_error"])
def test_corrupt_identity_preserves_only_own_trusted_results(chat_db, monkeypatch, record_property, fault):
    from app.services.agent.operation_store import OperationStore, OperationStoreUnavailable
    h = AcceptanceHarness(monkeypatch, chat_db, property_writer=record_property)
    message, rid, cid, store, saved = seed_two_tasks(h)
    before = deepcopy(store.load())
    original_params, original_load = OperationStore.task_params, OperationStore.load
    def corrupted(s, task):
        p = deepcopy(original_params(s, task)); p["positions"] = None
        return p
    monkeypatch.setattr(OperationStore, "task_params", corrupted)
    hits = []
    if fault == "ledger_read_error":
        def failed_load(s):
            if inspect.currentframe().f_back.f_code.co_name == "_saved_operations_for_task":
                hits.append(s.request_id)
                raise OperationStoreUnavailable("synthetic-completed-recovery-read-fault")
            return original_load(s)
        monkeypatch.setattr(OperationStore, "load", failed_load)
    h.trace["reservations"].clear(); h.frozen.clear()
    response = h.run(message, request_id=rid, conversation_id=cid)
    after = OperationStore(rid).load()
    record_property("ledger_before", json.dumps(before, ensure_ascii=False, default=str))
    record_property("ledger_after", json.dumps(after, ensure_ascii=False, default=str))
    assert h.payloads == [] and len(h.rows()) == 2 and h.trace["reservations"] == []
    assert after == before
    target = next(w["result"]["worker_results"][-1] for w in h.trace["workers"] if w["task"]["id"] == "t1")
    record_property("target_task_result", json.dumps(target, ensure_ascii=False, default=str))
    if fault == "prefix_boundary":
        assert [r["id"] for r in target.get("records", [])] == [saved["t1"]["id"]], target
    else:
        assert hits, "read fault did not reach damaged-identity completion recovery"
        assert any(s in target["answer"] for s in ("账本", "无法核验", "恢复受限", "读取失败")), target
        assert "本次没有" not in target["answer"], target
