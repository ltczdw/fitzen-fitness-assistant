"""Independent HTTP/worker acceptance; no business validator is called directly."""
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text
from evals.v2.safety import deny_network
from test_chat_http_write_guard import ChatHTTPHarness, chat_db, plan


class AcceptanceHarness(ChatHTTPHarness):
    def __init__(self, monkeypatch, engine, parsed=None, property_writer=lambda *a: None):
        self.payloads = []
        self.frozen = []
        self.expect_no_workout_freeze = False
        self.property_writer = property_writer
        super().__init__(monkeypatch, engine, parsed)
        from app.services.agent.operation_store import OperationStore
        original = OperationStore.save_task_params
        def capture_freeze(store, task_id, params):
            self.frozen.append(deepcopy(params))
            return original(store, task_id, params)
        monkeypatch.setattr(OperationStore,"save_task_params",capture_freeze)

    async def call_tool(self, name, arguments, **kwargs):
        if name == "save_workout_record":
            self.payloads.append(deepcopy(arguments["record"]))
        return await super().call_tool(name, arguments, **kwargs)

    async def request(self, message, *, request_id=None, conversation_id=None):
        from httpx import ASGITransport, AsyncClient
        with deny_network() as audit:
            async with AsyncClient(transport=ASGITransport(app=self.app, raise_app_exceptions=False),
                                   base_url="http://api.test") as client:
                response = await client.post("/api/chat", json={"message": message,
                    "request_id": request_id or str(uuid4()), "conversation_id": conversation_id or str(uuid4()),
                    "user_id": "quality-user"})
            assert audit["blocked_connections"] == audit["blocked_dns"] == 0, audit
        self.property_writer("http_status", str(response.status_code))
        self.property_writer("save_dispatches", str(self.calls.count("save_workout_record")))
        self.property_writer("workout_rows", str(len(self.rows())))
        self.property_writer("save_payloads", json.dumps(self.payloads, ensure_ascii=False, default=str))
        self.property_writer("frozen_params", json.dumps(self.frozen, ensure_ascii=False, default=str))
        self.write_evidence(self.property_writer, response.json() if response.status_code == 200 else {"status": response.status_code, "body": response.text})
        assert response.status_code == 200, response.text
        return response.json()

    def assert_no_write(self, response):
        super().assert_no_write(response)
        if self.expect_no_workout_freeze:
            assert not any(p.get("kind") == "workout" for p in self.frozen), self.frozen


def record(**overrides):
    return {"exercise": "卧推", "sets": 4, "reps": 8, "weight_kg": 60, **overrides}


def make_harness(monkeypatch, db, message, records, record_property):
    parsed = plan(message, records[0])
    parsed["tasks"][0]["workouts"] = records
    harness = AcceptanceHarness(monkeypatch, db, parsed, record_property)
    harness.expect_no_workout_freeze = True
    return harness


@pytest.mark.parametrize("message,weight", [
    ("记录卧推60kg，4组8次", 4), ("记录卧推60kg，4组8次", 8),
    ("记录卧推120斤，4组8次", 120), ("记录卧推100磅，4组8次", 100),
    ("记录卧推60，4组8次", 4),
])
def test_r011_weight_requires_weight_role_and_unit(chat_db, monkeypatch, record_property, message, weight):
    h = make_harness(monkeypatch, chat_db, message, [record(weight_kg=weight)], record_property)
    h.assert_no_write(h.run(message))


@pytest.mark.parametrize("mode", ["weight", "counts", "whole_action"])
def test_r011_values_are_bound_to_action_in_one_span(chat_db, monkeypatch, record_property, mode):
    message = "记录卧推60kg，4组8次，深蹲80kg，3组5次"
    bench = record()
    if mode == "weight": bench["weight_kg"] = 80
    if mode == "counts": bench.update(sets=3, reps=5)
    if mode == "whole_action": bench.update(weight_kg=80, sets=3, reps=5)
    h = make_harness(monkeypatch, chat_db, message, [bench], record_property)
    h.assert_no_write(h.run(message))


@pytest.mark.parametrize("details", [
    [{"weight_kg": 62.5, "reps": 6}, {"weight_kg": 60, "reps": 8}],
    [{"weight_kg": 60, "reps": 6}, {"weight_kg": 62.5, "reps": 8}],
    [{"weight_kg": 60, "reps": 8}, {"weight_kg": 60, "reps": 8}],
], ids=["swapped_positions", "mismatched_pairs", "duplicated_first"])
def test_r011_set_facts_require_position_and_pairing(chat_db, monkeypatch, record_property, details):
    message = "记录卧推共2组，第一组60kg8次，第二组62.5kg6次"
    h = make_harness(monkeypatch, chat_db, message, [record(sets=2, set_details=details)], record_property)
    h.assert_no_write(h.run(message))


@pytest.mark.parametrize("message,expected", [
    ("记录杠铃卧推60kg，4组8次", (60, 4, 8)),
    ("记录卧推六十公斤，四组八次", (60, 4, 8)),
    ("记录卧推120斤，4组8次", (60, 4, 8)),
    ("记录卧推100磅，4组8次", (45.359237, 4, 8)),
    ("记录自重俯卧撑3组10次", (0, 3, 10)),
    ("记录卧推60kg，4x8", (60, 4, 8)),
    ("记录卧推60kg，4组每组8", (60, 4, 8)),
    ("记录今天的训练，我做了4组，每组8的10kg哑铃卧推", (10, 4, 8)),
])
def test_r011_supported_legal_controls(chat_db, monkeypatch, record_property, message, expected):
    # The model owns the record decision; the application still proves every
    # number/name against the frozen source before the real HTTP save.
    exercise = ("哑铃卧推" if "哑铃卧推" in message else "杠铃卧推" if "杠铃卧推" in message
                else "俯卧撑" if "俯卧撑" in message else "卧推")
    candidate = record(exercise=exercise, weight_kg=expected[0], sets=expected[1], reps=expected[2])
    h = AcceptanceHarness(monkeypatch, chat_db, plan(message, candidate), record_property)
    h.run(message)
    rows = h.rows()
    assert len(rows) == 1, rows
    assert float(rows[0][1]) == pytest.approx(expected[0], abs=.01)
    assert (rows[0][2], rows[0][3]) == expected[1:]
    assert len(h.payloads) == 1


def test_r011_legal_heterogeneous_facts_survive_freeze_save_read(chat_db, monkeypatch, record_property):
    from app.services.agent.operation_store import OperationStore
    message = "记录杠铃卧推共2组，第一组60kg8次，第二组62.5kg6次"
    details = [{"weight_kg": 60, "reps": 8}, {"weight_kg": 62.5, "reps": 6}]
    h = make_harness(monkeypatch, chat_db, message, [record(sets=2, set_details=details)], record_property)
    frozen = []
    original = OperationStore.save_task_params
    def spy(store, task_id, params):
        frozen.append(deepcopy(params))
        return original(store, task_id, params)
    monkeypatch.setattr(OperationStore, "save_task_params", spy)
    h.run(message)
    facts = lambda items: [(x["weight_kg"], x["reps"]) for x in items]
    assert len(frozen) == len(h.payloads) == len(h.rows()) == 1
    assert facts(frozen[0]["records"][0]["set_details"]) == facts(details)
    assert facts(h.payloads[0]["set_details"]) == facts(details)
    assert facts(json.loads(h.rows()[0][4])) == facts(details)


@pytest.mark.parametrize("field,value,message", [
    ("weight_kg", True, "记录卧推1kg，4组8次"),
    ("sets", True, "记录卧推60kg，1组8次"),
    ("reps", True, "记录卧推60kg，4组1次"),
    ("reps", 8.5, "记录卧推60kg，4组8次"),
    ("sets", None, "记录卧推60kg，4组8次"),
    ("weight_kg", float("nan"), "记录卧推60kg，4组8次"),
    ("weight_kg", float("inf"), "记录卧推60kg，4组8次"),
    ("weight_kg", "not-a-number", "记录卧推60kg，4组8次"),
    ("set_details", [{"weight_kg": 60, "reps": None}], "记录卧推60kg，4组8次"),
], ids=["bool_weight", "bool_sets", "bool_reps", "fractional_reps", "null_sets",
        "nan_weight", "infinite_weight", "text_weight", "invalid_details"])
def test_r011_invalid_model_values_never_reach_save(chat_db, monkeypatch, record_property, field, value, message):
    h = make_harness(monkeypatch, chat_db, message, [record(**{field: value})], record_property)
    h.assert_no_write(h.run(message))


def seed_cache(h, message, records, *, source=None, validated=None, completed=()):
    """Persist real synthetic ledger state, then resume through HTTP, not a mocked worker."""
    from app.schemas.chat import ChatRequest
    from app.services.memory import compact
    from app.services.agent.operation_store import OperationStore, workout_operation_identity
    rid, cid = str(uuid4()), str(uuid4())
    request = ChatRequest(message=message, request_id=rid, conversation_id=cid, user_id="quality-user")
    tid, _ = compact.begin_turn(request)
    store = OperationStore(rid)
    task = {"id": "t1", "task_type": "workout_record", "worker": "workout_record", "text": message,
            "action": "record", "side_effect": True, "depends_on": [], "payload": {"write_authorization": "authorized"}}
    base = datetime(2026, 10, 3, 3, tzinfo=timezone.utc)
    store.save_plan([task], time_base=base)
    params = {"kind": "workout", "records": records, "llm_attempts": 0}
    if source is not None: params["source_text"] = source
    if validated is not None: params["validated"] = validated
    store.save_task_params("t1", params)
    for index in completed:
        keyed = {**records[index], "idempotency_key": f"workout-task-{rid}-t1-{index}"}
        op_id, fingerprint = workout_operation_identity(keyed)
        store.reserve_operation(op_id, kind="workout", key=keyed["idempotency_key"], fingerprint=fingerprint, params=keyed)
        saved = h.agent.graph.tools.save_workout_record(user_id="quality-user", record=keyed)["record"]
        store.mark_operation(op_id, "saved", result=saved)
    compact.fail_turn(tid)
    return rid, cid, store


@pytest.mark.parametrize("validated", [None, True], ids=["legacy_source_conflict", "inconsistent_validated_marker"])
def test_r011_cache_source_must_match_frozen_authorized_plan(chat_db, monkeypatch, record_property, validated):
    message = "记录卧推60kg，4组8次"
    forged = record(weight_kg=999, date="2026-10-03")
    h = AcceptanceHarness(monkeypatch, chat_db, property_writer=record_property)
    rid, cid, _ = seed_cache(h, message, [forged], source="记录卧推999kg，4组8次", validated=validated)
    h.assert_no_write(h.run(message, request_id=rid, conversation_id=cid))


def test_r011_legacy_filter_keeps_original_operation_position(chat_db, monkeypatch, record_property):
    message = "记录卧推60kg，4组8次，深蹲80kg，3组5次"
    records = [record(weight_kg=999, date="2026-10-03"),
               record(exercise="深蹲", weight_kg=80, sets=3, reps=5, date="2026-10-03")]
    h = AcceptanceHarness(monkeypatch, chat_db, property_writer=record_property)
    rid, cid, _ = seed_cache(h, message, records)
    response = h.run(message, request_id=rid, conversation_id=cid)
    assert len(h.rows()) == len(h.payloads) == 1
    assert h.payloads[0]["exercise"] == "深蹲"
    assert h.payloads[0]["idempotency_key"] == f"workout-task-{rid}-t1-1", h.payloads
    assert any(x in response["answer"] for x in ("无法", "未保存", "来源")), response


def test_r011_legacy_completed_result_preserved_without_resave(chat_db, monkeypatch, record_property):
    message = "记录卧推60kg，4组8次，深蹲80kg，3组5次"
    records = [record(date="2026-10-03"), record(exercise="深蹲", weight_kg=999, sets=3, reps=5, date="2026-10-03")]
    h = AcceptanceHarness(monkeypatch, chat_db, property_writer=record_property)
    rid, cid, store = seed_cache(h, message, records, completed=(0,))
    before = h.rows()
    response = h.run(message, request_id=rid, conversation_id=cid)
    assert h.rows() == before
    assert h.payloads == [], h.payloads
    assert any(x in response["answer"] for x in ("未保存", "无法", "来源")), response
    operations = __import__("app.services.agent.operation_store", fromlist=["OperationStore"]).OperationStore(rid).load()["operations"]
    assert any(op["status"] == "saved" and op["result"]["exercise"] == "卧推" for op in operations.values())


def test_r011_legacy_completed_after_invalid_position_is_not_resaved(chat_db, monkeypatch, record_property):
    message = "记录卧推60kg，4组8次，深蹲80kg，3组5次"
    records = [record(weight_kg=999,date="2026-10-03"),
        record(exercise="深蹲",weight_kg=80,sets=3,reps=5,date="2026-10-03")]
    h = AcceptanceHarness(monkeypatch,chat_db,property_writer=record_property)
    rid,cid,_ = seed_cache(h,message,records,completed=(1,))
    before = h.rows()
    h.run(message,request_id=rid,conversation_id=cid)
    assert h.payloads == [] and h.rows() == before, {"payloads":h.payloads,"rows":h.rows()}


def test_r011_partial_cache_pending_record_keeps_original_index(chat_db, monkeypatch, record_property):
    message = "记录卧推60kg，4组8次，深蹲80kg，3组5次"
    records = [record(date="2026-10-03"),record(weight_kg=999,date="2026-10-03"),
        record(exercise="深蹲",weight_kg=80,sets=3,reps=5,date="2026-10-03")]
    h = AcceptanceHarness(monkeypatch,chat_db,property_writer=record_property)
    rid,cid,_ = seed_cache(h,message,records,completed=(0,))
    h.run(message,request_id=rid,conversation_id=cid)
    assert len(h.rows()) == 2 and len(h.payloads) == 1
    assert h.payloads[0]["idempotency_key"] == f"workout-task-{rid}-t1-2", h.payloads


@pytest.mark.parametrize("fault", ["unreadable", "missing"])
def test_r011_cache_plan_unreadable_blocks_pending_write(chat_db, monkeypatch, record_property, fault):
    from app.services.agent.operation_store import OperationStore, OperationStoreUnavailable
    message = "记录卧推60kg，4组8次"
    h = AcceptanceHarness(monkeypatch, chat_db, property_writer=record_property)
    rid, cid, _ = seed_cache(h, message, [record(date="2026-10-03")])
    original, reads = OperationStore.stored_plan, []
    def unreadable_after_routing(store):
        reads.append(store.request_id)
        if len(reads) > 1:
            if fault == "unreadable": raise OperationStoreUnavailable("quality-plan-read-failure")
            return None
        return original(store)
    monkeypatch.setattr(OperationStore, "stored_plan", unreadable_after_routing)
    h.assert_no_write(h.run(message, request_id=rid, conversation_id=cid))
    assert len(reads) >= 2


@pytest.mark.parametrize("field,value", [
    ("sets", float("nan")), ("sets", float("inf")),
    ("weight_kg", "not-a-number"), ("reps", None),
], ids=["nan_sets", "infinite_sets", "text_weight", "null_reps"])
def test_r011_malformed_legacy_accessor_result_fails_closed(chat_db, monkeypatch, record_property, field, value):
    from app.services.agent.operation_store import OperationStore
    message = "记录卧推60kg，4组8次"
    h = AcceptanceHarness(monkeypatch, chat_db, property_writer=record_property)
    rid, cid, _ = seed_cache(h, message, [record(date="2026-10-03")])
    original = OperationStore.task_params
    def damaged_cache(store, task_id):
        params = deepcopy(original(store, task_id))
        params["records"][0][field] = value
        return params
    monkeypatch.setattr(OperationStore, "task_params", damaged_cache)
    record_property("cache_corruption_boundary", "mocked storage accessor, persisted plan retained")
    h.assert_no_write(h.run(message, request_id=rid, conversation_id=cid))


def test_r011_cached_operation_parameter_conflict_zero_dispatch(chat_db, monkeypatch, record_property):
    from app.services.agent.operation_store import workout_operation_identity
    message = "记录卧推60kg，4组8次"
    h = AcceptanceHarness(monkeypatch, chat_db, property_writer=record_property)
    cached = record(date="2026-10-03")
    rid, cid, store = seed_cache(h, message, [cached], source=message, validated=True)
    previous = {**cached, "weight_kg": 65, "idempotency_key": f"workout-task-{rid}-t1-0"}
    op_id, fingerprint = workout_operation_identity(previous)
    store.reserve_operation(op_id, kind="workout", key=previous["idempotency_key"], fingerprint=fingerprint, params=previous)
    response = h.run(message, request_id=rid, conversation_id=cid)
    assert "冲突" in response["answer"], response
    assert h.payloads == [] and h.rows() == []


def test_r011_verified_cache_retry_preserves_clock_params_and_identity(chat_db, monkeypatch, record_property):
    from app.core import clock
    from app.services.memory import compact
    from app.services.agent.operation_store import OperationStore
    message = "记录昨天卧推60kg，4组8次"
    initial = datetime(2026, 10, 3, 3, tzinfo=timezone.utc)
    monkeypatch.setattr(clock, "utc_now", lambda: initial)
    h = AcceptanceHarness(monkeypatch, chat_db, plan(message, record()), record_property)
    rid, cid = str(uuid4()), str(uuid4())
    h.run(message, request_id=rid, conversation_id=cid)
    state = OperationStore(rid).load()
    before = deepcopy(state)
    assert len(h.rows()) == 1
    # Simulate interrupted final response, leaving verified parameters and completed operation.
    from app.db import session as db
    from app.db.memory_models import MemoryTurn
    from sqlalchemy import select
    with db.SessionLocal() as s, s.begin():
        row = s.scalar(select(MemoryTurn).where(MemoryTurn.request_id == rid))
        row.status = "failed"; row.response = None
    monkeypatch.setattr(clock, "utc_now", lambda: initial + timedelta(days=1))
    extractor = AsyncMock(side_effect=AssertionError("verified cache must not re-extract"))
    monkeypatch.setattr(h.agent.graph.workout_extraction, "run_many", extractor)
    h.run(message, request_id=rid, conversation_id=cid)
    after = OperationStore(rid).load()
    assert len(h.rows()) == len(h.payloads) == 1
    assert after["time_base"] == before["time_base"]
    assert after["tasks"] == before["tasks"]
    assert after["operations"] == before["operations"]
    extractor.assert_not_awaited()
    with chat_db.connect() as conn:
        assert str(conn.execute(text("SELECT workout_date FROM workout_records")).scalar()) == "2026-10-02"
