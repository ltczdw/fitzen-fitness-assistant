"""Real ASGI entry, deterministic model outputs, isolated SQLite/PostgreSQL.
Red R011 tests are defect evidence until business implementation is repaired.
"""
import asyncio
import json
from copy import deepcopy
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from quality_isolation import isolated_database
from evals.v2.safety import deny_network


@pytest.fixture(params=["sqlite", "postgres"])
def chat_db(request, monkeypatch, tmp_path, record_property):
    with isolated_database(monkeypatch, tmp_path, request.param) as engine:
        record_property("dialect", engine.dialect.name)
        record_property("database", engine.url.database)
        yield engine


class ChatHTTPHarness:
    def __init__(self, monkeypatch, engine, parsed=None):
        import app.api.chat as chat_module
        import app.main as app_main
        from app.core.config import settings
        from app.services.agent import execution
        self.app = app_main.app
        self.agent = chat_module.agent
        self.engine = engine
        self.calls = []
        self.trace = {"routes": [], "workers": [], "source_checks": [],
                      "plans": [], "freezes": [], "reservations": [], "dispatches": []}
        from app.services.agent.operation_store import OperationStore
        import app.services.agent.multi_agent_graph as graph_module
        original_route = self.agent._route
        async def route(*args, **kwargs):
            decision = await original_route(*args, **kwargs)
            self.trace["routes"].append(deepcopy({"degraded_reason": decision.degraded_reason,
                "parsed_input": decision.parsed_input, "tasks": decision.task_specs}))
            return decision
        monkeypatch.setattr(self.agent, "_route", route)
        original_worker = self.agent.graph._run_worker
        async def worker(name, state):
            self.trace["workers"].append(deepcopy({"name": name, "task": state.get("current_task"),
                "raw_message": state.get("raw_message"), "tasks": state.get("tasks")}))
            result = await original_worker(name, state)
            self.trace["workers"][-1]["result"] = deepcopy(result)
            return result
        monkeypatch.setattr(self.agent.graph, "_run_worker", worker)
        original_check = graph_module.validate_workout_records_against_source
        def check(records, source):
            captured = {"source": source, "candidates": deepcopy(records)}
            kept, rejected = original_check(records, source)
            captured.update(kept=deepcopy(kept), rejected=deepcopy(rejected))
            self.trace["source_checks"].append(captured)
            return kept, rejected
        monkeypatch.setattr(graph_module, "validate_workout_records_against_source", check)
        for method, bucket in (("save_plan", "plans"), ("save_task_params", "freezes"),
                               ("reserve_operation", "reservations")):
            original = getattr(OperationStore, method)
            def capture(store, *args, _original=original, _bucket=bucket, **kwargs):
                self.trace[_bucket].append(deepcopy({"args": args, "kwargs": kwargs}))
                return _original(store, *args, **kwargs)
            monkeypatch.setattr(OperationStore, method, capture)
        monkeypatch.setattr(self.agent.graph, "parse_input", AsyncMock(return_value=deepcopy(parsed)))
        async def extract(message):
            return fixture_workout_extraction(message)
        monkeypatch.setattr(self.agent.graph.workout_extraction, "run_many", extract)
        async def semantic(messages, schema, *, name=None, **kwargs):
            if name != "return_request_understanding":
                raise AssertionError(name)
            payload = json.loads(messages[-1]["content"])
            query = str(payload.get("query") or "")
            return schema.model_validate({
                "tasks": [{"task_type": "training_qa", "action": "question", "text": query,
                           "is_negated": False, "is_hypothetical": False,
                           "is_third_party": False}],
                "safety": {"level": "normal", "evidence": "", "categories": [],
                           "reason": "scripted normal"},
            })
        monkeypatch.setattr(self.agent.graph.llm, "configured", lambda: True)
        monkeypatch.setattr(self.agent.graph.llm, "complete_structured", semantic)
        monkeypatch.setattr(self.agent, "_run_planner_shadow", AsyncMock())
        monkeypatch.setattr(execution.memory_service, "load", AsyncMock(return_value={"memory_backend": "fixture"}))
        monkeypatch.setattr(self.agent.graph.rag, "search", lambda *a, **k: [])
        monkeypatch.setattr(self.agent.graph, "_run_authorized_model_tool", AsyncMock(return_value=None))
        monkeypatch.setattr(settings, "planner_shadow_enabled", False)
        monkeypatch.setattr(self.agent.graph, "_call_mcp", self.call_tool)

    async def call_tool(self, name, arguments, **kwargs):
        self.calls.append(name)
        self.trace["dispatches"].append(deepcopy({"name": name, "arguments": arguments}))
        if name in {"get_user_profile", "get_nutrition_summary"}:
            return {}
        if name == "get_training_history":
            return []
        return getattr(self.agent.graph.tools, name)(**arguments)

    async def request(self, message, *, request_id=None, conversation_id=None):
        from httpx import AsyncClient, ASGITransport
        with deny_network() as audit:
            async with AsyncClient(transport=ASGITransport(app=self.app), base_url="http://api.test") as client:
                response = await client.post("/api/chat", json={
                    "message": message, "request_id": request_id or str(uuid4()),
                    "conversation_id": conversation_id or str(uuid4()), "user_id": "quality-user"})
            assert audit["blocked_connections"] == audit["blocked_dns"] == 0, audit
        assert response.status_code == 200, response.text
        return response.json()

    def run(self, message, **kwargs):
        return asyncio.run(self.request(message, **kwargs))

    def rows(self):
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT exercise, weight_kg, sets, reps, set_details FROM workout_records ORDER BY id")).all()

    def write_evidence(self, record_property, response=None):
        record_property("actual_application_trace", json.dumps(self.trace, ensure_ascii=False, default=str))
        record_property("actual_workout_sql", json.dumps([list(r) for r in self.rows()], ensure_ascii=False, default=str))
        if response is not None:
            record_property("actual_http_response", json.dumps(response, ensure_ascii=False, default=str))

    def assert_structured_path(self, source, candidates):
        assert any(r["degraded_reason"] is None and r["parsed_input"].get("authorization_validated")
                   for r in self.trace["routes"]), self.trace["routes"]
        assert any(w["name"] == "workout_record" for w in self.trace["workers"]), self.trace["workers"]
        assert any(c["source"] == source and
                   [{k: r.get(k) for k in ("exercise", "weight_kg", "sets", "reps")} for r in c["candidates"]] ==
                   [{k: r.get(k) for k in ("exercise", "weight_kg", "sets", "reps")} for r in candidates]
                   for c in self.trace["source_checks"]), self.trace["source_checks"]

    def assert_no_write(self, response):
        saves = [n for n in self.calls if n in {"save_workout_record", "calculate_nutrition", "calculate_nutrition_batch"}]
        with self.engine.connect() as conn:
            counts = [conn.execute(text("SELECT COUNT(*) FROM " + table)).scalar()
                      for table in ("workout_records", "user_food_log")]
        # Collect all evidence before checking so one failed assertion does not hide SQL state.
        assert saves == [] and counts == [0, 0], {"saves": saves, "row_counts": counts}
        answer = response["answer"]
        assert any(word in answer for word in ("确认", "补充", "缺少", "未保存", "未记录", "不能", "不足", "不保存", "无法")), answer
        assert "已记录" not in answer, answer


def plan(message, record):
    return {"tasks": [{"task_type": "workout_record", "action": "record", "text": message,
                       "workouts": [record], "is_negated": False, "is_hypothetical": False,
                       "is_third_party": False}]}


def workout_plan(message):
    """Semantic record decision with no model entities: extraction still applies."""
    return {"tasks": [{"task_type": "workout_record", "action": "record", "text": message,
                       "is_negated": False, "is_hypothetical": False, "is_third_party": False}]}


def fixture_workout_extraction(message):
    """Deterministic entity fixture for the save-pipeline cases.

    The R011 cases below cover source/authorization boundaries, not the remote
    extraction model. Every returned record is still checked by the real
    ``validate_workout_records_against_source`` gate.
    """
    def item(exercise, weight, sets, reps, set_details=None):
        record = {"exercise": exercise, "weight_kg": weight, "sets": sets, "reps": reps}
        if set_details is not None:
            record["set_details"] = set_details
        return {"record": record, "missing_fields": [], "llm_attempts": 0}

    if "第一组" in message:
        return [item("卧推", 60, 2, 8,
                     [{"weight_kg": 60, "reps": 8}, {"weight_kg": 62.5, "reps": 6}])]
    if "自重俯卧撑" in message:
        return [item("俯卧撑", 0, 3, 10)]
    if "120斤" in message or "六十公斤" in message or "卧推60kg" in message:
        return [item("卧推", 60, 4, 8)]
    return []


@pytest.mark.parametrize("case,message,record", [
    ("all", "帮我记录一下今天训练", {"exercise": "卧推", "weight_kg": 999, "sets": 4, "reps": 8}),
    ("weight", "记录卧推60kg，4组8次", {"exercise": "卧推", "weight_kg": 999, "sets": 4, "reps": 8}),
    ("sets", "记录卧推60kg，4组8次", {"exercise": "卧推", "weight_kg": 60, "sets": 9, "reps": 8}),
    ("reps", "记录卧推60kg，4组8次", {"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 20}),
])
def test_r011_forged_entities_zero_dispatch_zero_rows(chat_db, monkeypatch, case, message, record):
    harness = ChatHTTPHarness(monkeypatch, chat_db, plan(message, record))
    harness.assert_no_write(harness.run(message))


def test_r011_cross_task_entity_cannot_be_borrowed(chat_db, monkeypatch, record_property):
    source = "记录深蹲80kg，3组8次"
    parsed = plan(source, {"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8})
    parsed["tasks"].insert(0, {"task_type": "training_qa", "action": "question",
                              "text": "卧推60kg4组8次怎么发力", "is_negated": False,
                              "is_hypothetical": False, "is_third_party": False})
    harness = ChatHTTPHarness(monkeypatch, chat_db, parsed)
    response = harness.run("卧推60kg4组8次怎么发力，然后" + source)
    harness.write_evidence(record_property, response)
    harness.assert_structured_path(source, parsed["tasks"][1]["workouts"])
    assert not harness.trace["freezes"] and not harness.trace["reservations"], harness.trace
    harness.assert_no_write(response)


def test_r011_invalid_cross_task_fixture_has_no_rule_write_repair(chat_db, monkeypatch, record_property):
    source = "记录深蹲80kg，3组8次"
    parsed = plan(source, {"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8})
    parsed["tasks"].insert(0, {"task_type": "training_qa", "action": "question",
                              "text": "卧推60kg4组8次怎么发力"})
    h = ChatHTTPHarness(monkeypatch, chat_db, parsed)
    response = h.run("卧推60kg4组8次怎么发力，然后" + source)
    h.write_evidence(record_property, response)
    # An invalid model plan no longer falls back to an authorized rule write.
    assert h.trace["routes"][0]["degraded_reason"], h.trace
    assert not [d for d in h.trace["dispatches"] if d["name"] == "save_workout_record"]
    assert h.rows() == []
    h.assert_no_write(response)


def test_r011_unvalidated_old_stored_params_cannot_dispatch(chat_db, monkeypatch):
    from app.services.agent.operation_store import OperationStore
    message = "帮我记录一下今天训练"
    forged = {"exercise": "卧推", "weight_kg": 999, "sets": 4, "reps": 8}
    harness = ChatHTTPHarness(monkeypatch, chat_db, plan(message, forged))
    seen = []
    def old_params(store, task_id):
        seen.append(task_id)
        return {"kind": "workout", "records": [forged], "llm_attempts": 0}
    monkeypatch.setattr(OperationStore, "task_params", old_params)
    response = harness.run(message)
    assert seen, "worker never reached the cached-parameter boundary"
    harness.assert_no_write(response)


def test_r011_invalid_params_not_frozen_before_validation(chat_db, monkeypatch):
    from app.services.agent.operation_store import OperationStore
    message = "帮我记录一下今天训练"
    harness = ChatHTTPHarness(monkeypatch, chat_db, plan(message, {"exercise": "卧推", "weight_kg": 999, "sets": 4, "reps": 8}))
    original = OperationStore.save_task_params
    frozen = []
    def spy(store, task_id, params):
        frozen.append(deepcopy(params))
        return original(store, task_id, params)
    monkeypatch.setattr(OperationStore, "save_task_params", spy)
    response = harness.run(message)
    assert not frozen, "unvalidated model entities frozen in operation ledger"
    harness.assert_no_write(response)


@pytest.mark.parametrize("message,weight,sets,reps", [
    ("记录卧推60kg，4组8次", 60, 4, 8),
    ("记录卧推六十公斤，四组八次", 60, 4, 8),
    ("记录自重俯卧撑，3组10次", 0, 3, 10),
    ("记录杠铃卧推60kg，4组8次", 60, 4, 8),
    ("记录卧推120斤，4组8次", 60, 4, 8),
])
def test_r011_legal_controls(chat_db, monkeypatch, message, weight, sets, reps):
    harness = ChatHTTPHarness(monkeypatch, chat_db, workout_plan(message))
    response = harness.run(message)
    rows = harness.rows()
    assert len(rows) == 1 and (rows[0][1], rows[0][2], rows[0][3]) == (weight, sets, reps), rows
    assert harness.calls.count("save_workout_record") == 1


def test_r011_legal_structured_alias_and_per_set_control(chat_db, monkeypatch):
    message = "记录杠铃卧推，第一组60公斤8次，第二组62.5公斤6次"
    record = {"exercise": "卧推", "weight_kg": 60, "sets": 2, "reps": 8,
              "set_details": [{"weight_kg": 60, "reps": 8}, {"weight_kg": 62.5, "reps": 6}]}
    harness = ChatHTTPHarness(monkeypatch, chat_db, plan(message, record))
    harness.run(message)
    import json
    rows = harness.rows()
    assert len(rows) == 1, rows
    assert rows[0][4] is not None, "structured per-set facts were dropped before saving"
    assert [(s["weight_kg"], s["reps"]) for s in json.loads(rows[0][4])] == [(60, 8), (62.5, 6)]


def test_r011_legal_replay_retains_id_and_dispatch_count(chat_db, monkeypatch):
    message = "记录卧推60kg，4组8次"
    harness = ChatHTTPHarness(monkeypatch, chat_db, workout_plan(message))
    rid, cid = str(uuid4()), str(uuid4())
    async def scenario():
        first = await harness.request(message, request_id=rid, conversation_id=cid)
        second = await harness.request(message, request_id=rid, conversation_id=cid)
        return first, second
    first, second = asyncio.run(scenario())
    assert len(harness.rows()) == 1
    assert harness.calls.count("save_workout_record") == 1
    assert second["replayed"] is True


@pytest.mark.parametrize("message", ["不记了，查看上次记录", "可以记录跑步吗", "记录卧推60kg，4组8次？", "要是昨天吃了炸鸡，今天是不是该少吃点？"])
def test_forbidden_inputs_zero_dispatch_zero_rows(chat_db, monkeypatch, message):
    harness = ChatHTTPHarness(monkeypatch, chat_db)
    harness.run(message)
    assert not any(n in {"save_workout_record", "calculate_nutrition", "calculate_nutrition_batch"} for n in harness.calls)
    with chat_db.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM workout_records")).scalar() == 0
        assert conn.execute(text("SELECT COUNT(*) FROM user_food_log")).scalar() == 0


@pytest.mark.parametrize("message,name,weight,parsed", [
    ("不要记录鸡蛋，然后记录100克米饭", "米饭", 100, {"tasks": [
        {"task_type": "nutrition_record", "action": "record", "text": "不要记录鸡蛋",
         "is_negated": True, "is_hypothetical": False, "is_third_party": False},
        {"task_type": "nutrition_record", "action": "record", "text": "记录100克米饭",
         "is_negated": False, "is_hypothetical": False, "is_third_party": False,
         "foods": [{"name": "米饭", "quantity": 100, "unit": "克", "weight_g": 100,
                    "meal_type": None, "record_date": None}]},
    ]}),
    ("记录两个鸡蛋，再安排晚餐", "鸡蛋", 100, {"tasks": [
        {"task_type": "nutrition_record", "action": "record", "text": "记录两个鸡蛋",
         "is_negated": False, "is_hypothetical": False, "is_third_party": False,
         "foods": [{"name": "鸡蛋", "quantity": 2, "unit": "个", "weight_g": None,
                    "meal_type": None, "record_date": None}]},
        {"task_type": "nutrition_advice", "action": "question", "text": "再安排晚餐",
         "is_negated": False, "is_hypothetical": False, "is_third_party": False},
    ]}),
])
def test_food_authorization_boundaries(chat_db, monkeypatch, message, name, weight, parsed):
    harness = ChatHTTPHarness(monkeypatch, chat_db, parsed)
    harness.run(message)
    with chat_db.connect() as conn:
        rows = conn.execute(text("SELECT raw_food_name, weight_g FROM user_food_log")).all()
    assert len(rows) == 1 and name in rows[0][0] and float(rows[0][1]) == weight, rows
    assert len(harness.rows()) == 0


def test_conversation_delete_lifecycle(chat_db, monkeypatch):
    harness = ChatHTTPHarness(monkeypatch, chat_db)
    cid = str(uuid4())
    message = "记录卧推60kg，4组8次"
    harness = ChatHTTPHarness(monkeypatch, chat_db, workout_plan(message))
    async def scenario():
        await harness.request(message, conversation_id=cid)
        from httpx import AsyncClient, ASGITransport
        with deny_network() as audit:
            async with AsyncClient(transport=ASGITransport(app=harness.app), base_url="http://api.test") as client:
                deleted = await client.delete("/api/chat/conversations/" + cid, params={"user_id": "quality-user"})
                after = await client.get("/api/chat/conversations", params={"user_id": "quality-user"})
                history = await client.get("/api/chat/history", params={"user_id": "quality-user", "conversation_id": cid})
            assert audit["blocked_connections"] == audit["blocked_dns"] == 0
        return deleted, after, history
    deleted, after, history = asyncio.run(scenario())
    assert deleted.status_code == 200
    assert cid not in [r["conversation_id"] for r in after.json()["conversations"]]
    assert history.status_code == 404 or not history.json().get("messages"), history.text
    assert len(harness.rows()) == 1
