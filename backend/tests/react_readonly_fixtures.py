"""Tests for react readonly fixtures."""
from __future__ import annotations

import asyncio
import copy
import inspect
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from app.core import clock
from app.core.config import settings
from tests.conftest import _USER_FOOD_LOG_DDL
from tests.react_supervisor_fixtures import tool_call

QUERY = "结合我的情况给我训练建议"
URGENT_SOURCE = "胸痛并呼吸困难，帮我查训练建议"
FINAL = "基于你的档案与已读取信息：建议每周训练三次，循序渐进。"
PREPARATION = "主管预备文字-不得流出"
FIXED_LOCAL = datetime(2026, 10, 5, 12, tzinfo=ZoneInfo("Asia/Shanghai"))
BUSINESS_TABLES = (
    "workout_records", "workout_sessions", "workout_session_exercises",
    "workout_session_sets", "nutrition_logs", "user_food_log",
    "training_plans", "user_profiles",
)


def pytest_sessionstart(session):
    # Optional -p plugin for this fixed B+A/regression run. Existing conftest,
    # test source and network assertions remain untouched.
    from evals.v2.safety import deny_network
    session._phase_b_network_context = deny_network()
    session._phase_b_network_audit = session._phase_b_network_context.__enter__()


def pytest_sessionfinish(session, exitstatus):
    audit = session._phase_b_network_audit
    session._phase_b_network_context.__exit__(None, None, None)
    emit("SESSION_ISOLATION", "network", **audit)
    if audit != {"blocked_connections": 0, "blocked_dns": 0}:
        session.exitstatus = 1


def obs(value):
    if inspect.isawaitable(value):
        value = asyncio.run(value)
    return value.to_dict() if hasattr(value, "to_dict") else value


def emit(case, control, **values):
    # Descriptive observations only; no full user/model messages, credentials,
    # connection strings or personal data are included in this evidence.
    print("B_EVIDENCE=" + json.dumps({"case": case, "control": control, **values},
                                    ensure_ascii=False, default=str, sort_keys=True))


def check_all(case, checks):
    emit(case, "assertions", checks=checks)
    failed = [name for name, value in checks.items() if not value]
    assert not failed, f"{case}: failed contract checks: {failed}"


def chunks(query=QUERY, *, sufficient=True):
    from app.services.rag.retrieval import RetrievedChunk
    return [RetrievedChunk(
        "训练学基础", query + "。循序渐进，逐步增加，每周训练。", 0.9,
        "phase-b-synthetic-document", {"rerank_status": "model",
                                      "rerank_score": 0.98 if sufficient else 0.01,
                                      "page": 7}, "phase-b-chunk-1")]


def answer_script():
    return [
        {"role": "assistant", "content": PREPARATION,
         "tool_calls": [tool_call("t1", "read_profile", {})]},
        {"role": "assistant", "content": PREPARATION,
         "tool_calls": [tool_call("t2", "answer_training", {"query": QUERY})]},
        {"role": "assistant", "content": FINAL},
    ]


def semantic_task(task_type, action, text, *, negated=False, hypothetical=False, third_party=False):
    """One scripted model decision; the application validator stays real."""
    return {
        "task_type": task_type, "action": action, "text": text,
        "is_negated": negated, "is_hypothetical": hypothetical,
        "is_third_party": third_party,
    }


def semantic_outline(source, tasks=None, safety=None):
    """Structured RequestUnderstanding payload for one synthetic request."""
    return {"tasks": [dict(task) for task in (
        tasks if tasks is not None else [semantic_task("training_qa", "question", source)])],
        "safety": dict(safety or {
            "level": "normal", "evidence": "", "categories": [],
            "reason": "scripted normal",
        })}


def semantic_query(messages):
    """Read the original query from dialogue_context.semantic_messages."""
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(str(message.get("content") or ""))
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("query"), str):
            return payload["query"]
    return ""


def default_semantics(source):
    """Controlled, schema/source-validated understanding for fixture requests."""
    if source == URGENT_SOURCE:
        return semantic_outline(
            source,
            [semantic_task("safety_triage", "question", source)],
            safety={"level": "urgent", "evidence": "胸痛并呼吸困难",
                    "categories": ["current_symptom"],
                    "reason": "scripted current symptom"})
    return semantic_outline(source, [semantic_task("training_qa", "question", source)])


def tool_results(messages):
    return [json.loads(m["content"]) for m in messages if m.get("role") == "tool"]


def parse_sse(response):
    result, current = [], None
    for line in response.iter_lines():
        if line.startswith("event:"):
            current = line.partition(":")[2].strip()
        elif line.startswith("data:"):
            assert current, "SSE data must follow event line"
            result.append({"event": current, "data": json.loads(line.partition(":")[2])})
    return result


def sse_body(events):
    dones = [e["data"] for e in events if e["event"] == "done"]
    assert len(dones) == 1, f"expected exactly one done: {[e['event'] for e in events]}"
    return {**dones[0], "answer": "".join(e["data"]["content"] for e in events
                                         if e["event"] == "token")}


class ScriptLLM:
    def __init__(self, env, script=None, policy=None, final=FINAL, semantic=default_semantics,
                 semantic_error=None, configured=True):
        self.env, self.script, self.policy = env, list(script or []), policy
        self.final, self.calls = final, []
        self._configured = bool(configured)
        if semantic is False:
            semantic = None
        self.semantic, self.semantic_error = semantic, semantic_error
        self.semantic_calls = []
        self.has_structured_adapter = semantic is not None or semantic_error is not None
        if self.has_structured_adapter:
            self.complete_structured = self._complete_structured

    def configured(self):
        return self._configured

    async def _complete_structured(self, messages, schema, *, name=None, temperature=0.0,
                                   validator=None, **kwargs):
        """Real protocol shape: schema-validated model result + production validator."""
        self.semantic_calls.append({"messages": copy.deepcopy(messages), "name": name,
                                    "temperature": temperature, "source": semantic_query(messages)})
        from app.services.telemetry import model_attempt
        with model_attempt("mock-scripted-structured", streaming=False) as attempt:
            attempt.observe({"usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                       "total_tokens": 2}})
            if self.semantic_error is not None:
                raise self.semantic_error
            raw = self.semantic(semantic_query(messages)) if callable(self.semantic) else self.semantic
            if isinstance(raw, Exception):
                raise raw
            value = raw if isinstance(raw, schema) else schema.model_validate(copy.deepcopy(raw))
            if validator is not None:
                validator(value)
            return value

    async def complete_message(self, messages, **kwargs):
        entry = {"messages": copy.deepcopy(messages), "tools": copy.deepcopy(kwargs.get("tools")),
                 "repos_before": self.env.counts(), "response": None}
        self.calls.append(entry)
        if kwargs.get("tools"):
            index = sum(bool(c["tools"]) for c in self.calls) - 1
            if self.policy:
                value = self.policy(index, messages)
                if inspect.isawaitable(value):
                    value = await value
            else:
                value = self.script[index] if index < len(self.script) else {
                    "role": "assistant", "content": FINAL}
        else:
            value = {"role": "assistant", "content": self.final}
        entry["response"] = copy.deepcopy(value)
        return copy.deepcopy(value)

    async def complete_with_tools(self, *args, **kwargs):
        self.env.forbidden.append("complete_with_tools")
        raise AssertionError("readonly worker must not nest the legacy tool loop")


class BEnv:
    def __init__(self, directory, mp, *, enabled=True):
        self.mp = mp
        directory.mkdir(parents=True, exist_ok=True)
        from app.db import session as db
        from app.db import (memory_models, model_budget_models, quality_models,
                            workout_session_models)  # noqa: F401
        from app.db.models import Base
        from app.main import app
        from app.services.agent import agent as am
        self.agent, self.graph, self.app = am.agent, am.agent.graph, app
        self.engine = create_engine("sqlite:///" + str(directory / "phaseb.db"))
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as conn:
            conn.execute(text(_USER_FOOD_LOG_DDL))
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        mp.setattr(db, "engine", self.engine)
        mp.setattr(db, "SessionLocal", self.factory)
        mp.setattr(db, "DB_READY", True)
        mp.setattr(settings, "memory_read_backend", "compact_v1")
        mp.setattr(settings, "react_readonly_enabled", enabled)
        mp.setattr(settings, "planner_shadow_enabled", False)
        mp.setattr(clock, "utc_now", lambda: FIXED_LOCAL.astimezone(timezone.utc))
        mp.setattr(settings, "app_timezone", "Asia/Shanghai")
        # External memory retrieval is explicit mock; durable begin/finish are real.
        self.load = AsyncMock(return_value={"memory_backend": "compact_v1"})
        mp.setattr(am.memory_service, "load", self.load)
        mp.setattr(am.memory_service, "append_turn", AsyncMock())
        mp.setattr(am.memory_service, "remember_explicit", AsyncMock())
        # Existing parser/shadow model boundaries are mock and are counted, so a
        # forbidden readonly parse cannot disappear from the evidence.
        self.parse = AsyncMock(return_value=None)
        self.shadow = AsyncMock(return_value=(None, "unavailable"))
        mp.setattr(self.graph, "parse_input", self.parse)
        mp.setattr(self.graph, "plan_shadow", self.shadow)
        self.legacy = AsyncMock(return_value={"worker_results": [{"status": "complete",
                                "answer": "已按legacy流程处理。"}], "citations": [],
                                "model_used": False, "safety_level": "normal"})
        mp.setattr(self.graph, "run", self.legacy)
        self.forbidden = []
        self.repos = {"profile": [], "workouts": [], "nutrition": [], "plan": []}
        self.handlers, self.states, self.sql, self.mode_calls = [], [], [], []
        self.lifecycle = {"begin": [], "finish": [], "fail": [], "diagnostics": []}
        self._observe()
        self.mock_rag(chunks())
        self.llm = self.mock_llm(answer_script())
        # compact lock_profile may create this infrastructure anchor. Include it
        # in the seed baseline; no business data loads occur while preparing.
        self.seed_profile()
        self.reset_counts()
        self.client = TestClient(app)

    def _observe(self):
        from app.db import profile_repository, plan_repository, nutrition_sync, repositories
        for key, module, attr in (("profile", profile_repository, "get_profile"),
                                  ("workouts", repositories, "list_workouts"),
                                  ("nutrition", nutrition_sync, "synced_nutrition_summary"),
                                  ("plan", plan_repository, "get_plan")):
            original = getattr(module, attr)
            def wrap(*args, _fn=original, _key=key, **kwargs):
                self.repos[_key].append({"args": copy.deepcopy(args), "kwargs": copy.deepcopy(kwargs)})
                return _fn(*args, **kwargs)
            self.mp.setattr(module, attr, wrap)
        from app.services.agent import react_adapters as ra, react_readonly_workers as rw
        for module, names in ((ra, ("read_profile", "read_workout_history", "read_nutrition_summary",
                                    "read_current_plan")),
                              (rw, ("answer_training", "advise_nutrition", "plan_training"))):
            for name in names:
                fn = getattr(module, name)
                if inspect.iscoroutinefunction(fn):
                    async def spy(args, context, _fn=fn, _name=name):
                        entry = {"name": _name, "args": copy.deepcopy(args),
                                 "context": copy.deepcopy(context), "observation": None}
                        self.handlers.append(entry)
                        try:
                            value = await _fn(args, context)
                        except Exception as exc:
                            entry["exception"] = type(exc).__name__
                            raise
                        entry["observation"] = copy.deepcopy(obs(value))
                        return value
                else:
                    def spy(args, context, _fn=fn, _name=name):
                        entry = {"name": _name, "args": copy.deepcopy(args),
                                 "context": copy.deepcopy(context), "observation": None}
                        self.handlers.append(entry)
                        try:
                            value = _fn(args, context)
                        except Exception as exc:
                            entry["exception"] = type(exc).__name__
                            raise
                        entry["observation"] = copy.deepcopy(obs(value))
                        return value
                self.mp.setattr(module, name, spy)
        from app.services.agent import react_graph as rg, react_runtime as rr
        real_invoke = rg.invoke_react
        async def invoke(*args, **kwargs):
            state = await real_invoke(*args, **kwargs)
            self.states.append(copy.deepcopy(state))
            return state
        self.mp.setattr(rg, "invoke_react", invoke)
        real_select = rr.select_mode
        def select(*args, **kwargs):
            mode = real_select(*args, **kwargs)
            self.mode_calls.append(mode)
            return mode
        self.mp.setattr(rr, "select_mode", select)
        from app.services.agent import agent as am
        from app.db import quality_repository as qr
        for key, module, attr in (("begin", am, "begin_turn"), ("finish", am, "finish_turn"),
                                  ("fail", am, "fail_turn"), ("diagnostics", qr, "record_run")):
            original = getattr(module, attr)
            def spy(*args, _fn=original, _key=key, **kwargs):
                self.lifecycle[_key].append(True)
                return _fn(*args, **kwargs)
            self.mp.setattr(module, attr, spy)
        event.listen(self.engine, "before_cursor_execute", self._sql)

    def _sql(self, conn, cursor, statement, parameters, context, executemany):
        # statement has placeholders; parameters never enter the evidence logs.
        self.sql.append(statement)

    def counts(self):
        return {key: len(value) for key, value in self.repos.items()}

    def reset_counts(self):
        for value in self.repos.values():
            value.clear()
        for value in self.lifecycle.values():
            value.clear()
        self.handlers.clear()
        self.sql.clear()
        self.parse.reset_mock()
        self.shadow.reset_mock()
        self.legacy.reset_mock()
        if hasattr(self.llm, "semantic_calls"):
            self.llm.semantic_calls.clear()

    def mock_llm(self, script=None, *, policy=None, final=FINAL, semantic=default_semantics,
                 semantic_error=None, configured=True):
        llm = ScriptLLM(self, script, policy, final, semantic=semantic,
                        semantic_error=semantic_error, configured=configured)
        self.mp.setattr(self.graph, "llm", llm)
        self.llm = llm
        return llm

    def mock_rag(self, hits, callback=None):
        from app.services.rag import online
        self.rag_calls = []
        async def search(rag, query, **kwargs):
            self.rag_calls.append({"query_chars": len(query)})
            if callback:
                return await callback(query)
            return copy.deepcopy(hits)
        self.mp.setattr(online, "search", search)

    def seed_profile(self, owner="demo-user", days=3, goal="增肌"):
        from app.db.profile_repository import save_profile
        return save_profile(owner, {"available_days": days, "goal": goal})

    def seed_workout(self, owner, exercise, age, **values):
        from app.db.models import WorkoutRecordModel
        with self.factory.begin() as session:
            session.add(WorkoutRecordModel(user_id=owner, workout_date=clock.today()-timedelta(days=age),
                                           exercise=exercise, sets=4, reps=8, weight_kg=60, **values))

    def seed_nutrition(self, owner="demo-user", kcal=100, age=0):
        from app.db.models import NutritionLogModel
        with self.factory.begin() as session:
            session.add(NutritionLogModel(user_id=owner, log_date=clock.today()-timedelta(days=age),
                                         description="合成手工营养记录", calories=kcal,
                                         protein_g=10, carbs_g=5, fat_g=2))

    def seed_food(self, owner="demo-user", kcal=500, age=0):
        from uuid import uuid4
        with self.engine.begin() as conn:
            conn.execute(text("INSERT INTO user_food_log (user_id,meal_group_id,record_time,raw_food_name,"
                              "food_name,weight_g,total_kcal,total_protein,total_carb,total_fat,idempotency_key) "
                              "VALUES (:u,:g,:t,'鸡胸肉','鸡胸肉',200,:k,40,10,8,:g)"),
                         {"u": owner, "g": "synthetic-"+str(uuid4()), "k": kcal,
                          "t": (clock.utc_now()-timedelta(days=age)).replace(tzinfo=None)})

    def seed_plan(self, owner="demo-user"):
        from app.db.plan_repository import save_plan
        return save_plan(owner, {"plan_name": "合成全身计划", "days_per_week": 3,
                                 "sessions": [{"day": "周一", "items": ["卧推"]}]})

    def snapshot(self):
        result = {}
        with self.engine.connect() as conn:
            for table in BUSINESS_TABLES:
                rows = []
                for row in conn.execute(text(f'SELECT * FROM "{table}"')).mappings():
                    item = dict(row)
                    if table == "user_profiles":
                        # memory_epoch and its timestamp belong to infrastructure.
                        item.pop("memory_epoch", None)
                        item.pop("updated_at", None)
                    rows.append(item)
                result[table] = sorted(rows, key=lambda r: json.dumps(r, sort_keys=True, default=str))
        return result

    def post(self, request_id, *, message=QUERY, mode="react_readonly", conversation="demo-conversation", **extra):
        payload = {"message": message, "conversation_id": conversation, "request_id": request_id, **extra}
        if mode is not None:
            payload["mode"] = mode
        response = self.client.post("/api/chat", json=payload)
        assert response.status_code == 200, f"HTTP {response.status_code}: {response.text[:250]}"
        return response.json()

    def stream(self, request_id, *, message=QUERY, mode="react_readonly", conversation="demo-conversation"):
        payload = {"message": message, "mode": mode, "conversation_id": conversation, "request_id": request_id}
        with self.client.stream("POST", "/api/chat/stream", json=payload) as response:
            assert response.status_code == 200
            return parse_sse(response)

    def close(self):
        self.client.close()
        self.engine.dispose()


@contextmanager
def environment(tmp_path, monkeypatch, label, **kwargs):
    with monkeypatch.context() as mp:
        env = BEnv(tmp_path / label, mp, **kwargs)
        try:
            yield env
        finally:
            env.close()


def self_check(tmp_path, monkeypatch):
    from app.services.agent import react_adapters
    from app.services.agent.react_models import DomainObservation, ToolStatus
    with environment(tmp_path, monkeypatch, "helper-check") as env:
        env.seed_workout("demo-user", "卧推", 1)
        env.seed_food()
        env.seed_nutrition()
        env.seed_plan()
        context = {"user_id": "demo-user"}
        assert obs(react_adapters.read_profile({}, context))["result"]["available_days"] == 3
        assert obs(react_adapters.read_nutrition_summary({"days": 7}, context))["result"]["calories"] == 600
        assert obs(DomainObservation(status=ToolStatus.COMPLETED, result={"ok": True}))["status"] == "completed"
        assert obs(react_adapters.record_workout({}, context))["status"] == "rejected"
        before = env.snapshot()
        env.seed_profile(days=5)
        assert env.snapshot() != before, "snapshot must detect UPDATE"
        env.seed_profile(days=3)
        assert env.snapshot() == before
        fake = SimpleNamespace(iter_lines=lambda: iter([
            'event: token', 'data: {"content":"ok"}', '',
            'event: done', 'data: {"status":"complete"}', '']))
        assert sse_body(parse_sse(fake))["answer"] == "ok"
        emit("HELPER", "seed-sync-async-snapshot-sse", passed=True, tables=len(before))
