"""A2 reference-reuse regression: one RequestUnderstanding, one child RAG read.

The HTTP cases drive the real legacy graph, parser, consulting/advice worker and
RAG reference boundary. Only model wire and remote MCP are mocked; storage is a
temporary SQLite file. The small unit cases pin the coverage helper's exact
source/span rules.
"""
from __future__ import annotations

import asyncio
import copy
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from app.services.agent.semantics import (
    request_understanding, remember_understanding, validate_understanding,
)
from tests.react_phase_b_codex_support import (
    BUSINESS_TABLES, chunks, environment, sse_body,
)

TRAIN_SOURCE = "记录卧推60kg，4组8次，然后告诉我怎么恢复"
TRAIN_WRITE = "记录卧推60kg，4组8次"
TRAIN_READ = "然后告诉我怎么恢复"
FOOD_SOURCE = "记录午餐鸡胸肉200克，然后建议晚餐怎么吃"
FOOD_WRITE = "记录午餐鸡胸肉200克"
FOOD_READ = "然后建议晚餐怎么吃"


def semantic_task(kind, action, text, *, history_requirement="none"):
    return {"task_type": kind, "action": action, "text": text,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            "history_requirement": history_requirement}


def mixed_tasks(domain):
    safety = {"level": "normal", "evidence": "", "categories": [],
              "reason": "scripted normal"}
    if domain == "training":
        return {"tasks": [semantic_task("workout_record", "record", TRAIN_WRITE),
                          semantic_task("training_qa", "question", TRAIN_READ)],
                "safety": safety}
    return {"tasks": [semantic_task("nutrition_record", "record", FOOD_WRITE),
                      semantic_task("nutrition_advice", "question", FOOD_READ)],
            "safety": safety}


def business_dml(sql):
    import re
    return [statement for statement in sql
            if re.match(r"\s*(INSERT|UPDATE|DELETE)\b", statement, re.I)
            and any(re.search(r"\b" + table + r"\b", statement, re.I)
                    for table in BUSINESS_TABLES)]


@pytest.fixture(autouse=True)
def no_network_and_pending(monkeypatch):
    from evals.v2.safety import deny_network
    from app.services.agent import food_calculator
    monkeypatch.setattr(food_calculator, "_pending_food", {})
    with deny_network() as audit:
        yield
    assert audit == {"blocked_connections": 0, "blocked_dns": 0}


class GraphModel:
    """Fake wire with the real structured validator and real correction loop."""

    def __init__(self, raw):
        self.raw = copy.deepcopy(raw)
        self.semantic_posts, self.structured_calls, self.other_posts = [], [], []
        self.answer_prompts = []
        self.entity_weight = 60

    def configured(self):
        return True

    async def complete_structured(self, messages, schema, **kwargs):
        from app.services.llm.structured import complete_structured
        self.structured_calls.append(schema.__name__)
        return await complete_structured(self, messages, schema, **kwargs)

    async def complete_message(self, messages, **kwargs):
        choice = kwargs.get("tool_choice")
        name = ((choice.get("function") or {}).get("name")
                if isinstance(choice, dict) else None)
        if name == "return_request_understanding":
            self.semantic_posts.append(copy.deepcopy(messages))
            raw = self.raw
            # A redundant child understanding would be counted even if it can
            # produce a valid subset; the assertion must fail on the extra call.
            if isinstance(raw, dict):
                query = json.loads(messages[-1]["content"])["query"]
                matching = [task for task in raw.get("tasks", []) if task["text"] == query]
                if len(matching) == 1 and len(raw.get("tasks", [])) > 1:
                    raw = {"tasks": matching}
            arguments = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": "semantic-result", "type": "function",
                 "function": {"name": name, "arguments": arguments}}]}
        self.other_posts.append(copy.deepcopy(messages))
        if kwargs.get("tools"):
            return {"role": "assistant", "content": "建议循序渐进，并根据恢复情况调整。"}
        return {"role": "assistant", "content": "建议循序渐进，并根据恢复情况调整。"}

    async def complete(self, messages, **kwargs):
        self.other_posts.append(copy.deepcopy(messages))
        system = str(messages[0].get("content", ""))
        if "records" in system and "weight_kg" in system:
            return json.dumps({"records": [{"exercise": "卧推", "weight_kg": self.entity_weight,
                                          "sets": 4, "reps": 8}]})
        if "foods" in system and "weight_g" in system:
            return json.dumps({"foods": [{"source_text": "鸡胸肉200克", "name": "鸡胸肉",
                "quantity": 1, "unit": "g", "weight_g": 200, "meal_type": "午餐",
                "record_date": None, "is_negated": False}]}, ensure_ascii=False)
        if "健身科普答疑Agent" in system or "饮食营养方案" in system:
            self.answer_prompts.append(system)
        return "建议循序渐进，并根据恢复情况调整。"


@contextmanager
def real_graph_http_env(tmp_path, monkeypatch, raw, label):
    from app.core import clock
    from app.core.config import settings

    with environment(tmp_path, monkeypatch, label) as env:
        mp = env.mp
        mp.setattr(clock, "utc_now", lambda: datetime.now(timezone.utc))
        model = GraphModel(raw)
        mp.setattr(env.graph, "llm", model)
        mp.setattr(env.graph.input_parser, "llm", model)
        mp.setattr(env.graph.input_parser.workout_extractor, "llm", model)
        mp.setattr(env.graph.workout_extraction, "llm", model)
        mp.setattr(env.graph.workout_extraction.model_extractor, "llm", model)
        mp.setattr(settings, "input_parser_enabled", True)
        mp.setattr(settings, "workout_parser_enabled", True)
        mp.setattr(settings, "react_write_enabled", False)
        parse = AsyncMock(wraps=type(env.graph).parse_input.__get__(env.graph))
        legacy = AsyncMock(wraps=type(env.graph).run.__get__(env.graph))
        mp.setattr(env.graph, "parse_input", parse)
        mp.setattr(env.graph, "run", legacy)
        mp.setattr(env.graph.mcp, "call_tool", AsyncMock(return_value=None))
        mp.setattr(env.graph.mcp, "complete_with_tools", AsyncMock(return_value=None))
        with env.engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE food_nutrition (id INTEGER PRIMARY KEY, food_name TEXT, "
                "category TEXT, preparation_state TEXT, energy_kcal NUMERIC, protein_g NUMERIC, "
                "carb_g NUMERIC, fat_g NUMERIC, fiber_g NUMERIC, is_active BOOLEAN, "
                "source_version TEXT)"))
            conn.execute(text(
                "INSERT INTO food_nutrition VALUES "
                "(1,'鸡胸肉','肉','熟',167,31,0,3.6,0,TRUE,'reuse-v1')"))
        env.model, env.parse, env.legacy = model, parse, legacy
        env.reset_counts()
        yield env


def request(env, transport, rid, source):
    if transport == "json":
        return env.post(rid, message=source, mode="legacy")
    return sse_body(env.stream(rid, message=source, mode="legacy"))


def seed_reference_history(env):
    from app.db.memory_models import MemoryTurn
    from app.schemas.chat import ChatRequest
    from app.services.memory import compact
    tid, _ = compact.begin_turn(ChatRequest(
        message="我练卧推时肩部不适，恢复期间避免负重",
        conversation_id="demo-conversation", request_id="reference-history"))
    compact.finish_turn(tid, "后续建议会结合这个限制。", {"answer": "后续建议会结合这个限制。"})
    with env.factory() as session:
        history = copy.deepcopy(session.get(MemoryTurn, tid).messages)
    history = [{**message, "turn_id": tid} for message in history]
    user_id = next(message["message_id"] for message in history if message["role"] == "user")
    return history, user_id


@pytest.mark.parametrize("transport", ["json", "sse"])
@pytest.mark.parametrize("domain", ["training", "nutrition"])
def test_a2_real_legacy_graph_reuses_parent_understanding_for_child_rag(
        tmp_path, monkeypatch, domain, transport):
    source = TRAIN_SOURCE if domain == "training" else FOOD_SOURCE
    with real_graph_http_env(tmp_path, monkeypatch, mixed_tasks(domain), "a2-" + domain) as env:
        before = env.snapshot()
        body = request(env, transport, "a2-" + domain, source)
        after = env.snapshot()
        metrics = body.get("metrics", {})
        assert metrics.get("execution_mode") == "legacy", metrics
        assert metrics.get("mode_reason") == "requested_legacy"
        assert env.parse.await_count == 1
        assert env.legacy.await_count == 1 and not env.states and not env.handlers
        tasks = env.legacy.await_args.kwargs["planned_tasks"]
        assert len(tasks) == 2
        writes = [task for task in tasks if task.get("side_effect")]
        assert len(writes) == 1
        assert writes[0]["payload"]["write_authorization"] == "authorized"
        assert env.model.structured_calls == ["RequestUnderstanding"]
        assert len(env.model.semantic_posts) == 1
        assert env.model.answer_prompts, "consulting/advice worker did not execute"
        assert business_dml(env.sql), "no real business SQL dispatched"
        if domain == "training":
            rows = after["workout_records"]
            assert len(rows) == len(before["workout_records"]) + 1
            assert (rows[0]["exercise"], rows[0]["weight_kg"], rows[0]["sets"], rows[0]["reps"]) == (
                "卧推", 60, 4, 8)
            assert len(after["workout_sessions"]) == 1
            assert len(after["workout_session_exercises"]) == 1
            assert len(after["workout_session_sets"]) == 4
        else:
            rows = after["user_food_log"]
            assert len(rows) == len(before["user_food_log"]) + 1
            assert rows[0]["food_name"] == "鸡胸肉" and rows[0]["weight_g"] == 200
            assert rows[0]["total_kcal"] == 334 and rows[0]["total_protein"] == 62
            assert rows[0]["food_id"] == 1 and rows[0]["match_method"] == "exact"

        counts = (len(env.model.structured_calls), len(env.model.semantic_posts),
                  len(env.model.other_posts), env.parse.await_count, env.legacy.await_count,
                  env.graph.mcp.call_tool.await_count,
                  env.graph.mcp.complete_with_tools.await_count)
        env.sql.clear()
        replay = request(env, transport, "a2-" + domain, source)
        assert replay["replayed"] is True and replay["answer"] == body["answer"]
        assert env.snapshot() == after
        assert counts == (len(env.model.structured_calls), len(env.model.semantic_posts),
                          len(env.model.other_posts), env.parse.await_count, env.legacy.await_count,
                          env.graph.mcp.call_tool.await_count,
                          env.graph.mcp.complete_with_tools.await_count)
        assert not business_dml(env.sql)


@pytest.mark.parametrize("transport", ["json", "sse"])
@pytest.mark.parametrize("reference_source", ["parent_full_text", "child_full_text", "unrelated_record_span"])
def test_a2_parent_reference_reaches_child_rag_without_second_understanding(
        tmp_path, monkeypatch, transport, reference_source):
    write = "记录卧推60kg，4组8次"
    child = "然后按上次的限制给我恢复建议"
    parent = write + "，" + child
    rewritten = "卧推后有肩部不适、恢复期间避免负重时应如何安排恢复"
    raw = {"tasks": [semantic_task("workout_record", "record", write),
                     semantic_task("training_qa", "question", child,
                                   history_requirement="required")],
           "safety": {"level": "normal", "evidence": "", "categories": [],
                      "reason": "scripted normal"}}
    with real_graph_http_env(tmp_path, monkeypatch, raw, "reference-" + reference_source) as env:
        history, user_id = seed_reference_history(env)
        source_text = {"parent_full_text": parent, "child_full_text": child,
                       "unrelated_record_span": write}[reference_source]
        raw["references"] = [{
            "source_text": source_text, "query": rewritten, "evidence_ids": [user_id],
            "context_ids": [], "fact_fields": [], "ambiguous": False,
        }]
        env.load.return_value = {"memory_backend": "compact_v1", "short_memory": history}
        env.model.raw = copy.deepcopy(raw)
        observed = []

        async def search(pipeline, query, **kwargs):
            observed.append({"query": query, "skip_rewrite": kwargs.get("skip_rewrite")})
            return chunks(query=query)

        env.mp.setattr("app.services.rag.online.search", search)
        before = env.snapshot()
        body = request(env, transport, "reference-" + reference_source, parent)
        expected = child if reference_source == "unrelated_record_span" else rewritten
        assert observed == [{"query": expected, "skip_rewrite": expected != child}], observed
        assert body["metrics"].get("execution_mode") == "legacy"
        assert body["metrics"].get("mode_reason") == "requested_legacy"
        assert env.parse.await_count == 1 and env.legacy.await_count == 1 and not env.states
        assert env.model.structured_calls == ["RequestUnderstanding"]
        assert len(env.model.semantic_posts) == 1
        planned = env.legacy.await_args.kwargs["planned_tasks"]
        assert [item["text"] for item in planned] == [write, child]
        assert planned[0]["payload"]["write_authorization"] == "authorized"
        rows = env.snapshot()["workout_records"]
        assert len(rows) == len(before["workout_records"]) + 1
        assert (rows[0]["exercise"], rows[0]["weight_kg"], rows[0]["sets"], rows[0]["reps"]) == (
            "卧推", 60, 4, 8)
        assert len(env.snapshot()["workout_session_sets"]) == 4
        assert business_dml(env.sql)

        after = env.snapshot()
        counts = (len(env.model.structured_calls), len(env.model.semantic_posts),
                  len(env.model.other_posts), env.parse.await_count, env.legacy.await_count,
                  len(observed))
        env.sql.clear()
        replay = request(env, transport, "reference-" + reference_source, parent)
        assert replay["replayed"] is True and replay["answer"] == body["answer"]
        assert env.snapshot() == after
        assert counts == (len(env.model.structured_calls), len(env.model.semantic_posts),
                          len(env.model.other_posts), env.parse.await_count, env.legacy.await_count,
                          len(observed))
        assert not business_dml(env.sql)


class FakeUnderstanding:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def configured(self):
        return True

    async def complete_structured(self, messages, schema, **kwargs):
        self.calls += 1
        query = json.loads(messages[-1]["content"])["query"]
        raw = self.raw(query) if callable(self.raw) else {
            "tasks": [semantic_task("training_qa", "question", query)]}
        value = raw if isinstance(raw, schema) else schema.model_validate(copy.deepcopy(raw))
        if kwargs.get("validator") is not None:
            kwargs["validator"](value)
        return value


def child_history():
    return [{"message_id": "u1", "role": "user", "content": "我上周练过深蹲"}]


def remember_parent(source, child, references=None, history=None):
    raw = {"tasks": [semantic_task("workout_record", "record", TRAIN_WRITE),
                     semantic_task("training_qa", "question", child)]}
    if references:
        raw["references"] = references
    value = validate_understanding(source, raw, context={"short_memory": history or []})
    remember_understanding(source, value)
    return value


@pytest.fixture(autouse=True)
def request_scope():
    token = request_understanding.set({})
    try:
        yield
    finally:
        request_understanding.reset(token)


def test_a2_validated_child_with_empty_references_reuses_without_model_call():
    from app.services.rag.reference import resolve_reference_model
    from app.services.agent.semantics import understanding_covers
    remember_parent(TRAIN_SOURCE, TRAIN_READ)
    llm = FakeUnderstanding(None)
    result = asyncio.run(resolve_reference_model(
        TRAIN_READ, [], llm, parent_source=TRAIN_SOURCE))
    assert result == TRAIN_READ and llm.calls == 0
    assert understanding_covers(TRAIN_READ, source=TRAIN_SOURCE)


def test_a2_explicit_valid_reference_still_rewrites_without_second_call():
    from app.services.rag.reference import resolve_reference_model
    history = child_history()
    remember_parent(TRAIN_SOURCE, TRAIN_READ, references=[{
        "source_text": TRAIN_READ, "query": "深蹲后如何恢复", "evidence_ids": ["u1"],
        "context_ids": [], "fact_fields": [], "ambiguous": False,
    }], history=history)
    llm = FakeUnderstanding(None)
    result = asyncio.run(resolve_reference_model(
        TRAIN_READ, history, llm, parent_source=TRAIN_SOURCE))
    assert result == "深蹲后如何恢复" and llm.calls == 0


def test_a2_exact_child_reference_precedes_parent_full_text_reference():
    from app.services.rag.reference import resolve_reference_model
    history = child_history()
    remember_parent(TRAIN_SOURCE, TRAIN_READ, references=[
        {"source_text": TRAIN_SOURCE, "query": "父全文改写", "evidence_ids": ["u1"],
         "context_ids": [], "fact_fields": [], "ambiguous": False},
        {"source_text": TRAIN_READ, "query": "子任务改写", "evidence_ids": ["u1"],
         "context_ids": [], "fact_fields": [], "ambiguous": False},
    ], history=history)
    llm = FakeUnderstanding(None)
    result = asyncio.run(resolve_reference_model(
        TRAIN_READ, history, llm, parent_source=TRAIN_SOURCE))
    assert result == "子任务改写" and llm.calls == 0


def test_a2_unknown_parent_source_does_not_borrow_another_cache_entry():
    from app.services.rag.reference import resolve_reference_model
    history = child_history()
    remember_parent(TRAIN_SOURCE, TRAIN_READ, references=[
        {"source_text": TRAIN_SOURCE, "query": "父全文改写", "evidence_ids": ["u1"],
         "context_ids": [], "fact_fields": [], "ambiguous": False},
        {"source_text": TRAIN_READ, "query": "子任务改写", "evidence_ids": ["u1"],
         "context_ids": [], "fact_fields": [], "ambiguous": False},
    ], history=history)
    llm = FakeUnderstanding(None)
    result = asyncio.run(resolve_reference_model(
        TRAIN_READ, history, llm, parent_source="另一个未缓存的父请求"))
    assert result == TRAIN_READ and llm.calls == 1


def test_a2_uncovered_query_keeps_real_understanding_path():
    from app.services.rag.reference import resolve_reference_model
    remember_parent(TRAIN_SOURCE, TRAIN_READ)
    llm = FakeUnderstanding(None)
    result = asyncio.run(resolve_reference_model(
        "晚餐怎么吃", [], llm, parent_source=TRAIN_SOURCE))
    assert result == "晚餐怎么吃" and llm.calls == 1


def test_a2_similar_substring_is_not_coverage():
    from app.services.rag.reference import resolve_reference_model
    from app.services.agent.semantics import understanding_covers
    remember_parent(TRAIN_SOURCE, TRAIN_READ)
    assert not understanding_covers("怎么恢复", source=TRAIN_SOURCE)
    llm = FakeUnderstanding(None)
    result = asyncio.run(resolve_reference_model(
        "怎么恢复", [], llm, parent_source=TRAIN_SOURCE))
    assert result == "怎么恢复" and llm.calls == 1


def test_a2_unvalidated_cache_or_new_request_does_not_cover():
    from app.services.agent.semantics import understanding_covers
    remember_parent(TRAIN_SOURCE, TRAIN_READ)
    assert understanding_covers(TRAIN_READ, source=TRAIN_SOURCE)
    # A manually inserted non-validated child entry must not count.
    request_understanding.set({TRAIN_SOURCE: {"tasks": [
        semantic_task("training_qa", "question", TRAIN_READ)]}})
    assert not understanding_covers(TRAIN_READ, source=TRAIN_SOURCE)
    # A new request has a fresh cache; the previous request's coverage is gone.
    request_understanding.set({})
    assert not understanding_covers(TRAIN_READ, source=TRAIN_SOURCE)
