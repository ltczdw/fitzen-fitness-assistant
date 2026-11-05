import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from app.services.agent.semantics import (
    RequestUnderstanding, understand, get_understanding, remember_understanding,
    request_understanding, validate_understanding, semantic_candidates,
)
from app.services.agent.task_planner import plan_tasks_from_llm, plan_tasks_rule
from app.services.rag.reference import resolve_reference_model
from evals.v2.safety import deny_network


@pytest.fixture(autouse=True)
def scoped():
    token = request_understanding.set({})
    with deny_network():
        yield
    request_understanding.reset(token)


def outline(source, *, action="record", kind="nutrition_record", safety=None, **extra):
    return {"tasks": [{"task_type": kind, "action": action, "text": source,
                        "is_negated": False, "is_hypothetical": False, "is_third_party": False}],
            "safety": safety or {"level": "normal", "evidence": "", "categories": [],
                                 "reason": "scripted normal"},
            **extra}


class Model:
    def __init__(self, result=None, *, available=True):
        self.configured = lambda: available
        self.calls = 0
        self.result = result

    async def complete_structured(self, messages, schema, **kwargs):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return schema.model_validate(deepcopy(self.result))


@pytest.mark.parametrize("source", ["不要忘了给我记上中午一个鸡蛋", "中午鸡蛋一个，收进午餐", "朋友推荐的鸡蛋，我午餐吃了一个，帮我存上"])
def test_one_model_decision_is_shared_by_planner_react_and_memory(source):
    model = Model(outline(source))
    first = asyncio.run(understand(model, source))
    second = asyncio.run(understand(model, source))
    tasks, error = semantic_candidates(source)
    assert first == second and model.calls == 1
    assert error is None and tasks[0]["side_effect"]
    rule_tasks = plan_tasks_rule(source)
    # The planner's shared business projection (target, authorization, source
    # flags) must stay identical; the extra semantic/dialogue payload is the
    # planner-owned provenance, not a second model decision or write approval.
    projected = deepcopy(rule_tasks)
    for task in projected:
        for key in ("semantic_available", "semantic", "source_start", "source_end"):
            task["payload"].pop(key)
    # Compare every original field, including flags, facts and authorization.
    # Only the explicitly checked planner additions below are projected out.
    assert projected == tasks
    payload = rule_tasks[0]["payload"]
    assert payload["semantic_available"] is True
    assert payload["semantic"]["text"] == tasks[0]["text"]
    assert payload["semantic"]["source_task_type"] == "nutrition_record"
    assert payload["source_start"] == 0
    assert payload["source_end"] == len(source)
    planned = plan_tasks_from_llm(source, {"tasks": first["tasks"]})
    assert planned[0]["side_effect"]


def test_questions_and_flags_are_not_changed_by_text_patterns():
    source = "记录中午鸡蛋是什么意思"
    result = asyncio.run(understand(Model(outline(source, action="question")), source))
    assert not result["validated_tasks"][0]["side_effect"]


@pytest.mark.parametrize("error", [RuntimeError("offline"), ValueError("bad response")])
def test_model_failure_produces_no_business_candidates(error):
    source = "记录两个鸡蛋"
    assert asyncio.run(understand(Model(error), source)) is None
    assert get_understanding(source) is None
    assert all(not t["side_effect"] for t in plan_tasks_rule(source))


def test_unconfigured_model_is_not_called_or_replaced_by_keyword_rules():
    model = Model(available=False)
    assert asyncio.run(understand(model, "记录卧推60kg4组8次")) is None
    assert model.calls == 0


def test_reference_without_fixed_pronoun_and_arbitrary_action_names():
    source = "同样的条件下换成它，膝盖能承受吗"
    history = [{"message_id": "u1", "role": "user", "content": "保加利亚分腿蹲，膝盖有旧伤，不能负重"}]
    rewritten = "膝盖有旧伤且不能负重时，保加利亚分腿蹲的安全边界"
    raw = outline(source, action="question", kind="training_qa", references=[{
        "source_text": source, "query": rewritten, "evidence_ids": ["u1"], "ambiguous": False,
    }], critical_history_ids=["u1"])
    model = Model(raw)
    asyncio.run(understand(model, source, context={"short_memory": history}))
    assert asyncio.run(resolve_reference_model(source, history, model)) == rewritten
    assert model.calls == 1
    # Retrieval rewriting must never mutate executable task text.
    assert semantic_candidates(source)[0][0]["text"] == source


def test_ambiguous_reference_asks_and_never_guesses():
    source = "那一个帮我记一下"
    raw = outline(source, references=[{"source_text": source, "query": source, "evidence_ids": [], "ambiguous": True}])
    result = asyncio.run(understand(Model(raw), source))
    assert result["needs_clarification"]
    assert asyncio.run(resolve_reference_model(source, [], Model())) == source


@pytest.mark.parametrize("change", [
    {"tasks": [{"task_type": "nutrition_record", "action": "record", "text": "模型编造鸡胸肉", "is_negated": False, "is_hypothetical": False, "is_third_party": False}]},
    {"memory_requests": [{"action": "remember", "quote": "模型编造的事实", "key": "load", "confidence": 1}]},
    {"critical_history_ids": ["assistant-id"]},
    {"critical_fact_fields": ["foreign-owner-fact"]},
    {"references": [{"source_text": "鸡蛋", "query": "米饭", "evidence_ids": ["foreign"], "ambiguous": False}]},
])
def test_unknown_evidence_is_rejected(change):
    source = "鸡蛋一枚"
    assert asyncio.run(understand(Model({**outline(source), **change}), source)) is None


def test_model_fact_span_must_be_inside_the_same_original_task():
    source = "帮我记录午餐：一枚鸡蛋"
    raw = outline(source)
    raw["tasks"][0]["fact_text"] = "一枚鸡蛋"
    result = validate_understanding(source, raw)
    remember_understanding(source, result)
    from app.services.agent.react_graph import _strip_command_prefix
    start = source.index("一枚鸡蛋")
    assert _strip_command_prefix(source, (0, len(source))) == (start, len(source))
    raw["tasks"][0]["fact_text"] = "另一餐米饭"
    with pytest.raises(ValueError):
        validate_understanding(source, raw)


def test_memory_only_when_model_interprets_explicit_request(compact_db):
    from app.services.memory import compact
    from app.schemas.chat import ChatRequest
    from app.db.memory_models import UserFact
    source = "往后请按我对花生过敏来安排饮食"
    raw = outline(source, kind="training_qa", action="answer", memory_requests=[{
        "action": "remember", "quote": "我对花生过敏", "key": "allergy:peanut", "confidence": .99,
    }], current_is_critical=True)
    value = validate_understanding(source, raw)
    tid, _ = compact.begin_turn(ChatRequest(message=source, conversation_id="memory-semantic"))
    compact.finish_turn(tid, "收到", {"answer": "收到"}, semantic_values=[(source, value)])
    with compact_db() as session:
        rows = list(session.scalars(select(UserFact)))
        assert len(rows) == 1 and rows[0].status == "active"
        assert rows[0].source_quote == "我对花生过敏"
    # Same completed turn is an idempotent replay, with no second fact.
    compact.finish_turn(tid, "收到", {"answer": "收到"}, semantic_values=[(source, value)])
    with compact_db() as session:
        assert len(list(session.scalars(select(UserFact)))) == 1


def commit_memory(source, raw, *, context=None):
    from app.services.memory import compact
    from app.schemas.chat import ChatRequest
    value = validate_understanding(source, raw, context=context)
    tid, _ = compact.begin_turn(ChatRequest(message=source, conversation_id="memory-semantic"))
    compact.finish_turn(tid, "收到", {"answer": "收到"}, semantic_values=[(source, value)])
    return tid


def test_explicit_replacement_keeps_old_source_and_retires_only_known_target(compact_db):
    from app.services.memory import compact
    from app.db.memory_models import UserFact
    first = "以后参考我的卧推工作重量60公斤"
    commit_memory(first, outline(first, action="answer", kind="training_qa", memory_requests=[{
        "action": "remember", "quote": "我的卧推工作重量60公斤", "key": "workload:bench_press", "confidence": 1,
    }]))
    facts = compact.current_facts()
    second = "把这个记忆更新成我的卧推工作重量65公斤"
    commit_memory(second, outline(second, action="answer", kind="training_qa", memory_requests=[{
        "action": "replace", "quote": "我的卧推工作重量65公斤", "key": "workload:bench_press", "target_fields": list(facts), "confidence": 1,
    }]), context={"long_memory": facts})
    with compact_db() as session:
        rows = list(session.scalars(select(UserFact)))
        assert len(rows) == 2
        assert {r.status for r in rows} == {"active", "superseded"}
        assert next(r for r in rows if r.status == "active").value == "我的卧推工作重量65公斤"
        assert next(r for r in rows if r.status == "superseded").source_quote == "我的卧推工作重量60公斤"


def test_conflicting_memory_without_explicit_replace_asks_instead_of_overwriting():
    source = "也记一下我的卧推工作重量65公斤"
    facts = {"known": {"text": "60公斤", "source_quote": "60公斤", "source_message_id": "u1"}}
    raw = outline(source, action="answer", kind="training_qa", memory_requests=[{
        "action": "remember", "quote": "我的卧推工作重量65公斤", "key": "workload:bench_press", "target_fields": ["known"], "confidence": .99,
    }])
    value = validate_understanding(source, raw, context={"long_memory": facts})
    assert value["needs_clarification"]


def test_no_model_result_cannot_create_fact_from_remember_keyword(compact_db):
    from app.services.memory import compact
    from app.schemas.chat import ChatRequest
    from app.db.memory_models import UserFact
    tid, _ = compact.begin_turn(ChatRequest(message="记住我卧推60公斤"))
    compact.finish_turn(tid, "模型不可用", {"answer": "模型不可用"})
    with compact_db() as session:
        assert list(session.scalars(select(UserFact))) == []


def test_memory_transaction_rolls_back_all_updates_on_bad_second_fact(compact_db):
    from app.services.memory import compact
    from app.schemas.chat import ChatRequest
    from app.db.memory_models import UserFact, MemoryTurn
    source = "记下我偏好晨练且对花生过敏"
    value = validate_understanding(source, outline(source, action="answer", kind="training_qa", memory_requests=[{
        "action": "remember", "quote": "我偏好晨练", "key": "preference:training_time", "confidence": 1,
    }]))
    value["memory_requests"].append({"action": "replace", "quote": "不存在的原文", "key": "bad", "confidence": 1})
    tid, _ = compact.begin_turn(ChatRequest(message=source))
    with pytest.raises(ValueError):
        compact.finish_turn(tid, "收到", {"answer": "收到"}, semantic_values=[(source, value)])
    with compact_db() as session:
        assert list(session.scalars(select(UserFact))) == []
        assert session.get(MemoryTurn, tid).status != "completed"


def test_critical_context_is_selected_by_model_evidence_without_keyword_trigger():
    from app.services.context.budget import ContextBudgetManager
    source = "继续说刚才的问题"
    history = [{"message_id": "u1", "role": "user", "content": "医生要求这一阶段暂停大负荷"},
               {"message_id": "a1", "role": "assistant", "content": "收到"}]
    raw = outline(source, action="question", kind="training_qa", critical_history_ids=["u1"])
    remember_understanding(source, validate_understanding(source, raw, context={"short_memory": history}))
    budget = ContextBudgetManager(window=4000, output=200, margin=0)
    messages = budget.assemble(system="健身助手", query=source, state={"short_memory": history}, hits=[])
    assert any("暂停大负荷" in m["content"] for m in messages)
    assert all(set(m) == {"role", "content"} for m in messages)


def test_profile_snapshot_cannot_be_replaced_as_a_long_term_fact():
    source = "更新我的记忆：喜欢晨练"
    raw = outline(source, action="answer", kind="training_qa", memory_requests=[{
        "action": "replace", "quote": "喜欢晨练", "key": "preference:training_time", "target_fields": ["profile"], "confidence": 1,
    }])
    with pytest.raises(ValueError, match="target"):
        validate_understanding(source, raw, context={"long_memory": {"profile": {"source": "saved_profile", "facts": {"goal": "增肌"}}}})


def test_primary_parser_reuses_understanding_and_skips_entities_when_asking():
    from app.services.agent.deepseek_input_parser import PlannerLLM
    source = "午餐那一份，帮我记一下"
    model = Model(outline(source, needs_clarification=True, clarification="你指的是哪份食物，大约多少克？"))
    model.complete = AsyncMock(side_effect=AssertionError("clarification must not extract entities"))
    result = asyncio.run(PlannerLLM(model).parse(source))
    assert result["tasks"][0]["text"] == source
    assert get_understanding(source)["needs_clarification"]
    assert model.calls == 1
    model.complete.assert_not_called()


def test_estimate_override_is_bound_into_operation_fingerprint():
    from app.services.agent.operation_store import food_operation_identity
    rows = [{"raw_text": "300克神秘盖饭", "requested_name": "神秘盖饭", "weight_g": 300}]
    one = food_operation_identity(rows, key="stable", nutrition_overrides={"300克神秘盖饭": {"kcal": 180}})
    two = food_operation_identity(rows, key="stable", nutrition_overrides={"300克神秘盖饭": {"kcal": 181}})
    assert one[0] == two[0] and one[1] != two[1]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True])
def test_invalid_estimate_is_never_written(compact_db, monkeypatch, bad):
    from app.services.agent.food_calculator import calculate_nutrition
    from app.services.agent.food_parser import FoodInput
    from app.db.food_repository import resolve_food
    entry = FoodInput("300克神秘盖饭", "神秘盖饭", 1, "克", 300, None)
    override = {"kcal": bad, "protein_g": 7, "carbs_g": 25, "fat_g": 6, "fiber_g": 1}
    result = calculate_nutrition("demo-user", [entry], raw_message=entry.raw_text,
                                 nutrition_overrides={entry.raw_text: override})
    assert result["status"] != "calculated"
    from sqlalchemy import text
    with compact_db() as session:
        assert session.execute(text("SELECT COUNT(*) FROM user_food_log")).scalar_one() == 0


def test_async_compatibility_bridge_keeps_memory_budget_and_cache_context():
    from contextvars import ContextVar
    from app.services.agent.intent import _run_async
    marker = ContextVar("semantic-test-marker", default=None)
    async def check():
        token = marker.set("request-owned")
        try:
            async def child():
                assert marker.get() == "request-owned"
                remember_understanding("scope", {"validated_tasks": []})
                return marker.get()
            assert _run_async(child) == "request-owned"
            assert get_understanding("scope") is not None
        finally:
            marker.reset(token)
    asyncio.run(check())


def test_frozen_original_understanding_is_restored_only_for_bound_source():
    import hashlib
    from app.services.agent.react_write_runtime import WriteRuntimeSession
    source = "午餐300克神秘盖饭，帮我保存"
    value = validate_understanding(source, outline(source))
    snapshot = {"tasks": [{"task_type": "react_controller", "parameters": {
        "input_fingerprint": hashlib.sha256(source.encode()).hexdigest(), "semantic_understanding": value}}]}
    WriteRuntimeSession._restore_source_understanding(snapshot, "不同请求")
    assert get_understanding("不同请求") is None
    WriteRuntimeSession._restore_source_understanding(snapshot, source)
    assert semantic_candidates(source)[1] is None


@pytest.mark.parametrize("status,expected", [("committed", True), ("reserved", False)])
def test_restored_ai_label_requires_committed_operation(status, expected):
    from app.services.agent.react_write_runtime import WriteRuntimeSession
    session = WriteRuntimeSession.__new__(WriteRuntimeSession)
    session.source = "午餐300克神秘盖饭，帮我保存"
    session.checkpoint = {}
    session.ai_estimate_committed = False
    snapshot = {"revision": 2, "checkpoint": {}, "operations": [{"kind": "nutrition", "status": status,
        "parameters": {"groups": [{"foods": [{"source": "ai"}]}]}}]}
    session._apply_snapshot(snapshot)
    assert session.ai_estimate_committed is expected


def test_primary_model_parser_and_legacy_ai_food_save_with_real_sql(compact_db, monkeypatch):
    import json
    from sqlalchemy import text
    from test_retry_operations import RetryScenario
    from app.schemas.chat import ChatRequest
    from app.services.agent.deepseek_input_parser import PlannerLLM
    from app.services import nutrition_ai
    source = "别忘了帮我把午餐300克神秘盖饭记上"
    model = Model(outline(source))
    model.complete = AsyncMock(return_value=json.dumps({"foods": [{"source_text": "300克神秘盖饭", "name": "神秘盖饭",
        "quantity": 1, "unit": "克", "weight_g": 300, "meal_type": "午餐", "record_date": None, "is_negated": False}]}))
    estimate = AsyncMock(return_value={"per_100g": {"kcal": 180, "protein_g": 7, "carbs_g": 25, "fat_g": 6, "fiber_g": 1}})
    monkeypatch.setattr(nutrition_ai, "estimate_food_nutrition", estimate)
    scenario = RetryScenario(monkeypatch, compact_db)
    scenario.agent.graph.llm = model
    monkeypatch.setattr(scenario.agent.graph, "parse_input", PlannerLLM(model).parse)
    response = asyncio.run(scenario.respond(ChatRequest(message=source, conversation_id="legacy-semantic", request_id="legacy-ai-semantic")))
    assert "AI估算" in response.answer, response.answer
    assert scenario.tool_calls.count("calculate_nutrition") == 1
    estimate.assert_awaited_once()
    assert model.calls == 1
    with compact_db() as db:
        row = db.execute(text("SELECT weight_g, is_estimated, match_method FROM user_food_log")).one()
        assert row.weight_g == 300 and row.is_estimated and row.match_method == "ai_estimate"


def test_emergency_gate_uses_request_local_model_safety_without_second_call(monkeypatch):
    from app.services.agent import intent
    source = "现在胸痛并且呼吸困难"
    safety = {"level": "urgent", "evidence": source, "categories": ["current_symptom"],
              "reason": "scripted current emergency"}
    value = validate_understanding(source, outline(source, action="question", kind="safety_triage",
                                                    safety=safety))
    remember_understanding(source, value)
    model = AsyncMock(side_effect=AssertionError("emergency must reuse the request-local judgment"))
    monkeypatch.setattr(intent, "classify_intent_llm", model)
    result = intent.classify_intent(source)
    assert result.name == "symptom_triage" and result.safety_level == "urgent"
    model.assert_not_called()


def test_memory_replacement_race_never_supersedes_a_newer_source(compact_db):
    from app.services.memory import compact
    from app.db.memory_models import UserFact
    def update(value, *, context=None, targets=None):
        source = f"更新并记住我的卧推工作重量{value}公斤"
        return commit_memory(source, outline(source, action="answer", kind="training_qa", memory_requests=[{
            "action": "replace" if targets else "remember", "quote": f"我的卧推工作重量{value}公斤",
            "key": "load:bench", "target_fields": targets or [], "confidence": 1}]), context=context)
    update(60)
    original = compact.current_facts()
    update(65, context={"long_memory": original}, targets=list(original))
    update(70, context={"long_memory": original}, targets=list(original))
    with compact_db() as db:
        rows = list(db.scalars(select(UserFact)))
        assert [row.value for row in rows if row.status == "active"] == ["我的卧推工作重量65公斤"]
        assert [row.value for row in rows if row.status == "pending_confirmation"] == ["我的卧推工作重量70公斤"]


def test_only_requested_memory_target_quotes_are_retained():
    source = "把记忆更新为晨练"
    facts = {"wanted": {"source_message_id": "u1", "source_quote": "晚练"},
             "unrelated": {"source_message_id": "u2", "source_quote": "私人事实"}}
    value = validate_understanding(source, outline(source, action="answer", kind="training_qa", memory_requests=[{
        "action": "replace", "quote": "晨练", "key": "preference:time", "target_fields": ["wanted"], "confidence": 1}]), context={"long_memory": facts})
    assert set(value["memory_target_sources"]) == {"wanted"}


@pytest.mark.parametrize("flag", ["is_negated", "is_hypothetical", "is_third_party"])
@pytest.mark.parametrize("value", ["false", 0, 1])
def test_model_semantic_flags_cannot_be_coerced_from_strings_or_numbers(flag, value):
    source = "午餐一个鸡蛋，帮我记上"
    raw = outline(source)
    raw["tasks"][0][flag] = value
    with pytest.raises(ValueError):
        validate_understanding(source, raw)
    assert not any(t["side_effect"] for t in plan_tasks_rule(source))
