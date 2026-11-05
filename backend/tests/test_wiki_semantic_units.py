"""Semantic-unit assembly: exact sources, conditions, budget and recovery."""
from __future__ import annotations

from app.services.context.token_counter import TokenCounter

import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.db.memory_models import MemoryTurn
from app.db.wiki_models import WikiPage
from app.schemas.chat import ChatRequest
from app.services.agent import dialogue_context, semantics
from app.services.context.budget import (
    ContextBlock, ContextBudgetExceeded, ContextBudgetManager,
)
from app.services.memory import compact, semantic_units
from app.services.memory import wiki_context as wc
from app.services.memory.selection import fit_units
from test_wiki_addressed_context import add_turn, load, memory_db, pid


def selector(plans, extraction=None, semantic=None):
    async def call(messages, schema, *, name, **kwargs):
        if name == "return_wiki_read_plan":
            value = next(plans)
        elif name == "return_request_understanding":
            value = semantic
        elif name == "return_wiki_semantic_units":
            value = extraction
        elif name == "return_dialogue_answer":
            value = {"answer": "按原文回答。", "requires_grounding": False}
        else:
            raise AssertionError(name)
        result = schema.model_validate(value)
        if kwargs.get("validator"):
            kwargs["validator"](result)
        return result
    return SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(side_effect=call))


def prepare_units(monkeypatch, memory, page_id, extraction):
    monkeypatch.setattr(settings, "wiki_semantic_units_enabled", True)
    llm = selector(iter([{"page_ids": [page_id]}]), extraction)
    asyncio.run(wc.prepare(llm, "继续", memory, semantics.PROMPT))
    return llm


def extraction_for(page_id, message_id, quote, **overrides):
    unit = {
        "unit_id": "u1",
        "message_id": message_id,
        "quote": quote,
        "claim": overrides.pop("claim", "一条可独立装配的记忆"),
        "subject": overrides.pop("subject", "本人训练"),
        "conditions": overrides.pop("conditions", []),
        "validity_text": overrides.pop("validity_text", None),
        "relevance": overrides.pop("relevance", 0.8),
        "depends_on": overrides.pop("depends_on", []),
        "complete": True,
    }
    unit.update(overrides)
    return {"pages": [{"page_id": page_id, "units": [unit],
                       "complete": True, "reason": ""}]}


def test_model_units_keep_exact_source_conditions_negation_and_validity(memory_db, monkeypatch):
    quote = "我周一在室内训练，但不对花生过敏，持续到月底。"
    tid = add_turn(quote, "已记录。")
    memory = load("我什么时候训练")
    chosen = pid(memory, tid)
    extraction = extraction_for(
        chosen, tid + ":user", quote,
        conditions=[
            {"condition": "室内训练", "negated": False, "quote": "室内训练"},
            {"condition": "不对花生过敏", "negated": True, "quote": "不对花生过敏"},
        ],
        validity_text="持续到月底", relevance=0.91)
    prepare_units(monkeypatch, memory, chosen, extraction)
    assert len(memory["wiki_units"]) == 1
    unit = memory["wiki_units"][0]
    assert unit["origin"] == "model_extracted"
    assert unit["source_start"] is not None and unit["source_end"] is not None
    assert quote[unit["source_start"]:unit["source_end"]] == quote
    assert unit["quote"] == quote
    assert any(item["negated"] and item["quote"] == "不对花生过敏"
               for item in unit["conditions"])
    assert unit["validity"]["text"] == "持续到月底"
    assert unit["revision"] == memory["wiki_loaded_pages"][chosen]["revision"]
    assert unit["content_hash"] == memory["wiki_loaded_pages"][chosen]["content_hash"]
    rendered = semantic_units.format_unit_content(unit)
    assert "否定条件" in rendered and quote in rendered


def test_forged_quote_or_page_falls_back_to_full_original(memory_db, monkeypatch):
    tid = add_turn("原始完整内容", "这是一段需要完整保留的原文，不能只留伪造片段。")
    memory = load("继续")
    chosen = pid(memory, tid)
    bad = extraction_for(chosen, tid + ":user", "伪造片段")
    prepare_units(monkeypatch, memory, chosen, bad)
    unit = memory["wiki_units"][0]
    assert unit["origin"] == "extended_fragment"
    assert unit["semantic_completeness"] == "full_original"
    assert any("原始完整内容" in item["content"] for item in unit["messages"])
    assert memory["wiki_unit_failures"]

    # A page id outside the offered catalog is rejected by the same validator.
    memory2 = load("继续")
    bad_page = {"pages": [{"page_id": "invented", "units": [], "complete": True,
                           "reason": ""}]}
    prepare_units(monkeypatch, memory2, chosen, bad_page)
    assert memory2["wiki_units"][0]["origin"] == "extended_fragment"

    unconfirmed = extraction_for(chosen, tid + ":user", "原始完整内容")
    unconfirmed["pages"][0]["complete"] = False
    memory3 = load("继续")
    prepare_units(monkeypatch, memory3, chosen, unconfirmed)
    assert memory3["wiki_units"][0]["semantic_completeness"] == "full_original"


def test_model_failure_keeps_full_original_instead_of_shrinking(memory_db, monkeypatch):
    tid = add_turn("不可抽取的原文", "完整原文必须保留，不因模型失败而消失。")
    memory = load("继续")
    chosen = pid(memory, tid)
    monkeypatch.setattr(settings, "wiki_semantic_units_enabled", True)

    async def call(messages, schema, *, name, **kwargs):
        if name == "return_wiki_read_plan":
            value = {"page_ids": [chosen]}
        elif name == "return_wiki_semantic_units":
            raise semantic_units.StructuredOutputError(
                "semantic_unit_model_unavailable", [{"type": "fixture"}])
        else:
            raise AssertionError(name)
        return schema.model_validate(value)

    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(side_effect=call))
    asyncio.run(wc.prepare(llm, "继续", memory, semantics.PROMPT))
    unit = memory["wiki_units"][0]
    assert unit["origin"] == "extended_fragment"
    assert any("完整原文必须保留" in item["content"] for item in unit["messages"])
    assert memory["wiki_unit_failures"][0]["code"] == "semantic_unit_model_unavailable"


def test_large_page_is_read_for_units_when_whole_page_cannot_enter_context(memory_db, monkeypatch):
    background = "背景原文" * 700
    quote = "周三只做两组徒手训练。"
    tid = add_turn("我的训练安排", background + quote)
    memory = load("我的训练安排")
    chosen = pid(memory, tid)
    entry = next(item for item in memory["wiki_catalog"]["entries"]
                 if item["page_id"] == chosen)
    assert entry["estimated_tokens"] > 2000  # larger than the final test budget
    monkeypatch.setattr(settings, "wiki_semantic_units_enabled", True)
    monkeypatch.setattr(settings, "wiki_source_read_max_tokens", 14000)
    monkeypatch.setattr(settings, "wiki_semantic_unit_max_input_tokens", 16000)
    extraction = extraction_for(chosen, tid + ":assistant", quote, relevance=0.95)
    llm = selector(iter([{"page_ids": [chosen]}]), extraction)
    asyncio.run(wc.prepare(llm, "我的训练安排", memory, semantics.PROMPT))
    # The page source was read/extracted under the independent source budget;
    # whole-page cost is allowed to exceed the final context budget.
    assert quote in json.dumps(memory["short_memory"], ensure_ascii=False)
    assert memory["wiki_read_tokens"] > 2000
    assert memory["wiki_read_tokens"] <= memory["wiki_read_limit"]
    unit = memory["wiki_units"][0]
    assert unit["origin"] == "model_extracted" and unit["quote"] == quote

    final = ContextBudgetManager(window=1200, output=100, margin=0, counter=TokenCounter(tokenizer_path=""))
    whole = final.assemble(system="规则", query="继续",
                           state={**memory, "wiki_units": []}, hits=[])
    packed = final.assemble(system="规则", query="继续", state=memory, hits=[])
    assert quote not in json.dumps(whole, ensure_ascii=False)
    assert quote in json.dumps(packed, ensure_ascii=False)
    assert background not in json.dumps(packed, ensure_ascii=False)
    assert packed.report["used_tokens"] <= packed.report["input_budget"]
    full = ContextBudgetManager(window=30000, output=100, margin=0).assemble(
        system="规则", query="继续", state=memory, hits=[])
    assert quote in json.dumps(full, ensure_ascii=False)

    # The final dialogue answer path uses the same unit projection and budget.
    import app.services.context.budget as budget_module
    dialogue_budget = ContextBudgetManager(window=5600, output=100, margin=0)
    monkeypatch.setattr(budget_module, "context_budget", dialogue_budget)
    dialogue_llm = SimpleNamespace(
        configured=lambda: True,
        complete_structured=AsyncMock(return_value={
            "answer": "按原文回答。", "requires_grounding": False}))
    answer = asyncio.run(dialogue_context.conversation_answer(
        dialogue_llm, ChatRequest(message="继续", conversation_id="wiki-chat"), memory,
        {"dialogue_ids": [tid + ":assistant"], "reply_mode": "conversation",
         "tasks": [], "reply_purpose": "general"}))
    final_messages = json.dumps(dialogue_llm.complete_structured.await_args.args[0],
                                ensure_ascii=False)
    assert answer["answer"] == "按原文回答。"
    assert quote in final_messages and background not in final_messages

    # Cold recovery re-reads the same bounded source range without extraction
    # and still uses the final unit budget, not the source-page budget.
    frozen = wc.freeze_context(memory, {"dialogue_ids": [tid + ":assistant"]})
    fresh = load("我的训练安排")
    no_model = SimpleNamespace(
        configured=lambda: True,
        complete_structured=AsyncMock(side_effect=AssertionError("re-extracted")))
    asyncio.run(wc.prepare_cached(
        "继续", fresh, {"dialogue_ids": [tid + ":assistant"],
                        "__wiki_context__": frozen}, semantics.PROMPT))
    recovered = final.assemble(system="规则", query="继续", state=fresh, hits=[])
    recovered_text = json.dumps(recovered, ensure_ascii=False)
    no_model.complete_structured.assert_not_awaited()
    assert quote in recovered_text and background not in recovered_text


def test_extraction_cap_falls_back_without_truncating_large_source(memory_db, monkeypatch):
    content = "长原文" * 1500
    tid = add_turn("继续", content)
    memory = load("继续")
    chosen = pid(memory, tid)
    monkeypatch.setattr(settings, "wiki_semantic_units_enabled", True)
    monkeypatch.setattr(settings, "wiki_source_read_max_tokens", 14000)
    monkeypatch.setattr(settings, "wiki_semantic_unit_max_input_tokens", 500)
    llm = selector(iter([{"page_ids": [chosen]}]), extraction_for(
        chosen, tid + ":assistant", content, relevance=0.5))
    asyncio.run(wc.prepare(llm, "继续", memory, semantics.PROMPT))
    unit = memory["wiki_units"][0]
    assert unit["origin"] == "extended_fragment"
    assert unit["semantic_completeness"] == "full_original"
    assert content in json.dumps(unit["messages"], ensure_ascii=False)
    assert any(item["code"] == "semantic_unit_input_budget"
               for item in memory["wiki_unit_failures"])


def test_source_time_is_provided_to_extracted_and_fallback_answer_context():
    unit = {"unit_id": "old-state", "quote": "我今天感冒发烧，暂停训练。",
            "temporal": {"source_at": "2026-09-01T12:00:00Z"}}
    rendered = semantic_units.unit_messages(unit, as_of="2026-11-02T12:00:00Z")
    text = json.dumps(rendered, ensure_ascii=False)
    assert "2026-09-01" in text and "2026-11-02" in text
    assert unit["quote"] in text and "不能据此认定本轮仍成立" in text
    unit.update(origin="extended_fragment", messages=[{"role": "user", "content": unit["quote"]}])
    fallback = semantic_units.unit_messages(unit, as_of="2026-11-02T12:00:00Z")
    assert "2026-09-01" in fallback[0]["content"]
    assert unit["quote"] in fallback[0]["content"]


def test_large_page_short_unit_fits_where_whole_page_does_not():
    long_quote = "背景原文" * 400
    unit_quote = "周三只做两组徒手训练。"
    page = {"page_id": "page-1", "turn_id": "turn-1", "conversation_id": "c1",
            "revision": 1, "content_hash": "abc", "completed_at": "2026-01-01",
            "wiki_importance": 0.8, "wiki_required": False, "wiki_root": True,
            "wiki_view": "current",
            "messages": [{"role": "user", "content": long_quote + unit_quote,
                          "message_id": "turn-1:user", "app_meta": {"wiki_page_id": "page-1"}}]}
    whole_state = {"short_memory": page["messages"]}
    unit = {
        **{key: page[key] for key in ("page_id", "turn_id", "conversation_id",
                                      "revision", "content_hash", "completed_at",
                                      "wiki_view")},
        "unit_id": "unit-1", "message_id": "turn-1:user", "role": "user",
        "quote": unit_quote, "source_start": len(long_quote),
        "source_end": len(long_quote) + len(unit_quote),
        "quote_hash": semantic_units.quote_digest(unit_quote),
        "claim": "短训练安排", "subject": "训练", "conditions": [],
        "validity": {"effective_at": None, "invalid_at": None, "text": None},
        "dependencies": [], "origin": "model_extracted",
        "provenance_status": "partial", "fact_validity": "unknown",
        "superseded_by": [], "conflicts_with": [], "value": 0.9,
        "importance": 0.9, "required": False, "root": True,
        "semantic_completeness": "model_asserted", "messages": [],
    }
    unit_state = {"short_memory": page["messages"], "wiki_units": [unit]}
    manager = ContextBudgetManager(window=1200, output=100, margin=0, counter=TokenCounter(tokenizer_path=""))
    whole = manager.assemble(system="规则", query="继续", state=whole_state, hits=[])
    assert unit_quote not in json.dumps(whole, ensure_ascii=False)
    packed = manager.assemble(system="规则", query="继续", state=unit_state, hits=[])
    text = json.dumps(packed, ensure_ascii=False)
    assert unit_quote in text and long_quote not in text
    assert packed.report["used_tokens"] <= packed.report["input_budget"]


def test_dependencies_missing_required_overflow_and_optional_skips_are_explicit():
    result = fit_units(["A"], ["A"], {"A": ["missing"]}, {"A": 10}, 100)
    assert result["insufficient"] and result["reason"] == "missing_dependency"
    assert result["accepted"] == []
    result = fit_units(["A"], ["A"], {}, {"A": 200}, 100)
    assert result["insufficient"] and result["reason"] == "required_over_budget"
    result = fit_units(["A"], [], {"A": ["missing"]}, {"A": 10}, 100)
    assert not result["insufficient"] and result["accepted"] == []
    assert result["skipped"] == ["A"] and result["missing_dependencies"] == {"A": ["missing"]}
    deep = fit_units(["A"], ["A"], {"A": ["B"], "B": ["missing"]},
                     {"A": 1, "B": 1}, 100)
    assert deep["reason"] == "missing_dependency"
    assert deep["missing_dependencies"] == {"B": ["missing"]}


def test_final_budget_excludes_optional_units_with_missing_strong_dependencies():
    manager = ContextBudgetManager(window=5000, output=100, margin=0)
    blocks = [
        ContextBlock('broken', 'recent', [{'role': 'user', 'content': 'incomplete'}],
                     40, group_id='unit:broken', depends_on=('unit:missing',)),
        ContextBlock('consumer', 'recent', [{'role': 'assistant', 'content': 'consumer'}],
                     40, group_id='unit:consumer', depends_on=('unit:broken',)),
        ContextBlock('independent', 'query', [{'role': 'user', 'content': 'current query'}],
                     100, required=True),
    ]
    result = manager.build(blocks)
    assert list(result) == [{'role': 'user', 'content': 'current query'}]


def test_greedy_recomputes_shared_dependency_and_handles_zero_cost_and_ties():
    # Static density would choose C then D and skip A. Recomputing after the
    # shared prerequisite B makes A dense enough to fit; D is then skipped.
    result = fit_units(
        ["A", "C", "D"], [], {"A": ["B"], "C": ["B"]},
        {"A": 50, "B": 50, "C": 10, "D": 60}, 160,
        values={"A": 0.9, "C": 0.8, "D": 0.7})
    assert set(result["accepted"]) == {"A", "B", "C"}
    assert result["skipped"] == ["D"]
    assert result["selected_tokens"] == 110

    shared = fit_units(["A", "C"], [], {"A": ["B"], "C": ["B"]},
                       {"A": 10, "B": 20, "C": 10}, 100,
                       values={"A": 0.9, "C": 0.8})
    assert set(shared["accepted"]) == {"A", "B", "C"}
    assert shared["selected_tokens"] == 40

    zero = fit_units(["Z", "Y", "X"], [], {},
                     {"Z": 10, "Y": 10, "X": 0}, 10,
                     values={"Z": 0.5, "Y": 0.5, "X": 0.5})
    assert "X" in zero["accepted"]
    assert len(set(zero["accepted"]) & {"Y", "Z"}) == 1
    assert zero["accepted"] == ["X", "Y"]

    cycle = fit_units(["A", "C"], [], {"A": ["B"], "B": ["A"]},
                      {"A": 20, "B": 20, "C": 10}, 40,
                      values={"A": 0.9, "C": 0.8})
    assert cycle["accepted"] == ["C"]
    assert cycle["skipped"] == ["A"]


def test_structured_fields_and_conflict_both_sides_reach_final_context(monkeypatch):
    memory = {
        "long_memory": {
            "allergy": {"text": "我对花生过敏", "source_quote": "我对花生过敏",
                        "source_message_id": "u1", "kind": "fact", "domain": "general"},
        },
        "memory_conflicts": {"medication": [
            {"source_message_id": "u2", "source_quote": "我吃布洛芬"},
            {"source_message_id": "u3", "source_quote": "我不吃布洛芬"},
        ]},
    }
    units = semantic_units.structured_units(
        memory, essential_fields={"medication"}, protected=())
    assert any(unit["unit_id"] == "structured:allergy" and
               unit["message_id"] == "u1" for unit in units)
    conflict = next(unit for unit in units if unit["unit_id"] == "conflict:medication")
    assert conflict["required"] and len(conflict["conditions"]) == 2
    from app.services.agent.semantics import request_understanding
    token = request_understanding.set({"query": {"critical_fact_fields": ["medication"]}})
    try:
        packed = ContextBudgetManager(window=4000, output=100, margin=0).assemble(
            system="规则", query="用药冲突", state=memory, hits=[])
    finally:
        request_understanding.reset(token)
    text = json.dumps(packed, ensure_ascii=False)
    assert "我吃布洛芬" in text and "我不吃布洛芬" in text
    assert "我对花生过敏" in text


def test_later_round_cannot_silently_drop_already_required_page(memory_db, monkeypatch):
    required_turn = add_turn("必要来源", "必要原文。")
    other_turn = add_turn("可选来源", "可选原文。")
    memory = load("继续")
    required_page = pid(memory, required_turn)
    other_page = pid(memory, other_turn)
    llm = selector(iter([
        {"page_ids": [required_page], "required_page_ids": [required_page],
         "done": False},
        {"page_ids": [other_page], "required_page_ids": [], "done": True},
    ]))
    asyncio.run(wc.prepare(llm, "继续", memory, semantics.PROMPT))
    assert required_page in memory["wiki_required_pages"]
    assert required_turn in {item["turn_id"] for item in memory["short_memory"]}
    assert other_turn in {item["turn_id"] for item in memory["short_memory"]}
    assert llm.complete_structured.await_count == 2


def test_task_scope_keeps_selected_unit_and_strong_shared_dependency(memory_db, monkeypatch):
    a_tid = add_turn("任务A第一句：周一练腿。", "任务A第二句：周三练背。")
    b_tid = add_turn("共用依据B：我在家训练。", "收到。")
    memory = load("周一练什么")
    a, b = pid(memory, a_tid), pid(memory, b_tid)
    extraction = {"pages": [
        {"page_id": a, "complete": True, "reason": "", "units": [
            {"unit_id": "a-user", "message_id": a_tid + ":user",
             "quote": "任务A第一句：周一练腿。", "claim": "周一练腿",
             "subject": "训练", "conditions": [], "validity_text": None,
             "relevance": 0.9, "depends_on": [], "complete": True},
            {"unit_id": "a-assistant", "message_id": a_tid + ":assistant",
             "quote": "任务A第二句：周三练背。", "claim": "周三练背",
             "subject": "训练", "conditions": [], "validity_text": None,
             "relevance": 0.5, "depends_on": [], "complete": True},
        ]},
        {"page_id": b, "complete": True, "reason": "", "units": [
            {"unit_id": "b-user", "message_id": b_tid + ":user",
             "quote": "共用依据B：我在家训练。", "claim": "在家训练",
             "subject": "条件", "conditions": [], "validity_text": None,
             "relevance": 0.8, "depends_on": [], "complete": True},
        ]},
    ]}
    monkeypatch.setattr(settings, "wiki_semantic_units_enabled", True)
    llm = selector(iter([{"page_ids": [a], "dependencies": {a: [b]}}]), extraction)
    asyncio.run(wc.prepare(llm, "周一练什么", memory, semantics.PROMPT))
    selected = dialogue_context.answer_memory(
        memory, {"dialogue_ids": [a_tid + ":user"]})
    ids = {unit["unit_id"] for unit in selected["wiki_units"]}
    assert len(ids) == 2
    assert any("任务A第一句" in json.dumps(unit, ensure_ascii=False) for unit in selected["wiki_units"])
    assert not any("任务A第二句" in json.dumps(unit, ensure_ascii=False)
                   for unit in selected["wiki_units"])
    assert any("共用依据B" in json.dumps(unit, ensure_ascii=False)
               for unit in selected["wiki_units"])
    frozen = wc.freeze_context(memory, {"dialogue_ids": [a_tid + ":user"]})
    assert {unit["unit_id"] for unit in frozen["units"]} == ids


def test_final_budget_reports_missing_dependency_and_required_overflow():
    missing = ContextBlock("unit-A", "semantic", [{"role": "system", "content": "必要单元"}],
                           95, required=True, group_id="unit:A",
                           depends_on=("unit:absent",))
    with pytest.raises(ContextBudgetExceeded) as error:
        ContextBudgetManager(window=1000, output=100, margin=0).build([missing])
    assert error.value.reason == "missing_dependency"

    big = ContextBlock("unit-B", "semantic",
                       [{"role": "system", "content": "必要" * 600}],
                       95, required=True, group_id="unit:B")
    with pytest.raises(ContextBudgetExceeded) as error:
        ContextBudgetManager(window=400, output=100, margin=0).build([big])
    assert error.value.reason == "required_over_budget"

    optional = ContextBlock("unit-C", "semantic",
                            [{"role": "system", "content": "可选" * 500}],
                            40, group_id="unit:C", depends_on=("unit:absent",))
    small = ContextBlock("unit-D", "semantic",
                         [{"role": "system", "content": "小单元"}], 40, group_id="unit:D")
    packed = ContextBudgetManager(window=500, output=100, margin=0).build([optional, small])
    assert "小单元" in json.dumps(packed, ensure_ascii=False)
    assert packed.report["dropped_blocks"]


def test_history_view_units_are_labeled_not_current(memory_db, monkeypatch):
    from uuid import uuid4
    from app.db.memory_models import MemoryState
    old = add_turn("旧安排", "以前周三练腿。")
    with memory_db.factory.begin() as s:
        s.add(MemoryState(
            state_id=str(uuid4()), owner_id="demo-user", conversation_id="wiki-chat",
            scope="conversation", logical_key="training:old-plan", kind="transient",
            domain="training", version=1, value="过期的旧安排", status="active",
            valid_from=compact.now(), expires_at=compact.now().replace(year=2020),
            memory_epoch=0, source_turn_id=old, source_message_id=old + ":user"))
    memory = load("以前怎么安排")
    history = wc.catalog("wiki-chat", 0, "", 16000, initial=True, include_history=True)
    memory["wiki_catalog"] = history
    chosen = pid(memory, old)
    monkeypatch.setattr(settings, "wiki_semantic_units_enabled", True)
    extraction = extraction_for(chosen, old + ":assistant", "以前周三练腿。",
                                validity_text=None)
    llm = selector(iter([{"page_ids": [chosen], "include_history": True}]), extraction)
    asyncio.run(wc.prepare(llm, "以前怎么安排", memory, semantics.PROMPT))
    assert memory["wiki_units"][0]["wiki_view"] == "history"
    rendered = semantic_units.format_unit_content(memory["wiki_units"][0])
    assert "历史证据，非当前状态" in rendered


def test_response_path_reports_missing_unit_dependency():
    page_id, message_id, unit_id = "p1", "t1:user", "u1"
    memory = {
        "wiki_prepared": True, "wiki_conversation_id": "c1", "memory_epoch": 0,
        "long_memory": {}, "retrieved_history": [], "wiki_dependencies": {},
        "wiki_loaded_pages": {"p1": {"page_id": page_id, "turn_id": "t1",
                                     "conversation_id": "c1", "revision": 1,
                                     "content_hash": "h", "messages": []}},
        "short_memory": [{"role": "user", "content": "继续", "message_id": message_id,
                          "turn_id": "t1", "app_meta": {"wiki_page_id": page_id,
                                                         "wiki_group": "wiki:p1"}}],
        "wiki_units": [{
            "unit_id": unit_id, "page_id": page_id, "turn_id": "t1",
            "conversation_id": "c1", "revision": 1, "content_hash": "h",
            "message_id": message_id, "role": "user", "quote": "继续",
            "source_start": 0, "source_end": 2, "quote_hash": "x",
            "claim": "继续", "subject": "", "conditions": [],
            "validity": {}, "dependencies": ["missing-unit"], "origin": "model_extracted",
            "provenance_status": "partial", "fact_validity": "unknown",
            "superseded_by": [], "conflicts_with": [], "value": 1.0,
            "importance": 1.0, "required": True, "root": True,
            "semantic_completeness": "model_asserted", "messages": [],
        }],
    }
    understanding = {"dialogue_ids": [message_id], "reply_mode": "conversation",
                     "tasks": [], "reply_purpose": "general"}
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock())
    result = asyncio.run(dialogue_context.conversation_answer(
        llm, ChatRequest(message="继续", conversation_id="c1"), memory, understanding))
    assert result["status"] == "context_budget"
    assert result["error"] == "semantic_unit_dependency_missing"
    llm.complete_structured.assert_not_awaited()


def test_execution_entry_maps_semantic_unit_insufficiency(monkeypatch):
    from app.services.agent.agent import FitnessAgent
    agent = FitnessAgent()
    monkeypatch.setattr(agent, "_restore_persisted_semantics", AsyncMock())
    monkeypatch.setattr(
        semantics, "understand",
        AsyncMock(side_effect=wc.WikiReadError("wiki_semantic_unit_missing_dependency")))
    response = asyncio.run(agent._understand_or_limit(
        ChatRequest(message="继续"), {"wiki_semantic_attempted": True}, "normal"))
    assert response is not None and response.intent == "clarification"
    assert "语义单元" in response.answer and "未新增记录" in response.answer


def test_real_agent_final_prompt_uses_units_not_whole_page(memory_db, monkeypatch):
    from test_retry_operations import RetryScenario
    from app.services.agent import execution
    source = "我什么时候训练？"
    quote = "我周一在室内训练，但不对花生过敏。"
    tid = add_turn(quote, "背景回答" * 100)
    memory = load(source)
    chosen = pid(memory, tid)
    raw = {"tasks": [{"task_type": "training_qa", "action": "question", "text": source,
                      "is_negated": False, "is_hypothetical": False,
                      "is_third_party": False, "history_requirement": "required",
                      "reply_mode": "conversation", "goal_role": "result"}],
           "dialogue_ids": [tid + ":user"], "reply_mode": "conversation",
           "relation": "follow_up",
           "safety": {"level": "normal", "evidence": "", "categories": [],
                      "reason": "scripted"}}
    extraction = extraction_for(chosen, tid + ":user", quote, relevance=0.9)
    llm = selector(iter([{"page_ids": [chosen]}]), extraction, semantic=raw)
    monkeypatch.setattr(settings, "wiki_semantic_units_enabled", True)
    scenario = RetryScenario(monkeypatch, memory_db.factory)
    monkeypatch.setattr(execution.memory_service, "load", AsyncMock(return_value=memory))
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured",
                        llm.complete_structured)
    response = asyncio.run(scenario.respond(
        ChatRequest(message=source, conversation_id="wiki-chat")))
    assert response.answer == "按原文回答。"
    last = llm.complete_structured.await_args_list[-1]
    rendered = json.dumps(last.args[0], ensure_ascii=False)
    assert quote in rendered
    assert "背景回答" * 20 not in rendered
    assert [call.kwargs["name"] for call in llm.complete_structured.await_args_list] == [
        "return_request_understanding", "return_wiki_read_plan",
        "return_wiki_semantic_units", "return_dialogue_answer"]


def test_structured_units_keep_validity_supersession_and_conflict_orthogonal():
    memory = {"long_memory": {
        "old_fact": {"text": "旧的训练时间", "source_quote": "以前晚上训练",
                     "source_message_id": "u1", "kind": "fact",
                     "valid_to": "2020-01-01T00:00:00+00:00",
                     "superseded_by": ["new-fact-id"]},
    }}
    item = semantic_units.structured_units(memory)[0]
    assert item["origin"] == "structured"
    assert item["fact_validity"] == "expired"
    assert item["superseded_by"] == ["new-fact-id"]
    assert item["conflicts_with"] == []
    assert item["message_id"] == "u1"
    assert "以前晚上训练" == item["quote"]


def test_cold_resume_freezes_addresses_and_revalidates_sources(memory_db, monkeypatch):
    tid = add_turn("原任务", "周三只做两组徒手训练。")
    memory = load("继续")
    chosen = pid(memory, tid)
    extraction = extraction_for(chosen, tid + ":assistant", "周三只做两组徒手训练。",
                                relevance=0.8)
    prepare_units(monkeypatch, memory, chosen, extraction)
    understanding = {"dialogue_ids": [tid + ":assistant"]}
    frozen = wc.freeze_context(memory, understanding)
    assert frozen and frozen["units"]
    assert '"quote":' not in json.dumps(frozen, ensure_ascii=False)
    assert all("quote_hash" in unit for unit in frozen["units"])
    understanding["__wiki_context__"] = frozen
    fresh = load("继续")
    no_model = SimpleNamespace(configured=lambda: True,
                               complete_structured=AsyncMock(
                                   side_effect=AssertionError("cold resume re-extracted")))
    asyncio.run(wc.prepare_cached("继续", fresh, understanding, semantics.PROMPT))
    assert fresh["wiki_units"][0]["quote"] == "周三只做两组徒手训练。"
    no_model.complete_structured.assert_not_awaited()

    tampered = json.loads(json.dumps(frozen, ensure_ascii=False))
    tampered["units"][0]["quote_hash"] = "0" * 64
    tampered_understanding = {**understanding, "__wiki_context__": tampered}
    with pytest.raises(wc.WikiReadError, match="wiki_cached_unit_quote_changed"):
        asyncio.run(wc.prepare_cached("继续", load("继续"), tampered_understanding,
                                      semantics.PROMPT))

    with memory_db.factory.begin() as s:
        s.get(MemoryTurn, tid).hidden_at = compact.now()
    with pytest.raises(wc.WikiReadError):
        asyncio.run(wc.prepare_cached("继续", load("继续"), understanding, semantics.PROMPT))

    with memory_db.factory.begin() as s:
        s.get(WikiPage, chosen).revision += 1
    with pytest.raises(wc.WikiReadError):
        asyncio.run(wc.prepare_cached("继续", load("继续"), understanding, semantics.PROMPT))


def _temporal_unit(unit_id, value, *, sensitivity, stability="time_sensitive",
                   source_at="2026-01-01T00:00:00+00:00", required=False,
                   view="current", basis="最近"):
    return {
        "unit_id": unit_id, "value": value, "importance": value,
        "required": required, "wiki_view": view,
        "temporal": {"sensitivity": sensitivity, "stability": stability,
                     "basis_quote": basis if stability == "time_sensitive" else None,
                     "source_at": source_at},
    }


def test_temporal_factor_is_bounded_explicit_and_never_infers_time(monkeypatch):
    as_of = "2026-11-01T00:00:00+00:00"
    fresh = _temporal_unit("fresh", .9, sensitivity=1.0,
                           source_at="2026-10-31T00:00:00+00:00")
    old = _temporal_unit("old", .9, sensitivity=1.0,
                         source_at="2020-01-01T00:00:00+00:00")
    stable = _temporal_unit("stable", .9, sensitivity=0.0,
                            stability="stable", basis=None,
                            source_at="2000-01-01T00:00:00+00:00")
    missing = _temporal_unit("missing", .9, sensitivity=1.0, source_at=None)
    future = _temporal_unit("future", .9, sensitivity=1.0,
                            source_at="2030-01-01T00:00:00+00:00")
    ancient = _temporal_unit("ancient", .9, sensitivity=1.0,
                             source_at="1900-01-01T00:00:00+00:00")
    history = _temporal_unit("history", .9, sensitivity=1.0,
                             source_at="2020-01-01T00:00:00+00:00",
                             view="history")
    required = _temporal_unit("required", .9, sensitivity=1.0,
                              source_at="2020-01-01T00:00:00+00:00",
                              required=True)

    fresh_value, fresh_signal = semantic_units.assess_unit(fresh, as_of=as_of)
    old_value, old_signal = semantic_units.assess_unit(old, as_of=as_of)
    assert fresh_value > old_value
    assert old_signal["status"] == "decayed"
    assert 0.85 <= old_signal["factor"] < 1.0
    assert semantic_units.assess_unit(stable, as_of=as_of)[1]["status"] == "stable"
    assert semantic_units.assess_unit(stable, as_of=as_of)[0] == stable["value"]
    assert semantic_units.assess_unit(missing, as_of=as_of)[1]["status"] == "missing_time"
    assert semantic_units.assess_unit(missing, as_of=as_of)[0] == missing["value"]
    assert semantic_units.assess_unit(future, as_of=as_of)[1]["status"] == "future_time"
    assert semantic_units.assess_unit(future, as_of=as_of)[0] == future["value"]
    ancient_value, ancient_signal = semantic_units.assess_unit(ancient, as_of=as_of)
    assert ancient_value == round(ancient["value"] * .85, 6)
    assert ancient_signal["factor"] == .85
    assert semantic_units.assess_unit(history, as_of=as_of)[1]["status"] == "history"
    assert semantic_units.assess_unit(history, as_of=as_of)[0] == history["value"]
    assert semantic_units.assess_unit(required, as_of=as_of)[1]["status"] == "required"
    assert semantic_units.assess_unit(required, as_of=as_of)[0] == required["value"]

    # The request as_of is fixed: changing it changes the score deterministically.
    later_value, _ = semantic_units.assess_unit(fresh, as_of="2026-12-01T00:00:00+00:00")
    assert later_value < fresh_value
    monkeypatch.setattr(settings, "wiki_temporal_decay_enabled", False)
    assert semantic_units.assess_unit(old, as_of=as_of)[1]["status"] == "disabled"
    assert semantic_units.assess_unit(old, as_of=as_of)[0] == old["value"]


def test_time_sensitive_decay_changes_final_selection_but_stable_and_history_do_not(
        memory_db, monkeypatch):
    old_tid = add_turn("旧短期状态：最近改成早上训练。", "收到。")
    new_tid = add_turn("新短期状态：这周改成早上训练。", "收到。")
    stable_tid = add_turn("长期事实：我对花生过敏。", "收到。")
    with memory_db.factory.begin() as s:
        s.get(MemoryTurn, old_tid).completed_at = compact.now() - timedelta(days=400)
        s.get(MemoryTurn, stable_tid).completed_at = compact.now() - timedelta(days=400)
        s.get(MemoryTurn, new_tid).completed_at = compact.now() - timedelta(days=1)
    memory = load("我什么时候训练")
    old_page, new_page, stable_page = (pid(memory, old_tid), pid(memory, new_tid),
                                       pid(memory, stable_tid))
    extraction = {"pages": [
        {"page_id": old_page, "complete": True, "reason": "", "units": [{
            "unit_id": "old-sensitive", "message_id": old_tid + ":user",
            "quote": "旧短期状态：最近改成早上训练。", "claim": "旧下午训练",
            "subject": "训练时间", "conditions": [], "validity_text": None,
            "temporal_stability": "time_sensitive", "temporal_sensitivity": 1.0,
            "temporal_basis_quote": "最近", "relevance": .95, "depends_on": [],
            "complete": True}]},
        {"page_id": new_page, "complete": True, "reason": "", "units": [{
            "unit_id": "new-sensitive", "message_id": new_tid + ":user",
            "quote": "新短期状态：这周改成早上训练。", "claim": "这周早上训练",
            "subject": "训练时间", "conditions": [], "validity_text": None,
            "temporal_stability": "time_sensitive", "temporal_sensitivity": 1.0,
            "temporal_basis_quote": "这周", "relevance": .90, "depends_on": [],
            "complete": True}]},
        {"page_id": stable_page, "complete": True, "reason": "", "units": [{
            "unit_id": "stable-allergy", "message_id": stable_tid + ":user",
            "quote": "长期事实：我对花生过敏。", "claim": "花生过敏",
            "subject": "过敏", "conditions": [], "validity_text": None,
            "temporal_stability": "stable", "temporal_sensitivity": 0.0,
            "temporal_basis_quote": None, "relevance": .82, "depends_on": [],
            "complete": True}]},
    ]}
    monkeypatch.setattr(settings, "wiki_semantic_units_enabled", True)
    llm = selector(iter([{"page_ids": [old_page, new_page, stable_page]}]), extraction)
    asyncio.run(wc.prepare(llm, "我什么时候训练", memory, semantics.PROMPT))
    memory["wiki_as_of"] = compact.now().isoformat()

    units = {unit["claim"]: unit for unit in memory["wiki_units"]}
    assert units["旧下午训练"]["temporal"]["stability"] == "time_sensitive"
    assert units["花生过敏"]["temporal"]["stability"] == "stable"
    old_value = semantic_units.unit_value(units["旧下午训练"], as_of=memory["wiki_as_of"])
    new_value = semantic_units.unit_value(units["这周早上训练"], as_of=memory["wiki_as_of"])
    stable_value = semantic_units.unit_value(units["花生过敏"], as_of=memory["wiki_as_of"])
    assert old_value < stable_value < new_value

    # Same real final-assembly entry: choose a budget that fits exactly two.
    two_state = {**memory, "wiki_units": [units["这周早上训练"], units["花生过敏"]]}
    two = ContextBudgetManager(window=100000, output=100, margin=0).assemble(
        system="规则", query="我什么时候训练", state=two_state, hits=[])
    window = two.report["used_tokens"] + 110
    packed = ContextBudgetManager(window=window, output=100, margin=0).assemble(
        system="规则", query="我什么时候训练", state=memory, hits=[])
    by_id = {unit["unit_id"]: claim for claim, unit in units.items()}
    accepted = {by_id[item.split("unit:", 1)[1]]
                for item in packed.report["semantic_unit_selection"]["accepted"]}
    signals = {by_id[item.split("unit:", 1)[1]]: signal for item, signal
               in packed.report["semantic_unit_selection"]["temporal_signals"].items()}
    assert accepted == {"这周早上训练", "花生过敏"}
    assert "旧下午训练" not in accepted
    assert signals["旧下午训练"]["status"] == "decayed"
    assert signals["花生过敏"]["status"] == "stable"
    assert signals["这周早上训练"]["status"] == "decayed"

    # Turning the time factor off is the executable same-budget ablation.
    monkeypatch.setattr(settings, "wiki_temporal_decay_enabled", False)
    off = ContextBudgetManager(window=window, output=100, margin=0).assemble(
        system="规则", query="我什么时候训练", state=memory, hits=[])
    off_accepted = {by_id[item.split("unit:", 1)[1]]
                    for item in off.report["semantic_unit_selection"]["accepted"]}
    assert "旧下午训练" in off_accepted
    assert off_accepted != accepted
    assert off.report["used_tokens"] <= off.report["input_budget"]


def test_value_ranking_is_an_executable_same_budget_ablation():
    costs = {"A": 100, "B": 10, "C": 10}
    values = {"A": 1.0, "B": .8, "C": .79}
    density = fit_units(["A", "B", "C"], [], {}, costs, 100, values=values,
                        ranking="density")
    value = fit_units(["A", "B", "C"], [], {}, costs, 100, values=values,
                      ranking="value")
    assert density["accepted"] == ["B", "C"]
    assert value["accepted"] == ["A"]
    assert density["selected_tokens"] == 20 and density["remaining_tokens"] == 80
    assert value["selected_tokens"] == 100 and value["remaining_tokens"] == 0
    assert density["ranking"] == "density" and value["ranking"] == "value"


def test_cached_optional_wiki_error_only_blocks_required_history():
    base_task = {"task_type": "workout_record", "action": "record",
                 "text": "记录深蹲", "is_negated": False,
                 "is_hypothetical": False, "is_third_party": False,
                 "history_requirement": "none"}
    self_contained = {
        "wiki_catalog": {"entries": []}, "wiki_error": "wiki_database_unavailable",
        "wiki_conversation_id": "c1", "memory_epoch": 0,
        "tasks": [base_task],
    }
    asyncio.run(wc.prepare_cached(
        "记录深蹲", self_contained, {"tasks": [base_task]}, semantics.PROMPT))
    assert self_contained.get("wiki_prepared") is not True

    requires_history = {**self_contained,
                        "tasks": [{**base_task, "history_requirement": "required"}]}
    with pytest.raises(wc.WikiReadError, match="wiki_database_unavailable"):
        asyncio.run(wc.prepare_cached(
            "记录深蹲", requires_history,
            {"tasks": [{**base_task, "history_requirement": "required"}]},
            semantics.PROMPT))


def test_invalid_temporal_signal_falls_back_to_complete_original(memory_db, monkeypatch):
    quote = "这周我把训练改到早上。"
    tid = add_turn(quote, "收到。")
    memory = load("继续")
    chosen = pid(memory, tid)
    bad = extraction_for(
        chosen, tid + ":user", quote,
        temporal_stability="time_sensitive", temporal_sensitivity=.5,
        temporal_basis_quote="原文没有的时间词")
    prepare_units(monkeypatch, memory, chosen, bad)
    unit = memory["wiki_units"][0]
    assert unit["origin"] == "extended_fragment"
    assert unit["semantic_completeness"] == "full_original"
    assert any("这周我把训练改到早上" in item["content"] for item in unit["messages"])
    assert memory["wiki_unit_failures"]


def test_frozen_units_keep_temporal_binding_and_reject_tamper(memory_db, monkeypatch):
    quote = "这周我把训练改到早上。"
    tid = add_turn(quote, "收到。")
    memory = load("继续")
    chosen = pid(memory, tid)
    extraction = extraction_for(
        chosen, tid + ":user", quote,
        temporal_stability="time_sensitive", temporal_sensitivity=.8,
        temporal_basis_quote="这周")
    prepare_units(monkeypatch, memory, chosen, extraction)
    unit = memory["wiki_units"][0]
    assert unit["temporal"]["basis_quote"] == "这周"
    understanding = {"dialogue_ids": [tid + ":user"]}
    frozen = wc.freeze_context(memory, understanding)
    assert frozen["as_of"] == memory["wiki_as_of"]
    record = frozen["units"][0]["temporal"]
    assert record["sensitivity"] == .8 and record["stability"] == "time_sensitive"
    assert '"basis_quote":' not in json.dumps(frozen, ensure_ascii=False)
    assert record["basis_hash"] == semantic_units.quote_digest("这周")
    fresh = load("继续")
    asyncio.run(wc.prepare_cached(
        "继续", fresh, {**understanding, "__wiki_context__": frozen}, semantics.PROMPT))
    recovered = fresh["wiki_units"][0]
    assert recovered["temporal"]["sensitivity"] == .8
    assert recovered["temporal"]["basis_quote"] == "这周"

    tampered = json.loads(json.dumps(frozen, ensure_ascii=False))
    tampered["units"][0]["temporal"]["basis_hash"] = "0" * 64
    with pytest.raises(wc.WikiReadError, match="wiki_cached_unit_temporal_changed"):
        asyncio.run(wc.prepare_cached(
            "继续", load("继续"), {**understanding, "__wiki_context__": tampered},
            semantics.PROMPT))
