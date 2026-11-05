"""Regressions for the 2026-10-29 live synthetic failures.

These tests use the real semantic/lifecycle/dialogue components with an
isolated SQLite database and scripted model replies. No provider or live
database is contacted.
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select

from app.core.config import settings
from app.db.memory_models import MemoryState
from app.schemas.chat import ChatRequest
from app.services.agent import dialogue_context as dc
from app.services.agent import semantics
from app.services.memory import compact, lifecycle as lc, wiki_context as wc
from tests.test_wiki_addressed_context import memory_db  # noqa: F401 - shared isolated fixture


def safety(level: str = "normal") -> dict:
    return {"level": level, "evidence": "", "categories": [], "reason": "scripted"}


def task(source: str, *, kind: str = "chitchat", action: str = "question", **extra) -> dict:
    return {
        "task_type": kind,
        "action": action,
        "text": source,
        "is_negated": extra.pop("is_negated", False),
        "is_hypothetical": extra.pop("is_hypothetical", False),
        "is_third_party": extra.pop("is_third_party", False),
        "history_requirement": extra.pop("history_requirement", "none"),
        **extra,
    }


def outline(source: str, **extra) -> dict:
    return {"tasks": [task(source, **extra.pop("task_extra", {}))],
            "safety": safety(), **extra}


# ---------------------------------------------------------------------------
# USER-LIVE29-01: a fact selected by fact_fields carries its own source turn.
# That provenance is not a current-history dialogue ID.
# ---------------------------------------------------------------------------

def test_cross_session_fact_recall_accepts_real_fact_source_provenance():
    source = "我让你保存的训练时间偏好是什么？只引用已经保存的那条，不要猜，也不要新增记忆。"
    fact = "semantic:e852eefc410a89fd:28bf36bcec4b6b476ca2bd15793799d85859bab67da68f45"
    turn_id = "c4e308ca-514a-4d93-b2c5-3308524b582b"
    memory = {
        "short_memory": [],
        "retrieved_history": [{
            "page_id": "wiki-source-page", "turn_id": turn_id,
            "content": json.dumps([
                {"message_id": turn_id + ":user", "role": "user", "content": "我一般在晚上训练。"},
                {"message_id": turn_id + ":assistant", "role": "assistant", "content": "收到。"},
            ], ensure_ascii=False),
        }],
        "long_memory": {
            fact: {
                "text": "我一般在晚上训练",
                "source_quote": "我一般在晚上训练",
                "source_turn_id": turn_id,
                "source_message_id": turn_id + ":user",
                "kind": "fact",
                "domain": "general",
                "fact_id": "62bf22ce-9c55-4bc6-891b-6bf125a6f97f",
            }
        },
        "current_states": {},
    }
    raw = {
        "tasks": [task(
            source, action="question", reply_mode="conversation",
            reply_purpose="fact_recall", goal_role="result", quoted=False,
            fact_text=None, fact_fields=[fact], critical_fact_fields=[],
            context_ids=["c4e308ca-514a-4d93-b2c5-3308524b582b:user",
                         "c4e308ca-514a-4d93-b2c5-3308524b582b:assistant"],
        )],
        "safety": safety(),
        "reply_mode": "conversation",
        "reply_purpose": "fact_recall",
        "dialogue_ids": ["c4e308ca-514a-4d93-b2c5-3308524b582b:user",
                         "c4e308ca-514a-4d93-b2c5-3308524b582b:assistant"],
        "relation": "follow_up",
    }
    value = semantics.validate_understanding(source, raw, context=memory)
    selected = value["tasks"][0]
    assert selected["fact_fields"] == [fact]
    # The user source is already selected through fact_fields; the addressed
    # assistant original remains legal historical evidence, not a user fact.
    assert set(selected["context_ids"]) <= {turn_id + ":assistant"}
    assert set(value["dialogue_ids"]) <= {turn_id + ":assistant"}


def test_cross_session_fact_recall_renders_the_selected_current_fact():
    source = "我让你保存的训练时间偏好是什么？只引用已经保存的那条，不要猜，也不要新增记忆。"
    fact = "semantic:e852eefc410a89fd:28bf36bcec4b6b476ca2bd15793799d85859bab67da68f45"
    memory = {"short_memory": [], "long_memory": {fact: {
        "text": "我一般在晚上训练", "source_quote": "我一般在晚上训练",
        "source_turn_id": "c4e308ca-514a-4d93-b2c5-3308524b582b",
        "source_message_id": "c4e308ca-514a-4d93-b2c5-3308524b582b:user",
        "kind": "fact", "domain": "general", "fact_id": "f-1",
    }}}
    raw = {
        "tasks": [task(source, action="question", reply_mode="conversation",
                       reply_purpose="fact_recall", goal_role="result", quoted=False,
                       fact_text=None, fact_fields=[fact], critical_fact_fields=[],
                       context_ids=["c4e308ca-514a-4d93-b2c5-3308524b582b:user"])],
        "safety": safety(), "reply_mode": "conversation",
        "reply_purpose": "fact_recall", "relation": "follow_up",
    }
    value = semantics.validate_understanding(source, raw, context=memory)

    async def complete(messages, schema, **kwargs):
        return schema.model_validate({"evidence": [{
            "source_id": "fact:" + fact, "quote": "我一般在晚上训练",
        }]})

    llm = SimpleNamespace(configured=lambda: True,
                          complete_structured=AsyncMock(side_effect=complete))
    result = asyncio.run(dc.conversation_answer(
        llm, ChatRequest(message=source), memory, value))
    assert result and "我一般在晚上训练" in result["answer"]


def test_selected_cross_source_cold_resume_reloads_bound_pages(memory_db):
    from tests.test_wiki_addressed_context import add_turn, load, pid, prepare
    tid = add_turn("我喜欢早晨训练。", "收到这条信息。", cid="cold-cross-source")
    source = "请回顾另一个会话里我说喜欢什么时候训练，只引用原话，不要保存。"
    memory = load(source, cid="cold-cross-reader")
    prepare(memory, [{"page_ids": [pid(memory, tid)]}])
    raw = {
        "tasks": [task(source, action="question", reply_mode="conversation",
                       reply_purpose="fact_recall", goal_role="result", quoted=False,
                       context_ids=[tid + ":user", tid + ":assistant"])],
        "safety": safety(), "reply_mode": "conversation",
        "reply_purpose": "fact_recall", "dialogue_ids": [tid + ":user", tid + ":assistant"],
    }
    value = semantics.validate_understanding(source, raw, context=memory)
    frozen = wc.freeze_context(memory, value)
    assert frozen and frozen["pages"]
    value["__wiki_context__"] = frozen
    fresh = load(source, cid="cold-cross-reader")
    asyncio.run(wc.prepare_cached(source, fresh, value, semantics.PROMPT))
    assert any(item["message_id"] == tid + ":user"
               for item in dc.project_addressed_history(fresh))


def test_forged_fact_source_is_still_rejected():
    source = "我让你保存的训练时间偏好是什么？"
    raw = outline(source, task_extra={
        "reply_mode": "conversation", "reply_purpose": "fact_recall",
        "context_ids": ["forged:user"], "fact_fields": ["semantic:not-present"],
    })
    raw["safety"] = safety()
    raw["reply_purpose"] = "fact_recall"
    with pytest.raises(semantics.SemanticValidationError):
        semantics.validate_understanding(source, raw, context={"short_memory": [], "long_memory": {}})


# ---------------------------------------------------------------------------
# USER-SIM29-01: a non-writing constraint may share the nutrition domain.
# Only action=record/confirm is a write.
# ---------------------------------------------------------------------------

def test_hypothetical_negative_constraint_is_not_a_write_conflict():
    source = "假如明晚吃180克鸡肉，这样说会不会被保存？我现在没吃，不要登记。"
    first = "假如明晚吃180克鸡肉，这样说会不会被保存？"
    second = "我现在没吃，不要登记。"
    raw = {
        "tasks": [
            task(first, kind="nutrition_record", action="question",
                 is_hypothetical=True, goal_role="result", quoted=False,
                 reply_mode="conversation", reply_purpose="execution_status",
                 fact_text=None, context_ids=[], fact_fields=[], critical_fact_fields=[]),
            task(second, kind="nutrition_record", action="answer", is_negated=True,
                 goal_role="constraint", quoted=False, reply_mode="conversation",
                 reply_purpose="general", fact_text=None, context_ids=[],
                 fact_fields=[], critical_fact_fields=[]),
        ],
        "safety": safety(),
        "reply_mode": "conversation",
        "reply_purpose": "execution_status",
        "relation": "new",
    }
    value = semantics.validate_understanding(source, raw)
    constraint = value["tasks"][1]
    assert constraint["goal_role"] == "constraint"
    assert constraint["is_negated"] is True
    assert constraint["action"] == "answer"
    assert all(not item["side_effect"] for item in value["validated_tasks"])
    assert all(item["action"] != "record" for item in value["validated_tasks"])


def test_hypothetical_answer_through_real_agent_has_zero_business_sql(compact_db, monkeypatch):
    from test_retry_operations import RetryScenario
    from sqlalchemy import text
    from app.services.agent import semantics as semantic_module

    source = "假如明晚吃180克鸡肉，这样说会不会被保存？我现在没吃，不要登记。"
    raw = {
        "tasks": [
            task("假如明晚吃180克鸡肉，这样说会不会被保存？", kind="nutrition_record",
                 action="question", is_hypothetical=True, goal_role="result", quoted=False,
                 reply_mode="conversation", reply_purpose="execution_status",
                 fact_text=None, context_ids=[], fact_fields=[], critical_fact_fields=[]),
            task("我现在没吃，不要登记。", kind="nutrition_record", action="answer",
                 is_negated=True, goal_role="constraint", quoted=False, reply_mode="conversation",
                 reply_purpose="general", fact_text=None, context_ids=[],
                 fact_fields=[], critical_fact_fields=[]),
        ],
        "safety": safety(), "reply_mode": "conversation",
        "reply_purpose": "execution_status", "relation": "new",
    }

    async def complete(messages, schema, *, name=None, **kwargs):
        if name == "return_request_understanding":
            value = schema.model_validate(raw)
            if kwargs.get("validator"):
                kwargs["validator"](value)
            return value
        if name == "return_dialogue_answer":
            return schema.model_validate({
                "answer": "不会保存。", "requires_grounding": False,
                "status_kind": "save_permission",
            })
        raise AssertionError(name)

    scenario = RetryScenario(monkeypatch, compact_db)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", complete)
    token = semantic_module.request_understanding.set({})
    try:
        response = asyncio.run(scenario.respond(ChatRequest(message=source)))
    finally:
        semantic_module.request_understanding.reset(token)
    assert response.answer == "不会保存。"
    assert not scenario.saves and not scenario.tool_calls
    with compact_db() as session:
        for table in ("user_food_log", "nutrition_logs", "workout_records", "user_facts"):
            assert session.execute(text("SELECT COUNT(*) FROM " + table)).scalar() == 0


def test_real_nutrition_record_still_requires_an_authorized_write():
    source = "今天早餐实际吃了45克燕麦，请保存成今天早餐的饮食记录。"
    raw = outline(source, task_extra={"kind": "nutrition_record", "action": "record"})
    raw["tasks"][0]["fact_text"] = "今天早餐实际吃了45克燕麦"
    value = semantics.validate_understanding(source, raw)
    assert value["validated_tasks"][0]["side_effect"] is True


# ---------------------------------------------------------------------------
# USER-LIVE29-02: optional Wiki addressing failure must not block a
# self-contained record, while a real history-only goal still fails closed.
# ---------------------------------------------------------------------------

class WikiFailureModel:
    def __init__(self, understanding: dict):
        self.understanding = understanding
        self.calls: list[str] = []

    def configured(self):
        return True

    async def complete_structured(self, messages, schema, *, name, **kwargs):
        self.calls.append(name)
        if name == "return_wiki_read_plan":
            return schema.model_validate({"query": "燕麦", "page_ids": [], "done": False})
        if name == "return_request_understanding":
            value = schema.model_validate(self.understanding)
            if kwargs.get("validator"):
                kwargs["validator"](value)
            return value
        raise AssertionError(name)


def wiki_memory() -> dict:
    return {
        "short_memory": [],
        "long_memory": {},
        "current_states": {},
        "all_current_states": {},
        "persistent_constraints": {},
        "all_persistent_constraints": {},
        "versioned_view": True,
        "memory_epoch": 0,
        "wiki_conversation_id": "live-wiki",
        "wiki_catalog": {
            "entries": [{
                "page_id": "page-1", "path": "/old", "title": "旧页", "revision": 1,
                "turn_id": "old-turn", "conversation_id": "live-wiki",
                "content_hash": "hash", "completed_at": None,
                "estimated_tokens": 16, "preview": "旧内容", "references": [],
                "critical": False, "current_anchor": False, "wiki_view": "current",
            }],
            "query": "燕麦", "next_offset": 0, "has_more": False,
            "has_history": False, "catalog_tokens": 16,
        },
    }


def run_with_wiki_round_failure(monkeypatch, understanding: dict, source: str):
    monkeypatch.setattr(settings, "wiki_memory_max_rounds", 2)
    monkeypatch.setattr(wc, "catalog", lambda *args, **kwargs: {
        "entries": [], "query": "燕麦", "next_offset": 0, "has_more": False,
        "has_history": False, "catalog_tokens": 0,
    })
    model = WikiFailureModel(understanding)
    return asyncio.run(semantics.understand(model, source, context=wiki_memory())), model


def test_wiki_round_limit_does_not_block_self_contained_record(monkeypatch):
    source = "今天早餐实际吃了45克燕麦，请保存成今天早餐的饮食记录。"
    raw = outline(source, task_extra={"kind": "nutrition_record", "action": "record"})
    raw["tasks"][0]["fact_text"] = "今天早餐实际吃了45克燕麦"
    result, model = run_with_wiki_round_failure(monkeypatch, raw, source)
    assert result is not None
    assert result["validated_tasks"][0]["side_effect"] is True
    assert model.calls == ["return_request_understanding"]


def test_missing_history_requirement_in_wiki_request_fails_closed(monkeypatch):
    source = "请解释一般成年人力量训练后如何安排休息，只讲通用原则。"
    raw = outline(source, task_extra={"kind": "training_qa", "action": "question",
                                      "reply_mode": "grounded", "reply_purpose": "general"})
    raw["tasks"][0].pop("history_requirement")
    model = WikiFailureModel(raw)
    monkeypatch.setattr(settings, "wiki_memory_max_rounds", 2)
    result = asyncio.run(semantics.understand(model, source, context=wiki_memory()))
    assert result is None
    assert model.calls == ["return_request_understanding"]


def test_wiki_round_limit_still_blocks_explicit_history_followup(monkeypatch):
    source = "继续刚才那个训练计划，把第二天的动作换成无器械的。"
    raw = outline(source, task_extra={"kind": "training_qa", "action": "question",
                                      "reply_mode": "conversation", "reply_purpose": "general",
                                      "history_requirement": "required"},
                   relation="follow_up")
    with pytest.raises(wc.WikiReadError):
        run_with_wiki_round_failure(monkeypatch, raw, source)


# ---------------------------------------------------------------------------
# USER-LIVE29-03: canonical transient state identity; reference does not
# create a new state or extend TTL; clear removes aliases.
# ---------------------------------------------------------------------------

def commit_state(source: str, cid: str, state: dict, *, memory: dict | None = None) -> str:
    value = semantics.validate_understanding(source, {
        "tasks": [task(source, kind="chitchat", action="answer",
                       reply_mode="conversation", reply_purpose="general")],
        "safety": safety(), "context_states": [state],
    }, context=memory or {})
    tid, _ = compact.begin_turn(ChatRequest(message=source, conversation_id=cid))
    compact.finish_turn(tid, "收到", {"answer": "收到"}, semantic_values=[(source, value)])
    return tid


@pytest.mark.parametrize("source,key,action,scope", [
    ("手边只有弹力带", "available_equipment", "set", "conversation"),
    ("按我刚才说的当前器械给建议", "training:available_equipment", "reference", "conversation"),
    ("解除刚才的临时器械条件", "training:available_equipment", "clear", "conversation"),
])
def test_transient_state_identity_is_canonical_and_reference_does_not_renew(compact_db, source, key, action, scope):
    cid = "live-state"
    first = commit_state("手边只有弹力带", cid, {
        "key": "available_equipment", "quote": "手边只有弹力带",
        "action": "set", "scope": "conversation", "domain": "training",
    })
    before = lc.snapshot(cid)
    original = before["current_states"]["context:training:available_equipment"]
    with compact_db() as session:
        count_before = session.scalar(select(func.count()).select_from(MemoryState).where(
            MemoryState.kind == "transient")) or 0
    memory = lc.snapshot(cid)
    if action == "reference":
        commit_state(source, cid, {
            "key": key, "quote": source, "action": action, "scope": scope,
            "domain": "training",
        }, memory=memory)
    elif action == "clear":
        # Simulate the pre-fix alias that must not survive an explicit clear.
        with compact_db() as session, session.begin():
            session.add(MemoryState(
                state_id="legacy-alias", owner_id="demo-user", conversation_id=cid,
                scope="conversation", logical_key="training:training:available_equipment",
                kind="transient", domain="training", version=0, value="旧别名",
                status="active", valid_from=compact.now(),
                expires_at=compact.now() + timedelta(hours=1),
                source_turn_id=first, source_message_id=first + ":user", memory_epoch=0,
            ))
        commit_state(source, cid, {
            "key": key, "quote": source, "action": action, "scope": scope,
            "domain": "training",
        }, memory=memory)
    after = lc.snapshot(cid)
    if action == "reference":
        assert after["current_states"]["context:training:available_equipment"]["expires_at"] == original["expires_at"]
        with compact_db() as session:
            assert session.scalar(select(func.count()).select_from(MemoryState).where(
                MemoryState.kind == "transient")) == count_before
    elif action == "clear":
        assert not after["current_states"]
        with compact_db() as session:
            alias = session.get(MemoryState, "legacy-alias")
            assert alias.status != "active"
    else:
        assert after["current_states"]["context:training:available_equipment"]["expires_at"]


@pytest.mark.parametrize("alias", [
    "training:" * 8 + "available_equipment",
    "training:context:training:available_equipment",
])
def test_clear_removes_legacy_alias_frames_within_field_length(compact_db, alias):
    cid = "legacy-alias-frame"
    first = commit_state("器械只有弹力带", cid, {
        "key": "available_equipment", "quote": "器械只有弹力带",
        "action": "set", "scope": "conversation", "domain": "training",
    })
    with compact_db.begin() as session:
        session.add(MemoryState(
            state_id="legacy-frame-" + str(len(alias)), owner_id="demo-user",
            conversation_id=cid, scope="conversation", logical_key=alias,
            kind="transient", domain="training", version=0, value="旧别名",
            status="active", valid_from=compact.now(),
            expires_at=compact.now() + timedelta(hours=1),
            source_turn_id=first, source_message_id=first + ":user", memory_epoch=0,
        ))
    commit_state("解除器械限制", cid, {
        "key": "available_equipment", "quote": "解除器械限制",
        "action": "clear", "scope": "conversation", "domain": "training",
    }, memory=lc.snapshot(cid))
    assert not lc.snapshot(cid)["current_states"]
    with compact_db() as session:
        row = session.scalar(select(MemoryState).where(MemoryState.logical_key == alias))
        assert row is not None and row.status != "active"


def test_clear_chooses_latest_canonical_clear_before_alias_resurrection(compact_db):
    cid = "canonical-latest"
    first = commit_state("器械只有弹力带", cid, {
        "key": "available_equipment", "quote": "器械只有弹力带",
        "action": "set", "scope": "conversation", "domain": "training",
    })
    with compact_db.begin() as session:
        session.add(MemoryState(
            state_id="older-active-alias", owner_id="demo-user", conversation_id=cid,
            scope="conversation", logical_key="training:training:available_equipment",
            kind="transient", domain="training", version=0, value="旧别名",
            status="active", valid_from=compact.now(),
            expires_at=compact.now() + timedelta(hours=1),
            source_turn_id=first, source_message_id=first + ":user", memory_epoch=0,
        ))
    commit_state("解除器械限制", cid, {
        "key": "available_equipment", "quote": "解除器械限制",
        "action": "clear", "scope": "conversation", "domain": "training",
    }, memory=lc.snapshot(cid))
    assert not lc.snapshot(cid)["current_states"]


@pytest.mark.parametrize("other_count", [12, 2501])
def test_write_canonical_lookup_preserves_unrelated_active_fields(compact_db, other_count):
    cid = "write-canonical-target"
    first = commit_state("器械只有弹力带", cid, {
        "key": "available_equipment", "quote": "器械只有弹力带",
        "action": "set", "scope": "conversation", "domain": "training",
    })
    with compact_db.begin() as session:
        session.add_all([MemoryState(
            state_id=f"other-active-{i}", owner_id="demo-user",
            conversation_id=cid, scope="conversation",
            logical_key=f"training:other_attribute_{i}", kind="transient",
            domain="training", version=0, value="其他器械",
            status="active", valid_from=compact.now(),
            expires_at=compact.now() + timedelta(hours=1),
            source_turn_id=first, source_message_id=first + ":user",
            memory_epoch=0,
        ) for i in range(other_count)])
    memory = lc.snapshot(cid)
    commit_state("解除器械限制", cid, {
        "key": "available_equipment", "quote": "解除器械限制",
        "action": "clear", "scope": "conversation", "domain": "training",
    }, memory=memory)
    with compact_db() as session:
        retained = session.scalar(select(func.count()).select_from(MemoryState).where(
            MemoryState.conversation_id == cid,
            MemoryState.state_id.like("other-active-%"),
            MemoryState.status == "active"))
        assert retained == other_count
    assert "context:training:available_equipment" not in lc.snapshot(cid)["current_states"]


def test_canonical_framing_is_case_sensitive_and_deep_mixed_framing_normalizes(compact_db):
    cid = "framing-case"
    first = commit_state("器械只有弹力带", cid, {
        "key": "available_equipment", "quote": "器械只有弹力带",
        "action": "set", "scope": "conversation", "domain": "training",
    })
    alias = "training:training:context:training:context:training:available_equipment"
    assert lc.canonical_context_key("training", alias) == "available_equipment"
    with compact_db.begin() as session:
        session.add(MemoryState(
            state_id="mixed-frame-active", owner_id="demo-user",
            conversation_id=cid, scope="conversation", logical_key=alias,
            kind="transient", domain="training", version=0, value="旧器械",
            status="active", valid_from=compact.now(),
            expires_at=compact.now() + timedelta(hours=1),
            source_turn_id=first, source_message_id=first + ":user",
            memory_epoch=0,
        ))
    # Clear the canonical identity; the mixed-frame alias must not survive.
    commit_state("解除器械限制", cid, {
        "key": "available_equipment", "quote": "解除器械限制",
        "action": "clear", "scope": "conversation", "domain": "training",
    }, memory=lc.snapshot(cid))
    assert not lc.snapshot(cid)["current_states"]

    # A case-different non-framing attribute is a distinct identity.
    other = commit_state("另一条器械条件", cid, {
        "key": "available_equipment", "quote": "另一条器械条件",
        "action": "set", "scope": "conversation", "domain": "training",
    })
    with compact_db.begin() as session:
        session.add(MemoryState(
            state_id="case-sensitive-attribute", owner_id="demo-user",
            conversation_id=cid, scope="conversation",
            logical_key="training:Training:available_equipment",
            kind="transient", domain="training", version=0, value="大小写属性",
            status="active", valid_from=compact.now(),
            expires_at=compact.now() + timedelta(hours=1),
            source_turn_id=other, source_message_id=other + ":user",
            memory_epoch=0,
        ))
    assert len(lc.snapshot(cid)["current_states"]) == 2


def test_late_old_clear_cannot_override_newer_alias_and_max_version_wins(compact_db):
    cid = "version-max"
    first = commit_state("器械只有弹力带", cid, {
        "key": "available_equipment", "quote": "器械只有弹力带",
        "action": "set", "scope": "conversation", "domain": "training",
    })
    with compact_db.begin() as session:
        session.add(MemoryState(
            state_id="newer-alias", owner_id="demo-user", conversation_id=cid,
            scope="conversation",
            logical_key="training:training:available_equipment",
            kind="transient", domain="training", version=9, value="更新的器械",
            status="active", valid_from=compact.now(),
            expires_at=compact.now() + timedelta(hours=1),
            source_turn_id=first, source_message_id=first + ":user",
            memory_epoch=0,
        ))
    clear_item = {
        "kind": "transient", "conversation_id": cid, "scope": "conversation",
        "domain": "training", "logical_key": "training:available_equipment",
        "value": "解除", "status": "cleared",
        "source_turn_id": first, "source_message_id": first + ":user",
    }
    with compact_db() as session, session.begin():
        lc.write_versions(session, [clear_item], owner="demo-user", epoch=0,
                          version=5, instant=compact.now())
    assert lc.snapshot(cid)["current_states"]["context:training:available_equipment"]["text"] == "更新的器械"
    with compact_db() as session, session.begin():
        lc.write_versions(session, [clear_item], owner="demo-user", epoch=0,
                          version=10, instant=compact.now())
    assert not lc.snapshot(cid)["current_states"]


def test_apply_labels_reference_keeps_stored_identity_without_local_ttl(compact_db):
    cid = "live-state-reference"
    commit_state("手边只有弹力带", cid, {
        "key": "available_equipment", "quote": "手边只有弹力带",
        "action": "set", "scope": "conversation", "domain": "training",
    })
    memory = lc.snapshot(cid)
    source = "按我刚才说的当前器械给建议"
    value = semantics.validate_understanding(source, {
        "tasks": [task(source, kind="chitchat", action="answer",
                       reply_mode="conversation", reply_purpose="general")],
        "safety": safety(),
        "context_states": [{"key": "context:training:available_equipment", "quote": source,
                            "action": "reference", "scope": "conversation", "domain": "training"}],
    }, context=memory)
    before = memory["current_states"]["context:training:available_equipment"]
    lc.apply_labels(memory, value)
    assert memory["lifecycle_local_states"] == {}
    assert memory["lifecycle_cleared_states"] == []
    assert memory["all_current_states"]["context:training:available_equipment"]["expires_at"] == before["expires_at"]


# ---------------------------------------------------------------------------
# USER-LIVE29-04: explicit output format is carried by the structured result,
# not inferred from a word inside the answer path.
# ---------------------------------------------------------------------------

def test_single_sentence_format_is_a_structured_task_field():
    source = "姐姐发来一句：“我今天壶铃硬拉40公斤，2组，每组10次。”请替她复述成一句话，这是她的训练，别存到我的记录里。"
    raw = outline(source, task_extra={
        "quoted": True, "is_third_party": True, "reply_mode": "conversation",
        "reply_purpose": "general", "output_format": "single_sentence",
    })
    value = semantics.validate_understanding(source, raw)
    assert value["tasks"][0]["output_format"] == "single_sentence"


def test_dialogue_model_receives_single_sentence_contract_and_no_extra_write():
    source = "姐姐发来一句：“我今天壶铃硬拉40公斤，2组，每组10次。”请替她复述成一句话，这是她的训练，别存到我的记录里。"
    prompts: list[str] = []

    async def complete(messages, schema, **kwargs):
        prompts.append(messages[0]["content"])
        assert "一句话" in messages[0]["content"] or "one sentence" in messages[0]["content"].lower()
        value = schema.model_validate({
            "answer": "姐姐今天壶铃硬拉40公斤，2组×10次。",
            "requires_grounding": False,
        })
        if kwargs.get("validator"):
            kwargs["validator"](value)
        return value

    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(side_effect=complete))
    result = asyncio.run(dc.conversation_answer(
        llm, ChatRequest(message=source), {},
        {"reply_mode": "conversation", "reply_purpose": "general", "quoted": True,
         "is_third_party": True, "output_format": "single_sentence", "tasks": []},
    ))
    assert result and result["answer"] == "姐姐今天壶铃硬拉40公斤，2组×10次。"
    assert prompts


def test_single_sentence_rejects_real_versions_but_keeps_decimals():
    assert dc._single_sentence_answer("姐姐说：“我今天壶铃硬拉40公斤，2组，每组10次。”")
    assert dc._single_sentence_answer("“姐姐今天做了壶铃硬拉40公斤，2组，每组10次。”")
    assert dc._single_sentence_answer("重量40.5公斤，2组×10次。")
    assert not dc._single_sentence_answer("版本一：40公斤，2组×10次。版本二：35公斤，3组×8次。")
    assert not dc._single_sentence_answer("第一句。第二句。")


def test_planner_and_react_scope_preserve_structured_output_format():
    from app.services.agent.react_readonly_workers import _dialogue_scope
    from app.services.agent.task_planner import plan_tasks_rule
    source = "姐姐发来一句：“我今天壶铃硬拉40公斤，2组，每组10次。”请替她复述成一句话，这是她的训练，别存到我的记录里。"
    raw = outline(source, task_extra={
        "quoted": True, "is_third_party": True, "reply_mode": "conversation",
        "reply_purpose": "general", "output_format": "single_sentence",
    })
    value = semantics.validate_understanding(source, raw)
    semantics.remember_understanding(source, value)
    planned = plan_tasks_rule(source)
    assert planned and planned[0]["payload"]["dialogue"]["output_format"] == "single_sentence"
    assert planned[0]["payload"]["semantic"]["output_format"] == "single_sentence"
    assert planned[0]["payload"]["dialogue"]["history_requirement"] == "none"
    assert _dialogue_scope({"semantic_scope": planned[0]["payload"]["semantic"]})[
        "output_format"] == "single_sentence"


def test_required_history_requirement_propagates_to_planner_and_react_scope():
    from app.services.agent.react_readonly_workers import _dialogue_scope
    from app.services.agent.task_planner import plan_tasks_rule
    source = "照上次重量做了3组，每组10次，帮我保存今天的卧推。"
    raw = outline(source, task_extra={
        "kind": "workout_record", "action": "record",
        "fact_text": "照上次重量做了3组，每组10次",
        "history_requirement": "required",
    })
    value = semantics.validate_understanding(source, raw)
    semantics.remember_understanding(source, value)
    planned = plan_tasks_rule(source)
    assert planned[0]["payload"]["semantic"]["history_requirement"] == "required"
    assert _dialogue_scope({"semantic_scope": planned[0]["payload"]["semantic"]})[
        "history_requirement"] == "required"


def test_default_dialogue_output_is_not_silently_shrunk():
    prompts: list[str] = []
    answer = "第一段说明。第二段说明。"

    async def complete(messages, schema, **kwargs):
        prompts.append(messages[0]["content"])
        value = schema.model_validate({"answer": answer, "requires_grounding": False})
        if kwargs.get("validator"):
            kwargs["validator"](value)
        return value

    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(side_effect=complete))
    result = asyncio.run(dc.conversation_answer(
        llm, ChatRequest(message="给我一个正常回答"), {},
        {"reply_mode": "conversation", "reply_purpose": "general", "tasks": []},
    ))
    assert result and result["answer"] == answer
    assert "恰好一句" not in prompts[0]


# ---------------------------------------------------------------------------
# USER-LIVE29-05/06: accounting export happens while the isolated factory is
# alive, before dispose/drop, and even when the run failed.
# ---------------------------------------------------------------------------

def _import_runner():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import run_ai_hybrid_simulation as runner
    return runner


def test_finish_database_exports_quota_ledger_and_sources_before_dispose(compact_db, tmp_path, monkeypatch):
    from app.db import memory_models  # registers user_facts/memory tables
    from app.db.models import Base
    from app.db.model_budget_models import ModelCallLedger, ModelQuota
    runner = _import_runner()
    monkeypatch.setattr(runner, "OUT", tmp_path)
    actions: list[str] = []
    Base.metadata.create_all(compact_db.kw["bind"], checkfirst=True)
    for model in (ModelQuota, ModelCallLedger):
        model.__table__.create(compact_db.kw["bind"], checkfirst=True)
    with compact_db() as session, session.begin():
        session.add(ModelQuota(
            quota_id="quota-1", kind="task", calls=2, tokens=1234,
            max_calls=12, max_tokens=64000,
            created_at=compact.now(), expires_at=compact.now() + timedelta(hours=1),
        ))
        session.add(ModelCallLedger(
            call_id="call-1", scope_id="scope-1", kind="task", capability="chat",
            model="synthetic", stage_id="stage-1", quota_ids=["quota-1"],
            reserved_tokens=100, status="started", usage=None,
            started_at=compact.now(), finished_at=None,
        ))
    engine = SimpleNamespace(dispose=lambda: actions.append("dispose"))
    temp = SimpleNamespace(created=True)
    temp.drop = lambda: (actions.append("drop"), setattr(temp, "created", False))
    report = {"execution_complete": False, "error_type": "RemoteProtocolError",
              "stop_code": "provider_async_failure"}
    runner.finish_database(report, compact_db, engine, temp, fixture_ready=True)
    assert actions == ["dispose", "drop"]
    assert report["temporary_database_removed"] is True
    assert "accounting_export_error_type" not in report, report
    exported = json.loads((tmp_path / "task-quota-audit.json").read_text(encoding="utf-8"))
    assert exported["readonly_export"] is True
    assert exported["quotas"][0]["calls"] == 2 and exported["quotas"][0]["tokens"] == 1234
    assert exported["ledger"][0]["status"] == "started"
    assert exported["ledger"][0]["usage"] is None


def test_finish_database_cleans_up_when_accounting_export_fails(compact_db, tmp_path, monkeypatch):
    runner = _import_runner()
    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(runner, "export_accounting", lambda factory: (_ for _ in ()).throw(ValueError("export failure")))
    actions: list[str] = []
    engine = SimpleNamespace(dispose=lambda: actions.append("dispose"))
    temp = SimpleNamespace(created=True)
    temp.drop = lambda: (actions.append("drop"), setattr(temp, "created", False))
    report = {"execution_complete": False, "error_type": "RemoteProtocolError"}
    runner.finish_database(report, compact_db, engine, temp, fixture_ready=True)
    assert actions == ["dispose", "drop"]
    assert report["accounting_export_error_type"] == "ValueError"
    assert report["temporary_database_removed"] is True


def test_finish_database_reports_cleanup_failure_after_dispose(compact_db, tmp_path, monkeypatch):
    runner = _import_runner()
    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(runner, "export_fixture", lambda factory: {"public_vectors": 0})
    monkeypatch.setattr(runner, "export_accounting", lambda factory: {"quotas": [], "ledger": []})
    actions: list[str] = []
    engine = SimpleNamespace(dispose=lambda: actions.append("dispose"))
    temp = SimpleNamespace(created=True)
    temp.drop = lambda: (_ for _ in ()).throw(RuntimeError("drop failure"))
    report = {"execution_complete": False, "error_type": "RemoteProtocolError"}
    runner.finish_database(report, compact_db, engine, temp, fixture_ready=True)
    assert actions == ["dispose"]
    assert report["cleanup_error_type"] == "RuntimeError"
    assert "temporary_database_removed" not in report or report["temporary_database_removed"] is False


def test_finish_database_reports_dispose_failure_and_still_drops(compact_db, tmp_path, monkeypatch):
    runner = _import_runner()
    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(runner, "export_fixture", lambda factory: {"public_vectors": 0})
    monkeypatch.setattr(runner, "export_accounting", lambda factory: {"quotas": [], "ledger": []})
    actions: list[str] = []
    engine = SimpleNamespace(dispose=lambda: (_ for _ in ()).throw(RuntimeError("dispose failure")))
    temp = SimpleNamespace(created=True)
    temp.drop = lambda: (actions.append("drop"), setattr(temp, "created", False))
    report = {"execution_complete": False, "error_type": "RemoteProtocolError"}
    runner.finish_database(report, compact_db, engine, temp, fixture_ready=True)
    assert actions == ["drop"]
    assert report["dispose_error_type"] == "RuntimeError"
    assert report["temporary_database_removed"] is True


def test_main_finally_exports_real_unknown_ledger_after_mock_remote_protocol_error(tmp_path, monkeypatch):
    """Exercise main/run_cases failure and the shared finally, not a hand report."""
    import httpx
    import logging
    import sys
    from types import SimpleNamespace
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import quality_isolation
    from app.db import memory_models  # noqa: F401 - registers memory tables
    from app.db import session as db
    from app.db.model_budget_models import ModelCallLedger, ModelQuota
    from app.db.models import Base, UserProfileModel

    runner = _import_runner()
    monkeypatch.setattr(runner, "OUT", tmp_path)
    db_path = tmp_path / "simulation.db"
    engine = create_engine("sqlite:///" + str(db_path))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    actions: list[str] = []
    real_dispose = engine.dispose
    engine.dispose = lambda: (actions.append("dispose"), real_dispose())
    Base.metadata.create_all(engine)
    ModelQuota.__table__.create(engine, checkfirst=True)
    ModelCallLedger.__table__.create(engine, checkfirst=True)

    class FakeTemp:
        created = False
        name = "fake-simulation-db"
        url = "sqlite:///" + str(db_path)

        def create(self):
            self.created = True

        def drop(self):
            actions.append("drop")
            self.created = False

    monkeypatch.setattr(quality_isolation, "TempPGDatabase", FakeTemp)
    monkeypatch.setattr(quality_isolation, "checked_engine", lambda _url: engine)
    monkeypatch.setattr(db, "init_db", lambda: bool(Base.metadata.create_all(engine) or True))
    monkeypatch.setattr(runner, "prepare_public_knowledge", lambda *_a, **_k: {})
    monkeypatch.setattr(runner, "require_live_knowledge", lambda *_a, **_k: {})
    from app.services.rag import ingestion
    monkeypatch.setattr(ingestion, "backfill_missing_vectors",
                        lambda **_k: SimpleNamespace(as_dict=lambda: {}))
    monkeypatch.setattr(runner.shared, "endpoint_map", lambda _settings: {})
    monkeypatch.setattr(sys, "argv", ["run_ai_hybrid_simulation.py", "--live",
                                      "--allow-unpriced-limited"])

    async def fail_run(report, factory, limiter, sql):
        with factory() as session, session.begin():
            session.add(ModelQuota(
                quota_id="q-failed", kind="task", calls=1, tokens=0,
                max_calls=12, max_tokens=64000,
                created_at=compact.now(), expires_at=compact.now() + timedelta(hours=1)))
            session.add(ModelCallLedger(
                call_id="c-failed", scope_id="scope-failed", kind="task",
                capability="chat", model="synthetic", stage_id="stage-failed",
                quota_ids=["q-failed"], reserved_tokens=100, status="started",
                usage=None, started_at=compact.now(), finished_at=None))
        limiter.state["calls"].append({"number": 1, "capability": "chat",
                                       "status": "dispatched_cost_unknown"})
        raise httpx.RemoteProtocolError("mock remote protocol error")

    monkeypatch.setattr(runner, "run_cases", fail_run)
    try:
        exit_code = runner.main()
    finally:
        logging.disable(logging.NOTSET)
    report = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["error_type"] == "RemoteProtocolError"
    assert report["temporary_database_removed"] is True
    assert actions == ["dispose", "drop"]
    assert report["retained_accounting"] == {"quotas": 1, "ledger": 1, "facts": 0}
    audit = json.loads((tmp_path / "task-quota-audit.json").read_text(encoding="utf-8"))
    assert audit["quotas"][0]["calls"] == 1
    assert audit["ledger"][0]["status"] == "started"
    assert audit["ledger"][0]["usage"] is None
    request_ledger = json.loads((tmp_path / "request-ledger.json").read_text(encoding="utf-8"))
    assert request_ledger["calls"][0]["status"] == "dispatched_cost_unknown"
    with factory() as session:
        assert session.query(ModelCallLedger).count() == 1
        assert session.query(ModelQuota).one().calls == 1
