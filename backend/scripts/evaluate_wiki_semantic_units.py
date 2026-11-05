"""Offline end-to-end evaluation for Wiki semantic-unit assembly.

Unlike a pure block fixture, this runner creates an isolated SQLite database,
commits real ``MemoryTurn``/Wiki pages, calls the real ``compact.load`` and
``wiki_context.prepare`` (which calls ``read_pages``), then measures the final
``ContextBudgetManager`` assembly and a dialogue answer call.  Models are
scripted, provider credentials are not used, and the database is temporary.

The report is structural only: no real answer-quality or extraction-accuracy
claim is made.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import timedelta
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

# The script lives under backend/scripts; make backend importable for the
# existing isolated-database helper and application modules.
BACKEND = Path(__file__).resolve().parents[1]
for path in (BACKEND, BACKEND / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.core.config import settings
from app.db.memory_models import MemoryTurn, UserFact
from app.db.wiki_models import WikiPage
from app.schemas.chat import ChatRequest
from app.services.agent import dialogue_context, semantics
from app.services.context.budget import ContextBudgetExceeded, ContextBudgetManager
from app.services.memory import compact, semantic_units
from app.services.memory import wiki_context as wc


class Patch:
    """Minimal monkeypatch context used by tests.quality_isolation."""

    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value):
        self._undo.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def context(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        for target, name, value in reversed(self._undo):
            setattr(target, name, value)
        self._undo.clear()
        return False


def add_turn(cid, message, answer):
    tid, _ = compact.begin_turn(ChatRequest(message=message, conversation_id=cid))
    compact.finish_turn(tid, answer, {"answer": answer})
    return tid


def load(cid, query="继续"):
    legacy = SimpleNamespace(initialize=AsyncMock(return_value=False),
                             load=AsyncMock(return_value={}))
    return asyncio.run(compact.CompactMemory(legacy).load("demo-user", cid, query))


def page_id(memory, turn_id):
    return next(entry["page_id"] for entry in memory["wiki_catalog"]["entries"]
                if entry["turn_id"] == turn_id)


def backdate(turn_id, days):
    with compact.session() as s, s.begin():
        s.get(MemoryTurn, turn_id).completed_at = compact.now() - timedelta(days=days)


class FakeLLM:
    def __init__(self, plans, quotes=None, *, fail_extraction=False, details=None):
        self.plans = iter(plans)
        self.quotes = dict(quotes or {})
        self.details = dict(details or {})
        self.fail_extraction = fail_extraction
        self.calls = []
        self.last_messages = []

    def configured(self):
        return True

    async def complete_structured(self, messages, schema, *, name, **kwargs):
        self.calls.append(name)
        self.last_messages = messages
        if name == "return_wiki_read_plan":
            value = next(self.plans)
        elif name == "return_wiki_semantic_units":
            if self.fail_extraction:
                raise semantic_units.StructuredOutputError(
                    "semantic_unit_model_unavailable", [{"type": "fixture"}])
            payload = json.loads(messages[-1]["content"])
            pages = []
            for page in payload["pages"]:
                wanted = self.quotes.get(page["page_id"])
                found = None
                if wanted:
                    found = next((message for message in page["messages"]
                                  if wanted in str(message.get("content") or "")), None)
                if found is None:
                    pages.append({"page_id": page["page_id"], "units": [],
                                  "complete": False,
                                  "reason": "fixture source not confirmed"})
                    continue
                detail = self.details.get(page["page_id"], {})
                pages.append({
                    "page_id": page["page_id"], "complete": True, "reason": "",
                    "units": [{
                        "unit_id": "fixture-" + page["page_id"],
                        "message_id": found["message_id"], "quote": wanted,
                        "claim": detail.get("claim", "fixture unit"),
                        "subject": detail.get("subject", "fixture"),
                        "conditions": detail.get("conditions", []),
                        "validity_text": detail.get("validity_text"),
                        "temporal_stability": detail.get("temporal_stability", "stable"),
                        "temporal_sensitivity": detail.get("temporal_sensitivity", 0.0),
                        "temporal_basis_quote": detail.get("temporal_basis_quote"),
                        "relevance": detail.get("relevance", 0.9),
                        "depends_on": [], "complete": True,
                    }],
                })
            value = {"pages": pages}
        elif name == "return_dialogue_answer":
            self.last_messages = messages
            value = {"answer": "按已读原文回答。", "requires_grounding": False}
        else:
            raise AssertionError(name)
        result = schema.model_validate(value)
        if kwargs.get("validator"):
            kwargs["validator"](result)
        return result


def assemble(memory, *, window=1200, understanding=None, units=True):
    state = dialogue_context.answer_memory(memory, understanding) if understanding else memory
    if not units:
        state = {**state, "wiki_units": [], "wiki_selected_units": []}
    manager = ContextBudgetManager(window=window, output=100, margin=0)
    result = manager.assemble(system="规则", query="继续", state=state, hits=[])
    return result, json.dumps(result, ensure_ascii=False)


def prepare(memory, plan, quotes, *, fail=False, query="继续", details=None):
    llm = FakeLLM([plan], quotes, fail_extraction=fail, details=details)
    asyncio.run(wc.prepare(llm, query, memory, semantics.PROMPT))
    return llm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    options = parser.parse_args()
    from tests.quality_isolation import isolated_database

    patches = Patch()
    cases = {}
    with TemporaryDirectory() as directory:
        with isolated_database(patches, Path(directory), "sqlite"):
            # Settings are test-only: independent source and extraction budgets.
            settings.wiki_enabled = True
            settings.wiki_semantic_units_enabled = True
            settings.wiki_source_read_max_tokens = 14000
            settings.wiki_semantic_unit_max_input_tokens = 16000
            settings.llm_context_window_tokens = 65536

            # 1. Large page: real prepare/read, final budget cannot fit page.
            cid = "eval-large"
            background = "背景原文" * 700
            short = "周一在室内训练，但不对花生过敏，持续到月底。"
            large_turn = add_turn(cid, "我的训练安排", background + short)
            memory = load(cid, "我的训练安排")
            large_page = page_id(memory, large_turn)
            entry = next(item for item in memory["wiki_catalog"]["entries"]
                         if item["page_id"] == large_page)
            prepare(memory, {"page_ids": [large_page]},
                    {large_page: short}, query="我的训练安排",
                    details={large_page: {
                        "conditions": [
                            {"condition": "室内训练", "negated": False,
                             "quote": "室内训练"},
                            {"condition": "不对花生过敏", "negated": True,
                             "quote": "不对花生过敏"},
                        ],
                        "validity_text": "持续到月底",
                    }})
            whole, whole_text = assemble(memory, units=False)
            units, unit_text = assemble(memory, units=True)
            full_whole, full_whole_text = assemble(memory, window=30000, units=False)
            full_units, full_unit_text = assemble(memory, window=30000, units=True)
            cases["large_page"] = {
                "catalog_estimated_tokens": entry["estimated_tokens"],
                "source_read_tokens": memory["wiki_read_tokens"],
                "source_read_limit": memory["wiki_read_limit"],
                "whole_page_covered": short in whole_text,
                "unit_covered": short in unit_text,
                "unit_hides_full_background": background not in unit_text,
                "whole_tokens": whole.report["used_tokens"],
                "unit_tokens": units.report["used_tokens"],
                "full_fit_whole_covered": short in full_whole_text,
                "full_fit_unit_covered": short in full_unit_text,
                "full_fit_tokens": {"whole": full_whole.report["used_tokens"],
                                    "units": full_units.report["used_tokens"]},
                "condition_retained": "室内训练" in unit_text,
                "negation_retained": "不对花生过敏" in unit_text,
                "validity_retained": "持续到月底" in unit_text,
            }

            # 2. Temporal value signal: same pages, fixed as_of, one budget that
            # fits two units. GT is the current short-term fact plus the stable
            # fact; the stale sensitive fact must not win merely by age/order.
            temporal_cid = "eval-temporal"
            old_turn = add_turn(temporal_cid, "旧短期状态：最近改成早上训练。", "收到。")
            new_turn = add_turn(temporal_cid, "新短期状态：这周改成早上训练。", "收到。")
            stable_turn = add_turn(temporal_cid, "长期事实：我对花生过敏。", "收到。")
            backdate(old_turn, 400)
            backdate(stable_turn, 400)
            backdate(new_turn, 1)
            temporal_memory = load(temporal_cid, "我什么时候训练")
            old_page = page_id(temporal_memory, old_turn)
            new_page = page_id(temporal_memory, new_turn)
            stable_page = page_id(temporal_memory, stable_turn)
            prepare(
                temporal_memory,
                {"page_ids": [old_page, new_page, stable_page]},
                {old_page: "旧短期状态：最近改成早上训练。",
                 new_page: "新短期状态：这周改成早上训练。",
                 stable_page: "长期事实：我对花生过敏。"},
                query="我什么时候训练",
                details={
                    old_page: {"claim": "旧下午训练",
                               "temporal_stability": "time_sensitive",
                               "temporal_sensitivity": 1.0,
                               "temporal_basis_quote": "最近", "relevance": .95},
                    new_page: {"claim": "这周早上训练",
                               "temporal_stability": "time_sensitive",
                               "temporal_sensitivity": 1.0,
                               "temporal_basis_quote": "这周", "relevance": .90},
                    stable_page: {"claim": "花生过敏",
                                  "temporal_stability": "stable",
                                  "temporal_sensitivity": 0.0, "relevance": .82},
                })
            temporal_memory["wiki_as_of"] = compact.now().isoformat()
            temporal_units = {unit["claim"]: unit
                              for unit in temporal_memory["wiki_units"]}
            by_claim = {unit["unit_id"]: claim
                        for claim, unit in temporal_units.items()}
            two_state = {**temporal_memory, "wiki_units": [
                temporal_units["这周早上训练"], temporal_units["花生过敏"]]}
            two = assemble(two_state, window=100000)[0]
            temporal_window = two.report["used_tokens"] + 110
            time_on = assemble(temporal_memory, window=temporal_window)[0]
            on_accepted = {by_claim[item.split("unit:", 1)[1]]
                           for item in time_on.report["semantic_unit_selection"]["accepted"]}
            on_signals = {by_claim[item.split("unit:", 1)[1]]: signal
                          for item, signal in time_on.report["semantic_unit_selection"][
                              "temporal_signals"].items()}
            settings.wiki_temporal_decay_enabled = False
            try:
                time_off = assemble(temporal_memory, window=temporal_window)[0]
            finally:
                settings.wiki_temporal_decay_enabled = True
            off_accepted = {by_claim[item.split("unit:", 1)[1]]
                            for item in time_off.report["semantic_unit_selection"]["accepted"]}
            cases["temporal_value_ablation"] = {
                "as_of": temporal_memory["wiki_as_of"],
                "same_budget_tokens": temporal_window - 100,
                "gt_expected": ["这周早上训练", "花生过敏"],
                "time_on_accepted": sorted(on_accepted),
                "time_on_gt_pass": on_accepted == {"这周早上训练", "花生过敏"},
                "time_off_accepted": sorted(off_accepted),
                "time_off_gt_pass": off_accepted == {"这周早上训练", "花生过敏"},
                "time_on_used": time_on.report["used_tokens"],
                "time_off_used": time_off.report["used_tokens"],
                "signals": on_signals,
                "stable_factor_is_one": on_signals["花生过敏"]["factor"] == 1.0,
                "old_sensitive_decayed": on_signals["旧下午训练"]["status"] == "decayed",
            }

            # 3. Same-budget ranking ablation through the real final assembly.
            from app.services.context.budget import ContextBlock
            ranking_blocks = [
                ContextBlock("query", "query",
                             [{"role": "user", "content": "current query"}], 999,
                             required=True),
                ContextBlock("unit:A", "semantic",
                             [{"role": "system", "content": "A" * 1000}], 40,
                             importance=1.0, group_id="unit:A"),
                ContextBlock("unit:B", "semantic",
                             [{"role": "system", "content": "B" * 100}], 40,
                             importance=.8, group_id="unit:B"),
                ContextBlock("unit:C", "semantic",
                             [{"role": "system", "content": "C" * 100}], 40,
                             importance=.79, group_id="unit:C"),
            ]
            def rank_run(mode):
                settings.wiki_unit_ranking = mode
                try:
                    packed = ContextBudgetManager(
                        window=1200, output=100, margin=0).build(ranking_blocks)
                finally:
                    settings.wiki_unit_ranking = "density"
                selection = packed.report["semantic_unit_selection"]
                return {
                    "accepted": selection.get("accepted", []),
                    "skipped": selection.get("skipped", []),
                    "used": packed.report["used_tokens"],
                }
            density = rank_run("density")
            value = rank_run("value")
            cases["ranking_ablation"] = {
                "same_budget_tokens": 1100,
                "gt_density_small_complete_units": ["unit:B", "unit:C"],
                "density": density,
                "value": value,
                "density_gt_pass": density["accepted"] == ["unit:B", "unit:C"],
                "value_gt_pass": value["accepted"] == ["unit:A"],
                "both_within_budget": (density["used"] <= 1100
                                       and value["used"] <= 1100),
            }

            # 4. Deep dependency through actual read_pages/prepare.
            dep_cid = "eval-deep"
            dep_turn = add_turn(dep_cid, "共用依据", "我在家训练。")
            main_turn = add_turn(dep_cid, "主任务", "周一练腿。")
            dep_memory = load(dep_cid, "周一练什么")
            main_page = page_id(dep_memory, main_turn)
            dep_page = page_id(dep_memory, dep_turn)
            prepare(dep_memory, {"page_ids": [main_page],
                                 "dependencies": {main_page: [dep_page]}},
                    {main_page: "周一练腿。", dep_page: "我在家训练。"},
                    query="周一练什么")
            scoped_understanding = {"dialogue_ids": [main_turn + ":user"]}
            selected = dialogue_context.answer_memory(dep_memory, scoped_understanding)
            dep_final, dep_text = assemble(
                dep_memory, window=3000, understanding=scoped_understanding)
            cases["deep_dependency"] = {
                "source_read_tokens": dep_memory["wiki_read_tokens"],
                "selected_unit_pages": sorted({unit["page_id"] for unit in selected["wiki_units"]}),
                "main_covered": "周一练腿。" in dep_text,
                "dependency_covered": "我在家训练。" in dep_text,
                "final_tokens": dep_final.report["used_tokens"],
            }

            # 3. Source tamper after a real prepare must fail the source check.
            tampered = None
            with compact.session() as s, s.begin():
                s.get(WikiPage, large_page).revision += 1
            try:
                with compact.session() as s:
                    wc.validate_loaded(s, memory, {large_turn}, memory["memory_epoch"])
            except wc.WikiReadError as exc:
                tampered = str(exc)
            cases["source_tamper"] = {"rejected": tampered is not None,
                                      "code": tampered or ""}

            # 4. Model failure keeps a complete original fallback.
            fail_cid = "eval-failure"
            fail_turn = add_turn(fail_cid, "模型失败原文", "完整原文必须保留。")
            fail_memory = load(fail_cid, "继续")
            fail_page = page_id(fail_memory, fail_turn)
            prepare(fail_memory, {"page_ids": [fail_page]}, {}, fail=True)
            fail_unit = fail_memory["wiki_units"][0]
            cases["model_failure"] = {
                "origin": fail_unit["origin"],
                "completeness": fail_unit["semantic_completeness"],
                "full_text_retained": "完整原文必须保留。" in json.dumps(
                    fail_unit["messages"], ensure_ascii=False),
            }

            # 5. Extraction input cap is independent of the source-read budget.
            cap_cid = "eval-extraction-cap"
            cap_turn = add_turn(cap_cid, "抽取预算", "长原文" * 1500)
            cap_memory = load(cap_cid, "继续")
            cap_page = page_id(cap_memory, cap_turn)
            settings.wiki_semantic_unit_max_input_tokens = 500
            prepare(cap_memory, {"page_ids": [cap_page],
                                 "required_page_ids": [cap_page]}, {}, query="继续")
            cap_unit = cap_memory["wiki_units"][0]
            cap_insufficiency = ""
            try:
                assemble(cap_memory, window=400)
            except ContextBudgetExceeded as exc:
                cap_insufficiency = exc.reason
            cases["extraction_cap"] = {
                "source_read_tokens": cap_memory["wiki_read_tokens"],
                "origin": cap_unit["origin"],
                "completeness": cap_unit["semantic_completeness"],
                "failure_code": (cap_memory.get("wiki_unit_failures") or [{}])[0].get("code", ""),
                "final_insufficiency_reported": cap_insufficiency,
            }
            settings.wiki_semantic_unit_max_input_tokens = 16000

            # 6. Cold recovery: no extraction call, source/version checks, final units.
            recovery_cid = "eval-recovery"
            recovery_turn = add_turn(recovery_cid, "恢复任务", "周三只做两组徒手训练。")
            recovery_memory = load(recovery_cid, "继续")
            recovery_page = page_id(recovery_memory, recovery_turn)
            prepare(recovery_memory, {"page_ids": [recovery_page]},
                    {recovery_page: "周三只做两组徒手训练。"}, query="继续")
            recovery_understanding = {"dialogue_ids": [recovery_turn + ":assistant"]}
            frozen = wc.freeze_context(recovery_memory, recovery_understanding)
            fresh = load(recovery_cid, "继续")
            no_model = FakeLLM([])
            asyncio.run(wc.prepare_cached(
                "继续", fresh, {**recovery_understanding, "__wiki_context__": frozen},
                semantics.PROMPT))
            recovered, recovered_text = assemble(fresh, window=2000)
            cases["cold_recovery"] = {
                "frozen_has_no_bodies": '"quote":' not in json.dumps(frozen, ensure_ascii=False),
                "unit_covered": "周三只做两组徒手训练。" in recovered_text,
                "model_calls": no_model.calls,
            }

            # 7. Real dialogue final-call chain with the recovered/assembled units.
            dialogue = FakeLLM([], {})
            answer = asyncio.run(dialogue_context.conversation_answer(
                dialogue, ChatRequest(message="继续", conversation_id=recovery_cid),
                fresh, {"dialogue_ids": recovery_understanding["dialogue_ids"],
                        "reply_mode": "conversation", "tasks": [],
                        "reply_purpose": "general"}))
            cases["final_call_chain"] = {
                "answer": answer["answer"],
                "unit_in_final_prompt": "周三只做两组徒手训练。" in json.dumps(
                    dialogue.last_messages, ensure_ascii=False),
                "model_calls": dialogue.calls,
            }

            # 8. Structured conflict: both task-relevant sides are required;
            # an unrelated conflict is not leaked into the task context.
            conflict_cid = "eval-conflict"
            conflict_turn = add_turn(conflict_cid, "冲突来源", "记录。")
            with compact.session() as s, s.begin():
                for field, value, quote in (
                        ("medication_conflict", "布洛芬", "我吃布洛芬"),
                        ("medication_conflict", "不吃布洛芬", "我不吃布洛芬"),
                        ("unrelated_conflict", "旧信息", "旧无关信息"),
                        ("unrelated_conflict", "新信息", "新无关信息")):
                    s.add(UserFact(
                        fact_id=str(uuid4()), field=field, value=value,
                        source_turn_id=conflict_turn,
                        source_message_id=conflict_turn + ":user",
                        source_quote=quote, status="active",
                        valid_from=compact.now() - timedelta(seconds=1),
                        memory_epoch=0))
            conflict_memory = load(conflict_cid, "继续")
            token = semantics.request_understanding.set({
                "fixture": {"critical_fact_fields": ["medication_conflict"]}})
            try:
                conflict_packed = ContextBudgetManager(
                    window=3000, output=100, margin=0).assemble(
                        system="规则", query="用药冲突", state=conflict_memory, hits=[])
            finally:
                semantics.request_understanding.reset(token)
            conflict_text = json.dumps(conflict_packed, ensure_ascii=False)
            cases["conflict_and_false_positive"] = {
                "recall": "我吃布洛芬" in conflict_text
                and "我不吃布洛芬" in conflict_text,
                "unrelated_false_positive": "旧无关信息" in conflict_text
                or "新无关信息" in conflict_text,
                "final_tokens": conflict_packed.report["used_tokens"],
            }

    payload = {
        "evaluation": "offline end-to-end isolated SQLite fixture; scripted model",
        "cases": cases,
        "limits": [
            "Scripted extraction and answer model; no real provider answer quality.",
            "Synthetic pages and scripted units; structural/ablation checks only.",
            "The final chain uses the real dialogue budget path with a scripted answer model.",
            "Source-read capacity is bounded by WIKI_SOURCE_READ_MAX_TOKENS; extraction "
            "input by WIKI_SEMANTIC_UNIT_MAX_INPUT_TOKENS; final answer by model input budget.",
            "The temporal value case compares same-budget time-on/time-off and the ranking "
            "case compares density/value ordering; these are structural GT checks, not "
            "answer-quality or truth evidence.",
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if options.output:
        path = Path(options.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
