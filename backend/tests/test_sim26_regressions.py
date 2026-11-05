"""SIM26 regressions: ordinary phase allocation and read-only intent.

All model output is either the saved offline fixture from the 2026-10-25
simulation or a provider-boundary MockTransport response. SQL is an isolated
temporary database; no real provider, resident service or personal data is used.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from tests.react_phase_c2_3_support import (
    C3HTTPHarness,
    business_keys,
    ledger_snapshot,
    mock_provider_transport,
)


SIM26_PHASE_SOURCE = (
    "成年增肌新手，无伤病，器械只有弹力带。排今天一节18分钟训练，"
    "其中热身3分钟、放松2分钟，主体和组间转换也必须计入总计18分钟。仅建议，不保存。"
)
SIM26_NEGATED_SOURCE = "今天可以拿深蹲90公斤3组5次举例讲解，但千万不要记为我的实际训练。"


def _quantity_seconds(quantity):
    from app.services.agent.training_timing import duration_limit_seconds
    return duration_limit_seconds(quantity)


def _plan_information(phase_limits=(), *, source_time="18分钟", available_days=None):
    from app.services.agent.task_information import PlanInformation
    from app.services.agent.training_timing import duration_limit_seconds
    import re

    quantity = re.search(r"[0-9零〇一二两三四五六七八九十百半.]+(?:小时|钟头|分钟|分|秒|minutes?|mins?|min|hours?|hrs?|h|seconds?|secs?|s)", source_time)
    quantity_text = quantity.group(0) if quantity else source_time
    facts = [
        {"field": "goal", "source_id": "current", "quote": "成年增肌新手"},
        {"field": "experience", "source_id": "current", "quote": "成年增肌新手"},
        {"field": "equipment", "source_id": "current", "quote": "器械只有弹力带"},
        {"field": "session_time", "source_id": "current", "quote": source_time,
         "quantity_quote": quantity_text, "seconds": duration_limit_seconds(quantity_text)},
    ]
    if available_days is not None:
        facts.append({"field": "available_days", "source_id": "current",
                      "quote": available_days[0], "quantity_quote": available_days[1],
                      "count": available_days[2]})
    return PlanInformation(scope="session", facts=facts, phase_limits=list(phase_limits))


def _phase_limit(phase, relation, context_quote, quantity_quote=None, *,
                 seconds=None, source_id="current", qualifiers=()):
    quantity = quantity_quote or context_quote
    return {"phase": phase, "relation": relation, "source_id": source_id,
            "context_quote": context_quote, "quantity_quote": quantity,
            "seconds": seconds if seconds is not None else _quantity_seconds(quantity),
            "qualifiers": list(qualifiers)}


def _plan_llm(info):
    return SimpleNamespace(complete_structured=AsyncMock(return_value=info))


# ---------------------------------------------------------------------------
# SIM26-01: model-provided phase allocations are source/quantity bound
# ---------------------------------------------------------------------------


def test_sim26_saved_plan_information_accepts_ordinary_phase_allocations():
    from app.services.agent.task_information import check_plan_information

    info = _plan_information([
        _phase_limit("warmup", "exact", "其中热身3分钟", "3分钟"),
        _phase_limit("cooldown", "exact", "放松2分钟", "2分钟"),
    ])
    details, clarification = asyncio.run(check_plan_information(
        _plan_llm(info), {"raw_message": SIM26_PHASE_SOURCE, "task_text": SIM26_PHASE_SOURCE}))
    assert clarification is None
    assert details["known"]["session_time"] == "18分钟"
    assert details["session_time_seconds"] == 1080
    assert details["phase_limits"] == [
        {"phase": "warmup", "relation": "exact", "seconds": 180, "quote": "其中热身3分钟",
         "quantity_quote": "3分钟"},
        {"phase": "cooldown", "relation": "exact", "seconds": 120, "quote": "放松2分钟",
         "quantity_quote": "2分钟"},
    ]


@pytest.mark.parametrize("quantity,seconds", [
    ("3分钟", 180), ("三分钟", 180), ("3 分钟", 180), ("1分30秒", 90),
    ("90秒", 90), ("0.5小时", 1800), ("2 minutes", 120), ("90 seconds", 90),
    ("半小时", 1800),
])
def test_sim26_phase_quantity_units_are_converted_and_checked(quantity, seconds):
    from app.services.agent.task_information import check_plan_information

    context = "热身" + quantity
    source = f"成年增肌新手，无伤病，器械只有弹力带，总共30分钟。{context}。"
    info = _plan_information([_phase_limit("warmup", "exact", context, quantity)],
                             source_time="30分钟")
    details, clarification = asyncio.run(check_plan_information(
        _plan_llm(info), {"raw_message": source, "task_text": source}))
    assert clarification is None, quantity
    assert details["phase_limits"] == [
        {"phase": "warmup", "relation": "exact", "seconds": seconds,
         "quote": context, "quantity_quote": quantity}]


@pytest.mark.parametrize("relation,context,quantity,seconds", [
    ("at_most", "热身最多3分钟", "3分钟", 180),
    ("at_most", "warmup no more than 5 min", "5 min", 300),
    ("at_least", "热身至少3分钟", "3分钟", 180),
    ("at_least", "warmup at least 3 minutes", "3 minutes", 180),
])
def test_sim26_model_bound_relations_are_kept_with_canonical_seconds(relation, context, quantity, seconds):
    from app.services.agent.task_information import check_plan_information

    source = f"成年增肌新手，无伤病，器械只有弹力带，总共30分钟。{context}。"
    info = _plan_information([_phase_limit("warmup", relation, context, quantity)],
                             source_time="30分钟")
    details, clarification = asyncio.run(check_plan_information(
        _plan_llm(info), {"raw_message": source, "task_text": source}))
    assert clarification is None
    assert details["phase_limits"][0]["relation"] == relation
    assert details["phase_limits"][0]["seconds"] == seconds


@pytest.mark.parametrize("case", ["foreign_source", "quantity_outside", "seconds_mismatch",
                                  "short_context", "duplicate_conflict"])
def test_sim26_phase_limit_structural_failures_close(case):
    from app.services.agent.task_information import check_plan_information

    source = "成年增肌新手，无伤病，器械只有弹力带，总共30分钟。热身3分钟。"
    limit = _phase_limit("warmup", "exact", "热身3分钟", "3分钟")
    if case == "foreign_source":
        limit["source_id"] = "history:0"
    elif case == "quantity_outside":
        limit["quantity_quote"] = "5分钟"
    elif case == "seconds_mismatch":
        limit["seconds"] = 181
    elif case == "short_context":
        limit["context_quote"] = "3分钟"
    elif case == "duplicate_conflict":
        info = _plan_information([limit, {**limit, "seconds": 200}], source_time="30分钟")
        _, clarification = asyncio.run(check_plan_information(
            _plan_llm(info), {"raw_message": source, "task_text": source}))
        assert clarification is not None
        assert clarification.category == "service_unavailable"
        return
    info = _plan_information([limit], source_time="30分钟")
    _, clarification = asyncio.run(check_plan_information(
        _plan_llm(info), {"raw_message": source, "task_text": source}))
    assert clarification is not None
    assert clarification.category == "service_unavailable"


@pytest.mark.parametrize("qualifier", ["negated", "approximate", "range"])
def test_sim26_qualified_phase_is_soft_and_does_not_close_the_plan(qualifier):
    from app.services.agent.task_information import check_plan_information

    source = "成年增肌新手，无伤病，器械只有弹力带，总共30分钟。热身3分钟。"
    limit = _phase_limit("warmup", "exact", "热身3分钟", "3分钟")
    limit["qualifiers"] = [qualifier]
    info = _plan_information([limit], source_time="30分钟")
    details, clarification = asyncio.run(check_plan_information(
        _plan_llm(info), {"raw_message": source, "task_text": source}))
    # A qualified phase is not promoted to a hard budget and must not erase the
    # whole plan. The plan worker still sees the original wording.
    assert clarification is None
    assert details["phase_limits"] == []
    assert details["session_time_seconds"] == 1800


def test_sim26_phase_windows_use_complete_context_not_neighbour_phases():
    from app.services.agent.task_information import check_plan_information

    source = ("成年增肌新手，无伤病，器械只有弹力带，总共最多30分钟，"
              "热身3分钟、放松2分钟，主体10分钟；组间休息3分钟。")
    info = _plan_information([
        _phase_limit("warmup", "exact", "热身3分钟", "3分钟"),
        _phase_limit("cooldown", "exact", "放松2分钟", "2分钟"),
        _phase_limit("main", "exact", "主体10分钟", "10分钟"),
        _phase_limit("breaks", "exact", "组间休息3分钟", "3分钟"),
    ], source_time="30分钟")
    details, clarification = asyncio.run(check_plan_information(
        _plan_llm(info), {"raw_message": source, "task_text": source}))
    assert clarification is None
    assert [(item["phase"], item["relation"], item["seconds"])
            for item in details["phase_limits"]] == [
        ("warmup", "exact", 180), ("cooldown", "exact", 120),
        ("main", "exact", 600), ("breaks", "exact", 180)]


def test_sim26_conflicting_duplicate_phase_fails_closed():
    from app.services.agent.task_information import check_plan_information

    source = "成年增肌新手，无伤病，器械只有弹力带，总共30分钟。热身3分钟；热身4分钟。"
    info = _plan_information([
        _phase_limit("warmup", "exact", "热身3分钟", "3分钟"),
        _phase_limit("warmup", "exact", "热身4分钟", "4分钟"),
    ], source_time="30分钟")
    _, clarification = asyncio.run(check_plan_information(
        _plan_llm(info), {"raw_message": source, "task_text": source}))
    assert clarification is not None


def test_sim26_unbound_phase_limit_fails_closed_as_service_unavailable():
    from app.services.agent.task_information import check_plan_information

    source = "成年增肌新手，无伤病，器械只有弹力带，总共30分钟。"
    info = _plan_information([_phase_limit("warmup", "exact", "热身3分钟", "3分钟")],
                             source_time="30分钟")
    _, clarification = asyncio.run(check_plan_information(
        _plan_llm(info), {"raw_message": source, "task_text": source}))
    assert clarification is not None
    assert clarification.category == "service_unavailable"


def _timed_plan_json():
    return json.dumps({"sessions": [{
        "label": "训练1",
        "warmup": {"seconds": 180, "activity": "关节活动与轻缓热身"},
        "main": {"seconds": 600, "exercises": [
            {"name": "弹力带深蹲", "sets": 3, "reps": 12},
            {"name": "弹力带划船", "sets": 3, "reps": 12},
            {"name": "弹力带推举", "sets": 3, "reps": 10},
        ]},
        "breaks": {"seconds": 180},
        "cooldown": {"seconds": 120, "activity": "轻缓放松与拉伸"},
        "time_consistency": "consistent",
    }]}, ensure_ascii=False)


def _wire_plan_worker(monkeypatch, graph, info):
    from app.services.rag.retrieval import RetrievedChunk
    from app.services.agent.semantics import RequestUnderstanding

    hit = RetrievedChunk("训练安排", "循序渐进，规律训练。", .95, "official",
                         {"rerank_score": .95, "rerank_status": "model"}, "synthetic")
    monkeypatch.setattr(graph, "_readonly_retrieval", AsyncMock(return_value=[hit]))
    monkeypatch.setattr(graph.llm, "complete_message", AsyncMock(return_value={
        "role": "assistant", "content": "", "tool_calls": [{
            "type": "function",
            "function": {"name": "return_timed_training_plan", "arguments": _timed_plan_json()},
        }],
    }))
    async def complete_structured(messages, schema, *, name=None, **kwargs):
        if name == "return_request_understanding":
            payload = json.loads(messages[-1]["content"])
            source = str(payload.get("query") or "")
            value = RequestUnderstanding.model_validate({
                "tasks": [{"task_type": "chitchat", "action": "answer", "text": source,
                           "is_negated": False, "is_hypothetical": False,
                           "is_third_party": False, "history_requirement": "none"}],
                "safety": {"level": "normal", "evidence": "", "categories": [],
                           "reason": "scripted normal"},
            })
            return value
        return info
    monkeypatch.setattr(graph.llm, "configured", lambda: True)
    monkeypatch.setattr(graph.llm, "complete_structured", complete_structured)
    return info


# ---------------------------------------------------------------------------
# SIM26 original flow: real worker/renderer and real HTTP, zero business writes
# ---------------------------------------------------------------------------


def test_sim26_original_input_reaches_real_plan_worker_and_renderer(compact_db, monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.services.agent.training_timing import TimedTrainingPlan

    plan = TimedTrainingPlan.model_validate_json(_timed_plan_json()).sessions[0]
    assert plan.total_seconds == 1080
    assert (plan.warmup.seconds, plan.cooldown.seconds) == (180, 120)
    assert plan.main.seconds + plan.breaks.seconds == 780
    graph = FitnessMultiAgentGraph()
    info = _plan_information([
        _phase_limit("warmup", "exact", "其中热身3分钟", "3分钟"),
        _phase_limit("cooldown", "exact", "放松2分钟", "2分钟"),
    ])
    _wire_plan_worker(monkeypatch, graph, info)
    state = {"user_id": "demo-user", "raw_message": SIM26_PHASE_SOURCE,
             "task_text": "安排训练", "task_type": "training_plan", "task_action": "question"}
    with compact_db() as session:
        before = session.scalar(text("SELECT COUNT(*) FROM training_plans"))
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]
    with compact_db() as session:
        after = session.scalar(text("SELECT COUNT(*) FROM training_plans"))
    assert result["status"] == "complete" and result["model_used"] is True
    answer = result["answer"]
    assert "| 准备／热身 | 3分钟 |" in answer
    assert "| 主体动作 | 10分钟 |" in answer
    assert "| 组间休息与转换（穿插进行） | 3分钟 |" in answer
    assert "| 收尾／放松 | 2分钟 |" in answer
    assert "合计：18分钟，不超过18分钟。" in answer
    assert "不在上述总量外追加热身、休息或收尾" in answer
    assert before == after == 0


def test_sim26_http_plan_flow_renders_without_business_save(compact_db, monkeypatch):
    import httpx

    from app.api import chat
    from app.main import app
    from app.services.agent.agent import FitnessAgent
    from app.services.agent.execution_state import PlannerDecision
    from app.services.agent.intent import IntentResult

    agent = FitnessAgent()
    info = _plan_information([
        _phase_limit("warmup", "exact", "其中热身3分钟", "3分钟"),
        _phase_limit("cooldown", "exact", "放松2分钟", "2分钟"),
    ])
    _wire_plan_worker(monkeypatch, agent.graph, info)
    specs = [dict(id="plan", task_type="training_plan", worker="plan_adjust",
                  action="question", text=SIM26_PHASE_SOURCE, side_effect=False,
                  depends_on=[], payload={})]
    monkeypatch.setattr(agent, "_route", AsyncMock(return_value=PlannerDecision(
        IntentResult("plan_adjust", 1.0), {}, tuple(specs))))
    monkeypatch.setattr(chat, "agent", agent)

    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://fixture") as api:
            return await api.post("/api/chat", json={
                "message": SIM26_PHASE_SOURCE,
                "conversation_id": "sim26-plan-http",
                "request_id": "sim26-plan-http-1",
            })

    with compact_db() as session:
        before = session.scalar(text("SELECT COUNT(*) FROM training_plans"))
    wire = asyncio.run(send())
    with compact_db() as session:
        after = session.scalar(text("SELECT COUNT(*) FROM training_plans"))
    assert wire.status_code == 200
    payload = wire.json()
    assert payload["intent"] == "training_plan"
    assert "| 准备／热身 | 3分钟 |" in payload["answer"]
    assert "| 收尾／放松 | 2分钟 |" in payload["answer"]
    assert "合计：18分钟，不超过18分钟。" in payload["answer"]
    assert before == after == 0


# ---------------------------------------------------------------------------
# SIM26-02: no-commit consultation intent comes from completed application goals
# ---------------------------------------------------------------------------


def _runtime_without_io():
    from app.services.agent.react_write_runtime import WriteRuntimeSession

    runtime = WriteRuntimeSession.__new__(WriteRuntimeSession)
    runtime.ai_estimate_committed = False
    runtime.termination_reason = ""
    runtime.receipts = []
    runtime.committed_observations = []
    runtime.resume_goal_results = {}
    runtime.resume_covered = []
    runtime.frozen_parameters = {}
    runtime.response_ids = {}
    runtime._response_ids = lambda: {"request_id": "r", "message_id": "m", "run_id": "u"}
    runtime._citations = lambda state: []
    return runtime


def _read_state(task_type, *, answer="", status="completed"):
    return {
        "final_answer": answer,
        "intent_candidates": [{"id": "goal-1", "task_type": task_type,
                               "action": "question", "side_effect": False}],
        "goal_results": {"goal-1": {"status": status, "result": {}}},
        "covered_goals": ["goal-1"],
    }


@pytest.mark.parametrize("task_type", ["training_qa", "nutrition_advice", "training_plan"])
def test_sim26_no_commit_intent_uses_completed_application_goal(task_type):
    runtime = _runtime_without_io()
    result = runtime._committed_payload(_read_state(task_type, answer="咨询答复"))
    assert result["intent"] == task_type
    assert result["current_task"]["task_type"] == task_type
    assert result["receipt_refs"] == []
    assert result["mode_reason"] == "read_only_completed"
    assert result["final_answer"] == "咨询答复"


def test_sim26_no_commit_without_read_goal_never_claims_nutrition_record():
    runtime = _runtime_without_io()
    result = runtime._committed_payload({"final_answer": ""})
    assert result["intent"] != "nutrition_record"
    assert result["current_task"]["task_type"] != "nutrition_record"
    assert "已完成记录" not in result["final_answer"]
    assert result["receipt_refs"] == []


def test_sim26_direct_answer_uses_single_frozen_read_task():
    runtime = _runtime_without_io()
    runtime.frozen_parameters = {"task_outline": [
        {"id": "goal-1", "task_type": "nutrition_advice"}]}
    result = runtime._committed_payload({"final_answer": "建议..."})
    assert result["intent"] == "nutrition_advice"
    assert result["receipt_refs"] == []


@pytest.mark.parametrize("proof,expected", [
    ("workout_db_v1", "workout_record"),
    ("nutrition_db_v1", "nutrition_record"),
])
def test_sim26_known_commit_proof_keeps_trusted_intent(proof, expected):
    runtime = _runtime_without_io()
    runtime.receipts = ["real-receipt"]
    runtime.committed_observations = [{
        "status": "completed", "commit_state": "committed",
        "result": {"proof_type": proof},
    }]
    result = runtime._committed_payload({"final_answer": "已完成记录。"})
    assert result["intent"] == expected
    assert result["current_task"]["task_type"] == expected
    assert result["receipt_refs"] == ["real-receipt"]
    assert result["mode_reason"] == "write_executed"


def test_sim26_unknown_commit_proof_is_not_nutrition_success():
    runtime = _runtime_without_io()
    runtime.receipts = ["unknown-receipt"]
    runtime.committed_observations = [{
        "status": "completed", "commit_state": "committed",
        "result": {"proof_type": "forged_proof"},
    }]
    result = runtime._committed_payload({"final_answer": ""})
    assert result["intent"] != "nutrition_record"
    assert result["receipt_refs"] == []
    assert "已完成记录" not in result["final_answer"]


def test_sim26_limited_payload_is_not_a_write_success():
    runtime = _runtime_without_io()
    runtime.receipts = []
    runtime.termination_reason = "supervisor_turn_limit"
    result = runtime._limited_payload({})
    assert result["intent"] == "execution_limited"
    assert result["react_write_status"] == "limited"
    assert result["receipt_refs"] == []


# ---------------------------------------------------------------------------
# SIM26-02: real react_write JSON/SSE/replay over isolated SQL
# ---------------------------------------------------------------------------


@pytest.fixture
def sim26_env(tmp_path, monkeypatch):
    from tests.react_phase_c1_support import isolated
    from tests.react_phase_c2_2_support import ensure_catalog

    with isolated(monkeypatch, tmp_path / "sim26", "sqlite") as engine:
        ensure_catalog(engine)
        yield engine


def _provider_message(message):
    return {
        "choices": [{
            "message": message,
            "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
    }


def _understanding_message(source, task_type, *, action="question", fact_text=None,
                           is_negated=False, is_hypothetical=False):
    raw = {
        "tasks": [{
            "task_type": task_type, "action": action, "text": source,
            "is_negated": is_negated, "is_hypothetical": is_hypothetical,
            "is_third_party": False, "fact_text": fact_text,
            "reply_mode": "grounded", "reply_purpose": "general",
            "context_ids": [], "fact_fields": [], "history_requirement": "none",
        }],
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "scripted normal"},
        "reply_mode": "grounded", "reply_purpose": "general", "relation": "new",
        "current_is_critical": False, "critical_fact_fields": [],
        "critical_history_ids": [], "dialogue_ids": [], "references": [],
        "memory_requests": [], "needs_clarification": False,
    }
    return _provider_message({
        "role": "assistant", "content": "", "tool_calls": [{
            "id": "call-understanding", "type": "function",
            "function": {"name": "return_request_understanding",
                         "arguments": json.dumps(raw, ensure_ascii=False)},
        }],
    })


def _tool_message(name, arguments, *, call_id="call-tool"):
    return _provider_message({
        "role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(arguments, ensure_ascii=False)},
        }],
    })


def _answer_message(content):
    return _provider_message({"role": "assistant", "content": content})


def _hit(title, content):
    return SimpleNamespace(
        title=title, source="knowledge://sim26", chunk_id="chunk-" + title,
        content=content, score=1.0,
        metadata={"rerank_status": "model", "rerank_score": 1.0, "page": 1})


def _install_provider(monkeypatch, *, understanding, supervisor, answer,
                      extraction=None, plan_info=None, seen=None):
    import httpx

    def handler(request):
        body = json.loads(request.content.decode("utf-8"))
        tools = {item.get("function", {}).get("name")
                 for item in body.get("tools") or []}
        messages = body.get("messages") or []
        if seen is not None:
            seen.append({"tools": sorted(tools), "messages": messages})
        if "return_request_understanding" in tools:
            return httpx.Response(200, json=understanding)
        if "return_plan_information" in tools:
            assert plan_info is not None
            return httpx.Response(200, json=plan_info)
        if tools & {"record_workout", "record_nutrition", "answer_training",
                    "advise_nutrition", "plan_training"}:
            return httpx.Response(200, json=supervisor)
        if extraction is not None and not tools and any(
                "训练记录解析器" in str(message.get("content") or "")
                for message in messages):
            return httpx.Response(200, json=extraction)
        return httpx.Response(200, json=answer)

    return mock_provider_transport(monkeypatch, [handler])


def _install_rag(monkeypatch):
    import app.services.rag.online as rag_online

    async def search(pipeline, query, **kwargs):
        return [_hit("训练强度指南", "训练强度应循序渐进，注意动作质量和恢复。"),
                _hit("训练恢复", "组间休息与动作质量需要兼顾。")]

    monkeypatch.setattr(rag_online, "search", search, raising=False)


def test_sim26_native_negated_training_is_read_only_json_sse_and_replay(sim26_env, monkeypatch):
    source = SIM26_NEGATED_SOURCE
    # Saved answer from the 2026-10-25 native_negated live turn (call 109).
    answer_text = (
        "以“深蹲90公斤3组5次”为例，这属于偏力量型的安排：5次一组的次数偏低，"
        "重量相对较大，通常对应每组接近力竭、但还能保持动作质量的水平。它是否符合你，"
        "不能只看90公斤这个数字，而要看最后一组第5次是否还能在姿势不变的前提下完成；"
        "如果第3次就很吃力或动作变形，说明当前重量偏高。\n\n"
        "判断是否合适，可以用“还能不能再做一次”作标准：每组做完5次后，"
        "感觉再完成1次会很困难但并非完全做不到，通常比较合适；若5次轻松完成、"
        "还能多做3次以上，重量偏轻；若做不到5次或必须借力、憋气硬顶，则偏重。"
        "组间休息建议2至3分钟，大重量深蹲需要更充分恢复，休息不足会让后两组动作质量明显下降。\n\n"
        "动作要点是脚掌稳定踩地、膝盖与脚尖方向一致、下蹲时保持躯干稳定、核心收紧，"
        "起身时不要塌腰或膝盖内扣。训练前先做几组轻重量热身，逐步加到工作重量，"
        "不要直接上90公斤。\n\n"
        "调整时按完成情况处理：3组都能规范完成5次且有余力，下次可小幅加重，"
        "比如增加2.5至5公斤，或保持重量增加到3组6至8次；只有前两组达标、"
        "第三组明显变形，先维持重量并把休息延长到3分钟；连第一组都难以规范完成5次，"
        "就降低重量，优先保证动作质量。以上为示例性思路，具体重量需结合个人情况确定。"
        "若出现明显肿胀、剧烈疼痛、麻木、关节变形或活动受限，应停止训练并寻求专业帮助。"
    )
    _install_rag(monkeypatch)
    seen_urls = _install_provider(
        monkeypatch,
        understanding=_understanding_message(source, "training_qa",
                                            is_negated=True, is_hypothetical=True),
        supervisor=_tool_message("answer_training", {"query": source}),
        answer=_answer_message(answer_text),
    )
    harness = C3HTTPHarness(monkeypatch, sim26_env)
    before = business_keys(sim26_env)
    rid = "sim26-native-negated"
    status, payload = harness.post_json(source, request_id=rid)
    assert status == 200
    assert payload["intent"] == "training_qa"
    assert payload["answer"] == answer_text
    assert payload["metrics"]["receipt_refs"] == []
    assert payload["metrics"]["react_write"] == "completed"
    assert payload["request_id"] == rid
    assert payload["message_id"] and payload["run_id"]
    assert business_keys(sim26_env) == before
    snapshot = ledger_snapshot(sim26_env, rid)
    assert snapshot["request"]["status"] == "completed"
    assert snapshot["operations"] == []
    calls_after_first = len(seen_urls)
    assert calls_after_first >= 2

    replay_status, replay = harness.post_json(source, request_id=rid)
    assert replay_status == 200 and replay["replayed"] is True
    assert replay["intent"] == payload["intent"]
    assert replay["answer"] == payload["answer"]
    assert replay["request_id"] == payload["request_id"]
    assert replay["message_id"] == payload["message_id"]
    assert replay["run_id"] == payload["run_id"]
    assert replay["metrics"]["receipt_refs"] == []
    assert len(seen_urls) == calls_after_first
    assert business_keys(sim26_env) == before

    sse_status, events, _ = harness.stream(source, request_id=rid)
    assert sse_status == 200 and events
    sse_final = events[-1]
    sse_answer = "".join(str(frame.get("content") or "") for frame in events)
    assert sse_final["intent"] == payload["intent"]
    assert sse_final["request_id"] == payload["request_id"]
    assert sse_final["message_id"] == payload["message_id"]
    assert sse_final["run_id"] == payload["run_id"]
    assert sse_answer == payload["answer"]
    assert sse_final["metrics"]["receipt_refs"] == []
    assert len(seen_urls) == calls_after_first
    assert business_keys(sim26_env) == before


def test_sim26_native_nutrition_advice_is_read_only(sim26_env, monkeypatch):
    source = "根据我今天的饮食给点建议，只讲通用原则，不要保存任何记录。"
    answer_text = "建议优先保证蛋白质和蔬菜摄入，主食按活动量调整；本答复不保存记录。"
    _install_rag(monkeypatch)
    seen_urls = _install_provider(
        monkeypatch,
        understanding=_understanding_message(source, "nutrition_advice"),
        supervisor=_tool_message("advise_nutrition", {"query": source}),
        answer=_answer_message(answer_text),
    )
    harness = C3HTTPHarness(monkeypatch, sim26_env)
    before = business_keys(sim26_env)
    status, payload = harness.post_json(source, request_id="sim26-nutrition-advice")
    assert status == 200
    assert payload["intent"] == "nutrition_advice"
    assert payload["answer"] == answer_text
    assert payload["metrics"]["receipt_refs"] == []
    assert business_keys(sim26_env) == before
    assert len(seen_urls) >= 2


def test_sim26_native_workout_commit_keeps_workout_intent_and_replays(sim26_env, monkeypatch):
    source = "请记录今天哑铃划船24公斤，3组，每组12次。"
    fact = "今天哑铃划船24公斤，3组，每组12次"
    extraction = _provider_message({
        "role": "assistant",
        "content": json.dumps({"records": [{
            "exercise": "哑铃划船", "weight_kg": 24, "sets": 3, "reps": 12,
            "inferred_fields": [], "record_date": None,
        }]}, ensure_ascii=False),
    })
    seen_urls = _install_provider(
        monkeypatch,
        understanding=_understanding_message(source, "workout_record", action="record",
                                            fact_text=fact),
        supervisor=_tool_message("record_workout", {
            "source_ref": {"start": source.index(fact), "end": source.index(fact) + len(fact),
                           "exact_text": fact}}),
        answer=_answer_message("unused"),
        extraction=extraction,
    )
    harness = C3HTTPHarness(monkeypatch, sim26_env)
    before = business_keys(sim26_env)
    rid = "sim26-native-workout"
    status, payload = harness.post_json(source, request_id=rid)
    assert status == 200
    assert payload["intent"] == "workout_record"
    assert payload["metrics"]["receipt_refs"]
    after = business_keys(sim26_env)
    assert len(after["workout_records"]) == len(before["workout_records"]) + 1
    snapshot = ledger_snapshot(sim26_env, rid)
    assert any(operation["kind"] == "workout" for operation in snapshot["operations"])
    receipts = list(payload["metrics"]["receipt_refs"])
    ids = (payload["request_id"], payload["message_id"], payload["run_id"])
    calls_after_first = len(seen_urls)

    replay_status, replay = harness.post_json(source, request_id=rid)
    assert replay_status == 200 and replay["replayed"] is True
    assert replay["intent"] == payload["intent"]
    assert replay["answer"] == payload["answer"]
    assert replay["metrics"]["receipt_refs"] == receipts
    assert (replay["request_id"], replay["message_id"], replay["run_id"]) == ids
    assert len(seen_urls) == calls_after_first
    assert business_keys(sim26_env) == after


def test_sim26_native_nutrition_commit_keeps_nutrition_intent(sim26_env, monkeypatch):
    fact = ("今天午餐：牛肉面一碗，热量520千卡，蛋白质25克，"
            "碳水60克，脂肪15克")
    source = "帮我记录" + fact + "。"
    seen_urls = _install_provider(
        monkeypatch,
        understanding=_understanding_message(source, "nutrition_record", action="record",
                                            fact_text=fact),
        supervisor=_tool_message("record_nutrition", {
            "source_ref": {"start": source.index(fact), "end": source.index(fact) + len(fact),
                           "exact_text": fact}}),
        answer=_answer_message("unused"),
    )
    harness = C3HTTPHarness(monkeypatch, sim26_env)
    before = business_keys(sim26_env)
    status, payload = harness.post_json(source, request_id="sim26-native-nutrition")
    assert status == 200
    assert payload["intent"] == "nutrition_record"
    assert payload["metrics"]["receipt_refs"]
    after = business_keys(sim26_env)
    assert len(after["nutrition_logs"]) == len(before["nutrition_logs"]) + 1
    assert len(after["user_food_log"]) == len(before["user_food_log"])
    receipts = list(payload["metrics"]["receipt_refs"])
    ids = (payload["request_id"], payload["message_id"], payload["run_id"])
    calls_after_first = len(seen_urls)
    replay_status, replay = harness.post_json(source, request_id="sim26-native-nutrition")
    assert replay_status == 200 and replay["replayed"] is True
    assert replay["intent"] == payload["intent"]
    assert replay["answer"] == payload["answer"]
    assert replay["metrics"]["receipt_refs"] == receipts
    assert (replay["request_id"], replay["message_id"], replay["run_id"]) == ids
    assert len(seen_urls) == calls_after_first
    assert business_keys(sim26_env) == after
