"""B9 修复回归：症状表达族、条件不产生伪计划任务、总量与阶段分别有来源。

所有断言针对离线纯函数与 mock 后的应用入口；不调用真实模型、向量或数据库。
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from evals.v2.safety import deny_network


@pytest.fixture(autouse=True)
def offline():
    with deny_network():
        yield


# --- AI-SIM-08：症状表达族 -------------------------------------------------


@pytest.mark.parametrize("source", [
    "胸口突然疼得很厉害，连呼吸都费劲",
    "现在呼吸都费劲",
    "呼吸费劲",
    "每次呼吸都很费力",
    "胸口突然疼",
    "胸部突然疼痛，吸不上气。",
])
def test_symptom_expression_variants_are_model_urgent(source):
    from app.services.agent.semantics import safety_for, validate_understanding
    raw = {"tasks": [_task("safety_triage", source)],
           "safety": {"level": "urgent", "evidence": source,
                      "categories": ["current_symptom"], "reason": "scripted current symptom"}}
    value = validate_understanding(source, raw)
    assert value["safety"]["level"] == "urgent"
    from app.services.agent import semantics
    token = semantics.request_understanding.set({})
    try:
        semantics.remember_understanding(source, value)
        assert safety_for(source)["level"] == "urgent"
    finally:
        semantics.request_understanding.reset(token)


@pytest.mark.parametrize("source", [
    "我没有胸口突然疼，也没有呼吸费劲。",
    "如何预防胸部突然疼痛？",
    "以前胸口突然疼，现在已经好了。",
    "跑步时正常喘气需要注意什么？",
    "普通喘气不影响训练。",
    "呼吸不费劲。",
    "我没有突然胸口疼。",
    "我没有突然呼吸费劲。",
])
def test_symptom_variants_keep_contextual_exemptions(source):
    from app.services.agent import semantics
    raw = {"tasks": [_task("training_qa", source)],
           "safety": {"level": "normal", "evidence": source,
                      "categories": ["negated", "prevention", "medical_history"],
                      "reason": "scripted negation/history/consultation context"}}
    value = semantics.validate_understanding(source, raw)
    token = semantics.request_understanding.set({})
    try:
        semantics.remember_understanding(source, value)
        assert semantics.safety_for(source)["level"] == "normal"
    finally:
        semantics.request_understanding.reset(token)


def test_general_science_context_is_not_urgent():
    from app.services.agent import semantics
    source = "科普一下：胸痛可能是什么原因？"
    raw = {"tasks": [_task("training_qa", source)],
           "safety": {"level": "medical", "evidence": "胸痛可能是什么原因",
                      "categories": ["consultation"], "reason": "general science question"}}
    value = semantics.validate_understanding(source, raw)
    token = semantics.request_understanding.set({})
    try:
        semantics.remember_understanding(source, value)
        assert semantics.safety_for(source)["level"] == "medical"
    finally:
        semantics.request_understanding.reset(token)


def test_negated_chest_does_not_mask_real_breathing_symptom():
    from app.services.agent import semantics
    source = "我没有胸口疼，但现在呼吸都费劲。"
    raw = {"tasks": [_task("safety_triage", source)],
           "safety": {"level": "urgent", "evidence": "现在呼吸都费劲",
                      "categories": ["current_symptom"], "reason": "current breathing symptom"}}
    value = semantics.validate_understanding(source, raw)
    token = semantics.request_understanding.set({})
    try:
        semantics.remember_understanding(source, value)
        assert semantics.safety_for(source)["level"] == "urgent"
    finally:
        semantics.request_understanding.reset(token)


@pytest.mark.parametrize("transport", ["json", "sse"])
@pytest.mark.parametrize("mode", ["legacy", "react_readonly", "react_write"])
def test_new_urgent_wording_uses_structured_safety_and_no_tool(compact_db, monkeypatch, transport, mode):
    import httpx
    from app.api import chat
    from app.main import app
    from app.services.agent.semantics import RequestUnderstanding
    from test_retry_operations import RetryScenario
    source = "胸口突然疼得很厉害，连呼吸都费劲"
    scenario = RetryScenario(monkeypatch, compact_db)
    if mode == "react_write":
        from app.core.config import settings
        monkeypatch.setattr(settings, "react_write_enabled", True)
        monkeypatch.setattr(settings, "memory_read_backend", "compact_v1")
    raw = RequestUnderstanding.model_validate({
        "tasks": [{"task_type": "safety_triage", "action": "question", "text": source,
                   "is_negated": False, "is_hypothetical": False, "is_third_party": False}],
        "safety": {"level": "urgent", "evidence": source,
                   "categories": ["current_symptom"], "reason": "scripted urgent"},
    })
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", lambda: True)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", AsyncMock(return_value=raw))
    monkeypatch.setattr(chat, "agent", scenario.agent)

    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as api:
            return await api.post("/api/chat/stream" if transport == "sse" else "/api/chat",
                                  json={"message": source, "mode": mode})
    wire = asyncio.run(send())
    assert wire.status_code == 200
    response = wire.json() if transport == "json" else json.loads([
        line[6:] for line in wire.text.splitlines() if line.startswith("data: ")][-1])
    assert response["safety_level"] == "urgent"
    assert not scenario.tool_calls


# --- AI-SIM-09-R5：条件不是独立任务 ---------------------------------------


def _task(task_type, text, *, action="question", mode=None, **extra):
    item = {"task_type": task_type, "action": action, "text": text,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False}
    if mode is not None:
        item["reply_mode"] = mode
    item.update(extra)
    return item


def test_conditions_attach_to_adjacent_review_and_single_plan_remains():
    from app.services.agent import semantics
    prefix, first, last = ("只建议，不保存。家里只有弹力带。", "先整理已有训练记录，",
                           "再给新手每周两次每次15分钟的增肌周计划。")
    source = prefix + first + last
    raw = {"tasks": [
        _task("training_qa", prefix + first, mode="conversation"),
        _task("training_plan", last),
    ]}
    checked = semantics.validate_understanding(source, raw)
    assert [t["task_type"] for t in checked["validated_tasks"]] == ["training_qa", "training_plan"]
    # 语义角色：回顾是 conversation 只读，计划是 grounded 独立结果。
    assert checked["validated_tasks"][0]["side_effect"] is False
    assert all(source[t["source_start"]:t["source_end"]] == t["text"] for t in checked["validated_tasks"])


def test_two_explicit_plan_requests_still_produce_two_plans():
    from app.services.agent import semantics
    source = "先给我一个增肌计划，再给我一个减脂计划。"
    raw = {"tasks": [
        _task("training_plan", "先给我一个增肌计划，", plan_result=True,
              result_scope="independent", plan_request_quote="给我一个增肌计划"),
        _task("training_plan", "再给我一个减脂计划。", plan_result=True,
              result_scope="independent", plan_request_quote="给我一个减脂计划"),
    ]}
    checked = semantics.validate_understanding(source, raw)
    assert [t["task_type"] for t in checked["validated_tasks"]] == ["training_plan", "training_plan"]


def test_two_legitimate_english_plans_are_kept():
    from app.services.agent import semantics
    parts = ["First build a strength routine; ", "then outline a mobility session."]
    raw = {"tasks": [
        _task("training_plan", parts[0], plan_result=True,
              result_scope="independent", plan_request_quote="build a strength routine"),
        _task("training_plan", parts[1], plan_result=True,
              result_scope="independent", plan_request_quote="outline a mobility session"),
    ]}
    checked = semantics.validate_understanding("".join(parts), raw)
    assert [t["task_type"] for t in checked["validated_tasks"]] == ["training_plan", "training_plan"]


def test_semantics_prompt_keeps_conditions_out_of_plan_tasks():
    from app.services.agent import semantics
    assert "本身不是一个训练计划结果" in semantics.PROMPT
    assert "器械、经验、时长、频率、只建议不保存等只是训练计划的条件" in semantics.PROMPT


# --- AI-SIM-09-R5：总量与阶段分别有来源 -----------------------------------


def test_quantity_parser_accepts_numeric_units_not_semantic_context():
    from app.services.agent.training_timing import duration_limit_seconds
    assert duration_limit_seconds("18分钟") == 1080
    assert duration_limit_seconds("3分钟") == 180
    assert duration_limit_seconds("2分钟") == 120
    assert duration_limit_seconds("0.5小时") == 1800
    # The parser is a numeric/unit converter, not a semantic reader.
    assert duration_limit_seconds("总共最多18分钟") is None
    assert duration_limit_seconds("热身固定3分钟") is None


def _plan_information(phase_limits=()):
    from app.services.agent.task_information import PlanInformation
    return PlanInformation(facts=[
        {"field": "goal", "source_id": "current", "quote": "增肌"},
        {"field": "experience", "source_id": "current", "quote": "训练新手"},
        {"field": "equipment", "source_id": "current", "quote": "哑铃"},
        {"field": "session_time", "source_id": "current", "quote": "总共最多18分钟",
         "quantity_quote": "18分钟", "seconds": 1080},
    ], phase_limits=phase_limits)


def _phase(phase, relation, context_quote, quantity_quote, *, seconds=None,
           source_id="current", qualifiers=()):
    from app.services.agent.training_timing import duration_limit_seconds
    return {"phase": phase, "relation": relation, "source_id": source_id,
            "context_quote": context_quote, "quantity_quote": quantity_quote,
            "seconds": seconds if seconds is not None else duration_limit_seconds(quantity_quote),
            "qualifiers": list(qualifiers)}


def test_plan_information_binds_model_phase_fields_and_seconds():
    from app.services.agent.task_information import check_plan_information
    source = "我是增肌训练新手，家里只有哑铃，总共最多18分钟，热身固定3分钟、收尾固定2分钟，均包含在总量。"
    info = _plan_information([
        _phase("warmup", "exact", "热身固定3分钟", "3分钟"),
        _phase("cooldown", "exact", "收尾固定2分钟", "2分钟"),
    ])
    llm = SimpleNamespace(complete_structured=AsyncMock(return_value=info))
    details, missing = asyncio.run(check_plan_information(llm, {"raw_message": source, "task_text": source}))
    assert missing is None
    assert details["known"]["session_time"] == "总共最多18分钟"
    assert details["session_time_seconds"] == 1080
    assert details["phase_limits"] == [
        {"phase": "warmup", "relation": "exact", "seconds": 180,
         "quote": "热身固定3分钟", "quantity_quote": "3分钟"},
        {"phase": "cooldown", "relation": "exact", "seconds": 120,
         "quote": "收尾固定2分钟", "quantity_quote": "2分钟"},
    ]


def test_phase_limits_accept_reordered_phases_and_hour_units():
    from app.services.agent.task_information import PlanInformation, check_plan_information
    source = "我是增肌训练新手，只有哑铃，总共1.5小时。收尾固定2分钟，热身固定3分钟。"
    info = PlanInformation(facts=[
        {"field": "goal", "source_id": "current", "quote": "增肌"},
        {"field": "experience", "source_id": "current", "quote": "训练新手"},
        {"field": "equipment", "source_id": "current", "quote": "哑铃"},
        {"field": "session_time", "source_id": "current", "quote": "总共1.5小时",
         "quantity_quote": "1.5小时", "seconds": 5400},
    ], phase_limits=[
        _phase("cooldown", "exact", "收尾固定2分钟", "2分钟"),
        _phase("warmup", "exact", "热身固定3分钟", "3分钟"),
    ])
    llm = SimpleNamespace(complete_structured=AsyncMock(return_value=info))
    details, clarification = asyncio.run(check_plan_information(llm, {"raw_message": source}))
    assert clarification is None
    assert details["known"]["session_time"] == "总共1.5小时"
    assert details["session_time_seconds"] == 5400
    assert details["phase_limits"] == [
        {"phase": "cooldown", "relation": "exact", "seconds": 120,
         "quote": "收尾固定2分钟", "quantity_quote": "2分钟"},
        {"phase": "warmup", "relation": "exact", "seconds": 180,
         "quote": "热身固定3分钟", "quantity_quote": "3分钟"},
    ]


@pytest.mark.parametrize("case", ["foreign_source", "quantity_outside", "seconds_mismatch",
                                  "short_context"])
def test_phase_contract_structural_failures_close(case):
    """Unverifiable source/number/context proof must never become a budget."""
    from app.services.agent.task_information import check_plan_information
    source = "我是增肌训练新手，只有哑铃，总共最多18分钟。热身固定3分钟。"
    limit = _phase("warmup", "exact", "热身固定3分钟", "3分钟")
    if case == "foreign_source":
        limit["source_id"] = "history:0"
    elif case == "quantity_outside":
        limit["quantity_quote"] = "5分钟"
    elif case == "seconds_mismatch":
        limit["seconds"] = 181
    elif case == "short_context":
        limit["context_quote"] = "3分钟"
    info = _plan_information([limit])
    llm = SimpleNamespace(complete_structured=AsyncMock(return_value=info))
    _, clarification = asyncio.run(check_plan_information(llm, {"raw_message": source}))
    assert clarification is not None
    assert clarification.category == "service_unavailable"


@pytest.mark.parametrize("qualifier,context,quantity,seconds", [
    ("negated", "热身不要求固定3分钟", "3分钟", 180),
    ("approximate", "热身大约3分钟", "3分钟", 180),
    ("range", "热身两三分钟", "两三分钟", 150),
])
def test_qualified_phase_is_soft_and_original_wording_still_reaches_generation(
        monkeypatch, qualifier, context, quantity, seconds):
    """A source-bound soft qualifier is not a hard budget or a plan failure.

    The original quantity/context stay in the current source and the plan
    worker still receives that source; only the machine-checked phase budget
    is withheld. Foreign sources, numbers and seconds mismatches remain hard
    failures in the test above.
    """
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.services.agent.task_information import check_plan_information
    source = f"我是增肌训练新手，只有哑铃，总共最多18分钟。{context}。"
    info = _plan_information([_phase("warmup", "exact", context, quantity,
                                     seconds=seconds, qualifiers=[qualifier])])
    details, clarification = asyncio.run(check_plan_information(
        SimpleNamespace(complete_structured=AsyncMock(return_value=info)),
        {"raw_message": source, "task_text": source}))
    assert clarification is None
    assert details["phase_limits"] == []
    assert details["known"]["session_time"] == "总共最多18分钟"
    assert details["session_time_seconds"] == 1080
    assert context in source and quantity in context

    from app.services.rag.retrieval import RetrievedChunk
    graph = FitnessMultiAgentGraph()
    captured = []
    monkeypatch.setattr(graph, "_readonly_retrieval", AsyncMock(return_value=[
        RetrievedChunk("训练", "循序渐进，规律训练。", .95, "official",
                       {"rerank_score": .95, "rerank_status": "model"}, "synthetic")]))
    async def forced(messages, **kwargs):
        captured.append(messages)
        return {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "return_timed_training_plan",
                                              "arguments": _raw()}}]}
    monkeypatch.setattr(graph.llm, "complete_message", forced)
    monkeypatch.setattr("app.services.agent.task_information.check_plan_information",
                        AsyncMock(return_value=(details, None)))
    state = {"user_id": "demo-user", "raw_message": source, "task_text": "安排训练",
             "task_type": "training_plan", "task_action": "question"}
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]
    assert result["status"] == "complete" and result["model_used"]
    rendered = json.dumps(captured, ensure_ascii=False)
    # The original soft wording is still part of what the generator sees; the
    # application only withheld the machine-checked hard phase budget above.
    assert context in rendered


def test_duplicate_identical_phase_constraints_dedupe():
    from app.services.agent.task_information import check_plan_information
    source = "我是增肌训练新手，只有哑铃，总共最多18分钟，热身固定3分钟。"
    limit = _phase("warmup", "exact", "热身固定3分钟", "3分钟")
    info = _plan_information([limit, limit])
    llm = SimpleNamespace(complete_structured=AsyncMock(return_value=info))
    details, clarification = asyncio.run(check_plan_information(llm, {"raw_message": source}))
    assert clarification is None and len(details["phase_limits"]) == 1


def test_conflicting_same_phase_constraints_fail_closed():
    from app.services.agent.task_information import check_plan_information
    source = "我是增肌训练新手，只有哑铃，总共最多18分钟。热身固定3分钟；热身至少3分钟。"
    info = _plan_information([
        _phase("warmup", "exact", "热身固定3分钟", "3分钟"),
        _phase("warmup", "at_least", "热身至少3分钟", "3分钟"),
    ])
    llm = SimpleNamespace(complete_structured=AsyncMock(return_value=info))
    _, clarification = asyncio.run(check_plan_information(llm, {"raw_message": source}))
    assert clarification is not None


def test_unbound_phase_limit_fails_closed_as_service_unavailable():
    from app.services.agent.task_information import check_plan_information
    source = "我是增肌训练新手，家里只有哑铃，总共最多18分钟。"
    info = _plan_information([_phase("warmup", "exact", "热身固定3分钟", "3分钟")])
    llm = SimpleNamespace(complete_structured=AsyncMock(return_value=info))
    _, clarification = asyncio.run(check_plan_information(llm, {"raw_message": source, "task_text": source}))
    assert clarification.category == "service_unavailable"


def test_minimum_session_seconds_reserves_positive_unconstrained_phases():
    from app.services.agent.training_timing import minimum_session_seconds
    assert minimum_session_seconds([
        {"phase": "warmup", "relation": "exact", "seconds": 540},
        {"phase": "cooldown", "relation": "exact", "seconds": 540},
    ]) == 1082  # 540 + 1(main) + 1(breaks) + 540
    assert minimum_session_seconds([
        {"phase": "warmup", "relation": "exact", "seconds": 180},
        {"phase": "cooldown", "relation": "exact", "seconds": 120},
    ]) == 302
    assert minimum_session_seconds([]) == 4
    assert minimum_session_seconds([
        {"phase": "warmup", "relation": "exact", "seconds": 180},
        {"phase": "warmup", "relation": "exact", "seconds": 180},
    ]) == 183  # 去重，不把同阶段加两遍
    assert minimum_session_seconds([
        {"phase": "warmup", "relation": "at_most", "seconds": 200},
    ]) == 4  # 上限只保留正数最低值


def _session(warmup=180, main=600, breaks=180, cooldown=120):
    return {
        "label": "训练1",
        "warmup": {"seconds": warmup, "activity": "关节活动与热身"},
        "main": {"seconds": main, "exercises": [{"name": "哑铃深蹲", "sets": 3, "reps": 8}]},
        "breaks": {"seconds": breaks},
        "cooldown": {"seconds": cooldown, "activity": "轻缓放松"},
        "time_consistency": "consistent",
    }


def _raw(warmup=180, main=600, breaks=180, cooldown=120):
    return json.dumps({"sessions": [_session(warmup, main, breaks, cooldown)]}, ensure_ascii=False)


PHASE_LIMITS = [
    {"phase": "warmup", "relation": "exact", "seconds": 180},
    {"phase": "cooldown", "relation": "exact", "seconds": 120},
]


def test_render_enforces_source_bound_phase_limits():
    from app.services.agent.training_timing import render_timed_plan
    answer = render_timed_plan(_raw(), limit_seconds=1080, expected_sessions=1, phase_limits=PHASE_LIMITS)
    assert "3分钟" in answer and "2分钟" in answer


def test_wrong_fixed_phase_is_rejected_even_when_total_is_within_limit():
    from app.services.agent.training_timing import render_timed_plan
    with pytest.raises(ValueError, match="phase_limit_violation"):
        render_timed_plan(_raw(warmup=200, main=580), limit_seconds=1080, expected_sessions=1,
                          phase_limits=PHASE_LIMITS)


def test_at_most_and_at_least_relations_are_enforced():
    from app.services.agent.training_timing import render_timed_plan
    limits = [{"phase": "warmup", "relation": "at_most", "seconds": 200},
              {"phase": "cooldown", "relation": "at_least", "seconds": 60}]
    assert render_timed_plan(_raw(warmup=180, main=600), limit_seconds=1080, expected_sessions=1,
                             phase_limits=limits)
    with pytest.raises(ValueError, match="phase_limit_violation"):
        render_timed_plan(_raw(warmup=220, main=560), limit_seconds=1080, expected_sessions=1,
                          phase_limits=limits)


def _wire_plan_worker(monkeypatch, graph, raw, information):
    monkeypatch.setattr("app.services.agent.task_information.check_plan_information",
                        AsyncMock(return_value=(information, None)))
    _wire_retrieval_and_model(monkeypatch, graph, raw)


def _wire_retrieval_and_model(monkeypatch, graph, raw):
    from app.services.rag.retrieval import RetrievedChunk
    hit = RetrievedChunk("训练", "循序渐进，规律训练。", .95, "official",
                         {"rerank_score": .95, "rerank_status": "model"}, "synthetic")
    monkeypatch.setattr(graph, "_readonly_retrieval", AsyncMock(return_value=[hit]))
    forced = AsyncMock(return_value={"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "return_timed_training_plan", "arguments": raw}}]})
    monkeypatch.setattr(graph.llm, "complete_message", forced)


def _phase_information():
    return {"scope": "session",
            "known": {"goal": "增肌", "experience": "训练新手", "equipment": "哑铃",
                      "session_time": "总共最多18分钟"},
            "session_time_seconds": 1080,
            "phase_limits": [
                {"phase": "warmup", "relation": "exact", "seconds": 180, "quote": "热身固定3分钟"},
                {"phase": "cooldown", "relation": "exact", "seconds": 120, "quote": "收尾固定2分钟"},
            ]}


def test_plan_worker_renders_honoring_phase_budgets(monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    graph = FitnessMultiAgentGraph()
    _wire_plan_worker(monkeypatch, graph, _raw(), _phase_information())
    state = {"user_id": "demo-user", "raw_message": "我是增肌训练新手，家里只有哑铃，总共最多18分钟，热身固定3分钟、收尾固定2分钟。",
             "task_text": "安排训练", "task_type": "training_plan", "task_action": "question"}
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]
    assert result["status"] == "complete" and result["model_used"]
    assert "| 收尾／放松 | 2分钟 |" in result["answer"]


def test_plan_worker_rejects_wrong_fixed_phase(monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    graph = FitnessMultiAgentGraph()
    _wire_plan_worker(monkeypatch, graph, _raw(warmup=200, main=580), _phase_information())
    state = {"user_id": "demo-user", "raw_message": "安排训练", "task_text": "安排训练",
             "task_type": "training_plan", "task_action": "question"}
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]
    assert result["status"] == "error" and result["model_used"]
    assert "时间分配未通过校验" in result["answer"]


def test_impossible_phase_total_combination_fails_closed_with_clarification(monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    graph = FitnessMultiAgentGraph()
    information = {"scope": "session",
                   "known": {"goal": "增肌", "experience": "训练新手", "equipment": "哑铃",
                             "session_time": "总共最多18分钟"},
                   "session_time_seconds": 1080,
                   "phase_limits": [
                       {"phase": "warmup", "relation": "exact", "seconds": 600, "quote": "热身固定10分钟"},
                       {"phase": "cooldown", "relation": "exact", "seconds": 600, "quote": "收尾固定10分钟"},
                   ]}
    monkeypatch.setattr("app.services.agent.task_information.check_plan_information",
                        AsyncMock(return_value=(information, None)))
    state = {"user_id": "demo-user", "raw_message": "安排训练", "task_text": "安排训练",
             "task_type": "training_plan", "task_action": "question"}
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]
    assert result["status"] == "needs_confirmation"
    assert "超过了总时长上限" in result["answer"]
    assert "session_time" not in result["answer"]


def test_equal_total_still_reserves_positive_unconstrained_phases(monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    graph = FitnessMultiAgentGraph()
    information = {"scope": "session",
                   "known": {"goal": "增肌", "experience": "训练新手", "equipment": "哑铃",
                             "session_time": "总共最多18分钟"},
                   "session_time_seconds": 1080,
                   "phase_limits": [
                       {"phase": "warmup", "relation": "exact", "seconds": 540},
                       {"phase": "cooldown", "relation": "exact", "seconds": 540},
                   ]}
    monkeypatch.setattr("app.services.agent.task_information.check_plan_information",
                        AsyncMock(return_value=(information, None)))
    retrieval = AsyncMock(side_effect=AssertionError("Impossible allocation must stop before retrieval"))
    monkeypatch.setattr(graph, "_readonly_retrieval", retrieval)
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke({"raw_message": "安排训练"}))["result"]
    assert result["status"] == "needs_confirmation"
    retrieval.assert_not_awaited()


def test_full_extraction_18_3_2_to_worker_result(monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    graph = FitnessMultiAgentGraph()
    source = "我是增肌训练新手，家里只有哑铃，总共最多18分钟，热身固定3分钟、收尾固定2分钟，均包含在总量。"
    info = _plan_information([
        _phase("warmup", "exact", "热身固定3分钟", "3分钟"),
        _phase("cooldown", "exact", "收尾固定2分钟", "2分钟"),
    ])
    monkeypatch.setattr(graph.llm, "complete_structured", AsyncMock(return_value=info))
    _wire_retrieval_and_model(monkeypatch, graph, _raw())
    state = {"user_id": "demo-user", "raw_message": source, "task_text": "安排训练",
             "task_type": "training_plan", "task_action": "question"}
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]
    assert result["status"] == "complete" and result["model_used"]
    assert "| 准备／热身 | 3分钟 |" in result["answer"]
    assert "| 收尾／放松 | 2分钟 |" in result["answer"]
    assert "不超过18分钟" in result["answer"]


def test_plan_conditions_are_extracted_from_full_current_source(monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.services.agent.task_information import PlanInformation
    graph = FitnessMultiAgentGraph()
    source = "只建议，不保存。家里只有弹力带。先整理已有训练记录，再给训练新手每周两次每次15分钟的增肌周计划。"
    task_text = "再给训练新手每周两次每次15分钟的增肌周计划。"
    info = PlanInformation(scope="weekly", scope_quote="每周两次", facts=[
        {"field": "goal", "source_id": "current", "quote": "增肌"},
        {"field": "experience", "source_id": "current", "quote": "训练新手"},
        {"field": "equipment", "source_id": "current", "quote": "弹力带"},
        {"field": "session_time", "source_id": "current", "quote": "每次15分钟",
         "quantity_quote": "15分钟", "seconds": 900},
        {"field": "available_days", "source_id": "current", "quote": "每周两次",
         "quantity_quote": "两次", "count": 2},
    ])
    monkeypatch.setattr(graph.llm, "complete_structured", AsyncMock(return_value=info))
    first = _session(warmup=120, main=600, breaks=120, cooldown=60)
    second = {**_session(warmup=120, main=600, breaks=120, cooldown=60), "label": "训练2"}
    _wire_retrieval_and_model(monkeypatch, graph,
                             json.dumps({"sessions": [first, second]}, ensure_ascii=False))
    state = {"user_id": "demo-user", "raw_message": source, "task_text": task_text,
             "task_type": "training_plan", "task_action": "question"}
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]
    assert result["status"] == "complete" and result["model_used"]
    assert result["answer"].count("| 准备／热身 | 2分钟 |") == 2
    assert "以上2次训练累计30分钟" in result["answer"]


def test_half_hour_beginner_runs_actual_worker(monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.services.agent.task_information import PlanInformation
    graph = FitnessMultiAgentGraph()
    source = "我是减脂训练新手，只能徒手，每次0.5小时。"
    info = PlanInformation(facts=[
        {"field": "goal", "source_id": "current", "quote": "减脂"},
        {"field": "experience", "source_id": "current", "quote": "训练新手"},
        {"field": "equipment", "source_id": "current", "quote": "徒手"},
        {"field": "session_time", "source_id": "current", "quote": "0.5小时",
         "quantity_quote": "0.5小时", "seconds": 1800},
    ])
    monkeypatch.setattr(graph.llm, "complete_structured", AsyncMock(return_value=info))
    # 单次 <= 1800 秒的四阶段方案
    _wire_retrieval_and_model(monkeypatch, graph, _raw(warmup=180, main=1200, breaks=180, cooldown=120))
    state = {"user_id": "demo-user", "raw_message": source, "task_text": "安排训练",
             "task_type": "training_plan", "task_action": "question"}
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]
    assert result["status"] == "complete" and result["model_used"]
    assert "不超过30分钟" in result["answer"]
