"""Purpose, domain and side effect are independent; providers and SQL isolated."""
import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select, text

from app.db.memory_models import UserFact
from app.schemas.chat import ChatRequest
from app.services.agent import dialogue_context as dialogue, semantics
from app.services.memory import compact
from evals.v2.safety import deny_network
from test_dialogue_continuity import task
from test_dialogue_contract import provider, wire
from test_retry_operations import RetryScenario


@pytest.fixture(autouse=True)
def isolate():
    token = semantics.request_understanding.set({})
    with deny_network():
        yield
    semantics.request_understanding.reset(token)


def safety(level="normal", evidence="", categories=(), reason="scripted safety"):
    """Provider-bound structured safety result; not an application default."""
    return {"level": level, "evidence": evidence,
            "categories": list(categories), "reason": reason}


def intent(source, purpose, *, child_purpose=None, kind="safety_triage", fields=(),
           decisions=(), assessment=None):
    return {"tasks": [{**task(source, kind, "answer"), "reply_mode": "conversation",
                       "reply_purpose": child_purpose or purpose, "fact_fields": list(fields)}],
            "reply_mode": "conversation", "reply_purpose": purpose,
            "memory_requests": list(decisions),
            "safety": assessment if assessment is not None else safety()}


def decision(quote, *, key="personal_fact", action="remember", targets=()):
    return {"action": action, "quote": quote, "key": key, "confidence": .99, "target_fields": list(targets)}


def test_parent_and_child_share_the_same_purpose_schema():
    parent = semantics.RequestUnderstanding.model_json_schema()["properties"]["reply_purpose"]["enum"]
    child = semantics.SemanticTask.model_json_schema()["properties"]["reply_purpose"]["enum"]
    assert set(child) == set(parent) == {"general", "execution_status", "fact_recall", "memory_update"}


@pytest.mark.parametrize("quote", ["我对核桃过敏", "我的膝盖曾受伤", "我正在按医嘱服药", "我家只有弹力带", "我偏好晨练"])
@pytest.mark.parametrize("legacy_child", [False, True])
def test_personal_memory_domain_cannot_force_medical_assessment(quote, legacy_child):
    source = "请长期记住：" + quote
    raw = intent(source, "memory_update", child_purpose="general" if legacy_child else None,
                 decisions=[decision(quote)])
    value = semantics.validate_understanding(source, raw)
    assert value["reply_purpose"] == value["tasks"][0]["reply_purpose"] == "memory_update"
    assert value["validated_tasks"][0]["task_type"] != "safety_triage"
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock())
    result = asyncio.run(dialogue.conversation_answer(llm, ChatRequest(message=source), {}, value))
    assert result["memory_update_requested"] and "已记住" not in result["answer"]
    llm.complete_structured.assert_not_awaited()
    assert raw["tasks"][0]["task_type"] == "safety_triage"


@pytest.mark.parametrize("quote", ["我对核桃过敏", "我的膝盖曾受伤", "我正在按医嘱服药", "我家只有弹力带", "我偏好晨练"])
def test_historical_recall_only_renders_actual_source_for_any_domain(quote):
    source = "只复述我以前提供的信息，不做判断"
    memory = {"long_memory": {"own": {"text": quote, "source_message_id": "u"}}}
    value = semantics.validate_understanding(source, intent(source, "fact_recall", fields=["own"]), context=memory)
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(return_value={
        "evidence": [{"source_id": "fact:own", "quote": quote}]}))
    result = asyncio.run(dialogue.conversation_answer(llm, ChatRequest(message=source), memory, value))
    assert result["answer"] == "你提供过的信息是：\n“" + quote + "”"
    assert llm.complete_structured.await_args.args[1] is dialogue.RecallSelection


def test_child_only_recall_is_not_sent_to_free_form_dialogue():
    source = "复述旧记录"
    raw = intent(source, "general", child_purpose="fact_recall")
    value = semantics.validate_understanding(source, raw)
    assert value["reply_purpose"] == "fact_recall"
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock())
    result = asyncio.run(dialogue.conversation_answer(llm, ChatRequest(message=source), {}, value))
    assert "没有找到" in result["answer"]
    llm.complete_structured.assert_not_awaited()


@pytest.mark.parametrize("parent,child", [
    ("execution_status", "fact_recall"), ("execution_status", "memory_update"),
    ("fact_recall", "execution_status"), ("fact_recall", "memory_update"),
    ("memory_update", "execution_status"), ("memory_update", "fact_recall"),
])
def test_all_different_explicit_parent_and_child_purposes_are_rejected(parent, child):
    source = "请记住我喜欢晨练"
    raw = intent(source, parent, child_purpose=child, kind="chitchat", decisions=[decision("我喜欢晨练")])
    with pytest.raises(semantics.SemanticValidationError) as exc:
        semantics.validate_understanding(source, raw)
    assert exc.value.validation_errors[0]["type"] == "semantic_purpose_conflict"


@pytest.mark.parametrize("repair", [False, True])
def test_explicit_purpose_conflict_uses_original_two_attempt_allowance(monkeypatch, repair):
    quote, source = "我对核桃过敏", "复述我以前提供的过敏记录"
    bad = intent(source, "execution_status", child_purpose="fact_recall", fields=["own"])
    good = intent(source, "fact_recall", fields=["own"])
    memory = {"long_memory": {"own": {"text": quote, "source_message_id": "u"}}}
    def respond(body, number):
        name = body["tool_choice"]["function"]["name"]
        if name == "return_request_understanding":
            return wire(good if repair and number == 2 else bad)
        assert repair and name == "select_personal_record_sources"
        return wire({"evidence": [{"source_id": "fact:own", "quote": quote}]}, name=name)
    client, seen = provider(monkeypatch, respond)
    value = asyncio.run(semantics.understand(client, source, context=memory))
    assert len(seen) == 2 and "semantic_purpose_conflict" in seen[1]["messages"][0]["content"]
    if repair:
        result = asyncio.run(dialogue.conversation_answer(client, ChatRequest(message=source), memory, value))
        assert result["answer"] == "你提供过的信息是：\n“" + quote + "”" and len(seen) == 3
    else:
        assert value is None and semantics.get_understanding(source) is None
    assert all(b["tool_choice"]["function"]["name"] != "return_dialogue_answer" for b in seen)


@pytest.mark.parametrize("kind,action,purpose", [
    ("safety_triage", "question", "general"), ("safety_triage", "question", "execution_status"),
    ("workout_record", "record", "fact_recall"), ("nutrition_record", "confirm", "memory_update"),
])
def test_purpose_does_not_grant_a_write_or_free_medical_answer(kind, action, purpose):
    source = "我的膝盖现在疼，请判断能否运动"
    raw = intent(source, purpose, kind=kind)
    raw["tasks"][0]["action"] = action
    with pytest.raises(semantics.SemanticValidationError):
        semantics.validate_understanding(source, raw)


@pytest.mark.parametrize("mutation", ["foreign_fact", "invented_quote", "missing_decision", "different_task_quote", "conflicting_purpose"])
def test_memory_purpose_preserves_source_and_decision_checks(mutation):
    source = "请记住我偏好晨练；这是另一个问题"
    raw = intent(source, "memory_update", decisions=[decision("我偏好晨练")])
    if mutation == "foreign_fact": raw["tasks"][0]["fact_fields"] = ["foreign"]
    elif mutation == "invented_quote": raw["memory_requests"][0]["quote"] = "我喜欢夜跑"
    elif mutation == "missing_decision": raw["memory_requests"] = []
    elif mutation == "different_task_quote":
        raw["reply_purpose"] = "general"
        raw["reply_mode"] = "grounded"
        raw["tasks"][0]["text"] = "这是另一个问题"
        raw["tasks"].append(task("请记住我偏好晨练"))
    else: raw["tasks"][0]["reply_purpose"] = "fact_recall"
    with pytest.raises(semantics.SemanticValidationError):
        semantics.validate_understanding(source, raw)


def test_mixed_recall_and_actual_assessment_keep_separate_routes():
    from app.services.agent.task_planner import plan_tasks_from_llm
    read, assess = "复述既有伤病记录", "评估我今天膝痛是否可以训练"
    source = read + "；" + assess
    raw = intent(read, "fact_recall")
    raw.update(reply_mode="grounded", reply_purpose="general")
    raw["tasks"].append(task(assess, "safety_triage"))
    value = semantics.validate_understanding(source, raw)
    semantics.remember_understanding(source, value)
    planned = plan_tasks_from_llm(source, {"tasks": value["validated_tasks"]})
    assert value["reply_mode"] == "grounded" and len(planned) == 2
    assert planned[0]["payload"]["dialogue"]["reply_purpose"] == "fact_recall"
    assert planned[1]["task_type"] == "safety_triage" and "dialogue" not in planned[1]["payload"]
    bad = copy.deepcopy(raw)
    bad["reply_purpose"] = "fact_recall"
    with pytest.raises(semantics.SemanticValidationError):
        semantics.validate_understanding(source, bad)


@pytest.mark.parametrize("child_purpose", ["general", "fact_recall"])
def test_explicit_grounded_medical_assessment_cannot_be_hidden_by_top_recall(child_purpose):
    source = "我现在膝盖疼，请评估能否继续训练"
    raw = intent(source, "fact_recall", child_purpose=child_purpose)
    raw["tasks"][0]["reply_mode"] = "grounded"
    with pytest.raises(semantics.SemanticValidationError):
        semantics.validate_understanding(source, raw)


@pytest.mark.parametrize("quote", ["我对核桃过敏", "我家只有弹力带", "我偏好晨练"])
def test_real_agent_memory_first_save_duplicate_replace_and_recall(compact_db, monkeypatch, quote):
    from app.core.config import settings
    from app.services.agent import execution
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    scenario = RetryScenario(monkeypatch, compact_db)
    monkeypatch.setattr(execution.memory_service, "load", AsyncMock(side_effect=lambda *a: {
        "long_memory": compact.current_facts(), "short_memory": [], "memory_backend": "fixture"}))
    requests = []
    def response(body, number):
        name = body["tool_choice"]["function"]["name"]
        if name == "return_request_understanding":
            return wire(requests[-1])
        assert name == "select_personal_record_sources"
        field, fact = next(iter(compact.current_facts().items()))
        return wire({"evidence": [{"source_id": "fact:" + field, "quote": fact["source_quote"]}]}, name=name)
    client, seen = provider(monkeypatch, response)
    monkeypatch.setattr(scenario.agent.graph, "llm", client)
    monkeypatch.setattr(scenario.agent.graph.input_parser, "llm", client)
    monkeypatch.setattr(scenario.agent.graph, "parse_input", FitnessMultiAgentGraph.parse_input.__get__(scenario.agent.graph))
    monkeypatch.setattr(settings, "input_parser_enabled", True)
    async def send(source, raw):
        requests.append(raw)
        request = ChatRequest(message=source)
        first = await scenario.respond(request)
        count = len(seen)
        replay = await scenario.respond(request)
        assert replay.replayed and replay.answer == first.answer and len(seen) == count
        return first
    source = "请长期记住：" + quote
    first = asyncio.run(send(source, intent(source, "memory_update", decisions=[decision(quote)])))
    assert first.metrics["memory_update_status"] == "committed" and quote in first.answer
    duplicate = asyncio.run(send(source, intent(source, "memory_update", child_purpose="general", decisions=[decision(quote, key="another_key")])))
    assert duplicate.metrics["memory_update_status"] == "committed"
    with compact_db() as s:
        assert len(list(s.scalars(select(UserFact)))) == 1
    latest = "我更喜欢午后锻炼"
    source = "请替换之前的记忆：" + latest
    targets = list(compact.current_facts())
    replaced = asyncio.run(send(source, intent(source, "memory_update", decisions=[decision(latest, action="replace", targets=targets)])))
    assert latest in replaced.answer and replaced.metrics["memory_update_status"] == "committed"
    source = "按最新信息复述记忆"
    recalled = asyncio.run(send(source, intent(source, "fact_recall", fields=list(compact.current_facts()))))
    assert latest in recalled.answer and quote not in recalled.answer
    assert len(seen) == 5  # one semantic call per turn, plus source selection for recall
    assert not scenario.saves
    with compact_db() as s:
        rows = list(s.scalars(select(UserFact)))
        assert len(rows) == 2 and sorted(r.status for r in rows) == ["active", "superseded"]
        for table in ("workout_records", "user_food_log", "nutrition_logs"):
            assert s.execute(text("SELECT count(*) FROM " + table)).scalar() == 0


@pytest.mark.parametrize("source,evidence,remember", [
    ("我现在胸口疼，还喘不过气，请记住", "我现在胸口疼，还喘不过气", True),
    ("突然讲话含糊，回顾我的旧记录", "突然讲话含糊", False),
])
def test_model_urgent_safety_preempts_memory_recall_and_workers(
        compact_db, monkeypatch, source, evidence, remember):
    """One required semantic judgment, then urgent stops every business path.

    The wording alone has no authority: the provider mock returns the explicit
    urgent assessment with a current-source evidence span. Even a memory
    request (first case) or a recall request (second case) must not reach its
    handler, model worker, or save path.
    """
    scenario = RetryScenario(monkeypatch, compact_db)
    raw = intent(
        source, "general", kind="chitchat",
        decisions=[decision(evidence, key="current-symptom")] if remember else [],
        assessment=safety("urgent", evidence, ["current_symptom"],
                          "scripted current symptom"))
    client, seen = provider(monkeypatch, lambda body, number: wire(raw))
    monkeypatch.setattr(scenario.agent.graph, "llm", client)
    result = asyncio.run(scenario.respond(ChatRequest(message=source)))

    # The request did buy one metered semantic model call; urgent stops every
    # business path, but model_used reports the actual model attempt.
    assert result.safety_level == "urgent" and result.model_used
    assert result.metrics["safety_status"] == "assessed"
    assert len(seen) == 1
    assert seen[0]["tool_choice"]["function"]["name"] == "return_request_understanding"
    scenario.agent.graph.parse_input.assert_not_awaited()
    assert scenario.tool_calls == [] and scenario.saves == []
    with compact_db() as s:
        for table in ("user_facts", "workout_records", "user_food_log", "nutrition_logs"):
            assert s.execute(text("SELECT count(*) FROM " + table)).scalar() == 0


@pytest.mark.parametrize("source,assessment,expected", [
    # The exact wording that the provider can call urgent still stays normal
    # when the model binds a normal judgment to this source.
    ("我现在胸口疼，还喘不过气，请记住",
     safety("normal", "", ["negated"], "scripted historical/negated context"), "normal"),
    # A negated first clause does not exempt a new current symptom when the
    # model binds urgent evidence to the new clause.
    ("我没有胸口疼，但现在突然说话含糊",
     safety("urgent", "现在突然说话含糊", ["current_symptom"],
            "scripted current symptom after a negated clause"), "urgent"),
])
def test_safety_follows_model_evidence_not_word_cutoff(monkeypatch, source, assessment, expected):
    raw = intent(source, "general", kind="chitchat", assessment=assessment)
    client, seen = provider(monkeypatch, lambda body, number: wire(raw))
    value = asyncio.run(semantics.understand(client, source))
    assert value is not None and value["safety"]["level"] == expected
    assert semantics.safety_for(source)["level"] == expected
    assert len(seen) == 1


@pytest.mark.parametrize("assessment", [
    None,
    safety("urgent", "历史记录里的胸痛", ["current_symptom"], "foreign evidence"),
    safety("medical", "", ["medical_history"], "non-normal without evidence"),
    safety("normal", "原文没有的片段", [], "normal with foreign evidence"),
], ids=["missing", "foreign", "invalid-non-normal", "invalid-normal"])
def test_missing_invalid_or_foreign_safety_is_unavailable_not_normal(monkeypatch, assessment):
    source = "我偏好晨练"
    raw = intent(source, "general", kind="chitchat", assessment=assessment)
    if assessment is None:
        raw.pop("safety")
    client, seen = provider(monkeypatch, lambda body, number: wire(raw))
    value = asyncio.run(semantics.understand(client, source))

    decision = semantics.safety_for(source)
    assert decision["status"] == "unavailable" and decision["level"] == "unknown"
    if assessment is None:
        # Missing safety is a schema-valid but unusable judgment. The shared
        # structured path allows exactly one bounded correction attempt; after
        # it still cannot assess safety, the request remains explicitly
        # unavailable instead of falling back to normal.
        assert value is None and semantics.semantic_attempt_failed(source)
        assert len(seen) == 2
    else:
        # Evidence not bound to this source is a rejected model attempt, not a
        # normal default and not a usable partial understanding. The shared
        # structured path allows exactly one bounded correction attempt.
        assert value is None and semantics.semantic_attempt_failed(source)
        assert len(seen) == 2


def test_medical_safety_still_commits_an_explicit_memory(compact_db, monkeypatch):
    """Fail-closed must not become a blanket non-normal write rejection."""
    source = "请长期记住：我对核桃过敏"
    scenario = RetryScenario(monkeypatch, compact_db)
    raw = intent(source, "memory_update", decisions=[decision("我对核桃过敏")],
                 assessment=safety("medical", "我对核桃过敏", ["medical_history"],
                                   "scripted historical medical context"))
    client, seen = provider(monkeypatch, lambda body, number: wire(raw))
    monkeypatch.setattr(scenario.agent.graph, "llm", client)
    result = asyncio.run(scenario.respond(ChatRequest(message=source)))

    assert len(seen) == 1
    assert result.metrics["memory_update_status"] == "committed"
    assert "已记住" in result.answer
    with compact_db() as s:
        assert len(list(s.scalars(select(UserFact)))) == 1


def test_missing_safety_never_saves_memory_via_real_entry(compact_db, monkeypatch):
    source = "请长期记住：我偏好晨练"
    scenario = RetryScenario(monkeypatch, compact_db)
    raw = intent(source, "memory_update", decisions=[decision("我偏好晨练")])
    raw.pop("safety")
    client, seen = provider(monkeypatch, lambda body, number: wire(raw))
    monkeypatch.setattr(scenario.agent.graph, "llm", client)
    result = asyncio.run(scenario.respond(ChatRequest(message=source)))

    assert result.intent == "execution_limited"
    assert result.metrics["mode_reason"] == "safety_assessment_unavailable"
    assert result.safety_level != "urgent" and not result.model_used
    # One bounded correction attempt is allowed before the explicit limit.
    assert len(seen) == 2
    scenario.agent.graph.parse_input.assert_not_awaited()
    assert scenario.saves == [] and scenario.tool_calls == []
    with compact_db() as s:
        assert s.execute(text("SELECT count(*) FROM user_facts")).scalar() == 0
