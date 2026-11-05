"""Cross-domain dialogue tests: all external clients mocked, SQL isolated."""
import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from sqlalchemy import select, text

from app.core import clock
from app.db.memory_models import MemoryTurn, UserFact
from app.schemas.chat import ChatRequest
from app.services.agent import dialogue_context as dialogue, record_clarifications as records
from app.services.agent.semantics import (
    validate_understanding, understand, request_understanding, remember_understanding,
)
from app.services.memory import compact
from test_retry_operations import RetryScenario


def task(text, kind="training_qa", action="question"):
    return {"task_type": kind, "action": action, "text": text,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False}


def outline(source, **kwargs):
    value = {"tasks": [task(source)], **kwargs}
    value.setdefault("safety", {"level": "normal", "evidence": "", "categories": [],
                                "reason": "scripted normal"})
    return value


def shared_model(raw, answer="可以，我会按你刚才选择的第二项继续。", *, grounding=False):
    async def structured(messages, schema, *, name, **kwargs):
        if name == "return_request_understanding":
            return schema.model_validate(raw)
        if name == "return_dialogue_answer":
            return schema.model_validate({"answer": answer, "requires_grounding": grounding})
        raise AssertionError("Unexpected model boundary: " + name)
    return SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(side_effect=structured))


def history(factory, *, cid="context", user="我需要居家训练方案", assistant="你希望每周训练几次？"):
    tid, _ = compact.begin_turn(ChatRequest(message=user, conversation_id=cid))
    compact.finish_turn(tid, assistant, {"answer": assistant, "intent": "clarification"})
    with factory() as s:
        messages = copy.deepcopy(s.get(MemoryTurn, tid).messages)
    return tid, messages


@pytest.fixture(autouse=True)
def fresh_semantics():
    token = request_understanding.set({})
    yield
    request_understanding.reset(token)


def test_shared_understanding_receives_both_speakers_and_excludes_internal_messages():
    source = "第二个，再简短一点"
    context = {"short_memory": [
        {"role": "user", "content": "给两个方案", "message_id": "u"},
        {"role": "assistant", "content": "方案一A；方案二B", "message_id": "a"},
        {"role": "tool", "content": "PRIVATE-TOOL-LEDGER", "message_id": "t"},
    ]}
    llm = shared_model(outline(source, dialogue_ids=["u", "a"], relation="follow_up", reply_mode="conversation"))
    result = asyncio.run(understand(llm, source, context=context))
    payload = json.loads(llm.complete_structured.await_args.args[0][-1]["content"])
    assert [(m["role"], m["message_id"]) for m in payload["history"]] == [("user", "u"), ("assistant", "a")]
    assert "PRIVATE-TOOL-LEDGER" not in json.dumps(payload)
    wire = llm.complete_structured.await_args.args[0]
    assert [m["role"] for m in wire] == ["system", "user", "assistant", "user"]
    assert "方案一A；方案二B" in wire[2]["content"]
    assert sum("方案一A；方案二B" in m["content"] for m in wire) == 1
    assert result["reply_mode"] == "conversation"


@pytest.mark.parametrize("field", ["evidence_ids", "context_ids"])
def test_assistant_reference_is_context_and_never_user_fact(field):
    source = "那个动作换成徒手可以吗"
    raw = outline(source, references=[{"source_text": source, "query": "直臂下压可以换成徒手动作吗", field: ["a"]}])
    result = validate_understanding(source, raw, context={"short_memory": [{"message_id": "a", "role": "assistant", "content": "第二个动作是直臂下压"}]})
    assert result["references"][0]["evidence_ids"] == []
    assert result["references"][0]["context_ids"] == ["a"]
    assert not result["validated_tasks"][0]["side_effect"]


@pytest.mark.parametrize("field", ["dialogue_ids", "critical_history_ids"])
def test_unknown_dialogue_or_assistant_as_user_constraint_is_rejected(field):
    source = "继续"
    identifier = "unknown" if field == "dialogue_ids" else "a"
    with pytest.raises(ValueError):
        validate_understanding(source, outline(source, **{field: [identifier]}), context={"short_memory": [{"message_id": "a", "role": "assistant", "content": "建议"}]})


def test_conversation_mode_cannot_authorize_write_or_historical_fact():
    source = "把刚才那个记下来"
    raw = {"tasks": [task(source, "workout_record", "record")], "reply_mode": "conversation",
           "references": [{"source_text": source, "query": "卧推60kg4组8次", "context_ids": ["a"]}]}
    result = validate_understanding(source, raw, context={"short_memory": [{"message_id": "a", "role": "assistant", "content": "建议卧推60kg4组8次"}]})
    assert result["reply_mode"] == "grounded"
    assert result["tasks"][0]["text"] == source
    # The generated retrieval rewrite has no workout payload or source span.
    from app.services.agent.workout_extractor import validate_workout_records_against_source
    kept, dropped = validate_workout_records_against_source([{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}], source)
    assert not kept and dropped


def test_budget_keeps_the_complete_referenced_pair_under_pressure():
    from app.services.context.budget import ContextBudgetManager
    source = "选第二种"
    messages = [{"role": "user", "content": "两种选择", "message_id": "u"},
                {"role": "assistant", "content": "第一种周一练，第二种周三练", "message_id": "a"},
                {"role": "user", "content": "无关" * 1500, "message_id": "u2"},
                {"role": "assistant", "content": "无关回答" * 1500, "message_id": "a2"}]
    value = validate_understanding(source, outline(source, dialogue_ids=["a"], reply_mode="conversation"), context={"short_memory": messages})
    remember_understanding(source, value)
    result = ContextBudgetManager(window=1400, output=100, margin=0).assemble(system="助手", query=source, state={"short_memory": messages}, hits=[])
    assert [m["role"] for m in result if m["content"] in {"两种选择", "第一种周一练，第二种周三练"}] == ["user", "assistant"]
    assert not any("无关" in m["content"] for m in result)


def test_latest_question_is_not_compressed_before_understanding():
    original = [{"role": "user", "content": "两个选择是什么"},
                {"role": "assistant", "content": "开头，必须保留的第二个选项，结尾"}]
    excerpt = [{"role": "user", "content": "两个选择是什么"},
               {"role": "assistant", "content": "开头[省略]结尾"}]
    rows = [SimpleNamespace(messages=original, window_summary=json.dumps(excerpt)),
            SimpleNamespace(messages=original, window_summary=json.dumps(excerpt))]
    result = compact.recent_dialogue(rows)
    assert result[1]["content"] == excerpt[1]["content"]
    assert result[-1]["content"] == original[-1]["content"]


def test_full_referenced_pair_restored_from_sql_instead_of_compressed_excerpt(compact_db):
    tid, messages = history(compact_db, assistant="开头\n中间完整的第二方案\n结尾")
    messages[-1]["content"] = "[压缩摘录]开头，结尾"
    request = ChatRequest(message="第二方案是什么", conversation_id="context")
    live, _ = compact.begin_turn(request)
    token = compact.current_turn.set(live)
    try:
        result = asyncio.run(dialogue.restore(request, {"short_memory": messages}, {"dialogue_ids": [tid + ":assistant"]}))
    finally:
        compact.current_turn.reset(token)
    assert "中间完整的第二方案" in result["short_memory"][-1]["content"]
    assert result["short_memory"][-1]["role"] == "assistant"


@pytest.mark.parametrize("mutation", ["hidden", "deleted", "epoch", "conversation"])
def test_restore_rechecks_privacy_and_conversation_at_use_time(compact_db, mutation):
    tid, messages = history(compact_db)
    request = ChatRequest(message="三次", conversation_id="context")
    live, _ = compact.begin_turn(request)
    with compact_db() as s, s.begin():
        row = s.get(MemoryTurn, tid)
        if mutation == "hidden": row.hidden_at = clock.utc_now()
        if mutation == "deleted": row.deleted_at = clock.utc_now()
        if mutation == "epoch": row.memory_epoch += 1
        if mutation == "conversation": row.conversation_id = "foreign"
    token = compact.current_turn.set(live)
    try:
        with pytest.raises(dialogue.DialogueContextUnavailable):
            asyncio.run(dialogue.restore(request, {"short_memory": messages}, {"dialogue_ids": [tid + ":assistant"]}))
    finally:
        compact.current_turn.reset(token)


@pytest.mark.parametrize("source,relation,answer", [
    ("第二个，整理成表格", "follow_up", "方案二：每周三次居家训练。"),
    ("不是第二个，是第一个", "correction", "你选择的是第一项。"),
    ("都可以", "clarification", "收到，你对两个选项都没有偏好。"),
    ("先不继续这个，聊一下刚才说的饮食记录功能", "new", "可以，我们继续讨论饮食记录功能。"),
])
def test_real_agent_continues_dialogue_without_rag_or_any_business_write(compact_db, monkeypatch, source, relation, answer):
    tid, messages = history(compact_db, user="两个方案都是什么", assistant="第一项A，第二项B，你选哪个？")
    llm = shared_model(outline(source, dialogue_ids=[tid + ":user", tid + ":assistant"], relation=relation, reply_mode="conversation"), answer)
    scenario = RetryScenario(monkeypatch, compact_db)
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", llm.configured)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", llm.complete_structured)
    from app.services.agent import execution
    monkeypatch.setattr(execution.memory_service, "load", AsyncMock(return_value={"short_memory": messages, "memory_backend": "fixture"}))
    async def parse(query):
        value = await understand(scenario.agent.graph.llm, query)
        return {"tasks": [{k: t[k] for k in ("task_type", "action", "text", *records.FLAGS)} for t in value["tasks"]]}
    monkeypatch.setattr(scenario.agent.graph, "parse_input", AsyncMock(side_effect=parse))
    forbidden_rag = AsyncMock(side_effect=AssertionError("conversation must not retrieve"))
    monkeypatch.setattr(scenario.agent.graph.rag, "search", forbidden_rag)
    request = ChatRequest(message=source, conversation_id="context")
    response = asyncio.run(scenario.respond(request))
    assert response.intent == "conversation" and response.answer == answer
    prompt = llm.complete_structured.await_args.args[0]
    assert {m["role"] for m in prompt} == {"system", "user", "assistant"}
    assert any(m["role"] == "assistant" and m["content"] == messages[-1]["content"] for m in prompt)
    assert not scenario.tool_calls and not forbidden_rag.called
    with compact_db() as s:
        assert s.execute(text("SELECT COUNT(*) FROM workout_records")).scalar() == 0
        assert s.execute(text("SELECT COUNT(*) FROM user_food_log")).scalar() == 0
        assert list(s.scalars(select(UserFact))) == []


def test_new_knowledge_falls_back_to_grounded_worker_and_resolved_query(compact_db, monkeypatch):
    from app.services.rag.retrieval import RetrievedChunk
    tid, messages = history(compact_db, user="帮我安排居家训练", assistant="每周能练几次？")
    source = "每周三次"
    query = "居家训练每周三次如何安排"
    raw = outline(source, dialogue_ids=[tid + ":user", tid + ":assistant"], reply_mode="conversation", references=[{
        "source_text": source, "query": query, "context_ids": [tid + ":assistant"], "evidence_ids": [tid + ":user"]}])
    llm = shared_model(raw, grounding=True)
    scenario = RetryScenario(monkeypatch, compact_db)
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", llm.configured)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", llm.complete_structured)
    from app.services.agent import execution
    monkeypatch.setattr(execution.memory_service, "load", AsyncMock(return_value={"short_memory": messages, "memory_backend": "fixture"}))
    async def parse(query):
        value = await understand(scenario.agent.graph.llm, query)
        return {"tasks": [{k: t[k] for k in ("task_type", "action", "text", *records.FLAGS)} for t in value["tasks"]]}
    monkeypatch.setattr(scenario.agent.graph, "parse_input", AsyncMock(side_effect=parse))
    seen = []
    def retrieve(query, **kwargs):
        seen.append(query)
        return [RetrievedChunk("居家训练", "居家训练每周三次，逐步增加，循序渐进", 1.0, "fixture", {"rerank_status": "model", "rerank_score": .95}, "fixture-1")]
    monkeypatch.setattr(scenario.agent.graph.rag, "search", retrieve)
    monkeypatch.setattr(scenario.agent.graph.mcp, "complete_with_tools", AsyncMock(return_value={"content": "可以每周安排三次，并逐步增加训练量。"}))
    response = asyncio.run(scenario.respond(ChatRequest(message=source, conversation_id="context")))
    assert seen == [query] and response.intent != "conversation"
    assert "三次" in response.answer and not scenario.saves


def record_model(proposed, patches=None, *, accept=True):
    async def structured(messages, schema, *, name, **kwargs):
        payload = json.loads(messages[-1]["content"])
        if name == "return_request_understanding":
            source = str(payload.get("query") or "")
            return schema.model_validate({
                "tasks": [task(source, "training_qa", "question")],
                "safety": {"level": "normal", "evidence": "", "categories": [],
                           "reason": "scripted record context"},
            })
        if name == "return_record_proposals":
            return schema.model_validate({"records": proposed})
        assert name == "return_record_reply"
        return schema.model_validate({"draft_id": payload["draft_id"], "action": "update" if patches else "accept",
            "accepts_all": accept, "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            "patches": patches or []})
    return SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(side_effect=structured))


def create_record_draft(source, proposed, *, kind="nutrition_record"):
    request = ChatRequest(message=source, conversation_id="food-context")
    tid, _ = compact.begin_turn(request)
    token = compact.current_turn.set(tid)
    try:
        question = asyncio.run(records.create_question(record_model(proposed), request,
            {"tasks": [task(source, kind, "record")], "needs_clarification": True}))
    finally:
        compact.current_turn.reset(token)
    compact.finish_turn(tid, question, {"answer": question, "intent": "clarification"})
    return tid, question


def test_food_missing_amount_reuses_same_controller_and_real_sql_save(compact_db, monkeypatch):
    proposed = [{"kind": "nutrition", "source_quote": "鸡蛋", "name": "鸡蛋", "meal_type": "午餐", "meal_quote": "中午"}]
    tid, question = create_record_draft("帮我记录中午鸡蛋", proposed)
    assert "分量待补充" in question and "午餐" in question
    llm = record_model(proposed, [{"index": 0, "source_quote": "50克", "weight_g": 50}])
    scenario = RetryScenario(monkeypatch, compact_db)
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", llm.configured)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", llm.complete_structured)
    # Boundary food lookup only; calculator and SQL logging remain real.
    from app.services.agent import food_calculator
    lookup = lambda name: SimpleNamespace(
        food_id=None, food_name=name, energy_kcal=144, protein_g=12.6,
        carb_g=1.2, fat_g=9.6, fiber_g=0, match_method="exact", confidence=1.0)
    monkeypatch.setattr(food_calculator, "resolve_food", lookup)
    from app.db import food_repository
    monkeypatch.setattr(food_repository, "resolve_food", lookup)
    request = ChatRequest(message="50克", conversation_id="food-context")
    response = asyncio.run(scenario.respond(request))
    assert "已记录" in response.answer, response.answer
    with compact_db() as s:
        rows = s.execute(text("SELECT food_name, weight_g, meal_type FROM user_food_log")).all()
    assert [tuple(r) for r in rows] == [("鸡蛋", 50, "午餐")]
    replayed = asyncio.run(scenario.respond(request))
    assert replayed.replayed
    with compact_db() as s:
        assert s.execute(text("SELECT COUNT(*) FROM user_food_log")).scalar() == 1


def test_food_name_and_meal_correction_come_from_current_reply(compact_db):
    proposed = [{"kind": "nutrition", "source_quote": "50克鸡蛋", "name": "鸡蛋", "weight_g": 50,
                 "meal_type": "午餐", "meal_quote": "午餐"}]
    create_record_draft("记录午餐50克鸡蛋", proposed)
    request = ChatRequest(message="改成晚餐的牛肉", conversation_id="food-context")
    tid, _ = compact.begin_turn(request)
    token = compact.current_turn.set(tid)
    try:
        result = asyncio.run(records.continue_request(record_model(proposed, [{"index": 0, "source_quote": "晚餐的牛肉", "name": "牛肉", "meal_type": "晚餐", "meal_quote": "晚餐"}], accept=False), request))
    finally:
        compact.current_turn.reset(token)
    assert "牛肉" in result["answer"] and "晚餐" in result["answer"] and "source" not in result


def test_cross_domain_patch_and_assistant_food_number_cannot_be_used_as_fact(compact_db):
    proposed = [{"kind": "nutrition", "source_quote": "鸡蛋", "name": "鸡蛋"}]
    create_record_draft("记录鸡蛋", proposed)
    request = ChatRequest(message="按你说的", conversation_id="food-context")
    tid, _ = compact.begin_turn(request)
    token = compact.current_turn.set(tid)
    try:
        llm = record_model(proposed, [{"index": 0, "source_quote": "按你说的", "weight_g": 999}])
        with pytest.raises(records.DraftError):
            asyncio.run(records.continue_request(llm, request))
    finally:
        compact.current_turn.reset(token)


def test_shared_controller_keeps_two_foods_in_one_meal(compact_db):
    proposed = [
        {"kind": "nutrition", "source_quote": "50克鸡蛋", "name": "鸡蛋", "weight_g": 50, "meal_type": "午餐", "meal_quote": "午餐"},
        {"kind": "nutrition", "source_quote": "100克米饭", "name": "米饭", "weight_g": 100, "meal_type": "午餐", "meal_quote": "午餐"},
    ]
    create_record_draft("记录午餐50克鸡蛋和100克米饭", proposed)
    request = ChatRequest(message="对，都保存", conversation_id="food-context")
    tid, _ = compact.begin_turn(request)
    token = compact.current_turn.set(tid)
    try:
        result = asyncio.run(records.continue_request(record_model(proposed), request))
    finally:
        compact.current_turn.reset(token)
    assert len(result["parsed_input"]["tasks"]) == 1
    assert len(result["parsed_input"]["tasks"][0]["foods"]) == 2
    from app.services.agent.planner_validation import validate_planner_output
    checked = validate_planner_output(result["source"], result["parsed_input"])
    assert not checked.fallback_required


def test_food_cancellation_uses_shared_terminal_state(compact_db):
    proposed = [{"kind": "nutrition", "source_quote": "鸡蛋", "name": "鸡蛋"}]
    origin, _ = create_record_draft("记录鸡蛋", proposed)
    request = ChatRequest(message="这次就不记录了", conversation_id="food-context")
    tid, _ = compact.begin_turn(request)
    llm = record_model(proposed)
    async def cancel(messages, schema, **kwargs):
        payload = json.loads(messages[-1]["content"])
        return schema.model_validate({"draft_id": payload["draft_id"], "action": "cancel", "accepts_all": False,
            "is_negated": True, "is_hypothetical": False, "is_third_party": False})
    llm.complete_structured.side_effect = cancel
    token = compact.current_turn.set(tid)
    try:
        result = asyncio.run(records.continue_request(llm, request))
    finally:
        compact.current_turn.reset(token)
    assert result["intent"] == "clarification_cancelled" and "训练" not in result["answer"]
    with compact_db() as s:
        assert s.get(MemoryTurn, origin).messages[0]["app_meta"][records.KEY]["state"] == "cancelled"
        assert s.execute(text("SELECT COUNT(*) FROM user_food_log")).scalar() == 0


def test_confirmation_after_topic_switch_cannot_silently_consume_an_old_draft(compact_db):
    proposed = [{"kind": "nutrition", "source_quote": "50克鸡蛋", "name": "鸡蛋", "weight_g": 50, "meal_type": "午餐", "meal_quote": "午餐"}]
    create_record_draft("记录午餐50克鸡蛋", proposed)
    history(compact_db, cid="food-context", user="另外我想调整计划", assistant="你是打算先从居家训练开始吗？")
    request = ChatRequest(message="是", conversation_id="food-context")
    tid, _ = compact.begin_turn(request)
    token = compact.current_turn.set(tid)
    try:
        result = asyncio.run(records.continue_request(record_model(proposed), request))
    finally:
        compact.current_turn.reset(token)
    assert "source" not in result and result["intent"] == "clarification"
    with compact_db() as s:
        assert s.execute(text("SELECT COUNT(*) FROM user_food_log")).scalar() == 0


def test_existing_workout_v1_state_and_same_request_binding_survive_shared_controller(compact_db):
    from test_workout_clarifications import create_pending, continue_pending
    _, origin, _ = create_pending(compact_db)
    with compact_db() as s, s.begin():
        row = s.get(MemoryTurn, origin)
        messages = copy.deepcopy(row.messages)
        old = messages[0]["app_meta"].pop(records.KEY)
        for r in old["records"]:
            r.pop("kind", None)
        messages[0]["app_meta"]["workout_clarification_v1"] = old
        row.messages = messages
    _, live, result = continue_pending()
    assert result["source"] and result["origin"] == origin
    with compact_db() as s, s.begin():
        row = s.get(MemoryTurn, live)
        messages = copy.deepcopy(row.messages)
        messages[0]["app_meta"]["workout_continuation_v1"] = messages[0]["app_meta"].pop(records.BINDING)
        row.messages = messages
    llm = record_model([])
    _, _, resumed = continue_pending(llm=llm)
    assert resumed == result and llm.complete_structured.await_count == 0
