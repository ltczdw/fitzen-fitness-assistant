"""Shared AI response contracts: model boundaries mocked, real isolated storage."""
import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import select, text

from app.core.config import settings
from app.db.memory_models import MemoryTurn, UserFact
from app.schemas.chat import ChatRequest
from app.services.agent import dialogue_context as dialogue, record_clarifications as drafts, semantics
from app.services.memory import compact
from app.services.mcp.executor import ToolExecutionGateway, ToolSchemaError, TOOL_SCHEMAS
from app.services.security.input_guard import inspect_input
from evals.v2.safety import deny_network
from test_retry_operations import RetryScenario
from test_dialogue_continuity import history, record_model, task, outline


@pytest.fixture(autouse=True)
def isolated():
    token = semantics.request_understanding.set({})
    with deny_network():
        yield
    semantics.request_understanding.reset(token)


def _safety_task(source):
    return {"task_type": "safety_triage", "action": "question", "text": source,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False}


@pytest.mark.parametrize("source", [
    "我现在胸口疼，还喘不过气，能继续跑吗？", "突然胸部疼痛，吸不上气。", "现在没法呼吸。",
    "他刚刚失去了意识，叫不醒。", "我半边身体突然使不上劲。", "突然讲话含糊。",
    "我没有胸口疼，但现在喘不上气。", "不是没有胸痛，现在还在疼。",
])
def test_symptom_families_are_model_urgent(source):
    result = semantics.validate_understanding(source, {
        "tasks": [_safety_task(source)],
        "safety": {"level": "urgent", "evidence": source,
                   "categories": ["current_symptom"], "reason": "scripted current symptom"},
    })
    assert result["safety"]["level"] == "urgent"


@pytest.mark.parametrize("source", [
    "我没有胸口疼，也没有喘不过气。", "我没有出现过胸口疼。", "如何预防胸部疼痛？",
    "假设有人喘不过气，应该怎样预防？", "以前胸口疼，现在已经好了。", "跑步时正常喘气需要注意什么？",
    "去年胸口疼，早已经好了。", "以前胸部疼痛早就恢复了。", "小时候喘不上气，早已消失了。",
])
def test_symptom_context_is_not_a_literal_keyword_alarm(source):
    result = semantics.validate_understanding(source, {
        "tasks": [_safety_task(source)],
        "safety": {"level": "normal", "evidence": source,
                   "categories": ["negated", "prevention", "medical_history"],
                   "reason": "scripted context"},
    })
    assert result["safety"]["level"] == "normal"


@pytest.mark.parametrize("transport", ["json", "sse"])
@pytest.mark.parametrize("mode", ["legacy", "react_readonly", "react_write"])
def test_urgent_http_uses_structured_safety_and_no_tool(compact_db, monkeypatch, transport, mode):
    import httpx
    from app.api import chat
    from app.main import app
    from app.services.agent.semantics import RequestUnderstanding
    scenario = RetryScenario(monkeypatch, compact_db)
    if mode == "react_write":
        from app.core.config import settings
        monkeypatch.setattr(settings, "react_write_enabled", True)
        monkeypatch.setattr(settings, "memory_read_backend", "compact_v1")
    source = "现在胸部疼痛，透不过气，能练吗？"
    raw = RequestUnderstanding.model_validate({
        "tasks": [{"task_type": "safety_triage", "action": "question", "text": source,
                   "is_negated": False, "is_hypothetical": False, "is_third_party": False}],
        "safety": {"level": "urgent", "evidence": source,
                   "categories": ["current_symptom"], "reason": "scripted urgent"},
    })
    from app.services.telemetry import model_attempt

    async def structured(*_args, **_kwargs):
        with model_attempt("mock-structured") as attempt:
            attempt.observe({"usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                       "total_tokens": 2}})
            return raw

    monkeypatch.setattr(scenario.agent.graph.llm, "configured", lambda: True)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", structured)
    monkeypatch.setattr(chat, "agent", scenario.agent)
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as api:
            return await api.post("/api/chat/stream" if transport == "sse" else "/api/chat",
                json={"message": source, "mode": mode})
    wire = asyncio.run(send())
    assert wire.status_code == 200
    response = wire.json() if transport == "json" else json.loads([
        line[6:] for line in wire.text.splitlines() if line.startswith("data: ")][-1])
    assert response["safety_level"] == "urgent" and response["model_used"] is True
    assert not scenario.tool_calls


NUTRIENTS = dict(kcal=116, protein_g=2.6, carbs_g=25.9, fat_g=.3, fiber_g=.3)


@pytest.mark.parametrize("name", ["calculate_nutrition", "calculate_nutrition_batch"])
@pytest.mark.parametrize("remote", [False, True])
def test_one_nutrition_contract_for_local_and_remote_paths(monkeypatch, name, remote):
    from app.services.mcp.client import MCPClientBridge
    bridge = MCPClientBridge()
    arguments = {"raw_message": "synthetic", "nutrition_overrides": {"150克熟米饭": NUTRIENTS},
                 "entries" if name == "calculate_nutrition" else "groups": []}
    remote_call = AsyncMock(return_value={"status": "calculated"} if remote else None)
    local_call = Mock(return_value={"status": "calculated"})
    monkeypatch.setattr(settings, "mcp_url", "http://mock.fixture" if remote else "")
    bridge.call_tool = remote_call
    if remote:
        bridge._tool_schemas[name] = deepcopy(TOOL_SCHEMAS[name])
    result = asyncio.run(bridge.execute_tool(name, arguments, local_handler=local_call,
        trusted_user_id="demo-user", write_authorized=True))
    assert result == {"status": "calculated"}
    if remote:
        assert remote_call.await_args.args[1]["nutrition_overrides"] == arguments["nutrition_overrides"]
    else:
        assert local_call.call_args.kwargs["nutrition_overrides"] == arguments["nutrition_overrides"]
    assert local_call.call_count == (0 if remote else 1)


@pytest.mark.parametrize("change", [{"typo_field": {}}, {"nutrition_overrides": {"x": {**NUTRIENTS, "kcal": -1}}},
                                  {"nutrition_overrides": {"x": {**NUTRIENTS, "fat_g": True}}}])
def test_bad_nutrition_arguments_reject_before_dispatch(change):
    operation = AsyncMock()
    arguments = {"entries": [], "raw_message": "synthetic", **change}
    with pytest.raises(ToolSchemaError):
        asyncio.run(ToolExecutionGateway().execute("calculate_nutrition", arguments, schema=None,
            operation=operation, allowed_tools={"calculate_nutrition"}, write_authorized=True,
            trusted_user_id="demo-user"))
    operation.assert_not_awaited()


def test_remote_discovery_cannot_weaken_common_argument_contract():
    with pytest.raises(ToolSchemaError):
        ToolExecutionGateway._validate("calculate_nutrition", {"entries": [], "raw_message": "x", "unknown": 1},
            {"type": "object"}, trusted_user_id="demo-user")


def save_memory(quote, key, *, action="remember", targets=None):
    source = "请长期记住或更新：" + quote
    tid, _ = compact.begin_turn(ChatRequest(message=source, conversation_id="fact-scope"))
    raw = outline(source, reply_purpose="memory_update", memory_requests=[{
        "action": action, "quote": quote, "key": key, "confidence": .99, "target_fields": targets or []}])
    value = semantics.validate_understanding(source, raw, context={"long_memory": compact.current_facts()})
    response = {"answer": "已收到请求。", "intent": "conversation", "metrics": {"memory_update_requested": True}}
    compact.finish_turn(tid, response["answer"], response, semantic_values=[(source, value)])
    return tid


@pytest.mark.parametrize("quote", ["我对花生过敏", "我家只有弹力带", "我喜欢早上训练"])
def test_same_explicit_content_ignores_model_key_variation(compact_db, quote):
    save_memory(quote, "first-key")
    save_memory(quote, "different-key")
    with compact_db() as session:
        rows = list(session.scalars(select(UserFact)))
        assert len(rows) == 1 and rows[0].status == "active" and rows[0].source_quote == quote


def test_replacement_reuses_existing_fact_and_retires_only_authorized_targets(compact_db):
    save_memory("我家只有哑铃", "equipment")
    target = list(compact.current_facts())
    save_memory("我家有弹力带", "other-equipment")
    tid = save_memory("我家有弹力带", "third-key", action="replace", targets=target)
    with compact_db() as session:
        rows = list(session.scalars(select(UserFact)))
        assert len(rows) == 2
        assert {r.source_quote for r in rows if r.status == "active"} == {"我家有弹力带"}
        actual = next(r for r in rows if r.status == "active")
        meta = session.get(MemoryTurn, tid).messages[0]["app_meta"]["fact_update"]
        assert meta["fact_id"] == actual.fact_id and meta["field"] == actual.field


@pytest.mark.parametrize("mutation", ["hidden", "deleted", "epoch"])
def test_invalid_old_source_cannot_be_reused_as_duplicate(compact_db, mutation):
    tid = save_memory("我喜欢晨练", "key-one")
    from app.core import clock
    with compact_db() as session, session.begin():
        row = session.get(MemoryTurn, tid)
        if mutation == "epoch": row.memory_epoch += 1
        elif mutation == "hidden": row.hidden_at = clock.utc_now()
        else: row.deleted_at = clock.utc_now()
    save_memory("我喜欢晨练", "key-two")
    assert len(compact.current_facts()) == 1
    with compact_db() as session:
        assert len(list(session.scalars(select(UserFact)))) == 2


@pytest.mark.parametrize("question", ["我昨天的体重？", "我上次测的血压？", "我的跑步成绩是多少？"])
def test_unknown_personal_records_cannot_generate_invented_navigation(question):
    # Irrelevant data exists: the model must select nothing, rather than append
    # advice or turn unrelated facts into the requested measurement.
    memory = {"long_memory": {"own": {"source_quote": "我喜欢早上训练", "source_message_id": "u"}}}
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(return_value={"evidence": []}))
    value = semantics.validate_understanding(question, outline(question, reply_purpose="fact_recall"), context=memory)
    result = asyncio.run(dialogue.conversation_answer(llm, ChatRequest(message=question), memory, value))
    assert result["answer"] == "我没有找到与你这个问题对应的实际记录，无法给出具体信息。"
    assert llm.complete_structured.await_args.args[1] is dialogue.RecallSelection


@pytest.mark.parametrize("selection", [
    {"evidence": [], "answer": "去身体历史页面查询"},
    {"evidence": [{"source_id": "foreign", "quote": "80kg"}]},
    {"evidence": [{"source_id": "fact:own", "quote": "我重80kg"}]},
])
def test_personal_recall_rejects_free_text_or_forged_sources(selection):
    source = "我上次体重多少？"
    memory = {"long_memory": {"own": {"source_quote": "我喜欢晨练", "source_message_id": "u"}}}
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(return_value=selection))
    result = asyncio.run(dialogue.conversation_answer(llm, ChatRequest(message=source), memory,
        {"reply_mode": "conversation", "reply_purpose": "fact_recall"}))
    assert result["answer"] == "暂时无法继续这段对话，请稍后重试。"


@pytest.mark.parametrize("question", ["把刚才两条训练整理成清单", "换成表格，保留原来的数字", "只复述我提供的数据"])
@pytest.mark.parametrize("kind", ["workout_record", "nutrition_record"])
def test_readonly_domain_label_does_not_force_knowledge_worker(question, kind):
    memory = {"short_memory": [{"message_id": "u1", "role": "user", "content": "卧推55kg，3组10次"}]}
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(return_value={
        "answer": "卧推55kg，3组10次", "requires_grounding": False}))
    value = semantics.validate_understanding(question,
        {"tasks": [task(question, kind, "answer")], "reply_mode": "conversation"}, context=memory)
    assert value["reply_mode"] == "conversation"
    assert not value["validated_tasks"][0]["side_effect"]
    result = asyncio.run(dialogue.conversation_answer(llm, ChatRequest(message=question), memory, value))
    assert result["answer"] == "卧推55kg，3组10次"


@pytest.mark.parametrize("kind", ["training_qa", "training_plan", "nutrition_advice"])
def test_new_advice_keeps_grounded_worker(kind):
    memory = {"short_memory": [{"message_id": "u", "role": "user", "content": "我一周练三次"}]}
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(return_value={
        "answer": "需要新知识", "requires_grounding": True}))
    result = asyncio.run(dialogue.conversation_answer(llm, ChatRequest(message="新建议"), memory,
        {"reply_mode": "grounded", "tasks": [task("新建议", kind)]}))
    assert result is None
    llm.complete_structured.assert_not_awaited()


def test_grounded_mixed_request_keeps_independent_goals():
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock())
    result = asyncio.run(dialogue.conversation_answer(llm, ChatRequest(message="复述并给新计划"),
        {"short_memory": [{"role": "user", "content": "以前的信息"}]},
        {"reply_mode": "grounded", "tasks": [task("复述"), task("给新计划", "training_plan")]}))
    assert result is None
    llm.complete_structured.assert_not_awaited()


def test_partial_entity_uses_existing_draft_even_when_router_did_not_ask(compact_db, monkeypatch):
    source = "帮我记录今天晚餐的米饭。"
    raw = outline(source)
    raw["tasks"] = [task(source, "nutrition_record", "record")]
    raw["safety"] = {"level": "normal", "evidence": "", "categories": [], "reason": "scripted normal"}
    model = record_model([{"kind": "nutrition", "source_quote": "米饭", "name": "米饭",
                           "meal_type": "晚餐", "meal_quote": "晚餐", "date_quote": "今天"}])
    scenario = RetryScenario(monkeypatch, compact_db)
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", model.configured)
    from app.services.agent.semantics import RequestUnderstanding
    record_structured = model.complete_structured
    async def structured(messages, schema, *, name=None, **kwargs):
        if name == "return_request_understanding":
            return RequestUnderstanding.model_validate({
                "tasks": [{**raw["tasks"][0], "text": source}],
                "safety": {"level": "normal", "evidence": "", "categories": [],
                           "reason": "fixture"},
            })
        return await record_structured(messages, schema, name=name, **kwargs)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", structured)
    async def parse(query):
        value = semantics.validate_understanding(query, raw)
        semantics.remember_understanding(query, value)
        return {"tasks": [{**raw["tasks"][0], "foods": [{"name": "米饭", "quantity": None,
                    "unit": None, "weight_g": None, "meal_type": "晚餐"}]}]}
    monkeypatch.setattr(scenario.agent.graph, "parse_input", AsyncMock(side_effect=parse))
    request = ChatRequest(message=source, conversation_id="partial-food")
    response = asyncio.run(scenario.respond(request))
    assert response.intent == "clarification" and "米饭" in response.answer and "分量待补充" in response.answer
    assert "食物名称" not in response.answer
    with compact_db() as session:
        rows = list(session.scalars(select(MemoryTurn)))
        draft = rows[0].messages[0]["app_meta"][drafts.KEY]
        assert draft["records"][0]["meal_type"] == "晚餐"
        assert draft["records"][0]["record_date"]
        assert session.execute(text("SELECT COUNT(*) FROM user_food_log")).scalar() == 0

    # A later turn adopts only that frozen draft, keeping its meal and date.
    update = record_model([], [{"index": 0, "source_quote": "150克", "weight_g": 150}])
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", update.complete_structured)
    from app.services.agent import food_calculator
    from app.db import food_repository
    lookup = lambda name: SimpleNamespace(food_id=None, food_name=name, energy_kcal=116,
        protein_g=2.6, carb_g=25.9, fat_g=.3, fiber_g=.3, match_method="exact", confidence=1.)
    monkeypatch.setattr(food_calculator, "resolve_food", lookup)
    monkeypatch.setattr(food_repository, "resolve_food", lookup)
    followup = ChatRequest(message="150克", conversation_id="partial-food")
    saved = asyncio.run(scenario.respond(followup))
    assert "已记录" in saved.answer
    replayed = asyncio.run(scenario.respond(followup))
    assert replayed.replayed and replayed.answer == saved.answer
    with compact_db() as session:
        values = session.execute(text("SELECT food_name,weight_g,meal_type,record_time FROM user_food_log")).all()
        assert len(values) == 1 and tuple(values[0])[:3] == ("米饭", 150, "晚餐")
        assert str(values[0][3]).startswith(draft["records"][0]["record_date"])


@pytest.mark.parametrize("transport", ["json", "sse"])
def test_schema_failure_has_safe_final_response_and_preserves_earlier_records(compact_db, monkeypatch, transport):
    import httpx
    from app.api import chat
    from app.main import app
    from app.db import repositories
    repositories.create_workout("demo-user", "深蹲", 3, 8, 50, idempotency_key="earlier-task")
    source = "记录卧推60kg4组8次"
    item = {**task(source, "workout_record", "record"), "workouts": [
        {"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}]}
    scenario = RetryScenario(monkeypatch, compact_db, parsed_input={"tasks": [item]})
    monkeypatch.setattr(scenario.agent.graph, "run", AsyncMock(side_effect=ToolSchemaError("synthetic schema error")))
    monkeypatch.setattr(chat, "agent", scenario.agent)
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as api:
            return await api.post("/api/chat/stream" if transport == "sse" else "/api/chat", json={"message": source})
    wire = asyncio.run(send())
    assert wire.status_code == 200
    if transport == "json":
        response = wire.json()
    else:
        frames = [json.loads(line[6:]) for line in wire.text.splitlines() if line.startswith("data: ")]
        response = {**frames[-1], "answer": "".join(f.get("content", "") for f in frames)}
    assert response["intent"] == "error" and response["metrics"]["error_code"] == "tool_schema_rejected"
    assert "已保存的记录会保留" in response["answer"]
    with compact_db() as session:
        assert session.execute(text("SELECT COUNT(*) FROM workout_records")).scalar() == 1


def test_fact_recall_can_only_render_sourced_quotes():
    quote = "我家有哑铃和弹力带"
    memory = {"long_memory": {"own": {"source_quote": quote, "source_message_id": "u"}}}
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(return_value={
        "evidence": [{"source_id": "fact:own", "quote": quote}]}))
    response = asyncio.run(dialogue.conversation_answer(llm, ChatRequest(message="我有哪些器械？"), memory,
        {"reply_mode": "conversation", "reply_purpose": "fact_recall"}))
    assert response["answer"] == "你提供过的信息是：\n“我家有哑铃和弹力带”"


def test_fact_deleted_during_model_await_cannot_be_returned(compact_db):
    tid = save_memory("我家有哑铃", "equipment")
    facts = compact.current_facts()
    key = next(iter(facts))
    request = ChatRequest(message="我有什么器械？", conversation_id="another")
    live, _ = compact.begin_turn(request)
    async def model(*args, **kwargs):
        from app.core import clock
        with compact_db() as session, session.begin():
            session.get(MemoryTurn, tid).deleted_at = clock.utc_now()
        return {"evidence": [{"source_id": "fact:" + key, "quote": "我家有哑铃"}]}
    llm = SimpleNamespace(configured=lambda: True, complete_structured=model)
    token = compact.current_turn.set(live)
    try:
        response = asyncio.run(dialogue.conversation_answer(llm, request, {"long_memory": facts},
            {"reply_mode": "conversation", "reply_purpose": "fact_recall"}))
    finally:
        compact.current_turn.reset(token)
    assert response["answer"] == "暂时无法继续这段对话，请稍后重试。"


def test_concurrent_same_content_commits_once_under_profile_lock(compact_db):
    from concurrent.futures import ThreadPoolExecutor
    quote = "我喜欢使用弹力带"
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda key: save_memory(quote, key), ["parallel-one", "parallel-two"]))
    with compact_db() as session:
        rows = list(session.scalars(select(UserFact)))
        assert len(rows) == 1 and rows[0].source_quote == quote and rows[0].status == "active"


def test_real_agent_readonly_record_domain_uses_history_without_rag(compact_db, monkeypatch):
    original = "卧推55kg3组10次；深蹲70kg4组8次"
    tid, messages = history(compact_db, user=original, assistant="两条训练已记录。")
    source = "只将刚才两条训练换成清单，不要保存。"
    scenario = RetryScenario(monkeypatch, compact_db)
    from app.services.agent import execution
    monkeypatch.setattr(execution.memory_service, "load", AsyncMock(return_value={"short_memory": messages, "memory_backend": "fixture"}))
    async def parse(query):
        raw = {"tasks": [{**task(query, "workout_record", "answer"), "is_negated": True}],
               "reply_mode": "conversation", "dialogue_ids": [tid + ":user", tid + ":assistant"]}
        value = semantics.validate_understanding(query, raw, context={"short_memory": messages})
        semantics.remember_understanding(query, value)
        return {"tasks": [{k: t[k] for k in ("task_type", "action", "text", *drafts.FLAGS)} for t in value["tasks"]]}
    monkeypatch.setattr(scenario.agent.graph, "parse_input", AsyncMock(side_effect=parse))
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", lambda: True)
    async def structured(messages, schema, *, name=None, **kwargs):
        if name == "return_request_understanding":
            payload = json.loads(messages[-1]["content"])
            source_value = str(payload.get("query") or "")
            return schema.model_validate({
                "tasks": [{"task_type": "training_qa", "action": "question",
                           "text": source_value, "is_negated": False,
                           "is_hypothetical": False, "is_third_party": False}],
                "safety": {"level": "normal", "evidence": "", "categories": [],
                           "reason": "scripted normal"},
            })
        return {"answer": original, "requires_grounding": False}
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", AsyncMock(side_effect=structured))
    forbidden = Mock(side_effect=AssertionError("pure context must not retrieve"))
    monkeypatch.setattr(scenario.agent.graph.rag, "search", forbidden)
    result = asyncio.run(scenario.respond(ChatRequest(message=source, conversation_id="context")))
    assert result.intent == "conversation" and result.answer == original
    forbidden.assert_not_called()
    assert not scenario.tool_calls


@pytest.mark.parametrize("name", ["calculate_nutrition", "calculate_nutrition_batch"])
def test_nutrition_override_dispatch_really_saves_once_and_replays(compact_db, monkeypatch, name):
    from dataclasses import asdict
    from app.services.agent.food_parser import FoodInput
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.services.mcp.client import MCPClientBridge
    monkeypatch.setattr(settings, "mcp_url", "")
    entry = FoodInput(raw_text="150克合成食物", requested_name="合成食物", quantity=1,
        unit="g", weight_g=150, portion_name=None, meal_type="晚餐")
    arguments = {"raw_message": entry.raw_text, "idempotency_key": "override-contract",
                 "nutrition_overrides": {entry.raw_text: NUTRIENTS}}
    arguments["entries" if name == "calculate_nutrition" else "groups"] = (
        [asdict(entry)] if name == "calculate_nutrition" else [{"meal_type": "晚餐", "entries": [asdict(entry)]}])
    tools = FitnessMultiAgentGraph().tools
    bridge = MCPClientBridge()
    async def execute():
        return await bridge.execute_tool(name, arguments, local_handler=getattr(tools, name),
            trusted_user_id="demo-user", write_authorized=True)
    first, replay = asyncio.run(execute()), asyncio.run(execute())
    assert first["status"] == replay["status"] == "calculated"
    assert first["log_ids"] == replay["log_ids"] and first["totals"]["kcal"] == 174
    with compact_db() as session:
        rows = session.execute(text("SELECT weight_g,is_estimated,meal_type FROM user_food_log")).all()
        assert len(rows) == 1 and tuple(rows[0]) == (150, 1, "晚餐")


def test_write_timeout_stays_unknown_and_does_not_fallback():
    from app.db.storage import CommitUnknownError
    operation = AsyncMock(side_effect=asyncio.TimeoutError)
    fallback = AsyncMock(return_value={"status": "calculated"})
    with pytest.raises(CommitUnknownError):
        asyncio.run(ToolExecutionGateway().execute("calculate_nutrition", {"entries": [], "raw_message": "synthetic"},
            schema=None, operation=operation, fallback=fallback, allowed_tools={"calculate_nutrition"},
            write_authorized=True, trusted_user_id="demo-user"))
    fallback.assert_not_awaited()


@pytest.mark.parametrize("name", ["calculate_nutrition", "calculate_nutrition_batch"])
def test_actual_mcp_registered_function_executes_override(compact_db, monkeypatch, name):
    from dataclasses import asdict
    from app.services.agent.food_parser import FoodInput
    from app.services.mcp.server import mcp
    entry = FoodInput(raw_text="100克合成谷物", requested_name="合成谷物", quantity=1,
        unit="g", weight_g=100, portion_name=None, meal_type="早餐")
    arguments = {"user_id": "demo-user", "raw_message": entry.raw_text, "idempotency_key": "server-override",
                 "nutrition_overrides": {entry.raw_text: NUTRIENTS}}
    arguments["entries" if name == "calculate_nutrition" else "groups"] = ([asdict(entry)]
        if name == "calculate_nutrition" else [{"meal_type": "早餐", "entries": [asdict(entry)]}])
    # In-process MCP dispatch, real registered function and calculator/SQL;
    # no server process or network transport is started.
    asyncio.run(mcp.call_tool(name, arguments))
    with compact_db() as session:
        rows = session.execute(text("SELECT weight_g,is_estimated,meal_type FROM user_food_log")).all()
        assert len(rows) == 1 and tuple(rows[0]) == (100, 1, "早餐")


def test_recall_source_errors_use_existing_two_attempt_repair_contract(monkeypatch):
    from test_dialogue_contract import provider, wire
    quote = "我家有弹力带"
    memory = {"long_memory": {"owned": {"source_quote": quote, "source_message_id": "u"}}}
    def respond(body, number):
        return wire({"evidence": [{"source_id": "foreign" if number == 1 else "fact:owned", "quote": quote}]},
                    "select_personal_record_sources")
    client, seen = provider(monkeypatch, respond)
    result = asyncio.run(dialogue.conversation_answer(client, ChatRequest(message="我有什么器械？"), memory,
        {"reply_mode": "conversation", "reply_purpose": "fact_recall"}))
    assert result["answer"] == "你提供过的信息是：\n“我家有弹力带”"
    assert len(seen) == 2
