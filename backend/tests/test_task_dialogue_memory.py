"""Mixed-task behavior through the real agent/graph; providers mocked, SQL isolated."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import select, text

from app.db.memory_models import MemoryTurn, UserFact
from app.schemas.chat import ChatRequest
from app.services.agent import dialogue_context as dialogue, semantics
from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
from app.services.memory import compact
from evals.v2.safety import deny_network
from test_ai_common_causes import save_memory
from test_dialogue_continuity import history, task
from test_retry_operations import RetryScenario
from tests.plan_fixture_support import plan_message


@pytest.fixture(autouse=True)
def isolated():
    token = semantics.request_understanding.set({})
    with deny_network():
        yield
    semantics.request_understanding.reset(token)


def semantic_envelope(messages):
    """Minimal source-bound model understanding for the early safety gate."""
    payload = json.loads(messages[-1]["content"])
    source = str(payload.get("query") or "")
    return {"tasks": [{"task_type": "training_qa", "action": "question", "text": source,
                       "is_negated": False, "is_hypothetical": False,
                       "is_third_party": False}],
            "safety": {"level": "normal", "evidence": "", "categories": [],
                       "reason": "scripted normal"}}


def mixed_scenario(compact_db, monkeypatch, *, nutrition=False, reverse=False, fail_dialogue=False):
    past = "午餐90克米饭，晚餐110克鸡蛋" if nutrition else "划船40公斤3组12次，硬拉65公斤4组6次"
    tid, messages = history(compact_db, user=past, assistant="已收到以上内容。")
    recap = "回顾刚才两餐及克数" if nutrition else "回顾刚才两项训练的重量"
    plan = "给一个通用的周训练计划示例"
    clauses = [plan, recap] if reverse else [recap, plan]
    source = "；".join(clauses)
    items = [({**task(clause, "nutrition_record" if nutrition else "workout_record", "answer"),
               "reply_mode": "conversation", "context_ids": [tid + ":user", tid + ":assistant"]}
              if clause == recap else task(clause, "training_plan")) for clause in clauses]
    scenario = RetryScenario(monkeypatch, compact_db)
    from app.services.agent import execution
    monkeypatch.setattr(execution.memory_service, "load", AsyncMock(return_value={"short_memory": messages, "memory_backend": "fixture"}))
    async def parse(query):
        value = semantics.validate_understanding(query, {"tasks": items, "reply_mode": "grounded"}, context={"short_memory": messages})
        semantics.remember_understanding(query, value)
        return {"tasks": [{k: t[k] for k in ("task_type", "action", "text", "is_negated", "is_hypothetical", "is_third_party")} for t in value["tasks"]]}
    monkeypatch.setattr(scenario.agent.graph, "parse_input", AsyncMock(side_effect=parse))
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", lambda: True)
    async def structured(messages, schema, *, name, **kwargs):
        if name == "return_request_understanding":
            return schema.model_validate(semantic_envelope(messages))
        if name == "return_dialogue_answer":
            if fail_dialogue:
                raise ValueError("synthetic invalid reply")
            wire = json.dumps(messages, ensure_ascii=False)
            assert past in wire and plan not in messages[-1]["content"]
            return {"answer": past, "requires_grounding": False}
        if name == "return_plan_information":
            return {"scope": "generic", "scope_quote": plan, "facts": []}
        raise AssertionError(name)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", AsyncMock(side_effect=structured))
    retrieval = AsyncMock(return_value=[])
    monkeypatch.setattr(scenario.agent.graph, "_readonly_retrieval", retrieval)
    return scenario, ChatRequest(message=source, conversation_id="context"), past, recap, plan, retrieval


@pytest.mark.parametrize("nutrition", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_recap_survives_independent_plan_evidence_failure(compact_db, monkeypatch, nutrition, reverse):
    scenario, request, past, recap, plan, retrieval = mixed_scenario(compact_db, monkeypatch, nutrition=nutrition, reverse=reverse)
    result = asyncio.run(scenario.respond(request))
    assert past in result.answer and "匹配的知识依据" in result.answer
    assert result.answer.index(past) < result.answer.index("匹配的知识依据") if not reverse else result.answer.index("匹配的知识依据") < result.answer.index(past)
    assert [r["status"] for r in result.metrics["task_outcomes"]] == (["insufficient_evidence", "complete"] if reverse else ["complete", "insufficient_evidence"])
    assert len(retrieval.await_args_list) == 1 and retrieval.await_args.args[0] == plan + " 健身房"
    assert not scenario.saves
    with compact_db() as s:
        assert s.execute(text("SELECT COUNT(*) FROM workout_records")).scalar() == 0
        assert s.execute(text("SELECT COUNT(*) FROM user_food_log")).scalar() == 0


def test_one_dialogue_failure_does_not_cancel_other_task(compact_db, monkeypatch):
    scenario, request, past, _, _, retrieval = mixed_scenario(compact_db, monkeypatch, fail_dialogue=True)
    result = asyncio.run(scenario.respond(request))
    assert "暂时无法" in result.answer and "匹配的知识依据" in result.answer
    assert [r["status"] for r in result.metrics["task_outcomes"]] == ["unavailable", "insufficient_evidence"]
    assert retrieval.await_count == 1 and not scenario.saves


def test_failed_recap_keeps_independent_successful_plan(compact_db, monkeypatch):
    from app.services.agent import multi_agent_graph as graph_module
    scenario, request, *_ = mixed_scenario(compact_db, monkeypatch, fail_dialogue=True)
    monkeypatch.setattr(graph_module, "assess_rag_evidence", lambda *a, **kw: SimpleNamespace(answerable=True))
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_message", AsyncMock(return_value=plan_message("每周两次的通用示例")))
    result = asyncio.run(scenario.respond(request))
    assert "暂时无法" in result.answer and "每周两次的通用示例" in result.answer
    assert [r["status"] for r in result.metrics["task_outcomes"]] == ["unavailable", "complete"]


def test_saved_record_and_recap_survive_plan_failure_and_replay_once(compact_db, monkeypatch):
    record, recap, plan = "记录今天卧推40公斤3组12次", "回顾刚记录的训练", "给一个通用周计划示例"
    source = "；".join((record, recap, plan))
    scenario = RetryScenario(monkeypatch, compact_db, workout_records=lambda _: [{"exercise": "卧推", "weight_kg": 40, "sets": 3, "reps": 12}])
    async def parse(query):
        items = [task(record, "workout_record", "record"), {**task(recap), "reply_mode": "conversation"}, task(plan, "training_plan")]
        value = semantics.validate_understanding(query, {"tasks": items})
        semantics.remember_understanding(query, value)
        parsed = {"tasks": [{k: t[k] for k in ("task_type", "action", "text", "is_negated", "is_hypothetical", "is_third_party")} for t in value["tasks"]]}
        parsed["tasks"][0]["workouts"] = [{"exercise": "卧推", "weight_kg": 40, "sets": 3, "reps": 12}]
        return parsed
    monkeypatch.setattr(scenario.agent.graph, "parse_input", AsyncMock(side_effect=parse))
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", lambda: True)
    async def structured(messages, schema, *, name, **kwargs):
        if name == "return_request_understanding":
            return schema.model_validate(semantic_envelope(messages))
        if name == "return_plan_information":
            return {"scope": "generic", "scope_quote": plan, "facts": []}
        if name == "return_dialogue_answer":
            context = json.dumps(messages, ensure_ascii=False)
            assert "已记录" in context and "40" in context
            return {"answer": "刚才记录了卧推40公斤，3组12次。", "requires_grounding": False}
        raise AssertionError(name)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", AsyncMock(side_effect=structured))
    monkeypatch.setattr(scenario.agent.graph, "_readonly_retrieval", AsyncMock(return_value=[]))
    request = ChatRequest(message=source)
    first = asyncio.run(scenario.respond(request))
    count = scenario.agent.graph.llm.complete_structured.await_count
    replay = asyncio.run(scenario.respond(request))
    assert "刚才记录了卧推40公斤" in first.answer and "匹配的知识依据" in first.answer
    assert [r["status"] for r in first.metrics["task_outcomes"]] == ["complete", "complete", "insufficient_evidence"]
    assert replay.replayed and first.answer == replay.answer and scenario.agent.graph.llm.complete_structured.await_count == count
    assert len(scenario.saves) == 1
    with compact_db() as s:
        assert s.execute(text("SELECT weight_kg,sets,reps FROM workout_records")).all() == [(40,3,12)]


@pytest.mark.parametrize("stream", [False, True])
def test_mixed_json_sse_replay_is_exact_and_zero_new_model(compact_db, monkeypatch, stream):
    from app.api import chat
    from app.main import app
    scenario, request, past, *_ = mixed_scenario(compact_db, monkeypatch)
    monkeypatch.setattr(chat, "agent", scenario.agent)
    def decode(wire, is_stream):
        assert wire.status_code == 200
        if not is_stream:
            return wire.json()
        frames = [json.loads(line[6:]) for line in wire.text.splitlines() if line.startswith("data: ")]
        return {**frames[-1], "answer": "".join(f.get("content", "") for f in frames)}
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            payload = request.model_dump(mode="json")
            first = decode(await client.post("/api/chat/stream" if stream else "/api/chat", json=payload), stream)
            calls = scenario.agent.graph.llm.complete_structured.await_count
            second = decode(await client.post("/api/chat" if stream else "/api/chat/stream", json=payload), not stream)
            assert scenario.agent.graph.llm.complete_structured.await_count == calls
            return first, second
    first, replay = asyncio.run(send())
    assert past in first["answer"] and replay["answer"] == first["answer"] and replay["replayed"]
    assert first["metrics"]["task_outcomes"] == replay["metrics"]["task_outcomes"]
    assert not scenario.saves


@pytest.mark.parametrize("quote", ["我对核桃过敏", "我的训练目标是减脂", "我家有一副哑铃", "我习惯早上训练"])
@pytest.mark.parametrize("purpose", ["general", "execution_status"])
def test_unrelated_memories_are_absent_from_every_answer_channel(quote, purpose):
    marker = "UNRELATED_PRIVATE_" + quote
    memory = {"profile": {"note": marker}, "long_memory": {"unrelated": {"text": marker}},
              "short_memory": [{"message_id": "other", "role": "user", "content": marker}],
              "retrieved_history": [{"turn_id": "old", "content": marker}],
              "semantic_memory": [{"text": marker, "category": "constraint", "speaker": "user"}],
              "memory_summary": marker}
    async def structured(messages, schema, **kwargs):
        assert marker not in json.dumps(messages, ensure_ascii=False)
        return {"answer": "不会保存。", "requires_grounding": False}
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(side_effect=structured))
    result = asyncio.run(dialogue.conversation_answer(llm, ChatRequest(message="这只是转述，不要保存。"), memory,
        {"reply_mode": "conversation", "reply_purpose": purpose}))
    assert result["answer"] == "不会保存。" and marker in memory["long_memory"]["unrelated"]["text"]


def test_explicitly_selected_fact_and_dialogue_keep_roles():
    memory = {"long_memory": {"wanted": {"text": "我只有弹力带"}, "other": {"text": "UNRELATED"}},
              "short_memory": [{"message_id": "u", "role": "user", "content": "给两个选项"},
                               {"message_id": "a", "role": "assistant", "content": "选项甲和乙"},
                               {"message_id": "x", "role": "user", "content": "UNRELATED"}]}
    async def structured(messages, schema, **kwargs):
        wire = json.dumps(messages, ensure_ascii=False)
        assert "我只有弹力带" in wire and "UNRELATED" not in wire
        assert [m["role"] for m in messages if m["content"] in {"给两个选项", "选项甲和乙"}] == ["user", "assistant"]
        return {"answer": "选择乙。", "requires_grounding": False}
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(side_effect=structured))
    result = asyncio.run(dialogue.conversation_answer(llm, ChatRequest(message="结合我现有器械选一个"), memory,
        {"reply_mode": "conversation", "dialogue_ids": ["u", "a"], "fact_fields": ["wanted"]}))
    assert result["answer"] == "选择乙。"


@pytest.mark.parametrize("selector", ["dialogue_ids", "references", "critical_history_ids", "critical_fact_fields"])
def test_recall_selection_uses_top_level_and_task_sources_consistently(selector):
    wanted, unrelated = "我家只有弹力带", "UNRELATED_PRIVATE"
    memory = {"short_memory": [{"message_id": "u", "role": "user", "content": wanted},
                               {"message_id": "other", "role": "user", "content": unrelated}],
              "long_memory": {"equipment": {"text": wanted, "source_message_id": "u"},
                              "other": {"text": unrelated, "source_message_id": "other"}}}
    source = "我有哪些器械"
    raw = {"tasks": [task(source)], "reply_mode": "conversation", "reply_purpose": "fact_recall"}
    if selector == "references":
        raw[selector] = [{"source_text": source, "query": source, "evidence_ids": ["u"]}]
    else:
        raw[selector] = ["equipment" if selector == "critical_fact_fields" else "u"]
    value = semantics.validate_understanding(source, raw, context=memory)
    async def structured(messages, schema, **kwargs):
        assert unrelated not in json.dumps(messages, ensure_ascii=False)
        return {"evidence": [{"source_id": "fact:equipment" if selector == "critical_fact_fields" else "history:u", "quote": wanted}]}
    llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(side_effect=structured))
    result = asyncio.run(dialogue.conversation_answer(llm, ChatRequest(message=source), memory, value))
    assert wanted in result["answer"]


@pytest.mark.parametrize("kind,action", [("workout_record", "record"), ("nutrition_record", "confirm"), ("safety_triage", "question")])
def test_task_conversation_never_bypasses_writes_or_medical_assessment(kind, action):
    source = "记录今天训练卧推40公斤3组12次" if kind != "safety_triage" else "我的膝盖疼怎么办"
    with pytest.raises(semantics.SemanticValidationError):
        semantics.validate_understanding(source, {"tasks": [{**task(source, kind, action), "reply_mode": "conversation"}]})


@pytest.mark.parametrize("field,value", [("context_ids", ["foreign"]), ("fact_fields", ["foreign"])])
def test_task_sources_must_belong_to_supplied_context(field, value):
    source = "整理已有内容"
    with pytest.raises(semantics.SemanticValidationError):
        semantics.validate_understanding(source, {"tasks": [{**task(source), "reply_mode": "conversation", field: value}]})


def test_raw_parser_metadata_cannot_select_read_worker():
    from app.services.agent.task_planner import plan_tasks_from_llm
    source = "帮我分析训练"
    parsed = {"tasks": [{**task(source), "reply_mode": "conversation", "dialogue": {"reply_mode": "conversation"}}]}
    result = plan_tasks_from_llm(source, parsed)
    assert result and "dialogue" not in result[0]["payload"]


def test_source_occurrence_survives_write_sort_and_identical_read_text():
    from app.services.agent.planner_validation import validate_planner_output
    from app.services.agent.task_planner import plan_tasks_from_llm
    read, write = "回顾刚才安排", "记录今天卧推40公斤3组12次"
    source = "；".join((read, write, read))
    items = [{**task(read), "reply_mode": "conversation"}, task(write, "workout_record", "record"), task(read)]
    value = semantics.validate_understanding(source, {"tasks": items})
    semantics.remember_understanding(source, value)
    parsed = {"tasks": [{k: t[k] for k in ("task_type", "action", "text", "is_negated", "is_hypothetical", "is_third_party")} for t in value["tasks"]]}
    parsed["tasks"][1]["workouts"] = [{"exercise": "卧推", "weight_kg": 40, "sets": 3, "reps": 12}]
    checked = validate_planner_output(source, parsed)
    assert not checked.fallback_required and checked.tasks[0]["action"] == "record"
    planned = plan_tasks_from_llm(source, {"authorization_validated": True, "tasks": checked.tasks})
    reads = [t for t in planned if t["text"] == read]
    assert reads[0]["payload"]["source_start"] == 0 and "dialogue" in reads[0]["payload"]
    assert reads[0]["depends_on"] == [] and "dialogue" not in reads[1]["payload"]
    assert reads[1]["payload"]["source_start"] == source.rfind(read)
    result = FitnessMultiAgentGraph._finalize_node({"planned_tasks": planned, "worker_results": [
        {"task_id": t["id"], "task_type": t["task_type"], "answer": str(t["payload"]["source_start"]), "status": "complete"} for t in planned]})
    assert result["final_answer"].split("\n\n") == ["0", str(source.index(write)), str(source.rfind(read))]


def test_trusted_task_span_must_still_match_actual_source():
    from app.services.agent.task_planner import plan_tasks_from_llm
    source = "回顾刚才安排"
    assert plan_tasks_from_llm(source, {"authorization_validated": True, "tasks": [
        {**task(source), "source_start": 1, "source_end": len(source)}]}) is None


def test_grounded_advice_retains_selected_safety_context(monkeypatch):
    graph = FitnessMultiAgentGraph()
    memory = {"allergy": {"text": "我对核桃过敏"}, "injury": {"text": "我的膝盖受过伤"},
              "unrelated": {"text": "UNRELATED_PRIVATE"}}
    worker = SimpleNamespace(ainvoke=AsyncMock(return_value={"result": {"status": "complete", "answer": "建议", "model_used": False}}))
    graph.workers["nutrition"] = worker
    task = {"id": "t", "task_type": "nutrition_advice", "action": "question", "text": "饮食怎么安排",
            "payload": {"semantic": {"fact_fields": [], "critical_fact_fields": ["allergy", "injury"]}}}
    asyncio.run(graph._run_worker("nutrition", {"user_id": "demo-user", "raw_message": "饮食怎么安排",
        "long_memory": memory, "current_task": task}))
    sent = worker.ainvoke.await_args.args[0]["long_memory"]
    assert set(sent) == {"allergy", "injury"}
    assert "unrelated" not in sent


def test_selected_fact_visibility_is_rechecked_after_generation(compact_db):
    save_memory("我家有弹力带", "equipment")
    memory = {"long_memory": compact.current_facts()}
    field, fact = next(iter(memory["long_memory"].items()))
    request = ChatRequest(message="按我现有器械解释刚才的话", conversation_id="fact-scope")
    tid, _ = compact.begin_turn(request)
    token = compact.current_turn.set(tid)
    async def structured(messages, schema, **kwargs):
        with compact_db() as s, s.begin():
            s.get(MemoryTurn, fact["source_turn_id"]).hidden_at = compact.clock.utc_now()
        return {"answer": "我家有弹力带", "requires_grounding": False}
    try:
        llm = SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(side_effect=structured))
        result = asyncio.run(dialogue.conversation_answer(llm, request, memory,
            {"reply_mode": "conversation", "fact_fields": [field]}))
        assert "弹力带" not in result["answer"] and result["status"] == "unavailable"
    finally:
        compact.current_turn.reset(token)


def test_receipt_punctuation_does_not_rewrite_fact_source(compact_db):
    tid = save_memory("我家只有弹力带。", "equipment")
    with compact_db() as s:
        row = s.get(MemoryTurn, tid)
        assert row.response["answer"] == row.messages[-1]["content"] == "已记住：我家只有弹力带。"
        assert s.scalar(select(UserFact)).source_quote == "我家只有弹力带。"
