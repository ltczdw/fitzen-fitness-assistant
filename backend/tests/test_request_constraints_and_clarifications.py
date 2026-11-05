"""General source binding and real HTTP draft metadata, all providers mocked."""
import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import text

from app.schemas.chat import ChatRequest
from app.services.agent import semantics, record_clarifications as drafts
from evals.v2.safety import deny_network


def task(kind, text, *, mode="grounded", purpose="general", action="question"):
    return dict(task_type=kind, action=action, text=text, reply_mode=mode, reply_purpose=purpose,
                is_negated=False, is_hypothetical=False, is_third_party=False)


@pytest.mark.parametrize("prefix,first,last,kind", [
    ("只建议，不保存。家里只有弹力带。", "先整理已有训练记录，", "再给新手每周两次每次15分钟的增肌周计划。", "training_plan"),
    ("我对牛奶过敏。不要记录。", "先复述之前的午餐，", "再给一个不含奶的晚餐搭配建议。", "nutrition_advice"),
    ("只用表格回答，我只能徒手。", "先回顾最近的动作，", "再给减脂初学者每天10分钟的训练安排。", "training_plan"),
])
def test_separated_conditions_keep_contiguous_source_and_independent_goals(prefix, first, last, kind):
    source = prefix + first + last
    raw = {"tasks": [task("training_qa", prefix + first, mode="conversation"), task(kind, last)]}
    with deny_network():
        checked = semantics.validate_understanding(source, raw)
    assert len(checked["validated_tasks"]) == 2
    assert all(source[t["source_start"]:t["source_end"]] == t["text"] for t in checked["validated_tasks"])
    broken = copy.deepcopy(raw)
    broken["tasks"][1]["text"] = prefix + last
    with pytest.raises(semantics.SemanticValidationError) as error:
        semantics.validate_understanding(source, broken)
    assert "删除中间部分后拼接" in error.value.validation_errors[0]["hint"]
    omitted = {"tasks": [task(kind, last)]}
    # A sufficiently long discarded prefix cannot disappear through correction.
    with pytest.raises(semantics.SemanticValidationError):
        semantics.validate_understanding(prefix * 5 + last, omitted)


@pytest.mark.parametrize("role,source,target", [
    ("训练方案", "仅建议。家里只有弹力带；先回顾记录，再给每周两次15分钟的训练。", "再给每周两次15分钟的训练。"),
    ("饮食营养方案", "我对牛奶过敏。先列出午餐，再给不含奶的晚餐建议。", "再给不含奶的晚餐建议。"),
])
def test_planning_prompt_receives_full_current_conditions_without_changing_task(role, source, target):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    state = {"raw_message": source, "task_text": target, "task_context": []}
    with deny_network():
        messages = FitnessMultiAgentGraph()._planning_prompt(role, state, [])
    assert source in json.dumps(messages, ensure_ascii=False)
    assert state["task_context"] == []
    assert any(m["role"] == "user" and target in m["content"] for m in messages)


@pytest.mark.parametrize("kind,source,proposal,expected", [
    ("nutrition_record", "请记录今天晚餐吃的豆腐。", {"kind": "nutrition", "source_quote": "豆腐", "name": "豆腐",
      "meal_type": "晚餐", "meal_quote": "晚餐"}, "food_amount"),
    ("workout_record", "请记录今天卧推4组8次。", {"kind": "workout", "source_quote": "卧推4组8次", "exercise": "卧推",
      "sets": 4, "reps": 8}, "weight_kg"),
])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("router_asks", [False, True])
def test_http_first_question_preview_and_replay_keep_metadata(compact_db, monkeypatch, kind, source, proposal, expected, stream, router_asks):
    from app.api import chat
    from app.main import app
    from test_retry_operations import RetryScenario
    scenario = RetryScenario(monkeypatch, compact_db)
    agent = scenario.agent
    monkeypatch.setattr(chat, "agent", agent)
    async def parse(query):
        raw = {"tasks": [task(kind, query, action="record")],
               "safety": {"level": "normal", "evidence": "", "categories": [],
                          "reason": "scripted normal"},
               "needs_clarification": router_asks, "clarification": "请补充数据"}
        checked = semantics.validate_understanding(query, raw)
        semantics.remember_understanding(query, checked)
        return {"tasks": [{k: t[k] for k in ("task_type", "action", "text", *drafts.FLAGS)} for t in checked["tasks"]]}
    monkeypatch.setattr(agent.graph, "parse_input", AsyncMock(side_effect=parse))
    async def structured(messages, schema, *, name, **kw):
        if name == "return_request_understanding":
            payload = json.loads(messages[-1]["content"])
            query = payload["query"]
            return schema.model_validate({
                "tasks": [task(kind, query, action="record")],
                "safety": {"level": "normal", "evidence": "", "categories": [],
                           "reason": "scripted normal"},
            })
        if name == "return_record_proposals":
            return schema.model_validate({"records": [proposal]})
        assert name == "return_record_reply"
        payload = json.loads(messages[-1]["content"])
        return schema.model_validate({"draft_id": payload["draft_id"], "action": "unclear", "accepts_all": False,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False})
    calls = AsyncMock(side_effect=structured)
    monkeypatch.setattr(agent.graph.llm, "configured", lambda: True)
    monkeypatch.setattr(agent.graph.llm, "complete_structured", calls)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
            async def send(payload, streaming):
                wire = await api.post("/api/chat/stream" if streaming else "/api/chat", json=payload)
                assert wire.status_code == 200
                if not streaming:
                    return wire.json()
                frames = [json.loads(line[6:]) for line in wire.text.splitlines() if line.startswith("data: ")]
                return {**frames[-1], "answer": "".join(f.get("content", "") for f in frames)}
            first = await send(ChatRequest(message=source, conversation_id="draft-metadata").model_dump(), stream)
            assert first["clarifications"][0]["task_type"] == kind
            assert [f["name"] for f in first["clarifications"][0]["fields"]] == [expected]
            assert first["answer"] == first["clarifications"][0]["question"]
            followup = ChatRequest(message="还没想好", conversation_id="draft-metadata").model_dump()
            result = await send(followup, not stream)
            assert result["clarifications"][0]["fields"] == first["clarifications"][0]["fields"]
            count = calls.await_count
            replay = await send(followup, stream)
            assert replay["clarifications"] == result["clarifications"] and calls.await_count == count
        with compact_db() as s:
            assert s.scalar(text("SELECT COUNT(*) FROM workout_records")) == 0
            assert s.scalar(text("SELECT COUNT(*) FROM user_food_log")) == 0
    with deny_network():
        asyncio.run(run())


def test_completed_preview_and_service_failure_do_not_invent_missing_fields():
    record = dict(kind="workout", exercise="卧推", weight_kg=10, sets=3, reps=8)
    question = "以上全部正确并保存吗？"
    result = drafts.question_details([record], question)
    assert result[0]["category"] == "ambiguous_reference" and result[0]["fields"] == []
    unavailable = drafts.unavailable_response()
    assert unavailable.clarifications[0].category == "service_unavailable"
    assert unavailable.clarifications[0].fields == []


@pytest.mark.parametrize("failure", [False, True])
def test_plan_continuation_retains_bound_fields_or_reports_service_failure(compact_db, failure):
    from test_task_information import draft_fixture, begin
    from app.services.agent import planning_questions as planning
    from app.services.memory import compact
    initial, _ = draft_fixture(compact_db)
    request = ChatRequest(message="还没想好", conversation_id=initial.conversation_id)
    tid = begin(request)
    async def decide(messages, schema, **kwargs):
        if failure:
            raise ValueError("synthetic model failure")
        data = json.loads(messages[-1]["content"])
        return schema.model_validate({"draft_id": data["draft_id"], "action": "unclear"})
    token = compact.current_turn.set(tid)
    try:
        with deny_network():
            result = asyncio.run(planning.continue_request(SimpleNamespace(complete_structured=decide), request))
        assert result["clarifications"]
        if failure:
            assert result["clarifications"][0]["category"] == "service_unavailable"
            assert result["clarifications"][0]["fields"] == []
        else:
            assert [f["name"] for f in result["clarifications"][0]["fields"]] == ["goal", "experience"]
            assert result["answer"] == result["clarifications"][0]["question"]
    finally:
        compact.current_turn.reset(token)
