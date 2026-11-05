"""Complete training time allocations, actual worker and HTTP/replay boundaries."""
import asyncio
import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import text

from app.services.agent.training_timing import duration_limit_seconds, weekly_sessions, render_timed_plan
from evals.v2.safety import deny_network


@pytest.fixture(autouse=True)
def offline():
    with deny_network():
        yield


@pytest.mark.parametrize("quote, seconds", [
    ("20分钟", 1200), ("十五分钟", 900), ("半小时", 1800),
    ("一个半小时", 5400), ("1小时20分钟30秒", 4830), ("22.5 minutes", 1350),
    ("３０分", 1800), ("15至20分钟", 1200), ("15分钟到半小时", 1800),
    ("三十秒", 30), ("1.5h", 5400), ("0.1分钟", 6), ("两分半", 150), (".5小时", 1800),
])
def test_explicit_grounded_duration_units_and_ranges(quote, seconds):
    assert duration_limit_seconds(quote) == seconds


@pytest.mark.parametrize("quote", ["", "随便", "一会儿", "-20分钟", "0分钟", "25小时",
    "15或20分钟", "不是15分钟而是20分钟", "20分钟另加5分钟", "20分钟不包括收尾", "20分钟20分钟", "一百二分钟", "十五二分钟",
    "每次总共20分钟，其中含30秒收尾", "每次最多1小时，理想20分钟", "每次一点五小时", "半个半小时", "一半小时", "1.5个半小时"])
def test_unknown_or_conflicting_duration_never_uses_a_default(quote):
    assert duration_limit_seconds(quote) is None


@pytest.mark.parametrize("quote, count", [("一周两练", 2), ("每周三回", 3), ("5", 5), ("每周2天", 2),
    ("不确定", None), ("每周2天每天2次", None), ("1.5次", None), ("0次", None), ("每周一点五次", None), ("每周2点5次", None)])
def test_weekly_count_uses_grounded_quantity(quote, count):
    assert weekly_sessions(quote) == count


def plan_payload(count=2):
    session = dict(warmup=dict(seconds=120, activity="关节活动和轻负荷热身"),
        main=dict(seconds=660, exercises=[dict(name="哑铃深蹲", sets=2, reps=10), dict(name="哑铃划船", sets=2, reps=10)]),
        breaks=dict(seconds=300),
        cooldown=dict(seconds=120, activity="轻缓活动与放松"),
        time_consistency="consistent")
    return dict(sessions=[dict(label=f"训练{i+1}", **copy.deepcopy(session)) for i in range(count)])


def raw_plan(count=2):
    return json.dumps(plan_payload(count), ensure_ascii=False)


def test_renderer_includes_every_phase_and_real_sum_once_per_session():
    answer = render_timed_plan(raw_plan(), limit_seconds=1200, expected_sessions=2)
    assert answer.count("| 收尾／放松 | 2分钟 |") == 2
    assert answer.count("合计：20分钟，不超过20分钟。") == 2
    assert "累计40分钟" in answer and "用时为建议估算，并非实测" in answer
    assert "穿插进行" in answer and "哑铃深蹲" in answer and '"seconds"' not in answer


@pytest.mark.parametrize("fault", ["missing", "zero", "negative", "bool", "fraction", "string", "over", "extra", "duplicate_day", "wrong_count", "empty_activity"])
def test_incomplete_or_over_limit_output_is_rejected_without_filling_a_phase(fault):
    value = plan_payload()
    session = value["sessions"][0]
    if fault == "missing":
        del session["cooldown"]
    elif fault in {"zero", "negative", "bool", "fraction", "string"}:
        session["cooldown"]["seconds"] = {"zero": 0, "negative": -1, "bool": True, "fraction": 1.5, "string": "120"}[fault]
    elif fault == "over":
        session["cooldown"]["seconds"] += 1
    elif fault == "extra":
        session["extra_cooldown"] = dict(seconds=120, activity="加在总量外")
    elif fault == "duplicate_day":
        value["sessions"][1]["label"] = session["label"]
    elif fault == "wrong_count":
        value["sessions"].pop()
    elif fault == "empty_activity":
        session["cooldown"]["activity"] = "  "
    with pytest.raises(ValueError):
        render_timed_plan(json.dumps(value), limit_seconds=1200, expected_sessions=2)


def test_duplicate_field_or_non_json_prose_cannot_bypass_the_contract():
    duplicate = raw_plan().replace('"seconds": 120', '"seconds": 1, "seconds": 120', 1)
    for raw in (duplicate, "热身3分钟，主体17分钟，练完再放松", "```json\n" + raw_plan() + "\n```"):
        with pytest.raises(ValueError):
            render_timed_plan(raw, limit_seconds=1200, expected_sessions=2)


@pytest.mark.parametrize("field", ["introduction", "notes", "main"])
def test_unknown_top_level_or_phase_fields_are_rejected_by_schema(field):
    value = plan_payload()
    if field in {"introduction", "notes"}:
        value[field] = "另外再做五分钟"
    else:
        value["sessions"][0][field]["activity"] = "额外进行5 min放松"
    with pytest.raises(ValueError):
        render_timed_plan(json.dumps(value), limit_seconds=1200, expected_sessions=2)


@pytest.mark.parametrize("prose", ["额外拉伸15分钟", "add 15 minutes outside the limit",
                                   "再活动20分钟", "不要持续保持10分钟，只做轻缓拉伸"])
def test_model_owned_prose_is_rendered_without_local_reinterpretation(prose):
    value = plan_payload()
    value["sessions"][0]["cooldown"]["activity"] = prose
    answer = render_timed_plan(json.dumps(value), limit_seconds=1200, expected_sessions=2)
    assert prose in answer


def test_model_time_consistency_is_required_and_literal():
    value = plan_payload()
    del value["sessions"][0]["time_consistency"]
    with pytest.raises(ValueError):
        render_timed_plan(json.dumps(value), limit_seconds=1200, expected_sessions=2)
    value = plan_payload()
    value["sessions"][0]["time_consistency"] = "inconsistent"
    with pytest.raises(ValueError):
        render_timed_plan(json.dumps(value), limit_seconds=1200, expected_sessions=2)


@pytest.mark.parametrize("prose", ["若干秒", "放松片刻", "轻缓拉伸一会儿"])
def test_indefinite_prose_time_is_allowed_and_rendered_from_structured_seconds(prose):
    value = plan_payload()
    value["sessions"][0]["cooldown"]["activity"] = prose
    answer = render_timed_plan(json.dumps(value), limit_seconds=1200, expected_sessions=2)
    assert "| 收尾／放松 | 2分钟 |" in answer


def test_timed_exercise_is_rendered_from_fields_and_cannot_exceed_active_budget():
    value = plan_payload(1)
    value["sessions"][0]["main"]["exercises"] = [dict(name="平板支撑", sets=3, hold_seconds=30)]
    raw = json.dumps(value)
    assert "平板支撑：3组，每组30秒" in render_timed_plan(raw, limit_seconds=1200, expected_sessions=1)
    value["sessions"][0]["main"]["exercises"][0]["hold_seconds"] = 300
    with pytest.raises(ValueError, match="timed_exercises_exceed_main_phase"):
        render_timed_plan(json.dumps(value), limit_seconds=1200, expected_sessions=1)


def hits():
    from app.services.rag.retrieval import RetrievedChunk
    return [RetrievedChunk("训练", "循序渐进，规律训练。", .95, "official",
        {"rerank_score": .95, "rerank_status": "model"}, "synthetic")]


def install_normal_semantics(monkeypatch, graph):
    async def semantic(messages, schema, *, name=None, **kwargs):
        if name != "return_request_understanding":
            raise AssertionError(name)
        payload = json.loads(messages[-1]["content"])
        source = str(payload.get("query") or "")
        return schema.model_validate({
            "tasks": [{"task_type": "training_qa", "action": "question", "text": source,
                       "is_negated": False, "is_hypothetical": False,
                       "is_third_party": False}],
            "safety": {"level": "normal", "evidence": "", "categories": [],
                       "reason": "scripted normal"},
        })
    monkeypatch.setattr(graph.llm, "configured", lambda: True)
    monkeypatch.setattr(graph.llm, "complete_structured", semantic)


def wire_worker(monkeypatch, graph, raw, *, quote="每次20分钟", scope="weekly"):
    from app.services.agent.training_timing import duration_limit_seconds, weekly_sessions
    import re
    match = re.search(r"[0-9零〇一二两三四五六七八九十百半.]+(?:小时|钟头|分钟|分|秒|minutes?|mins?|min|hours?|hrs?|h|seconds?|secs?|s)", quote)
    quantity = match.group(0) if match else ""
    available_quote = "每周两次"
    information = dict(
        scope=scope,
        known=dict(goal="增肌", experience="新手", equipment="哑铃",
                   session_time=quote, available_days=available_quote),
        session_time_seconds=duration_limit_seconds(quantity) if quantity else None,
        available_days_count=weekly_sessions(available_quote),
        phase_limits=[],
    )
    monkeypatch.setattr("app.services.agent.task_information.check_plan_information", AsyncMock(return_value=(information, None)))
    retrieval = AsyncMock(return_value=hits())
    monkeypatch.setattr(graph, "_readonly_retrieval", retrieval)
    mcp = AsyncMock(return_value=None)
    monkeypatch.setattr(graph.mcp, "complete_with_tools", mcp)
    plain = AsyncMock(return_value=raw)
    monkeypatch.setattr(graph.llm, "complete", plain)
    forced = AsyncMock(return_value={"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "return_timed_training_plan", "arguments": raw}}]})
    monkeypatch.setattr(graph.llm, "complete_message", forced)
    complete = forced if quote else plain
    return complete, retrieval, mcp


@pytest.mark.parametrize("source", [
    "先回顾卧推，再给新手哑铃增肌周计划，每周两次每次20分钟，不保存。",
    "不保存；只有哑铃，新手增肌每周两次每次20分钟。先回顾卧推，再排周计划。",
    "先回顾卧推。我是增肌新手，只有哑铃每周两次每次20分钟。再给周计划，不保存。",
])
def test_actual_worker_checks_and_renders_the_contract_in_any_condition_order(monkeypatch, source):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    graph = FitnessMultiAgentGraph()
    complete, _, mcp = wire_worker(monkeypatch, graph, raw_plan())
    state = dict(user_id="demo-user", raw_message=source, task_text="再给周计划", task_type="training_plan", task_action="question")
    result = asyncio.run(graph.workers["plan_adjust"].ainvoke(state))["result"]
    assert result["status"] == "complete" and result["model_used"]
    assert result["answer"].count("| 收尾／放松 | 2分钟 |") == 2
    assert complete.await_count == 1 and mcp.await_count == 0
    assert complete.call_args.kwargs["tool_choice"]["function"]["name"] == "return_timed_training_plan"
    messages = complete.call_args.args[0]
    assert source in json.dumps(messages, ensure_ascii=False)
    assert '"time_limit_seconds": 1200' in json.dumps(messages, ensure_ascii=False).replace('\\"', '"')
    assert state.get("task_context") is None


@pytest.mark.parametrize("raw", ["旧的18至20分钟自由文本，未提收尾", '{"sessions": []}'])
def test_worker_bounds_correction_then_fails_an_invalid_timing_plan(monkeypatch, raw):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    graph = FitnessMultiAgentGraph()
    complete, _, _ = wire_worker(monkeypatch, graph, raw)
    value = asyncio.run(graph.workers["plan_adjust"].ainvoke(dict(user_id="demo-user", raw_message="训练安排")))["result"]
    assert value["status"] == "error" and value["model_used"]
    assert "时间分配未通过校验" in value["answer"] and raw not in value["answer"]
    # One bounded structured correction is allowed; the same invalid result on
    # the second attempt fails closed instead of adding a third call.
    assert complete.await_count == 2


def test_ambiguous_limit_asks_only_for_missing_field_before_retrieval(monkeypatch):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    graph = FitnessMultiAgentGraph()
    complete, retrieval, _ = wire_worker(monkeypatch, graph, raw_plan(), quote="随便练一会儿")
    value = asyncio.run(graph.workers["plan_adjust"].ainvoke(dict(raw_message="训练安排")))["result"]
    assert value["status"] == "needs_confirmation"
    assert value["clarification"]["fields"] == []
    assert value["clarification"]["category"] == "service_unavailable"
    assert "每周" not in value["answer"]
    complete.assert_not_awaited(); retrieval.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_saved_write_then_plan_validation_and_opposite_transport_replay(compact_db, monkeypatch, stream, valid):
    from app.api import chat
    from app.main import app
    from app.schemas.chat import ChatRequest
    from app.services.agent.execution_state import PlannerDecision
    from app.services.agent.intent import IntentResult
    from test_retry_operations import RetryScenario
    scenario = RetryScenario(monkeypatch, compact_db, workout_records=lambda message: [
        dict(exercise="卧推", weight_kg=40, sets=3, reps=8)])
    agent = scenario.agent
    monkeypatch.setattr(chat, "agent", agent)
    write, plan = "记录卧推40kg3组8次", "再给新手哑铃增肌每周两次每次20分钟周计划"
    specs = [dict(id="saved", task_type="workout_record", worker="workout_record", action="record", text=write,
        side_effect=True, depends_on=[], payload={"write_authorization": "authorized"}),
        dict(id="plan", task_type="training_plan", worker="plan_adjust", action="question", text=plan, depends_on=["saved"])]
    monkeypatch.setattr(agent, "_route", AsyncMock(return_value=PlannerDecision(IntentResult("plan_adjust", 1), {}, tuple(specs))))
    payload = plan_payload()
    if not valid:
        del payload["sessions"][0]["cooldown"]
    complete, _, _ = wire_worker(monkeypatch, agent.graph, json.dumps(payload))
    install_normal_semantics(monkeypatch, agent.graph)
    request = ChatRequest(message=write + "，" + plan, conversation_id="timed-plan-http")
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
            async def send(streaming):
                wire = await api.post("/api/chat/stream" if streaming else "/api/chat", json=request.model_dump())
                assert wire.status_code == 200
                if not streaming:
                    return wire.json()
                frames = [json.loads(line[6:]) for line in wire.text.splitlines() if line.startswith("data: ")]
                return {**frames[-1], "answer": "".join(frame.get("content", "") for frame in frames)}
            first = await send(stream)
            assert [(o["task_id"], o["status"]) for o in first["metrics"]["task_outcomes"]] == [("saved", "complete"), ("plan", "complete" if valid else "error")]
            assert "已记录" in first["answer"]
            assert ("| 收尾／放松 |" in first["answer"]) is valid
            calls = complete.await_count
            replay = await send(not stream)
            assert replay["replayed"] and replay["answer"] == first["answer"]
            assert complete.await_count == calls and len(scenario.saves) == 1
        with compact_db() as session:
            assert session.scalar(text("SELECT COUNT(*) FROM workout_records")) == 1
            assert session.scalar(text("SELECT COUNT(*) FROM workout_session_sets")) == 3
    asyncio.run(run())
