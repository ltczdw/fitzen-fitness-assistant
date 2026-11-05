"""Numeric source binding and direct-save regression; all model calls mocked."""
import asyncio
import copy
import json
from unittest.mock import AsyncMock
import pytest
from sqlalchemy import text
from app.schemas.chat import ChatRequest
from app.services.agent.workout_extractor import (
    preprocess_workout,
    validate_workout_candidate,
    validate_workout_records_against_source,
)
from test_retry_operations import RetryScenario
from tests.plan_fixture_support import plan_message


def workout(exercise="哑铃卧推", weight=10, sets=4, reps=10):
    return {"exercise": exercise, "weight_kg": weight, "sets": sets, "reps": reps}


@pytest.mark.parametrize("notation", ["10kg*10次*4", "10kg×10次×4", "10kg*10*4", "10公斤＊10次＊4组", "10kg×4组×10次", "10kg 4x10", "10kg，4组每组10", "10*10*4", "10kg*10,4组", "10公斤×10，四组"])
def test_source_notation_keeps_model_values_including_trailing_sets(notation):
    candidate = workout()
    kept, dropped = validate_workout_records_against_source([candidate], "记录哑铃卧推" + notation)
    assert kept == [candidate] and not dropped


def test_screenshot_two_actions_are_sourced_without_confirmation():
    candidates = [workout(), workout("直臂下压", 20, 5)]
    kept, dropped = validate_workout_records_against_source(candidates,
        "帮我记录哑铃卧推10kg*10次*4，直臂下压20kg*10次*5")
    assert kept == candidates and not dropped


@pytest.mark.parametrize("source,candidates", [
    ("帮我记录，哑铃弯举10kg*10*4，引体向上自重*10*4",
     [workout("哑铃弯举", 10, 4, 10), workout("引体向上", 0, 4, 10)]),
    ("记录引体向上自重*10次*4", [workout("引体向上", 0, 4, 10)]),
    ("记录双杠臂屈伸徒手*10*4", [workout("双杠臂屈伸", 0, 4, 10)]),
    ("记录引体向上自重*4x10", [workout("引体向上", 0, 10, 4)]),
])
def test_bodyweight_notation_is_a_complete_three_factor_chain(source, candidates):
    kept, dropped = validate_workout_records_against_source(candidates, source)
    assert kept == candidates and not dropped


def test_bodyweight_notation_rejects_reversed_counts():
    candidate = workout("引体向上", 0, 10, 4)
    kept, dropped = validate_workout_records_against_source([candidate], "记录引体向上自重*10*4")
    assert not kept and dropped[0]["fields"] == ["sets", "reps"]


def candidate_state(candidate, message):
    return validate_workout_candidate(
        candidate, message,
        {"normalized_text": message, "inferred_fields": [], "number_mentions": []})


def test_explicit_bodyweight_never_asks_for_a_weight():
    message = "帮我记录，哑铃弯举10kg*10*4，引体向上自重*10*4"
    parsed = candidate_state(
        {"exercise": "引体向上", "weight_kg": None, "sets": 4, "reps": 10}, message)
    assert parsed.status == "valid"
    assert (parsed.record["weight_kg"], parsed.record["sets"], parsed.record["reps"]) == (0.0, 4, 10)
    assert "weight_kg" in parsed.inferred_fields


def test_bodyweight_marker_does_not_fill_another_actions_missing_weight():
    message = "帮我记录，哑铃弯举*10*4，引体向上自重*10*4"
    parsed = candidate_state(
        {"exercise": "哑铃弯举", "weight_kg": None, "sets": 4, "reps": 10}, message)
    assert parsed.status == "incomplete" and parsed.missing_fields == ("weight_kg",)


def test_rule_parser_accepts_explicit_self_weight_as_zero():
    parsed = preprocess_workout("引体向上自重*10*4")
    assert parsed.status == "valid" and parsed.record["weight_kg"] == 0.0


@pytest.mark.parametrize("names", [("自定义推举甲", "自定义划船乙"), ("蜘蛛弯举", "绳索面拉"), ("直臂下压", "哑铃卧推"), ("TRX反向划船", "Landmine Press")])
def test_each_reported_literal_name_gets_its_own_source_scope(names):
    candidates = [workout(names[0]), workout(names[1], 20, 5)]
    source = f"记录{names[0]}10kg*10次*4，{names[1]}20kg*10次*5"
    kept, dropped = validate_workout_records_against_source(candidates, source)
    assert kept == candidates and not dropped


@pytest.mark.parametrize("field,value", [("weight_kg", 20), ("sets", 5), ("reps", 7)])
def test_dynamic_actions_still_cannot_borrow_each_others_values(field, value):
    first, second = workout("自定义动作甲"), workout("自定义动作乙", 20, 5, 7)
    first[field] = value
    kept, dropped = validate_workout_records_against_source([first, second],
        "记录自定义动作甲10kg*10次*4，自定义动作乙20kg*7次*5")
    assert kept == [second] and len(dropped) == 1 and field in dropped[0]["fields"]


@pytest.mark.parametrize("source,candidate", [
    ("记录直臂下压20kg*10次", workout("直臂下压", 20, 5)),
    ("记录直臂下压20kg*10次*5", workout("直臂下压", 20, 10, 5)),
    ("记录直臂下压20kg*10次*5", workout("绳索直臂下压", 20, 5)),
    ("记录哑铃卧推10kg，3.5组10次", workout(sets=3)),
    ("记录哑铃卧推10kg*10次*4*9", workout()),
    ("记录哑铃卧推-10kg，4组10次", workout()),
    ("记录哑铃卧推10kg，4组-10次", workout()),
    ("记录哑铃卧推10kg，4组10次", workout(weight=4)),
])
def test_incomplete_invented_ambiguous_or_invalid_values_remain_blocked(source, candidate):
    kept, dropped = validate_workout_records_against_source([candidate], source)
    assert not kept and dropped


def test_same_source_occurrence_cannot_be_consumed_twice():
    candidate = workout("直臂下压", 20, 5)
    kept, dropped = validate_workout_records_against_source([candidate, copy.deepcopy(candidate)], "记录直臂下压20kg*10次*5")
    assert len(kept) == len(dropped) == 1


def test_numbered_set_pairs_still_must_match_in_order():
    source = "记录直臂下压共2组，第一组20kg10次，第二组25kg8次"
    candidate = workout("直臂下压", 20, 2)
    candidate["set_details"] = [{"weight_kg": 25, "reps": 8}, {"weight_kg": 20, "reps": 10}]
    assert validate_workout_records_against_source([candidate], source)[1]
    candidate["set_details"].reverse()
    assert validate_workout_records_against_source([candidate], source)[0] == [candidate]


@pytest.mark.parametrize("transport", ["json", "sse"])
def test_first_message_saves_both_and_runs_following_plan_with_no_confirmation(compact_db, monkeypatch, transport):
    from app.services.agent.task_information import PlanInformation
    from app.services.rag.retrieval import RetrievedChunk
    record_text = "帮我记录哑铃卧推10kg*10次*4，直臂下压20kg*10次*5"
    plan_text = "再帮我安排一下明天的动作，通用示例即可"
    source = record_text + "，" + plan_text
    flags = {"is_negated": False, "is_hypothetical": False, "is_third_party": False}
    parsed = {"tasks": [
        {"task_type": "workout_record", "action": "record", "text": record_text,
         **flags, "workouts": [workout(), workout("直臂下压", 20, 5)]},
        {"task_type": "training_plan", "action": "question", "text": plan_text, **flags}]}
    scenario = RetryScenario(monkeypatch, compact_db, parsed_input=parsed)
    # The shared understanding entry must also carry a source-bound safety
    # assessment; restoring only parse_input leaves the safety gate without a
    # model judgment and the write correctly stops before execution.
    semantic_tasks = [{
        key: task[key] for key in (
            "task_type", "action", "text", "is_negated", "is_hypothetical",
            "is_third_party")}
        | {"history_requirement": "none", "reply_mode": "grounded",
           "reply_purpose": "general"}
        for task in parsed["tasks"]]
    understanding = {"tasks": semantic_tasks,
                     "safety": {"level": "normal", "evidence": "", "categories": [],
                                "reason": "scripted normal"}}
    async def shared_semantics(messages, schema, *, name=None, temperature=0.0,
                               validator=None, **kwargs):
        if schema is PlanInformation:
            value = PlanInformation(scope="generic", scope_quote="通用示例")
        else:
            value = schema.model_validate(copy.deepcopy(understanding))
        if validator is not None:
            validator(value)
        return value
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", lambda: True)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured",
                        shared_semantics)
    monkeypatch.setattr(scenario.agent.graph.rag, "search", lambda *args, **kwargs: [
        RetrievedChunk("训练安排", "明天根据恢复安排训练，逐步增加，每周循序渐进", 1, "fixture", {"rerank_status": "model", "rerank_score": .99}, "fitness-plan")])
    planning = AsyncMock(return_value=plan_message("明天根据今天训练的恢复情况安排动作"))
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_message", planning)
    request = ChatRequest(message=source, conversation_id="direct-source", request_id="source-direct")
    async def run():
        if transport == "json":
            response = await scenario.respond(request)
            return response.answer
        events = [event async for event in scenario.agent.respond_stream(request)]
        return "".join(e["data"]["content"] for e in events if e["event"] == "token")
    answer = asyncio.run(run())
    assert "已记录" in answer and "明天根据今天训练" in answer, answer
    assert "确认" not in answer and planning.await_count == 1
    prompt = "\n".join(m["content"] for m in planning.await_args.args[0])
    assert '"sets": 5' in prompt and '"exercise": "直臂下压"' in prompt
    with compact_db() as s:
        rows = s.execute(text("SELECT exercise, weight_kg, sets, reps FROM workout_records ORDER BY id")).all()
    assert [tuple(r) for r in rows] == [("哑铃卧推", 10, 4, 10), ("直臂下压", 20, 5, 10)]
    replay = asyncio.run(run())
    assert replay == answer and len(scenario.saves) == 2 and planning.await_count == 1


@pytest.mark.parametrize("transport", ["json", "sse"])
def test_real_http_and_model_transport_extract_then_save_without_confirmation(compact_db, monkeypatch, transport):
    import app.api.chat as chat_api
    from tests.react_phase_c2_3_support import C3HTTPHarness, mock_provider_transport
    from test_chat_write_pilot import native
    from app.services.rag.retrieval import RetrievedChunk
    flags = {"is_negated": False, "is_hypothetical": False, "is_third_party": False}
    record_text = "帮我记录哑铃卧推10kg*10次*4，直臂下压20kg*10次*5"
    plan_text = "再帮我安排一下明天的动作，通用示例即可"
    source = record_text + "，" + plan_text
    def semantic_response(tasks):
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
            "id": "synthetic-source-result", "type": "function", "function": {
                "name": "return_request_understanding", "arguments": json.dumps({
                    "tasks": [{**task, "history_requirement": "none",
                               "reply_mode": "grounded", "reply_purpose": "general"}
                              for task in tasks],
                    "safety": {"level": "normal", "evidence": "", "categories": [],
                               "reason": "scripted normal"},
                    "needs_clarification": False}, ensure_ascii=False)}}]}}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 4}}
    def plan_information_response():
        value = {"scope": "generic", "scope_quote": "通用示例", "facts": [], "phase_limits": []}
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
            "id": "synthetic-plan-information", "type": "function", "function": {
                "name": "return_plan_information", "arguments": json.dumps(value, ensure_ascii=False)}}]}}],
                "usage": {"prompt_tokens": 6, "completion_tokens": 4}}
    seen = mock_provider_transport(monkeypatch, [
        semantic_response([
            {"task_type": "workout_record", "action": "record", "text": record_text, **flags},
            {"task_type": "training_plan", "action": "question", "text": plan_text, **flags}]),
        native({"records": [workout(), workout("直臂下压", 20, 5)]}),
        plan_information_response(),
        {"choices": [{"message": plan_message("明天根据恢复情况安排动作")}],
         "usage": {"prompt_tokens": 6, "completion_tokens": 4}},
        # The plan worker's exact task span is already covered by the parent
        # RequestUnderstanding, so no second semantic call is authorized here.
    ])
    scenario = RetryScenario(monkeypatch, compact_db)
    scenario.restore_provider_semantics(monkeypatch)
    # Restore the actual planner/extractor; only provider HTTP is simulated.
    monkeypatch.setattr(scenario.agent.graph, "parse_input", scenario.agent.graph.input_parser.plan)
    monkeypatch.setattr(chat_api, "agent", scenario.agent)
    monkeypatch.setattr(scenario.agent.graph.rag, "search", lambda *args, **kwargs: [
        RetrievedChunk("训练安排", "明天根据恢复安排训练，逐步增加，循序渐进", 1, "fixture", {"rerank_status": "model", "rerank_score": .99}, "fitness-plan")])
    harness = C3HTTPHarness(monkeypatch, compact_db.kw["bind"], write_enabled=False)
    def send():
        if transport == "json":
            status, payload = harness.post_json(source, mode="legacy", request_id="http-source")
            return status, payload["answer"]
        status, events, _ = harness.stream(source, mode="legacy", request_id="http-source")
        return status, "".join(e.get("content", "") for e in events)
    status, answer = send()
    assert status == 200 and "已记录" in answer and "明天根据恢复" in answer, answer
    assert "确认" not in answer and len(seen) == 4
    with compact_db() as s:
        rows = s.execute(text("SELECT exercise, weight_kg, sets, reps FROM workout_records ORDER BY id")).all()
    assert [tuple(r) for r in rows] == [("哑铃卧推", 10, 4, 10), ("直臂下压", 20, 5, 10)]
    assert send() == (status, answer) and len(seen) == 4 and len(scenario.saves) == 2
