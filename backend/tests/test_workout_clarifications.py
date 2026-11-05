"""Mocked AI + real temporary SQL tests for multi-turn workout recording."""
import asyncio
import copy
import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select, text

from app.core import clock
from app.db.memory_models import MemoryTurn
from app.schemas.chat import ChatRequest
from app.services.agent import workout_clarifications as draft
from app.services.agent.semantics import remember_understanding, validate_understanding
from app.services.memory import compact
from test_retry_operations import RetryScenario

SOURCE = "帮我记录哑铃卧推10*10*4，直臂下压20*10*5，再帮我安排一下明天的动作"
WRITE = "帮我记录哑铃卧推10*10*4，直臂下压20*10*5"
READ = "再帮我安排一下明天的动作"


def task(kind, text, action="record"):
    return {"task_type": kind, "action": action, "text": text,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False}


def normal_safety():
    return {"level": "normal", "evidence": "", "categories": [], "reason": "scripted normal"}


def semantic():
    return {"tasks": [task("workout_record", WRITE), task("training_plan", READ, "question")],
            "safety": normal_safety(),
            "needs_clarification": True, "clarification": "请确认格式和日期"}


def proposal():
    return {"records": [
        {"source_quote": "哑铃卧推10*10*4", "exercise": "哑铃卧推", "weight_kg": 10, "sets": 4, "reps": 10},
        {"source_quote": "直臂下压20*10*5", "exercise": "直臂下压", "weight_kg": 20, "sets": 5, "reps": 10},
    ]}


def reply(draft_id, action="accept", accepts_all=True, **kwargs):
    return {"draft_id": draft_id, "action": action, "accepts_all": accepts_all,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False, **kwargs}


def model(proposed=None, decision=None):
    async def structured(messages, schema, *, name, **kwargs):
        payload = json.loads(messages[-1]["content"])
        if name == "return_record_proposals":
            return schema.model_validate(proposed if proposed is not None else proposal())
        if name == "return_request_understanding":
            return schema.model_validate({"tasks": [task("training_plan", payload["query"], "question")],
                                          "safety": normal_safety()})
        assert name == "return_record_reply"
        value = decision(payload) if callable(decision) else decision
        return schema.model_validate(value or reply(payload["draft_id"]))
    return SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(side_effect=structured))


def create_pending(factory, *, proposed=None, cid="multi", request_id="origin"):
    request = ChatRequest(message=SOURCE, conversation_id=cid, request_id=request_id)
    tid, _ = compact.begin_turn(request)
    token = compact.current_turn.set(tid)
    try:
        question = asyncio.run(draft.create_question(model(proposed), request, semantic()))
    finally:
        compact.current_turn.reset(token)
    if question is not None:
        compact.finish_turn(tid, question, {"answer": question, "intent": "clarification"})
    return request, tid, question


def continue_pending(message="都是", *, cid="multi", request_id="confirmation", llm=None):
    request = ChatRequest(message=message, conversation_id=cid, request_id=request_id)
    tid, _ = compact.begin_turn(request)
    token = compact.current_turn.set(tid)
    try:
        result = asyncio.run(draft.continue_request(llm or model(), request))
    finally:
        compact.current_turn.reset(token)
    return request, tid, result


def finish(tid, result):
    compact.finish_turn(tid, result["answer"], {"answer": result["answer"], "intent": result["intent"]})


def read_draft(factory, tid):
    with factory() as s:
        return copy.deepcopy(s.get(MemoryTurn, tid).messages[0]["app_meta"][draft.KEY])


def unknown_action_semantic(source):
    return {"tasks": [{**task("workout_record", source), "action": "question",
                       "reply_purpose": "general", "goal_role": "result"}],
            "needs_clarification": True,
            "clarification": "“哑铃玩具”是动作名称吗？请确认动作名称。"}


def test_unknown_exercise_clarification_keeps_numbers_for_the_next_reply(compact_db):
    source = "帮我记录哑铃玩具10kg*10*4"
    proposed = {"records": [{"source_quote": "哑铃玩具10kg*10*4", "exercise": "哑铃玩具",
                             "weight_kg": 10, "sets": 4, "reps": 10}]}
    request = ChatRequest(message=source, conversation_id="unknown-action", request_id="unknown-origin")
    tid, _ = compact.begin_turn(request)
    token = compact.current_turn.set(tid)
    try:
        question = asyncio.run(draft.create_question(model(proposed), request,
                                                     unknown_action_semantic(source)))
    finally:
        compact.current_turn.reset(token)
    assert question and "哑铃玩具" in question and "10 kg" in question
    compact.finish_turn(tid, question, {"answer": question, "intent": "clarification"})

    correction = ChatRequest(message="是弯举", conversation_id="unknown-action",
                             request_id="unknown-correction")
    tid2, _ = compact.begin_turn(correction)
    token = compact.current_turn.set(tid2)
    try:
        assert draft.has_pending(correction)
        result = asyncio.run(draft.continue_request(model(decision=lambda p: reply(
            p["draft_id"], "update", True,
            patches=[{"index": 0, "source_quote": "是弯举", "exercise": "弯举"}])), correction))
    finally:
        compact.current_turn.reset(token)
    workout = result["parsed_input"]["tasks"][0]["workouts"][0]
    assert workout["exercise"] == "弯举"
    assert (workout["weight_kg"], workout["sets"], workout["reps"]) == (10, 4, 10)
    assert "哑铃玩具" not in result["source"]


def test_needs_record_draft_only_for_unresolved_record_writes():
    assert draft.needs_record_draft(unknown_action_semantic("帮我记录哑铃玩具10kg*10*4"))
    read = {"tasks": [{"task_type": "workout_record", "action": "question",
                       "text": "我昨天卧推的记录是什么", "reply_purpose": "fact_recall",
                       "goal_role": "result"}], "needs_clarification": True}
    assert not draft.needs_record_draft(read)
    assert not draft.needs_record_draft({**unknown_action_semantic("记录卧推"),
                                         "needs_clarification": False})


def test_record_question_that_is_a_read_does_not_create_a_draft(compact_db):
    source = "我昨天卧推的记录是什么"
    semantics = {"tasks": [{**task("workout_record", source), "action": "question",
                            "reply_purpose": "fact_recall", "goal_role": "result"}],
                 "needs_clarification": True}
    request = ChatRequest(message=source, conversation_id="read-action", request_id="read-origin")
    tid, _ = compact.begin_turn(request)
    token = compact.current_turn.set(tid)
    try:
        assert asyncio.run(draft.create_question(model(), request, semantics)) is None
    finally:
        compact.current_turn.reset(token)


@pytest.mark.parametrize("message", ["是", "都是"])
def test_full_preview_confirmation_saves_both_and_keeps_plan_and_replay(compact_db, monkeypatch, message):
    scenario = RetryScenario(monkeypatch, compact_db)
    llm = model()
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", llm.configured)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", llm.complete_structured)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete", AsyncMock(return_value="明天根据恢复情况安排训练。"))
    async def parse(source):
        assert source == SOURCE  # confirmation must not re-extract unbound history
        understanding = validate_understanding(source, semantic())
        remember_understanding(source, understanding)
        return {"tasks": [{k: t[k] for k in ("task_type", "action", "text", *draft.FLAGS)}
                          for t in understanding["tasks"]]}
    monkeypatch.setattr(scenario.agent.graph, "parse_input", AsyncMock(side_effect=parse))
    first = ChatRequest(message=SOURCE, conversation_id="e2e", request_id="e2e-origin")
    preview = asyncio.run(scenario.respond(first))
    assert preview.intent == "clarification" and not scenario.saves
    assert "10 kg × 10 次 × 4 组" in preview.answer and "20 kg × 10 次 × 5 组" in preview.answer
    confirmation = ChatRequest(message=message, conversation_id="e2e", request_id="e2e-confirm")
    response = asyncio.run(scenario.respond(confirmation))
    assert "已记录" in response.answer, response.answer
    with compact_db() as s:
        rows = s.execute(text("SELECT exercise, weight_kg, sets, reps FROM workout_records ORDER BY exercise")).all()
        turn = s.scalar(select(MemoryTurn).where(MemoryTurn.request_id == confirmation.request_id))
        assert turn.messages[0]["content"] == message
        assert turn.operations["plan"][-1]["task_type"] == "training_plan"
        assert len(turn.operations["plan"][-1]["depends_on"]) == 2
    assert set(map(tuple, rows)) == {("哑铃卧推", 10.0, 4, 10), ("直臂下压", 20.0, 5, 10)}
    calls = llm.complete_structured.await_count
    replayed = asyncio.run(scenario.respond(confirmation))
    assert replayed.answer == response.answer and len(scenario.saves) == 2
    assert llm.complete_structured.await_count == calls
    assert scenario.agent.graph.parse_input.await_count == 1


@pytest.mark.parametrize("action,flags", [
    ("unclear", {}), ("accept", {"is_negated": True}),
    ("accept", {"is_hypothetical": True}), ("accept", {"is_third_party": True}),
])
def test_partial_negative_hypothetical_third_party_never_claim(compact_db, action, flags):
    _, origin, _ = create_pending(compact_db)
    _, tid, result = continue_pending(llm=model(decision=lambda p: reply(p["draft_id"], action=action, **flags)))
    assert "source" not in result and result["intent"] == "clarification"
    assert read_draft(compact_db, origin)["state"] == "waiting"
    with compact_db() as s:
        assert s.execute(text("SELECT COUNT(*) FROM workout_records")).scalar() == 0


def test_cancel_and_unrelated(compact_db):
    _, origin, _ = create_pending(compact_db)
    _, tid, result = continue_pending("今天吃什么", llm=model(decision=lambda p: reply(p["draft_id"], "unrelated", False)))
    assert result is None and read_draft(compact_db, origin)["state"] == "waiting"
    _, tid, result = continue_pending("取消", request_id="cancel", llm=model(decision=lambda p: reply(p["draft_id"], "cancel", False)))
    assert result["intent"] == "clarification_cancelled"
    finish(tid, result)
    _, _, result = continue_pending(request_id="after-cancel")
    assert result is None and read_draft(compact_db, origin)["state"] == "cancelled"


def test_correction_previews_then_accepts_new_frozen_values(compact_db):
    _, origin, _ = create_pending(compact_db)
    patches = [{"index": 1, "source_quote": "第二个改成25kg", "weight_kg": 25}]
    _, tid, result = continue_pending("第二个改成25kg", llm=model(decision=lambda p: reply(p["draft_id"], "update", False, patches=patches)))
    assert "25 kg" in result["answer"] and "source" not in result
    finish(tid, result)
    _, _, accepted = continue_pending(request_id="accept-updated")
    assert "25kg" in accepted["source"] and "20kg" not in accepted["source"]
    changed = read_draft(compact_db, origin)["records"][1]
    assert changed["field_sources"]["weight_kg"]["message_id"] == tid + ":user"


def test_missing_information_can_be_supplied_and_saved_without_another_question(compact_db):
    proposed = proposal()
    proposed["records"][1]["weight_kg"] = None
    create_pending(compact_db, proposed=proposed)
    _, _, accepted = continue_pending("第二个重量20kg，其余按刚才的记录", llm=model(decision=lambda p: reply(
        p["draft_id"], "update", True, patches=[{"index": 1, "source_quote": "第二个重量20kg", "weight_kg": 20}])))
    assert accepted["parsed_input"]["tasks"][1]["workouts"][0]["weight_kg"] == 20


@pytest.mark.parametrize("mutation", ["quote", "number", "name", "future", "bool", "nan"])
def test_model_cannot_create_facts_without_original_user_source(compact_db, mutation):
    proposed = proposal()
    record = proposed["records"][0]
    if mutation == "quote": record["source_quote"] = "卧推99kg"
    if mutation == "number": record["weight_kg"] = 99
    if mutation == "name": record["exercise"] = "深蹲"
    if mutation == "future": record["date_quote"] = "明天"
    if mutation == "bool": record["sets"] = True
    if mutation == "nan": record["weight_kg"] = float("nan")
    try:
        _, origin, question = create_pending(compact_db, proposed=proposed)
        assert question is None
        with compact_db() as s:
            assert draft.KEY not in s.get(MemoryTurn, origin).messages[0]["app_meta"]
    except draft.DraftError:
        pass


@pytest.mark.parametrize("mutation", ["owner", "convo", "epoch", "hidden", "deleted", "expired", "source", "question"])
def test_scope_privacy_ttl_and_question_visibility_block_old_draft(compact_db, mutation):
    _, origin, _ = create_pending(compact_db)
    with compact_db() as s, s.begin():
        row = s.get(MemoryTurn, origin)
        meta = copy.deepcopy(row.messages)
        value = meta[0]["app_meta"][draft.KEY]
        if mutation == "owner": value["owner"] = "another-user"
        if mutation == "convo": row.conversation_id = "another-conversation"
        if mutation == "epoch": row.memory_epoch += 1
        if mutation == "hidden": row.hidden_at = clock.utc_now()
        if mutation == "deleted": row.deleted_at = clock.utc_now()
        if mutation == "expired": value["expires_at"] = 0
        if mutation == "source": meta[0]["content"] = "其他训练"
        if mutation == "question": meta[-1]["content"] = "没有展示草稿"
        row.messages = meta
    llm = model()
    _, _, result = continue_pending(llm=llm)
    assert result is None and llm.complete_structured.await_count == 0


def test_bad_reply_target_and_unsourced_correction_cannot_claim(compact_db):
    _, origin, _ = create_pending(compact_db)
    with pytest.raises(draft.DraftError):
        continue_pending(llm=model(decision=reply("forged-target")))
    with pytest.raises(draft.DraftError):
        continue_pending("改成25kg", request_id="bad-number", llm=model(decision=lambda p: reply(
            p["draft_id"], "update", True, patches=[{"index": 0, "source_quote": "25kg", "weight_kg": 99}])))
    assert read_draft(compact_db, origin)["state"] == "waiting"


def test_one_claim_per_draft_and_same_request_reuses_binding_even_after_ttl(compact_db, monkeypatch):
    _, origin, _ = create_pending(compact_db)
    _, _, accepted = continue_pending()
    assert accepted.get("source") and read_draft(compact_db, origin)["state"] == "claimed"
    llm = model()
    _, _, again = continue_pending(llm=llm)
    assert again == accepted and llm.complete_structured.await_count == 0
    _, _, other = continue_pending(request_id="another-confirmation")
    assert other is None
    original_now = clock.utc_now()
    monkeypatch.setattr(clock, "utc_now", lambda: original_now + timedelta(hours=2))
    _, _, after_ttl = continue_pending(llm=llm)
    assert after_ttl == accepted and llm.complete_structured.await_count == 0


def test_model_outage_preserves_draft_and_budget_failure_propagates(compact_db):
    from app.services.llm.budget import ModelBudgetError
    _, origin, _ = create_pending(compact_db)
    llm = model()
    llm.complete_structured.side_effect = ConnectionError("private provider body")
    _, _, result = continue_pending(llm=llm)
    assert "草稿已保留" in result["answer"] and "private" not in result["answer"]
    assert read_draft(compact_db, origin)["state"] == "waiting"
    llm.complete_structured.side_effect = ModelBudgetError("request_limit")
    with pytest.raises(ModelBudgetError):
        continue_pending(request_id="budget", llm=llm)


def test_plain_confirmation_without_draft_does_not_invoke_or_authorize_model(compact_db):
    llm = model()
    _, _, result = continue_pending(llm=llm)
    assert result is None and llm.complete_structured.await_count == 0


def test_claim_is_atomic_across_two_worker_threads(compact_db):
    from concurrent.futures import ThreadPoolExecutor
    create_pending(compact_db)
    original = None
    attempts = []
    for i in range(2):
        request = ChatRequest(message="是", conversation_id="multi", request_id=f"racing-{i}")
        tid, _ = compact.begin_turn(request)
        token = compact.current_turn.set(tid)
        try:
            original = draft._load(request)
        finally:
            compact.current_turn.reset(token)
        attempts.append((request, tid))
    def claim(attempt):
        request, tid = attempt
        token = compact.current_turn.set(tid)
        try:
            return draft._transition(request, original, draft.WorkoutReply.model_validate(reply(original["id"])))
        except draft.DraftError:
            return None
        finally:
            compact.current_turn.reset(token)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, attempts))
    assert sum(bool(r and r.get("source")) for r in results) == 1
    with compact_db() as s:
        bindings = [r.messages[0]["app_meta"].get(draft.BINDING) for r in s.scalars(select(MemoryTurn))]
        assert sum(bool(b) for b in bindings) == 1


def test_partial_save_retry_reuses_adopted_command_without_model_or_duplicate(compact_db, monkeypatch):
    from app.db.storage import StorageUnavailableError
    create_pending(compact_db)
    scenario = RetryScenario(monkeypatch, compact_db, fail={"直臂下压": True})
    llm = model()
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", llm.configured)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", llm.complete_structured)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete", AsyncMock(return_value="训练建议"))
    request = ChatRequest(message="都是", conversation_id="multi", request_id="partial-save")
    with pytest.raises(StorageUnavailableError):
        asyncio.run(scenario.respond(request))
    assert len(scenario.saves) == 1 and scenario.saves[0]["exercise"] == "哑铃卧推"
    calls = llm.complete_structured.await_count
    scenario.fail.clear()
    response = asyncio.run(scenario.respond(request))
    assert "已记录" in response.answer
    assert len(scenario.saves) == 2
    assert sum(c.kwargs.get("name") == "return_record_reply" for c in llm.complete_structured.await_args_list) == 1
    assert not scenario.agent.graph.parse_input.called
    with compact_db() as s:
        assert s.execute(text("SELECT COUNT(*) FROM workout_records")).scalar() == 2


def test_new_draft_supersedes_old_one(compact_db):
    _, origin, _ = create_pending(compact_db)
    _, new_origin, _ = create_pending(compact_db, request_id="new-origin")
    assert read_draft(compact_db, origin)["state"] == "superseded"
    _, _, accepted = continue_pending()
    assert accepted["origin"] == new_origin


def test_complete_question_is_replayed_over_sse_without_another_model_call(compact_db, monkeypatch):
    create_pending(compact_db)
    scenario = RetryScenario(monkeypatch, compact_db)
    llm = model()
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", llm.configured)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete_structured", llm.complete_structured)
    monkeypatch.setattr(scenario.agent.graph.llm, "complete", AsyncMock(return_value="训练建议"))
    request = ChatRequest(message="是", conversation_id="multi", request_id="sse-confirmation")
    async def collect():
        return [event async for event in scenario.agent.respond_stream(request)]
    events = asyncio.run(collect())
    assert events[-1]["event"] == "done" and events[-1]["data"]["intent"] != "clarification"
    assert "已记录" in "".join(e["data"]["content"] for e in events if e["event"] == "token")
    calls = llm.complete_structured.await_count
    replay = asyncio.run(collect())
    assert replay[-1]["data"]["replayed"] and len(scenario.saves) == 2
    assert llm.complete_structured.await_count == calls


def test_duplicate_source_occurrence_cannot_create_two_records(compact_db):
    proposed = proposal()
    proposed["records"] = [proposed["records"][0], proposed["records"][0]]
    with pytest.raises(draft.DraftError):
        create_pending(compact_db, proposed=proposed)


def test_confirmation_across_midnight_keeps_preview_date(compact_db, monkeypatch):
    from datetime import datetime, timezone
    before = datetime(2026, 10, 18, 15, 59, tzinfo=timezone.utc)
    monkeypatch.setattr(clock, "utc_now", lambda: before)
    _, origin, _ = create_pending(compact_db)
    day = read_draft(compact_db, origin)["records"][0]["record_date"]
    monkeypatch.setattr(clock, "utc_now", lambda: before + timedelta(minutes=2))
    _, _, accepted = continue_pending()
    assert accepted["parsed_input"]["tasks"][0]["workouts"][0]["record_date"] == day


def test_proposal_outage_reports_failure_without_an_unbound_confirmation(compact_db, monkeypatch):
    scenario = RetryScenario(monkeypatch, compact_db)
    async def parse(source):
        understanding = validate_understanding(source, semantic())
        remember_understanding(source, understanding)
        return {"tasks": [{k: t[k] for k in ("task_type", "action", "text", *draft.FLAGS)}
                          for t in understanding["tasks"]]}
    monkeypatch.setattr(scenario.agent.graph, "parse_input", AsyncMock(side_effect=parse))
    monkeypatch.setattr(scenario.agent.graph.llm, "configured", lambda: False)
    response = asyncio.run(scenario.respond(ChatRequest(message=SOURCE, request_id="no-proposal")))
    assert response.intent == "execution_limited"
    assert "安全语义判断" in response.answer
    assert "请确认格式和日期" not in response.answer and not scenario.saves
