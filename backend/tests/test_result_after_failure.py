"""Failure turns remain history but must not replace a valid result locator."""
import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core import clock
from app.core.config import settings
from app.db.memory_models import MemoryTurn
from app.schemas.chat import ChatRequest
from app.services.agent.result_reference import result_sources, validate_result_reference
from app.services.agent.semantics import SemanticTask
from app.services.llm.budget import ModelBudgetError
from app.services.memory import compact
from app.services.memory.recent_window import load_window
from tests.test_wiki_addressed_context import memory_db, add_turn


def limit_turn(message="给我具体的食物", cid="wiki-chat", *, intent="execution_limited"):
    tid, _ = compact.begin_turn(ChatRequest(message=message, conversation_id=cid))
    compact.finish_turn(tid, "FAILED_ATTEMPT_ONLY", {
        "answer": "FAILED_ATTEMPT_ONLY", "intent": intent,
        "metrics": {"execution_mode": "limited", "react_status": "limited"}})
    return tid


@pytest.mark.parametrize("failures", [1, 12])
@pytest.mark.parametrize("unrelated", [0, 100])
def test_latest_result_skips_failure_before_limit_and_reuses_loaded_rows(memory_db, monkeypatch, failures, unrelated):
    monkeypatch.setattr(settings, "memory_short_window_turns", 10)
    good = add_turn("给我制定减脂饮食计划", "GOOD_DIET_RESULT")
    failed = [limit_turn() for _ in range(failures)]
    for i in range(unrelated):
        add_turn("其他会话", "PRIVATE_OTHER_RESULT", cid="other-chat")
    memory_db.queries.clear()
    memory = load_window("wiki-chat", 0)
    sources = result_sources(memory)
    assert sources[0]["message_id"] == good + ":assistant"
    assert all(s["message_id"] not in {tid + ":assistant" for tid in failed} for s in sources)
    assert "FAILED_ATTEMPT_ONLY" in json.dumps(memory["short_memory"])
    assert "PRIVATE_OTHER_RESULT" not in json.dumps(memory)
    # Window LIMIT10 plus at most one SQL LIMIT1 lookup outside the window.
    assert len(memory_db.queries) == (1 if failures < 10 else 2)
    assert all("LIMIT" in sql.upper() for sql in memory_db.queries)
    # The same source must validate, not merely appear in the prompt.
    task = SemanticTask.model_validate({
        "task_type": "nutrition_advice", "action": "question", "text": "给我具体的食物",
        "is_negated": False, "is_hypothetical": False, "is_third_party": False, "quoted": False,
        "result_reference": {"message_id": good + ":assistant", "selection": "latest", "quote": "具体的食物"}})
    assert validate_result_reference(task, "follow_up", memory) is None
    assert good + ":assistant" in task.context_ids


@pytest.mark.parametrize("invisible", ["hidden", "deleted", "epoch", "other_conversation"])
def test_result_lookup_cannot_revive_invisible_answer_after_failure(memory_db, monkeypatch, invisible):
    monkeypatch.setattr(settings, "memory_short_window_turns", 10)
    good = add_turn("原目标", "INVISIBLE_RESULT")
    limit_turn()
    with memory_db.factory.begin() as session:
        row = session.get(MemoryTurn, good)
        if invisible == "hidden": row.hidden_at = compact.now()
        elif invisible == "deleted": row.deleted_at = compact.now()
        elif invisible == "epoch": row.memory_epoch = 1
        else: row.conversation_id = "other-chat"
    memory_db.queries.clear()
    assert compact.latest_result_dialogue("wiki-chat", 0) == []
    assert len(memory_db.queries) == 1
    memory_db.queries.clear()
    memory = load_window("wiki-chat", 0)
    assert result_sources(memory) == []
    assert "INVISIBLE_RESULT" not in json.dumps(memory)
    assert len(memory_db.queries) == 2


def test_legacy_result_is_retained_and_execution_mode_alone_marks_failure(memory_db, monkeypatch):
    monkeypatch.setattr(settings, "memory_short_window_turns", 10)
    good = add_turn("旧版本饮食目标", "LEGACY_RESULT_WITHOUT_MARKER")
    limit_turn(intent="clarification")
    assert result_sources({"recent_result_dialogue": compact.latest_result_dialogue("wiki-chat", 0)})[0]["message_id"] == good + ":assistant"
    assert result_sources(load_window("wiki-chat", 0))[0]["message_id"] == good + ":assistant"


@pytest.mark.parametrize("transport", ["json", "sse"])
@pytest.mark.parametrize("window", [0, 10])
def test_diet_failure_then_concrete_followup_succeeds_and_replays(tmp_path, monkeypatch, transport, window):
    from evals.v2.safety import deny_network
    from tests.react_phase_a_support import tool_call
    from tests.react_phase_b_codex_support import environment, chunks, semantic_outline, semantic_task, sse_body
    source, followup = "给我制定减脂饮食计划", "给我具体的食物"
    with deny_network() as audit, environment(tmp_path, monkeypatch, "result-after-failure") as env:
        env.mp.setattr(settings, "wiki_enabled", True)
        env.mp.setattr(settings, "memory_short_window_turns", window)
        service = compact.CompactMemory(SimpleNamespace(initialize=AsyncMock(side_effect=AssertionError("no Redis"))))
        env.load.side_effect = service.load
        env.mock_rag(chunks(source))
        routed = []
        selected = []
        async def structured(messages, schema, **kwargs):
            payload = json.loads(messages[-1]["content"])
            routed.append(payload)
            if failure[0]:
                raise ModelBudgetError("model_prepare_token_budget_exhausted")
            task = semantic_task("nutrition_advice", "question", payload["query"])
            task["delivery_type"] = "plan" if payload["query"] == source else "advice"
            task["history_requirement"] = "none"
            if payload["query"] == followup:
                latest = payload["result_sources"][0]
                selected.append(latest)
                task.update(result_reference={"message_id": latest["message_id"], "selection": "latest", "quote": followup},
                            context_ids=[latest["user_message_id"], latest["message_id"]], history_requirement="required")
            value = schema.model_validate(semantic_outline(payload["query"], [task]) | {
                "relation": "follow_up" if payload["query"] == followup else "new"})
            if kwargs.get("validator"): kwargs["validator"](value)
            return value
        failure = [False]
        llm = env.mock_llm(final="原饮食计划", policy=lambda index, messages: {
            "role": "assistant", "content": "", "tool_calls": [
                tool_call("diet-" + str(index), "advise_nutrition", {"query": source})]})
        env.mp.setattr(llm, "complete_structured", structured)
        base = clock.utc_now()
        env.mp.setattr(clock, "utc_now", lambda: base)
        first = env.post("diet-success", message=source)
        assert first["metrics"]["react_status"] == "complete"
        target = routed[0]["result_sources"]
        assert target == []
        with env.factory() as session:
            good = session.query(MemoryTurn).filter_by(request_id="diet-success").one().turn_id
        failure[0] = True
        env.mp.setattr(clock, "utc_now", lambda: base + timedelta(seconds=1))
        stopped = env.post("diet-failed", message=followup)
        assert stopped["metrics"]["mode_reason"] == "semantic_budget_stopped"
        failure[0] = False
        env.mp.setattr(clock, "utc_now", lambda: base + timedelta(seconds=2))
        llm.final = "具体食物示例：燕麦、鸡蛋、豆腐。"
        before = env.snapshot()
        result = (env.post("diet-followup", message=followup) if transport == "json" else
                  sse_body(env.stream("diet-followup", message=followup)))
        assert result["answer"] == llm.final, result
        assert result["metrics"]["react_status"] == "complete"
        assert selected[0]["message_id"] == good + ":assistant"
        assert all(h["name"] != "plan_training" for h in env.handlers)
        prompt = json.dumps([call for call in llm.calls if not call["tools"]][-1]["messages"], ensure_ascii=False)
        assert source in prompt and followup in prompt
        assert "Token 预算不足" not in prompt
        count, semantic_count = len(llm.calls), len(routed)
        replay = env.post("diet-followup", message=followup)
        assert replay["answer"] == result["answer"]
        assert (len(llm.calls), len(routed)) == (count, semantic_count)
        assert env.snapshot() == before
    assert audit == {"blocked_connections": 0, "blocked_dns": 0}
