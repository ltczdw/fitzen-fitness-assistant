"""B03 regression: shared semantics decide a mixed write before ReAct runs.

Every provider boundary is mocked and storage is a fresh temporary SQLite file.
The test drives the real HTTP JSON/SSE endpoints, the real mode selection and
the real source/write authorization gate. The legacy dispatch itself is a
controlled fixture because the full legacy graph is covered by its own matrix;
it performs real DML only after the validated write task reaches ``graph.run``.
"""
from __future__ import annotations

import copy
import json

import pytest

from tests.react_phase_b_codex_support import (
    FINAL, answer_script, default_semantics, environment,
    semantic_outline as _support_semantic_outline,
    semantic_task, sse_body,
)


def semantic_outline(source, tasks=None):
    """Add the request-level safety judgment this batch now requires.

    The protected formal support remains untouched; ordinary B03 cases express
    the model's normal safety result at the same provider boundary.
    """
    value = _support_semantic_outline(source, tasks)
    for task in value.get("tasks") or []:
        task.setdefault("history_requirement", "none")
    value["safety"] = {"level": "normal", "evidence": "", "categories": [],
                       "reason": "scripted normal"}
    return value

TRAIN_MESSAGE = "记录卧推60kg，4组8次，然后告诉我怎么恢复"
TRAIN_WRITE = "记录卧推60kg，4组8次"
TRAIN_READ = "然后告诉我怎么恢复"
NUTRITION_MESSAGE = "午餐吃了鸡胸肉200克，帮我记下来，再给我一些蛋白质建议"
NUTRITION_WRITE = "午餐吃了鸡胸肉200克，帮我记下来"
NUTRITION_READ = "再给我一些蛋白质建议"


@pytest.fixture(autouse=True)
def b03_network_and_pending(monkeypatch):
    from evals.v2.safety import deny_network
    from app.services.agent import food_calculator
    monkeypatch.setattr(food_calculator, "_pending_food", {})
    with deny_network() as audit:
        yield
    assert audit == {"blocked_connections": 0, "blocked_dns": 0}


def training_semantics(source):
    return semantic_outline(source, [
        {**semantic_task("workout_record", "record", TRAIN_WRITE),
         "workout_sources": [{"exercise": "卧推", "quote": "卧推60kg，4组8次"}]},
        semantic_task("training_qa", "question", TRAIN_READ),
    ])


def training_parser_input():
    return {"tasks": [
        {"task_type": "workout_record", "action": "record", "text": TRAIN_WRITE,
         "is_negated": False, "is_hypothetical": False, "is_third_party": False,
         "workouts": [{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}]},
        {"task_type": "training_qa", "action": "question", "text": TRAIN_READ,
         "is_negated": False, "is_hypothetical": False, "is_third_party": False},
    ]}


def nutrition_semantics(source):
    return semantic_outline(source, [
        semantic_task("nutrition_record", "record", NUTRITION_WRITE),
        semantic_task("nutrition_advice", "question", NUTRITION_READ),
    ])


def nutrition_parser_input():
    return {"tasks": [
        {"task_type": "nutrition_record", "action": "record", "text": NUTRITION_WRITE,
         "is_negated": False, "is_hypothetical": False, "is_third_party": False,
         "foods": [{"source_text": "鸡胸肉200克", "name": "鸡胸肉", "quantity": 1,
                    "unit": "克", "weight_g": 200, "meal_type": "午餐",
                    "record_date": None, "is_negated": False}]},
        {"task_type": "nutrition_advice", "action": "question", "text": NUTRITION_READ,
         "is_negated": False, "is_hypothetical": False, "is_third_party": False},
    ]}


def write_tasks(tasks, task_type):
    return [task for task in tasks if task.get("task_type") == task_type]


def assert_authorized(task):
    payload = task.get("payload") or {}
    assert payload.get("write_authorization") == "authorized", payload
    assert task.get("side_effect") is True


def test_b03_training_write_routes_whole_request_legacy_json_sse_replay(tmp_path, monkeypatch):
    from app.db.repositories import create_workout

    with environment(tmp_path, monkeypatch, "b03-training") as env:
        before, routed = env.snapshot(), []

        async def controlled_legacy(*args, **kwargs):
            tasks = copy.deepcopy(kwargs["planned_tasks"])
            routed.extend(tasks)
            create_workout("demo-user", "卧推", 4, 8, 60, note="b03-semantic-fixture")
            return {"worker_results": [{"status": "complete", "answer": "已记录卧推并给出恢复建议。"}],
                    "citations": [], "model_used": False, "safety_level": "normal"}

        env.legacy.side_effect = controlled_legacy
        env.parse.return_value = training_parser_input()
        env.mock_llm(semantic=training_semantics)
        body = env.post("b03-train-json", message=TRAIN_MESSAGE, mode="legacy")
        after = env.snapshot()
        writes = write_tasks(routed, "workout_record")
        assert body["metrics"].get("execution_mode") == "legacy", body
        assert body["metrics"].get("mode_reason") == "requested_legacy", body["metrics"]
        assert env.legacy.await_count == 1
        assert env.parse.await_count == 1
        assert len(env.llm.semantic_calls) == 1
        assert not env.states and not env.handlers
        assert writes and all(task.get("action") == "record" for task in writes)
        assert_authorized(writes[0])
        assert len(after["workout_records"]) == len(before["workout_records"]) + 1
        assert after["workout_records"][-1]["exercise"] == "卧推"
        assert after["workout_records"][-1]["weight_kg"] == 60
        assert len(after["workout_sessions"]) == len(before["workout_sessions"]) + 1

        models, parsers, legacies, rows = (len(env.llm.semantic_calls), env.parse.await_count,
                                           env.legacy.await_count, env.snapshot())
        replay = env.post("b03-train-json", message=TRAIN_MESSAGE, mode="legacy")
        assert replay["replayed"] is True
        assert replay["answer"] == body["answer"]
        assert replay["metrics"].get("execution_mode") == "legacy"
        assert (len(env.llm.semantic_calls), env.parse.await_count,
                env.legacy.await_count, env.snapshot()) == (models, parsers, legacies, rows)

        env.mock_llm(semantic=training_semantics)
        env.reset_counts()
        before_stream = env.snapshot()
        events = env.stream("b03-train-sse", message=TRAIN_MESSAGE, mode="legacy")
        stream = sse_body(events)
        assert [event["event"] for event in events][-1] == "done"
        assert stream["metrics"].get("execution_mode") == "legacy", stream["metrics"]
        assert stream["metrics"].get("mode_reason") == "requested_legacy"
        assert len(env.llm.semantic_calls) == 1 and env.parse.await_count == 1
        assert env.legacy.await_count == 1 and not env.states
        assert len(env.snapshot()["workout_records"]) == len(before_stream["workout_records"]) + 1
        assert env.snapshot()["workout_records"][-1]["exercise"] == "卧推"


def test_b03_nutrition_write_routes_whole_request_legacy_json_sse(tmp_path, monkeypatch):
    from app.db.nutrition_repository import create_nutrition

    with environment(tmp_path, monkeypatch, "b03-nutrition") as env:
        before, routed = env.snapshot(), []

        async def controlled_legacy(*args, **kwargs):
            tasks = copy.deepcopy(kwargs["planned_tasks"])
            routed.extend(tasks)
            create_nutrition("demo-user", "午餐", "鸡胸肉200克", calories=300,
                             protein_g=62, carbs_g=0, fat_g=7)
            return {"worker_results": [{"status": "complete", "answer": "已记录鸡胸肉并给出蛋白质建议。"}],
                    "citations": [], "model_used": False, "safety_level": "normal"}

        env.legacy.side_effect = controlled_legacy
        env.parse.return_value = nutrition_parser_input()
        env.mock_llm(semantic=nutrition_semantics)
        body = env.post("b03-food-json", message=NUTRITION_MESSAGE, mode="react_readonly")
        after = env.snapshot()
        writes = write_tasks(routed, "nutrition_record")
        assert body["metrics"].get("execution_mode") == "legacy", body["metrics"]
        assert body["metrics"].get("mode_reason") == "contains_write_or_confirmation"
        assert env.legacy.await_count == 1 and env.parse.await_count == 1
        assert len(env.llm.semantic_calls) == 1
        assert not env.states and not env.handlers
        assert writes and all(task.get("action") == "record" for task in writes)
        assert_authorized(writes[0])
        assert (writes[0].get("payload") or {}).get("foods")[0]["name"] == "鸡胸肉"
        assert len(after["nutrition_logs"]) == len(before["nutrition_logs"]) + 1
        assert after["nutrition_logs"][-1]["description"] == "鸡胸肉200克"

        env.mock_llm(semantic=nutrition_semantics)
        env.reset_counts()
        before_stream = env.snapshot()
        stream = sse_body(env.stream("b03-food-sse", message=NUTRITION_MESSAGE))
        assert stream["metrics"].get("execution_mode") == "legacy"
        assert stream["metrics"].get("mode_reason") == "contains_write_or_confirmation"
        assert len(env.llm.semantic_calls) == 1 and env.parse.await_count == 1
        assert env.legacy.await_count == 1 and not env.states
        assert len(env.snapshot()["nutrition_logs"]) == len(before_stream["nutrition_logs"]) + 1


def test_b03_pure_read_reuses_one_semantic_decision_without_legacy_parse(tmp_path, monkeypatch):
    message = "训练后怎么恢复"
    with environment(tmp_path, monkeypatch, "b03-pure") as env:
        before = env.snapshot()
        env.mock_llm(answer_script(), semantic=lambda source: semantic_outline(
            source, [semantic_task("training_qa", "question", source)]))
        body = env.post("b03-pure", message=message, mode="react_readonly")
        state = env.states[-1]
        assert body["metrics"].get("execution_mode") == "react_readonly", body["metrics"]
        assert body["metrics"].get("mode_reason") == "pure_read"
        assert body["metrics"].get("react_status") == "complete"
        assert body["answer"] and body["model_used"] is True
        assert len(env.llm.semantic_calls) == 1
        assert env.parse.await_count == 0 and env.shadow.await_count == 0
        assert env.legacy.await_count == 0 and not env.forbidden
        assert [task.get("text") for task in state["intent_candidates"]] == [message]
        assert env.snapshot() == before


def test_b03_question_about_a_record_is_pure_read_without_write(tmp_path, monkeypatch):
    message = "记录卧推60kg，4组8次是什么意思"
    with environment(tmp_path, monkeypatch, "b03-question") as env:
        before = env.snapshot()
        env.mock_llm(answer_script(), semantic=lambda source: semantic_outline(
            source, [semantic_task("workout_record", "question", source)]))
        body = env.post("b03-question", message=message, mode="react_readonly")
        assert body["metrics"].get("execution_mode") == "react_readonly", body["metrics"]
        assert body["metrics"].get("mode_reason") == "pure_read"
        assert body["metrics"].get("react_status") == "complete"
        assert len(env.llm.semantic_calls) == 1
        assert env.parse.await_count == 0 and env.legacy.await_count == 0
        assert not env.handlers or all(h["name"] != "record_workout" for h in env.handlers)
        assert env.snapshot() == before


def test_b03_shared_understanding_receives_application_memory_context(tmp_path, monkeypatch):
    message = "训练后怎么恢复"
    memory = {
        "memory_backend": "compact_v1",
        "short_memory": [{"message_id": "h1", "role": "user", "content": "我对花生过敏"}],
        "long_memory": {"allergy": {"text": "花生过敏", "source_message_id": "h1",
                                    "source_quote": "我对花生过敏"}},
    }
    with environment(tmp_path, monkeypatch, "b03-context") as env:
        env.load.return_value = memory
        before = env.snapshot()
        env.mock_llm(answer_script(), semantic=lambda source: semantic_outline(
            source, [semantic_task("training_qa", "question", source)]))
        body = env.post("b03-context", message=message, mode="react_readonly")
        assert body["metrics"].get("execution_mode") == "react_readonly"
        call = env.llm.semantic_calls[0]
        payload = json.loads(call["messages"][-1]["content"])
        assert payload["active_facts"] == memory["long_memory"]
        assert payload["history"][0] == {"message_id": "h1", "role": "user"}
        assert any(message.get("role") == "user" and "我对花生过敏" in message.get("content", "")
                   for message in call["messages"])
        assert env.snapshot() == before


def test_b03_ambiguous_reference_asks_instead_of_react_or_guessing(tmp_path, monkeypatch):
    message = "那一个帮我记一下"
    question = "你指的是哪一条记录？请补充动作或餐次。"

    def semantic(source):
        raw = semantic_outline(source, [semantic_task("training_qa", "question", source)])
        raw["references"] = [{"source_text": source, "query": source, "ambiguous": True}]
        raw["clarification"] = question
        return raw

    with environment(tmp_path, monkeypatch, "b03-ambiguous") as env:
        before = env.snapshot()
        env.mock_llm(answer_script(), semantic=semantic)
        body = env.post("b03-ambiguous", message=message, mode="react_readonly")
        assert body["answer"] == question
        assert body["intent"] == "clarification"
        assert len(env.llm.semantic_calls) == 1
        assert env.legacy.await_count == 0 and not env.states and not env.handlers
        assert env.snapshot() == before


@pytest.mark.parametrize("flag", ["is_negated", "is_hypothetical", "is_third_party"])
def test_b03_flagged_record_intent_stays_legacy_with_zero_business_write(tmp_path, monkeypatch, flag):
    message = "记录卧推60kg，4组8次，然后告诉我怎么恢复"
    flagged = semantic_task("workout_record", "record", TRAIN_WRITE)
    flagged[flag] = True
    with environment(tmp_path, monkeypatch, "b03-flag-" + flag) as env:
        before, routed = env.snapshot(), []

        async def controlled_legacy(*args, **kwargs):
            routed.extend(copy.deepcopy(kwargs["planned_tasks"]))
            return {"worker_results": [{"status": "complete", "answer": "该记录未保存。"}],
                    "citations": [], "model_used": False, "safety_level": "normal"}

        env.legacy.side_effect = controlled_legacy
        env.parse.return_value = {"tasks": [
            {"task_type": "workout_record", "action": "record", "text": TRAIN_WRITE,
             "is_negated": False, "is_hypothetical": False, "is_third_party": False,
             flag: True},
            {"task_type": "training_qa", "action": "question", "text": TRAIN_READ,
             "is_negated": False, "is_hypothetical": False, "is_third_party": False},
        ]}
        env.mock_llm(semantic=lambda source: semantic_outline(
            source, [flagged, semantic_task("training_qa", "question", TRAIN_READ)]))
        body = env.post("b03-flag-" + flag, message=message, mode="legacy")
        writes = write_tasks(routed, "workout_record")
        assert body["metrics"].get("execution_mode") == "legacy"
        assert body["metrics"].get("mode_reason") == "requested_legacy"
        assert env.legacy.await_count == 1 and not env.states
        assert writes and (writes[0].get("payload") or {}).get("write_authorization") == "rejected_unauthorized"
        assert writes[0].get("side_effect") is False
        assert env.snapshot() == before


@pytest.mark.parametrize("semantic_source,error", [
    ("invalid_span", None),
    ("spoofed_fact", None),
    ("configured_failure", RuntimeError("synthetic semantic outage")),
    ("invalid_json", ValueError("synthetic invalid structured output")),
])
def test_b03_semantic_failure_is_explicit_and_never_pure_read(tmp_path, monkeypatch, semantic_source, error):
    message = "记录卧推60kg，4组8次，然后告诉我怎么恢复"
    with environment(tmp_path, monkeypatch, "b03-fail-" + semantic_source) as env:
        before = env.snapshot()
        if semantic_source == "invalid_span":
            semantic = lambda source: semantic_outline(
                source, [semantic_task("workout_record", "record", "记录硬拉100kg")])
            env.mock_llm(semantic=semantic)
        elif semantic_source == "spoofed_fact":
            def semantic(source):
                raw = semantic_outline(source, [semantic_task("workout_record", "record", TRAIN_WRITE)])
                raw["tasks"][0]["fact_text"] = "记录硬拉100kg"
                return raw
            env.mock_llm(semantic=semantic)
        else:
            env.mock_llm(semantic_error=error)
        body = env.post("b03-fail-" + semantic_source, message=message, mode="react_readonly")
        assert len(env.llm.semantic_calls) == 1
        assert body["metrics"].get("execution_mode") == "limited", body["metrics"]
        assert body["metrics"].get("mode_reason") == "safety_assessment_unavailable"
        assert body["metrics"].get("react_status") == "limited"
        assert body["intent"] == "execution_limited"
        assert env.legacy.await_count == 0 and not env.states and not env.handlers
        assert env.snapshot() == before


@pytest.mark.parametrize("case", ["no_adapter", "no_config"])
def test_b03_missing_structured_semantics_is_limited_with_zero_write(tmp_path, monkeypatch, case):
    message = "记录卧推60kg，4组8次，然后告诉我怎么恢复"
    with environment(tmp_path, monkeypatch, "b03-no-semantics-" + case) as env:
        before = env.snapshot()
        if case == "no_adapter":
            env.mock_llm(semantic=False)
            assert not hasattr(env.llm, "complete_structured")
        else:
            env.mock_llm(semantic=default_semantics, configured=False)
            assert hasattr(env.llm, "complete_structured")
            assert env.llm.configured() is False
        body = env.post("b03-no-semantics-" + case, message=message, mode="react_readonly")
        assert env.llm.semantic_calls == []
        assert body["metrics"].get("execution_mode") == "limited", body["metrics"]
        assert body["metrics"].get("mode_reason") == "safety_assessment_unavailable"
        assert body["metrics"].get("react_status") == "limited"
        assert body["intent"] == "execution_limited" and body["model_used"] is False
        assert env.parse.await_count == 0 and env.shadow.await_count == 0
        assert env.legacy.await_count == 0 and not env.states and not env.handlers
        assert env.snapshot() == before


def test_b03_budget_stop_is_limited_with_zero_write(tmp_path, monkeypatch):
    from app.services.llm.budget import ModelBudgetError

    message = "记录卧推60kg，4组8次，然后告诉我怎么恢复"
    with environment(tmp_path, monkeypatch, "b03-budget") as env:
        before = env.snapshot()
        env.mock_llm(semantic_error=ModelBudgetError("model_budget_exhausted:request"))
        body = env.post("b03-budget", message=message, mode="react_readonly")
        assert len(env.llm.semantic_calls) == 1
        assert body["metrics"].get("execution_mode") == "limited"
        assert body["metrics"].get("mode_reason") == "semantic_budget_stopped"
        assert body["metrics"].get("react_status") == "limited"
        assert body["intent"] == "execution_limited" and body["model_used"] is False
        assert env.parse.await_count == 0 and env.legacy.await_count == 0
        assert not env.states and not env.handlers
        assert env.snapshot() == before
