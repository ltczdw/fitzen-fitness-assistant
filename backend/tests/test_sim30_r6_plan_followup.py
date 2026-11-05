"""SIM30 R6: a preview edit may shorten only the named day."""
import json

import pytest

from app.services.agent.task_information import (
    PlanFact, PlanInformation, PlanInformationUnbound, _resolve_plan_information,
)
from app.services.agent.training_timing import parse_timed_plan, render_timed_plan


def _session(label, minutes):
    return {
        "label": label,
        "warmup": {"seconds": 240, "activity": "走动与关节活动"},
        "main": {
            "seconds": (minutes * 60) - 600,
            "exercises": [
                {"name": "哑铃深蹲", "sets": 2, "reps": 10},
                {"name": "墙壁俯卧撑", "sets": 2, "reps": 10},
            ],
        },
        "breaks": {"seconds": 240},
        "cooldown": {"seconds": 120, "activity": "慢走与拉伸"},
        "time_consistency": "consistent",
    }


def _raw(first=30, second=25):
    return json.dumps({"sessions": [_session("周二", first), _session("周六", second)]})


def test_preview_edit_keeps_the_unmodified_day_at_its_own_limit():
    rendered = render_timed_plan(
        _raw(), limit_seconds=1500, expected_sessions=2,
        expected_weekdays=["周二", "周六"], session_limits=[1800, 1500])
    first, second = rendered.split("周六")
    assert "不超过30分钟" in first
    assert "不超过25分钟" in second
    assert "每次上限" not in rendered


def test_a_session_over_its_own_limit_still_fails():
    with pytest.raises(ValueError, match="session_time_exceeded"):
        parse_timed_plan(
            _raw(first=30, second=30), limit_seconds=1500, expected_sessions=2,
            session_limits=[1800, 1500])


def _information(day_label):
    return PlanInformation(
        scope="session",
        facts=[PlanFact(field="session_time", source_id="current",
                        quote="把第二天缩到25分钟", quantity_quote="25分钟",
                        seconds=1500, day_label=day_label)],
    )


def test_session_time_can_name_one_preview_day():
    source = "把第二天缩到25分钟"
    details = _resolve_plan_information(
        _information("周六"), {"current": source}, {"current": source},
        source, source, {"周二", "周六"})
    assert details["session_time_seconds"] == 1500
    assert details["session_time_day"] == "周六"


def test_a_day_outside_the_current_preview_is_rejected():
    source = "把第二天缩到25分钟"
    with pytest.raises(PlanInformationUnbound):
        _resolve_plan_information(
            _information("周日"), {"current": source}, {"current": source},
            source, source, {"周二", "周六"})


def _preview_understanding(dialogue_ids):
    return {
        "tasks": [{
            "task_type": "training_plan", "delivery_type": "preview",
            "goal_role": "result", "preview_ref": "preview-1",
            "history_requirement": "none", "context_ids": [],
        }],
        "dialogue_ids": list(dialogue_ids),
        "critical_history_ids": [],
    }


def test_preview_edit_uses_application_state_not_wiki():
    from app.services.agent import semantics

    wrapper = {"status": "ok",
               "state": {"preview_id": "preview-1", "message_id": "turn-1:assistant"}}
    value = _preview_understanding(["turn-9:assistant"])
    assert semantics._history_required(value, {"current_plan_preview": wrapper}) is False
    # A non-preview goal still treats an unknown address as history-dependent.
    value["tasks"][0] = {"task_type": "chitchat", "history_requirement": "none",
                         "context_ids": [], "goal_role": "result"}
    assert semantics._history_required(value, {"current_plan_preview": wrapper}) is True


def test_cached_preview_request_does_not_read_wiki(monkeypatch):
    import asyncio

    from app.services.memory import wiki_context

    async def forbidden(*args, **kwargs):
        raise AssertionError("preview-only request must not read Wiki pages")

    monkeypatch.setattr(wiki_context, "read_pages", forbidden)
    wrapper = {"status": "ok",
               "state": {"preview_id": "preview-1", "message_id": "turn-1:assistant"}}
    memory = {"wiki_catalog": {"entries": [{"page_id": "p1"}]},
              "current_plan_preview": wrapper}
    asyncio.run(wiki_context.prepare_cached(
        "把第二天缩到25分钟", memory, _preview_understanding(["turn-9:assistant"]),
        "semantic prompt"))


def test_missing_safety_gets_a_bounded_correction():
    import asyncio

    from app.services.agent import semantics
    from app.services.llm.client import LLMClient

    source = "给我讲讲深蹲的通用要点"
    task = {"task_type": "chitchat", "action": "question", "text": source,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            "goal_role": "result", "history_requirement": "none",
            "reply_mode": "conversation", "reply_purpose": "general"}
    without = {"tasks": [task], "reply_mode": "conversation", "reply_purpose": "general"}
    with_safety = {**without, "safety": {"level": "normal", "evidence": "",
                                        "categories": [], "reason": "普通动作问题"}}

    class Fake(LLMClient):
        def __init__(self):
            self.calls = 0
            self.tool_schemas = []

        def configured(self):
            return True

        async def complete_message(self, messages, **kwargs):
            self.calls += 1
            self.tool_schemas.append(kwargs["tools"][0]["function"]["parameters"])
            payload = without if self.calls == 1 else with_safety
            return {"tool_calls": [{"type": "function", "function": {
                "name": "return_request_understanding",
                "arguments": json.dumps(payload, ensure_ascii=False)}}]}

    client = Fake()
    context = {"short_memory": [], "long_memory": {}}
    result = asyncio.run(semantics.understand(client, source, context=context))
    assert client.calls == 2
    schema = client.tool_schemas[0]
    assert "safety" in schema["required"]
    assert schema["properties"]["safety"]["$ref"] == "#/$defs/SafetyAssessment"
    assert "anyOf" not in schema["properties"]["safety"]
    assert result is not None
    assert result["safety"]["level"] == "normal"


def test_repeated_missing_safety_is_not_assessed_or_retried_again():
    import asyncio
    from app.services.agent import semantics
    from app.services.llm.client import LLMClient

    source = "请说明力量训练的基本动作"
    task = {"task_type": "chitchat", "action": "question", "text": source,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            "goal_role": "result", "history_requirement": "none",
            "reply_mode": "conversation", "reply_purpose": "general"}

    class Fake(LLMClient):
        def __init__(self):
            self.calls = 0

        def configured(self):
            return True

        async def complete_message(self, messages, **kwargs):
            self.calls += 1
            return {"tool_calls": [{"type": "function", "function": {
                "name": "return_request_understanding",
                "arguments": json.dumps({"tasks": [task]}, ensure_ascii=False)}}]}

    client = Fake()
    token = semantics.request_understanding.set({})
    try:
        context = {"short_memory": [], "long_memory": {}}
        assert asyncio.run(semantics.understand(client, source, context=context)) is None
        assert client.calls == 2
        assert context["semantic_failure_code"] == "semantic_safety_missing"
        assert semantics.safety_for(source)["status"] == "unavailable"
        assert asyncio.run(semantics.understand(client, source, context=context)) is None
        assert client.calls == 2
    finally:
        semantics.request_understanding.reset(token)


def test_plan_confirm_conversation_is_a_capability_boundary_not_a_write():
    from app.services.agent import semantics
    from app.services.agent.task_models import is_plan_save_capability_task

    source = "请把刚才对话里改成第二天25分钟的两天预览保存成正式计划。"
    raw = {
        "tasks": [{
            "task_type": "training_plan", "action": "confirm", "text": source,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            "plan_result": False, "goal_role": "result", "history_requirement": "required",
            "reply_mode": "conversation", "delivery_type": "preview",
            "preview_ref": "preview-1",
        }],
        "safety": {"level": "normal", "evidence": "", "categories": [],
                   "reason": "保存边界"},
    }
    context = {
        "current_plan_preview": {"status": "ok",
                                 "state": {"preview_id": "preview-1",
                                           "message_id": "turn-1:assistant"}},
        "short_memory": [], "long_memory": {},
    }
    value = semantics.validate_understanding(source, raw, context=context)
    task = value["tasks"][0]
    assert task["action"] == "confirm"
    assert is_plan_save_capability_task(task) is True


def test_preview_session_limits_apply_only_the_named_day():
    from app.services.agent.multi_agent_graph import (
        _preview_session_limits, _preview_weekdays,
    )

    base = {
        "weekdays": ["周二", "周六"],
        "days": [{"label": "周二", "minutes": 30},
                 {"label": "周六", "minutes": 30}],
    }
    assert _preview_session_limits(base, 2, "preview", 1500, "周六") == [1800, 1500]
    # No resolved day: the requested cap applies to the whole preview.
    assert _preview_session_limits(base, 2, "preview", 1500, None) == [1500, 1500]
    # Unknown day is not silently treated as the named day.
    assert _preview_session_limits(base, 2, "preview", 1500, "周日") == [1500, 1500]
    # Reversed wording must not reorder the stored sessions.
    assert _preview_weekdays(base, "preview", "周六25分钟，周二30分钟") == ["周二", "周六"]


def test_production_client_exposes_the_format_retry_option():
    import inspect

    from app.services.llm.client import LLMClient

    assert "allow_json_retry" in inspect.signature(LLMClient.complete_structured).parameters


def test_preview_result_drops_redundant_assistant_pointer():
    from app.services.agent import semantics

    source = ("沿用刚才的两天预览，把第二天缩到25分钟，保留热身和收尾，"
              "减少一个动作并说明删了什么。只改这份预览，不保存。")
    raw = {
        "tasks": [{
            "task_type": "training_plan", "action": "answer", "text": source,
            "delivery_type": "preview", "preview_ref": "preview-1",
            "goal_role": "result", "history_requirement": "none",
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        }],
        "dialogue_ids": ["turn-9:assistant", "turn-8:user"],
        "safety": {"level": "normal", "evidence": "", "categories": [], "reason": "x"},
    }
    context = {
        "short_memory": [
            {"message_id": "turn-8:user", "role": "user", "content": "目标保持基础力量"},
            {"message_id": "turn-9:assistant", "role": "assistant", "content": "previous preview",
             "app_meta":{"plan_preview":{"preview_id":"preview-1"}}},
        ],
        "current_plan_preview": {"status": "ok",
                                 "state": {"preview_id": "preview-1",
                                           "message_id": "turn-9:assistant"}},
        "long_memory": {},
    }
    value = semantics.validate_understanding(source, raw, context=context)
    assert value["dialogue_ids"] == ["turn-8:user"]
