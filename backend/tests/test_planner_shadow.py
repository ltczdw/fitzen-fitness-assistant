import asyncio
from unittest.mock import AsyncMock
import pytest
from app.core.config import settings
from app.schemas.chat import ChatRequest
from app.services.agent import agent as agent_module
from app.services.agent.deepseek_input_parser import ShadowPlannerOutput
from app.services.agent.deepseek_input_parser import PlannerLLM
from app.services.agent.planner_shadow import find_differences
from app.services.llm.client import LLMGatewayError, LLMErrorCategory
from app.services.agent.task_planner import plan_tasks_rule
from app.services.agent.intent import classify_intent_rule


def test_shadow_schema_rejects_entity_fields_and_requires_safety_flags():
    with pytest.raises(ValueError):
        ShadowPlannerOutput.model_validate({"tasks":[{"task_type":"nutrition_record","action":"record","text":"记录鸡蛋","is_negated":False,"is_hypothetical":False,"is_third_party":False,"foods":[{"name":"鸡蛋"}]}]})
    with pytest.raises(ValueError):
        ShadowPlannerOutput.model_validate({"tasks":[{"task_type":"nutrition_record","action":"record","text":"记录鸡蛋"}]})


def test_shadow_observation_cannot_replace_the_primary_model_plan(monkeypatch):
    monkeypatch.setattr(settings,"planner_shadow_enabled",True)
    agent=agent_module.FitnessAgent()
    message="记录两个鸡蛋，再安排晚餐"
    rule=plan_tasks_rule(message,classify_intent_rule(message))
    agent._parse_input=AsyncMock(return_value=None)
    agent.graph.plan_shadow=AsyncMock(return_value=({"tasks":[{
        "task_type":"workout_record", "action":"record", "text":"malicious shadow output",
        "is_negated":False, "is_hypothetical":False, "is_third_party":False,
    }]},"ok"))
    async def run():
        token=agent_module._request_route.set(None)
        try:
            assert await agent._run_planner_shadow(message) is None
            decision=await agent._route(ChatRequest(message=message),{})
            # Shadow output cannot authorize writes. An unavailable primary
            # model yields an explicit nonwriting fallback.
            fields=("task_type","worker","action","text")
            assert [tuple(task.get(field) for field in fields) for task in decision.task_specs] == \
                   [tuple(task.get(field) for field in fields) for task in rule]
            assert decision.parsed_input=={}
        finally:
            agent_module._request_route.reset(token)
    asyncio.run(run())
    agent._parse_input.assert_awaited_once_with(message)
    agent.graph.plan_shadow.assert_awaited_once()


def test_shadow_comparison_cannot_be_returned_to_caller(monkeypatch):
    monkeypatch.setattr(settings,"planner_shadow_enabled",True)
    agent=agent_module.FitnessAgent()
    agent.graph.plan_shadow=AsyncMock(return_value=({"tasks":[{"task_type":"workout_record","action":"record","text":"rewrite","is_negated":False,"is_hypothetical":False,"is_third_party":False}]},"ok"))
    async def run():
        assert await agent._run_planner_shadow("昨天练了卧推") is None
    asyncio.run(run())
    agent.graph.plan_shadow.assert_awaited_once()


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (LLMErrorCategory.CONFIG, "unconfigured"),
        (LLMErrorCategory.TRANSPORT, "timeout"),
        (LLMErrorCategory.UPSTREAM, "upstream_http_402"),
        (LLMErrorCategory.PROTOCOL, "provider_protocol_error"),
    ],
)
def test_shadow_status_separates_gateway_failures(category, expected):
    parser = PlannerLLM()
    parser.llm.configured = lambda: True
    parser.llm.complete = AsyncMock(side_effect=LLMGatewayError("sanitized", category, status_code=402))
    result, status = asyncio.run(parser.plan_shadow("记录两个鸡蛋"))
    assert result is None
    assert status == expected


def test_shadow_status_separates_unconfigured_and_invalid_json():
    parser = PlannerLLM()
    parser.llm.configured = lambda: False
    parser.llm.complete = AsyncMock(side_effect=AssertionError("unconfigured client must not call upstream"))
    result, status = asyncio.run(parser.plan_shadow("记录鸡蛋"))
    assert result is None
    assert status == "unconfigured"
    parser.llm.configured = lambda: True
    parser.llm.complete = AsyncMock(return_value="not JSON")
    result, status = asyncio.run(parser.plan_shadow("记录鸡蛋"))
    assert result is None
    assert status == "invalid_json"


def test_comparison_reports_required_difference_dimensions():
    source="我昨天吃了鸡蛋，分析晚餐，再说假如的话"
    rules=[{"task_type":"nutrition_advice","action":"question","text":source}]
    shadow=[
        {"task_type":"nutrition_record","action":"record","text":"昨天吃了鸡蛋"},
        {"task_type":"nutrition_advice","action":"answer","text":"改写后的晚餐分析"},
    ]
    differences=find_differences(source,rules,shadow)
    assert "task_count" in differences
    assert "task_type" in differences
    assert "action" in differences
    assert "text_not_contiguous_original" in differences
    assert "llm_record_rule_non_record" in differences


def test_comparison_aligns_tasks_by_source_spans_not_list_position():
    source="补记昨天卧推，再记录今天深蹲"
    rules=[{"task_type":"workout_record","action":"record","text":source}]
    llm=[
        {"task_type":"workout_record","action":"record","text":"补记昨天卧推"},
        {"task_type":"workout_record","action":"record","text":"今天深蹲"},
    ]
    differences=find_differences(source,rules,llm)
    assert "task_count" in differences
    assert "llm_record_rule_non_record" not in differences


def test_comparison_does_not_call_separate_record_commands_or_meals_one_meal_splits():
    source="不要记录鸡蛋；记录午餐米饭，再帮我算蛋白质"
    rules=[{"task_type":"nutrition_advice","action":"question","text":source}]
    llm=[
        {"task_type":"nutrition_record","action":"record","text":"不要记录鸡蛋","is_negated":True},
        {"task_type":"nutrition_record","action":"record","text":"记录午餐米饭"},
        {"task_type":"nutrition_advice","action":"question","text":"再帮我算蛋白质"},
    ]
    differences=find_differences(source,rules,llm)
    assert "llm_record_with_unsafe_flag" in differences
    assert "one_meal_split_into_multiple_food_tasks" not in differences

    two_meals="记录今天早餐两个鸡蛋，午餐一碗米饭"
    two_meal_llm=[
        {"task_type":"nutrition_record","action":"record","text":"记录今天早餐两个鸡蛋"},
        {"task_type":"nutrition_record","action":"record","text":"午餐一碗米饭"},
    ]
    two_meal_rules=[{"task_type":"nutrition_record","action":"record","text":two_meals}]
    assert "one_meal_split_into_multiple_food_tasks" not in find_differences(two_meals,two_meal_rules,two_meal_llm)
