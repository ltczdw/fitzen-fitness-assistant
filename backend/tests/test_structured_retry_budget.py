import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import settings
from app.services.agent.deepseek_input_parser import PlannerLLM, ParsedFoodResponse
from app.services.llm.client import LLMClient


def test_invalid_structure_gets_one_correction():
    llm = type('FakeLLM', (), {'configured': lambda self: True})()
    llm.complete = AsyncMock(side_effect=['not json', '{"foods":[{"name":"synthetic-food"}]}'])
    result = asyncio.run(PlannerLLM(llm)._call_json('original instructions', ParsedFoodResponse, 'synthetic request'))
    assert result['foods'][0]['name'] == 'synthetic-food'
    assert llm.complete.await_count == 2
    second = llm.complete.call_args_list[1].args[0]
    assert '不新增任务或实体' in second[0]['content']
    assert llm.complete.call_args_list[0].args[0][1] == second[1]


def test_repeated_invalid_structure_stops_after_two():
    llm = type('FakeLLM', (), {'configured': lambda self: True})()
    llm.complete = AsyncMock(return_value='{"foods":[{"weight_g":-1}]}')
    assert asyncio.run(PlannerLLM(llm)._call_json('instructions', ParsedFoodResponse, 'synthetic request')) is None
    assert llm.complete.await_count == 2


@pytest.mark.parametrize('status', [401, 402, 503])
def test_transport_and_structure_repair_share_two_physical_calls(monkeypatch, status):
    monkeypatch.setattr(settings, 'llm_api_key', 'mock-not-real')
    monkeypatch.setattr(settings, 'llm_base_url', f'https://structured-{status}.invalid/v1')
    monkeypatch.setattr(settings, 'llm_model', 'mock')
    monkeypatch.setattr(settings, 'llm_retry_delays', '0,0,0')
    seen = []
    original = httpx.AsyncClient

    def handle(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(status)
        return httpx.Response(200, json={'choices': [{'message': {'content': 'invalid JSON'}}],
                                       'usage': {'prompt_tokens': 4, 'completion_tokens': 2}})

    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    llm = LLMClient()
    llm.admission.token = AsyncMock()
    assert asyncio.run(PlannerLLM(llm)._call_json('instructions', ParsedFoodResponse, 'synthetic request')) is None
    assert len(seen) == (2 if status == 503 else 1)


def test_valid_structure_requires_one_call():
    llm = type('FakeLLM', (), {'configured': lambda self: True})()
    llm.complete = AsyncMock(return_value='{"foods":[]}')
    assert asyncio.run(PlannerLLM(llm)._call_json('instructions', ParsedFoodResponse, 'synthetic request')) == {'foods': []}
    assert llm.complete.await_count == 1
