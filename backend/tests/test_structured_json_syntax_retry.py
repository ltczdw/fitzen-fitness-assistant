"""Tests for structured json syntax retry."""
import asyncio
import json

import pytest
from pydantic import BaseModel

from app.services.llm.structured import StructuredOutputError, complete_structured


class Result(BaseModel):
    value: int


def _message(arguments: str, name: str = "result"):
    return {"tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def complete_message(self, messages, **kwargs):
        self.prompts.append(messages)
        return self.responses.pop(0)


def test_json_syntax_failure_can_use_one_final_format_retry():
    client = FakeClient([
        _message('{"value":'),
        _message('{"value":1,}'),
        _message('{"value": 1}'),
    ])
    result = asyncio.run(complete_structured(
        client, [{"role": "user", "content": "return value 1"}], Result,
        name="result", allow_json_retry=True))
    assert result == Result(value=1)
    assert len(client.prompts) == 3
    assert "FINAL FORMAT RETRY" in str(client.prompts[-1][0]["content"])


def test_default_keeps_two_attempts_and_fails_closed():
    client = FakeClient([_message('{"value":'), _message('{"value":1,}')])
    with pytest.raises(StructuredOutputError) as raised:
        asyncio.run(complete_structured(
            client, [{"role": "user", "content": "x"}], Result, name="result"))
    assert raised.value.code == "structured_invalid_json"
    assert len(client.prompts) == 2


def test_schema_error_does_not_get_the_extra_json_attempt():
    client = FakeClient([
        _message('{"value":"bad"}'),
        _message('{"value":"still-bad"}'),
        _message('{"value": 1}'),
    ])
    with pytest.raises(StructuredOutputError) as raised:
        asyncio.run(complete_structured(
            client, [{"role": "user", "content": "x"}], Result,
            name="result", allow_json_retry=True))
    assert raised.value.code == "structured_invalid_arguments"
    assert len(client.prompts) == 2


def test_initial_schema_failure_does_not_enable_the_format_attempt():
    client = FakeClient([
        _message('{"value":"bad"}'),
        _message('{"value":'),
        _message('{"value": 1}'),
    ])
    with pytest.raises(StructuredOutputError) as raised:
        asyncio.run(complete_structured(
            client, [{"role": "user", "content": "x"}], Result,
            name="result", allow_json_retry=True))
    assert raised.value.code == "structured_invalid_json"
    assert len(client.prompts) == 2


def test_third_invalid_json_still_fails_closed():
    client = FakeClient([
        _message('{"value":'),
        _message('{"value":1,}'),
        _message('{"value":1,}'),
    ])
    with pytest.raises(StructuredOutputError) as raised:
        asyncio.run(complete_structured(
            client, [{"role": "user", "content": "x"}], Result,
            name="result", allow_json_retry=True))
    assert raised.value.code == "structured_invalid_json"
    assert len(client.prompts) == 3


def test_prompt_contains_one_top_level_object_example():
    from app.services.agent import semantics
    assert "只示意结构" in semantics.PROMPT
    assert "tasks 数组闭合后不得再出现第二个顶层对象" in semantics.PROMPT
    json.loads('{"tasks": [], "safety": {}}')  # sanity: example shape is valid JSON
