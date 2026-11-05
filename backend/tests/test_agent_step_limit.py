import asyncio
from unittest.mock import AsyncMock

import pytest

from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
from app.services.llm.client import LLMClient


TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {}}}]
CALL = {"content": "", "tool_calls": [
    {"id": "call", "function": {"name": "read", "arguments": "{}"}}
]}


@pytest.mark.parametrize("requested,expected", [(6, 6), (99, 6), (2, 2)])
def test_exhausted_loop_stops_before_last_tools(monkeypatch, requested, expected):
    llm = LLMClient()
    monkeypatch.setattr(llm, "configured", lambda: True)
    model = AsyncMock(return_value=CALL)
    execute = AsyncMock(return_value={"count": 1})
    monkeypatch.setattr(llm, "_complete_message", model)

    result = asyncio.run(llm._complete_with_tools_impl(
        [], tools=TOOLS, tool_executor=execute, max_tool_rounds=requested,
    ))

    assert model.await_count == expected
    assert execute.await_count == expected - 1
    assert result["error"] == "max_tool_rounds"
    assert len(result["tool_results"]) == expected - 1


def test_final_answer_on_sixth_step_is_success(monkeypatch):
    llm = LLMClient()
    monkeypatch.setattr(llm, "configured", lambda: True)
    model = AsyncMock(side_effect=[CALL] * 5 + [{"content": "done"}])
    monkeypatch.setattr(llm, "_complete_message", model)
    result = asyncio.run(llm._complete_with_tools_impl(
        [], tools=TOOLS, tool_executor=AsyncMock(return_value={}),
    ))
    assert model.await_count == 6
    assert result["content"] == "done"
    assert "error" not in result


@pytest.mark.parametrize("write", [False, True])
def test_worker_limit_preserves_results_and_does_not_fallback(monkeypatch, write):
    graph = FitnessMultiAgentGraph()
    monkeypatch.setattr(graph.llm, "configured", lambda: True)
    completed = [{"name": "save_workout_record", "result": {"saved": True}}]
    bridge = AsyncMock(return_value={
        "error": "max_tool_rounds", "content": "limit reached",
        "tool_results": completed,
    })
    monkeypatch.setattr(graph.mcp, "complete_with_tools", bridge)
    fallback = AsyncMock()
    monkeypatch.setattr(graph, "_complete_or_fallback", fallback)

    async def node(state):
        if write:
            await graph._run_authorized_model_tool(
                state, "save_workout_record", [{"record": {}}] * 9,
            )
        else:
            await graph._complete_or_fallback_with_mcp([], "fallback", state)
        pytest.fail("Exhausted task must not continue to a write fallback")

    result = asyncio.run(graph._new_worker(node).ainvoke({"user_id": "fixture"}))
    assert result["result"]["status"] == "max_tool_rounds"
    assert result["result"]["tool_results"] == completed
    fallback.assert_not_called()
    if write:
        assert bridge.call_args.kwargs["max_tool_rounds"] == 6
