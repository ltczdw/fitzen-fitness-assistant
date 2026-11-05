import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
from app.services.memory import compact


def test_tool_observation_is_persisted(monkeypatch):
    graph = object.__new__(FitnessMultiAgentGraph)
    graph.mcp = SimpleNamespace(call_tool=AsyncMock(return_value={"saved": True}))
    append = Mock()
    monkeypatch.setattr(compact, "append_observation", append)
    async def run():
        token = compact.current_turn.set("turn-observation")
        try:
            assert await graph._call_mcp("create_workout", {"exercise": "test"}) == {"saved": True}
        finally:
            compact.current_turn.reset(token)
    asyncio.run(run())
    append.assert_called_once_with("turn-observation", "create_workout", {"exercise": "test"}, {"saved": True})
