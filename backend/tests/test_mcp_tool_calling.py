import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.llm.client import LLMClient
from app.services.mcp.client import MCPClientBridge


class MCPToolCallingTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_exposes_all_registered_mcp_tools(self):
        from app.services.mcp.server import mcp

        tools = await mcp.list_tools()
        self.assertEqual(len(tools), 9)
        self.assertIn("calculate_nutrition", {tool.name for tool in tools})

    async def test_bridge_converts_tools_list_to_model_schemas(self):
        discovered = [SimpleNamespace(
            name="get_training_history",
            description="读取训练历史",
            input_schema={"type": "object", "properties": {"user_id": {"type": "string"}}},
        )]

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def list_tools(self):
                return discovered

        with patch("app.services.mcp.client.Client", return_value=FakeClient()), \
             patch("app.services.mcp.client.settings.mcp_url", "http://mcp.fixture"):
            bridge = MCPClientBridge()
            result = await bridge.list_tools()

        self.assertEqual(result[0]["type"], "function")
        self.assertEqual(result[0]["function"]["name"], "get_training_history")
        self.assertEqual(result[0]["function"]["parameters"], discovered[0].input_schema)

    async def test_llm_executes_tool_calls_and_returns_final_answer(self):
        # Use an isolated provider identity so this test cannot inherit an OPEN
        # breaker from another test. Production sharing and the dedicated
        # breaker tests remain unchanged.
        with patch("app.services.llm.client.settings.llm_base_url",
                   "https://mcp-isolation.invalid/v1"), \
             patch("app.services.llm.client.settings.llm_model", "mcp-isolation-model"), \
             patch("app.services.llm.client.settings.llm_api_key", "mcp-isolation-key"):
            llm = LLMClient()
            llm.configured = lambda: True
            responses = [
                {"content": "", "tool_calls": [{"id": "call-1", "function": {"name": "get_training_history", "arguments": '{"user_id":"wrong-user"}'}}]},
                {"content": "你最近完成了 3 次训练。", "tool_calls": []},
            ]

            async def fake_complete_message(*_args, **_kwargs):
                return responses.pop(0)

            llm._complete_message = fake_complete_message
            calls = []

            async def execute(name, arguments):
                calls.append((name, arguments))
                return {"count": 3}

            result = await llm.complete_with_tools(
                [{"role": "user", "content": "我最近训练了几次？"}],
                tools=[{"type": "function", "function": {"name": "get_training_history", "parameters": {}}}],
                tool_executor=execute,
            )

        self.assertEqual(result["content"], "你最近完成了 3 次训练。")
        self.assertEqual(calls, [("get_training_history", {"user_id": "wrong-user"})])
        self.assertEqual([message["role"] for message in result["messages"]], ["user", "assistant", "tool", "assistant"])

    async def test_bridge_context_overrides_model_user_id(self):
        bridge = MCPClientBridge()
        bridge.list_tools = _async_value([{"type": "function", "function": {"name": "get_user_profile", "parameters": {}}}])
        calls = []

        async def fake_call(name, arguments):
            calls.append((name, arguments))
            return {"user_id": arguments["user_id"]}

        bridge.call_tool = fake_call

        class FakeLLM:
            async def complete_with_tools(self, _messages, *, tools, tool_executor, **_kwargs):
                self.tools = tools
                await tool_executor("get_user_profile", {"user_id": "model-chosen-user"})
                return {"content": "ok"}

        await bridge.complete_with_tools(FakeLLM(), [{"role": "user", "content": "读取资料"}], tool_context={"user_id": "real-user"})
        self.assertEqual(calls, [("get_user_profile", {"user_id": "real-user"})])

    async def test_read_only_mcp_loop_blocks_write_tools(self):
        bridge = MCPClientBridge()
        bridge.list_tools = _async_value([{"type": "function", "function": {"name": "save_workout_record", "parameters": {}}}])
        testcase = self

        class FakeLLM:
            async def complete_with_tools(self, _messages, *, tool_executor, **_kwargs):
                with testcase.assertRaises(PermissionError):
                    await tool_executor("save_workout_record", {"record": {}})
                return {"content": "已拒绝"}

        result = await bridge.complete_with_tools(
            FakeLLM(),
            [{"role": "user", "content": "读取训练计划"}],
            blocked_tools={"save_workout_record"},
        )
        self.assertEqual(result["content"], "已拒绝")


def _async_value(value):
    async def getter(*_args, **_kwargs):
        return value

    return getter


if __name__ == "__main__":
    unittest.main()
