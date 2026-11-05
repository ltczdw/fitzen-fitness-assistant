from collections.abc import Callable
from typing import Any

import logging
from app.db.storage import CommitUnknownError
from app.services.mcp.executor import READ_TOOLS, TOOL_SCHEMAS, WRITE_TOOLS, ToolExecutionGateway

logger = logging.getLogger(__name__)
import httpx2
from fastmcp import Client
from fastmcp.client.transports.http import StreamableHttpTransport

from app.core.config import settings


class MCPClientBridge:
    """Agent 的 MCP 客户端；服务不可用时由上层切换到本地工具。"""

    def __init__(self) -> None:
        self._tool_cache: list[dict[str, Any]] | None = None
        self._tool_schemas: dict[str, dict[str, Any]] = {}
        self._executor = ToolExecutionGateway()

    @staticmethod
    def _http_client_factory(**kwargs):
        # A local MCP endpoint must not be sent through a machine-wide proxy.
        # httpx2 inherits proxy settings by default, which can turn localhost
        # MCP requests into opaque 502 responses on Windows.
        kwargs["trust_env"] = False
        return httpx2.AsyncClient(**kwargs)

    @classmethod
    def _new_client(cls):
        return Client(
            StreamableHttpTransport(
                settings.mcp_url,
                httpx_client_factory=cls._http_client_factory,
            ),
            timeout=settings.mcp_timeout_seconds,
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any | None:
        """Low-level remote transport; application tool calls use execute_tool()."""
        if not settings.mcp_url:
            return None
        dispatched = False
        try:
            async with self._new_client() as client:
                dispatched = True
                result = await client.call_tool(name, arguments)
            if getattr(result, "is_error", False):
                raise RuntimeError("MCP returned a tool error")
            data = getattr(result, "data", None)
            if data is None:
                data = getattr(result, "structured_content", None)
            if data is None and name in WRITE_TOOLS:
                raise RuntimeError("MCP write result missing")
            return data
        except Exception as exc:
            logger.exception("mcp_tool_failed", extra={"tool_name": name, "dispatched": dispatched})
            if dispatched and name in WRITE_TOOLS:
                raise CommitUnknownError("Remote write outcome unknown; local replay refused") from exc
            return None

    async def list_tools(self, *, force_refresh: bool = False) -> list[dict[str, Any]]:
        """Discover the remote MCP tools and expose OpenAI-compatible schemas."""
        if self._tool_cache is not None and not force_refresh:
            return list(self._tool_cache)
        if not settings.mcp_url:
            return []
        try:
            async with self._new_client() as client:
                discovered = await client.list_tools()
            tools: list[dict[str, Any]] = []
            for tool in discovered:
                name = str(getattr(tool, "name", "") or "").strip()
                if not name:
                    continue
                schema = getattr(tool, "input_schema", None) or getattr(tool, "parameters", None)
                if not isinstance(schema, dict):
                    schema = TOOL_SCHEMAS.get(name, {})
                if schema:
                    self._tool_schemas[name] = schema
                tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(getattr(tool, "description", "") or ""),
                        "parameters": schema,
                    },
                })
            self._tool_cache = tools
            return list(tools)
        except Exception:
            logger.exception("mcp_tool_discovery_failed")
            return []

    async def execute_tool(
        self, name: str, arguments: dict[str, Any], *,
        local_handler: Callable[..., Any] | None = None,
        trusted_user_id: str | None = None, write_authorized: bool = False,
        request_id: str | None = None, task_id: str | None = None,
    ) -> Any:
        """Run one known tool through the shared gateway, with safe local fallback."""
        async def remote(args):
            if not settings.mcp_url:
                return None
            return await self.call_tool(name, args)

        async def local(args):
            if local_handler is None:
                return None
            import asyncio
            return await asyncio.to_thread(local_handler, **args)

        return await self._executor.execute(
            name, arguments, schema=self._tool_schemas.get(name) or TOOL_SCHEMAS.get(name),
            operation=remote, fallback=local if local_handler is not None else None,
            allowed_tools={name}, write_authorized=write_authorized,
            trusted_user_id=trusted_user_id, request_id=request_id, task_id=task_id,
        )

    async def complete_with_tools(
        self,
        llm,
        messages: list[dict[str, Any]],
        *,
        max_tool_rounds: int = 6,
        allowed_tools: set[str] | None = None,
        blocked_tools: set[str] | None = None,
        authorized_write_tools: set[str] | None = None,
        tool_context: dict[str, Any] | None = None,
        argument_rewriter: Callable[[str, dict[str, Any], int], dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Let the model choose discovered MCP tools through a tool-call loop."""
        tools = await self.list_tools()
        # Keep blocked tools in the model-visible schema so the executor can
        # return an explicit permission error; execute() remains the hard gate.
        visible_names = set(allowed_tools) if allowed_tools is not None else set(READ_TOOLS)
        known_tools = [tool for tool in tools
                       if tool.get("function", {}).get("name") in (READ_TOOLS | WRITE_TOOLS)]
        if allowed_tools is not None:
            tools = [tool for tool in known_tools if tool.get("function", {}).get("name") in visible_names]
        else:
            # Keep explicitly blocked writes visible so the model receives a
            # clear refusal, while ordinary read turns only advertise read tools.
            blocked_visible = set(blocked_tools or ()) & set(WRITE_TOOLS)
            tools = [tool for tool in known_tools
                     if tool.get("function", {}).get("name") in (set(READ_TOOLS) | blocked_visible)]
        if not tools:
            return None

        authorized_write_tools = authorized_write_tools or set()
        call_index = 0

        async def execute(name: str, arguments: dict[str, Any]) -> Any:
            nonlocal call_index
            if blocked_tools and name in blocked_tools:
                raise PermissionError(f"MCP tool is blocked for this task: {name}")
            if name not in visible_names:
                raise PermissionError(f"MCP tool is not allowed for this task: {name}")
            arguments = dict(arguments)
            if argument_rewriter is not None:
                arguments = argument_rewriter(name, arguments, call_index)
            call_index += 1
            schema = self._tool_schemas.get(name) or TOOL_SCHEMAS.get(name)
            result = await self._executor.execute(
                name, arguments, schema=schema,
                operation=lambda safe_args: self.call_tool(name, safe_args),
                allowed_tools=visible_names, blocked_tools=blocked_tools,
                write_authorized=name in authorized_write_tools,
                trusted_user_id=(tool_context or {}).get("user_id"),
                request_id=(tool_context or {}).get("request_id"),
                task_id=(tool_context or {}).get("task_id"),
            )
            return result

        return await llm.complete_with_tools(
            messages,
            tools=tools,
            tool_executor=execute,
            max_tool_rounds=max_tool_rounds,
        )
