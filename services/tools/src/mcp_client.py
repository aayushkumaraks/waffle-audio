"""MCP (Model Context Protocol) client implementation for connecting to external tool servers."""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MCPToolDefinition:
    """Represents a tool exposed by an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


class MCPClientManager:
    """Manages connections and tool execution across configured MCP servers."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    async def list_tools_for_server(
        self, config: MCPServerConfig
    ) -> list[MCPToolDefinition]:
        """Connect to an MCP server and query its exposed tools."""
        if config.url:
            return await self._list_tools_sse(config)
        elif config.command:
            return await self._list_tools_stdio(config)
        else:
            raise ValueError(f"Server '{config.name}' must have either 'command' or 'url' specified.")

    async def _list_tools_stdio(
        self, config: MCPServerConfig
    ) -> list[MCPToolDefinition]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        merged_env = os.environ.copy()
        merged_env.update(config.env)

        server_params = StdioServerParameters(
            command=config.command or "",
            args=config.args,
            env=merged_env,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=self.timeout)
                response = await asyncio.wait_for(session.list_tools(), timeout=self.timeout)

                tools: list[MCPToolDefinition] = []
                for tool in response.tools:
                    tools.append(
                        MCPToolDefinition(
                            name=tool.name,
                            description=tool.description or f"Tool provided by {config.name}",
                            input_schema=getattr(tool, "input_schema", getattr(tool, "inputSchema", {})) or {"type": "object", "properties": {}},
                            server_name=config.name,
                        )
                    )
                return tools

    async def _list_tools_sse(
        self, config: MCPServerConfig
    ) -> list[MCPToolDefinition]:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        async with sse_client(config.url or "", headers=config.headers) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=self.timeout)
                response = await asyncio.wait_for(session.list_tools(), timeout=self.timeout)

                tools: list[MCPToolDefinition] = []
                for tool in response.tools:
                    tools.append(
                        MCPToolDefinition(
                            name=tool.name,
                            description=tool.description or f"Tool provided by {config.name}",
                            input_schema=getattr(tool, "input_schema", getattr(tool, "inputSchema", {})) or {"type": "object", "properties": {}},
                            server_name=config.name,
                        )
                    )
                return tools

    async def call_tool(
        self, config: MCPServerConfig, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """Call a specific tool on the given MCP server."""
        if config.url:
            return await self._call_tool_sse(config, tool_name, arguments)
        elif config.command:
            return await self._call_tool_stdio(config, tool_name, arguments)
        else:
            raise ValueError(f"Server '{config.name}' must have either 'command' or 'url' specified.")

    async def _call_tool_stdio(
        self, config: MCPServerConfig, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        merged_env = os.environ.copy()
        merged_env.update(config.env)

        server_params = StdioServerParameters(
            command=config.command or "",
            args=config.args,
            env=merged_env,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=self.timeout)
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments=arguments),
                    timeout=self.timeout,
                )

                text_outputs: list[str] = []
                for c in result.content:
                    if hasattr(c, "text"):
                        text_outputs.append(c.text)
                    else:
                        text_outputs.append(str(c))

                output = "\n".join(text_outputs).strip()
                if getattr(result, "is_error", getattr(result, "isError", False)):
                    return f"Tool execution error: {output}"
                return output or "Tool executed successfully with no textual output."

    async def _call_tool_sse(
        self, config: MCPServerConfig, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        async with sse_client(config.url or "", headers=config.headers) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=self.timeout)
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments=arguments),
                    timeout=self.timeout,
                )

                text_outputs: list[str] = []
                for c in result.content:
                    if hasattr(c, "text"):
                        text_outputs.append(c.text)
                    else:
                        text_outputs.append(str(c))

                output = "\n".join(text_outputs).strip()
                if getattr(result, "is_error", getattr(result, "isError", False)):
                    return f"Tool execution error: {output}"
                return output or "Tool executed successfully with no textual output."

    def list_tools_sync(self, config: MCPServerConfig) -> list[MCPToolDefinition]:
        """Synchronous wrapper for list_tools_for_server."""
        return asyncio.run(self.list_tools_for_server(config))

    def call_tool_sync(
        self, config: MCPServerConfig, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """Synchronous wrapper for call_tool."""
        return asyncio.run(self.call_tool(config, tool_name, arguments))
