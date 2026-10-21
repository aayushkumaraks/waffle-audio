"""Central Tool Manager uniting MCP servers and DuckDuckGo search fallback."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from constants import (
    DUCKDUCKGO_MAX_RESULTS,
    DUCKDUCKGO_SEARCH_ENABLED,
    MCP_CONFIG_PATH,
    TOOLS_ENABLED,
)
from .mcp_client import MCPClientManager, MCPServerConfig, MCPToolDefinition
from .search import DuckDuckGoSearch

logger = logging.getLogger(__name__)


class ToolManager:
    """Discovers, caches, and routes tool calls across MCP servers and DuckDuckGo."""

    def __init__(
        self,
        config_path: Path | str | None = None,
        enable_search: bool = DUCKDUCKGO_SEARCH_ENABLED,
        enable_tools: bool = TOOLS_ENABLED,
        search_max_results: int = DUCKDUCKGO_MAX_RESULTS,
        timeout: float = 30.0,
    ) -> None:
        self.config_path = Path(config_path or MCP_CONFIG_PATH)
        self.enable_search = enable_search
        self.enable_tools = enable_tools
        self.search = DuckDuckGoSearch(max_results=search_max_results)
        self.mcp_client = MCPClientManager(timeout=timeout)

        self._lock = threading.Lock()
        self._servers: dict[str, MCPServerConfig] = {}
        # tool_name -> (server_config, tool_definition)
        self._tools: dict[str, tuple[MCPServerConfig, MCPToolDefinition]] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Load configuration and discover available MCP tools."""
        with self._lock:
            self._load_config()
            self._discover_all_tools()
            self._initialized = True

    def refresh(self) -> None:
        """Reload configuration and rediscover all tools."""
        self.initialize()

    def _load_config(self) -> None:
        """Read MCP configuration from disk."""
        self._servers.clear()
        if not self.config_path.exists():
            # Create empty template if it does not exist
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                self.config_path.write_text(json.dumps({"mcpServers": {}}, indent=2))
            except Exception as exc:
                logger.warning("Could not create empty MCP config at %s: %s", self.config_path, exc)
            return

        try:
            raw_text = self.config_path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
            servers_data = data.get("mcpServers", {})

            for name, cfg in servers_data.items():
                self._servers[name] = MCPServerConfig(
                    name=name,
                    command=cfg.get("command"),
                    args=cfg.get("args", []),
                    env=cfg.get("env", {}),
                    url=cfg.get("url"),
                    headers=cfg.get("headers", {}),
                )
            logger.info("Loaded %d MCP server configurations from %s.", len(self._servers), self.config_path)
        except Exception as exc:
            logger.error("Failed to parse MCP config from %s: %s", self.config_path, exc)

    def _discover_all_tools(self) -> None:
        """Query each configured MCP server for its exposed tools."""
        self._tools.clear()
        if not self.enable_tools:
            logger.info("Tool calling is disabled via configuration.")
            return

        for name, server_cfg in self._servers.items():
            try:
                tools = self.mcp_client.list_tools_sync(server_cfg)
                for tool in tools:
                    self._tools[tool.name] = (server_cfg, tool)
                logger.info("Discovered %d tools from MCP server '%s'.", len(tools), name)
            except Exception as exc:
                logger.warning("Failed to discover tools from MCP server '%s': %s", name, exc)

    @property
    def installed_tool_names(self) -> set[str]:
        """Return the set of names of currently installed MCP tools."""
        with self._lock:
            return set(self._tools.keys())

    def get_ollama_tools(self) -> list[dict[str, Any]]:
        """Return full tools schema in Ollama/OpenAI format."""
        if not self.enable_tools:
            return []

        with self._lock:
            if not self._initialized:
                self._load_config()
                self._discover_all_tools()
                self._initialized = True

            tools_list: list[dict[str, Any]] = []

            # 1. Add all discovered MCP tools
            for tool_name, (_, tool_def) in self._tools.items():
                tools_list.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_def.name,
                            "description": tool_def.description,
                            "parameters": tool_def.input_schema,
                        },
                    }
                )

            # 2. Add DuckDuckGo Search as a first-class search tool
            if self.enable_search:
                tools_list.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "duckduckgo_search",
                            "description": (
                                "Search the web using DuckDuckGo. Use this to find real-time facts, "
                                "current information, news, weather, general details, or when no other "
                                "installed tool can answer the user's question."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query string to look up.",
                                    }
                                },
                                "required": ["query"],
                            },
                        },
                    }
                )

            return tools_list

    def get_server_tools(self, server_name: str) -> list[dict[str, Any]]:
        """Return tool definitions belonging to a specific MCP server."""
        with self._lock:
            if not self._initialized:
                self._load_config()
                self._discover_all_tools()
                self._initialized = True

            res = []
            for tool_name, (server_cfg, tool_def) in self._tools.items():
                if server_cfg.name == server_name:
                    res.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool_def.name,
                                "description": tool_def.description,
                                "parameters": tool_def.input_schema,
                            },
                        }
                    )
            return res

    def _build_search_query(self, name: str, args: dict[str, Any]) -> str:
        """Construct an effective search query from a function name and arguments."""
        # 1. Check for explicit search query keys
        for key in ("query", "q", "search_query", "input", "prompt", "text"):
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # 2. Extract values from arguments, filtering out internal or empty values
        val_parts: list[str] = []
        for k, v in args.items():
            if isinstance(v, (str, int, float, bool)) and str(v).strip():
                val_parts.append(str(v).strip())

        # 3. Clean up tool name into human-readable intent
        import re
        clean_name = re.sub(r"^(get_|fetch_|check_|lookup_|search_|show_|read_)", "", name.lower())
        clean_name = clean_name.replace("_", " ").replace("-", " ").strip()

        if val_parts:
            combined_vals = " ".join(val_parts)
            if clean_name and clean_name not in combined_vals.lower():
                return f"{clean_name} {combined_vals}"
            return combined_vals

        return clean_name or name

    def execute_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Execute a tool by name, with automatic DuckDuckGo fallback."""
        args = arguments or {}
        name = (name or "").strip()

        logger.info("Executing tool: %s with arguments: %s", name, args)

        # 1. Built-in DuckDuckGo Search
        if name == "duckduckgo_search":
            query = self._build_search_query(name, args)
            return self.search.search(str(query))

        # 2. Check if tool is provided by an installed MCP server
        with self._lock:
            if not self._initialized and not self._tools:
                self._load_config()
                self._discover_all_tools()
                self._initialized = True
            tool_entry = self._tools.get(name)

        if tool_entry is not None:
            server_cfg, tool_def = tool_entry
            # Normalize missing required arguments for common MCP tools
            if name == "get_current_time" and "timezone" not in args:
                args = dict(args)
                args["timezone"] = "Etc/UTC"
            elif name == "convert_time":
                args = dict(args)
                if "source_timezone" not in args:
                    args["source_timezone"] = "Etc/UTC"
                if "target_timezone" not in args:
                    args["target_timezone"] = "Etc/UTC"
            try:
                result = self.mcp_client.call_tool_sync(server_cfg, name, args)
                return result
            except Exception as exc:
                logger.warning(
                    "MCP tool '%s' on server '%s' failed: %s. Falling back to DuckDuckGo search.",
                    name,
                    server_cfg.name,
                    exc,
                )
                query = self._build_search_query(name, args)
                fallback = self.search.search(str(query))
                return f"[Tool '{name}' error: {exc}]\nFallback DuckDuckGo Search:\n{fallback}"

        # 3. No installed tool matches this function -> Fallback to DuckDuckGo search!
        logger.info(
            "No installed MCP tool found for function '%s'. Falling back to DuckDuckGo search.",
            name,
        )
        query = self._build_search_query(name, args)
        ddg_result = self.search.search(str(query))
        return (
            f"No installed tool matching function '{name}'. "
            f"DuckDuckGo search results for '{query}':\n{ddg_result}"
        )

    def fallback_search(self, query: str) -> str:
        """Explicitly run a DuckDuckGo search."""
        return self.search.search(query)

    # ------------------------------------------------------------------
    # Configuration Management (Used by install_tool.py script)
    # ------------------------------------------------------------------

    def list_installed_servers(self) -> dict[str, dict[str, Any]]:
        """Return the dictionary of configured servers and their current status."""
        with self._lock:
            self._load_config()
            result: dict[str, dict[str, Any]] = {}
            for name, cfg in self._servers.items():
                server_tools = [
                    t.name for t_name, (s, t) in self._tools.items() if s.name == name
                ]
                result[name] = {
                    "command": cfg.command,
                    "args": cfg.args,
                    "env": cfg.env,
                    "url": cfg.url,
                    "tools": server_tools,
                }
            return result

    def add_server(
        self,
        name: str,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> list[str]:
        """Register a new MCP server, test its connectivity, and save to config.

        Returns the list of tool names discovered on the newly added server.
        """
        name = name.strip()
        if not name:
            raise ValueError("Server name cannot be empty.")

        new_config = MCPServerConfig(
            name=name,
            command=command,
            args=args or [],
            env=env or {},
            url=url,
            headers=headers or {},
        )

        # Test server connectivity and retrieve tools
        discovered_tools = self.mcp_client.list_tools_sync(new_config)
        tool_names = [t.name for t in discovered_tools]

        with self._lock:
            self._load_config()
            # Update disk config
            disk_data = {}
            if self.config_path.exists():
                try:
                    disk_data = json.loads(self.config_path.read_text(encoding="utf-8"))
                except Exception:
                    disk_data = {}
            if "mcpServers" not in disk_data:
                disk_data["mcpServers"] = {}

            server_entry: dict[str, Any] = {}
            if url:
                server_entry["url"] = url
                if headers:
                    server_entry["headers"] = headers
            else:
                server_entry["command"] = command
                server_entry["args"] = args or []
                if env:
                    server_entry["env"] = env

            disk_data["mcpServers"][name] = server_entry
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(disk_data, indent=2), encoding="utf-8")

            # Update in-memory state
            self._servers[name] = new_config
            for tool in discovered_tools:
                self._tools[tool.name] = (new_config, tool)

        logger.info("Successfully added MCP server '%s' with tools: %s", name, tool_names)
        return tool_names

    def remove_server(self, name: str) -> bool:
        """Remove an MCP server from configuration and runtime state."""
        with self._lock:
            self._load_config()
            if self.config_path.exists():
                try:
                    disk_data = json.loads(self.config_path.read_text(encoding="utf-8"))
                    if "mcpServers" in disk_data and name in disk_data["mcpServers"]:
                        del disk_data["mcpServers"][name]
                        self.config_path.write_text(json.dumps(disk_data, indent=2), encoding="utf-8")
                except Exception as exc:
                    logger.error("Failed to update config file on server removal: %s", exc)

            if name in self._servers:
                del self._servers[name]
                # Remove tools belonging to this server
                self._tools = {
                    t_name: (s, t) for t_name, (s, t) in self._tools.items() if s.name != name
                }
                return True
            return False
