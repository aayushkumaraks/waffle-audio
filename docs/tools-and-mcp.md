# Model Context Protocol (MCP) & Tools Integration

Waffle Audio integrates with the **Model Context Protocol (MCP)**, the open standard for connecting AI assistants to data sources, developer tools, and external services.

---

## 1. How Tools Work in Waffle Audio

Tool execution follows a three-tier architecture:

1. **Dynamic Intent Gating**:
   During regular chit-chat, tool schemas are omitted from system context to keep TTFT ultra-fast (<350 ms) and prevent model confusion. When intent analysis detects keywords or patterns ("search the web", "weather in London", "check sqlite database", "fetch url", etc.), schemas are dynamically injected for that turn.

2. **Spoken Bridge Phrases**:
   While a tool call or DuckDuckGo query is executing (which may take 500 ms to 2 seconds), Waffle immediately speaks an on-the-go conversational bridge phrase (e.g. *"Checking that for you..."*, *"Looking it up on the web..."*). This ensures there is never awkward silence during tool execution.

3. **Installed MCP Tool vs DuckDuckGo Fallback**:
   - If an installed MCP tool matches the query, Waffle invokes that tool directly.
   - If no installed tool matches or the query requires general web knowledge, news, or weather, Waffle falls back to **DuckDuckGo Search** automatically.

---

## 2. Managing Tools with `install_tool.py`

Waffle Audio includes an interactive tool manager script:

```bash
uv run python install_tool.py
```

Or run via bash:
```bash
./install_tool.sh
```

### Curated Presets:

You can install curated MCP tools directly from presets:

```bash
# Web page scraper & markdown content extractor
uv run python install_tool.py --preset fetch

# SQLite database querying tool
uv run python install_tool.py --preset sqlite

# Accurate local and international time & timezone queries
uv run python install_tool.py --preset time

# Local filesystem file reader & inspector
uv run python install_tool.py --preset filesystem

# Persistent knowledge graph memory
uv run python install_tool.py --preset memory

# GitHub repositories, issues, and PR management
uv run python install_tool.py --preset github
```

### Inspecting Installed Tools:

```bash
uv run python install_tool.py --list
```

### Testing DuckDuckGo Fallback Search:

```bash
uv run python install_tool.py --test-search "James Webb telescope recent discoveries"
```

### Previewing Spoken Bridge Phrases:

```bash
uv run python install_tool.py --preview-phrases
```

---

## 3. Configuration File (`mcp_servers.json`)

Installed tools are stored in standard MCP configuration format in `mcp_servers.json` in the project root:

```json
{
  "mcpServers": {
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    },
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time"]
    },
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "data.db"]
    }
  }
}
```

To add custom MCP servers (either `stdio` or `sse`), simply add their definitions to `mcp_servers.json` or use `install_tool.py`. Waffle automatically discovers and initializes all configured servers on startup.

---

## 4. Unit Test Verification

Tool execution and MCP server management are validated via:
- `tests/test_tools.py`: Tool discovery, JSON schema extraction, execution dispatch, DuckDuckGo search fallback, emotional bridge phrases.
- `tests/test_install_tool.py`: Preset installation, custom stdio/SSE registration, server removal, and CLI argument parsing.

```bash
uv run pytest tests/test_tools.py tests/test_install_tool.py
```
