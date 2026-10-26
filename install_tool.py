#!/usr/bin/env python3
"""Waffle Audio — Tool & MCP Server Installer Script.

Allows users to install, configure, test, and manage MCP (Model Context Protocol) tools
and search providers for the Waffle Audio assistant.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from constants import GITHUB_PERSONAL_ACCESS_TOKEN, MCP_CONFIG_PATH
from services.tools import ToolManager

console = Console()

# Curated MCP Presets
PRESETS: dict[str, dict[str, Any]] = {
    "fetch": {
        "title": "Web Fetch & HTML to Markdown Reader",
        "description": "Enables the assistant to read static web pages and extract markdown content.",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "env": {},
    },
    "sqlite": {
        "title": "SQLite Database Explorer",
        "description": "Allows the assistant to query and inspect SQLite database tables.",
        "command": "uvx",
        "args": ["mcp-server-sqlite", "--db-path", "data.db"],
        "env": {},
    },
    "time": {
        "title": "Time & Timezone Tool",
        "description": "Provides local time, UTC time, and timezone conversion capabilities.",
        "command": "uvx",
        "args": ["mcp-server-time"],
        "env": {},
    },
    "filesystem": {
        "title": "Local Filesystem Access",
        "description": "Allows the assistant to read, list, and examine files in allowed directories.",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
        "env": {},
    },
    "memory": {
        "title": "Knowledge Graph Memory",
        "description": "Enables long-term persistent entity memory across conversations.",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "env": {},
    },
    "github": {
        "title": "GitHub Integration",
        "description": "Allows inspecting repositories, issues, and pull requests via GitHub token.",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_PERSONAL_ACCESS_TOKEN},
    },
    "puppeteer": {
        "title": "Headless Browser Automation (Puppeteer)",
        "description": "Allows interacting with dynamic web pages, taking screenshots, and clicking.",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
        "env": {},
    },
    "weather": {
        "title": "Live Weather & Forecasts (Open-Meteo)",
        "description": "Fetches current temperature, forecasts, and weather conditions worldwide without API keys.",
        "command": "uvx",
        "args": ["chuk-mcp-open-meteo"],
        "env": {},
    },
}


def _print_header() -> None:
    banner = Text()
    banner.append(" 🧇 WAFFLE AUDIO ", style="bold bright_white on dark_blue")
    banner.append("  MCP Tool Installer & Manager\n", style="bold cyan")
    banner.append("Manage tools, external integrations, and DuckDuckGo search for your voice assistant.", style="dim italic")
    console.print(Panel(banner, border_style="blue", padding=(0, 1)))


def list_tools(manager: ToolManager) -> None:
    """Display installed MCP servers and their discovered tools."""
    manager.initialize()
    servers = manager.list_installed_servers()

    table = Table(title="[bold cyan]Installed MCP Servers & Tools[/bold cyan]", border_style="cyan")
    table.add_column("Server Name", style="bold green")
    table.add_column("Type / Transport", style="cyan")
    table.add_column("Command / Target", style="dim")
    table.add_column("Tools Discovered", style="yellow")

    if not servers:
        table.add_row(
            "[dim italic]None configured[/dim italic]",
            "—",
            f"Config file: {manager.config_path}",
            "Fallback: DuckDuckGo Search active",
        )
    else:
        for name, info in servers.items():
            if info.get("url"):
                transport = "SSE / HTTP"
                target = info["url"]
            else:
                transport = "stdio"
                cmd_parts = [info.get("command") or ""] + (info.get("args") or [])
                target = " ".join(cmd_parts)

            tool_list = info.get("tools") or [t["function"]["name"] for t in manager.get_server_tools(name)]
            tool_display = ", ".join(tool_list) if tool_list else "[yellow]Connected (schema pending)[/yellow]"
            table.add_row(name, transport, target, tool_display)

    console.print(table)

    # Show built-in DuckDuckGo Search status
    ddg_panel = Panel(
        "[bold green]✓ DuckDuckGo Search Fallback:[/] [bright_white]Active[/]\n"
        "[dim]Whenever a requested tool is not installed or when general web information is needed, "
        "the assistant automatically searches DuckDuckGo.[/dim]",
        border_style="green",
        title="[bold green]Built-in Web Fallback[/]",
    )
    console.print(ddg_panel)

    # Show Emotional Filler Phrases status
    phrase_panel = Panel(
        "[bold magenta]✓ Emotional Filler Phrases:[/] [bright_white]Active (Synthesized on the go)[/]\n"
        "[dim]Plays conversational emotional speech fillers ('Hmm, let me search that...', 'Uhh, checking...') "
        "on the go using the active voice model and TTS pipeline while tools execute in parallel.[/dim]",
        border_style="magenta",
        title="[bold magenta]Conversational Latency Masking[/]",
    )
    console.print(phrase_panel)



def install_preset(manager: ToolManager, preset_key: str) -> None:
    """Install a pre-configured MCP tool preset."""
    if preset_key not in PRESETS:
        console.print(f"[bold red]Unknown preset '{preset_key}'.[/bold red] Available: {', '.join(PRESETS.keys())}")
        return

    preset = PRESETS[preset_key]
    console.print(f"\n[bold cyan]Installing Preset:[/] [bold green]{preset['title']}[/]")
    console.print(f"[dim]{preset['description']}[/dim]\n")

    command = preset["command"]
    args = list(preset["args"])
    env = dict(preset["env"])

    # If preset requires additional configuration (e.g. GitHub token)
    if "GITHUB_PERSONAL_ACCESS_TOKEN" in env:
        token = Prompt.ask("Enter your GitHub Personal Access Token (or leave blank to skip)", default="")
        if token:
            env["GITHUB_PERSONAL_ACCESS_TOKEN"] = token
        else:
            del env["GITHUB_PERSONAL_ACCESS_TOKEN"]

    with console.status(f"[bold cyan]Connecting to '{preset_key}' MCP server and querying tools...[/bold cyan]", spinner="dots"):
        try:
            tools = manager.add_server(
                name=preset_key,
                command=command,
                args=args,
                env=env,
            )
            console.print(f"[bold green]✓ Successfully installed MCP tool '{preset_key}'![/bold green]")
            if tools:
                console.print(f"[green]Discovered {len(tools)} tool(s):[/green] [cyan]{', '.join(tools)}[/cyan]")
            else:
                console.print("[yellow]Server registered successfully. Tools will be loaded on demand.[/yellow]")
        except Exception as exc:
            console.print(f"[bold red]Failed to install tool preset '{preset_key}':[/bold red] {exc}")
            console.print("[dim yellow]Tip: Ensure the required runner ('uvx' or 'npx') is available on your PATH.[/dim yellow]")


def install_custom(manager: ToolManager) -> None:
    """Prompt user for custom MCP server details."""
    console.print("\n[bold cyan]Add Custom MCP Server[/bold cyan]")
    mode = Prompt.ask("Select transport type", choices=["stdio", "sse"], default="stdio")

    name = Prompt.ask("Enter a unique name for this MCP server").strip()
    if not name:
        console.print("[bold red]Name cannot be empty.[/bold red]")
        return

    if mode == "stdio":
        command = Prompt.ask("Executable command (e.g., 'uvx', 'npx', 'python', 'node')").strip()
        args_raw = Prompt.ask("Arguments (space-separated, e.g. 'mcp-server-sqlite --db-path ./app.db')", default="").strip()
        args = args_raw.split() if args_raw else []

        env_raw = Prompt.ask("Environment variables (key=value, comma-separated, optional)", default="").strip()
        env: dict[str, str] = {}
        if env_raw:
            for item in env_raw.split(","):
                if "=" in item:
                    k, v = item.split("=", 1)
                    env[k.strip()] = v.strip()

        with console.status(f"[bold cyan]Connecting to '{name}' and validating tools...[/bold cyan]", spinner="dots"):
            try:
                tools = manager.add_server(name=name, command=command, args=args, env=env)
                console.print(f"[bold green]✓ Successfully installed custom MCP server '{name}'![/bold green]")
                console.print(f"[green]Discovered tools:[/] [cyan]{', '.join(tools) or 'None'}[/cyan]")
            except Exception as exc:
                console.print(f"[bold red]Failed to connect to MCP server '{name}':[/bold red] {exc}")
                save_anyway = Confirm.ask("Do you want to save the configuration anyway?", default=False)
                if save_anyway:
                    # Write to config without validating
                    manager.config_path.parent.mkdir(parents=True, exist_ok=True)
                    data = {}
                    if manager.config_path.exists():
                        try:
                            data = json.loads(manager.config_path.read_text(encoding="utf-8"))
                        except Exception:
                            data = {}
                    data.setdefault("mcpServers", {})[name] = {"command": command, "args": args, "env": env}
                    manager.config_path.write_text(json.dumps(data, indent=2))
                    console.print(f"[yellow]Configuration for '{name}' saved to {manager.config_path}.[/yellow]")
    else:
        url = Prompt.ask("Enter SSE URL (e.g. 'http://localhost:8000/sse')").strip()
        with console.status(f"[bold cyan]Connecting to '{name}' at {url}...[/bold cyan]", spinner="dots"):
            try:
                tools = manager.add_server(name=name, url=url)
                console.print(f"[bold green]✓ Successfully registered SSE MCP server '{name}'![/bold green]")
                console.print(f"[green]Discovered tools:[/] [cyan]{', '.join(tools) or 'None'}[/cyan]")
            except Exception as exc:
                console.print(f"[bold red]Failed to connect to SSE server '{name}':[/bold red] {exc}")


def remove_tool(manager: ToolManager, name: str | None = None) -> None:
    """Remove an installed MCP tool."""
    servers = manager.list_installed_servers()
    if not servers:
        console.print("[yellow]No MCP servers currently installed to remove.[/yellow]")
        return

    if not name:
        server_names = list(servers.keys())
        name = Prompt.ask("Select server to remove", choices=server_names)

    if manager.remove_server(name):
        console.print(f"[bold green]✓ Removed MCP server '{name}' successfully.[/bold green]")
    else:
        console.print(f"[bold red]Server '{name}' was not found in configuration.[/bold red]")


def test_tool(manager: ToolManager, name: str | None = None) -> None:
    """Test calling an installed tool or search."""
    servers = manager.list_installed_servers()
    if not servers:
        console.print("[yellow]No custom MCP servers installed. Testing DuckDuckGo search instead.[/yellow]")
        test_search(manager, "What is the latest space discovery?")
        return

    if not name:
        name = Prompt.ask("Select server to test", choices=list(servers.keys()))

    with console.status(f"[bold cyan]Testing MCP server '{name}'...[/bold cyan]", spinner="dots"):
        try:
            manager.refresh()
            server_tools = manager.get_server_tools(name)
            console.print(f"[bold green]✓ Server '{name}' is active![/bold green]")
            console.print(f"[cyan]Available function schemas ({len(server_tools)}):[/cyan]")
            for t in server_tools:
                fn = t.get("function", {})
                console.print(f"  • [bold]{fn.get('name')}[/bold]: {fn.get('description')}")
        except Exception as exc:
            console.print(f"[bold red]Test failed for '{name}':[/bold red] {exc}")


def test_search(manager: ToolManager, query: str | None = None) -> None:
    """Test the DuckDuckGo search fallback."""
    q = query or Prompt.ask("Enter a search query to test", default="current Python version")
    with console.status(f"[bold cyan]Querying DuckDuckGo for: '{q}'...[/bold cyan]", spinner="dots"):
        results = manager.fallback_search(q)
    console.print(Panel(results, title=f"[bold green]DuckDuckGo Search Output for '{q}'[/]", border_style="green"))


def preview_phrases() -> None:
    """Display dictionary of emotional filler phrases and contextual emotions."""
    from services.tools.src.phrases import FILLER_PHRASES

    table = Table(
        title="[bold magenta]Emotional Filler Phrases Dictionary (Synthesized on the go)[/bold magenta]",
        border_style="magenta",
    )
    table.add_column("Category", style="bold green")
    table.add_column("ID", style="cyan")
    table.add_column("Spoken Sentence", style="white")
    table.add_column("Emotion", style="yellow")
    table.add_column("Synthesis", style="bold bright_cyan")

    for cat, p_list in FILLER_PHRASES.items():
        for p in p_list:
            table.add_row(cat, p["id"], f'"{p["text"]}"', p["emotion"], "On the go (Active TTS)")

    console.print(table)


def interactive_menu(manager: ToolManager) -> None:
    """Run interactive terminal menu."""
    _print_header()

    while True:
        console.print("\n[bold cyan]Select an option:[/bold cyan]")
        console.print("  [bold green][1][/bold green] List installed MCP tools & servers")
        console.print("  [bold green][2][/bold green] Install a tool from popular presets (fetch, sqlite, filesystem, etc.)")
        console.print("  [bold green][3][/bold green] Install a custom MCP tool (stdio or SSE)")
        console.print("  [bold green][4][/bold green] Test an installed MCP server")
        console.print("  [bold green][5][/bold green] Test DuckDuckGo search fallback")
        console.print("  [bold green][6][/bold green] Preview emotional filler phrases (on-the-go TTS)")
        console.print("  [bold green][7][/bold green] Remove an installed tool")
        console.print("  [bold red][0][/bold red] Exit")

        choice = Prompt.ask("\nEnter choice", choices=["0", "1", "2", "3", "4", "5", "6", "7"], default="1")

        if choice == "0":
            console.print("[dim]Goodbye![/dim]")
            break
        elif choice == "1":
            list_tools(manager)
        elif choice == "2":
            console.print("\n[bold cyan]Available Presets:[/bold cyan]")
            for key, preset in PRESETS.items():
                console.print(f"  • [bold green]{key:12}[/bold green] - {preset['title']}")
            preset_choice = Prompt.ask("Enter preset name", choices=list(PRESETS.keys()))
            install_preset(manager, preset_choice)
        elif choice == "3":
            install_custom(manager)
        elif choice == "4":
            test_tool(manager)
        elif choice == "5":
            test_search(manager)
        elif choice == "6":
            preview_phrases()
        elif choice == "7":
            remove_tool(manager)



def main() -> None:
    parser = argparse.ArgumentParser(description="Waffle Audio MCP Tool & Search Installer")
    parser.add_argument("--list", action="store_true", help="List all configured MCP tools and servers")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()), help="Install an MCP tool from preset")
    parser.add_argument("--add", type=str, help="Add custom MCP server by name")
    parser.add_argument("--command", type=str, help="Command executable for custom stdio MCP server (e.g. uvx, npx)")
    parser.add_argument("--args", nargs="*", default=[], help="Arguments for the MCP server command")
    parser.add_argument("--url", type=str, help="URL for SSE-based MCP server")
    parser.add_argument("--remove", type=str, help="Remove an installed MCP tool by name")
    parser.add_argument("--test", type=str, help="Test an installed MCP tool by name")
    parser.add_argument("--test-search", type=str, help="Test DuckDuckGo search fallback with a query")
    parser.add_argument("--preview-phrases", action="store_true", help="List emotional filler phrases dictionary and contextual emotions")
    parser.add_argument("--config", type=str, default=str(MCP_CONFIG_PATH), help="Path to mcp_servers.json config file")

    args = parser.parse_args()
    manager = ToolManager(config_path=Path(args.config))

    # If no flags provided, run interactive UI
    if not any([args.list, args.preset, args.add, args.remove, args.test, args.test_search, args.preview_phrases]):
        interactive_menu(manager)
        return

    if args.preview_phrases:
        preview_phrases()
    elif args.list:
        list_tools(manager)

    elif args.preset:
        install_preset(manager, args.preset)
    elif args.add:
        if args.url:
            manager.add_server(name=args.add, url=args.url)
            console.print(f"[bold green]✓ Registered SSE MCP server '{args.add}' successfully.[/bold green]")
        elif args.command:
            tools = manager.add_server(name=args.add, command=args.command, args=args.args)
            console.print(f"[bold green]✓ Installed MCP server '{args.add}' with tools: {tools}[/bold green]")
        else:
            console.print("[bold red]Error: --command or --url required when using --add.[/bold red]")
            sys.exit(1)
    elif args.remove:
        remove_tool(manager, args.remove)
    elif args.test:
        test_tool(manager, args.test)
    elif args.test_search:
        test_search(manager, args.test_search)


if __name__ == "__main__":
    main()
