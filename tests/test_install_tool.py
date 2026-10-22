"""Unit tests for install_tool.py script and MCP management."""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import install_tool
from install_tool import (
    PRESETS,
    _print_header,
    install_custom,
    install_preset,
    list_tools,
    main,
    preview_phrases,
    remove_tool,
)


class TestInstallTool(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_manager = MagicMock()
        self.mock_manager.config_path = Path("/tmp/fake_mcp.json")

    def test_presets_structure(self) -> None:
        """Presets dictionary must have valid keys, commands, and args."""
        self.assertIn("fetch", PRESETS)
        self.assertIn("sqlite", PRESETS)
        self.assertIn("filesystem", PRESETS)
        for key, preset in PRESETS.items():
            self.assertIn("title", preset)
            self.assertIn("description", preset)
            self.assertIn("command", preset)
            self.assertIn("args", preset)
            self.assertIn("env", preset)
            self.assertIsInstance(preset["args"], list)
            self.assertIsInstance(preset["env"], dict)

    def test_print_header(self) -> None:
        """_print_header should execute without errors."""
        with patch.object(install_tool.console, "print") as mock_print:
            _print_header()
            mock_print.assert_called_once()

    def test_list_tools_empty(self) -> None:
        """list_tools should display empty table when no servers are installed."""
        self.mock_manager.list_installed_servers.return_value = {}
        with patch.object(install_tool.console, "print") as mock_print:
            list_tools(self.mock_manager)
            self.mock_manager.initialize.assert_called_once()
            self.assertGreaterEqual(mock_print.call_count, 1)

    def test_list_tools_with_servers(self) -> None:
        """list_tools should display rows for stdio and sse servers."""
        self.mock_manager.list_installed_servers.return_value = {
            "fetch": {"command": "uvx", "args": ["mcp-server-fetch"], "tools": ["fetch_url"]},
            "remote": {"url": "http://localhost:8000/sse", "tools": []},
        }
        self.mock_manager.get_server_tools.return_value = []
        with patch.object(install_tool.console, "print") as mock_print:
            list_tools(self.mock_manager)
            self.mock_manager.initialize.assert_called_once()
            self.assertGreaterEqual(mock_print.call_count, 1)

    def test_install_preset_unknown(self) -> None:
        """Installing an unknown preset should print an error and not call add_server."""
        with patch.object(install_tool.console, "print") as mock_print:
            install_preset(self.mock_manager, "unknown_preset_xyz")
            self.mock_manager.add_server.assert_not_called()
            mock_print.assert_called()

    def test_install_preset_success(self) -> None:
        """Installing a valid preset should call manager.add_server."""
        self.mock_manager.add_server.return_value = ["fetch_url"]
        with patch.object(install_tool.console, "print"):
            install_preset(self.mock_manager, "fetch")
            self.mock_manager.add_server.assert_called_once()
            call_kwargs = self.mock_manager.add_server.call_args[1]
            self.assertEqual(call_kwargs["name"], "fetch")
            self.assertEqual(call_kwargs["command"], "uvx")

    def test_install_preset_github_with_token(self) -> None:
        """Installing GitHub preset should prompt for token and include it."""
        self.mock_manager.add_server.return_value = ["github_tool"]
        with patch("rich.prompt.Prompt.ask", return_value="ghp_test_token123"):
            install_preset(self.mock_manager, "github")
            self.mock_manager.add_server.assert_called_once()
            call_kwargs = self.mock_manager.add_server.call_args[1]
            self.assertEqual(call_kwargs["env"].get("GITHUB_PERSONAL_ACCESS_TOKEN"), "ghp_test_token123")

    def test_install_preset_failure_handles_exception(self) -> None:
        """Installation failure should be caught and logged cleanly."""
        self.mock_manager.add_server.side_effect = RuntimeError("Failed to spawn uvx")
        with patch.object(install_tool.console, "print") as mock_print:
            install_preset(self.mock_manager, "sqlite")
            mock_print.assert_called()

    def test_install_custom_stdio(self) -> None:
        """install_custom with stdio transport should query command and args."""
        self.mock_manager.add_server.return_value = ["custom_tool"]
        with patch("rich.prompt.Prompt.ask") as mock_ask, patch.object(install_tool.console, "print"):
            mock_ask.side_effect = [
                "stdio",           # mode
                "my_server",       # name
                "node",            # command
                "index.js --prod", # args_raw
                "API_KEY=123",     # env_raw
            ]
            install_custom(self.mock_manager)
            self.mock_manager.add_server.assert_called_once_with(
                name="my_server",
                command="node",
                args=["index.js", "--prod"],
                env={"API_KEY": "123"},
            )

    def test_install_custom_sse(self) -> None:
        """install_custom with sse transport should query url and register."""
        self.mock_manager.add_server.return_value = ["sse_tool"]
        with patch("rich.prompt.Prompt.ask") as mock_ask, patch.object(install_tool.console, "print"):
            mock_ask.side_effect = [
                "sse",                           # mode
                "remote_sse",                    # name
                "http://localhost:8080/events",  # url
            ]
            install_custom(self.mock_manager)
            self.mock_manager.add_server.assert_called_once_with(
                name="remote_sse",
                url="http://localhost:8080/events",
            )

    def test_remove_tool_none_installed(self) -> None:
        """remove_tool should notify when no servers are installed."""
        self.mock_manager.list_installed_servers.return_value = {}
        with patch.object(install_tool.console, "print") as mock_print:
            remove_tool(self.mock_manager)
            mock_print.assert_called_with("[yellow]No MCP servers currently installed to remove.[/yellow]")

    def test_remove_tool_with_name_success(self) -> None:
        """remove_tool with explicit name should call remove_server."""
        self.mock_manager.list_installed_servers.return_value = {"fetch": {}}
        self.mock_manager.remove_server.return_value = True
        with patch.object(install_tool.console, "print") as mock_print:
            remove_tool(self.mock_manager, name="fetch")
            self.mock_manager.remove_server.assert_called_once_with("fetch")
            mock_print.assert_called()

    def test_remove_tool_not_found(self) -> None:
        """remove_tool with missing server should print not found."""
        self.mock_manager.list_installed_servers.return_value = {"fetch": {}}
        self.mock_manager.remove_server.return_value = False
        with patch.object(install_tool.console, "print") as mock_print:
            remove_tool(self.mock_manager, name="nonexistent")
            mock_print.assert_called()

    def test_test_tool_fallback_to_search(self) -> None:
        """test_tool with no servers installed should fall back to search."""
        self.mock_manager.list_installed_servers.return_value = {}
        self.mock_manager.fallback_search.return_value = "Search result"
        with patch.object(install_tool, "test_search") as mock_search:
            install_tool.test_tool(self.mock_manager)
            mock_search.assert_called_once()

    def test_test_tool_with_server(self) -> None:
        """test_tool with server name should refresh and print tools."""
        self.mock_manager.list_installed_servers.return_value = {"fetch": {}}
        self.mock_manager.get_server_tools.return_value = [
            {"function": {"name": "fetch_url", "description": "Fetches a URL"}}
        ]
        with patch.object(install_tool.console, "print"):
            install_tool.test_tool(self.mock_manager, name="fetch")
            self.mock_manager.refresh.assert_called_once()
            self.mock_manager.get_server_tools.assert_called_once_with("fetch")

    def test_test_search(self) -> None:
        """test_search should call manager.fallback_search with query."""
        self.mock_manager.fallback_search.return_value = "Python 3.11 info"
        with patch.object(install_tool.console, "print"):
            install_tool.test_search(self.mock_manager, query="Python 3.11")
            self.mock_manager.fallback_search.assert_called_once_with("Python 3.11")

    def test_preview_phrases(self) -> None:
        """preview_phrases should display table without errors."""
        with patch.object(install_tool.console, "print") as mock_print:
            preview_phrases()
            mock_print.assert_called_once()

    def test_main_cli_flags(self) -> None:
        """main() CLI argument parser should handle flags correctly."""
        with patch("sys.argv", ["install_tool.py", "--preview-phrases"]), patch.object(install_tool, "preview_phrases") as mock_pp:
            main()
            mock_pp.assert_called_once()

        with patch("sys.argv", ["install_tool.py", "--list"]), patch.object(install_tool, "list_tools") as mock_lt:
            main()
            mock_lt.assert_called_once()

        with patch("sys.argv", ["install_tool.py", "--preset", "fetch"]), patch.object(install_tool, "install_preset") as mock_ip:
            main()
            mock_ip.assert_called_once()

        with patch("sys.argv", ["install_tool.py", "--test-search", "hello"]), patch.object(install_tool, "test_search") as mock_ts:
            main()
            mock_ts.assert_called_once()


if __name__ == "__main__":
    unittest.main()
