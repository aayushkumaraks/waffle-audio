"""Unit tests for install_tool.py MCP Tool Installer script."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import install_tool
from install_tool import (
    PRESETS,
    install_preset,
    list_tools,
    remove_tool,
)


class TestInstallToolScript(unittest.TestCase):
    def test_presets_structure(self) -> None:
        self.assertIn("fetch", PRESETS)
        self.assertIn("sqlite", PRESETS)
        self.assertIn("weather", PRESETS)
        for name, preset in PRESETS.items():
            self.assertIn("title", preset)
            self.assertIn("command", preset)
            self.assertIn("args", preset)

    def test_list_tools_runs_without_error(self) -> None:
        mock_manager = MagicMock()
        mock_manager.list_installed_servers.return_value = {
            "fetch": {
                "command": "uvx",
                "args": ["mcp-server-fetch"],
                "tools": ["fetch"],
            }
        }
        # Calling list_tools should succeed
        with patch.object(install_tool.console, "print"):
            list_tools(mock_manager)
            mock_manager.initialize.assert_called_once()

    def test_install_preset_invokes_tool_manager(self) -> None:
        mock_manager = MagicMock()
        mock_manager.list_installed_servers.return_value = {}

        with patch.object(install_tool.console, "print"):
            install_preset(mock_manager, "fetch")
            mock_manager.add_server.assert_called_once()
            call_kwargs = mock_manager.add_server.call_args[1]
            self.assertEqual(call_kwargs["name"], "fetch")
            self.assertEqual(call_kwargs["command"], "uvx")

    def test_remove_tool_invokes_tool_manager(self) -> None:
        mock_manager = MagicMock()
        mock_manager.list_installed_servers.return_value = {"sqlite": {}}
        with patch.object(install_tool.console, "print"), patch("rich.prompt.Confirm.ask", return_value=True):
            remove_tool(mock_manager, "sqlite")
            mock_manager.remove_server.assert_called_once_with("sqlite")

    def test_test_tool_invokes_execute(self) -> None:
        mock_manager = MagicMock()
        mock_manager.list_installed_servers.return_value = {"fetch": {}}
        mock_manager.get_server_tools.return_value = [{"function": {"name": "fetch", "description": "Fetch URL"}}]

        with patch.object(install_tool.console, "print"):
            install_tool.test_tool(mock_manager, "fetch")
            mock_manager.refresh.assert_called_once()
            mock_manager.get_server_tools.assert_called_once_with("fetch")

    def test_test_search_invokes_ddg(self) -> None:
        mock_manager = MagicMock()
        mock_manager.fallback_search.return_value = "Search result snippets."

        with patch.object(install_tool.console, "print"):
            install_tool.test_search(mock_manager, "Python 3.13")
            mock_manager.fallback_search.assert_called_once_with("Python 3.13")


if __name__ == "__main__":
    unittest.main()
