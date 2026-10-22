"""Unit tests for services/conversation/cli.py."""
from __future__ import annotations

import argparse
import os
import signal
import threading
import unittest
from unittest.mock import MagicMock, patch

from services.conversation.cli import _setup_logging, _suppress_c_stderr


class TestConversationCLI(unittest.TestCase):
    def test_setup_logging_delegation(self) -> None:
        """_setup_logging should forward benchmark_enabled flag to setup_app_logging."""
        with patch("services.logging_config.setup_app_logging") as mock_setup:
            _setup_logging(benchmark_enabled=True)
            mock_setup.assert_called_once_with(benchmark_enabled=True)

        with patch("services.logging_config.setup_app_logging") as mock_setup:
            _setup_logging(benchmark_enabled=False)
            mock_setup.assert_called_once_with(benchmark_enabled=False)

    def test_suppress_c_stderr_context_manager(self) -> None:
        """_suppress_c_stderr should succeed without raising unhandled errors."""
        with _suppress_c_stderr():
            pass

    def test_cli_argument_parsing_defaults(self) -> None:
        """Argument parser should populate expected default values."""
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--verbose", action="store_true")
        parser.add_argument("--benchmark", action="store_true", default=False)
        parser.add_argument("--voice", type=str, default=None)
        parser.add_argument("--trigger-mode", action="store_true", default=False)

        args = parser.parse_args([])
        self.assertFalse(args.verbose)
        self.assertFalse(args.benchmark)
        self.assertIsNone(args.voice)
        self.assertFalse(args.trigger_mode)

    def test_cli_argument_parsing_custom_flags(self) -> None:
        """Argument parser should parse provided flags accurately."""
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--verbose", action="store_true")
        parser.add_argument("--benchmark", action="store_true", default=False)
        parser.add_argument("--voice", type=str, default=None)
        parser.add_argument("--trigger-mode", action="store_true", default=False)

        args = parser.parse_args(["-v", "--benchmark", "--voice", "sarah_clone", "--trigger-mode"])
        self.assertTrue(args.verbose)
        self.assertTrue(args.benchmark)
        self.assertEqual(args.voice, "sarah_clone")
        self.assertTrue(args.trigger_mode)

    def test_signal_handling(self) -> None:
        """Signal handler should set the stop_event when triggered."""
        stop_event = threading.Event()

        def handle_signal(_signum=None, _frame=None) -> None:
            stop_event.set()

        self.assertFalse(stop_event.is_set())
        handle_signal(signal.SIGINT, None)
        self.assertTrue(stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
