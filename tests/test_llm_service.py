"""Unit tests for services/llm/src/llm_service.py and services/llm/app.py."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from models import Message
from services.llm.src.llm_service import (
    LLMListener,
    LLMService,
    ServiceAlreadyStarted,
    ServiceNotStarted,
)


class TestLLMService(unittest.TestCase):
    def test_llm_service_lifecycle_and_errors(self) -> None:
        mock_provider = MagicMock()
        service = LLMService(mock_provider)

        # Not started operations raise ServiceNotStarted
        with self.assertRaises(ServiceNotStarted):
            service.generate([Message(role="user", content="Hi")])
        with self.assertRaises(ServiceNotStarted):
            service.stop()

        service.start()
        self.assertTrue(service._started)
        mock_provider.start.assert_called_once()

        # Double start raises ServiceAlreadyStarted
        with self.assertRaises(ServiceAlreadyStarted):
            service.start()

        # Stop service
        service.stop()
        self.assertFalse(service._started)
        mock_provider.stop.assert_called_once()

    def test_llm_service_listener_forwarding(self) -> None:
        mock_provider = MagicMock()
        service = LLMService(mock_provider)

        class DummyListener(LLMListener):
            def __init__(self) -> None:
                self.started = False
                self.updated_text = ""
                self.completed_text = ""
                self.tool_started_called = False
                self.tool_completed_called = False
                self.followup_recorded = False

            def on_generation_started(self) -> None:
                self.started = True

            def on_generation_updated(self, text: str) -> None:
                self.updated_text = text

            def on_generation_completed(self, text: str) -> None:
                self.completed_text = text

            def on_tool_started(
                self, tool_name: str, arguments: dict, phrase: str | None = None
            ) -> None:
                self.tool_started_called = True

            def on_tool_completed(
                self,
                tool_name: str,
                result: str,
                duration: float = 0.0,
                fallback: bool = False,
                server: str = "mcp",
            ) -> None:
                self.tool_completed_called = True

            def record_followup_llm(self, first_token_dur: float, full_dur: float) -> None:
                self.followup_recorded = True

        listener = DummyListener()
        service.add_listener(listener)
        self.assertIn(listener, service._listeners)

        # Trigger callbacks
        service.on_generation_started()
        self.assertTrue(listener.started)

        service.on_generation_updated("token chunk")
        self.assertEqual(listener.updated_text, "token chunk")

        service.on_generation_completed("final text")
        self.assertEqual(listener.completed_text, "final text")

        service.on_tool_started("test_tool", {"arg": "val"}, "searching...")
        self.assertTrue(listener.tool_started_called)

        service.on_tool_completed("test_tool", "result", duration=0.5, fallback=False, server="mcp")
        self.assertTrue(listener.tool_completed_called)

        service.record_followup_llm(0.2, 0.8)
        self.assertTrue(listener.followup_recorded)

        service.remove_listener(listener)
        self.assertNotIn(listener, service._listeners)

    def test_llm_service_interrupt(self) -> None:
        mock_provider = MagicMock()
        service = LLMService(mock_provider)
        service._generation_queue.put([Message(role="user", content="Pending")])

        service.interrupt()
        mock_provider.abort.assert_called_once()
        self.assertTrue(service._generation_queue.empty())

    def test_llm_app_console_listener(self) -> None:
        from services.llm.app import ConsoleListener
        listener = ConsoleListener()
        listener.on_generation_started()
        listener.on_generation_updated("chunk")
        listener.on_generation_completed("final")


if __name__ == "__main__":
    unittest.main()
