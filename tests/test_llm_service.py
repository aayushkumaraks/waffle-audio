"""Unit tests for services/llm: LLMService and LLMProvider."""
from __future__ import annotations

import queue
import time
import unittest
from unittest.mock import MagicMock, patch

from models import Message
from services.llm.src.llm_provider import LLMProviderConfig
from services.llm.src.llm_service import (
    _STOP,
    LLMListener,
    LLMService,
    ServiceAlreadyStarted,
    ServiceNotStarted,
)


class DummyLLMListener(LLMListener):
    def __init__(self) -> None:
        self.started = False
        self.updates: list[str] = []
        self.completed_text: str | None = None
        self.tools_started: list[tuple] = []
        self.tools_completed: list[tuple] = []
        self.followups: list[tuple] = []

    def on_generation_started(self) -> None:
        self.started = True

    def on_generation_updated(self, text: str) -> None:
        self.updates.append(text)

    def on_generation_completed(self, text: str) -> None:
        self.completed_text = text

    def on_tool_started(self, tool_name: str, arguments: dict, phrase: str | None = None) -> None:
        self.tools_started.append((tool_name, arguments, phrase))

    def on_tool_completed(
        self,
        tool_name: str,
        result: str,
        duration: float = 0.0,
        fallback: bool = False,
        server: str = "mcp",
    ) -> None:
        self.tools_completed.append((tool_name, result, duration, fallback, server))

    def record_followup_llm(self, first_token_dur: float, full_dur: float) -> None:
        self.followups.append((first_token_dur, full_dur))


class TestLLMService(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = MagicMock()
        self.service = LLMService(self.provider)

    def tearDown(self) -> None:
        if self.service._started:
            self.service.stop()

    def test_lifecycle_start_and_stop(self) -> None:
        """Service start and stop lifecycle updates state and delegates to provider."""
        self.assertFalse(self.service._started)
        self.service.start()
        self.assertTrue(self.service._started)
        self.provider.start.assert_called_once()

        # Starting an already started service raises ServiceAlreadyStarted
        with self.assertRaises(ServiceAlreadyStarted):
            self.service.start()

        self.service.stop()
        self.assertFalse(self.service._started)
        self.provider.stop.assert_called_once()

        # Stopping an unstarted service raises ServiceNotStarted
        with self.assertRaises(ServiceNotStarted):
            self.service.stop()

    def test_generate_requires_started(self) -> None:
        """Calling generate on unstarted service raises ServiceNotStarted."""
        with self.assertRaises(ServiceNotStarted):
            self.service.generate([Message(role="user", content="Hi")])

    def test_generate_enqueues_messages(self) -> None:
        """Generate enqueues messages when service is started."""
        self.service._started = True
        msg = Message(role="user", content="Hello")
        self.service.generate([msg])

        self.assertEqual(self.service._generation_queue.qsize(), 1)
        item = self.service._generation_queue.get_nowait()
        self.assertEqual(item, [msg])

    def test_abort_clears_queue_and_calls_provider_abort(self) -> None:
        """Abort signals provider and empties generation queue."""
        self.service._started = True
        self.service.generate([Message(role="user", content="msg1")])

        self.service.abort()
        self.provider.abort.assert_called_once()
        self.assertEqual(self.service._generation_queue.qsize(), 0)

    def test_listener_registration(self) -> None:
        """Listeners can be registered and removed."""
        listener = DummyLLMListener()
        self.service.add_listener(listener)
        self.assertIn(listener, self.service._listeners)

        self.service.remove_listener(listener)
        self.assertNotIn(listener, self.service._listeners)

    def test_callbacks_forwarded_to_listeners(self) -> None:
        """Service callbacks are forwarded cleanly to all registered listeners."""
        listener = DummyLLMListener()
        self.service.add_listener(listener)

        self.service.on_generation_started()
        self.assertTrue(listener.started)

        self.service.on_generation_updated("word ")
        self.assertEqual(listener.updates, ["word "])

        self.service.on_generation_completed("word done")
        self.assertEqual(listener.completed_text, "word done")

        self.service.on_tool_started("calc", {"a": 1}, phrase="Thinking...")
        self.assertEqual(len(listener.tools_started), 1)
        self.assertEqual(listener.tools_started[0], ("calc", {"a": 1}, "Thinking..."))

        self.service.on_tool_completed("calc", "result=2", duration=0.1, fallback=False, server="mcp")
        self.assertEqual(len(listener.tools_completed), 1)
        self.assertEqual(listener.tools_completed[0], ("calc", "result=2", 0.1, False, "mcp"))

        self.service.record_followup_llm(0.12, 0.45)
        self.assertEqual(listener.followups, [(0.12, 0.45)])

    def test_worker_processes_generation_queue(self) -> None:
        """Worker thread consumes queued message and invokes provider.generate."""
        self.service.start()
        msg = Message(role="user", content="Test prompt")
        self.service.generate([msg])

        # Give worker a moment to execute provider.generate
        time.sleep(0.1)
        self.provider.generate.assert_called_with(messages=[msg], callbacks=self.service)


class TestLLMProviderConfig(unittest.TestCase):
    def test_default_config(self) -> None:
        """LLMProviderConfig should instantiate with valid defaults."""
        config = LLMProviderConfig()
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.base_url)
        self.assertIsInstance(config.tools_enabled, bool)

    def test_intent_pattern_matching(self) -> None:
        """TOOL_RELATED_INTENT_PATTERN matches tool-related queries."""
        from services.llm.src.llm_provider import TOOL_RELATED_INTENT_PATTERN

        self.assertIsNotNone(TOOL_RELATED_INTENT_PATTERN.search("search for latest news"))
        self.assertIsNotNone(TOOL_RELATED_INTENT_PATTERN.search("check the weather in Tokyo"))
        self.assertIsNotNone(TOOL_RELATED_INTENT_PATTERN.search("check the time now"))



if __name__ == "__main__":
    unittest.main()
