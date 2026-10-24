"""Unit tests for models/message.py data models."""
from __future__ import annotations

import unittest
from models import Message


class TestMessageModel(unittest.TestCase):
    def test_message_initialization(self) -> None:
        msg = Message(role="user", content="Hello, assistant!")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello, assistant!")
        self.assertIsNone(msg.tool_calls)

    def test_message_with_tool_calls(self) -> None:
        tool_calls = [{"function": {"name": "test_tool", "arguments": {}}}]
        msg = Message(role="assistant", content="", tool_calls=tool_calls)
        self.assertEqual(msg.role, "assistant")
        self.assertEqual(msg.tool_calls, tool_calls)

    def test_message_slots(self) -> None:
        msg = Message(role="system", content="System prompt")
        # Since Message uses slots=True, assigning unknown attributes should raise AttributeError
        with self.assertRaises(AttributeError):
            msg.extra_attribute = "disallowed"  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
