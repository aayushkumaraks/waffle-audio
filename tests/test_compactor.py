"""Tests for the ContextCompactor background conversation compaction system."""
import time
import unittest
from unittest.mock import MagicMock, patch

from models import Message
from services.conversation.src.compactor import ContextCompactor
from services.conversation.src.manager import ConversationManager


class TestContextCompactor(unittest.TestCase):
    def setUp(self) -> None:
        self.compactor = ContextCompactor(
            enabled=True,
            max_turns=3,  # 6 messages
            char_limit=200,
            preserve_turns=1,  # 2 messages
            use_llm=True,
        )

    def test_should_not_compact_when_disabled(self) -> None:
        self.compactor.enabled = False
        history = [Message(role="user", content=f"msg {i}") for i in range(10)]
        self.assertFalse(self.compactor.should_compact(history))

    def test_should_not_compact_short_history(self) -> None:
        # 2 messages <= preserve window (1 turn * 2 = 2 messages)
        history = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        self.assertFalse(self.compactor.should_compact(history))

    def test_should_compact_when_turn_limit_reached(self) -> None:
        # max_turns=3 -> 6 messages
        history = [
            Message(role="user", content="msg 1"),
            Message(role="assistant", content="reply 1"),
            Message(role="user", content="msg 2"),
            Message(role="assistant", content="reply 2"),
            Message(role="user", content="msg 3"),
            Message(role="assistant", content="reply 3"),
        ]
        self.assertTrue(self.compactor.should_compact(history))

    def test_should_compact_when_char_limit_reached(self) -> None:
        # 4 messages (> preserve_count=2), total chars > 200
        history = [
            Message(role="user", content="A" * 120),
            Message(role="assistant", content="B" * 120),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        self.assertTrue(self.compactor.should_compact(history))

    def test_heuristic_summary_generation(self) -> None:
        messages = [
            Message(role="user", content="What is the weather in Paris?"),
            Message(role="assistant", content="It is sunny and twenty degrees in Paris."),
            Message(role="user", content="Check the time in Tokyo."),
            Message(role="assistant", content="It is nine PM in Tokyo."),
        ]
        summary = self.compactor._heuristic_summarizer(messages)
        self.assertIn("weather in Paris", summary)
        self.assertIn("twenty degrees", summary)

    @patch("services.conversation.src.compactor.httpx.Client")
    def test_compact_async_runs_on_separate_thread(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "User asked about Paris weather and Tokyo time."}
        }
        mock_client.post.return_value = mock_resp
        mock_client.__enter__.return_value = mock_client
        mock_client_cls.return_value = mock_client

        history = [
            Message(role="user", content="What is the weather in Paris?"),
            Message(role="assistant", content="It is sunny in Paris."),
            Message(role="user", content="What time is it in Tokyo?"),
            Message(role="assistant", content="It is 9 PM in Tokyo."),
            Message(role="user", content="What is the capital of France?"),
            Message(role="assistant", content="The capital is Paris."),
        ]

        result_compacted = []
        result_snapshot_len = 0
        callback_called = False

        def on_compacted(compacted: list[Message], snap_len: int) -> None:
            nonlocal result_compacted, result_snapshot_len, callback_called
            result_compacted = compacted
            result_snapshot_len = snap_len
            callback_called = True

        launched = self.compactor.compact_async(history, on_compacted)
        self.assertTrue(launched)

        # Wait briefly for background thread to execute
        timeout = 2.0
        start = time.time()
        while not callback_called and (time.time() - start) < timeout:
            time.sleep(0.05)

        self.assertTrue(callback_called)
        self.assertEqual(result_snapshot_len, 6)
        # Should contain 1 system summary message + 2 preserved recent messages
        self.assertEqual(len(result_compacted), 3)
        self.assertEqual(result_compacted[0].role, "system")
        self.assertIn("Prior Context Summary", result_compacted[0].content)
        self.assertEqual(result_compacted[1].content, "What is the capital of France?")
        self.assertEqual(result_compacted[2].content, "The capital is Paris.")

    @patch("services.conversation.src.compactor.httpx.Client")
    def test_manager_merges_new_messages_arriving_during_compaction(
        self, mock_client_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "Earlier conversation discussed Paris and Tokyo."}
        }
        mock_client.post.return_value = mock_resp
        mock_client.__enter__.return_value = mock_client
        mock_client_cls.return_value = mock_client

        mock_stt = MagicMock()
        mock_llm = MagicMock()
        mock_tts = MagicMock()

        manager = ConversationManager(
            stt=mock_stt,
            llm=mock_llm,
            tts=mock_tts,
            compactor=self.compactor,
        )

        # Seed 6 messages into manager history
        for i in range(3):
            manager._history.append(Message(role="user", content=f"question {i}"))
            manager._history.append(Message(role="assistant", content=f"answer {i}"))

        original_snapshot_len = len(manager._history)

        # Launch compaction
        self.compactor.compact_async(manager._history, manager._on_compaction_completed)

        # Simulate a new turn arriving while compaction is in-flight on the separate thread
        manager._history.append(Message(role="user", content="New urgent question"))
        manager._history.append(Message(role="assistant", content="New quick answer"))

        # Wait for compactor thread to finish and merge
        timeout = 2.0
        start = time.time()
        while self.compactor.is_compacting and (time.time() - start) < timeout:
            time.sleep(0.05)

        # The manager history should contain:
        # [0]: system summary
        # [1]: question 2 (preserved from snapshot)
        # [2]: answer 2 (preserved from snapshot)
        # [3]: New urgent question (newly arrived during async compaction)
        # [4]: New quick answer (newly arrived during async compaction)
        self.assertEqual(len(manager._history), 5)
        self.assertEqual(manager._history[0].role, "system")
        self.assertIn("Prior Context Summary", manager._history[0].content)
        self.assertEqual(manager._history[-2].content, "New urgent question")
        self.assertEqual(manager._history[-1].content, "New quick answer")

    def test_fast_heuristic_default_does_not_call_http(self) -> None:
        """When use_llm is False (default), compaction runs instantly without any HTTP/Ollama network requests."""
        fast_compactor = ContextCompactor(
            enabled=True,
            max_turns=2,
            char_limit=100,
            preserve_turns=1,
            use_llm=False,
        )

        history = [
            Message(role="user", content="Where is Berlin?"),
            Message(role="assistant", content="Berlin is the capital of Germany."),
            Message(role="user", content="What is the currency?"),
            Message(role="assistant", content="The currency is the Euro."),
        ]

        result_compacted = []
        callback_called = False

        def on_compacted(compacted: list[Message], snap_len: int) -> None:
            nonlocal result_compacted, callback_called
            result_compacted = compacted
            callback_called = True

        launched = fast_compactor.compact_async(history, on_compacted)
        self.assertTrue(launched)

        # Wait briefly for thread to finish (should be immediate)
        start = time.time()
        while not callback_called and (time.time() - start) < 1.0:
            time.sleep(0.02)

        self.assertTrue(callback_called)
        self.assertEqual(len(result_compacted), 3)
        self.assertIn("Berlin", result_compacted[0].content)

    def test_abort_cancels_background_compaction(self) -> None:
        """Calling abort() signals the background compactor thread to exit cleanly without modifying history."""
        slow_compactor = ContextCompactor(
            enabled=True,
            max_turns=2,
            char_limit=100,
            preserve_turns=1,
            use_llm=True,
        )

        callback_invoked = False

        def on_compacted(compacted: list[Message], snap_len: int) -> None:
            nonlocal callback_invoked
            callback_invoked = True

        history = [
            Message(role="user", content="User message 1"),
            Message(role="assistant", content="Assistant reply 1"),
            Message(role="user", content="User message 2"),
            Message(role="assistant", content="Assistant reply 2"),
        ]

        # Simulate slow summary execution and trigger abort while in-flight
        def slow_summary(messages):
            time.sleep(0.05)
            return "simulated summary"

        with patch.object(slow_compactor, "_summarize_messages", side_effect=slow_summary):
            launched = slow_compactor.compact_async(history, on_compacted)
            self.assertTrue(launched)
            # Abort while in-flight
            slow_compactor.abort()

        time.sleep(0.1)
        self.assertFalse(callback_invoked)

    def test_manager_aborts_compactor_on_trigger_llm(self) -> None:
        """When a user utterance arrives, manager aborts in-flight compaction so all compute is given to generation."""
        mock_stt = MagicMock()
        mock_llm = MagicMock()
        mock_tts = MagicMock()
        mock_compactor = MagicMock()

        manager = ConversationManager(
            stt=mock_stt,
            llm=mock_llm,
            tts=mock_tts,
            compactor=mock_compactor,
        )

        manager._trigger_llm("Can you help me?")
        mock_compactor.abort.assert_called_once()


if __name__ == "__main__":
    unittest.main()
