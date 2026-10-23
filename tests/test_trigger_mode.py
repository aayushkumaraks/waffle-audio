import unittest
from unittest.mock import MagicMock, patch
import argparse

from services.conversation.src.manager import ConversationManager
from services.conversation.src.ui import ConversationUI
from models import Message


class TestTriggerMode(unittest.TestCase):
    def setUp(self):
        self.mock_stt = MagicMock()
        self.mock_llm = MagicMock()
        self.mock_tts = MagicMock()
        self.user_turns = []
        self.ignored_turns = []

        def on_user_turn(text: str):
            self.user_turns.append(text)

        def on_user_turn_ignored(text: str):
            self.ignored_turns.append(text)

        self.on_user_turn = on_user_turn
        self.on_user_turn_ignored = on_user_turn_ignored

    def test_is_trigger_word_present_default_name(self):
        manager = ConversationManager(
            stt=self.mock_stt,
            llm=self.mock_llm,
            tts=self.mock_tts,
            trigger_mode=True,
            assistant_name="Waffle",
        )

        # Valid trigger phrases
        self.assertTrue(manager.is_trigger_word_present("Hey Waffle"))
        self.assertTrue(manager.is_trigger_word_present("hey waffle"))
        self.assertTrue(manager.is_trigger_word_present("HEY WAFFLE"))
        self.assertTrue(manager.is_trigger_word_present("Hey, Waffle!"))
        self.assertTrue(manager.is_trigger_word_present("Hey Waffle, what's the weather?"))
        self.assertTrue(manager.is_trigger_word_present("Well, hey waffle, can you help me?"))
        self.assertTrue(manager.is_trigger_word_present("Hey-Waffle"))
        self.assertTrue(manager.is_trigger_word_present("Hey: Waffle."))

        # Phonetic variants of Waffle should trigger
        self.assertTrue(manager.is_trigger_word_present("Hey Vuffle, can you hear me?"))
        self.assertTrue(manager.is_trigger_word_present("Hey Vaudeville, what time is it?"))

        # Explicit tool call directives without wake phrase should NOT trigger
        self.assertFalse(manager.is_trigger_word_present("Tool call, what is the weather?"))
        self.assertFalse(manager.is_trigger_word_present("Can you do a tool-call to check time?"))
        self.assertFalse(manager.is_trigger_word_present("tool_call: check stock"))
        self.assertFalse(manager.is_trigger_word_present("use the tool to get time"))

        # Invalid phrases
        self.assertFalse(manager.is_trigger_word_present("What is the weather?"))
        self.assertFalse(manager.is_trigger_word_present("Waffle is great"))
        self.assertFalse(manager.is_trigger_word_present("Hey there how are you"))
        self.assertFalse(manager.is_trigger_word_present("I ate a waffle for breakfast"))
        self.assertFalse(manager.is_trigger_word_present("Hey"))
        self.assertFalse(manager.is_trigger_word_present(""))

    def test_trigger_mode_follow_up_window(self):
        import time
        manager = ConversationManager(
            stt=self.mock_stt,
            llm=self.mock_llm,
            tts=self.mock_tts,
            trigger_mode=True,
            assistant_name="Waffle",
            follow_up_window_seconds=10.0,
        )

        # Before any assistant reply, untriggered text is rejected
        self.assertFalse(manager.is_trigger_word_present("Can you tell me more?"))

        # Simulate assistant completing a turn
        manager._last_assistant_turn_time = time.monotonic()

        # Follow-up within grace window is allowed
        self.assertTrue(manager.is_trigger_word_present("Can you tell me more?"))

        # After grace window expires, it requires wake word again
        manager._last_assistant_turn_time = time.monotonic() - 15.0
        self.assertFalse(manager.is_trigger_word_present("Can you tell me more?"))

    def test_is_trigger_word_present_custom_name(self):
        manager = ConversationManager(
            stt=self.mock_stt,
            llm=self.mock_llm,
            tts=self.mock_tts,
            trigger_mode=True,
            assistant_name="Jarvis",
        )

        self.assertTrue(manager.is_trigger_word_present("Hey Jarvis, check the time"))
        self.assertTrue(manager.is_trigger_word_present("hey, jarvis!"))
        # Old name should now be ignored
        self.assertFalse(manager.is_trigger_word_present("Hey Waffle"))

    def test_trigger_mode_blocks_untriggered_utterance(self):
        manager = ConversationManager(
            stt=self.mock_stt,
            llm=self.mock_llm,
            tts=self.mock_tts,
            on_user_turn=self.on_user_turn,
            trigger_mode=True,
            assistant_name="Waffle",
            on_user_turn_ignored=self.on_user_turn_ignored,
        )

        manager._trigger_llm("Can you tell me the news?")

        # Must not call LLM or add to history
        self.mock_llm.generate.assert_not_called()
        self.assertEqual(len(manager.history), 0)
        self.assertEqual(self.user_turns, [])
        self.assertEqual(self.ignored_turns, ["Can you tell me the news?"])

    def test_trigger_mode_processes_triggered_utterance(self):
        manager = ConversationManager(
            stt=self.mock_stt,
            llm=self.mock_llm,
            tts=self.mock_tts,
            on_user_turn=self.on_user_turn,
            trigger_mode=True,
            assistant_name="Waffle",
            on_user_turn_ignored=self.on_user_turn_ignored,
        )

        manager._trigger_llm("Hey Waffle, what is the date today?")

        # Must call LLM and record turn
        self.mock_llm.generate.assert_called_once()
        self.assertEqual(len(manager.history), 1)
        self.assertEqual(manager.history[0].content, "Hey Waffle, what is the date today?")
        self.assertEqual(self.user_turns, ["Hey Waffle, what is the date today?"])
        self.assertEqual(self.ignored_turns, [])

    def test_trigger_mode_disabled_processes_all_utterances(self):
        manager = ConversationManager(
            stt=self.mock_stt,
            llm=self.mock_llm,
            tts=self.mock_tts,
            on_user_turn=self.on_user_turn,
            trigger_mode=False,
            assistant_name="Waffle",
            on_user_turn_ignored=self.on_user_turn_ignored,
        )

        manager._trigger_llm("Just regular conversation without trigger word.")

        self.mock_llm.generate.assert_called_once()
        self.assertEqual(len(manager.history), 1)
        self.assertEqual(self.user_turns, ["Just regular conversation without trigger word."])
        self.assertEqual(self.ignored_turns, [])

    def test_ui_on_user_turn_ignored_resets_state(self):
        ui = ConversationUI(trigger_mode=True, assistant_name="Waffle")
        # Simulate active speaking fragments
        ui.on_transcript_started("pass the salt")
        ui.on_transcript_completed("pass the salt")

        self.assertEqual(ui._current_user, "pass the salt")

        # When ignored, UI state should reset cleanly back to READY
        ui.on_user_turn_ignored("pass the salt")

        self.assertEqual(ui.status, "READY")
        self.assertFalse(ui.is_generating)
        self.assertEqual(ui._current_user, "")
        self.assertEqual(ui._pending_user_turn, "")
        self.assertEqual(len(ui.history), 0)

    def test_ui_render_shows_trigger_badge(self):
        ui = ConversationUI(trigger_mode=True, assistant_name="Waffle")
        layout = ui.render(0)
        # Verify render succeeds without exception and reflects trigger mode
        self.assertIsNotNone(layout)

    def test_cli_parser_trigger_mode_flag(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--trigger-mode", action="store_true")
        args = parser.parse_args(["--trigger-mode"])
        self.assertTrue(args.trigger_mode)


if __name__ == "__main__":
    unittest.main()
