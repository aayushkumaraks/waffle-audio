import unittest

from services.conversation.src.ui import ConversationUI


class ConversationUITests(unittest.TestCase):
    def setUp(self):
        self.ui = ConversationUI()

    def test_initial_state_is_ready(self):
        self.assertEqual(self.ui.status, "READY")
        self.assertEqual(self.ui.history, [])
        self.assertFalse(self.ui.is_generating)

    def test_transcript_updates_transition_to_listening(self):
        self.ui.on_transcript_started("Hello")
        self.assertEqual(self.ui.status, "LISTENING")

        self.ui.on_transcript_updated("Hello there")
        self.assertEqual(self.ui.status, "LISTENING")

        self.ui.on_transcript_completed("Hello there, Waffle")
        self.assertEqual(self.ui.status, "LISTENING")

    def test_generation_flow_and_history_commitment(self):
        self.ui.on_transcript_started("What is the time?")
        self.ui.on_transcript_completed("What is the time?")

        self.ui.on_generation_started()
        self.assertEqual(self.ui.status, "PROCESSING")
        self.assertTrue(self.ui.is_generating)
        self.assertEqual(len(self.ui.history), 1)
        self.assertEqual(self.ui.history[0]["role"], "user")
        self.assertEqual(self.ui.history[0]["text"], "What is the time?")

        self.ui.on_generation_updated("It is currently")
        self.assertEqual(self.ui.status, "PROCESSING")

        self.ui.on_generation_completed("It is currently 5 PM.")
        self.assertFalse(self.ui.is_generating)
        self.assertEqual(len(self.ui.history), 2)
        self.assertEqual(self.ui.history[1]["role"], "assistant")
        self.assertEqual(self.ui.history[1]["text"], "It is currently 5 PM.")

    def test_sync_audio_playback_transitions(self):
        self.ui.on_transcript_started("Test")
        self.ui.on_generation_started()
        self.ui.on_generation_completed("Response")

        # Audio is still playing
        self.ui.sync_audio_playback(is_playing=True)
        self.assertEqual(self.ui.status, "PROCESSING")

        # Audio finishes
        self.ui.sync_audio_playback(is_playing=False)
        self.assertEqual(self.ui.status, "READY")

    def test_render_all_statuses(self):
        for status in ["READY", "LISTENING", "PROCESSING"]:
            self.ui.set_status(status)
            layout = self.ui.render(anim_frame=3)
            self.assertIsNotNone(layout)
            self.assertIsNotNone(layout["header"])
            self.assertIsNotNone(layout["body"])
            self.assertIsNotNone(layout["footer"])
            self.assertIsNotNone(layout["header"]["status_box"])


if __name__ == "__main__":
    unittest.main()
