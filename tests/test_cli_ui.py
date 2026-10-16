import unittest
import numpy as np

from services.audio.src import AudioPlaybackListener, AudioPlayer
from services.conversation.src.ui import ConversationUI


class MockPlaybackListener(AudioPlaybackListener):
    def __init__(self):
        self.started = False
        self.updates = []
        self.completed = False

    def on_playback_started(self) -> None:
        self.started = True

    def on_playback_text_updated(self, text: str) -> None:
        self.updates.append(text)

    def on_playback_completed(self) -> None:
        self.completed = True


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

    def test_text_does_not_show_before_audio_plays(self):
        self.ui.on_transcript_started("What is the time?")
        self.ui.on_transcript_completed("What is the time?")

        self.ui.on_generation_started()
        self.assertEqual(self.ui.status, "PROCESSING")
        self.assertTrue(self.ui.is_generating)
        # User message committed
        self.assertEqual(len(self.ui.history), 1)

        # While LLM is generating, text must NOT show up in current assistant display yet
        self.ui.on_generation_updated("It is currently")
        self.assertEqual(self.ui._current_assistant, "")

        self.ui.on_generation_completed("It is currently 5 PM.")
        # After LLM finishes generating, text is still NOT committed until audio plays
        self.assertEqual(self.ui._current_assistant, "")
        self.assertEqual(len(self.ui.history), 1)

        # Audio starts playing: text appears as audio plays
        self.ui.on_playback_text_updated("It is currently")
        self.assertEqual(self.ui._current_assistant, "It is currently")

        self.ui.on_playback_text_updated("It is currently 5 PM.")
        self.assertEqual(self.ui._current_assistant, "It is currently 5 PM.")

        # Audio finishes: turn is committed to history
        self.ui.on_playback_completed()
        self.assertEqual(len(self.ui.history), 2)
        self.assertEqual(self.ui.history[1]["role"], "assistant")
        self.assertEqual(self.ui.history[1]["text"], "It is currently 5 PM.")
        self.assertEqual(self.ui.status, "READY")

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

    def test_verbose_layout_and_benchmark_metrics(self):
        verbose_ui = ConversationUI(verbose=True)
        layout = verbose_ui.render(anim_frame=0)
        self.assertIsNotNone(layout["benchmark"])

        verbose_ui.on_transcript_started("How fast are you?")
        verbose_ui.on_transcript_completed("How fast are you?")
        verbose_ui.on_generation_started()
        verbose_ui.on_generation_updated("I am")
        verbose_ui.on_generation_completed("I am very fast.")
        verbose_ui.on_playback_started()
        verbose_ui.on_playback_text_updated("I am very fast.")
        verbose_ui.on_playback_completed()

        bm = verbose_ui.latest_benchmark
        self.assertIsNotNone(bm)
        self.assertIn("gate_dur", bm)
        self.assertIn("first_token_dur", bm)
        self.assertIn("full_llm_dur", bm)
        self.assertIn("tts_first_dur", bm)
        self.assertIn("e2e_dur", bm)
        self.assertGreaterEqual(bm["e2e_dur"], 0.0)

        # Re-render with populated benchmark
        layout_with_bm = verbose_ui.render(anim_frame=1)
        self.assertIsNotNone(layout_with_bm["benchmark"])


class AudioPlayerPlaybackTests(unittest.TestCase):
    def test_audio_chunk_queuing_with_text(self):
        player = AudioPlayer()
        listener = MockPlaybackListener()
        player.add_playback_listener(listener)

        fake_audio = np.zeros(1024, dtype=np.float32)
        player.on_audio_chunk(fake_audio, 16000, text="Hello world")

        item = player._playback_queue.get_nowait()
        self.assertEqual(len(item), 3)
        self.assertEqual(item[1], 16000)
        self.assertEqual(item[2], "Hello world")


if __name__ == "__main__":
    unittest.main()
