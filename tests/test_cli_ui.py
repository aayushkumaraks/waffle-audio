import unittest
import numpy as np

from rich.console import Console
from services.audio.src import AudioPlaybackListener, AudioPlayer
from services.conversation.src.sentence_gate import SentenceGate
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

    def test_dialogue_autoscroll_overflow_keeps_latest_turns(self):
        console = Console(width=80, height=20)
        # Push 15 turns of conversation to exceed available body height
        for i in range(1, 16):
            self.ui.on_transcript_started(f"User message turn {i}")
            self.ui.on_transcript_completed(f"User message turn {i}")
            self.ui.on_generation_started()
            self.ui.on_generation_completed(f"Waffle answer turn {i}")
            self.ui.on_playback_started()
            self.ui.on_playback_text_updated(f"Waffle answer turn {i}")
            self.ui.on_playback_completed()

        layout = self.ui.render(anim_frame=0, console=console)
        body_panel = layout["body"].renderable
        plain_text = body_panel.renderable.plain

        # Latest turns MUST be present at the bottom
        self.assertIn("User message turn 15", plain_text)
        self.assertIn("Waffle answer turn 15", plain_text)

        # Earliest turns should have scrolled out of visible window
        self.assertNotIn("User message turn 1\n", plain_text)

        # Panel subtitle should indicate older lines are hidden
        self.assertIsNotNone(body_panel.subtitle)
        self.assertIn("older line", str(body_panel.subtitle))

    def test_dialogue_in_progress_speech_visible_at_bottom(self):
        console = Console(width=80, height=20)
        # Push 10 completed turns
        for i in range(1, 11):
            self.ui.on_transcript_started(f"User turn {i}")
            self.ui.on_transcript_completed(f"User turn {i}")
            self.ui.on_playback_text_updated(f"Waffle turn {i}")
            self.ui.on_playback_completed()

        # User is actively speaking
        self.ui.on_transcript_started("Active in-progress user utterance")
        self.ui.on_transcript_updated("Active in-progress user utterance that is long")

        layout = self.ui.render(anim_frame=0, console=console)
        body_panel = layout["body"].renderable
        plain_text = body_panel.renderable.plain

        self.assertIn("Active in-progress user utterance that is long", plain_text)
        self.assertIn("👤 You (speaking):", plain_text)

    def test_multi_line_user_transcript_accumulates_and_displays_fully(self):
        # Line 1 of speech
        self.ui.on_transcript_started("Tell me about the solar system")
        self.ui.on_transcript_completed("Tell me about the solar system")

        # Line 2 of speech (longer utterance spanning multiple STT lines)
        self.ui.on_transcript_started("and how many planets")
        self.ui.on_transcript_updated("and how many planets orbit the sun?")
        self.ui.on_transcript_completed("and how many planets orbit the sun?")

        # In-progress display must contain BOTH line 1 and line 2 (wrapped to terminal width)
        console = Console(width=80, height=20)
        layout = self.ui.render(anim_frame=0, console=console)
        body_panel = layout["body"].renderable
        plain_text = body_panel.renderable.plain
        normalized_text = plain_text.replace("\n", " ")
        self.assertIn("Tell me about the solar system and how many planets orbit the sun?", normalized_text)

        # When generation starts, history must preserve the complete multi-line utterance
        self.ui.on_generation_started()
        self.assertEqual(len(self.ui.history), 1)
        self.assertEqual(
            self.ui.history[0]["text"],
            "Tell me about the solar system and how many planets orbit the sun?",
        )

    def test_on_user_turn_syncs_sentence_gate_text(self):
        self.ui.on_user_turn("What is quantum computing and how does it work?")
        self.ui.on_generation_started()
        self.assertEqual(len(self.ui.history), 1)
        self.assertEqual(
            self.ui.history[0]["text"],
            "What is quantum computing and how does it work?",
        )




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


class SentenceGateTests(unittest.TestCase):
    def test_notify_speech_active_prevents_premature_fire(self):
        calls = []
        gate = SentenceGate(callback=calls.append, gap_timeout=0.1, quick_fire_timeout=0.05)
        # Line 1 arrives
        gate.submit("Hello I was thinking of")
        # User is actively speaking line 2 before silence fires
        gate.notify_speech_active()
        import time
        time.sleep(0.12)
        # Gate must NOT have fired while speech is active
        self.assertEqual(calls, [])

        # User finishes line 2
        gate.submit("how neural networks work")
        time.sleep(0.15)
        # Gate fires once with both lines joined
        self.assertEqual(calls, ["Hello I was thinking of how neural networks work"])

    def test_fragments_within_interval_accumulate_single_llm_call(self):
        calls = []
        gate = SentenceGate(callback=calls.append, gap_timeout=0.15, quick_fire_timeout=0.08)
        import time
        gate.submit("Sentence one.")
        time.sleep(0.04)
        gate.submit("Sentence two.")
        time.sleep(0.04)
        gate.submit("Sentence three.")
        time.sleep(0.2)
        self.assertEqual(calls, ["Sentence one. Sentence two. Sentence three."])
        self.assertEqual(len(calls), 1)

    def test_incomplete_eight_words_does_not_quick_fire(self):
        calls = []
        gate = SentenceGate(callback=calls.append, gap_timeout=1.5, quick_fire_timeout=0.4)
        delay = gate._compute_delay("one two three four five six seven eight")
        # Score is 0.50, which is below 0.70 threshold; delay must NOT be quick_fire_timeout (0.4)
        self.assertGreaterEqual(delay, 1.0)


if __name__ == "__main__":
    unittest.main()

