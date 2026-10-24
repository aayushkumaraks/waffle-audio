"""Unit tests for services/audio/src/audio_player.py audio output and playback listener system."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from services.audio.src.audio_player import AudioPlaybackListener, AudioPlayer


class TestAudioPlayer(unittest.TestCase):
    def test_audio_player_listener_management(self) -> None:
        player = AudioPlayer()

        class DummyListener(AudioPlaybackListener):
            def on_playback_started(self) -> None:
                pass

            def on_playback_text_updated(self, text: str) -> None:
                pass

            def on_playback_completed(self) -> None:
                pass

        listener = DummyListener()
        player.add_playback_listener(listener)
        self.assertIn(listener, player._playback_listeners)

        player.remove_playback_listener(listener)
        self.assertNotIn(listener, player._playback_listeners)

    def test_audio_player_monitor_management(self) -> None:
        player = AudioPlayer()
        cb = MagicMock()
        player.add_playback_monitor(cb)
        self.assertIn(cb, player._monitor_callbacks)

        player.remove_playback_monitor(cb)
        self.assertNotIn(cb, player._monitor_callbacks)

    def test_is_playing_states(self) -> None:
        player = AudioPlayer()
        self.assertFalse(player.is_playing)

        player.on_synthesis_started()
        self.assertTrue(player.is_playing)

        player.on_synthesis_completed()
        self.assertFalse(player.is_playing)

        # Upstream busy check
        player.set_upstream_busy(lambda: True)
        self.assertTrue(player.is_playing)

        player.set_upstream_busy(lambda: False)
        self.assertFalse(player.is_playing)

    def test_interrupt_clears_queue_and_resets(self) -> None:
        player = AudioPlayer()
        player.on_synthesis_started()
        chunk = np.zeros(512, dtype=np.float32)
        player.on_audio_chunk(chunk, 24000, "hello")
        self.assertFalse(player._playback_queue.empty())

        player.interrupt()
        self.assertTrue(player._playback_queue.empty())
        self.assertFalse(player.is_playing)

    @patch("sounddevice.OutputStream")
    def test_start_stop_lifecycle(self, mock_output_stream: MagicMock) -> None:
        player = AudioPlayer()
        player.start()
        self.assertIsNotNone(player._worker_thread)
        self.assertTrue(player._worker_thread.is_alive())

        player.stop()
        self.assertIsNone(player._worker_thread)


if __name__ == "__main__":
    unittest.main()
