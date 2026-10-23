"""Unit tests for services/audio/src/audio_player.py."""
from __future__ import annotations

import queue
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from services.audio.src.audio_player import (
    _STOP,
    AudioPlaybackListener,
    AudioPlayer,
)


class DummyPlaybackListener(AudioPlaybackListener):
    def __init__(self) -> None:
        self.started = False
        self.texts: list[str] = []
        self.completed = False

    def on_playback_started(self) -> None:
        self.started = True

    def on_playback_text_updated(self, text: str) -> None:
        self.texts.append(text)

    def on_playback_completed(self) -> None:
        self.completed = True


class TestAudioPlayer(unittest.TestCase):
    def setUp(self) -> None:
        self.player = AudioPlayer()

    def tearDown(self) -> None:
        if self.player._started if hasattr(self.player, "_started") else (self.player._worker_thread is not None):
            self.player.stop()

    def test_listener_management(self) -> None:
        """Listeners can be added and removed."""
        listener = DummyPlaybackListener()
        self.player.add_playback_listener(listener)
        self.assertIn(listener, self.player._playback_listeners)

        self.player.remove_playback_listener(listener)
        self.assertNotIn(listener, self.player._playback_listeners)

    def test_monitor_callback_management(self) -> None:
        """Echo cancellation monitor callbacks can be added and removed."""
        cb = MagicMock()
        self.player.add_playback_monitor(cb)
        self.assertIn(cb, self.player._monitor_callbacks)

        self.player.remove_playback_monitor(cb)
        self.assertNotIn(cb, self.player._monitor_callbacks)

    def test_upstream_busy_predicate(self) -> None:
        """Upstream busy predicate reports status correctly and handles exceptions."""
        self.assertFalse(self.player._is_upstream_busy())

        self.player.set_upstream_busy(lambda: True)
        self.assertTrue(self.player._is_upstream_busy())

        self.player.set_upstream_busy(lambda: False)
        self.assertFalse(self.player._is_upstream_busy())

        # Faulty predicate
        def bad():
            raise RuntimeError("Boom")
        self.player.set_upstream_busy(bad)
        self.assertFalse(self.player._is_upstream_busy())

    def test_on_synthesis_started_sets_synthesizing(self) -> None:
        """on_synthesis_started sets internal synthesizing event."""
        self.assertFalse(self.player._synthesizing.is_set())
        self.player.on_synthesis_started()
        self.assertTrue(self.player._synthesizing.is_set())

    def test_on_audio_chunk_queues_item(self) -> None:
        """on_audio_chunk places audio buffer and metadata into playback queue."""
        chunk = np.zeros(1024, dtype=np.float32)
        self.player.on_audio_chunk(chunk, 24000, text="Hello world")
        self.assertEqual(self.player._playback_queue.qsize(), 1)
        item = self.player._playback_queue.get_nowait()
        self.assertEqual(len(item), 3)
        np.testing.assert_array_equal(item[0], chunk)
        self.assertEqual(item[1], 24000)
        self.assertEqual(item[2], "Hello world")

    def test_interrupt_clears_queue_and_state(self) -> None:
        """Interrupt purges queued chunks and resets turn active."""
        chunk = np.zeros(512, dtype=np.float32)
        self.player.on_audio_chunk(chunk, 24000, text="one")
        self.player.on_audio_chunk(chunk, 24000, text="two")
        self.assertEqual(self.player._playback_queue.qsize(), 2)

        self.player.interrupt()

        self.assertTrue(self.player._interrupted.is_set())
        self.assertFalse(self.player._turn_active)
        self.assertEqual(self.player._accumulated_text, "")
        self.assertEqual(self.player._playback_queue.qsize(), 0)

    def test_is_playing_reflects_events(self) -> None:
        """is_playing is True if playing event, synthesizing event, or non-empty queue."""
        self.assertFalse(self.player.is_playing)

        self.player._playing.set()
        self.assertTrue(self.player.is_playing)
        self.player._playing.clear()

        self.player._synthesizing.set()
        self.assertTrue(self.player.is_playing)
        self.player._synthesizing.clear()

        self.player._playback_queue.put_nowait((np.zeros(10, dtype=np.float32), 16000, "text"))
        self.assertTrue(self.player.is_playing)
        self.player.interrupt()
        self.assertFalse(self.player.is_playing)

    def test_post_interrupt_chunk_is_not_dropped(self) -> None:
        """When interrupt() is called, any chunk arriving for the subsequent turn (e.g. tool filler) is played."""
        written_chunks: list[np.ndarray] = []

        class MockStream:
            def __init__(self, **kwargs):
                self.stopped = False
            def start(self):
                self.stopped = False
            def stop(self):
                self.stopped = True
            def close(self):
                self.stopped = True
            def write(self, data):
                written_chunks.append(data.copy())

        with patch("sounddevice.OutputStream", MockStream):
            self.player.start()

            # First turn chunk
            self.player.on_audio_chunk(np.ones(2048, dtype=np.float32), 24000, "turn 1")
            time.sleep(0.03)

            # Barge-in happens
            self.player.interrupt()
            time.sleep(0.02)

            # Subsequent turn arrives (e.g. tool filler phrase)
            self.player.on_synthesis_started()
            self.player.on_audio_chunk(np.full(1024, 5.0, dtype=np.float32), 24000, "filler phrase")
            self.player.on_synthesis_completed()

            # Allow playback worker to process
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if any(np.isclose(c[0, 0], 5.0) for c in written_chunks):
                    break
                time.sleep(0.02)

            self.assertTrue(
                any(np.isclose(c[0, 0], 5.0) for c in written_chunks),
                "Subsequent turn chunk (value 5.0) was incorrectly discarded after interrupt!",
            )


if __name__ == "__main__":
    unittest.main()
