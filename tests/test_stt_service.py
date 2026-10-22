"""Unit tests for services/stt: STTService and adapters."""
from __future__ import annotations

import queue
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from services.stt.src.stt_service import (
    AudioQueueFull,
    ServiceAlreadyStarted,
    ServiceNotStarted,
    STTConfig,
    STTListener,
    STTService,
    _MoonshineListenerAdapter,
)


class DummySTTListener(STTListener):
    def __init__(self) -> None:
        self.started_lines: list[str] = []
        self.updated_lines: list[str] = []
        self.completed_lines: list[str] = []

    def on_transcript_started(self, text: str) -> None:
        self.started_lines.append(text)

    def on_transcript_updated(self, text: str) -> None:
        self.updated_lines.append(text)

    def on_transcript_completed(self, text: str) -> None:
        self.completed_lines.append(text)


class TestMoonshineAdapter(unittest.TestCase):
    def test_adapter_event_forwarding(self) -> None:
        """_MoonshineListenerAdapter maps event types to STTListener methods."""
        listener = DummySTTListener()
        adapter = _MoonshineListenerAdapter(listener)

        fake_event = SimpleNamespace(line=SimpleNamespace(text="Hello world"))

        adapter.on_line_started(fake_event)
        self.assertEqual(listener.started_lines, ["Hello world"])

        adapter.on_line_text_changed(fake_event)
        self.assertEqual(listener.updated_lines, ["Hello world"])

        adapter.on_line_completed(fake_event)
        self.assertEqual(listener.completed_lines, ["Hello world"])


class TestSTTService(unittest.TestCase):
    def setUp(self) -> None:
        self.config = STTConfig(queue_size=2)
        self.service = STTService(self.config)

    def tearDown(self) -> None:
        if self.service._stream is not None:
            with patch("moonshine_voice.Transcriber"):
                self.service.stop()

    def test_listener_registration(self) -> None:
        """Listeners can be registered and removed before start."""
        listener = DummySTTListener()
        self.service.add_listener(listener)
        self.assertIn(listener, self.service._listeners)

        self.service.remove_listener(listener)
        self.assertNotIn(listener, self.service._listeners)

    def test_push_not_started_raises_error(self) -> None:
        """Pushing audio when service is not started raises ServiceNotStarted."""
        audio = np.zeros(512, dtype=np.float32)
        with self.assertRaises(ServiceNotStarted):
            self.service.push(audio, 16000)

    def test_stop_idempotent_when_not_started(self) -> None:
        """Stopping an unstarted service returns gracefully (idempotent)."""
        self.assertIsNone(self.service._stream)
        self.service.stop()
        self.assertIsNone(self.service._stream)

    @patch("services.stt.src.stt_service.Transcriber")
    @patch("services.stt.src.stt_service.get_model_for_language")
    def test_start_and_stop_lifecycle(self, mock_get_model, mock_transcriber_cls) -> None:
        """start() initializes Transcriber and Stream; stop() cleans them up."""
        mock_get_model.return_value = ("/fake/path", "fake_arch")
        mock_transcriber = MagicMock()
        mock_stream = MagicMock()
        mock_transcriber.create_stream.return_value = mock_stream
        mock_transcriber_cls.return_value = mock_transcriber

        listener = DummySTTListener()
        self.service.add_listener(listener)

        self.service.start()
        self.assertIsNotNone(self.service._stream)
        mock_transcriber.create_stream.assert_called_once()
        mock_stream.start.assert_called_once()
        mock_stream.add_listener.assert_called_once()

        # Double start raises ServiceAlreadyStarted
        with self.assertRaises(ServiceAlreadyStarted):
            self.service.start()

        # Stop service
        self.service.stop()
        self.assertIsNone(self.service._stream)
        self.assertIsNone(self.service._transcriber)
        mock_stream.stop.assert_called_once()

    @patch("services.stt.src.stt_service.Transcriber")
    @patch("services.stt.src.stt_service.get_model_for_language")
    def test_push_audio_and_queue_full(self, mock_get_model, mock_transcriber_cls) -> None:
        """push() queues audio; raises AudioQueueFull when buffer is exceeded."""
        mock_get_model.return_value = ("/fake/path", "fake_arch")
        mock_transcriber = MagicMock()
        mock_stream = MagicMock()
        mock_transcriber.create_stream.return_value = mock_stream
        mock_transcriber_cls.return_value = mock_transcriber

        self.service.start()
        audio = np.zeros(512, dtype=np.float32)

        # Queue size is 2
        self.service.push(audio, 16000)
        self.service.push(audio, 16000)

        # 3rd push should exceed queue size and raise AudioQueueFull
        with self.assertRaises(AudioQueueFull):
            self.service.push(audio, 16000)

        self.service.stop()

    @patch("services.stt.src.stt_service.Transcriber")
    @patch("services.stt.src.stt_service.get_model_for_language")
    def test_listener_removal_during_streaming(self, mock_get_model, mock_transcriber_cls) -> None:
        """Removing listener while streaming removes adapter from stream."""
        mock_get_model.return_value = ("/fake/path", "fake_arch")
        mock_transcriber = MagicMock()
        mock_stream = MagicMock()
        mock_transcriber.create_stream.return_value = mock_stream
        mock_transcriber_cls.return_value = mock_transcriber

        listener = DummySTTListener()
        self.service.add_listener(listener)
        self.service.start()

        self.service.remove_listener(listener)
        mock_stream.remove_listener.assert_called_once()
        self.assertNotIn(listener, self.service._listeners)

        self.service.stop()


if __name__ == "__main__":
    unittest.main()
