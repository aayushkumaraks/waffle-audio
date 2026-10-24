"""Unit tests for services/stt/src/stt_service.py and services/stt/app.py."""
from __future__ import annotations

import unittest
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


class TestSTTService(unittest.TestCase):
    def test_moonshine_listener_adapter_events(self) -> None:
        class DummyListener(STTListener):
            def __init__(self) -> None:
                self.started = ""
                self.updated = ""
                self.completed = ""

            def on_transcript_started(self, text: str) -> None:
                self.started = text

            def on_transcript_updated(self, text: str) -> None:
                self.updated = text

            def on_transcript_completed(self, text: str) -> None:
                self.completed = text

        listener = DummyListener()
        adapter = _MoonshineListenerAdapter(listener)

        mock_event = MagicMock()
        mock_event.line.text = "Testing line"

        adapter.on_line_started(mock_event)
        self.assertEqual(listener.started, "Testing line")

        adapter.on_line_text_changed(mock_event)
        self.assertEqual(listener.updated, "Testing line")

        adapter.on_line_completed(mock_event)
        self.assertEqual(listener.completed, "Testing line")

    @patch("services.stt.src.stt_service.get_model_for_language")
    @patch("services.stt.src.stt_service.Transcriber")
    def test_stt_service_lifecycle_and_double_start(
        self, mock_transcriber_class: MagicMock, mock_get_model: MagicMock
    ) -> None:
        mock_get_model.return_value = ("fake/model/path", "fake_arch")

        mock_transcriber = MagicMock()
        mock_stream = MagicMock()
        mock_transcriber.create_stream.return_value = mock_stream
        mock_transcriber_class.return_value = mock_transcriber

        service = STTService(STTConfig())
        service.start()
        self.assertIsNotNone(service._stream)

        # Double start raises ServiceAlreadyStarted
        with self.assertRaises(ServiceAlreadyStarted):
            service.start()

        service.stop()
        self.assertIsNone(service._stream)

    def test_stt_push_not_started_raises_error(self) -> None:
        service = STTService(STTConfig())
        with self.assertRaises(ServiceNotStarted):
            service.push(np.zeros(160, dtype=np.float32), 16000)

    @patch("services.stt.src.stt_service.get_model_for_language")
    @patch("services.stt.src.stt_service.Transcriber")
    def test_stt_push_queue_full(
        self, mock_transcriber_class: MagicMock, mock_get_model: MagicMock
    ) -> None:
        import queue

        mock_get_model.return_value = ("fake/model/path", "fake_arch")
        mock_transcriber = MagicMock()
        mock_transcriber.create_stream.return_value = MagicMock()
        mock_transcriber_class.return_value = mock_transcriber

        service = STTService(STTConfig(queue_size=1))
        service.start()

        audio = np.zeros(160, dtype=np.float32)
        with patch.object(service._audio_queue, "put_nowait", side_effect=queue.Full):
            with self.assertRaises(AudioQueueFull):
                service.push(audio, 16000)

        service.stop()

    def test_stt_app_console_listener(self) -> None:
        from services.stt.app import ConsoleListener
        listener = ConsoleListener()
        # Should execute methods without errors
        listener.on_transcript_started("hello")
        listener.on_transcript_updated("hello world")
        listener.on_transcript_completed("hello world.")


if __name__ == "__main__":
    unittest.main()
