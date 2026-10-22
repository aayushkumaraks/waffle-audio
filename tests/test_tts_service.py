"""Unit tests for services/tts: BaseTTSService, TTSService factory, and HF caching."""
from __future__ import annotations

import queue
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from services.tts.src.base_tts_service import BaseTTSService
from services.tts.src.hf_cache import _set_offline, load_local_first
from services.tts.src.tts_service import (
    TTSService,
    _normalize_backend,
)
from services.tts.src.tts_types import (
    ServiceAlreadyStarted,
    ServiceNotStarted,
    SpeechQueueFull,
    TTSConfig,
    TTSListener,
)


class DummyTTSListener(TTSListener):
    def __init__(self) -> None:
        self.started = False
        self.chunks: list[tuple[np.ndarray, int, str]] = []
        self.completed = False

    def on_synthesis_started(self) -> None:
        self.started = True

    def on_audio_chunk(self, audio: np.ndarray, sample_rate: int, text: str = "") -> None:
        self.chunks.append((audio, sample_rate, text))

    def on_synthesis_completed(self) -> None:
        self.completed = True


class MockConcreteTTS(BaseTTSService):
    """Concrete mock subclass of BaseTTSService for testing base queue & streaming mechanics."""

    def __init__(self, config: TTSConfig) -> None:
        super().__init__(config)
        self.loaded = False
        self.unloaded = False

    def _load_engine(self) -> None:
        self.loaded = True

    def _unload_engine(self) -> None:
        self.unloaded = True

    def _stream_audio(self, text: str):
        # Returns (iterator of audio chunks, sample_rate)
        def stream():
            yield np.ones(512, dtype=np.float32)
            yield np.full(512, 2.0, dtype=np.float32)
        return stream(), 24000


class TestTTSServiceFactory(unittest.TestCase):
    def test_normalize_backend(self) -> None:
        """Backend strings are normalized to 'pocket' or 'kokoro'."""
        self.assertEqual(_normalize_backend("kokoro"), "kokoro")
        self.assertEqual(_normalize_backend("KOKORO_ONNX"), "kokoro")
        self.assertEqual(_normalize_backend("pocket"), "pocket")
        self.assertEqual(_normalize_backend("pocket_tts"), "pocket")
        self.assertEqual(_normalize_backend(None), "pocket")
        self.assertEqual(_normalize_backend(""), "pocket")

    def test_tts_factory_backends(self) -> None:
        """TTSService creates appropriate backend instance."""
        from services.tts.src.kokoro_tts_service import KokoroTTSService
        from services.tts.src.pocket_tts_service import PocketTTSService

        pocket_svc = TTSService(TTSConfig(backend="pocket"))
        self.assertIsInstance(pocket_svc, PocketTTSService)

        kokoro_svc = TTSService(TTSConfig(backend="kokoro"))
        self.assertIsInstance(kokoro_svc, KokoroTTSService)

    def test_tts_factory_invalid_backend(self) -> None:
        """Unsupported backend raises ValueError."""
        with self.assertRaises(ValueError):
            TTSService(TTSConfig(backend="non_existent_engine"))

    def test_dynamic_getattr(self) -> None:
        """services.tts.src.tts_service exposes KokoroTTSService and PocketTTSService dynamically."""
        import services.tts.src.tts_service as tts_module

        self.assertIsNotNone(getattr(tts_module, "KokoroTTSService"))
        self.assertIsNotNone(getattr(tts_module, "PocketTTSService"))
        with self.assertRaises(AttributeError):
            getattr(tts_module, "FakeTTSService")


class TestBaseTTSService(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TTSConfig(queue_size=2)
        self.service = MockConcreteTTS(self.config)

    def tearDown(self) -> None:
        if self.service._started:
            self.service.stop()

    def test_lifecycle(self) -> None:
        """Start and stop manage engine loading and worker thread."""
        self.assertFalse(self.service._started)
        self.service.start()
        self.assertTrue(self.service._started)
        self.assertTrue(self.service.loaded)

        with self.assertRaises(ServiceAlreadyStarted):
            self.service.start()

        self.service.stop()
        self.assertFalse(self.service._started)
        self.assertTrue(self.service.unloaded)

        with self.assertRaises(ServiceNotStarted):
            self.service.stop()

    def test_speak_queue_full(self) -> None:
        """speak() adds text and raises SpeechQueueFull when bounded queue is full."""
        self.service.start()
        self.service.speak("Sentence one")
        self.service.speak("Sentence two")

        with self.assertRaises(SpeechQueueFull):
            self.service.speak("Sentence three")

    def test_interrupt(self) -> None:
        """interrupt() advances epoch, purges speech queue, and resets pending count."""
        self.service.start()
        self.service.speak("Pending item 1")
        self.service.speak("Pending item 2")

        orig_epoch = self.service._epoch
        self.service.interrupt()
        self.assertGreater(self.service._epoch, orig_epoch)
        self.assertEqual(self.service._speech_queue.qsize(), 0)
        self.assertEqual(self.service._pending, 0)

    def test_synthesize_direct(self) -> None:
        """synthesize_direct streams and concatenates audio synchronously."""
        self.service.start()
        audio, rate = self.service.synthesize_direct("Test direct synthesis")
        self.assertEqual(rate, 24000)
        self.assertEqual(len(audio), 1024)
        np.testing.assert_array_equal(audio[:512], np.ones(512, dtype=np.float32))
        np.testing.assert_array_equal(audio[512:], np.full(512, 2.0, dtype=np.float32))

    def test_synthesize_direct_not_started(self) -> None:
        """synthesize_direct on stopped service raises ServiceNotStarted."""
        with self.assertRaises(ServiceNotStarted):
            self.service.synthesize_direct("Test")

    def test_direct_chunk_streaming(self) -> None:
        """Chunks from TTS stream are dispatched directly to listeners without latency buffering."""
        listener = DummyTTSListener()
        self.service.add_listener(listener)
        self.service.start()
        self.service.speak("Hello world")

        # Wait for worker thread to process speech queue
        import time
        for _ in range(50):
            if listener.completed:
                break
            time.sleep(0.02)

        self.assertTrue(listener.completed)
        self.assertEqual(len(listener.chunks), 2)
        audio, rate, text = listener.chunks[0]
        self.assertEqual(rate, 24000)
        self.assertEqual(len(audio), 512)
        self.assertEqual(text, "Hello world")
        self.assertEqual(len(listener.chunks[1][0]), 512)


class TestHFCache(unittest.TestCase):
    def test_set_offline(self) -> None:
        """_set_offline sets HF_HUB_OFFLINE in env and hf_constants."""
        import os
        from huggingface_hub import constants as hf_constants

        _set_offline(True)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertTrue(hf_constants.HF_HUB_OFFLINE)

        _set_offline(False)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")
        self.assertFalse(hf_constants.HF_HUB_OFFLINE)

    def test_load_local_first_success(self) -> None:
        """load_local_first returns result when offline load succeeds."""
        mock_loader = MagicMock(return_value="cached_model")
        result = load_local_first(mock_loader)
        self.assertEqual(result, "cached_model")
        mock_loader.assert_called_once()

    def test_load_local_first_fallback(self) -> None:
        """load_local_first retries online when offline attempt fails."""
        mock_loader = MagicMock(side_effect=[FileNotFoundError("Cache miss"), "online_model"])
        with patch("services.tts.src.hf_cache._ensure_token"):
            result = load_local_first(mock_loader)
            self.assertEqual(result, "online_model")
            self.assertEqual(mock_loader.call_count, 2)


if __name__ == "__main__":
    unittest.main()
