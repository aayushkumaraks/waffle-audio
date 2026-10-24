"""Unit tests for services/tts modules: types, hf_cache, base, pocket, kokoro, and tts_service."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from services.tts.src.tts_types import (
    ServiceAlreadyStarted,
    ServiceNotStarted,
    TTSConfig,
    TTSListener,
)
from services.tts.src.hf_cache import load_local_first
from services.tts.src.tts_service import TTSService, _normalize_backend


class TestTTSTypesAndCache(unittest.TestCase):
    def test_tts_config_defaults(self) -> None:
        cfg = TTSConfig()
        self.assertEqual(cfg.speed, 1.0)
        self.assertEqual(cfg.queue_size, 16)

    def test_normalize_backend(self) -> None:
        self.assertEqual(_normalize_backend("kokoro"), "kokoro")
        self.assertEqual(_normalize_backend("kokoro_onnx"), "kokoro")
        self.assertEqual(_normalize_backend("pocket"), "pocket")
        self.assertEqual(_normalize_backend("pocket_tts"), "pocket")
        self.assertEqual(_normalize_backend(""), "pocket")
        self.assertEqual(_normalize_backend(None), "pocket")

    def test_hf_cache_load_local_first_success(self) -> None:
        loader = MagicMock(return_value="model_instance")
        result = load_local_first(loader)
        self.assertEqual(result, "model_instance")
        self.assertEqual(loader.call_count, 1)

    def test_hf_cache_load_local_first_retry_online(self) -> None:
        calls = 0

        def flaky_loader():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("Local cache missing")
            return "online_model"

        with patch("services.tts.src.hf_cache._ensure_token"):
            result = load_local_first(flaky_loader)
            self.assertEqual(result, "online_model")
            self.assertEqual(calls, 2)


class TestTTSServiceFactory(unittest.TestCase):
    @patch("services.tts.src.pocket_tts_service.PocketTTSService.__init__", return_value=None)
    def test_factory_creates_pocket_by_default(self, mock_pocket_init: MagicMock) -> None:
        service = TTSService(TTSConfig(backend="pocket"))
        from services.tts.src.pocket_tts_service import PocketTTSService
        self.assertIsInstance(service, PocketTTSService)

    @patch("services.tts.src.kokoro_tts_service.KokoroTTSService.__init__", return_value=None)
    def test_factory_creates_kokoro_when_requested(self, mock_kokoro_init: MagicMock) -> None:
        service = TTSService(TTSConfig(backend="kokoro"))
        from services.tts.src.kokoro_tts_service import KokoroTTSService
        self.assertIsInstance(service, KokoroTTSService)

    def test_factory_invalid_backend_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            TTSService(TTSConfig(backend="unsupported_engine"))


class TestBaseTTSServicePlumbing(unittest.TestCase):
    def test_base_tts_service_lifecycle_and_interrupt(self) -> None:
        from services.tts.src.base_tts_service import BaseTTSService

        class MockConcreteTTS(BaseTTSService):
            def _load_engine(self) -> None:
                pass

            def _unload_engine(self) -> None:
                pass

            def _stream_audio(self, text: str):
                return iter([np.zeros(240, dtype=np.float32)]), 24000

        service = MockConcreteTTS(TTSConfig(queue_size=2))

        # Not started errors
        with self.assertRaises(ServiceNotStarted):
            service.speak("hello")
        with self.assertRaises(ServiceNotStarted):
            service.stop()

        service.start()
        self.assertTrue(service._started)

        # Double start raises ServiceAlreadyStarted
        with self.assertRaises(ServiceAlreadyStarted):
            service.start()

        # Listeners
        class DummyListener(TTSListener):
            def on_synthesis_started(self) -> None:
                pass

            def on_audio_chunk(self, audio, sample_rate: int, text: str = "") -> None:
                pass

            def on_synthesis_completed(self) -> None:
                pass

        listener = DummyListener()
        service.add_listener(listener)
        self.assertIn(listener, service._listeners)
        service.remove_listener(listener)
        self.assertNotIn(listener, service._listeners)

        # Direct synthesis
        audio, rate = service.synthesize_direct("Quick test")
        self.assertEqual(rate, 24000)
        self.assertEqual(len(audio), 240)

        # Interrupt clears speech queue
        service.speak("Queued speech")
        service.interrupt()
        self.assertEqual(service._speech_queue.qsize(), 0)

        service.stop()
        self.assertFalse(service._started)


class TestTTSApp(unittest.TestCase):
    def test_completion_listener(self) -> None:
        from services.tts.app import _CompletionListener
        listener = _CompletionListener()
        self.assertFalse(listener.completed.is_set())
        listener.on_synthesis_started()
        listener.on_audio_chunk(np.zeros(100), 24000)
        self.assertEqual(listener.chunk_count, 1)
        listener.on_synthesis_completed()
        self.assertTrue(listener.completed.is_set())


if __name__ == "__main__":
    unittest.main()
