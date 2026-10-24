"""Unit tests for services/audio_processing modules: resampler, echo cancellation, barge-in."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from services.audio_processing.src.resampler import resample_audio
from services.audio_processing.src.echo_cancellation import (
    AECConfig,
    AcousticEchoCanceller,
    PassthroughEchoCanceller,
)
from services.audio_processing.src.barge_in import BargeInController


class TestResampler(unittest.TestCase):
    def test_resample_same_rate_or_empty(self) -> None:
        empty = np.zeros(0, dtype=np.float32)
        self.assertEqual(len(resample_audio(empty, 16000, 24000)), 0)

        audio = np.ones(100, dtype=np.float32)
        res = resample_audio(audio, 16000, 16000)
        self.assertIs(res, audio)

    def test_resample_upsample_and_downsample(self) -> None:
        src = np.sin(np.linspace(0, 2 * np.pi, 160)).astype(np.float32)
        # 160 samples at 16kHz is 10ms -> at 24kHz should be 240 samples
        up = resample_audio(src, 16000, 24000)
        self.assertEqual(len(up), 240)

        # Downsample back to 16kHz
        down = resample_audio(up, 24000, 16000)
        self.assertEqual(len(down), 160)

    def test_resample_fallback_linear(self) -> None:
        src = np.ones(100, dtype=np.float32)
        with patch("services.audio_processing.src.resampler._SOXR_AVAILABLE", False), patch(
            "services.audio_processing.src.resampler._SCIPY_AVAILABLE", False
        ):
            out = resample_audio(src, 1000, 2000)
            self.assertEqual(len(out), 200)


class TestEchoCancellation(unittest.TestCase):
    def test_passthrough_canceller(self) -> None:
        canceller = PassthroughEchoCanceller()
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        canceller.feed_reference(audio, 16000)
        out = canceller.process(audio, 16000)
        np.testing.assert_array_equal(out, audio)
        canceller.reset()

    def test_acoustic_echo_canceller_disabled(self) -> None:
        cfg = AECConfig(enabled=False)
        canceller = AcousticEchoCanceller(cfg)
        audio = np.ones(50, dtype=np.float32)
        canceller.feed_reference(audio, 16000)
        out = canceller.process(audio, 16000)
        np.testing.assert_array_equal(out, audio)

    def test_acoustic_echo_canceller_adaptive_reduction(self) -> None:
        cfg = AECConfig(enabled=True, filter_length=64, step_size=0.1)
        canceller = AcousticEchoCanceller(cfg)

        # Synthetic reference audio
        ref = np.sin(np.linspace(0, 100, 1600)).astype(np.float32)
        canceller.feed_reference(ref, 16000)

        # Microphone picks up the exact same signal (100% echo)
        mic = ref.copy()
        cleaned = canceller.process(mic, 16000)

        self.assertEqual(len(cleaned), len(mic))
        # NLMS should adaptively reduce the energy over time
        initial_energy = np.mean(cleaned[:100] ** 2)
        final_energy = np.mean(cleaned[-100:] ** 2)
        self.assertLess(final_energy, initial_energy + 1e-4)

        canceller.reset()
        self.assertEqual(canceller._weights.sum(), 0.0)
        self.assertEqual(len(canceller._ref_buffer), 0)


class TestBargeInController(unittest.TestCase):
    def test_barge_in_controller_lifecycle(self) -> None:
        mock_player = MagicMock()
        mock_manager = MagicMock()
        mock_tts = MagicMock()
        mock_llm = MagicMock()
        barge_in_callback = MagicMock()

        controller = BargeInController(
            player=mock_player,
            manager=mock_manager,
            tts=mock_tts,
            llm=mock_llm,
            on_barge_in=barge_in_callback,
        )

        self.assertTrue(controller.enabled)
        controller.enabled = False
        self.assertFalse(controller.enabled)
        controller.enabled = True

        mock_player.is_playing = True
        controller.on_speech_started()

        # Should interrupt all active services
        mock_player.interrupt.assert_called_once()
        mock_manager.barge_in.assert_called_once()
        mock_tts.interrupt.assert_called_once()
        barge_in_callback.assert_called_once()
        controller.on_speech_ended()



if __name__ == "__main__":
    unittest.main()
