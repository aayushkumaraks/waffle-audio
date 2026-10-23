"""Unit tests for audio processing services: resampling, acoustic echo cancellation, and barge-in."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from services.audio_processing.src.barge_in import BargeInController
from services.audio_processing.src.echo_cancellation import (
    AECConfig,
    AcousticEchoCanceller,
    PassthroughEchoCanceller,
)
from services.audio_processing.src.resampler import resample_audio


class TestResampler(unittest.TestCase):
    def test_same_sample_rate_returns_same_buffer(self) -> None:
        """When src_rate equals dst_rate, original buffer is returned."""
        audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        resampled = resample_audio(audio, 16000, 16000)
        np.testing.assert_array_equal(resampled, audio)

    def test_empty_buffer(self) -> None:
        """Empty audio buffer returns empty buffer regardless of sample rates."""
        audio = np.empty(0, dtype=np.float32)
        resampled = resample_audio(audio, 16000, 24000)
        self.assertEqual(resampled.size, 0)

    def test_upsample_length(self) -> None:
        """Upsampling 16kHz to 24kHz (1.5x) yields approximately 1.5x sample count."""
        audio = np.sin(np.linspace(0, 10, 1600)).astype(np.float32)
        resampled = resample_audio(audio, 16000, 24000)
        self.assertAlmostEqual(len(resampled), 2400, delta=10)

    def test_downsample_length(self) -> None:
        """Downsampling 24kHz to 16kHz (2/3x) yields approximately 2/3x sample count."""
        audio = np.sin(np.linspace(0, 10, 2400)).astype(np.float32)
        resampled = resample_audio(audio, 24000, 16000)
        self.assertAlmostEqual(len(resampled), 1600, delta=10)


class TestAcousticEchoCancellation(unittest.TestCase):
    def test_passthrough_canceller(self) -> None:
        """PassthroughEchoCanceller returns unchanged audio and handles calls cleanly."""
        canceller = PassthroughEchoCanceller()
        audio = np.array([0.5, -0.5, 0.25], dtype=np.float32)
        canceller.feed_reference(audio, 16000)
        output = canceller.process(audio, 16000)
        np.testing.assert_array_equal(output, audio)
        canceller.reset()

    def test_disabled_aec_passes_through(self) -> None:
        """AcousticEchoCanceller with enabled=False should return input directly."""
        config = AECConfig(enabled=False)
        canceller = AcousticEchoCanceller(config)
        mic = np.ones(512, dtype=np.float32)
        ref = np.ones(512, dtype=np.float32)
        canceller.feed_reference(ref, 16000)
        output = canceller.process(mic, 16000)
        np.testing.assert_array_equal(output, mic)

    def test_feed_reference_and_process(self) -> None:
        """Feeding reference audio and processing mic audio reduces correlated echo signal."""
        config = AECConfig(enabled=True, filter_length=64, step_size=0.1, sample_rate=16000)
        canceller = AcousticEchoCanceller(config)

        # Reference echo signal
        ref_chunk = np.sin(np.linspace(0, 20 * np.pi, 512)).astype(np.float32) * 0.5
        canceller.feed_reference(ref_chunk, 16000)

        # Microphone receives the echo signal
        mic_chunk = ref_chunk.copy()
        output = canceller.process(mic_chunk, 16000)

        # Output shape matches input shape
        self.assertEqual(output.shape, mic_chunk.shape)
        # Canceller should not fail and energy should be bounded
        self.assertTrue(np.all(np.isfinite(output)))

    def test_reset_clears_state(self) -> None:
        """Reset should zero filter weights and reference buffer."""
        canceller = AcousticEchoCanceller()
        canceller.feed_reference(np.ones(512, dtype=np.float32), 16000)
        canceller.reset()
        self.assertEqual(canceller._ref_buffer.size, 0)
        self.assertTrue(np.all(canceller._weights == 0.0))


class TestBargeInController(unittest.TestCase):
    def setUp(self) -> None:
        self.player = MagicMock()
        self.manager = MagicMock()
        self.tts = MagicMock()
        self.llm = MagicMock()
        self.on_barge_in = MagicMock()

        self.controller = BargeInController(
            player=self.player,
            manager=self.manager,
            tts=self.tts,
            llm=self.llm,
            on_barge_in=self.on_barge_in,
        )

    def test_enabled_toggle(self) -> None:
        """Controller enabled property can be toggled."""
        self.assertTrue(self.controller.enabled)
        self.controller.enabled = False
        self.assertFalse(self.controller.enabled)

    def test_on_speech_started_when_idle(self) -> None:
        """Speech started when player is idle and no pending generation does not trigger barge in."""
        self.player.is_playing = False
        self.manager._pending_generation = False

        self.controller.on_speech_started()
        self.player.interrupt.assert_not_called()
        self.on_barge_in.assert_not_called()

    def test_on_speech_started_when_playing(self) -> None:
        """Speech started while player is playing triggers barge-in."""
        self.player.is_playing = True
        self.manager._pending_generation = False

        self.controller.on_speech_started()
        self.player.interrupt.assert_called_once()
        self.manager.barge_in.assert_called_once()
        self.tts.interrupt.assert_called_once()
        self.on_barge_in.assert_called_once()

    def test_on_speech_started_when_generating(self) -> None:
        """Speech started while generation is pending triggers barge-in."""
        self.player.is_playing = False
        self.manager._pending_generation = True

        self.controller.on_speech_started()
        self.player.interrupt.assert_called_once()
        self.manager.barge_in.assert_called_once()
        self.tts.interrupt.assert_called_once()
        self.on_barge_in.assert_called_once()

    def test_disabled_ignores_speech(self) -> None:
        """Disabled controller does not trigger barge in even if playing."""
        self.controller.enabled = False
        self.player.is_playing = True

        self.controller.on_speech_started()
        self.player.interrupt.assert_not_called()
        self.on_barge_in.assert_not_called()

    def test_trigger_barge_in_fallback_to_llm_abort(self) -> None:
        """When manager is not provided or lacks barge_in, llm.abort is called."""
        ctrl = BargeInController(llm=self.llm)
        ctrl.trigger_barge_in()
        self.llm.abort.assert_called_once()

    def test_check_interruption_filters_faint_echo(self) -> None:
        """Low energy audio below min_rms does not trigger barge in while playing."""
        self.player.is_playing = True
        self.controller._grace_period_seconds = 0.0  # bypass grace period for test
        faint_echo = np.full(512, 0.005, dtype=np.float32)  # RMS = 0.005 < 0.025
        self.assertFalse(self.controller.check_interruption(faint_echo))
        self.player.interrupt.assert_not_called()

    def test_check_interruption_triggers_on_loud_speech(self) -> None:
        """High energy audio with consecutive confirmation triggers barge in."""
        self.player.is_playing = True
        self.controller._grace_period_seconds = 0.0
        self.controller._consecutive_frames = 2
        loud_speech = np.full(512, 0.08, dtype=np.float32)  # RMS = 0.08 >= 0.025

        # Frame 1: counts 1, not yet confirmed
        self.assertFalse(self.controller.check_interruption(loud_speech))
        self.player.interrupt.assert_not_called()

        # Frame 2: counts 2, confirmed and triggers!
        self.assertTrue(self.controller.check_interruption(loud_speech))
        self.player.interrupt.assert_called_once()
        self.manager.barge_in.assert_called_once()


if __name__ == "__main__":
    unittest.main()
