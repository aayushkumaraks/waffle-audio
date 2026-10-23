"""Instant Interruption (Barge-In) Controller.

Coordinates stopping assistant speech immediately when user speech is confirmed:
1. Stops and flushes the sounddevice output buffer.
2. Aborts LLM token streaming.
3. Purges the pending text-to-TTS queue.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np

from constants import (
    BARGE_IN_CONSECUTIVE_FRAMES,
    BARGE_IN_MIN_RMS,
)

logger = logging.getLogger(__name__)


class BargeInController:
    """Interruption controller that wires VAD detection to audio/LLM/TTS cancellation."""

    def __init__(
        self,
        *,
        player: Any = None,
        manager: Any = None,
        tts: Any = None,
        llm: Any = None,
        on_barge_in: Callable[[], None] | None = None,
        min_rms: float = BARGE_IN_MIN_RMS,
        consecutive_frames: int = BARGE_IN_CONSECUTIVE_FRAMES,
        grace_period_seconds: float = 0.25,
    ) -> None:
        self._player = player
        self._manager = manager
        self._tts = tts
        self._llm = llm
        self._on_barge_in = on_barge_in
        self._enabled = True
        self._min_rms = max(0.0, float(min_rms))
        self._consecutive_frames = max(1, int(consecutive_frames))
        self._grace_period_seconds = max(0.0, float(grace_period_seconds))
        self._speech_frame_count = 0
        self._playback_start_time: float | None = None
        self._was_playing = False
        self._last_audio: Any = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val

    @property
    def min_rms(self) -> float:
        return self._min_rms

    @min_rms.setter
    def min_rms(self, val: float) -> None:
        self._min_rms = max(0.0, float(val))

    def feed_audio(self, audio: Any) -> None:
        """Cache the most recent microphone frame for energy and echo analysis."""
        if audio is not None and hasattr(audio, "__len__") and len(audio) > 0:
            self._last_audio = audio

    def on_speech_started(self, audio: Any = None) -> None:
        """Called when VAD confirms user voice activity."""
        if not self._enabled:
            return

        if audio is not None:
            self.check_interruption(audio)
            return

        if self._last_audio is not None and self._player is not None and getattr(self._player, "is_playing", False):
            self.check_interruption(self._last_audio)
            return

        is_busy = False
        if self._player is not None and getattr(self._player, "is_playing", False):
            is_busy = True

        if self._manager is not None and getattr(self._manager, "_pending_generation", False):
            is_busy = True

        if is_busy:
            self.trigger_barge_in()

    def on_speech_ended(self) -> None:
        """Called when user speech finishes."""
        self._speech_frame_count = 0

    def check_interruption(self, audio: Any = None) -> bool:
        """Evaluate whether audio constitutes genuine user barge-in."""
        if not self._enabled:
            return False

        is_playing = bool(self._player is not None and getattr(self._player, "is_playing", False))
        is_generating = bool(self._manager is not None and getattr(self._manager, "_pending_generation", False))

        if not (is_playing or is_generating):
            self._speech_frame_count = 0
            self._was_playing = False
            self._playback_start_time = None
            return False

        now = time.monotonic()
        if is_playing:
            if not self._was_playing or self._playback_start_time is None:
                self._playback_start_time = now
                self._was_playing = True
            # Grace period after speech playback onset (skip transient room echoes)
            if (now - self._playback_start_time) < self._grace_period_seconds:
                return False
        else:
            self._was_playing = False
            self._playback_start_time = None

        target_audio = audio if audio is not None else self._last_audio
        if target_audio is not None and hasattr(target_audio, "__len__") and len(target_audio) > 0:
            arr = np.asarray(target_audio, dtype=np.float32)
            rms = float(np.sqrt(np.mean(np.square(arr))))
            if rms < self._min_rms:
                self._speech_frame_count = 0
                return False

        self._speech_frame_count += 1
        if self._speech_frame_count < self._consecutive_frames:
            return False

        self._speech_frame_count = 0
        self.trigger_barge_in()
        return True

    def trigger_barge_in(self) -> None:
        """Execute instant interruption signals across all active pipeline stages."""
        logger.info("Executing instant barge-in interruption signals.")
        self._speech_frame_count = 0
        self._was_playing = False
        self._playback_start_time = None

        # 1. Stop and flush audio playback output buffer
        if self._player is not None and hasattr(self._player, "interrupt"):
            try:
                self._player.interrupt()
            except Exception:
                logger.exception("Error interrupting audio player on barge-in.")

        # 2. Abort LLM token streaming
        if self._manager is not None and hasattr(self._manager, "barge_in"):
            try:
                self._manager.barge_in()
            except Exception:
                logger.exception("Error triggering conversation manager barge-in.")
        elif self._llm is not None and hasattr(self._llm, "abort"):
            try:
                self._llm.abort()
            except Exception:
                logger.exception("Error aborting LLM on barge-in.")

        # 3. Purge text-to-TTS queue
        if self._tts is not None and hasattr(self._tts, "interrupt"):
            try:
                self._tts.interrupt()
            except Exception:
                logger.exception("Error interrupting TTS on barge-in.")

        # 4. Notify UI / external callbacks
        if self._on_barge_in is not None:
            try:
                self._on_barge_in()
            except Exception:
                logger.exception("Error in on_barge_in callback.")

