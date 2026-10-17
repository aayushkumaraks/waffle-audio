"""Instant Interruption (Barge-In) Controller.

Coordinates stopping assistant speech immediately when user speech is confirmed:
1. Stops and flushes the sounddevice output buffer.
2. Aborts LLM token streaming.
3. Purges the pending text-to-TTS queue.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

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
    ) -> None:
        self._player = player
        self._manager = manager
        self._tts = tts
        self._llm = llm
        self._on_barge_in = on_barge_in
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val

    def on_speech_started(self) -> None:
        """Called immediately when VAD confirms user voice activity."""
        if not self._enabled:
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
        pass

    def trigger_barge_in(self) -> None:
        """Execute instant interruption signals across all active pipeline stages."""
        logger.info("Executing instant barge-in interruption signals.")

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
