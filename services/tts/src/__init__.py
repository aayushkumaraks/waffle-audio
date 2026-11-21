"""Public exports for the TTS service package."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .tts_service import (
    SpeechQueueFull,
    TTSConfig,
    TTSListener,
    TTSService,
    TTSServiceError,
    ServiceAlreadyStarted,
    ServiceNotStarted,
)

if TYPE_CHECKING:
    from .emotion_tts_service import EmotionTTSService, EmotionVoiceWorker


def __getattr__(name: str) -> Any:
    """Lazy-load emotion TTS classes so importing services.tts.src does not eagerly load PyTorch/PocketTTS."""
    if name in ("EmotionTTSService", "EmotionVoiceWorker"):
        from .emotion_tts_service import EmotionTTSService, EmotionVoiceWorker

        return EmotionTTSService if name == "EmotionTTSService" else EmotionVoiceWorker
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "EmotionTTSService",
    "EmotionVoiceWorker",
    "SpeechQueueFull",
    "TTSConfig",
    "TTSListener",
    "TTSService",
    "TTSServiceError",
    "ServiceAlreadyStarted",
    "ServiceNotStarted",
]
