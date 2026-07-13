"""Public exports for the TTS service package."""

from .tts_service import (
    SpeechQueueFull,
    TTSConfig,
    TTSListener,
    TTSService,
    TTSServiceError,
    ServiceAlreadyStarted,
    ServiceNotStarted,
)

__all__ = [
    "SpeechQueueFull",
    "TTSConfig",
    "TTSListener",
    "TTSService",
    "TTSServiceError",
    "ServiceAlreadyStarted",
    "ServiceNotStarted",
]
