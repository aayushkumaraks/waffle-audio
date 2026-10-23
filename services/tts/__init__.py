from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .src.base_tts_service import BaseTTSService
from .src.tts_service import TTSService
from .src.tts_types import (
    AudioBuffer,
    ServiceAlreadyStarted,
    ServiceNotStarted,
    SpeechQueueFull,
    TTSConfig,
    TTSListener,
    TTSServiceError,
)

if TYPE_CHECKING:
    from .src.kokoro_tts_service import KokoroTTSService
    from .src.pocket_tts_service import PocketTTSService


def __getattr__(name: str) -> Any:
    """Lazy export of backend services so they are only loaded when requested."""
    if name == "KokoroTTSService":
        from .src.kokoro_tts_service import KokoroTTSService

        return KokoroTTSService
    if name == "PocketTTSService":
        from .src.pocket_tts_service import PocketTTSService

        return PocketTTSService
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "AudioBuffer",
    "BaseTTSService",
    "KokoroTTSService",
    "PocketTTSService",
    "ServiceAlreadyStarted",
    "ServiceNotStarted",
    "SpeechQueueFull",
    "TTSConfig",
    "TTSListener",
    "TTSService",
    "TTSServiceError",
]
