"""Feature-flagged TTS factory used by the rest of the application."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import constants
from constants import TTS_BACKEND
from .tts_types import (
    AudioBuffer,
    ServiceAlreadyStarted,
    ServiceNotStarted,
    SpeechQueueFull,
    TTSConfig,
    TTSListener,
    TTSServiceError,
)

if TYPE_CHECKING:
    from .kokoro_tts_service import KokoroTTSService
    from .pocket_tts_service import PocketTTSService

def _normalize_backend(name: str | None) -> str:
    """Normalize backend identifier to 'pocket' or 'kokoro', defaulting to 'pocket'."""
    val = (name or "").strip().lower()
    if val in {"kokoro", "kokoro_onnx"}:
        return "kokoro"
    if val in {"pocket", "pockettts", "pocket_tts"}:
        return "pocket"
    return "pocket" if not val else val

# Determine backend from constants.py, defaulting to pocket
_ACTIVE_BACKEND = _normalize_backend(getattr(constants, "TTS_BACKEND", TTS_BACKEND))

if _ACTIVE_BACKEND == "kokoro":
    from .kokoro_tts_service import KokoroTTSService  # type: ignore[assignment]
else:
    from .pocket_tts_service import PocketTTSService  # type: ignore[assignment]


class TTSService:
    """Create the selected engine without changing application call sites.

    Default backend is 'pocket' (PocketTTS). Set TTS_BACKEND in constants.py or
    VOICECHAT_TTS_BACKEND=kokoro in the environment to switch engines.
    """

    def __new__(cls, config: TTSConfig | None = None) -> Any:
        config = config or TTSConfig()
        backend = _normalize_backend(config.backend or TTS_BACKEND)
        if backend == "kokoro":
            from .kokoro_tts_service import KokoroTTSService
            return KokoroTTSService(config)
        if backend == "pocket":
            from .pocket_tts_service import PocketTTSService
            return PocketTTSService(config)
        raise ValueError(f"TTS backend must be 'pocket' or 'kokoro', got '{backend}'.")


def __getattr__(name: str) -> Any:
    """Lazy export of individual backend service classes if requested directly."""
    if name == "KokoroTTSService":
        from .kokoro_tts_service import KokoroTTSService
        return KokoroTTSService
    if name == "PocketTTSService":
        from .pocket_tts_service import PocketTTSService
        return PocketTTSService
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "AudioBuffer",
    "ServiceAlreadyStarted",
    "ServiceNotStarted",
    "SpeechQueueFull",
    "TTSConfig",
    "TTSListener",
    "TTSService",
    "TTSServiceError",
]

