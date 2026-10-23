"""Feature-flagged TTS factory used by the rest of the application."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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

logger = logging.getLogger(__name__)


def _normalize_backend(name: str | None) -> str:
    """Normalize backend identifier to 'pocket' or 'kokoro', defaulting to 'pocket'."""
    val = (name or "").strip().lower()
    if val in {"kokoro", "kokoro_onnx"}:
        return "kokoro"
    if val in {"pocket", "pockettts", "pocket_tts"}:
        return "pocket"
    return "pocket" if not val else val


class TTSService:
    """Create the selected engine without changing application call sites.

    Default backend is 'pocket' (PocketTTS). Set TTS_BACKEND in constants.py or
    VOICECHAT_TTS_BACKEND=kokoro in the environment to switch engines.
    """

    def __new__(cls, config: TTSConfig | None = None) -> Any:
        config = config or TTSConfig()
        backend = _normalize_backend(config.backend or TTS_BACKEND)
        logger.info(
            "Initializing TTS service (backend: '%s', voice: '%s', language: '%s')...",
            backend,
            config.voice or "default",
            config.language or "default",
        )
        if backend == "kokoro":
            logger.info("Conditionally loading KokoroTTSService backend...")
            from .kokoro_tts_service import KokoroTTSService

            return KokoroTTSService(config)
        if backend == "pocket":
            logger.info("Conditionally loading PocketTTSService backend...")
            from .pocket_tts_service import PocketTTSService

            return PocketTTSService(config)
        raise ValueError(f"TTS backend must be 'pocket' or 'kokoro', got '{backend}'.")


def __getattr__(name: str) -> Any:
    """Lazy export of individual backend service classes if requested directly."""
    if name == "KokoroTTSService":
        logger.info("Conditionally loading KokoroTTSService via module attribute access...")
        from .kokoro_tts_service import KokoroTTSService

        return KokoroTTSService
    if name == "PocketTTSService":
        logger.info("Conditionally loading PocketTTSService via module attribute access...")
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

