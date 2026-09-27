"""Feature-flagged TTS factory used by the rest of the application."""
from __future__ import annotations
from constants import TTS_BACKEND
from .kokoro_tts_service import KokoroTTSService
from .pocket_tts_service import PocketTTSService
from .tts_types import AudioBuffer, ServiceAlreadyStarted, ServiceNotStarted, SpeechQueueFull, TTSConfig, TTSListener, TTSServiceError
class TTSService:
    """Create the selected engine without changing application call sites.

    ``VOICECHAT_TTS_BACKEND=kokoro`` is the default. Set it to ``pocket`` to
    enable the new Pocket TTS backend.
    """
    def __new__(cls, config: TTSConfig | None = None):
        config = config or TTSConfig()
        backend = (config.backend or TTS_BACKEND).lower()
        if backend == "kokoro": return KokoroTTSService(config)
        if backend == "pocket": return PocketTTSService(config)
        raise ValueError("VOICECHAT_TTS_BACKEND must be 'kokoro' or 'pocket'.")
__all__ = ["AudioBuffer", "ServiceAlreadyStarted", "ServiceNotStarted", "SpeechQueueFull", "TTSConfig", "TTSListener", "TTSService", "TTSServiceError"]
