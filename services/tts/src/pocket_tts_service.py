"""Pocket TTS backend implementation."""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from constants import (
    POCKET_TTS_CACHE_DIR,
    POCKET_TTS_DEFAULT_LANGUAGE,
    POCKET_TTS_DEFAULT_VOICE,
    resolve_voice_spec,
)

# Pocket TTS uses Hugging Face downloads for model weights and bundled voices.
# Pin their cache to a project-owned, persistent directory before importing it.
os.environ.setdefault("HF_HOME", str(Path(POCKET_TTS_CACHE_DIR).resolve()))

from .base_tts_service import BaseTTSService
from .hf_cache import load_local_first
from .tts_types import AudioBuffer, TTSConfig

if TYPE_CHECKING:
    from pocket_tts import TTSModel

logger = logging.getLogger(__name__)


class PocketTTSService(BaseTTSService):
    def __init__(self, config: TTSConfig) -> None:
        super().__init__(config)
        self._engine: Any = None
        self._voice_state: object | None = None

    def _load_engine(self) -> None:
        t0 = time.perf_counter()
        voice_target = resolve_voice_spec(self._config.voice or POCKET_TTS_DEFAULT_VOICE)
        language = self._config.language or POCKET_TTS_DEFAULT_LANGUAGE
        logger.info(
            "Loading Pocket TTS model weights and voice '%s' (language='%s', quantize=%s)...",
            voice_target,
            language,
            self._config.quantize,
        )

        def load() -> tuple[Any, object]:
            logger.info("Importing pocket_tts engine...")
            from pocket_tts import TTSModel

            logger.info("Loading TTSModel (language='%s', quantize=%s)...", language, self._config.quantize)
            engine = TTSModel.load_model(language=language, quantize=self._config.quantize)
            logger.info("Computing voice state for '%s'...", voice_target)
            voice_state = engine.get_state_for_audio_prompt(voice_target)
            return engine, voice_state

        # Local cache first (no network); contact Hugging Face only if data is missing.
        engine, self._voice_state = load_local_first(load)
        self._engine = engine
        elapsed = time.perf_counter() - t0
        logger.info(
            "Pocket TTS model and voice '%s' loaded successfully in %.2fs (sample_rate=%dHz).",
            voice_target,
            elapsed,
            getattr(engine, "sample_rate", 24000),
        )

    def _unload_engine(self) -> None:
        self._engine = None
        self._voice_state = None

    def _stream_audio(self, text: str) -> tuple[Iterator[AudioBuffer], int]:
        assert self._engine is not None and self._voice_state is not None

        def stream() -> Iterator[AudioBuffer]:
            for chunk in self._engine.generate_audio_stream(self._voice_state, text):
                yield chunk.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=True)

        return stream(), self._engine.sample_rate

