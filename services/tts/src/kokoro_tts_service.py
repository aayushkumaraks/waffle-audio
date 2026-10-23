"""Kokoro backend retained for the feature-flag rollout."""
from __future__ import annotations

import asyncio
from collections.abc import Iterator
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from constants import KOKORO_DEFAULT_LANGUAGE, KOKORO_DEFAULT_VOICE
from .base_tts_service import BaseTTSService
from .tts_types import AudioBuffer, TTSConfig

if TYPE_CHECKING:
    from kokoro_onnx import Kokoro

logger = logging.getLogger(__name__)


class KokoroTTSService(BaseTTSService):
    def __init__(self, config: TTSConfig) -> None:
        super().__init__(config)
        self._engine: Any = None

    def _load_engine(self) -> None:
        t0 = time.perf_counter()
        logger.info(
            "Loading Kokoro TTS ONNX model from '%s' (voices='%s')...",
            self._config.model_path,
            self._config.voices_path,
        )
        from kokoro_onnx import Kokoro

        self._engine = Kokoro(model_path=self._config.model_path, voices_path=self._config.voices_path)
        elapsed = time.perf_counter() - t0
        logger.info("Kokoro TTS ONNX engine loaded successfully in %.2fs.", elapsed)

    def _unload_engine(self) -> None:
        self._engine = None

    def _stream_audio(self, text: str) -> tuple[Iterator[AudioBuffer], int]:
        assert self._engine is not None
        chunks: list[AudioBuffer] = []
        sample_rate = 24000

        async def collect() -> None:
            nonlocal sample_rate
            async for audio, sample_rate in self._engine.create_stream(
                text=text,
                voice=self._config.voice or KOKORO_DEFAULT_VOICE,
                speed=self._config.speed,
                lang=self._config.language or KOKORO_DEFAULT_LANGUAGE,
            ):
                chunks.append(audio.astype(np.float32, copy=False))

        asyncio.run(collect())
        return iter(chunks), sample_rate

