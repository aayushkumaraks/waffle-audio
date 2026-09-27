"""Pocket TTS backend implementation."""
from __future__ import annotations
import os
from collections.abc import Iterator
from pathlib import Path
import numpy as np
from constants import POCKET_TTS_CACHE_DIR, POCKET_TTS_DEFAULT_LANGUAGE, POCKET_TTS_DEFAULT_VOICE

# Pocket TTS uses Hugging Face downloads for model weights and bundled voices.
# Pin their cache to a project-owned, persistent directory before importing it.
os.environ.setdefault("HF_HOME", str(Path(POCKET_TTS_CACHE_DIR).resolve()))

from pocket_tts import TTSModel
from .base_tts_service import BaseTTSService
from .tts_types import AudioBuffer, TTSConfig
class PocketTTSService(BaseTTSService):
    def __init__(self, config: TTSConfig) -> None:
        super().__init__(config); self._engine: TTSModel | None = None; self._voice_state: object | None = None
    def _load_engine(self) -> None:
        engine = TTSModel.load_model(language=self._config.language or POCKET_TTS_DEFAULT_LANGUAGE, quantize=self._config.quantize)
        self._voice_state = engine.get_state_for_audio_prompt(self._config.voice or POCKET_TTS_DEFAULT_VOICE); self._engine = engine
    def _unload_engine(self) -> None:
        self._engine = None; self._voice_state = None
    def _stream_audio(self, text: str) -> tuple[Iterator[AudioBuffer], int]:
        assert self._engine is not None and self._voice_state is not None
        def stream() -> Iterator[AudioBuffer]:
            for chunk in self._engine.generate_audio_stream(self._voice_state, text):
                yield chunk.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
        return stream(), self._engine.sample_rate
