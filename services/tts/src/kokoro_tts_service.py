"""Kokoro backend retained for the feature-flag rollout."""
from __future__ import annotations
import asyncio
import queue
import threading
from collections.abc import Iterator
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kokoro_onnx import Kokoro
from constants import KOKORO_DEFAULT_LANGUAGE, KOKORO_DEFAULT_VOICE
from .base_tts_service import BaseTTSService
from .tts_types import AudioBuffer, TTSConfig

_STREAM_END = object()


class KokoroTTSService(BaseTTSService):
    def __init__(self, config: TTSConfig) -> None:
        super().__init__(config); self._engine: Kokoro | None = None
    def _load_engine(self) -> None:
        from kokoro_onnx import Kokoro
        self._engine = Kokoro(model_path=self._config.model_path, voices_path=self._config.voices_path)
    def _unload_engine(self) -> None: self._engine = None
    def _stream_audio(self, text: str) -> tuple[Iterator[AudioBuffer], int]:
        assert self._engine is not None
        chunk_queue: queue.Queue[AudioBuffer | Exception | object] = queue.Queue()
        rate_ready = threading.Event()
        sample_rate_holder = [24000]

        def _producer() -> None:
            async def _run() -> None:
                async for audio, sr in self._engine.create_stream(
                    text=text,
                    voice=self._config.voice or KOKORO_DEFAULT_VOICE,
                    speed=self._config.speed,
                    lang=self._config.language or KOKORO_DEFAULT_LANGUAGE,
                ):
                    sample_rate_holder[0] = sr
                    rate_ready.set()
                    chunk_queue.put(audio.astype(np.float32, copy=False))

            try:
                asyncio.run(_run())
            except Exception as exc:
                chunk_queue.put(exc)
            finally:
                rate_ready.set()
                chunk_queue.put(_STREAM_END)

        thread = threading.Thread(target=_producer, name="kokoro-stream", daemon=True)
        thread.start()
        rate_ready.wait(timeout=10.0)

        def _iter_chunks() -> Iterator[AudioBuffer]:
            try:
                while True:
                    item = chunk_queue.get()
                    if item is _STREAM_END:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                thread.join(timeout=2.0)

        return _iter_chunks(), sample_rate_holder[0]
