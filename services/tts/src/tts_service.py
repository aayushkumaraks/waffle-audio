from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TypeAlias

import numpy as np
import numpy.typing as npt
from kokoro_onnx import Kokoro

from constants import TTS_DEFAULT_VOICE, TTS_DEFAULT_LANGUAGE

logger = logging.getLogger(__name__)

_STOP = object()

# Kokoro's ONNX model hard-caps input at 510 phonemes.  A conservative
# character ceiling well below that limit; real counts vary by language
# and vocabulary, but ~200 chars is safe for English prose.
_MAX_CHUNK_CHARS = 200


def _split_text(text: str) -> list[str]:
    """
    Split text into chunks that each fit within Kokoro's phoneme limit.

    Strategy:
    1. Split on sentence-ending punctuation (. ! ?).
    2. Any chunk still above _MAX_CHUNK_CHARS is further split on
       clause boundaries (, ; :) or, as a last resort, on whitespace.
    """
    # Split on sentence boundaries, keeping the delimiter with its sentence.
    raw = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks: list[str] = []
    for sentence in raw:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= _MAX_CHUNK_CHARS:
            chunks.append(sentence)
        else:
            # Further split on clause punctuation.
            clauses = re.split(r'(?<=[,;:])\s+', sentence)
            current = ""
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                if current and len(current) + 1 + len(clause) > _MAX_CHUNK_CHARS:
                    chunks.append(current)
                    current = clause
                else:
                    current = (current + " " + clause).strip() if current else clause
            if current:
                chunks.append(current)

    return chunks or [text]

AudioBuffer: TypeAlias = npt.NDArray[np.float32]
QueueItem: TypeAlias = str | object

@dataclass(slots=True)
class TTSConfig:
    model_path: str
    voices_path: str

class TTSListener(ABC):
    """Receives streaming synthesis events."""

    @abstractmethod
    def on_synthesis_started(self) -> None:
        ...

    @abstractmethod
    def on_audio_chunk(
        self,
        audio: AudioBuffer,
        sample_rate: int,
    ) -> None:
        ...

    @abstractmethod
    def on_synthesis_completed(self) -> None:
        ...


@dataclass(slots=True)
class TTSConfig:
    model_path: str
    voices_path: str

    voice: str = TTS_DEFAULT_VOICE
    speed: float = 1.0
    language: str = TTS_DEFAULT_LANGUAGE

    queue_size: int = 1


class TTSServiceError(Exception):
    """Base exception for the TTS service."""


class ServiceAlreadyStarted(TTSServiceError):
    """Raised when attempting to start an already running service."""


class ServiceNotStarted(TTSServiceError):
    """Raised when attempting to use or stop a service that has not been started."""


class SpeechQueueFull(TTSServiceError):
    """Raised when a speech request cannot be queued."""


class TTSService:
    """Production-ready streaming text-to-speech service."""

    def __init__(
        self,
        config: Optional[TTSConfig] = None,
    ) -> None:
        self._config = config or TTSConfig()

        self._listeners: list[TTSListener] = []

        self._speech_queue: queue.Queue[QueueItem] = queue.Queue(
            maxsize=self._config.queue_size,
        )

        self._worker_thread: Optional[threading.Thread] = None

        self._shutdown = threading.Event()

        self._engine: Optional[Kokoro] = None

    def start(self) -> None:
        """Start the TTS service."""

        if self._engine is not None:
            raise ServiceAlreadyStarted(
                "TTSService is already started."
            )

        self._shutdown.clear()

        self._engine = Kokoro(
            model_path=self._config.model_path,
            voices_path=self._config.voices_path,
        )

        self._worker_thread = threading.Thread(
            target=self._process_speech_queue,
            name="tts-worker",
            daemon=True,
        )

        self._worker_thread.start()

        logger.info("TTSService started.")

    def stop(self) -> None:
        """Stop the TTS service."""

        if self._engine is None:
            raise ServiceNotStarted(
                "TTSService is not started."
            )

        logger.info("Stopping TTSService...")

        self._shutdown.set()

        try:
            self._speech_queue.put_nowait(_STOP)
        except queue.Full:
            pass

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5)
            self._engine = None

            if self._worker_thread.is_alive():
                logger.error(
                    "Worker thread failed to stop within timeout."
                )

        self._worker_thread = None

        while True:
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("TTSService stopped.")

    def speak(self, text: str) -> None:
        """Queue text for speech synthesis."""

        if self._engine is None:
            raise ServiceNotStarted(
                "TTSService is not started."
            )

        try:
            self._speech_queue.put_nowait(text)
        except queue.Full:
            logger.warning(
                "Speech queue full. Dropping speech request."
            )
            raise SpeechQueueFull(
                "Speech queue is full."
            )

    def add_listener(
        self,
        listener: TTSListener,
    ) -> None:
        """Register a listener."""

        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(
        self,
        listener: TTSListener,
    ) -> None:
        """Remove a registered listener."""

        if listener in self._listeners:
            self._listeners.remove(listener)

    async def _synthesize(
        self,
        text: str,
    ) -> None:

        assert self._engine is not None


        logger.info("Synthesizing: %s", text)

        async for audio, sample_rate in self._engine.create_stream(
            text=text,
            voice=self._config.voice,
            speed=self._config.speed,
            lang=self._config.language,
        ):
            self._dispatch_audio_chunk(
                audio,
                sample_rate,
            )

        logger.info("Synthesis complete.")

    def _process_speech_queue(self) -> None:
        """Process queued speech requests."""

        while True:
            
            logger.info("Worker waiting for speech...")

            item = self._speech_queue.get()

            if item is _STOP:
                break

            assert isinstance(item, str)
            text = item

            try:
                chunks = _split_text(text)
                for chunk in chunks:
                    asyncio.run(self._synthesize(chunk))

            except Exception:
                logger.exception("Speech synthesis failed.")

    def _dispatch_synthesis_started(self) -> None:
        for listener in tuple(self._listeners):
            listener.on_synthesis_started()

    def _dispatch_audio_chunk(
        self,
        audio: AudioBuffer,
        sample_rate: int,
    ) -> None:
        print(f"Dispatching audio chunk: {len(audio)} samples @ {sample_rate} Hz")
        for listener in tuple(self._listeners):
            listener.on_audio_chunk(
                audio,
                sample_rate,
            )

    def _dispatch_synthesis_completed(self) -> None:
        for listener in tuple(self._listeners):
            listener.on_synthesis_completed()