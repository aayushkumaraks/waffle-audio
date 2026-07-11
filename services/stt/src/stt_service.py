from __future__ import annotations

import numpy as np
import numpy.typing as npt

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Optional, TypeAlias
from abc import ABC, abstractmethod

from moonshine_voice import Stream, TranscriptEventListener, Transcriber
from moonshine_voice.download import get_model_for_language

logger = logging.getLogger(__name__)

_STOP = object()
AudioBuffer: TypeAlias = npt.NDArray[np.float32]
AudioChunk: TypeAlias = tuple[AudioBuffer, int]
QueueItem: TypeAlias = AudioChunk | object


class STTListener(ABC):
    @abstractmethod
    def on_transcript_started(self, text: str) -> None:
        pass

    @abstractmethod
    def on_transcript_updated(self, text: str) -> None:
        pass

    @abstractmethod
    def on_transcript_completed(self, text: str) -> None:
        pass

class _MoonshineListenerAdapter(TranscriptEventListener):
    def __init__(self, listener: STTListener) -> None:
        self._listener = listener

    def on_line_started(self, event) -> None:
        self._listener.on_transcript_started(event.line.text)

    def on_line_text_changed(self, event) -> None:
        self._listener.on_transcript_updated(event.line.text)

    def on_line_completed(self, event) -> None:
        self._listener.on_transcript_completed(event.line.text)


@dataclass(slots=True)
class STTConfig:
    language: str = "en"
    update_interval: float = 0.5
    queue_size: int = 200


class STTServiceError(Exception):
    """Base exception for the STT service."""


class ServiceAlreadyStarted(STTServiceError):
    pass


class ServiceNotStarted(STTServiceError):
    pass


class AudioQueueFull(STTServiceError):
    pass


class STTService:
    """Production-ready wrapper around Moonshine streaming transcription."""

    def __init__(self, config: Optional[STTConfig] = None) -> None:
        self._config = config or STTConfig()

        self._transcriber: Optional[Transcriber] = None
        self._stream: Optional[Stream] = None
        
        self._listener_adapters: dict[
            STTListener,
            _MoonshineListenerAdapter,
        ] = {}
        self._listeners: set[STTListener] = set()

        self._audio_queue: queue.Queue[QueueItem] = queue.Queue(
            maxsize=self._config.queue_size
        )

        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    def start(self) -> None:
        if self._stream is not None:
            raise ServiceAlreadyStarted("STTService is already running.")

        logger.info("Loading Moonshine model...")

        model_path, model_arch = get_model_for_language(
            self._config.language
        )

        self._transcriber = Transcriber(
            model_path=model_path,
            model_arch=model_arch,
            update_interval=self._config.update_interval,
        )

        self._stream = self._transcriber.create_stream(
            update_interval=self._config.update_interval
        )

        self._stream.start()

        for listener in self._listeners:
            adapter = _MoonshineListenerAdapter(listener)
            self._listener_adapters[listener] = adapter
            self._stream.add_listener(adapter)

        self._shutdown.clear()

        self._worker_thread = threading.Thread(
            target=self._process_audio_queue,
            name="stt-worker",
            daemon=True,
        )

        self._worker_thread.start()

        logger.info("STTService started.")

    def stop(self) -> None:
        if self._stream is None:
            return

        logger.info("Stopping STTService...")

        self._shutdown.set()

        try:
            self._audio_queue.put_nowait(_STOP)
        except queue.Full:
            pass

        if self._worker_thread is not None:

            self._worker_thread.join(timeout=5)

            if self._worker_thread.is_alive():
                logger.error("Worker thread failed to stop within timeout.")

        try:
            self._stream.stop()
        finally:
            self._stream.close()

        self._stream = None
        self._transcriber = None
        self._worker_thread = None

        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("STTService stopped.")

    def add_listener(self, listener: STTListener) -> None:
        if listener in self._listeners:
            return

        self._listeners.add(listener)

        if self._stream is None:
            return

        adapter = _MoonshineListenerAdapter(listener)
        self._listener_adapters[listener] = adapter

        self._stream.add_listener(adapter)

    def remove_listener(self, listener: STTListener) -> None:
        self._listeners.discard(listener)

        adapter = self._listener_adapters.pop(listener, None)

        if adapter is None:
            return

        if self._stream is not None:
            self._stream.remove_listener(adapter)

    def push(self, audio_data: AudioBuffer, sample_rate: int) -> None:
        """
        Queue normalized float32 audio samples for transcription.
        """

        self._require_started()

        try:
            self._audio_queue.put_nowait((audio_data, sample_rate))
        except queue.Full:
            logger.warning("Audio queue full. Dropping audio chunk.")
            raise AudioQueueFull("Audio queue is full.")

    def _process_audio_queue(self) -> None:
        while True:
            item = self._audio_queue.get()

            if item is _STOP:
                break

            batch: list[AudioChunk] = [item]
            stop_requested = False

            while True:
                try:
                    queued = self._audio_queue.get_nowait()
                except queue.Empty:
                    break

                if queued is _STOP:
                    stop_requested = True
                    break

                batch.append(queued)

            audio_chunks = [audio for audio, _ in batch]
            sample_rate = batch[0][1]

            audio_data = (
                audio_chunks[0]
                if len(audio_chunks) == 1
                else np.concatenate(audio_chunks)
            )

            try:
                if self._stream is not None:
                    self._stream.add_audio(audio_data, sample_rate)
            except Exception:
                logger.exception("Error while processing audio batch.")

            if stop_requested:
                break

    def _require_started(self) -> None:
        if self._stream is None:
            raise ServiceNotStarted("STTService has not been started.")

    def __enter__(self) -> STTService:
        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ) -> None:
        self.stop()