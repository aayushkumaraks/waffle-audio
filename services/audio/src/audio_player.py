from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np
import sounddevice as sd

from services.tts.src.tts_service import (
    AudioBuffer,
    TTSListener,
)

logger = logging.getLogger(__name__)

_STOP = object()

AudioChunk: TypeAlias = tuple[AudioBuffer, int, str]


class AudioPlaybackListener(ABC):
    """Receives events synchronized with physical audio playback."""

    @abstractmethod
    def on_playback_started(self) -> None:
        ...

    @abstractmethod
    def on_playback_text_updated(self, text: str) -> None:
        ...

    @abstractmethod
    def on_playback_completed(self) -> None:
        ...


class AudioPlayer(TTSListener):
    def __init__(self) -> None:
        self._playback_queue: queue.Queue[AudioChunk | object] = queue.Queue()

        self._worker_thread: threading.Thread | None = None

        self._shutdown = threading.Event()
        self._playing = threading.Event()
        self._synthesizing = threading.Event()
        self._playback_listeners: list[AudioPlaybackListener] = []

    def add_playback_listener(self, listener: AudioPlaybackListener) -> None:
        if listener not in self._playback_listeners:
            self._playback_listeners.append(listener)

    def remove_playback_listener(self, listener: AudioPlaybackListener) -> None:
        if listener in self._playback_listeners:
            self._playback_listeners.remove(listener)

    @property
    def is_playing(self) -> bool:
        """True while audio is actively being played back or still pending.

        Stays true across the gaps between streamed sentences so the speaker
        output of one sentence is not picked up as user speech.
        """
        return (
            self._playing.is_set()
            or self._synthesizing.is_set()
            or not self._playback_queue.empty()
        )

    def start(self) -> None:
        self._shutdown.clear()

        self._worker_thread = threading.Thread(
            target=self._playback_worker,
            name="audio-player-worker",
            daemon=True,
        )

        self._worker_thread.start()

    def stop(self) -> None:
        self._shutdown.set()

        self._playback_queue.put_nowait(_STOP)

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5)

        self._worker_thread = None

    def on_synthesis_started(self) -> None:
        self._synthesizing.set()

    def on_audio_chunk(
        self,
        audio: AudioBuffer,
        sample_rate: int,
        text: str = "",
    ) -> None:
        self._playback_queue.put_nowait((audio, sample_rate, text))

    def on_synthesis_completed(self) -> None:
        self._synthesizing.clear()

    def _playback_worker(self) -> None:
        stream: sd.OutputStream | None = None
        stream_rate: int | None = None
        accumulated_text = ""
        try:
            while True:
                item = self._playback_queue.get()

                if item is _STOP:
                    break

                if len(item) == 3:
                    audio, sample_rate, text = item
                else:
                    audio, sample_rate = item[0], item[1]
                    text = ""

                try:
                    if stream is None or stream_rate != sample_rate:
                        if stream is not None:
                            stream.close()
                        stream = sd.OutputStream(
                            samplerate=sample_rate,
                            channels=2,
                            dtype="float32",
                            blocksize=512,
                        )
                        stream.start()
                        stream_rate = sample_rate

                    self._playing.set()
                    self._dispatch_playback_started()

                    text = text.strip()
                    if text:
                        accumulated_text = (accumulated_text + " " + text).strip()
                        self._dispatch_playback_text_updated(accumulated_text)

                    stream.write(np.column_stack((audio, audio)))

                except Exception:
                    logger.exception("Audio playback failed.")
                finally:
                    self._playing.clear()
                    if self._playback_queue.empty() and not self._synthesizing.is_set():
                        self._dispatch_playback_completed()
                        accumulated_text = ""
        finally:
            if stream is not None:
                stream.close()

    def _dispatch_playback_started(self) -> None:
        for listener in tuple(self._playback_listeners):
            try:
                listener.on_playback_started()
            except Exception:
                logger.exception("Error in playback started listener.")

    def _dispatch_playback_text_updated(self, text: str) -> None:
        for listener in tuple(self._playback_listeners):
            try:
                listener.on_playback_text_updated(text)
            except Exception:
                logger.exception("Error in playback text listener.")

    def _dispatch_playback_completed(self) -> None:
        for listener in tuple(self._playback_listeners):
            try:
                listener.on_playback_completed()
            except Exception:
                logger.exception("Error in playback completed listener.")
