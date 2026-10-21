from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import Callable, TypeAlias

import numpy as np
import sounddevice as sd

from constants import AUDIO_OUTPUT_DEVICE
from services.tts.src.tts_types import (
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
    """Plays TTS audio and reports playback events per *response turn*.

    ``on_playback_completed`` fires once, when every queued chunk has played,
    TTS is no longer synthesizing, and the optional upstream source (the TTS
    service's pending queue) is idle.  Gaps between streamed clauses therefore
    do not end the turn.
    """

    # Frames written per stream.write call; bounds interrupt latency
    # (2048 frames is ~85 ms at 24 kHz) without cross-thread stream aborts.
    _WRITE_SLICE = 2048

    def __init__(self) -> None:
        self._playback_queue: queue.Queue[tuple | object] = queue.Queue()

        self._worker_thread: threading.Thread | None = None

        self._shutdown = threading.Event()
        self._playing = threading.Event()
        self._synthesizing = threading.Event()
        self._interrupted = threading.Event()
        self._playback_listeners: list[AudioPlaybackListener] = []
        self._monitor_callbacks: list[Callable[[AudioBuffer, int], None]] = []
        self._upstream_busy: Callable[[], bool] | None = None

        self._state_lock = threading.Lock()
        self._turn_active = False
        self._accumulated_text = ""

    def add_playback_listener(self, listener: AudioPlaybackListener) -> None:
        if listener not in self._playback_listeners:
            self._playback_listeners.append(listener)

    def remove_playback_listener(self, listener: AudioPlaybackListener) -> None:
        if listener in self._playback_listeners:
            self._playback_listeners.remove(listener)

    def add_playback_monitor(self, callback: Callable[[AudioBuffer, int], None]) -> None:
        """Register a callback that receives reference audio for echo cancellation."""
        if callback not in self._monitor_callbacks:
            self._monitor_callbacks.append(callback)

    def remove_playback_monitor(self, callback: Callable[[AudioBuffer, int], None]) -> None:
        if callback in self._monitor_callbacks:
            self._monitor_callbacks.remove(callback)

    def set_upstream_busy(self, predicate: Callable[[], bool] | None) -> None:
        """Register a predicate reporting queued-but-not-yet-synthesized speech."""
        self._upstream_busy = predicate

    def _is_upstream_busy(self) -> bool:
        if self._upstream_busy is None:
            return False
        try:
            return bool(self._upstream_busy())
        except Exception:
            logger.exception("Upstream busy predicate failed.")
            return False

    def interrupt(self) -> None:
        """Stop playback immediately and discard all queued audio.

        Any pending queue items are purged, ongoing stream write breaks
        within _WRITE_SLICE frames, and turn is closed without dispatching
        on_playback_completed.
        """
        with self._state_lock:
            self._interrupted.set()
            self._turn_active = False
            self._accumulated_text = ""
            stop_seen = False
            while True:
                try:
                    item = self._playback_queue.get_nowait()
                except queue.Empty:
                    break
                if item is _STOP:
                    stop_seen = True
            if stop_seen:
                self._playback_queue.put_nowait(_STOP)
            self._synthesizing.clear()

    @property
    def is_playing(self) -> bool:
        """True while audio is playing or more speech for this turn is pending.

        Stays true across the gaps between streamed clauses so the speaker
        output of one clause is not picked up as user speech.
        """
        return (
            self._playing.is_set()
            or self._synthesizing.is_set()
            or not self._playback_queue.empty()
            or self._is_upstream_busy()
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
        # The last chunk may already have finished playing; close the turn now.
        self._maybe_complete_turn()

    def _maybe_complete_turn(self) -> None:
        with self._state_lock:
            if not self._turn_active:
                return
            if (
                self._playing.is_set()
                or self._synthesizing.is_set()
                or not self._playback_queue.empty()
                or self._is_upstream_busy()
            ):
                return
            self._turn_active = False
            finished_text = self._accumulated_text
            self._accumulated_text = ""
        logger.info("[AUDIO BENCHMARK] Audio playback completed for turn (spoken='%.40s...')", finished_text)
        self._dispatch_playback_completed()


    def _playback_worker(self) -> None:
        stream: sd.OutputStream | None = None
        stream_rate: int | None = None
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

                with self._state_lock:
                    if self._interrupted.is_set():
                        self._interrupted.clear()
                        continue
                    self._turn_active = True
                    self._playing.set()
                    text = text.strip()
                    if text:
                        self._accumulated_text = (
                            self._accumulated_text + " " + text
                        ).strip()
                    accumulated = self._accumulated_text

                try:
                    if stream is None or stream_rate != sample_rate:
                        if stream is not None:
                            stream.close()
                            stream = None
                        stream = sd.OutputStream(
                            samplerate=sample_rate,
                            channels=2,
                            dtype="float32",
                            blocksize=512,
                            device=AUDIO_OUTPUT_DEVICE,
                        )
                        stream.start()
                        stream_rate = sample_rate

                    self._dispatch_playback_started()
                    self._dispatch_playback_monitor(audio, sample_rate)
                    if text:
                        self._dispatch_playback_text_updated(accumulated)

                    stereo = np.column_stack((audio, audio)).astype(np.float32, copy=False)
                    for start in range(0, len(stereo), self._WRITE_SLICE):
                        if self._interrupted.is_set():
                            break  # interrupted mid-chunk
                        stream.write(stereo[start:start + self._WRITE_SLICE])

                except Exception:
                    logger.exception("Audio playback failed.")
                    # Recreate the device stream on the next chunk.
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass
                    stream = None
                    stream_rate = None
                finally:
                    self._playing.clear()
                    self._maybe_complete_turn()
        finally:
            if stream is not None:
                stream.close()

    def _dispatch_playback_monitor(self, audio: AudioBuffer, sample_rate: int) -> None:
        for cb in tuple(self._monitor_callbacks):
            try:
                cb(audio, sample_rate)
            except Exception:
                logger.exception("Error in playback monitor callback.")


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
