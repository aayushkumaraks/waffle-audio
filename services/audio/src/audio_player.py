from __future__ import annotations

import logging
import queue
import threading
from typing import TypeAlias

import numpy as np

import sounddevice as sd

from services.tts.src.tts_service import (
    AudioBuffer,
    TTSListener,
)

logger = logging.getLogger(__name__)

_STOP = object()

AudioChunk: TypeAlias = tuple[AudioBuffer, int]


class AudioPlayer(TTSListener):
    def __init__(self) -> None:
        self._playback_queue: queue.Queue[AudioChunk | object] = queue.Queue()

        self._worker_thread: threading.Thread | None = None

        self._shutdown = threading.Event()
        self._playing = threading.Event()

    @property
    def is_playing(self) -> bool:
        """True while audio is actively being played back."""
        return self._playing.is_set()

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
        pass

    def on_audio_chunk(
        self,
        audio: AudioBuffer,
        sample_rate: int,
    ) -> None:
        self._playback_queue.put_nowait((audio, sample_rate))

    def on_synthesis_completed(self) -> None:
        pass

    def _playback_worker(self) -> None:
        while True:
            item = self._playback_queue.get()

            if item is _STOP:
                break

            audio, sample_rate = item

            logger.info(
                "Playing audio (%d samples @ %d Hz)",
                len(audio),
                sample_rate,
            )

            try:
                stereo_audio = np.column_stack((audio, audio))

                logger.info("Calling blocking play")
                logger.info(
                    "audio.shape=%s dtype=%s",
                    audio.shape,
                    audio.dtype,
                )

                self._playing.set()
                try:
                    with sd.OutputStream(
                        samplerate=sample_rate,
                        channels=2,
                        dtype="float32",
                    ) as stream:
                        stream.write(stereo_audio)
                finally:
                    self._playing.clear()

                logger.info("Returned from blocking play")

                logger.info("Playback finished.")

            except Exception:
                logger.exception("Audio playback failed.")