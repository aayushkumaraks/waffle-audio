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
        stream: sd.OutputStream | None = None
        stream_rate: int | None = None
        try:
            while True:
                item = self._playback_queue.get()

                if item is _STOP:
                    break

                audio, sample_rate = item
                try:
                    # Keep one stream open for adjacent TTS chunks. Reopening the
                    # device for every generated chunk causes timing gaps and lets
                    # microphone input leak back into the conversation loop.
                    if stream is None or stream_rate != sample_rate:
                        if stream is not None:
                            stream.close()
                        stream = sd.OutputStream(
                            samplerate=sample_rate,
                            channels=2,
                            dtype="float32",
                        )
                        stream.start()
                        stream_rate = sample_rate

                    self._playing.set()
                    stream.write(np.column_stack((audio, audio)))
                except Exception:
                    logger.exception("Audio playback failed.")
                finally:
                    # The queue is intentionally serialized, so this clears only
                    # after the current chunk has reached the output device.
                    self._playing.clear()
        finally:
            if stream is not None:
                stream.close()
