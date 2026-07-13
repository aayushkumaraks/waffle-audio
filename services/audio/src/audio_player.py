from __future__ import annotations
import queue
import threading
from typing import TypeAlias
from services.tts.src.tts_service import (
    AudioBuffer,
    TTSListener,
)
import sounddevice as sd

_STOP = object()

AudioChunk: TypeAlias = tuple[AudioBuffer, int]

class AudioPlayer(TTSListener):
    
    def on_synthesis_started(self) -> None:
        pass

    def on_audio_chunk(
        self,
        audio: AudioBuffer,
        sample_rate: int,
    ) -> None:
        self._playback_queue.put_nowait(
            (audio, sample_rate)
        )

    def on_synthesis_completed(self) -> None:
        pass

    def __init__(self) -> None:
        self._playback_queue: queue.Queue[AudioChunk | object] = queue.Queue()

        self._worker_thread: threading.Thread | None = None

        self._shutdown = threading.Event()

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

    def _playback_worker(self) -> None:
        while True:

            item = self._playback_queue.get()

            if item is _STOP:
                break

            audio, sample_rate = item

            sd.play(audio, sample_rate)
            sd.wait()