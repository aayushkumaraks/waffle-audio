"""Queue and listener plumbing shared by TTS backends."""
from __future__ import annotations
import logging, queue, threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
import numpy as np
from .tts_types import AudioBuffer, ServiceAlreadyStarted, ServiceNotStarted, SpeechQueueFull, TTSConfig, TTSListener
logger = logging.getLogger(__name__)
_STOP = object()
class BaseTTSService(ABC):
    def __init__(self, config: TTSConfig) -> None:
        self._config, self._listeners = config, []
        self._speech_queue: queue.Queue[str | object] = queue.Queue(config.queue_size)
        self._worker_thread: threading.Thread | None = None
        self._engine_lock, self._started = threading.Lock(), False
    def start(self) -> None:
        if self._started: raise ServiceAlreadyStarted("TTSService is already started.")
        self._load_engine(); self._started = True
        self._worker_thread = threading.Thread(target=self._process_speech_queue, name="tts-worker", daemon=True); self._worker_thread.start()
    def stop(self) -> None:
        if not self._started: raise ServiceNotStarted("TTSService is not started.")
        try: self._speech_queue.put_nowait(_STOP)
        except queue.Full: pass
        if self._worker_thread: self._worker_thread.join(timeout=5)
        self._worker_thread = None; self._unload_engine(); self._started = False
    def speak(self, text: str) -> None:
        if not self._started: raise ServiceNotStarted("TTSService is not started.")
        try: self._speech_queue.put_nowait(text)
        except queue.Full as exc: raise SpeechQueueFull("Speech queue is full.") from exc
    def add_listener(self, listener: TTSListener) -> None:
        if listener not in self._listeners: self._listeners.append(listener)
    def remove_listener(self, listener: TTSListener) -> None:
        if listener in self._listeners: self._listeners.remove(listener)
    def synthesize_direct(self, text: str) -> tuple[AudioBuffer, int]:
        if not self._started: raise ServiceNotStarted("TTSService is not started.")
        with self._engine_lock:
            chunks, rate = self._stream_audio(text)
            chunks = list(chunks)
        return (np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32), rate)
    def _process_speech_queue(self) -> None:
        while True:
            item = self._speech_queue.get()
            if item is _STOP: return
            try:
                with self._engine_lock:
                    self._dispatch_synthesis_started(); chunks, rate = self._stream_audio(item)
                    for audio in chunks: self._dispatch_audio_chunk(audio, rate)
                    self._dispatch_synthesis_completed()
            except Exception: logger.exception("Speech synthesis failed.")
    def _dispatch_synthesis_started(self) -> None:
        for listener in tuple(self._listeners): listener.on_synthesis_started()
    def _dispatch_audio_chunk(self, audio: AudioBuffer, rate: int) -> None:
        for listener in tuple(self._listeners): listener.on_audio_chunk(audio, rate)
    def _dispatch_synthesis_completed(self) -> None:
        for listener in tuple(self._listeners): listener.on_synthesis_completed()
    @abstractmethod
    def _load_engine(self) -> None: ...
    @abstractmethod
    def _unload_engine(self) -> None: ...
    @abstractmethod
    def _stream_audio(self, text: str) -> tuple[Iterator[AudioBuffer], int]: ...
