"""Queue and listener plumbing shared by TTS backends."""
from __future__ import annotations
import logging, queue, threading, time
from abc import ABC, abstractmethod
from collections.abc import Iterator
import numpy as np
from constants import is_benchmark_enabled
from .tts_types import AudioBuffer, ServiceAlreadyStarted, ServiceNotStarted, SpeechQueueFull, TTSConfig, TTSListener
logger = logging.getLogger(__name__)
_STOP = object()
class BaseTTSService(ABC):
    def __init__(self, config: TTSConfig) -> None:
        self._config, self._listeners = config, []
        self._speech_queue: queue.Queue[str | object] = queue.Queue(config.queue_size)
        self._worker_thread: threading.Thread | None = None
        self._engine_lock, self._started = threading.Lock(), False
        # Number of texts queued or currently being synthesized.
        self._pending, self._pending_lock = 0, threading.Lock()
        # Bumped on interrupt so in-flight synthesis stops emitting audio.
        self._epoch = 0
    @property
    def is_busy(self) -> bool:
        """True while any text is queued or still being synthesized."""
        with self._pending_lock: return self._pending > 0
    def _change_pending(self, delta: int) -> None:
        with self._pending_lock: self._pending = max(0, self._pending + delta)
    def start(self) -> None:
        if self._started:
            raise ServiceAlreadyStarted("TTSService is already started.")
        logger.info("Starting %s engine...", self.__class__.__name__)
        t0 = time.perf_counter()
        self._load_engine()
        self._started = True
        logger.info("%s engine started successfully in %.2fs.", self.__class__.__name__, time.perf_counter() - t0)
        self._worker_thread = threading.Thread(
            target=self._process_speech_queue, name="tts-worker", daemon=True
        )
        self._worker_thread.start()

    def stop(self) -> None:
        if not self._started:
            raise ServiceNotStarted("TTSService is not started.")
        logger.info("Stopping %s engine...", self.__class__.__name__)
        try:
            self._speech_queue.put_nowait(_STOP)
        except queue.Full:
            pass
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        self._worker_thread = None
        self._unload_engine()
        self._started = False
        logger.info("%s engine stopped.", self.__class__.__name__)
    def speak(self, text: str) -> None:
        if not self._started: raise ServiceNotStarted("TTSService is not started.")
        self._change_pending(1)
        try: self._speech_queue.put_nowait(text)
        except queue.Full as exc:
            self._change_pending(-1)
            raise SpeechQueueFull("Speech queue is full.") from exc
    def interrupt(self) -> None:
        """Purge pending text and stop the in-flight synthesis for instant barge-in."""
        self._epoch += 1
        stop_seen = False
        while True:
            try:
                item = self._speech_queue.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                stop_seen = True
                continue
            self._change_pending(-1)
        if stop_seen:
            self._speech_queue.put_nowait(_STOP)

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
            epoch = self._epoch
            t_synth_start = time.perf_counter()
            t_first_chunk: float | None = None
            total_samples = 0
            sample_rate = 24000
            try:
                with self._engine_lock:
                    self._dispatch_synthesis_started()
                    chunks, rate = self._stream_audio(item)
                    first = True
                    for audio in chunks:
                        if self._epoch != epoch:
                            break  # interrupted: drop the rest of this utterance
                        if first:
                            t_first_chunk = time.perf_counter()
                        total_samples += len(audio)
                        sample_rate = rate
                        self._dispatch_audio_chunk(audio, rate, text=item if first else "")
                        first = False
            except Exception:
                logger.exception("Speech synthesis failed.")
            finally:
                if is_benchmark_enabled():
                    t_synth_end = time.perf_counter()
                    synth_dur = t_synth_end - t_synth_start
                    first_chunk_latency = (t_first_chunk - t_synth_start) if t_first_chunk else synth_dur
                    audio_dur = (total_samples / sample_rate) if sample_rate > 0 else 0.0
                    rtf = (synth_dur / audio_dur) if audio_dur > 0 else 0.0
                    logger.info(
                        "[TTS BENCHMARK] Synthesized '%s' in %.3fs (first audio chunk: %.3fs, audio: %.2fs, RTF: %.2fx, %dHz)",
                        (item[:40] + "...") if isinstance(item, str) and len(item) > 40 else item,
                        synth_dur,
                        first_chunk_latency,
                        audio_dur,
                        rtf,
                        sample_rate,
                    )
                # Decrement before signalling completion so listeners that
                # check is_busy inside on_synthesis_completed see the final state.
                self._change_pending(-1)
                self._dispatch_synthesis_completed()

    def _dispatch_synthesis_started(self) -> None:
        for listener in tuple(self._listeners): listener.on_synthesis_started()
    def _dispatch_audio_chunk(self, audio: AudioBuffer, rate: int, text: str = "") -> None:
        for listener in tuple(self._listeners):
            try:
                listener.on_audio_chunk(audio, rate, text=text)
            except TypeError:
                listener.on_audio_chunk(audio, rate)
    def _dispatch_synthesis_completed(self) -> None:
        for listener in tuple(self._listeners): listener.on_synthesis_completed()
    @abstractmethod
    def _load_engine(self) -> None: ...
    @abstractmethod
    def _unload_engine(self) -> None: ...
    @abstractmethod
    def _stream_audio(self, text: str) -> tuple[Iterator[AudioBuffer], int]: ...
