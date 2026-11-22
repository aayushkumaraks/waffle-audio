"""Emotion-aware PocketTTS service with FIFO ordering, voice-state caching, and idle offloading."""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import numpy as np

from constants import (
    AVAILABLE_EMOTIONS,
    AVAILABLE_GENDERS,
    CLONED_VOICES_DIR,
    POCKET_TTS_CACHE_DIR,
    POCKET_TTS_DEFAULT_LANGUAGE,
    TTS_QUANTIZE,
    get_emotion_voice_path,
)
from services.tts.src.tts_types import (
    AudioBuffer,
    ServiceAlreadyStarted,
    ServiceNotStarted,
    SpeechQueueFull,
    TTSListener,
    TTSServiceError,
)

# Pin Hugging Face cache before pocket_tts / huggingface_hub are lazily imported
os.environ.setdefault("HF_HOME", str(Path(POCKET_TTS_CACHE_DIR).resolve()))

if TYPE_CHECKING:
    from pocket_tts import TTSModel

logger = logging.getLogger(__name__)

_STOP = object()
_T = TypeVar("_T")


def load_local_first(loader: Callable[[], _T]) -> _T:
    """Lazy wrapper around hf_cache.load_local_first so huggingface_hub is not imported eagerly."""
    from services.tts.src.hf_cache import load_local_first as _load_local_first

    return _load_local_first(loader)


class EmotionVoiceWorker:
    """Manages an individual emotion voice state in memory (load, cache, idle offload)."""

    def __init__(
        self,
        emotion: str,
        gender: str,
        model_path: Path,
        engine_getter: Any,
        engine_lock: threading.Lock,
        dispatch_callbacks: dict[str, Any] | None = None,
        queue_size: int = 16,
    ) -> None:
        self.emotion = emotion
        self.gender = gender
        self.model_path = model_path
        self._get_engine = engine_getter
        self._engine_lock = engine_lock
        self._dispatch = dispatch_callbacks or {}

        self._voice_state: object | None = None
        self._state_lock = threading.Lock()

        self.last_used_time: float = time.time()
        self.is_synthesizing: bool = False

    @property
    def is_loaded(self) -> bool:
        with self._state_lock:
            return self._voice_state is not None

    def get_state(self) -> object | None:
        with self._state_lock:
            return self._voice_state

    def load(self) -> None:
        """Load the voice state tensor into memory under the engine lock."""
        with self._state_lock:
            if self._voice_state is not None:
                return

            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Emotion voice model not found: {self.model_path}. "
                    "Run 'python generate_emotion_voices.py' to generate."
                )

            t0 = time.perf_counter()
            engine = self._get_engine()
            if engine is None:
                raise RuntimeError("Cannot load voice state before TTS engine is initialized.")

            with self._engine_lock:
                state = engine.get_state_for_audio_prompt(str(self.model_path.resolve()))
            self._voice_state = state
            self.last_used_time = time.time()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "Loaded [%s_%s] voice model into memory in %.2f ms (%s)",
                self.gender,
                self.emotion,
                elapsed_ms,
                self.model_path.name,
            )

    def offload(self) -> bool:
        """Offload the voice state from memory to release RAM without blocking on GC."""
        with self._state_lock:
            if self._voice_state is None:
                return False

            self._voice_state = None
            logger.info("Offloaded [%s_%s] voice model from memory.", self.gender, self.emotion)
            return True

    def stop(self) -> None:
        """Release cached voice state memory."""
        self.offload()


class EmotionTTSService:
    """Coordinates multi-emotion PocketTTS synthesis with strict FIFO ordering and idle offloading."""

    def __init__(
        self,
        gender: str = "female",
        language: str = POCKET_TTS_DEFAULT_LANGUAGE,
        quantize: bool = TTS_QUANTIZE,
        offload_idle_seconds: float = 120.0,
        models_dir: Path | str | None = None,
        queue_size: int = 16,
    ) -> None:
        gender_clean = gender.strip().lower()
        if gender_clean not in AVAILABLE_GENDERS:
            raise ValueError(f"Invalid gender '{gender}'. Must be one of {AVAILABLE_GENDERS}.")

        self.gender = gender_clean
        self.language = language
        self.quantize = quantize
        self.offload_idle_seconds = offload_idle_seconds
        self.models_dir = Path(models_dir) if models_dir else CLONED_VOICES_DIR

        self._engine: TTSModel | None = None
        self._engine_lock = threading.Lock()
        self._started = False
        self._listeners: list[TTSListener] = []

        # Single FIFO synthesis queue preserves sentence ordering across emotion switches
        self._speech_queue: queue.Queue[tuple[str, str] | object] = queue.Queue(queue_size)
        self._worker_thread: threading.Thread | None = None
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._epoch = 0

        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()

        dispatch_cbs = {
            "started": self._dispatch_synthesis_started,
            "chunk": self._dispatch_audio_chunk,
            "completed": self._dispatch_synthesis_completed,
        }

        self._workers: dict[str, EmotionVoiceWorker] = {}
        for emotion in AVAILABLE_EMOTIONS:
            model_path = get_emotion_voice_path(self.gender, emotion, self.models_dir)
            worker = EmotionVoiceWorker(
                emotion=emotion,
                gender=self.gender,
                model_path=model_path,
                engine_getter=lambda: self._engine,
                engine_lock=self._engine_lock,
                dispatch_callbacks=dispatch_cbs,
                queue_size=queue_size,
            )
            self._workers[emotion] = worker

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_busy(self) -> bool:
        """True while any text is queued or currently being synthesized."""
        with self._pending_lock:
            return self._pending > 0

    def _change_pending(self, delta: int) -> None:
        with self._pending_lock:
            self._pending = max(0, self._pending + delta)

    def add_listener(self, listener: TTSListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: TTSListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def start(self) -> None:
        """Start the TTS engine, preload all emotion voice states, and launch FIFO synthesis worker."""
        if self._started:
            raise ServiceAlreadyStarted("EmotionTTSService is already started.")

        logger.info("Initializing PocketTTS engine for EmotionTTSService (gender=%s)...", self.gender)

        def _loader() -> TTSModel:
            from pocket_tts import TTSModel

            return TTSModel.load_model(language=self.language, quantize=self.quantize)

        self._engine = load_local_first(_loader)

        try:
            for worker in self._workers.values():
                worker.load()
        except Exception as exc:
            for worker in self._workers.values():
                worker.offload()
            with self._engine_lock:
                self._engine = None
            raise TTSServiceError(
                f"Failed to load emotion voice models for gender '{self.gender}': {exc}"
            ) from exc

        self._started = True
        self._worker_thread = threading.Thread(
            target=self._process_speech_queue,
            name="tts-emotion-worker",
            daemon=True,
        )
        self._worker_thread.start()

        # Start background watchdog for idle offloading if enabled
        if self.offload_idle_seconds > 0:
            self._watchdog_stop.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name="tts-offload-watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()

        logger.info("EmotionTTSService started successfully for gender '%s'.", self.gender)

    def stop(self) -> None:
        """Stop synthesis worker, watchdog, and unload engine & voice states."""
        if not self._started:
            return

        self._started = False
        self._epoch += 1
        self._watchdog_stop.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=2.0)
            self._watchdog_thread = None

        try:
            self._speech_queue.put_nowait(_STOP)
        except queue.Full:
            self.interrupt()
            try:
                self._speech_queue.put_nowait(_STOP)
            except queue.Full:
                pass

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        self._worker_thread = None

        for worker in self._workers.values():
            worker.stop()

        with self._engine_lock:
            self._engine = None

        with self._pending_lock:
            self._pending = 0

        logger.info("EmotionTTSService stopped.")

    def speak(self, text: str, emotion: str = "neutral") -> None:
        """Speak text with the requested emotion voice model in strict FIFO order."""
        if not self._started:
            raise ServiceNotStarted("EmotionTTSService is not started.")

        emotion_key = (emotion or "neutral").strip().lower()
        if emotion_key not in self._workers:
            logger.warning("Unknown emotion '%s', falling back to 'neutral'.", emotion)
            emotion_key = "neutral"

        self._change_pending(1)
        try:
            self._speech_queue.put_nowait((text, emotion_key))
        except queue.Full as exc:
            self._change_pending(-1)
            raise SpeechQueueFull(f"Speech queue is full (emotion={emotion_key}).") from exc

    def interrupt(self) -> None:
        """Purge pending text and stop any in-flight synthesis for instant barge-in."""
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

    def _process_speech_queue(self) -> None:
        while True:
            item = self._speech_queue.get()
            if item is _STOP:
                return

            text, emotion = item
            worker = self._workers[emotion]
            epoch = self._epoch

            try:
                if self._epoch != epoch:
                    continue

                worker.is_synthesizing = True
                worker.last_used_time = time.time()

                # Lazy auto-reload if this emotion was offloaded while idle
                if not worker.is_loaded:
                    worker.load()

                engine = self._engine
                state = worker.get_state()
                if engine is None or state is None:
                    logger.error("Failed to acquire engine or state for emotion %s", emotion)
                    continue

                with self._engine_lock:
                    if self._epoch != epoch:
                        continue
                    self._dispatch_synthesis_started(emotion)
                    first = True
                    for chunk in engine.generate_audio_stream(state, text):
                        if self._epoch != epoch:
                            break
                        audio = chunk.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
                        self._dispatch_audio_chunk(
                            audio, engine.sample_rate, text=text if first else "", emotion=emotion
                        )
                        first = False
            except Exception:
                logger.exception("Synthesis failed for emotion '%s'", emotion)
            finally:
                worker.is_synthesizing = False
                worker.last_used_time = time.time()
                # Decrement pending before signalling completion so AudioPlayer.is_playing
                # sees the accurate upstream busy state.
                self._change_pending(-1)
                self._dispatch_synthesis_completed(emotion)

    def offload_emotion(self, emotion: str) -> bool:
        """Explicitly offload a specific emotion model from memory."""
        worker = self._workers.get(emotion.strip().lower())
        if worker:
            return worker.offload()
        return False

    def offload_idle(self) -> list[str]:
        """Offload any non-synthesizing models that have exceeded idle timeout."""
        if self.offload_idle_seconds <= 0:
            return []

        now = time.time()
        offloaded: list[str] = []
        for emotion, worker in self._workers.items():
            # Keep neutral pinned resident for instant response; offload happy/sad/angry when idle
            if emotion == "neutral":
                continue

            if worker.is_loaded and not worker.is_synthesizing:
                if now - worker.last_used_time >= self.offload_idle_seconds:
                    if worker.offload():
                        offloaded.append(emotion)
        return offloaded

    def get_statuses(self) -> dict[str, dict[str, Any]]:
        """Return status dictionary for UI monitoring."""
        now = time.time()
        return {
            emotion: {
                "loaded": worker.is_loaded,
                "idle_seconds": round(now - worker.last_used_time, 1),
                "synthesizing": worker.is_synthesizing,
                "model_file": worker.model_path.name,
            }
            for emotion, worker in self._workers.items()
        }

    def _watchdog_loop(self) -> None:
        """Background loop to periodically check and offload idle models."""
        while not self._watchdog_stop.wait(5.0):
            try:
                self.offload_idle()
            except Exception:
                logger.debug("Error during idle offloading check", exc_info=True)

    def _dispatch_synthesis_started(self, emotion: str) -> None:
        for listener in tuple(self._listeners):
            listener.on_synthesis_started()

    def _dispatch_audio_chunk(
        self, audio: AudioBuffer, rate: int, text: str = "", emotion: str = "neutral"
    ) -> None:
        for listener in tuple(self._listeners):
            try:
                listener.on_audio_chunk(audio, rate, text=text, emotion=emotion)
            except TypeError:
                try:
                    listener.on_audio_chunk(audio, rate, text=text)
                except TypeError:
                    listener.on_audio_chunk(audio, rate)

    def _dispatch_synthesis_completed(self, emotion: str) -> None:
        for listener in tuple(self._listeners):
            listener.on_synthesis_completed()
