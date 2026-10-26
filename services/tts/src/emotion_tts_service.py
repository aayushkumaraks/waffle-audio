"""Multi-threaded, in-memory emotion TTS service for PocketTTS with idle offloading."""
from __future__ import annotations

import gc
import logging
import os
import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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

# Pin Hugging Face cache
os.environ.setdefault("HF_HOME", str(Path(POCKET_TTS_CACHE_DIR).resolve()))

from pocket_tts import TTSModel
from services.tts.src.hf_cache import load_local_first
from services.tts.src.tts_types import (
    AudioBuffer,
    ServiceAlreadyStarted,
    ServiceNotStarted,
    SpeechQueueFull,
    TTSListener,
    TTSServiceError,
)

logger = logging.getLogger(__name__)

_STOP = object()


class EmotionVoiceWorker:
    """Dedicated worker thread managing an individual emotion voice model in memory."""

    def __init__(
        self,
        emotion: str,
        gender: str,
        model_path: Path,
        engine_getter: Any,
        engine_lock: threading.Lock,
        dispatch_callbacks: dict[str, Any],
        queue_size: int = 16,
    ) -> None:
        self.emotion = emotion
        self.gender = gender
        self.model_path = model_path
        self._get_engine = engine_getter
        self._engine_lock = engine_lock
        self._dispatch = dispatch_callbacks

        self._queue: queue.Queue[tuple[str, str] | object] = queue.Queue(queue_size)
        self._thread: threading.Thread | None = None
        self._voice_state: object | None = None
        self._state_lock = threading.Lock()
        self._started = False

        self.last_used_time: float = time.time()
        self.is_synthesizing: bool = False

    @property
    def is_loaded(self) -> bool:
        with self._state_lock:
            return self._voice_state is not None

    def load(self) -> None:
        """Load the voice state tensor into memory."""
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
        """Offload the voice state from memory to release RAM."""
        with self._state_lock:
            if self._voice_state is None:
                return False

            self._voice_state = None
            gc.collect()
            logger.info("Offloaded [%s_%s] voice model from memory.", self.gender, self.emotion)
            return True

    def start(self) -> None:
        """Start the worker thread."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run,
            name=f"tts-worker-{self.emotion}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop worker thread and release memory."""
        if not self._started:
            return
        self._started = False
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self.offload()

    def submit(self, text: str) -> None:
        """Enqueue speech text for this emotion."""
        if not self._started:
            raise ServiceNotStarted(f"Worker for {self.emotion} is not started.")
        try:
            self._queue.put_nowait((text, self.emotion))
        except queue.Full as exc:
            raise SpeechQueueFull(f"Speech queue for {self.emotion} is full.") from exc

    def _run(self) -> None:
        while self._started:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _STOP:
                break

            text, emotion = item
            try:
                self.is_synthesizing = True
                self.last_used_time = time.time()

                # Ensure loaded in memory (lazy auto-reload on demand if was offloaded)
                if not self.is_loaded:
                    self.load()

                engine = self._get_engine()
                if engine is None:
                    continue

                with self._state_lock:
                    state = self._voice_state

                if state is None:
                    logger.error("Failed to acquire state for emotion %s", emotion)
                    continue

                self._dispatch["started"](emotion)

                # Stream audio chunks under engine lock
                with self._engine_lock:
                    first = True
                    for chunk in engine.generate_audio_stream(state, text):
                        audio = chunk.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
                        self._dispatch["chunk"](audio, engine.sample_rate, text=text if first else "", emotion=emotion)
                        first = False

                self._dispatch["completed"](emotion)
            except Exception:
                logger.exception("Synthesis failed in worker %s", self.emotion)
                self._dispatch["completed"](emotion)
            finally:
                self.is_synthesizing = False
                self.last_used_time = time.time()


class EmotionTTSService:
    """Coordinates multi-emotion PocketTTS workers with thread-per-emotion and idle offloading."""

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

        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()

        # Create workers for all available emotions
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

    def add_listener(self, listener: TTSListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: TTSListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def start(self) -> None:
        """Start the TTS engine and all emotion voice workers."""
        if self._started:
            raise ServiceAlreadyStarted("EmotionTTSService is already started.")

        logger.info("Initializing PocketTTS engine for EmotionTTSService (gender=%s)...", self.gender)

        def _loader() -> TTSModel:
            return TTSModel.load_model(language=self.language, quantize=self.quantize)

        self._engine = load_local_first(_loader)

        # Start each worker thread and load their states in parallel threads
        load_threads: list[threading.Thread] = []
        for emotion, worker in self._workers.items():
            worker.start()
            # Parallel loading thread for fast startup
            t = threading.Thread(
                target=worker.load,
                name=f"loader-{emotion}",
                daemon=True,
            )
            load_threads.append(t)
            t.start()

        for t in load_threads:
            t.join(timeout=10.0)

        self._started = True

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
        """Stop all workers and unload engine."""
        if not self._started:
            return

        self._started = False
        self._watchdog_stop.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=2.0)
            self._watchdog_thread = None

        for worker in self._workers.values():
            worker.stop()

        with self._engine_lock:
            self._engine = None

        gc.collect()
        logger.info("EmotionTTSService stopped.")

    def speak(self, text: str, emotion: str = "neutral") -> None:
        """Speak text with the requested emotion voice model."""
        if not self._started:
            raise ServiceNotStarted("EmotionTTSService is not started.")

        emotion_key = (emotion or "neutral").strip().lower()
        if emotion_key not in self._workers:
            logger.warning("Unknown emotion '%s', falling back to 'neutral'.", emotion)
            emotion_key = "neutral"

        worker = self._workers[emotion_key]
        worker.submit(text)

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

    def _dispatch_audio_chunk(self, audio: AudioBuffer, rate: int, text: str = "", emotion: str = "neutral") -> None:
        for listener in tuple(self._listeners):
            try:
                listener.on_audio_chunk(audio, rate, text=text)
            except TypeError:
                listener.on_audio_chunk(audio, rate)

    def _dispatch_synthesis_completed(self, emotion: str) -> None:
        for listener in tuple(self._listeners):
            listener.on_synthesis_completed()
