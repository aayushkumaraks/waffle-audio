"""Shared public contracts for all text-to-speech backends."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias
import numpy as np
import numpy.typing as npt
from constants import KOKORO_MODEL_PATH, KOKORO_VOICES_PATH, TTS_QUANTIZE
AudioBuffer: TypeAlias = npt.NDArray[np.float32]
@dataclass(slots=True)
class TTSConfig:
    backend: str | None = None
    voice: str | None = None
    language: str | None = None
    speed: float = 1.0
    quantize: bool = TTS_QUANTIZE
    model_path: str = KOKORO_MODEL_PATH
    voices_path: str = KOKORO_VOICES_PATH
    queue_size: int = 1
class TTSListener(ABC):
    @abstractmethod
    def on_synthesis_started(self) -> None: ...
    @abstractmethod
    def on_audio_chunk(self, audio: AudioBuffer, sample_rate: int) -> None: ...
    @abstractmethod
    def on_synthesis_completed(self) -> None: ...
class TTSServiceError(Exception): pass
class ServiceAlreadyStarted(TTSServiceError): pass
class ServiceNotStarted(TTSServiceError): pass
class SpeechQueueFull(TTSServiceError): pass
