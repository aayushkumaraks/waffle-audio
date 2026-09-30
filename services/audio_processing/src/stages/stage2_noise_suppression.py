from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

AudioBuffer = npt.NDArray[np.float32]


class NoiseSuppressor(Protocol):
    """Contract for speech-preserving background-noise suppression."""

    def process(self, audio: AudioBuffer, sample_rate: int) -> AudioBuffer:
        """Return an enhanced audio signal at the same sample rate."""
        ...


@dataclass(slots=True)
class NoiseSuppressionConfig:
    enabled: bool = False


class PassthroughNoiseSuppressor:
    """Stage 2 placeholder that preserves the current audio signal."""

    def process(self, audio: AudioBuffer, sample_rate: int) -> AudioBuffer:
        return audio
