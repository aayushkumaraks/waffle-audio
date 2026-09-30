from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

AudioBuffer = npt.NDArray[np.float32]


class NoiseSuppressor(Protocol):
    """Contract for speech-preserving background-noise suppression."""

    def process(self, audio: AudioBuffer, sample_rate: int) -> AudioBuffer:
        """Return an enhanced audio signal at the same sample rate."""
        ...


class PassthroughNoiseSuppressor:
    """ boundary; preserves audio until a real suppressor is selected."""

    def process(self, audio: AudioBuffer, sample_rate: int) -> AudioBuffer:
        return audio
