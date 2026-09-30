from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

AudioBuffer = npt.NDArray[np.float32]


class EchoCanceller(Protocol):
    """Contract for acoustic echo cancellation."""

    def process(
        self,
        microphone_audio: AudioBuffer,
        reference_audio: AudioBuffer,
        sample_rate: int,
    ) -> AudioBuffer:
        """Remove reference/playback echo from microphone audio."""
        ...


class PassthroughEchoCanceller:
    """Stage 3 placeholder; no echo cancellation is applied yet."""

    def process(
        self,
        microphone_audio: AudioBuffer,
        reference_audio: AudioBuffer,
        sample_rate: int,
    ) -> AudioBuffer:
        return microphone_audio
