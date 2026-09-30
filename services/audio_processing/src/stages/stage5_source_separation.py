from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

AudioBuffer = npt.NDArray[np.float32]


class TargetSpeakerExtractor(Protocol):
    """Contract for extracting the target speaker from mixed audio."""

    def process(
        self,
        audio: AudioBuffer,
        sample_rate: int,
    ) -> AudioBuffer:
        """Return a target-speaker-focused signal."""
        ...


class PassthroughTargetSpeakerExtractor:
    """Stage 5 placeholder; no source separation is performed yet."""

    def process(
        self,
        audio: AudioBuffer,
        sample_rate: int,
    ) -> AudioBuffer:
        return audio
