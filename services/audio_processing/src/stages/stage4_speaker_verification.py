from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

AudioBuffer = npt.NDArray[np.float32]


class TargetSpeakerVerifier(Protocol):
    """Contract for deciding whether speech belongs to the target speaker."""

    def is_target_speaker(
        self,
        audio: AudioBuffer,
        sample_rate: int,
    ) -> bool:
        """Return True when the audio matches the enrolled speaker."""
        ...


@dataclass(slots=True)
class SpeakerVerificationConfig:
    enabled: bool = False
    similarity_threshold: float = 0.70


class PassthroughTargetSpeakerVerifier:
    """Stage 4 placeholder; accepts all speech until verification is enabled."""

    def is_target_speaker(
        self,
        audio: AudioBuffer,
        sample_rate: int,
    ) -> bool:
        return True
