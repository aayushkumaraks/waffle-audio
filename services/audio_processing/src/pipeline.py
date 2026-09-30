from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .stages.stage2_noise_suppression import (
    NoiseSuppressor,
    PassthroughNoiseSuppressor,
)
from .stages.stage3_echo_cancellation import (
    EchoCanceller,
    PassthroughEchoCanceller,
)
from .stages.stage4_speaker_verification import (
    PassthroughTargetSpeakerVerifier,
    TargetSpeakerVerifier,
)
from .stages.stage5_source_separation import (
    PassthroughTargetSpeakerExtractor,
    TargetSpeakerExtractor,
)
from .vad_service import VADService

AudioBuffer = npt.NDArray[np.float32]


@dataclass(slots=True)
class AudioPipelineConfig:
    """Feature switches for the staged audio pipeline."""

    noise_suppression_enabled: bool = False
    echo_cancellation_enabled: bool = False
    target_speaker_enabled: bool = False
    source_separation_enabled: bool = False


class AudioPipeline:
    """Apply audio stages in a fixed order before STT.

    Stage 1 VAD is currently active. Later stages are explicit extension
    points and remain pass-through until their implementations are enabled.
    """

    def __init__(
        self,
        vad: VADService,
        *,
        config: AudioPipelineConfig | None = None,
        noise_suppressor: NoiseSuppressor | None = None,
        echo_canceller: EchoCanceller | None = None,
        target_speaker_verifier: TargetSpeakerVerifier | None = None,
        target_speaker_extractor: TargetSpeakerExtractor | None = None,
    ) -> None:
        self.config = config or AudioPipelineConfig()
        self.vad = vad
        self.noise_suppressor = noise_suppressor or PassthroughNoiseSuppressor()
        self.echo_canceller = echo_canceller or PassthroughEchoCanceller()
        self.target_speaker_verifier = (
            target_speaker_verifier or PassthroughTargetSpeakerVerifier()
        )
        self.target_speaker_extractor = (
            target_speaker_extractor or PassthroughTargetSpeakerExtractor()
        )

    def start(self) -> None:
        self.vad.start()

    def stop(self) -> None:
        self.vad.stop()

    def reset(self) -> None:
        self.vad.reset()

    def process(
        self,
        microphone_audio: AudioBuffer,
        sample_rate: int,
        *,
        reference_audio: AudioBuffer | None = None,
    ) -> AudioBuffer:
        audio = np.asarray(microphone_audio, dtype=np.float32).reshape(-1)

        if self.config.echo_cancellation_enabled and reference_audio is not None:
            audio = self.echo_canceller.process(
                audio,
                np.asarray(reference_audio, dtype=np.float32).reshape(-1),
                sample_rate,
            )

        if self.config.noise_suppression_enabled:
            audio = self.noise_suppressor.process(audio, sample_rate)

        if self.config.source_separation_enabled:
            audio = self.target_speaker_extractor.process(audio, sample_rate)

        audio = self.vad.process(audio, sample_rate)

        if (
            self.config.target_speaker_enabled
            and audio.size
            and not self.target_speaker_verifier.is_target_speaker(audio, sample_rate)
        ):
            return np.empty(0, dtype=np.float32)

        return audio
