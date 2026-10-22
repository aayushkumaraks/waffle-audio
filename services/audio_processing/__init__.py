"""Audio preprocessing services: VAD, Acoustic Echo Cancellation, and Barge-In."""
from .src.barge_in import BargeInController
from .src.echo_cancellation import AcousticEchoCanceller, AECConfig
from .src.resampler import resample_audio
from .src.vad_service import (
    SileroVADBackend,
    VADBackend,
    VADConfig,
    VADListener,
    VADService,
)

__all__ = [
    "AECConfig",
    "AcousticEchoCanceller",
    "BargeInController",
    "SileroVADBackend",
    "VADBackend",
    "VADConfig",
    "VADListener",
    "VADService",
    "resample_audio",
]