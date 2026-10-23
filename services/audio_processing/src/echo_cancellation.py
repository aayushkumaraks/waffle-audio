"""Acoustic Echo Cancellation (AEC) for speaker-microphone loopback isolation.

Subtracts speaker playback output (reference audio) from microphone input
to prevent the voice assistant from hearing its own voice when playing over
loudspeakers without headphones.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .resampler import resample_audio

logger = logging.getLogger(__name__)

AudioBuffer = npt.NDArray[np.float32]


class EchoCanceller(Protocol):
    """Protocol for acoustic echo cancellation."""

    def feed_reference(self, reference_audio: AudioBuffer, sample_rate: int) -> None:
        """Feed speaker playback audio as reference signal."""
        ...

    def process(self, microphone_audio: AudioBuffer, sample_rate: int) -> AudioBuffer:
        """Subtract reference echo from microphone input."""
        ...

    def reset(self) -> None:
        """Reset internal filter state."""
        ...


@dataclass(slots=True)
class AECConfig:
    enabled: bool = True
    filter_length: int = 256
    step_size: float = 0.05
    leakage: float = 0.999
    sample_rate: int = 16_000


class AcousticEchoCanceller:
    """Normalized Least Mean Squares (NLMS) adaptive acoustic echo canceller."""

    def __init__(self, config: AECConfig | None = None) -> None:
        self._config = config or AECConfig()
        self._filter_len = self._config.filter_length
        self._mu = self._config.step_size
        self._weights = np.zeros(self._filter_len, dtype=np.float32)
        self._ref_buffer = np.empty(0, dtype=np.float32)
        self._ref_history = deque(maxlen=self._filter_len * 4)
        self._lock = threading.Lock()

    def feed_reference(self, reference_audio: AudioBuffer, sample_rate: int) -> None:
        """Buffer speaker playback audio for echo cancellation."""
        if not self._config.enabled or reference_audio.size == 0:
            return

        ref_16k = resample_audio(
            reference_audio.astype(np.float32, copy=False).flatten(),
            sample_rate,
            self._config.sample_rate,
        )
        with self._lock:
            self._ref_buffer = np.concatenate((self._ref_buffer, ref_16k))
            # Keep reference buffer bounded to ~1.0 second
            max_samples = self._config.sample_rate
            if self._ref_buffer.size > max_samples:
                self._ref_buffer = self._ref_buffer[-max_samples:]

    def process(self, microphone_audio: AudioBuffer, sample_rate: int) -> AudioBuffer:
        """Cancel acoustic echo from microphone input using NLMS adaptive filtering."""
        if not self._config.enabled or microphone_audio.size == 0:
            return microphone_audio

        mic_16k = resample_audio(
            microphone_audio.astype(np.float32, copy=False).flatten(),
            sample_rate,
            self._config.sample_rate,
        )

        n_samples = len(mic_16k)
        with self._lock:
            buf_len = len(self._ref_buffer)
            if buf_len < n_samples:
                ref_slice = np.zeros(n_samples, dtype=np.float32)
                if buf_len > 0:
                    ref_slice[n_samples - buf_len:] = self._ref_buffer
                    self._ref_buffer = np.empty(0, dtype=np.float32)
            else:
                ref_slice = self._ref_buffer[:n_samples].copy()
                self._ref_buffer = self._ref_buffer[n_samples:]

        if np.max(np.abs(ref_slice)) < 1e-5:
            with self._lock:
                self._weights *= (self._config.leakage ** n_samples)
                self._ref_history.clear()
            return resample_audio(mic_16k, self._config.sample_rate, sample_rate)

        # Run adaptive NLMS filter
        cleaned = np.zeros(n_samples, dtype=np.float32)
        eps = 1e-6

        # Extend reference history for convolution window
        history_list = list(self._ref_history)
        full_ref = np.concatenate(
            (np.array(history_list, dtype=np.float32), ref_slice)
        ) if history_list else ref_slice

        h_len = self._filter_len
        for i in range(n_samples):
            idx = len(history_list) + i
            if idx >= h_len:
                x_vec = full_ref[idx - h_len : idx][::-1]
            else:
                x_vec = np.zeros(h_len, dtype=np.float32)
                avail = full_ref[:idx][::-1]
                x_vec[: len(avail)] = avail

            # Estimated echo
            d_hat = float(np.dot(self._weights, x_vec))
            # Error = microphone input - estimated echo
            err = mic_16k[i] - d_hat
            cleaned[i] = err

            # Normalized LMS weight update
            norm = float(np.dot(x_vec, x_vec)) + eps
            self._weights = (
                self._config.leakage * self._weights + (self._mu * err / norm) * x_vec
            )

        # Update reference history
        for s in ref_slice:
            self._ref_history.append(float(s))

        # Resample back to original sample rate if needed
        return resample_audio(cleaned, self._config.sample_rate, sample_rate)

    def reset(self) -> None:
        """Reset internal filter weights and reference buffer."""
        with self._lock:
            self._weights.fill(0.0)
            self._ref_buffer = np.empty(0, dtype=np.float32)
            self._ref_history.clear()


class PassthroughEchoCanceller:
    """Pass-through echo canceller when AEC is disabled or unneeded."""

    def feed_reference(self, reference_audio: AudioBuffer, sample_rate: int) -> None:
        pass

    def process(self, microphone_audio: AudioBuffer, sample_rate: int) -> AudioBuffer:
        return microphone_audio

    def reset(self) -> None:
        pass
