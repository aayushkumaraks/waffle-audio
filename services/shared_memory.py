"""Zero-copy shared memory audio buffer for multi-process isolation.

Allows transferring raw NumPy float32 PCM audio buffers across Python processes
using multiprocessing.shared_memory without JSON, base64, or TCP socket overhead.
"""

from __future__ import annotations

import logging
from multiprocessing import shared_memory

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

AudioBuffer = npt.NDArray[np.float32]


class SharedAudioBuffer:
    """Zero-copy ring buffer backed by POSIX shared memory."""

    def __init__(
        self,
        name: str = "waffle_audio_shm",
        capacity_samples: int = 16000 * 5,  # 5 seconds @ 16kHz
        create: bool = False,
    ) -> None:
        self.name = name
        self.capacity_samples = capacity_samples
        self.bytes_size = capacity_samples * 4  # float32 = 4 bytes
        self._is_creator = create

        if create:
            # Clean up existing segment with same name if stale
            try:
                existing = shared_memory.SharedMemory(name=name)
                existing.close()
                existing.unlink()
            except FileNotFoundError:
                pass

            self._shm = shared_memory.SharedMemory(
                name=name,
                create=True,
                size=self.bytes_size + 16,  # 16 bytes header for write_idx / sample_rate
            )
            self._header = np.ndarray((4,), dtype=np.int32, buffer=self._shm.buf[:16])
            self._header[0] = 0  # write_idx
            self._header[1] = 16000  # sample_rate
            self._header[2] = 0  # unread_samples
        else:
            self._shm = shared_memory.SharedMemory(name=name)
            self._header = np.ndarray((4,), dtype=np.int32, buffer=self._shm.buf[:16])

        self._audio_data = np.ndarray(
            (capacity_samples,),
            dtype=np.float32,
            buffer=self._shm.buf[16 : 16 + self.bytes_size],
        )

    def write(self, audio: AudioBuffer, sample_rate: int = 16000) -> None:
        """Write raw audio frames directly into shared memory without copying."""
        flat = audio.astype(np.float32, copy=False).flatten()
        n = len(flat)
        if n == 0 or n > self.capacity_samples:
            return

        w_idx = int(self._header[0])
        end_idx = w_idx + n
        if end_idx <= self.capacity_samples:
            self._audio_data[w_idx:end_idx] = flat
        else:
            first = self.capacity_samples - w_idx
            self._audio_data[w_idx:] = flat[:first]
            self._audio_data[: n - first] = flat[first:]

        self._header[0] = end_idx % self.capacity_samples
        self._header[1] = sample_rate
        self._header[2] = min(self.capacity_samples, int(self._header[2]) + n)

    def read_latest(self, n_samples: int) -> tuple[AudioBuffer, int]:
        """Read the latest n_samples directly from shared memory."""
        w_idx = int(self._header[0])
        sample_rate = int(self._header[1])
        n = min(n_samples, self.capacity_samples)

        start_idx = (w_idx - n) % self.capacity_samples
        if start_idx + n <= self.capacity_samples:
            out = self._audio_data[start_idx : start_idx + n].copy()
        else:
            first = self.capacity_samples - start_idx
            out = np.concatenate(
                (self._audio_data[start_idx:], self._audio_data[: n - first])
            )

        return out, sample_rate

    def close(self) -> None:
        """Close shared memory handle."""
        try:
            self._shm.close()
            if self._is_creator:
                self._shm.unlink()
        except Exception:
            pass
