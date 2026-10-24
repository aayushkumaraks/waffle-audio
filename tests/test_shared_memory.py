"""Unit tests for services/shared_memory.py zero-copy audio ring buffer."""
from __future__ import annotations

import unittest
import numpy as np

from services.shared_memory import SharedAudioBuffer


class TestSharedAudioBuffer(unittest.TestCase):
    def test_shared_memory_create_write_read(self) -> None:
        shm_name = "test_waffle_shm_unit"
        writer = SharedAudioBuffer(name=shm_name, capacity_samples=1000, create=True)
        try:
            reader = SharedAudioBuffer(name=shm_name, capacity_samples=1000, create=False)

            # Write 100 samples
            audio = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
            writer.write(audio, sample_rate=16000)

            read_audio, sample_rate = reader.read_latest(100)
            self.assertEqual(sample_rate, 16000)
            self.assertEqual(len(read_audio), 100)
            np.testing.assert_allclose(read_audio, audio, atol=1e-5)

            reader.close()
        finally:
            writer.close()

    def test_shared_memory_wraparound(self) -> None:
        shm_name = "test_waffle_shm_wrap"
        writer = SharedAudioBuffer(name=shm_name, capacity_samples=200, create=True)
        try:
            # First write 150 samples
            part1 = np.ones(150, dtype=np.float32) * 1.0
            writer.write(part1, sample_rate=16000)

            # Next write 100 samples (triggers wraparound past capacity 200)
            part2 = np.ones(100, dtype=np.float32) * 2.0
            writer.write(part2, sample_rate=16000)

            # Read latest 120 samples: 20 samples from part1 (value 1.0), 100 samples from part2 (value 2.0)
            out, rate = writer.read_latest(120)
            self.assertEqual(rate, 16000)
            self.assertEqual(len(out), 120)
            np.testing.assert_allclose(out[:20], 1.0, atol=1e-5)
            np.testing.assert_allclose(out[20:], 2.0, atol=1e-5)
        finally:
            writer.close()

    def test_shared_memory_empty_and_oversized_write(self) -> None:
        shm_name = "test_waffle_shm_edge"
        buf = SharedAudioBuffer(name=shm_name, capacity_samples=100, create=True)
        try:
            # Empty write should be a no-op
            buf.write(np.zeros(0, dtype=np.float32))

            # Oversized write (more than capacity) should be ignored
            oversized = np.ones(200, dtype=np.float32)
            buf.write(oversized)

            out, _ = buf.read_latest(50)
            self.assertEqual(len(out), 50)
            np.testing.assert_allclose(out, 0.0)
        finally:
            buf.close()


if __name__ == "__main__":
    unittest.main()
