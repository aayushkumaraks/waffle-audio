import unittest

import numpy as np

from services.audio_processing.src.vad_service import VADConfig, VADService


class FakeVADBackend:
    def __init__(self, events):
        self.events = iter(events)
        self.reset_count = 0

    def process(self, audio):
        return next(self.events, None)

    def reset(self):
        self.reset_count += 1


class VADServiceTests(unittest.TestCase):
    def _service(self, events):
        config = VADConfig(
            pre_speech_padding_ms=100,
            frame_samples=512,
        )
        service = VADService(config, backend=FakeVADBackend(events))
        service.start()
        return service

    def test_idle_audio_is_not_forwarded(self):
        service = self._service([None, None, None])
        audio = np.ones(512, dtype=np.float32)

        output = service.process(audio, 16_000)

        self.assertEqual(output.size, 0)
        self.assertFalse(service.is_speech)

    def test_speech_start_includes_pre_roll(self):
        service = self._service([None, {"start": 512}])

        first = np.full(512, 1.0, dtype=np.float32)
        second = np.full(512, 2.0, dtype=np.float32)

        self.assertEqual(service.process(first, 16_000).size, 0)

        output = service.process(second, 16_000)

        np.testing.assert_array_equal(
            output,
            np.concatenate([first, second]),
        )
        self.assertTrue(service.is_speech)

    def test_speech_is_forwarded_until_end(self):
        service = self._service(
            [{"start": 0}, None, {"end": 1024}, None]
        )

        first = np.full(512, 1.0, dtype=np.float32)
        second = np.full(512, 2.0, dtype=np.float32)
        third = np.full(512, 3.0, dtype=np.float32)
        fourth = np.full(512, 4.0, dtype=np.float32)

        np.testing.assert_array_equal(
            service.process(first, 16_000),
            first,
        )
        np.testing.assert_array_equal(
            service.process(second, 16_000),
            second,
        )
        np.testing.assert_array_equal(
            service.process(third, 16_000),
            third,
        )

        self.assertFalse(service.is_speech)
        self.assertEqual(service.process(fourth, 16_000).size, 0)

    def test_arbitrary_input_chunk_sizes_are_supported(self):
        service = self._service([None, {"start": 512}])

        first = np.ones(300, dtype=np.float32)
        second = np.full(724, 2.0, dtype=np.float32)

        self.assertEqual(service.process(first, 16_000).size, 0)

        output = service.process(second, 16_000)

        self.assertEqual(output.size, 1024)
        np.testing.assert_array_equal(
            output[:300],
            first,
        )

    def test_wrong_sample_rate_is_rejected(self):
        service = self._service([])
        with self.assertRaises(ValueError):
            service.process(np.zeros(512, dtype=np.float32), 48_000)

    def test_reset_clears_stream_state(self):
        backend = FakeVADBackend([{"start": 0}])
        service = VADService(backend=backend)
        service.start()

        service.process(np.ones(512, dtype=np.float32), 16_000)
        self.assertTrue(service.is_speech)

        service.reset()

        self.assertFalse(service.is_speech)
        self.assertGreaterEqual(backend.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
