import unittest

import numpy as np

from services.audio_processing.src.pipeline import AudioPipeline, AudioPipelineConfig


class FakeVAD:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.reset_count = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def reset(self):
        self.reset_count += 1

    def process(self, audio, sample_rate):
        return audio


class AudioPipelineTests(unittest.TestCase):
    def test_stage_one_remains_active_when_later_stages_are_disabled(self):
        vad = FakeVAD()
        pipeline = AudioPipeline(vad)

        pipeline.start()
        audio = np.ones(512, dtype=np.float32)

        output = pipeline.process(audio, 16_000)

        np.testing.assert_array_equal(output, audio)
        self.assertTrue(vad.started)

        pipeline.stop()
        self.assertTrue(vad.stopped)

    def test_all_future_stages_default_to_disabled(self):
        config = AudioPipelineConfig()

        self.assertFalse(config.noise_suppression_enabled)
        self.assertFalse(config.echo_cancellation_enabled)
        self.assertFalse(config.target_speaker_enabled)
        self.assertFalse(config.source_separation_enabled)

    def test_reset_delegates_to_vad(self):
        vad = FakeVAD()
        pipeline = AudioPipeline(vad)
        pipeline.start()

        pipeline.reset()

        self.assertEqual(vad.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
