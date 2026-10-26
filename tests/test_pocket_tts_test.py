"""Unit tests for pocket_tts_test.py standalone smoke test script."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

import pocket_tts_test


class TestPocketTTSTestScript(unittest.TestCase):
    def test_main_execution(self) -> None:
        mock_pocket = MagicMock()
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_voice_state = MagicMock()
        mock_model.get_state_for_audio_prompt.return_value = mock_voice_state

        # Mock generate_audio_stream yielding audio tensors
        mock_tensor = MagicMock()
        mock_numpy = np.zeros(240, dtype=np.float32)
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = mock_numpy

        mock_model.generate_audio_stream.return_value = [mock_tensor, mock_tensor]
        mock_pocket.TTSModel.load_model.return_value = mock_model

        with patch.dict("sys.modules", {"pocket_tts": mock_pocket}), patch("builtins.print"):
            pocket_tts_test.main()

        mock_pocket.TTSModel.load_model.assert_called_once()
        mock_model.get_state_for_audio_prompt.assert_called_once()
        mock_model.generate_audio_stream.assert_called_once()


if __name__ == "__main__":
    unittest.main()
