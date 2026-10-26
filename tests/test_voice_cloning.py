"""Tests for voice cloning and cloned voice resolution using standard unittest."""
from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import constants
from constants import (
    POCKET_TTS_DEFAULT_VOICE,
    list_cloned_voices,
    resolve_voice_spec,
)
from services.tts.src.tts_types import TTSConfig


class TestVoiceCloning(unittest.TestCase):
    def test_list_cloned_voices_empty(self) -> None:
        """Empty directory should return an empty list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(list_cloned_voices(tmp_dir), [])

    def test_list_cloned_voices_nonexistent(self) -> None:
        """Non-existent directory should return an empty list without error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent = Path(tmp_dir) / "does_not_exist"
            self.assertEqual(list_cloned_voices(non_existent), [])

    def test_list_cloned_voices_discovery(self) -> None:
        """Should discover only .safetensors files, ignoring hidden files and other extensions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir)
            (p / "voice_b.safetensors").write_text("fake")
            (p / "voice_a.safetensors").write_text("fake")
            (p / "voice_c.wav").write_text("fake")
            (p / ".hidden.safetensors").write_text("fake")

            voices = list_cloned_voices(p)
            self.assertEqual(voices, ["voice_a", "voice_b"])

    def test_resolve_voice_spec_default(self) -> None:
        """None or empty string should resolve to the default voice."""
        self.assertEqual(resolve_voice_spec(None), POCKET_TTS_DEFAULT_VOICE)
        self.assertEqual(resolve_voice_spec(""), POCKET_TTS_DEFAULT_VOICE)
        self.assertEqual(resolve_voice_spec("   "), POCKET_TTS_DEFAULT_VOICE)

    def test_resolve_voice_spec_direct_file(self) -> None:
        """Direct file path should resolve to its absolute path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            voice_file = Path(tmp_dir) / "custom.safetensors"
            voice_file.write_text("fake")

            resolved = resolve_voice_spec(str(voice_file))
            self.assertEqual(resolved, str(voice_file.resolve()))

    def test_resolve_voice_spec_from_cloned_voices_dir(self) -> None:
        """Voice name should resolve to the .safetensors file in CLONED_VOICES_DIR."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir)
            cloned_file = p / "my_custom_voice.safetensors"
            cloned_file.write_text("fake")

            orig_dir = constants.CLONED_VOICES_DIR
            try:
                constants.CLONED_VOICES_DIR = p
                resolved = resolve_voice_spec("my_custom_voice")
                self.assertEqual(resolved, str(cloned_file.resolve()))
            finally:
                constants.CLONED_VOICES_DIR = orig_dir

    def test_resolve_voice_spec_predefined_fallback(self) -> None:
        """A voice name that is not a local file should be returned as-is (predefined catalog voice)."""
        self.assertEqual(resolve_voice_spec("anna"), "anna")
        self.assertEqual(resolve_voice_spec("alba"), "alba")

    def test_sanitize_voice_name(self) -> None:
        """Voice name should be sanitized to alphanumeric and underscore/hyphen."""
        from clone_voice import _sanitize_voice_name

        self.assertEqual(_sanitize_voice_name("My Cool Voice!"), "My_Cool_Voice")
        self.assertEqual(_sanitize_voice_name("alex-01"), "alex-01")
        self.assertEqual(_sanitize_voice_name("...///..."), "custom_voice")

    def test_pocket_tts_service_resolves_voice(self) -> None:
        """PocketTTSService should use resolve_voice_spec when loading the voice state."""
        from services.tts.src.pocket_tts_service import PocketTTSService

        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir)
            cloned_file = p / "cloned_test.safetensors"
            cloned_file.write_text("fake")

            orig_dir = constants.CLONED_VOICES_DIR
            try:
                constants.CLONED_VOICES_DIR = p
                config = TTSConfig(backend="pocket", voice="cloned_test")
                service = PocketTTSService(config)

                mock_engine = MagicMock()
                mock_engine.sample_rate = 24000
                mock_engine.get_state_for_audio_prompt.return_value = {"state": "mock"}

                mock_pocket = MagicMock()
                mock_pocket.TTSModel.load_model.return_value = mock_engine
                with patch.dict("sys.modules", {"pocket_tts": mock_pocket}):
                    service._load_engine()

                mock_engine.get_state_for_audio_prompt.assert_called_once_with(str(cloned_file.resolve()))
            finally:
                constants.CLONED_VOICES_DIR = orig_dir

    def test_cli_voice_argument(self) -> None:
        """CLI argument parser should accept --voice."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--voice",
            type=str,
            default=os.getenv("VOICECHAT_TTS_VOICE"),
        )
        args = parser.parse_args(["--voice", "sarah_clone"])
        self.assertEqual(args.voice, "sarah_clone")

    def test_record_audio_from_microphone_mocked(self) -> None:
        """_record_audio_from_microphone should complete cleanly using InputStream and save a valid wav."""
        import numpy as np
        from clone_voice import _record_audio_from_microphone

        class MockInputStream:
            def __init__(self, *args, callback=None, **kwargs):
                self.callback = callback

            def __enter__(self):
                if self.callback:
                    # Feed mock audio chunks
                    self.callback(np.zeros((512, 1), dtype=np.int16), 512, None, None)
                return self

            def __exit__(self, *args):
                pass

        with patch("sounddevice.InputStream", MockInputStream), patch("time.sleep", return_value=None):
            wav_path = _record_audio_from_microphone(duration_seconds=0.1, sample_rate=16000)
            try:
                self.assertTrue(wav_path.is_file())
                self.assertGreater(wav_path.stat().st_size, 44)
            finally:
                if wav_path.is_file():
                    wav_path.unlink()


if __name__ == "__main__":
    unittest.main()
