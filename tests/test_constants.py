"""Unit tests for constants.py configuration and environment parsing."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import constants
from constants import (
    POCKET_TTS_DEFAULT_VOICE,
    _float_from_env,
    _int_from_env,
    is_benchmark_enabled,
    list_cloned_voices,
    resolve_voice_spec,
    set_benchmark_enabled,
)


class TestConstants(unittest.TestCase):
    def test_int_from_env_valid(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"TEST_INT_VAR": "42"}):
            self.assertEqual(_int_from_env("TEST_INT_VAR", 10), 42)

    def test_int_from_env_default(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_int_from_env("NON_EXISTENT_INT", 100), 100)

    def test_int_from_env_invalid_raises_value_error(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"INVALID_INT": "not_an_int"}):
            with self.assertRaises(ValueError):
                _int_from_env("INVALID_INT", 10)

    def test_float_from_env_valid(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"TEST_FLOAT_VAR": "3.14"}):
            self.assertAlmostEqual(_float_from_env("TEST_FLOAT_VAR", 1.0), 3.14)

    def test_float_from_env_default(self) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_float_from_env("NON_EXISTENT_FLOAT", 2.5), 2.5)

    def test_float_from_env_invalid_raises_value_error(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"INVALID_FLOAT": "abc"}):
            with self.assertRaises(ValueError):
                _float_from_env("INVALID_FLOAT", 1.0)

    def test_list_cloned_voices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir)
            (p / "charlie.safetensors").write_text("dummy")
            (p / "alice.safetensors").write_text("dummy")
            (p / "bob.wav").write_text("dummy")
            voices = list_cloned_voices(p)
            self.assertEqual(voices, ["alice", "charlie"])

    def test_resolve_voice_spec(self) -> None:
        self.assertEqual(resolve_voice_spec(""), POCKET_TTS_DEFAULT_VOICE)
        self.assertEqual(resolve_voice_spec("anna"), "anna")

        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir)
            v_file = p / "custom.safetensors"
            v_file.write_text("dummy")
            resolved = resolve_voice_spec(str(v_file))
            self.assertEqual(resolved, str(v_file.resolve()))

    def test_benchmark_enabled_toggle(self) -> None:
        initial = is_benchmark_enabled()
        try:
            set_benchmark_enabled(True)
            self.assertTrue(is_benchmark_enabled())
            set_benchmark_enabled(False)
            self.assertFalse(is_benchmark_enabled())
        finally:
            set_benchmark_enabled(initial)


if __name__ == "__main__":
    unittest.main()
