"""Unit tests for constants.py module configuration, environment loading, and helpers."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from constants import (
    _int_from_env,
    _load_env_file,
    is_benchmark_enabled,
    resolve_voice_spec,
    set_benchmark_enabled,
)


class TestConstants(unittest.TestCase):
    def test_int_from_env_valid(self) -> None:
        with patch.dict(os.environ, {"TEST_INT_VAR": "42"}):
            self.assertEqual(_int_from_env("TEST_INT_VAR", 10), 42)

    def test_int_from_env_default_when_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_int_from_env("MISSING_VAR", 99), 99)

    def test_int_from_env_raises_value_error_on_invalid(self) -> None:
        with patch.dict(os.environ, {"INVALID_INT": "not-an-int"}):
            with self.assertRaises(ValueError):
                _int_from_env("INVALID_INT", 10)

    def test_load_env_file_missing_file_does_not_raise(self) -> None:
        # Should cleanly return without crashing
        _load_env_file("/path/that/does/not/exist/.env")

    def test_load_env_file_parses_and_preserves_existing(self) -> None:
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".env") as f:
            f.write("# This is a comment\n")
            f.write("EMPTY_VAL=\n")
            f.write("PLAIN_VAL=hello\n")
            f.write('QUOTED_VAL="quoted string"\n')
            f.write("SINGLE_QUOTED='single quoted'\n")
            f.write("ALREADY_SET=new_value\n")
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"ALREADY_SET": "original_value"}):
                _load_env_file(temp_path)
                self.assertEqual(os.environ.get("PLAIN_VAL"), "hello")
                self.assertEqual(os.environ.get("QUOTED_VAL"), "quoted string")
                self.assertEqual(os.environ.get("SINGLE_QUOTED"), "single quoted")
                self.assertEqual(os.environ.get("ALREADY_SET"), "original_value")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_benchmark_enabled_toggle(self) -> None:
        orig = is_benchmark_enabled()
        try:
            set_benchmark_enabled(True)
            self.assertTrue(is_benchmark_enabled())
            set_benchmark_enabled(False)
            self.assertFalse(is_benchmark_enabled())
        finally:
            set_benchmark_enabled(orig)

    def test_resolve_voice_spec(self) -> None:
        # Voice spec helper in constants
        self.assertTrue(len(resolve_voice_spec("alba")) > 0)
        self.assertEqual(resolve_voice_spec("custom_path.wav"), "custom_path.wav")


if __name__ == "__main__":
    unittest.main()
