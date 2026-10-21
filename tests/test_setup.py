"""Unit tests for setup wizard, .env loading, and user personalization."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from constants import _load_env_file
from services.llm.src.llm_provider import LLMProvider, LLMProviderConfig
import setup


class TestSetupWizard(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.env_file = Path(self.tmp_dir.name) / ".env"

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_prompt_with_default_non_interactive(self) -> None:
        val = setup.prompt_with_default("Ollama URL", "http://localhost:11434", non_interactive=True)
        self.assertEqual(val, "http://localhost:11434")

    def test_prompt_without_default_non_interactive(self) -> None:
        val = setup.prompt_without_default("Your Name", non_interactive=True)
        self.assertEqual(val, "")

    @patch("builtins.input", return_value="")
    def test_prompt_with_default_interactive_empty_input(self, mock_input) -> None:
        val = setup.prompt_with_default("Model", "qwen3.5:4b", non_interactive=False)
        self.assertEqual(val, "qwen3.5:4b")

    @patch("builtins.input", return_value="custom_model")
    def test_prompt_with_default_interactive_custom_input(self, mock_input) -> None:
        val = setup.prompt_with_default("Model", "qwen3.5:4b", non_interactive=False)
        self.assertEqual(val, "custom_model")

    @patch("builtins.input", return_value="")
    def test_prompt_without_default_interactive_empty_input(self, mock_input) -> None:
        val = setup.prompt_without_default("Your Location", non_interactive=False)
        self.assertEqual(val, "")

    @patch("builtins.input", return_value="London, UK")
    def test_prompt_without_default_interactive_custom_input(self, mock_input) -> None:
        val = setup.prompt_without_default("Your Location", non_interactive=False)
        self.assertEqual(val, "London, UK")

    def test_run_setup_creates_env_file(self) -> None:
        setup.run_setup(env_path=self.env_file, force=True, non_interactive=True)
        self.assertTrue(self.env_file.is_file())
        content = self.env_file.read_text(encoding="utf-8")

        # Verify defaults are present
        self.assertIn('ASSISTANT_NAME="Waffle"', content)
        self.assertIn('OLLAMA_MODEL="qwen3.5:4b"', content)
        self.assertIn('VOICECHAT_TTS_VOICE="anna"', content)
        self.assertIn('AUDIO_INPUT_DEVICE="default"', content)
        self.assertIn('GITHUB_PERSONAL_ACCESS_TOKEN=""', content)

        # Verify fields without defaults are present and empty
        self.assertIn('USER_NAME=""', content)
        self.assertIn('USER_LOCATION=""', content)
        self.assertIn('USER_TIMEZONE=""', content)

    def test_run_setup_skips_when_env_exists_without_force(self) -> None:
        self.env_file.write_text("CUSTOM_KEY=custom_value\n", encoding="utf-8")
        setup.run_setup(env_path=self.env_file, force=False, non_interactive=True)
        content = self.env_file.read_text(encoding="utf-8")
        self.assertEqual(content, "CUSTOM_KEY=custom_value\n")


class TestEnvLoadingAndPersonalization(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.env_file = Path(self.tmp_dir.name) / ".env"

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_load_env_file_parses_cleanly(self) -> None:
        self.env_file.write_text(
            '# Comment line\n'
            'TEST_UNQUOTED=hello\n'
            'TEST_DOUBLE_QUOTES="world with spaces"\n'
            'TEST_SINGLE_QUOTES=\'single quoted\'\n',
            encoding="utf-8",
        )
        # Clear any prior env values
        for k in ["TEST_UNQUOTED", "TEST_DOUBLE_QUOTES", "TEST_SINGLE_QUOTES"]:
            os.environ.pop(k, None)

        _load_env_file(self.env_file)
        self.assertEqual(os.environ.get("TEST_UNQUOTED"), "hello")
        self.assertEqual(os.environ.get("TEST_DOUBLE_QUOTES"), "world with spaces")
        self.assertEqual(os.environ.get("TEST_SINGLE_QUOTES"), "single quoted")

    def test_llm_provider_builds_personalized_prompt(self) -> None:
        with patch("services.llm.src.llm_provider.ASSISTANT_NAME", "Jarvis"), \
             patch("services.llm.src.llm_provider.USER_NAME", "Tony Stark"), \
             patch("services.llm.src.llm_provider.USER_LOCATION", "Malibu, CA"), \
             patch("services.llm.src.llm_provider.USER_TIMEZONE", "America/Los_Angeles"):

            prompt = LLMProvider._build_default_system_prompt()
            self.assertIn("named Jarvis", prompt)
            self.assertIn("speaking with Tony Stark", prompt)
            self.assertIn("location is Malibu, CA", prompt)
            self.assertIn("timezone is America/Los_Angeles", prompt)


if __name__ == "__main__":
    unittest.main()
