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
        self.assertIn('VOICECHAT_PERSONALITY="natural"', content)
        self.assertIn('OLLAMA_MODEL="qwen3.5:4b"', content)
        self.assertIn('VOICECHAT_TTS_VOICE="anna"', content)
        self.assertIn('AUDIO_INPUT_DEVICE="default"', content)
        self.assertIn('VOICECHAT_BARGE_IN_ENABLED="true"', content)
        self.assertIn('VOICECHAT_AEC_ENABLED="true"', content)
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

    @patch("urllib.request.urlopen")
    def test_list_ollama_models_success(self, mock_urlopen) -> None:
        import io
        mock_response = io.BytesIO(b'{"models": [{"name": "qwen3.5:4b"}, {"name": "llama3:latest"}]}')
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response

        models = setup._list_ollama_models("http://localhost:11434")
        self.assertEqual(models, ["qwen3.5:4b", "llama3:latest"])

    @patch("urllib.request.urlopen", side_effect=Exception("Connection refused"))
    def test_list_ollama_models_handles_error(self, mock_urlopen) -> None:
        models = setup._list_ollama_models("http://localhost:11434")
        self.assertEqual(models, [])

    @patch("setup._list_ollama_models", return_value=["qwen3.5:4b", "gemma4:e4b"])
    @patch("builtins.input", side_effect=[
        "",           # User name
        "",           # User location
        "",           # User timezone
        "",           # Assistant name (default: Waffle)
        "",           # Personality (default: natural)
        "",           # TTS Backend (default: pocket)
        "",           # Voice (default: anna)
        "",           # Speed (default: 1.0)
        "",           # Mic (default: default)
        "",           # Trigger mode (default: false)
        "",           # Barge in (default: true)
        "",           # AEC (default: true)
        "",           # Ollama url (default)
        "2",          # Model selection: index 2 -> gemma4:e4b
        "",           # GitHub token
    ])
    def test_interactive_model_selection_from_api_list(self, mock_input, mock_models) -> None:
        setup.run_setup(env_path=self.env_file, force=True, non_interactive=False)
        content = self.env_file.read_text(encoding="utf-8")
        self.assertIn('OLLAMA_MODEL="gemma4:e4b"', content)
        self.assertIn('VOICECHAT_PERSONALITY="natural"', content)

    @patch("setup._list_ollama_models", return_value=["qwen3.5:4b"])
    @patch("builtins.input", side_effect=[
        "",           # User name
        "",           # User location
        "",           # User timezone
        "",           # Assistant name
        "4",          # Personality selection: index 4 -> annoyed
        "",           # TTS Backend
        "",           # Voice
        "",           # Speed
        "",           # Mic
        "",           # Trigger mode
        "",           # Barge in (default: true)
        "",           # AEC (default: true)
        "",           # Ollama url
        "custom_mistral:7b",  # Model selection: custom name
        "",           # GitHub token
    ])
    def test_interactive_model_selection_custom_name(self, mock_input, mock_models) -> None:
        setup.run_setup(env_path=self.env_file, force=True, non_interactive=False)
        content = self.env_file.read_text(encoding="utf-8")
        self.assertIn('OLLAMA_MODEL="custom_mistral:7b"', content)
        self.assertIn('VOICECHAT_PERSONALITY="annoyed"', content)


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

    def test_load_env_file_overrides_empty_environment_variables(self) -> None:
        self.env_file.write_text(
            'USER_NAME="Aayush"\n'
            'USER_LOCATION="New Delhi, India"\n',
            encoding="utf-8",
        )
        # Pre-set empty environment variables (as often occurs in subshells)
        os.environ["USER_NAME"] = ""
        os.environ["USER_LOCATION"] = "   "

        _load_env_file(self.env_file)
        self.assertEqual(os.environ.get("USER_NAME"), "Aayush")
        self.assertEqual(os.environ.get("USER_LOCATION"), "New Delhi, India")

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

    def test_llm_provider_when_no_user_name_prohibits_taking_name(self) -> None:
        """When no user name is provided, LLM must be explicitly forbidden from taking/inventing user name."""
        with patch("services.llm.src.llm_provider.USER_NAME", ""):
            prompt = LLMProvider._build_default_system_prompt()
            self.assertIn("The user's name is not provided. Never assume, invent, or use any name to address or refer to the user.", prompt)
            self.assertNotIn("speaking with", prompt)
            self.assertIn("Never address the user by any name unless their name is explicitly provided", prompt)

    def test_llm_provider_when_no_assistant_name_avoids_hardcoding(self) -> None:
        """When no assistant name is provided in .env/config, assistant must not hardcode 'Waffle'."""
        with patch("services.llm.src.llm_provider.ASSISTANT_NAME", ""):
            prompt = LLMProvider._build_default_system_prompt()
            self.assertIn("You are a helpful voice assistant. You speak in a natural", prompt)
            self.assertNotIn("named", prompt)

    def test_llm_provider_instance_config_overrides(self) -> None:
        """LLMProviderConfig overrides environment variables dynamically."""
        config = LLMProviderConfig(assistant_name="Friday", user_name="Bruce Wayne")
        provider = LLMProvider(config)
        req = provider._build_request([])
        system_content = req["messages"][0]["content"]
        self.assertIn("named Friday", system_content)
        self.assertIn("speaking with Bruce Wayne", system_content)


if __name__ == "__main__":
    unittest.main()

