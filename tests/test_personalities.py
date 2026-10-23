"""Unit tests for voice assistant personalities system."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from services.llm.src.personalities import (
    DEFAULT_PERSONALITY,
    PERSONALITIES,
    Personality,
    build_personality_prompt,
    get_personality,
    list_personalities,
)
from services.llm.src.llm_provider import LLMProvider, LLMProviderConfig


class TestPersonalities(unittest.TestCase):
    def test_required_personalities_present(self) -> None:
        """Verify the 4 required and bonus popular personalities exist."""
        required = ["natural", "professional", "playful", "annoyed"]
        for p_id in required:
            self.assertIn(p_id, PERSONALITIES)

        # Bonus popular personalities
        popular = ["sarcastic", "empathetic", "enthusiastic", "concise", "philosophical"]
        for p_id in popular:
            self.assertIn(p_id, PERSONALITIES)

    def test_list_personalities(self) -> None:
        """Verify list_personalities returns all registered personalities in order."""
        plist = list_personalities()
        self.assertGreaterEqual(len(plist), 9)
        self.assertEqual(plist[0].id, "natural")

    def test_get_personality_by_id(self) -> None:
        p = get_personality("annoyed")
        self.assertEqual(p.id, "annoyed")
        self.assertEqual(p.name, "Annoyed & Disrespectful")

    def test_get_personality_case_insensitive(self) -> None:
        p = get_personality("PLAYFUL")
        self.assertEqual(p.id, "playful")

    def test_get_personality_by_index_string(self) -> None:
        p1 = get_personality("1")
        self.assertEqual(p1.id, "natural")
        p4 = get_personality("4")
        self.assertEqual(p4.id, "annoyed")

    def test_get_personality_fallback_on_unknown(self) -> None:
        p = get_personality("non_existent_personality_xyz")
        self.assertEqual(p.id, DEFAULT_PERSONALITY)

    def test_get_personality_none_fallback(self) -> None:
        p = get_personality(None)
        self.assertEqual(p.id, DEFAULT_PERSONALITY)

    def test_build_personality_prompt_with_name(self) -> None:
        prompt = build_personality_prompt(assistant_name="Waffle", personality_id="annoyed")
        self.assertIn("You are a helpful voice assistant named Waffle.", prompt)
        self.assertIn("annoyed, cynical, and disrespectful tone", prompt)

    def test_build_personality_prompt_without_name(self) -> None:
        prompt = build_personality_prompt(assistant_name="", personality_id="playful")
        self.assertIn("You are a helpful voice assistant.", prompt)
        self.assertNotIn("named", prompt)
        self.assertIn("playful, witty, and spontaneous", prompt)

    def test_build_personality_prompt_natural(self) -> None:
        prompt = build_personality_prompt(assistant_name="Jarvis", personality_id="natural")
        self.assertIn("named Jarvis", prompt)
        self.assertIn("natural, calm, and friendly", prompt)

    def test_build_personality_prompt_professional(self) -> None:
        prompt = build_personality_prompt(assistant_name="Friday", personality_id="professional")
        self.assertIn("named Friday", prompt)
        self.assertIn("professional, courteous, and accurate tone", prompt)


class TestLLMProviderPersonalityIntegration(unittest.TestCase):
    def test_provider_default_personality_from_env(self) -> None:
        with patch("services.llm.src.llm_provider.ASSISTANT_PERSONALITY", "annoyed"):
            prompt = LLMProvider._build_default_system_prompt()
            self.assertIn("annoyed, cynical, and disrespectful tone", prompt)

    def test_provider_config_overrides_personality(self) -> None:
        with patch("services.llm.src.llm_provider.ASSISTANT_PERSONALITY", "natural"):
            config = LLMProviderConfig(assistant_name="TestBot", personality="sarcastic")
            provider = LLMProvider(config)
            req = provider._build_request([])
            system_content = req["messages"][0]["content"]
            self.assertIn("named TestBot", system_content)
            self.assertIn("sharp wit, dry sarcasm, and clever irony", system_content)

    def test_provider_tool_prompt_respects_personality(self) -> None:
        prompt = LLMProvider._build_tool_system_prompt(personality="concise")
        self.assertIn("razor-sharp brevity", prompt)

    def test_provider_followup_prompt_respects_personality(self) -> None:
        prompt = LLMProvider._build_followup_system_prompt(personality="philosophical")
        self.assertIn("thoughtful, wise, and philosophical tone", prompt)


if __name__ == "__main__":
    unittest.main()
