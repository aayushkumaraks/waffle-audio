"""Filler phrases and emotional speech cues played during tool execution.

Synthesizes conversational filler sentences on the go using the active TTS pipeline,
ensuring consistent voice model, voice cloning, and audio engine characteristics.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Callable

from constants import FILLER_PHRASES_ENABLED, is_benchmark_enabled

logger = logging.getLogger(__name__)

# Dictionary of contextual filler phrases with natural emotional cues and vocal tones
FILLER_PHRASES: dict[str, list[dict[str, str]]] = {
    "search": [
        {"id": "search_1", "text": "let me check.", "emotion": "curious"},
        {"id": "search_2", "text": "Searching for it.", "emotion": "focused"},
        {"id": "search_3", "text": "looking it up.", "emotion": "thoughtful"},
        {"id": "search_4", "text": "let me search that for you.", "emotion": "curious"},
        {"id": "search_5", "text": "looking that up online right now.", "emotion": "casual"},
        {"id": "search_6", "text": "searching for it on the web.", "emotion": "focused"},
        {"id": "search_7", "text": "let me see what I can find.", "emotion": "thoughtful"},
        {"id": "search_8", "text": "checking that out for you.", "emotion": "helpful"},
        {"id": "search_9", "text": "searching for it now.", "emotion": "attentive"},
        {"id": "search_10", "text": "let me search that up.", "emotion": "casual"},
    ],
    "weather": [
        {"id": "weather_1", "text": "let me check the weather.", "emotion": "curious"},
        {"id": "weather_2", "text": "looking up the forecast for you.", "emotion": "thoughtful"},
        {"id": "weather_3", "text": "checking current weather conditions.", "emotion": "attentive"},
        {"id": "weather_4", "text": "let me see what the weather looks like.", "emotion": "curious"},
        {"id": "weather_5", "text": "checking the forecast right now.", "emotion": "focused"},
        {"id": "weather_6", "text": "pulling up the weather report.", "emotion": "helpful"},
        {"id": "weather_7", "text": "let me look up the weather for that.", "emotion": "casual"},
    ],
    "time": [
        {"id": "time_1", "text": "let me check the time.", "emotion": "quick"},
        {"id": "time_2", "text": "checking the clock.", "emotion": "casual"},
        {"id": "time_3", "text": "looking up the current time for you.", "emotion": "helpful"},
        {"id": "time_4", "text": "let me check what time it is.", "emotion": "thoughtful"},
        {"id": "time_5", "text": "looking up the time zone now.", "emotion": "focused"},
    ],
    "database": [
        {"id": "db_1", "text": "querying the database now.", "emotion": "focused"},
        {"id": "db_2", "text": "let me look up those records for you.", "emotion": "helpful"},
        {"id": "db_3", "text": "checking the data.", "emotion": "analytical"},
        {"id": "db_4", "text": "pulling up the database details.", "emotion": "calm"},
        {"id": "db_5", "text": "searching through the records.", "emotion": "thoughtful"},
    ],
    "filesystem": [
        {"id": "fs_1", "text": "checking the files for you.", "emotion": "thoughtful"},
        {"id": "fs_2", "text": "looking through the directory.", "emotion": "casual"},
        {"id": "fs_3", "text": "scanning those files real quick.", "emotion": "eager"},
        {"id": "fs_4", "text": "checking the local files.", "emotion": "focused"},
        {"id": "fs_5", "text": "looking up the folder contents.", "emotion": "analytical"},
    ],
    "fetch": [
        {"id": "fetch_1", "text": "fetching that page for you.", "emotion": "curious"},
        {"id": "fetch_2", "text": "pulling up that website.", "emotion": "eager"},
        {"id": "fetch_3", "text": "reading that link now.", "emotion": "thoughtful"},
        {"id": "fetch_4", "text": "checking that web page.", "emotion": "focused"},
    ],
    "general": [
        {"id": "gen_1", "text": "let me check.", "emotion": "thoughtful"},
        {"id": "gen_2", "text": "looking it up.", "emotion": "curious"},
        {"id": "gen_3", "text": "Searching for it.", "emotion": "focused"},
        {"id": "gen_4", "text": "let me look into that.", "emotion": "helpful"},
        {"id": "gen_5", "text": "checking that right now.", "emotion": "attentive"},
        {"id": "gen_6", "text": "give me a moment while I check.", "emotion": "calm"},
        {"id": "gen_7", "text": "let me see.", "emotion": "thoughtful"},
        {"id": "gen_8", "text": "working on that now.", "emotion": "focused"},
        {"id": "gen_9", "text": "looking into that for you.", "emotion": "helpful"},
    ],
}


class PhraseManager:
    """Manages emotional filler phrases and contextual on-the-go speech generation."""

    def __init__(
        self,
        tts: Any = None,
        speak_callback: Callable[[str], None] | None = None,
        enabled: bool = FILLER_PHRASES_ENABLED,
    ) -> None:
        self.enabled = enabled
        self._tts = tts
        self._speak_callback = speak_callback

        self._lock = threading.Lock()
        # category -> list of recently used phrase IDs to cycle without immediate repeats
        self._recent_usage: dict[str, list[str]] = {}

    def set_tts(self, tts: Any) -> None:
        """Register the active TTSService to synthesize phrases on the go."""
        self._tts = tts

    def set_speak_callback(self, callback: Callable[[str], None] | None) -> None:
        """Register a callback (e.g. tts.speak) to synthesize phrases on the go."""
        self._speak_callback = callback

    def match_category(self, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        """Map tool name and arguments to the most accurate category."""
        target = (tool_name or "").lower()
        args_str = " ".join(f"{k} {v}" for k, v in (arguments or {}).items()).lower()
        combined = f"{target} {args_str}"

        if any(w in combined for w in ["weather", "forecast", "meteo", "rain", "temperature", "climate"]):
            return "weather"
        if any(w in combined for w in ["search", "duckduckgo", "ddgs", "google", "lookup", "query_web", "find"]):
            return "search"
        if any(w in combined for w in ["sql", "sqlite", "database", "query_db", "db", "table", "records"]):
            return "database"
        if any(w in combined for w in ["file", "dir", "directory", "read_file", "path", "folder", "filesystem"]):
            return "filesystem"
        if any(w in combined for w in ["time", "clock", "date", "timezone", "now"]):
            return "time"
        if any(w in combined for w in ["fetch", "url", "http", "curl", "browser", "puppeteer", "webpage"]):
            return "fetch"

        return "general"

    def select_phrase(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, str]:
        """Pick the most accurate phrase, avoiding repeats and randomizing among similar phrases."""
        with self._lock:
            category = self.match_category(tool_name, arguments)
            phrase_list = FILLER_PHRASES.get(category) or FILLER_PHRASES["general"]

            used_ids = self._recent_usage.setdefault(category, [])

            # Filter out recently used phrases in this category
            unused_phrases = [p for p in phrase_list if p["id"] not in used_ids]

            # If all phrases in category have been used, reset history for this category
            if not unused_phrases:
                used_ids.clear()
                unused_phrases = phrase_list

            # Randomize among candidate unused phrases
            chosen = random.choice(unused_phrases)
            used_ids.append(chosen["id"])

            return {
                "id": chosen["id"],
                "text": chosen["text"],
                "emotion": chosen["emotion"],
                "category": category,
            }

    def play_filler(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        callback: Callable[[str], None] | None = None,
    ) -> dict[str, str] | None:
        """Pick a contextual phrase and generate/speak it on the go using the active TTS pipeline."""
        if not self.enabled:
            return None

        t_start = time.perf_counter()
        chosen = self.select_phrase(tool_name, arguments)
        phrase_text = chosen["text"]

        speak_fn = callback or self._speak_callback
        if speak_fn is not None:
            try:
                speak_fn(phrase_text)
                if is_benchmark_enabled():
                    elapsed = time.perf_counter() - t_start
                    logger.info(
                        "[FILLER BENCHMARK] Spoke filler phrase [%s / %s] '%s' for tool '%s' via callback in %.4fs",
                        chosen.get("category", "general"),
                        chosen.get("emotion", "neutral"),
                        phrase_text,
                        tool_name,
                        elapsed,
                    )
            except Exception as exc:
                logger.warning("Error dispatching filler phrase to speak callback: %s", exc)
        elif self._tts is not None and hasattr(self._tts, "speak"):
            try:
                self._tts.speak(phrase_text)
                if is_benchmark_enabled():
                    elapsed = time.perf_counter() - t_start
                    logger.info(
                        "[FILLER BENCHMARK] Spoke filler phrase [%s / %s] '%s' for tool '%s' via TTSService in %.4fs",
                        chosen.get("category", "general"),
                        chosen.get("emotion", "neutral"),
                        phrase_text,
                        tool_name,
                        elapsed,
                    )
            except Exception as exc:
                logger.warning("Error dispatching filler phrase to TTSService: %s", exc)
        else:
            logger.debug(
                "No TTS pipeline or speak callback attached; phrase '%s' selected but not spoken.",
                phrase_text,
            )

        return chosen

