"""Asynchronous background context compactor for small-weight LLMs."""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Callable, Sequence

import httpx

from constants import (
    CONTEXT_COMPACTOR_CHAR_LIMIT,
    CONTEXT_COMPACTOR_ENABLED,
    CONTEXT_COMPACTOR_MAX_TURNS,
    CONTEXT_COMPACTOR_PRESERVE_TURNS,
    CONTEXT_COMPACTOR_USE_LLM,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
)
from models import Message

logger = logging.getLogger(__name__)


class ContextCompactor:
    """Manages non-blocking asynchronous conversation context compaction.

    Monitors conversation length/character volume. When thresholds are crossed,
    spawns a separate background worker thread to compact older turns into a concise
    rolling summary, preserving recent dialogue turns verbatim.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        enabled: bool = CONTEXT_COMPACTOR_ENABLED,
        max_turns: int = CONTEXT_COMPACTOR_MAX_TURNS,
        char_limit: int = CONTEXT_COMPACTOR_CHAR_LIMIT,
        preserve_turns: int = CONTEXT_COMPACTOR_PRESERVE_TURNS,
        use_llm: bool = CONTEXT_COMPACTOR_USE_LLM,
        timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.enabled = enabled
        self.max_turns = max_turns
        self.char_limit = char_limit
        self.preserve_turns = preserve_turns
        self.use_llm = use_llm
        self.timeout = timeout

        self._lock = threading.Lock()
        self._abort_event = threading.Event()
        self._is_compacting = False
        self._last_compaction_time: float = 0.0

    def abort(self) -> None:
        """Signal immediate abort of running compaction to yield all compute to live turns."""
        self._abort_event.set()

    @property
    def is_compacting(self) -> bool:
        """Return True if background compaction is actively in progress."""
        with self._lock:
            return self._is_compacting

    def should_compact(self, history: Sequence[Message]) -> bool:
        """Determine whether conversation history has exceeded limits and requires compaction."""
        if not self.enabled:
            return False

        with self._lock:
            if self._is_compacting:
                return False

        non_system = [m for m in history if m.role != "system"]
        preserve_count = max(2, self.preserve_turns * 2)

        # Do not compact if we do not have messages beyond the preserved window
        if len(non_system) <= preserve_count:
            return False

        # 1. Turn count limit (turns = user + assistant pairs)
        if len(non_system) >= self.max_turns * 2:
            return True

        # 2. Total character volume limit across history content
        total_chars = sum(len(m.content or "") for m in history)
        if total_chars >= self.char_limit:
            return True

        return False

    def compact_async(
        self,
        history: list[Message],
        on_compacted: Callable[[list[Message], int], None],
    ) -> bool:
        """Spawn a background worker thread to compact older messages if threshold is reached.

        Returns True if a compaction thread was launched, False otherwise.
        """
        if not self.should_compact(history):
            return False

        with self._lock:
            if self._is_compacting:
                return False
            self._is_compacting = True
            self._abort_event.clear()

        snapshot = list(history)
        thread = threading.Thread(
            target=self._run_worker,
            args=(snapshot, on_compacted),
            name="context-compactor",
            daemon=True,
        )
        thread.start()
        logger.info(
            "Launched background context compactor thread for history with %d messages.",
            len(snapshot),
        )
        return True

    def _run_worker(
        self,
        snapshot: list[Message],
        on_compacted: Callable[[list[Message], int], None],
    ) -> None:
        """Executed inside the dedicated background thread."""
        try:
            if self._abort_event.is_set():
                return

            preserve_count = max(2, self.preserve_turns * 2)
            if len(snapshot) <= preserve_count:
                return

            to_compact = snapshot[:-preserve_count]
            preserve = snapshot[-preserve_count:]

            if self._abort_event.is_set():
                return

            if self.use_llm:
                summary = self._summarize_messages(to_compact)
            else:
                summary = self._heuristic_summarizer(to_compact)

            if self._abort_event.is_set():
                logger.info("Context compaction aborted by signal.")
                return

            if not summary:
                logger.warning("Context compactor produced empty summary; skipping update.")
                return

            summary_message = Message(
                role="system",
                content=f"[Prior Context Summary: {summary}]",
            )
            compacted_history = [summary_message] + list(preserve)

            self._last_compaction_time = time.perf_counter()
            logger.info(
                "Background context compaction finished: %d older messages compacted into %d chars.",
                len(to_compact),
                len(summary),
            )
            on_compacted(compacted_history, len(snapshot))

        except Exception as exc:
            logger.warning("Error during background context compaction: %s", exc)
        finally:
            with self._lock:
                self._is_compacting = False

    def _summarize_messages(self, messages: Sequence[Message]) -> str:
        """Summarize older messages via fast LLM call with reliable heuristic fallback."""
        formatted_dialogue = self._format_dialogue(messages)
        if not formatted_dialogue.strip():
            return ""

        # Attempt 1: Call Ollama with low token limit
        try:
            summary = self._call_ollama_summarizer(formatted_dialogue)
            if summary and len(summary.strip()) > 10:
                return summary.strip()
        except Exception as exc:
            logger.debug("Ollama summarizer call failed: %s. Using heuristic summarizer.", exc)

        # Attempt 2: Deterministic heuristic summarizer
        return self._heuristic_summarizer(messages)

    def _format_dialogue(self, messages: Sequence[Message]) -> str:
        """Format a message sequence into clean text for the summarizer."""
        dialogue_lines: list[str] = []
        for m in messages:
            content = (m.content or "").strip()
            if not content:
                continue

            if m.role == "user":
                dialogue_lines.append(f"User: {content}")
            elif m.role == "assistant":
                dialogue_lines.append(f"Assistant: {content}")
            elif m.role == "system":
                # Existing prior summary
                clean_content = content.replace("[Prior Context Summary:", "").rstrip("]")
                dialogue_lines.append(f"Previous Context: {clean_content.strip()}")
            elif m.role == "tool":
                # Condense tool outputs to prevent bloated prompts
                truncated = content[:200] + "..." if len(content) > 200 else content
                dialogue_lines.append(f"Tool Data: {truncated}")

        return "\n".join(dialogue_lines)

    def _call_ollama_summarizer(self, dialogue: str) -> str:
        """Invoke Ollama in non-streaming mode with strict brevity instructions."""
        if self._abort_event.is_set():
            return ""

        prompt = (
            "Summarize the key facts, user preferences, names, and topics from this prior conversation "
            "into 1-2 concise, factual sentences. Do not use markdown, bullets, or pleasantries.\n\n"
            f"Conversation:\n{dialogue}"
        )

        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {
                        "num_predict": 100,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            # Clean any stray formatting or conversational tags
            content = re.sub(r"^(here is a summary:?|summary:?)\s*", "", content, flags=re.I)
            return content.strip()

    def _heuristic_summarizer(self, messages: Sequence[Message]) -> str:
        """Fast, robust keyword and topic extraction when the LLM is busy or unreachable."""
        user_topics: list[str] = []
        assistant_points: list[str] = []

        for m in messages:
            text = (m.content or "").strip()
            if not text:
                continue
            if m.role == "user":
                # Clean filler question phrases
                clean = re.sub(
                    r"^(hey|hello|can you|could you|please|what is|tell me about)\b\s*",
                    "",
                    text,
                    flags=re.I,
                ).strip(" ?.!:,")
                if clean:
                    user_topics.append(clean[:60])
            elif m.role == "assistant":
                # Take the first concise sentence
                first_sentence = re.split(r"[.!?]", text)[0].strip()
                if first_sentence and len(first_sentence) > 10:
                    assistant_points.append(first_sentence[:80])

        summary_parts = []
        if user_topics:
            topics_joined = ", ".join(user_topics[:3])
            summary_parts.append(f"User inquired about: {topics_joined}")
        if assistant_points:
            answers_joined = "; ".join(assistant_points[:2])
            summary_parts.append(f"Key responses discussed: {answers_joined}")

        return ". ".join(summary_parts) + "." if summary_parts else "Prior conversation covered general topics."
