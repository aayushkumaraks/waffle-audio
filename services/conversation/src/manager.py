from __future__ import annotations

import logging
import queue
import re
import threading
import time

from typing import Callable

from constants import (
    ASSISTANT_NAME,
    SENTENCE_GATE_GAP_TIMEOUT,
    SENTENCE_GATE_QUICK_FIRE_TIMEOUT,
)
from models import Message

from services.stt.src import STTListener, STTService
from services.llm.src import LLMListener, LLMService
from services.tts.src import TTSService
from .compactor import ContextCompactor
from .sentence_gate import ClauseChunker, SentenceGate

logger = logging.getLogger(__name__)


class _STTListenerAdapter(STTListener):
    """Adapts STT events to ConversationManager callbacks."""

    def __init__(self, manager: "ConversationManager") -> None:
        self._manager = manager

    def on_transcript_started(self, text: str) -> None:
        self._manager._handle_transcript_started(text)

    def on_transcript_updated(self, text: str) -> None:
        self._manager._handle_transcript_updated(text)

    def on_transcript_completed(self, text: str) -> None:
        self._manager._handle_transcript_completed(text)


class ConversationManager(LLMListener):
    """Coordinates the conversation between application services."""

    def __init__(
        self,
        stt: STTService,
        llm: LLMService,
        tts: TTSService,
        on_user_turn: Callable[[str], None] | None = None,
        gap_timeout: float = SENTENCE_GATE_GAP_TIMEOUT,
        quick_fire_timeout: float = SENTENCE_GATE_QUICK_FIRE_TIMEOUT,
        max_history_turns: int = 8,
        compactor: ContextCompactor | None = None,
        trigger_mode: bool = False,
        assistant_name: str = ASSISTANT_NAME,
        on_user_turn_ignored: Callable[[str], None] | None = None,
        follow_up_window_seconds: float = 15.0,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._on_user_turn = on_user_turn
        self._max_history_turns = max_history_turns
        self._compactor = compactor or ContextCompactor()
        self._trigger_mode = trigger_mode
        self._assistant_name = assistant_name.strip() if assistant_name else "Waffle"
        self._on_user_turn_ignored = on_user_turn_ignored
        self._follow_up_window_seconds = follow_up_window_seconds
        self._last_assistant_turn_time: float = 0.0

        self._stt_listener = _STTListenerAdapter(self)

        self._history: list[Message] = []

        self._gate = SentenceGate(
            callback=self._trigger_llm,
            gap_timeout=gap_timeout,
            quick_fire_timeout=quick_fire_timeout,
        )
        self._chunker = ClauseChunker()
        self._partial_reply = ""
        self._tool_args: dict[str, dict] = {}
        self._tool_notes: list[str] = []

        self._started = False
        self._pending_generation = False
        self._state_lock = threading.Lock()

    def start(self) -> None:
        """Start listening for conversation events."""

        if self._started:
            return

        self._stt.add_listener(self._stt_listener)
        self._llm.add_listener(self)

        self._started = True

        logger.info("ConversationManager started.")

    def stop(self) -> None:
        """Stop listening for conversation events."""

        if not self._started:
            return

        self._gate.cancel()
        self._stt.remove_listener(self._stt_listener)
        self._llm.remove_listener(self)

        self._started = False

        logger.info("ConversationManager stopped.")

    def reset(self) -> None:
        """Clear the current conversation history."""

        self._history.clear()

        logger.info("Conversation history cleared.")

    @property
    def history(self) -> list[Message]:
        """Returns a copy of the conversation history."""

        return self._history.copy()

    def submit_user_text(self, text: str) -> None:
        """Submit a finalized user utterance to the conversation pipeline."""

        text = text.strip()
        if not text:
            logger.debug("Ignoring empty user text.")
            return

        self._trigger_llm(text)

    # ------------------------------------------------------------------
    # STT Event Handlers
    # ------------------------------------------------------------------

    def _handle_transcript_started(self, text: str) -> None:
        logger.debug("Speech started: %s", text)
        self._compactor.abort()
        self._gate.notify_speech_active()

    def _handle_transcript_updated(self, text: str) -> None:
        logger.debug("Transcript updated: %s", text)
        self._gate.notify_speech_active()

    def _handle_transcript_completed(self, text: str) -> None:
        logger.info("Transcript completed: %s", text)

        text = text.strip()

        if not text:
            logger.debug("Ignoring empty transcript.")
            return

        self._gate.submit(text)

    @property
    def trigger_mode(self) -> bool:
        """Returns whether the assistant only processes utterances with the trigger word."""
        return self._trigger_mode

    @trigger_mode.setter
    def trigger_mode(self, value: bool) -> None:
        self._trigger_mode = bool(value)

    @property
    def assistant_name(self) -> str:
        """Returns the assistant's persona name used for trigger word matching."""
        return self._assistant_name

    @assistant_name.setter
    def assistant_name(self, value: str) -> None:
        self._assistant_name = value.strip() if value else "Waffle"

    def is_trigger_word_present(self, text: str) -> bool:
        """Check if 'Hey <ASSISTANT_NAME>', an active follow-up, or a tool call is spoken."""
        if not text:
            return False

        # 1. Explicit tool invocation directive (e.g. "tool call", "tool-call", "use tool")
        if re.search(
            r"\b(?:tool[\s_-]*calls?|use\s+(?:a\s+|the\s+)?tool|call\s+(?:a\s+|the\s+)?tool)\b",
            text,
            re.IGNORECASE,
        ):
            return True

        # 2. Check if within active conversation follow-up grace window
        if (
            self._last_assistant_turn_time > 0
            and (time.monotonic() - self._last_assistant_turn_time) <= self._follow_up_window_seconds
        ):
            return True

        # 3. Check for wake phrase 'Hey <ASSISTANT_NAME>'
        if not self._assistant_name:
            return False
        clean_name = self._assistant_name.strip()
        escaped_name = re.sub(r"\s+", r"\\s+", re.escape(clean_name))
        pattern = rf"\bhey\b[,\s\-\.!?:;]*{escaped_name}\b"
        return bool(re.search(pattern, text, re.IGNORECASE))

    @property
    def compactor(self) -> ContextCompactor:
        """Returns the active ContextCompactor instance."""
        return self._compactor

    def _trigger_compaction_if_needed(self) -> None:
        """Evaluate history and launch background context compaction on a separate thread if threshold is met."""
        with self._state_lock:
            history_snapshot = list(self._history)

        if self._compactor.should_compact(history_snapshot):
            self._compactor.compact_async(history_snapshot, self._on_compaction_completed)

    def _on_compaction_completed(
        self, compacted_history: list[Message], original_snapshot_len: int
    ) -> None:
        """Callback invoked by the background compactor thread when compaction finishes."""
        with self._state_lock:
            # Preserve any new turns that were appended during the background compaction
            newly_arrived = (
                self._history[original_snapshot_len:]
                if len(self._history) > original_snapshot_len
                else []
            )
            self._history = compacted_history + newly_arrived
            logger.info(
                "ConversationManager history compacted: now %d message(s) (preserved %d newly arrived).",
                len(self._history),
                len(newly_arrived),
            )

    def _prune_history(self) -> None:
        """Maintain a strict sliding context window (last N turns) to keep TTFT low."""
        max_messages = self._max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def _trigger_llm(self, text: str) -> None:
        """Called by SentenceGate once the utterance looks complete."""
        if self._trigger_mode and not self.is_trigger_word_present(text):
            logger.info(
                "Trigger mode active: ignoring utterance without trigger word 'Hey %s': %s",
                self._assistant_name,
                text,
            )
            if self._on_user_turn_ignored is not None:
                try:
                    self._on_user_turn_ignored(text)
                except Exception as exc:
                    logger.warning("Error in on_user_turn_ignored callback: %s", exc)
            return

        # Yield all resources to the active user turn by aborting background compaction
        self._compactor.abort()
        self._tool_args.clear()
        self._tool_notes.clear()

        self._history.append(
            Message(
                role="user",
                content=text,
            )
        )
        self._prune_history()

        logger.info(
            "[CONVERSATION] Processing user turn (history len=%d, text len=%d): '%s'",
            len(self._history),
            len(text),
            text,
        )

        if self._on_user_turn is not None:
            try:
                self._on_user_turn(text)
            except Exception as exc:
                logger.warning("Error in on_user_turn callback: %s", exc)


        logger.debug(
            "Conversation history now contains %d message(s).",
            len(self._history),
        )

        with self._state_lock:
            self._pending_generation = True
            self._partial_reply = ""
        try:
            self._llm.generate(self._history)
        except queue.Full:
            with self._state_lock:
                self._pending_generation = False
            logger.warning(
                "LLM queue is full — dropping utterance: %s", text
            )

    # ------------------------------------------------------------------
    # LLM Event Handlers & Adaptive Clause Gating
    # ------------------------------------------------------------------

    @property
    def _spoken_chars(self) -> int:
        return self._chunker.spoken_chars

    @_spoken_chars.setter
    def _spoken_chars(self, val: int) -> None:
        self._chunker._cursor = val

    def on_generation_started(self) -> None:
        logger.debug("LLM generation started.")
        self._chunker.reset()

    def on_generation_updated(self, text: str) -> None:
        logger.debug("LLM generation updated.")

        with self._state_lock:
            if not self._pending_generation:
                return
            self._partial_reply = text

        for chunk in self._chunker.process_stream(text):
            self._speak(chunk)

    def _speak(self, text: str) -> None:
        try:
            self._tts.speak(text)
        except Exception:
            logger.exception("Could not queue speech.")

    def on_tool_started(
        self, tool_name: str, arguments: dict, phrase: str | None = None
    ) -> None:
        self._tool_args[tool_name] = arguments

    def on_tool_completed(
        self,
        tool_name: str,
        result: str,
        duration: float = 0.0,
        fallback: bool = False,
        server: str = "mcp",
    ) -> None:
        args = self._tool_args.pop(tool_name, {})
        excerpt = re.sub(r"\s+", " ", str(result or "")).strip()[:500]
        self._tool_notes.append(f"Used {tool_name} with {args}. Result: {excerpt}")

    def on_generation_completed(self, text: str) -> None:
        with self._state_lock:
            if not self._pending_generation:
                logger.debug(
                    "Ignoring LLM completion not initiated by ConversationManager."
                )
                return
            self._pending_generation = False

        logger.info("LLM generation completed: %s", text)

        remainder = self._chunker.finalize(text)
        if remainder:
            self._speak(remainder)

        history_text = text
        if self._tool_notes:
            history_text = f"(Tool context: {' | '.join(self._tool_notes)})\n{text}"
            self._tool_notes.clear()
        self._history.append(
            Message(
                role="assistant",
                content=history_text,
            )
        )
        self._prune_history()
        self._last_assistant_turn_time = time.monotonic()
        self._trigger_compaction_if_needed()

        logger.debug(
            "Conversation history now contains %d message(s).",
            len(self._history),
        )

    def barge_in(self) -> None:
        """Instantly interrupt assistant speech and ongoing LLM generation."""
        logger.info("Barge-in: interrupting LLM generation and flushing TTS/audio.")
        self._compactor.abort()
        with self._state_lock:
            was_pending = self._pending_generation
            self._pending_generation = False
            partial = self._partial_reply.strip()
            self._partial_reply = ""
        self._chunker.reset()

        # Keep user/assistant alternation in the LLM context: record the part
        # of the reply that was generated before the interruption.
        if was_pending and partial:
            self._history.append(Message(role="assistant", content=partial))
            self._prune_history()

        # Abort LLM stream if supported
        if hasattr(self._llm, "abort"):
            try:
                self._llm.abort()
            except Exception:
                logger.exception("Error aborting LLM generation on barge-in.")

        # Interrupt TTS queue if supported
        if hasattr(self._tts, "interrupt"):
            try:
                self._tts.interrupt()
            except Exception:
                logger.exception("Error interrupting TTS on barge-in.")