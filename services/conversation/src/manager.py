from __future__ import annotations

import logging
import re
import queue
import threading

from typing import Callable

from constants import SENTENCE_GATE_GAP_TIMEOUT, SENTENCE_GATE_QUICK_FIRE_TIMEOUT
from models import Message

from services.stt.src import STTListener, STTService
from services.llm.src import LLMListener, LLMService
from services.tts.src import TTSService
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
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._on_user_turn = on_user_turn
        self._max_history_turns = max_history_turns

        self._stt_listener = _STTListenerAdapter(self)

        self._history: list[Message] = []

        self._gate = SentenceGate(
            callback=self._trigger_llm,
            gap_timeout=gap_timeout,
            quick_fire_timeout=quick_fire_timeout,
        )
        self._chunker = ClauseChunker()
        self._partial_reply = ""

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

    def _prune_history(self) -> None:
        """Maintain a strict sliding context window (last N turns) to keep TTFT low."""
        max_messages = self._max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def _trigger_llm(self, text: str) -> None:
        """Called by SentenceGate once the utterance looks complete."""
        self._history.append(
            Message(
                role="user",
                content=text,
            )
        )
        self._prune_history()

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

        self._history.append(
            Message(
                role="assistant",
                content=text,
            )
        )
        self._prune_history()

        logger.debug(
            "Conversation history now contains %d message(s).",
            len(self._history),
        )

    def barge_in(self) -> None:
        """Instantly interrupt assistant speech and ongoing LLM generation."""
        logger.info("Barge-in: interrupting LLM generation and flushing TTS/audio.")
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