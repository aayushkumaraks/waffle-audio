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
from .sentence_gate import SentenceGate

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
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._on_user_turn = on_user_turn

        self._stt_listener = _STTListenerAdapter(self)

        self._history: list[Message] = []

        self._gate = SentenceGate(
            callback=self._trigger_llm,
            gap_timeout=gap_timeout,
            quick_fire_timeout=quick_fire_timeout,
        )

        self._started = False
        self._pending_generation = False
        self._spoken_chars = 0
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

    def _trigger_llm(self, text: str) -> None:
        """Called by SentenceGate once the utterance looks complete."""
        self._history.append(
            Message(
                role="user",
                content=text,
            )
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

        try:
            self._llm.generate(self._history)
            with self._state_lock:
                self._pending_generation = True
        except queue.Full:
            logger.warning(
                "LLM queue is full — dropping utterance: %s", text
            )

    # ------------------------------------------------------------------
    # LLM Event Handlers
    # ------------------------------------------------------------------

    # A sentence ends at . ! or ? (plus optional closing quotes) followed by
    # whitespace; requiring the following space avoids splitting on decimals.
    _SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s")
    # Avoid sending tiny fragments (e.g. "Sure.") to TTS on their own.
    _MIN_SPEECH_CHARS = 30

    def on_generation_started(self) -> None:
        logger.debug("LLM generation started.")
        self._spoken_chars = 0

    def on_generation_updated(self, text: str) -> None:
        logger.debug("LLM generation updated.")

        with self._state_lock:
            if not self._pending_generation:
                return

        boundary = None
        for match in self._SENTENCE_END.finditer(text, self._spoken_chars):
            boundary = match.end()

        if boundary is None:
            return

        chunk = text[self._spoken_chars:boundary].strip()
        if len(chunk) < self._MIN_SPEECH_CHARS:
            return

        self._spoken_chars = boundary
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

        remainder = text[self._spoken_chars:].strip()
        self._spoken_chars = 0
        if remainder:
            self._speak(remainder)

        self._history.append(
            Message(
                role="assistant",
                content=text,
            )
        )

        logger.debug(
            "Conversation history now contains %d message(s).",
            len(self._history),
        )