"""Emotion-aware conversation manager coordinating STT, LLM, and EmotionTTSService."""
from __future__ import annotations

import logging
import queue
import re
import threading
from typing import Callable

from constants import SENTENCE_GATE_GAP_TIMEOUT, SENTENCE_GATE_QUICK_FIRE_TIMEOUT
from models import Message
from services.conversation.src.emotion_parser import (
    extract_turn_emotion,
    parse_sentence_emotion,
    strip_emotion_tags,
)
from services.conversation.src.sentence_gate import SentenceGate
from services.llm.src import LLMListener, LLMService
from services.stt.src import STTListener, STTService
from services.tts.src import EmotionTTSService

logger = logging.getLogger(__name__)


class _EmotionSTTAdapter(STTListener):
    def __init__(self, manager: "EmotionConversationManager") -> None:
        self._manager = manager

    def on_transcript_started(self, text: str) -> None:
        self._manager._handle_transcript_started(text)

    def on_transcript_updated(self, text: str) -> None:
        self._manager._handle_transcript_updated(text)

    def on_transcript_completed(self, text: str) -> None:
        self._manager._handle_transcript_completed(text)


class EmotionConversationManager(LLMListener):
    """Coordinates conversation turns with real-time emotion parsing and voice switching."""

    # Sentence boundary: . ! ? followed by whitespace
    _SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s")
    _MIN_SPEECH_CHARS = 20

    def __init__(
        self,
        stt: STTService,
        llm: LLMService,
        tts: EmotionTTSService,
        on_user_turn: Callable[[str], None] | None = None,
        on_emotion_change: Callable[[str], None] | None = None,
        gap_timeout: float = SENTENCE_GATE_GAP_TIMEOUT,
        quick_fire_timeout: float = SENTENCE_GATE_QUICK_FIRE_TIMEOUT,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._on_user_turn = on_user_turn
        self._on_emotion_change = on_emotion_change

        self._stt_listener = _EmotionSTTAdapter(self)
        self._history: list[Message] = []
        self._gate = SentenceGate(
            callback=self._trigger_llm,
            gap_timeout=gap_timeout,
            quick_fire_timeout=quick_fire_timeout,
        )

        self._started = False
        self._pending_generation = False
        self._spoken_chars = 0
        self._active_emotion: str = "neutral"
        self._state_lock = threading.Lock()

    @property
    def active_emotion(self) -> str:
        return self._active_emotion

    def start(self) -> None:
        if self._started:
            return
        self._stt.add_listener(self._stt_listener)
        self._llm.add_listener(self)
        self._started = True
        logger.info("EmotionConversationManager started.")

    def stop(self) -> None:
        if not self._started:
            return
        self._gate.cancel()
        self._stt.remove_listener(self._stt_listener)
        self._llm.remove_listener(self)
        self._started = False
        logger.info("EmotionConversationManager stopped.")

    def reset(self) -> None:
        self._history.clear()
        self._active_emotion = "neutral"
        logger.info("Emotion conversation history cleared.")

    @property
    def history(self) -> list[Message]:
        return self._history.copy()

    def submit_user_text(self, text: str) -> None:
        text = text.strip()
        if text:
            self._trigger_llm(text)

    # ------------------------------------------------------------------
    # STT Event Handlers
    # ------------------------------------------------------------------

    def _handle_transcript_started(self, text: str) -> None:
        self._gate.notify_speech_active()

    def _handle_transcript_updated(self, text: str) -> None:
        self._gate.notify_speech_active()

    def _handle_transcript_completed(self, text: str) -> None:
        text = text.strip()
        if text:
            self._gate.submit(text)

    def _trigger_llm(self, text: str) -> None:
        self._history.append(Message(role="user", content=text))
        if self._on_user_turn is not None:
            try:
                self._on_user_turn(text)
            except Exception as exc:
                logger.warning("Error in on_user_turn: %s", exc)

        try:
            self._llm.generate(self._history)
            with self._state_lock:
                self._pending_generation = True
        except queue.Full:
            logger.warning("LLM queue full, dropping utterance: %s", text)

    # ------------------------------------------------------------------
    # LLM Event Handlers with Emotion Parsing
    # ------------------------------------------------------------------

    def on_generation_started(self) -> None:
        self._spoken_chars = 0
        self._active_emotion = "neutral"
        if self._on_emotion_change:
            self._on_emotion_change("neutral")

    def on_generation_updated(self, text: str) -> None:
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
        self._process_chunk(chunk)

    def on_generation_completed(self, text: str) -> None:
        with self._state_lock:
            if not self._pending_generation:
                return
            self._pending_generation = False

        remainder = text[self._spoken_chars:].strip()
        self._spoken_chars = 0
        if remainder:
            self._process_chunk(remainder)

        # Store clean text in history for natural continuation
        clean_full_text = strip_emotion_tags(text)
        self._history.append(Message(role="assistant", content=clean_full_text))

    def _process_chunk(self, chunk: str) -> None:
        new_emotion, clean_text = parse_sentence_emotion(chunk, current_emotion=self._active_emotion)
        if new_emotion != self._active_emotion:
            self._active_emotion = new_emotion
            if self._on_emotion_change:
                try:
                    self._on_emotion_change(new_emotion)
                except Exception:
                    pass

        if clean_text:
            try:
                self._tts.speak(clean_text, emotion=self._active_emotion)
            except Exception:
                logger.exception("Could not queue emotional speech.")
