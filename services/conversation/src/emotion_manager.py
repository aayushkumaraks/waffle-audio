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
    extract_latest_emotion,
    split_by_emotion,
    strip_emotion_tags,
)
from services.conversation.src.sentence_gate import ClauseChunker, SentenceGate
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

    def __init__(
        self,
        stt: STTService,
        llm: LLMService,
        tts: EmotionTTSService,
        on_user_turn: Callable[[str], None] | None = None,
        on_emotion_change: Callable[[str], None] | None = None,
        gap_timeout: float = SENTENCE_GATE_GAP_TIMEOUT,
        quick_fire_timeout: float = SENTENCE_GATE_QUICK_FIRE_TIMEOUT,
        max_history_turns: int = 8,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._on_user_turn = on_user_turn
        self._on_emotion_change = on_emotion_change
        self._max_history_turns = max_history_turns

        self._stt_listener = _EmotionSTTAdapter(self)
        self._history: list[Message] = []
        # Keep [emotion] tags in the internal LLM history so small models keep emitting tags
        self._llm_history: list[Message] = []
        self._gate = SentenceGate(
            callback=self._trigger_llm,
            gap_timeout=gap_timeout,
            quick_fire_timeout=quick_fire_timeout,
        )
        self._chunker = ClauseChunker()

        self._started = False
        self._pending_generation = False
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
        self._llm_history.clear()
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

    def _prune_history(self) -> None:
        max_messages = self._max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
        if len(self._llm_history) > max_messages:
            self._llm_history = self._llm_history[-max_messages:]

    def _trigger_llm(self, text: str) -> None:
        user_msg = Message(role="user", content=text)
        self._history.append(user_msg)
        self._llm_history.append(user_msg)
        self._prune_history()
        if self._on_user_turn is not None:
            try:
                self._on_user_turn(text)
            except Exception as exc:
                logger.warning("Error in on_user_turn: %s", exc)

        # Set _pending_generation before enqueueing to avoid racing with fast LLM callbacks
        with self._state_lock:
            self._pending_generation = True
        try:
            self._llm.generate(list(self._llm_history))
        except queue.Full:
            with self._state_lock:
                self._pending_generation = False
            logger.warning("LLM queue full, dropping utterance: %s", text)

    # ------------------------------------------------------------------
    # LLM Event Handlers with Emotion Parsing
    # ------------------------------------------------------------------

    def on_generation_started(self) -> None:
        self._chunker.reset()
        self._active_emotion = "neutral"
        if self._on_emotion_change:
            try:
                self._on_emotion_change("neutral")
            except Exception:
                pass

    def on_generation_updated(self, text: str) -> None:
        with self._state_lock:
            if not self._pending_generation:
                return

        for chunk in self._chunker.process_stream(text):
            self._process_chunk(chunk)

    def on_generation_completed(self, text: str) -> None:
        with self._state_lock:
            if not self._pending_generation:
                return
            self._pending_generation = False

        remainder = self._chunker.finalize(text)
        if remainder:
            self._process_chunk(remainder)

        logger.info("[EMOTION LLM RAW] %r", text)

        # Store clean text in public history, and canonical-tagged text in _llm_history
        # so small models never drift into omitting or inventing non-canonical [emotion] tags.
        clean_full_text = strip_emotion_tags(text)
        segments = split_by_emotion(text.strip(), current_emotion="neutral")
        if segments:
            tagged_text = " ".join(f"[{emo}] {seg}" for emo, seg in segments)
        else:
            tagged_text = f"[{self._active_emotion}] {clean_full_text}" if clean_full_text else ""

        self._history.append(Message(role="assistant", content=clean_full_text))
        if tagged_text:
            self._llm_history.append(Message(role="assistant", content=tagged_text))
        self._prune_history()

    def on_generation_failed(self, error: str) -> None:
        with self._state_lock:
            self._pending_generation = False
        self._chunker.reset()
        logger.warning("EmotionConversationManager recovered from LLM failure: %s", error)

    def _process_chunk(self, chunk: str) -> None:
        # Split on emotion tags so multi-sentence chunks with mid-chunk emotion switches
        # preserve every emotion transition in order.
        for new_emotion, clean_text in split_by_emotion(chunk, current_emotion=self._active_emotion):
            if new_emotion != self._active_emotion:
                self._active_emotion = new_emotion
                if self._on_emotion_change:
                    try:
                        self._on_emotion_change(new_emotion)
                    except Exception:
                        pass

            if clean_text:
                logger.info("[EMOTION ROUTE] emotion=%s text=%r", self._active_emotion, clean_text)
                try:
                    self._tts.speak(clean_text, emotion=self._active_emotion)
                except Exception:
                    logger.exception("Could not queue emotional speech.")

    def barge_in(self) -> None:
        """Interrupt active LLM generation and flush pending TTS."""
        with self._state_lock:
            self._pending_generation = False
        self._chunker.reset()
        if hasattr(self._llm, "abort"):
            try:
                self._llm.abort()
            except Exception:
                logger.exception("Error aborting LLM generation on barge-in.")
        if hasattr(self._tts, "interrupt"):
            try:
                self._tts.interrupt()
            except Exception:
                logger.exception("Error interrupting TTS on barge-in.")
