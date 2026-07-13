from __future__ import annotations

import logging
from models import Message

from services.stt.src import STTListener, STTService
from services.llm.src import LLMListener, LLMService
from services.tts.src import TTSService

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
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts

        self._stt_listener = _STTListenerAdapter(self)

        self._history: list[Message] = []

        self._started = False

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

    # ------------------------------------------------------------------
    # STT Event Handlers
    # ------------------------------------------------------------------

    def _handle_transcript_started(self, text: str) -> None:
        logger.debug("Speech started: %s", text)

    def _handle_transcript_updated(self, text: str) -> None:
        logger.debug("Transcript updated: %s", text)

    def _handle_transcript_completed(self, text: str) -> None:
        logger.info("Transcript completed: %s", text)

        text = text.strip()

        if not text:
            logger.debug("Ignoring empty transcript.")
            return

        self._history.append(
            Message(
                role="user",
                content=text,
            )
        )

        logger.debug(
            "Conversation history now contains %d message(s).",
            len(self._history),
        )

        self._llm.generate(self._history)

    # ------------------------------------------------------------------
    # LLM Event Handlers
    # ------------------------------------------------------------------

    def on_generation_started(self) -> None:
        logger.debug("LLM generation started.")

    def on_generation_updated(self, text: str) -> None:
        logger.debug("LLM generation updated.")

    def on_generation_completed(self, text: str) -> None:
        logger.info("LLM generation completed.")

        self._tts.speak(text)

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