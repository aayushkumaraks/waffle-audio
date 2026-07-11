from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from services.stt.src import STTListener, STTService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Message:
    """Represents a single message in the conversation."""

    role: Literal["user", "assistant"]
    content: str


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


class ConversationManager:
    """Coordinates the conversation between application services."""

    def __init__(self, stt: STTService) -> None:
        self._stt = stt

        self._stt_listener = _STTListenerAdapter(self)

        self._history: list[Message] = []

        self._started = False

    def start(self) -> None:
        """Start listening for conversation events."""

        if self._started:
            return

        self._stt.add_listener(self._stt_listener)
        self._started = True

        logger.info("ConversationManager started.")

    def stop(self) -> None:
        """Stop listening for conversation events."""

        if not self._started:
            return

        self._stt.remove_listener(self._stt_listener)
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