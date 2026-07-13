from abc import ABC, abstractmethod
from typing import Sequence
import queue
import threading
import logging

logger = logging.getLogger(__name__)

from models import Message

from .llm_provider import LLMCallbacks, LLMProvider

_STOP = object()


class LLMListener(ABC):
    """Receives generation events from the LLM service."""

    @abstractmethod
    def on_generation_started(self) -> None:
        ...

    @abstractmethod
    def on_generation_updated(self, text: str) -> None:
        ...

    @abstractmethod
    def on_generation_completed(self, text: str) -> None:
        ...


class LLMServiceError(Exception):
    """Base exception for all LLM service errors."""


class ServiceAlreadyStarted(LLMServiceError):
    """Raised when attempting to start an already running service."""


class ServiceNotStarted(LLMServiceError):
    """Raised when attempting to use or stop a service that has not been started."""


class LLMService(LLMCallbacks):
    """Provider-agnostic streaming LLM service."""

    def __init__(
        self,
        provider: LLMProvider,
    ) -> None:
        self._provider = provider

        self._listeners: list[LLMListener] = []

        self._generation_queue: queue.Queue[Sequence[Message]] = queue.Queue(
            maxsize=1,
        )

        self._worker_thread: threading.Thread | None = None

        self._shutdown = threading.Event()

        self._started = False

    def start(self) -> None:
        """Start the LLM service."""

        if self._started:
            raise ServiceAlreadyStarted("LLMService is already started.")

        self._provider.start()
        self._shutdown.clear()

        self._worker_thread = threading.Thread(
            target=self._process_generation_queue,
            name="llm-worker",
            daemon=True,
        )

        self._worker_thread.start()

        self._started = True

    def stop(self) -> None:
        """Stop the LLM service."""

        if not self._started:
            raise ServiceNotStarted("LLMService is not started.")
        
        self._shutdown.set()

        self._generation_queue.put_nowait(_STOP)

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5)

        self._worker_thread = None

        self._provider.stop()

        self._started = False

    def generate(self, messages: Sequence[Message]) -> None:
        if not self._started:
            raise ServiceNotStarted("LLMService is not started.")

        self._generation_queue.put_nowait(list(messages))

    def add_listener(self, listener: LLMListener) -> None:
        """Register a listener."""

        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: LLMListener) -> None:
        """Remove a registered listener."""

        if listener in self._listeners:
            self._listeners.remove(listener)

    def on_generation_started(self) -> None:
        """Dispatch generation started."""

        for listener in tuple(self._listeners):
            listener.on_generation_started()

    def on_generation_updated(self, text: str) -> None:
        """Dispatch generation updates."""

        for listener in tuple(self._listeners):
            listener.on_generation_updated(text)

    def on_generation_completed(self, text: str) -> None:
        """Dispatch generation completed."""

        for listener in tuple(self._listeners):
            listener.on_generation_completed(text)

    def _process_generation_queue(self):
        while True:
            item = self._generation_queue.get()

            if item is _STOP:
                break

            try:
                self._provider.generate(
                    messages=item,
                    callbacks=self,
                )
            except Exception:
                logger.exception("LLM generation failed.")