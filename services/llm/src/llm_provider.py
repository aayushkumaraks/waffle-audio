import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from models import Message
from constants import OLLAMA_BASE_URL, OLLAMA_KEEP_ALIVE, OLLAMA_MODEL, OLLAMA_NUM_PREDICT

logger = logging.getLogger(__name__)


class LLMCallbacks(ABC):
    """Receives streaming events from an LLM provider."""

    @abstractmethod
    def on_generation_started(self) -> None:
        ...

    @abstractmethod
    def on_generation_updated(self, text: str) -> None:
        ...

    @abstractmethod
    def on_generation_completed(self, text: str) -> None:
        ...


@dataclass(slots=True)
class LLMProviderConfig:
    """Configuration for the local Ollama provider."""

    base_url: str = OLLAMA_BASE_URL
    model: str = OLLAMA_MODEL


class LLMProviderError(Exception):
    """Raised when the LLM provider encounters an error."""


class LLMProvider:
    """Communicates with the local Ollama server."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self._config = config

        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=httpx.Timeout(60.0),
        )
        self._abort_event = threading.Event()

    def abort(self) -> None:
        """Signal immediate abort of active LLM streaming."""
        self._abort_event.set()


    def start(self) -> None:
        """Verify that the Ollama server is reachable."""

        try:
            response = self._client.get("/api/version")
            response.raise_for_status()

        except httpx.HTTPError as exc:
            raise LLMProviderError(
                f"Unable to connect to Ollama at {self._config.base_url}."
            ) from exc

        # Load the model into memory now (in the background) so the first
        # user turn does not pay the cold-start cost.
        threading.Thread(
            target=self._warm_up, name="llm-warmup", daemon=True
        ).start()

    def _warm_up(self) -> None:
        try:
            self._client.post(
                "/api/chat",
                json={
                    "model": self._config.model,
                    "messages": [],
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                },
                timeout=httpx.Timeout(120.0),
            )
            logger.info("LLM model warmed up.")
        except httpx.HTTPError:
            logger.debug("LLM warm-up failed.", exc_info=True)

    def stop(self) -> None:
        """Release provider resources."""
        self.abort()
        self._client.close()

    def generate(
        self,
        messages: Sequence[Message],
        callbacks: LLMCallbacks,
    ) -> None:
        """Generate a streamed response with instant abort support."""
        self._abort_event.clear()
        payload = self._build_request(messages)

        try:
            with self._client.stream(
                "POST",
                "/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()

                generation_started = False
                generated_text = ""

                for line in response.iter_lines():
                    if self._abort_event.is_set():
                        logger.info("LLM generation aborted by signal.")
                        break

                    if not line:
                        continue

                    chunk = json.loads(line)

                    content = (
                        chunk.get("message", {})
                        .get("content", "")
                    )

                    if not content:
                        continue

                    if not generation_started:
                        callbacks.on_generation_started()
                        generation_started = True

                    generated_text += content

                    callbacks.on_generation_updated(generated_text)

                if generation_started and not self._abort_event.is_set():
                    callbacks.on_generation_completed(generated_text)

        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            if not self._abort_event.is_set():
                raise LLMProviderError(
                    "Failed to generate response."
                ) from exc

    _SYSTEM_PROMPT = (
        "You are a helpful voice assistant named Waffle. You speak in a natural, calm, and friendly conversational tone.\n"
        "Your responses are read aloud by a text-to-speech engine (Kokoro/Pocket TTS), so strictly adhere to these speech rules:\n"
        "1. Reply in 2-3 short, spoken sentences only.\n"
        "2. Output plain spoken English words only. NEVER output markdown tables, bullet points, headers, emojis, URLs, or code blocks.\n"
        "3. NEVER include asterisked or bracketed actions or expressive tags (e.g. do NOT write *laughs*, *smiles*, [sighs], or *chuckles*).\n"
        "4. Spell out numbers, measurements, and symbols in full words (e.g., 'three kilometers' not '3km', 'percent' not '%').\n"
        "5. Use only standard conversational punctuation (periods, commas, question marks). Do not use asterisks, hashes, backticks, or dashes."
    )

    def _build_request(
        self,
        messages: Sequence[Message],
    ) -> dict[str, Any]:
        """Build an Ollama chat request with sliding window guard."""
        # Enforce sliding window: retain at most the last 16 messages (8 turns)
        sliding_messages = messages[-16:] if len(messages) > 16 else messages

        system_message = {"role": "system", "content": self._SYSTEM_PROMPT}

        return {
            "model": self._config.model,
            "messages": [system_message] + [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in sliding_messages
            ],
            "stream": True,
            "think": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"num_predict": OLLAMA_NUM_PREDICT},
        }