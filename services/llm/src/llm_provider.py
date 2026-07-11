import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from models import Message


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

    base_url: str = "http://172.19.96.1:11434"
    model: str = "qwen3.5:4b"


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

    def start(self) -> None:
        """Verify that the Ollama server is reachable."""

        try:
            response = self._client.get("/api/version")
            response.raise_for_status()

        except httpx.HTTPError as exc:
            raise LLMProviderError(
                f"Unable to connect to Ollama at {self._config.base_url}."
            ) from exc

    def stop(self) -> None:
        """Release provider resources."""

        self._client.close()

    def generate(
        self,
        messages: Sequence[Message],
        callbacks: LLMCallbacks,
    ) -> None:
        """Generate a streamed response."""

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

                if generation_started:
                    callbacks.on_generation_completed(generated_text)

        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMProviderError(
                "Failed to generate response."
            ) from exc

    def _build_request(
        self,
        messages: Sequence[Message],
    ) -> dict[str, Any]:
        """Build an Ollama chat request."""

        return {
            "model": self._config.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
            "stream": True,
            "think": False,
        }