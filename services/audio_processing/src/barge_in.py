from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class BargeInConfig:
    enabled: bool = False
    minimum_speech_ms: int = 120


class BargeInController:
    """Boundary for user interruption of TTS."""

    def __init__(
        self,
        config: BargeInConfig | None = None,
        *,
        on_interrupt: Callable[[], None] | None = None,
    ) -> None:
        self.config = config or BargeInConfig()
        self._on_interrupt = on_interrupt

    def trigger(self) -> None:
        if self.config.enabled and self._on_interrupt is not None:
            self._on_interrupt()
