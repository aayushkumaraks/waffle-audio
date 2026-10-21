from dataclasses import dataclass
from typing import Any, Literal


@dataclass(slots=True)
class Message:
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_calls: list[dict[str, Any]] | None = None
