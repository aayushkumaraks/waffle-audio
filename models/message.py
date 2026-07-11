from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: str