"""Public exports for the conversation manager package."""

from .compactor import ContextCompactor
from .manager import ConversationManager
from .sentence_gate import ClauseChunker, SentenceGate
from .ui import ConversationUI

__all__ = [
    "ClauseChunker",
    "ContextCompactor",
    "ConversationManager",
    "SentenceGate",
    "ConversationUI",
]

