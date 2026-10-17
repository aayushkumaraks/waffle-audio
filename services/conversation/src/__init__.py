"""Public exports for the conversation manager package."""

from .manager import ConversationManager
from .sentence_gate import ClauseChunker, SentenceGate
from .ui import ConversationUI

__all__ = ["ClauseChunker", "ConversationManager", "SentenceGate", "ConversationUI"]

