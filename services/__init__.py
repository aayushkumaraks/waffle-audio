"""Public exports for all services with lazy loading for fast startup and low overhead."""

import importlib
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Audio
    "AudioPlayer": (".audio.src", "AudioPlayer"),
    # STT
    "AudioQueueFull": (".stt.src", "AudioQueueFull"),
    "STTConfig": (".stt.src", "STTConfig"),
    "STTListener": (".stt.src", "STTListener"),
    "STTService": (".stt.src", "STTService"),
    "STTServiceError": (".stt.src", "STTServiceError"),
    # LLM
    "LLMCallbacks": (".llm.src", "LLMCallbacks"),
    "LLMListener": (".llm.src", "LLMListener"),
    "LLMProvider": (".llm.src", "LLMProvider"),
    "LLMProviderConfig": (".llm.src", "LLMProviderConfig"),
    "LLMProviderError": (".llm.src", "LLMProviderError"),
    "LLMService": (".llm.src", "LLMService"),
    "LLMServiceError": (".llm.src", "LLMServiceError"),
    # TTS
    "SpeechQueueFull": (".tts.src", "SpeechQueueFull"),
    "TTSConfig": (".tts.src", "TTSConfig"),
    "TTSListener": (".tts.src", "TTSListener"),
    "TTSService": (".tts.src", "TTSService"),
    "TTSServiceError": (".tts.src", "TTSServiceError"),
    # Conversation
    "ConversationManager": (".conversation.src", "ConversationManager"),
    # Unified Pipeline & IPC
    "UnifiedPipeline": (".pipeline", "UnifiedPipeline"),
    "SharedAudioBuffer": (".shared_memory", "SharedAudioBuffer"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path, __name__)
        val = getattr(module, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

