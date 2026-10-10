# ---------------------------------------------------------------------------
# Static application constants
# ---------------------------------------------------------------------------

import os


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc

# Ollama / LLM
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://172.19.96.1:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")

# Kokoro backend is retained for existing users and local model files.
TTS_MODEL_PATH: str = "voiceModels/kokoro-v1.0.onnx"
TTS_VOICES_PATH: str = "voiceModels/voices-v1.0.bin"
KOKORO_MODEL_DOWNLOAD_URL: str = "https://github.com/thewh1teagle/kokoro-onnx/releases/latest/download/kokoro-v1.0.onnx"
KOKORO_VOICES_DOWNLOAD_URL: str = "https://github.com/thewh1teagle/kokoro-onnx/releases/latest/download/voices-v1.0.bin"

# TTS backend selection
TTS_BACKEND: str = os.getenv("VOICECHAT_TTS_BACKEND", "pocket").lower()
KOKORO_MODEL_PATH: str = "voiceModels/kokoro-v1.0.onnx"
KOKORO_VOICES_PATH: str = "voiceModels/voices-v1.0.bin"
KOKORO_DEFAULT_VOICE: str = "af_sarah"
KOKORO_DEFAULT_LANGUAGE: str = "en-us"
POCKET_TTS_DEFAULT_VOICE: str = os.getenv("VOICECHAT_TTS_VOICE", "alba")
POCKET_TTS_DEFAULT_LANGUAGE: str = os.getenv("VOICECHAT_TTS_LANGUAGE", "english")
POCKET_TTS_CACHE_DIR: str = os.getenv("VOICECHAT_POCKET_TTS_CACHE_DIR", "voiceModels/pocket-tts-cache")
TTS_QUANTIZE: bool = os.getenv("VOICECHAT_TTS_QUANTIZE", "false").lower() in {"1", "true", "yes", "on"}

# HTTP API
API_HOST: str = os.getenv("VOICECHAT_API_HOST", "0.0.0.0")
API_PORT: int = _int_from_env("VOICECHAT_API_PORT", 3000)
API_CORS_ALLOW_ORIGINS: str = os.getenv("VOICECHAT_CORS_ALLOW_ORIGINS", "*")
