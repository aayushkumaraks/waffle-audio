# ---------------------------------------------------------------------------
# Static application constants
#
# All hard-coded URLs, model identifiers, file paths, and default values
# live here so they can be changed in one place.
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

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = "http://172.19.96.1:11434"
OLLAMA_MODEL: str = "qwen3.5:4b"

# ---------------------------------------------------------------------------
# Kokoro TTS -- local model files
# ---------------------------------------------------------------------------

TTS_MODEL_PATH: str = "voiceModels/kokoro-v1.0.onnx"
TTS_VOICES_PATH: str = "voiceModels/voices-v1.0.bin"

# ---------------------------------------------------------------------------
# Kokoro TTS -- download URLs (used by standalone test/setup scripts)
# ---------------------------------------------------------------------------

KOKORO_MODEL_DOWNLOAD_URL: str = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/latest/download/"
    "kokoro-v1.0.onnx"
)
KOKORO_VOICES_DOWNLOAD_URL: str = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/latest/download/"
    "voices-v1.0.bin"
)


# ---------------------------------------------------------------------------
# TTS backend selection
# ---------------------------------------------------------------------------

TTS_BACKEND: str = os.getenv("VOICECHAT_TTS_BACKEND", "pocket").lower()
# TTS_BACKEND: str = os.getenv("VOICECHAT_TTS_BACKEND", "kokoro").lower()
KOKORO_MODEL_PATH: str = "voiceModels/kokoro-v1.0.onnx"
KOKORO_VOICES_PATH: str = "voiceModels/voices-v1.0.bin"
KOKORO_DEFAULT_VOICE: str = "af_sarah"
KOKORO_DEFAULT_LANGUAGE: str = "en-us"
POCKET_TTS_DEFAULT_VOICE: str = "anna"
POCKET_TTS_DEFAULT_LANGUAGE: str = "english"
POCKET_TTS_CACHE_DIR: str = os.getenv(
    "VOICECHAT_POCKET_TTS_CACHE_DIR", "voiceModels/pocket-tts-cache"
)
TTS_QUANTIZE: bool = os.getenv("VOICECHAT_TTS_QUANTIZE", "false").lower() in {
    "1", "true", "yes", "on"
}

# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

API_HOST: str = os.getenv("VOICECHAT_API_HOST", "0.0.0.0")
API_PORT: int = _int_from_env("VOICECHAT_API_PORT", 3000)

# Browser origins allowed to call the API during local development and when
# the SPA is served separately. Same-origin production (/ui) does not require
# a CORS entry. Override this with a comma-separated list in production.
API_CORS_ALLOW_ORIGINS: str = os.getenv(
    "VOICECHAT_CORS_ALLOW_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:4173,http://127.0.0.1:4173,"
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:8000,http://127.0.0.1:8000",
)
