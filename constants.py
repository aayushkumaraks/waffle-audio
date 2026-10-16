# ---------------------------------------------------------------------------
# Static application constants
#
# All hard-coded URLs, model identifiers, file paths, and default values
# live here so they can be changed in one place.
# ---------------------------------------------------------------------------

import os
from pathlib import Path


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

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://172.19.96.1:11434") # WSL to HOST
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
# Keep the model resident in memory so turns never pay a cold-load penalty.
OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
# Hard cap on reply length (replies are spoken aloud, 2-3 short sentences).
OLLAMA_NUM_PREDICT: int = _int_from_env("OLLAMA_NUM_PREDICT", 160)

# ---------------------------------------------------------------------------
# Sentence Gate (silence interval before LLM invocation)
# ---------------------------------------------------------------------------

SENTENCE_GATE_GAP_TIMEOUT: float = float(os.getenv("VOICECHAT_GAP_TIMEOUT", "2.0"))
SENTENCE_GATE_QUICK_FIRE_TIMEOUT: float = float(os.getenv("VOICECHAT_QUICK_FIRE_TIMEOUT", "0.8"))

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
CLONED_VOICES_DIR: Path = Path(os.getenv("VOICECHAT_CLONED_VOICES_DIR", "cloned_voices"))
TTS_QUANTIZE: bool = os.getenv("VOICECHAT_TTS_QUANTIZE", "false").lower() in {
    "1", "true", "yes", "on"
}


def list_cloned_voices(directory: Path | str | None = None) -> list[str]:
    """Return a sorted list of cloned voice names found in the directory."""
    target_dir = Path(directory) if directory is not None else CLONED_VOICES_DIR
    if not target_dir.is_dir():
        return []
    return sorted(
        p.stem
        for p in target_dir.glob("*.safetensors")
        if p.is_file() and not p.name.startswith(".")
    )


def resolve_voice_spec(voice: str | None, default: str = POCKET_TTS_DEFAULT_VOICE) -> str:
    """Resolve a voice name or file path for TTS engines.

    If given a name matching an existing cloned voice in CLONED_VOICES_DIR or an
    existing file path, returns the resolved file path string. Otherwise returns
    the voice name/identifier.
    """
    v = (voice or "").strip()
    if not v:
        v = default

    direct_path = Path(v)
    if direct_path.is_file():
        return str(direct_path.resolve())

    safetensors_path = CLONED_VOICES_DIR / f"{v}.safetensors"
    if safetensors_path.is_file():
        return str(safetensors_path.resolve())

    named_file = CLONED_VOICES_DIR / v
    if named_file.is_file():
        return str(named_file.resolve())

    return v

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
