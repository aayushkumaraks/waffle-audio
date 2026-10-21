# ---------------------------------------------------------------------------
# Static application constants
#
# All hard-coded URLs, model identifiers, file paths, and default values
# live here so they can be changed in one place.
# ---------------------------------------------------------------------------

import os
from pathlib import Path


def _load_env_file(filepath: Path | str = ".env") -> None:
    """Load key-value pairs from a .env file into os.environ if present.

    Preserves any variables already explicitly set in the process environment.
    """
    env_path = Path(filepath)
    if not env_path.is_file():
        return
    try:
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if len(value) >= 2 and (
                        (value[0] == '"' and value[-1] == '"')
                        or (value[0] == "'" and value[-1] == "'")
                    ):
                        value = value[1:-1]
                    if key not in os.environ:
                        os.environ[key] = value
    except Exception:
        pass


# Load .env before defining any constants
_load_env_file(".env")


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc

# ---------------------------------------------------------------------------
# Personalization & User Identity (Loaded from .env or empty defaults)
# ---------------------------------------------------------------------------

USER_NAME: str = os.getenv("USER_NAME", "").strip()
USER_LOCATION: str = os.getenv("USER_LOCATION", "").strip()
USER_TIMEZONE: str = os.getenv("USER_TIMEZONE", "").strip()
ASSISTANT_NAME: str = os.getenv("ASSISTANT_NAME", "Waffle").strip() or "Waffle"

# ---------------------------------------------------------------------------
# Audio Hardware Configuration
# ---------------------------------------------------------------------------

AUDIO_INPUT_DEVICE: int | str | None = os.getenv("AUDIO_INPUT_DEVICE", "").strip() or None
if isinstance(AUDIO_INPUT_DEVICE, str) and AUDIO_INPUT_DEVICE.lower() in {"default", "none", ""}:
    AUDIO_INPUT_DEVICE = None
elif isinstance(AUDIO_INPUT_DEVICE, str) and AUDIO_INPUT_DEVICE.isdigit():
    AUDIO_INPUT_DEVICE = int(AUDIO_INPUT_DEVICE)

AUDIO_OUTPUT_DEVICE: int | str | None = os.getenv("AUDIO_OUTPUT_DEVICE", "").strip() or None
if isinstance(AUDIO_OUTPUT_DEVICE, str) and AUDIO_OUTPUT_DEVICE.lower() in {"default", "none", ""}:
    AUDIO_OUTPUT_DEVICE = None
elif isinstance(AUDIO_OUTPUT_DEVICE, str) and AUDIO_OUTPUT_DEVICE.isdigit():
    AUDIO_OUTPUT_DEVICE = int(AUDIO_OUTPUT_DEVICE)

# ---------------------------------------------------------------------------
# External Tokens & Keys
# ---------------------------------------------------------------------------

GITHUB_PERSONAL_ACCESS_TOKEN: str = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "").strip()

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
# Keep the model resident in memory so turns never pay a cold-load penalty.
OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
# Hard cap on reply length (replies are spoken aloud, 2-3 short sentences).
OLLAMA_NUM_PREDICT: int = _int_from_env("OLLAMA_NUM_PREDICT", 160)
OLLAMA_NUM_CTX: int = _int_from_env("OLLAMA_NUM_CTX", 2048)

# ---------------------------------------------------------------------------
# MCP & Tool Calling
# ---------------------------------------------------------------------------

MCP_CONFIG_PATH: Path = Path(os.getenv("VOICECHAT_MCP_CONFIG", "mcp_servers.json"))
TOOLS_ENABLED: bool = os.getenv("VOICECHAT_TOOLS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
DUCKDUCKGO_SEARCH_ENABLED: bool = os.getenv("VOICECHAT_DDGS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
DUCKDUCKGO_MAX_RESULTS: int = _int_from_env("VOICECHAT_DDGS_MAX_RESULTS", 4)
FILLER_PHRASES_ENABLED: bool = os.getenv("VOICECHAT_FILLER_PHRASES_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# Context Compactor (Asynchronous Background Conversation Compaction)
# ---------------------------------------------------------------------------

CONTEXT_COMPACTOR_ENABLED: bool = os.getenv("VOICECHAT_COMPACTOR_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
# Trigger compaction when history reaches or exceeds this number of turns (turn = user+assistant pair)
CONTEXT_COMPACTOR_MAX_TURNS: int = _int_from_env("VOICECHAT_COMPACTOR_MAX_TURNS", 6)
# Trigger compaction if total characters across history exceeds this limit (~500-700 tokens)
CONTEXT_COMPACTOR_CHAR_LIMIT: int = _int_from_env("VOICECHAT_COMPACTOR_CHAR_LIMIT", 2500)
# Recent turns (pairs) to always preserve verbatim at the end of the history
CONTEXT_COMPACTOR_PRESERVE_TURNS: int = _int_from_env("VOICECHAT_COMPACTOR_PRESERVE_TURNS", 2)
# When false (default), uses fast non-blocking heuristic summarizer to prevent Ollama inference contention
CONTEXT_COMPACTOR_USE_LLM: bool = os.getenv("VOICECHAT_COMPACTOR_USE_LLM", "false").lower() in {"1", "true", "yes", "on"}



# ---------------------------------------------------------------------------
# Sentence Gate (silence interval before LLM invocation)
# ---------------------------------------------------------------------------

SENTENCE_GATE_GAP_TIMEOUT: float = float(os.getenv("VOICECHAT_GAP_TIMEOUT", "0.5"))
SENTENCE_GATE_QUICK_FIRE_TIMEOUT: float = float(os.getenv("VOICECHAT_QUICK_FIRE_TIMEOUT", "0.25"))

# ---------------------------------------------------------------------------
# Barge-in / echo cancellation (experimental, opt-in)
#
# When disabled (default) the CLI microphone is muted while the assistant is
# speaking, so the assistant can never hear and answer itself.  When enabled,
# the microphone stays live during playback for interruption detection; this
# needs headphones or a good acoustic setup, because the software echo
# canceller cannot fully remove loudspeaker echo.
# ---------------------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}

BARGE_IN_ENABLED: bool = os.getenv("VOICECHAT_BARGE_IN_ENABLED", "false").lower() in _TRUTHY
AEC_ENABLED: bool = os.getenv("VOICECHAT_AEC_ENABLED", "false").lower() in _TRUTHY

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
VOICECHAT_TTS_VOICE: str = os.getenv("VOICECHAT_TTS_VOICE", "").strip() or POCKET_TTS_DEFAULT_VOICE
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
