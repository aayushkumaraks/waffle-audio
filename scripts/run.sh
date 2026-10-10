#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$APP_DIR"

if [[ ! -x "$APP_DIR/.venv/bin/python" && ! -f "$APP_DIR/.venv/Scripts/python.exe" ]]; then
  echo "Waffle Audio dependencies are not installed. Run scripts/install.sh first." >&2
  exit 1
fi
# Load user-selected runtime settings without executing arbitrary shell syntax.
if [[ -f "$APP_DIR/.env" ]]; then
  while IFS='=' read -r key value; do
    case "$key" in VOICECHAT_TTS_BACKEND|VOICECHAT_TTS_VOICE|VOICECHAT_API_HOST|VOICECHAT_API_PORT|VOICECHAT_CORS_ALLOW_ORIGINS) export "$key=$value" ;; esac
  done < "$APP_DIR/.env"
fi
export VOICECHAT_TTS_BACKEND="${VOICECHAT_TTS_BACKEND:-pocket}"
if [[ -n "${VOICECHAT_TTS_VOICE:-}" ]]; then
  export WAFFLE_AUDIO_TTS_VOICE="$VOICECHAT_TTS_VOICE"
fi
if ! curl --silent --fail "${OLLAMA_URL:-http://127.0.0.1:11434}/api/version" >/dev/null 2>&1; then
  echo "Ollama is not reachable. Start Ollama and ensure model qwen3.5:4b is installed." >&2
  echo "Try: ollama serve  (then: ollama pull qwen3.5:4b)" >&2
  read -r -p "Start the Waffle Audio API anyway? [y/N] " answer || answer=""
  [[ "$answer" =~ ^([Yy]|[Yy][Ee][Ss])$ ]] || exit 1
fi
if [[ -x "$APP_DIR/.venv/bin/python" ]]; then
  exec uv run python -m services.api.app
else
  exec uv run python -m services.api.app
fi
