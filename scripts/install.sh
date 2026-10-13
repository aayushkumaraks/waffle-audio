#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="https://github.com/aayushkumaraks/waffle-audio.git"
APP_DIR="${WAFFLE_AUDIO_DIR:-$HOME/waffle-audio}"
say() { printf '\n[Waffle Audio] %s\n' "$*"; }
fail() { say "ERROR: $*" >&2; exit 1; }
read_env() {
  local key="$1"
  if [[ -n "${!key:-}" ]]; then printf '%s' "${!key}"; return; fi
  if [[ -f .env ]]; then
    sed -n "s/^$key=//p" .env | tail -n 1
  fi
}
upsert_env() {
  local key="$1" value="$2" tmp
  tmp="$(mktemp)"
  if [[ -f .env ]]; then grep -v -E "^$key=" .env > "$tmp" || true; fi
  printf '%s=%s\n' "$key" "$value" >> "$tmp"
  mv "$tmp" .env
}
prompt_env() {
  local key="$1" prompt="$2" default="$3" value
  value="$(read_env "$key")"
  if [[ -z "$value" ]]; then
    read -r -p "$prompt [$default]: " value || value=""
    value="${value:-$default}"
  fi
  printf '%s' "$value"
}

command -v docker >/dev/null 2>&1 || fail "Install Docker Desktop/Engine first: https://docs.docker.com/get-docker/"
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required. Update Docker Desktop or install the Compose plugin."
if [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" pull --ff-only || fail "Could not update $APP_DIR; check local changes."
elif [[ -e "$APP_DIR" ]]; then
  fail "$APP_DIR exists but is not a Git repository. Set WAFFLE_AUDIO_DIR to another path."
else
  command -v git >/dev/null 2>&1 || fail "Git is required to download the project."
  git clone "$REPO_URL" "$APP_DIR" || fail "Clone failed."
fi
cd "$APP_DIR"

say "Configure Ollama. The URL must be reachable FROM the Docker container."
ollama_url="$(prompt_env OLLAMA_BASE_URL 'Ollama base URL' 'http://host.docker.internal:11434')"
[[ "$ollama_url" =~ ^https?://[^[:space:]]+$ ]] || fail "OLLAMA_BASE_URL must be an http(s) URL without spaces."
ollama_model="$(prompt_env OLLAMA_MODEL 'Ollama model name' 'qwen3.5:4b')"
[[ "$ollama_model" =~ ^[A-Za-z0-9._:/-]+$ ]] || fail "OLLAMA_MODEL contains unsupported characters."

voice="$(prompt_env VOICECHAT_TTS_VOICE 'Pocket TTS voice' 'alba')"
[[ "$voice" =~ ^[A-Za-z0-9._/-]+$ ]] || fail "Invalid voice identifier."
tts_backend="$(read_env VOICECHAT_TTS_BACKEND)"
tts_backend="${tts_backend:-pocket}"

upsert_env OLLAMA_BASE_URL "$ollama_url"
upsert_env OLLAMA_MODEL "$ollama_model"
upsert_env VOICECHAT_TTS_BACKEND "$tts_backend"
upsert_env VOICECHAT_TTS_VOICE "$voice"

say "Building the Docker image. This validates dependency installation before launch."
docker compose build waffle-audio || fail "Docker image build failed. Review the build output above."

say "Checking Ollama connectivity and running a minimal chat request from inside the image."
if docker compose run --rm --no-deps --entrypoint python waffle-audio-cli scripts/validate-ollama.py; then
  :
else
  status=$?
  if [[ "$status" -eq 2 ]]; then
    if command -v ollama >/dev/null 2>&1 && [[ -t 0 ]]; then
      read -r -p "Pull '$ollama_model' on this host now? [y/N] " pull_model || pull_model=""
      if [[ "$pull_model" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
        ollama pull "$ollama_model" || fail "ollama pull failed."
        docker compose run --rm --no-deps --entrypoint python waffle-audio-cli scripts/validate-ollama.py || fail "Ollama validation still failed."
      else
        fail "Install the model with: ollama pull $ollama_model, then rerun the installer."
      fi
    else
      fail "Install the model on the host with: ollama pull $ollama_model, then rerun the installer."
    fi
  else
    fail "Ollama validation failed. Check that Ollama is running and OLLAMA_BASE_URL is reachable from Docker."
  fi
fi

say "Configuration and Docker build validated."
if [[ "$(uname -s)" == "Linux" ]]; then
  if [[ -e /dev/snd ]]; then
    say "Linux audio device detected. CLI mode will map /dev/snd when launched."
  else
    say "No /dev/snd found. Use web mode for browser microphone/speaker access, or configure a Linux audio device."
  fi
else
  say "On macOS, use web mode: Docker Desktop does not provide portable host microphone passthrough to Linux containers."
fi
say "One-click scripts: $APP_DIR/scripts/"
if [[ -t 0 ]]; then
  read -r -p 'Run Waffle Audio now? [y/N] ' run_now || run_now=""
  if [[ "$run_now" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then exec "$APP_DIR/scripts/run.sh"; fi
  read -r -p 'Start web/API container in background with Docker restart policy? [y/N] ' service || service=""
  if [[ "$service" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then exec "$APP_DIR/scripts/create-service.sh"; fi
fi
say "No action taken. Run later: cd \"$APP_DIR\" && ./scripts/run.sh"
