#!/usr/bin/env bash
set -Eeuo pipefail
REPO_URL="https://github.com/aayushkumaraks/waffle-audio.git"
APP_DIR="${WAFFLE_AUDIO_DIR:-$HOME/waffle-audio}"
say() { printf '\n[Waffle Audio] %s\n' "$*"; }
fail() { say "ERROR: $*" >&2; exit 1; }
command -v docker >/dev/null 2>&1 || fail "Install Docker Desktop/Engine first: https://docs.docker.com/get-docker/"
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required. Update Docker Desktop or install the Compose plugin."
if [[ -d "$APP_DIR/.git" ]]; then git -C "$APP_DIR" pull --ff-only || fail "Could not update $APP_DIR; check local changes."
elif [[ -e "$APP_DIR" ]]; then fail "$APP_DIR exists but is not a Git repository. Set WAFFLE_AUDIO_DIR to another path."
else command -v git >/dev/null 2>&1 || fail "Git is required to download the project."; git clone "$REPO_URL" "$APP_DIR" || fail "Clone failed."; fi
cd "$APP_DIR"
cat <<'VOICES'

Pocket TTS voice selection (default: alba)
English: alba, anna, azelma, bill_boerst, caro_davy, charles, cosette,
         eponine, eve, fantine, george, jane, jean, javert, marius, mary,
         michael, paul, peter_yearsley, stuart_bell, vera
French: estelle | German: juergen | Italian: giovanni
Portuguese: rafael | Spanish: lola
Catalog and licenses: https://huggingface.co/kyutai/tts-voices
VOICES
read -r -p 'Pocket TTS voice [alba]: ' voice || voice=""
voice="${voice:-alba}"
[[ "$voice" =~ ^[A-Za-z0-9._/-]+$ ]] || fail "Invalid voice identifier."
if [[ -f .env ]]; then grep -v -E '^(VOICECHAT_TTS_BACKEND|VOICECHAT_TTS_VOICE)=' .env > .env.tmp || true; mv .env.tmp .env; fi
printf 'VOICECHAT_TTS_BACKEND=pocket\nVOICECHAT_TTS_VOICE=%s\n' "$voice" >> .env
say "Configuration saved. Docker builds the runtime and installs all Python/system dependencies inside the image."
say "Ollama must be installed and running on the host with qwen3.5:4b pulled: ollama pull qwen3.5:4b"
if [[ "$(uname -s)" == "Linux" ]] && [[ ! -e /dev/snd ]]; then say "No /dev/snd found. CLI microphone mode may not be available from this container; web mode uses browser audio."; fi
say "One-click scripts: $APP_DIR/scripts/"
if [[ -t 0 ]]; then
  read -r -p 'Run Waffle Audio now? [y/N] ' run_now || run_now=""
  if [[ "$run_now" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then exec "$APP_DIR/scripts/run.sh"; fi
  read -r -p 'Start web/API container in background with Docker restart policy? [y/N] ' service || service=""
  if [[ "$service" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then exec "$APP_DIR/scripts/create-service.sh"; fi
fi
say "No action taken. Run later: cd \"$APP_DIR\" && ./scripts/run.sh"
