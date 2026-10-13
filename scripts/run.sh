#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$APP_DIR"
command -v docker >/dev/null 2>&1 || { echo "Docker is required. Install Docker Desktop or Docker Engine first." >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "Docker Compose v2 is required." >&2; exit 1; }

cli_available=false
if [[ "$(uname -s)" == "Linux" && -e /dev/snd ]]; then cli_available=true; fi

printf '\nWaffle Audio\n============\n'
if [[ "$cli_available" == true ]]; then
  printf '1) CLI voice assistant (Linux /dev/snd audio device)\n2) Web UI + API (browser microphone; recommended cross-platform)\n'
  read -r -p 'Choose a mode [1/2]: ' mode || mode=""
  case "$mode" in
    1)
      echo "Starting CLI with the Linux audio device mapped into the container."
      docker compose --profile cli run --build --rm --device /dev/snd:/dev/snd waffle-audio-cli
      ;;
    2)
      docker compose --profile web up --build -d waffle-audio
      echo "Web UI: http://localhost:${VOICECHAT_API_PORT:-3000}/ui"
      echo "API docs: http://localhost:${VOICECHAT_API_PORT:-3000}/docs"
      echo "Logs: docker compose logs -f waffle-audio"
      ;;
    *) echo "No mode selected. Scripts are located at $SCRIPT_DIR"; exit 0 ;;
  esac
else
  if [[ "$(uname -s)" == "Linux" ]]; then
    echo "No /dev/snd device detected; CLI microphone access is unavailable in this setup."
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    echo "macOS Docker Desktop does not provide portable host-microphone passthrough to Linux containers."
  else
    echo "This platform does not have a supported direct CLI audio-device mapping."
  fi
  echo "Use web mode: your browser handles microphone and speaker access."
  read -r -p 'Start Web UI + API? [Y/n]: ' mode || mode=""
  case "$mode" in
    ""|[Yy]|[Yy][Ee][Ss])
      docker compose --profile web up --build -d waffle-audio
      echo "Web UI: http://localhost:${VOICECHAT_API_PORT:-3000}/ui"
      echo "API docs: http://localhost:${VOICECHAT_API_PORT:-3000}/docs"
      echo "Logs: docker compose logs -f waffle-audio"
      ;;
    *) echo "No mode selected. Scripts are located at $SCRIPT_DIR"; exit 0 ;;
  esac
fi
