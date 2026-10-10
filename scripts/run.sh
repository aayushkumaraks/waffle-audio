#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$APP_DIR"
command -v docker >/dev/null 2>&1 || { echo "Docker is required. Install Docker Desktop or Docker Engine first." >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "Docker Compose v2 is required." >&2; exit 1; }
printf '\nWaffle Audio\n============\n1) CLI voice assistant (microphone and speakers)\n2) Web UI + API (use your browser microphone)\n'
read -r -p 'Choose a mode [1/2]: ' mode || mode=""
case "$mode" in
  1) echo "Starting CLI. Audio-device access depends on your Docker host; see README for platform notes."; docker compose --profile cli run --rm waffle-audio-cli ;;
  2) docker compose --profile web up --build -d waffle-audio; echo "Web UI: http://localhost:${VOICECHAT_API_PORT:-3000}/ui"; echo "API docs: http://localhost:${VOICECHAT_API_PORT:-3000}/docs"; echo "Logs: docker compose logs -f waffle-audio" ;;
  *) echo "No mode selected. One-click scripts are in: $SCRIPT_DIR"; exit 0 ;;
esac
