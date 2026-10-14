#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$APP_DIR"
command -v docker >/dev/null 2>&1 || { echo "Docker is required." >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "Docker Compose v2 is required." >&2; exit 1; }
echo "This starts the web/API container in the background with restart: unless-stopped."
read -r -p 'Continue? [y/N] ' answer || answer=""
[[ "$answer" =~ ^([Yy]|[Yy][Ee][Ss])$ ]] || { echo "No changes made. Scripts are in $SCRIPT_DIR"; exit 0; }
docker compose --profile web up --build -d waffle-audio
echo "Web UI: http://localhost:${VOICECHAT_API_PORT:-3000}/ui"
echo "Stop: docker compose --profile web down"
