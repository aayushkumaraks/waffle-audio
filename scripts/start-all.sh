#!/usr/bin/env bash
set -Eeuo pipefail

say() { printf '[Waffle Audio] %s\n' "$*"; }

say "Starting API backend on 0.0.0.0:${VOICECHAT_API_PORT:-3000}..."
uv run --frozen --no-sync python -m services.api.app &
API_PID=$!

say "Starting Webapp on 0.0.0.0:${VOICECHAT_WEBAPP_PORT:-5173}..."
if [ -d "/app/tools/webapp" ]; then
  (cd /app/tools/webapp && npm run start -- --host 0.0.0.0 --port "${VOICECHAT_WEBAPP_PORT:-5173}") &
  WEBAPP_PID=$!
fi

cleanup() {
  say "Shutting down services..."
  kill -TERM "$API_PID" "${WEBAPP_PID:-}" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
  wait "${WEBAPP_PID:-}" 2>/dev/null || true
}

trap cleanup SIGTERM SIGINT SIGHUP

wait -n "$API_PID" "${WEBAPP_PID:-}"
cleanup
