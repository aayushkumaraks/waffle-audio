#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d node_modules ]; then
  echo "Installing webapp dependencies..."
  npm install
fi

exec npm run dev -- --host 0.0.0.0
