#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d node_modules ]; then
  echo "Installing webapp dependencies..."
  npm ci
fi

exec npm run start
