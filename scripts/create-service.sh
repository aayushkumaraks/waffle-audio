#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OS="$(uname -s)"
case "$OS" in
  Darwin)
    LABEL="com.waffle-audio.server"; PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"; mkdir -p "$(dirname "$PLIST")"
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>Label</key><string>$LABEL</string><key>ProgramArguments</key><array><string>$APP_DIR/scripts/run.sh</string></array><key>WorkingDirectory</key><string>$APP_DIR</string><key>RunAtLoad</key><true/><key>KeepAlive</key><false/><key>StandardOutPath</key><string>$APP_DIR/waffle-audio.log</string><key>StandardErrorPath</key><string>$APP_DIR/waffle-audio-error.log</string></dict></plist>
EOF
    launchctl unload "$PLIST" >/dev/null 2>&1 || true; launchctl load "$PLIST"
    echo "Login service installed: $PLIST"; echo "Remove: launchctl unload '$PLIST' && rm '$PLIST'" ;;
  Linux)
    command -v systemctl >/dev/null 2>&1 || { echo "systemd unavailable; run scripts/run.sh manually." >&2; exit 1; }
    UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"; UNIT="$UNIT_DIR/waffle-audio.service"; mkdir -p "$UNIT_DIR"
    cat > "$UNIT" <<EOF
[Unit]
Description=Waffle Audio local voice assistant
After=network-online.target
[Service]
Type=simple
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/scripts/run.sh
Restart=on-failure
RestartSec=5
[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload; systemctl --user enable --now waffle-audio.service
    echo "Login service installed. Status: systemctl --user status waffle-audio.service"; echo "Remove: systemctl --user disable --now waffle-audio.service && rm '$UNIT'" ;;
  MINGW*|MSYS*|CYGWIN*) echo "Use PowerShell: .\\scripts\\create-service.ps1"; exit 1 ;;
  *) echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac
