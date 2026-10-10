#!/usr/bin/env bash
# Waffle Audio installer for macOS, Linux, and Git Bash/WSL on Windows.
set -Eeuo pipefail

REPO_URL="https://github.com/aayushkumaraks/waffle-audio.git"
APP_DIR="${WAFFLE_AUDIO_DIR:-$HOME/waffle-audio}"
VOICE_DEFAULT="alba"
OLLAMA_MODEL="qwen3.5:4b"

say() { printf '\n[Waffle Audio] %s\n' "$*"; }
warn() { printf '\n[Waffle Audio] WARNING: %s\n' "$*" >&2; }
fail() { printf '\n[Waffle Audio] ERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
confirm() { local prompt="$1" answer; read -r -p "$prompt [y/N] " answer || return 1; [[ "$answer" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; }

case "$(uname -s)" in
  Darwin) OS="macos" ;;
  Linux) OS="linux"; if grep -qi microsoft /proc/version 2>/dev/null; then OS="wsl"; fi ;;
  MINGW*|MSYS*|CYGWIN*) OS="windows-git-bash" ;;
  *) fail "Unsupported OS. On Windows use scripts/install.ps1 or Git Bash/WSL." ;;
esac
say "Detected platform: $OS"
say "Checking prerequisites..."

PYTHON=""
for candidate in python3.11 python3 python; do
  if have "$candidate" && "$candidate" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 12) else 1)' >/dev/null 2>&1; then PYTHON="$(command -v "$candidate")"; break; fi
done
if [[ -z "$PYTHON" ]]; then
  warn "Python 3.11 was not found."
  case "$OS" in
    macos) say "Install with: brew install python@3.11" ;;
    linux|wsl) say "Install Python 3.11 using your distribution's package manager or pyenv." ;;
    windows-git-bash) say "Install from https://www.python.org/downloads/windows/ and enable the PATH option." ;;
  esac
  fail "Python 3.11 is required. Install it and rerun this script."
fi
say "Python: $("$PYTHON" --version) ($PYTHON)"

if ! have uv; then
  say "uv was not found; installing it using the official installer."
  if [[ "$OS" == "windows-git-bash" ]]; then
    have powershell.exe || fail "Install uv using PowerShell: irm https://astral.sh/uv/install.ps1 | iex"
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command 'irm https://astral.sh/uv/install.ps1 | iex' || fail "uv installation failed. See https://docs.astral.sh/uv/getting-started/installation/"
  else
    have curl || fail "curl is required to install uv."
    curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv installation failed."
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  fi
fi
have uv || fail "uv is not on PATH. Open a new terminal or add uv's install directory to PATH."
say "uv: $(uv --version)"

if ! have git; then fail "Git is required. Install Git and rerun this script."; fi
if ! have ollama; then
  warn "Ollama is not installed (required for local LLM responses)."
  case "$OS" in
    macos) say "Install from https://ollama.com/download/mac (or brew install --cask ollama)" ;;
    linux|wsl) say "Install with: curl -fsSL https://ollama.com/install.sh | sh" ;;
    windows-git-bash) say "Install from https://ollama.com/download/windows" ;;
  esac
  confirm "Continue setup without Ollama for now?" || fail "Install Ollama and rerun the installer."
else say "Ollama: $(ollama --version 2>/dev/null | head -n 1)"; fi

case "$OS" in
  macos)
    if have brew && ! brew list --versions portaudio >/dev/null 2>&1; then
      if confirm "Install PortAudio with Homebrew?"; then brew install portaudio || warn "PortAudio install failed; audio I/O may not work."; fi
    elif ! have brew; then warn "Homebrew not found. Install it from https://brew.sh if audio devices fail."; fi
    ;;
  linux)
    if have apt-get && ! dpkg -s libportaudio2 >/dev/null 2>&1; then
      if confirm "Install Linux audio prerequisites via apt (requires sudo)?"; then sudo apt-get update && sudo apt-get install -y libportaudio2 portaudio19-dev pulseaudio-utils || warn "Audio package installation failed."; fi
    elif have dnf && ! rpm -q portaudio >/dev/null 2>&1; then
      if confirm "Install PortAudio via dnf (requires sudo)?"; then sudo dnf install -y portaudio portaudio-devel || warn "PortAudio installation failed."; fi
    elif have pacman && ! pacman -Q portaudio >/dev/null 2>&1; then
      if confirm "Install PortAudio via pacman (requires sudo)?"; then sudo pacman -Sy --needed --noconfirm portaudio || warn "PortAudio installation failed."; fi
    fi
    ;;
  wsl) warn "WSL audio access depends on WSLg and a PulseAudio-capable PortAudio host API. Native Windows or Linux may be easier if devices are missing." ;;
esac

if [[ -d "$APP_DIR/.git" ]]; then
  say "Updating existing repository at $APP_DIR"
  git -C "$APP_DIR" pull --ff-only || fail "Could not update repository; check local changes."
elif [[ -e "$APP_DIR" ]]; then fail "$APP_DIR exists but is not a Git repository. Set WAFFLE_AUDIO_DIR to another location."
else say "Cloning Waffle Audio into $APP_DIR"; git clone "$REPO_URL" "$APP_DIR" || fail "Could not clone repository."; fi
cd "$APP_DIR"
say "Installing Python dependencies with uv..."
uv sync || fail "Dependency installation failed. Review the output above and retry."

cat <<'VOICES'

Pocket TTS voices
-----------------
Choose a voice identifier. Default: alba.
English: alba, anna, azelma, bill_boerst, caro_davy, charles, cosette,
         eponine, eve, fantine, george, jane, jean, javert, marius, mary,
         michael, paul, peter_yearsley, stuart_bell, vera
French: estelle | German: juergen | Italian: giovanni
Portuguese: rafael | Spanish: lola
Voice catalog and individual licenses: https://huggingface.co/kyutai/tts-voices
VOICES
read -r -p "Pocket TTS voice [$VOICE_DEFAULT]: " selected_voice || selected_voice=""
selected_voice="${selected_voice:-$VOICE_DEFAULT}"
[[ "$selected_voice" =~ ^[A-Za-z0-9._/-]+$ ]] || fail "Invalid voice identifier."

# Runtime configuration uses environment variables; the launcher reads .env.
ENV_FILE="$APP_DIR/.env"
touch "$ENV_FILE"
set_env() {
  local key="$1" value="$2" tmp
  tmp="$(mktemp)"
  grep -v "^${key}=" "$ENV_FILE" > "$tmp" || true
  printf '%s=%s\n' "$key" "$value" >> "$tmp"
  mv "$tmp" "$ENV_FILE"
}
set_env VOICECHAT_TTS_BACKEND pocket
set_env VOICECHAT_TTS_VOICE "$selected_voice"
say "Selected Pocket TTS voice: $selected_voice"

if have ollama; then
  if ! curl --silent --fail http://127.0.0.1:11434/api/version >/dev/null 2>&1; then warn "Ollama API is not reachable at 127.0.0.1:11434. Start Ollama before running Waffle Audio."; fi
  if ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "$OLLAMA_MODEL"; then
    if confirm "Pull required Ollama model '$OLLAMA_MODEL' now?"; then ollama pull "$OLLAMA_MODEL" || warn "Model pull failed; run 'ollama pull $OLLAMA_MODEL' before starting."; else warn "Model not pulled; run 'ollama pull $OLLAMA_MODEL' before starting."; fi
  fi
fi

say "Installation complete: $APP_DIR"
say "One-click scripts are located at: $APP_DIR/scripts/"
if confirm "Start Waffle Audio now?"; then exec "$APP_DIR/scripts/run.sh"; fi
if confirm "Configure Waffle Audio to start on login?"; then exec "$APP_DIR/scripts/create-service.sh"; fi
say "No action taken. Your scripts remain at: $APP_DIR/scripts/"
say "Run later with: cd \"$APP_DIR\" && ./scripts/run.sh"
