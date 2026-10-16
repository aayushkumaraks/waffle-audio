#!/usr/bin/env bash
# ==============================================================================
# Waffle Audio — One-Click Install & Run Script
# ==============================================================================
set -e

# ANSI styling
BOLD="\033[1m"
GREEN="\033[38;5;46m"
CYAN="\033[38;5;51m"
YELLOW="\033[38;5;226m"
RED="\033[38;5;196m"
RESET="\033[0m"

echo -e "${BOLD}${CYAN}"
cat << "EOF"
  ██╗    ██╗ █████╗ ███████╗███████╗██╗     ███████╗
  ██║    ██║██╔══██╗██╔════╝██╔════╝██║     ██╔════╝
  ██║ █╗ ██║███████║█████╗  █████╗  ██║     █████╗  
  ██║███╗██║██╔══██║██╔══╝  ██╔══╝  ██║     ██╔══╝  
  ╚███╔███╔╝██║  ██║██║     ██║     ███████╗███████╗
EOF
echo -e "${RESET}${BOLD}          Voice-to-Voice AI Assistant (CLI)${RESET}\n"

# Ensure PATH includes standard user local bin directories
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 1. Check & Install uv if not present
echo -e "${CYAN}[1/4]${RESET} Checking package manager (uv)..."
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv not found. Installing uv automatically...${RESET}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}Error: Failed to install uv. Please install uv manually: https://docs.astral.sh/uv/${RESET}"
        exit 1
    fi
    echo -e "${GREEN}✓ uv installed successfully.${RESET}"
else
    echo -e "${GREEN}✓ uv is ready ($(uv --version))${RESET}"
fi

# 2. Check PortAudio system dependency on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "${CYAN}[2/4]${RESET} Checking audio library (PortAudio)..."
    if ! ldconfig -p 2>/dev/null | grep -q "libportaudio\.so" && ! dpkg -l 2>/dev/null | grep -q "libportaudio2"; then
        echo -e "${YELLOW}PortAudio was not detected. Attempting installation via apt...${RESET}"
        if command -v sudo &> /dev/null; then
            sudo apt update && sudo apt install -y libportaudio2 portaudio19-dev pulseaudio-utils || true
        fi
    else
        echo -e "${GREEN}✓ PortAudio library found.${RESET}"
    fi
else
    echo -e "${CYAN}[2/4]${RESET} Checking audio subsystem: non-Linux OS (${OSTYPE}), using bundled drivers."
fi

# 3. Synchronize Python environment and dependencies
echo -e "${CYAN}[3/4]${RESET} Synchronizing Python virtual environment dependencies..."
uv sync
echo -e "${GREEN}✓ Dependencies synchronized.${RESET}"

# 4. Check Ollama service & model
echo -e "${CYAN}[4/4]${RESET} Checking Ollama service and model..."
OLLAMA_HOST="${VOICECHAT_OLLAMA_HOST:-127.0.0.1:11434}"
MODEL_NAME="${VOICECHAT_MODEL:-qwen3.5:4b}"

if ! curl -s "http://${OLLAMA_HOST}/api/tags" &> /dev/null; then
    if command -v ollama &> /dev/null; then
        echo -e "${YELLOW}Ollama service is not running. Starting background ollama serve...${RESET}"
        ollama serve >/dev/null 2>&1 &
        sleep 3
    else
        echo -e "${YELLOW}Warning: Ollama not reachable at ${OLLAMA_HOST}.${RESET}"
        echo -e "${YELLOW}Please ensure Ollama is installed and running with: ollama serve${RESET}"
    fi
fi

if curl -s "http://${OLLAMA_HOST}/api/tags" &> /dev/null; then
    if ! curl -s "http://${OLLAMA_HOST}/api/tags" | grep -q "${MODEL_NAME}"; then
        echo -e "${YELLOW}Model '${MODEL_NAME}' not found in Ollama. Pulling model now...${RESET}"
        ollama pull "${MODEL_NAME}" || true
    else
        echo -e "${GREEN}✓ Ollama model '${MODEL_NAME}' is ready.${RESET}"
    fi
fi

# Launch the CLI application
echo -e "\n${BOLD}${GREEN}Starting Waffle Audio CLI...${RESET}\n"
exec uv run python -m services.conversation.cli "$@"
