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

# 4. Check Ollama service & model using constants.py
echo -e "${CYAN}[4/4]${RESET} Checking Ollama service and model..."
OLLAMA_BASE_URL=$(uv run --no-sync python -c "import constants; print(constants.OLLAMA_BASE_URL)" 2>/dev/null || echo "http://172.19.96.1:11434")
MODEL_NAME=$(uv run --no-sync python -c "import constants; print(getattr(constants, 'OLLAMA_MODEL', 'qwen3.5:4b'))" 2>/dev/null || echo "qwen3.5:4b")
OLLAMA_BASE_URL="${OLLAMA_BASE_URL%/}"

if curl -s "${OLLAMA_BASE_URL}/api/tags" &> /dev/null; then
    if curl -s "${OLLAMA_BASE_URL}/api/tags" | grep -q "${MODEL_NAME}"; then
        echo -e "${GREEN}✓ Ollama model '${MODEL_NAME}' is ready at ${OLLAMA_BASE_URL}.${RESET}"
    else
        echo -e "${YELLOW}Model '${MODEL_NAME}' not found at ${OLLAMA_BASE_URL}. Pulling model now...${RESET}"
        ollama pull "${MODEL_NAME}" || true
    fi
else
    echo -e "${YELLOW}Warning: Ollama not reachable at ${OLLAMA_BASE_URL}.${RESET}"
    echo -e "${YELLOW}Please ensure Ollama is installed and running at ${OLLAMA_BASE_URL}.${RESET}"
fi

# Launch the CLI application
echo -e "\n${BOLD}${GREEN}Starting Waffle Audio CLI...${RESET}\n"
exec uv run python -m services.conversation.cli "$@"
