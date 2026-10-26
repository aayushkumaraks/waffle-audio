#!/usr/bin/env bash
# ==============================================================================
# Waffle Audio — One-Click Emotional Voice Assistant Script
# ==============================================================================
set -e

# ANSI styling
BOLD="\033[1m"
GREEN="\033[38;5;46m"
CYAN="\033[38;5;51m"
YELLOW="\033[38;5;226m"
RED="\033[38;5;196m"
MAGENTA="\033[38;5;201m"
RESET="\033[0m"

echo -e "${BOLD}${MAGENTA}"
cat << "EOF"
  ██╗    ██╗ █████╗ ███████╗███████╗██╗     ███████╗
  ██║    ██║██╔══██╗██╔════╝██╔════╝██║     ██╔════╝
  ██║ █╗ ██║███████║█████╗  █████╗  ██║     █████╗  
  ██║███╗██║██╔══██║██╔══╝  ██╔══╝  ██║     ██╔══╝  
  ╚███╔███╔╝██║  ██║██║     ██║     ███████╗███████╗
EOF
echo -e "${RESET}${BOLD}     Emotional Voice-to-Voice AI Assistant (CLI)${RESET}\n"

# Ensure PATH includes standard user local bin directories
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 1. Check & Install uv if not present
echo -e "${CYAN}[1/5]${RESET} Checking package manager (uv)..."
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
    echo -e "${CYAN}[2/5]${RESET} Checking audio library (PortAudio)..."
    if ! ldconfig -p 2>/dev/null | grep -q "libportaudio\.so" && ! dpkg -l 2>/dev/null | grep -q "libportaudio2"; then
        echo -e "${YELLOW}PortAudio was not detected. Attempting installation via apt...${RESET}"
        if command -v sudo &> /dev/null; then
            sudo apt update && sudo apt install -y libportaudio2 portaudio19-dev pulseaudio-utils || true
        fi
    else
        echo -e "${GREEN}✓ PortAudio library found.${RESET}"
    fi
else
    echo -e "${CYAN}[2/5]${RESET} Checking audio subsystem: non-Linux OS (${OSTYPE}), using bundled drivers."
fi

# 3. Synchronize Python environment and dependencies
echo -e "${CYAN}[3/5]${RESET} Synchronizing Python virtual environment dependencies..."
uv sync
echo -e "${GREEN}✓ Dependencies synchronized.${RESET}"

# 4. Check Ollama service & model
echo -e "${CYAN}[4/5]${RESET} Checking Ollama service and model..."
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

# 5. Check and ensure emotion voice safetensors exist
echo -e "${CYAN}[5/5]${RESET} Verifying emotion voice models..."
MISSING_MODELS=0
for emo in neutral happy sad angry; do
    if [ ! -f "cloned_voices/${emo}_female.safetensors" ] || [ ! -f "cloned_voices/${emo}_male.safetensors" ]; then
        MISSING_MODELS=1
        break
    fi
done

if [ "$MISSING_MODELS" -eq 1 ]; then
    echo -e "${YELLOW}Emotion voice safetensors missing. Generating now from voices/output/...${RESET}"
    uv run python generate_emotion_voices.py
fi
echo -e "${GREEN}✓ All emotion voice models verified in cloned_voices/.${RESET}"

for arg in "$@"; do
    if [[ "$arg" == "--verbose" || "$arg" == "-v" ]]; then
        echo -e "${CYAN}[INFO]${RESET} Verbose latency profiling & README benchmark comparison enabled."
    fi
done

# Gender selection
GENDER_ARG=""
for arg in "$@"; do
    if [[ "$arg" == "--gender" || "$arg" == --gender=* ]]; then
        GENDER_ARG="$arg"
        break
    fi
done

if [ -z "$GENDER_ARG" ] && [ -t 0 ]; then
    echo -e "\n${BOLD}${MAGENTA}Select Assistant Voice Gender:${RESET}"
    echo -e "  ${BOLD}[1]${RESET} Female (Emotions: neutral, happy, sad, angry)"
    echo -e "  ${BOLD}[2]${RESET} Male   (Emotions: neutral, happy, sad, angry)"
    echo -ne "\n${BOLD}Enter choice [1-2, default: 1 (Female)]: ${RESET}"
    read -r choice || choice="1"
    choice=$(echo "$choice" | tr -d '[:space:]')
    if [ "$choice" == "2" ] || [ "$choice" == "male" ] || [ "$choice" == "m" ]; then
        GENDER_ARG="--gender=male"
        echo -e "${GREEN}✓ Selected Gender: Male${RESET}"
    else
        GENDER_ARG="--gender=female"
        echo -e "${GREEN}✓ Selected Gender: Female${RESET}"
    fi
fi

EXTRA_ARGS=()
if [ -n "$GENDER_ARG" ]; then
    EXTRA_ARGS+=("$GENDER_ARG")
fi

# Launch Emotional Voice Assistant CLI
echo -e "\n${BOLD}${GREEN}Starting Waffle Audio Emotional Voice Assistant...${RESET}\n"
exec uv run python -m services.conversation.emotion_cli "${EXTRA_ARGS[@]}" "$@"
