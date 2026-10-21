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

# Check if .env file exists or if --setup was explicitly requested
SETUP_REQUESTED=false
CLEANED_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--setup" ]]; then
        SETUP_REQUESTED=true
    else
        CLEANED_ARGS+=("$arg")
    fi
done

if [ ! -f .env ] || [ "$SETUP_REQUESTED" = true ]; then
    echo -e "\n${CYAN}[Setup]${RESET} Launching interactive configuration setup..."
    if [ "$SETUP_REQUESTED" = true ]; then
        uv run python setup.py --force
    else
        uv run python setup.py
    fi
fi

# 4. Check Ollama service & model using constants.py / .env
echo -e "${CYAN}[4/4]${RESET} Checking Ollama service and model..."
OLLAMA_BASE_URL=$(uv run --no-sync python -c "import constants; print(constants.OLLAMA_BASE_URL)" 2>/dev/null || echo "http://localhost:11434")
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

for arg in "$@"; do
    if [[ "$arg" == "--verbose" || "$arg" == "-v" ]]; then
        echo -e "${CYAN}[INFO]${RESET} Verbose latency profiling & README benchmark comparison enabled."
    fi
done

# 5. Look for cloned voices and offer voice selection if not explicitly provided
VOICE_ARG=()
HAS_VOICE_FLAG=false
for arg in "$@"; do
    if [[ "$arg" == "--voice" || "$arg" == --voice=* ]]; then
        HAS_VOICE_FLAG=true
        break
    fi
done

if [ "$HAS_VOICE_FLAG" = false ]; then
    CLONED_DIR="cloned_voices"
    CLONED_VOICES=()
    if [ -d "$CLONED_DIR" ]; then
        while IFS= read -r f; do
            if [ -n "$f" ]; then
                name=$(basename "$f" .safetensors)
                CLONED_VOICES+=("$name")
            fi
        done < <(find "$CLONED_DIR" -maxdepth 1 -name "*.safetensors" 2>/dev/null | sort)
    fi

    if [ ${#CLONED_VOICES[@]} -gt 0 ]; then
        echo -e "\n${BOLD}${CYAN}Cloned Voices Detected:${RESET}"
        echo -e "  ${BOLD}[0]${RESET} Default voice"
        idx=1
        for v in "${CLONED_VOICES[@]}"; do
            echo -e "  ${BOLD}[$idx]${RESET} $v (cloned)"
            ((idx++))
        done

        if [ -t 0 ]; then
            echo -ne "\n${BOLD}Would you like to use a cloned voice for your assistant? [0-${#CLONED_VOICES[@]}, default: 0]: ${RESET}"
            read -r choice || choice=""
            choice=$(echo "$choice" | tr -d '[:space:]')
            if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#CLONED_VOICES[@]}" ]; then
                selected_voice="${CLONED_VOICES[$((choice-1))]}"
                echo -e "${GREEN}✓ Selected voice: ${BOLD}${selected_voice}${RESET}"
                VOICE_ARG=("--voice" "${selected_voice}")
            elif [ -n "$choice" ] && [ "$choice" != "0" ] && [ "$choice" != "n" ] && [ "$choice" != "no" ]; then
                # Match by typed voice name
                matched_voice=""
                for v in "${CLONED_VOICES[@]}"; do
                    if [ "$choice" == "$v" ]; then
                        matched_voice="$v"
                        break
                    fi
                done
                if [ -n "$matched_voice" ]; then
                    echo -e "${GREEN}✓ Selected voice: ${BOLD}${matched_voice}${RESET}"
                    VOICE_ARG=("--voice" "${matched_voice}")
                else
                    echo -e "${YELLOW}Using default voice.${RESET}"
                fi
            else
                echo -e "${GREEN}Using default voice.${RESET}"
            fi
        fi
    else
        echo -e "${CYAN}[Voice]${RESET} Using default voice. (Tip: Clone custom voices anytime with: ${BOLD}uv run python clone_voice.py${RESET})"
    fi
fi

# Launch the CLI application
echo -e "\n${BOLD}${GREEN}Starting Waffle Audio CLI...${RESET}\n"
exec uv run python -u -m services.conversation.cli "${CLEANED_ARGS[@]}" "${VOICE_ARG[@]}"
