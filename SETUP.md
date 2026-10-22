# Waffle Audio Setup

Quick start guide for installing Waffle Audio. For full details and troubleshooting, see the [Installation & Setup Guide](docs/setup.md).

---

## 1. Quick Install (One-Click)

The fastest way to install everything is simply running the launcher:

```bash
./run.sh
```

On Windows:
```cmd
run.bat
```

`run.sh` automatically:
- Installs `uv` package manager if missing.
- Installs PortAudio audio system libraries on Linux.
- Synchronizes Python 3.11 virtual environment dependencies.
- Runs the interactive setup wizard (`setup.py`) on first launch to configure your microphone, detect Ollama models via `/api/tags`, and personalize your assistant.

---

## 2. Manual Setup

### 1. Install System Audio Dependencies (Linux / WSL2)
```bash
sudo apt update && sudo apt install -y libportaudio2 portaudio19-dev pulseaudio-utils
```

### 2. Synchronize Virtual Environment
```bash
uv sync
```

### 3. Ensure Ollama is Installed & Running
```bash
ollama pull qwen3.5:4b
```

### 4. Run the Configuration Wizard
```bash
uv run python setup.py
```
*Queries Ollama's `/api/tags` endpoint dynamically, presenting an interactive numbered model selector with pre-populated current settings.*

### 5. Verify Installation (Unit Tests)
```bash
uv run pytest
```
*Runs the 211 deterministic unit test suite to verify all audio drivers, STT, LLM, TTS, tools, and API components are fully operational.*

---

## 3. Detailed Documentation

- **[Full Installation Guide](docs/setup.md)**: Prerequisites, PortAudio, WSLg audio, Ollama connection, and troubleshooting.
- **[Configuration Reference](docs/configuration.md)**: Complete list of all `.env` options and default values.
- **[Running Guide](docs/running.md)**: Starting the CLI, trigger mode, verbose profiling, and API server.
