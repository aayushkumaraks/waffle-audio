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
- Runs the interactive setup wizard (`setup.py`) on first launch to configure your microphone, detect Ollama host connectivity, and personalize your assistant.

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
*Probes reachable Ollama endpoints, queries detected microphone devices, and configures `.env`.*

### 5. Verify Installation (Unit Tests)
```bash
PYTHONPATH=. uv run pytest tests/
```
*Runs the 111 deterministic unit test suite to verify all audio drivers, STT, LLM, TTS, tools, and conversation components are fully operational.*

---

## 3. Emotional Voice Models Setup (Optional)

Waffle Audio includes an experimental multi-emotion voice cloning system:
1. Emotion clips (`voices/output/*.wav`) from the CREMA-D dataset provide audio prompts for 4 emotional states (`neutral`, `happy`, `sad`, `angry`) across female (actor 1003) and male (actor 1011) voices.
2. Generate all emotion `.safetensors` models into `cloned_voices/`:
```bash
uv run python generate_emotion_voices.py
```
*(Note: `./run-with-emotion.sh` automatically checks and generates missing models on startup).*
3. Run the emotional voice assistant:
```bash
./run-with-emotion.sh
```

---

## 4. Detailed Documentation

- **[Full Installation Guide](docs/setup.md)**: Prerequisites, PortAudio, WSLg audio, Ollama connection, and troubleshooting.
- **[Configuration Reference](docs/configuration.md)**: Complete list of all `.env` options and default values.
- **[Running Guide](docs/running.md)**: Starting the CLI, trigger mode, verbose profiling, and API server.
- **[Voice Cloning Guide](docs/voice-cloning.md)**: Custom voice cloning and audio model setup.
