# Setup & Installation Guide

This guide covers everything you need to install, configure, and prepare Waffle Audio on Linux, Windows (WSL2 / native), or macOS.

---

## 1. System Requirements

- **Python**: Python 3.11 (`3.11.x` is strictly required; Python 3.12+ is not yet supported by certain upstream audio dependencies).
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/) (Astral's ultra-fast Python package and project manager).
- **LLM Runtime**: [Ollama](https://ollama.com/) running locally or accessible on your local network/WSL host.
- **Audio Subsystem**:
  - **Linux / WSL2**: PortAudio (`libportaudio2`, `portaudio19-dev`, `pulseaudio-utils`).
  - **macOS**: Built-in CoreAudio drivers.
  - **Windows**: Bundled WASAPI / MME drivers.
- **Hardware Recommendations**:
  - Minimum 8 GB RAM (16 GB recommended).
  - GPU acceleration (NVIDIA CUDA or Apple Silicon Metal) for Ollama model inference is recommended for sub-second latencies, but CPU inference also works.

---

## 2. One-Click Automated Setup

The easiest way to get started is running the automated launcher script:

### On Linux / WSL2:
```bash
./run.sh
```

### On Windows (CMD / PowerShell):
```cmd
run.bat
```

What `./run.sh` does automatically:
1. Detects or installs `uv` if missing.
2. Checks for PortAudio system dependencies on Linux/WSL and installs them via `apt` if needed.
3. Synchronizes the virtual environment with exact dependencies via `uv sync`.
4. Launches the interactive **Setup Wizard** (`setup.py`) on your first run.
5. Checks whether Ollama is reachable and ensures the default model (`qwen3.5:4b`) is installed.
6. Launches the assistant CLI with real-time UI.

---

## 3. Manual Installation (Step-by-Step)

If you prefer installing dependencies manually or running in a custom environment:

### Step 3.1: Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### Step 3.2: Install Audio System Libraries (Linux / WSL2)
```bash
sudo apt update
sudo apt install -y libportaudio2 portaudio19-dev pulseaudio-utils
```

### Step 3.3: Clone Repository & Sync Dependencies
```bash
git clone https://github.com/aayushkumaraks/waffle-audio.git
cd waffle-audio
uv sync
```

### Step 3.4: Install & Start Ollama
1. Download Ollama from [ollama.com](https://ollama.com/download) or install via terminal:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. Pull the recommended conversational model:
   ```bash
   ollama pull qwen3.5:4b
   ```
3. Verify Ollama is running:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Step 3.5: Run the Setup Wizard
Run the interactive onboarding wizard to configure your `.env` file:
```bash
uv run python setup.py
```

To run non-interactively with safe default values:
```bash
uv run python setup.py --non-interactive
```

To reconfigure an existing `.env` file at any time:
```bash
uv run python setup.py --force
```

---

## 4. WSL2 Specific Notes & Audio Passthrough

If you are running inside Windows Subsystem for Linux (WSL2):

1. **WSLg Audio**: Modern WSL2 on Windows 11 includes WSLg with native PulseAudio redirection. Sound output and microphone capture pass through automatically to Windows default devices.
2. **Ollama on Windows Host**: If Ollama is installed on your Windows host rather than inside WSL, `setup.py` automatically detects your Windows host gateway IP (via `ip route show default`) and sets `OLLAMA_BASE_URL` to `http://<WSL_HOST_IP>:11434`.
3. **Microphone Permissions**: Ensure Windows Settings -> Privacy & Security -> Microphone has granted access to Windows Subsystem for Linux.

---

## 5. Troubleshooting Installation

| Problem | Cause | Solution |
|---|---|---|
| `No module named '_sounddevice'` or `PortAudio not found` | Missing system C library | Run `sudo apt install libportaudio2 portaudio19-dev` |
| `requires-python >=3.11,<3.12` error | Python 3.12+ active in system | `uv` automatically downloads Python 3.11 when running `uv sync`. Ensure you execute commands via `uv run`. |
| `Ollama not reachable` | Ollama daemon not started | Start Ollama: `ollama serve`. On WSL2, verify firewall allows port 11434 or configure `OLLAMA_HOST=0.0.0.0` in Windows. |
| Model download error on first speech turn | Hugging Face cache permission | Ensure write permissions in `voiceModels/pocket-tts-cache/`. Models download automatically on first speech. |
