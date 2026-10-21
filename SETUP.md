# Setup

Follow these steps once before running the app for the first time.

> [!TIP]
> After setup, you can launch the voice assistant with one click using `./run.sh` (or `run.bat` on Windows).

## 1. Prerequisites

| Requirement | Minimum version | Notes                                                   |
|-------------|-----------------|---------------------------------------------------------|
| Python      | 3.11            | Check with `python --version`                           |
| uv          | latest          | https://docs.astral.sh/uv/getting-started/installation/ |
| Ollama      | latest          | https://ollama.com                                      |
| PortAudio   | latest          | Required by `sounddevice`                               |

### Install PortAudio

PortAudio is a system library, not a Python package.

**Debian / Ubuntu / WSL2**

```bash
sudo apt update
sudo apt install libportaudio2 portaudio19-dev pulseaudio-utils
```

**macOS**

```bash
brew install portaudio
```

**Windows (native)**

No additional installation is required. The `sounddevice` wheel bundles PortAudio.

### WSL2 compatibility

The application captures microphone input using **PortAudio** through the
Python `sounddevice` package.

Some Ubuntu releases (particularly Ubuntu 22.04's `libportaudio2` package)
do **not** expose the PulseAudio host API required by WSLg. In that case,
the application will fail to detect any audio devices even though WSLg and
PulseAudio are running correctly.

Verify your PortAudio installation:

```bash
uv run python - <<'EOF'
import sounddevice as sd

print("Host APIs:", sd.query_hostapis())
print("Devices:")
print(sd.query_devices())
EOF
```

A working WSL installation should include a **PulseAudio** host API, for example:

```
Host APIs:
ALSA
OSS
PulseAudio
```

If you only see:

```
ALSA
OSS
```

and the device list is empty, your PortAudio package does not support
PulseAudio. This is a known limitation of some Ubuntu packages.

Recommended options:

- Ubuntu 24.04+ or newer
- Build a newer PortAudio release with PulseAudio support
- Use a native Linux installation instead of WSL

---

## 2. Install Python dependencies

```bash
uv sync
```

`uv` reads `pyproject.toml` and creates an isolated virtual environment automatically. This installs all required pipeline components, including `scipy` for polyphase FIR resampling, Silero VAD, Moonshine STT, and Pocket TTS (optional `soxr` is supported for SIMD-accelerated resampling).

---

## 3. Pull the Ollama model

The app uses `qwen3.5:4b` by default.

```bash
ollama pull qwen3.5:4b
```

Start the Ollama server:

```bash
ollama serve
```

### WSL2 note

If you run Ollama on your Windows host while running Waffle Audio inside WSL2, the setup script (`setup.py` or `./run.sh`) automatically tests connectivity and detects your Windows host gateway IP. You can also configure `OLLAMA_BASE_URL` in `.env`.

---

## 4. Initial Setup & Personalization Wizard

Run the interactive setup wizard once before your first launch (or simply run `./run.sh`, which launches it automatically if `.env` does not exist):

```bash
uv run python setup.py
```

The wizard guides you through three quick configuration stages:
1. **User Personalization** (no defaults):
   - Your preferred name (`USER_NAME`)
   - Your city or location (`USER_LOCATION`, e.g. `New Delhi`, `London`)
   - Your timezone (`USER_TIMEZONE`, e.g. `Asia/Kolkata`, `America/New_York`)
2. **Assistant & Audio Hardware** (smart defaults provided):
   - Assistant name (`ASSISTANT_NAME`, default: `Waffle`)
   - TTS voice (`VOICECHAT_TTS_VOICE`, default: `anna`, showing all cloned voices)
   - Microphone input device index (`AUDIO_INPUT_DEVICE`, default: `default`, listing all detected sounddevice inputs)
3. **LLM Runtime & External API Tokens** (defaults provided):
   - Ollama base URL (`OLLAMA_BASE_URL`, default: `http://localhost:11434` or detected WSL host)
   - Ollama model (`OLLAMA_MODEL`, default: `qwen3.5:4b`)
   - GitHub personal access token (`GITHUB_PERSONAL_ACCESS_TOKEN`, default: none/empty)

Your settings are written to `.env`. This file is git-ignored and safe for local credentials. You can copy [`.env.example`](.env.example) to `.env` to configure manually, or re-run setup anytime with:

```bash
uv run python setup.py --force
```

---

## 5. TTS backend and voice cloning

Pocket TTS is the default backend (`VOICECHAT_TTS_BACKEND=pocket`) and supports real-time voice cloning.
Its cache is persisted at `voiceModels/pocket-tts-cache/`; override with `VOICECHAT_POCKET_TTS_CACHE_DIR` if needed.

To clone any voice and use it for your assistant:
```bash
uv run python clone_voice.py
```
> [!NOTE]
> Pocket TTS voice cloning requires accepting terms on Hugging Face at https://huggingface.co/kyutai/pocket-tts. You will be prompted for your Hugging Face token on first use if not already authenticated.

To switch back to the Kokoro backend, set `VOICECHAT_TTS_BACKEND=kokoro`. Kokoro uses the existing model files in `voiceModels/`:
```
voiceModels/
├── kokoro-v1.0.onnx
└── voices-v1.0.bin
```

---

## 6. Verify audio

Run:

```bash
uv run python - <<'EOF'
import sounddevice as sd

print("Host APIs:", sd.query_hostapis())
print("Devices:")
print(sd.query_devices())
EOF
```

Expected output:

- `PulseAudio` appears in the host API list.
- Your microphone and speakers appear in the device list.

If the device list is empty:

1. Verify WSLg microphone access is enabled.
2. Run `pactl info` and confirm `RDPSource` exists.
3. If only `ALSA` and `OSS` are listed, your PortAudio installation does not support PulseAudio.

---

Setup is complete. See [RUN.md](RUN.md) to start the app, or launch immediately with `./run.sh` (`run.bat` on Windows).
