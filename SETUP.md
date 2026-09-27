# Setup

Follow these steps once before running the app for the first time.

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

`uv` reads `pyproject.toml` and creates an isolated virtual environment automatically.

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

The default `base_url` in `services/llm/src/llm_provider.py` points to the
Windows host IP as seen from inside WSL2.

If your Ollama server is running elsewhere, update
`LLMProviderConfig.base_url`.

---

## 4. Choose a TTS backend

Kokoro remains the default and uses the existing model files in `voiceModels/`:

```
voiceModels/
├── kokoro-v1.0.onnx
└── voices-v1.0.bin
```

To opt into Pocket TTS, set `VOICECHAT_TTS_BACKEND=pocket` before starting the
app. It downloads and caches its model and the default `alba` voice on first use;
no files need to be placed in `voiceModels/`. Its cache is persisted at
`voiceModels/pocket-tts-cache/`; override that location with
`VOICECHAT_POCKET_TTS_CACHE_DIR` if needed.

---

## 5. Verify audio

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

Setup is complete. See [RUN.md](RUN.md) to start the app.
