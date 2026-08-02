# Setup

Follow these steps once before running the app for the first time.

## 1. Prerequisites

| Requirement | Minimum version | Notes                                                  |
|-------------|----------------|---------------------------------------------------------|
| Python      | 3.11           | Check with `python --version`                           |
| uv          | latest         | https://docs.astral.sh/uv/getting-started/installation/ |
| Ollama      | latest         | https://ollama.com                                      |
| PortAudio   | any            | Required by sounddevice (see below)                     |

### Install PortAudio

PortAudio is a system library, not a Python package.

**Debian / Ubuntu / WSL2:**
```
sudo apt install libportaudio2
```

**macOS:**
```
brew install portaudio
```

**Windows (native):**
PortAudio is bundled with the `sounddevice` wheel on Windows; no separate install
is needed.

---

## 2. Install Python dependencies

From the project root:

```
uv sync
```

uv reads `pyproject.toml` and creates an isolated virtual environment automatically.

---

## 3. Pull the Ollama model

The app uses `qwen3.5:4b` by default. Pull it before the first run:

```
ollama pull qwen3.5:4b
```

Make sure the Ollama server is running:

```
ollama serve
```

### WSL2 note

The default `base_url` in `services/llm/src/llm_provider.py` points to
`http://172.19.96.1:11434`, which is the Windows host IP as seen from inside WSL2.
If you are running Ollama on a different host or port, update `LLMProviderConfig.base_url`
in that file.

---

## 4. Place voice model files

The Kokoro TTS engine requires two model files in the `voiceModels/` directory:

```
voiceModels/
  kokoro-v1.0.onnx
  voices-v1.0.bin
```

Download them from the Kokoro ONNX releases page:
https://github.com/thewh1teagle/kokoro-onnx/releases

---

## 5. Verify audio devices

The app uses your system default microphone and speakers. Run the following to confirm
sounddevice can see your devices:

```
uv run python -c "import sounddevice; print(sounddevice.query_devices())"
```

If no devices appear, check that PortAudio is installed correctly (step 1).

---

Setup is complete. See [RUN.md](RUN.md) to start the app.
