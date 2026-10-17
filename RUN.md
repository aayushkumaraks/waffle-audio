# Running the app

## One-Click Start (CLI App)

The fastest way to install dependencies, verify models, and start the voice assistant:

```bash
./run.sh
```

On Windows:
```cmd
run.bat
```

Or start the CLI app directly with `uv`:

```bash
uv run python -m services.conversation.cli
```

---

## Emotional Voice Assistant (Multi-Emotion Voice Models)

Run the emotion-aware voice assistant with dynamic emotion detection, dedicated per-emotion worker threads, in-memory caching, and automatic idle offloading:

```bash
./run-with-emotion.sh
```

On Windows:
```cmd
run-with-emotion.bat
```

Or start directly with `uv`:
```bash
# Interactive gender selection:
uv run python -m services.conversation.emotion_cli

# Specify gender explicitly (female or male):
uv run python -m services.conversation.emotion_cli --gender female
uv run python -m services.conversation.emotion_cli --gender male

# Enable verbose latency profiling and README benchmark timeline:
./run-with-emotion.sh --verbose
uv run python -m services.conversation.emotion_cli --gender female --verbose
```

### Emotion Features
- **Gender Voice Selection**: Choose either **Female** or **Male** voice model set at startup.
- **4 Emotion States**: Every emotion (`neutral`, `happy`, `sad`, `angry`) has its own `.safetensors` model in `cloned_voices/`.
- **Thread-Per-Emotion Workers**: Each emotion model runs in a dedicated thread (`tts-worker-neutral`, `tts-worker-happy`, `tts-worker-sad`, `tts-worker-angry`).
- **In-Memory Caching with Automatic Offloading**: Models stay in RAM for rapid response and automatically offload to release memory when idle (`--offload-idle <seconds>`).
- **Real-Time LLM Emotion Tagging & Parsing**: The LLM specifies its emotional tone via tags like `[happy]`, `[sad]`, `[angry]`, or `[neutral]`, which are dynamically parsed to route speech chunks to the matching emotion voice worker.
- **Verbose Latency Profiling**: Pass `--verbose` or `-v` to show live pipeline latency metrics and compare against the README warm turn benchmarks.

### CLI Interface Features
- **Top-Right Status Badge**: Always shows `● READY`, `🎤 LISTENING`, or `⠋ PROCESSING`.
- **Real-Time Conversation**: Live speech transcription and streaming voice responses synchronized with audio playback.
- **Dynamic Auto-Scrolling**: Keeps dialogue panel anchored at the bottom as turns arrive, with an indicator for scrolled lines.
- **Multi-Line Speech Accumulation**: Continuous user speech spanning multiple segments is joined into a single coherent prompt.
- **ASCII Animations**: Dynamic header waveforms and processing spinners.
- **Zero Log Clutter**: All engine chatter is written to `logs/voicechat.log`.
- **Verbose Latency Profiling**: Pass `--verbose` (e.g. `./run.sh --verbose`) to display real-time latency timelines and compare with README benchmarks.

---

## Voice Cloning & Custom Voices (Pocket TTS)

Waffle Audio supports instant voice cloning with Pocket TTS. Cloned voices are saved to `cloned_voices/<voice_name>.safetensors`.

### 1. Clone a voice

Run the interactive cloning wizard:

```bash
uv run python clone_voice.py
```

You can record 10-15 seconds directly through your microphone or provide an existing audio file (`.wav`, `.mp3`, etc.):

```bash
# Clone from an audio file:
uv run python clone_voice.py --name alex --audio sample.wav

# Clone from microphone with custom duration:
uv run python clone_voice.py --name alex --record --duration 12

# List all cloned voices:
uv run python clone_voice.py --list
```

> [!NOTE]
> Pocket TTS voice cloning uses gated weights from [Hugging Face (kyutai/pocket-tts)](https://huggingface.co/kyutai/pocket-tts). Simply accept the terms on Hugging Face and provide your token when prompted (or via `HF_TOKEN`).

### 2. Run with your cloned voice

When you launch with `./run.sh` (or `run.bat`), it detects your cloned voices and asks if you would like to use one:

```text
Cloned Voices Detected:
  [0] Default voice
  [1] alex (cloned)

Would you like to use a cloned voice for your assistant? [0-1, default: 0]: 1
✓ Selected voice: alex
```

Or pass `--voice <name>` directly:

```bash
./run.sh --voice alex
# or
uv run python -m services.conversation.cli --voice alex
```

## Start Ollama (Manual)

The LLM service requires Ollama to be running before the app starts.

```
ollama serve
```

If Ollama is already running as a background service (e.g., started automatically on
login), you can skip this step.

---

## Start the API server

From the project root:

```
uv run python -m services.api.app
```

You should see output similar to:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:3000 (Press CTRL+C to quit)
```

Swagger docs are available at:

```
http://localhost:3000/docs
```

The **browser webapp** is served from the same port — no separate server needed:

```
http://localhost:3000/ui
```

All services and the webapp are on this same server and port.

---

## Stop the app

Press `Ctrl+C`. The server shuts down and all services are stopped cleanly.

---

## Main endpoints

| Area         | Method + Path               | Purpose |
|--------------|-----------------------------|---------|
| **Webapp**   | `GET /ui`                   | Browser voice chat UI |
| Health       | `GET /health`               | Service health and API port |
| LLM          | `POST /llm/generate`        | Generate one response from chat messages |
| TTS          | `POST /tts/speak`           | Queue text for speech (server-side playback) |
| TTS          | `POST /tts/synthesize`      | Synthesize text and return WAV for browser playback |
| STT          | `POST /stt/push`            | Push audio samples for transcription |
| STT          | `GET /stt/transcripts`      | Read captured transcript events |
| STT          | `DELETE /stt/transcripts`   | Clear transcript events |
| Conversation | `POST /conversation/respond`| Submit finalized user text into conversation flow |
| Conversation | `GET /conversation/history` | Read conversation history |
| Conversation | `DELETE /conversation/history` | Clear conversation history |
| Audio        | `GET /audio/status`         | Playback state |

---

## Configuration

Use environment variables to configure the API bind address and port.

```bash
VOICECHAT_API_HOST=0.0.0.0 VOICECHAT_API_PORT=3000 uv run python -m services.api.app
```

For browser clients, CORS is enabled by default for all origins (`*`).
To restrict origins, pass a comma-separated allowlist:

```bash
VOICECHAT_CORS_ALLOW_ORIGINS=http://localhost:8080,http://127.0.0.1:5500 uv run python -m services.api.app
```

| Setting        | Location                                     | Default              |
|----------------|----------------------------------------------|----------------------|
| API host       | `constants.py` (`VOICECHAT_API_HOST`)        | 0.0.0.0              |
| API port       | `constants.py` (`VOICECHAT_API_PORT`)        | 3000                 |
| CORS origins   | `constants.py` (`VOICECHAT_CORS_ALLOW_ORIGINS`) | *                 |
| Ollama URL     | `services/llm/src/llm_provider.py`           | http://172.19.96.1:11434 |
| Ollama model   | `services/llm/src/llm_provider.py`           | qwen3.5:4b           |
| TTS voice      | `services/tts/src/tts_service.py`            | af_sarah             |
| TTS speed      | `services/tts/src/tts_service.py`            | 1.0                  |
| STT language   | `services/stt/src/stt_service.py`            | en                   |
| Silence gate   | `constants.py` (`VOICECHAT_GAP_TIMEOUT`, `VOICECHAT_QUICK_FIRE_TIMEOUT`) | 2.0 s gap, 0.8 s quick-fire |

---

## Troubleshooting

**"Unable to connect to Ollama"**
Ollama is not running or is on a different address. Start it with `ollama serve` and
check `LLMProviderConfig.base_url` in `services/llm/src/llm_provider.py`.

**"Address already in use"**
Another process is already using the configured API port. Change `VOICECHAT_API_PORT`
to an open port.

**No audio input / output**
Run `uv run python -c "import sounddevice; print(sounddevice.query_devices())"` to list
devices. On Linux/WSL2 ensure PortAudio is installed (`sudo apt install libportaudio2`).

**Moonshine model download is slow**
The Moonshine model is downloaded automatically on first start and cached locally.
Subsequent starts are fast.

**The assistant does not respond**
Check the terminal for logged warnings. A common cause is the STT transcribing silence
or noise; the sentence gate will discard empty transcripts automatically.
