# Running Waffle Audio

Quick reference for running Waffle Audio. For full details, see the [Running & CLI Guide](docs/running.md).

---

## 1. One-Click Start (CLI App)

The fastest way to install dependencies, verify models, and start the voice assistant:

```bash
./run.sh
```

On Windows:
```cmd
run.bat
```

> [!TIP]
> On your first run, `./run.sh` automatically launches the interactive **Setup Wizard** (`setup.py`) to personalize your assistant and detect your microphone and Ollama setup. You can re-run the wizard anytime with `./run.sh --setup` or `uv run python setup.py --force`.

Or start the CLI app directly with `uv`:

```bash
uv run python -m services.conversation.cli
```

---

## 2. Common CLI Flags

| Command / Flag | Purpose |
|---|---|
| `./run.sh -v` / `--verbose` | Show real-time turn latency timeline and compare with benchmarks |
| `./run.sh --benchmark` | Enable latency metrics, tool execution logging, and session averages |
| `./run.sh --trigger-mode` | Only respond when wake phrase `"Hey <ASSISTANT_NAME>"` is spoken |
| `./run.sh --voice <name>` | Start with a specific voice or cloned voice from `cloned_voices/` |
| `./run.sh --setup` | Re-run the interactive setup wizard before starting |

---

## 3. Emotional Voice Assistant (Multi-Emotion Voice Models)

Run the emotion-aware voice assistant with dynamic emotion detection, FIFO synthesis over cached `.safetensors` voice states, playback-synced emotion badges, and automatic idle offloading:

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

# Re-run interactive .env / Ollama host setup (useful on WSL):
./run-with-emotion.sh --setup

# Enable verbose latency profiling and README benchmark timeline:
./run-with-emotion.sh --verbose
uv run python -m services.conversation.emotion_cli --gender female --verbose
```

> [!WARNING]
> **Experimental**: The emotional voice assistant is currently in an experimental state. Prompt compliance with emotion tags and switching latency may vary depending on hardware and Ollama model size.

### How It Works

```
User Speech --> STT (Moonshine) --> LLM with [emotion] prompt --> EmotionParser (split_by_emotion)
                                                                        |
                                                       FIFO Synthesis Queue (EmotionTTSService)
                                                        ├── [neutral] voice state (pinned in RAM)
                                                        ├── [happy]   voice state (cached / auto-offload)
                                                        ├── [sad]     voice state (cached / auto-offload)
                                                        └── [angry]   voice state (cached / auto-offload)
                                                                        |
                                                          AudioPlayer (with live emotion sync)
                                                                        |
                                                      EmotionConversationUI (per-segment badges)
```

1. **Voice Conditioning & Model Extraction**:
   - Audio clips for 4 core emotions (`neutral`, `happy`, `sad`, `angry`) across female (CREMA-D actor 1003) and male (actor 1011) speakers are processed using PocketTTS's Mimi speech encoder.
   - Acoustic conditioning tensors are saved into individual `cloned_voices/<emotion>_<gender>.safetensors` files.
2. **Single-Gender Model Activation**:
   - At startup, the user chooses either **Female** or **Male**. Only the selected gender's 4 emotion voice states are loaded into memory.
3. **FIFO Synthesis with Cached Voice States**:
   - A single FIFO synthesis worker (`tts-emotion-worker`) processes sentences in exact order across emotion switches, preventing out-of-order audio and `AudioPlayer` state races while supporting `is_busy` and `interrupt()`.
   - Resident voice states switch in sub-10ms without reloading the PocketTTS engine.
4. **Intelligent Memory Offloading**:
   - An idle watchdog tracks when each voice state was last used. If a non-neutral voice state remains idle beyond `--offload-idle <seconds>` (default `120s`), its tensors are released from RAM.
   - When an offloaded emotion is requested again, it reloads from disk automatically. The `neutral` state remains resident as a baseline fallback.
5. **LLM Emotion Parsing, History Retention & Live UI Sync**:
   - `EMOTION_SYSTEM_PROMPT` instructs the LLM when to choose `[neutral]`, `[happy]`, `[sad]`, or `[angry]`. Internal LLM history preserves tags across turns to prevent multi-turn tag drift, while TTS and UI display clean spoken text.
   - `EmotionParser` splits multi-emotion chunks in order (and normalizes common synonyms like `[cheerful]`, `[frustrated]`, `[sorry]`). `AudioPlayer` forwards the active emotion per spoken chunk so the UI displays playback-synced badges and per-segment colored history.

### Emotion Features
- **Gender Voice Selection**: Choose either **Female** or **Male** voice model set at startup (or via `--gender`).
- **4 Emotion States**: Every emotion (`neutral`, `happy`, `sad`, `angry`) has its own `.safetensors` model in `cloned_voices/`.
- **Strict FIFO Multi-Emotion Playback**: Sentences always play in order even when emotions change mid-reply.
- **In-Memory Caching with Automatic Offloading**: Models stay in RAM for rapid response and automatically offload when idle (`--offload-idle <seconds>`).
- **Live Playback-Synced UI**: Header badge, footer status, and per-segment chat history reflect the exact emotion being spoken without duplicate rows.
- **Verbose Latency Profiling**: Pass `--verbose` or `-v` to show live pipeline latency metrics and compare against the README warm turn benchmarks.

---

## 4. Voice Cloning

Clone any custom voice in 10–15 seconds:

```bash
uv run python clone_voice.py
```

See [docs/voice-cloning.md](docs/voice-cloning.md) for full instructions.

---

## 5. MCP Tools & Search

Manage installed Model Context Protocol (MCP) tools and web search:

```bash
uv run python install_tool.py
```

See [docs/tools-and-mcp.md](docs/tools-and-mcp.md) for presets and options.

---

## 6. HTTP API Server & Web App UI

Start the headless FastAPI server process:

```bash
uv run python -m services.api.app
```

Interactive Swagger documentation is available at: `http://localhost:3000/docs`

To start the React Web UI:

```bash
bash ./tools/webapp/start.sh
```

Or open `http://localhost:3000/ui` in your browser when the API server is active with the built webapp bundle.

---

## 7. Running Unit Tests

Run the complete 200+ test suite across all services and tools:

```bash
# Run via pytest (recommended)
PYTHONPATH=. uv run pytest tests/

# Or run via Python unittest runner
uv run python -m unittest discover tests
```

---

## 8. Full Documentation

- **[Installation & Setup](docs/setup.md)**
- **[CLI & Running Guide](docs/running.md)**
- **[Configuration Reference](docs/configuration.md)**
- **[System Architecture](docs/architecture.md)**
- **[Audio Processing](docs/audio-processing.md)**
- **[Tools & MCP](docs/tools-and-mcp.md)**
- **[Latency Benchmarks](docs/benchmarks.md)**
- **[Voice Cloning](docs/voice-cloning.md)**
