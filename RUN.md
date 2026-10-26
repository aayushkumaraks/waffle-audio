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

> [!WARNING]
> **Experimental & Unstable**: The emotional voice assistant is currently in an experimental, unstable state. Live multi-threaded voice model switching, memory offloading under high load, and prompt compliance with emotion tags may vary depending on hardware and LLM model size.

### How It Works

```
User Speech --> STT (Moonshine) --> LLM with [emotion] prompt --> EmotionParser
                                                                        |
                                                    +-------------------+-------------------+
                                                    |                   |                   |
                                            [neutral worker]    [happy worker]      [sad/angry worker]
                                            (Thread 1 / RAM)    (Thread 2 / RAM)    (Thread 3,4 / RAM)
                                                    \                   |                   /
                                                     +------------------+------------------+
                                                                        |
                                                                  AudioPlayer
```

1. **Voice Conditioning & Model Extraction**:
   - Audio clips for 4 core emotions (`neutral`, `happy`, `sad`, `angry`) across female (CREMA-D actor 1003) and male (actor 1011) speakers are processed using PocketTTS's Mimi speech encoder.
   - Acoustic conditioning tensors are saved into individual `cloned_voices/<emotion>_<gender>.safetensors` files.
2. **Single-Gender Model Activation**:
   - At startup, the user chooses either **Female** or **Male**. Only the selected gender's 4 emotion models are loaded into memory and worker threads.
3. **Thread-Per-Emotion Synthesis Architecture**:
   - Each emotion model is assigned a dedicated worker thread (`tts-worker-neutral`, `tts-worker-happy`, `tts-worker-sad`, `tts-worker-angry`) with its own synthesis task queue.
   - Models are loaded resident in memory for sub-10ms switching latency.
4. **Intelligent Memory Offloading**:
   - An idle watchdog tracks when each model was last used. If a non-neutral emotion model remains idle beyond the timeout (`--offload-idle <seconds>`, default 120s), its tensors are offloaded from RAM and garbage collected.
   - When an offloaded emotion is requested again, the model auto-reloads from disk in milliseconds. The neutral model remains resident as a baseline fallback.
5. **LLM Emotion Parsing & Speech Routing**:
   - The LLM is instructed via system prompt to prepend emotional tags (`[happy]`, `[sad]`, `[angry]`, `[neutral]`).
   - `EmotionParser` and `EmotionConversationManager` stream-parse sentences, strip brackets so the user hears natural speech without bracket syntax, and route the text to the matching emotion worker thread.

### Emotion Features
- **Gender Voice Selection**: Choose either **Female** or **Male** voice model set at startup.
- **4 Emotion States**: Every emotion (`neutral`, `happy`, `sad`, `angry`) has its own `.safetensors` model in `cloned_voices/`.
- **Thread-Per-Emotion Workers**: Each emotion model runs in a dedicated thread (`tts-worker-neutral`, `tts-worker-happy`, `tts-worker-sad`, `tts-worker-angry`).
- **In-Memory Caching with Automatic Offloading**: Models stay in RAM for rapid response and automatically offload to release memory when idle (`--offload-idle <seconds>`).
- **Real-Time LLM Emotion Tagging & Parsing**: The LLM specifies its emotional tone via tags like `[happy]`, `[sad]`, `[angry]`, or `[neutral]`, which are dynamically parsed to route speech chunks to the matching emotion voice worker.
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

Run the complete 111-test suite across all services and tools:

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
