# Waffle-Audio

**Talk to a local AI the way you talk to a person. Private, fast, and fully yours.**

Waffle-Audio is a real-time, voice-to-voice assistant that runs entirely on your local machine. Speak naturally, and it answers out loud in **sub-second latency (0.83s)**. No cloud APIs, no subscriptions, and your voice never leaves your computer.

---

## Highlights

- **Private & 100% Offline**: Speech recognition (Moonshine), language model (Ollama), and speech synthesis (Pocket TTS / Kokoro) run locally on your device.
- **Dynamic Ollama Model Discovery**: Interactive setup wizard queries Ollama's live `/api/tags` endpoint to present a numbered model selector, pre-populating your current settings.
- **Sub-Second Response (0.83s)**: Adaptive clause chunking streams speech tokens into voice on the first ~3 words, starting playback before the LLM finishes generating.
- **Instant Interruption (Barge-In)**: Acoustic Echo Cancellation (AEC) subtracts speaker audio from the microphone so you can interrupt Waffle mid-sentence with <50ms reaction.
- **Understands Natural Turn-Taking**: Adaptive sentence gate distinguishes quick punctuation pauses (250ms) from natural thinking pauses ("I want to... uh... check the weather") without interrupting you prematurely.
- **Model Context Protocol (MCP) & Web Search**: Dynamically invokes installed MCP tools (SQLite, filesystem, GitHub, time) and automatically falls back to DuckDuckGo search for real-time news and facts.
- **10-Second Voice Cloning**: Clone any voice from a quick 10–15s mic recording or audio file to give your assistant a custom persona.
- **Asynchronous Context Compactor**: Background summarizer rolls older conversation history into compact memory, preventing small LLMs from slowing down over long sessions.
- **Comprehensive Unit Test Suite**: 211 deterministic unit tests covering every service and script with 100% test passing rate across `pytest` and `unittest`.
- **Rich Terminal UI**: Animated ASCII waveforms, dynamic auto-scrolling dialogue, status badges, and real-time latency timelines.

---

## Performance Snapshot

Measured directly from live sessions on the default local stack (Moonshine STT, Ollama `qwen3.5:4b`, Pocket TTS):

| Pipeline Stage / Metric | Baseline | Target | Measured Live | Delta |
|---|---:|---:|---:|---|
| **Sentence Gate (Punctuation)** | 0.80 s | 0.40 s | **0.25 s** | *-0.15 s* |
| **LLM Time To First Token (TTFT)** | 0.70 s | 0.45 s | **0.35 s** | *-0.10 s* |
| **TTS First Audio Out (TTFB)** | 1.80 s | 1.00 s | **0.58 s** | *-0.42 s* |
| **Total Conversational E2E Latency** | **3.30 s** | **1.60 s** | **0.83 s** | **~75% faster (sub-second!)** |
| **Average Warm Multi-Sentence Turn** | 3.30 s | 2.10 s | **1.12 s** | **~66% faster** |

*For detailed latency stage breakdowns and profiling logs, see [docs/benchmarks.md](docs/benchmarks.md).*

---

## Quickstart (One-Click)

The fastest way to install dependencies, verify your environment, and start talking:

### Linux / WSL2:
```bash
./run.sh
```

### Windows:
```cmd
run.bat
```

> [!NOTE]
> On first launch, `./run.sh` automatically runs the interactive **Setup Wizard** (`setup.py`) to personalize your assistant, detect audio devices, and query Ollama for available models. You can re-run setup anytime with `./run.sh --setup` or `uv run python setup.py --force`.

Or launch directly with `uv`:
```bash
uv run python -m services.conversation.cli
```

---

## Common CLI Options

```bash
# Real-time latency profiling timeline in the terminal UI
./run.sh --verbose

# Enable structured JSONL latency tracking and session averages
./run.sh --benchmark

# Wake phrase mode: only responds when "Hey [ASSISTANT_NAME]" is spoken
./run.sh --trigger-mode

# Launch with a specific custom or cloned voice
./run.sh --voice my_voice

# Reconfigure .env settings before launch
./run.sh --setup
```

---

## Core Capabilities

### 1. Voice Cloning & Custom Voices
Clone any voice in seconds from your microphone or an audio file:
```bash
uv run python clone_voice.py
```
Cloned voices are saved to `cloned_voices/<name>.safetensors` and detected automatically on startup. See [docs/voice-cloning.md](docs/voice-cloning.md).

### 2. Tools & Web Search (Model Context Protocol)
Install and manage external tools with the built-in installer:
```bash
uv run python install_tool.py --preset fetch      # Web scraper & reader
uv run python install_tool.py --preset sqlite     # Query SQLite databases
uv run python install_tool.py --preset time       # Current time & timezones
uv run python install_tool.py --preset filesystem # Local file inspection
uv run python install_tool.py --preset github     # GitHub repos & issues
```
If no installed tool matches, Waffle automatically queries DuckDuckGo search and speaks the answer aloud with conversational bridge phrases. See [docs/tools-and-mcp.md](docs/tools-and-mcp.md).

### 3. Headless HTTP API
Start the FastAPI backend service for network integration:
```bash
uv run python -m services.api.app
```
Interactive Swagger docs: `http://localhost:3000/docs`. See [docs/running.md](docs/running.md).

### 4. Running Unit Tests
Waffle Audio features a comprehensive 211-test suite with 100% pass rate covering all services and scripts:
```bash
# Run complete test suite with pytest
uv run pytest

# Or run via Python standard unittest discovery
uv run python -m unittest discover tests
```

---

## Project Structure

```
├── run.sh                  # One-click installer & launcher (Linux/WSL)
├── run.bat                 # One-click launcher (Windows)
├── setup.py                # Interactive setup wizard with Ollama API model discovery
├── clone_voice.py          # Interactive voice cloning CLI
├── install_tool.py         # MCP tool installer & DuckDuckGo search tester
├── install_tool.sh         # Convenience launcher for install_tool.py
├── constants.py            # Environment loader & central fallback configuration
├── .env.example            # Annotated template configuration file
├── mcp_servers.json        # Configured Model Context Protocol servers
├── services/               # Core application services
│   ├── conversation/       # Conversation manager, sentence gate, clause chunker, UI
│   ├── audio/              # Hardware audio output & playback interrupt
│   ├── audio_processing/   # Silero VAD, Acoustic Echo Canceller (NLMS), Barge-in
│   ├── stt/                # Moonshine streaming speech-to-text
│   ├── llm/                # Ollama streaming provider & intent-gated tools
│   ├── tts/                # Pocket TTS & Kokoro streaming speech synthesis
│   ├── tools/              # MCP client manager, search engine & bridge phrases
│   ├── api/                # FastAPI HTTP service & endpoints
│   └── pipeline.py         # Unified in-process async pipeline
├── docs/                   # Detailed documentation guides
│   ├── setup.md            # Prerequisites & system installation
│   ├── running.md          # CLI usage, trigger mode & API guide
│   ├── configuration.md    # Complete .env & constants reference
│   ├── architecture.md     # Pipeline design & technical architecture
│   ├── audio-processing.md # VAD, AEC, and Barge-In audio layer
│   ├── tools-and-mcp.md    # MCP tool calling & web search fallback
│   ├── benchmarks.md       # Latency benchmarks & profiling guide
│   └── voice-cloning.md    # Custom voice cloning walkthrough
├── cloned_voices/          # User-saved .safetensors voice embeddings
├── tests/                  # Comprehensive unit & integration test suite (211 tests)
└── pyproject.toml          # uv project dependencies & pytest configuration (Python 3.11)
```

---

## Documentation Index

- **[Installation & Setup](docs/setup.md)**: Prerequisites, PortAudio, WSLg audio, and Ollama configuration.
- **[Running & CLI Guide](docs/running.md)**: CLI flags, trigger mode, verbose profiling, and API endpoints.
- **[Configuration Reference](docs/configuration.md)**: Exhaustive reference of all `.env` options and fallback defaults.
- **[Technical Architecture](docs/architecture.md)**: Streaming pipeline, latency optimizations, and event flow.
- **[Audio Processing](docs/audio-processing.md)**: Silero VAD, NLMS echo cancellation, and sub-50ms barge-in.
- **[Tools & MCP](docs/tools-and-mcp.md)**: Installing tools, DuckDuckGo search fallback, and bridge phrases.
- **[Performance Benchmarks](docs/benchmarks.md)**: Turn breakdowns, hardware profiling, and latency comparisons.
- **[Voice Cloning](docs/voice-cloning.md)**: Cloning custom voices from mic or audio files.
