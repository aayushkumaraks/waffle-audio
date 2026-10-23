# Waffle-Audio

**Talk to a local AI the way you talk to a person. Private, fast, and fully yours.**

Waffle-Audio is a real-time, voice-to-voice assistant that runs entirely on your local machine. Speak naturally, and it answers out loud in **sub-second latency (0.83s)**. No cloud APIs, no subscriptions, and your voice never leaves your computer.

---

## Highlights

- **Private & 100% Offline**: Speech recognition (Moonshine), language model (Ollama), and speech synthesis (Pocket TTS / Kokoro) run locally on your device.
- **Automatic Ollama & Host Discovery**: Setup wizard automatically detects reachable Ollama endpoints and WSL host IP gateways, verifying model availability before launch.
- **Sub-Second Response (0.83s)**: Adaptive clause chunking streams speech tokens into voice on the first ~3 words, starting playback before the LLM finishes generating.
- **Instant Interruption (Barge-In)**: Acoustic Echo Cancellation (AEC) subtracts speaker audio from the microphone so you can interrupt Waffle mid-sentence with <50ms reaction.
- **Understands Natural Turn-Taking**: Adaptive sentence gate distinguishes quick punctuation pauses (250ms) from natural thinking pauses ("I want to... uh... check the weather") without interrupting you prematurely.
- **Model Context Protocol (MCP) & Web Search**: Dynamically invokes installed MCP tools (SQLite, filesystem, GitHub, time) and automatically falls back to DuckDuckGo search for real-time news and facts.
- **10-Second Voice Cloning**: Clone any voice from a quick 10–15s mic recording or audio file to give your assistant a custom persona.
- **Asynchronous Context Compactor**: Background summarizer rolls older conversation history into compact memory, preventing small LLMs from slowing down over long sessions.
- **Comprehensive Unit Test Suite**: 111 deterministic unit tests covering core services, tools, setup, and CLI with 100% test passing rate across `pytest` and `unittest`.
- **Rich Terminal UI & Web App**: Interactive CLI with animated ASCII waveforms, dynamic auto-scrolling dialogue, and a modern React Web UI single-page application.

---

## Performance & Latency Benchmarks

Waffle Audio is profiled to achieve sub-second, human-like conversational responsiveness on local consumer hardware.

All metrics are measured directly from live sessions on the default local stack:
- **Speech Recognition (STT)**: Moonshine (streaming local ONNX)
- **Language Model (LLM)**: Ollama `qwen3.5:4b` (GPU accelerated, 2048 context window)
- **Speech Synthesis (TTS)**: Pocket TTS (`anna` voice)
- **Audio Pipeline**: Unified in-process event loop with raw NumPy array buffers

### 1. End-to-End Conversational Latency

End-to-End (E2E) latency measures the elapsed time from the exact millisecond user speech ceases until the first sound waves leave the speaker:

| Conversational Mode / Turn Type | Unoptimized Baseline | Target Reference | Measured Live | Improvement vs Baseline |
|---|---:|---:|---:|---|
| **Conversational Fast Path** (Short turn, e.g. "It feels surreal talking to a robot") | 3.30 s | 1.60 s | **0.78 s – 0.83 s** | **~75% faster (sub-second!)** |
| **Average Warm Turn** (Multi-sentence detailed explanation) | 3.30 s | 2.10 s | **1.12 s – 1.23 s** | **~66% faster** |
| **First Turn (Cold Start)** (Model warmup & Hugging Face cache) | 15.00 s | 6.80 s | **5.40 s** | **~64% faster** |

---

### 2. Stage-by-Stage Latency Breakdown (`./run.sh --verbose`)

Granular timeline recorded from verified live two-turn user interaction:

| Pipeline Stage | Turn 1 (Detailed Answer) | Turn 2 (Conversational) | Target Reference | Performance Delta |
|---|---:|---:|---:|---|
| **1. Sentence Gate Silence Wait** | 0.25 s | 0.25 s | 0.40 s | **-0.15 s (Faster! ✓)** |
| **2. LLM Time To First Token (TTFT)** | 0.37 s | 0.32 s | 0.45 s | **-0.13 s (Faster! ✓)** |
| **3. LLM Full Reply Generation** | 0.87 s | 0.61 s | 1.47 s | **-0.86 s (Faster! ✓)** |
| **4. TTS Time To First Audio (TTFB)** | 1.16 s | 0.53 s | 1.82 s | **-1.29 s (Faster! ✓)** |
| **★ Total Conversational E2E Latency** | **1.41 s** | **0.78 s** | **2.10 s** | **-1.32 s (Sub-Second! ✓)** |
| **Total Turn Duration** (Full Speech Complete) | 16.72 s | 3.84 s | — | Full spoken answer |

> [!TIP]
> **How Sub-Second Playback Works**: Traditional voice assistants wait for the LLM to complete its entire answer before sending text to TTS. Waffle Audio uses **Adaptive Clause Chunking (`ClauseChunker`)**: the first clause dispatches speech synthesis on the first ~3–4 words in <200ms TTFB, streaming audio into the speaker queue while Ollama is still generating subsequent sentences.

---

### 3. Tool Calling & Real-Time Search Timings

When external tools or DuckDuckGo searches are executed, Waffle synthesizes natural conversational bridge phrases on-the-go to keep the interaction fluid:

| Tool Pipeline Stage | Metric / Timing | Description |
|---|---:|---|
| **Intent Analysis & Tool Decision** | **<25 ms** | Dynamically evaluates intent before Ollama generation |
| **Spoken Bridge Phrase Audio Start** | **~0.50 s** | Synthesizes context-aware phrase (*"Looking that up on the web..."*) |
| **DuckDuckGo Web Search Execution** | **1.88 s** (1883 ms) | Average execution time over live queries (min: 1.77s, max: 2.00s) |
| **Follow-Up LLM Time To First Token** | **0.40 s** | Ingests search findings and starts token generation |
| **Follow-Up LLM Full Response** | **1.15 s** | Generates concise, spoken-friendly factual answer |
| **Overall Tool Turn Latency** | **4.38 s** | Acknowledges query immediately with speech while background lookup completes |

---

### 4. What Makes Waffle Audio Fast

| Optimization | Architectural Mechanism | Impact |
|---|---|---|
| **Adaptive Clause Chunking** | Splits streaming LLM tokens on punctuation (`,`, `;`, `:`) and conjunctions; fast path triggers on 3 words | Begins speech playback in **<0.58s** before the LLM finishes generating |
| **Unified In-Process Event Loop** | Passes contiguous NumPy `float32` arrays directly in memory | Eliminates TCP network serialization and JSON marshaling overhead |
| **GPU VRAM Context Capping** | Limits `OLLAMA_NUM_CTX` to 2048 tokens | Keeps model weights and KV-cache 100% in GPU VRAM (3.06 GB vs 14.3 GB), preventing slow CPU offloading |
| **Ollama Model Residency** | Sets `OLLAMA_KEEP_ALIVE=30m` | Keeps the model resident in memory across turns, avoiding cold-load penalties |
| **Acoustic Echo Cancellation (AEC)** | Normalized Least Mean Squares (NLMS) filter cancels speaker output from microphone input | Enables instant barge-in without audio loopback or self-interruption |
| **Asynchronous Context Compactor** | Heuristic background worker summarizes older turns (threshold: 6 turns) | Prevents context bloating from degrading LLM generation speed |
| **Adaptive Sentence Gate** | Quick-fires on terminal punctuation in 250 ms (vs 500 ms for thinking pauses) | Eliminates 300–500 ms of dead air at the end of user utterances |
| **SIMD Audio Resampling** | Leverages `soxr` and `scipy.signal.resample_poly` polyphase FIR filters | Downsamples 48 kHz / 24 kHz streams to 16 kHz with near-zero CPU usage |

*For deep dive turn breakdowns and benchmark logging formats, see [docs/benchmarks.md](docs/benchmarks.md).*

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
> On first launch, `./run.sh` automatically runs the interactive **Setup Wizard** (`setup.py`) to personalize your assistant, detect audio devices, and configure Ollama. You can re-run setup anytime with `./run.sh --setup` or `uv run python setup.py --force`.

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

### 3. Headless HTTP API & Web App UI
Start the FastAPI backend service:
```bash
uv run python -m services.api.app
```
Interactive Swagger docs: `http://localhost:3000/docs`.

Start the React Web UI:
```bash
bash ./tools/webapp/start.sh
```
Or open `http://localhost:3000/ui` when the API server is running with the built bundle. See [tools/webapp/README.md](tools/webapp/README.md).

### 4. Running Unit Tests
Waffle Audio features a comprehensive 111-test suite with 100% pass rate:
```bash
# Run complete test suite with pytest
PYTHONPATH=. uv run pytest tests/

# Or run via Python standard unittest discovery
uv run python -m unittest discover tests
```

---

## Project Structure

```
├── run.sh                  # One-click installer & launcher (Linux/WSL)
├── run.bat                 # One-click launcher (Windows)
├── setup.py                # Interactive setup wizard with Ollama host detection
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
├── tools/
│   └── webapp/             # React SPA web application (Vite, WebRTC, voice controls)
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
├── tests/                  # 111 unit tests covering all core services & tools
└── pyproject.toml          # uv project dependencies (Python 3.11)
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
