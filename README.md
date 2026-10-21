# Waffle-Audio

**Talk to a local AI the way you'd talk to a person. Private, fast, and fully yours.**

Waffle-Audio is a real-time voice-to-voice assistant that runs on your own machine.
Speak naturally, and it answers out loud in about two seconds. No cloud APIs, no
accounts, no subscriptions, and your voice never leaves your computer.

## Why Waffle-Audio

- **Private by design.** Speech recognition, the language model and the voice all run
  locally. Your conversations are never uploaded anywhere.
- **Feels conversational.** Replies are streamed sentence by sentence into speech, so
  Waffle starts talking while it is still thinking.
- **Understands how people actually talk.** An adaptive sentence gate with clause chunking waits through natural pauses ("I want to... uh... test how fast you are") and synthesizes speech in rapid clause chunks without interrupting you prematurely.
- **Natural turn-taking with instant barge-in.** Acoustic Echo Cancellation (AEC) cancels speaker audio from the microphone, allowing you to interrupt Waffle at any time mid-sentence with sub-50ms reaction.
- **Offline after first run.** Models are loaded from the local cache; the network is
  only touched when a model is missing.
- **Modular and hackable.** STT, LLM and TTS are independent services. Swap the model,
  the voice or the backend without touching the rest.
- **Terminal or web.** Use the live voice CLI, the HTTP/WebRTC API, or the web app.

## Performance

Measured directly from a real live session on the default stack (Moonshine STT, Ollama `qwen3.5:4b`, Kokoro/Pocket TTS), comparing initial baselines against the latest in-process async pipeline and adaptive clause-gated streaming.

### End-to-end latency (end of your speech to Waffle starting to speak)

| Version | Avg warm turn | Conversational fast-path | First turn (cold start) |
|---|---:|---:|---:|
| Before optimisation | 3.3 s | — | 15.0 s |
| Initial target | 2.1 s | 1.6 s | 6.8 s |
| **Current (Measured Live)** | **1.12 s** | **0.83 s** | **5.4 s** |
| Overall Improvement | **~66% faster** | **~75% faster (sub-second)** | **~64% faster** |

### Verified Live Session Turn Breakdown (`./run.sh --verbose`)

Verified live pipeline measurements from real two-turn user interaction:

| Pipeline Stage | Turn 1 (Longer Response) | Turn 2 (Conversational) | Target Reference | Performance vs Target |
|---|---:|---:|---:|---|
| **1. Sentence Gate** | 0.25 s | 0.25 s | 0.40 s | **-0.15s (Faster! ✓)** |
| **2. LLM First Token (TTFT)** | 0.37 s | 0.35 s | 0.45 s | **-0.10s (Faster! ✓)** |
| **3. LLM Full Reply** | 0.87 s | 0.61 s | 1.47 s | **-0.86s (Faster! ✓)** |
| **4. TTS First Speech** | 1.16 s | 0.58 s | 1.82 s | **-1.24s (Faster! ✓)** |
| **5. Total E2E Latency** | **1.41 s** | **0.83 s** | **2.10 s** | **-1.27s (Faster! ✓)** |

Notes:

- Turn #1 ("Hello, I'm checking whether this is working...") generated a multi-sentence response in 1.41s E2E.
- Turn #2 ("It feels surreal talking to a robot.") reached full speech playback in **0.83s E2E**, well below the 1.0s threshold.
- STT runs continuously while you talk, with silence detection folded directly into the 0.25s sentence gate wait.
- Results depend on your hardware and Ollama host GPU.

### What makes it fast

| Optimisation | Effect |
|---|---|
| Adaptive clause chunking & fast path | Speech starts on the first ~3-4 words (<200ms TTFB), streaming ahead of full sentence |
| Unified in-process async pipeline | Zero network/TCP serialization latency; passes raw NumPy audio buffers in-memory |
| Capped Ollama context (2048 tokens) | Keeps model 100% in GPU VRAM (3.06 GB vs 14.3 GB), preventing slow CPU KV offloading |
| Acoustic Echo Cancellation (AEC) & Barge-in | Natural two-way conversation; user can interrupt at any point without audio feedback |
| Optimized VAD & pre-roll ring buffer | 200ms trailing silence cut dead air; 200ms pre-roll preserves initial unvoiced consonants |
| Polyphase FIR & SIMD resampling | Ultra-low CPU overhead downsampling audio via `soxr` / `scipy.signal.resample_poly` |
| Ollama keep-alive and background warm-up | First-turn wait cut from ~15 s to ~5.4 s |
| Sliding context window (6–8 turns) | Keeps prompt history compact (max 16 messages), preventing LLM token slowdown |
| Faster sentence gate | Terminal punctuation quick-fires in 250ms; unpunctuated sentences wait only 500ms |
| Local-first model loading | No network request on startup |

## How it works

```
Microphone --> [ NLMS Echo Canceller ] --> VAD (Silero) --> STT (Moonshine) --> ClauseGate / LLM (Ollama) --> TTS (Kokoro / Pocket TTS) --> Speakers
                      ^                                                                                                                         |
                      +----------------------------- (Playback Reference Audio Loopback) -------------------------------------------------------+
```

When speech is detected during assistant playback or generation, the **BargeInController** instantly stops playback, aborts streaming LLM inference, and purges the TTS queue.

| Stage | Technology         | Notes                                      |
|-------|--------------------|--------------------------------------------|
| STT   | Moonshine          | Streaming speech-to-text, runs on CPU/GPU  |
| LLM   | Ollama (local)     | Default model: qwen3.5:4b                  |
| TTS   | Kokoro / Pocket TTS | Selectable streaming neural text-to-speech |
| Audio | sounddevice / PortAudio | Cross-platform audio I/O with AEC loopback |

## Key features

- **Instant Interruption (Barge-In)**: Speak at any time to interrupt Waffle mid-sentence. The VAD and AEC detect your voice, stopping speaker playback in <50ms, aborting LLM generation, and clearing TTS queues.
- **Acoustic Echo Cancellation (AEC)**: Normalized Least Mean Squares (NLMS) filter cancels speaker output from the microphone feed so Waffle never interrupts itself or loops back speech.
- **Adaptive Clause Chunking**: `ClauseChunker` splits streaming LLM tokens on punctuation (`,`, `;`, `:`) and conjunctions (`and`, `but`, `because`), with a first-chunk fast path (3 words) for sub-200ms TTFB.
- **Interactive Setup Wizard & Onboarding**: Automatically detects and prompts for configuration on first run (`setup.py`), saving to a git-ignored `.env` file with seamless fallback to non-sensitive defaults in `constants.py`.
- **User Personalization & Dynamic Prompting**: Speaks with you by name (`USER_NAME`), localizes responses to your city (`USER_LOCATION`), and anchors date/time queries to your timezone (`USER_TIMEZONE`).
- **Asynchronous Background Context Compactor**: Background worker (`ConversationCompactor`) periodically rolls older conversation history into concise summaries when turn count reaches threshold (`COMPACTOR_THRESHOLD = 6`), keeping inference fast without stalling active speech.
- **Dynamic Intent-Gated Tool Calling**: Ommits heavy tool schemas from prompt context during normal conversation; dynamically injects Model Context Protocol (MCP) tool schemas only when natural search/fetch intents ("search", "find recent", "check the web", "weather", etc.) are detected.
- **Model Context Protocol (MCP) Tool Calling**: Native integration with external MCP servers (stdio and SSE) exposing standard tools for databases, filesystem, web fetching, and more.
- **DuckDuckGo Search Fallback**: If the user asks for something and no installed tool matches that function, or when real-time facts/news/weather are needed, Waffle automatically queries DuckDuckGo search to get the details.
- **Tool Installer & Manager (`install_tool.py`)**: One-click installer with curated presets (fetch, sqlite, time, filesystem, memory, github) and interactive management of MCP tools.
- **Unified In-Process Async Pipeline**: Consolidates audio, VAD, STT, LLM, TTS, and player into a single async loop with zero serialization overhead, alongside POSIX shared memory (`SharedAudioBuffer`) for multi-process deployments.
- **Speech-Targeted LLM Formatting**: System prompt enforces natural conversational speech with strictly no markdown, emojis, asterisks, or numbered lists.


## Quick Start (One-Click)

Run the one-click install & run script from the project root:

```bash
./run.sh
```

On Windows:
```cmd
run.bat
```

> [!NOTE]
> On your very first run, `./run.sh` automatically launches the interactive **Setup Wizard** (`setup.py`) to personalize your assistant and detect your microphone and Ollama setup. You can re-run the wizard anytime with `./run.sh --setup` or `uv run python setup.py --force`.

Or run directly with `uv`:
```bash
uv run python -m services.conversation.cli
```

### Setup Wizard & Personalization

When `.env` is absent, the onboarding wizard prompts you to configure:
1. **User Personalization** (no defaults required): Your name, your city/location (e.g. `New Delhi`, `San Francisco`), and your local timezone (e.g. `Asia/Kolkata`, `America/New_York`).
2. **Assistant & Audio Hardware** (with smart defaults): Assistant name (`Waffle`), voice (`anna` / cloned voices), and microphone device index.
3. **LLM Runtime & Tokens** (with defaults): Reachable Ollama base URL (auto-detected host IP on WSL2 or `http://localhost:11434`), model (`qwen3.5:4b`), and optional GitHub personal access token for tools.

Configuration is saved directly to `.env` (which is safely ignored by git). See [`.env.example`](.env.example) for manual setup.

### CLI Interface Features

- **Top-Right Status Badge**: Always shows whether the assistant is `● READY`, `🎤 LISTENING`, or `⠋ PROCESSING`.
- **Real-Time Conversation**: Live transcription appears as you speak, and assistant responses stream word-by-word synchronously as the audio plays.
- **Dynamic Auto-Scrolling**: The conversation window automatically scrolls down so newest utterances and live responses are always visible, indicating hidden lines with a border subtitle.
- **Multi-Line Speech Accumulation**: Continuous speech across multiple Moonshine segments is seamlessly accumulated into a single turn without fragmentation.
- **Trigger Mode (`--trigger-mode`)**: When enabled (e.g. `./run.sh --trigger-mode`), Waffle only processes and responds when the wake phrase `"Hey <ASSISTANT_NAME>"` is spoken (e.g. `"Hey Waffle, what's the weather?"`), ignoring background speech.
- **ASCII Animations & Loaders**: Animated audio waveforms in the header and visual spinners/loaders during speech synthesis and model inference.
- **No Log Spam**: All backend and driver logs are cleanly redirected to `logs/voicechat.log`, keeping the terminal UI pristine.
- **Verbose Latency Profiling**: Pass `--verbose` (e.g. `./run.sh --verbose`) to show a real-time pipeline latency timeline in the UI and compare against README benchmarks.
- **Conversation Summary**: Full conversation history and benchmark timeline are displayed when you exit with `Ctrl+C`.

### Voice Cloning & Custom Voices

Clone any voice to give your assistant a custom voice:

```bash
uv run python clone_voice.py
```

Record 10-15 seconds from your mic or supply an audio file (`.wav`, `.mp3`). `./run.sh` and `run.bat` automatically detect your cloned voices and prompt you on startup, or you can start with `--voice <name>`:

```bash
./run.sh --voice my_voice
```

### Tools & MCP (Model Context Protocol) Integration

Waffle Audio integrates with the **Model Context Protocol (MCP)**, allowing your assistant to invoke tools whenever you ask questions or need real-time data:

- **Dynamic Intent Gating**: Tool schemas are omitted during standard chit-chat to keep LLM context light and latency fast. When you speak a tool-related keyword or phrase ("search the web for...", "look up recent news in...", "check the weather", "what is today's date?"), tool instructions and schemas are injected dynamically.
- **Installed MCP Tools**: If an installed MCP tool matches the requested function (e.g. SQLite database, filesystem, web fetcher, time), Waffle calls that tool directly.
- **DuckDuckGo Search Fallback**: If no installed tool matches the requested function, or when general web search / real-time facts are needed, Waffle automatically searches DuckDuckGo and reads the answer aloud.

#### Installing Tools to the App

Use the included tool installer script (`install_tool.py` or `./install_tool.sh`):

```bash
# Launch the interactive tool installer
uv run python install_tool.py

# Or install from curated presets directly:
uv run python install_tool.py --preset fetch       # Web page reader & markdown extractor
uv run python install_tool.py --preset sqlite      # Query SQLite database tables
uv run python install_tool.py --preset time        # Accurate time & timezone conversions
uv run python install_tool.py --preset filesystem  # Local file inspection
uv run python install_tool.py --preset memory      # Persistent knowledge graph memory
uv run python install_tool.py --preset github      # Inspect GitHub repos, issues, and PRs

# List installed tools & fallback status
uv run python install_tool.py --list

# Test DuckDuckGo search fallback
uv run python install_tool.py --test-search "current space discoveries"
```

All installed tools are saved in `mcp_servers.json` in standard MCP format and automatically discovered on startup.

#### Spoken Bridge Phrases (On-The-Go TTS)

While a tool call or DuckDuckGo search runs in the background, Waffle immediately speaks conversational bridge phrases with natural inflection (e.g. *"Looking that up for you..."*, *"Searching for it on the web..."*):
- **Synthesized On The Go**: Generated directly through the active TTS pipeline (Pocket TTS or Kokoro) using the exact same configured voice or cloned voice model, ensuring audio consistency without any pre-saved clips.
- **Context-Aware Selection**: Categorizes actions into `search`, `database`, `filesystem`, `time`, `fetch`, and `general` to pick the most accurate phrase for the invoked tool.
- **De-duplication & Randomization**: Remembers recently used phrases to cycle through similar alternatives without repeating itself.
- **Preview Phrases**:
  ```bash
  uv run python install_tool.py --preview-phrases # View phrases and categories
  ```

### Tuning & Configuration Reference

All options can be configured in `.env` (created automatically by `setup.py`) or passed as environment variables. If `.env` is absent, `constants.py` supplies non-sensitive fallback defaults.

| Environment variable | Default | Purpose |
|---|---|---|
| `USER_NAME` | `""` | User's preferred name for assistant personalization |
| `USER_LOCATION` | `""` | User's city/location (appended to general weather/news queries) |
| `USER_TIMEZONE` | `""` | User's timezone identifier (e.g. `America/New_York`, `Asia/Kolkata`) |
| `ASSISTANT_NAME` | `Waffle` | Voice assistant persona name |
| `AUDIO_INPUT_DEVICE` | `default` | Audio input device index or name (from PortAudio/sounddevice) |
| `AUDIO_OUTPUT_DEVICE` | `default` | Audio output playback device index or name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint (auto-detected host IP in WSL2) |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Ollama model name |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded in GPU VRAM |
| `OLLAMA_NUM_PREDICT` | `160` | Maximum tokens per reply |
| `VOICECHAT_TTS_BACKEND` | `pocket` | `pocket` or `kokoro` |
| `VOICECHAT_TTS_VOICE` | `anna` | Default TTS voice or cloned voice name |
| `VOICECHAT_GAP_TIMEOUT` | `0.8` | Seconds of silence before finalizing user turn |
| `VOICECHAT_QUICK_FIRE_TIMEOUT` | `0.3` | Silence timeout when sentence ends with terminal punctuation |
| `VOICECHAT_AEC_ENABLED` | `true` | Enable acoustic echo cancellation for microphone loopback |
| `VOICECHAT_BARGE_IN_ENABLED` | `true` | Enable instant interruption on detected user speech |
| `VOICECHAT_TRIGGER_MODE` | `false` | When true, only responds when 'Hey ASSISTANT_NAME' is spoken |
| `VOICECHAT_TOOL_CALLING_ENABLED` | `true` | Enable MCP and DuckDuckGo tool execution |
| `VOICECHAT_COMPACTOR_ENABLED` | `true` | Run background context compactor for long conversations |
| `VOICECHAT_COMPACTOR_THRESHOLD` | `6` | Number of turns before triggering background compaction |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | `""` | GitHub personal access token for GitHub MCP server preset |

## Project structure

```
.env.example          Template environment file
setup.py              Interactive onboarding and environment setup wizard
install_tool.py       CLI installer and manager for MCP server tools
install_tool.sh       Bash runner for install_tool.py
mcp_servers.json      Model Context Protocol server configuration
constants.py          Environment loader and non-sensitive fallback defaults
services/
  conversation/       Orchestrates STT -> LLM -> TTS
    src/
      manager.py          ConversationManager & sliding history window
      compactor.py        Async background context compactor for small LLMs
      sentence_gate.py    Adaptive silence-gap gate & ClauseChunker
      ui.py               Rich terminal UI controller & renderer
    cli.py                Live microphone voice assistant with AEC & Barge-In
  tools/              MCP tool integration & web search engine
    src/
      manager.py          ToolManager (orchestrates MCP tools & search fallback)
      mcp_client.py       Client for Model Context Protocol servers
      search.py           DuckDuckGo web, news, and weather search fallback
      phrases.py          Spoken bridge phrases during tool calls
  audio_processing/
    src/
      vad_service.py      Silero VAD with speech callbacks & pre-roll ring buffer
      echo_cancellation.py NLMS Acoustic Echo Canceller with delay reference
      barge_in.py         BargeInController (stops playback, aborts LLM, purges TTS)
      resampler.py        Fast polyphase / soxr SIMD audio resampling
  stt/                Moonshine streaming transcription wrapper
  llm/                Ollama HTTP provider + streaming abort & intent-gated tools
  tts/                Feature-flagged Kokoro and Pocket TTS services with queue purge
  audio/              Blocking audio playback worker with instant interrupt
  pipeline.py         Unified in-process async pipeline
  shared_memory.py    Zero-copy POSIX shared memory audio ring buffer
models/               Shared data models (Message)
voiceModels/          Kokoro ONNX model files (not committed)
pyproject.toml        Dependencies managed by uv
```

## Quick start

See [SETUP.md](SETUP.md) for first-time installation and [RUN.md](RUN.md) to start the
app.

## HTTP API

All services are exposed on one HTTP server process.

- Default port: `3000`
- Configurable via environment variable: `VOICECHAT_API_PORT`
- Host configurable via: `VOICECHAT_API_HOST`

Start the API server:

```bash
uv run python -m services.api.app
```

Set `VOICECHAT_TTS_BACKEND=pocket` to use Pocket TTS. Omitting it (or setting
`kokoro`) retains the existing Kokoro backend.

Then open:

```text
http://localhost:3000/docs
```


## Voice activity detection

The microphone path uses Silero VAD before Moonshine with immediate speech onset callbacks for barge-in interruption. VAD only gates audio admission to STT and triggers barge-in; it preserves existing STTListener events and ConversationManager / SentenceGate behavior.

The default configuration is:

| Setting | Default |
|---|---:|
| Sample rate | 16 kHz |
| VAD threshold | 0.5 |
| Minimum silence | 200 ms |
| Speech padding | 100 ms |
| Pre-speech padding | 200 ms |
| Backend frame | 512 samples |

This stage detects speech versus non-speech with a 200ms trailing window and 200ms audio pre-roll to prevent initial consonant clipping. Acoustic Echo Cancellation (AEC) subtracts speaker reference audio before VAD analysis to prevent false triggers during assistant speech.
