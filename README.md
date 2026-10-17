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
- **Understands how people actually talk.** An adaptive sentence gate waits through
  natural pauses ("I want to... uh... test how fast you are") instead of interrupting you.
- **No feedback loops.** The microphone is muted while Waffle speaks, so it never
  answers itself.
- **Offline after first run.** Models are loaded from the local cache; the network is
  only touched when a model is missing.
- **Modular and hackable.** STT, LLM and TTS are independent services. Swap the model,
  the voice or the backend without touching the rest.
- **Terminal or web.** Use the live voice CLI, the HTTP/WebRTC API, or the web app.

## Performance

Measured from a real session on the default stack (Moonshine STT, Ollama
`qwen3.5:4b`, Pocket TTS), over four warm turns after the latest optimisations.

### End-to-end latency (end of your speech to Waffle starting to speak)

| Version | Avg warm turn | First turn (cold start) |
|---|---:|---:|
| Before optimisation | 3.3 s | 15.0 s |
| **Current** | **2.1 s** | **6.8 s** |
| Improvement | **~37% faster** | **~55% faster** |

### Time spent in each module (average per warm turn)

| Module | What it does | Time |
|---|---|---:|
| Sentence gate | Waits for you to finish your thought | 0.4 s (complete sentence) to ~1.4 s (trailing off) |
| LLM, first token | Ollama starts replying | ~0.45 s |
| LLM, full reply | Whole 2-3 sentence response generated | 1.47 s |
| TTS, first sentence | Pocket TTS synthesises the opening sentence | 1.82 s (about 3.6 s of audio) |
| TTS, next sentence | Synthesised while the first one is playing | 1.39 s (hidden behind playback) |
| TTS, prompt processing | Text prep before the first audio chunk | 0.17 s |
| TTS speed | Audio produced vs. real time | 2.5x faster than real time |
| STT (Moonshine) | Streaming transcription while you speak | Not separately timed |

Notes:

- Gate and first-token times are derived from request timestamps, because the two are
  not logged separately.
- STT runs continuously while you talk, so its cost is folded into the gate wait.
- Results depend on your hardware and Ollama host. Your numbers will differ.

### What makes it fast

| Optimisation | Effect |
|---|---|
| Sentence-by-sentence TTS streaming | Speech starts ~0.5 s after the first token, not after the full reply |
| Ollama keep-alive and background warm-up | First-turn wait cut from ~15 s to ~7 s |
| Faster sentence gate | Unpunctuated sentences no longer wait ~2.5 s |
| Local-first model loading | No network request on startup |
| Mic copy fix in the CLI | Clean audio into STT, accurate transcripts |

## How it works

```
Microphone --> VAD (Silero) --> STT (Moonshine) --> LLM (Ollama) --> TTS (Kokoro / Pocket TTS) --> Speakers
```

| Stage | Technology         | Notes                                      |
|-------|--------------------|--------------------------------------------|
| STT   | Moonshine          | Streaming speech-to-text, runs on CPU/GPU  |
| LLM   | Ollama (local)     | Default model: qwen3.5:4b                  |
| TTS   | Kokoro / Pocket TTS | Selectable streaming neural text-to-speech |
| Audio | sounddevice / PortAudio | Cross-platform audio I/O              |

## Key features

- Echo cancellation: microphone input is muted while the assistant is speaking,
  including the gaps between streamed sentences
- Adaptive sentence gate: fragments the user speaks across short pauses are
  accumulated before the LLM call, avoiding premature or duplicate responses
- Streaming responses: the reply text is shown as it is generated and each finished
  sentence is spoken immediately
- Conversation history: every turn is tracked and printed on exit
- Voice-friendly LLM prompt: the model is instructed to reply in 2-3 short plain
  sentences with no emojis or markdown

## Quick Start (One-Click)

Run the one-click install & run script from the project root:

```bash
./run.sh
```

On Windows:
```cmd
run.bat
```

Or run directly with `uv`:
```bash
uv run python -m services.conversation.cli
```

### CLI Interface Features

- **Top-Right Status Badge**: Always shows whether the assistant is `● READY`, `🎤 LISTENING`, or `⠋ PROCESSING`.
- **Real-Time Conversation**: Live transcription appears as you speak, and assistant responses stream word-by-word synchronously as the audio plays.
- **Dynamic Auto-Scrolling**: The conversation window automatically scrolls down so newest utterances and live responses are always visible, indicating hidden lines with a border subtitle.
- **Multi-Line Speech Accumulation**: Continuous speech across multiple Moonshine segments is seamlessly accumulated into a single turn without fragmentation.
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

### Emotional Voice Assistant (One-Click)

Run the emotional voice assistant with dynamic emotion switching, thread-per-emotion models, in-memory caching, and idle offloading:

```bash
./run-with-emotion.sh
```

On Windows:
```cmd
run-with-emotion.bat
```

- **Gender Selection**: Select Female or Male at startup.
- **Emotions Supported**: `neutral`, `happy`, `sad`, `angry`.
- **Thread-Per-Emotion Workers**: In-memory caching with automatic offload when idle (`--offload-idle 120`).
- **Live Emotion Parsing**: Real-time extraction of LLM emotion tags (`[happy]`, `[sad]`, `[angry]`, `[neutral]`) routing speech to the matching voice model.

### Tuning

| Environment variable | Default | Purpose |
|---|---|---|
| `OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model in memory |
| `OLLAMA_NUM_PREDICT` | `160` | Maximum tokens per reply |
| `VOICECHAT_TTS_BACKEND` | `pocket` | `pocket` or `kokoro` |
| `VOICECHAT_GAP_TIMEOUT` | `2.0` | Seconds of silence before finalizing user turn |
| `VOICECHAT_QUICK_FIRE_TIMEOUT` | `0.8` | Silence timeout when sentence ends with terminal punctuation |

## Project structure

```
services/
  conversation/   Orchestrates STT -> LLM -> TTS
    src/
      manager.py          ConversationManager
      sentence_gate.py    Adaptive silence-gap gate
      ui.py               Rich terminal UI controller & renderer
    cli.py                Live microphone voice assistant
  stt/            Moonshine streaming transcription wrapper
  llm/            Ollama HTTP provider + queue-based service
  tts/            Feature-flagged Kokoro and Pocket TTS services
  audio/          Blocking audio playback worker
models/           Shared data models (Message)
voiceModels/      Kokoro ONNX model files (not committed)
pyproject.toml    Dependencies managed by uv
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

The WebRTC microphone path uses Silero VAD before Moonshine. VAD only gates audio admission to STT; it does not change the existing STTListener events or ConversationManager / SentenceGate behavior.

The default Stage 1 configuration is:

| Setting | Default |
|---|---:|
| Sample rate | 16 kHz |
| VAD threshold | 0.5 |
| Minimum silence | 300 ms |
| Speech padding | 100 ms |
| Pre-speech padding | 200 ms |
| Backend frame | 512 samples |

This stage detects speech versus non-speech. It does not distinguish the user's voice from another speaker such as a TV. Target-speaker verification and noise suppression are planned as subsequent stages.
