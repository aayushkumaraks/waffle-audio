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

Measured from a real session on the default stack (Moonshine STT, Ollama
`qwen3.5:4b`, Pocket TTS), over four warm turns after the latest optimisations.

### End-to-end latency (end of your speech to Waffle starting to speak)

| Version | Avg warm turn | First turn (cold start) |
|---|---:|---:|
| Before optimisation | 3.3 s | 15.0 s |
| **Current** | **1.6 s** | **5.4 s** |
| Improvement | **~50% faster** | **~64% faster** |

### Time spent in each module (average per warm turn)

| Module | What it does | Time |
|---|---|---:|
| Sentence gate & VAD | Waits for natural completion with 200ms trailing silence | 0.3 s (punctuated) to ~0.8 s (unpunctuated gap) |
| Clause chunker | Emits first clause (3 words min) for immediate synthesis | < 0.2 s |
| LLM, first token | Ollama starts streaming tokens | ~0.35 s |
| LLM, full reply | Whole 2-3 sentence response generated | 1.15 s |
| TTS, first chunk | Pocket TTS synthesises the opening clause | 0.65 s (starts playing immediately) |
| TTS, next clauses | Synthesised concurrently while the first one is playing | Hidden behind playback |
| TTS speed | Audio produced vs. real time | 2.5x - 3.0x faster than real time |
| STT (Moonshine) | Streaming transcription with 200ms pre-roll | Folded into conversational flow |

Notes:

- Gate and first-token times are derived from request timestamps, because the two are
  not logged separately.
- STT runs continuously while you talk, so its cost is folded into the gate wait.
- Results depend on your hardware and Ollama host. Your numbers will differ.

### What makes it fast

| Optimisation | Effect |
|---|---|
| Adaptive clause chunking & fast path | Speech starts on the first ~3-4 words (<200ms TTFB), streaming ahead of full sentence |
| Unified in-process async pipeline | Zero network/TCP serialization latency; passes raw NumPy audio buffers in-memory |
| Acoustic Echo Cancellation (AEC) & Barge-in | Natural two-way conversation; user can interrupt at any point without audio feedback |
| Optimized VAD & pre-roll ring buffer | 200ms trailing silence cut dead air; 200ms pre-roll preserves initial unvoiced consonants |
| Polyphase FIR & SIMD resampling | Ultra-low CPU overhead downsampling audio via `soxr` / `scipy.signal.resample_poly` |
| Ollama keep-alive and background warm-up | First-turn wait cut from ~15 s to ~5.4 s |
| Sliding context window (6–8 turns) | Keeps prompt history compact (max 16 messages), preventing LLM token slowdown |
| Faster sentence gate | Terminal punctuation quick-fires in 300ms; unpunctuated sentences wait only 800ms |
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
- **Unified In-Process Async Pipeline**: Consolidates audio, VAD, STT, LLM, TTS, and player into a single async loop with zero serialization overhead, alongside POSIX shared memory (`SharedAudioBuffer`) for multi-process deployments.
- **Sliding Conversation Context**: Retains a rolling 6–8 turns (max 16 messages) to maintain immediate context while preventing inference degradation over long sessions.
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

### Tuning

| Environment variable | Default | Purpose |
|---|---|---|
| `OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model in memory |
| `OLLAMA_NUM_PREDICT` | `160` | Maximum tokens per reply |
| `VOICECHAT_TTS_BACKEND` | `pocket` | `pocket` or `kokoro` |
| `VOICECHAT_GAP_TIMEOUT` | `0.8` | Seconds of silence before finalizing user turn |
| `VOICECHAT_QUICK_FIRE_TIMEOUT` | `0.3` | Silence timeout when sentence ends with terminal punctuation |
| `VOICECHAT_AEC_ENABLED` | `true` | Enable acoustic echo cancellation for microphone loopback |
| `VOICECHAT_BARGE_IN_ENABLED` | `true` | Enable instant interruption on detected user speech |

## Project structure

```
services/
  conversation/   Orchestrates STT -> LLM -> TTS
    src/
      manager.py          ConversationManager & sliding history window
      sentence_gate.py    Adaptive silence-gap gate & ClauseChunker
      ui.py               Rich terminal UI controller & renderer
    cli.py                Live microphone voice assistant with AEC & Barge-In
  audio_processing/
    src/
      vad_service.py      Silero VAD with speech callbacks & pre-roll ring buffer
      echo_cancellation.py NLMS Acoustic Echo Canceller with delay reference
      barge_in.py         BargeInController (stops playback, aborts LLM, purges TTS)
      resampler.py        Fast polyphase / soxr SIMD audio resampling
  stt/            Moonshine streaming transcription wrapper
  llm/            Ollama HTTP provider + streaming abort support
  tts/            Feature-flagged Kokoro and Pocket TTS services with queue purge
  audio/          Blocking audio playback worker with instant interrupt
  pipeline.py     Unified in-process async pipeline
  shared_memory.py Zero-copy POSIX shared memory audio ring buffer
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
