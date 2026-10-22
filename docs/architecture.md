# System Architecture & Technical Design

Waffle Audio is built as a modular, low-latency voice-to-voice assistant runtime. It streams audio and tokens in real time directly on your local hardware without cloud API dependencies.

---

## 1. High-Level Data Flow

```
+----------------------------------------------------------------------------------------------------------------+
|                                              Microphone Input                                                  |
+----------------------------------------------------------------------------------------------------------------+
                                                        |
                                                        v
                                       +---------------------------------+
                                       | Acoustic Echo Canceller (NLMS)  |<-- [Playback Loopback Reference]
                                       +---------------------------------+
                                                        |
                                                        v
                                       +---------------------------------+
                                       |    Silero VAD (Speech/Silence)  |--> (Speech Onset: Barge-In Trigger)
                                       +---------------------------------+
                                                        |
                                                        v
                                       +---------------------------------+
                                       |      Moonshine STT Engine       |
                                       +---------------------------------+
                                                        |
                                                        v
                                       +---------------------------------+
                                       |  Sentence Gate & Clause Chunker |
                                       +---------------------------------+
                                                        |
                                                        v
                                       +---------------------------------+
                                       |  LLM Provider (Ollama / Qwen)   |<-- [MCP Tools & DDG Fallback Search]
                                       +---------------------------------+
                                                        | (Streaming Tokens)
                                                        v
                                       +---------------------------------+
                                       | TTS Engine (Pocket TTS / Kokoro)|
                                       +---------------------------------+
                                                        | (Audio Chunks)
                                                        v
+----------------------------------------------------------------------------------------------------------------+
|                                              Speaker Playback                                                  |
+----------------------------------------------------------------------------------------------------------------+
```

---

## 2. Core Architectural Pillars

### A. Sub-Second End-to-End Latency
Traditional voice assistants serialize the entire pipeline: they wait for the user to finish speaking, run full transcription, send the complete text to an LLM, wait for the entire response to finish generating, and only then generate TTS audio. This incurs 4–8 seconds of latency.

Waffle Audio achieves **sub-second E2E latency (0.83s conversational turn)** through concurrent streaming:
1. **Streaming STT**: Moonshine transcribes audio continuously as you speak.
2. **Adaptive Sentence Gate**: Terminal punctuation quick-fires in 250 ms, sending speech to the LLM before prolonged silence.
3. **Adaptive Clause Chunking (`ClauseChunker`)**: As Ollama streams response tokens, the chunker splits tokens on punctuation (`,`, `;`, `:`) and conjunctions (`and`, `but`, `because`). The **first-chunk fast path** fires on the first ~3–4 words, dispatching speech synthesis in <200 ms TTFB.
4. **Streaming TTS**: Pocket TTS synthesizes raw PCM audio buffers from clauses on the fly, streaming audio directly to the speaker queue while Ollama is still generating subsequent sentences.

### B. Two-Way Natural Conversation (AEC & Barge-In)
- **Acoustic Echo Cancellation (AEC)**: The assistant maintains a circular playback reference buffer. When sound comes out of the speakers, the NLMS adaptive filter estimates the room transfer function and cancels speaker audio from the microphone stream.
- **Instant Interruption (Barge-In)**: When the user speaks while Waffle is talking, the `BargeInController` immediately:
  1. Interrupts `AudioPlayer` playback in <50 ms.
  2. Aborts active streaming LLM generation via HTTP connection drops and Ollama cancellation.
  3. Purges `TTSService` speech synthesis queues.
  4. Resets the turn accumulator for the incoming user utterance.

### C. Asynchronous Background Context Compaction
Small local models (such as `qwen3.5:4b`) slow down when context length balloons, and CPU KV-cache swapping causes dramatic token generation lag.
- The `ContextCompactor` runs asynchronously in the background.
- When conversation history reaches threshold (`VOICECHAT_COMPACTOR_MAX_TURNS = 6`), it rolls older turns into structured memory summaries while preserving the most recent 2 turns verbatim.
- Compaction runs out-of-band without locking the active voice turn or blocking speech synthesis.

### D. Intent-Gated Tool Integration (MCP & Search)
- Passing heavy tool definitions and JSON schemas on every conversational turn inflates TTFT (Time To First Token) and confuses small local models during casual conversation.
- Waffle uses **Dynamic Intent Gating**: standard chit-chat passes zero tool definitions to the LLM.
- Only when search, lookup, real-time facts, weather, or tool-related intents are detected (`has_tool_intent`) are MCP tools and DuckDuckGo search schemas dynamically injected.
- While the background tool execution completes, Waffle immediately speaks natural bridge phrases (*"Checking that for you..."*, *"Looking it up on the web..."*) to keep the conversational flow natural and engaging.

---

## 3. Package Structure & Modular Packaging

All service packages are cleanly isolated with standard `__init__.py` modules and unified lazy exports in `services/__init__.py`. Dynamic submodule resolution ensures that imports like `import services.conversation` or `from services.tts import ...` resolve cleanly without circular dependencies:

```
services/
  api/                    FastAPI service exposing HTTP and WebRTC endpoints
  audio/                  Hardware audio output & playback worker with interrupt
  audio_processing/       VAD, Acoustic Echo Canceller, Barge-In, and audio resamplers
  conversation/           ConversationManager, SentenceGate, ClauseChunker, and Rich UI
  llm/                    Ollama provider with token streaming and intent gating
  stt/                    Moonshine streaming speech-to-text service
  tools/                  MCP client manager, DuckDuckGo search, and bridge phrases
  tts/                    Pocket TTS & Kokoro TTS backends with audio streaming
  pipeline.py             Unified in-process async pipeline
  shared_memory.py        POSIX shared memory ring buffer for zero-copy IPC
```

---

## 4. Test Architecture & Verification

The codebase features a 211-test deterministic test suite organized in `tests/`:

```
tests/
  test_api_service.py              FastAPI endpoints, audio WAV conversion, one-shot listener
  test_audio_service.py            AudioPlayer playback listener, monitors, interrupt lifecycle
  test_audio_processing.py         Acoustic Echo Canceller (NLMS), Barge-In controller, resampler
  test_vad_service.py              Silero VAD speech detection & pre-speech padding buffer
  test_llm_service.py              LLMService lifecycle, queueing, instant abort, intent matching
  test_stt_service.py              Moonshine STT wrapper, streaming adapter, queue overflow
  test_tts_service.py              BaseTTSService, TTSService factory, local-first HF caching
  test_voice_cloning.py            Voice embedding resolution, mic recording, voice sanitization
  test_tools.py                    MCP server management, DuckDuckGo fallback, phrases
  test_install_tool.py             Presets installer, custom MCP registration, CLI flags
  test_conversation_cli.py         Logging redirect, C-level stderr suppression, CLI arguments
  test_cli_ui.py                   Rich terminal UI waveform, status badges, scrolling text
  test_compactor.py                ContextCompactor heuristics & background worker
  test_pipeline_optimizations.py   ClauseChunker fast path, early clause dispatch
  test_benchmark_logging.py        Per-turn JSONL latency logging & benchmark metrics
  test_trigger_mode.py             Trigger word gate & follow-up window state machine
  test_setup.py                    Ollama /api/tags model discovery, .env generation
  test_constants.py                Environment variable parsing & fallback defaults
```

All tests are executable via `uv run pytest` or `uv run python -m unittest discover tests`.
