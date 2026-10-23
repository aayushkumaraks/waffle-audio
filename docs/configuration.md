# Configuration & Environment Variables Reference

Waffle Audio is configured via environment variables or a local `.env` file in the project root.

If `.env` is absent, the application automatically loads safe, sensible fallback defaults defined in `constants.py`. You can generate or reconfigure `.env` at any time with:

```bash
uv run python setup.py --force
```

---

## Complete Variables Reference Table

### 1. User Personalization & Identity

| Variable | Type | Default | Description |
|---|---|---|---|
| `USER_NAME` | string | `""` | User's preferred name for personalized greetings and address. If empty, assistant never invents or guesses a name. |
| `USER_LOCATION` | string | `""` | User's city and country (e.g. `New Delhi, India` or `San Francisco, USA`). Appended to local weather and news queries. |
| `USER_TIMEZONE` | string | `""` | User's local timezone (e.g. `Asia/Kolkata` or `America/New_York`). Anchors time queries and system prompts. |
| `ASSISTANT_NAME` | string | `"Waffle"` | Assistant persona name. Used in trigger mode phrase `"Hey <ASSISTANT_NAME>"`. |

---

### 2. Text-to-Speech (TTS) Engine

| Variable | Type | Default | Description |
|---|---|---|---|
| `VOICECHAT_TTS_BACKEND` | string | `"pocket"` | TTS synthesis backend: `"pocket"` (ultra-fast Kyutai streaming) or `"kokoro"` (high-quality ONNX). |
| `VOICECHAT_TTS_VOICE` | string | `"anna"` | Active voice name: `"anna"`, `"af_sarah"`, or any cloned voice name located in `cloned_voices/`. |
| `VOICECHAT_TTS_SPEED` | float | `1.0` | Playback speed multiplier (e.g. `1.0`, `1.15`, `1.25`). |
| `VOICECHAT_TTS_QUANTIZE` | boolean | `false` | When true, runs quantized model weights to save RAM/VRAM. |
| `VOICECHAT_TTS_QUEUE_SIZE` | integer | `16` | Maximum synthesized audio chunk queue buffer size. |
| `VOICECHAT_POCKET_TTS_CACHE_DIR` | string | `"voiceModels/pocket-tts-cache"` | Persistent local directory for caching Hugging Face TTS weights. |

---

### 3. Audio Hardware & Interaction Modes

| Variable | Type | Default | Description |
|---|---|---|---|
| `AUDIO_INPUT_DEVICE` | integer/string | `"default"` | Audio input device index or substring match from sounddevice/PortAudio. |
| `AUDIO_OUTPUT_DEVICE`| integer/string | `"default"` | Audio playback output device index or substring match. |
| `VOICECHAT_TRIGGER_MODE` | boolean | `false` | When true, assistant only processes speech preceded by `"Hey <ASSISTANT_NAME>"`. |
| `VOICECHAT_FOLLOW_UP_WINDOW` | float | `15.0` | Seconds after assistant turn in trigger mode where speech continues without repeating wake phrase. |
| `VOICECHAT_BARGE_IN_ENABLED` | boolean | `true` | Allows user to interrupt assistant speech mid-sentence. |
| `VOICECHAT_AEC_ENABLED` | boolean | `true` | Enables Normalized Least Mean Squares (NLMS) acoustic echo cancellation for open speaker playback. |

---

### 4. Voice Activity Detection & Silence Gates

| Variable | Type | Default | Description |
|---|---|---|---|
| `VOICECHAT_GAP_TIMEOUT` | float | `0.5` | Seconds of silence required before finalizing user turn on unpunctuated phrases. |
| `VOICECHAT_QUICK_FIRE_TIMEOUT` | float | `0.25` | Seconds of silence when utterance ends with terminal punctuation (`?`, `!`, `.`). |
| `VOICECHAT_VAD_THRESHOLD` | float | `0.5` | Silero VAD probability threshold for speech activity detection (0.0 to 1.0). |
| `VOICECHAT_VAD_MIN_SILENCE_MS` | integer | `200` | Minimum trailing silence duration in milliseconds before speech completion callback fires. |
| `VOICECHAT_VAD_SPEECH_PAD_MS` | integer | `100` | Speech padding added around voice boundaries in milliseconds. |
| `VOICECHAT_VAD_PRE_SPEECH_PAD_MS` | integer | `200` | Audio pre-roll buffer in milliseconds prepended to prevent clipping initial consonants. |

---

### 5. Speech-to-Text (STT - Moonshine)

| Variable | Type | Default | Description |
|---|---|---|---|
| `VOICECHAT_STT_LANGUAGE` | string | `"en"` | Language code for Moonshine streaming transcription model. |
| `VOICECHAT_STT_UPDATE_INTERVAL` | float | `0.5` | Interval in seconds between streaming transcript event updates. |
| `VOICECHAT_STT_QUEUE_SIZE` | integer | `200` | STT raw audio input queue capacity. |

---

### 6. LLM & Ollama Runtime

| Variable | Type | Default | Description |
|---|---|---|---|
| `OLLAMA_BASE_URL` | string | `"http://localhost:11434"` | Reachable Ollama base endpoint. Auto-detects WSL host IP inside WSL2. |
| `OLLAMA_MODEL` | string | `"qwen3.5:4b"` | Ollama model name. Selectable interactively via Ollama `/api/tags` in `setup.py`. |
| `OLLAMA_KEEP_ALIVE` | string | `"30m"` | Time to keep model resident in GPU VRAM to avoid cold loading latency. |
| `OLLAMA_NUM_PREDICT` | integer | `160` | Maximum token reply cap (keeps answers concise and conversational). |
| `OLLAMA_NUM_CTX` | integer | `2048` | Context window size in tokens (prevents slow CPU KV offloading). |

---

### 7. Conversation Context Compactor

| Variable | Type | Default | Description |
|---|---|---|---|
| `VOICECHAT_MAX_HISTORY_TURNS` | integer | `8` | Maximum user-assistant message pairs retained in active LLM context. |
| `VOICECHAT_COMPACTOR_ENABLED` | boolean | `true` | Enables background worker to compact older turns into summaries. |
| `VOICECHAT_COMPACTOR_MAX_TURNS` | integer | `6` | Turn threshold before triggering background compaction. |
| `VOICECHAT_COMPACTOR_CHAR_LIMIT`| integer | `2500` | History character limit triggering compaction (~500-700 tokens). |
| `VOICECHAT_COMPACTOR_PRESERVE_TURNS` | integer | `2` | Number of most recent turns to always preserve verbatim. |
| `VOICECHAT_COMPACTOR_USE_LLM` | boolean | `false` | When true, uses Ollama for summaries; when false, uses fast heuristic compactor to prevent GPU contention. |

---

### 8. Tools & Model Context Protocol (MCP)

| Variable | Type | Default | Description |
|---|---|---|---|
| `VOICECHAT_TOOLS_ENABLED` | boolean | `true` | Enables Model Context Protocol (MCP) server execution. |
| `VOICECHAT_DDGS_ENABLED` | boolean | `true` | Enables DuckDuckGo fallback search for real-time facts/weather/news. |
| `VOICECHAT_DDGS_MAX_RESULTS` | integer | `4` | Maximum search results retrieved per query. |
| `VOICECHAT_FILLER_PHRASES_ENABLED`| boolean | `true` | Speaks natural bridge phrases (*"Looking that up..."*) during background tool calls. |
| `VOICECHAT_MCP_CONFIG` | string | `"mcp_servers.json"` | Path to MCP servers configuration file. |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | string | `""` | GitHub personal access token for GitHub MCP tool preset. |

---

### 9. Network & API Service

| Variable | Type | Default | Description |
|---|---|---|---|
| `VOICECHAT_API_HOST` | string | `"0.0.0.0"` | Host binding for FastAPI server. |
| `VOICECHAT_API_PORT` | integer | `3000` | Port for FastAPI server. |
| `VOICECHAT_CORS_ALLOW_ORIGINS` | string | `*` | Comma-separated allowed CORS origins for HTTP API. |
| `BENCHMARK_ENABLED` | boolean | `false` | Globally enables latency benchmarking and structured logging. |

---

## 10. Configuration Validation & Testing

Configuration loading, fallback defaults, and environment parsing are verified in:
- `tests/test_setup.py`: Validates interactive prompt parsing, Ollama connectivity detection, audio input enumerations, and `.env` file generation.
- `tests/test_benchmark_logging.py`: Validates dynamic benchmark flag toggling and constants updates.

```bash
PYTHONPATH=. uv run pytest tests/test_setup.py tests/test_benchmark_logging.py
```
