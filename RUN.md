# Running the app

## Start Ollama

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
| Silence gate   | `services/conversation/src/sentence_gate.py` | 2.0 s gap, 0.4 s quick-fire |

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
