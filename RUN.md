# Running Waffle Audio

## 1. Start Ollama

Ollama manages local LLM inference and must be running on the host machine before Waffle Audio starts:

```bash
ollama serve
```

Ensure the target model is pulled:
```bash
ollama pull qwen3.5:4b
```

---

## 2. Option A: Run via Docker (Recommended)

Docker provides a reproducible, cross-platform runtime for the Python backend and Node/React webapp.

```bash
# Start background web services
docker compose --profile web up --build -d

# View real-time logs
docker compose logs -f waffle-audio

# Stop containers
docker compose --profile web down
```

### URLs

- **Web UI**: `http://localhost:3000` (or `http://localhost:3000/ui`)
- **Webapp (Vite)**: `http://localhost:5173`
- **API Docs (Swagger)**: `http://localhost:3000/docs`
- **Health Endpoint**: `http://localhost:3000/health`

### Terminal CLI Mode (Linux Only)

For direct host microphone/speaker audio passthrough on Linux:
```bash
docker compose --profile cli run --build --rm --device /dev/snd:/dev/snd waffle-audio-cli
```

---

## 3. Option B: Run Locally (Without Docker)

### Start API Backend
From the repository root:
```bash
uv sync
uv run python -m services.api.app
```

### Start Web Frontend
In another terminal:
```bash
cd tools/webapp
npm ci
npm run dev -- --host 0.0.0.0
```

---

## Main Endpoints

| Area | Method + Path | Purpose |
|---|---|---|
| **Webapp** | `GET /` & `GET /ui` | Browser voice chat UI |
| **Health** | `GET /health` | Service health and active configuration |
| **WebRTC** | `POST /webrtc/offer` | Real-time bidirectional WebRTC voice stream |
| **LLM** | `POST /llm/generate` | Generate text response from message history |
| **TTS** | `POST /tts/synthesize` | Synthesize speech audio (WAV) for browser playback |
| **TTS** | `POST /tts/speak` | Server-side audio playback worker |
| **STT** | `POST /stt/push` | Push PCM audio samples for Moonshine transcription |
| **STT** | `GET /stt/transcripts` | Read captured transcript events |
| **Conversation** | `GET /conversation/history` | Read current conversation turn history |
| **Conversation** | `DELETE /conversation/history`| Reset active conversation history |
