# Waffle Audio

A local, real-time voice-to-voice AI assistant that runs entirely on your own hardware. 

Waffle Audio transcribes user speech using streaming STT, generates conversational responses with a local language model, and synthesizes natural streaming speech—delivering a responsive, low-latency conversational experience with both a modern browser interface and a Linux terminal CLI.

```text
Browser Mic ──▶ WebRTC / VAD ──▶ STT (Moonshine) ──▶ LLM (Ollama) ──▶ TTS (Pocket TTS) ──▶ Browser Speakers
CLI Mic     ──▶ STT (Moonshine) ──▶ ConversationManager ──▶ Ollama ──▶ TTS (Pocket TTS) ──▶ Host Speakers
```

---

## Key Features

- **Full Local Privacy**: Audio processing, transcription, LLM inference, and speech synthesis run locally without external cloud dependencies.
- **Production Containerization**: Multi-stage Docker image packaging the FastAPI backend, compiled React SPA, and Vite development server into a single portable runtime.
- **Cross-Platform Browser Mode**: Browser-based microphone and speaker audio using WebRTC, supported across macOS, Windows, and Linux.
- **Local Network Accessible**: Exposed on `0.0.0.0` with configurable WebRTC UDP media port forwarding (`50000-50100/udp`) so phones, tablets, or other computers on your LAN can connect.
- **Linux Audio Passthrough CLI**: Dedicated low-latency CLI mode with direct `/dev/snd` hardware access for Linux desktop and embedded setups.
- **Persistent Voice Caching**: Pocket TTS model weights and voice embeddings persist in a dedicated Docker volume, preventing redownloads across restarts.

---

## Quick Install (One-Click)

### Prerequisites

1. **[Docker Desktop](https://docs.docker.com/get-docker/)** (macOS / Windows) or **Docker Engine + Docker Compose v2** (Linux).
2. **[Ollama](https://ollama.com)** installed and running on the host machine.
3. **Git**.

### Installation Command

**macOS / Linux / WSL:**
```bash
curl -fsSL https://raw.githubusercontent.com/aayushkumaraks/waffle-audio/main/scripts/install.sh -o install-waffle-audio.sh
bash install-waffle-audio.sh
```

**Windows PowerShell:**
```powershell
irm https://raw.githubusercontent.com/aayushkumaraks/waffle-audio/main/scripts/install.ps1 -OutFile install-waffle-audio.ps1
powershell -ExecutionPolicy Bypass -File .\install-waffle-audio.ps1
```

The installer verifies Docker and Compose v2, prompts for configuration (saving values to `.env`), builds the multi-stage image, tests host Ollama reachability, and validates model inference before launching.

---

## Accessing the Application

When running in **Web Mode**, the container exposes both the compiled production interface and the development server:

| Service | URL | Description |
|---|---|---|
| **Web UI (Production)** | [`http://localhost:3000`](http://localhost:3000) or [`/ui`](http://localhost:3000/ui) | Optimized, compiled React SPA served directly by FastAPI. |
| **Webapp (Vite Server)** | [`http://localhost:5173`](http://localhost:5173) | Vite server running concurrently with hot module reload support. |
| **API Docs (Swagger)** | [`http://localhost:3000/docs`](http://localhost:3000/docs) | Interactive OpenAPI documentation. |
| **Health Check** | [`http://localhost:3000/health`](http://localhost:3000/health) | Service uptime and backend status monitor. |

### Accessing From Other Devices on Your Local Network (LAN)

Both port `3000` and port `5173` bind to `0.0.0.0`. To connect from another computer, smartphone, or tablet on your Wi-Fi/Ethernet:
1. Find your host machine's local IP address (e.g. `192.168.1.100`).
2. Open `http://192.168.1.100:3000` in the client browser.
3. For WebRTC microphone streaming over LAN, the container forwards UDP ports `50000-50100/udp` and automatically resolves client candidate IPs. If needed, you can explicitly set `VOICECHAT_WEBRTC_ADVERTISED_IP=192.168.1.100` in `.env`.

> [!NOTE]
> Modern web browsers restrict microphone access to **secure contexts** (`localhost`, `127.0.0.1`, or `https://`). When connecting via a remote LAN IP, you may need to configure a reverse proxy with TLS (e.g., Caddy or Nginx) or enable insecure origins in browser flags (e.g., `chrome://flags/#unsafely-treat-insecure-origin-as-secure`).

---

## Docker & Compose Commands

### Start Services

```bash
# Start Web UI + API in the background
docker compose --profile web up --build -d

# View live container logs
docker compose logs -f waffle-audio

# Check container health and running ports
docker compose ps

# Stop containers
docker compose --profile web down
```

### Run Terminal CLI Mode (Linux Only)

On Linux hosts with direct `/dev/snd` hardware access:
```bash
docker compose --profile cli run --build --rm --device /dev/snd:/dev/snd waffle-audio-cli
```

### Windows & WSL 2 Usage

If you use WSL 2 on Windows:
- Ensure WSL 2 integration is enabled in **Docker Desktop Settings** → **Resources** → **WSL integration** → toggle your distro **ON**.
- Alternatively, invoke commands using `docker.exe` directly from WSL:
  ```bash
  docker.exe compose --profile web up -d
  ```

---

## Configuration

Settings can be customized in `.env` or passed as environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama endpoint reachable from inside the container. |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Ollama model tag for conversation generation. |
| `VOICECHAT_API_PORT` | `3000` | Host port mapped to the FastAPI backend and Web UI. |
| `VOICECHAT_WEBAPP_PORT` | `5173` | Host port mapped to the Vite webapp server. |
| `VOICECHAT_TTS_BACKEND` | `pocket` | TTS engine (`pocket` or `kokoro`). |
| `VOICECHAT_TTS_VOICE` | `anna` | Voice identifier for Pocket TTS. |
| `VOICECHAT_WEBRTC_UDP_PORT_MIN` | `50000` | Lower bound for WebRTC RTP/RTCP UDP allocation. |
| `VOICECHAT_WEBRTC_UDP_PORT_MAX` | `50100` | Upper bound for WebRTC RTP/RTCP UDP allocation. |
| `VOICECHAT_WEBRTC_ADVERTISED_IP` | *(empty)* | Optional static IP advertised in ICE candidates for complex LAN/NAT setups. |
| `VOICECHAT_CORS_ALLOW_ORIGINS` | `*` | Allowed browser origins for cross-origin API requests. |

### Pocket TTS Voice Options

- **English**: `alba`, `anna`, `azelma`, `bill_boerst`, `caro_davy`, `charles`, `cosette`, `eponine`, `eve`, `fantine`, `george`, `jane`, `jean`, `javert`, `marius`, `mary`, `michael`, `paul`, `peter_yearsley`, `stuart_bell`, `vera`
- **French**: `estelle`
- **German**: `juergen`
- **Italian**: `giovanni`
- **Portuguese**: `rafael`
- **Spanish**: `lola`

---

## Local Development (Without Docker)

If developing locally without Docker, Python 3.11 and Node.js 20+ are required.

### 1. Backend Setup

```bash
# Install Python dependencies with uv
uv sync

# Run tests
uv run python -m unittest discover tests

# Start API server
uv run python -m services.api.app
```

### 2. Webapp Setup

```bash
cd tools/webapp

# Install dependencies
npm ci

# Start Vite dev server with host binding
npm run dev -- --host 0.0.0.0

# Build production bundle for static serving
npm run build
```

---

## Project Structure

```text
├── Dockerfile                    # Multi-stage production container build (Node + Python)
├── docker-compose.yml            # Services for Web mode (dual port + WebRTC UDP) and CLI
├── constants.py                  # Environment-aware static configuration & defaults
├── scripts/
│   ├── start-all.sh              # Container supervisor running FastAPI and Vite concurrently
│   ├── install.sh / install.ps1  # Automated host setup and Ollama validation scripts
│   ├── run.sh / run.ps1          # Platform-aware interactive launchers
│   ├── create-service.sh/.ps1    # Background service manager
│   └── validate-ollama.py        # Ollama connectivity and test inference checker
├── services/
│   ├── api/                      # FastAPI REST & WebRTC signaling endpoints
│   ├── audio/                    # Sound playback worker
│   ├── audio_processing/         # Silero VAD speech detection
│   ├── conversation/             # Turn coordinator & quiet CLI runner
│   ├── llm/                      # Ollama provider client
│   ├── stt/                      # Moonshine streaming speech-to-text
│   └── tts/                      # Pocket TTS and Kokoro synthesis engines
└── tools/
    └── webapp/                   # React 19 single-page application (Vite)
```

---

## License

This project is open-source under the [MIT License](LICENSE).
