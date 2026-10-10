# Waffle Audio

A local voice assistant with two run modes: a microphone-driven CLI using `ConversationManager`, or a browser UI served by the FastAPI API. Docker provides a consistent Python/system runtime across macOS, Windows, and Linux.

## Quick install

Install [Docker Desktop](https://docs.docker.com/get-docker/) (or Docker Engine plus Compose v2 on Linux) and Git. Then:

**macOS / Linux**

```bash
curl -fsSL https://raw.githubusercontent.com/aayushkumaraks/waffle-audio/main/scripts/install.sh -o install-waffle-audio.sh
bash install-waffle-audio.sh
```

**Windows PowerShell**

```powershell
irm https://raw.githubusercontent.com/aayushkumaraks/waffle-audio/main/scripts/install.ps1 -OutFile install-waffle-audio.ps1
powershell -ExecutionPolicy Bypass -File .\install-waffle-audio.ps1
```

The installer asks which Pocket TTS voice to use, saves it to `.env`, then offers to run now or start the web/API container in the background. Python packages and system dependencies are installed inside the Docker image; Python and `uv` are not required on the host.

## Choose a run mode

Run `scripts/run.sh` (macOS/Linux) or `scripts/run.ps1` (Windows PowerShell). The menu offers:

1. **CLI voice assistant** — uses the existing `ConversationManager` path, prints only recognized user speech and assistant replies, and suppresses routine service logs.
2. **Web UI + API** — builds and starts the API container, which serves the UI and docs.

Web mode URLs:

- UI: `http://localhost:3000/ui`
- API docs: `http://localhost:3000/docs`
- Health: `http://localhost:3000/health`

Override the host port with `VOICECHAT_API_PORT` (for example, `VOICECHAT_API_PORT=3100`).

## CLI audio notes

Docker makes the application runtime cross-platform, but microphone/speaker passthrough is controlled by the host OS and Docker implementation. Native Docker Desktop on macOS/Windows does not expose host audio devices to Linux containers in a portable way. Use **web mode** on those platforms; browser audio is captured by the browser and sent to the API. The CLI container is best-effort on Linux hosts with audio devices exposed to the container; for ALSA, configure device access for your system (often `/dev/snd`) before using CLI mode.

## Dependencies and LLM

- Docker Engine/Desktop with Docker Compose v2
- Git for the initial checkout
- [Ollama](https://ollama.com) running on the host with `qwen3.5:4b` pulled:

```bash
ollama pull qwen3.5:4b
```

Ollama stays on the host because it manages the local LLM and its model cache. The container reaches it at `host.docker.internal:11434` (the Compose file adds the host-gateway mapping for Linux). Override `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in `.env` if needed. On Linux, configure Ollama to listen on an address reachable from Docker, not only loopback.

## Container commands

```bash
# Web + API (background)
docker compose --profile web up --build -d

# Follow logs
docker compose logs -f waffle-audio

# Stop containers
docker compose --profile web down

# Run CLI interactively
docker compose --profile cli run --rm waffle-audio-cli

# Rebuild after code changes
docker compose build --no-cache
```

The Pocket TTS cache is stored in a named Docker volume so the model/voice downloads persist between runs. The web service uses `restart: unless-stopped`. `scripts/create-service.sh` / `.ps1` starts that background service; stopping it is `docker compose --profile web down`.

## Configuration

Copy/edit `.env` as needed. Common variables:

| Variable | Default | Purpose |
|---|---|---|
| `VOICECHAT_TTS_BACKEND` | `pocket` | TTS backend; existing Kokoro code remains available |
| `VOICECHAT_TTS_VOICE` | `alba` | Pocket TTS voice identifier |
| `VOICECHAT_API_PORT` | `3000` | Published host port |
| `VOICECHAT_CORS_ALLOW_ORIGINS` | `*` | Browser origin allowlist |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` in Compose | Host Ollama endpoint |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Local model name |

Voice options include English `alba`, `anna`, `azelma`, `bill_boerst`, `caro_davy`, `charles`, `cosette`, `eponine`, `eve`, `fantine`, `george`, `jane`, `jean`, `javert`, `marius`, `mary`, `michael`, `paul`, `peter_yearsley`, `stuart_bell`, `vera`; French `estelle`; German `juergen`; Italian `giovanni`; Portuguese `rafael`; Spanish `lola`. Catalog and license information: [Pocket TTS voices](https://huggingface.co/kyutai/tts-voices).

## Architecture

```text
Browser microphone → WebRTC/VAD → STT (Moonshine) → LLM (Ollama) → TTS (Pocket TTS) → Browser speakers
CLI microphone → STT (Moonshine) → ConversationManager → Ollama → Pocket TTS → Host speakers
```

The API and browser UI share one FastAPI process. The CLI is a separate entry point that reuses `ConversationManager`. The existing Kokoro implementation remains in the codebase.

## Development

```bash
uv sync
uv run pytest
```

The Python project requires Python 3.11. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`; end users do not need Python installed when using Docker.
