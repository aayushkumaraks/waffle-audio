# Waffle Audio

A local voice assistant with two run modes: a microphone-driven CLI using `ConversationManager` on supported Linux hosts, or a browser UI served by the FastAPI API. Docker provides a consistent Python/system runtime across macOS, Windows, and Linux.

## Quick install

Install [Docker Desktop](https://docs.docker.com/get-docker/) (or Docker Engine plus Compose v2 on Linux), Git, and [Ollama](https://ollama.com) on the host. Then:

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

The installer asks for `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and the Pocket TTS voice only when those settings are not already present in the environment or `.env`. It writes the values to `.env`, builds the Docker image, then verifies Ollama reachability, model availability, and a minimal real chat completion from inside the container. If the configured model is missing, it offers to run `ollama pull` when the host CLI is available. A failed build or LLM check stops installation instead of launching a broken setup.

The default container-reachable Ollama URL is `http://host.docker.internal:11434`. On Linux, Ollama must listen on an address reachable from Docker; if needed, set `OLLAMA_BASE_URL` to that address. Do not use `localhost` unless Ollama is running inside the same container.

## Choose a run mode

Run `scripts/run.sh` (macOS/Linux) or `scripts/run.ps1` (Windows PowerShell).

- **Linux with `/dev/snd`:** choose CLI mode to map the host audio device into the container, or use web mode.
- **macOS / Windows Docker Desktop:** the script offers web mode only. Docker Desktop does not provide portable host microphone passthrough to Linux containers; the browser handles microphone and speaker access.
- **Linux without `/dev/snd`:** use web mode or configure an audio device first.

Web mode URLs (default host port `3000`):

- UI: `http://localhost:3000/ui`
- API docs: `http://localhost:3000/docs`
- Health: `http://localhost:3000/health`

Override the host port with `VOICECHAT_API_PORT` (for example, `VOICECHAT_API_PORT=3100`).

## Dependencies and LLM

Ollama runs on the host and manages the local LLM and its model cache. The default model is `qwen3.5:4b`. If needed, run:

```bash
ollama pull qwen3.5:4b
```

Compose adds the `host.docker.internal` host-gateway mapping for Linux. Configure `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in `.env` to override the defaults.

## Container commands

```bash
# Web + API (background)
docker compose --profile web up --build -d

# Follow logs
docker compose logs -f waffle-audio

# Stop containers
docker compose --profile web down

# CLI on Linux only, with an available /dev/snd device
docker compose --profile cli run --build --rm --device /dev/snd:/dev/snd waffle-audio-cli

# Rebuild after code changes
docker compose build --no-cache
```

The Pocket TTS cache is stored in a named Docker volume so model/voice downloads persist between runs. The web service uses `restart: unless-stopped`. `scripts/create-service.sh` / `.ps1` starts that background service; stop it with `docker compose --profile web down`.

## Configuration

Common variables:

| Variable | Default | Purpose |
|---|---|---|
| `VOICECHAT_TTS_BACKEND` | `pocket` | TTS backend; existing Kokoro code remains available |
| `VOICECHAT_TTS_VOICE` | `alba` | Pocket TTS voice identifier |
| `VOICECHAT_API_PORT` | `3000` | Published host port |
| `VOICECHAT_CORS_ALLOW_ORIGINS` | `*` in Compose | Browser origin allowlist |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` in Compose | Host Ollama endpoint reachable from the container |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Local model name |

Voice options include English `alba`, `anna`, `azelma`, `bill_boerst`, `caro_davy`, `charles`, `cosette`, `eponine`, `eve`, `fantine`, `george`, `jane`, `jean`, `javert`, `marius`, `mary`, `michael`, `paul`, `peter_yearsley`, `stuart_bell`, `vera`; French `estelle`; German `juergen`; Italian `giovanni`; Portuguese `rafael`; Spanish `lola`. Catalog and license information: [Pocket TTS voices](https://huggingface.co/kyutai/tts-voices).

## Architecture

```text
Browser microphone → WebRTC/VAD → STT (Moonshine) → LLM (Ollama) → TTS (Pocket TTS) → Browser speakers
CLI microphone → STT (Moonshine) → ConversationManager → Ollama → Pocket TTS → Host speakers
```

The API and browser UI share one FastAPI process. The CLI is a separate entry point that reuses `ConversationManager`. The existing Kokoro implementation remains in the codebase.

## Docker validation

GitHub Actions validates the Compose configuration and builds the image for `linux/amd64` and `linux/arm64` when relevant files change in a pull request or on `main`.

## Development

```bash
uv sync
uv run pytest
```

The Python project requires Python 3.11. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`; end users do not need Python installed on the host.
