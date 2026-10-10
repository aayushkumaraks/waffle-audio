# Waffle Audio

A local, real-time voice assistant: microphone audio is transcribed, a locally hosted LLM generates a reply, and speech is played back. The API also serves a browser UI.

## One-click setup

**macOS / Linux:** open a terminal and run:

```bash
curl -fsSL https://raw.githubusercontent.com/aayushkumaraks/waffle-audio/main/scripts/install.sh -o install-waffle-audio.sh
bash install-waffle-audio.sh
```

**Windows:** open PowerShell and run:

```powershell
irm https://raw.githubusercontent.com/aayushkumaraks/waffle-audio/main/scripts/install.ps1 -OutFile install-waffle-audio.ps1
powershell -ExecutionPolicy Bypass -File .\install-waffle-audio.ps1
```

The installer checks Python 3.11, Git, `uv`, Ollama and audio prerequisites, installs Python dependencies, shows Pocket TTS voices, and asks whether to run now or configure launch-at-login. Windows users should use the native PowerShell scripts; Bash scripts on Windows require Git Bash or WSL.

Scripts are installed in `~/waffle-audio/scripts` by default (or the path set with `WAFFLE_AUDIO_DIR`):

| Action | macOS / Linux / Git Bash | Windows PowerShell |
|---|---|---|
| Install | `scripts/install.sh` | `scripts/install.ps1` |
| Run | `scripts/run.sh` | `scripts/run.ps1` |
| Start on login | `scripts/create-service.sh` | `scripts/create-service.ps1` |

Pocket TTS defaults to the `alba` voice. The installer offers a voice list and saves your choice in `.env`. Voices and licenses may change; see the [Pocket TTS voice catalog](https://huggingface.co/kyutai/tts-voices).

## Requirements and configuration

- Python **3.11** (the project currently requires `>=3.11,<3.12`)
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com), with model `qwen3.5:4b`
- Microphone and speakers; PortAudio may need a system package on macOS/Linux

The installer can install `uv`, common PortAudio packages, and the Ollama model after asking. Install Ollama separately where needed and start it before running the app. WSL audio support depends on WSLg and the available PortAudio host API.

Useful settings in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `VOICECHAT_TTS_BACKEND` | `pocket` | TTS engine selection (`pocket` or `kokoro`) |
| `VOICECHAT_TTS_VOICE` | `alba` | Pocket TTS voice identifier |
| `VOICECHAT_API_HOST` | `0.0.0.0` | API bind address |
| `VOICECHAT_API_PORT` | `3000` | API port |
| `VOICECHAT_CORS_ALLOW_ORIGINS` | `*` | Comma-separated browser origin allowlist, or `*` |
| `OLLAMA_BASE_URL` | project default | Ollama endpoint; set this if Ollama is not at the configured address |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Ollama model name |

## Use the app

Start the API from the repository root:

```bash
uv run python -m services.api.app
```

- Browser UI: `http://localhost:3000/ui`
- API documentation: `http://localhost:3000/docs`
- Health check: `http://localhost:3000/health`

Stop a foreground run with `Ctrl+C`. See [RUN.md](RUN.md) for endpoints and troubleshooting, or [SETUP.md](SETUP.md) for detailed audio and WSL notes.

## Architecture

```text
Microphone → VAD (Silero) → STT (Moonshine) → LLM (Ollama) → TTS → Speakers
```

- **STT:** Moonshine streaming transcription
- **LLM:** local Ollama model
- **TTS:** Pocket TTS by default; the existing Kokoro backend remains available in code
- **Audio:** `sounddevice` / PortAudio
- **API/UI:** FastAPI server with browser voice chat UI

The conversation manager uses an adaptive sentence gate to avoid premature responses after short pauses. The WebRTC microphone path uses Silero VAD to gate speech before transcription; it does not identify speakers or replace noise suppression.

## Development

```bash
uv sync
uv run pytest
```

Python dependencies are defined in `pyproject.toml` and locked in `uv.lock`.
