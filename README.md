# Waffle-Audio

A real-time, voice-to-voice AI assistant that runs entirely on your local machine.

You speak into a microphone. The assistant transcribes what you say, generates a reply
using a locally hosted language model, and speaks the reply back through your speakers.

## How it works

```
Microphone --> STT (Moonshine) --> LLM (Ollama) --> TTS (Kokoro) --> Speakers
```

| Stage | Technology         | Notes                                      |
|-------|--------------------|--------------------------------------------|
| STT   | Moonshine          | Streaming speech-to-text, runs on CPU/GPU  |
| LLM   | Ollama (local)     | Default model: qwen3.5:4b                  |
| TTS   | Kokoro ONNX        | Neural text-to-speech, ONNX runtime        |
| Audio | sounddevice / PortAudio | Cross-platform audio I/O              |

## Key features

- Echo cancellation: microphone input is muted while the assistant is speaking
- Adaptive sentence gate: fragments the user speaks across short pauses are
  accumulated before the LLM call, avoiding premature or duplicate responses
- Conversation history: every turn is tracked and printed on exit
- Voice-friendly LLM prompt: the model is instructed to reply in 1-2 plain
  sentences with no emojis or markdown

## Project structure

```
services/
  conversation/   Orchestrates STT -> LLM -> TTS
    src/
      manager.py          ConversationManager
      sentence_gate.py    Adaptive silence-gap gate
  stt/            Moonshine streaming transcription wrapper
  llm/            Ollama HTTP provider + queue-based service
  tts/            Kokoro ONNX synthesis service
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

Then open:

```text
http://localhost:3000/docs
```
