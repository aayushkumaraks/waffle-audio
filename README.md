# Waffle-Audio

A real-time, voice-to-voice AI assistant that runs entirely on your local machine.

You speak into a microphone. The assistant transcribes what you say, generates a reply
using a locally hosted language model, and speaks the reply back through your speakers.

## How it works

```
Microphone --> VAD (Silero) --> STT (Moonshine) --> LLM (Ollama) --> TTS (Kokoro / Pocket TTS) --> Speakers
```

| Stage | Technology         | Notes                                      |
|-------|--------------------|--------------------------------------------|
| STT   | Moonshine          | Streaming speech-to-text, runs on CPU/GPU  |
| LLM   | Ollama (local)     | Default model: qwen3.5:4b                  |
| TTS   | Kokoro / Pocket TTS | Selectable streaming neural text-to-speech |
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
  tts/            Feature-flagged Kokoro and Pocket TTS services
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

Set `VOICECHAT_TTS_BACKEND=pocket` to use Pocket TTS. Omitting it (or setting
`kokoro`) retains the existing Kokoro backend.

Then open:

```text
http://localhost:3000/docs
```


## Voice activity detection

The WebRTC microphone path uses Silero VAD before Moonshine. VAD only gates audio admission to STT; it does not change the existing STTListener events or ConversationManager / SentenceGate behavior.

The default VAD configuration is:

| Setting | Default |
|---|---:|
| Sample rate | 16 kHz |
| VAD threshold | 0.5 |
| Minimum silence | 300 ms |
| Speech padding | 100 ms |
| Pre-speech padding | 200 ms |
| Backend frame | 512 samples |

This component detects speech versus non-speech. It does not distinguish the user's voice from another speaker such as a TV. Target-speaker verification and noise suppression can be added independently.
