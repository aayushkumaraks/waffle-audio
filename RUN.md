# Running the app

## Start Ollama

The LLM service requires Ollama to be running before the app starts.

```
ollama serve
```

If Ollama is already running as a background service (e.g., started automatically on
login), you can skip this step.

---

## Start the app

From the project root:

```
uv run python -m services.conversation.app
```

You should see output similar to:

```
2026-08-02 09:00:00,000 INFO services.tts.src.tts_service: TTSService started.
2026-08-02 09:00:00,010 INFO services.conversation.src.manager: ConversationManager started.
2026-08-02 09:00:00,020 INFO services.stt.src.stt_service: Loading Moonshine model...
2026-08-02 09:00:02,000 INFO services.stt.src.stt_service: STTService started.
2026-08-02 09:00:02,010 INFO __main__: Listening... Press Ctrl+C to stop.
```

Once you see "Listening...", speak into your microphone. The assistant will reply
through your speakers.

---

## Stop the app

Press `Ctrl+C`. The app will stop cleanly and print the full conversation history:

```
Conversation History
--------------------
1. [user] Hello, how are you?
2. [assistant] I am doing well, thank you for asking.
```

---

## Configuration

All configuration is in source; there is no separate config file.

| Setting        | Location                                     | Default              |
|----------------|----------------------------------------------|----------------------|
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

**No audio input / output**
Run `uv run python -c "import sounddevice; print(sounddevice.query_devices())"` to list
devices. On Linux/WSL2 ensure PortAudio is installed (`sudo apt install libportaudio2`).

**Moonshine model download is slow**
The Moonshine model is downloaded automatically on first start and cached locally.
Subsequent starts are fast.

**The assistant does not respond**
Check the terminal for logged warnings. A common cause is the STT transcribing silence
or noise; the sentence gate will discard empty transcripts automatically.
