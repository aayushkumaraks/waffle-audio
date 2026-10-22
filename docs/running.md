# Running Waffle Audio & CLI User Guide

This guide covers running the conversational voice assistant, CLI options, trigger modes, benchmarking tools, and the background HTTP API service.

---

## 1. Quick Launch (Recommended)

Run the assistant with default configuration:

```bash
./run.sh
```

On Windows:
```cmd
run.bat
```

Or run directly with `uv`:
```bash
uv run python -m services.conversation.cli
```

---

## 2. Command Line Arguments & Flags

The CLI supports flags to customize runtime behavior:

| Flag | Short | Default | Description |
|---|---|---|---|
| `--verbose` | `-v` | `false` | Displays real-time turn timeline and compares latencies against README benchmarks |
| `--benchmark` | | `false` | Enables latency benchmarking, tool metrics logging, and post-session latency averages |
| `--voice <name>` | | from `.env` | Select TTS voice (built-in e.g. `anna`, `af_sarah`, or custom cloned voice name) |
| `--trigger-mode` | | `false` | Wake phrase mode: only processes speech when `"Hey <ASSISTANT_NAME>"` is spoken |
| `--setup` | | `false` | Force launches interactive setup wizard before starting assistant |

### Examples:

1. **Verbose Profiling & Benchmark Mode**:
   ```bash
   ./run.sh -v --benchmark
   ```
   *Displays real-time timeline breakdowns in the terminal UI showing Sentence Gate duration, TTFT, TTS start time, and total E2E latency on each turn.*

2. **Trigger Word Mode**:
   ```bash
   ./run.sh --trigger-mode
   ```
   *Waffle ignores background conversations and room chatter until you say `"Hey Waffle"`. Once triggered, a 15-second follow-up conversational window allows seamless natural dialogue.*

3. **Custom Cloned Voice**:
   ```bash
   ./run.sh --voice Hariom_voice
   ```

---

## 3. CLI Terminal UI Features

When running the CLI, Waffle Audio renders a rich terminal interface:

- **Header Status Badge**:
  - `● READY` (Cyan): System idle, listening for audio.
  - `🎤 LISTENING` (Green): Speech detected by Silero VAD; audio streaming to Moonshine STT.
  - `⠋ PROCESSING` (Yellow): Sentence gate closed; streaming LLM tokens & synthesizing speech.
- **Waveform Animation**: Dynamic ASCII audio waveform animates in the header during speech.
- **Live Streaming Transcript**: As you speak, transcribed words update in real time.
- **Synchronized Audio & Text Playback**: Assistant responses stream word-by-word synchronously as audio plays out of your speakers.
- **Auto-Scrolling Window**: The terminal dynamically scrolls to keep current dialogue and responses in view.
- **Zero Log Spam**: All driver warnings and background logs are cleanly redirected to `logs/voicechat.log`.
- **Exit Summary**: When you press `Ctrl+C`, the CLI presents a clean summary of turns and latency metrics.

---

## 4. Conversational Turn-Taking & Interruption (Barge-In)

Waffle Audio is designed for natural two-way conversation:

1. **Natural Pauses (Sentence Gate)**:
   - When you finish a sentence with punctuation (`?`, `!`, `.`), Waffle quick-fires in ~250 ms.
   - For unpunctuated pauses or hesitation ("I was wondering... um..."), Waffle waits 500 ms before deciding you have finished.
2. **Instant Interruption (Barge-In)**:
   - You do not need to wait for Waffle to finish speaking.
   - Speak at any moment: Acoustic Echo Cancellation (AEC) subtracts speaker audio so Waffle hears you clearly, stops playback in <50 ms, cancels ongoing LLM generation, purges TTS queues, and immediately listens to your new statement.

---

## 5. HTTP API Server

For headless, programmatic, or remote network integration, Waffle Audio exposes a FastAPI server:

### Starting the API:
```bash
uv run python -m services.api.app
```

The API starts on `http://0.0.0.0:3000` (configurable via `VOICECHAT_API_PORT` and `VOICECHAT_API_HOST`).

### Interactive Documentation (Swagger UI):
Open your browser at:
```text
http://localhost:3000/docs
```

### Main Endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Service health status and runtime configuration |
| `/audio/status` | `GET` | Current speaker playback state (`is_playing`) |
| `/llm/generate` | `POST` | Generate one-shot completion from chat messages |
| `/tts/speak` | `POST` | Queue text for server-side speaker playback |
| `/tts/synthesize` | `POST` | Synthesize text and return raw WAV audio bytes |
| `/stt/push` | `POST` | Push raw audio samples for speech-to-text |
| `/stt/transcripts` | `GET` | Retrieve transcribed text events |
| `/conversation/respond` | `POST` | Submit finalized text into conversational pipeline |
| `/conversation/barge-in` | `POST` | Instant interrupt: stops playback, aborts LLM, clears TTS |
| `/conversation/history` | `GET` | Retrieve conversation turn history |
| `/conversation/history` | `DELETE`| Clear active conversation history |

---

## 6. Embedded Python Usage (`UnifiedPipeline`)

If you are integrating Waffle directly into a Python application without HTTP or CLI overhead:

```python
import asyncio
from services.pipeline import UnifiedPipeline

async def main():
    pipeline = UnifiedPipeline()
    await pipeline.start()

    # The pipeline runs VAD, STT, LLM, TTS, AEC, and Player in-process
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 7. Running the Unit Test Suite

Waffle Audio includes 211 deterministic unit tests covering all components:

```bash
# Run complete test suite with pytest
uv run pytest

# Run with standard Python unittest runner
uv run python -m unittest discover tests

# Run specific test modules
uv run pytest tests/test_api_service.py
uv run pytest tests/test_audio_processing.py
uv run pytest tests/test_install_tool.py
```
