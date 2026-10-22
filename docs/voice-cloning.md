# Voice Cloning & Custom Voices Guide

Waffle Audio allows you to clone any custom voice from a short 10–15 second audio sample, giving your assistant your own voice, a friend's voice, or a persona of your choice.

---

## 1. How Voice Cloning Works

Voice cloning uses Kyutai Pocket TTS / Kokoro voice embeddings. A reference audio sample is analyzed to extract a speaker state embedding, which is then saved as a `.safetensors` file inside the `cloned_voices/` directory.

---

## 2. Cloning a Voice with `clone_voice.py`

Launch the voice cloning CLI:

```bash
uv run python clone_voice.py
```

### Option A: Record Directly from Microphone
1. Select option `[1] Record from microphone`.
2. Speak clearly into your microphone for 10–15 seconds (reading a passage, story, or speaking naturally).
3. Listen to the generated preview audio (`<name>_preview.wav`).
4. Save the voice.

### Option B: Use an Audio File
1. Place a clean `.wav` or `.mp3` file (10–30 seconds of clear speech without background music or noise) in your project.
2. Select option `[2] Load audio file`.
3. Enter the path to your audio file.
4. Name the voice and save.

---

## 3. Using Your Cloned Voice

Once saved, cloned voices are stored in `cloned_voices/<voice_name>.safetensors`.

### 1. Automatic Startup Prompt:
When you launch `./run.sh` or `run.bat`, Waffle automatically detects all `.safetensors` files in `cloned_voices/` and prompts you to select your preferred voice.

### 2. Via CLI Flag:
Pass the name of your cloned voice directly to the launcher:
```bash
./run.sh --voice my_voice
```

### 3. Via `.env` Configuration:
Set your default voice in `.env`:
```env
VOICECHAT_TTS_VOICE="my_voice"
```
Or run `uv run python setup.py --force` to choose your cloned voice as the default.

---

## 4. Unit Test Verification

Voice cloning and voice embedding resolution are verified with `tests/test_voice_cloning.py`:
- Cloned voice directory scanning (`list_cloned_voices`)
- Specifier resolution fallback (`resolve_voice_spec`)
- Voice name sanitization and microphone recording mocks

```bash
uv run pytest tests/test_voice_cloning.py
```
