# Audio processing architecture

The audio-processing layer is organized around independent capabilities rather than rollout stages.

## Pipeline boundary

Microphone audio is converted to 16 kHz mono PCM and passed through preprocessing before reaching STT.

```text
Microphone (PortAudio / WebRTC)
    |
    v
VAD
    |
    v
Moonshine STT
    |
    v
ConversationManager
```

The processing layer exposes independent provider boundaries for:

- `VADService` — speech activity detection
- `NoiseSuppressor` — background-noise suppression
- `EchoCanceller` — acoustic echo cancellation
- `TargetSpeakerVerifier` — target-speaker verification
- `TargetSpeakerExtractor` — source separation
- `BargeInController` — interruption coordination

Each WebRTC peer owns its streaming VAD state, so one connection cannot reset another connection's speech-detection state.

## Provider independence

Capability protocols are intentionally small. Production implementations can be replaced independently, while pass-through implementations keep the pipeline composable during provider selection.

Audio-processing components remain independent of STT provider internals, LLM orchestration, TTS provider internals, and WebRTC signaling details. This keeps the processing layer testable and lets individual providers be evaluated without restructuring the rest of the voice pipeline.

## VAD Thresholds & Pre-Roll Audio

Voice activity detection is tuned for responsive real-time conversation:

- **Dynamic Silence Timeout**: The speech-end trailing silence window is set to 200 ms (within the 180–220 ms range). Shorter trailing windows eliminate dead air at the end of user utterances without prematurely segmenting speech.
- **Rolling Audio Pre-roll Ring Buffer**: `VADService` maintains a rolling 150–200 ms audio ring buffer (`pre_speech_padding_ms=200`). When voice activity is confirmed, the pre-roll frames are prepended to the STT audio stream. This prevents aggressive VAD decisions from clipping initial unvoiced consonants (`p`, `t`, `k`, `s`), maintaining high speech recognition accuracy.
- **Instant Speech Detection Signals**: `VADService` dispatches immediate callbacks on the first frame where speech activity begins, enabling instant barge-in and playback interruption.

## Resampling Efficiency

Audio stream resampling (e.g. converting 48 kHz WebRTC microphone audio down to 16 kHz mono for STT and Moonshine) is implemented via `resample_audio` in `services/audio_processing/src/resampler.py`:

- Prioritizes `soxr` when available for ultra-fast SIMD-accelerated resampling.
- Uses `scipy.signal.resample_poly` for polyphase FIR anti-aliasing filter downsampling when `soxr` is not installed, achieving high throughput without the CPU overhead or latency of heavy libraries like `librosa`.
- Provides deterministic fallback to linear interpolation if neither library is present.

## Acoustic Echo Cancellation (AEC)

Acoustic echo cancellation is provided by `AcousticEchoCanceller` in `services/audio_processing/src/echo_cancellation.py`:

- **Normalized Least Mean Squares (NLMS)** adaptive filter estimates room impulse response between speaker playback and microphone input.
- Reference audio emitted by the audio player is fed into the canceller via a circular delay line (`feed_playback_reference`).
- The predicted echo signal is subtracted from incoming microphone frames prior to VAD processing, preventing the assistant from hearing its own voice and mistaking it for user speech or false interruptions.
- Adapts filter coefficients with step-size learning rate $\mu$ and regularized power normalization $\epsilon$.

## Instant Interruption (Barge-In)

The `BargeInController` in `services/audio_processing/src/barge_in.py` orchestrates real-time conversational turn-taking:

- Hooks into `VADService`'s `VADListener.on_speech_started` event.
- When speech is detected while the assistant is speaking or processing:
  1. Calls `AudioPlayer.interrupt()` to immediately stop active PortAudio stream playback and purge the queued audio buffers.
  2. Calls `LLMService.abort()` / `LLMProvider.abort()` to terminate in-flight streaming generation and stop Ollama processing.
  3. Calls `BaseTTSService.interrupt()` to clear the speech synthesis queue and cancel queued sentences.
  4. Resets `ConversationManager` and `ClauseChunker` state for the incoming user turn.
- An atomic cancellation token ensures thread-safe and asyncio-safe teardown with sub-50ms reaction time.

## Unified In-Process Pipeline & Zero-Copy IPC

For local execution, `UnifiedPipeline` (`services/pipeline.py`) eliminates network serialization and TCP handshakes:

- Consolidates VAD, AEC, STT, LLM, TTS, Player, and Barge-In in a single asynchronous process.
- Audio chunks are passed directly in memory as contiguous NumPy `float32` arrays.
- For multi-process architectures where model isolation is desired, `SharedAudioBuffer` (`services/shared_memory.py`) provides zero-copy shared memory ring buffers backed by POSIX shared memory (`multiprocessing.shared_memory`).

## Audio Hardware Routing & Device Selection

Audio stream acquisition and playback honor user-selected audio device indices configured via environment variables or `.env`:

- `AUDIO_INPUT_DEVICE`: Specifies the microphone device index (integer) or exact device name substring passed directly to `sounddevice.InputStream(device=...)`. When unset or `"default"`, uses the system default audio input device.
- `AUDIO_OUTPUT_DEVICE`: Specifies the speaker/headphone device index or name passed to `sounddevice.OutputStream(device=...)` in `AudioPlayer`.
- During onboarding, `setup.py` queries `sounddevice.query_devices()` to enumerate and present all detected input devices (e.g. PulseAudio `RDPSource` under WSLg or hardware USB microphones) with their respective PortAudio device indices.



