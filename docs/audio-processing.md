# Audio processing architecture

The audio-processing layer is organized around independent capabilities rather than rollout stages.

## Pipeline boundary

Browser microphone audio is converted to 16 kHz mono PCM and passed through preprocessing before reaching STT.

```text
WebRTC microphone
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

