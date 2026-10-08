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
