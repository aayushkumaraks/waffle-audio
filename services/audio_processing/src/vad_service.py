from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

AudioBuffer = npt.NDArray[np.float32]


class VADBackend(Protocol):
    """Stateful streaming VAD backend."""

    def process(self, audio: AudioBuffer) -> dict[str, float] | None:
        """Process exactly one backend-sized audio frame."""
        ...

    def reset(self) -> None:
        """Reset streaming state."""
        ...


@dataclass(slots=True)
class VADConfig:
    """Configuration for streaming voice activity detection."""

    enabled: bool = True
    sample_rate: int = 16_000
    threshold: float = 0.5
    min_silence_duration_ms: int = 300
    speech_pad_ms: int = 100
    pre_speech_padding_ms: int = 200
    frame_samples: int = 512


class SileroVADBackend:
    """Silero VAD implementation using the bundled ONNX model.

    Silero's ONNX model runs on CPU here deliberately. VAD is a small,
    latency-sensitive preprocessing step and should not compete with the
    application's STT/LLM/TTS workloads for GPU resources.
    """

    def __init__(self, config: VADConfig) -> None:
        from silero_vad import VADIterator, load_silero_vad

        self._iterator_cls = VADIterator
        self._config = config
        self._model = load_silero_vad(onnx=True)
        self._iterator = self._new_iterator()

    def _new_iterator(self):
        return self._iterator_cls(
            self._model,
            threshold=self._config.threshold,
            sampling_rate=self._config.sample_rate,
            min_silence_duration_ms=self._config.min_silence_duration_ms,
            speech_pad_ms=self._config.speech_pad_ms,
        )

    def process(self, audio: AudioBuffer) -> dict[str, float] | None:
        result = self._iterator(audio, return_seconds=False)
        if result is None:
            return None

        return {
            key: float(value)
            for key, value in result.items()
            if key in {"start", "end"}
        }

    def reset(self) -> None:
        self._iterator.reset_states()


class VADService:
    """Provider-independent streaming VAD gate.

    The service deliberately does not know anything about STT or
    ConversationManager. It accepts 16 kHz mono PCM and returns only audio
    belonging to speech regions.

    Non-speech audio is held in a short pre-roll buffer so the beginning of an
    utterance is not clipped while the VAD is deciding that speech started.
    While speech is active, audio is passed through unchanged, including the
    trailing silence required by streaming STT to finalize an utterance.
    """

    def __init__(
        self,
        config: VADConfig | None = None,
        *,
        backend: VADBackend | None = None,
    ) -> None:
        self._config = config or VADConfig()
        self._backend = backend

        self._input_buffer = np.empty(0, dtype=np.float32)
        self._pre_roll: deque[AudioBuffer] = deque(
            maxlen=self._pre_roll_capacity()
        )
        self._in_speech = False
        self._started = False

    @property
    def is_speech(self) -> bool:
        return self._in_speech

    def start(self) -> None:
        if self._started:
            return

        if self._backend is None and self._config.enabled:
            self._backend = SileroVADBackend(self._config)

        self._reset_stream_state()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return

        self._reset_stream_state()
        self._started = False

    def reset(self) -> None:
        """Reset VAD state without unloading the model."""
        self._require_started()
        self._reset_stream_state()

    def process(
        self,
        audio: AudioBuffer,
        sample_rate: int,
    ) -> AudioBuffer:
        """Return only audio accepted by the VAD.

        Input may contain arbitrary chunk sizes. Silero receives its required
        512-sample frames internally, while the returned audio remains
        float32 mono at the configured sample rate.
        """
        self._require_started()

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if sample_rate != self._config.sample_rate:
            raise ValueError(
                f"VAD expects {self._config.sample_rate} Hz audio, "
                f"received {sample_rate} Hz."
            )

        if not self._config.enabled:
            return audio.copy()

        if audio.size == 0:
            return np.empty(0, dtype=np.float32)

        self._input_buffer = np.concatenate((self._input_buffer, audio))

        output: list[AudioBuffer] = []
        frame_size = self._config.frame_samples

        while self._input_buffer.size >= frame_size:
            frame = self._input_buffer[:frame_size]
            self._input_buffer = self._input_buffer[frame_size:]

            event = self._backend.process(frame)  # type: ignore[union-attr]

            if self._in_speech:
                # Keep passing audio through while speech is active. This
                # includes the silence that causes Silero to emit "end".
                output.append(frame.copy())

                if event is not None and "end" in event:
                    self._in_speech = False
                    self._pre_roll.clear()
                continue

            # We are idle: retain a small amount of audio in case speech
            # begins during the current frame.
            self._pre_roll.append(frame.copy())

            if event is not None and "start" in event:
                output.extend(self._pre_roll)
                self._pre_roll.clear()
                self._in_speech = True

        if not output:
            return np.empty(0, dtype=np.float32)

        return np.concatenate(output).astype(np.float32, copy=False)

    def _pre_roll_capacity(self) -> int:
        return max(
            1,
            int(
                np.ceil(
                    self._config.pre_speech_padding_ms
                    / (self._config.frame_samples / self._config.sample_rate * 1000)
                )
            ),
        )

    def _reset_stream_state(self) -> None:
        self._input_buffer = np.empty(0, dtype=np.float32)
        self._pre_roll.clear()
        self._in_speech = False

        if self._backend is not None:
            self._backend.reset()

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("VADService has not been started.")
