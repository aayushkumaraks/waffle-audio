"""Smoke-test Pocket TTS streaming independently of the application."""

import time

import numpy as np
from pocket_tts import TTSModel

from constants import POCKET_TTS_DEFAULT_LANGUAGE, POCKET_TTS_DEFAULT_VOICE


def main() -> None:
    print("Loading Pocket TTS...")
    init_start = time.perf_counter()
    model = TTSModel.load_model(language=POCKET_TTS_DEFAULT_LANGUAGE)
    voice_state = model.get_state_for_audio_prompt(POCKET_TTS_DEFAULT_VOICE)
    print(f"Pocket TTS initialized in {time.perf_counter() - init_start:.3f}s\n")

    print("Streaming...\n")
    synthesis_start = time.perf_counter()
    first_chunk_time: float | None = None
    chunk_count = total_samples = 0

    for audio in model.generate_audio_stream(voice_state, "Hello world. " * 20):
        now = time.perf_counter()
        samples = audio.detach().cpu().numpy().reshape(-1).astype(np.float32)
        chunk_count += 1
        total_samples += len(samples)
        if first_chunk_time is None:
            first_chunk_time = now
            print(f"First audio after {now - synthesis_start:.3f}s\n")
        print(f"Chunk {chunk_count:02d} | {len(samples):6d} samples")

    total_time = time.perf_counter() - synthesis_start
    print("\nDone.")
    print(f"Chunks           : {chunk_count}")
    print(f"Total samples    : {total_samples}")
    print(f"Sample rate      : {model.sample_rate}")
    print(f"Synthesis time   : {total_time:.3f}s")


if __name__ == "__main__":
    main()
