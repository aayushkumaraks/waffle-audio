"""Manual smoke test for either feature-flagged TTS backend.

Examples:
    uv run python -m services.tts.app
    VOICECHAT_TTS_BACKEND=pocket uv run python -m services.tts.app
    uv run python -m services.tts.app --backend kokoro
"""

from __future__ import annotations

import argparse
import threading

from services.audio.src.audio_player import AudioPlayer
from services.tts.src.tts_service import TTSConfig, TTSListener, TTSService


class _CompletionListener(TTSListener):
    """Makes the smoke test wait for real synthesis completion."""

    def __init__(self) -> None:
        self.completed = threading.Event()
        self.chunk_count = 0

    def on_synthesis_started(self) -> None:
        print("Synthesis started.")

    def on_audio_chunk(self, audio, sample_rate: int) -> None:
        self.chunk_count += 1
        print(f"Chunk {self.chunk_count}: {len(audio)} samples @ {sample_rate} Hz")

    def on_synthesis_completed(self) -> None:
        self.completed.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a configured TTS backend.")
    parser.add_argument("--backend", choices=("kokoro", "pocket"), help="Override VOICECHAT_TTS_BACKEND for this run.")
    parser.add_argument("--text", default="Hello. This is a test of the TTS service.")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    tts = TTSService(TTSConfig(backend=args.backend))
    player = AudioPlayer()
    completion = _CompletionListener()
    tts.add_listener(player)
    tts.add_listener(completion)

    player.start()
    started = False
    try:
        tts.start()
        started = True
        print(f"Testing {type(tts).__name__}.")
        tts.speak(args.text)
        if not completion.completed.wait(args.timeout):
            raise TimeoutError(f"TTS did not complete within {args.timeout:g} seconds.")
        print(f"Synthesis completed with {completion.chunk_count} audio chunks.")
    finally:
        if started:
            tts.stop()
        player.stop()


if __name__ == "__main__":
    main()
