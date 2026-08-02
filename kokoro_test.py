import asyncio
import tempfile
import time
from pathlib import Path

from kokoro_onnx import Kokoro

from constants import KOKORO_MODEL_DOWNLOAD_URL as MODEL_URL, KOKORO_VOICES_DOWNLOAD_URL as VOICES_URL


def download(url: str, destination: Path) -> None:
    import urllib.request

    if destination.exists():
        return

    print(f"Downloading {destination.name}...")
    urllib.request.urlretrieve(url, destination)


async def main() -> None:
    model_dir = Path(tempfile.gettempdir()) / "kokoro-onnx"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "kokoro-v1.0.onnx"
    voices_path = model_dir / "voices-v1.0.bin"

    download(MODEL_URL, model_path)
    download(VOICES_URL, voices_path)

    print("Loading Kokoro...")
    init_start = time.perf_counter()

    tts = Kokoro(
        model_path=str(model_path),
        voices_path=str(voices_path),
    )

    init_time = time.perf_counter() - init_start
    print(f"Kokoro initialized in {init_time:.3f}s\n")

    print("Available voices:")
    print(tts.get_voices())
    print()

    text = "Hello world. " * 200

    print("Streaming...\n")

    synthesis_start = time.perf_counter()
    previous_chunk_time = synthesis_start
    first_chunk_time = None

    chunk_count = 0
    total_samples = 0
    sample_rate = None

    async for audio, sr in tts.create_stream(
        text=text,
        voice="af_sarah",
    ):
        now = time.perf_counter()

        chunk_count += 1
        total_samples += len(audio)
        sample_rate = sr

        if first_chunk_time is None:
            first_chunk_time = now
            print(
                f"First audio after "
                f"{first_chunk_time - synthesis_start:.3f}s\n"
            )

        print(
            f"Chunk {chunk_count:02d} | "
            f"{len(audio):6d} samples | "
            f"{audio.nbytes / 1024:7.1f} KB | "
            f"+{now - previous_chunk_time:.3f}s"
        )

        print(
            f"    dtype={audio.dtype}, "
            f"shape={audio.shape}, "
            f"min={audio.min():.3f}, "
            f"max={audio.max():.3f}"
        )

        previous_chunk_time = now

    total_time = time.perf_counter() - synthesis_start

    print("\nDone.")
    print(f"Chunks           : {chunk_count}")
    print(f"Total samples    : {total_samples}")
    print(f"Sample rate      : {sample_rate}")
    print(f"Synthesis time   : {total_time:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())