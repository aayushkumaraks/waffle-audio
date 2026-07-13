import tempfile
import time
from pathlib import Path

from services.tts.src.tts_service import TTSConfig, TTSService

from services.audio.src.audio_player import AudioPlayer


MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/latest/download/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/latest/download/voices-v1.0.bin"


def download(url: str, destination: Path) -> None:
    import urllib.request

    if destination.exists():
        return

    print(f"Downloading {destination.name}...")
    urllib.request.urlretrieve(url, destination)


def main() -> None:
    model_dir = Path(tempfile.gettempdir()) / "kokoro-onnx"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "kokoro-v1.0.onnx"
    voices_path = model_dir / "voices-v1.0.bin"

    download(MODEL_URL, model_path)
    download(VOICES_URL, voices_path)

    config = TTSConfig(
        model_path=str(model_path),
        voices_path=str(voices_path),
    )

    player = AudioPlayer()
    player.start()

    tts = TTSService(config)
    tts.add_listener(player)

    tts.start()

    tts.speak("Hello. This is a test of the TTS service.")

    time.sleep(5)

    tts.stop()
    player.stop()


if __name__ == "__main__":
    main()