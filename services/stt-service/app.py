import time
import sounddevice as sd
import numpy as np

from src.stt_service import STTListener, STTService

class ConsoleListener(STTListener):
    def on_transcript_started(self, text: str) -> None:
        print(f"Started: {text}")

    def on_transcript_updated(self, text: str) -> None:
        print(f"Updating: {text}")

    def on_transcript_completed(self, text: str) -> None:
        print(f"Completed: {text}")


service = STTService()
service.start()
service.add_listener(ConsoleListener())


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)

    audio = indata.astype(np.float32).flatten()
    service.push(audio, 16000)


stream = sd.InputStream(
    samplerate=16000,
    channels=1,
    dtype="float32",
    blocksize=512,
    callback=audio_callback,
)

stream.start()

print("Listening... Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    stream.stop()
    stream.close()
    service.stop()