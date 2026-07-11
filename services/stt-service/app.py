import time
import sounddevice as sd
import numpy as np

from moonshine_voice import TranscriptEventListener

from src.stt_service import STTService


class ConsoleListener(TranscriptEventListener):
    def on_line_started(self, event):
        print(f"Started: {event.line.text}")

    def on_line_text_changed(self, event):
        print(f"Updating: {event.line.text}")

    def on_line_completed(self, event):
        print(f"Completed: {event.line.text}")


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