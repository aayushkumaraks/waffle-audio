"""Interactive microphone CLI using the shared ConversationManager pipeline."""
from __future__ import annotations

import logging
import signal
import threading

import numpy as np
import sounddevice as sd

from services.audio.src import AudioPlayer
from services.conversation.src import ConversationManager
from services.llm.src import LLMListener, LLMProvider, LLMProviderConfig, LLMService
from services.stt.src import AudioQueueFull, STTListener, STTService
from services.tts.src import TTSConfig, TTSService

# Keep library/worker chatter out of the end-user terminal. Errors still appear.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


class ConsoleListener(STTListener, LLMListener):
    def on_transcript_started(self, text: str) -> None:
        pass

    def on_transcript_updated(self, text: str) -> None:
        pass

    def on_transcript_completed(self, text: str) -> None:
        text = text.strip()
        if text:
            print(f"\nYou: {text}", flush=True)

    def on_generation_started(self) -> None:
        pass

    def on_generation_updated(self, text: str) -> None:
        pass

    def on_generation_completed(self, text: str) -> None:
        print(f"Waffle: {text}", flush=True)


def main() -> None:
    stop_event = threading.Event()
    stt = STTService()
    llm = LLMService(LLMProvider(LLMProviderConfig()))
    tts = TTSService(TTSConfig())
    player = AudioPlayer()
    tts.add_listener(player)
    manager = ConversationManager(stt=stt, llm=llm, tts=tts)
    console = ConsoleListener()
    stt.add_listener(console)
    llm.add_listener(console)

    def stop(_signum=None, _frame=None):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)

    stream = None
    try:
        print("Waffle Audio CLI is ready. Speak naturally; press Ctrl+C to quit.", flush=True)
        player.start()
        tts.start()
        llm.start()
        manager.start()
        stt.start()

        def audio_callback(indata, frames, time_info, status):
            if status:
                logging.getLogger(__name__).warning("Audio input: %s", status)
            if player.is_playing:
                return
            try:
                stt.push(np.asarray(indata, dtype=np.float32).reshape(-1), 16000)
            except AudioQueueFull:
                pass

        stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=512, callback=audio_callback)
        stream.start()
        while not stop_event.wait(0.25):
            pass
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logging.getLogger(__name__).error("Could not start CLI audio pipeline: %s", exc)
        raise
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        stt.stop()
        manager.stop()
        llm.stop()
        tts.stop()
        player.stop()
        print("Waffle Audio stopped.", flush=True)


if __name__ == "__main__":
    main()
