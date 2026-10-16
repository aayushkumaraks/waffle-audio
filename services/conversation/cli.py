"""Interactive voice assistant CLI with Rich terminal UI and real-time audio pipeline."""
from __future__ import annotations

import contextlib
import logging
import os
import signal
import threading
import time

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live

from services.audio.src import AudioPlayer
from services.conversation.src import ConversationManager, ConversationUI
from services.llm.src import LLMProvider, LLMProviderConfig, LLMService
from services.stt.src import AudioQueueFull, STTService
from services.tts.src import TTSConfig, TTSService

# Suppress ONNX runtime logging
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")


def _setup_logging() -> None:
    """Redirect application logs to a file so the terminal UI stays completely clean."""
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/voicechat.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Remove any existing console/stream handlers
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(file_handler)

    # Silence verbose third-party loggers from console output
    for name in [
        "httpx",
        "httpcore",
        "onnxruntime",
        "moonshine_voice",
        "urllib3",
        "services",
        "pocket_tts",
        "kokoro_onnx",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)


@contextlib.contextmanager
def _suppress_c_stderr():
    """Suppress C-level ALSA / PortAudio device warning noise on Linux/WSL."""
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(null_fd, 2)
        os.close(null_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
    except Exception:
        yield


def main() -> None:
    _setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Waffle Audio CLI...")

    stop_event = threading.Event()

    def handle_signal(_signum=None, _frame=None) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    ui = ConversationUI()

    # Initialize audio and conversation services
    stt = STTService()
    provider = LLMProvider(LLMProviderConfig())
    llm = LLMService(provider)
    tts = TTSService(TTSConfig())
    player = AudioPlayer()

    tts.add_listener(player)
    manager = ConversationManager(stt=stt, llm=llm, tts=tts)

    # Wire UI listeners to real-time events
    stt.add_listener(ui)
    llm.add_listener(ui)

    # Start backend services
    player.start()
    tts.start()
    llm.start()
    manager.start()
    stt.start()

    def audio_callback(indata, frames, time_info, status):
        if player.is_playing:
            return
        try:
            # Copy flattened audio to prevent buffer reuse race conditions
            audio = indata.astype(np.float32).flatten()
            stt.push(audio, 16000)
        except AudioQueueFull:
            pass

    stream = None
    try:
        with _suppress_c_stderr():
            stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="float32",
                blocksize=512,
                callback=audio_callback,
            )
            stream.start()

        # Run alternate screen live UI with 10 Hz refresh
        console = Console()
        with Live(ui.render(0), console=console, screen=True, refresh_per_second=10) as live:
            frame = 0
            while not stop_event.is_set():
                ui.sync_audio_playback(player.is_playing)
                frame += 1
                live.update(ui.render(frame))
                time.sleep(0.08)

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.exception("Unexpected error in CLI loop: %s", exc)
    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        stt.stop()
        manager.stop()
        llm.stop()
        tts.stop()
        player.stop()

        # Print clean conversation summary after restoring normal terminal
        print("\n" + "=" * 60)
        print("WAFFLE AUDIO — Conversation History")
        print("=" * 60)
        history = ui.history
        if not history:
            print("No conversation messages recorded.")
        else:
            for idx, msg in enumerate(history, start=1):
                role_label = "You" if msg["role"] == "user" else "Waffle"
                print(f"{idx}. [{role_label}] {msg['text']}")
        print("=" * 60)
        print("Waffle Audio stopped cleanly.\n")


if __name__ == "__main__":
    main()
