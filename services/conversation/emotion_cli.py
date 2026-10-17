"""Interactive emotional voice assistant CLI with Rich terminal UI, multi-threaded voice models, and live emotion switching."""
from __future__ import annotations

import argparse
import contextlib
import logging
import os
import signal
import sys
import threading
import time

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live

from constants import AVAILABLE_GENDERS
from services.audio.src import AudioPlayer
from services.conversation.src.emotion_manager import EmotionConversationManager
from services.conversation.src.emotion_ui import EmotionConversationUI
from services.llm.src import LLMProvider, LLMProviderConfig, LLMService
from services.stt.src import AudioQueueFull, STTService
from services.tts.src import EmotionTTSService

# Suppress ONNX runtime logging
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")


def _setup_logging() -> None:
    """Redirect application logs to a file so the terminal UI stays completely clean."""
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/emotion_voicechat.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(file_handler)

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


def prompt_user_gender() -> str:
    """Interactively ask user to choose voice gender (Female or Male)."""
    print("\n" + "=" * 60)
    print("  WAFFLE AUDIO — Emotional Voice Assistant")
    print("=" * 60)
    print("Select Assistant Voice Gender:")
    print("  [1] Female (Emotions: neutral, happy, sad, angry)")
    print("  [2] Male   (Emotions: neutral, happy, sad, angry)")
    print("=" * 60)

    if not sys.stdin.isatty():
        return "female"

    try:
        choice = input("Enter choice [1-2, default: 1 (Female)]: ").strip().lower()
        if choice in ("2", "male", "m"):
            print("✓ Selected Voice Gender: Male\n")
            return "male"
        print("✓ Selected Voice Gender: Female\n")
        return "female"
    except (EOFError, KeyboardInterrupt):
        return "female"


def main() -> None:
    parser = argparse.ArgumentParser(description="Waffle Audio Emotional Voice Assistant CLI")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display verbose information in terminal",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default=os.getenv("VOICECHAT_EMOTION_GENDER"),
        choices=["female", "male", "Female", "Male"],
        help="Gender for the emotional voice models (female or male)",
    )
    parser.add_argument(
        "--offload-idle",
        type=float,
        default=120.0,
        help="Idle timeout in seconds before offloading inactive emotion models from memory (default: 120s)",
    )
    args, _ = parser.parse_known_args()

    # Determine gender: CLI argument, environment, or interactive prompt
    gender = args.gender
    if not gender:
        gender = prompt_user_gender()
    gender = gender.lower()

    _setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Waffle Audio Emotion CLI (gender=%s, offload_idle=%s)...", gender, args.offload_idle)

    stop_event = threading.Event()

    def handle_signal(_signum=None, _frame=None) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    ui = EmotionConversationUI(gender=gender, verbose=args.verbose)

    # 1. Initialize STT
    stt = STTService()

    # 2. Initialize LLM with Emotion System Prompt
    llm_config = LLMProviderConfig(system_prompt=LLMProvider.EMOTION_SYSTEM_PROMPT)
    provider = LLMProvider(llm_config)
    llm = LLMService(provider)

    # 3. Initialize Multi-Threaded Emotion TTS Service
    emotion_tts = EmotionTTSService(
        gender=gender,
        offload_idle_seconds=args.offload_idle,
    )

    # 4. Initialize Audio Player
    player = AudioPlayer()
    emotion_tts.add_listener(player)

    # 5. Initialize Emotion Conversation Manager
    manager = EmotionConversationManager(
        stt=stt,
        llm=llm,
        tts=emotion_tts,
        on_user_turn=ui.on_user_turn,
        on_emotion_change=ui.set_emotion,
    )

    # Wire UI listeners
    stt.add_listener(ui)
    llm.add_listener(ui)
    player.add_playback_listener(ui)

    # Start backend services
    player.start()
    emotion_tts.start()
    llm.start()
    manager.start()
    stt.start()

    def audio_callback(indata, frames, time_info, status):
        if player.is_playing:
            return
        try:
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

        console = Console()
        with Live(ui.render(0, console=console), console=console, screen=True, refresh_per_second=10) as live:
            frame = 0
            while not stop_event.is_set():
                ui.sync_audio_playback(player.is_playing)
                ui.update_worker_statuses(emotion_tts.get_statuses())
                frame += 1
                live.update(ui.render(frame, console=console))
                time.sleep(0.08)

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.exception("Unexpected error in Emotion CLI loop: %s", exc)
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
        emotion_tts.stop()
        player.stop()

        print("\n" + "=" * 70)
        print("WAFFLE AUDIO — Emotional Conversation Summary")
        print("=" * 70)
        history = ui.history
        if not history:
            print("No conversation messages recorded.")
        else:
            for idx, msg in enumerate(history, start=1):
                if msg["role"] == "user":
                    print(f"{idx}. [You] {msg['text']}")
                else:
                    emo_tag = f" [{msg.get('emotion', 'neutral').upper()}]"
                    print(f"{idx}. [Waffle{emo_tag}] {msg['text']}")

        if args.verbose and ui.benchmarks:
            from services.conversation.src.emotion_ui import (
                README_TARGET_E2E,
                README_TARGET_FIRST_TOKEN,
                README_TARGET_GATE,
                README_TARGET_LLM_FULL,
                README_TARGET_TTS_FIRST,
            )
            print("\n" + "=" * 70)
            print("LATENCY TIMELINE & README BENCHMARK COMPARISON")
            print("=" * 70)
            print(
                f"README Reference: Warm Turn E2E = {README_TARGET_E2E}s | Cold Start = 6.8s | Before Opt = 3.3s\n"
            )
            for i, bm in enumerate(ui.benchmarks, start=1):
                print(f"Turn #{i}:")
                print(f"  User       : \"{bm['user_text']}\"")
                print(f"  Assistant  : \"{bm['assistant_text']}\"")
                print("  Timeline Breakdown:")
                print(
                    f"    1. Sentence Gate      : {bm['gate_dur']:5.2f}s  (Target: {README_TARGET_GATE:4.2f}s  |  Δ {bm['gate_dur'] - README_TARGET_GATE:+.2f}s)"
                )
                print(
                    f"    2. LLM First Token    : {bm['first_token_dur']:5.2f}s  (Target: {README_TARGET_FIRST_TOKEN:4.2f}s  |  Δ {bm['first_token_dur'] - README_TARGET_FIRST_TOKEN:+.2f}s)"
                )
                print(
                    f"    3. LLM Full Reply     : {bm['full_llm_dur']:5.2f}s  (Target: {README_TARGET_LLM_FULL:4.2f}s  |  Δ {bm['full_llm_dur'] - README_TARGET_LLM_FULL:+.2f}s)"
                )
                print(
                    f"    4. TTS First Speech   : {bm['tts_first_dur']:5.2f}s  (Target: {README_TARGET_TTS_FIRST:4.2f}s  |  Δ {bm['tts_first_dur'] - README_TARGET_TTS_FIRST:+.2f}s)"
                )
                print(
                    f"    5. Total E2E Latency  : {bm['e2e_dur']:5.2f}s  (Target: {README_TARGET_E2E:4.2f}s  |  Δ {bm['e2e_dur'] - README_TARGET_E2E:+.2f}s)"
                )
                print(f"    Total Turn Duration   : {bm['total_dur']:5.2f}s\n")

        print("=" * 70)
        print("Waffle Audio stopped cleanly.\n")


if __name__ == "__main__":
    main()
