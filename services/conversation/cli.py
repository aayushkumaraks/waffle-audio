"""Interactive voice assistant CLI with Rich terminal UI, low-latency audio, and verbose benchmark timeline."""
from __future__ import annotations

import argparse
import contextlib
import logging
import os
import signal
import threading
import time
from rich.console import Console

# Suppress ONNX runtime logging
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")


def _setup_logging() -> None:
    """Redirect application and benchmark logs to logs/ directory with immediate flushing."""
    from services.logging_config import setup_app_logging
    setup_app_logging()



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
    # 1. Start console and spinner IMMEDIATELY so the user gets instant feedback
    console = Console()
    status = console.status("[bold cyan]Initializing Waffle Audio pipeline...[/bold cyan]", spinner="dots")
    status.start()

    parser = argparse.ArgumentParser(description="Waffle Audio Voice Assistant CLI")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display real-time turn timeline and compare latencies against README benchmarks",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=os.getenv("VOICECHAT_TTS_VOICE"),
        help="Voice to use for TTS (predefined name, cloned voice name from cloned_voices/, or path to .safetensors)",
    )
    parser.add_argument(
        "--trigger-mode",
        action="store_true",
        default=os.getenv("VOICECHAT_TRIGGER_MODE", "false").lower() in ("true", "1", "yes"),
        help="Only process and respond when the trigger word 'Hey ASSISTANT_NAME' is spoken",
    )
    args, _ = parser.parse_known_args()

    _setup_logging()
    logger = logging.getLogger(__name__)

    # Pipe application and service logs to console during startup alongside spinner
    class _StartupLogHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                msg = self.format(record)
                console.log(f"[dim]{msg}[/dim]")
            except Exception:
                pass

    startup_log_handler = _StartupLogHandler()
    startup_log_handler.setLevel(logging.INFO)
    startup_log_handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s]: %(message)s"))
    logging.getLogger().addHandler(startup_log_handler)

    stop_event = threading.Event()

    def handle_signal(_signum=None, _frame=None) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    vad = None
    stt = None
    manager = None
    llm = None
    tts = None
    player = None
    stream = None
    ui = None
    Live = None
    README_TARGET_E2E = 2.10
    README_TARGET_FIRST_TOKEN = 0.45
    README_TARGET_GATE = 0.40
    README_TARGET_LLM_FULL = 1.47
    README_TARGET_TTS_FIRST = 1.82

    try:
        console.log(
            f"[bold cyan]Starting Waffle Audio CLI[/bold cyan] (verbose={args.verbose}, voice={args.voice or 'default'})"
        )
        logger.info("Starting Waffle Audio CLI (verbose=%s, voice=%s)...", args.verbose, args.voice or "default")

        status.update("[bold cyan]Loading core drivers & constants...[/bold cyan]")
        import numpy as np
        import sounddevice as sd
        from rich.live import Live as _Live
        Live = _Live
        from constants import (
            AEC_ENABLED,
            ASSISTANT_NAME,
            AUDIO_INPUT_DEVICE,
            BARGE_IN_ENABLED,
            OLLAMA_MODEL,
            TRIGGER_MODE,
        )
        console.log("[green]✓[/green] Core audio drivers loaded")

        # Initialize audio and conversation services
        status.update("[bold cyan]Initializing Voice Activity Detection (VAD) & Echo Cancellation...[/bold cyan]")
        from services.audio_processing.src import (
            AcousticEchoCanceller,
            BargeInController,
            VADConfig,
            VADService,
        )
        vad = VADService(VADConfig(min_silence_duration_ms=200, pre_speech_padding_ms=200))
        aec = AcousticEchoCanceller()
        console.log("[green]✓[/green] VAD & Echo Cancellation initialized")

        status.update("[bold cyan]Initializing Speech-to-Text (Moonshine ONNX)...[/bold cyan]")
        from services.stt.src import AudioQueueFull, STTService
        stt = STTService()
        console.log("[green]✓[/green] STT Service initialized")

        status.update(f"[bold cyan]Connecting to LLM ({OLLAMA_MODEL}) & loading MCP tools...[/bold cyan]")
        from services.llm.src import LLMProvider, LLMProviderConfig, LLMService
        provider = LLMProvider(LLMProviderConfig())
        llm = LLMService(provider)
        tool_count = len(provider.tool_manager.get_ollama_tools())
        console.log(f"[green]✓[/green] LLM Service initialized (model: {OLLAMA_MODEL}, tools available: {tool_count})")


        status.update("[bold cyan]Loading Text-to-Speech engine...[/bold cyan]")
        from services.tts.src import TTSConfig, TTSService
        tts = TTSService(TTSConfig(voice=args.voice))
        console.log("[green]✓[/green] TTS Service initialized")

        status.update("[bold cyan]Initializing Audio Playback system...[/bold cyan]")
        from services.audio.src import AudioPlayer
        player = AudioPlayer()
        player.set_upstream_busy(lambda: getattr(tts, "is_busy", False))
        player.add_playback_monitor(aec.feed_reference)
        tts.add_listener(player)
        provider.set_tts(tts)
        console.log("[green]✓[/green] Audio Player & Filler Phrases initialized")


        status.update("[bold cyan]Wiring Conversation Pipeline & Barge-In Controller...[/bold cyan]")
        from services.conversation.src import ConversationManager, ConversationUI
        from services.conversation.src.ui import (
            README_TARGET_E2E as _E2E,
            README_TARGET_FIRST_TOKEN as _FT,
            README_TARGET_GATE as _GATE,
            README_TARGET_LLM_FULL as _LF,
            README_TARGET_TTS_FIRST as _TF,
        )
        README_TARGET_E2E = _E2E
        README_TARGET_FIRST_TOKEN = _FT
        README_TARGET_GATE = _GATE
        README_TARGET_LLM_FULL = _LF
        README_TARGET_TTS_FIRST = _TF

        is_trigger_mode = bool(args.trigger_mode or TRIGGER_MODE)
        if is_trigger_mode:
            console.log(
                f"[yellow]Trigger mode ACTIVE[/yellow]: Assistant will only respond when 'Hey {ASSISTANT_NAME}' is spoken"
            )
            logger.info("Trigger mode active (trigger word: 'Hey %s')", ASSISTANT_NAME)

        ui = ConversationUI(
            verbose=args.verbose,
            trigger_mode=is_trigger_mode,
            assistant_name=ASSISTANT_NAME,
        )
        manager = ConversationManager(
            stt=stt,
            llm=llm,
            tts=tts,
            on_user_turn=ui.on_user_turn,
            trigger_mode=is_trigger_mode,
            assistant_name=ASSISTANT_NAME,
            on_user_turn_ignored=ui.on_user_turn_ignored,
        )
        barge_in = BargeInController(
            player=player,
            manager=manager,
            tts=tts,
            llm=llm,
            on_barge_in=ui.on_barge_in,
        )
        barge_in.enabled = BARGE_IN_ENABLED
        vad.add_listener(barge_in)

        stt.add_listener(ui)
        llm.add_listener(ui)
        player.add_playback_listener(ui)
        console.log("[green]✓[/green] Real-time event listeners wired")

        # Start backend services
        status.update("[bold cyan]Starting Voice Activity Detection worker...[/bold cyan]")
        vad.start()
        console.log("[green]✓[/green] VAD service started")

        status.update("[bold cyan]Starting Audio Player worker...[/bold cyan]")
        player.start()
        console.log("[green]✓[/green] Audio Player started")

        status.update("[bold cyan]Starting TTS engine (loading models & voices)...[/bold cyan]")
        tts.start()
        console.log("[green]✓[/green] TTS service started")

        status.update("[bold cyan]Starting LLM worker thread...[/bold cyan]")
        llm.start()
        console.log("[green]✓[/green] LLM service started")

        status.update("[bold cyan]Starting Conversation Manager...[/bold cyan]")
        manager.start()
        console.log("[green]✓[/green] Conversation Manager started")

        status.update("[bold cyan]Starting STT engine (loading Moonshine model)...[/bold cyan]")
        stt.start()
        console.log("[green]✓[/green] STT service started")

        def audio_callback(indata, frames, time_info, status_flag):
            if not BARGE_IN_ENABLED and player.is_playing:
                return
            try:
                audio = indata.astype(np.float32).flatten()
                if BARGE_IN_ENABLED and AEC_ENABLED:
                    cleaned = aec.process(audio, 16000)
                else:
                    cleaned = audio

                if BARGE_IN_ENABLED:
                    vad.process(cleaned, 16000)

                stt.push(cleaned, 16000)
            except AudioQueueFull:
                pass
            except Exception:
                logger.exception("Error in CLI audio callback.")

        status.update("[bold cyan]Opening microphone input stream (16 kHz)...[/bold cyan]")
        with _suppress_c_stderr():
            stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="float32",
                blocksize=512,
                callback=audio_callback,
                device=AUDIO_INPUT_DEVICE,
            )
            stream.start()
        console.log("[green]✓[/green] Microphone audio stream active")

        status.update("[bold green]All services ready. Launching UI...[/bold green]")
        time.sleep(0.3)
    finally:
        # Take the spinner off and detach console log handler once initialization completes
        logging.getLogger().removeHandler(startup_log_handler)
        status.stop()

    try:
        # Run alternate screen live UI with 10 Hz refresh
        with Live(ui.render(0, console=console), console=console, screen=True, refresh_per_second=10) as live:
            frame = 0
            while not stop_event.is_set():
                ui.sync_audio_playback(player.is_playing)
                frame += 1
                live.update(ui.render(frame, console=console))
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

        if vad is not None:
            vad.stop()
        if stt is not None:
            stt.stop()
        if manager is not None:
            manager.stop()
        if llm is not None:
            llm.stop()
        if tts is not None:
            tts.stop()
        if player is not None:
            player.stop()

        # Print clean conversation summary after restoring normal terminal
        print("\n" + "=" * 70)
        print("WAFFLE AUDIO — Conversation History")
        print("=" * 70)
        history = ui.history
        if not history:
            print("No conversation messages recorded.")
        else:
            for idx, msg in enumerate(history, start=1):
                role_label = "You" if msg["role"] == "user" else "Waffle"
                print(f"{idx}. [{role_label}] {msg['text']}")

        # Output benchmark summary and log to logs/
        if ui is not None and hasattr(ui, "benchmark_tracker"):
            summary_card = ui.benchmark_tracker.log_session_summary()
            if ui.benchmarks:
                print("\n" + summary_card)
                print("[dim]Detailed records appended to logs/voicechat.log, logs/benchmark.log, and logs/benchmarks.jsonl[/dim]\n")

        if args.verbose and ui.benchmarks:
            print("\n" + "=" * 70)
            print("PER-TURN LATENCY TIMELINE & README BENCHMARK COMPARISON")
            print("=" * 70)
            print(
                f"README Reference: Warm Turn E2E = {README_TARGET_E2E}s | Cold Start = 6.8s | Before Opt = 3.3s\n"
            )
            for i, bm in enumerate(ui.benchmarks, start=1):
                turn_type_str = f" [{bm.get('turn_type', 'conversation').upper().replace('_', ' ')}]" if bm.get('turn_type') else ""
                print(f"Turn #{i}{turn_type_str}:")
                print(f"  User       : \"{bm['user_text']}\"")
                print(f"  Assistant  : \"{bm['assistant_text']}\"")
                print("  Timeline Breakdown:")
                print(
                    f"    1. Sentence Gate      : {bm['gate_dur']:5.2f}s  (Target: {README_TARGET_GATE:4.2f}s  |  Δ {bm['gate_dur'] - README_TARGET_GATE:+.2f}s)"
                )
                print(
                    f"    2. LLM First Token    : {bm['first_token_dur']:5.2f}s  (Target: {README_TARGET_FIRST_TOKEN:4.2f}s  |  Δ {bm['first_token_dur'] - README_TARGET_FIRST_TOKEN:+.2f}s)"
                )
                if bm.get("total_tool_dur"):
                    tools_desc = ", ".join(t.get("name", "tool") for t in bm.get("tools", [])) or "tool"
                    print(
                        f"    ↳ Tool Execution      : {bm['total_tool_dur']:5.2f}s  ({tools_desc})"
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
