"""Voice cloning script for Waffle Audio (Pocket TTS backend).

This script allows users to clone a voice from an audio file or by recording
directly from their microphone, export the voice conditioning state as a
.safetensors embedding file, and verify it with a synthesized preview.

Cloned voices are stored in ``cloned_voices/<voice_name>.safetensors`` and can
be selected automatically in ``./run.sh`` or by passing ``--voice <voice_name>``
to the assistant.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

# Configure Hugging Face home to match project cache
from constants import (
    CLONED_VOICES_DIR,
    POCKET_TTS_CACHE_DIR,
    POCKET_TTS_DEFAULT_LANGUAGE,
    list_cloned_voices,
)

os.environ.setdefault("HF_HOME", str(Path(POCKET_TTS_CACHE_DIR).resolve()))

# ANSI terminal colors
BOLD = "\033[1m"
GREEN = "\033[38;5;46m"
CYAN = "\033[38;5;51m"
YELLOW = "\033[38;5;226m"
RED = "\033[38;5;196m"
DIM = "\033[2m"
RESET = "\033[0m"


class _Spinner:
    """Threaded terminal spinner to show active background progress."""

    def __init__(self, message: str) -> None:
        self.message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0
        start = time.time()
        while not self._stop.is_set():
            char = chars[idx % len(chars)]
            elapsed = time.time() - start
            print(f"\r  {CYAN}{char}{RESET} {self.message} {DIM}({elapsed:4.1f}s){RESET}", end="", flush=True)
            idx += 1
            time.sleep(0.08)

    def __enter__(self) -> "_Spinner":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop.set()
        self._thread.join()
        # Clear the spinner line
        print("\r" + " " * 80 + "\r", end="", flush=True)


def _check_and_prompt_hf_token() -> None:
    """Ensure Hugging Face token is present for downloading voice-cloning weights."""
    from huggingface_hub import get_token, login

    if get_token():
        return

    print(f"\n{YELLOW}{BOLD}! Hugging Face Authentication Required for Voice Cloning !{RESET}", flush=True)
    print(
        f"Pocket TTS uses gated voice-cloning weights from {CYAN}https://huggingface.co/kyutai/pocket-tts{RESET}.\n"
        "1. Accept the model terms on the page above (instant, free).\n"
        "2. Create or copy an Access Token from https://huggingface.co/settings/tokens (Read role).",
        flush=True,
    )

    if sys.stdin is not None and sys.stdin.isatty():
        try:
            token = input(f"\n{BOLD}Enter your Hugging Face token (leave empty to skip): {RESET}").strip()
            if token:
                os.environ["HF_TOKEN"] = token
                try:
                    login(token=token, add_to_git_credential=False)
                    print(f"{GREEN}✓ Logged in to Hugging Face successfully.{RESET}\n", flush=True)
                except Exception as exc:
                    print(f"{YELLOW}Warning during login: {exc}{RESET}\n", flush=True)
        except EOFError:
            pass


def _record_audio_from_microphone(duration_seconds: float = 10.0, sample_rate: int = 16000) -> Path:
    """Record speech from user's microphone using non-blocking InputStream and return temporary WAV path."""
    try:
        import sounddevice as sd
    except ImportError:
        print(f"{RED}Error: sounddevice is required for microphone recording.{RESET}", flush=True)
        sys.exit(1)

    print(f"\n{CYAN}{BOLD}[Microphone Recording]{RESET}", flush=True)
    print(f"Prepare to speak for {BOLD}{duration_seconds:.0f} seconds{RESET}.", flush=True)
    print("Tip: Read a book excerpt or speak naturally in a clear, quiet environment.", flush=True)

    for i in range(3, 0, -1):
        print(f"{YELLOW}Starting in {i}...{RESET}", flush=True)
        time.sleep(1.0)

    print(f"\n{GREEN}{BOLD}● RECORDING NOW... (Speak clearly into your microphone){RESET}\n", flush=True)

    recorded_chunks: list[np.ndarray] = []
    latest_level = 0.0

    def audio_callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        nonlocal latest_level
        chunk = indata.copy()
        recorded_chunks.append(chunk)
        # Compute simple peak level for visual mic activity
        latest_level = float(np.max(np.abs(chunk))) / 32768.0

    start_time = time.time()
    bar_len = 24

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=512,
        callback=audio_callback,
    ):
        while time.time() - start_time < duration_seconds:
            elapsed = time.time() - start_time
            remaining = max(0.0, duration_seconds - elapsed)
            progress = min(1.0, elapsed / duration_seconds)
            filled = int(progress * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)

            # Volume meter visual
            meter_bars = int(min(1.0, latest_level * 5.0) * 5)
            mic_meter = "▂▄▆█"[:meter_bars].ljust(4)

            print(
                f"\r  [{CYAN}{bar}{RESET}] {elapsed:4.1f}s / {duration_seconds:4.1f}s  "
                f"Mic: {GREEN}{mic_meter}{RESET} (remaining: {remaining:4.1f}s)",
                end="",
                flush=True,
            )
            time.sleep(0.08)

    bar = "█" * bar_len
    print(
        f"\r  [{GREEN}{bar}{RESET}] {duration_seconds:4.1f}s / {duration_seconds:4.1f}s  "
        f"Mic: {GREEN}DONE{RESET} (remaining:  0.0s)",
        flush=True,
    )

    if not recorded_chunks:
        print(f"\n{RED}Error: No audio samples were captured from microphone.{RESET}", flush=True)
        sys.exit(1)

    recording = np.concatenate(recorded_chunks, axis=0)
    print(f"\n{GREEN}✓ Recording finished ({duration_seconds:.1f}s, {len(recording):,} samples captured).{RESET}\n", flush=True)

    # Write to temporary WAV
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", prefix="voice_recording_", delete=False)
    wav_path = Path(temp_wav.name)
    temp_wav.close()

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(recording.tobytes())

    return wav_path


def _play_wav_file(wav_path: Path) -> None:
    """Play a WAV file through speakers cleanly with duration timeout."""
    try:
        import sounddevice as sd
        with wave.open(str(wav_path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16)
            if channels > 1:
                data = data.reshape(-1, channels)

        duration = len(data) / sample_rate
        sd.play(data, samplerate=sample_rate)
        time.sleep(duration + 0.15)
        sd.stop()
    except Exception as exc:
        print(f"{YELLOW}Could not play audio playback: {exc}{RESET}", flush=True)


def _sanitize_voice_name(name: str) -> str:
    """Sanitize voice name to safe alphanumeric/underscore string."""
    cleaned = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "custom_voice"


def clone_voice(
    audio_path: Path | str,
    voice_name: str,
    output_dir: Path | str = CLONED_VOICES_DIR,
    language: str = POCKET_TTS_DEFAULT_LANGUAGE,
    play_sample: bool = True,
) -> Path:
    """Extract voice characteristics from audio and export to .safetensors embedding."""
    from pocket_tts import TTSModel, export_model_state

    audio_path = Path(audio_path).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Source audio file not found: {audio_path}")

    voice_name = _sanitize_voice_name(voice_name)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_safetensors = out_dir / f"{voice_name}.safetensors"

    print(f"{CYAN}{BOLD}[1/4] Checking model & credentials...{RESET}", flush=True)
    _check_and_prompt_hf_token()

    print(f"Loading Pocket TTS model weights ({BOLD}{language}{RESET})...", flush=True)
    with _Spinner(f"Initializing Pocket TTS ({language})"):
        try:
            model = TTSModel.load_model(language=language)
        except Exception as exc:
            print(f"\r{RED}Error loading Pocket TTS model: {exc}{RESET}", flush=True)
            raise

    if not getattr(model, "has_voice_cloning", True):
        msg = (
            f"\n{RED}{BOLD}Voice cloning is not enabled in the current Pocket TTS weights.{RESET}\n"
            "Kyutai requires accepting terms at https://huggingface.co/kyutai/pocket-tts.\n"
            "Please visit that URL, click 'Agree and access repository', then run:\n"
            f"  {CYAN}uvx hf auth login{RESET}  (or set {CYAN}HF_TOKEN{RESET})\n"
            "and run this script again."
        )
        print(msg, flush=True)
        raise RuntimeError("Pocket TTS voice cloning weights not authorized.")

    print(f"{GREEN}✓ Model loaded successfully.{RESET}\n", flush=True)

    # 2. Extract acoustic embeddings
    print(
        f"{CYAN}{BOLD}[2/4] Cloning voice from '{audio_path.name}'...{RESET}\n"
        f"      Encoding speech with Mimi & projecting to speaker latent space...",
        flush=True,
    )
    start_time = time.perf_counter()
    with _Spinner("Extracting speaker characteristics"):
        state = model.get_state_for_audio_prompt(audio_path)
    extract_time = time.perf_counter() - start_time
    print(f"{GREEN}✓ Voice embedding extracted successfully in {extract_time:.2f}s.{RESET}\n", flush=True)

    # 3. Export safetensors
    print(f"{CYAN}{BOLD}[3/4] Exporting voice state to '{out_safetensors.name}'...{RESET}", flush=True)
    export_model_state(state, out_safetensors)
    size_mb = out_safetensors.stat().st_size / (1024 * 1024)
    print(f"{GREEN}✓ Saved cloned voice state: {BOLD}{out_safetensors}{RESET} ({size_mb:.2f} MB).\n", flush=True)

    # 4. Generate preview synthesis
    test_text = f"Hello! This is my newly cloned voice, {voice_name}, speaking for Waffle Audio."
    preview_wav = out_dir / f"{voice_name}_preview.wav"
    print(f"{CYAN}{BOLD}[4/4] Synthesizing speech preview...{RESET}", flush=True)
    print(f"      Text: \"{test_text}\"", flush=True)

    try:
        sample_rate = model.sample_rate
        audio_chunks: list[np.ndarray] = []
        with _Spinner("Synthesizing preview audio stream"):
            for chunk in model.generate_audio_stream(state, test_text):
                samples = chunk.detach().cpu().numpy().reshape(-1).astype(np.float32)
                audio_chunks.append(samples)

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            pcm_data = (np.clip(full_audio, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(str(preview_wav), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_data.tobytes())

            print(f"{GREEN}✓ Preview synthesized and saved to {preview_wav.name}.{RESET}", flush=True)

            if play_sample:
                print(f"{CYAN}▶ Playing preview through speakers...{RESET}", flush=True)
                _play_wav_file(preview_wav)
                print(f"{GREEN}✓ Playback complete.{RESET}\n", flush=True)
    except Exception as exc:
        print(f"{YELLOW}Warning during preview synthesis: {exc}{RESET}\n", flush=True)

    return out_safetensors


def interactive_mode() -> None:
    """Run an interactive wizard to guide the user through voice cloning."""
    print(f"{BOLD}{CYAN}")
    print("=" * 60)
    print("        WAFFLE AUDIO — VOICE CLONING WIZARD")
    print("=" * 60)
    print(f"{RESET}", flush=True)
    print("Clone any voice to use as your personal AI voice assistant.\n", flush=True)

    # 1. Ask for voice name
    default_name = f"voice_{int(time.time()) % 10000}"
    name_input = input(f"{BOLD}Enter a name for this cloned voice [{default_name}]: {RESET}").strip()
    voice_name = _sanitize_voice_name(name_input or default_name)

    # 2. Source selection
    print(f"\n{BOLD}Select audio source:{RESET}", flush=True)
    print("  [1] Record from microphone (recommended 10-20s)", flush=True)
    print("  [2] Provide an existing audio file (.wav, .mp3, etc.)", flush=True)

    choice = input(f"\n{BOLD}Choose [1 or 2, default: 1]: {RESET}").strip() or "1"

    temp_audio_to_clean: Path | None = None
    source_audio: Path

    if choice == "2":
        path_str = input(f"\n{BOLD}Enter the path to your audio file: {RESET}").strip()
        path = Path(path_str).expanduser().resolve()
        if not path.is_file():
            print(f"{RED}Error: File not found at '{path}'.{RESET}", flush=True)
            sys.exit(1)
        source_audio = path
    else:
        dur_str = input(f"\n{BOLD}Recording duration in seconds [default: 10]: {RESET}").strip() or "10"
        try:
            duration = max(3.0, float(dur_str))
        except ValueError:
            duration = 10.0

        recorded_wav = _record_audio_from_microphone(duration_seconds=duration)
        temp_audio_to_clean = recorded_wav
        source_audio = recorded_wav

        # Ask if user wants to hear their recording
        hear = input(f"{BOLD}Play back your recording to check it? [y/N]: {RESET}").strip().lower()
        if hear in ("y", "yes"):
            print("▶ Playing recording...", flush=True)
            _play_wav_file(recorded_wav)
            print("✓ Playback complete.\n", flush=True)

    try:
        out_path = clone_voice(
            audio_path=source_audio,
            voice_name=voice_name,
            output_dir=CLONED_VOICES_DIR,
            play_sample=True,
        )
        print(f"{BOLD}{GREEN}🎉 Voice '{voice_name}' cloned successfully!{RESET}", flush=True)
        print("To use this voice for your assistant:", flush=True)
        print(f"  • Run {CYAN}./run.sh{RESET} and select '{voice_name}' from the menu", flush=True)
        print(f"  • Or run: {CYAN}uv run python -m services.conversation.cli --voice {voice_name}{RESET}\n", flush=True)
    finally:
        if temp_audio_to_clean and temp_audio_to_clean.is_file():
            try:
                temp_audio_to_clean.unlink()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone a voice for Waffle Audio (Pocket TTS)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", type=str, help="Name of the cloned voice")
    parser.add_argument("--audio", type=str, help="Path to reference audio file (.wav, .mp3, etc.)")
    parser.add_argument("--record", action="store_true", help="Record audio sample directly from microphone")
    parser.add_argument("--duration", type=float, default=10.0, help="Microphone recording duration in seconds")
    parser.add_argument("--output-dir", type=str, default=str(CLONED_VOICES_DIR), help="Directory to save cloned voices")
    parser.add_argument("--language", type=str, default=POCKET_TTS_DEFAULT_LANGUAGE, help="Pocket TTS language model")
    parser.add_argument("--no-play-sample", action="store_true", help="Do not play sample playback after cloning")
    parser.add_argument("--list", action="store_true", help="List all currently available cloned voices")

    args = parser.parse_args()

    if args.list:
        voices = list_cloned_voices(args.output_dir)
        print(f"\n{BOLD}Cloned Voices in {args.output_dir}:{RESET}", flush=True)
        if not voices:
            print("  (No cloned voices found. Run 'uv run python clone_voice.py' to clone one!)", flush=True)
        else:
            for idx, v in enumerate(voices, start=1):
                print(f"  [{idx}] {v}", flush=True)
        print(flush=True)
        return

    # If no flags are provided, launch interactive wizard
    if not args.name and not args.audio and not args.record:
        interactive_mode()
        return

    # Non-interactive CLI mode
    voice_name = _sanitize_voice_name(args.name or f"voice_{int(time.time()) % 10000}")
    temp_audio_to_clean: Path | None = None

    if args.record:
        source_audio = _record_audio_from_microphone(duration_seconds=args.duration)
        temp_audio_to_clean = source_audio
    elif args.audio:
        source_audio = Path(args.audio).expanduser().resolve()
        if not source_audio.is_file():
            print(f"{RED}Error: Source audio file '{args.audio}' does not exist.{RESET}", flush=True)
            sys.exit(1)
    else:
        print(f"{RED}Error: Specify either --audio <path> or --record.{RESET}", flush=True)
        sys.exit(1)

    try:
        clone_voice(
            audio_path=source_audio,
            voice_name=voice_name,
            output_dir=args.output_dir,
            language=args.language,
            play_sample=not args.no_play_sample,
        )
    finally:
        if temp_audio_to_clean and temp_audio_to_clean.is_file():
            try:
                temp_audio_to_clean.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
