"""Generate PocketTTS voice cloning safetensor files for emotion audio clips.

Scans `voices/output/*.wav` and exports corresponding `.safetensors` embeddings
directly into `cloned_voices/`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from constants import (
    CLONED_VOICES_DIR,
    POCKET_TTS_CACHE_DIR,
    POCKET_TTS_DEFAULT_LANGUAGE,
)

# Configure HF cache
os.environ.setdefault("HF_HOME", str(Path(POCKET_TTS_CACHE_DIR).resolve()))

from pocket_tts import TTSModel, export_model_state
from services.tts.src.hf_cache import load_local_first


def generate_all_emotion_voices(
    input_dir: Path | str = "voices/output",
    output_dir: Path | str = CLONED_VOICES_DIR,
    language: str = POCKET_TTS_DEFAULT_LANGUAGE,
    overwrite: bool = False,
) -> list[Path]:
    in_dir = Path(input_dir).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {in_dir}")

    wav_files = sorted(in_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav audio clips found in {in_dir}")

    print(f"Found {len(wav_files)} audio clips in {in_dir}:")
    for wf in wav_files:
        print(f"  - {wf.name}")

    print(f"\nLoading Pocket TTS model weights ({language})...")
    t0 = time.time()

    def _loader() -> TTSModel:
        return TTSModel.load_model(language=language)

    model = load_local_first(_loader)
    print(f"Pocket TTS model loaded in {time.time() - t0:.2f}s.")

    if not getattr(model, "has_voice_cloning", True):
        raise RuntimeError("Pocket TTS voice cloning is not enabled.")

    generated_paths: list[Path] = []
    for wf in wav_files:
        stem = wf.stem  # e.g., 'angry_female', 'happy_male'
        out_safetensors = out_dir / f"{stem}.safetensors"

        if out_safetensors.is_file() and not overwrite and out_safetensors.stat().st_size > 1000:
            print(f"✓ Already exists: {out_safetensors.name} ({out_safetensors.stat().st_size / 1024 / 1024:.2f} MB), skipping.")
            generated_paths.append(out_safetensors)
            continue

        print(f"Extracting voice conditioning for '{wf.name}'...", end=" ", flush=True)
        t_start = time.perf_counter()
        state = model.get_state_for_audio_prompt(wf)
        export_model_state(state, out_safetensors)
        t_elapsed = time.perf_counter() - t_start
        size_mb = out_safetensors.stat().st_size / (1024 * 1024)
        print(f"done in {t_elapsed:.2f}s -> {out_safetensors.name} ({size_mb:.2f} MB)")
        generated_paths.append(out_safetensors)

    print(f"\nAll {len(generated_paths)} emotion voice models generated in {out_dir}.")
    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate emotion voice models as safetensors")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="voices/output",
        help="Directory containing emotion .wav clips",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CLONED_VOICES_DIR),
        help="Directory to save generated .safetensors files",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing .safetensors files",
    )
    args = parser.parse_args()

    generate_all_emotion_voices(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.force,
    )


if __name__ == "__main__":
    main()
