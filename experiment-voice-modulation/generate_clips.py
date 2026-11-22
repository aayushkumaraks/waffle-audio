"""Generate WAV voice clips for every female emotion voice clone tensor.

Scans `cloned_voices/*_female.safetensors`, synthesizes a standardized prompt
with Pocket TTS for each tensor, and saves 16-bit PCM WAV files plus metadata
into `experiment-voice-modulation/output/`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# Ensure repository root is importable when running from any directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import (
    CLONED_VOICES_DIR,
    POCKET_TTS_CACHE_DIR,
    POCKET_TTS_DEFAULT_LANGUAGE,
    TTS_QUANTIZE,
)

# Pin Hugging Face cache before importing pocket_tts
os.environ.setdefault("HF_HOME", str((REPO_ROOT / POCKET_TTS_CACHE_DIR).resolve()))

from services.tts.src.hf_cache import load_local_first

DEFAULT_TEXT = (
    "Hello! I wanted to tell you all about what happened earlier today. "
    "We walked down along the quiet river path as the afternoon sun began to dip behind the hills, "
    "and the entire sky slowly turned a bright, glowing orange. "
    "On the way back, we stopped at a small corner bakery, shared a warm cup of tea, "
    "and talked for nearly an hour about our plans for the coming season. "
    "It was one of those simple moments that really stays with you long after the day is over."
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "output"


def discover_female_tensors(cloned_voices_dir: Path) -> list[Path]:
    """Return sorted list of all female emotion voice clone safetensor files."""
    if not cloned_voices_dir.is_dir():
        raise FileNotFoundError(f"Cloned voices directory not found: {cloned_voices_dir}")
    tensors = sorted(
        p for p in cloned_voices_dir.glob("*_female.safetensors")
        if p.is_file() and not p.name.startswith(".")
    )
    if not tensors:
        raise FileNotFoundError(
            f"No *_female.safetensors files found in {cloned_voices_dir}"
        )
    return tensors


def synthesize_tensor_to_wav(
    model,
    tensor_path: Path,
    text: str,
    output_wav: Path,
    seed: int | None = 42,
) -> dict:
    """Synthesize `text` using a single voice clone tensor and write a 16-bit PCM WAV."""
    import torch

    if seed is not None:
        torch.manual_seed(seed)

    t_load = time.perf_counter()
    voice_state = model.get_state_for_audio_prompt(str(tensor_path.resolve()))
    load_ms = (time.perf_counter() - t_load) * 1000.0

    t_synth = time.perf_counter()
    chunks: list[np.ndarray] = []
    for chunk in model.generate_audio_stream(voice_state, text):
        arr = chunk.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
        if arr.size > 0:
            chunks.append(arr)
    synth_s = time.perf_counter() - t_synth

    if not chunks:
        raise RuntimeError(f"Pocket TTS returned no audio chunks for {tensor_path.name}")

    audio = np.concatenate(chunks, axis=0)
    sample_rate = int(model.sample_rate)
    duration_s = float(len(audio) / sample_rate)

    # Save 16-bit PCM WAV (clipped safely to [-1.0, 1.0])
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_wav), sample_rate, pcm16)

    emotion = tensor_path.stem.removesuffix("_female")
    return {
        "emotion": emotion,
        "tensor_file": str(tensor_path.relative_to(REPO_ROOT))
        if tensor_path.is_relative_to(REPO_ROOT)
        else str(tensor_path),
        "wav_file": str(output_wav.name),
        "sample_rate": sample_rate,
        "num_samples": int(len(audio)),
        "duration_s": round(duration_s, 3),
        "load_ms": round(load_ms, 2),
        "synthesis_s": round(synth_s, 3),
        "rtf": round(synth_s / duration_s, 3) if duration_s > 0 else None,
        "peak_abs": round(float(np.max(np.abs(audio))), 4),
    }


def generate_clips(
    cloned_voices_dir: Path | str | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    text: str = DEFAULT_TEXT,
    language: str = POCKET_TTS_DEFAULT_LANGUAGE,
    quantize: bool = TTS_QUANTIZE,
    seed: int | None = 42,
) -> dict:
    """Generate WAV clips for all female emotion tensors and write metadata.json."""
    voices_dir = (
        Path(cloned_voices_dir).resolve()
        if cloned_voices_dir is not None
        else (REPO_ROOT / CLONED_VOICES_DIR).resolve()
    )
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = discover_female_tensors(voices_dir)
    print(f"Discovered {len(tensors)} female emotion voice clone tensors in {voices_dir}:")
    for t in tensors:
        print(f"  - {t.name} ({t.stat().st_size / (1024 * 1024):.2f} MB)")

    print(f"\nLoading Pocket TTS model (language={language}, quantize={quantize})...")
    t0 = time.perf_counter()

    def _loader():
        from pocket_tts import TTSModel
        return TTSModel.load_model(language=language, quantize=quantize)

    model = load_local_first(_loader)
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s (sample_rate={model.sample_rate} Hz).\n")

    clips_meta: list[dict] = []
    for tensor_path in tensors:
        wav_path = out_dir / f"{tensor_path.stem}.wav"
        print(f"Synthesizing [{tensor_path.stem}] -> {wav_path.name}...", end=" ", flush=True)
        info = synthesize_tensor_to_wav(model, tensor_path, text, wav_path, seed=seed)
        clips_meta.append(info)
        print(
            f"done ({info['duration_s']:.2f}s audio in {info['synthesis_s']:.2f}s, "
            f"RTF={info['rtf']}, peak={info['peak_abs']:.3f})"
        )

    metadata = {
        "text": text,
        "language": language,
        "quantize": quantize,
        "seed": seed,
        "output_dir": str(out_dir),
        "clips": clips_meta,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved metadata to {meta_path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate WAV clips for all female emotion voice clone tensors."
    )
    parser.add_argument(
        "--cloned-voices-dir",
        type=str,
        default=None,
        help="Directory containing *_female.safetensors files (default: cloned_voices/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write generated WAV clips and metadata.json.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=DEFAULT_TEXT,
        help="Prompt text to synthesize across every female emotion tensor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Torch RNG seed used before each synthesis for reproducibility.",
    )
    args = parser.parse_args()

    generate_clips(
        cloned_voices_dir=args.cloned_voices_dir,
        output_dir=args.output_dir,
        text=args.text,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
