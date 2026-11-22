"""Generate PocketTTS voice cloning safetensor files for emotion audio clips.

Scans `voices/output/*.wav`, applies per-emotion acoustic conditioning tuned from
voice-modulation analysis (loudness leveling and RBJ shelving EQ to counter Pocket
TTS's spectral darkening, sad low-end boominess, and whisper attenuation), and
exports corresponding `.safetensors` embeddings directly into `cloned_voices/`.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile

from constants import (
    CLONED_VOICES_DIR,
    POCKET_TTS_CACHE_DIR,
    POCKET_TTS_DEFAULT_LANGUAGE,
)

# Configure HF cache
os.environ.setdefault("HF_HOME", str(Path(POCKET_TTS_CACHE_DIR).resolve()))

from pocket_tts import TTSModel, export_model_state
from services.tts.src.hf_cache import load_local_first

# Per-emotion conditioning profiles calibrated against voices/output/ reference clips:
# - neutral: mild +2.5 dB presence lift to counter Pocket TTS's ~-370 Hz centroid darkening
# - happy:   +3.5 dB high-shelf brightness & +1 dB hotter active RMS for energetic delivery
# - sad:     -4.5 dB low-shelf (<300 Hz) to tame excessive sub-500 Hz boominess (66% -> balanced warm)
#            while keeping sad distinctly darker than neutral
# - whisper: +6.7 dB active RMS normalization (from ~-36.2 dBFS -> -29.5 dBFS) to prevent Pocket TTS
#            from doubling the whisper attenuation (-13.7 dB -> ~-6 to -8 dB) while preserving breathiness
EMOTION_CLONE_PROFILES: dict[str, dict[str, float]] = {
    "neutral": {
        "target_active_dbfs": -29.0,
        "low_shelf_hz": 250.0,
        "low_shelf_db": 0.0,
        "high_shelf_hz": 2200.0,
        "high_shelf_db": 2.5,
    },
    "happy": {
        "target_active_dbfs": -28.0,
        "low_shelf_hz": 250.0,
        "low_shelf_db": -1.5,
        "high_shelf_hz": 2000.0,
        "high_shelf_db": 3.5,
    },
    "sad": {
        "target_active_dbfs": -29.5,
        "low_shelf_hz": 300.0,
        "low_shelf_db": -4.5,
        "high_shelf_hz": 2500.0,
        "high_shelf_db": 2.0,
    },
    "whisper": {
        "target_active_dbfs": -29.5,
        "low_shelf_hz": 300.0,
        "low_shelf_db": -2.0,
        "high_shelf_hz": 2500.0,
        "high_shelf_db": 1.5,
    },
}

DEFAULT_PROFILE: dict[str, float] = EMOTION_CLONE_PROFILES["neutral"]


def _biquad_shelf_sos(
    sr: int,
    freq_hz: float,
    gain_db: float,
    kind: str,
    slope: float = 0.707,
) -> np.ndarray:
    """Design a single RBJ biquad low-shelf or high-shelf filter in SOS format."""
    if abs(gain_db) < 1e-6:
        return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    a_amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq_hz / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / 2.0 * math.sqrt((a_amp + 1.0 / a_amp) * (1.0 / slope - 1.0) + 2.0)
    two_sqrt_a_alpha = 2.0 * math.sqrt(a_amp) * alpha

    if kind == "low":
        b0 = a_amp * ((a_amp + 1.0) - (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha)
        b1 = 2.0 * a_amp * ((a_amp - 1.0) - (a_amp + 1.0) * cos_w0)
        b2 = a_amp * ((a_amp + 1.0) - (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha)
        a0 = (a_amp + 1.0) + (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha
        a1 = -2.0 * ((a_amp - 1.0) + (a_amp + 1.0) * cos_w0)
        a2 = (a_amp + 1.0) + (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha
    elif kind == "high":
        b0 = a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha)
        b1 = -2.0 * a_amp * ((a_amp - 1.0) + (a_amp + 1.0) * cos_w0)
        b2 = a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha)
        a0 = (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha
        a1 = 2.0 * ((a_amp - 1.0) - (a_amp + 1.0) * cos_w0)
        a2 = (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha
    else:
        raise ValueError(f"Unsupported shelf kind: {kind}")

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _measure_active_rms_dbfs(audio: np.ndarray, sr: int) -> float:
    """Compute mean RMS (dBFS) over active frames within 35 dB of peak frame RMS."""
    frame_len = max(1, int(round(sr * 0.032)))
    hop_len = max(1, int(round(sr * 0.010)))
    if len(audio) < frame_len:
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        return 20.0 * math.log10(rms + 1e-12)

    num_frames = 1 + (len(audio) - frame_len) // hop_len
    strides = (audio.strides[0] * hop_len, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(num_frames, frame_len), strides=strides
    )
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    frame_db = 20.0 * np.log10(frame_rms + 1e-12)
    peak_db = float(np.max(frame_db))
    active = frame_db[frame_db >= peak_db - 35.0]
    return float(np.mean(active)) if active.size > 0 else float(np.mean(frame_db))


def _emotion_from_stem(stem: str) -> str:
    """Extract emotion name from filename stem like 'happy_female' or 'whisper_female'."""
    return stem.split("_", 1)[0].strip().lower()


def prepare_conditioning_tensor(
    wav_path: Path,
    target_sr: int,
    emotion: str,
    preprocess: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Load WAV, apply per-emotion EQ + active-RMS leveling, and return [1, samples] float32 tensor."""
    sr, data = wavfile.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        audio = data.astype(np.float64) / float(np.iinfo(data.dtype).max)
    else:
        audio = data.astype(np.float64)

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Resample to model sample rate (24 kHz) if needed
    if sr != target_sr:
        gcd = math.gcd(int(sr), int(target_sr))
        audio = signal.resample_poly(audio, target_sr // gcd, sr // gcd)
        sr = target_sr

    rms_before = _measure_active_rms_dbfs(audio, sr)
    profile = EMOTION_CLONE_PROFILES.get(emotion, DEFAULT_PROFILE)
    applied_gain_db = 0.0

    if preprocess:
        # 1. Gentle 60 Hz rumble high-pass filter
        hp_sos = signal.butter(2, 60.0 / (0.5 * sr), btype="highpass", output="sos")
        audio = signal.sosfilt(hp_sos, audio)

        # 2. Per-emotion low-shelf and high-shelf tonal conditioning
        if abs(profile["low_shelf_db"]) > 1e-6:
            ls_sos = _biquad_shelf_sos(sr, profile["low_shelf_hz"], profile["low_shelf_db"], "low")
            audio = signal.sosfilt(ls_sos, audio)
        if abs(profile["high_shelf_db"]) > 1e-6:
            hs_sos = _biquad_shelf_sos(sr, profile["high_shelf_hz"], profile["high_shelf_db"], "high")
            audio = signal.sosfilt(hs_sos, audio)

        # 3. Active-RMS loudness leveling
        rms_after_eq = _measure_active_rms_dbfs(audio, sr)
        applied_gain_db = profile["target_active_dbfs"] - rms_after_eq
        # Clamp extreme gain to avoid amplifying noise floor excessively
        applied_gain_db = max(-12.0, min(12.0, applied_gain_db))
        audio = audio * (10.0 ** (applied_gain_db / 20.0))

        # 4. Soft peak limiter to prevent clipping (<= 0.95)
        peak = float(np.max(np.abs(audio)))
        if peak > 0.95:
            audio = np.tanh(audio / peak * 1.25) / math.tanh(1.25) * 0.95

    rms_after = _measure_active_rms_dbfs(audio, sr)
    tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # [1, samples]
    stats = {
        "rms_before_dbfs": round(rms_before, 2),
        "rms_after_dbfs": round(rms_after, 2),
        "gain_db": round(applied_gain_db, 2),
        "low_shelf_db": profile["low_shelf_db"] if preprocess else 0.0,
        "high_shelf_db": profile["high_shelf_db"] if preprocess else 0.0,
    }
    return tensor, stats


def generate_all_emotion_voices(
    input_dir: Path | str = "voices/output",
    output_dir: Path | str = CLONED_VOICES_DIR,
    language: str = POCKET_TTS_DEFAULT_LANGUAGE,
    overwrite: bool = False,
    preprocess: bool = True,
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
    print(f"Pocket TTS model loaded in {time.time() - t0:.2f}s (sample_rate={model.sample_rate} Hz).")

    if not getattr(model, "has_voice_cloning", True):
        raise RuntimeError("Pocket TTS voice cloning is not enabled.")

    generated_paths: list[Path] = []
    for wf in wav_files:
        stem = wf.stem  # e.g., 'whisper_female', 'happy_male'
        emotion = _emotion_from_stem(stem)
        out_safetensors = out_dir / f"{stem}.safetensors"

        if out_safetensors.is_file() and not overwrite and out_safetensors.stat().st_size > 1000:
            print(f"✓ Already exists: {out_safetensors.name} ({out_safetensors.stat().st_size / 1024 / 1024:.2f} MB), skipping.")
            generated_paths.append(out_safetensors)
            continue

        print(f"Extracting [{emotion}] voice conditioning for '{wf.name}'...", end=" ", flush=True)
        t_start = time.perf_counter()
        audio_tensor, stats = prepare_conditioning_tensor(
            wf,
            target_sr=int(model.sample_rate),
            emotion=emotion,
            preprocess=preprocess,
        )
        state = model.get_state_for_audio_prompt(audio_tensor)
        export_model_state(state, out_safetensors)
        t_elapsed = time.perf_counter() - t_start
        size_mb = out_safetensors.stat().st_size / (1024 * 1024)
        print(
            f"done in {t_elapsed:.2f}s -> {out_safetensors.name} ({size_mb:.2f} MB) "
            f"[EQ: LS {stats['low_shelf_db']:+.1f} dB / HS {stats['high_shelf_db']:+.1f} dB, "
            f"RMS: {stats['rms_before_dbfs']:.1f} -> {stats['rms_after_dbfs']:.1f} dBFS ({stats['gain_db']:+.1f} dB)]"
        )
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
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable per-emotion EQ and loudness conditioning before cloning",
    )
    args = parser.parse_args()

    generate_all_emotion_voices(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.force,
        preprocess=not args.no_preprocess,
    )


if __name__ == "__main__":
    main()
