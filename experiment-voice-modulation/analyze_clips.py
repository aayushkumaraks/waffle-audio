"""Analyze pitch (F0) and tone (spectral, timbral, prosodic) differences across WAV clips.

Uses NumPy and SciPy only (no extra dependencies) to compute:
  - Pitch (F0) via YIN-inspired cumulative mean normalized difference (CMNDF)
    with parabolic interpolation and median filtering over voiced frames.
  - Tone / Timbre: spectral centroid, bandwidth, 85% rolloff, spectral flatness,
    band energy distribution (low / mid / high), spectral tilt (dB/octave),
    zero-crossing rate (ZCR), and harmonics-to-noise ratio (HNR).
  - Dynamics & Pacing: duration, active speech ratio, RMS loudness (dBFS),
    and dynamic range (P95 - P10 active frame dB).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import wavfile

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "output"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "voices" / "output"

# Analysis parameters tuned for female speech
FRAME_MS = 32.0
HOP_MS = 10.0
F0_MIN_HZ = 75.0
F0_MAX_HZ = 500.0
YIN_THRESHOLD = 0.20


def load_wav_mono_float(wav_path: Path) -> tuple[int, np.ndarray]:
    """Load WAV file as mono float64 normalized to [-1.0, 1.0]."""
    sr, data = wavfile.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        audio = data.astype(np.float64) / max_val
    else:
        audio = data.astype(np.float64)
    # Remove DC offset
    audio = audio - np.mean(audio)
    return int(sr), audio


def frame_signal(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Slice 1D signal into overlapping frames of shape (num_frames, frame_len)."""
    if len(audio) < frame_len:
        pad = np.zeros(frame_len - len(audio), dtype=np.float64)
        audio = np.concatenate([audio, pad])
    num_frames = 1 + (len(audio) - frame_len) // hop_len
    strides = (audio.strides[0] * hop_len, audio.strides[0])
    return np.lib.stride_tricks.as_strided(
        audio, shape=(num_frames, frame_len), strides=strides
    ).copy()


def yin_f0_and_hnr(
    frames: np.ndarray,
    sr: int,
    active_mask: np.ndarray,
    f0_min: float = F0_MIN_HZ,
    f0_max: float = F0_MAX_HZ,
    threshold: float = YIN_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate per-frame F0 (Hz), voicing mask, and HNR (dB) using YIN CMNDF + autocorrelation."""
    num_frames, frame_len = frames.shape
    tau_min = max(2, int(sr / f0_max))
    tau_max = min(frame_len // 2, int(sr / f0_min))

    f0 = np.full(num_frames, np.nan, dtype=np.float64)
    voiced = np.zeros(num_frames, dtype=bool)
    hnr_db = np.full(num_frames, np.nan, dtype=np.float64)

    # Next power of 2 for fast FFT-based autocorrelation
    n_fft = 1 << (2 * frame_len - 1).bit_length()

    for i in range(num_frames):
        if not active_mask[i]:
            continue
        x = frames[i] - np.mean(frames[i])
        energy = np.dot(x, x)
        if energy <= 1e-12:
            continue

        # Autocorrelation via FFT: r[tau] = sum_{j=0}^{N-1-tau} x[j] * x[j + tau]
        spec = np.fft.rfft(x, n=n_fft)
        r = np.fft.irfft(spec * np.conj(spec), n=n_fft)[: tau_max + 1]

        # Difference function d(tau) = sum_{j=0}^{W-1} (x[j] - x[j+tau])^2
        # using exact cumulative energy for window length W = frame_len - tau_max
        w = frame_len - tau_max
        x_sq = x * x
        csum = np.concatenate(([0.0], np.cumsum(x_sq)))
        e0 = csum[w]  # sum_{j=0}^{w-1} x[j]^2

        # Cross term via FFT on x[:w] and x[:frame_len]
        x1 = np.zeros(n_fft, dtype=np.float64)
        x1[:w] = x[:w]
        x2 = np.zeros(n_fft, dtype=np.float64)
        x2[:frame_len] = x
        cross = np.fft.irfft(np.conj(np.fft.rfft(x1)) * np.fft.rfft(x2), n=n_fft)[: tau_max + 1]

        taus = np.arange(tau_max + 1)
        e_tau = csum[taus + w] - csum[taus]
        d = np.maximum(0.0, e0 + e_tau - 2.0 * cross)

        # Cumulative mean normalized difference function (CMNDF)
        cmndf = np.ones_like(d)
        running_sum = np.cumsum(d[1:])
        nz = running_sum > 1e-12
        cmndf[1:][nz] = d[1:][nz] * np.arange(1, tau_max + 1)[nz] / running_sum[nz]

        # Search for first local minimum below threshold in [tau_min, tau_max]
        segment = cmndf[tau_min : tau_max + 1]
        below = np.where(segment < threshold)[0]
        if below.size > 0:
            idx = below[0]
            # Walk down to the local minimum
            while idx + 1 < len(segment) and segment[idx + 1] < segment[idx]:
                idx += 1
            tau = tau_min + idx
            # Parabolic interpolation around tau
            if 1 <= tau < tau_max:
                s0, s1, s2 = cmndf[tau - 1], cmndf[tau], cmndf[tau + 1]
                denom = s0 - 2.0 * s1 + s2
                if abs(denom) > 1e-12:
                    delta = 0.5 * (s0 - s2) / denom
                    tau_refined = tau + np.clip(delta, -0.5, 0.5)
                else:
                    tau_refined = float(tau)
            else:
                tau_refined = float(tau)

            pitch = sr / tau_refined
            if f0_min <= pitch <= f0_max:
                f0[i] = pitch
                voiced[i] = True

                # Normalized autocorrelation peak for HNR estimate
                r_norm = float(np.clip(r[tau] / (r[0] + 1e-12), 1e-6, 1.0 - 1e-6))
                hnr_db[i] = 10.0 * math.log10(r_norm / (1.0 - r_norm))
        else:
            # Unvoiced / breathy frame: still record HNR from best autocorrelation peak
            r_seg = r[tau_min : tau_max + 1] / (r[0] + 1e-12)
            r_peak = float(np.clip(np.max(r_seg), 1e-6, 1.0 - 1e-6))
            hnr_db[i] = 10.0 * math.log10(r_peak / (1.0 - r_peak))

    # Median-filter voiced F0 to suppress isolated octave jumps
    if np.sum(voiced) >= 5:
        voiced_idx = np.where(voiced)[0]
        smoothed = signal.medfilt(f0[voiced_idx], kernel_size=5)
        f0[voiced_idx] = smoothed

    return f0, voiced, hnr_db


def analyze_wav(wav_path: Path) -> dict[str, Any]:
    """Compute comprehensive pitch, tone, dynamics, and pacing metrics for one WAV file."""
    sr, audio = load_wav_mono_float(wav_path)
    duration_s = len(audio) / sr

    frame_len = int(round(sr * FRAME_MS / 1000.0))
    hop_len = int(round(sr * HOP_MS / 1000.0))
    frames = frame_signal(audio, frame_len, hop_len)

    # Frame RMS and active speech detection (-35 dB relative to peak frame RMS)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    frame_db = 20.0 * np.log10(frame_rms + 1e-12)
    peak_db = float(np.max(frame_db))
    active_mask = frame_db >= (peak_db - 35.0)
    active_ratio = float(np.mean(active_mask))

    # Pitch & HNR
    f0, voiced_mask, hnr_db = yin_f0_and_hnr(frames, sr, active_mask)
    voiced_f0 = f0[voiced_mask]
    voiced_ratio_of_active = (
        float(np.sum(voiced_mask) / max(1, int(np.sum(active_mask))))
    )

    if voiced_f0.size > 0:
        f0_mean = float(np.mean(voiced_f0))
        f0_median = float(np.median(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_p05, f0_p25, f0_p75, f0_p95 = [
            float(x) for x in np.percentile(voiced_f0, [5, 25, 75, 95])
        ]
        f0_iqr = f0_p75 - f0_p25
        f0_range_semitones = float(12.0 * math.log2(max(f0_p95, 1.0) / max(f0_p05, 1.0)))
    else:
        f0_mean = f0_median = f0_std = f0_p05 = f0_p25 = f0_p75 = f0_p95 = f0_iqr = f0_range_semitones = float("nan")

    # Spectral / tonal features on active frames
    window = np.hanning(frame_len)
    active_frames = frames[active_mask] if np.any(active_mask) else frames
    windowed = active_frames * window[None, :]
    mag = np.abs(np.fft.rfft(windowed, axis=1))
    power = mag * mag + 1e-20
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)

    mag_sum = np.sum(mag, axis=1) + 1e-12
    centroid = np.sum(mag * freqs[None, :], axis=1) / mag_sum
    bandwidth = np.sqrt(
        np.sum(mag * ((freqs[None, :] - centroid[:, None]) ** 2), axis=1) / mag_sum
    )

    # 85% spectral rolloff
    cumsum_pow = np.cumsum(power, axis=1)
    total_pow = cumsum_pow[:, -1:]
    rolloff_idx = np.argmax(cumsum_pow >= 0.85 * total_pow, axis=1)
    rolloff_85 = freqs[rolloff_idx]

    # Spectral flatness (geometric mean / arithmetic mean of power spectrum)
    geo_mean = np.exp(np.mean(np.log(power), axis=1))
    arith_mean = np.mean(power, axis=1)
    flatness = geo_mean / (arith_mean + 1e-20)

    # Band energy shares: Low (<500 Hz), Mid (500-2000 Hz), Presence (2000-4000 Hz), High (>4000 Hz)
    mean_power_spec = np.mean(power, axis=0)
    spec_total = float(np.sum(mean_power_spec)) + 1e-20
    low_share = float(np.sum(mean_power_spec[freqs < 500.0]) / spec_total)
    mid_share = float(np.sum(mean_power_spec[(freqs >= 500.0) & (freqs < 2000.0)]) / spec_total)
    pres_share = float(np.sum(mean_power_spec[(freqs >= 2000.0) & (freqs < 4000.0)]) / spec_total)
    high_share = float(np.sum(mean_power_spec[freqs >= 4000.0]) / spec_total)

    # Spectral tilt (dB per octave) across 150 Hz - 6000 Hz
    tilt_mask = (freqs >= 150.0) & (freqs <= min(6000.0, sr * 0.45))
    log2_f = np.log2(freqs[tilt_mask])
    spec_db = 10.0 * np.log10(mean_power_spec[tilt_mask] + 1e-20)
    slope_db_per_oct = float(np.polyfit(log2_f, spec_db, 1)[0])

    # Zero-crossing rate on active frames
    zcr = np.mean(np.abs(np.diff(np.signbit(active_frames), axis=1)), axis=1)

    # Loudness & dynamics
    overall_rms_db = float(20.0 * math.log10(float(np.sqrt(np.mean(audio * audio) + 1e-12))))
    active_db = frame_db[active_mask] if np.any(active_mask) else frame_db
    active_rms_db = float(np.mean(active_db))
    dyn_range_db = float(np.percentile(active_db, 95) - np.percentile(active_db, 10))

    # HNR across active frames and voiced frames
    active_hnr = hnr_db[active_mask & ~np.isnan(hnr_db)]
    voiced_hnr = hnr_db[voiced_mask & ~np.isnan(hnr_db)]

    emotion = wav_path.stem.removesuffix("_female")
    return {
        "emotion": emotion,
        "file": wav_path.name,
        "sample_rate": sr,
        "duration_s": round(duration_s, 3),
        "active_speech_pct": round(active_ratio * 100.0, 1),
        "pitch": {
            "voiced_pct_of_active": round(voiced_ratio_of_active * 100.0, 1),
            "mean_hz": round(f0_mean, 1) if not math.isnan(f0_mean) else None,
            "median_hz": round(f0_median, 1) if not math.isnan(f0_median) else None,
            "std_hz": round(f0_std, 1) if not math.isnan(f0_std) else None,
            "p05_hz": round(f0_p05, 1) if not math.isnan(f0_p05) else None,
            "p25_hz": round(f0_p25, 1) if not math.isnan(f0_p25) else None,
            "p75_hz": round(f0_p75, 1) if not math.isnan(f0_p75) else None,
            "p95_hz": round(f0_p95, 1) if not math.isnan(f0_p95) else None,
            "iqr_hz": round(f0_iqr, 1) if not math.isnan(f0_iqr) else None,
            "range_p05_p95_semitones": round(f0_range_semitones, 2)
            if not math.isnan(f0_range_semitones)
            else None,
        },
        "tone": {
            "spectral_centroid_hz": round(float(np.mean(centroid)), 1),
            "spectral_bandwidth_hz": round(float(np.mean(bandwidth)), 1),
            "spectral_rolloff85_hz": round(float(np.mean(rolloff_85)), 1),
            "spectral_flatness": round(float(np.mean(flatness)), 4),
            "spectral_tilt_db_per_oct": round(slope_db_per_oct, 2),
            "zcr": round(float(np.mean(zcr)), 4),
            "hnr_active_db": round(float(np.mean(active_hnr)), 2) if active_hnr.size else None,
            "hnr_voiced_db": round(float(np.mean(voiced_hnr)), 2) if voiced_hnr.size else None,
            "band_energy_pct": {
                "low_lt_500hz": round(low_share * 100.0, 1),
                "mid_500_2000hz": round(mid_share * 100.0, 1),
                "presence_2000_4000hz": round(pres_share * 100.0, 1),
                "high_gt_4000hz": round(high_share * 100.0, 1),
            },
        },
        "dynamics": {
            "overall_rms_dbfs": round(overall_rms_db, 2),
            "active_mean_dbfs": round(active_rms_db, 2),
            "dynamic_range_db": round(dyn_range_db, 2),
        },
    }


def add_baseline_deltas(results: list[dict[str, Any]], baseline_emotion: str = "neutral") -> None:
    """Annotate each clip's metrics with deltas relative to the neutral baseline."""
    baseline = next((r for r in results if r["emotion"] == baseline_emotion), None)
    if baseline is None:
        return

    base_f0 = baseline["pitch"]["median_hz"]
    base_centroid = baseline["tone"]["spectral_centroid_hz"]
    base_rms = baseline["dynamics"]["active_mean_dbfs"]
    base_dur = baseline["duration_s"]
    base_hnr = baseline["tone"]["hnr_active_db"]

    for r in results:
        cur_f0 = r["pitch"]["median_hz"]
        if base_f0 and cur_f0:
            f0_diff_hz = round(cur_f0 - base_f0, 1)
            f0_diff_st = round(12.0 * math.log2(cur_f0 / base_f0), 2)
        else:
            f0_diff_hz = None
            f0_diff_st = None

        cur_hnr = r["tone"]["hnr_active_db"]
        hnr_diff = round(cur_hnr - base_hnr, 2) if (base_hnr is not None and cur_hnr is not None) else None

        r["delta_vs_neutral"] = {
            "median_f0_hz": f0_diff_hz,
            "median_f0_semitones": f0_diff_st,
            "spectral_centroid_hz": round(r["tone"]["spectral_centroid_hz"] - base_centroid, 1),
            "active_loudness_db": round(r["dynamics"]["active_mean_dbfs"] - base_rms, 2),
            "hnr_active_db": hnr_diff,
            "duration_s": round(r["duration_s"] - base_dur, 3),
            "duration_pct": round(((r["duration_s"] / base_dur) - 1.0) * 100.0, 1) if base_dur else None,
        }


def build_reference_comparison(
    generated: list[dict[str, Any]],
    reference: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute per-emotion fidelity deltas: generated clip minus reference clip."""
    ref_by_emo = {r["emotion"]: r for r in reference}
    comparisons: list[dict[str, Any]] = []

    for gen in generated:
        emo = gen["emotion"]
        ref = ref_by_emo.get(emo)
        if ref is None:
            continue

        g_p, r_p = gen["pitch"], ref["pitch"]
        g_t, r_t = gen["tone"], ref["tone"]
        g_d, r_d = gen["dynamics"], ref["dynamics"]

        if g_p["median_hz"] and r_p["median_hz"]:
            f0_hz_diff = round(g_p["median_hz"] - r_p["median_hz"], 1)
            f0_st_diff = round(12.0 * math.log2(g_p["median_hz"] / r_p["median_hz"]), 2)
        else:
            f0_hz_diff = None
            f0_st_diff = None

        if g_p["range_p05_p95_semitones"] is not None and r_p["range_p05_p95_semitones"] is not None:
            range_st_diff = round(g_p["range_p05_p95_semitones"] - r_p["range_p05_p95_semitones"], 2)
        else:
            range_st_diff = None

        if g_t["hnr_active_db"] is not None and r_t["hnr_active_db"] is not None:
            hnr_diff = round(g_t["hnr_active_db"] - r_t["hnr_active_db"], 2)
        else:
            hnr_diff = None

        comparisons.append({
            "emotion": emo,
            "generated_file": gen["file"],
            "reference_file": ref["file"],
            "generated": {
                "duration_s": gen["duration_s"],
                "voiced_pct": g_p["voiced_pct_of_active"],
                "median_f0_hz": g_p["median_hz"],
                "f0_std_hz": g_p["std_hz"],
                "f0_range_st": g_p["range_p05_p95_semitones"],
                "centroid_hz": g_t["spectral_centroid_hz"],
                "rolloff85_hz": g_t["spectral_rolloff85_hz"],
                "flatness": g_t["spectral_flatness"],
                "tilt_db_per_oct": g_t["spectral_tilt_db_per_oct"],
                "hnr_active_db": g_t["hnr_active_db"],
                "low_lt_500hz_pct": g_t["band_energy_pct"]["low_lt_500hz"],
                "active_rms_dbfs": g_d["active_mean_dbfs"],
            },
            "reference": {
                "duration_s": ref["duration_s"],
                "voiced_pct": r_p["voiced_pct_of_active"],
                "median_f0_hz": r_p["median_hz"],
                "f0_std_hz": r_p["std_hz"],
                "f0_range_st": r_p["range_p05_p95_semitones"],
                "centroid_hz": r_t["spectral_centroid_hz"],
                "rolloff85_hz": r_t["spectral_rolloff85_hz"],
                "flatness": r_t["spectral_flatness"],
                "tilt_db_per_oct": r_t["spectral_tilt_db_per_oct"],
                "hnr_active_db": r_t["hnr_active_db"],
                "low_lt_500hz_pct": r_t["band_energy_pct"]["low_lt_500hz"],
                "active_rms_dbfs": r_d["active_mean_dbfs"],
            },
            "delta_gen_minus_ref": {
                "voiced_pct": round(g_p["voiced_pct_of_active"] - r_p["voiced_pct_of_active"], 1),
                "median_f0_hz": f0_hz_diff,
                "median_f0_semitones": f0_st_diff,
                "f0_range_st": range_st_diff,
                "centroid_hz": round(g_t["spectral_centroid_hz"] - r_t["spectral_centroid_hz"], 1),
                "rolloff85_hz": round(g_t["spectral_rolloff85_hz"] - r_t["spectral_rolloff85_hz"], 1),
                "flatness": round(g_t["spectral_flatness"] - r_t["spectral_flatness"], 4),
                "tilt_db_per_oct": round(g_t["spectral_tilt_db_per_oct"] - r_t["spectral_tilt_db_per_oct"], 2),
                "hnr_active_db": hnr_diff,
                "low_lt_500hz_pct": round(
                    g_t["band_energy_pct"]["low_lt_500hz"] - r_t["band_energy_pct"]["low_lt_500hz"], 1
                ),
                "active_rms_db": round(g_d["active_mean_dbfs"] - r_d["active_mean_dbfs"], 2),
            },
        })
    return comparisons


def _format_dataset_tables(results: list[dict[str, Any]], heading_prefix: str) -> list[str]:
    """Build Markdown tables for a single set of WAV analysis results."""
    lines: list[str] = [
        f"## {heading_prefix} — Pitch (F0) & Voicing",
        "",
        "| Emotion | Voiced % | Median F0 (Hz) | Δ vs Neutral | Mean F0 (Hz) | F0 Std (Hz) | P5–P95 Range (st) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        p = r["pitch"]
        d = r.get("delta_vs_neutral", {})
        delta_str = (
            f"{d['median_f0_hz']:+.1f} Hz ({d['median_f0_semitones']:+.2f} st)"
            if d.get("median_f0_hz") is not None
            else "n/a"
        )
        lines.append(
            f"| **{r['emotion']}** | {p['voiced_pct_of_active']:.1f}% | "
            f"{p['median_hz'] if p['median_hz'] is not None else 'n/a'} | {delta_str} | "
            f"{p['mean_hz'] if p['mean_hz'] is not None else 'n/a'} | "
            f"{p['std_hz'] if p['std_hz'] is not None else 'n/a'} | "
            f"{p['range_p05_p95_semitones'] if p['range_p05_p95_semitones'] is not None else 'n/a'} |"
        )

    lines.extend([
        "",
        f"## {heading_prefix} — Tone, Timbre & Spectral Balance",
        "",
        "| Emotion | Centroid (Hz) | Δ Centroid | Rolloff 85% (Hz) | Flatness | Tilt (dB/oct) | HNR Active (dB) | ZCR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in results:
        t = r["tone"]
        d = r.get("delta_vs_neutral", {})
        c_delta = f"{d['spectral_centroid_hz']:+.1f} Hz" if d.get("spectral_centroid_hz") is not None else "n/a"
        lines.append(
            f"| **{r['emotion']}** | {t['spectral_centroid_hz']:.1f} | {c_delta} | "
            f"{t['spectral_rolloff85_hz']:.1f} | {t['spectral_flatness']:.4f} | "
            f"{t['spectral_tilt_db_per_oct']:+.2f} | "
            f"{t['hnr_active_db'] if t['hnr_active_db'] is not None else 'n/a'} | {t['zcr']:.4f} |"
        )

    lines.extend([
        "",
        f"## {heading_prefix} — Band Energy & Dynamics",
        "",
        "| Emotion | <500 Hz | 0.5–2 kHz | 2–4 kHz | >4 kHz | Active RMS (dBFS) | Δ Loudness | Dyn Range (dB) | Duration |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in results:
        b = r["tone"]["band_energy_pct"]
        dyn = r["dynamics"]
        d = r.get("delta_vs_neutral", {})
        l_delta = f"{d['active_loudness_db']:+.2f} dB" if d.get("active_loudness_db") is not None else "n/a"
        lines.append(
            f"| **{r['emotion']}** | {b['low_lt_500hz']:.1f}% | {b['mid_500_2000hz']:.1f}% | "
            f"{b['presence_2000_4000hz']:.1f}% | {b['high_gt_4000hz']:.1f}% | "
            f"{dyn['active_mean_dbfs']:.2f} | {l_delta} | {dyn['dynamic_range_db']:.2f} | "
            f"{r['duration_s']:.2f}s |"
        )
    return lines


def format_markdown_report(
    generated: list[dict[str, Any]],
    reference: list[dict[str, Any]] | None = None,
    comparison: list[dict[str, Any]] | None = None,
) -> str:
    """Format a human-readable Markdown comparison report."""
    lines: list[str] = ["# Female Emotion Voice Modulation Analysis", ""]
    lines.extend(_format_dataset_tables(generated, "1. Generated Clips"))

    if reference:
        lines.append("")
        lines.extend(_format_dataset_tables(reference, "2. Reference Clips (voices/output/)"))

    if comparison:
        lines.extend([
            "",
            "## 3. Clone Fidelity: Generated vs Reference (Generated − Reference)",
            "",
            "| Emotion | Gen / Ref F0 (Hz) | Δ F0 (st) | Δ Range (st) | Δ Voiced % | Gen / Ref Centroid (Hz) | Δ Centroid | Δ <500 Hz | Δ HNR (dB) | Δ Loudness (dB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for c in comparison:
            g, r, d = c["generated"], c["reference"], c["delta_gen_minus_ref"]
            f0_pair = (
                f"{g['median_f0_hz']} / {r['median_f0_hz']}"
                if (g["median_f0_hz"] is not None or r["median_f0_hz"] is not None)
                else "n/a"
            )
            f0_st = f"{d['median_f0_semitones']:+.2f}" if d["median_f0_semitones"] is not None else "n/a"
            rng_st = f"{d['f0_range_st']:+.2f}" if d["f0_range_st"] is not None else "n/a"
            hnr_s = f"{d['hnr_active_db']:+.2f}" if d["hnr_active_db"] is not None else "n/a"
            lines.append(
                f"| **{c['emotion']}** | {f0_pair} | {f0_st} | {rng_st} | "
                f"{d['voiced_pct']:+.1f}% | {g['centroid_hz']:.0f} / {r['centroid_hz']:.0f} | "
                f"{d['centroid_hz']:+.1f} Hz | {d['low_lt_500hz_pct']:+.1f}% | "
                f"{hnr_s} | {d['active_rms_db']:+.2f} |"
            )

    return "\n".join(lines) + "\n"


def _analyze_wav_dir(target_dir: Path) -> list[dict[str, Any]]:
    wav_files = sorted(target_dir.glob("*_female.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No *_female.wav files found in {target_dir}")
    wav_files.sort(key=lambda p: (0 if p.stem.startswith("neutral") else 1, p.name))
    results = [analyze_wav(wf) for wf in wav_files]
    add_baseline_deltas(results, baseline_emotion="neutral")
    return results


def analyze_directory(
    input_dir: Path | str = DEFAULT_OUTPUT_DIR,
    reference_dir: Path | str | None = DEFAULT_REFERENCE_DIR,
) -> dict[str, Any]:
    """Analyze generated *_female.wav files and optionally compare against reference WAVs."""
    in_dir = Path(input_dir).resolve()
    generated = _analyze_wav_dir(in_dir)

    reference: list[dict[str, Any]] | None = None
    comparison: list[dict[str, Any]] | None = None
    if reference_dir is not None:
        ref_dir = Path(reference_dir).resolve()
        if ref_dir.is_dir() and list(ref_dir.glob("*_female.wav")):
            reference = _analyze_wav_dir(ref_dir)
            comparison = build_reference_comparison(generated, reference)

    payload = {
        "generated_dir": str(in_dir),
        "reference_dir": str(Path(reference_dir).resolve()) if reference is not None else None,
        "generated": generated,
        "reference": reference,
        "comparison": comparison,
    }

    json_path = in_dir / "analysis_report.json"
    md_path = in_dir / "analysis_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_report = format_markdown_report(generated, reference, comparison)
    md_path.write_text(md_report, encoding="utf-8")

    print(md_report)
    print(f"Saved JSON report to {json_path}")
    print(f"Saved Markdown report to {md_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze pitch and tone differences across generated female emotion WAV clips."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory containing generated *_female.wav files.",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default=str(DEFAULT_REFERENCE_DIR),
        help="Directory containing reference *_female.wav files (default: voices/output/).",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip comparison against reference WAV files.",
    )
    args = parser.parse_args()
    analyze_directory(
        input_dir=args.input_dir,
        reference_dir=None if args.no_reference else args.reference_dir,
    )


if __name__ == "__main__":
    main()
