"""High-efficiency audio resampling using soxr or scipy.signal.resample_poly.

Avoids slower libraries like librosa and provides high-performance polyphase
down/up-sampling to preserve acoustic fidelity for STT/WebRTC.
"""

from __future__ import annotations

import math
import numpy as np
import numpy.typing as npt

_SOXR_AVAILABLE = False
try:
    import soxr
    _SOXR_AVAILABLE = True
except ImportError:
    pass

_SCIPY_AVAILABLE = False
try:
    from scipy.signal import resample_poly
    _SCIPY_AVAILABLE = True
except ImportError:
    pass


def resample_audio(
    audio: npt.NDArray[np.float32],
    src_rate: int,
    dst_rate: int,
) -> npt.NDArray[np.float32]:
    """Resample 1D float32 audio buffer from src_rate to dst_rate efficiently."""
    if src_rate == dst_rate or audio.size == 0:
        return audio

    if _SOXR_AVAILABLE:
        try:
            return soxr.resample(audio, src_rate, dst_rate).astype(np.float32, copy=False)
        except Exception:
            pass

    if _SCIPY_AVAILABLE:
        try:
            factor = math.gcd(src_rate, dst_rate)
            up = dst_rate // factor
            down = src_rate // factor
            return resample_poly(audio, up, down).astype(np.float32, copy=False)
        except Exception:
            pass

    # Fast linear interpolation fallback
    n_out = max(1, round(len(audio) * dst_rate / src_rate))
    return np.interp(
        np.linspace(0, len(audio) - 1, n_out),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)
