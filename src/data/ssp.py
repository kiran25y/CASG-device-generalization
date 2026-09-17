"""Signal-level band harmonisation (SSP) — the device-agnostic front end.

Motivation (measured on our own manifests):

    device                        native sr    band after a naive 16 kHz resample
    HospitalStethoscope (Seoul)      4,000     0-2 kHz   (top 6 kHz of the mel is empty)
    HospitalStethoscope_sungbook     4,000     0-2 kHz
    Litt3200                         4,000     0-2 kHz
    AKGC417L / LittC2SE             44,100     0-8 kHz   (full)
    Meditron                  4k/10k/44.1k     mixed
    smartphone (iPhone)             48,000     0-8 kHz   (full)

Sample rate is therefore an almost perfect device label: a network can read the
capture device straight off the upper mel bands, and any "device invariance" we
claim on that input is confounded.

SSP removes the cue at the signal level: every clip is zero-phase low-pass
filtered to a COMMON cutoff, decimated to a COMMON working rate, then resampled
to the model rate. After this the usable band is identical for every corpus and
the only remaining differences are the ones we actually want to model (the
device transfer function inside the passband, noise floor, handling artefacts).

Chain:  native_sr --LPF(cutoff)--> native_sr --resample--> work_sr --resample--> model_sr

The intermediate decimation to `work_sr` is what makes the erasure explicit and
irreversible; the final upsample to `model_sr` adds no information but keeps the
25 ms / 10 ms framing contract that AST's positional embeddings expect.

Reference: Jeong et al., IEEE JBHI (this cohort) use cutoff=1000, work_sr=4000
and report it as the dominant driver of stethoscope->smartphone transfer
(+0.075 F1 vs. +0.019 for their feature-level module). We expose the cutoff so
it can be chosen on our own multi-device data rather than inherited.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from scipy.signal import butter, sosfiltfilt
    _HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    _HAVE_SCIPY = False

_SOS_CACHE: dict = {}


def _sos(sr: int, cutoff: float, order: int):
    """Cached second-order-sections for a Butterworth low-pass."""
    key = (int(sr), float(cutoff), int(order))
    if key not in _SOS_CACHE:
        wn = float(cutoff) / (0.5 * float(sr))
        if not (0.0 < wn < 1.0):                     # cutoff at/above Nyquist
            _SOS_CACHE[key] = None
        else:
            _SOS_CACHE[key] = butter(int(order), wn, btype="low", output="sos")
    return _SOS_CACHE[key]


def lowpass(x: np.ndarray, sr: int, cutoff: float, order: int = 8) -> np.ndarray:
    """Zero-phase Butterworth low-pass. No-op if cutoff >= Nyquist or the clip
    is too short for filtfilt's edge padding."""
    if not _HAVE_SCIPY:
        raise RuntimeError("SSP requires scipy — pip install 'scipy>=1.10'")
    sos = _sos(sr, cutoff, order)
    if sos is None:
        return x                                     # already band-limited below cutoff
    padlen = 3 * (2 * sos.shape[0] + 1)
    if x.shape[-1] <= padlen:
        return x                                     # too short to filter safely
    return np.ascontiguousarray(sosfiltfilt(sos, x).astype(np.float32))


class BandHarmonizer:
    """Apply SSP to a mono waveform.

    Parameters come from ``cfg.audio.ssp``:
        enabled    : bool  — off reproduces the legacy "naive combination" front end
        cutoff_hz  : float — common passband ceiling (e.g. 1000 or 2000)
        work_sr    : int   — common decimation rate (>= 2 * cutoff_hz)
        order      : int   — Butterworth order (8, matching the reference work)
    """

    def __init__(self, cfg):
        a = cfg.audio
        s = getattr(a, "ssp", None)
        self.enabled = bool(getattr(s, "enabled", False)) if s is not None else False
        self.cutoff = float(getattr(s, "cutoff_hz", 2000.0)) if s is not None else 0.0
        self.work_sr = int(getattr(s, "work_sr", 4000)) if s is not None else 0
        self.order = int(getattr(s, "order", 8)) if s is not None else 8
        self.model_sr = int(a.sample_rate)
        if self.enabled:
            if self.work_sr < 2 * self.cutoff:
                raise ValueError(
                    f"ssp.work_sr ({self.work_sr}) must be >= 2*cutoff_hz "
                    f"({2 * self.cutoff}) or the decimation aliases the passband")
            if self.model_sr < self.work_sr:
                raise ValueError(
                    f"audio.sample_rate ({self.model_sr}) must be >= ssp.work_sr "
                    f"({self.work_sr})")

    # -- numpy stage: runs at the file's native rate, before any resampling ----
    def harmonise(self, x: np.ndarray, native_sr: int) -> tuple:
        """Return (waveform at work_sr, work_sr). Pure numpy/scipy."""
        if not self.enabled:
            return x, native_sr
        x = lowpass(x, native_sr, self.cutoff, self.order)
        if native_sr != self.work_sr:
            x = _resample_np(x, native_sr, self.work_sr)
        return x, self.work_sr

    @property
    def effective_fmax(self) -> float:
        """Mel fmax implied by this configuration."""
        if not self.enabled:
            return float(self.model_sr) / 2.0
        return float(self.cutoff)


def _resample_np(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resample via torchaudio (kept consistent with the rest of the
    pipeline) with a scipy fallback."""
    if sr_in == sr_out:
        return x
    try:
        import torchaudio.functional as AF
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        return AF.resample(t, sr_in, sr_out).numpy()
    except Exception:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr_in), int(sr_out))
        return np.ascontiguousarray(
            resample_poly(x, int(sr_out) // g, int(sr_in) // g).astype(np.float32))


# ---------------------------------------------------------------------------
# Spectral descriptors — used by scripts/spectral_analysis.py to CHOOSE the
# cutoff from data instead of inheriting it, and to produce the paper figure.
# ---------------------------------------------------------------------------
_NAN3 = {"centroid": np.nan, "bandwidth": np.nan, "rolloff": np.nan,
         "hf_frac": np.nan}


def spectral_descriptors(x: np.ndarray, sr: int, roll_pct: float = 0.85,
                         power: bool = False, hf_split: float = 2000.0) -> dict:
    """Frame-wise spectral centroid / bandwidth / roll-off + high-band energy
    fraction, summarised by the median over frames.

    `power=False` (default) weights by MAGNITUDE |X|, which is the librosa and
    reference-paper convention. Weighting by |X|^2 instead crushes every
    statistic toward the low-frequency peak of a lung-sound spectrum and hides
    the between-device differences these descriptors exist to expose.

    `hf_frac` is the fraction of spectral energy above `hf_split` Hz. This is
    the device cue in its rawest form: it is identically 0 for any corpus
    sampled at 2*hf_split or below, and non-zero for the wideband ones, so a
    network can read capture hardware straight off it. Driving it to a common
    value across devices is precisely what SSP is for.
    """
    n_fft, hop = 1024, 256
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames < 1:
        return dict(_NAN3)
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    hi = freqs > hf_split
    cen, bw, roll, hff = [], [], [], []
    for i in range(n_frames):
        seg = x[i * hop: i * hop + n_fft] * win
        spec = np.abs(np.fft.rfft(seg))
        w = spec ** 2 if power else spec
        tot = w.sum()
        if tot <= 1e-12:
            continue
        c = float((freqs * w).sum() / tot)
        cen.append(c)
        bw.append(float(np.sqrt(((freqs - c) ** 2 * w).sum() / tot)))
        cs = np.cumsum(w)
        roll.append(float(freqs[int(np.searchsorted(cs, roll_pct * tot))]))
        e = spec ** 2
        hff.append(float(e[hi].sum() / max(e.sum(), 1e-12)))
    if not cen:
        return dict(_NAN3)
    return {"centroid": float(np.median(cen)),
            "bandwidth": float(np.median(bw)),
            "rolloff": float(np.median(roll)),
            "hf_frac": float(np.median(hff))}
