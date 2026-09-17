"""Device-style augmentation engine (the coverage workhorse) + SpecAugment.

Each random draw = a synthetic 'pseudo-device' transfer function. For the
supervised-contrastive objective, two independent passes give two views of the
SAME clip through two DIFFERENT synthetic devices (same content, different
style) — this is what makes the invariance device-label-free.

Band awareness
--------------
The simulator now runs AFTER SSP, so all frequency-domain parameters are
expressed as fractions of a nominal 8 kHz design band and rescaled to the
actual harmonised band. Without this, a configured 1.5-7.5 kHz low-pass sweep
is a no-op once every clip is band-limited to 2 kHz, and the augmentation
silently stops covering device roll-off — the single most important axis of
device variation.
"""
from __future__ import annotations
from typing import Dict
import torch
import torchaudio.functional as AF

_DESIGN_BAND_HZ = 8000.0        # band the config ranges were written against


def _rand(a, b):
    if b < a:
        a, b = b, a
    if b - a < 1e-9:
        return float(a)
    return float(torch.empty(1).uniform_(a, b).item())


class DeviceSimulator:
    """Randomised device transfer function on a mono waveform [T] or [1,T].

    hard_scale > 1 widens every range (the 'extreme' extrapolation tier).
    band_hz clamps/rescales the frequency-domain ranges to the usable band.
    """

    def __init__(self, cfg: Dict, sample_rate: int, hard_scale: float = 1.0,
                 band_hz: float | None = None):
        self.c = cfg
        self.sr = int(sample_rate)
        self.s = float(hard_scale)
        nyq = self.sr / 2.0
        self.band = float(min(band_hz if band_hz else nyq, nyq))
        self.fscale = self.band / _DESIGN_BAND_HZ
        self.fmax_safe = self.band * 0.98            # keep biquads off Nyquist

    def _f(self, hz):
        """Rescale a design-band frequency into the actual band."""
        return max(1.0, min(float(hz) * self.fscale, self.fmax_safe))

    # ---- ACPL acquisition-state layout --------------------------------- #
    # theta is a fixed-length vector in [0, 1]: for each transform block an
    # applied-flag followed by its parameters normalised by their own sampling
    # range. Fixed n_bands EQ keeps the length constant. When a block is not
    # applied its parameters are set to the identity position (gain 0 dB ->
    # 0.5, low-pass -> 1.0 i.e. fully open, noise SNR -> 1.0 i.e. clean).
    # Raw ||theta_hat - theta||^2 over mixed dB/Hz/bits units would be
    # dominated by the Hz terms; this normalisation is what makes the ACPL
    # acquisition loss trainable.
    def theta_dim(self) -> int:
        # flags(7) + gain(1) + eq(3*n_bands) + lp,hp,tilt,bits,snr(5)
        return 7 + 1 + 3 * int(self.c["eq"]["n_bands"]) + 5

    @staticmethod
    def _unit(v, a, b):
        if b - a < 1e-9:
            return 0.0
        return float(min(1.0, max(0.0, (v - a) / (b - a))))

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        out, _ = self.call_with_theta(wav)
        return out

    def call_with_theta(self, wav: torch.Tensor):
        sq = wav.dim() == 1
        if sq:
            wav = wav.unsqueeze(0)
        s = self.s
        c = self.c
        nb = int(c["eq"]["n_bands"])
        flags = [0.0] * 7
        # identity positions
        th_gain = 0.5
        th_eq = []
        for _ in range(nb):
            th_eq += [0.0, 0.5, 0.0]                  # f0, gain(0 dB), Q
        th_lp, th_hp, th_tilt, th_bits, th_snr = 1.0, 0.0, 0.5, 1.0, 1.0

        if torch.rand(1).item() < c["gain"]["p"]:
            g = c["gain"]
            lo_, hi_ = g["min_db"] * s, g["max_db"] * s
            v = _rand(lo_, hi_)
            wav = AF.gain(wav, v)
            flags[0] = 1.0
            th_gain = self._unit(v, lo_, hi_)

        if torch.rand(1).item() < c["eq"]["p"]:
            e = c["eq"]
            lo, hi = self._f(e["f_min"]), self._f(e["f_max"])
            flags[1] = 1.0
            th_eq = []
            for _ in range(nb):
                f0 = _rand(lo, hi)
                gv = _rand(-e["gain_db"] * s, e["gain_db"] * s)
                qv = _rand(e["q_min"], e["q_max"])
                wav = AF.equalizer_biquad(wav, self.sr, f0, gain=gv, Q=qv)
                th_eq += [self._unit(f0, lo, hi),
                          self._unit(gv, -e["gain_db"] * s, e["gain_db"] * s),
                          self._unit(qv, e["q_min"], e["q_max"])]

        if torch.rand(1).item() < c["lowpass"]["p"]:
            lp = c["lowpass"]
            lo_, hi_ = self._f(lp["cutoff_min"]), self._f(lp["cutoff_max"])
            cut = _rand(lo_, hi_)
            wav = AF.lowpass_biquad(wav, self.sr, cut)
            flags[2] = 1.0
            th_lp = self._unit(cut, lo_, hi_)

        if torch.rand(1).item() < c["highpass"]["p"]:
            hp = c["highpass"]
            # low-frequency roll-off is physical, not band-relative: keep as-is
            cut = min(_rand(hp["cutoff_min"], hp["cutoff_max"]), self.band * 0.5)
            cut = max(1.0, cut)
            wav = AF.highpass_biquad(wav, self.sr, cut)
            flags[3] = 1.0
            th_hp = self._unit(cut, hp["cutoff_min"], hp["cutoff_max"])

        if torch.rand(1).item() < c["tilt"]["p"]:
            t = c["tilt"]
            a = max(-0.999, min(0.999, _rand(t["alpha_min"] * s, t["alpha_max"] * s)))
            wav = wav - a * torch.nn.functional.pad(wav, (1, 0))[..., :-1]
            flags[4] = 1.0
            th_tilt = self._unit(a, t["alpha_min"] * s, t["alpha_max"] * s)

        if torch.rand(1).item() < c["codec"]["p"]:
            cc = c["codec"]
            bits = max(2, int(round(_rand(cc["bits_min"], cc["bits_max"]) / max(s, 1.0))))
            lv = 2 ** (bits - 1)
            wav = torch.round(wav.clamp(-1, 1) * lv) / lv
            flags[5] = 1.0
            th_bits = self._unit(bits, 2, cc["bits_max"])

        if torch.rand(1).item() < c["noise"]["p"]:
            nz = c["noise"]
            lo_, hi_ = nz["snr_min"] / max(s, 1.0), nz["snr_max"] / max(s, 1.0)
            snr = _rand(lo_, hi_)
            sp = wav.pow(2).mean().clamp_min(1e-12)
            n = torch.randn_like(wav)
            n = n * torch.sqrt(sp / (10 ** (snr / 10)) / n.pow(2).mean().clamp_min(1e-12))
            wav = wav + n
            flags[6] = 1.0
            th_snr = self._unit(snr, lo_, hi_)

        wav = wav / (wav.abs().max() + 1e-8)
        if not torch.isfinite(wav).all():
            wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        theta = torch.tensor(flags + [th_gain] + th_eq +
                             [th_lp, th_hp, th_tilt, th_bits, th_snr],
                             dtype=torch.float32)
        return (wav.squeeze(0) if sq else wav), theta


class SpecAugment:
    def __init__(self, freq_mask, time_mask, n_masks=2, fill="mean"):
        self.f, self.t, self.n = freq_mask, time_mask, n_masks
        self.fill = fill

    def sample_masks(self, nm: int, nt: int):
        """Draw one mask set (positions + widths) without applying it."""
        ms = []
        for _ in range(self.n):
            f0 = fw = t0 = tw = 0
            if self.f > 0 and nm > 1:
                fw = int(torch.randint(0, min(self.f, nm) + 1, (1,)).item())
                if fw:
                    f0 = int(torch.randint(0, nm - fw + 1, (1,)).item())
            if self.t > 0 and nt > 1:
                tw = int(torch.randint(0, min(self.t, nt) + 1, (1,)).item())
                if tw:
                    t0 = int(torch.randint(0, nt - tw + 1, (1,)).item())
            ms.append((f0, fw, t0, tw))
        return ms

    def apply_masks(self, mel: torch.Tensor, ms) -> torch.Tensor:
        mel = mel.clone()
        v = mel.mean() if self.fill == "mean" else 0.0
        for f0, fw, t0, tw in ms:
            if fw:
                mel[..., f0:f0 + fw, :] = v
            if tw:
                mel[..., :, t0:t0 + tw] = v
        return mel

    def paired(self, ma: torch.Tensor, mb: torch.Tensor):
        """SHARED mask positions/widths for two paired views; each view is
        filled with its own mean. v2 corrected pipeline."""
        assert ma.shape == mb.shape, "paired views must share a shape"
        ms = self.sample_masks(ma.shape[-2], ma.shape[-1])
        return self.apply_masks(ma, ms), self.apply_masks(mb, ms)

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.clone()
        nm, nt = mel.shape[-2], mel.shape[-1]
        # fill with the spectrogram mean rather than 0 — with per-clip
        # standardisation 0 is the mean anyway, but this stays correct if the
        # normalisation changes.
        v = mel.mean() if self.fill == "mean" else 0.0
        for _ in range(self.n):
            if self.f > 0 and nm > 1:
                w = int(torch.randint(0, min(self.f, nm) + 1, (1,)).item())
                if w:
                    f0 = int(torch.randint(0, nm - w + 1, (1,)).item())
                    mel[..., f0:f0 + w, :] = v
            if self.t > 0 and nt > 1:
                w = int(torch.randint(0, min(self.t, nt) + 1, (1,)).item())
                if w:
                    t0 = int(torch.randint(0, nt - w + 1, (1,)).item())
                    mel[..., :, t0:t0 + w] = v
        return mel

    def sample_plan(self, nm, nt):
        """Draw mask coordinates once; entries are (axis, start, width)."""
        plan = []
        for _ in range(self.n):
            if self.f > 0 and nm > 1:
                w = int(torch.randint(0, min(self.f, nm) + 1, (1,)).item())
                if w:
                    start = int(torch.randint(0, nm - w + 1, (1,)).item())
                    plan.append(("frequency", start, w))
            if self.t > 0 and nt > 1:
                w = int(torch.randint(0, min(self.t, nt) + 1, (1,)).item())
                if w:
                    start = int(torch.randint(0, nt - w + 1, (1,)).item())
                    plan.append(("time", start, w))
        return tuple(plan)

    def apply_plan(self, mel, plan):
        """Apply coordinates with this view's own pre-mask fill value."""
        out = mel.clone()
        value = out.mean() if self.fill == "mean" else 0.0
        for axis, start, width in plan:
            if axis == "frequency":
                out[..., start:start + width, :] = value
            elif axis == "time":
                out[..., :, start:start + width] = value
            else:
                raise ValueError("Unknown SpecAugment mask axis")
        return out

    def apply_pair(self, first, second):
        """Shared positions; preserve distinct acquisition content and means."""
        if first.shape[-2:] != second.shape[-2:]:
            raise ValueError("Paired SpecAugment requires equal frequency/time dimensions")
        plan = self.sample_plan(*first.shape[-2:])
        return self.apply_plan(first, plan), self.apply_plan(second, plan)
