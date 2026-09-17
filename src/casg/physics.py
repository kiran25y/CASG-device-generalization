"""CASG Module B: physics-aware acquisition simulator (protocol section 6).

Design contract (differs deliberately from the old DeviceSimulator):
  * SAMPLING is separated from APPLICATION. ``sample()`` draws a parameter
    dict; ``apply(wav, params)`` is DETERMINISTIC given that dict (additive
    noise carries its own integer seed). This is what makes the style
    triplet possible: the same style params applied to two crops, and a
    provably-different style applied to one of them.
  * Transforms are split into a STYLE-CORE set (smooth in-band frequency
    response, roll-off, sensor-noise profile, mild compression,
    quantization) that defines acquisition style, and a NUISANCE set
    (mains hum, rare soft clipping) that improves robustness but does not
    define the style embedding.
  * A normalized style descriptor r(params) supports controlled negatives:
    d_sim(a, b) = ||r(a) - r(b)||_2 and ``sample_far`` resamples the style
    core until the descriptor distance clears a threshold.
  * All frequency parameters live inside the harmonized band (default
    50-2000 Hz), so the simulator models residual in-band acquisition
    variation instead of re-introducing the high-frequency shortcut that
    harmonization removed. Per-clip normalization cancels global gain, so
    gain is not a style dimension here (protocol section 5, interaction
    note).
  * Environmental-noise bank and codec transforms are NOT implemented in
    this first version because no audited asset bank / offline codec cache
    exists in the repo; they are skipped explicitly (protocol allows this
    if recorded). ``self.skipped`` documents it.
"""
from __future__ import annotations
import math
from typing import Dict, Optional
import torch

_TWO_PI = 2.0 * math.pi


def _u(gen, a, b):
    if b < a:
        a, b = b, a
    return float(a + (b - a) * torch.rand(1, generator=gen).item())


class PhysicsSimulator:
    STYLE_KEYS = ("resp", "lp", "hp", "noise", "comp", "quant")
    NUISANCE_KEYS = ("hum", "clip")

    def __init__(self, cfg: Optional[Dict], sample_rate: int,
                 band_hz: float = 2000.0, seed: Optional[int] = None):
        c = dict(cfg or {})
        self.sr = int(sample_rate)
        self.band = float(min(band_hz or 2000.0, self.sr / 2.0))
        self.f_lo = 50.0
        self.n_ctrl = int(c.get("resp_ctrl_points", 8))
        self.p = {"resp": 0.80, "lp": 0.40, "hp": 0.40, "noise": 0.45,
                  "hum": 0.15, "comp": 0.15, "clip": 0.05, "quant": 0.10}
        self.p.update({k: float(v) for k, v in (c.get("probs") or {}).items()})
        self.resp_db = float(c.get("resp_max_db", 8.0))
        self.snr_range = tuple(c.get("snr_db", (15.0, 40.0)))
        self.snr_hard = tuple(c.get("snr_hard_db", (8.0, 15.0)))
        self.p_snr_hard = float(c.get("p_snr_hard", 0.10))
        self.delta_style = float(c.get("delta_style", 1.0))
        self.far_tries = int(c.get("far_tries", 8))
        self.gen = torch.Generator()
        self.gen.manual_seed(int(seed) if seed is not None else
                             int(torch.randint(0, 2**31 - 1, (1,)).item()))
        # log-spaced control frequencies for the smooth response
        self.ctrl_f = torch.logspace(math.log10(self.f_lo),
                                     math.log10(self.band), self.n_ctrl)
        self.desc_f = torch.logspace(math.log10(self.f_lo),
                                     math.log10(self.band), 32)
        self.skipped = ["environmental_noise_bank (no audited asset bank)",
                        "codec (no offline codec cache)"]

    # ------------------------------------------------------------ sampling
    def _on(self, key, gen=None):
        g = gen if gen is not None else self.gen
        return bool(torch.rand(1, generator=g).item() < self.p[key])

    def sample_style(self, gen=None) -> Dict:
        g = gen if gen is not None else self.gen
        s: Dict = {}
        if self._on("resp", g):
            s["resp"] = {"gains_db": [max(-self.resp_db, min(self.resp_db,
                          _u(g, -self.resp_db, self.resp_db)))
                          for _ in range(self.n_ctrl)]}
        if self._on("lp", g):
            s["lp"] = {"cutoff": _u(g, 1200.0, min(2000.0, self.band))}
        if self._on("hp", g):
            s["hp"] = {"cutoff": _u(g, 20.0, 150.0)}
        if self._on("noise", g):
            hard = torch.rand(1, generator=g).item() < self.p_snr_hard
            lo, hi = (self.snr_hard if hard else self.snr_range)
            s["noise"] = {"snr_db": _u(g, lo, hi),
                          "alpha": _u(g, 0.0, 2.0),   # 0 white .. 1 pink .. 2 brown
                          "seed": int(torch.randint(0, 2**31 - 1, (1,),
                                                    generator=g).item())}
        if self._on("comp", g):
            s["comp"] = {"amount": _u(g, 0.5, 3.0)}   # tanh drive
        if self._on("quant", g):
            s["quant"] = {"bits": int(round(_u(g, 12, 16)))
                          if torch.rand(1, generator=g).item() > 0.15
                          else int(round(_u(g, 8, 12)))}
        return s

    def sample_nuisance(self, gen=None) -> Dict:
        g = gen if gen is not None else self.gen
        n: Dict = {}
        if self._on("hum", g):
            n["hum"] = {"f0": 50.0 if torch.rand(1, generator=g).item() < 0.5 else 60.0,
                        "n_harm": int(_u(g, 1, 3.99)),
                        "amp_db": _u(g, -45.0, -30.0),
                        "phase": _u(g, 0.0, _TWO_PI)}
        if self._on("clip", g):
            n["clip"] = {"thresh": _u(g, 0.75, 0.98)}
        return n

    def sample(self, gen=None) -> Dict:
        return {"style": self.sample_style(gen),
                "nuisance": self.sample_nuisance(gen)}

    # ---------------------------------------------------------- descriptor
    def _response_curve(self, style: Dict, freqs: torch.Tensor) -> torch.Tensor:
        """Composite linear-amplitude response at ``freqs`` (style-core only)."""
        amp = torch.ones_like(freqs)
        r = style.get("resp")
        if r is not None:
            lf = torch.log10(freqs.clamp_min(1.0))
            lc = torch.log10(self.ctrl_f)
            g = torch.tensor(r["gains_db"], dtype=torch.float32)
            idx = torch.clamp(torch.searchsorted(lc, lf), 1, len(lc) - 1)
            w = (lf - lc[idx - 1]) / (lc[idx] - lc[idx - 1]).clamp_min(1e-9)
            db = g[idx - 1] * (1 - w) + g[idx] * w                # smooth interp
            amp = amp * 10.0 ** (db / 20.0)
        if "lp" in style:
            fc = style["lp"]["cutoff"]
            amp = amp / torch.sqrt(1.0 + (freqs / fc) ** 4)       # ~2nd order
        if "hp" in style:
            fc = style["hp"]["cutoff"]
            ratio = fc / freqs.clamp_min(1e-3)
            amp = amp / torch.sqrt(1.0 + ratio ** 4)
        return amp

    def descriptor(self, params: Dict) -> torch.Tensor:
        s = params["style"] if "style" in params else params
        resp = self._response_curve(s, self.desc_f)
        db = 20.0 * torch.log10(resp.clamp_min(1e-6)) / 24.0      # ~[-1,1]
        nz = s.get("noise")
        lo, hi = self.snr_hard[0], self.snr_range[1]
        extra = torch.tensor([
            1.0 if nz else 0.0,
            ((nz["snr_db"] - lo) / (hi - lo)) if nz else 1.0,     # clean -> 1
            (nz["alpha"] / 2.0) if nz else 0.0,
            (s.get("comp", {}).get("amount", 0.0)) / 3.0,
            1.0 - (s.get("quant", {}).get("bits", 16) - 8) / 8.0,
        ], dtype=torch.float32)
        return torch.cat([db, extra])

    def style_distance(self, pa: Dict, pb: Dict) -> float:
        return float(torch.linalg.norm(self.descriptor(pa) - self.descriptor(pb)))

    # ------------------------------------------------------------- apply
    def apply(self, wav: torch.Tensor, params: Dict) -> torch.Tensor:
        """Deterministic given ``params``. wav [T] or [1, T] -> same shape."""
        sq = wav.dim() == 1
        w = wav.unsqueeze(0) if sq else wav
        w = w.float()
        s, n = params.get("style", {}), params.get("nuisance", {})
        T = w.shape[-1]

        # ---- frequency-domain style core (response x roll-offs) ----------
        if any(k in s for k in ("resp", "lp", "hp")):
            spec = torch.fft.rfft(w, dim=-1)
            freqs = torch.fft.rfftfreq(T, d=1.0 / self.sr)
            amp = self._response_curve(s, freqs.clamp_min(1e-3))
            spec = spec * amp.to(spec.dtype).unsqueeze(0)
            w = torch.fft.irfft(spec, n=T, dim=-1)

        # ---- additive colored sensor noise (own deterministic seed) ------
        if "noise" in s:
            nz = s["noise"]
            g = torch.Generator().manual_seed(int(nz["seed"]))
            white = torch.randn(w.shape, generator=g)
            nsp = torch.fft.rfft(white, dim=-1)
            freqs = torch.fft.rfftfreq(T, d=1.0 / self.sr).clamp_min(1.0)
            shape = freqs ** (-float(nz["alpha"]) / 2.0)          # 1/f^a power
            nsp = nsp * shape.to(nsp.dtype).unsqueeze(0)
            noise = torch.fft.irfft(nsp, n=T, dim=-1)
            sp = w.pow(2).mean().clamp_min(1e-12)
            np_ = noise.pow(2).mean().clamp_min(1e-12)
            noise = noise * torch.sqrt(sp / (10 ** (nz["snr_db"] / 10.0)) / np_)
            w = w + noise

        # ---- mains hum (nuisance) ----------------------------------------
        if "hum" in n:
            h = n["hum"]
            t = torch.arange(T, dtype=torch.float32) / self.sr
            amp = (10 ** (h["amp_db"] / 20.0)) * w.abs().max().clamp_min(1e-6)
            hum = torch.zeros(T)
            for k in range(1, int(h["n_harm"]) + 1):
                hum = hum + (amp / k) * torch.sin(_TWO_PI * h["f0"] * k * t
                                                 + h["phase"] * k)
            w = w + hum.unsqueeze(0)

        # ---- mild compression / soft clip / quantization -----------------
        if "comp" in s:
            a = float(s["comp"]["amount"])
            peak = w.abs().max().clamp_min(1e-8)
            w = torch.tanh(w / peak * a) / math.tanh(a) * peak
        if "clip" in n:
            th = float(n["clip"]["thresh"]) * w.abs().max().clamp_min(1e-8)
            w = w.clamp(-th, th)
        if "quant" in s:
            lv = 2 ** (int(s["quant"]["bits"]) - 1)
            peak = w.abs().max().clamp_min(1e-8)
            w = torch.round(w / peak * lv) / lv * peak

        w = w / (w.abs().max() + 1e-8)
        if not torch.isfinite(w).all():
            w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        return w.squeeze(0) if sq else w

    # -------------------------------------------------------- conveniences
    def sample_apply(self, wav):
        p = self.sample()
        return self.apply(wav, p), p

    def sample_far(self, ref_params: Dict, gen=None) -> Dict:
        """Style-core provably different from ``ref_params``; own nuisance.
        Deterministic when ``gen`` is supplied (v3 hard gate)."""
        best, best_d = None, -1.0
        for _ in range(self.far_tries):
            cand = {"style": self.sample_style(gen),
                    "nuisance": self.sample_nuisance(gen)}
            d = self.style_distance(ref_params, cand)
            if d >= self.delta_style:
                return cand
            if d > best_d:
                best, best_d = cand, d
        return best                                   # farthest found

    def sample_apply_far(self, wav, ref_params):
        p = self.sample_far(ref_params)
        return self.apply(wav, p), p


class PhysicsAdapter:
    """Drop-in replacement for the old DeviceSimulator's __call__ so that the
    matched baselines (ERM-Physics, CF-only-Physics, ...) train under the SAME
    physics simulator as CASG (protocol section 14 fairness rule). The ACPL
    theta interface is deliberately unsupported: theta regression belongs to
    the retired objective."""

    def __init__(self, sim: PhysicsSimulator):
        self.sim = sim

    def __call__(self, wav):
        return self.sim.sample_apply(wav)[0]

    def theta_dim(self):
        raise RuntimeError("PhysicsAdapter has no theta: acpl+physics is not "
                           "a supported combination (protocol section 14).")

    def call_with_theta(self, wav):
        raise RuntimeError("PhysicsAdapter has no theta: acpl+physics is not "
                           "a supported combination (protocol section 14).")
