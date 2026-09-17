"""Dataset + spawn-safe loaders.

Reads WAV directly (no torchaudio.load / torchcodec), any bit depth, then:
  1. SSP band harmonisation at the file's NATIVE rate  (src/data/ssp.py)
  2. resample to the model rate
  3. repeat-pad / random-crop to a fixed length
  4. device-style augmentation -> log-mel -> SpecAugment

For contrastive training it emits TWO device-augmented views of the same clip
(SupCon positives). Every item also carries its row index so that per-clip
predictions can be dumped and all downstream metrics (per-device, worst-device,
AUROC, calibration) computed post-hoc without retraining.
"""
from __future__ import annotations
import os, wave
from typing import Dict, Optional
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .augment import DeviceSimulator, SpecAugment
from .ssp import BandHarmonizer


class MelExtractor:
    """Log-mel with an fmax that respects the SSP passband.

    With SSP on (cutoff 2 kHz) the 128 mel bins are packed into 0-2 kHz instead
    of being spread over 0-8 kHz, so all of the frequency resolution is spent on
    the clinically relevant band rather than on empty upper bands.
    """

    def __init__(self, cfg, fmax: Optional[float] = None):
        a = cfg.audio
        self.sr = int(a.sample_rate)
        self.pcen = bool(getattr(a, "pcen", False))
        fmax = float(a.fmax) if fmax is None else float(fmax)
        fmax = min(fmax, self.sr / 2.0)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=int(a.n_fft), win_length=int(a.win_length),
            hop_length=int(a.hop_length), n_mels=int(a.n_mels),
            f_min=float(a.fmin), f_max=fmax, power=2.0)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def __call__(self, wav):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        p = self.mel(wav)
        if self.pcen:
            m = p.clamp_min(1e-6)
            s, prev, frames = 0.04, m[..., :1], []
            for i in range(m.shape[-1]):
                prev = (1 - s) * prev + s * m[..., i:i + 1]
                frames.append(prev)
            sm = torch.cat(frames, -1)
            out = (m / (1e-6 + sm) ** 0.8 + 2.0) ** 0.5 - 2.0 ** 0.5
        else:
            out = self.to_db(p)
        return (out - out.mean()) / (out.std() + 1e-5)


def _decode_pcm(raw, sw):
    if sw == 1: return (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    if sw == 2: return np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if sw == 3:
        a = np.frombuffer(raw, np.uint8); a = a[:(len(a)//3)*3].reshape(-1, 3).astype(np.int32)
        v = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
        v = np.where(v >= (1 << 23), v - (1 << 24), v)
        return v.astype(np.float32) / (1 << 23)
    if sw == 4: return np.frombuffer(raw, "<i4").astype(np.float32) / (1 << 31)
    raise ValueError(f"bad sample width {sw}")


def _read_native(path):
    """Decode to float32 mono at the file's NATIVE sample rate.

    SSP must run before any resampling, so unlike the previous version this
    deliberately does NOT resample here.
    """
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if getattr(x, "ndim", 1) > 1:
            x = x.mean(1)
        return np.ascontiguousarray(x, np.float32), int(sr)
    except Exception:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate(); n = wf.getnframes()
            ch = wf.getnchannels(); sw = wf.getsampwidth()
            raw = wf.readframes(n)
        x = _decode_pcm(raw, sw)
        if ch > 1:
            x = x[:(len(x)//ch)*ch].reshape(-1, ch).mean(1)
        return np.ascontiguousarray(x, np.float32), int(sr)


def _stable_hash(s: str) -> int:
    """Process-independent 63-bit hash (Python's hash() is salted per run)."""
    import hashlib
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") >> 1


def _fix(w, L, train, mode="repeat"):
    """Crop / pad to exactly L samples.

    mode='repeat' tiles the clip instead of zero-padding. This matters: our
    clips are 2.2-3.1 s and the old 8 s zero-padded window was 62-73 % zeros,
    with the *amount* of padding differing systematically by device
    (Litt3200 0.725 vs smartphone 0.617) — the padding was itself a device
    fingerprint. Repeat-padding removes that cue and wastes no input.
    """
    T = w.shape[-1]
    if T == L:
        return w
    if T > L:
        st = int(torch.randint(0, T - L + 1, (1,)).item()) if train else (T - L) // 2
        return w[st:st + L]
    if mode == "repeat" and T > 0:
        reps = int(np.ceil(L / T))
        return w.repeat(reps)[:L]
    return torch.nn.functional.pad(w, (0, L - T))


class _LRUCache(dict):
    """Bounded waveform cache. Unbounded caching of 38 k clips across spawned
    workers is a direct route to host-RAM exhaustion."""

    def __init__(self, maxlen=6000):
        super().__init__()
        self.maxlen = int(maxlen)

    def put(self, k, v):
        if self.maxlen <= 0:
            return
        if len(self) >= self.maxlen:
            self.pop(next(iter(self)))
        self[k] = v


class RSCDataset(Dataset):
    def __init__(self, df, cfg, device_map, data_root, train, contrastive=False):
        self.df = df.reset_index(drop=True)
        if "cohort_uid" not in self.df.columns:                 # v2
            from . import manifest as _M
            self.df = _M.add_cohort_uid(_M.add_patient_uid(self.df))
        self.cfg = cfg
        self._physical_pids = None
        if train and bool(getattr(cfg.data, "physical_patient_donors", False)):
            # Reuse the cohort identity contract used by the split builder.
            if "cohort_uid" not in self.df.columns:
                from .manifest import add_cohort_uid
                if not {"source", "patient_id"}.issubset(self.df.columns):
                    raise ValueError("Physical donor keys require cohort_uid or source/patient_id")
                if self.df[["source", "patient_id"]].isna().any().any():
                    raise ValueError("Missing source/patient_id for physical donor keys")
                self.df = add_cohort_uid(self.df)
            keys = self.df["cohort_uid"]
            if keys.isna().any() or keys.astype(str).str.strip().eq("").any():
                raise ValueError("Missing physical-patient donor key")
            # Exact categorical IDs avoid truncating patient hashes. The mapping
            # is local to this dataset; do not use these IDs to align predictions.
            self._physical_pids = pd.factorize(keys.astype(str), sort=True)[0]
        self.dmap = device_map
        self.root = data_root
        self.train = train
        self.contrastive = contrastive
        self.ssp = BandHarmonizer(cfg)
        self.mel = MelExtractor(cfg, fmax=self.ssp.effective_fmax)
        self.sr = int(cfg.audio.sample_rate)
        self.L = int(cfg.audio.sample_rate * cfg.audio.clip_seconds)
        self.pad_mode = str(getattr(cfg.audio, "pad_mode", "repeat"))
        self.use_aug = bool(cfg.augment.enabled) and train
        self.spec = (SpecAugment(int(cfg.augment.spec_freq_mask),
                                 int(cfg.augment.spec_time_mask),
                                 int(cfg.augment.spec_n_masks))
                     if (train and cfg.augment.spec_augment) else None)
        # Device augmentation runs AFTER SSP, so its filter ranges are clamped
        # to the harmonised band instead of the original 16 kHz band — a 7.5 kHz
        # low-pass is meaningless once everything is band-limited to 2 kHz.
        band = self.ssp.effective_fmax
        self.sim = DeviceSimulator(cfg.augment.device_sim, self.sr, band_hz=band)
        ex = cfg.augment.extreme
        self.ex_sim = (DeviceSimulator(cfg.augment.device_sim, self.sr,
                                       float(ex.scale), band_hz=band)
                       if bool(ex.enabled) else None)
        self.ex_p = float(ex.p) if bool(ex.enabled) else 0.0
        # ACPL: emit the true normalised acquisition parameters with each view
        self.acpl = str(getattr(cfg.model, "method", "")).lower() in ("acpl", "cfsc")
        self.acpl_clean = self.acpl and bool(getattr(cfg.model,
                                                     "acpl_clean_anchor", False))
        self.theta_dim = self.sim.theta_dim()
        # CASG: physics-simulator far-style view pair (training only)
        self.casg = train and str(getattr(cfg.model, 'method', '')).lower() == 'casg'
        self.casg_style_views = self.casg and str(getattr(
            cfg.model, 'variant', 'casg_lite')).lower() == 'casg_full'
        self.casg_fold = str(getattr(cfg, 'casg_fold', 'dev'))
        self.casg_epoch = 0
        if self.casg:
            from src.casg.physics import PhysicsSimulator
            self.physics = PhysicsSimulator(dict(getattr(cfg, 'casg', {}) or {}), self.sr, band_hz=band)
        # matched physics baselines: augment.physics=true swaps the old simulator
        if train and not self.casg and bool(getattr(cfg.augment, 'physics', False)):
            from src.casg.physics import PhysicsSimulator, PhysicsAdapter
            if getattr(self, 'casg', False):      # casg_v3_views
                import zlib
                from src.casg.rng import sample_rng
                uid = str(r.get('sample_uid', r.get('filepath', i)))
                fold = str(getattr(self, 'casg_fold', 'dev'))
                base = int(getattr(self.cfg, 'seed', 0))
                ep = int(getattr(self, 'casg_epoch', 0))
                # two acquisition views, deterministic in the sample key
                pa = self.physics.sample(gen=sample_rng(base, fold, ep, uid, 0))
                pb = self.physics.sample_far(pa, gen=sample_rng(base, fold, ep, uid, 1))
                wa, wb = self.physics.apply(w, pa), self.physics.apply(w, pb)
                ma_clean, mb_clean = self.mel(wa), self.mel(wb)
                if self.spec is not None:
                    # ONE mask for the paired CF task views (v3 section 5)
                    g = sample_rng(base, fold, ep, uid, 99)
                    _st = torch.random.get_rng_state()
                    torch.random.manual_seed(int(torch.randint(
                        0, 2**31 - 1, (1,), generator=g).item()))
                    _mstate = torch.random.get_rng_state()
                    ma = self.spec(ma_clean)
                    torch.random.set_rng_state(_mstate)
                    mb = self.spec(mb_clean)
                    torch.random.set_rng_state(_st)
                else:
                    ma, mb = ma_clean, mb_clean
                out['mel'], out['mel2'] = ma, mb
                if getattr(self, 'casg_style_views', False):
                    # Full CASG: style discovery sees UNMASKED mel
                    out['mel_clean'], out['mel_clean2'] = ma_clean, mb_clean
                out['pid'] = torch.tensor(_stable_hash(str(r["cohort_uid"])), dtype=torch.long)  # v2
                out['uid'] = str(r.get('sample_uid', r.get('filepath', i)))
                return out

            if self.acpl:
                raise ValueError('acpl+physics is not a supported combination')
            self.sim = PhysicsAdapter(PhysicsSimulator(dict(getattr(cfg, 'casg', {}) or {}), self.sr, band_hz=band))
            self.ex_sim = None

        self._cache = _LRUCache(int(getattr(cfg.data, "cache_max_clips", 6000)))

    def __len__(self):
        return len(self.df)

    def _get(self, path):
        hit = self._cache.get(path)
        if hit is not None:
            return hit
        x, native_sr = _read_native(path)
        x, sr = self.ssp.harmonise(x, native_sr)          # <-- SSP, at native rate
        w = torch.from_numpy(np.ascontiguousarray(x, np.float32))
        if sr != self.sr:
            w = AF.resample(w, sr, self.sr)
        self._cache.put(path, w)
        return w

    def _augment(self, w):
        if self.ex_sim is not None and torch.rand(1).item() < self.ex_p:
            return self.ex_sim(w)
        return self.sim(w)

    def _view(self, w):
        m = self.mel(self._augment(w) if self.use_aug else w)
        return self.spec(m) if self.spec is not None else m

    def _view_theta(self, w):
        """View + its true acquisition parameters. Identity theta when the
        simulator is off (eval) or the extreme tier fires (its parameters live
        on a different scale, so it supervises as 'unspecified')."""
        ident = torch.zeros(self.theta_dim, dtype=torch.float32)
        if not self.use_aug:
            m = self.mel(w)
            return (self.spec(m) if self.spec is not None else m), ident
        if self.ex_sim is not None and torch.rand(1).item() < self.ex_p:
            aug, th = self.ex_sim(w), ident
        else:
            aug, th = self.sim.call_with_theta(w)
        m = self.mel(aug)
        return (self.spec(m) if self.spec is not None else m), th

    def __getitem__(self, i):
        r = self.df.iloc[i]
        w = _fix(self._get(os.path.join(self.root, r["filepath"])),
                 self.L, self.train, self.pad_mode)
        out = {"mel": self._view(w),
               "label4": torch.tensor(int(r["label4"]), dtype=torch.long),
               "label2": torch.tensor(int(r["label2"]), dtype=torch.long),
               "device_id": torch.tensor(int(self.dmap.get(r["device"], -1)), dtype=torch.long),
               "index": torch.tensor(int(i), dtype=torch.long)}
        if getattr(self, "casg", False):
            import zlib
            wa, _pa = self.physics.sample_apply(w)
            wb, _pb = self.physics.sample_apply_far(w, _pa)
            m1, m2 = self.mel(wa), self.mel(wb)
            if self.spec is not None:
                # Opt-in new-study setting; legacy configurations stay separate.
                if bool(getattr(self.cfg.augment, "shared_specaugment", False)):
                    m1, m2 = self.spec.apply_pair(m1, m2)
                else:
                    m1, m2 = self.spec.paired(m1, m2)      # v2: shared mask positions
            out["mel"], out["mel2"] = m1, m2
            # v2: donor key is ALWAYS the physical-patient (cohort) key
            out["pid"] = torch.tensor(_stable_hash(str(r["cohort_uid"])), dtype=torch.long)
            out["uid"] = str(r.get("sample_uid", r.get("filepath", i)))
            return out
        if self.acpl:
            m1, t1 = self._view_theta(w)
            m2, t2 = self._view_theta(w)
            out["mel"], out["theta"] = m1, t1
            out["mel2"], out["theta2"] = m2, t2       # two acquisition states
            if self.acpl_clean:
                out["mel0"] = self.spec(self.mel(w)) if self.spec is not None \
                              else self.mel(w)        # un-augmented anchor
        elif self.contrastive:
            out["mel2"] = self._view(w)               # second synthetic-device view
        return out


def make_loader(df, cfg, device_map, data_root, train, contrastive=False, batch_size=None):
    ds = RSCDataset(df, cfg, device_map, data_root, train, contrastive)
    bs = batch_size or int(cfg.optim.batch_size)
    sampler, shuffle = None, train
    mode = str(getattr(cfg.optim, "sampler", "none")).lower()
    if train and mode != "none":
        dev = df["device"].map(lambda d: device_map.get(d, 0)).to_numpy()
        y = df["label4"].to_numpy()
        key = {"device": dev, "class": y, "device_class": dev * 10 + y}[mode]
        _, inv = np.unique(key, return_inverse=True)
        w = (1.0 / np.bincount(inv))[inv]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(df), True)
        shuffle = False
    nw = int(cfg.data.num_workers)
    if not train:
        nw = 0
    if nw > 0 and contrastive:
        nw = min(nw, 4)
    kw = dict(batch_size=bs, shuffle=shuffle, sampler=sampler, num_workers=nw,
              drop_last=train, pin_memory=False)
    if nw > 0:
        import multiprocessing as mp
        kw.update(multiprocessing_context=mp.get_context("spawn"),
                  persistent_workers=True, prefetch_factor=2)
    return DataLoader(ds, **kw)
