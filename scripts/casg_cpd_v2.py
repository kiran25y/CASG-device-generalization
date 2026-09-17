"""CPD / CFV, waveform-faithful (protocol v3 16.1-16.2).

Supersedes scripts/casg_cpd.py, which perturbed the ALREADY-NORMALISED
log-Mel with a dB curve sampled on a LINEAR frequency grid. That is not the
transform the training simulator applies: it bypasses SSP, the mel filter
bank (whose bin centres are non-linear) and the per-clip renormalisation,
so it measured sensitivity to a perturbation the model was never trained
against. This version applies the shared perturbation bank to the
HARMONISED WAVEFORM and re-runs the full feature pipeline, exactly as
training does.

    python scripts/casg_cpd_v2.py --held_out Litt3200 --m 16 \
        --ckpt <ckpt> [<ckpt> ...]

Every model sees the identical bank, so this is a matched mechanism
comparison. Reports CPD (mean Jensen-Shannon divergence from the
unperturbed prediction) and CFV (variance of p(abnormal)) per checkpoint.
"""
import _boot, argparse, csv, os
import numpy as np
import torch
import torch.nn.functional as F
from src.utils import load_config, set_seed
from src.data import manifest as M
from src.data import dataset as DS
from src.data.dataset import RSCDataset

# Older copies of src/data/dataset.py resolve the manifest path inline inside
# __getitem__ and expose neither helper. Rather than require the box to be on
# the newest dataset.py (which would silently change the feature pipeline this
# script is supposed to reproduce exactly), fall back to equivalent local
# implementations. _resolve mirrors the inline logic; _fix_len mirrors the
# centre-crop / repeat-pad used at eval time (train=False).
_fix = getattr(DS, "_fix", None)
if _fix is None:
    def _fix(w, L, train, mode="repeat"):
        T = w.shape[-1]
        if T == L:
            return w
        if T > L:
            st = int(torch.randint(0, T - L + 1, (1,)).item()) if train \
                 else (T - L) // 2
            return w[st:st + L]
        if mode == "repeat" and T > 0:
            return w.repeat(int(np.ceil(L / T)))[:L]
        return F.pad(w, (0, L - T))

_resolve = getattr(DS, "resolve_audio_path", None)
if _resolve is None:
    def _resolve(filepath, data_root="", path_rewrites=None):
        raw = os.path.expanduser(str(filepath))
        p = os.path.normpath(raw if os.path.isabs(raw)
                             else os.path.join(str(data_root), raw))
        if os.path.exists(p):
            return p
        for old, new in sorted((path_rewrites or {}).items(),
                               key=lambda kv: len(str(kv[0])), reverse=True):
            old = os.path.normpath(os.path.expanduser(str(old)))
            new = os.path.normpath(os.path.expanduser(str(new)))
            if p == old:
                return new
            if p.startswith(old + os.sep):
                return os.path.join(new, p[len(old) + 1:])
        return p
from src.trainer import resolve_device
from src.casg.physics import PhysicsSimulator
from src.casg.rng import sample_rng
from scripts.casg_tta import build_model


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda x, y: (x * (x.clamp_min(1e-9).log() - y.clamp_min(1e-9).log())).sum(-1)
    return 0.5 * (kl(p, m) + kl(q, m))


@torch.no_grad()
def cpd_cfv(model, ds, sim, bank, dev, batch=16):
    """Waveform -> physics -> SSP-consistent mel -> model, per perturbation."""
    CPD, CFV = [], []
    idx = list(range(len(ds.df)))
    for s in range(0, len(idx), batch):
        chunk = idx[s:s + batch]
        waves = []
        for i in chunk:
            r = ds.df.iloc[i]
            path = _resolve(r["filepath"], ds.root,
                            getattr(ds, "path_rewrites", {}))
            waves.append(_fix(ds._get(path), int(ds.L), False,
                              getattr(ds, "pad_mode", "repeat")))
        clean = torch.stack([ds.mel(w) for w in waves]).to(dev)
        p0 = F.softmax(model(clean)["logits2"], -1)
        ds_, ps_ = [], []
        for params in bank:
            mels = torch.stack([ds.mel(sim.apply(w, params)) for w in waves]).to(dev)
            pk = F.softmax(model(mels)["logits2"], -1)
            ds_.append(js(p0, pk))
            ps_.append(pk[:, 1])
        CPD.append(torch.stack(ds_).mean(0).cpu())
        CFV.append(torch.stack(ps_).var(0).cpu())
    return float(torch.cat(CPD).mean()), float(torch.cat(CFV).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--held_out", required=True)
    ap.add_argument("--config", default="configs/casg_lite.yaml")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--bank_seed", type=int, default=20260827)
    ap.add_argument("--max_clips", type=int, default=0,
                    help="subsample the test fold for speed (0 = all)")
    ap.add_argument("--out", default="runs/cpd_report.csv")
    a = ap.parse_args()
    set_seed(0)
    cfg = load_config(a.config)
    dev = resolve_device(cfg)

    from scripts.train_contrastive import build_pool, _dev_filter
    seoul, pool = build_pool(cfg)
    seoul = _dev_filter(seoul, cfg, "train")
    test_df = (M.build_loo_folds(pool, seoul, [a.held_out], True)[a.held_out].test
               if a.held_out.lower() != "none" else
               M.build_tier3(M.load_manifest(cfg.data.test_manifest),
                             cfg.data.drop_tier3_patient))
    if a.max_clips and len(test_df) > a.max_clips:
        test_df = test_df.sample(a.max_clips, random_state=0).reset_index(drop=True)

    # bank applied to the WAVEFORM, identical for every model
    sim = PhysicsSimulator(dict(getattr(cfg, "casg", {}) or {}),
                           int(cfg.audio.sample_rate),
                           band_hz=float(cfg.audio.ssp.cutoff_hz))
    bank = [sim.sample(gen=sample_rng(a.bank_seed, "cpd", 0, "bank", k))
            for k in range(a.m)]
    print(f"[cpd] {len(bank)} waveform perturbations, {len(test_df)} clips, "
          f"held_out={a.held_out}")

    rows = []
    for cp in a.ckpt:
        ck = torch.load(cp, map_location="cpu")
        dmap = ck["device_map"]
        model = build_model(cfg, ck, dmap).to(dev).eval()
        ds = RSCDataset(test_df, cfg, dmap, cfg.data.data_root, train=False)
        cpd, cfv = cpd_cfv(model, ds, sim, bank, dev)
        rows.append({"ckpt": os.path.basename(cp), "held_out": a.held_out,
                     "M": a.m, "n_clips": len(test_df), "cpd": cpd, "cfv": cfv})
        print(f"{os.path.basename(cp):<52}CPD={cpd:.6f}  CFV={cfv:.6f}")

    # de-duplicating write: a rerun replaces its own rows instead of appending
    key = lambda r: (r["ckpt"], r["held_out"], r["M"])
    old = []
    if os.path.exists(a.out):
        with open(a.out) as fh:
            old = [r for r in csv.DictReader(fh)
                   if (r["ckpt"], r["held_out"], str(r["M"])) not in
                   {(k[0], k[1], str(k[2])) for k in map(key, rows)}]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(old + rows)
    print(f"written (de-duplicated) -> {a.out}")