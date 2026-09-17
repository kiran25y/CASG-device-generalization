"""Acquisition test-time augmentation (protocol v3 section 18, Mode A+).

    python scripts/casg_tta.py --ckpt runs/casg/ckpt_casg_ast_AKGC417L_seed0_casg_lite.pt \
        --held_out AKGC417L --m 8

Averages predictions over M plausible synthetic acquisitions of each test
clip, using ONE fixed perturbation bank shared by every model so the
comparison stays matched (same bank as the CPD/CFV metric of section 16).
This asks the model the same clinical question under several plausible
microphones and pools the answers — the deployment-time analogue of the
invariance the training objective targets.

Uses no target data of any kind: the perturbations come from the frozen
physics simulator priors, not from the target device.

    --m 0   evaluates the clean path only (sanity: reproduces the run's
            reported AUROC)

Writes runs/tta_report.csv (one row per checkpoint) and prints clean vs
TTA AUROC/AUPRC/ECE.
"""
import _boot, argparse, csv, glob, os, re
import numpy as np
import torch
import torch.nn.functional as F
from src.utils import load_config, set_seed
from src.data import manifest as M
from src.data.dataset import make_loader
from src.metrics import compute_metrics
from src.trainer import resolve_device
from src.casg.physics import PhysicsSimulator
from src.casg.rng import sample_rng

FREQS = None   # set in __main__ (shared perturbation grid)


def build_model(cfg, ck, dmap):
    sd = ck["state_dict"]
    if any(k.startswith("path_proj.") for k in sd) and \
       any(k.startswith("style_enc.") for k in sd):
        from src.casg.model import CASGNet
        m = CASGNet(cfg, n_devices=max(2, len(dmap)))
    elif any(k.startswith("film.") for k in sd):
        from src.models.acpl import ACPLNet
        td = int(sd["theta_head.weight"].shape[0])
        m = ACPLNet(cfg, n_devices=max(2, len(dmap)), theta_dim=td)
    else:
        from src.models import DeviceAgnosticNet
        m = DeviceAgnosticNet(cfg, n_devices=max(2, len(dmap)))
    m.load_state_dict(sd)
    return m


@torch.no_grad()
def predict(model, loader, dev, sim=None, bank=None):
    """bank=None -> clean path. Otherwise average over the perturbations."""
    P4, P2, Y4, Y2 = [], [], [], []
    for b in loader:
        mel = b["mel"].to(dev)
        o = model(mel)
        p4 = F.softmax(o["logits4"], -1)
        p2 = F.softmax(o["logits2"], -1)
        if bank:
            acc4, acc2 = p4.clone(), p2.clone()
            for params in bank:
                # perturb in the FEATURE domain: scale/shift the log-mel by the
                # simulator's frequency response, which is what the waveform
                # transform induces after harmonisation + mel.
                resp = sim._response_curve(params["style"], FREQS).to(dev)
                gain = (20.0 * torch.log10(resp.clamp_min(1e-6)) / 40.0)
                pert = mel + gain.view(1, 1, -1, 1)
                oo = model(pert)
                acc4 += F.softmax(oo["logits4"], -1)
                acc2 += F.softmax(oo["logits2"], -1)
            p4, p2 = acc4 / (1 + len(bank)), acc2 / (1 + len(bank))
        P4.append(p4.cpu()); P2.append(p2.cpu())
        Y4.append(b["label4"]); Y2.append(b["label2"])
    cat = lambda x: torch.cat(x).numpy()
    return cat(P4), cat(P2), cat(Y4), cat(Y2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--held_out", required=True,
                    help="device name, or 'none' for the frozen external site")
    ap.add_argument("--config", default="configs/casg_lite.yaml")
    ap.add_argument("--m", type=int, default=8, help="perturbations per clip")
    ap.add_argument("--bank_seed", type=int, default=20260827)
    ap.add_argument("--out", default="runs/tta_report.csv")
    a = ap.parse_args()
    set_seed(0)
    cfg = load_config(a.config)
    dev = resolve_device(cfg)

    # ---- test set exactly as training built it ---------------------------
    from scripts.train_contrastive import build_pool, _dev_filter, drop_test_cohorts
    seoul, pool = build_pool(cfg)
    seoul = _dev_filter(seoul, cfg, "train")
    if a.held_out.lower() != "none":
        test_df = M.build_loo_folds(pool, seoul, [a.held_out], True)[a.held_out].test
    else:
        test_df = M.build_tier3(M.load_manifest(cfg.data.test_manifest),
                                cfg.data.drop_tier3_patient)

    # ---- ONE shared perturbation bank ------------------------------------
    n_mels = int(cfg.audio.n_mels)
    globals()['FREQS'] = torch.linspace(50.0, float(cfg.audio.ssp.cutoff_hz), n_mels)
    sim = PhysicsSimulator(dict(getattr(cfg, "casg", {}) or {}),
                           int(cfg.audio.sample_rate),
                           band_hz=float(cfg.audio.ssp.cutoff_hz))
    bank = [sim.sample(gen=sample_rng(a.bank_seed, "tta", 0, "bank", k))
            for k in range(a.m)] if a.m > 0 else []
    print(f"[tta] {len(bank)} shared perturbations, test={len(test_df)} clips")

    rows = []
    for cp in a.ckpt:
        ck = torch.load(cp, map_location="cpu")
        dmap = ck["device_map"]
        model = build_model(cfg, ck, dmap).to(dev).eval()
        loader = make_loader(test_df, cfg, dmap, cfg.data.data_root, train=False)
        c4, c2, y4, y2 = predict(model, loader, dev)
        clean = compute_metrics(y4, c4.argmax(1), y2, c2.argmax(1), c4, c2)
        row = {"ckpt": os.path.basename(cp), "held_out": a.held_out, "M": a.m,
               "auroc_clean": clean["auroc2"], "auprc_clean": clean["auprc2"],
               "ece_clean": clean["ece2"]}
        if bank:
            t4, t2, _, _ = predict(model, loader, dev, sim, bank)
            tta = compute_metrics(y4, t4.argmax(1), y2, t2.argmax(1), t4, t2)
            row.update(auroc_tta=tta["auroc2"], auprc_tta=tta["auprc2"],
                       ece_tta=tta["ece2"],
                       gain=tta["auroc2"] - clean["auroc2"])
        rows.append(row)
        print(f"{os.path.basename(cp):<52} clean={row['auroc_clean']:.4f}"
              + (f"  tta={row['auroc_tta']:.4f}  gain={row['gain']:+.4f}"
                 f"  ECE {row['ece_clean']:.3f}->{row['ece_tta']:.3f}"
                 if bank else ""))

    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    if bank:
        g = [r["gain"] for r in rows]
        print(f"\nmean TTA gain over {len(rows)} checkpoint(s): {np.mean(g):+.4f}")
    print(f"appended -> {a.out}")
