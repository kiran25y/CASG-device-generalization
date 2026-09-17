"""Score the RESERVED Hospital-B smartphone arm. ONE-SHOT — read this first.

    python scripts/casg_score_frozen_phone.py

This arm (source `sungbook_iphone`, ~4,226 clips) was excluded from every
development decision by cfg.data.dev_exclude_sources and has never been
scored. It is the only evaluation in this study under SIMULTANEOUS device
shift and site shift.

Opening it is irreversible in the epistemic sense: once these numbers
exist, no hyperparameter, checkpoint-selection rule or analysis variant may
be chosen with knowledge of them, and the result must be reported whatever
it says. The script therefore:

  * refuses to run if freeze.json is absent (the design must be frozen
    BEFORE the arm is opened, or the arm proves nothing)
  * refuses to re-run over an existing output without --force
  * asserts the arm shares no physical patient (cohort_uid) with any
    training row of any fold, and contains only sungbook_iphone
  * evaluates the ALREADY-TRAINED checkpoints with a single forward pass;
    it cannot and does not modify any existing prediction file

Writes runs/frozen_phone/preds_<tag>.npz per checkpoint plus a summary
table, in the same format as every other arm so the existing bootstrap and
Holm machinery applies unchanged.
"""
import _boot, argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from src.utils import load_config, set_seed
from src.data import manifest as M
from src.data.dataset import make_loader
from src.metrics import compute_metrics
from src.trainer import resolve_device, Trainer
from scripts.casg_tta import build_model

METHODS = {
    "CASG-Lite":        ("runs/casg", "casg_ast_{d}_seed{s}_casg_lite"),
    "CF-only-Physics":  ("runs/casg", "casg_ast_{d}_seed{s}_cf_physics"),
    "ERM-Physics":      ("runs/erm", "erm_ast_{d}_seed{s}_phys"),
    "MixStyle-Physics": ("runs/mixstyle", "mixstyle_ast_{d}_seed{s}_phys"),
}
SEEDS = [0, 1, 2]
ARM = "sungbook_iphone"
OUTDIR = "runs/frozen_phone"


def build_arm(cfg):
    """The reserved arm: Hospital-B phone rows, same patient drop as Tier-3."""
    ep = str(getattr(cfg.data, "external_phone_manifest", "") or
             cfg.data.extra_manifest)
    df = M.add_cohort_uid(M.add_patient_uid(M.load_manifest(ep)))
    df = df[df["source"].astype(str).str.lower() == ARM].copy()
    if df.empty:
        raise SystemExit(f"no rows with source=={ARM} in {ep}")
    drop = str(getattr(cfg.data, "drop_tier3_patient", "") or "")
    if drop:
        # drop_tier3_patient is a patient_uid on the stethoscope manifest;
        # remove the same PHYSICAL patient here via cohort_uid.
        t3 = M.add_cohort_uid(M.add_patient_uid(
            M.load_manifest(cfg.data.test_manifest)))
        bad = set(t3.loc[t3["patient_uid"] == drop, "cohort_uid"])
        if bad:
            n0 = len(df)
            df = df[~df["cohort_uid"].isin(bad)].copy()
            print(f"[arm] dropped {n0 - len(df)} clip(s) of the Tier-3 "
                  f"outlier patient ({drop})")
    return df.reset_index(drop=True)


def assert_unseen(cfg, arm):
    """No physical patient of this arm may appear in ANY fold's training set."""
    from scripts.train_contrastive import build_pool, _dev_filter
    seoul, pool = build_pool(cfg)
    seoul = _dev_filter(seoul, cfg, "train")
    tr = M.add_cohort_uid(M.add_patient_uid(
        pd.concat([seoul, pool], ignore_index=True)))
    overlap = set(tr["cohort_uid"]) & set(arm["cohort_uid"])
    if overlap:
        raise SystemExit(f"ABORT: {len(overlap)} physical patient(s) of the "
                         f"reserved arm appear in training, e.g. "
                         f"{sorted(overlap)[:3]}. The arm is not unseen.")
    srcs = sorted(arm["source"].astype(str).str.lower().unique())
    assert srcs == [ARM], f"arm contaminated with sources {srcs}"
    print(f"[guard] arm is unseen: {len(arm)} clips, "
          f"{arm['cohort_uid'].nunique()} physical patients, "
          f"0 shared with training")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/casg_lite.yaml")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if not os.path.exists("freeze.json"):
        raise SystemExit("ABORT: freeze.json absent. The design must be frozen "
                         "before the reserved arm is opened, or the result is "
                         "not a held-out evaluation.")
    summary_path = os.path.join(OUTDIR, "frozen_phone_summary.csv")
    if os.path.exists(summary_path) and not a.force:
        raise SystemExit(f"{summary_path} exists — the arm has already been "
                         f"opened. Re-scoring cannot un-see the first result; "
                         f"pass --force only if you know why.")
    os.makedirs(OUTDIR, exist_ok=True)

    set_seed(0)
    cfg = load_config(a.config)
    dev = resolve_device(cfg)
    arm = build_arm(cfg)
    assert_unseen(cfg, arm)

    rows = []
    for name, (root, stem_t) in METHODS.items():
        for fold in ["AKGC417L", "Meditron", "LittC2SE", "Litt3200",
                     "smartphone", "none"]:
            for seed in SEEDS:
                stem = stem_t.format(d=fold, s=seed)
                cp = os.path.join(root, f"ckpt_{stem}.pt")
                if not os.path.exists(cp):
                    continue
                ck = torch.load(cp, map_location="cpu")
                dmap = ck["device_map"]
                model = build_model(cfg, ck, dmap).to(dev).eval()
                loader = make_loader(arm, cfg, dmap, cfg.data.data_root,
                                     train=False)
                tr = Trainer(model, cfg, train_df=None, device=dev)
                raw = tr.predict(loader)
                meta = arm.iloc[raw["index"].astype(int)].reset_index(drop=True)
                m = compute_metrics(raw["y4"], raw["prob4"].argmax(1),
                                    raw["y2"], raw["prob2"].argmax(1),
                                    raw["prob4"], raw["prob2"])
                np.savez_compressed(
                    os.path.join(OUTDIR, f"preds_{stem}_frozenphone.npz"),
                    prob4=raw["prob4"], prob2=raw["prob2"],
                    y4=raw["y4"], y2=raw["y2"],
                    device=meta["device"].to_numpy().astype(str),
                    patient=meta["cohort_uid"].to_numpy().astype(str),
                    source=meta["source"].to_numpy().astype(str),
                    method=name, held_out=fold, seed=seed, arm=ARM)
                rows.append({"method": name, "train_fold": fold, "seed": seed,
                             "auroc": m["auroc2"], "auprc": m["auprc2"],
                             "ece": m["ece2"], "n": len(raw["y2"])})
                print(f"{name:<18}{fold:<12}s{seed}  AUROC={m['auroc2']:.4f}"
                      f"  AUPRC={m['auprc2']:.4f}  ECE={m['ece2']:.4f}",
                      flush=True)

    if not rows:
        raise SystemExit("no checkpoints found")
    d = pd.DataFrame(rows)
    d.to_csv(summary_path, index=False)
    print("\n=== Reserved Hospital-B smartphone arm: mean over seeds ===")
    print(d.pivot_table(index="method", columns="train_fold",
                        values="auroc", aggfunc="mean").round(4).to_string())
    print("\n=== Mean over all training folds and seeds ===")
    print(d.groupby("method")[["auroc", "auprc", "ece"]].mean().round(4)
          .sort_values("auroc", ascending=False).to_string())
    print(f"\nwritten -> {summary_path} and {len(rows)} preds_*.npz in {OUTDIR}")
    print("This arm is now open. Report the result as it stands.")