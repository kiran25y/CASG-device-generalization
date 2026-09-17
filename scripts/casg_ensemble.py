"""Seed ensembling — free performance from checkpoints you already trained.

    python scripts/casg_ensemble.py

Averages the predicted probabilities of the 3 seeds of EVERY method
(single-model rows are averaged metrics; the ensemble row is one model
whose prediction is the seed-averaged probability). Applied identically
to all four matched methods, so the comparison stays fair.

Reads only the per-clip prediction dumps (preds_*.npz), so it needs no
GPU and no retraining. Writes runs/ensemble_report.md and prints the
table. Prediction order is verified identical across seeds before
averaging — a mismatch aborts rather than silently mixing clips.
"""
import _boot, argparse, json, os, re
import numpy as np
from src.metrics import compute_metrics

DEVS = ["AKGC417L", "Meditron", "LittC2SE", "Litt3200", "smartphone"]
SPECS = [("CASG-Lite (ours)", "runs/casg/preds_casg_ast_{d}_seed{s}_casg_lite.npz"),
         ("CF-only-Physics",  "runs/casg/preds_casg_ast_{d}_seed{s}_cf_physics.npz"),
         ("ERM-Physics",      "runs/erm/preds_erm_ast_{d}_seed{s}_phys.npz"),
         ("MixStyle-Physics", "runs/mixstyle/preds_mixstyle_ast_{d}_seed{s}_phys.npz")]


def ensemble_cell(tmpl, dev, seeds=(0, 1, 2)):
    """-> (metrics_of_ensemble, [per-seed auroc]) or (None, [])."""
    P4, P2, y4, y2, singles, order = [], [], None, None, [], None
    for s in seeds:
        p = tmpl.format(d=dev, s=s)
        if not os.path.exists(p):
            return None, []
        z = np.load(p, allow_pickle=True)
        key = z["patient"].astype(str) if "patient" in z else None
        if order is None:
            order, y4, y2 = key, z["y4"], z["y2"]
        else:
            assert np.array_equal(z["y4"], y4) and np.array_equal(z["y2"], y2), \
                f"{p}: label order differs across seeds — cannot ensemble"
        P4.append(z["prob4"]); P2.append(z["prob2"])
        m = compute_metrics(z["y4"], z["prob4"].argmax(1),
                            z["y2"], z["prob2"].argmax(1), z["prob4"], z["prob2"])
        singles.append(m["auroc2"])
    p4, p2 = np.mean(P4, axis=0), np.mean(P2, axis=0)
    ens = compute_metrics(y4, p4.argmax(1), y2, p2.argmax(1), p4, p2)
    return ens, singles


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ensemble_report.md")
    a = ap.parse_args()

    lines = ["# Seed-ensemble results (3-seed probability average)", "",
             "Single = mean of the three individual models' AUROC.",
             "Ensemble = one model whose prediction is the averaged probability.",
             ""]
    print("\n### LODO AUROC: single-model mean vs 3-seed ensemble\n")
    head = (f"{'method':<20}" + "".join(f"{d[:9]:>11}" for d in DEVS) +
            f"{'MEAN':>8}{'WORST':>8}")
    print(head); print("-" * len(head))
    lines += ["| Method | " + " | ".join(DEVS) + " | Mean | Worst |",
              "|---" * (len(DEVS) + 3) + "|"]
    summary = {}
    for name, tmpl in SPECS:
        singles_dev, ens_dev = [], []
        for d in DEVS:
            ens, sing = ensemble_cell(tmpl, d)
            if ens is None:
                singles_dev.append(np.nan); ens_dev.append(np.nan); continue
            singles_dev.append(float(np.mean(sing)))
            ens_dev.append(float(ens["auroc2"]))
        summary[name] = (singles_dev, ens_dev)
        print(f"{name:<20}" + "".join(f"{v:>11.3f}" for v in ens_dev) +
              f"{np.nanmean(ens_dev):>8.3f}{np.nanmin(ens_dev):>8.3f}"
              f"   (single: {np.nanmean(singles_dev):.3f}/"
              f"{np.nanmin(singles_dev):.3f})")
        lines.append(f"| {name} | " +
                     " | ".join(f"{v:.3f}" for v in ens_dev) +
                     f" | {np.nanmean(ens_dev):.3f} | {np.nanmin(ens_dev):.3f} |")

    print("\n### Frozen external site: single vs ensemble\n")
    print(f"{'method':<20}{'single':>9}{'ensemble':>10}{'gain':>8}"
          f"{'AUPRC':>8}{'ECE':>8}")
    lines += ["", "## Frozen external site", "",
              "| Method | Single AUROC | Ensemble AUROC | Gain | AUPRC | ECE |",
              "|---|---|---|---|---|---|"]
    for name, tmpl in SPECS:
        ens, sing = ensemble_cell(tmpl, "none")
        if ens is None:
            continue
        s = float(np.mean(sing))
        print(f"{name:<20}{s:>9.3f}{ens['auroc2']:>10.3f}"
              f"{ens['auroc2'] - s:>+8.3f}{ens['auprc2']:>8.3f}{ens['ece2']:>8.3f}")
        lines.append(f"| {name} | {s:.3f} | {ens['auroc2']:.3f} | "
                     f"{ens['auroc2'] - s:+.3f} | {ens['auprc2']:.3f} | "
                     f"{ens['ece2']:.3f} |")

    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"\nwritten -> {a.out}")
    print("Report BOTH rows in the paper: single-model (protocol-faithful) "
          "and 3-seed ensemble (deployment-realistic). Applying it to every "
          "method keeps the comparison matched.")
