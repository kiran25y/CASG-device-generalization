"""Build every paper table/figure from the per-clip prediction dumps.

Because `train_contrastive.py` saves `preds_<tag>.npz` for every run, all
reporting is post-hoc: adding a metric never costs a retrain.

  python scripts/report.py                          # all tables, markdown
  python scripts/report.py --latex                  # LaTeX bodies
  python scripts/report.py --metric auroc2 --baseline erm
  python scripts/report.py --curves --tsne          # figures (needs matplotlib)

Table layout follows the reference work on this cohort (Spec/Sens/Acc/F1/AUROC/
AUPRC, mean with 95% CI) so the two papers are directly comparable, with a
per-device axis and a worst-device column added for the LODO protocol.
"""
import _boot, argparse, glob, os, json, re
import numpy as np
import pandas as pd
from src.metrics import (compute_metrics, bootstrap_ci, paired_bootstrap_test,
                         auroc, auprc)

METRICS = ["specificity", "sensitivity", "accuracy", "f1_2", "auroc2", "auprc2"]
PRETTY = {"specificity": "Specificity", "sensitivity": "Sensitivity",
          "accuracy": "Accuracy", "f1_2": "F1", "auroc2": "AUROC",
          "auprc2": "AUPRC", "icbhi4": "ICBHI-4", "se4": "Se-4",
          "sp4": "Sp-4", "uar4": "UAR-4", "auroc4": "AUROC-4"}

# --- LODO grid hygiene -------------------------------------------------------
# Two ways this stops being a grid without anything erroring:
#
#  1. held_out="none" is the EXTERNAL run (trained on every device, evaluated on
#     the second hospital site). It is NOT a leave-one-device-out fold. Pooling
#     it in adds a sixth "device" column and shifts every method mean and every
#     paired-bootstrap p-value.
#  2. Extra seeds. Seeds 3-4 exist for a few ERM cells and were never removed,
#     so those cells average five runs while every other cell averages three.
#     Unbalanced cells are not comparable and the means are not either.
#
# Both are dropped here and announced, rather than left to surface later as a
# number nobody can reproduce.
LODO_DEVICES = ["AKGC417L", "Meditron", "LittC2SE", "Litt3200", "smartphone"]
EXTERNAL_HELD_OUT = "none"


def lodo_only(df, n_seeds=3, quiet=False):
    """Restrict a run table to the balanced leave-one-device-out grid."""
    if df is None or len(df) == 0:
        return df
    ext = int((df.held_out == EXTERNAL_HELD_OUT).sum())
    out = df[df.held_out.isin(LODO_DEVICES)].copy()
    extra = int((out.seed >= n_seeds).sum())
    out = out[out.seed < n_seeds].copy()
    if not quiet and (ext or extra):
        msg = []
        if ext:
            msg.append(f"{ext} external run(s) (held_out='{EXTERNAL_HELD_OUT}') "
                       f"- reported separately, never in the LODO table")
        if extra:
            msg.append(f"{extra} run(s) with seed >= {n_seeds} - unbalanced cells")
        print("  [grid] dropped " + "; ".join(msg))
    cnt = out.pivot_table(index="method", columns="held_out",
                          values="seed", aggfunc="count").fillna(0).astype(int)
    if not quiet and cnt.size and cnt.values.min() != cnt.values.max():
        print("  [grid] *** STILL UNBALANCED after filtering:")
        print(cnt.to_string())
    return out


def external_only(df, n_seeds=3):
    """The complementary slice: the frozen external-site evaluation."""
    if df is None or len(df) == 0:
        return df
    out = df[(df.held_out == EXTERNAL_HELD_OUT) & (df.seed < n_seeds)]
    return out.copy()
# -----------------------------------------------------------------------------


def load_runs(root="runs", pattern="preds_*.npz", include_tagged=False):
    """Load per-clip dumps from the MAIN grid only.

    Runs carrying a --tag_suffix (sweeps, ablations, backbone variants) used a
    different config — different SSP cutoff, backbone or ablated component — so
    mixing them into the main table silently compares across preprocessing
    settings. They are excluded unless --include_tagged is passed.
    """
    rows, skipped = [], []
    for f in sorted(glob.glob(os.path.join(root, "*", pattern))):
        base = os.path.basename(f)
        if base.startswith("preds_ood_"):
            continue
        # main-grid tags are exactly preds_<method>_<backbone>_<held>_seed<N>.npz
        stem = base[len("preds_"):-len(".npz")]
        if not re.fullmatch(r"[a-z]+_[a-z]+_[A-Za-z0-9]+_seed\d+", stem):
            if not include_tagged:
                skipped.append(base)
                continue
        z = np.load(f, allow_pickle=True)
        p4, p2 = z["prob4"], z["prob2"]
        m = compute_metrics(z["y4"], p4.argmax(1), z["y2"], p2.argmax(1), p4, p2)
        m["specificity"] = m["sp2"] * 100
        m["sensitivity"] = m["se2"] * 100
        m["accuracy"] = m["acc2"] * 100
        m.update({"method": str(z["method"]), "backbone": str(z["backbone"]),
                  "held_out": str(z["held_out"]), "seed": int(z["seed"]),
                  "file": f})
        # per-device (matters when the test set spans devices, e.g. held_out=none)
        dev = z["device"].astype(str)
        per = {}
        for d in sorted(set(dev.tolist())):
            k = dev == d
            per[d] = compute_metrics(z["y4"][k], p4[k].argmax(1),
                                     z["y2"][k], p2[k].argmax(1), p4[k], p2[k])
        m["_per_device"] = per
        m["_worst_auroc2"] = float(np.nanmin([v["auroc2"] for v in per.values()])) \
            if per else np.nan
        rows.append(m)
    if skipped:
        print(f"[report] EXCLUDED {len(skipped)} tagged run(s) (sweep/ablation, "
              f"different config): {sorted(set(skipped))[:4]}"
              f"{' ...' if len(skipped) > 4 else ''}")
    if not rows:
        raise SystemExit(f"no main-grid preds_*.npz under {root}/*/ — run training first")
    return pd.DataFrame(rows)


def fmt_ci(vals, pct=False, nd=3, n_boot=1000):
    m, lo, hi = bootstrap_ci(vals, n_boot=n_boot)
    if not np.isfinite(m):
        return "—"
    if pct:
        return f"{m:.2f} ({lo:.2f}–{hi:.2f})"
    return f"{m:.{nd}f} ({lo:.{nd}f}–{hi:.{nd}f})"


def table_deployment(df, held_out, latex=False):
    """Reference-style 6-metric table for one held-out device."""
    sub = df[df.held_out == held_out]
    if sub.empty:
        return f"\n(no runs with held_out={held_out})\n"
    out = [f"\n### Held-out device: {held_out}  (mean, 95% CI over seeds)\n",
           "| Method | " + " | ".join(PRETTY[m] for m in METRICS) + " |",
           "|---" * (len(METRICS) + 1) + "|"]
    for meth, g in sub.groupby("method"):
        cells = [fmt_ci(g[m].values, pct=m in ("specificity", "sensitivity", "accuracy"))
                 for m in METRICS]
        out.append(f"| {meth} (n={len(g)}) | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_main(df, metric="auroc2"):
    """methods x held-out device, + worst-device column."""
    devs = sorted(df.held_out.unique())
    out = [f"\n### Main table — {PRETTY.get(metric, metric)} (mean, 95% CI over seeds)\n",
           "| Method | " + " | ".join(devs) + " | worst-device |",
           "|---" * (len(devs) + 2) + "|"]
    for meth, g in df.groupby("method"):
        cells = [fmt_ci(g[g.held_out == d][metric].values) for d in devs]
        means = [g[g.held_out == d][metric].mean() for d in devs
                 if len(g[g.held_out == d])]
        means = [m for m in means if np.isfinite(m)]
        worst = f"{min(means):.3f}" if means else "—"
        out.append(f"| {meth} | " + " | ".join(cells) + f" | {worst} |")
    return "\n".join(out)


def significance(df, metric="auroc2", baseline="erm", n_boot=10000):
    piv = df.pivot_table(index=["held_out", "seed"], columns="method", values=metric)
    if baseline not in piv.columns:
        return f"\n(baseline '{baseline}' not present)\n"
    out = [f"\n### Paired bootstrap vs {baseline} on {PRETTY.get(metric, metric)}"
           f" (paired by device x seed)\n",
           "| Method | n | mean | baseline | delta | p | p_bonf | sig |",
           "|---|---|---|---|---|---|---|---|"]
    others = [c for c in piv.columns if c != baseline]
    nb = max(1, len(others))
    for m in others:
        d = piv[[baseline, m]].dropna()
        if len(d) < 2:
            out.append(f"| {m} | {len(d)} | — | — | — | — | — | insufficient |")
            continue
        p = paired_bootstrap_test(d[m].values, d[baseline].values, n_boot=n_boot)
        delta = d[m].mean() - d[baseline].mean()
        pb = min(1.0, p * nb)
        out.append(f"| {m} | {len(d)} | {d[m].mean():.4f} | {d[baseline].mean():.4f} "
                   f"| {delta:+.4f} | {p:.4f} | {pb:.4f} | "
                   f"{'YES' if pb < 0.05 else 'no'} |")
    return "\n".join(out)


def table_calibration(df):
    out = ["\n### Calibration (binary head)\n",
           "| Method | ECE | Brier |", "|---|---|---|"]
    for meth, g in df.groupby("method"):
        out.append(f"| {meth} | {fmt_ci(g['ece2'].values)} | {fmt_ci(g['brier2'].values)} |")
    return "\n".join(out)


def figures(df, outdir="runs/figures", curves=True, tsne=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figures] matplotlib unavailable ({e}); skipping")
        return
    os.makedirs(outdir, exist_ok=True)
    if curves:
        for held, g in df.groupby("held_out"):
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            for meth, gg in g.groupby("method"):
                z = np.load(gg.iloc[0]["file"], allow_pickle=True)
                y, s = z["y2"], z["prob2"][:, 1]
                o = np.argsort(-s)
                yy = y[o]
                tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
                tpr = tp / max(1, yy.sum()); fpr = fp / max(1, (1 - yy).sum())
                ax[0].plot(fpr, tpr, label=f"{meth} ({auroc(y, s):.3f})")
                prec = tp / np.arange(1, len(yy) + 1)
                ax[1].plot(tp / max(1, yy.sum()), prec, label=f"{meth} ({auprc(y, s):.3f})")
            ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
            ax[0].set(xlabel="FPR", ylabel="TPR", title=f"ROC — held-out {held}")
            ax[1].set(xlabel="Recall", ylabel="Precision", title=f"PR — held-out {held}")
            for a in ax:
                a.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"roc_pr_{held}.png"), dpi=180)
            plt.close(fig)
        print(f"[figures] ROC/PR -> {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--metric", default="auroc2")
    ap.add_argument("--baseline", default="erm")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--tsne", action="store_true")
    ap.add_argument("--out", default="runs/REPORT.md")
    ap.add_argument("--include_tagged", action="store_true",
                    help="also include sweep/ablation runs (different configs!)")
    ap.add_argument("--n_seeds", type=int, default=3,
                    help="seeds per cell; runs with a higher seed are dropped so "
                         "every cell has the same n")
    ap.add_argument("--no_filter", action="store_true",
                    help="disable LODO filtering (keeps held_out=none and extra "
                         "seeds in the main table -- for debugging only)")
    a = ap.parse_args()

    raw = load_runs(a.root, include_tagged=a.include_tagged)
    print(f"[report] loaded {len(raw)} runs: {raw.groupby('method').size().to_dict()}")
    ext_df = external_only(raw, a.n_seeds)
    df = raw if a.no_filter else lodo_only(raw, a.n_seeds)
    cnt = df.pivot_table(index="method", columns="held_out", values="seed", aggfunc="count")
    print("\nruns per (method, device) — every cell should equal the seed count:")
    print(cnt.fillna(0).astype(int).to_string())
    if cnt.fillna(0).values.min() != cnt.fillna(0).values.max():
        print("\n*** GRID INCOMPLETE — table below mixes cells with different n. ***")

    parts = [f"# Results report\n\nRuns: {len(df)}  "
             f"(methods: {sorted(df.method.unique())}, "
             f"devices: {sorted(df.held_out.unique())}, "
             f"seeds: {sorted(df.seed.unique())})\n",
             table_main(df, a.metric)]
    for d in sorted(df.held_out.unique()):
        parts.append(table_deployment(df, d))
    parts.append(significance(df, a.metric, a.baseline))
    parts.append(table_calibration(df))

    if len(ext_df):
        parts.append("\n### External site (held_out=none, frozen second hospital)\n")
        parts.append("Not a leave-one-device-out fold and never pooled into the "
                     "tables above.\n")
        parts.append(table_deployment(ext_df, "none").replace(
            "Held-out device: none", "External site"))

    # secondary 4-class view
    parts.append("\n### 4-class (ICBHI benchmark view)\n")
    g = df.groupby("method")[["icbhi4", "se4", "sp4", "uar4", "auroc4"]].agg(["mean", "std"])
    parts.append(g.round(4).to_string())

    txt = "\n".join(parts)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)
    print(f"\n[report] written -> {a.out}")

    if a.curves or a.tsne:
        figures(df, curves=a.curves, tsne=a.tsne)