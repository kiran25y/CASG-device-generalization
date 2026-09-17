"""Aggregate runs/comparison.csv into the paper table: per-method mean+/-std of
each metric (averaged over devices+seeds), plus Wilcoxon signed-rank +
Bonferroni on the method-vs-baseline deltas (paired by device+seed)."""
import _boot, argparse, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from itertools import combinations


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/comparison.csv")
    ap.add_argument("--metric", default="icbhi2")
    ap.add_argument("--baseline", default="erm")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    df = df[df["status"] == "ok"].copy()
    if df.empty: raise SystemExit("no successful runs yet")

    print(f"\n=== per-method summary (mean +/- std over device x seed) ===")
    for met in ["icbhi4", "icbhi2", "se4", "uar4", "se2"]:
        if met not in df: continue
        g = df.groupby("method")[met].agg(["mean", "std", "count"])
        print(f"\n[{met}]"); print(g.round(4).to_string())

    print(f"\n=== Wilcoxon + Bonferroni on {a.metric} (paired by device+seed) ===")
    piv = df.pivot_table(index=["held_out", "seed"], columns="method", values=a.metric)
    methods = [m for m in piv.columns if piv[m].notna().sum() >= 3]
    pairs = list(combinations(methods, 2)); nb = max(1, len(pairs))
    for x, y in pairs:
        d = piv[[x, y]].dropna()
        if len(d) < 3: continue
        try: stat, p = wilcoxon(d[x], d[y])
        except ValueError: p = 1.0
        print(f"  {x} ({d[x].mean():.4f}) vs {y} ({d[y].mean():.4f}): "
              f"p={p:.4f} p_bonf={min(1.0, p*nb):.4f} sig={'YES' if p*nb < 0.05 else 'no'}")
