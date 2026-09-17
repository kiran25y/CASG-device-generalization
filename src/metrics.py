"""Metrics.

Threshold-dependent : ICBHI score (Se, Sp), accuracy, per-class recall, UAR, F1
Threshold-free      : AUROC, AUPRC  (macro one-vs-rest for the 4-class task)
Calibration         : ECE, Brier
Aggregation         : per-device + worst-device, and bootstrap CIs

Why AUROC/AUPRC were added
--------------------------
Everything previously reported was argmax-based, so a model whose decision
threshold has drifted under prior shift is indistinguishable from a model that
cannot discriminate at all. Concretely: contrastive / held_out=smartphone /
seed 0 gave Se2 0.986 with Sp2 0.155 — that is a mis-placed operating point,
not a failure of discrimination, and the two require completely different
fixes. AUROC/AUPRC separate them. They are also the endpoints the reference
work on this cohort reports, so we need them for any comparison.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np


# ---------------------------------------------------------------------------
# threshold-dependent
# ---------------------------------------------------------------------------
def _se_sp(y_true, y_pred, abn_from=1):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    nm = y_true < abn_from
    ab = ~nm
    sp = float((y_pred[nm] == y_true[nm]).mean()) if nm.any() else float("nan")
    se = float((y_pred[ab] == y_true[ab]).mean()) if ab.any() else float("nan")
    return se, sp


def icbhi(y_true, y_pred):
    se, sp = _se_sp(y_true, y_pred)
    acc = float((np.asarray(y_pred) == np.asarray(y_true)).mean())
    return {"score": float(np.nanmean([se, sp])), "se": se, "sp": sp, "acc": acc}


def uar_f1(y_true, y_pred, n):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recs, f1s = [], []
    for c in range(n):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        r = tp / (tp + fn) if tp + fn else 0.0
        p = tp / (tp + fp) if tp + fp else 0.0
        recs.append(r)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(recs)), float(np.mean(f1s))


def binary_f1(y_true, y_pred, pos=1):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == pos) & (y_true == pos)).sum())
    fp = int(((y_pred == pos) & (y_true != pos)).sum())
    fn = int(((y_pred != pos) & (y_true == pos)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * p * r / (p + r)) if p + r else 0.0


# ---------------------------------------------------------------------------
# threshold-free
# ---------------------------------------------------------------------------
def auroc(y_true, score) -> float:
    """Binary AUROC via the rank (Mann-Whitney) identity; ties handled."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):                     # average ranks within ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auprc(y_true, score) -> float:
    """Average precision (step-wise, the sklearn `average_precision_score`
    convention)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.sum() == 0 or len(y) == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / y.sum()
    d_rec = np.diff(np.concatenate([[0.0], rec]))
    return float((prec * d_rec).sum())


def macro_ovr(y_true, prob, n_classes, fn) -> float:
    vals = []
    for c in range(n_classes):
        v = fn((np.asarray(y_true) == c).astype(int), np.asarray(prob)[:, c])
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def ece(y_true, prob_pos, n_bins=15) -> float:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_pos, dtype=float)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[0]) & (p <= edges[1])
        if not m.any():
            continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def brier(y_true, prob_pos) -> float:
    y = np.asarray(y_true).astype(float)
    p = np.asarray(prob_pos, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def compute_metrics(y4t, y4p, y2t, y2p,
                    prob4: Optional[np.ndarray] = None,
                    prob2: Optional[np.ndarray] = None) -> Dict[str, float]:
    y4t = np.asarray(y4t); y4p = np.asarray(y4p)
    y2t = np.asarray(y2t); y2p = np.asarray(y2p)
    valid = y4t >= 0

    if valid.any():
        m4 = icbhi(y4t[valid], y4p[valid])
        uar4, f14 = uar_f1(y4t[valid], y4p[valid], 4)
        i4, a4, se4, sp4 = m4["score"], m4["acc"], m4["se"], m4["sp"]
        if prob4 is not None:
            au4 = macro_ovr(y4t[valid], np.asarray(prob4)[valid], 4, auroc)
            ap4 = macro_ovr(y4t[valid], np.asarray(prob4)[valid], 4, auprc)
        else:
            au4 = ap4 = float("nan")
    else:
        i4 = a4 = se4 = sp4 = uar4 = f14 = au4 = ap4 = float("nan")

    m2 = icbhi(y2t, y2p)
    if prob2 is not None:
        p_pos = np.asarray(prob2)[:, 1]
        au2, ap2 = auroc(y2t, p_pos), auprc(y2t, p_pos)
        ece2, br2 = ece(y2t, p_pos), brier(y2t, p_pos)
    else:
        au2 = ap2 = ece2 = br2 = float("nan")

    return {"icbhi4": i4, "acc4": a4, "se4": se4, "sp4": sp4, "uar4": uar4,
            "f1_4": f14, "auroc4": au4, "auprc4": ap4,
            "icbhi2": m2["score"], "acc2": m2["acc"], "se2": m2["se"],
            "sp2": m2["sp"], "f1_2": binary_f1(y2t, y2p),
            "auroc2": au2, "auprc2": ap2, "ece2": ece2, "brier2": br2,
            "n": int(len(y2t))}


def per_device(devices: Sequence[str], y4t, y4p, y2t, y2p,
               prob4=None, prob2=None) -> Dict[str, Dict]:
    devices = np.asarray(devices)
    out: Dict[str, Dict] = {}
    for d in sorted(set(devices.tolist())):
        m = devices == d
        out[str(d)] = compute_metrics(
            np.asarray(y4t)[m], np.asarray(y4p)[m],
            np.asarray(y2t)[m], np.asarray(y2p)[m],
            None if prob4 is None else np.asarray(prob4)[m],
            None if prob2 is None else np.asarray(prob2)[m])
    real = [v for k, v in out.items() if not k.startswith("_")]
    for key in ("icbhi2", "se2", "auroc2", "auprc2", "f1_2"):
        vals = [v[key] for v in real if np.isfinite(v.get(key, np.nan))]
        out[f"_worst_{key}"] = float(np.min(vals)) if vals else float("nan")
    return out


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0):
    """Percentile CI of the mean over fold/seed-level scores."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(int(n_boot), len(v)))].mean(axis=1)
    return (float(v.mean()),
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def bootstrap_ci_clips(y_true, score, fn, n_boot: int = 1000, alpha: float = 0.05,
                       seed: int = 0):
    """Percentile CI of a clip-level metric by resampling clips."""
    y = np.asarray(y_true)
    s = np.asarray(score)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(fn(y[idx], s[idx]))
    if not stats:
        return float("nan"), float("nan"), float("nan")
    return (float(fn(y, s)),
            float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def paired_bootstrap_test(a: Sequence[float], b: Sequence[float],
                          n_boot: int = 10000, seed: int = 0) -> float:
    """Two-sided paired bootstrap p-value for mean(a) - mean(b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return float("nan")
    d = a - b
    obs = d.mean()
    rng = np.random.default_rng(seed)
    dc = d - obs                                  # centre under H0
    boot = dc[rng.integers(0, len(dc), size=(int(n_boot), len(dc)))].mean(axis=1)
    return float((np.abs(boot) >= abs(obs)).mean())
