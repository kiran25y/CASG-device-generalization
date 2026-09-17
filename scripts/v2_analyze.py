"""v2 Phase-1 analysis. Run ONLY after all 54 runs exist.

    python scripts/v2_analyze.py                # freeze + verify + analyse
    python scripts/v2_analyze.py --selection ema

Steps
  1. Freeze runs_v2: sha256 of every ckpt/preds/metrics + code commit
     -> runs_v2/freeze_v2.json (refuses to overwrite without --force).
  2. Alignment gate: per fold, all 9 prediction files (3 conditions x 3
     seeds) have identical uid / label / cohort vectors. Abort otherwise.
  3. Per-condition AUROC per fold (single mean+-SD, 3-seed ensemble),
     LODO mean and worst, Hospital-B stethoscope arm.
  4. Patient-clustered paired bootstrap (2,000 common draws, seed 20260914)
     on the ensemble for the three PRE-DECLARED contrasts:
        CASG - Identity      (does the transfer operation contribute?)
        CASG - CF-only       (does the whole package beat consistency alone?)
        Identity - CF-only   (does the extra branch/loss package contribute?)
     on LODO mean, LODO worst (minimum recomputed within each draw) and the
     stethoscope arm. Primary endpoint: LODO mean. Bonferroni for the two
     formal contrasts involving CASG -> 97.5% marginal intervals reported
     alongside 95%.
  5. Training diagnostics summary from metrics json: selected epoch,
     donor utilisation, cross-domain fraction, displacement, EMA epoch.
Writes runs_v2/analysis/*.csv and prints the tables.
"""
import _boot, argparse, glob, hashlib, json, os, subprocess, sys
import numpy as np, pandas as pd

FOLDS = ["AKGC417L", "Meditron", "LittC2SE", "Litt3200", "smartphone"]
ARMS = FOLDS + ["none"]
SEEDS = [0, 1, 2]
COND = {"CF-only": "cf_physics", "Identity": "casg_identity", "CASG": "casg_lite"}
CONTRASTS = [("CASG", "Identity"), ("CASG", "CF-only"), ("Identity", "CF-only")]


def sha(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0: return np.nan
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    ss = s[o]; i = 0
    while i < len(ss):                               # average tied ranks
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]: j += 1
        if j > i: r[o[i:j + 1]] = r[o[i:j + 1]].mean()
        i = j + 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def path(root, cond, arm, seed, sel):
    return os.path.join(root, "casg", f"preds_casg_ast_{arm}_seed{seed}_{COND[cond]}_v2"
                        + ("_ema" if sel == "ema" else "") + ".npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_v2")
    ap.add_argument("--selection", default="raw", choices=["raw", "ema"])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = os.path.join(a.root, "analysis"); os.makedirs(out, exist_ok=True)

    # ------------------------------------------------------------ 1. freeze
    fz = os.path.join(a.root, "freeze_v2.json")
    if os.path.exists(fz) and not a.force:
        print(f"freeze exists: {fz} (not overwritten)")
    else:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        rec = {"commit": commit, "files": {}}
        for p in sorted(glob.glob(os.path.join(a.root, "casg", "*"))):
            if p.endswith((".pt", ".npz", ".json")):
                rec["files"][p] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
        json.dump(rec, open(fz, "w"), indent=1)
        print(f"FROZEN {len(rec['files'])} files -> {fz}  commit {commit[:12]}")

    # ------------------------------------------------------ 2. alignment
    print(f"\n== alignment gate ({a.selection}) ==")
    data = {}
    for arm in ARMS:
        ref = None
        for cond in COND:
            for s in SEEDS:
                p = path(a.root, cond, arm, s, a.selection)
                if not os.path.exists(p):
                    print(f"*** MISSING {p}"); sys.exit(1)
                z = np.load(p, allow_pickle=True)
                key = (z["uid"].astype(str), z["y2"], z["cohort"].astype(str))
                if ref is None: ref = key
                else:
                    for nm, x, r in zip(("uid", "y2", "cohort"), key, ref):
                        if not np.array_equal(x, r):
                            print(f"*** ALIGNMENT FAILURE {arm}: {nm} differs in {p}"); sys.exit(1)
                data[(cond, arm, s)] = {"y": z["y2"], "p": z["prob2"][:, 1], "coh": key[2]}
        print(f"  {arm:<11} 9 files aligned, {len(ref[1])} clips, {len(np.unique(ref[2]))} cohorts")

    # ------------------------------------------------- 3. point estimates
    rows = []
    ens = {}
    for cond in COND:
        for arm in ARMS:
            singles = [auroc(data[(cond, arm, s)]["y"], data[(cond, arm, s)]["p"]) for s in SEEDS]
            pe = np.mean([data[(cond, arm, s)]["p"] for s in SEEDS], 0)
            ens[(cond, arm)] = pe
            rows.append({"condition": cond, "arm": arm, "single_mean": np.mean(singles),
                         "single_sd": np.std(singles, ddof=1), "ensemble": auroc(data[(cond, arm, 0)]["y"], pe)})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, f"auroc_{a.selection}.csv"), index=False)
    print(f"\n== ensemble AUROC ({a.selection} selection) ==")
    piv = df.pivot(index="condition", columns="arm", values="ensemble")[ARMS]
    piv["LODO_mean"] = piv[FOLDS].mean(1); piv["LODO_worst"] = piv[FOLDS].min(1)
    print(piv.round(4).to_string())
    print("\n== single-model mean ± SD ==")
    print(df.pivot(index="condition", columns="arm", values="single_mean")[ARMS].round(4).to_string())

    # ------------------------------------- 4. paired clustered bootstrap
    rng = np.random.default_rng(20260914)
    y = {arm: data[("CASG", arm, 0)]["y"] for arm in ARMS}
    coh = {arm: data[("CASG", arm, 0)]["coh"] for arm in ARMS}
    groups = {arm: {c: np.where(coh[arm] == c)[0] for c in np.unique(coh[arm])} for arm in ARMS}
    keys = {arm: list(groups[arm]) for arm in ARMS}
    stats = {c: {"mean": [], "worst": [], "steth": []} for c in COND}
    for _ in range(a.n_boot):
        idx = {}
        for arm in ARMS:                         # common draw across conditions
            pick = rng.integers(0, len(keys[arm]), len(keys[arm]))
            idx[arm] = np.concatenate([groups[arm][keys[arm][k]] for k in pick])
        for cond in COND:
            per = [auroc(y[f][idx[f]], ens[(cond, f)][idx[f]]) for f in FOLDS]
            stats[cond]["mean"].append(np.nanmean(per)); stats[cond]["worst"].append(np.nanmin(per))
            stats[cond]["steth"].append(auroc(y["none"][idx["none"]], ens[(cond, "none")][idx["none"]]))
    crows = []
    for A, B in CONTRASTS:
        for ep in ("mean", "worst", "steth"):
            d = np.array(stats[A][ep]) - np.array(stats[B][ep])
            est = {"mean": piv.loc[A, "LODO_mean"] - piv.loc[B, "LODO_mean"],
                   "worst": piv.loc[A, "LODO_worst"] - piv.loc[B, "LODO_worst"],
                   "steth": piv.loc[A, "none"] - piv.loc[B, "none"]}[ep]
            lo95, hi95 = np.nanpercentile(d, [2.5, 97.5]); lo975, hi975 = np.nanpercentile(d, [1.25, 98.75])
            crows.append({"contrast": f"{A} - {B}", "endpoint": ep, "estimate": est,
                          "lo95": lo95, "hi95": hi95, "lo97.5": lo975, "hi97.5": hi975,
                          "excludes0_95": bool(lo95 > 0 or hi95 < 0)})
    cd = pd.DataFrame(crows); cd.to_csv(os.path.join(out, f"paired_{a.selection}.csv"), index=False)
    print(f"\n== paired contrasts, {a.n_boot} common cohort-clustered draws ({a.selection}) ==")
    print(cd.round(4).to_string(index=False))

    # ------------------------------------------------- 5. diagnostics
    drows = []
    for cond in COND:
        for arm in ARMS:
            for s in SEEDS:
                mp = os.path.join(a.root, "casg", f"metrics_casg_ast_{arm}_seed{s}_{COND[cond]}_v2.json")
                if not os.path.exists(mp): continue
                m = json.load(open(mp)); dg = m.get("diagnostics") or []
                act = [d for d in dg if d.get("branch_active_steps", 0) > 0]
                drows.append({"condition": cond, "arm": arm, "seed": s,
                              "best_epoch": m.get("best_epoch"), "ema_best_epoch": m.get("ema_best_epoch"),
                              "pre_intervention_ckpt": (m.get("best_epoch", 99) is not None and m.get("best_epoch", 99) < 5),
                              "donor_util": np.mean([d["donor_util"] for d in act]) if act else np.nan,
                              "cross_domain": np.mean([d["cross_domain_frac"] for d in act]) if act else np.nan,
                              "disp_mean": np.mean([d["disp_mean"] for d in act]) if act else np.nan,
                              "disp_max": max([d["disp_max"] for d in act]) if act else np.nan})
    dd = pd.DataFrame(drows); dd.to_csv(os.path.join(out, "diagnostics.csv"), index=False)
    print("\n== training diagnostics (mean over seeds) ==")
    print(dd.groupby("condition")[["best_epoch", "donor_util", "cross_domain", "disp_mean", "disp_max"]]
          .mean().round(3).to_string())
    n_pre = int(dd["pre_intervention_ckpt"].sum())
    print(f"\ncheckpoints selected before the intervention epoch: {n_pre}  "
          f"(report this number; do not re-select)")
    print(f"\nwritten -> {out}/")