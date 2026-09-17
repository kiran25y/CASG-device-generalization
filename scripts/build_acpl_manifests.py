"""ACPL step 1: master patient table, overlap audit, frozen patient-safe splits.

Implements items 1-5 of the ACPL report's implementation order (Section 13).
Run BEFORE any training. Produces, under --out (default acpl_splits/):

  master_manifest.csv     every clip, one row, with cohort_uid / site / arm
  overlap_report.txt      true physical overlaps vs raw-ID collisions
  label_distribution.csv  per-arm 4-class counts (fills the report's Table 4)
  splits/E*.json          frozen clip-index splits for E1-E8
  FREEZE.sha256           hash of every split file; commit this

The single most important distinction here: a raw patient-ID collision between
unrelated corpora (Seoul id 105 vs ICBHI id 105) is NOT a physical overlap.
Physical identity is cohort_uid = device-invariant source prefix + id, exactly
as in src/data/manifest.py. The ACPL report's "91 overlapping patients between
Seoul stethoscope and ICBHI" is a raw-ID collision and is reported as such.

  python scripts/build_acpl_manifests.py \
      --train  manifests/train_manifest.csv \
      --icbhi  manifests/icbhi_manifest.csv \
      --extra  manifests/extra_manifest.csv \
      --sungbook_steth manifests/sungbook_test_manifest.csv \
      --out acpl_splits
"""
import _boot, argparse, hashlib, json, os, sys
import numpy as np
import pandas as pd
from src.data import manifest as M

SEED = 20260813          # frozen; changing it invalidates every split
VAL_FRACTION = 0.15      # patient-level validation share of each source pool

ICBHI_DEVICES = ["AKGC417L", "Meditron", "LittC2SE", "Litt3200"]


# --------------------------------------------------------------------------- #
def load_all(a):
    frames = []
    for path, tag in [(a.train, "seoul_steth"), (a.icbhi, "icbhi"),
                      (a.extra, "smartphone"), (a.sungbook_steth, "sungbook_steth")]:
        if not path:
            continue
        if not os.path.exists(path):
            print(f"  [warn] {tag}: {path} not found — skipped "
                  f"(E3/E4/E8 need the Sungbook stethoscope manifest)")
            continue
        df = M.load_manifest(path)
        df["_manifest"] = tag
        frames.append(df)
    if not frames:
        raise SystemExit("no manifest could be loaded — check the paths "
                         "printed above (they come from --config unless "
                         "overridden)")
    df = pd.concat(frames, ignore_index=True)
    df = M.add_cohort_uid(M.add_patient_uid(df))

    # site: resolve from source string; the smartphone manifest carries both
    # Seoul and Sungbook arms and MUST be split by site, not treated as one
    def site(src):
        s = str(src).lower()
        return "sungbook" if ("sungbook" in s or "sungbuk" in s) else \
               ("icbhi" if s.startswith("icbhi") else "seoul")
    df["site"] = df["source"].map(site)

    # arm = the report's Table 3 rows
    def arm(r):
        if r["_manifest"] == "icbhi":
            return f"icbhi_{r['device']}"
        if r["_manifest"] == "sungbook_steth":
            return "sungbook_steth"
        if str(r["device"]).lower() in ("smartphone", "iphone"):
            return f"{r['site']}_smartphone"
        return f"{r['site']}_steth"
    df["arm"] = df.apply(arm, axis=1)
    return df


def overlap_audit(df, out):
    """True overlap = shared cohort_uid. Raw-ID collision = same patient_id
    string across corpora with different cohort_uid. Only the former matters."""
    lines = ["OVERLAP AUDIT — physical (cohort_uid) vs raw-ID collisions", "=" * 64]
    arms = sorted(df.arm.unique())
    true_pairs = {}
    for i, a1 in enumerate(arms):
        for a2 in arms[i + 1:]:
            c1 = set(df[df.arm == a1].cohort_uid)
            c2 = set(df[df.arm == a2].cohort_uid)
            true = c1 & c2
            r1 = set(df[df.arm == a1].patient_id.astype(str))
            r2 = set(df[df.arm == a2].patient_id.astype(str))
            raw = r1 & r2
            if true or raw:
                lines.append(f"{a1:24s} <-> {a2:24s}  physical={len(true):4d}  "
                             f"raw-id-collisions={len(raw - set(x.split('_')[-1] for x in true)):4d}")
            if true:
                true_pairs[(a1, a2)] = true
    lines.append("")
    lines.append("Only PHYSICAL overlaps drive exclusion. Raw-ID collisions between")
    lines.append("unrelated corpora (e.g. Seoul id 105 vs ICBHI id 105) are distinct")
    lines.append("people and are NOT excluded. This resolves the report's Table 5:")
    lines.append("the Seoul<->ICBHI '91 overlapping patient IDs' are collisions.")
    txt = "\n".join(lines)
    open(os.path.join(out, "overlap_report.txt"), "w").write(txt)
    print(txt)
    return true_pairs


def label_distribution(df, out):
    tab = (df.groupby(["arm", "label4"]).size().unstack(fill_value=0)
             .rename(columns={0: "normal", 1: "crackle", 2: "wheeze", 3: "both"}))
    tab["clips"] = tab.sum(1)
    tab["patients"] = df.groupby("arm").cohort_uid.nunique()
    tab.to_csv(os.path.join(out, "label_distribution.csv"))
    print("\nPER-ARM LABEL DISTRIBUTION (fills report Table 4)")
    print(tab.to_string())
    return tab


# ---------------------------------------------------------------- splits ---- #
def _val_split(pool, rng):
    """Patient-level validation split inside a source pool."""
    pats = sorted(pool.cohort_uid.unique())
    rng.shuffle(pats)
    nval = max(1, int(len(pats) * VAL_FRACTION))
    val = set(pats[:nval])
    return (pool[~pool.cohort_uid.isin(val)].index.tolist(),
            pool[pool.cohort_uid.isin(val)].index.tolist())


def make_split(df, name, source_mask, target_mask, rng, purpose,
               adapt_budgets=None):
    """One frozen experiment split with hard patient exclusion.

    Rule (report Table 6): no cohort_uid may appear on both sides. Every source
    clip whose patient occurs anywhere in the target is dropped from source.
    """
    src = df[source_mask].copy()
    tgt = df[target_mask].copy()
    if not len(tgt):
        print(f"  [skip] {name}: target empty (missing manifest?)")
        return None
    excl = set(tgt.cohort_uid)
    dropped = src[src.cohort_uid.isin(excl)]
    src = src[~src.cohort_uid.isin(excl)]
    tr_idx, va_idx = _val_split(src, rng)

    split = {"name": name, "purpose": purpose, "seed": SEED,
             "train_idx": tr_idx, "val_idx": va_idx,
             "test_idx": tgt.index.tolist(),
             "excluded_source_clips": int(len(dropped)),
             "excluded_patients": int(dropped.cohort_uid.nunique()),
             "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(tgt),
             "n_test_patients": int(tgt.cohort_uid.nunique()),
             "n_test_normal_patients":
                 int(tgt[tgt.label4 == 0].cohort_uid.nunique())}

    # adaptation pools for E5-E7: patient-disjoint from the evaluation half
    if adapt_budgets:
        pats = sorted(tgt.cohort_uid.unique())
        rng.shuffle(pats)
        half = set(pats[: len(pats) // 2])           # adaptation half
        pool = tgt[tgt.cohort_uid.isin(half)]
        ev = tgt[~tgt.cohort_uid.isin(half)]
        split["adapt_pool_idx"] = pool.index.tolist()
        split["adapt_eval_idx"] = ev.index.tolist()
        split["adapt_budgets"] = adapt_budgets
        split["note"] = ("Adaptation clips are drawn ONLY from adapt_pool_idx; "
                         "evaluation ONLY on adapt_eval_idx. Patient-disjoint "
                         "by construction.")
    return split


def build_splits(df, out):
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.join(out, "splits"), exist_ok=True)
    A = df.arm
    seoul_steth = A == "seoul_steth"
    seoul_phone = A == "seoul_smartphone"
    sung_phone = A == "sungbook_smartphone"
    sung_steth = A == "sungbook_steth"
    icbhi_all = A.str.startswith("icbhi_")

    specs = []
    # E1 — strict LODO over the four ICBHI devices (smartphone moved to E2)
    for dev in ICBHI_DEVICES:
        specs.append((f"E1_lodo_{dev}",
                      (seoul_steth | icbhi_all) & (df.device != dev),
                      icbhi_all & (df.device == dev),
                      "E1 strict LODO domain generalization", None))
    # E2 — Seoul cross-device DG (steth+ICBHI -> Seoul smartphone)
    specs.append(("E2_seoul_crossdevice", seoul_steth | icbhi_all, seoul_phone,
                  "E2 cross-device DG, zero target access", None))
    # E3 — cross-hospital DG (Seoul steth -> Sungbook steth)
    specs.append(("E3_crosshospital", seoul_steth, sung_steth,
                  "E3 cross-hospital DG, same device family", None))
    # E4 — joint device+hospital DG
    specs.append(("E4_joint", seoul_steth | icbhi_all, sung_phone,
                  "E4 joint device+hospital DG, hardest no-target setting", None))
    # E5-E7 — adaptation on Seoul smartphone (budgets sampled at train time
    # from the frozen pool, so the split file itself stays budget-agnostic)
    specs.append(("E5to7_adapt_seoul_phone", seoul_steth | icbhi_all, seoul_phone,
                  "E5 UDA / E6 few-shot / E7 supervised adaptation",
                  {"K_U": [100, 500, 1000], "K_L": [0, 1, 5, 10, 25, 50, 100,
                                                    500, 1000]}))
    # E8 — external validation after adaptation: frozen Sungbook smartphone
    specs.append(("E8_external_sungbook_phone", seoul_steth | icbhi_all,
                  sung_phone, "E8 FROZEN external. Never tune on this.", None))

    written = []
    for name, sm, tm, purpose, budgets in specs:
        sp = make_split(df, name, sm, tm, rng, purpose, budgets)
        if sp is None:
            continue
        p = os.path.join(out, "splits", name + ".json")
        json.dump(sp, open(p, "w"))
        written.append(p)
        print(f"  {name:28s} train {sp['n_train']:6d}  val {sp['n_val']:5d}  "
              f"test {sp['n_test']:5d} ({sp['n_test_patients']} pat, "
              f"{sp['n_test_normal_patients']} normal)  "
              f"excluded {sp['excluded_source_clips']} clips / "
              f"{sp['excluded_patients']} pat")
    return written


def freeze(paths, out):
    h = hashlib.sha256()
    lines = []
    for p in sorted(paths):
        d = hashlib.sha256(open(p, "rb").read()).hexdigest()
        h.update(d.encode())
        lines.append(f"{d}  {os.path.basename(p)}")
    lines.append(f"{h.hexdigest()}  COMBINED")
    open(os.path.join(out, "FREEZE.sha256"), "w").write("\n".join(lines))
    print(f"\nFROZEN. Combined hash {h.hexdigest()[:16]}…  — commit "
          f"{out}/FREEZE.sha256; any change to any split invalidates it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/contrastive.yaml",
                    help="manifest paths default from here; CLI flags override")
    ap.add_argument("--train", default="")
    ap.add_argument("--icbhi", default="")
    ap.add_argument("--extra", default="")
    ap.add_argument("--sungbook_steth", default="")
    ap.add_argument("--out", default="acpl_splits")
    a = ap.parse_args()

    # The training config already knows where the manifests live — including
    # test_manifest, which IS the Sungbook stethoscope set the ACPL report
    # lists as TBD. Guessing paths on the CLI is how the first run failed.
    from src.utils import load_config
    _cfg = load_config(a.config).data
    a.train = a.train or str(_cfg.train_manifest)
    a.icbhi = a.icbhi or str(_cfg.icbhi_manifest)
    a.extra = a.extra or str(_cfg.extra_manifest)
    a.sungbook_steth = a.sungbook_steth or str(getattr(_cfg, "test_manifest", ""))
    print(f"[manifests] train={a.train}  icbhi={a.icbhi}\n"
          f"            extra={a.extra}  sungbook_steth={a.sungbook_steth}")
    os.makedirs(a.out, exist_ok=True)

    df = load_all(a)
    df.to_csv(os.path.join(a.out, "master_manifest.csv"), index=False)
    print(f"master manifest: {len(df)} clips, "
          f"{df.cohort_uid.nunique()} physical patients, arms: "
          f"{sorted(df.arm.unique())}")

    overlap_audit(df, a.out)
    label_distribution(df, a.out)
    written = build_splits(df, a.out)
    freeze(written + [os.path.join(a.out, "master_manifest.csv")], a.out)
