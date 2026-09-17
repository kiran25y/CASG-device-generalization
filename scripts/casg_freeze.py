"""Freeze the CASG experiment (protocol v3 section 24.4).

    python scripts/casg_freeze.py

Writes freeze.json recording, for the record and for the paper's
reproducibility statement:

  * git commit + dirty-file list (or a warning if not a repo)
  * sha256 of every config, every source file under src/ and scripts/,
    every manifest, and the frozen-site exclusion list
  * sha256 + size of every checkpoint and prediction file in the frozen
    method set, grouped by (method, fold, seed)
  * the experimental design actually executed: methods, folds, seeds
  * an explicit statement of what may no longer be changed

After this file exists, any hyperparameter chosen using frozen-test or
external-hospital results is post-hoc and must be labelled exploratory in
the paper. scripts/casg_verify_preds.py re-checks these hashes later.
"""
import _boot, datetime, hashlib, json, os, re, subprocess, sys
from collections import defaultdict

OUT = "freeze.json"
METHODS = {
    "CASG-Lite":        ("runs/casg", "casg_ast_{d}_seed{s}_casg_lite"),
    "CF-only-Physics":  ("runs/casg", "casg_ast_{d}_seed{s}_cf_physics"),
    "ERM-Physics":      ("runs/erm", "erm_ast_{d}_seed{s}_phys"),
    "MixStyle-Physics": ("runs/mixstyle", "mixstyle_ast_{d}_seed{s}_phys"),
}
FOLDS = ["AKGC417L", "Meditron", "LittC2SE", "Litt3200", "smartphone", "none"]
SEEDS = [0, 1, 2]


def sha(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def hash_tree(root, exts):
    out = {}
    for d, _, files in os.walk(root):
        if "__pycache__" in d:
            continue
        for f in sorted(files):
            if f.endswith(exts):
                p = os.path.join(d, f)
                out[p] = sha(p)
    return out


if __name__ == "__main__":
    if os.path.exists(OUT) and "--force" not in sys.argv:
        raise SystemExit(f"{OUT} already exists — the experiment is frozen. "
                         f"Pass --force only if you intend to invalidate that "
                         f"freeze (and say so in the paper).")

    fr = {"frozen_utc": datetime.datetime.utcnow().isoformat() + "Z",
          "design": {"methods": sorted(METHODS), "folds": FOLDS, "seeds": SEEDS,
                     "n_runs_expected": len(METHODS) * len(FOLDS) * len(SEEDS)},
          "git": {"commit": git("rev-parse", "HEAD"),
                  "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                  "dirty": git("status", "--porcelain").splitlines()},
          "code": {}, "configs": {}, "manifests": {}, "artifacts": {},
          "missing": []}
    fr["code"].update(hash_tree("src", (".py",)))
    fr["code"].update(hash_tree("scripts", (".py",)))
    fr["configs"].update(hash_tree("configs", (".yaml", ".yml")))

    from src.utils import load_config
    cfg = load_config("configs/casg_lite.yaml")
    for k in ("train_manifest", "icbhi_manifest", "extra_manifest", "test_manifest"):
        p = str(getattr(cfg.data, k, "") or "")
        if p and os.path.exists(p):
            fr["manifests"][p] = sha(p)
    fr["frozen_sites"] = list(getattr(cfg.data, "dev_exclude_sources", []) or [])

    n_ck = n_pr = 0
    for name, (root, stem_t) in METHODS.items():
        entry = defaultdict(dict)
        for fold in FOLDS:
            for seed in SEEDS:
                stem = stem_t.format(d=fold, s=seed)
                cell = {}
                for tag, p in (("ckpt", os.path.join(root, f"ckpt_{stem}.pt")),
                               ("preds", os.path.join(root, f"preds_{stem}.npz")),
                               ("metrics", os.path.join(root, f"metrics_{stem}.json"))):
                    if os.path.exists(p):
                        cell[tag] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
                        n_ck += int(tag == "ckpt")
                        n_pr += int(tag == "preds")
                    else:
                        fr["missing"].append(p)
                entry[fold][str(seed)] = cell
        fr["artifacts"][name] = dict(entry)

    fr["counts"] = {"checkpoints": n_ck, "prediction_files": n_pr,
                    "missing": len(fr["missing"])}
    fr["statement"] = (
        "CASG-Lite and the three matched baselines were evaluated with three "
        "seeds on five leave-one-device-out folds plus the frozen external "
        "hospital. From this timestamp, no hyperparameter, checkpoint-selection "
        "rule, architecture choice or analysis variant may be selected using "
        "frozen-test or external-hospital results; anything so chosen is "
        "post-hoc and must be reported as exploratory.")

    json.dump(fr, open(OUT, "w"), indent=2, sort_keys=True)
    print(f"FROZEN -> {OUT}")
    print(f"  git commit : {fr['git']['commit'][:12] or 'n/a'}"
          f"{'  (WORKING TREE DIRTY)' if fr['git']['dirty'] else ''}")
    print(f"  code files : {len(fr['code'])}   configs: {len(fr['configs'])}"
          f"   manifests: {len(fr['manifests'])}")
    print(f"  checkpoints: {n_ck}   prediction files: {n_pr}")
    if fr["missing"]:
        print(f"  *** {len(fr['missing'])} expected artifact(s) missing, e.g. "
              f"{fr['missing'][0]}")
    if fr["git"]["dirty"]:
        print("  *** uncommitted changes: commit first so the recorded hash "
              "identifies the exact code used.")