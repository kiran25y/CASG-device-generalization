"""Prediction-file integrity gate (audit item 4). Run BEFORE any comparison.

    python scripts/casg_verify_preds.py

Ensembling and paired bootstraps are only valid if every prediction file
for a given fold describes the SAME clips in the SAME order. This asserts,
for each fold, across all four methods and three seeds:

  * identical number of clips
  * identical binary and four-class label vectors, element-wise
  * identical patient-id vector, element-wise
  * identical source-file vector when present
  * probability rows that sum to 1 and contain no NaN

and, if freeze.json exists, that every file still matches its frozen
sha256. Any violation aborts: a silent row-order mismatch would corrupt
both the ensemble and every paired test computed from it.
"""
import _boot, hashlib, json, os, sys
import numpy as np

FOLDS = ["AKGC417L", "Meditron", "LittC2SE", "Litt3200", "smartphone", "none"]
SEEDS = [0, 1, 2]
SPECS = {"CASG-Lite":        "runs/casg/preds_casg_ast_{d}_seed{s}_casg_lite.npz",
         "CF-only-Physics":  "runs/casg/preds_casg_ast_{d}_seed{s}_cf_physics.npz",
         "ERM-Physics":      "runs/erm/preds_erm_ast_{d}_seed{s}_phys.npz",
         "MixStyle-Physics": "runs/mixstyle/preds_mixstyle_ast_{d}_seed{s}_phys.npz"}


def sha(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fail(msg):
    print(f"\n*** INTEGRITY FAILURE: {msg}")
    sys.exit(1)


if __name__ == "__main__":
    frozen = json.load(open("freeze.json")) if os.path.exists("freeze.json") else None
    print(f"freeze.json: {'loaded' if frozen else 'absent (hash check skipped)'}\n")
    print(f"{'fold':<12}{'files':>7}{'clips':>8}{'patients':>10}  labels/order")
    total = 0
    for d in FOLDS:
        ref = None
        n_files = 0
        for m, tmpl in SPECS.items():
            for s in SEEDS:
                p = tmpl.format(d=d, s=s)
                if not os.path.exists(p):
                    continue
                z = np.load(p, allow_pickle=True)
                n_files += 1
                total += 1
                y2, y4 = z["y2"], z["y4"]
                pat = z["patient"].astype(str) if "patient" in z else None
                src = z["source"].astype(str) if "source" in z else None
                p2 = z["prob2"]
                if not np.isfinite(p2).all():
                    fail(f"{p}: non-finite probabilities")
                if not np.allclose(p2.sum(1), 1.0, atol=1e-4):
                    fail(f"{p}: prob2 rows do not sum to 1")
                if ref is None:
                    ref = (p, y2, y4, pat, src)
                    continue
                rp, ry2, ry4, rpat, rsrc = ref
                if len(y2) != len(ry2):
                    fail(f"{p}: {len(y2)} clips vs {len(ry2)} in {rp}")
                if not np.array_equal(y2, ry2):
                    fail(f"{p}: binary labels differ from {rp} (row order?)")
                if not np.array_equal(y4, ry4):
                    fail(f"{p}: four-class labels differ from {rp}")
                if pat is not None and rpat is not None and not np.array_equal(pat, rpat):
                    fail(f"{p}: patient ids differ from {rp}")
                if src is not None and rsrc is not None and not np.array_equal(src, rsrc):
                    fail(f"{p}: source files differ from {rp}")
                if frozen:
                    for meth, tree in frozen.get("artifacts", {}).items():
                        cell = tree.get(d, {}).get(str(s), {})
                        rec = cell.get("preds")
                        if rec and os.path.basename(p) in tmpl.format(d=d, s=s):
                            if meth == m and rec["sha256"] != sha(p):
                                fail(f"{p}: sha256 differs from freeze.json — "
                                     f"the file changed after the freeze")
        if ref:
            _, y2, _, pat, _ = ref
            npat = len(np.unique(pat)) if pat is not None else -1
            print(f"{d:<12}{n_files:>7}{len(y2):>8}{npat:>10}  aligned")
    print(f"\nALL PREDICTION FILES ALIGNED ({total} files). Ensembling and "
          f"paired bootstraps are valid.")