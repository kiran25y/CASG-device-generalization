"""Prove the frozen-site fix on the real manifests (review B1). Repo root:

    python scripts/casg_topology_check.py

For every LODO fold it rebuilds the split exactly as training does and
asserts: no Hospital-B row in train or validation; the smartphone target is
Seoul-only; patient sets are disjoint. Then it reports the frozen external
arms. Fails loudly; safe to run while other jobs train (CPU, no GPU).
"""
import _boot
import pandas as pd
from src.utils import load_config
from src.data import manifest as M
from scripts.train_contrastive import _dev_filter, assert_no_frozen_site, internal_val

cfg = load_config("configs/casg.yaml")
bad = [str(x).lower() for x in getattr(cfg.data, "dev_exclude_sources", []) or []]
assert bad, "data.dev_exclude_sources is empty — run scripts/casg_fix_topology.py"
print(f"frozen source tags: {bad}\n")

seoul = _dev_filter(M.add_patient_uid(M.load_manifest(cfg.data.train_manifest)),
                    cfg, "train_manifest")
icbhi = M.add_patient_uid(M.load_manifest(cfg.data.icbhi_manifest))
extra = _dev_filter(M.add_patient_uid(M.load_manifest(cfg.data.extra_manifest)),
                    cfg, "extra_manifest")
pool = pd.concat([icbhi, extra], ignore_index=True)

def frozen_rows(df):
    if df is None or "source" not in df.columns:
        return 0
    src = df["source"].astype(str).str.lower()
    return int(src.apply(lambda v: any(b in v for b in bad)).sum())

print(f"{'fold':<12} {'train':>7} {'val':>6} {'test':>6} {'test pat':>9} "
      f"{'HospB in dev':>13}")
for dev in list(cfg.data.loo_devices):
    fold = M.build_loo_folds(pool, seoul, [dev],
                             bool(cfg.data.patient_exclusion))[dev]
    tr, va = internal_val(fold.train, float(cfg.data.internal_val_frac), 0)
    assert_no_frozen_site(cfg, tr, va)
    leak = frozen_rows(tr) + frozen_rows(va)
    assert leak == 0, f"{dev}: {leak} Hospital-B rows in development"
    ptr = set(M.add_cohort_uid(tr)["cohort_uid"])
    pte = set(M.add_cohort_uid(fold.test)["cohort_uid"])
    assert not (ptr & pte), f"{dev}: patient overlap {sorted(ptr & pte)[:3]}"
    if dev == "smartphone":
        srcs = sorted(fold.test["source"].astype(str).str.lower().unique())
        assert all("sungbook" not in s for s in srcs), \
            f"smartphone target still contains Hospital B: {srcs}"
        print(f"{'':<12} smartphone target sources: {srcs}")
    print(f"{dev:<12} {len(tr):>7} {len(va):>6} {len(fold.test):>6} "
          f"{len(pte):>9} {leak:>13}")

ext_steth = M.build_tier3(M.load_manifest(cfg.data.test_manifest),
                          cfg.data.drop_tier3_patient)
raw_extra = M.load_manifest(cfg.data.extra_manifest)
ext_phone = raw_extra[raw_extra["source"].astype(str).str.lower()
                      .str.contains("sungbook")]
print(f"\nFROZEN external arms (evaluated once, after the config is locked):")
print(f"  Hospital-B stethoscope : {len(ext_steth):5d} clips")
print(f"  Hospital-B smartphone  : {len(ext_phone):5d} clips "
      f"({ext_phone.patient_id.nunique()} patients)")
print("\nTOPOLOGY CHECK PASSED — no Hospital-B row reaches development.")
