"""Build manifests. Two cohorts only (all paths in configs/contrastive.yaml):

  Train  : Seoul stethoscope + ICBHI (4 contact stethoscopes)   [4-class]
  Held-out unseen device : iPhone (Seoul+Sungbook) -> 'smartphone'
  External test : Sungbook stethoscope (frozen)

7 capture devices, 5 leave-one-device-out folds. SPRSound and HF-Lung are
deliberately out of scope — see the note in configs/contrastive.yaml.

Run:  python scripts/build_manifests.py                  # everything
      python scripts/build_manifests.py --only icbhi     # just re-slice ICBHI

--only matters when a container has lost the generated ICBHI cycle clips but
the other manifests are already correct (e.g. after scripts/remap_paths.py):
rebuilding everything would re-derive labels and could change validated row
counts.
"""
import _boot, argparse, os
import pandas as pd
from src.utils import load_config
from src.data.adapter import build_unified_manifests, convert_named_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_boot.REPO, "configs", "contrastive.yaml"))
    ap.add_argument("--no_probe", action="store_true")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=["hospital", "icbhi", "sungbook", "iphone"],
                    help="rebuild only these; default is all")
    a = ap.parse_args()
    cfg = load_config(a.config); s = cfg.get("sources", {}); out = cfg.data.data_root
    probe = not a.no_probe; os.makedirs(out, exist_ok=True)
    want = (lambda k: True) if not a.only else (lambda k: k in a.only)
    G = lambda k: s.get(k, "")
    if a.only:
        print(f"[build] --only {a.only}: other manifests left untouched")

    # core: Seoul(train) + ICBHI(train) + Sungbook(external test)
    build_unified_manifests(out,
                            G("hospital_csv") if want("hospital") else "",
                            G("hospital_dir"),
                            G("icbhi_csv") if want("icbhi") else "",
                            G("icbhi_dir"),
                            G("sungbook_csv") if want("sungbook") else "",
                            G("sungbook_dir"), probe)

    # HELD-OUT smartphone = iPhone Seoul + Sungbook  -> extra_manifest.csv
    parts = []
    if want("iphone"):
        if G("iphone_seoul_dir") and os.path.isdir(G("iphone_seoul_dir")):
            parts.append(convert_named_dir(G("iphone_seoul_dir"), "seoul_iphone", "smartphone", probe))
        if G("iphone_sungbook_dir") and os.path.isdir(G("iphone_sungbook_dir")):
            parts.append(convert_named_dir(G("iphone_sungbook_dir"), "sungbook_iphone", "smartphone", probe))
    parts = [p for p in parts if p is not None and len(p) > 0]
    if parts:
        extra = pd.concat(parts, ignore_index=True)
        extra.to_csv(os.path.join(out, "extra_manifest.csv"), index=False)
        print(f"\n[build] extra_manifest.csv (HELD-OUT smartphone): {len(extra)} clips, "
              f"labels={extra.label4.value_counts().to_dict()}, patients={extra.patient_id.nunique()}")

    print(f"\n[build] done. Train stethoscopes, test unseen phone:\n"
          f"  python scripts/train_contrastive.py --method contrastive --held_out smartphone --seed 0")