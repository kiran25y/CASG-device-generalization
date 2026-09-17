"""Single training run for ANY method. Leave one device out (or --held_out none
for the frozen external Sungbook test). Same code path for every method so the
comparison is fair.

  python scripts/train_contrastive.py --method contrastive --held_out AKGC417L --seed 0
  python scripts/train_contrastive.py --method dann        --held_out AKGC417L --seed 0
  python scripts/train_contrastive.py --method erm         --held_out none     --seed 0

Oracle row (domain ADAPTATION upper bound — the deployment device IS in
training; reproduces the reference work's Setup 4 and is NOT a DG result):

  python scripts/train_contrastive.py --method dmixstyle --held_out none \
      --target_device smartphone --seed 0
"""
import _boot, argparse, os, json, copy
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.utils import load_config, set_seed
from src.data import manifest as M
from src.data.dataset import make_loader
from src.models import DeviceAgnosticNet, count_params
from src.metrics import compute_metrics, per_device
from src.trainer import Trainer, resolve_device
import torch
torch.multiprocessing.set_sharing_strategy("file_system")


def _dev_filter(df, cfg, what):
    """Protocol v2 4.6 — remove every frozen-site row from development."""
    bad = [str(x).lower() for x in getattr(cfg.data, "dev_exclude_sources", []) or []]
    if not bad or "source" not in df.columns:
        return df
    src = df["source"].astype(str).str.lower()
    mask = src.apply(lambda v: any(b in v for b in bad))
    if int(mask.sum()):
        print(f"[data] frozen-site guard: dropped {int(mask.sum())} {what} rows "
              f"({sorted(src[mask].unique())})")
    return df[~mask].reset_index(drop=True)


def assert_no_frozen_site(cfg, *frames):
    bad = [str(x).lower() for x in getattr(cfg.data, "dev_exclude_sources", []) or []]
    if not bad:
        return
    for f in frames:
        if f is None or "source" not in getattr(f, "columns", []):
            continue
        src = f["source"].astype(str).str.lower()
        hit = src.apply(lambda v: any(b in v for b in bad))
        if bool(hit.any()):
            raise SystemExit(
                f"FROZEN-SITE LEAK: {int(hit.sum())} rows from "
                f"{sorted(src[hit].unique())} reached a development split. "
                f"Protocol v2 section 4.6 forbids this. Aborting.")


def build_pool(cfg):
    seoul = M.add_patient_uid(M.load_manifest(cfg.data.train_manifest))
    icbhi = M.add_patient_uid(M.load_manifest(cfg.data.icbhi_manifest))
    ep = str(getattr(cfg.data, "extra_manifest", "") or "")
    if ep and os.path.exists(ep):
        extra = M.add_patient_uid(M.load_manifest(ep))
        extra = _dev_filter(extra, cfg, 'extra_manifest')
        icbhi = pd.concat([icbhi, extra], ignore_index=True)
        print(f"[data] +extra: {sorted(extra.device.unique())}")
    return seoul, icbhi


def internal_val(df, frac, seed):
    """Patient-disjoint source validation.

    Grouped on cohort_uid (device-INVARIANT physical identity), not
    patient_uid: 'seoul_steth_105' and 'seoul_iphone_105' are the same
    person, and grouping on the source-prefixed key let their stethoscope
    clips sit in train while their phone clips sat in validation.
    Protocol v3 section 4.5 requires physical-patient mapping here.
    """
    if frac <= 0:
        return df, None
    g = M.add_cohort_uid(df) if "cohort_uid" not in df.columns else df
    tr, va = next(GroupShuffleSplit(1, test_size=frac, random_state=seed)
                  .split(g, groups=g["cohort_uid"]))
    return (g.iloc[tr].reset_index(drop=True),
            g.iloc[va].reset_index(drop=True))


def drop_test_cohorts(train_df, test_df, enabled=True):
    """Remove every training row whose PHYSICAL patient appears in the test set.

    This is the fix for the external-test leak. Previously the `--held_out none`
    path concatenated the whole pool into training, which pulled in
    `sungbook_iphone` — 4,124 clips from 95 of the 100 Seongbuk test patients,
    recorded in immediate sequence at the same auscultation sites as the
    stethoscope test clips. `build_loo_folds` already keys exclusion on
    `cohort_uid`; the external-test path did not.
    """
    if not enabled:
        return train_df, []
    tr = M.add_cohort_uid(M.add_patient_uid(train_df))
    te = M.add_cohort_uid(M.add_patient_uid(test_df))
    bad = set(te["cohort_uid"])
    mask = tr["cohort_uid"].isin(bad)
    excluded = sorted(tr.loc[mask, "cohort_uid"].unique().tolist())
    return tr[~mask].reset_index(drop=True), excluded


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_boot.REPO, "configs", "contrastive.yaml"))
    ap.add_argument("--method", default=None)
    ap.add_argument("--held_out", default="AKGC417L")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--tag_suffix", default="", help="appended to the run tag")
    ap.add_argument("--target_device", default=None,
                    help="ORACLE ONLY: target domain for --method dmixstyle")
    ap.add_argument("--set", nargs="*", default=[],
                    help="config overrides, e.g. --set audio.ssp.cutoff_hz=1000")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.method:
        cfg["model"]["method"] = a.method
    if a.epochs is not None:
        cfg["optim"]["epochs"] = a.epochs
    if a.backbone:
        cfg["model"]["backbone"] = a.backbone
    for kv in a.set:                                # dotted-path overrides
        k, v = kv.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        try:
            node[parts[-1]] = json.loads(v)
        except Exception:
            node[parts[-1]] = v
    cfg["seed"] = a.seed
    set_seed(a.seed)

    method = str(cfg.model.method).lower()
    # variant collision guard: cf_physics / casg_lite / casg_full all
    # run under method='casg' and would otherwise share one run tag,
    # overwriting each other's checkpoints, metrics and predictions.
    if method == 'casg':
        _v = str(getattr(cfg.model, 'variant', 'casg_lite')).lower()
        a.tag_suffix = f'_{_v}' + a.tag_suffix
    out_dir = os.path.join(cfg.output_dir, method)
    os.makedirs(out_dir, exist_ok=True)

    seoul, pool = build_pool(cfg)
    seoul = _dev_filter(seoul, cfg, 'train_manifest')
    excl_ext = []
    if a.held_out.lower() != "none":
        fold = M.build_loo_folds(pool, seoul, [a.held_out],
                                 bool(cfg.data.patient_exclusion))[a.held_out]
        train_df, test_df = fold.train, fold.test
        print(f"[data] {method} leave-out {a.held_out}: train={len(train_df)} "
              f"test={len(test_df)} excl={len(fold.excluded_patients)} patients")
    else:
        train_all = pd.concat([seoul, pool], ignore_index=True)
        test_df = M.build_tier3(M.load_manifest(cfg.data.test_manifest),
                                cfg.data.drop_tier3_patient)
        train_df, excl_ext = drop_test_cohorts(
            train_all, test_df, bool(cfg.data.patient_exclusion))
        print(f"[data] {method} train-all + EXTERNAL test: "
              f"train={len(train_df)} (dropped {len(train_all)-len(train_df)} rows "
              f"from {len(excl_ext)} test-cohort patients) test={len(test_df)}")

    train_df, val_df = internal_val(train_df, float(cfg.data.internal_val_frac), a.seed)
    assert_no_frozen_site(cfg, train_df, val_df)   # protocol v3 4.6
    # v3 4.6/22: blinding by ACCESS CAPABILITY, not by instruction.
    _expmode = getattr(cfg, 'experiment', None)
    _mode = (_expmode.get('mode', 'dev') if isinstance(_expmode, dict)
             else getattr(_expmode, 'mode', 'dev')) if _expmode else 'dev'
    if str(_mode) == 'dev':
        from src.casg.access_guard import arm_dev_mode
        _root = str(cfg.data.data_root)
        _outer = {os.path.join(_root, str(q))
                   for q in test_df['filepath']} if test_df is not None else set()
        arm_dev_mode(getattr(cfg.data, 'dev_exclude_sources', []) or [], _outer)
    dmap = {d: i for i, d in enumerate(sorted(train_df["device"].unique()))}
    inv = {i: d for d, i in dmap.items()}
    root = cfg.data.data_root

    tgt_id = None
    if method == "dmixstyle":
        if not a.target_device:
            raise SystemExit("--method dmixstyle requires --target_device "
                             "(it is a domain-ADAPTATION oracle, not a DG method)")
        if a.target_device not in dmap:
            raise SystemExit(f"--target_device {a.target_device!r} is not in the "
                             f"training set (devices: {sorted(dmap)})")
        tgt_id = dmap[a.target_device]
        print(f"[oracle] directional MixStyle -> target '{a.target_device}' "
              f"(id {tgt_id}). This row is DA, not DG — label it as such.")

    if method == "casg":

        from src.casg.model import CASGNet

        model = CASGNet(cfg, n_devices=max(2, len(dmap)))

        model.verify_split()

        print(f"[casg] split_block={model.split_block} taps={model.taps} split-equivalence OK", flush=True)


    elif str(getattr(cfg.model, "method", "")).lower() in ("acpl", "cfsc"):
        from src.models.acpl import ACPLNet
        from src.data.augment import DeviceSimulator
        _td = DeviceSimulator(cfg.augment.device_sim,
                              int(cfg.audio.sample_rate)).theta_dim()
        model = ACPLNet(cfg, n_devices=max(2, len(dmap)), theta_dim=_td)
        print(f"[acpl] theta_dim={_td} lambda_acq/cf from cfg.loss", flush=True)
    else:
        model = DeviceAgnosticNet(cfg, n_devices=max(2, len(dmap)))
    print(f"[model] method={method} backbone={cfg.model.backbone} "
          f"params={count_params(model):,} mixstyle={model.mixstyle}", flush=True)

    tr = Trainer(model, cfg, train_df=train_df, device=resolve_device(cfg),
                 target_device_id=tgt_id)
    contrastive = method in ("contrastive", "erm2view", "cfc")  # need 2 views
    tl = make_loader(train_df, cfg, dmap, root, train=True, contrastive=contrastive)
    vl = make_loader(val_df, cfg, dmap, root, train=False) if val_df is not None else None

    bk = str(cfg.model.backbone)
    tag = f"{method}_{bk}_{a.held_out}_seed{a.seed}{a.tag_suffix}"
    resume_path = os.path.join(out_dir, f"resume_{tag}.pt")
    tr.fit(tl, vl, resume_path=resume_path)

    # ---- test -------------------------------------------------------------
    test_loader = make_loader(test_df, cfg, dmap, root, train=False)
    raw = tr.predict(test_loader)                      # single forward pass
    # align metadata to prediction order via the dataset row index rather than
    # assuming the loader preserved dataframe order
    test_meta = test_df.iloc[raw["index"].astype(int)].reset_index(drop=True)
    p4 = raw["prob4"].argmax(1)
    p2 = raw["prob2"].argmax(1)
    m = compute_metrics(raw["y4"], p4, raw["y2"], p2, raw["prob4"], raw["prob2"])
    m["_per_device"] = per_device(test_meta["device"].to_numpy(),
                                  raw["y4"], p4, raw["y2"], p2,
                                  raw["prob4"], raw["prob2"])

    torch.save({"state_dict": tr.model.state_dict(), "config": cfg.to_dict(),
                "device_map": dmap, "best_epoch": fit_info.get("best_epoch", -1),
                "selection": "raw"}, os.path.join(out_dir, f"ckpt_{tag}.pt"))
    # ---- v2: EMA checkpoint + predictions (secondary; raw stays primary) ----
    ema_sd = fit_info.get("ema_state_dict")
    if ema_sd is not None:
        torch.save({"state_dict": ema_sd, "config": cfg.to_dict(), "device_map": dmap,
                    "best_epoch": fit_info.get("ema_best_epoch", -1), "selection": "ema",
                    "ema_updates": fit_info.get("ema_updates"), "ema_beta": fit_info.get("ema_beta")},
                   os.path.join(out_dir, f"ckpt_{tag}_ema.pt"))
        _cur = copy.deepcopy(tr.model.state_dict()); tr.model.load_state_dict(ema_sd)
        raw_ema = tr.predict(test_loader); tr.model.load_state_dict(_cur)
        meta_ema = test_df.iloc[raw_ema["index"].astype(int)].reset_index(drop=True)
        np.savez_compressed(
            os.path.join(out_dir, f"preds_{tag}_ema.npz"),
            prob4=raw_ema["prob4"], prob2=raw_ema["prob2"], y4=raw_ema["y4"], y2=raw_ema["y2"],
            device=meta_ema["device"].to_numpy().astype(str),
            patient=meta_ema["patient_id"].to_numpy().astype(str),
            cohort=meta_ema["cohort_uid"].to_numpy().astype(str) if "cohort_uid" in meta_ema else meta_ema["patient_id"].to_numpy().astype(str),
            source=meta_ema["source"].to_numpy().astype(str),
            uid=(meta_ema["sample_uid"] if "sample_uid" in meta_ema else meta_ema["filepath"]).to_numpy().astype(str),
            method=method, backbone=bk, held_out=a.held_out, seed=a.seed, selection="ema")

    # per-clip dump: every downstream metric is recomputed from this, so a new
    # metric never costs a retrain.
    np.savez_compressed(
        os.path.join(out_dir, f"preds_{tag}.npz"),
        prob4=raw["prob4"], prob2=raw["prob2"], y4=raw["y4"], y2=raw["y2"],
        device=test_meta["device"].to_numpy().astype(str),
        patient=test_meta["patient_id"].to_numpy().astype(str),
        cohort=test_meta["cohort_uid"].to_numpy().astype(str) if "cohort_uid" in test_meta else test_meta["patient_id"].to_numpy().astype(str),
        source=test_meta["source"].to_numpy().astype(str),
        uid=(test_meta["sample_uid"] if "sample_uid" in test_meta else test_meta["filepath"]).to_numpy().astype(str),
        method=method, backbone=bk, held_out=a.held_out, seed=a.seed, selection="raw")

    flat = {k: v for k, v in m.items() if not k.startswith("_")}
    flat.update({"method": method, "backbone": bk, "held_out": a.held_out,
                 "seed": a.seed,
                 "best_val_score": float(fit_info["best_val"]),
                 "select_metric": str(cfg.optim.select_metric),
                 "best_epoch": fit_info.get("best_epoch", -1),
                 "ema_best_val_score": fit_info.get("ema_best_val"),
                 "ema_best_epoch": fit_info.get("ema_best_epoch", -1),
                 "ema_updates": fit_info.get("ema_updates"), "ema_beta": fit_info.get("ema_beta"),
                 "ema_start_epoch": fit_info.get("ema_start_epoch"),
                 "variant": str(getattr(cfg.model, "variant", "")),
                 "val_curve": fit_info.get("val_curve", []),
                 "diagnostics": fit_info.get("diagnostics", [])})
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w") as fh:
        json.dump(flat, fh, indent=2)

    print("\n==================== RESULT ====================")
    print(f"{method} | held-out {a.held_out}: "
          f"AUROC2={m['auroc2']:.4f} AUPRC2={m['auprc2']:.4f} F1_2={m['f1_2']:.4f} "
          f"| icbhi4={m['icbhi4']:.4f} Se4={m['se4']:.4f} Sp4={m['sp4']:.4f} "
          f"uar4={m['uar4']:.4f} | ECE={m['ece2']:.4f}")
    if "_per_device" in m:
        for d, dm in m["_per_device"].items():
            if not d.startswith("_"):
                print(f"  {d:30s} n={dm['n']:5d} AUROC2={dm['auroc2']:.4f} "
                      f"Se2={dm['se2']:.4f} Sp2={dm['sp2']:.4f}")
        print(f"  WORST-device AUROC2={m['_per_device']['_worst_auroc2']:.4f} "
              f"F1_2={m['_per_device']['_worst_f1_2']:.4f}")

    # Scope: hospital (Seoul + Sungbook + iPhone) and ICBHI only. The former
    # optional SPRSound OOD pass was removed — it contributed label4 = -1 rows
    # from a different population and confounded the device comparison.
