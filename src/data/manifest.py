"""Manifest handling + the three-tier evaluation protocol (§2, §3).

Implements, verbatim to the pipeline document:
  * §2.3  source-prefixed patient IDs (namespace-collision fix).
  * §2.2  LOO folds with per-fold patient exclusion (leak fix).
  * §3    Tier 1 GroupShuffleSplit, Tier 2 LOO, Tier 3 frozen external.
  * §2.5  Tier 3 outlier-patient drop.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REQUIRED_COLS = ["source", "patient_id", "filepath", "cycle_id", "device",
                 "hospital", "label4", "label2", "sample_rate", "duration"]


def load_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df


def add_patient_uid(df: pd.DataFrame) -> pd.DataFrame:
    """§2.3: prefix every patient_id by its source BEFORE any join/split.
    Prevents silent merge of unrelated Seoul(4-10080) and ICBHI(101-226) IDs."""
    df = df.copy()
    df["patient_uid"] = df["source"].astype(str) + "_" + df["patient_id"].astype(str)
    return df


_DEVICE_SUFFIXES = ("_iphone", "_smartphone", "_phone", "_steth", "_stethoscope")


def add_cohort_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Physical-patient key that is the SAME across capture devices at one site.
    e.g. source 'seoul' and 'seoul_iphone' both map to cohort 'seoul', so patient
    105 recorded on a stethoscope AND a phone shares one cohort_uid 'seoul_105'.
    This is what patient-exclusion must key on when a device is held out, or the
    same patient leaks between the stethoscope (train) and phone (test) sets."""
    df = df.copy()
    def cohort(src):
        src = str(src)
        for suf in _DEVICE_SUFFIXES:
            if src.endswith(suf):
                return src[: -len(suf)]
        return src
    df["cohort_uid"] = df["source"].map(cohort) + "_" + df["patient_id"].astype(str)
    return df


def build_tier1(seoul_df: pd.DataFrame, val_fraction: float,
                seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tier 1 (§3): in-domain Seoul val, GroupShuffleSplit by patient_uid."""
    df = add_patient_uid(seoul_df)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr_idx, va_idx = next(gss.split(df, groups=df["patient_uid"]))
    train, val = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
    assert set(train["patient_uid"]).isdisjoint(set(val["patient_uid"])), \
        "Tier 1 split is not patient-disjoint"
    return train.reset_index(drop=True), val.reset_index(drop=True)


def detect_cross_device_patients(icbhi_df: pd.DataFrame) -> pd.DataFrame:
    """§2.2 audit: patients recorded on >1 device (the leak source)."""
    df = add_patient_uid(icbhi_df)
    g = df.groupby("patient_uid")["device"].nunique()
    leaked = g[g > 1].index.tolist()
    return df[df["patient_uid"].isin(leaked)][["patient_uid", "device", "cycle_id"]]


@dataclass
class LOOFold:
    held_out_device: str
    train: pd.DataFrame
    test: pd.DataFrame
    excluded_patients: List[str] = field(default_factory=list)


def build_loo_folds(icbhi_df: pd.DataFrame, seoul_df: pd.DataFrame,
                    devices: List[str], patient_exclusion: bool = True
                    ) -> Dict[str, LOOFold]:
    """Tier 2 (§3): leave-one-device-out folds.

    For held-out device D: test = ICBHI rows on D; train = Seoul + ICBHI rows
    on the other devices. When `patient_exclusion` (§2.2), any training cycle
    whose patient_uid also appears in the test set is dropped -- the mandatory
    leak fix that removes cross-device patients from that fold's training set.
    """
    icbhi = add_cohort_uid(add_patient_uid(icbhi_df))
    seoul = add_cohort_uid(add_patient_uid(seoul_df))
    folds: Dict[str, LOOFold] = {}
    for dev in devices:
        test = icbhi[icbhi["device"] == dev].copy()
        pool_other = icbhi[icbhi["device"] != dev].copy()
        excluded: List[str] = []
        if patient_exclusion:
            # exclude by PHYSICAL patient (cohort_uid) so the same person recorded
            # on another device at the same site cannot leak into training
            test_patients = set(test["cohort_uid"])
            def drop(dfx):
                mask = dfx["cohort_uid"].isin(test_patients)
                return dfx[~mask].copy(), sorted(dfx.loc[mask, "cohort_uid"].unique().tolist())
            pool_other, ex1 = drop(pool_other)
            seoul_f, ex2 = drop(seoul)
            excluded = sorted(set(ex1) | set(ex2))
        else:
            seoul_f = seoul
        train = pd.concat([seoul_f, pool_other], ignore_index=True)
        # invariant: no PHYSICAL patient in both train and test of this fold
        assert set(train["cohort_uid"]).isdisjoint(set(test["cohort_uid"])) or not patient_exclusion
        folds[dev] = LOOFold(dev, train.reset_index(drop=True),
                             test.reset_index(drop=True), excluded)
    return folds


def build_tier3(sungbook_df: pd.DataFrame, drop_patient: str | None) -> pd.DataFrame:
    """Tier 3 (§3, §2.5): frozen external hospital, outlier patient dropped."""
    df = add_patient_uid(sungbook_df)
    if drop_patient:
        df = df[df["patient_uid"] != drop_patient].copy()
    return df.reset_index(drop=True)


def device_to_id(df: pd.DataFrame) -> Dict[str, int]:
    """Stable device->int map for the DANN device head (§4.1)."""
    return {d: i for i, d in enumerate(sorted(df["device"].unique()))}