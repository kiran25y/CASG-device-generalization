"""Adapter: v1 / real CSVs -> the final pipeline's unified manifests.

Handles the real-world schema variety seen across cohorts:
  * Seoul hospital  : path,label,patient_id,device,hospital,split      (string label)
  * ICBHI per-cycle : path,label,patient_id,device,begin,end           (string label, slicing)
  * Sungbook test   : filepath,basename,label_4cls,label_2cls,sample_rate,duration_sec
                      (INTEGER labels, no string label, no patient_id column)

Output columns (== src/data/manifest.REQUIRED_COLS):
  source, patient_id, filepath, cycle_id, device, hospital,
  label4, label2, sample_rate, duration, split
"""
from __future__ import annotations
import os
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

CLASS4 = {"normal": 0, "crackle": 1, "wheeze": 2, "both": 3, "abnormal": 1}
CLASS2 = {"normal": 0, "crackle": 1, "wheeze": 1, "both": 1, "abnormal": 1}

# order matters: prefer full paths over bare basenames
FILE_ALIASES = ("filepath", "file_path", "path", "audio_path", "wav_path",
                "filename", "file_id", "file", "wav", "audio", "basename")
LABEL_STR_ALIASES = ("label", "class", "target", "y", "class_name_4cls")
LABEL4_INT_ALIASES = ("label4", "label_4cls", "label_4", "y4")
LABEL2_INT_ALIASES = ("label2", "label_2cls", "label_2", "y2")
DEVICE_ALIASES = ("device", "device_name", "microphone", "stethoscope")
PID_ALIASES = ("patient_id", "patient", "pid", "subject_id", "subject", "id")
SR_ALIASES = ("sample_rate", "sr", "samplerate")
DUR_ALIASES = ("duration", "duration_sec", "dur", "length_sec")
BEGIN_ALIASES = ("begin", "start", "start_sec", "begin_sec")
END_ALIASES = ("end", "stop", "end_sec")


def _find(df: pd.DataFrame, aliases) -> Optional[str]:
    for c in aliases:
        if c in df.columns:
            return c
    return None


def _to_label4(v) -> int:
    """Accept a string label OR an already-numeric 4-class id."""
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v).strip()
    if s.lower() in CLASS4:
        return CLASS4[s.lower()]
    return int(float(s))  # numeric string like "0"


def _labels_from_row(row, str_col, i4_col, i2_col) -> Tuple[int, int]:
    if str_col is not None and pd.notna(row[str_col]) and \
            str(row[str_col]).strip().lower() in CLASS4:
        k = str(row[str_col]).strip().lower()
        return CLASS4[k], CLASS2[k]
    if i4_col is not None and pd.notna(row[i4_col]):
        l4 = _to_label4(row[i4_col])
        if i2_col is not None and pd.notna(row[i2_col]):
            l2 = int(row[i2_col])
        else:
            l2 = 0 if l4 == 0 else 1
        return l4, l2
    raise KeyError("no usable label column (need a string 'label' or integer "
                   "'label_4cls'/'label4')")


def _patient_id(row, pid_col, filename: str) -> str:
    if pid_col is not None and pd.notna(row[pid_col]):
        return str(row[pid_col])
    stem = Path(str(filename)).name
    # Sungbook/Seoul basenames look like "50013)1-1-N.wav" -> patient 50013
    for sep in (")", "_", "-", "."):
        if sep in stem:
            head = stem.split(sep)[0]
            if head:
                return head
    return Path(str(filename)).stem


def _abspath(audio_dir: str, name: str) -> str:
    """Resolve a manifest path. Relative -> join audio_dir. Absolute -> use as
    is, BUT if that absolute path doesn't exist and audio_dir is given, fall
    back to audio_dir/<basename> (handles CSVs carrying stale paths from another
    machine, e.g. ICBHI labels.per_cycle.csv pointing at /root/.ssh/...)."""
    p = Path(str(name))
    if p.is_absolute():
        try:
            exists = p.exists()
        except OSError:          # e.g. PermissionError on a protected stale path
            exists = False
        if exists or not audio_dir:
            return str(p)
        return str(Path(audio_dir) / p.name)
    return str(Path(audio_dir) / name)


_INDEX_CACHE = {}


def _basename_index(audio_dir: str) -> dict:
    """Walk audio_dir once and map basename -> full path (first match wins).
    Lets us find files even when the CSV path is stale or the WAVs sit in a
    subfolder (e.g. ICBHI's audio_and_txt_files/)."""
    if audio_dir in _INDEX_CACHE:
        return _INDEX_CACHE[audio_dir]
    idx = {}
    if audio_dir and os.path.isdir(audio_dir):
        for dp, dns, fns in os.walk(audio_dir):
            dns[:] = [d for d in dns if d != "__MACOSX"]
            for f in fns:
                if f.startswith("._"):          # AppleDouble resource fork
                    continue
                if f.lower().endswith(".wav") and f not in idx:
                    idx[f] = os.path.join(dp, f)
    _INDEX_CACHE[audio_dir] = idx
    return idx


def _locate(audio_dir: str, name: str) -> str:
    """Best-effort absolute path to an audio file."""
    direct = _abspath(audio_dir, name)
    if os.path.exists(direct):
        return direct
    if audio_dir:
        hit = _basename_index(audio_dir).get(Path(str(name)).name)
        if hit:
            return hit
    return direct


def _wav_info(path: str) -> Tuple[int, float]:
    try:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate(); n = wf.getnframes()
            return sr, (n / sr if sr else float("nan"))
    except Exception:
        return 0, float("nan")


def _pcm_to_float(raw: bytes, sampwidth: int) -> np.ndarray:
    """Decode PCM bytes of any common bit depth (8/16/24/32-bit) to float32."""
    if sampwidth == 1:                       # unsigned 8-bit
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sampwidth == 2:                       # signed 16-bit
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sampwidth == 3:                       # signed 24-bit (no native numpy dtype)
        a = np.frombuffer(raw, dtype=np.uint8)
        a = a[: (len(a) // 3) * 3].reshape(-1, 3).astype(np.int32)
        val = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
        val = np.where(val >= (1 << 23), val - (1 << 24), val)
        return val.astype(np.float32) / (1 << 23)
    if sampwidth == 4:                       # signed 32-bit
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / (1 << 31)
    raise ValueError(f"unsupported sample width: {sampwidth} bytes")


def _read_wav(path: str) -> Tuple[np.ndarray, int]:
    # Prefer soundfile (handles every WAV/FLAC subtype) when available.
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if getattr(x, "ndim", 1) > 1:
            x = x.mean(axis=1)
        return np.asarray(x, dtype=np.float32), int(sr)
    except Exception:
        pass
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate(); n = wf.getnframes()
        ch = wf.getnchannels(); sw = wf.getsampwidth()
        raw = wf.readframes(n)
    x = _pcm_to_float(raw, sw)
    if ch > 1:
        x = x[: (len(x) // ch) * ch].reshape(-1, ch).mean(axis=1)
    return x, sr


def _write_wav(path: str, x: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def convert_cohort(csv_path: str, audio_dir: str, source: str,
                   default_device: str, hospital_name: str,
                   probe_audio: bool = True, is_icbhi: bool = False,
                   out_dir: Optional[str] = None, slice_cycles: bool = True,
                   default_sr: int = 16000) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    fcol = _find(df, FILE_ALIASES)
    if fcol is None:
        raise KeyError(f"{csv_path}: no file/path column; got {list(df.columns)}")
    str_col = _find(df, LABEL_STR_ALIASES)
    i4_col = _find(df, LABEL4_INT_ALIASES)
    i2_col = _find(df, LABEL2_INT_ALIASES)
    if str_col is None and i4_col is None:
        raise KeyError(f"{csv_path}: no label column (string 'label' or int "
                       f"'label_4cls'); got {list(df.columns)}")
    dev_col = _find(df, DEVICE_ALIASES)
    pid_col = _find(df, PID_ALIASES)
    hosp_col = _find(df, ("hospital", "site", "cohort"))
    sr_col = _find(df, SR_ALIASES)
    dur_col = _find(df, DUR_ALIASES)
    b_col = _find(df, BEGIN_ALIASES) if is_icbhi else None
    e_col = _find(df, END_ALIASES) if is_icbhi else None
    do_slice = bool(is_icbhi and slice_cycles and b_col and e_col and out_dir)

    rows, counter, missing = [], {}, 0
    for _, r in df.iterrows():
        src_path = _locate(audio_dir, r[fcol])
        l4, l2 = _labels_from_row(r, str_col, i4_col, i2_col)
        patient = _patient_id(r, pid_col, r[fcol])
        device = str(r[dev_col]) if dev_col else default_device
        hospital = str(r[hosp_col]) if hosp_col else hospital_name
        cid = counter.get(patient, 0); counter[patient] = cid + 1

        if do_slice:
            if not os.path.exists(src_path):
                missing += 1
                continue
            try:
                x, sr = _read_wav(src_path)
            except Exception as exc:
                missing += 1
                if missing <= 3:
                    print(f"[adapter] skip unreadable {src_path}: {exc}")
                continue
            b = int(float(r[b_col]) * sr); e = int(float(r[e_col]) * sr)
            clip = x[max(0, b):max(b + 1, e)]
            fpath = os.path.abspath(os.path.join(out_dir, "audio", "icbhi",
                                    f"{patient}_{cid}_{device}.wav"))
            _write_wav(fpath, clip, sr)
            dur = len(clip) / sr if sr else float("nan")
        else:
            fpath = src_path
            if sr_col is not None and pd.notna(r[sr_col]):
                sr = int(r[sr_col])
                dur = float(r[dur_col]) if (dur_col and pd.notna(r[dur_col])) else \
                    (_wav_info(fpath)[1] if probe_audio else float("nan"))
            elif probe_audio:
                sr, dur = _wav_info(fpath)
            else:
                sr, dur = default_sr, float("nan")

        rows.append({"source": source, "patient_id": patient, "filepath": fpath,
                     "cycle_id": cid, "device": device, "hospital": hospital,
                     "label4": int(l4), "label2": int(l2),
                     "sample_rate": sr or default_sr, "duration": dur,
                     "split": "train"})
    if missing:
        print(f"[adapter] WARNING: {missing} {source} audio files not found "
              f"(check audio_dir); those rows were skipped.")
    return pd.DataFrame(rows)


def build_unified_manifests(out_dir: str, hospital_csv: str, hospital_dir: str,
                            icbhi_csv: str, icbhi_dir: str,
                            sungbook_csv: str, sungbook_dir: str,
                            probe_audio: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if hospital_csv and os.path.exists(hospital_csv):
        h = convert_cohort(hospital_csv, hospital_dir or "", "seoul",
                           "HospitalStethoscope", "seoul_combined", probe_audio)
        h.to_csv(os.path.join(out_dir, "train_manifest.csv"), index=False)
        print(f"[adapter] train_manifest.csv: {len(h)} rows, "
              f"{h['patient_id'].nunique()} patients, devices={sorted(h.device.unique())}")

    if icbhi_csv and os.path.exists(icbhi_csv):
        ic = convert_cohort(icbhi_csv, icbhi_dir or "", "icbhi", "ICBHI", "icbhi",
                            probe_audio, is_icbhi=True, out_dir=out_dir)
        if len(ic) == 0:
            raise FileNotFoundError(
                f"No ICBHI audio was found under icbhi_dir='{icbhi_dir}'. The CSV "
                f"paths are stale, so the WAVs are located by filename under that "
                f"directory. Point --icbhi_dir at the folder that actually contains "
                f"the .wav files (searched recursively). Locate them with:\n"
                f"    find /workspace -name '101_1b1_Al_sc_Meditron.wav'")
        ic.to_csv(os.path.join(out_dir, "icbhi_manifest.csv"), index=False)
        print(f"[adapter] icbhi_manifest.csv: {len(ic)} cycles, "
              f"devices={sorted(ic.device.unique())}, "
              f"labels={ic.label4.value_counts().to_dict()}")

    if sungbook_csv and os.path.exists(sungbook_csv):
        s = convert_cohort(sungbook_csv, sungbook_dir or "", "sungbook",
                           "HospitalStethoscope_sungbook", "sungbook", probe_audio)
        s.to_csv(os.path.join(out_dir, "test_manifest.csv"), index=False)
        print(f"[adapter] test_manifest.csv: {len(s)} rows, "
              f"{s['patient_id'].nunique()} patients")

    print(f"[adapter] done -> {out_dir}")


# ===========================================================================
# Extra device datasets (binary or disease labels, NO ICBHI 4-class labels).
# These enrich device diversity for the 2-class task + device-invariance.
# label4 is set to -1 (a "no 4-class label" mask); label2 is derived.
# ===========================================================================
import re as _re

DISEASE_NORMAL = {"healthy", "normal", "control", "hc"}


def convert_extra_cohort(csv_path: str, audio_dir: str, source: str,
                         default_device: str, probe_audio: bool = True,
                         disease: bool = False, default_sr: int = 16000) -> pd.DataFrame:
    """Binary (normal/abnormal) or disease-labelled cohort -> unified schema
    with label4 = -1 (masked) and label2 in {0,1}."""
    df = pd.read_csv(csv_path)
    fcol = _find(df, FILE_ALIASES)
    if fcol is None:
        raise KeyError(f"{csv_path}: no file column; got {list(df.columns)}")
    lblcol = _find(df, LABEL_STR_ALIASES + ("label",))
    devcol = _find(df, DEVICE_ALIASES)
    pidcol = _find(df, PID_ALIASES)
    srcol = _find(df, SR_ALIASES); durcol = _find(df, DUR_ALIASES)
    rows = []
    for _, r in df.iterrows():
        fpath = _locate(audio_dir, r[fcol])
        lab = str(r[lblcol]).strip().lower()
        if disease:
            l2 = 0 if lab in DISEASE_NORMAL else 1
        else:
            l2 = 0 if lab in ("normal", "n", "0") else 1
        # patient id: column -> P\d+ token in the name -> stem
        if pidcol is not None and pd.notna(r[pidcol]):
            patient = str(r[pidcol])
        else:
            m = _re.search(r"[Pp](\d+)", str(r[fcol]))
            patient = ("P" + m.group(1)) if m else Path(str(r[fcol])).stem
        device = str(r[devcol]) if devcol else default_device
        if srcol is not None and pd.notna(r[srcol]):
            sr = int(r[srcol]); dur = float(r[durcol]) if (durcol and pd.notna(r[durcol])) else float("nan")
        elif probe_audio:
            sr, dur = _wav_info(fpath)
        else:
            sr, dur = default_sr, float("nan")
        rows.append({"source": source, "patient_id": patient, "filepath": fpath,
                     "cycle_id": 0, "device": device, "hospital": source,
                     "label4": -1, "label2": int(l2),
                     "sample_rate": sr or default_sr, "duration": dur, "split": "train"})
    return pd.DataFrame(rows)


def build_extra_manifest(out_dir: str, specs) -> pd.DataFrame:
    """specs: list of dicts with keys csv, audio_dir, source, device, disease.
    Writes <out_dir>/extra_manifest.csv (appended datasets)."""
    os.makedirs(out_dir, exist_ok=True)
    parts = []
    for sp in specs:
        if not sp.get("csv") or not os.path.exists(sp["csv"]):
            print(f"[extra] skip missing {sp.get('csv')}"); continue
        d = convert_extra_cohort(sp["csv"], sp.get("audio_dir", ""), sp["source"],
                                 sp.get("device", sp["source"]),
                                 probe_audio=sp.get("probe", True),
                                 disease=sp.get("disease", False))
        print(f"[extra] {sp['source']}: {len(d)} clips, device={sorted(d.device.unique())}, "
              f"label2={d.label2.value_counts().to_dict()}, patients={d.patient_id.nunique()}")
        parts.append(d)
    if not parts:
        raise SystemExit("no extra datasets found")
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(os.path.join(out_dir, "extra_manifest.csv"), index=False)
    print(f"[extra] wrote {len(out)} rows -> {os.path.join(out_dir, 'extra_manifest.csv')}")
    return out


# ===========================================================================
# iPhone / smartphone recordings named with the Seoul/Sungbook convention
# "<patient>)<...>-<L>.wav", L in {N,C,W,B} -> normal/crackle/wheeze/both.
# Used as a HELD-OUT unseen device (full 4-class labels).
# ===========================================================================
import re as _re2
_L4 = {"n": 0, "normal": 0, "c": 1, "crackle": 1, "w": 2, "wheeze": 2,
       "b": 3, "both": 3, "a": 1, "abnormal": 1}


def _parse_named(fname: str):
    stem = os.path.splitext(os.path.basename(fname))[0]
    m = _re2.match(r"([A-Za-z0-9]+)\)", stem)
    pid = m.group(1) if m else (_re2.match(r"([A-Za-z0-9]+)[_\-.]", stem) or [None, stem])[1] \
        if _re2.match(r"([A-Za-z0-9]+)[_\-.]", stem) else stem
    lab = None
    for tok in _re2.split(r"[)_\-.]", stem)[::-1]:
        if tok.lower() in _L4:
            lab = _L4[tok.lower()]; break
    return pid, lab


def convert_named_dir(audio_dir: str, source: str, device: str,
                      probe_audio: bool = True, default_sr: int = 16000):
    """Scan a directory of WAVs named with the N/C/W/B convention -> unified rows."""
    wavs, skipped = [], 0
    for dp, dns, fns in os.walk(audio_dir):
        dns[:] = [d for d in dns if d != "__MACOSX"]   # macOS zip artefacts
        for f in fns:
            if f.startswith("._"):                     # AppleDouble, not audio
                skipped += 1
                continue
            if f.lower().endswith(".wav"):
                wavs.append(os.path.join(dp, f))
    if skipped:
        print(f"[named] skipped {skipped} AppleDouble/__MACOSX files")
    rows, nolab = [], 0
    for p in wavs:
        pid, lab = _parse_named(p)
        if lab is None:
            nolab += 1
            continue
        sr, dur = (_wav_info(p) if probe_audio else (default_sr, float("nan")))
        rows.append({"source": source, "patient_id": pid, "filepath": os.path.abspath(p),
                     "cycle_id": 0, "device": device, "hospital": source,
                     "label4": int(lab), "label2": (0 if lab == 0 else 1),
                     "sample_rate": sr or default_sr, "duration": dur, "split": "train"})
    df = pd.DataFrame(rows)
    print(f"[named] {source}: {len(df)} clips ({nolab} unlabeled skipped) "
          f"device={device} labels={df.label4.value_counts().to_dict() if len(df) else {}} "
          f"patients={df.patient_id.nunique() if len(df) else 0}")
    return df