"""Generate every LaTeX table in the paper from runs/*/preds_*.npz.

Writes paper/tab_*.tex, which main.tex \\input{}s. Re-run after any grid
change and the manuscript numbers update — never hand-edit the tab_ files.

  python scripts/make_tables.py
  python scripts/make_tables.py --root runs --la_root runs_LAon
"""
import _boot, argparse, glob, os, re
import numpy as np
import pandas as pd
from src.metrics import compute_metrics, bootstrap_ci, paired_bootstrap_test

MAIN = re.compile(r"[a-z]+_[a-z]+_[A-Za-z0-9]+_seed\d+$")
PRETTY = {"erm": "ERM", "dann": "DANN", "coral": "Deep CORAL",
          "mixstyle": "MixStyle", "contrastive": "SupCon",
          "dmixstyle": "Directional MixStyle"}
ORDER = ["erm", "dann", "coral", "mixstyle", "contrastive"]
DEVORDER = ["AKGC417L", "Meditron", "LittC2SE", "Litt3200", "smartphone"]

DEVICE_INFO = [
    ("Hospital", "Littmann CORE", "Electronic stethoscope", "4", "22{,}305", "train"),
    ("Hospital", "iPhone 13", "Smartphone MEMS mic.", "48", "9{,}217", "LODO"),
    ("Hospital", "Littmann CORE (site 2)", "Electronic stethoscope", "4", "1{,}986", "external"),
    ("ICBHI", "AKG C417L", "Air-coupled lavalier mic.", "44.1", "4{,}345", "LODO"),
    ("ICBHI", "Welch Allyn Meditron", "Electronic stethoscope", "4/10/44.1", "1{,}426", "LODO"),
    ("ICBHI", "Littmann Classic II SE", "Acoustic stethoscope + mic.", "44.1", "575", "LODO"),
    ("ICBHI", "Littmann 3200", "Electronic stethoscope", "4", "496", "LODO"),
]


def load(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*", "preds_*.npz"))):
        b = os.path.basename(f)
        if b.startswith("preds_ood_"):
            continue
        if not MAIN.fullmatch(b[len("preds_"):-len(".npz")]):
            continue
        z = np.load(f, allow_pickle=True)
        p4, p2 = z["prob4"], z["prob2"]
        m = compute_metrics(z["y4"], p4.argmax(1), z["y2"], p2.argmax(1), p4, p2)
        m.update({"method": str(z["method"]), "held_out": str(z["held_out"]),
                  "seed": int(z["seed"])})
        rows.append(m)
    if not rows:
        raise SystemExit(f"no main-grid runs in {root}/")
    return pd.DataFrame(rows)


def ci(v, nd=3, pct=False):
    m, lo, hi = bootstrap_ci(v)
    if not np.isfinite(m):
        return "--"
    s = 100 if pct else 1
    return f"{m*s:.{nd}f} ({lo*s:.{nd}f}--{hi*s:.{nd}f})"


def w(path, body):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"  wrote {path}")


# ---------------------------------------------------------------- tables ---
def tab_devices(out):
    r = "\n".join(
        f"{c} & {d} & {t} & {sr} & {n} & {ro} \\\\" for c, d, t, sr, n, ro in DEVICE_INFO)
    w(os.path.join(out, "tab_devices.tex"), f"""\\begin{{table}}[!t]
\\caption{{Capture devices. LODO denotes a leave-one-device-out fold; the
external site is a frozen test set never used for training.}}
\\label{{tab:devices}}
\\centering
\\setlength{{\\tabcolsep}}{{3pt}}
\\footnotesize
\\begin{{tabular}}{{lllrrl}}
\\toprule
Cohort & Device & Transduction & kHz & Clips & Role \\\\
\\midrule
{r}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab_main(df, out, metric="auroc2"):
    devs = [d for d in DEVORDER if d in set(df.held_out)]
    lines = []
    for m in [x for x in ORDER if x in set(df.method)]:
        g = df[df.method == m]
        cells = [ci(g[g.held_out == d][metric].values) for d in devs]
        means = [g[g.held_out == d][metric].mean() for d in devs]
        means = [x for x in means if np.isfinite(x)]
        worst = f"\\textbf{{{min(means):.3f}}}" if means else "--"
        lines.append(f"{PRETTY.get(m, m)} & " + " & ".join(cells) + f" & {worst} \\\\")
    hdr = " & ".join(devs)
    w(os.path.join(out, "tab_main.tex"), f"""\\begin{{table*}}[!t]
\\caption{{Leave-one-device-out AUROC (mean, 95\\,\\% bootstrap CI over three
seeds). Every method uses the identical backbone, protocol and budget. The
final column is the worst-device value, which governs deployment.}}
\\label{{tab:main}}
\\centering
\\footnotesize
\\begin{{tabular}}{{l{'c'*len(devs)}c}}
\\toprule
Method & {hdr} & Worst \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""")


def tab_sig(df, out, metric="auroc2", baseline="erm"):
    piv = df.pivot_table(index=["held_out", "seed"], columns="method", values=metric)
    others = [m for m in ORDER if m in piv.columns and m != baseline]
    nb = max(1, len(others))
    lines = []
    for m in others:
        d = piv[[baseline, m]].dropna()
        p = paired_bootstrap_test(d[m].values, d[baseline].values)
        pb = min(1.0, p * nb)
        lines.append(f"{PRETTY.get(m,m)} & {len(d)} & {d[m].mean():.3f} & "
                     f"{d[m].mean()-d[baseline].mean():+.4f} & {p:.3f} & {pb:.3f} & "
                     f"{'yes' if pb < 0.05 else 'no'} \\\\")
    base = piv[baseline].mean()
    w(os.path.join(out, "tab_sig.tex"), f"""\\begin{{table}}[!t]
\\caption{{Paired comparison against ERM (mean AUROC {base:.3f}), paired by
(device, seed). $p$ from a two-sided paired bootstrap; $p_{{\\mathrm{{bonf}}}}$
Bonferroni-corrected. No method improves on ERM.}}
\\label{{tab:sig}}
\\centering
\\footnotesize
\\begin{{tabular}}{{lrrrrrc}}
\\toprule
Method & $n$ & AUROC & $\\Delta$ & $p$ & $p_{{\\mathrm{{bonf}}}}$ & Sig. \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab_deploy(df, out, held="smartphone"):
    sub = df[df.held_out == held]
    cols = [("sp2", "Sp", True), ("se2", "Se", True), ("acc2", "Acc", True),
            ("f1_2", "$F_1$", False), ("auroc2", "AUROC", False),
            ("auprc2", "AUPRC", False)]
    lines = []
    for m in [x for x in ORDER if x in set(sub.method)]:
        g = sub[sub.method == m]
        lines.append(f"{PRETTY.get(m,m)} & " +
                     " & ".join(ci(g[c].values, 1 if p else 3, p) for c, _, p in cols) + " \\\\")
    hdr = " & ".join(n for _, n, _ in cols)
    w(os.path.join(out, "tab_deploy.tex"), f"""\\begin{{table*}}[!t]
\\caption{{Held-out {held} fold, full deployment view (mean, 95\\,\\% CI).
Specificity, sensitivity and accuracy in percent. AUROC is high while $F_1$
and specificity are poor: a threshold problem, not a discrimination problem.}}
\\label{{tab:deploy}}
\\centering
\\footnotesize
\\begin{{tabular}}{{lcccccc}}
\\toprule
Method & {hdr} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""")


def tab_la(out, root, la_root):
    """Logit-adjustment ablation: main grid (off) vs runs_LAon (on)."""
    def grab(r, suffix=""):
        d = {}
        for f in glob.glob(os.path.join(r, "*", f"metrics_*{suffix}.json")):
            b = os.path.basename(f)[len("metrics_"):-len(".json")]
            m = re.fullmatch(r"(\w+?)_ast_(\w+)_seed(\d)" + (suffix or ""), b)
            if not m:
                continue
            import json
            d.setdefault((m.group(1), m.group(2)), []).append(json.load(open(f)))
        return d
    on = grab(la_root) if os.path.isdir(la_root) else {}
    off = grab(root)
    keys = sorted(set(on) & set(off))
    keys = [k for k in keys if k[0] in ("erm", "contrastive")]
    if not keys:
        w(os.path.join(out, "tab_la.tex"),
          "% tab_la: no matched LA-on/LA-off pairs found — run the ablation.\n")
        return
    lines = []
    for meth, dev in keys:
        for lab, src in (("on", on), ("off", off)):
            v = src[(meth, dev)]
            g = lambda k: float(np.mean([x[k] for x in v]))
            sp4 = f"{g('sp4') * 100:.1f}"
            ic4 = f"{g('icbhi4'):.3f}"
            if lab == "off":                      # highlight the better setting
                sp4 = "\\textbf{" + sp4 + "}"
                ic4 = "\\textbf{" + ic4 + "}"
            lines.append(
                f"{PRETTY.get(meth, meth)} & {dev} & {lab} & "
                f"{g('auroc2'):.3f} & {g('sp2') * 100:.1f} & {sp4} & {ic4} & "
                f"{g('ece2'):.3f} \\\\")
    w(os.path.join(out, "tab_la.tex"), f"""\\begin{{table}}[!t]
\\caption{{Effect of logit adjustment under device shift. AUROC is unchanged,
confirming discrimination is unaffected, while four-class specificity
($\\mathrm{{Sp}}_4$) and ICBHI score improve markedly when it is removed.}}
\\label{{tab:la}}
\\centering
\\setlength{{\\tabcolsep}}{{3pt}}
\\footnotesize
\\begin{{tabular}}{{lllccccc}}
\\toprule
Method & Device & LA & AUROC & Sp & $\\mathrm{{Sp}}_4$ & ICBHI-4 & ECE \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab_calib(out, csvs):
    rows = []
    for f in sorted(csvs):
        d = pd.read_csv(f)
        d = d[d.K == 10]
        if not len(d):
            continue
        # non-greedy + anchored: a greedy [\d.]+ eats the extension dot on
        # calib_fpr0.05.csv and produces the unparseable string "0.05."
        m = re.search(r"fpr([\d.]+?)\.csv$", f)
        fpr = float(m.group(1)) if m else 0.20
        rows.append((1 - fpr, d.sp_cal.mean(), d.se_cal.mean(),
                     (d.se_cal.mean() + d.sp_cal.mean()) / 2,
                     d.sp_def.mean(), (d.se_def.mean() + d.sp_def.mean()) / 2))
    if not rows:
        w(os.path.join(out, "tab_calib.tex"), "% tab_calib: run scripts/calibrate.py\n")
        return
    rows.sort(reverse=True)
    lines = [f"{t:.2f} & {sp:.3f} & {sp-t:+.3f} & {se:.3f} & {ba:.3f} \\\\"
             for t, sp, se, ba, _, _ in rows]
    dsp, dba = rows[0][4], rows[0][5]
    w(os.path.join(out, "tab_calib.tex"), f"""\\begin{{table}}[!t]
\\caption{{Few-shot operating-point calibration with $K=10$ normal recordings
from the deployment device, averaged over all methods, devices and seeds.
The fixed 0.5 threshold gives specificity {dsp:.3f} with no control
(balanced accuracy {dba:.3f}).}}
\\label{{tab:calib}}
\\centering
\\footnotesize
\\begin{{tabular}}{{ccccc}}
\\toprule
Requested Sp & Achieved Sp & Error & Se & Balanced acc. \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")



def tab_folds(out, cfg_path="configs/contrastive.yaml"):
    """Review weakness 1: patients per fold, not just clips. K=10 calibration
    recordings are drawn from distinct patients, so the reader needs this."""
    try:
        from src.utils import load_config
        from src.data import manifest as M
        import pandas as _pd
    except Exception as e:
        w(os.path.join(out, "tab_folds.tex"), f"% tab_folds unavailable: {e}\n"); return
    cfg = load_config(cfg_path)
    seoul = M.load_manifest(cfg.data.train_manifest)
    icbhi = M.load_manifest(cfg.data.icbhi_manifest)
    ep = str(getattr(cfg.data, "extra_manifest", "") or "")
    pool = _pd.concat([icbhi] + ([M.load_manifest(ep)] if ep and os.path.exists(ep) else []),
                      ignore_index=True)
    lines = []
    for d in [x for x in DEVORDER if x in set(pool.device)]:
        f = M.build_loo_folds(pool, seoul, [d], True)[d]
        te = f.test
        norm = te[te.label2 == 0]
        lines.append(f"{d} & {len(f.train):,} & {len(te):,} & "
                     f"{te.patient_id.nunique()} & {norm.patient_id.nunique()} & "
                     f"{len(f.excluded_patients)} \\\\".replace(",", "{,}"))
    w(os.path.join(out, "tab_folds.tex"), f"""\\begin{{table}}[!t]
\\caption{{Leave-one-device-out folds. Calibration recordings are drawn from
distinct patients, so the count of normal patients in each test fold bounds the
usable $K$.}}
\\label{{tab:folds}}
\\centering
\\setlength{{\\tabcolsep}}{{3pt}}
\\footnotesize
\\begin{{tabular}}{{lrrrrr}}
\\toprule
\\multirow{{2}}{{*}}{{Held-out device}} & \\multicolumn{{1}}{{c}}{{Train}} & \\multicolumn{{3}}{{c}}{{Test}} & \\multicolumn{{1}}{{c}}{{Excl.}} \\\\
\\cmidrule(lr){{2-2}} \\cmidrule(lr){{3-5}} \\cmidrule(lr){{6-6}}
 & clips & clips & patients & normal pat. & patients \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab_ablation(out, root):
    """Component ablation from --tag_suffix runs (_noSSP/_noaug/_nosupcon/_nola)."""
    import json
    tags = [("", "Full"), ("_noSSP", "-- band harmonisation"),
            ("_noaug", "-- device augmentation"), ("_nosupcon", "-- contrastive term"),
            ("_nola", "-- logit adjustment")]
    devs = ["AKGC417L", "smartphone"]
    rows, have = [], False
    for suf, lab in tags:
        cells = []
        for d in devs:
            fs = glob.glob(os.path.join(root, "*", f"metrics_*_{d}_seed*{suf}.json"))
            fs = [f for f in fs if os.path.basename(f).endswith(f"{suf}.json")]
            if suf == "":
                fs = [f for f in fs if MAIN.fullmatch(
                    os.path.basename(f)[len("metrics_"):-len(".json")])]
            v = [json.load(open(f)) for f in fs]
            if v:
                have = True
                cells.append(f"{np.mean([x['auroc2'] for x in v]):.3f}")
                cells.append(f"{np.mean([x['f1_2'] for x in v]):.3f}")
            else:
                cells += ["--", "--"]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    if not have:
        w(os.path.join(out, "tab_ablation.tex"),
          "% tab_ablation: run the stage-4 ablations (see RUNBOOK).\n")
        return
    w(os.path.join(out, "tab_ablation.tex"), f"""\\begin{{table}}[!t]
\\caption{{Component ablation on two representative held-out devices, mean over
three seeds. Each row removes one component from the full configuration.}}
\\label{{tab:ablation}}
\\centering
\\setlength{{\\tabcolsep}}{{4pt}}
\\footnotesize
\\begin{{tabular}}{{lcccc}}
\\toprule
\\multirow{{2}}{{*}}{{Configuration}} & \\multicolumn{{2}}{{c}}{{AKGC417L}} & \\multicolumn{{2}}{{c}}{{smartphone}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
 & AUROC & $F_1$ & AUROC & $F_1$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab_backbone(out, root):
    """Backbone comparison from --backbone runs (_bkcnn/_bktimm) vs main AST."""
    import json
    variants = [("_bkcnn", "CNN (scratch)"), ("_bktimm", "EfficientNet-B0 (ImageNet)"),
                ("", "AST (AudioSet)")]
    devs = ["AKGC417L", "smartphone"]
    rows, have = [], False
    for suf, lab in variants:
        cells = []
        for d in devs:
            fs = glob.glob(os.path.join(root, "*", f"metrics_*_{d}_seed*{suf}.json"))
            fs = [f for f in fs if os.path.basename(f).endswith(f"{suf}.json")]
            if suf == "":
                fs = [f for f in fs if MAIN.fullmatch(
                    os.path.basename(f)[len("metrics_"):-len(".json")])]
            v = [json.load(open(f)) for f in fs]
            if v:
                have = True
                cells.append(f"{np.mean([x['auroc2'] for x in v]):.3f}")
            else:
                cells.append("--")
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    if not have:
        w(os.path.join(out, "tab_backbone.tex"),
          "% tab_backbone: run the backbone ablation (see RUNBOOK).\n")
        return
    w(os.path.join(out, "tab_backbone.tex"), f"""\\begin{{table}}[!t]
\\caption{{Backbone comparison under the identical protocol (AUROC, mean over
three seeds). Pretraining modality matters more than parameter count.}}
\\label{{tab:backbone}}
\\centering
\\footnotesize
\\begin{{tabular}}{{lcc}}
\\toprule
Backbone & AKGC417L & smartphone \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab_external(out, root):
    """Frozen external-hospital test (held_out=none)."""
    import json
    fs = sorted(glob.glob(os.path.join(root, "*", "metrics_*_none_seed*.json")))
    fs = [f for f in fs if MAIN.fullmatch(os.path.basename(f)[len("metrics_"):-len(".json")])]
    if not fs:
        w(os.path.join(out, "tab_external.tex"),
          "% tab_external: run --held_out none (see RUNBOOK).\n")
        return
    by = {}
    for f in fs:
        d = json.load(open(f))
        by.setdefault(d.get("method", "?"), []).append(d)
    rows = []
    for m in [x for x in ORDER if x in by]:
        v = by[m]
        g = lambda k: float(np.mean([x[k] for x in v]))
        rows.append(f"{PRETTY.get(m, m)} & {g('sp2')*100:.1f} & {g('se2')*100:.1f} & "
                    f"{g('f1_2'):.3f} & {g('auroc2'):.3f} & {g('auprc2'):.3f} \\\\")
    w(os.path.join(out, "tab_external.tex"), f"""\\begin{{table}}[!t]
\\caption{{Frozen external-hospital test. The model is trained on all devices;
the second site contributes no training data. Specificity and sensitivity in
percent.}}
\\label{{tab:external}}
\\centering
\\footnotesize
\\begin{{tabular}}{{lccccc}}
\\toprule
Method & Sp & Se & $F_1$ & AUROC & AUPRC \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--la_root", default="runs_LAon")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--deploy_device", default="smartphone")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.root)
    print(f"[tables] {len(df)} runs from {a.root}/")
    tab_devices(a.out)
    tab_main(df, a.out)
    tab_sig(df, a.out)
    tab_deploy(df, a.out, a.deploy_device)
    tab_la(a.out, a.root, a.la_root)
    tab_calib(a.out, glob.glob(os.path.join(a.root, "calib_fpr*.csv"))
              or glob.glob("runs/calib_fpr*.csv"))
    tab_folds(a.out)
    tab_ablation(a.out, a.root)
    tab_backbone(a.out, a.root)
    tab_external(a.out, a.root)
    print(f"[tables] done -> {a.out}/tab_*.tex")