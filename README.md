# CASG — Continuous Acquisition-Style Generalization

Code and analysis for a patient-disjoint, device-target benchmark for respiratory
sound classification, and for **CASG**: a training-only residual transfer of
frequency-conditioned token statistics inside an Audio Spectrogram Transformer (AST).

Four training pipelines are compared under one backbone, front end, acquisition
simulator, optimizer and seed set: **ERM**, **MixStyle**, **CF-only**
(counterfactual prediction consistency) and **CASG**.

---

## Results

Three-seed probability ensembles. Five leave-one-device-out targets plus an
external hospital withheld from all development.

| Endpoint | ERM | MixStyle | CF-only | **CASG** |
|---|---|---|---|---|
| Mean AUROC, five device targets | 0.809 | **0.811** | 0.810 | 0.811 |
| Worst-target AUROC | 0.731 | 0.737 | 0.731 | **0.744** |
| External stethoscope AUROC | 0.840 | 0.844 | **0.850** | 0.850 |
| External phone AUROC (phone held out) | 0.816 | 0.828 | **0.844** | 0.831 |
| External phone AP (phone held out) | 0.434 | 0.460 | 0.495 | **0.506** |
| Mean CPD x10^-3 (lower = less acquisition-sensitive) | 20.4 | 19.9 | 8.0 | **7.3** |

CASG attains the highest observed worst-target AUROC and the lowest modelled
acquisition sensitivity. Of 24 paired patient-clustered contrasts, two exclude
zero — both CASG over ERM on the joint unseen-device, unseen-site arm
(+0.015 AUROC, +0.072 AP), exploratory and unadjusted. **No CASG−CF-only
interval excludes zero**, so an incremental benefit of the token transfer over
consistency alone is not established. Per-arm values: `docs/VERIFIED_NUMBERS.md`.

## Method

Two physics-simulated acquisition views of each clip traverse shared AST blocks
1–9. From epoch 6, the anchor's 12×39 patch-token grid is combined with
frequency-conditioned statistics of a donor carrying the same four-class label
and a different patient key, through a residual AdaIN at rho = 0.5; class and
distillation tokens bypass the transfer. Anchor, paired and transferred maps then
pass through shared blocks 10–12, special-token pooling, a 768→512→256 projector
and four-class / binary heads. Classification is applied to the anchor and
transferred paths; symmetric stop-gradient KL consistency between the anchor and
each other path.

**Inference is one unmodified AST pass** — no simulator, no donor, no transfer.

## Benchmark

| Source | Clips | Patients | Role |
|---|---:|---:|---|
| Hospital A, stethoscope | 22,305 | 611 | development |
| Hospital A, phone | 4,991 | 98 | development / target |
| ICBHI 2017, four devices | 6,842 | 126 | development / targets |
| Hospital B, stethoscope | 1,966 | 99 | external evaluation |
| Hospital B, phone | 4,180 | 96 | external evaluation |

Partitions are keyed on a device-invariant physical-patient identifier, so a
child recorded on both a stethoscope and a phone is never split across train and
test. Patient counts are not additive across arms.

## Layout

```
src/casg/      physics simulator, AdaIN transfer, model split, training step
src/data/      manifests, band harmonization, dataset, augmentation
src/models/    AST backbone and baseline network
scripts/       training, freezing, verification, analysis, figures
configs/       casg_lite, cf_physics, casg_identity, casg_full
analysis/      aggregate result tables and the freeze record
figures/       publication figures (vector PDF)
manuscript/    LaTeX source
docs/          runbooks, verified-number reference, figure specification
```

## Usage

```bash
pip install -r requirements.txt

python scripts/train_contrastive.py --config configs/casg_lite.yaml \
    --held_out AKGC417L --seed 0
```

Analysis, given prediction dumps under `runs/`:

```bash
python scripts/casg_verify_preds.py    # alignment gate — must pass first
python scripts/casg_ensemble.py        # three-seed ensembles
python scripts/casg_figures.py         # all figures, vector + 600 dpi
```

`casg_verify_preds.py` asserts that every prediction file for a fold describes
the same clips in the same order. Ensembling and paired bootstraps are invalid
without it, so it runs first and aborts on any mismatch.

## Controlled follow-up (v2)

`docs/V2_RUNBOOK.md` pre-declares a study that isolates the token transfer with a
branch- and loss-matched **identity control**: identical branches, losses and
dropout, with the transfer replaced by an explicit identity operation.

```bash
python scripts/v2_patch.py --check && python scripts/v2_patch.py
python scripts/v2_gates.py     # seven runtime gates on tensors; all must pass
bash scripts/v2_launch.sh      # 3 conditions x 6 fits x 3 seeds
python scripts/v2_analyze.py
```

## Data availability

ICBHI 2017 is public under its own terms. The hospital cohorts are identifiable
pediatric clinical recordings collected under IRB approval (No. 2021-0017-02) and
**are not released here**: this repository contains no audio, no patient manifest
and no per-clip prediction. De-identified analysis data may be requested from the
corresponding author, subject to institutional approval.

## Reproducibility

`analysis/freeze.json` records the repository commit and the SHA-256 digest of
every source file, configuration, manifest, checkpoint and prediction file at the
configuration freeze (2026-09-07T02:37:24Z), together with the statement that no
hyperparameter, checkpoint-selection rule or analysis variant may be selected
using frozen-test results after that timestamp.



## Licence

Code released under the MIT Licence. The ICBHI database and the hospital
recordings are governed by their own terms and are not covered by it.
