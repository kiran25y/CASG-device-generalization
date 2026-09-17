# #!/usr/bin/env python3
# """Generate publication-quality CASG result figures from frozen predictions.

# This script reads the per-clip ``preds_*.npz`` files written by
# ``scripts/train_contrastive.py``.  It does not infer curves from aggregate
# tables and it does not run a checkpoint again.  The prediction dumps are the
# exact checkpoint outputs used by the reporting code, so using them avoids GPU
# work and prevents a new inference pass from silently changing preprocessing.

# Generated files
# ---------------

# ``fig3_lodo_forest``
#     Seed-ensemble AUROC and patient-clustered 95% confidence intervals on one
#     shared x-axis.
# ``fig5_roc`` and ``fig6_pr``
#     Six evaluation arms, thin solid mean curves over seeds.
# ``fig7_calibration``
#     Fifteen-bin reliability diagrams using pooled seed-by-clip predictions.
# ``fig6_external_curves``
#     Native-size ROC, precision-recall, and reliability panels for the frozen
#     Hospital-B stethoscope arm.
# ``fig8_frozen_phone``
#     Hospital-B smartphone AUROC with patient-clustered intervals computed from
#     the frozen per-clip prediction dumps.
# ``fig4_mechanism``
#     CPD and CFV means +/- one standard deviation from ``cpd_report.csv``.
# ``fig9_tsne``
#     A fixed, balanced four-method representation diagnostic.  Features are
#     extracted from saved checkpoints; no training occurs.

# Every figure is written as vector PDF, editable SVG, and 600-dpi PNG.  The PDF
# is the file to include in LaTeX.  The script fails on a missing or misaligned
# paper-grid input unless ``--allow-missing`` is explicitly supplied.

# Examples
# --------

# Run from the repository root::

#     python scripts/casg_verify_preds.py
#     python scripts/casg_paper_figures.py --root . --out figures_publication

# Generate only the discrimination curves::

#     python scripts/casg_paper_figures.py --only roc pr external

# Use fewer bootstrap replicates only for a quick rendering check::

#     python scripts/casg_paper_figures.py --bootstrap 100 --only lodo phone

# The legacy ``casg_lite`` suffix remains only in input filenames created by the
# completed experiment.  All displayed labels use ``CASG (ours)``.
# """

# from __future__ import annotations

# import argparse
# import csv
# import hashlib
# import json
# import math
# import re
# import sys
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Callable, Iterable, Sequence

# import numpy as np

# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from matplotlib.ticker import MultipleLocator

# try:
#     from sklearn.metrics import (
#         average_precision_score,
#         precision_recall_curve,
#         roc_auc_score,
#         roc_curve,
#     )
# except ImportError as exc:  # pragma: no cover - actionable runtime message
#     raise SystemExit(
#         "scikit-learn is required. Install the project requirements first: "
#         "python -m pip install -r requirements.txt"
#     ) from exc


# # IEEE Transactions nominal column widths in inches.
# ONE_COLUMN = 3.50
# TWO_COLUMN = 7.16

# FOLDS = ("AKGC417L", "Meditron", "LittC2SE", "Litt3200", "smartphone", "none")
# FOLD_LABEL = {
#     "AKGC417L": "AKG C417L",
#     "Meditron": "Meditron",
#     "LittC2SE": "Littmann C-II-SE",
#     "Litt3200": "Littmann 3200",
#     "smartphone": "Hospital-A phone",
#     "none": "Hospital-B stethoscope",
# }
# PHONE_ROW_LABEL = {
#     "AKGC417L": "AKG C417L held out",
#     "Meditron": "Meditron held out",
#     "LittC2SE": "Littmann C-II-SE held out",
#     "Litt3200": "Littmann 3200 held out",
#     "smartphone": "Phone held out\n(joint device + site)",
#     "none": "All development devices",
# }


# @dataclass(frozen=True)
# class MethodSpec:
#     key: str
#     label: str
#     short: str
#     color: str
#     linestyle: object
#     marker: str
#     main_pattern: str
#     phone_pattern: str
#     cpd_suffix: str


# # Four high-contrast academic colors requested for the manuscript.  Result
# # curves are all solid; markers remain available for forest and bar plots.
# METHODS: tuple[MethodSpec, ...] = (
#     MethodSpec(
#         key="erm",
#         label="ERM-Physics",
#         short="ERM",
#         color="#0072B2",
#         linestyle="-",
#         marker="o",
#         main_pattern="runs/erm/preds_erm_ast_{fold}_seed{seed}_phys.npz",
#         phone_pattern=(
#             "runs/frozen_phone/"
#             "preds_erm_ast_{fold}_seed{seed}_phys_frozenphone.npz"
#         ),
#         cpd_suffix="_phys",
#     ),
#     MethodSpec(
#         key="mixstyle",
#         label="MixStyle-Physics",
#         short="MixStyle",
#         color="#E3A500",  # gold/yellow
#         linestyle="-",
#         marker="s",
#         main_pattern=(
#             "runs/mixstyle/"
#             "preds_mixstyle_ast_{fold}_seed{seed}_phys.npz"
#         ),
#         phone_pattern=(
#             "runs/frozen_phone/"
#             "preds_mixstyle_ast_{fold}_seed{seed}_phys_frozenphone.npz"
#         ),
#         cpd_suffix="_phys",
#     ),
#     MethodSpec(
#         key="cf",
#         label="CF-only-Physics",
#         short="CF-only",
#         color="#009E73",
#         linestyle="-",
#         marker="^",
#         main_pattern=(
#             "runs/casg/"
#             "preds_casg_ast_{fold}_seed{seed}_cf_physics.npz"
#         ),
#         phone_pattern=(
#             "runs/frozen_phone/"
#             "preds_casg_ast_{fold}_seed{seed}_cf_physics_frozenphone.npz"
#         ),
#         cpd_suffix="_cf_physics",
#     ),
#     MethodSpec(
#         key="casg",
#         label="CASG (ours)",
#         short="CASG",
#         color="#D1495B",  # red
#         linestyle="-",
#         marker="D",
#         main_pattern=(
#             "runs/casg/"
#             "preds_casg_ast_{fold}_seed{seed}_casg_lite.npz"
#         ),
#         phone_pattern=(
#             "runs/frozen_phone/"
#             "preds_casg_ast_{fold}_seed{seed}_casg_lite_frozenphone.npz"
#         ),
#         cpd_suffix="_casg_lite",
#     ),
# )


# @dataclass
# class Prediction:
#     path: Path
#     y2: np.ndarray
#     probability: np.ndarray
#     patient: np.ndarray
#     y4: np.ndarray | None
#     source: np.ndarray | None
#     device: np.ndarray | None

#     @property
#     def n(self) -> int:
#         return int(self.y2.size)


# class InputError(RuntimeError):
#     """Raised when a paper input is missing, malformed, or misaligned."""


# def _as_text(a: np.ndarray) -> np.ndarray:
#     return np.asarray(a).astype(str).reshape(-1)


# def _scalar(z: np.lib.npyio.NpzFile, key: str) -> str | None:
#     if key not in z.files:
#         return None
#     value = np.asarray(z[key])
#     if value.size != 1:
#         return None
#     return str(value.reshape(-1)[0])


# def load_prediction(
#     path: Path,
#     *,
#     expected_method: MethodSpec | None = None,
#     expected_fold: str | None = None,
#     expected_seed: int | None = None,
# ) -> Prediction:
#     """Load and validate one prediction dump."""

#     if not path.is_file():
#         raise InputError(f"missing prediction file: {path}")
#     try:
#         with np.load(path, allow_pickle=True) as z:
#             missing = {"y2", "prob2", "patient"} - set(z.files)
#             if missing:
#                 raise InputError(f"{path}: missing arrays {sorted(missing)}")
#             y2 = np.asarray(z["y2"]).astype(int).reshape(-1).copy()
#             prob2 = np.asarray(z["prob2"], dtype=np.float64).copy()
#             patient = _as_text(z["patient"]).copy()
#             y4 = (
#                 np.asarray(z["y4"]).astype(int).reshape(-1).copy()
#                 if "y4" in z.files
#                 else None
#             )
#             source = _as_text(z["source"]).copy() if "source" in z.files else None
#             device = _as_text(z["device"]).copy() if "device" in z.files else None

#             recorded_method = _scalar(z, "method")
#             recorded_fold = _scalar(z, "held_out")
#             recorded_seed = _scalar(z, "seed")
#     except (OSError, ValueError, KeyError) as exc:
#         raise InputError(f"cannot read {path}: {exc}") from exc

#     if prob2.ndim != 2 or prob2.shape[1] != 2:
#         raise InputError(f"{path}: prob2 must have shape [N, 2], got {prob2.shape}")
#     if prob2.shape[0] != y2.size:
#         raise InputError(f"{path}: prob2/y2 length mismatch")
#     for name, array in (
#         ("patient", patient),
#         ("y4", y4),
#         ("source", source),
#         ("device", device),
#     ):
#         if array is not None and array.size != y2.size:
#             raise InputError(f"{path}: {name}/y2 length mismatch")
#     if not np.isfinite(prob2).all():
#         raise InputError(f"{path}: non-finite probabilities")
#     if (prob2 < -1e-7).any() or (prob2 > 1.0 + 1e-7).any():
#         raise InputError(f"{path}: probabilities outside [0, 1]")
#     if not np.allclose(prob2.sum(axis=1), 1.0, atol=1e-4, rtol=0.0):
#         raise InputError(f"{path}: probability rows do not sum to one")
#     if not set(np.unique(y2)).issubset({0, 1}):
#         raise InputError(f"{path}: y2 is not binary")
#     if np.unique(y2).size != 2:
#         raise InputError(f"{path}: both binary classes are required")
#     if np.any(np.char.str_len(patient) == 0):
#         raise InputError(f"{path}: empty patient identifier")

#     if expected_fold is not None and recorded_fold is not None:
#         if recorded_fold != expected_fold:
#             raise InputError(
#                 f"{path}: held_out={recorded_fold!r}, expected {expected_fold!r}"
#             )
#     if expected_seed is not None and recorded_seed is not None:
#         try:
#             if int(float(recorded_seed)) != int(expected_seed):
#                 raise InputError(
#                     f"{path}: seed={recorded_seed!r}, expected {expected_seed}"
#                 )
#         except ValueError as exc:
#             raise InputError(f"{path}: invalid seed metadata {recorded_seed!r}") from exc
#     # Method metadata in old runs may use the historical experiment key.  The
#     # exact filename is authoritative; do not rewrite or relabel the run file.
#     if expected_method is not None and recorded_method is not None:
#         allowed = {
#             expected_method.key,
#             expected_method.label,
#             expected_method.short,
#             "acpl" if expected_method.key == "casg" else expected_method.key,
#         }
#         if expected_method.key in {"cf", "casg"}:
#             allowed.add("casg")
#         if recorded_method not in allowed and expected_method.key not in recorded_method.lower():
#             raise InputError(
#                 f"{path}: method metadata {recorded_method!r} does not match "
#                 f"{expected_method.key!r}"
#             )

#     return Prediction(
#         path=path,
#         y2=y2,
#         probability=prob2[:, 1],
#         patient=patient,
#         y4=y4,
#         source=source,
#         device=device,
#     )


# def assert_aligned(predictions: Sequence[Prediction], context: str) -> None:
#     """Require identical evaluation rows across methods and seeds."""

#     if not predictions:
#         raise InputError(f"{context}: no prediction files")
#     ref = predictions[0]
#     for cur in predictions[1:]:
#         if cur.n != ref.n:
#             raise InputError(
#                 f"{context}: {cur.path} has {cur.n} rows; {ref.path} has {ref.n}"
#             )
#         fields = (
#             ("binary labels", ref.y2, cur.y2),
#             ("four-class labels", ref.y4, cur.y4),
#             ("patient identifiers", ref.patient, cur.patient),
#             ("source identifiers", ref.source, cur.source),
#             ("device identifiers", ref.device, cur.device),
#         )
#         for label, left, right in fields:
#             if left is None and right is None:
#                 continue
#             if left is None or right is None or not np.array_equal(left, right):
#                 raise InputError(
#                     f"{context}: {label} differ between {ref.path} and {cur.path}; "
#                     "seed averaging or paired comparisons would be invalid"
#                 )


# class PredictionStore:
#     def __init__(
#         self,
#         root: Path,
#         seeds: Sequence[int],
#         *,
#         allow_missing: bool,
#     ) -> None:
#         self.root = root.resolve()
#         self.seeds = tuple(int(s) for s in seeds)
#         self.allow_missing = bool(allow_missing)
#         self.cache: dict[tuple[str, str, int, bool], Prediction] = {}
#         self.used_paths: set[Path] = set()

#     def path(self, method: MethodSpec, fold: str, seed: int, phone: bool) -> Path:
#         pattern = method.phone_pattern if phone else method.main_pattern
#         return self.root / pattern.format(fold=fold, seed=seed)

#     def load(
#         self,
#         method: MethodSpec,
#         fold: str,
#         seed: int,
#         *,
#         phone: bool = False,
#     ) -> Prediction | None:
#         key = (method.key, fold, int(seed), phone)
#         if key in self.cache:
#             return self.cache[key]
#         path = self.path(method, fold, int(seed), phone)
#         if not path.is_file():
#             if self.allow_missing:
#                 return None
#             raise InputError(f"missing prediction file: {path}")
#         pred = load_prediction(
#             path,
#             expected_method=method,
#             expected_fold=fold,
#             expected_seed=int(seed),
#         )
#         self.cache[key] = pred
#         self.used_paths.add(path.resolve())
#         return pred

#     def seeded_cell(
#         self, method: MethodSpec, fold: str, *, phone: bool = False
#     ) -> list[tuple[int, Prediction]]:
#         """Return explicit ``(seed, prediction)`` pairs for one result cell."""

#         out = [
#             (seed, p)
#             for seed in self.seeds
#             if (p := self.load(method, fold, seed, phone=phone)) is not None
#         ]
#         if not self.allow_missing and len(out) != len(self.seeds):
#             raise InputError(
#                 f"{method.label}/{fold}: expected {len(self.seeds)} seeds, "
#                 f"found {len(out)}"
#             )
#         if out:
#             assert_aligned([p for _, p in out], f"{method.label}/{fold}")
#         return out

#     def cell(
#         self, method: MethodSpec, fold: str, *, phone: bool = False
#     ) -> list[Prediction]:
#         return [p for _, p in self.seeded_cell(method, fold, phone=phone)]

#     def validate_grid(self, *, phone: bool = False) -> int:
#         label = "frozen phone" if phone else "main"
#         count = 0
#         for fold in FOLDS:
#             group: list[Prediction] = []
#             for method in METHODS:
#                 cell = self.cell(method, fold, phone=phone)
#                 group.extend(cell)
#                 count += len(cell)
#             if group:
#                 assert_aligned(group, f"{label} grid, fold={fold}")
#             elif not self.allow_missing:
#                 raise InputError(f"{label} grid, fold={fold}: no prediction files")
#         return count

#     def ensemble(
#         self, method: MethodSpec, fold: str, *, phone: bool = False
#     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#         cell = self.cell(method, fold, phone=phone)
#         if not cell:
#             raise InputError(f"{method.label}/{fold}: no predictions")
#         assert_aligned(cell, f"{method.label}/{fold}")
#         return cell[0].y2, np.mean([p.probability for p in cell], axis=0), cell[0].patient


# def configure_style() -> None:
#     plt.rcParams.update(
#         {
#             "font.family": "serif",
#             "font.serif": [
#                 "Times New Roman",
#                 "Nimbus Roman",
#                 "Nimbus Roman No9 L",
#                 "DejaVu Serif",
#             ],
#             "mathtext.fontset": "dejavuserif",
#             "font.size": 8.0,
#             "axes.labelsize": 8.0,
#             "axes.titlesize": 8.4,
#             "axes.linewidth": 0.8,
#             "xtick.labelsize": 7.0,
#             "ytick.labelsize": 7.0,
#             "xtick.major.width": 0.8,
#             "ytick.major.width": 0.8,
#             "xtick.major.size": 3.0,
#             "ytick.major.size": 3.0,
#             "xtick.direction": "out",
#             "ytick.direction": "out",
#             "legend.fontsize": 7.0,
#             "legend.frameon": False,
#             "axes.spines.top": False,
#             "axes.spines.right": False,
#             "axes.axisbelow": True,
#             "grid.color": "#E6E6E6",
#             "grid.linewidth": 0.45,
#             "grid.alpha": 1.0,
#             "lines.solid_capstyle": "round",
#             "lines.dash_capstyle": "round",
#             "figure.facecolor": "white",
#             "axes.facecolor": "white",
#             "savefig.facecolor": "white",
#             "pdf.fonttype": 42,
#             "ps.fonttype": 42,
#             "svg.fonttype": "none",
#         }
#     )


# def save_figure(
#     fig: plt.Figure,
#     out_dir: Path,
#     stem: str,
#     *,
#     formats: Sequence[str],
#     dpi: int,
# ) -> list[Path]:
#     out_dir.mkdir(parents=True, exist_ok=True)
#     written: list[Path] = []
#     metadata = {"Creator": "casg_paper_figures.py", "Title": stem}
#     for extension in formats:
#         path = out_dir / f"{stem}.{extension}"
#         kwargs: dict[str, object] = {
#             "facecolor": "white",
#             "edgecolor": "none",
#         }
#         if extension == "png":
#             kwargs["dpi"] = dpi
#         elif extension == "pdf":
#             kwargs["metadata"] = metadata
#         fig.savefig(path, **kwargs)
#         if not path.is_file() or path.stat().st_size == 0:
#             raise RuntimeError(f"failed to write {path}")
#         written.append(path)
#     plt.close(fig)
#     print("[write] " + ", ".join(str(p) for p in written))
#     return written


# def _method_handles() -> list[Line2D]:
#     return [
#         Line2D(
#             [],
#             [],
#             color=m.color,
#             linestyle=m.linestyle,
#             linewidth=2.0 if m.key == "casg" else 1.6,
#             marker=m.marker,
#             markersize=4.0,
#             markerfacecolor=m.color if m.key == "casg" else "white",
#             markeredgewidth=1.0,
#             label=m.label,
#         )
#         for m in METHODS
#     ]


# def _curve_handles() -> list[Line2D]:
#     """Legend handles matching the uncluttered solid result curves."""

#     return [
#         Line2D(
#             [],
#             [],
#             color=m.color,
#             linestyle="-",
#             linewidth=1.55 if m.key == "casg" else 1.25,
#             label=m.label,
#         )
#         for m in METHODS
#     ]


# def _style_axis(ax: plt.Axes) -> None:
#     ax.grid(True)
#     ax.set_xlim(0.0, 1.0)
#     ax.set_xticks(np.linspace(0.0, 1.0, 6))
#     ax.set_yticks(np.linspace(0.0, 1.0, 6))


# def _roc_samples(cell: Sequence[Prediction], grid: np.ndarray) -> np.ndarray:
#     curves: list[np.ndarray] = []
#     for pred in cell:
#         fpr, tpr, _ = roc_curve(pred.y2, pred.probability, drop_intermediate=False)
#         curve = np.interp(grid, fpr, tpr)
#         curve[0], curve[-1] = 0.0, 1.0
#         curves.append(curve)
#     return np.asarray(curves)


# def _pr_samples(cell: Sequence[Prediction], grid: np.ndarray) -> np.ndarray:
#     curves: list[np.ndarray] = []
#     for pred in cell:
#         precision, recall, _ = precision_recall_curve(pred.y2, pred.probability)
#         # sklearn returns recall in descending order.  Interpolate only for
#         # cross-seed visualization; AUPRC itself is computed by the exact
#         # average_precision_score function, never by integrating this grid.
#         recall = recall[::-1]
#         precision = precision[::-1]
#         unique_recall, first = np.unique(recall, return_index=True)
#         curve = np.interp(grid, unique_recall, precision[first])
#         curves.append(curve)
#     return np.asarray(curves)


# def _plot_mean_curve(
#     ax: plt.Axes,
#     grid: np.ndarray,
#     curves: np.ndarray,
#     method: MethodSpec,
# ) -> None:
#     mean = curves.mean(axis=0)
#     ax.plot(
#         grid,
#         mean,
#         color=method.color,
#         linestyle="-",
#         linewidth=1.55 if method.key == "casg" else 1.25,
#         zorder=4 if method.key == "casg" else 3,
#     )


# def _core_metric_rows(store: PredictionStore) -> list[dict[str, object]]:
#     rows: list[dict[str, object]] = []
#     for fold in FOLDS:
#         for method in METHODS:
#             for seed, pred in store.seeded_cell(method, fold):
#                 rows.append(
#                     {
#                         "arm": fold,
#                         "arm_label": FOLD_LABEL[fold],
#                         "method": method.label,
#                         "seed": seed,
#                         "n_clips": pred.n,
#                         "n_patients": np.unique(pred.patient).size,
#                         "auroc": roc_auc_score(pred.y2, pred.probability),
#                         "auprc": average_precision_score(pred.y2, pred.probability),
#                         "ece_15bin": expected_calibration_error(
#                             pred.y2, pred.probability, n_bins=15
#                         ),
#                     }
#                 )
#     return rows


# def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
#     if not rows:
#         return
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with path.open("w", newline="", encoding="utf-8") as handle:
#         writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
#         writer.writeheader()
#         writer.writerows(rows)
#     print(f"[write] {path}")


# def figure_curves(
#     store: PredictionStore,
#     out_dir: Path,
#     *,
#     kind: str,
#     formats: Sequence[str],
#     dpi: int,
# ) -> None:
#     grid = np.linspace(0.0, 1.0, 401)
#     fig, axes = plt.subplots(2, 3, figsize=(TWO_COLUMN, 4.55))
#     fig.subplots_adjust(
#         left=0.075, right=0.992, top=0.965, bottom=0.135, wspace=0.31, hspace=0.43
#     )
#     letters = "abcdef"
#     for panel, (ax, fold) in enumerate(zip(axes.ravel(), FOLDS)):
#         reference: Prediction | None = None
#         for method in METHODS:
#             cell = store.cell(method, fold)
#             if not cell:
#                 continue
#             reference = reference or cell[0]
#             curves = (
#                 _roc_samples(cell, grid) if kind == "roc" else _pr_samples(cell, grid)
#             )
#             _plot_mean_curve(ax, grid, curves, method)

#         if kind == "roc":
#             ax.plot(
#                 [0, 1],
#                 [0, 1],
#                 color="#A8A8A8",
#                 linestyle=(0, (3, 2)),
#                 linewidth=0.8,
#                 zorder=1,
#             )
#             ax.set_xlabel("False-positive rate")
#             ax.set_ylabel("True-positive rate")
#         else:
#             if reference is not None:
#                 prevalence = float(reference.y2.mean())
#                 ax.axhline(
#                     prevalence,
#                     color="#A8A8A8",
#                     linestyle=(0, (3, 2)),
#                     linewidth=0.8,
#                     zorder=1,
#                 )
#             ax.set_xlabel("Recall")
#             ax.set_ylabel("Precision")
#         _style_axis(ax)
#         ax.set_ylim(0.0, 1.01)
#         ax.set_title(f"({letters[panel]}) {FOLD_LABEL[fold]}", pad=3.0)

#     fig.legend(
#         handles=_curve_handles(),
#         loc="lower center",
#         bbox_to_anchor=(0.5, 0.012),
#         ncol=4,
#         columnspacing=1.25,
#         handlelength=2.7,
#     )
#     save_figure(
#         fig,
#         out_dir,
#         "fig5_roc" if kind == "roc" else "fig6_pr",
#         formats=formats,
#         dpi=dpi,
#     )


# def expected_calibration_error(
#     y: np.ndarray, probability: np.ndarray, *, n_bins: int = 15
# ) -> float:
#     y = np.asarray(y, dtype=np.float64)
#     probability = np.asarray(probability, dtype=np.float64)
#     edges = np.linspace(0.0, 1.0, n_bins + 1)
#     total = 0.0
#     for index in range(n_bins):
#         if index == 0:
#             mask = (probability >= edges[index]) & (probability <= edges[index + 1])
#         else:
#             mask = (probability > edges[index]) & (probability <= edges[index + 1])
#         if mask.any():
#             total += float(mask.mean()) * abs(
#                 float(y[mask].mean()) - float(probability[mask].mean())
#             )
#     return float(total)


# def calibration_points(
#     y: np.ndarray, probability: np.ndarray, *, n_bins: int = 15
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     edges = np.linspace(0.0, 1.0, n_bins + 1)
#     confidence: list[float] = []
#     observed: list[float] = []
#     counts: list[int] = []
#     for index in range(n_bins):
#         if index == 0:
#             mask = (probability >= edges[index]) & (probability <= edges[index + 1])
#         else:
#             mask = (probability > edges[index]) & (probability <= edges[index + 1])
#         if mask.any():
#             confidence.append(float(probability[mask].mean()))
#             observed.append(float(y[mask].mean()))
#             counts.append(int(mask.sum()))
#     return np.asarray(confidence), np.asarray(observed), np.asarray(counts)


# def figure_calibration(
#     store: PredictionStore,
#     out_dir: Path,
#     *,
#     formats: Sequence[str],
#     dpi: int,
# ) -> None:
#     fig, axes = plt.subplots(2, 3, figsize=(TWO_COLUMN, 4.55))
#     fig.subplots_adjust(
#         left=0.075, right=0.992, top=0.965, bottom=0.105, wspace=0.31, hspace=0.42
#     )
#     letters = "abcdef"
#     for panel, (ax, fold) in enumerate(zip(axes.ravel(), FOLDS)):
#         ax.plot(
#             [0, 1],
#             [0, 1],
#             color="#A8A8A8",
#             linestyle=(0, (3, 2)),
#             linewidth=0.8,
#             zorder=1,
#         )
#         legend_handles: list[Line2D] = []
#         for method in METHODS:
#             cell = store.cell(method, fold)
#             if not cell:
#                 continue
#             y = np.concatenate([p.y2 for p in cell])
#             probability = np.concatenate([p.probability for p in cell])
#             x, observed, _ = calibration_points(y, probability, n_bins=15)
#             ece = expected_calibration_error(y, probability, n_bins=15)
#             ax.plot(
#                 x,
#                 observed,
#                 color=method.color,
#                 linestyle="-",
#                 linewidth=1.55 if method.key == "casg" else 1.25,
#                 zorder=4 if method.key == "casg" else 3,
#             )
#             legend_handles.append(
#                 Line2D(
#                     [],
#                     [],
#                     color=method.color,
#                     linestyle="-",
#                     linewidth=1.55 if method.key == "casg" else 1.25,
#                     label=(
#                         f"{method.label}: {ece:.3f}"
#                         if method.key == "casg"
#                         else f"{method.short}: {ece:.3f}"
#                     ),
#                 )
#             )
#         _style_axis(ax)
#         ax.set_ylim(0.0, 1.01)
#         ax.set_xlabel("Predicted $p$(abnormal)")
#         ax.set_ylabel("Observed frequency")
#         ax.set_title(f"({letters[panel]}) {FOLD_LABEL[fold]}", pad=3.0)
#         ax.legend(
#             handles=legend_handles,
#             title="ECE (15 bins)",
#             loc="upper left",
#             fontsize=5.4,
#             title_fontsize=5.5,
#             handlelength=1.8,
#             labelspacing=0.22,
#             borderpad=0.15,
#         )
#     save_figure(fig, out_dir, "fig7_calibration", formats=formats, dpi=dpi)


# def _stable_seed(base_seed: int, *parts: str) -> int:
#     payload = "|".join(parts).encode("utf-8")
#     value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
#     return int((value ^ int(base_seed)) % (2**63 - 1))


# def patient_clustered_ci(
#     y: np.ndarray,
#     probability: np.ndarray,
#     patient: np.ndarray,
#     *,
#     n_bootstrap: int,
#     random_seed: int,
#     metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
# ) -> tuple[float, float, float, int]:
#     """Percentile interval from resampling physical patients with replacement."""

#     y = np.asarray(y).astype(int)
#     probability = np.asarray(probability, dtype=np.float64)
#     patient = _as_text(patient)
#     unique, inverse = np.unique(patient, return_inverse=True)
#     if unique.size < 2:
#         raise InputError("patient-clustered CI requires at least two patients")
#     groups = [np.flatnonzero(inverse == index) for index in range(unique.size)]
#     observed = float(metric(y, probability))
#     rng = np.random.default_rng(random_seed)
#     samples: list[float] = []
#     attempts = 0
#     maximum_attempts = max(n_bootstrap * 5, n_bootstrap + 100)
#     while len(samples) < n_bootstrap and attempts < maximum_attempts:
#         attempts += 1
#         draw = rng.integers(0, unique.size, size=unique.size)
#         indices = np.concatenate([groups[index] for index in draw])
#         if np.unique(y[indices]).size != 2:
#             continue
#         samples.append(float(metric(y[indices], probability[indices])))
#     required = min(n_bootstrap, max(20, int(math.ceil(0.9 * n_bootstrap))))
#     if len(samples) < required:
#         raise InputError(
#             f"only {len(samples)} valid clustered bootstrap samples out of "
#             f"{n_bootstrap} requested"
#         )
#     low, high = np.percentile(np.asarray(samples), [2.5, 97.5])
#     return observed, float(low), float(high), int(unique.size)


# def _shared_forest_limits(rows: Sequence[dict[str, object]]) -> tuple[float, float]:
#     low = min(float(row["ci_low"]) for row in rows)
#     high = max(float(row["ci_high"]) for row in rows)
#     lower = max(0.0, math.floor((low - 0.015) / 0.05) * 0.05)
#     upper = min(1.0, math.ceil((high + 0.015) / 0.05) * 0.05)
#     if upper - lower < 0.25:
#         middle = (upper + lower) / 2.0
#         lower = max(0.0, middle - 0.125)
#         upper = min(1.0, middle + 0.125)
#     return lower, upper


# def _forest_plot(
#     rows: Sequence[dict[str, object]],
#     *,
#     title: str,
#     row_label: dict[str, str],
#     row_order: Sequence[str],
#     out_dir: Path,
#     stem: str,
#     formats: Sequence[str],
#     dpi: int,
#     highlight: str | None = None,
# ) -> None:
#     fig, ax = plt.subplots(figsize=(TWO_COLUMN, 3.25))
#     fig.subplots_adjust(left=0.235, right=0.985, top=0.76, bottom=0.19)
#     y_base = np.arange(len(row_order))[::-1].astype(float)
#     offsets = np.linspace(0.24, -0.24, len(METHODS))
#     lookup = {(str(r["method_key"]), str(r["arm"])): r for r in rows}
#     if highlight is not None and highlight in row_order:
#         index = row_order.index(highlight)
#         y = y_base[index]
#         ax.axhspan(y - 0.48, y + 0.48, color="#D55E00", alpha=0.07, zorder=0)
#     for method, offset in zip(METHODS, offsets):
#         x, y, left, right = [], [], [], []
#         for index, arm in enumerate(row_order):
#             row = lookup.get((method.key, arm))
#             if row is None:
#                 continue
#             value = float(row["auroc"])
#             x.append(value)
#             y.append(y_base[index] + offset)
#             left.append(value - float(row["ci_low"]))
#             right.append(float(row["ci_high"]) - value)
#         if not x:
#             continue
#         ax.errorbar(
#             x,
#             y,
#             xerr=np.asarray([left, right]),
#             fmt=method.marker,
#             color=method.color,
#             markersize=4.2,
#             markerfacecolor=method.color if method.key == "casg" else "white",
#             markeredgewidth=1.05,
#             linewidth=1.25,
#             capsize=2.1,
#             capthick=1.0,
#             label=method.label,
#             zorder=4 if method.key == "casg" else 3,
#         )
#     ax.set_yticks(y_base)
#     ax.set_yticklabels([row_label[arm] for arm in row_order])
#     ax.set_ylim(-0.6, len(row_order) - 0.4)
#     ax.set_xlim(*_shared_forest_limits(rows))
#     ax.xaxis.set_major_locator(MultipleLocator(0.05))
#     ax.grid(axis="x")
#     ax.set_xlabel("Binary AUROC")
#     fig.suptitle(title, fontsize=8.5, y=0.985)
#     fig.legend(
#         handles=_method_handles(),
#         loc="upper center",
#         bbox_to_anchor=(0.5, 0.925),
#         ncol=4,
#         columnspacing=1.15,
#         handlelength=2.3,
#     )
#     save_figure(fig, out_dir, stem, formats=formats, dpi=dpi)


# def figure_lodo(
#     store: PredictionStore,
#     out_dir: Path,
#     *,
#     n_bootstrap: int,
#     bootstrap_seed: int,
#     formats: Sequence[str],
#     dpi: int,
# ) -> None:
#     rows: list[dict[str, object]] = []
#     for fold in FOLDS:
#         for method in METHODS:
#             cell = store.cell(method, fold)
#             if not cell:
#                 continue
#             y, probability, patient = store.ensemble(method, fold)
#             value, low, high, n_patients = patient_clustered_ci(
#                 y,
#                 probability,
#                 patient,
#                 n_bootstrap=n_bootstrap,
#                 random_seed=_stable_seed(bootstrap_seed, "main", method.key, fold),
#             )
#             rows.append(
#                 {
#                     "arm": fold,
#                     "arm_label": FOLD_LABEL[fold],
#                     "method_key": method.key,
#                     "method": method.label,
#                     "auroc": value,
#                     "ci_low": low,
#                     "ci_high": high,
#                     "n_clips": y.size,
#                     "n_patients": n_patients,
#                     "seeds": ";".join(map(str, store.seeds)),
#                     "bootstrap_replicates": n_bootstrap,
#                 }
#             )
#     if not rows:
#         print("[skip] LODO figure: no complete cells")
#         return
#     write_csv(out_dir / "lodo_patient_clustered_ci.csv", rows)
#     _forest_plot(
#         rows,
#         title="Unseen-device and external-site discrimination (seed ensemble, 95% CI)",
#         row_label=FOLD_LABEL,
#         row_order=FOLDS,
#         out_dir=out_dir,
#         stem="fig3_lodo_forest",
#         formats=formats,
#         dpi=dpi,
#     )


# def figure_external(
#     store: PredictionStore,
#     out_dir: Path,
#     *,
#     formats: Sequence[str],
#     dpi: int,
# ) -> None:
#     fold = "none"
#     grid = np.linspace(0.0, 1.0, 401)
#     fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMN, 2.42))
#     fig.subplots_adjust(left=0.075, right=0.992, top=0.80, bottom=0.235, wspace=0.34)
#     for method in METHODS:
#         cell = store.cell(method, fold)
#         if not cell:
#             continue
#         _plot_mean_curve(axes[0], grid, _roc_samples(cell, grid), method)
#         _plot_mean_curve(axes[1], grid, _pr_samples(cell, grid), method)
#         pooled_y = np.concatenate([p.y2 for p in cell])
#         pooled_probability = np.concatenate([p.probability for p in cell])
#         x, observed, _ = calibration_points(pooled_y, pooled_probability, n_bins=15)
#         axes[2].plot(
#             x,
#             observed,
#             color=method.color,
#             linestyle="-",
#             linewidth=1.55 if method.key == "casg" else 1.25,
#         )

#     axes[0].plot([0, 1], [0, 1], color="#A8A8A8", ls=(0, (3, 2)), lw=0.8)
#     axes[0].set_title("(a) ROC")
#     axes[0].set_xlabel("False-positive rate")
#     axes[0].set_ylabel("True-positive rate")

#     reference = store.cell(METHODS[0], fold)
#     if reference:
#         axes[1].axhline(
#             float(reference[0].y2.mean()), color="#A8A8A8", ls=(0, (3, 2)), lw=0.8
#         )
#     axes[1].set_title("(b) Precision–recall")
#     axes[1].set_xlabel("Recall")
#     axes[1].set_ylabel("Precision")

#     axes[2].plot([0, 1], [0, 1], color="#A8A8A8", ls=(0, (3, 2)), lw=0.8)
#     axes[2].set_title("(c) Reliability")
#     axes[2].set_xlabel("Predicted $p$(abnormal)")
#     axes[2].set_ylabel("Observed frequency")
#     for ax in axes:
#         _style_axis(ax)
#         ax.set_ylim(0.0, 1.01)
#     fig.suptitle("Frozen Hospital-B stethoscope arm", fontsize=8.7, y=0.965)
#     fig.legend(
#         handles=_curve_handles(),
#         loc="lower center",
#         bbox_to_anchor=(0.5, 0.015),
#         ncol=4,
#         columnspacing=1.15,
#         handlelength=2.5,
#     )
#     save_figure(fig, out_dir, "fig6_external_curves", formats=formats, dpi=dpi)


# # ---------------------------------------------------------------------------
# # Representation diagnostic from saved checkpoints (no training)
# # ---------------------------------------------------------------------------

# TSNE_DEVICE_COLORS = {
#     "AKGC417L": "#0072B2",
#     "Meditron": "#E3A500",
#     "LittC2SE": "#009E73",
#     "Litt3200": "#CC79A7",
#     "smartphone": "#D1495B",
#     "HospitalStethoscope": "#666666",
# }
# TSNE_DEVICE_MARKERS = {
#     "AKGC417L": "o",
#     "Meditron": "s",
#     "LittC2SE": "^",
#     "Litt3200": "D",
#     "smartphone": "P",
#     "HospitalStethoscope": "X",
# }
# TSNE_DEVICE_LABELS = {
#     **FOLD_LABEL,
#     "HospitalStethoscope": "Littmann CORE",
# }


# def _method_by_key(key: str) -> MethodSpec:
#     try:
#         return next(method for method in METHODS if method.key == key)
#     except StopIteration as exc:  # pragma: no cover - guarded by argparse
#         raise InputError(f"unknown method key: {key}") from exc


# def _checkpoint_path(root: Path, method: MethodSpec, fold: str, seed: int) -> Path:
#     pred_path = root / method.main_pattern.format(fold=fold, seed=seed)
#     name = pred_path.name.replace("preds_", "ckpt_", 1)
#     return pred_path.with_name(name).with_suffix(".pt")


# def _repo_path(root: Path, raw: object) -> Path:
#     path = Path(str(raw)).expanduser()
#     return path if path.is_absolute() else root / path


# def _balanced_tsne_frame(
#     root: Path,
#     cfg: object,
#     *,
#     per_device: int,
#     random_seed: int,
# ) -> tuple[object, list[Path]]:
#     """Build one device/pathology-balanced development subset.

#     The subset is constructed once and reused for every checkpoint.  Frozen
#     Hospital-B sources are removed according to the checkpoint configuration.
#     Balancing both device and binary pathology prevents disease prevalence
#     from masquerading as acquisition clustering.
#     """

#     try:
#         import pandas as pd
#         from src.data import manifest as manifest_api
#     except ImportError as exc:  # pragma: no cover - server dependency
#         raise InputError(f"t-SNE needs pandas and the project data package: {exc}") from exc

#     frames: list[object] = []
#     paths: list[Path] = []
#     for field in ("train_manifest", "icbhi_manifest", "extra_manifest"):
#         raw = str(getattr(cfg.data, field, "") or "")
#         if not raw:
#             if field == "extra_manifest":
#                 continue
#             raise InputError(f"checkpoint config has no data.{field}")
#         path = _repo_path(root, raw).resolve()
#         if not path.is_file():
#             raise InputError(f"t-SNE manifest is missing: {path}")
#         frames.append(manifest_api.load_manifest(str(path)))
#         paths.append(path)

#     frame = pd.concat(frames, ignore_index=True)
#     frame = manifest_api.add_cohort_uid(manifest_api.add_patient_uid(frame))
#     blocked = [
#         str(value).lower()
#         for value in (getattr(cfg.data, "dev_exclude_sources", []) or [])
#     ]
#     if blocked:
#         source = frame["source"].astype(str).str.lower()
#         keep = ~source.apply(lambda value: any(token in value for token in blocked))
#         frame = frame.loc[keep].reset_index(drop=True)
#     if frame.empty:
#         raise InputError("no development samples remain for t-SNE")

#     frame["label2"] = frame["label2"].astype(int)
#     devices = sorted(frame["device"].astype(str).unique())
#     if len(devices) < 2:
#         raise InputError("t-SNE requires at least two recording devices")
#     strata = frame.groupby(["device", "label2"], observed=True).size()
#     required = [(device, label) for device in devices for label in (0, 1)]
#     absent = [key for key in required if key not in strata.index]
#     if absent:
#         raise InputError(
#             "cannot device/pathology-balance t-SNE; empty strata: "
#             + ", ".join(map(str, absent))
#         )
#     each = min(per_device // 2, min(int(strata.loc[key]) for key in required))
#     if each < 5:
#         raise InputError(
#             f"too few clips in a device/pathology stratum for t-SNE ({each})"
#         )

#     rng = np.random.default_rng(random_seed)
#     parts = []
#     for device, label in required:
#         group = frame[
#             (frame["device"].astype(str) == device) & (frame["label2"] == label)
#         ]
#         chosen = rng.choice(len(group), size=each, replace=False)
#         parts.append(group.iloc[chosen])
#     subset = pd.concat(parts, ignore_index=True)
#     print(
#         f"[t-SNE] balanced subset: {len(devices)} devices x 2 pathology groups "
#         f"x {each} clips = {len(subset)}"
#     )
#     return subset, paths


# def _restore_checkpoint_model(checkpoint: Path, torch_device: object):
#     try:
#         import torch
#     except ImportError as exc:  # pragma: no cover - server dependency
#         raise InputError(f"t-SNE checkpoint extraction needs PyTorch: {exc}") from exc
#     try:
#         from src.utils import Config
#     except ImportError as exc:  # pragma: no cover - repository layout
#         raise InputError(
#             f"cannot import the repository's src package from {checkpoint}: {exc}"
#         ) from exc

#     try:
#         saved = torch.load(checkpoint, map_location="cpu")
#     except Exception as exc:
#         raise InputError(f"cannot load checkpoint {checkpoint}: {exc}") from exc
#     required = {"state_dict", "config", "device_map"}
#     missing = required - set(saved)
#     if missing:
#         raise InputError(f"{checkpoint}: missing checkpoint fields {sorted(missing)}")
#     cfg = Config(saved["config"])
#     # All learned parameters are restored below.  Disabling constructor-time
#     # pretraining avoids an unnecessary network request to HuggingFace/timm.
#     cfg.model.timm_pretrained = False
#     state = saved["state_dict"]
#     device_map = saved["device_map"]
#     if any(key.startswith("path_proj.") for key in state) and any(
#         key.startswith("style_enc.") for key in state
#     ):
#         from src.casg.model import CASGNet

#         model = CASGNet(cfg, n_devices=max(2, len(device_map)))
#     else:
#         from src.models import DeviceAgnosticNet

#         model = DeviceAgnosticNet(cfg, n_devices=max(2, len(device_map)))
#     try:
#         model.load_state_dict(state, strict=True)
#     except RuntimeError as exc:
#         raise InputError(f"{checkpoint}: state-dict restore failed: {exc}") from exc
#     return model.to(torch_device).eval(), cfg, device_map


# def _extract_checkpoint_features(
#     model: object,
#     cfg: object,
#     device_map: dict,
#     subset: object,
#     *,
#     root: Path,
#     torch_device: object,
#     batch_size: int,
# ) -> np.ndarray:
#     try:
#         import torch
#         from src.data.dataset import make_loader
#     except ImportError as exc:  # pragma: no cover - server dependency
#         raise InputError(f"t-SNE feature extraction dependency failed: {exc}") from exc

#     data_root = _repo_path(root, cfg.data.data_root).resolve()
#     loader = make_loader(
#         subset,
#         cfg,
#         device_map,
#         str(data_root),
#         train=False,
#         batch_size=batch_size,
#     )
#     chunks: list[np.ndarray] = []
#     indices: list[np.ndarray] = []
#     try:
#         with torch.inference_mode():
#             for batch in loader:
#                 output = model(batch["mel"].to(torch_device))
#                 if not isinstance(output, dict) or "feat" not in output:
#                     raise InputError("model forward pass did not return the 'feat' tensor")
#                 chunks.append(output["feat"].detach().cpu().numpy())
#                 indices.append(batch["index"].detach().cpu().numpy().astype(int))
#     except torch.cuda.OutOfMemoryError as exc:
#         raise InputError(
#             "CUDA ran out of memory during t-SNE feature extraction; rerun with "
#             "--torch-device cpu or a smaller --tsne-batch-size"
#         ) from exc
#     features = np.concatenate(chunks, axis=0)
#     order = np.concatenate(indices)
#     if not np.array_equal(np.sort(order), np.arange(len(subset))):
#         raise InputError("t-SNE loader indices are missing, duplicated, or misaligned")
#     aligned = np.empty_like(features)
#     aligned[order] = features
#     if not np.isfinite(aligned).all() or aligned.ndim != 2:
#         raise InputError("checkpoint produced malformed t-SNE features")
#     return aligned


# def _load_tsne_cache(
#     path: Path, method_keys: Sequence[str]
# ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
#     if not path.is_file():
#         raise InputError(f"t-SNE feature cache is missing: {path}")
#     try:
#         with np.load(path, allow_pickle=False) as saved:
#             required = {"device", "patient", "pathology"} | {
#                 f"features_{key}" for key in method_keys
#             }
#             missing = required - set(saved.files)
#             if missing:
#                 raise InputError(f"{path}: missing cache arrays {sorted(missing)}")
#             metadata = {
#                 "device": _as_text(saved["device"]).copy(),
#                 "patient": _as_text(saved["patient"]).copy(),
#                 "pathology": np.asarray(saved["pathology"]).astype(int).reshape(-1),
#             }
#             features = {
#                 key: np.asarray(saved[f"features_{key}"], dtype=np.float32).copy()
#                 for key in method_keys
#             }
#     except (OSError, ValueError, KeyError) as exc:
#         raise InputError(f"cannot read t-SNE cache {path}: {exc}") from exc
#     n = metadata["device"].size
#     if any(array.size != n for array in metadata.values()):
#         raise InputError(f"{path}: metadata length mismatch")
#     for key, array in features.items():
#         if array.ndim != 2 or array.shape[0] != n or not np.isfinite(array).all():
#             raise InputError(f"{path}: malformed features_{key} array {array.shape}")
#     return features, metadata


# def _device_probe(
#     features: np.ndarray,
#     labels: np.ndarray,
#     patient: np.ndarray,
#     *,
#     random_seed: int,
# ) -> tuple[float, float, int]:
#     """Patient-grouped linear probe; balanced accuracy has chance 1/K."""

#     from sklearn.linear_model import LogisticRegression
#     from sklearn.metrics import balanced_accuracy_score
#     from sklearn.model_selection import GroupShuffleSplit
#     from sklearn.pipeline import make_pipeline
#     from sklearn.preprocessing import StandardScaler

#     classes = np.unique(labels)
#     splitter = GroupShuffleSplit(n_splits=5, test_size=0.30, random_state=random_seed)
#     scores: list[float] = []
#     for train, test in splitter.split(features, labels, groups=patient):
#         if np.unique(labels[train]).size != classes.size or np.unique(labels[test]).size != classes.size:
#             continue
#         classifier = make_pipeline(
#             StandardScaler(),
#             LogisticRegression(
#                 max_iter=2500,
#                 class_weight="balanced",
#                 solver="lbfgs",
#                 random_state=random_seed,
#             ),
#         )
#         classifier.fit(features[train], labels[train])
#         scores.append(float(balanced_accuracy_score(labels[test], classifier.predict(features[test]))))
#     if len(scores) < 3:
#         raise InputError("fewer than three valid patient-grouped device-probe splits")
#     return float(np.mean(scores)), float(np.std(scores)), len(scores)


# def _tsne_coordinates(
#     features: np.ndarray, *, perplexity: float, random_seed: int
# ) -> tuple[np.ndarray, float]:
#     import inspect

#     from sklearn.decomposition import PCA
#     from sklearn.manifold import TSNE
#     from sklearn.preprocessing import normalize

#     normalized = normalize(features, norm="l2")
#     components = min(50, normalized.shape[1], normalized.shape[0] - 1)
#     reduced = PCA(n_components=components, random_state=random_seed).fit_transform(normalized)
#     used_perplexity = min(float(perplexity), max(5.0, (len(reduced) - 1) / 3.0))
#     options: dict[str, object] = {
#         "n_components": 2,
#         "init": "pca",
#         "learning_rate": "auto",
#         "perplexity": used_perplexity,
#         "random_state": random_seed,
#     }
#     # scikit-learn renamed n_iter to max_iter in 1.5.  Support the repository's
#     # declared >=1.3 range without changing the numerical iteration budget.
#     iteration_key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
#     options[iteration_key] = 1000
#     coordinates = TSNE(**options).fit_transform(reduced)
#     return coordinates, used_perplexity


# def figure_tsne(
#     root: Path,
#     out_dir: Path,
#     *,
#     method_keys: Sequence[str],
#     fold: str,
#     seed: int,
#     per_device: int,
#     perplexity: float,
#     random_seed: int,
#     batch_size: int,
#     torch_device_name: str,
#     cache_path: Path | None,
#     formats: Sequence[str],
#     dpi: int,
# ) -> set[Path]:
#     """Extract real checkpoint features and draw a deterministic t-SNE panel."""

#     methods = [_method_by_key(key) for key in method_keys]
#     inputs: set[Path] = set()
#     if cache_path is not None:
#         cache_path = cache_path.resolve()
#         features, metadata = _load_tsne_cache(cache_path, method_keys)
#         inputs.add(cache_path)
#         print(f"[t-SNE] using frozen feature cache: {cache_path}")
#     else:
#         try:
#             import torch
#         except ImportError as exc:  # pragma: no cover - server dependency
#             raise InputError(f"t-SNE checkpoint extraction needs PyTorch: {exc}") from exc
#         if torch_device_name == "auto":
#             torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             torch_device = torch.device(torch_device_name)
#         if torch_device.type == "cuda" and not torch.cuda.is_available():
#             raise InputError("--torch-device requests CUDA, but CUDA is unavailable")
#         features = {}
#         subset = None
#         manifest_paths: list[Path] = []
#         reference_signature = None
#         for method in methods:
#             checkpoint = _checkpoint_path(root, method, fold, seed).resolve()
#             if not checkpoint.is_file():
#                 raise InputError(f"missing t-SNE checkpoint: {checkpoint}")
#             model, cfg, device_map = _restore_checkpoint_model(checkpoint, torch_device)
#             signature = (
#                 int(cfg.audio.sample_rate),
#                 int(cfg.audio.n_mels),
#                 int(cfg.audio.hop_length),
#                 float(cfg.audio.clip_seconds),
#                 float(getattr(cfg.audio, "fmax", 0.0)),
#             )
#             if reference_signature is None:
#                 reference_signature = signature
#                 subset, manifest_paths = _balanced_tsne_frame(
#                     root, cfg, per_device=per_device, random_seed=random_seed
#                 )
#             elif signature != reference_signature:
#                 raise InputError(
#                     f"{checkpoint}: preprocessing differs from the first t-SNE checkpoint"
#                 )
#             assert subset is not None
#             features[method.key] = _extract_checkpoint_features(
#                 model,
#                 cfg,
#                 device_map,
#                 subset,
#                 root=root,
#                 torch_device=torch_device,
#                 batch_size=batch_size,
#             )
#             inputs.add(checkpoint)
#             del model
#             if torch_device.type == "cuda":
#                 torch.cuda.empty_cache()
#         inputs.update(manifest_paths)
#         metadata = {
#             "device": subset["device"].astype(str).to_numpy(),
#             "patient": subset["cohort_uid"].astype(str).to_numpy(),
#             "pathology": subset["label2"].astype(int).to_numpy(),
#         }
#         cache_path = out_dir / "fig9_tsne_features.npz"
#         cache_path.parent.mkdir(parents=True, exist_ok=True)
#         payload: dict[str, np.ndarray] = dict(metadata)
#         payload.update(
#             {f"features_{key}": np.asarray(value, dtype=np.float32) for key, value in features.items()}
#         )
#         np.savez_compressed(cache_path, **payload)
#         print(f"[write] {cache_path}")

#     device = metadata["device"].astype(str)
#     patient = metadata["patient"].astype(str)
#     devices = [key for key in FOLDS[:-1] if key in set(device)]
#     devices.extend(sorted(set(device) - set(devices)))
#     encoded = np.asarray([devices.index(value) for value in device], dtype=int)
#     coordinates: dict[str, np.ndarray] = {}
#     probe_rows: list[dict[str, object]] = []
#     for method in methods:
#         coordinates[method.key], used_perplexity = _tsne_coordinates(
#             features[method.key], perplexity=perplexity, random_seed=random_seed
#         )
#         mean, sd, splits = _device_probe(
#             features[method.key], encoded, patient, random_seed=random_seed
#         )
#         probe_rows.append(
#             {
#                 "method_key": method.key,
#                 "method": method.label,
#                 "held_out_checkpoint": fold,
#                 "seed": seed,
#                 "n_clips": len(device),
#                 "n_patients": np.unique(patient).size,
#                 "n_devices": len(devices),
#                 "perplexity": used_perplexity,
#                 "device_probe_balanced_accuracy_mean": mean,
#                 "device_probe_balanced_accuracy_sd": sd,
#                 "patient_group_splits": splits,
#                 "balanced_chance": 1.0 / len(devices),
#             }
#         )

#     n_columns = min(2, len(methods))
#     n_rows = int(math.ceil(len(methods) / n_columns))
#     figure_height = 3.05 if n_rows == 1 else 5.45
#     fig, axes = plt.subplots(
#         n_rows,
#         n_columns,
#         figsize=(TWO_COLUMN, figure_height),
#         squeeze=False,
#     )
#     fig.subplots_adjust(
#         left=0.035,
#         right=0.80,
#         top=0.84 if n_rows == 1 else 0.91,
#         bottom=0.20 if n_rows == 1 else 0.10,
#         wspace=0.08,
#         hspace=0.39 if n_rows > 1 else 0.08,
#     )
#     flat_axes = axes.ravel()
#     for panel, method in enumerate(methods):
#         ax = flat_axes[panel]
#         z = coordinates[method.key]
#         for device_name in devices:
#             mask = device == device_name
#             ax.scatter(
#                 z[mask, 0],
#                 z[mask, 1],
#                 s=6.0,
#                 alpha=0.62,
#                 linewidths=0.0,
#                 marker=TSNE_DEVICE_MARKERS.get(device_name, "o"),
#                 color=TSNE_DEVICE_COLORS.get(device_name, "#666666"),
#                 rasterized=False,
#             )
#         row = probe_rows[panel]
#         ax.set_title(f"({chr(97 + panel)}) {method.label}", pad=4.0)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         for spine in ax.spines.values():
#             spine.set_visible(True)
#             spine.set_color("#9A9A9A")
#             spine.set_linewidth(0.6)
#         ax.text(
#             0.5,
#             -0.075 if n_rows == 1 else -0.09,
#             "patient-grouped device probe: "
#             f"{100 * float(row['device_probe_balanced_accuracy_mean']):.1f}% "
#             f"$\\pm$ {100 * float(row['device_probe_balanced_accuracy_sd']):.1f}%",
#             transform=ax.transAxes,
#             ha="center",
#             va="top",
#             fontsize=6.5 if n_rows == 1 else 5.9,
#         )
#     for ax in flat_axes[len(methods) :]:
#         ax.remove()
#     legend_handles = [
#         Line2D(
#             [],
#             [],
#             linestyle="none",
#             marker=TSNE_DEVICE_MARKERS.get(name, "o"),
#             color=TSNE_DEVICE_COLORS.get(name, "#666666"),
#             markersize=4.4,
#             label=TSNE_DEVICE_LABELS.get(name, name),
#         )
#         for name in devices
#     ]
#     fig.legend(
#         handles=legend_handles,
#         loc="center left",
#         bbox_to_anchor=(0.805, 0.49),
#         labelspacing=0.55,
#         handletextpad=0.35,
#     )
#     fig.suptitle(
#         "Balanced representation diagnostic "
#         f"(held-out {TSNE_DEVICE_LABELS.get(fold, fold)}, seed {seed})",
#         fontsize=8.7,
#         y=0.965,
#     )
#     write_csv(out_dir / "tsne_device_probe_metrics.csv", probe_rows)
#     save_figure(fig, out_dir, "fig9_tsne", formats=formats, dpi=dpi)
#     return inputs


# def _normalize_summary_method(name: str) -> str | None:
#     token = re.sub(r"[^a-z0-9]+", "", name.lower())
#     if token.startswith("erm"):
#         return "erm"
#     if token.startswith("mixstyle"):
#         return "mixstyle"
#     if token.startswith("cfonly"):
#         return "cf"
#     if token == "cf":
#         return "cf"
#     if token.startswith("casg"):
#         return "casg"
#     return None


# def validate_frozen_phone_summary(
#     root: Path,
#     rows: Sequence[dict[str, object]],
#     *,
#     tolerance: float = 5e-4,
#     allow_missing: bool = False,
# ) -> None:
#     path = root / "runs/frozen_phone/frozen_phone_summary.csv"
#     if not path.is_file():
#         print("[note] frozen_phone_summary.csv absent; per-clip files remain authoritative")
#         return
#     expected = {
#         (str(row["method_key"]), str(row["arm"]), int(row["seed"])): float(row["auroc"])
#         for row in rows
#     }
#     checked = 0
#     with path.open(newline="", encoding="utf-8-sig") as handle:
#         for item in csv.DictReader(handle):
#             method = _normalize_summary_method(str(item.get("method", "")))
#             fold = str(item.get("train_fold", item.get("held_out", "")))
#             seed_text = str(item.get("seed", ""))
#             if method is None or not seed_text:
#                 continue
#             key = (method, fold, int(float(seed_text)))
#             if key not in expected:
#                 continue
#             recorded = float(item["auroc"])
#             if abs(recorded - expected[key]) > tolerance:
#                 raise InputError(
#                     f"{path}: AUROC mismatch for {key}: CSV={recorded:.6f}, "
#                     f"recomputed={expected[key]:.6f}"
#                 )
#             checked += 1
#     if checked != len(expected) and not allow_missing:
#         raise InputError(
#             f"{path}: verified {checked} of {len(expected)} expected per-seed cells"
#         )
#     print(f"[check] frozen-phone summary agrees with {checked} recomputed cells")


# def figure_phone(
#     store: PredictionStore,
#     out_dir: Path,
#     *,
#     n_bootstrap: int,
#     bootstrap_seed: int,
#     formats: Sequence[str],
#     dpi: int,
# ) -> None:
#     rows: list[dict[str, object]] = []
#     per_seed_rows: list[dict[str, object]] = []
#     for fold in FOLDS:
#         for method in METHODS:
#             seeded_cell = store.seeded_cell(method, fold, phone=True)
#             if not seeded_cell:
#                 continue
#             for seed, pred in seeded_cell:
#                 per_seed_rows.append(
#                     {
#                         "arm": fold,
#                         "method_key": method.key,
#                         "method": method.label,
#                         "seed": seed,
#                         "auroc": roc_auc_score(pred.y2, pred.probability),
#                         "auprc": average_precision_score(pred.y2, pred.probability),
#                         "ece_15bin": expected_calibration_error(
#                             pred.y2, pred.probability, n_bins=15
#                         ),
#                         "n_clips": pred.n,
#                         "n_patients": np.unique(pred.patient).size,
#                     }
#                 )
#             y, probability, patient = store.ensemble(method, fold, phone=True)
#             value, low, high, n_patients = patient_clustered_ci(
#                 y,
#                 probability,
#                 patient,
#                 n_bootstrap=n_bootstrap,
#                 random_seed=_stable_seed(bootstrap_seed, "phone", method.key, fold),
#             )
#             rows.append(
#                 {
#                     "arm": fold,
#                     "arm_label": PHONE_ROW_LABEL[fold].replace("\n", " "),
#                     "method_key": method.key,
#                     "method": method.label,
#                     "auroc": value,
#                     "ci_low": low,
#                     "ci_high": high,
#                     "n_clips": y.size,
#                     "n_patients": n_patients,
#                     "seeds": ";".join(str(seed) for seed, _ in seeded_cell),
#                     "bootstrap_replicates": n_bootstrap,
#                 }
#             )
#     if not rows:
#         print("[skip] frozen-phone figure: no per-clip prediction files")
#         return
#     validate_frozen_phone_summary(
#         store.root, per_seed_rows, allow_missing=store.allow_missing
#     )
#     write_csv(out_dir / "frozen_phone_per_seed_metrics.csv", per_seed_rows)
#     write_csv(out_dir / "frozen_phone_patient_clustered_ci.csv", rows)
#     _forest_plot(
#         rows,
#         title="Frozen Hospital-B smartphone arm (seed ensemble, 95% CI)",
#         row_label=PHONE_ROW_LABEL,
#         row_order=FOLDS,
#         out_dir=out_dir,
#         stem="fig8_frozen_phone",
#         formats=formats,
#         dpi=dpi,
#         highlight="smartphone",
#     )


# def _method_for_checkpoint(name: str) -> MethodSpec | None:
#     stem = Path(name).stem
#     if stem.endswith("_cf_physics"):
#         return next(m for m in METHODS if m.key == "cf")
#     if stem.endswith("_casg_lite"):
#         return next(m for m in METHODS if m.key == "casg")
#     if stem.startswith("ckpt_mixstyle_ast_") and stem.endswith("_phys"):
#         return next(m for m in METHODS if m.key == "mixstyle")
#     if stem.startswith("ckpt_erm_ast_") and stem.endswith("_phys"):
#         return next(m for m in METHODS if m.key == "erm")
#     return None


# def figure_mechanism(
#     root: Path,
#     out_dir: Path,
#     *,
#     seeds: Sequence[int],
#     allow_missing: bool,
#     cpd_csv: Path | None,
#     formats: Sequence[str],
#     dpi: int,
#     render: bool = True,
# ) -> Path | None:
#     path = (cpd_csv or (root / "runs/cpd_report.csv")).resolve()
#     if not path.is_file():
#         if allow_missing:
#             print(f"[skip] mechanism figure: missing {path}")
#             return None
#         raise InputError(f"missing mechanism report: {path}")
#     records: dict[tuple[str, str, int], dict[str, object]] = {}
#     with path.open(newline="", encoding="utf-8-sig") as handle:
#         for item in csv.DictReader(handle):
#             method = _method_for_checkpoint(str(item.get("ckpt", "")))
#             fold = str(item.get("held_out", ""))
#             seed_match = re.search(r"_seed(\d+)(?:_|\.)", str(item.get("ckpt", "")))
#             if method is None or fold not in FOLDS or not seed_match:
#                 continue
#             seed = int(seed_match.group(1))
#             if seed not in seeds:
#                 continue
#             records[(method.key, fold, seed)] = {
#                 "method_key": method.key,
#                 "method": method.label,
#                 "arm": fold,
#                 "seed": seed,
#                 "cpd": float(item["cpd"]),
#                 "cfv": float(item["cfv"]),
#                 "M": int(float(item.get("M", 0) or 0)),
#                 "n_clips": int(float(item.get("n_clips", 0) or 0)),
#             }
#     expected = {(m.key, f, int(s)) for m in METHODS for f in FOLDS for s in seeds}
#     missing = sorted(expected - set(records))
#     if missing and not allow_missing:
#         preview = ", ".join(map(str, missing[:8]))
#         raise InputError(
#             f"{path}: incomplete CPD/CFV grid ({len(missing)} missing), e.g. {preview}"
#         )
#     for fold in FOLDS:
#         matched = [r for key, r in records.items() if key[1] == fold]
#         signatures = {(int(r["M"]), int(r["n_clips"])) for r in matched}
#         if len(signatures) > 1:
#             raise InputError(
#                 f"{path}: inconsistent perturbation count/sample count in fold {fold}: "
#                 f"{sorted(signatures)}"
#             )
#     if not records:
#         print(f"[skip] mechanism figure: no paper-grid rows in {path}")
#         return path
#     if not render:
#         print(f"[check] mechanism grid: {len(records)} valid CPD/CFV rows")
#         return path

#     fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN, 3.05))
#     fig.subplots_adjust(left=0.085, right=0.992, top=0.76, bottom=0.245, wspace=0.30)
#     x = np.arange(len(FOLDS), dtype=float)
#     width = 0.19
#     offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * width
#     summary_rows: list[dict[str, object]] = []
#     for axis, metric, label in zip(axes, ("cpd", "cfv"), ("CPD", "CFV")):
#         for method, offset in zip(METHODS, offsets):
#             means, standard_deviations = [], []
#             for fold in FOLDS:
#                 values = [
#                     float(records[(method.key, fold, int(seed))][metric]) * 1e3
#                     for seed in seeds
#                     if (method.key, fold, int(seed)) in records
#                 ]
#                 means.append(float(np.mean(values)) if values else np.nan)
#                 standard_deviations.append(float(np.std(values)) if values else np.nan)
#                 if values:
#                     summary_rows.append(
#                         {
#                             "metric": label,
#                             "arm": fold,
#                             "method_key": method.key,
#                             "method": method.label,
#                             "mean_x1e3": float(np.mean(values)),
#                             "sd_x1e3": float(np.std(values)),
#                             "n_seeds": len(values),
#                         }
#                     )
#             axis.bar(
#                 x + offset,
#                 means,
#                 width,
#                 yerr=standard_deviations,
#                 color=method.color,
#                 edgecolor="#202020",
#                 linewidth=0.45,
#                 alpha=1.0 if method.key == "casg" else 0.82,
#                 capsize=1.8,
#                 error_kw={"elinewidth": 0.8, "capthick": 0.8},
#                 zorder=3,
#             )
#         axis.set_xticks(x)
#         axis.set_xticklabels([FOLD_LABEL[f] for f in FOLDS], rotation=31, ha="right")
#         axis.set_ylabel(f"{label} ($\times 10^{{-3}}$)")
#         axis.set_title(f"({chr(97 + (0 if metric == 'cpd' else 1))}) {label}")
#         axis.grid(axis="y")
#         axis.set_ylim(bottom=0.0)
#     fig.legend(
#         handles=_method_handles(),
#         loc="upper center",
#         bbox_to_anchor=(0.5, 0.985),
#         ncol=4,
#         columnspacing=1.15,
#         handlelength=2.3,
#     )
#     write_csv(out_dir / "mechanism_summary.csv", summary_rows)
#     save_figure(fig, out_dir, "fig4_mechanism", formats=formats, dpi=dpi)
#     return path


# def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
#     digest = hashlib.sha256()
#     with path.open("rb") as handle:
#         while chunk := handle.read(chunk_size):
#             digest.update(chunk)
#     return digest.hexdigest()


# def write_input_manifest(
#     path: Path,
#     inputs: Iterable[Path],
#     *,
#     root: Path,
#     seeds: Sequence[int],
#     hash_inputs: bool,
# ) -> None:
#     entries = []
#     for item in sorted({p.resolve() for p in inputs}):
#         stat = item.stat()
#         try:
#             relative = str(item.relative_to(root.resolve()))
#         except ValueError:
#             relative = str(item)
#         record: dict[str, object] = {
#             "path": relative.replace("\\", "/"),
#             "bytes": stat.st_size,
#             "mtime_ns": stat.st_mtime_ns,
#         }
#         if hash_inputs:
#             record["sha256"] = file_sha256(item)
#         entries.append(record)
#     payload = {
#         "generator": "casg_paper_figures.py",
#         "display_method_name": "CASG (ours)",
#         "seeds": list(map(int, seeds)),
#         "inputs": entries,
#         "note": (
#             "Prediction files contain patient/source/device/label row metadata but "
#             "not a unique clip path. Alignment is therefore checked element-wise "
#             "using every saved row metadata field."
#         ),
#     }
#     path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
#     print(f"[write] {path}")


# def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#         description="Generate IEEE-size CASG result figures from frozen per-clip outputs.",
#     )
#     default_root = Path(__file__).resolve().parents[1]
#     parser.add_argument("--root", type=Path, default=default_root)
#     parser.add_argument("--out", type=Path, default=Path("figures_publication"))
#     parser.add_argument(
#         "--only",
#         nargs="+",
#         choices=(
#             "lodo",
#             "roc",
#             "pr",
#             "calibration",
#             "external",
#             "phone",
#             "mechanism",
#             "tsne",
#         ),
#         default=None,
#         help="subset to generate; omit for every figure",
#     )
#     parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
#     parser.add_argument("--bootstrap", type=int, default=2000)
#     parser.add_argument("--bootstrap-seed", type=int, default=20260909)
#     parser.add_argument("--dpi", type=int, default=600)
#     parser.add_argument(
#         "--formats", nargs="+", choices=("pdf", "svg", "png"), default=["pdf", "svg", "png"]
#     )
#     parser.add_argument("--cpd-csv", type=Path, default=None)
#     parser.add_argument(
#         "--tsne-methods",
#         nargs="+",
#         choices=tuple(method.key for method in METHODS),
#         default=["erm", "mixstyle", "cf", "casg"],
#         help="saved checkpoints shown in the representation diagnostic",
#     )
#     parser.add_argument("--tsne-fold", choices=FOLDS, default="smartphone")
#     parser.add_argument("--tsne-seed", type=int, default=0)
#     parser.add_argument("--tsne-per-device", type=int, default=240)
#     parser.add_argument("--tsne-perplexity", type=float, default=30.0)
#     parser.add_argument("--tsne-random-seed", type=int, default=20260909)
#     parser.add_argument("--tsne-batch-size", type=int, default=32)
#     parser.add_argument(
#         "--torch-device",
#         default="auto",
#         help="feature-extraction device: auto, cpu, cuda, or e.g. cuda:1",
#     )
#     parser.add_argument(
#         "--tsne-cache",
#         type=Path,
#         default=None,
#         help="reuse a fig9_tsne_features.npz cache instead of checkpoint inference",
#     )
#     parser.add_argument(
#         "--allow-missing",
#         action="store_true",
#         help="generate available cells instead of failing; do not use for final paper figures",
#     )
#     parser.add_argument("--hash-inputs", action="store_true")
#     parser.add_argument("--check-only", action="store_true")
#     args = parser.parse_args(argv)
#     if args.bootstrap < 20:
#         parser.error("--bootstrap must be at least 20")
#     if not args.seeds or len(set(args.seeds)) != len(args.seeds):
#         parser.error("--seeds must contain unique values")
#     if args.dpi < 300:
#         parser.error("--dpi must be at least 300")
#     if not args.tsne_methods or len(set(args.tsne_methods)) != len(args.tsne_methods):
#         parser.error("--tsne-methods must contain unique method keys")
#     if args.tsne_per_device < 20:
#         parser.error("--tsne-per-device must be at least 20")
#     if args.tsne_perplexity < 5:
#         parser.error("--tsne-perplexity must be at least 5")
#     if args.tsne_batch_size < 1:
#         parser.error("--tsne-batch-size must be positive")
#     return args


# def main(args: argparse.Namespace) -> int:
#     root = args.root.resolve()
#     if not root.is_dir():
#         raise InputError(f"repository root does not exist: {root}")
#     # When invoked as ``python scripts/casg_paper_figures.py``, Python places
#     # ``scripts/`` rather than the repository root at sys.path[0].  Register
#     # the explicit --root before any lazy imports of src.* used by t-SNE.
#     root_text = str(root)
#     if root_text not in sys.path:
#         sys.path.insert(0, root_text)
#     out_dir = args.out if args.out.is_absolute() else root / args.out
#     selected = set(
#         args.only
#         or (
#             "lodo",
#             "roc",
#             "pr",
#             "calibration",
#             "external",
#             "phone",
#             "mechanism",
#             "tsne",
#         )
#     )
#     main_figures = {"lodo", "roc", "pr", "calibration", "external"}
#     store = PredictionStore(root, args.seeds, allow_missing=args.allow_missing)
#     core_rows: list[dict[str, object]] = []

#     if selected & main_figures:
#         count = store.validate_grid(phone=False)
#         print(f"[check] main grid: {count} valid aligned prediction files")
#         if count:
#             core_rows = _core_metric_rows(store)
#     if "phone" in selected:
#         count = store.validate_grid(phone=True)
#         print(f"[check] frozen-phone grid: {count} valid aligned prediction files")

#     cpd_input: Path | None = None
#     tsne_inputs: set[Path] = set()
#     tsne_cache = args.tsne_cache
#     if tsne_cache is not None and not tsne_cache.is_absolute():
#         tsne_cache = root / tsne_cache
#     if args.check_only:
#         if "mechanism" in selected:
#             cpd_input = figure_mechanism(
#                 root,
#                 out_dir,
#                 seeds=args.seeds,
#                 allow_missing=args.allow_missing,
#                 cpd_csv=args.cpd_csv,
#                 formats=args.formats,
#                 dpi=args.dpi,
#                 render=False,
#             )
#         if "tsne" in selected:
#             if tsne_cache is not None:
#                 _load_tsne_cache(tsne_cache.resolve(), args.tsne_methods)
#                 print(f"[check] t-SNE feature cache: {tsne_cache.resolve()}")
#             else:
#                 checkpoints = [
#                     _checkpoint_path(
#                         root, _method_by_key(key), args.tsne_fold, args.tsne_seed
#                     ).resolve()
#                     for key in args.tsne_methods
#                 ]
#                 missing = [path for path in checkpoints if not path.is_file()]
#                 if missing:
#                     raise InputError(f"missing t-SNE checkpoint: {missing[0]}")
#                 print(f"[check] t-SNE checkpoints: {len(checkpoints)} present")
#         print("[check] validation completed; --check-only requested")
#         return 0
#     else:
#         configure_style()
#         if core_rows:
#             write_csv(out_dir / "core_per_seed_metrics.csv", core_rows)
#         if "lodo" in selected:
#             figure_lodo(
#                 store,
#                 out_dir,
#                 n_bootstrap=args.bootstrap,
#                 bootstrap_seed=args.bootstrap_seed,
#                 formats=args.formats,
#                 dpi=args.dpi,
#             )
#         if "roc" in selected:
#             figure_curves(store, out_dir, kind="roc", formats=args.formats, dpi=args.dpi)
#         if "pr" in selected:
#             figure_curves(store, out_dir, kind="pr", formats=args.formats, dpi=args.dpi)
#         if "calibration" in selected:
#             figure_calibration(store, out_dir, formats=args.formats, dpi=args.dpi)
#         if "external" in selected:
#             figure_external(store, out_dir, formats=args.formats, dpi=args.dpi)
#         if "phone" in selected:
#             figure_phone(
#                 store,
#                 out_dir,
#                 n_bootstrap=args.bootstrap,
#                 bootstrap_seed=args.bootstrap_seed,
#                 formats=args.formats,
#                 dpi=args.dpi,
#             )
#         if "mechanism" in selected:
#             cpd_input = figure_mechanism(
#                 root,
#                 out_dir,
#                 seeds=args.seeds,
#                 allow_missing=args.allow_missing,
#                 cpd_csv=args.cpd_csv,
#                 formats=args.formats,
#                 dpi=args.dpi,
#                 render=True,
#             )
#         if "tsne" in selected:
#             tsne_inputs = figure_tsne(
#                 root,
#                 out_dir,
#                 method_keys=args.tsne_methods,
#                 fold=args.tsne_fold,
#                 seed=args.tsne_seed,
#                 per_device=args.tsne_per_device,
#                 perplexity=args.tsne_perplexity,
#                 random_seed=args.tsne_random_seed,
#                 batch_size=args.tsne_batch_size,
#                 torch_device_name=args.torch_device,
#                 cache_path=tsne_cache,
#                 formats=args.formats,
#                 dpi=args.dpi,
#             )

#     manifest_inputs = set(store.used_paths)
#     if cpd_input is not None and cpd_input.is_file():
#         manifest_inputs.add(cpd_input)
#     manifest_inputs.update(tsne_inputs)
#     summary = root / "runs/frozen_phone/frozen_phone_summary.csv"
#     if "phone" in selected and summary.is_file():
#         manifest_inputs.add(summary.resolve())
#     freeze = root / "freeze.json"
#     if freeze.is_file():
#         manifest_inputs.add(freeze.resolve())
#     out_dir.mkdir(parents=True, exist_ok=True)
#     write_input_manifest(
#         out_dir / "figure_input_manifest.json",
#         manifest_inputs,
#         root=root,
#         seeds=args.seeds,
#         hash_inputs=args.hash_inputs,
#     )
#     print(
#         f"[done] Use the PDF files in LaTeX. PNG files are {args.dpi} dpi for "
#         "review; SVG files are editable."
#     )
#     return 0


# if __name__ == "__main__":
#     ARGS = parse_args()
#     try:
#         raise SystemExit(main(ARGS))
#     except InputError as exc:
#         print(f"\nINPUT VALIDATION FAILED: {exc}", file=sys.stderr)
#         raise SystemExit(2)

"""Publication figures for the CASG paper (IEEE JBHI, vector PDF + PNG).

    python scripts/casg_figures.py                 # everything available
    python scripts/casg_figures.py --only roc pr   # a subset
    python scripts/casg_figures.py --list

Figures
  arch        F1  CASG-Lite architecture / data flow            (schematic)
  protocol    F2  LODO folds, cohort keying, frozen site        (schematic)
  forest      F3  per-device AUROC, patient-clustered 95% CI    (preds_*.npz)
  mechanism   F4  CPD & CFV across all six arms                 (cpd_report.csv)
  roc         F5  ROC per arm, mean +/- SD over seeds           (preds_*.npz)
  pr          F6  precision-recall per arm, mean +/- SD         (preds_*.npz)
  calib       F7  reliability diagrams + ECE                    (preds_*.npz)
  phone       F8  reserved Hospital-B smartphone arm            (frozen_phone/)

Every data figure is computed from the frozen prediction dumps; nothing is
hard-coded. A missing input is reported and skipped, never substituted.

Design notes (these matter for print): colour-blind-safe palette, every
series additionally distinguished by line style AND marker so the figure
survives greyscale printing, minimum line width 1.5 pt, no dotted hairlines,
serif text matched to the IEEE body font, and exact IEEE column widths so
nothing is rescaled by LaTeX (rescaling is what makes curves look thin).
"""
import _boot, argparse, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- constants
COL1, COL2 = 3.5, 7.16          # IEEE single / double column width (inches)
OUT = "figures"
DEVS = ["AKGC417L", "Meditron", "LittC2SE", "Litt3200", "smartphone"]
ARMS = DEVS + ["none"]
ARM_LABEL = {"AKGC417L": "AKGC417L", "Meditron": "Meditron",
             "LittC2SE": "LittC2SE", "Litt3200": "Litt3200",
             "smartphone": "Smartphone", "none": "External site"}
SEEDS = [0, 1, 2]

# order = plotting order; ours last so it draws on top
SPECS = [
    ("ERM-Physics",      "runs/erm/preds_erm_ast_{d}_seed{s}_phys.npz"),
    ("MixStyle-Physics", "runs/mixstyle/preds_mixstyle_ast_{d}_seed{s}_phys.npz"),
    ("CF-only-Physics",  "runs/casg/preds_casg_ast_{d}_seed{s}_cf_physics.npz"),
    ("CASG-Lite (ours)", "runs/casg/preds_casg_ast_{d}_seed{s}_casg_lite.npz"),
]
# Four high-contrast academic colours. All curves are SOLID: dashed patterns
# break up at print size and were the cause of the "broken lines" appearance.
# Markers distinguish series in the forest, bar and reliability plots, where
# points are sparse enough for them to read cleanly.
STYLE = {
    "ERM-Physics":      dict(c="#0072B2", ls="-", m="o"),   # blue
    "MixStyle-Physics": dict(c="#E3A500", ls="-", m="s"),   # gold
    "CF-only-Physics":  dict(c="#009E73", ls="-", m="^"),   # green
    "CASG-Lite (ours)": dict(c="#D1495B", ls="-", m="D"),   # red
}
CPD_KEY = [("ERM-Physics", "erm_ast"), ("MixStyle-Physics", "mixstyle_ast"),
           ("CF-only-Physics", "cf_physics"), ("CASG-Lite (ours)", "casg_lite")]


def rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.6,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 600,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.antialiased": True, "patch.antialiased": True,
        "axes.axisbelow": True, "grid.color": "#E6E6E6",
        "grid.linewidth": 0.45,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def save(fig, name, dpi=600):
    """Vector PDF for LaTeX, editable SVG, and a high-resolution PNG."""
    os.makedirs(OUT, exist_ok=True)
    written = []
    for ext in ("pdf", "svg", "png"):
        p = os.path.join(OUT, f"{name}.{ext}")
        kw = {"facecolor": "white", "edgecolor": "none"}
        if ext == "png":
            kw["dpi"] = dpi
        fig.savefig(p, **kw)
        written.append(os.path.basename(p))
    plt.close(fig)
    print(f"  wrote {OUT}/{{{', '.join(written)}}}")


def load(tmpl, dev, seed):
    p = tmpl.format(d=dev, s=seed)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {"y2": z["y2"], "p": z["prob2"][:, 1],
            "pat": z["patient"].astype(str) if "patient" in z else
                   np.arange(len(z["y2"])).astype(str)}


# ------------------------------------------------------------ metric helpers
def _auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    r = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort") + 1.0
    # average ranks over ties
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = r[o[i:j + 1]].mean()
        i = j + 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) /
                 (pos.sum() * neg.sum()))


def roc_curve(y, s, grid):
    o = np.argsort(-s, kind="mergesort")
    y = np.asarray(y)[o]
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    P, N = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    return np.interp(grid, np.r_[0, fp / N, 1], np.r_[0, tp / P, 1])


def pr_curve(y, s, grid):
    o = np.argsort(-s, kind="mergesort")
    y = np.asarray(y)[o]
    tp = np.cumsum(y == 1)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / max((y == 1).sum(), 1)
    # step-interpolate precision at each recall level (max precision at >= r)
    out = np.empty_like(grid)
    for i, r in enumerate(grid):
        m = rec >= r
        out[i] = prec[m].max() if m.any() else prec[-1]
    return out


def cluster_ci(y, s, pat, n_boot=2000, seed=0):
    """Patient-clustered percentile CI for AUROC (resample patients)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(pat, return_inverse=True)
    idx_by = [np.where(inv == k)[0] for k in range(len(uniq))]
    obs, out = _auroc(y, s), []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[k] for k in pick])
        v = _auroc(y[idx], s[idx])
        if np.isfinite(v):
            out.append(v)
    if not out:
        return obs, np.nan, np.nan
    return obs, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def ece_bins(y, p, nb=12):
    edges = np.linspace(0, 1, nb + 1)
    xs, ys, ws = [], [], []
    for i in range(nb):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < nb - 1 else p <= 1)
        if m.sum() < 10:
            continue
        xs.append(p[m].mean()); ys.append(y[m].mean()); ws.append(m.sum())
    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)
    e = float((ws * np.abs(xs - ys)).sum() / ws.sum()) if len(ws) else np.nan
    return xs, ys, ws, e


# =============================================================== F1 schematic
def fig_arch():
    fig, ax = plt.subplots(figsize=(COL2, 3.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, txt, fc="#FFFFFF", ec="#333333", fs=7, lw=0.9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.35,rounding_size=1.2",
                     fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fs, zorder=3,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x1, y1, x2, y2, style="-|>", ls="-", c="#333333", lw=0.9):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                     mutation_scale=8, lw=lw, ls=ls, color=c,
                     shrinkA=1, shrinkB=1, zorder=1))

    box(0.5, 19, 13.5, 8, "Raw clip\n4 / 16 / 44.1 kHz", fc="#F2F2F2", fs=6.6)
    box(16.5, 19, 13, 8, "Band\nharmonisation\n2 kHz $\\rightarrow$ 16 kHz", fc="#EAF3FA")
    arrow(14, 23, 16.5, 23)

    # two acquisition views
    box(31, 30.5, 15, 8.5, "Acquisition\nsimulator $A_1$", fc="#EAF3FA")
    box(31, 7.5, 15, 8.5, "Acquisition\nsimulator $A_2$", fc="#EAF3FA")
    arrow(29.5, 23, 31, 34.7); arrow(29.5, 23, 31, 11.7)
    ax.text(38.5, 23, "same clip,\ntwo physically\nplausible devices",
            ha="center", va="center", fontsize=6.3, style="italic", color="#555555")

    box(48.5, 30.5, 13, 8.5, "AST\nblocks 1$-$9", fc="#FFF6E5")
    box(48.5, 7.5, 13, 8.5, "AST\nblocks 1$-$9", fc="#FFF6E5")
    arrow(46, 34.7, 48.5, 34.7); arrow(46, 11.7, 48.5, 11.7)

    # the intervention
    box(64, 19, 17, 9.5,
        "Frequency-aware\nresidual AdaIN\n(same class, other patient)",
        fc="#E8F5EE", ec="#009E73", lw=1.3, bold=True)
    arrow(61.5, 34.7, 70.0, 28.5, ls=(0, (3, 2)), c="#009E73", lw=1.1)
    ax.text(72.5, 16.4, "training only", fontsize=6.2, style="italic",
            color="#009E73", ha="center")

    box(84, 30.5, 14, 8.5, "AST\nblocks 10$-$12\n+ heads", fc="#FFF6E5")
    box(84, 7.5, 14, 8.5, "AST\nblocks 10$-$12\n+ heads", fc="#FFF6E5")
    arrow(61.5, 34.7, 84, 34.7); arrow(61.5, 11.7, 84, 11.7)
    arrow(81, 23, 84, 31.5, c="#009E73", lw=1.1)

    # losses
    ax.annotate("", xy=(91, 30.5), xytext=(91, 16),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.2, color="#C00000"))
    ax.text(97.6, 23.2, "$\\mathcal{L}_{CF}$\nsymmetric\nstop-grad JS",
            ha="center", va="center", fontsize=6.8, color="#C00000")
    arrow(91, 39.0, 91, 42.4, style="-|>")
    ax.text(94.6, 41.6, "$\\mathcal{L}_{cls}$", ha="center", va="center",
            fontsize=7.5)
    ax.text(50, 2.2, "Inference: one unmodified AST forward pass "
            "(simulator, second view and AdaIN are training-only)",
            ha="center", fontsize=6.8, style="italic", color="#444444")
    ax.add_patch(Rectangle((0.4, 0.4), 99.2, 45.2, fill=False,
                           ec="#CCCCCC", lw=0.7))
    save(fig, "fig1_architecture")


# =============================================================== F2 schematic
def fig_protocol():
    fig, ax = plt.subplots(figsize=(COL2, 2.75))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, w, h, t, fc, ec="#333333", fs=6.8, lw=0.9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.3,rounding_size=1.0",
                     fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs,
                zorder=3, fontweight="bold" if bold else "normal",
                linespacing=1.3)

    ax.text(0, 37.5, "Hospital A + ICBHI (development)", fontsize=7.5,
            fontweight="bold", ha="left")
    xs = np.linspace(1, 81, 5)
    for i, d in enumerate(DEVS):
        held = (i == 3)
        box(xs[i], 24, 17, 8.5,
            f"{ARM_LABEL[d]}\n{'HELD OUT (test)' if held else 'train'}",
            fc="#FDECEC" if held else "#EAF3FA",
            ec="#C00000" if held else "#333333",
            lw=1.4 if held else 0.9, bold=held, fs=6.5)
    ax.text(50, 20.4, "repeated five times: each device is the test fold exactly once",
            ha="center", fontsize=6.4, style="italic", color="#555555")

    box(1, 8, 44, 8.5,
        "Splits keyed on $\\mathtt{cohort\\_uid}$\n"
        "(physical patient, device-invariant)", fc="#F2F2F2")
    ax.text(23, 4.4, "prevents the same patient appearing on\n"
                     "stethoscope in train and phone in test",
            ha="center", fontsize=6.2, color="#555555", linespacing=1.3)

    box(51, 6.5, 47, 11.5,
        "Hospital B — FROZEN\nstethoscope arm 1,966 clips\n"
        "phone arm 4,180 clips\nexcluded from every development decision",
        fc="#FDECEC", ec="#C00000", lw=1.4, bold=True, fs=6.2)
    ax.text(75.5, 2.2, "capability-blinded: unreadable, not merely filtered",
            ha="center", fontsize=6.2, style="italic", color="#C00000")
    save(fig, "fig2_protocol")


# ================================================================== F3 forest
def fig_forest():
    rows = {}
    for name, tmpl in SPECS:
        for d in ARMS:
            ys, ps, pats = [], [], []
            for s in SEEDS:
                z = load(tmpl, d, s)
                if z is None:
                    continue
                ys.append(z["y2"]); ps.append(z["p"]); pats.append(z["pat"])
            if not ys:
                continue
            # pool seeds by averaging probabilities on the shared clip order
            if not all(len(a) == len(ys[0]) for a in ys):
                print(f"  [skip] {name}/{d}: seed length mismatch"); continue
            rows[(name, d)] = cluster_ci(ys[0], np.mean(ps, 0), pats[0])
    if not rows:
        print("  [skip] forest: no prediction files"); return

    fig, axes = plt.subplots(1, len(ARMS), figsize=(COL2, 2.5))
    off = np.linspace(0.30, -0.30, len(SPECS))
    for j, (ax, d) in enumerate(zip(axes, ARMS)):
        lo_all, hi_all = [], []
        for k, (name, _) in enumerate(SPECS):
            if (name, d) not in rows:
                continue
            v, lo, hi = rows[(name, d)]
            lo_all.append(lo); hi_all.append(hi)
            st = STYLE[name]
            ax.errorbar(v, off[k], xerr=[[v - lo], [hi - v]], fmt=st["m"],
                        color=st["c"], ms=4.2, lw=1.25, capsize=2.1,
                        capthick=1.0,
                        mfc=st["c"] if name.startswith("CASG") else "white",
                        mew=1.05, zorder=4 if name.startswith("CASG") else 3)
        # zoom to the data: a shared 0-1 axis makes every CI look identical
        if lo_all:
            a, b = min(lo_all), max(hi_all)
            pad = max(0.012, 0.10 * (b - a))
            ax.set_xlim(a - pad, b + pad)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3, prune="both"))
        ax.set_ylim(-0.55, 0.55); ax.set_yticks([])
        ax.set_title(ARM_LABEL[d], pad=3)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", lw=0.4, color="#EEEEEE", zorder=0)
        ax.set_axisbelow(True)
    axes[len(ARMS) // 2].set_xlabel("Binary AUROC (note: independent scale per panel)")
    handles = [Line2D([], [], color=STYLE[n]["c"], marker=STYLE[n]["m"],
                      ls="-", lw=1.25, ms=4.2,
                      mfc=STYLE[n]["c"] if n.startswith("CASG") else "white",
                      mew=1.05, label=n) for n, _ in SPECS]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Binary AUROC with patient-clustered 95% CI "
                 "(seed-averaged probabilities)", y=1.03, fontsize=8)
    save(fig, "fig3_lodo_forest")


# =============================================================== F4 mechanism
def fig_mechanism(csv="runs/cpd_report.csv"):
    if not os.path.exists(csv):
        print(f"  [skip] mechanism: {csv} missing"); return
    import csv as _csv
    rec = list(_csv.DictReader(open(csv)))
    rec = [r for r in rec if "seed11" not in r["ckpt"]]

    def bucket(c):
        for name, key in CPD_KEY:
            if key in c:
                return name
        return None
    vals = {}
    for r in rec:
        b = bucket(r["ckpt"])
        if b:
            vals.setdefault((b, r["held_out"]), []).append(
                (float(r["cpd"]), float(r["cfv"])))
    if not vals:
        print("  [skip] mechanism: nothing parsed"); return

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.35))
    w = 0.2
    for ax, j, lab in ((axes[0], 0, "CPD"), (axes[1], 1, "CFV")):
        x = np.arange(len(ARMS))
        for k, (name, _) in enumerate(SPECS):
            mu = [np.mean([v[j] for v in vals.get((name, d), [])]) * 1e3
                  if (name, d) in vals else np.nan for d in ARMS]
            sd = [np.std([v[j] for v in vals.get((name, d), [])]) * 1e3
                  if (name, d) in vals else np.nan for d in ARMS]
            st = STYLE[name]
            ax.bar(x + (k - 1.5) * w, mu, w, yerr=sd, capsize=1.6,
                   color=st["c"], edgecolor="black", linewidth=0.5,
                   alpha=1.0 if name.startswith("CASG") else 0.82,
                   error_kw=dict(lw=0.7), label=name if j == 0 else None,
                   zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABEL[d] for d in ARMS], rotation=28,
                           ha="right")
        ax.set_ylabel(f"{lab}  ($\\times 10^{{-3}}$)")
        ax.set_title(f"{lab} — lower is more acquisition-invariant", pad=3)
        ax.grid(axis="y", lw=0.4, color="#E5E5E5", zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.22)      # headroom for the legend
    axes[0].legend(loc="upper left", ncol=1, fontsize=6.3)
    save(fig, "fig4_mechanism")


# ============================================================== F5/F6 curves
def _curve_panel(kind):
    grid = np.linspace(0, 1, 401)        # finer grid => visibly smoother curves
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 4.55))
    any_drawn = False
    for ax, d in zip(axes.ravel(), ARMS):
        scores = []
        for name, tmpl in SPECS:
            ys, ps = [], []
            for s in SEEDS:
                z = load(tmpl, d, s)
                if z is None:
                    continue
                ys.append(z["y2"]); ps.append(z["p"])
            if not ys:
                continue
            # ENSEMBLE: average the three seeds' probabilities per clip, then
            # build one curve. This is the same quantity the tables report, so
            # the area under the plotted curve equals the tabulated value.
            y, pe = ys[0], np.mean(ps, 0)
            cs = [roc_curve(y, pe, grid) if kind == "roc" else pr_curve(y, pe, grid)]
            scores.append(_auroc(y, pe) if kind == "roc"
                          else float(np.trapz(cs[0], grid)))
            any_drawn = True
            mu = cs[0]
            st = STYLE[name]
            ax.plot(grid, mu, color=st["c"], ls="-",
                    lw=1.55 if name.startswith("CASG") else 1.25,
                    solid_capstyle="round",
                    label=name, zorder=4 if name.startswith("CASG") else 3)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, ls=(0, (2, 2)),
                    zorder=1)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        else:
            z = load(SPECS[0][1], d, 0)
            if z is not None:
                ax.axhline(float((z["y2"] == 1).mean()), color="#BBBBBB",
                           lw=0.8, ls=(0, (2, 2)), zorder=1)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(ARM_LABEL[d], pad=3)
        ax.grid(lw=0.4, color="#EEEEEE", zorder=0); ax.set_axisbelow(True)
        # The curves coincide: state the numeric spread so the overlap is
        # readable as a result rather than as a plotting failure.
        if scores:
            lab = "AUROC" if kind == "roc" else "AUPRC"
            ax.text(0.97, 0.06 if kind == "roc" else 0.90,
                    f"{lab} {min(scores):.3f}–{max(scores):.3f}\n"
                    f"across all four methods",
                    transform=ax.transAxes, ha="right",
                    va="bottom" if kind == "roc" else "top",
                    fontsize=5.9, color="#333333", linespacing=1.25,
                    bbox=dict(fc="white", ec="#DDDDDD", lw=0.5, pad=1.6))
    if not any_drawn:
        plt.close(fig); print(f"  [skip] {kind}: no prediction files"); return
    handles = [Line2D([], [], color=STYLE[n]["c"], ls="-",
                      lw=1.55 if n.startswith("CASG") else 1.25, label=n)
               for n, _ in SPECS]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.035), columnspacing=1.25,
               handlelength=2.7)
    fig.suptitle(("Receiver operating characteristic" if kind == "roc"
                  else "Precision-recall") +
                 ", three-seed probability ensemble", y=0.995, fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save(fig, "fig5_roc" if kind == "roc" else "fig6_pr")


def fig_roc(): _curve_panel("roc")
def fig_pr():  _curve_panel("pr")


# ================================================================= F7 calib
def fig_calib():
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 4.4))
    any_drawn = False
    for ax, d in zip(axes.ravel(), ARMS):
        ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, ls=(0, (2, 2)))
        for name, tmpl in SPECS:
            ys, ps = [], []
            for s in SEEDS:
                z = load(tmpl, d, s)
                if z is None:
                    continue
                ys.append(z["y2"]); ps.append(z["p"])
            if not ys:
                continue
            any_drawn = True
            # ENSEMBLE probability per clip, matching the tabulated ECE
            xs, yy, _, e = ece_bins(ys[0], np.mean(ps, 0))
            st = STYLE[name]
            ax.plot(xs, yy, color=st["c"], ls="-", marker=st["m"], ms=3.0,
                    lw=1.55 if name.startswith("CASG") else 1.25,
                    mfc=st["c"], mew=0.0, solid_capstyle="round",
                    label=f"{name.replace(' (ours)','')}: {e:.3f}",
                    zorder=4 if name.startswith("CASG") else 3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted $p$(abnormal)"); ax.set_ylabel("Observed frequency")
        ax.set_title(ARM_LABEL[d], pad=3)
        ax.legend(title="ECE (12 bins)", fontsize=5.6, title_fontsize=5.7,
                  loc="upper left", handlelength=1.8, labelspacing=0.22,
                  borderpad=0.15)
        ax.grid(lw=0.45, color="#E6E6E6", zorder=0); ax.set_axisbelow(True)
    if not any_drawn:
        plt.close(fig); print("  [skip] calib: no prediction files"); return
    fig.suptitle("Reliability diagrams (three-seed probability ensemble). "
                 "Points below the diagonal indicate overconfidence.",
                 y=0.995, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig7_calibration")


# ================================================================= F8 phone
def fig_phone(csv="runs/frozen_phone/frozen_phone_summary.csv"):
    if not os.path.exists(csv):
        print(f"  [skip] phone: {csv} missing"); return
    import csv as _csv
    rec = list(_csv.DictReader(open(csv)))
    name_map = {"CASG-Lite": "CASG-Lite (ours)", "CF-only-Physics": "CF-only-Physics",
                "ERM-Physics": "ERM-Physics", "MixStyle-Physics": "MixStyle-Physics"}
    vals = {}
    for r in rec:
        n = name_map.get(r["method"])
        if n:
            vals.setdefault((n, r["train_fold"]), []).append(float(r["auroc"]))

    fig, ax = plt.subplots(figsize=(COL1 * 1.5, 2.3))
    x = np.arange(len(ARMS)); w = 0.2
    for k, (name, _) in enumerate(SPECS):
        mu = [np.mean(vals.get((name, d), [np.nan])) for d in ARMS]
        sd = [np.std(vals.get((name, d), [np.nan])) for d in ARMS]
        st = STYLE[name]
        ax.bar(x + (k - 1.5) * w, mu, w, yerr=sd, capsize=1.6, color=st["c"],
               edgecolor="black", lw=0.5,
               alpha=1.0 if name.startswith("CASG") else 0.82,
               error_kw=dict(lw=0.7), label=name, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"trained w/o\n{ARM_LABEL[d]}" for d in ARMS],
                       fontsize=6.2)
    ax.set_ylim(0.75, 0.885); ax.set_ylabel("AUROC on reserved arm")
    ax.set_title("Reserved Hospital-B smartphone arm (4,180 clips, opened once)",
                 pad=16)
    ax.grid(axis="y", lw=0.4, color="#E5E5E5", zorder=0); ax.set_axisbelow(True)
    # legend above the axes so it cannot collide with bars or the annotation
    ax.legend(fontsize=6.2, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), frameon=False,
              columnspacing=1.1, handlelength=1.4)
    # shade and label the strictest cell
    i = ARMS.index("smartphone")
    ax.axvspan(i - 0.46, i + 0.46, color="#C00000", alpha=0.055, zorder=0)
    ax.annotate("unseen device AND unseen site", xy=(i, 0.845),
                xytext=(i - 1.35, 0.872), fontsize=6.0, color="#C00000",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#C00000",
                                shrinkA=2, shrinkB=3))
    save(fig, "fig8_frozen_phone")


# ======================================== F6 external-site curve triptych
def fig_external(arm="none"):
    """ROC + PR + reliability for ONE arm, drawn at final size.

    Replaces the practice of \\includegraphics[trim=...,clip] on the six-panel
    ROC/PR/calibration figures. Cropping a 7.16-in figure down to 0.29\\textwidth
    scales every stroke by about a third and slices off the axis labels; this
    renders the three panels natively at the width they will be printed, so
    line weights and font sizes are the ones chosen here.
    """
    grid = np.linspace(0, 1, 401)
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.42))
    drawn = False
    roc_s, pr_s = [], []

    for name, tmpl in SPECS:
        ys, ps = [], []
        for s in SEEDS:
            z = load(tmpl, arm, s)
            if z is None:
                continue
            ys.append(z["y2"]); ps.append(z["p"])
        if not ys:
            continue
        drawn = True
        st = STYLE[name]
        lw = 1.55 if name.startswith("CASG") else 1.25
        y, pe = ys[0], np.mean(ps, 0)              # three-seed ensemble
        R = roc_curve(y, pe, grid); P = pr_curve(y, pe, grid)
        roc_s.append(_auroc(y, pe))
        pr_s.append(float(np.trapz(P, grid)))
        for ax, C in ((axes[0], R), (axes[1], P)):
            ax.plot(grid, C, color=st["c"], ls="-", lw=lw,
                    solid_capstyle="round",
                    zorder=4 if name.startswith("CASG") else 3)
        xs, yy, _, e = ece_bins(y, pe)             # ensemble ECE, matches tables
        axes[2].plot(xs, yy, color=st["c"], ls="-", marker=st["m"],
                     ms=3.0, mfc=st["c"], mew=0.0, lw=lw,
                     solid_capstyle="round",
                     zorder=4 if name.startswith("CASG") else 3)
        axes[2].plot([], [], color=st["c"], ls="-", lw=lw,
                     label=f"{name.replace(' (ours)','')}: {e:.3f}")
    if not drawn:
        plt.close(fig); print(f"  [skip] external: no preds for arm={arm}"); return

    axes[0].plot([0, 1], [0, 1], color="#BBBBBB", lw=0.9, ls=(0, (2, 2)), zorder=1)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("(a) ROC", pad=4)
    axes[0].text(0.96, 0.05, f"AUROC {min(roc_s):.3f}–{max(roc_s):.3f}",
                 transform=axes[0].transAxes, ha="right", fontsize=6.2,
                 bbox=dict(fc="white", ec="#DDDDDD", lw=0.5, pad=1.6))

    z0 = load(SPECS[0][1], arm, 0)
    if z0 is not None:
        axes[1].axhline(float((z0["y2"] == 1).mean()), color="#BBBBBB", lw=0.9,
                        ls=(0, (2, 2)), zorder=1)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("(b) Precision–recall", pad=4)
    axes[1].text(0.96, 0.90, f"AUPRC {min(pr_s):.3f}–{max(pr_s):.3f}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=6.2,
                 bbox=dict(fc="white", ec="#DDDDDD", lw=0.5, pad=1.6))

    axes[2].plot([0, 1], [0, 1], color="#BBBBBB", lw=0.9, ls=(0, (2, 2)), zorder=1)
    axes[2].set_xlabel("Predicted $p$(abnormal)")
    axes[2].set_ylabel("Observed frequency")
    axes[2].set_title("(c) Reliability", pad=4)
    axes[2].legend(title="ECE (12 bins)", fontsize=5.8, title_fontsize=5.9,
                   loc="upper left", handlelength=1.6,
                   labelspacing=0.25, borderpad=0.2)

    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(lw=0.45, color="#E6E6E6", zorder=0); ax.set_axisbelow(True)

    handles = [Line2D([], [], color=STYLE[n]["c"], ls="-",
                      lw=1.55 if n.startswith("CASG") else 1.25, label=n)
               for n, _ in SPECS]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, f"fig6_external_curves{'' if arm == 'none' else '_' + arm}")


FIGS = {"arch": fig_arch, "protocol": fig_protocol, "forest": fig_forest,
        "mechanism": fig_mechanism, "roc": fig_roc, "pr": fig_pr,
        "calib": fig_calib, "phone": fig_phone, "external": fig_external}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[], choices=list(FIGS))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k in FIGS:
            print(" ", k)
        sys.exit(0)
    OUT = a.out
    rc()
    for k in (a.only or list(FIGS)):
        print(f"[{k}]")
        try:
            FIGS[k]()
        except Exception as e:
            print(f"  [FAIL] {k}: {type(e).__name__}: {e}")
    print(f"\nfigures in {OUT}/ — PDF for LaTeX, SVG editable, PNG at 600 dpi "
          f"(\\includegraphics[width=\\textwidth]{{fig5_roc.pdf}})")