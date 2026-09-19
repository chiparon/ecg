"""Render six publication figures only from independently verified full-cohort results."""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from methodology_supplement.plot import _foot, _save, _whiskers
from phase2.src.plot_phase2 import COLORS
from .common import (WORKSPACE, check_info, file_info, load_config, load_manifest,
                     read_json, require_freeze, save_json, stage_paths)

MODELS = {"resnet": "ResNet", "tcn": "TCN"}
METRICS = {"macro_auroc": "Macro AUROC", "macro_f1": "Macro F1", "ece": "Classwise ECE"}
CONTRASTS = ("E-S", "S-I", "E-I")
MODES = tuple(f"S_{index:02d}" for index in range(5))
SNRS = (10, 5, 0)
MODEL_COLORS = {"resnet": COLORS["electrode"], "tcn": COLORS["independent_rms"]}
MODE_COLORS = ("#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7A5195")
MODE_MARKERS = ("o", "s", "^", "D", "v")
FIGURE_IDS = ("signed_control_structure_effects", "signed_control_snr_curves",
              "signed_control_matrices", "subspace_q_distributions",
              "subspace_geometry", "input_validation")
UNCERTAINTY = ("Thin: paired patient 95% CI, conditional on fixed checkpoints/noises/modes. "
               "Thick: +/-1 training-seed SD (3 seeds); these are separate, not combined.")
DIRECTION = "Positive AUROC/F1: first condition better; positive ECE: first condition worse."


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _one(frame, **keys):
    selected = frame
    for key, value in keys.items():
        selected = selected.loc[selected[key].eq(value)]
    _require(len(selected) == 1, f"Expected exactly one figure cell: {keys}")
    return selected.iloc[0]


def _grid(frame, columns, values, name):
    expected = set(product(*values))
    actual = set(frame.loc[:, columns].itertuples(index=False, name=None))
    _require(len(frame) == len(expected) and actual == expected,
             f"Incomplete or duplicate {name} grid")


def _info_objects(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            yield value
        else:
            for item in value.values():
                yield from _info_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _info_objects(item)


def _finish(fig, title, footer, bottom=.16, top=.88, hspace=.4, wspace=.3):
    fig.suptitle(title, fontsize=15, fontweight="bold", y=.985)
    fig.subplots_adjust(left=.075, right=.96, bottom=bottom, top=top,
                        hspace=hspace, wspace=wspace)
    _foot(fig, footer)


def _structure(paths, summary, seeds, modes, figures):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row_index, (model, model_name) in enumerate(MODELS.items()):
        color = MODEL_COLORS[model]
        for column, (metric, metric_name) in enumerate(METRICS.items()):
            ax = axes[row_index, column]
            ax.axvline(0, color="#777777", linewidth=.8)
            ax.axhspan(-.38, .38, color="#F2F2F2", zorder=0)
            for y, contrast in enumerate(CONTRASTS):
                record = _one(summary, model=model, metric=metric, snr=0, contrast=contrast)
                _whiskers(ax, y, record, color, horizontal=True)
                for index, seed in enumerate(sorted(seeds.seed.unique())):
                    point = _one(seeds, model=model, metric=metric, snr=0, contrast=contrast, seed=seed)
                    ax.plot(100 * point.estimate, y - .23 + .035 * (index - 1), "|",
                            color=color, markersize=6, alpha=.85)
                if contrast != "E-I":
                    for index, mode in enumerate(MODES):
                        point = _one(modes, model=model, metric=metric, snr=0, contrast=contrast, mode=mode)
                        ax.plot(100 * point.estimate, y + .23 + .035 * (index - 2),
                                marker=MODE_MARKERS[index], color=MODE_COLORS[index],
                                markersize=4, linestyle="none")
            ax.set(yticks=range(3), yticklabels=CONTRASTS, ylim=(2.5, -.5),
                   xlabel="Paired difference (percentage points)",
                   title=f"{model_name} | {metric_name}" + ("\nCo-primary: E-S at 0 dB" if metric == "macro_auroc" else "\nSecondary metric"))
            ax.grid(axis="x", color="#E8E8E8", linewidth=.6)
    legend = [Line2D([], [], color="#555555", marker="|", linestyle="none", label="Training-seed estimates")]
    legend.extend(Line2D([], [], color=color, marker=marker, linestyle="none", label=mode)
                  for mode, color, marker in zip(MODES, MODE_COLORS, MODE_MARKERS))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.52, .945), ncol=6, frameon=False)
    _finish(fig, "0 dB structure effects | both architectures are co-primary",
            UNCERTAINTY + "\n" + DIRECTION + " E = Standard; S = equal five-mode mean; I = independent RMS.\n"
            "Deterministic row offsets: CI -0.065 / SD +0.065; seed ticks -0.23; individual mode points +0.23 (small fixed spacing).\n"
            "Modes are fixed descriptive controls, not extra training seeds. E-I has no mode dependence and is shown once. Grey row: E-S.",
            bottom=.19, top=.82, hspace=.5)
    _save(fig, FIGURE_IDS[0],
          "All three 0-dB contrasts and all three metrics for both co-primary architectures. "
          "Primary estimand: each architecture's 0-dB E-S macro AUROC. Points average paired metric differences, not probabilities. "
          + UNCERTAINTY + " " + DIRECTION + " Five individual fixed-mode estimates and three training-seed estimates are distinguished; redundant mode copies of E-I are omitted.",
          paths, figures, [paths["tables"] / f"{name}.csv" for name in
                           ("signed_control_summary", "signed_control_seed_effects", "signed_mode_summary")])


def _snr(paths, summary, figures):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for row_index, (metric, metric_name) in enumerate(METRICS.items()):
        for column, contrast in enumerate(CONTRASTS):
            ax = axes[row_index, column]
            ax.axhline(0, color="#777777", linewidth=.8)
            ax.axvspan(1.72, 2.3, color="#F2F2F2", zorder=0)
            for model_index, (model, name) in enumerate(MODELS.items()):
                offset = -.13 if model_index == 0 else .13
                records = [_one(summary, model=model, metric=metric, contrast=contrast, snr=snr) for snr in SNRS]
                ax.plot(np.arange(3) + offset, [100 * record.estimate for record in records],
                        color=MODEL_COLORS[model], linewidth=1, label=name)
                for x, record in enumerate(records):
                    _whiskers(ax, x + offset, record, MODEL_COLORS[model])
            ax.set(xticks=range(3), xticklabels=["10", "5", "0"], xlim=(-.35, 2.35),
                   xlabel="SNR (dB); increasing noise to right",
                   ylabel=f"{metric_name} difference (pp)", title=contrast)
            ax.grid(axis="y", color="#E8E8E8", linewidth=.6)
            if (row_index, column) == (0, 0):
                ax.set_title("E-S | primary at 0 dB", fontweight="bold")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.5, .952), ncol=2, frameon=False)
    _finish(fig, "Signed-control effects across the complete SNR grid",
            UNCERTAINTY + "\n" + DIRECTION + " All values in percentage points.\n"
            "Horizontal offsets are visual only: ResNet -0.13, TCN +0.13 tick units; CI -0.065 and SD +0.065 around each model.\n"
            "S = mean across all five frozen signed modes. Grey band marks 0 dB, not an uncertainty region.",
            bottom=.15, top=.89, hspace=.55)
    _save(fig, FIGURE_IDS[1],
          "Complete 54-cell summary: both architectures, three contrasts, three metrics and 10/5/0 dB. "
          "Increasing noise is rightward. " + UNCERTAINTY + " " + DIRECTION +
          " Fixed horizontal offsets separate models and uncertainty types without changing the SNR values.",
          paths, figures, [paths["tables"] / "signed_control_summary.csv"])


def _matrices(paths, matrices, controls, matrix_path, controls_path, figures):
    items = [("Standard A", matrices["standard"]), *[(item["mode"], item) for item in matrices["controls"]]]
    limit = max(float(np.max(np.abs(item["matrix"]))) for _, item in items)
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    sign_lookup = {item["mode"]: item for item in controls["controls"]}
    for ax, (name, item) in zip(axes.flat, items):
        values = np.asarray(item["matrix"], dtype=np.float64)
        image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        for lead, electrode in np.ndindex(values.shape):
            if values[lead, electrode] != 0:
                ax.text(electrode, lead, f"{values[lead, electrode]:.2g}", ha="center", va="center",
                        fontsize=7, color="white" if abs(values[lead, electrode]) > .6 * limit else "#222222")
        detail = "Frozen reference" if name == "Standard A" else f"Candidate {sign_lookup[name]['candidate_index']}"
        ax.set(xticks=range(len(matrices["electrode_order"])), xticklabels=matrices["electrode_order"],
               yticks=range(len(matrices["lead_order"])), yticklabels=matrices["lead_order"],
               title=f"{name} | rank {item['rank']}\n{detail}", xlabel="Source coordinate", ylabel="ECG lead")
    _finish(fig, "Actual lead-mapping matrices | rank and singular spectrum preserved",
            "All panels use the same coefficient colour scale; blank cells are exact zeros. A_signed = diag(s) A; no entries are reconstructed from performance.\n"
            "These are mathematical signed-direction controls, not physical acquisition-error simulations. Source coordinate labels follow the frozen Standard matrix.",
            bottom=.12, top=.89, hspace=.45, wspace=.4)
    fig.subplots_adjust(right=.89)
    fig.colorbar(image, cax=fig.add_axes([.92, .22, .015, .56]), label="Mapping coefficient")
    _save(fig, FIGURE_IDS[2],
          "Standard and all five actual signed mapping matrices with common colour scale, source/lead order and matrix rank. "
          "Candidate numbers are the pre-prediction accepted candidates, never performance-selected. "
          "Signs preserve rank and singular values; this is a mathematical mapping-direction intervention, not a physical acquisition-error simulator.",
          paths, figures, [matrix_path, controls_path])


def _q_averages(cfg, frame, freeze):
    """One value per ECG after the complete fixed-factor grid; nulls propagate."""
    columns = ["object", "snr", "noise_seed", "mode", "record_index"]
    frame = frame.copy()
    frame["snr"] = frame.snr.fillna(-1).astype(int)
    frame["noise_seed"] = frame.noise_seed.fillna(-1).astype(int)
    frame["mode"] = frame["mode"].fillna("")
    expected = {("clean", -1, -1, "")}
    for snr, noise in product(SNRS, cfg["phase1_noise_seeds"]):
        for suffix in ("noise", "input"):
            expected.update((f"{condition}_{suffix}", snr, noise, "") for condition in ("E", "I"))
            expected.update((f"S_{suffix}", snr, noise, mode) for mode in MODES)
    observed = set(frame[columns[:4]].itertuples(index=False, name=None))
    _require(observed == expected, "q figure requires exactly the complete clean/E/S/I grid")
    _require(not frame.duplicated(columns).any(), "Duplicate per-record q cell")
    with np.load(check_info(freeze["cohort"]), allow_pickle=False) as cohort:
        ids, patients = cohort["ids"], cohort["patient_ids"]
    indices = frame.record_index.to_numpy(dtype=np.int64)
    _require(np.all((indices >= 0) & (indices < len(ids))), "Out-of-cohort q record")
    _require(np.array_equal(frame.ecg_id.to_numpy(), ids[indices]) and
             np.array_equal(frame.patient_id.to_numpy(), patients[indices]), "q cohort identity changed")
    counts = frame.groupby(columns[:4], dropna=False).size()
    _require((counts == len(ids)).all(), "Missing ECGs in q source groups")
    values = frame.q.to_numpy(dtype=float)
    zero = frame.zero_norm.astype(str).str.lower().eq("true").to_numpy()
    _require(np.array_equal(np.isnan(values), zero) and np.isfinite(values[~zero]).all()
             and (values[~zero] >= 0).all(), "q null/zero-norm contract changed")

    def aggregate(selected, repetitions):
        groups = selected.groupby("record_index", sort=True).q
        count, valid = groups.size(), groups.count()
        _require(len(count) == len(ids) and (count == repetitions).all(), "Incomplete fixed-factor q average")
        # Do not silently drop null fixed-factor cells from a record's average.
        means = (groups.sum() / repetitions).where(valid.eq(repetitions))
        return {"values": means.to_numpy(dtype=float), "n_records": len(ids),
                "n_patients": len(np.unique(patients)), "n_source_rows": len(selected),
                "n_source_null": int(selected.q.isna().sum()), "n_null_record_means": int(means.isna().sum()),
                "fixed_values_per_record": repetitions}

    result = {("clean", -1, ""): aggregate(frame.loc[frame.object.eq("clean")], 1)}
    for snr, suffix in product(SNRS, ("noise", "input")):
        for condition in ("E", "I", "S"):
            selected = frame.loc[frame.object.eq(f"{condition}_{suffix}") & frame.snr.eq(snr)]
            result[(suffix, snr, condition)] = aggregate(selected, 30 if condition == "S" else 6)
            if condition == "S":
                for mode in MODES:
                    result[(suffix, snr, mode)] = aggregate(selected.loc[selected["mode"].eq(mode)], 6)
    return result


def _subspace(paths, averages, figures):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    categories = ("clean", "E", "S", "I", *MODES)
    colors = (COLORS["clean_only"], COLORS["electrode"], COLORS["mixed"], COLORS["independent_rms"], *MODE_COLORS)
    counts = []
    for row, suffix in enumerate(("noise", "input")):
        for column, snr in enumerate(SNRS):
            ax = axes[row, column]
            labels = []
            for position, (condition, color) in enumerate(zip(categories, colors)):
                key = ("clean", -1, "") if condition == "clean" else (suffix, snr, condition)
                item = averages[key]
                values = item["values"]
                finite = values[np.isfinite(values)]
                labels.append(("Clean ref." if condition == "clean" else ("S mean" if condition == "S" else condition)) +
                              f"\n{len(finite)}/{item['n_null_record_means']}")
                if len(finite):
                    box = ax.boxplot([finite], positions=[position], widths=.55, whis=(0, 100),
                                     showfliers=False, patch_artist=True,
                                     medianprops={"color": "#111111", "linewidth": 1.3},
                                     whiskerprops={"linewidth": .8}, capprops={"linewidth": .8})
                    box["boxes"][0].set(facecolor=color, alpha=.55)
                counts.append(dict(panel=f"{suffix}_{snr}", condition=condition,
                                   **{name: value for name, value in item.items() if name != "values"}))
            ax.set_yscale("symlog", linthresh=1e-14, linscale=.65)
            ax.set(xticks=range(len(categories)), xticklabels=labels,
                   ylabel=r"$q=\|Qv\|_F^2/\|v\|_F^2$", title=f"{'Actual residual noise' if suffix == 'noise' else 'Full noisy input'} | {snr} dB")
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="y", color="#E5E5E5", linewidth=.6)
            ax.axhline(0, color="#555555", linewidth=.7)
    all_values = np.concatenate([item["values"] for item in averages.values()])
    ymax = float(np.nanmax(all_values))
    for ax in axes.flat:
        ax.set_ylim(0, max(ymax * 1.25, 1e-13))
    _finish(fig, "Standard-column-space residual fraction | descriptive across ECG records",
            "Each box: across-ECG distribution of per-record means over 6 fixed noises; S mean also averages 5 modes (30 fixed values). Clean is unaveraged.\n"
            "Box = Q1/median/Q3; whiskers = observed min/max, not CIs. Tick counts = valid record means / null record means. Repeated noises are not patients.\n"
            "Null denominator remains null and propagates to the record mean; exact zero stays zero. Symlog retains the zero area (linear threshold 1e-14).\n"
            "Actual noise = float64(noisy float32) - float64(clean float32). q tests only ideal Standard column-space constraints, not complete physiological plausibility.",
            bottom=.19, top=.90, hspace=.45, wspace=.3)
    _save(fig, FIGURE_IDS[3],
          "Descriptive q distributions for clean, actual E/S/I residual noise and full noisy E/S/I inputs at all three SNRs; all five signed modes are separately shown. "
          "Per-ECG means use all six frozen noise realizations; the additional S mean uses all 30 mode/noise combinations. "
          "Box quartiles and full-range whiskers are across ECGs, not patient CIs, and fixed repeats are not counted as independent patients. "
          "Missing-denominator values are explicitly counted and propagated; zero is not replaced by a positive floor. "
          "Standard rank eight includes six unconstrained precordial coordinates and ideal limb relationships, not all physiological constraints.",
          paths, figures, [paths["tables"] / "subspace_diagnostics_per_record.csv"])
    figures[-1]["aggregation_counts"] = counts


def _geometry(paths, matrices, matrix_path, figures):
    fig = plt.figure(figsize=(15, 12))
    grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, .85))
    P = np.asarray(matrices["projection"]["P"])
    projectors = [("Standard P", P), *[(item["mode"] + " : S P S", np.asarray(item["projection"])) for item in matrices["controls"]]]
    limit = max(float(np.max(np.abs(value))) for _, value in projectors)
    for index, (name, value) in enumerate(projectors):
        ax = fig.add_subplot(grid[index // 3, index % 3])
        image = ax.imshow(value, vmin=-limit, vmax=limit, cmap="RdBu_r", aspect="equal")
        ax.set(xticks=range(12), xticklabels=matrices["lead_order"], yticks=range(12),
               yticklabels=matrices["lead_order"], title=name)
        ax.tick_params(axis="both", labelsize=7)
        ax.tick_params(axis="x", labelrotation=60)
    angles_ax, spectrum_ax, change_ax = [fig.add_subplot(grid[2, index]) for index in range(3)]
    standard = matrices["standard"]
    rank = standard["rank"]
    basis = np.linalg.svd(np.asarray(standard["matrix"]), full_matrices=False)[0][:, :rank]
    spectrum_ax.plot(range(1, 13), sorted(standard["eigenvalues"], reverse=True),
                     color="#333333", linewidth=2, label="Standard")
    changes = []
    covariance = np.asarray(standard["covariance"])
    for item, color, marker in zip(matrices["controls"], MODE_COLORS, MODE_MARKERS):
        signed_basis = np.linalg.svd(np.asarray(item["matrix"]), full_matrices=False)[0][:, :rank]
        cosines = np.linalg.svd(basis.T @ signed_basis, compute_uv=False)
        angles = np.degrees(np.arccos(np.clip(cosines, 0, 1)))
        angles_ax.plot(range(1, rank + 1), angles, color=color, marker=marker, markersize=4, label=item["mode"])
        spectrum_ax.plot(range(1, 13), sorted(item["eigenvalues"], reverse=True),
                         color=color, marker=marker, markersize=4, markerfacecolor="none", linewidth=.6, label=item["mode"])
        changes.append(np.linalg.norm(np.asarray(item["covariance"]) - covariance, "fro") /
                       np.linalg.norm(covariance, "fro"))
    angles_ax.set(title="Principal angles to Standard", xlabel="Ordered subspace angle", ylabel="Angle (degrees)",
                  xticks=range(1, rank + 1), ylim=(-2, 92))
    spectrum_ax.set(title="Unit-source covariance spectrum", xlabel="Descending eigenvalue index", ylabel="Eigenvalue", xticks=(1, 4, 8, 12))
    change_ax.bar(MODES, changes, color=MODE_COLORS, width=.6)
    change_ax.set(title="Covariance direction changes", ylabel=r"$\|SCS-C\|_F/\|C\|_F$", xlabel="Fixed signed mode")
    for ax in (angles_ax, spectrum_ax, change_ax):
        ax.grid(axis="y", color="#E8E8E8", linewidth=.6)
    angles_ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
    spectrum_ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    _finish(fig, "Exact subspace geometry | rotated directions, unchanged spectrum",
            "Top: actual 12 x 12 orthogonal projectors in ECG-lead coordinates, with a shared scale. Bottom: principal angles and covariance descriptors from actual matrices.\n"
            "C = A A^T; signed covariance = S C S. Spectrum curves overlap by design; principal-angle roundoff can produce numerical-near-zero angles.\n"
            "No literal low-dimensional plane or physical hardware is implied. Standard rank 8 includes six unconstrained precordial coordinates plus ideal limb relationships.",
            bottom=.14, top=.92, hspace=.6, wspace=.45)
    fig.subplots_adjust(right=.9)
    fig.colorbar(image, cax=fig.add_axes([.93, .47, .012, .4]), label="Projector coefficient")
    _save(fig, FIGURE_IDS[4],
          "Exact projector matrices P and all five SPS matrices, principal angles between actual rank-eight column spaces, "
          "unit-source covariance eigenvalues and relative covariance changes. Matrix-coordinate heatmaps and principal angles "
          "describe actual high-dimensional geometry without depicting a literal 12-D plane. Preserved eigenvalues do not imply preserved directions. "
          "These ideal Standard subspace constraints do not establish physiological plausibility or unique causality.",
          paths, figures, [matrix_path])


def _validation(cfg, paths, audit, figures):
    conditions = ("E", "I", *MODES)
    metrics = (("snr_abs_error_db", "Actual residual SNR error", "snr_abs_error_db"),
               ("max_lead_rms_relative_error", "Actual residual per-lead RMS error", "lead_rms_relative_error"),
               ("max_periodogram_peak_relative_error", "Actual residual per-lead PSD error", "periodogram_tolerance"))
    exact = (("max_base_sign_absolute_error", "Designed base sign invariant"),
             ("max_scaled_sign_absolute_error", "Designed scaled sign invariant"),
             ("max_designed_periodogram_peak_relative_error", "Designed per-lead PSD invariant"))
    grouped = audit.groupby(["condition", "snr"], sort=False)
    maxima = grouped[[field for field, _, _ in metrics] + [field for field, _ in exact]].max()
    _grid(maxima.reset_index(), ["condition", "snr"], [conditions, SNRS], "input audit")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    snr_colors = ("#595959", "#0072B2", "#D55E00")
    for column, (field, title, gate_name) in enumerate(metrics):
        ax = axes[0, column]
        gate = cfg["gates"][gate_name]
        ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
        for snr_index, (snr, color) in enumerate(zip(SNRS, snr_colors)):
            for x, condition in enumerate(conditions):
                value = maxima.loc[(condition, snr), field]
                if pd.notna(value):
                    ax.plot(x + (snr_index - 1) * .19, value / gate, marker=("o", "s", "^")[snr_index],
                            color=color, markersize=5, linestyle="none")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set(title=f"{title}\nFixed gate = {gate:g}", ylabel="Maximum error / fixed gate", ylim=(0, 2),
               xticks=range(len(conditions)), xticklabels=conditions)
        if column == 2:
            ax.text(1, .02, "N/A", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        ax.grid(axis="y", color="#E8E8E8", linewidth=.6)
    for column, (field, title) in enumerate(exact):
        ax = axes[1, column]
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1)
        applicable = MODES if column < 2 else ("E", *MODES)
        for snr_index, (snr, color) in enumerate(zip(SNRS, snr_colors)):
            for x, condition in enumerate(conditions):
                value = maxima.loc[(condition, snr), field]
                if condition in applicable:
                    _require(pd.notna(value) and value == 0, f"Exact designed invariant failed: {condition}/{snr}/{field}")
                    ax.plot(x + (snr_index - 1) * .19, value, marker=("o", "s", "^")[snr_index],
                            color=color, markersize=5, linestyle="none")
                elif snr_index == 0:
                    ax.text(x, .1, "N/A", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        ax.set(title=title + "\nExact zero required (not a tolerance test)",
               ylabel="Maximum absolute error", xticks=range(len(conditions)), xticklabels=conditions,
               ylim=(-.05, .10), yticks=[0])
    legend = [Line2D([], [], marker=marker, color=color, linestyle="none", label=f"{snr} dB")
              for snr, color, marker in zip(SNRS, snr_colors, ("o", "s", "^"))]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .95), ncol=3, frameon=False)
    n_records = audit.record_index.nunique()
    n_patients = audit.patient_id.nunique()
    _finish(fig, "Input validation | measured errors and exact designed invariants",
            f"Each point is the maximum over {n_records:,} ECG records and 6 fixed noises for that condition/SNR ({n_patients:,} unique patients); not an uncertainty estimate.\n"
            "Top: post-addition float64 residual measurements; dashed line = fixed gate. RMS is worst lead; PSD is worst bin difference / that lead's reference peak, then worst lead.\n"
            "Bottom: designed float32 sign and per-lead PSD identities are exactly zero. Independent RMS has no signed/PSD identity requirement (N/A).\n"
            "All seven conditions and 10/5/0 dB are retained; fixed horizontal offsets separate SNRs. Zero stays zero on the upper symlog axes.",
            bottom=.19, top=.82, hspace=.5)
    _save(fig, FIGURE_IDS[5],
          "Actual post-addition residual SNR, per-lead RMS and per-lead frequency-bin PSD errors are plotted against their frozen gates. "
          "Each point is the worst observed record/noise error within a condition and SNR, not a patient estimate or CI. "
          "Designed base/scaled sign identities and PSD invariance are separately displayed as exact-zero requirements. "
          "Nonapplicable independent-noise PSD and sign comparisons are explicitly labelled rather than filled with fake zeros.",
          paths, figures, [paths["tables"] / "input_validation.csv", Path(cfg["_config_path"])])


def run(config=None, stage="full"):
    _require(stage == "full", "Publication figures are full-cohort only")
    cfg = load_config(config)
    paths = stage_paths(cfg, stage)
    verification_path = paths["logs"] / "full_verification.json"
    verification = read_json(verification_path)
    _require(verification.get("status") == "passed" and verification.get("stage") == "full"
             and verification.get("config_sha256") == cfg["_config_sha256"],
             "Publication requires a passed, current full-cohort independent verification")
    for info in _info_objects(verification):
        check_info(info)
    freeze = require_freeze(cfg, stage)
    inputs = load_manifest(cfg, stage)
    statistics_path = paths["logs"] / "statistics.json"
    statistics = read_json(statistics_path)
    _require(statistics.get("status") == "completed" and statistics.get("stage") == "full"
             and statistics.get("config_sha256") == cfg["_config_sha256"], "Statistics are incomplete or stale")
    source_paths = [verification_path, statistics_path, paths["inputs"] / "manifest.json",
                    paths["logs"] / "freeze.json", Path(cfg["_config_path"])]
    for info in [*inputs["tables"].values(), *statistics["outputs"].values()]:
        source_paths.append(check_info(info))
    matrix_path, controls_path = check_info(freeze["matrices"]), check_info(freeze["sign_controls"])
    matrices, controls = read_json(matrix_path), read_json(controls_path)
    source_paths.extend([matrix_path, controls_path])
    _require([item["mode"] for item in matrices["controls"]] == list(MODES), "Matrix mode order changed")
    tables = {name: pd.read_csv(check_info(statistics["outputs"][name])) for name in
              ("signed_control_summary", "signed_control_seed_effects", "signed_mode_summary")}
    base_keys = ["model", "snr", "metric", "contrast"]
    base_values = [tuple(MODELS), SNRS, tuple(METRICS), CONTRASTS]
    _grid(tables["signed_control_summary"], base_keys, base_values, "summary")
    _grid(tables["signed_control_seed_effects"], [*base_keys, "seed"],
          [*base_values, cfg["phase1_training_seeds"]], "seed effects")
    _grid(tables["signed_mode_summary"], [*base_keys, "mode"], [*base_values, MODES], "mode effects")
    q = pd.read_csv(check_info(inputs["tables"]["subspace_diagnostics_per_record"]),
                    usecols=["object", "snr", "noise_seed", "mode", "record_index", "ecg_id", "patient_id", "q", "zero_norm"])
    averages = _q_averages(cfg, q, freeze)
    del q
    audit = pd.read_csv(check_info(inputs["tables"]["input_validation"]),
                        usecols=["condition", "snr", "noise_seed", "record_index", "patient_id", "snr_abs_error_db",
                                 "max_lead_rms_relative_error", "max_periodogram_peak_relative_error",
                                 "max_base_sign_absolute_error", "max_scaled_sign_absolute_error",
                                 "max_designed_periodogram_peak_relative_error"])
    _require(len(audit) == 126 * freeze["n_records"] and
             not audit.duplicated(["condition", "snr", "noise_seed", "record_index"]).any(), "Incomplete input audit")
    source_infos = [file_info(path) for path in dict.fromkeys(source_paths)]
    paths["figures"].mkdir(parents=True, exist_ok=True)
    manifest_path = paths["figures"] / "manifest.json"
    save_json(manifest_path, dict(status="running", stage=stage, config_sha256=cfg["_config_sha256"]))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    figures = []
    _structure(paths, tables["signed_control_summary"], tables["signed_control_seed_effects"], tables["signed_mode_summary"], figures)
    _snr(paths, tables["signed_control_summary"], figures)
    _matrices(paths, matrices, controls, matrix_path, controls_path, figures)
    _subspace(paths, averages, figures)
    _geometry(paths, matrices, matrix_path, figures)
    _validation(cfg, paths, audit, figures)
    _require(tuple(item["id"] for item in figures) == FIGURE_IDS, "Figure package incomplete")
    for info in source_infos:
        check_info(info)
    result = dict(status="completed", stage=stage, config_sha256=cfg["_config_sha256"], figure_count=6,
                  figures=figures, sources=source_infos, implementation=file_info(Path(__file__)),
                  implementation_dependencies=[file_info(path) for path in (
                      Path(__file__).with_name("common.py"), WORKSPACE / "methodology_supplement/plot.py",
                      WORKSPACE / "methodology_supplement/common.py", WORKSPACE / "phase2/src/plot_phase2.py")],
                  display_scale=100, metric_units="percentage points", uncertainty=UNCERTAINTY,
                  q_aggregation="Complete per-ECG fixed-factor averages, then descriptive across-ECG distributions; nulls propagate; no derived CSV.")
    save_json(manifest_path, result)
    print("Saved six full-cohort figures in PNG, PDF and SVG", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=("full",), default="full")
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
