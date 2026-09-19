"""Publication figures from verified tables; no fitting or post-hoc selection."""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from phase2.src.plot_phase2 import COLORS, SHORT_LABELS
from .common import WORKSPACE, file_info, load_config, read_json, save_json, stage_paths, write_csv

MODELS = {"resnet": "ResNet", "tcn": "TCN"}
NAMES = {"standard": "Standard A", "limb_only": "Limb only", "precordial_only": "Precordial only", "randomized": "Randomized mean"}
BANDS = {"band_0p05_5": "0.05–5 Hz", "band_5_15": "5–15 Hz", "band_15_40": "15–40 Hz", "band_0p5_40": "0.5–40 Hz"}
METRICS = {"macro_auroc": "Macro AUROC", "macro_f1": "Macro F1", "ece": "Classwise ECE"}
MATRIX_COLORS = dict(zip(NAMES, ("#595959", "#0072B2", "#D55E00", "#009E73")))


def _name(identifier):
    return NAMES.get(identifier, f"Random {int(identifier[-2:]) + 1}" if identifier.startswith("random_") else identifier)


def _whiskers(ax, x, row, color, horizontal=False, scale=100):
    mean, low, high, sd = (float(row[key]) * scale for key in ("estimate", "patient_ci_low", "patient_ci_high", "seed_sd"))
    if horizontal:
        ax.hlines(x - .065, low, high, color=color, linewidth=1.1, alpha=.7)
        ax.hlines(x + .065, mean - sd, mean + sd, color=color, linewidth=3)
        ax.plot(mean, x, "o", color=color, markersize=4)
    else:
        ax.vlines(x - .065, low, high, color=color, linewidth=1.1, alpha=.7)
        ax.vlines(x + .065, mean - sd, mean + sd, color=color, linewidth=3)
        ax.plot(x, mean, "o", color=color, markersize=4)


def _two_axes(ax, row, color, marker="o", size=38, alpha=1):
    x, y = 100 * row["x_mean"], 100 * row["y_mean"]
    ax.hlines(y, 100 * row["x_ci_low"], 100 * row["x_ci_high"], color=color, linewidth=.7, alpha=.35 * alpha)
    ax.vlines(x, 100 * row["y_ci_low"], 100 * row["y_ci_high"], color=color, linewidth=.7, alpha=.35 * alpha)
    ax.hlines(y, x - 100 * row["x_seed_sd"], x + 100 * row["x_seed_sd"], color=color, linewidth=1.6, alpha=.75 * alpha)
    ax.vlines(x, y - 100 * row["y_seed_sd"], y + 100 * row["y_seed_sd"], color=color, linewidth=1.6, alpha=.75 * alpha)
    ax.scatter([x], [y], c=[color], marker=marker, s=size, alpha=alpha, zorder=4)


def _foot(fig, text):
    fig.text(.02, .018, text, fontsize=8, color="#444444", va="bottom")


def _save(fig, stem, caption, paths, figures, source_tables):
    files = {}
    for suffix in ("png", "pdf", "svg"):
        destination = paths["figures"] / f"{stem}.{suffix}"
        fig.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
        files[suffix] = file_info(destination)
    plt.close(fig)
    figures.append(dict(id=stem, caption=caption, files=files, sources=[file_info(path) for path in source_tables]))


def _structural_tables(matrices, leads, electrodes, paths):
    overall, columns, rows, similarity = [], [], [], []
    for item in matrices:
        data = item["diagnostics"]
        overall.append(dict(matrix_id=item["id"], family=item["family"], rank=data["rank"],
                            right_nullity=data["right_nullity"], left_nullity=data["left_nullity"],
                            mean_abs_offdiagonal_covariance=data["mean_abs_offdiagonal_covariance"],
                            total_squared_gain=sum(data["column_squared_norms"])))
        for j, electrode in enumerate(electrodes):
            columns.append(dict(matrix_id=item["id"], electrode=electrode, affected_leads=data["column_nonzero_counts"][j],
                                propagation_l2=np.sqrt(data["column_squared_norms"][j])))
            for k, other in enumerate(electrodes):
                similarity.append(dict(matrix_id=item["id"], electrode=electrode, other_electrode=other,
                                       gram=data["column_gram"][j][k], cosine=data["column_cosine"][j][k]))
        for j, lead in enumerate(leads):
            rows.append(dict(matrix_id=item["id"], lead=lead, contributing_electrodes=data["row_nonzero_counts"][j],
                             row_squared_gain=data["row_squared_norms"][j]))
    for name, values in (("matrix_structure", overall), ("matrix_electrodes", columns), ("matrix_leads", rows), ("matrix_column_similarity", similarity)):
        write_csv(paths["tables"] / f"{name}.csv", values)


def _matrix_figures(cfg, paths, matrices, leads, electrodes, data, diagnostics, figures):
    matrix_source = paths["inputs"] / "matrices.json"
    table_source = paths["tables"] / "new_summary.csv"
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for ax, item in zip(axes.flat, matrices):
        image = ax.imshow(item["values"], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(9), electrodes, rotation=60, ha="right")
        ax.set_yticks(range(12), leads)
        ax.set_title(f"{_name(item['id'])} | rank {item['diagnostics']['rank']}")
    fig.subplots_adjust(wspace=.48, hspace=.45, right=.92, bottom=.14)
    fig.colorbar(image, cax=fig.add_axes([.945, .2, .012, .6]), label="Signed acquisition coefficient")
    _foot(fig, "Random controls preserve row/column degree and energy; only five frozen signed-wiring realizations are tested.")
    _save(fig, "matrix_heatmaps", "All eight concrete matrices within the four prespecified families; the five random controls are not independent training runs.", paths, figures, [matrix_source])

    fig, axes = plt.subplots(2, 4, figsize=(17, 11))
    for ax, item in zip(axes.flat, matrices):
        matrix = np.asarray(item["values"])
        left, right = np.linspace(1, 0, 9), np.linspace(1, 0, 12)
        for i, j in zip(*np.nonzero(matrix)):
            ax.plot([0, 1], [left[j], right[i]], color="#0072B2" if matrix[i, j] > 0 else "#D55E00",
                    alpha=.45, linewidth=.3 + abs(matrix[i, j]))
        ax.scatter(np.zeros(9), left, s=14, color="#444444", zorder=3)
        ax.scatter(np.ones(12), right, s=14, color="#444444", zorder=3)
        for j, name in enumerate(electrodes):
            ax.text(-.045, left[j], name, ha="right", va="center", fontsize=8)
        for j, name in enumerate(leads):
            ax.text(1.045, right[j], name, ha="left", va="center", fontsize=8)
        ax.set(xlim=(-.3, 1.3), ylim=(-.08, 1.1), title=_name(item["id"]))
        ax.axis("off")
    fig.subplots_adjust(wspace=.12, hspace=.15, bottom=.07)
    _foot(fig, "Electrodes → leads. Blue: positive; vermilion: negative. Width encodes |A|. Inactive electrodes have no edges.")
    _save(fig, "matrix_bipartite", "Signed electrode-to-lead propagation graphs, including disconnected inactive electrode nodes.", paths, figures, [matrix_source])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for item in matrices:
        color = MATRIX_COLORS[item["family"]]
        alpha = .42 if item["family"] == "randomized" else 1
        values = item["diagnostics"]
        axes[0].plot(range(9), np.sqrt(values["column_squared_norms"]), marker="o", markersize=3,
                     label=_name(item["id"]), color=color, alpha=alpha)
        axes[1].plot(range(9), values["column_nonzero_counts"], marker="o", markersize=3, color=color, alpha=alpha)
    axes[0].set_ylabel("Unit-source propagation gain ||A[:,e]||₂")
    axes[1].set_ylabel("Number of affected leads")
    for ax in axes:
        ax.set_xticks(range(9), electrodes, rotation=45)
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(fontsize=7, ncol=2)
    fig.subplots_adjust(bottom=.22, wspace=.25)
    _foot(fig, "Gains describe the unnormalized linear mapping, not the final per-record SNR-scaled noise. Random-control curves overlap by construction.")
    _save(fig, "matrix_unit_propagation", "Unit-electrode L2 propagation strength and lead coverage; inactive columns are reported as zero, not omitted.", paths, figures, [matrix_source])

    fig, ax = plt.subplots(figsize=(8, 4.7))
    for item in matrices:
        ax.plot(range(1, 10), item["diagnostics"]["singular_values"], marker="o", markersize=3,
                label=_name(item["id"]), color=MATRIX_COLORS[item["family"]], alpha=.55 if item["family"] == "randomized" else 1)
    ax.set(xlabel="Singular-value index", ylabel="Singular value", xticks=range(1, 10))
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.15)
    fig.subplots_adjust(bottom=.19)
    _foot(fig, "Both right and left nullities are reported in matrix_structure.csv. Polarity randomization can change rank.")
    _save(fig, "matrix_singular_values", "Singular-value spectra of all concrete acquisition matrices on a common linear scale.", paths, figures, [matrix_source])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for mi, (model, model_name) in enumerate(MODELS.items()):
        for si, snr in enumerate(cfg["ablation_snrs"]):
            ax = axes[mi, si]
            for index, source in enumerate(NAMES):
                selected = data[(data.analysis == "matrix") & (data.model == model) & (data.snr == snr)
                                & (data.source_id == source) & (data.metric == "macro_auroc") & (data.outcome == "structure_effect")]
                if len(selected) != 1:
                    raise ValueError("Incomplete matrix-forest cell")
                _whiskers(ax, index, selected.iloc[0], MATRIX_COLORS[source], horizontal=True)
            ax.axvline(0, color="#888888", linewidth=.7)
            ax.set_yticks(range(4), list(NAMES.values()))
            ax.invert_yaxis()
            ax.set(title=f"{model_name} | {snr} dB", xlabel="Electrode − independent-RMS AUROC (pp)")
    fig.subplots_adjust(bottom=.14, hspace=.5, wspace=.45)
    _foot(fig, "Thin interval: paired patient-cluster 95% CI. Thick interval: ±1 training-seed SD (n=3). Randomized mean averages five fixed matrices.")
    _save(fig, "matrix_structure_effects", "Matrix-specific structure contrasts at 10 and 0 dB; every matrix has its own per-record, per-lead RMS comparator.", paths, figures, [table_source])

    corr = diagnostics[diagnostics.condition.eq("electrode")].groupby("source_id").effective_cross_lead_correlation.mean()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    random_markers = dict(zip((f"random_{i:02d}" for i in range(5)), ("o", "s", "^", "D", "v")))
    for mi, (model, model_name) in enumerate(MODELS.items()):
        for si, snr in enumerate(cfg["ablation_snrs"]):
            ax = axes[mi, si]
            for item in matrices:
                analysis = "matrix_instance" if item["family"] == "randomized" else "matrix"
                subset = data[(data.analysis == analysis) & (data.model == model) & (data.snr == snr)
                              & (data.source_id == item["id"]) & (data.metric == "macro_auroc") & (data.outcome == "electrode_drop")]
                row = subset.iloc[0]
                x = corr[item["id"]]
                color = MATRIX_COLORS[item["family"]]
                ax.vlines(x, 100 * row.patient_ci_low, 100 * row.patient_ci_high, color=color, alpha=.6)
                ax.scatter(x, 100 * row.estimate, color=color, s=30, marker=random_markers.get(item["id"], "o"))
                if item["family"] != "randomized":
                    ax.annotate(_name(item["id"]), (x, 100 * row.estimate), xytext=(4, 4), textcoords="offset points", fontsize=7)
            ax.set(title=f"{model_name} | {snr} dB", xlabel="Mean |off-diagonal lead correlation|", ylabel="Clean − electrode AUROC (pp)")
            ax.grid(alpha=.12)
    handles = [Line2D([], [], marker=marker, linestyle="none", color=MATRIX_COLORS["randomized"], label=_name(identifier))
               for identifier, marker in random_markers.items()]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5, .98), fontsize=8)
    fig.subplots_adjust(top=.89, bottom=.16, hspace=.45, wspace=.27)
    _foot(fig, "Correlation uses actual 0-dB injected residuals and only active leads; SNR scaling otherwise preserves correlation.\nDifferent families can have different marginal energy profiles. Shapes identify the five fixed random matrices.")
    _save(fig, "matrix_correlation_drop", "Descriptive correlation-magnitude versus degradation plot; topology, rank and local energy profiles are not reduced to this single correlation statistic. Vertical whiskers: conditional paired patient 95% CIs.", paths, figures, [table_source, paths["tables"] / "noise_diagnostics.csv"])


def _snr_figures(cfg, paths, new, reused, figures):
    for metric, title in METRICS.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
        for mi, (model, model_name) in enumerate(MODELS.items()):
            for phase, frame, ax in (("phase1", new, axes[0, mi]), ("phase2", reused, axes[1, mi])):
                strategies = ["clean_only"] if phase == "phase1" else cfg["strategies"]
                for offset, strategy in enumerate(strategies):
                    subset = frame[(frame.analysis == "snr") & (frame.model == model) & (frame.strategy == strategy)
                                   & (frame.metric == metric) & (frame.outcome == "structure_effect")].sort_values("snr")
                    if len(subset) != 5:
                        raise ValueError("SNR figure must contain all five levels")
                    jitter = (offset - (len(strategies) - 1) / 2) * .18
                    ax.plot(subset.snr + jitter, 100 * subset.estimate, color=COLORS[strategy], linewidth=1, label=SHORT_LABELS[strategy])
                    for _, row in subset.iterrows():
                        _whiskers(ax, row.snr + jitter, row, COLORS[strategy])
                ax.axhline(0, color="#888888", linewidth=.7)
                ax.set_title(f"{model_name} | {'Phase 1: 3 seeds' if phase == 'phase1' else 'Phase 2: 5 seeds'}")
                ax.set_ylabel(f"Electrode − independent-RMS\n{title} (pp)")
                ax.set_xlim(21, -1)
                ax.set_xticks(cfg["snrs"])
                ax.grid(alpha=.12)
                if phase == "phase2":
                    ax.set_xlabel("SNR (dB); noise increases to the right")
                    ax.legend(fontsize=8, ncol=2, loc="upper left" if metric == "ece" else "lower left")
        fig.subplots_adjust(bottom=.16, hspace=.37, wspace=.3)
        sign = "Positive ECE differences mean worse calibration." if metric == "ece" else "Positive differences favor electrode-structured tests."
        _foot(fig, f"Thin: paired patient 95% CI; thick: ±1 training-seed SD. Fixed noises: phase 1 n=6; phase 2 n=5. Phases are not pooled.\n{sign} Small x offsets separate strategies at the same SNR.")
        _save(fig, f"snr_{metric}", f"Five-SNR {title} structure-effect curves, separately faceted by phase and architecture with both uncertainty levels.", paths, figures, [paths["tables"] / "new_summary.csv", paths["tables"] / "phase2_snr_summary.csv"])


def _band_figures(cfg, paths, data, diagnostics, figures):
    band_data = data[data.analysis.eq("band") & data.metric.eq("macro_auroc")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    heatmaps = []
    for model in MODELS:
        for snr in cfg["ablation_snrs"]:
            sub = band_data[band_data.model.eq(model) & band_data.snr.eq(snr)]
            values = np.array([[float(sub[sub.source_id.eq(band) & sub.outcome.eq(outcome)].estimate.iloc[0]) * 100
                                for outcome in ("independent_rms_drop", "electrode_drop")] for band in BANDS])
            heatmaps.append(values)
    limit = max(abs(value).max() for value in heatmaps)
    for ax, values, (model, snr) in zip(axes.flat, heatmaps, [(m, s) for m in MODELS for s in cfg["ablation_snrs"]]):
        image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks([0, 1], ["Independent RMS", "Electrode"])
        ax.set_yticks(range(4), list(BANDS.values()))
        ax.set_title(f"{MODELS[model]} | {snr} dB")
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", color="white" if abs(value) > .55 * limit else "black")
    fig.subplots_adjust(bottom=.15, hspace=.36, right=.88, wspace=.4)
    fig.colorbar(image, cax=fig.add_axes([.91, .22, .017, .6]), label="Clean − noisy AUROC (pp)")
    _foot(fig, "All four bands use the same 100-second generation / central 10-second crop protocol. Numeric uncertainty is in new_summary.csv.")
    _save(fig, "band_drop_heatmap", "Frequency-band by structure degradation under an identical long-source protocol; the 0.5–40-Hz branch is not silently equated to the legacy periodic branch.", paths, figures, [paths["tables"] / "new_summary.csv"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    band_colors = dict(zip(BANDS, ("#595959", "#0072B2", "#D55E00", "#009E73")))
    for mi, (model, name) in enumerate(MODELS.items()):
        for si, snr in enumerate(cfg["ablation_snrs"]):
            ax = axes[mi, si]
            for index, band in enumerate(BANDS):
                row = band_data[band_data.model.eq(model) & band_data.snr.eq(snr) & band_data.source_id.eq(band)
                                & band_data.outcome.eq("structure_effect")].iloc[0]
                _whiskers(ax, index, row, band_colors[band], horizontal=True)
            ax.axvline(0, color="#888888", linewidth=.7)
            ax.set_yticks(range(4), list(BANDS.values()))
            ax.invert_yaxis()
            ax.set(title=f"{name} | {snr} dB", xlabel="Electrode − independent-RMS AUROC (pp)")
    fig.subplots_adjust(bottom=.14, hspace=.45, wspace=.35)
    _foot(fig, "Thin: paired patient-cluster 95% CI; thick: ±1 training-seed SD (n=3). Six fixed noise realizations; no new significance tests.")
    _save(fig, "band_structure_effects", "Prespecified frequency-band structure contrasts at 10 and 0 dB, with distinct seed and conditional patient uncertainty.", paths, figures, [paths["tables"] / "new_summary.csv"])

    power = diagnostics[diagnostics.condition.eq("electrode")].groupby("source_id").power_fraction_0_5.mean()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for mi, (model, name) in enumerate(MODELS.items()):
        for si, snr in enumerate(cfg["ablation_snrs"]):
            ax = axes[mi, si]
            for band in BANDS:
                row = band_data[band_data.model.eq(model) & band_data.snr.eq(snr) & band_data.source_id.eq(band)
                                & band_data.outcome.eq("structure_effect")].iloc[0]
                x = 100 * power[band]
                ax.vlines(x, 100 * row.patient_ci_low, 100 * row.patient_ci_high, color=band_colors[band])
                ax.scatter(x, 100 * row.estimate, color=band_colors[band], s=30)
                ax.annotate(BANDS[band], (x, 100 * row.estimate), xytext=(3, 5), textcoords="offset points", fontsize=7)
            ax.axhline(0, color="#888888", linewidth=.7)
            ax.set(title=f"{name} | {snr} dB", xlabel="Observed noise power below 5 Hz (%)", ylabel="AUROC structure effect (pp)")
            ax.margins(x=.15)
    fig.subplots_adjust(bottom=.17, hspace=.45, wspace=.3)
    _foot(fig, "Power is measured after cropping, demeaning, scaling and float32 addition (0-dB audit); DC excluded.\nThe 10-second periodogram resolves 0.1 Hz, not 0.05 Hz. Scatter is descriptive; no regression is fitted.")
    _save(fig, "band_power_effect", "Measured low-frequency power share versus the structure contrast; target passband and observed cropped-record spectrum are kept distinct. Vertical whiskers: conditional paired patient 95% CIs; seed SD is shown in the band forest.", paths, figures, [paths["tables"] / "new_summary.csv", paths["tables"] / "noise_diagnostics.csv"])


def _plane_figures(cfg, paths, frame, figures):
    strategies = cfg["strategies"][1:]
    snr_colors = {value: plt.get_cmap("viridis")(.10 + .15 * i) for i, value in enumerate(cfg["snrs"])}
    for main in (True, False):
        selected = frame[frame.kind.eq("bandpass") if main else ~frame.kind.eq("bandpass")]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
        for mi, (model, name) in enumerate(MODELS.items()):
            for si, strategy in enumerate(strategies):
                ax = axes[mi, si]
                subset = selected[selected.model.eq(model) & selected.strategy.eq(strategy)]
                expected = 105 if main else 9
                if len(subset) != expected:
                    raise ValueError("Effect-plane panel is incomplete")
                for row in subset.to_dict("records"):
                    marker = "o" if main else {"bw": "o", "ma": "s", "em": "^"}[row["kind"]]
                    _two_axes(ax, row, snr_colors[row["snr"]], marker=marker, size=13 if main else 32, alpha=.7)
                ax.axvline(0, color="#aaaaaa", linewidth=.6)
                ax.axhline(0, color="#aaaaaa", linewidth=.6)
                ax.set_title(f"{name} | {SHORT_LABELS[strategy]}")
                if mi == 1:
                    ax.set_xlabel("Clean-only E − I AUROC (pp)")
                if si == 0:
                    ax.set_ylabel("Strategy − clean-only AUROC\nunder electrode tests (pp)")
        legend = [Line2D([], [], marker="o", linestyle="none", color=color, label=f"{snr} dB") for snr, color in snr_colors.items() if main or snr in (20, 10, 0)]
        if not main:
            legend += [Line2D([], [], marker=marker, linestyle="none", color="#444444", label=kind) for kind, marker in (("bw", "o"), ("ma", "s"), ("em", "^"))]
        fig.legend(handles=legend, loc="upper center", ncol=len(legend), bbox_to_anchor=(.5, .975), fontsize=8)
        fig.subplots_adjust(top=.89, bottom=.18, hspace=.28, wspace=.14)
        caution = "Gaussian: all 21 combinations × five SNRs." if main else "NSTDB: confounded replay sensitivity only; not isolated cross-lead-structure evidence."
        _foot(fig, caution + "\nThin: paired patient 95% CI; thick: ±1 training-seed SD (n=5). Axes share a baseline; no causal slope or independence is claimed.")
        stem = "evaluation_training_plane" if main else "evaluation_training_nstdb"
        _save(fig, stem, caution + " Horizontal effects use phase-two clean-only checkpoints; vertical effects compare strategies under the same electrode tests.", paths, figures, [paths["tables"] / "effect_plane.csv"])


def _pareto(paths, frame, figures):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (model, name) in zip(axes, MODELS.items()):
        sub = frame[frame.model.eq(model)]
        for row in sub.to_dict("records"):
            _two_axes(ax, row, COLORS[row["strategy"]], marker="o" if not row["point_dominated"] else "x", size=50)
            ax.annotate(SHORT_LABELS[row["strategy"]], (100 * row["x_mean"], 100 * row["y_mean"]),
                        xytext=(6, 7), textcoords="offset points", fontsize=9)
        frontier = sub[~sub.point_dominated.astype(bool)].sort_values("x_mean")
        ax.plot(100 * frontier.x_mean, 100 * frontier.y_mean, color="#888888", linestyle="--", linewidth=.8, zorder=1)
        ax.axvline(0, color="#aaaaaa", linewidth=.6)
        ax.axhline(0, color="#aaaaaa", linewidth=.6)
        ax.set(title=f"{name} | upper-left is better", xlabel="Clean AUROC cost (pp; lower is better)",
               ylabel="Joint-unseen retention gain (pp)")
        ax.margins(x=.25, y=.18)
    fig.subplots_adjust(bottom=.22, wspace=.27)
    _foot(fig, "Thin: paired patient 95% CI; thick: ±1 training-seed SD (n=5). Ratios are formed within checkpoint/draw before averaging.\nDashed frontier / × markers use point estimates within each model only; no significance or clinical acceptability is implied.")
    _save(fig, "pareto", "Clean-cost versus registered joint-unseen retention benefit. Lower clean cost and higher benefit define the upper-left direction; absolute AUROCs remain in pareto.csv.", paths, figures, [paths["tables"] / "pareto.csv"])


def run(config=None, stage="full"):
    if stage != "full":
        raise ValueError("Publication figures require the full selected experiment")
    cfg = load_config(config)
    paths = stage_paths(cfg, stage)
    for name in ("new_statistics", "reused_statistics"):
        report = read_json(paths["logs"] / f"{name}.json")
        if report["status"] != "completed" or report["config_sha256"] != cfg["_config_sha256"]:
            raise ValueError("Cannot plot incomplete or differently configured results")
    paths["figures"].mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    matrix_file = read_json(paths["inputs"] / "matrices.json")
    matrices = matrix_file["matrices"]
    new = pd.read_csv(paths["tables"] / "new_summary.csv")
    existing = pd.read_csv(paths["tables"] / "phase2_snr_summary.csv")
    diagnostics = pd.read_csv(paths["tables"] / "noise_diagnostics.csv")
    figures = []
    _structural_tables(matrices, matrix_file["lead_order"], matrix_file["electrode_order"], paths)
    _matrix_figures(cfg, paths, matrices, matrix_file["lead_order"], matrix_file["electrode_order"], new, diagnostics, figures)
    _snr_figures(cfg, paths, new, existing, figures)
    _band_figures(cfg, paths, new, diagnostics, figures)
    _plane_figures(cfg, paths, pd.read_csv(paths["tables"] / "effect_plane.csv"), figures)
    _pareto(paths, pd.read_csv(paths["tables"] / "pareto.csv"), figures)
    if len(figures) != 15:
        raise ValueError("Figure package incomplete")
    result = dict(status="completed", stage=stage, config_sha256=cfg["_config_sha256"],
                  figure_count=len(figures), figures=figures, implementation=file_info(Path(__file__)),
                  display_scale=100, display_units="percentage points for metric differences; percent for power fractions",
                  uncertainty="Seed SD and conditional paired patient CI are distinct and are never added together.")
    save_json(paths["figures"] / "manifest.json", result)
    print(f"Saved {len(figures)} full-cohort figures in PNG, PDF and SVG", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=("full",), default="full")
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
