"""Render the preregistered phase-2 figures from completed, persisted results only.

Run from the workspace with ``python -m phase2.src.plot_phase2 --config ...
--stage pilot|full``. Missing matrix cells are errors, never optional figures.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd

from phase1_ecg_robustness.src.plotting import (
    FigureWriter,
    KIND_LABELS,
    LABELS as CONDITION_LABELS,
    _errorbar,
    _legend,
    _panels,
)
from .common import (
    expected_runs,
    file_info,
    load_config,
    require_preregistration,
    run_directory,
    save_json,
    stage_paths,
)

STRATEGIES = ("clean_only", "independent_rms", "electrode", "mixed")
LABELS = {
    "clean_only": "Clean only",
    "independent_rms": "Independent RMS augmentation",
    "electrode": "Electrode augmentation",
    "mixed": "Mixed augmentation",
}
COLORS = dict(zip(STRATEGIES, ("#595959", "#0072B2", "#D55E00", "#009E73")))
SHORT_LABELS = dict(zip(STRATEGIES, ("Clean", "Ind. RMS", "Electrode", "Mixed")))
METRICS = {
    "macro_auroc": "Macro AUROC",
    "macro_ap": "Macro AP",
    "macro_f1": "Macro F1",
    "brier": "Brier score (lower is better)",
    "ece": "Classwise ECE (lower is better)",
}
OUTCOMES = {
    "absolute": "Absolute",
    "retention": "Noisy / clean",
    "drop": "Clean - noisy",
}
DESCRIPTORS = ["group_id", "kind", "combo_set", "combo_id", "condition", "snr"]
FAMILIES = {
    "clean",
    "gaussian_snr",
    "heldout_heatmaps",
    "primary_effects",
    "ranks_seeds",
    "calibration",
    "noise_stability",
    "nstdb_sensitivity",
    "training_validation",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _select(frame, **filters):
    mask = pd.Series(True, index=frame.index)
    for key, value in filters.items():
        mask &= frame[key].eq(value)
    return frame.loc[mask]


def _one(frame, **filters):
    selected = _select(frame, **filters)
    _require(
        len(selected) == 1, f"Expected one row for {filters}; found {len(selected)}"
    )
    return selected.iloc[0]


def _finite(values, context):
    values = np.asarray(values, dtype=float)
    _require(np.isfinite(values).all(), f"Undefined required figure values: {context}")
    return values


def _interval(
    ax, point, low, high, position, *, color, marker, label=None, horizontal=False
):
    """Draw asymmetric percentile intervals without assuming they enclose the point."""
    _finite([point, low, high], "required interval endpoints")
    _require(low <= high, "Reversed interval endpoints")
    if horizontal:
        ax.hlines(position, low, high, color=color, linewidth=1.5)
        ax.plot([low, high], [position, position], "|", color=color)
    else:
        ax.vlines(position, low, high, color=color, linewidth=1.5)
        ax.plot([position, position], [low, high], "_", color=color)
    if horizontal:
        ax.plot(
            point, position, marker=marker, color=color, linestyle="none", label=label
        )
    else:
        ax.plot(
            position, point, marker=marker, color=color, linestyle="none", label=label
        )


def _snr_axis(ax, snrs):
    snrs = sorted(set(int(s) for s in snrs), reverse=True)
    ax.set_xticks(
        snrs,
        [
            f"{s}\n{'unseen' if s in (15, 5) else 'pressure' if s == 0 else 'seen SNR'}"
            for s in snrs
        ],
    )
    ax.set_xlim(max(snrs) + 1.2, min(snrs) - 1.2)
    ax.set_xlabel("Nominal SNR (dB)")
    for snr in snrs:
        if snr in (15, 5):
            ax.axvline(snr, color="#999999", linestyle=":", linewidth=0.65, zorder=0)
        elif snr == 0:
            ax.axvline(snr, color="#A33A32", linestyle="--", linewidth=0.8, zorder=0)


class PlotInputs:
    def __init__(self, cfg, stage):
        self.cfg, self.stage = cfg, stage
        self.paths = stage_paths(cfg, stage)
        self.models = list(cfg["models"])
        self.seeds = list(cfg["stages"][stage]["seeds"])
        self.noise_seeds = list(cfg["stages"][stage]["test_noise_seeds"])
        self.snrs = list(cfg["stages"][stage]["test_snrs"])
        self.sources, self.frames = {}, {}
        self.protocol_paths = [Path(__file__), Path(cfg["_config_path"])]
        for name in ("evaluation", "statistics"):
            path = self.paths["logs"] / f"{name}_protocol.json"
            protocol = json.loads(path.read_text(encoding="utf-8"))
            _require(
                protocol.get("status") == "completed",
                f"Incomplete {name} protocol: {path}",
            )
            _require(
                protocol.get("config_sha256") == cfg["_config_sha256"],
                f"Stale {name} configuration",
            )
            self.protocol_paths.append(path)
        manifest_path = self.paths["test_inputs"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(manifest.get("status") == "completed", "Test cache is incomplete")
        _require(
            manifest.get("config_sha256") == cfg["_config_sha256"],
            "Test cache configuration differs",
        )
        self.protocol_paths.append(manifest_path)
        self.groups = pd.DataFrame(manifest["groups"])
        _require(
            not self.groups.empty and not self.groups.group_id.duplicated().any(),
            "Missing or duplicate registered groups",
        )
        self.summary = self.table(
            "seed_summary",
            DESCRIPTORS
            + [
                "model",
                "strategy",
                "metric",
                "outcome",
                "n_seeds",
                "mean",
                "std",
                "ci95_low",
                "ci95_high",
            ],
            ["model", "strategy", "group_id", "metric", "outcome"],
        )
        self.seed = self.table(
            "group_seed_metrics",
            DESCRIPTORS + ["model", "strategy", "seed", "metric", "outcome", "value"],
            ["model", "strategy", "seed", "group_id", "metric", "outcome"],
        )
        self.patient = self.table(
            "patient_ci",
            [
                "model",
                "strategy",
                "training_seed",
                "group_id",
                "metric",
                "outcome",
                "lhs",
                "rhs",
                "point",
                "ci95_low",
                "ci95_high",
            ],
        )
        self.patient["training_seed"] = self.patient.training_seed.astype(str)
        self.patient["rhs"] = self.patient.rhs.fillna("")
        self.effects = self.table(
            "paired_effects",
            [
                "model",
                "lhs",
                "rhs",
                "group_id",
                "metric",
                "outcome",
                "point",
                "sd",
                "n_seeds",
                "ci95_low",
                "ci95_high",
                "patient_ci95_low",
                "patient_ci95_high",
            ],
            ["model", "lhs", "rhs", "group_id", "metric", "outcome"],
        )
        self.primary = self.table(
            "primary_comparisons",
            [
                "model",
                "lhs",
                "rhs",
                "group_id",
                "metric",
                "outcome",
                "point",
                "n_seeds",
                "p_holm",
            ],
            ["model", "lhs", "rhs"],
        )
        self.rank = self.table(
            "rank_seed",
            ["model", "strategy", "seed", "group_id", "metric", "value", "rank"],
            ["model", "strategy", "seed", "group_id", "metric"],
        )
        self.rank_summary = self.table(
            "rank_summary",
            ["model", "strategy", "group_id", "metric", "mean_rank", "std_rank"],
            ["model", "strategy", "group_id", "metric"],
        )
        self.correlations = self.table(
            "rank_correlations",
            [
                "model",
                "seed",
                "metric",
                "group_lhs",
                "group_rhs",
                "comparison",
                "spearman",
                "kendall",
            ],
        )
        self.bins = self.table(
            "calibration_bins",
            [
                "model",
                "strategy",
                "seed",
                "group_id",
                "class_name",
                "bin_id",
                "count",
                "probability_sum",
                "target_sum",
            ],
            ["model", "strategy", "seed", "group_id", "class_name", "bin_id"],
        )
        self.noise = self.table(
            "noise_seed_metrics",
            DESCRIPTORS
            + [
                "model",
                "strategy",
                "n_training_seeds",
                "noise_seed",
                "metric",
                "outcome",
                "value",
            ],
            ["model", "strategy", "noise_seed", "group_id", "metric", "outcome"],
        )
        self.replay = self.table(
            "noise_replay_summary",
            [
                "model",
                "strategy",
                "group_id",
                "metric",
                "outcome",
                "n_noise_seeds",
                "mean",
                "std",
                "min",
                "max",
            ],
            ["model", "strategy", "group_id", "metric", "outcome"],
        )
        self.noise_contrasts = self.table(
            "noise_contrasts",
            [
                "level",
                "model",
                "lhs",
                "rhs",
                "group_id",
                "metric",
                "outcome",
                "seed",
                "noise_seed",
                "point",
                "std",
                "n_noise_seeds",
            ],
        )
        self.trajectories = []
        for strategy, model, seed in expected_runs(cfg, stage):
            path = run_directory(cfg, stage, strategy, model, seed) / "epochs.csv"
            frame = pd.read_csv(path)
            needed = {"epoch", "train_loss", "val_loss", "val_macro_auroc"}
            _require(
                needed <= set(frame),
                f"Missing trajectory fields in {path}: {needed - set(frame)}",
            )
            epochs = int(cfg["stages"][stage]["epochs"])
            _require(
                frame.epoch.tolist() == list(range(1, epochs + 1)),
                f"Incomplete epoch sequence: {path}",
            )
            _finite(frame[list(needed)].to_numpy(), str(path))
            self.trajectories.append((strategy, model, seed, frame, path))
        self.validate_coverage()

    def table(self, name, columns, keys=None):
        path = self.paths["tables"] / f"{name}.csv"
        frame = pd.read_csv(path)
        _require(not frame.empty, f"Mandatory table is empty: {path}")
        _require(
            set(columns) <= set(frame),
            f"Missing fields in {path}: {set(columns) - set(frame)}",
        )
        if keys:
            _require(
                not frame.duplicated(keys).any(),
                f"Duplicate scientific keys in {path}: {keys}",
            )
        self.sources[name], self.frames[name] = path, frame
        return frame

    def validate_coverage(self):
        base_keys = ["model", "strategy", "seed", "group_id"]
        expected = set(
            itertools.product(self.models, STRATEGIES, self.seeds, self.groups.group_id)
        )
        for metric in METRICS:
            rows = _select(self.seed, metric=metric, outcome="absolute")
            actual = set(rows[base_keys].itertuples(index=False, name=None))
            _require(
                actual == expected,
                f"Incomplete {metric} registered group/seed grid; missing={list(expected - actual)[:8]}",
            )
            _finite(rows.value, metric)
        for group in ("clean", "primary_joint"):
            for model, strategy in itertools.product(self.models, STRATEGIES):
                for outcome in (
                    ("absolute", "clean_cost")
                    if group == "clean"
                    else ("absolute", "retention", "drop")
                ):
                    rows = _select(
                        self.seed,
                        model=model,
                        strategy=strategy,
                        group_id=group,
                        metric="macro_auroc",
                        outcome=outcome,
                    )
                    _require(
                        set(rows.seed) == set(self.seeds),
                        f"Incomplete {group}/{outcome} seeds",
                    )
                    _finite(rows.value, f"{group}/{outcome}")
        summary = self.summary[
            self.summary.outcome.eq("absolute") & self.summary.metric.isin(METRICS)
        ]
        _require(
            (summary.n_seeds == len(self.seeds)).all(),
            "Summary has incomplete training seeds",
        )
        if len(self.seeds) == 1:
            _require(
                self.summary[["std", "ci95_low", "ci95_high"]].isna().all().all(),
                "Pilot must not contain invented training SD/t intervals",
            )
        for model, strategy, group, class_name in itertools.product(
            self.models, STRATEGIES, ("clean", "primary_joint"), self.cfg["class_order"]
        ):
            rows = _select(
                self.bins,
                model=model,
                strategy=strategy,
                group_id=group,
                class_name=class_name,
            )
            _require(
                set(rows.seed) == set(self.seeds), "Missing calibration checkpoint"
            )
            for seed in self.seeds:
                bins = rows[rows.seed == seed]
                _require(
                    set(bins.bin_id) == set(range(15)),
                    "Calibration must persist every one of the 15 bins",
                )
                _require(
                    (bins["count"] >= 0).all() and bins["count"].sum() > 0,
                    "Invalid calibration bin counts",
                )
        self.gaussian_groups = self.groups[
            (self.groups.kind == "bandpass")
            & (self.groups.combo_id == "aggregate")
            & (self.groups.snr >= 0)
        ]
        _require(not self.gaussian_groups.empty, "No aggregate Gaussian groups")
        for condition, combo_set in itertools.product(
            self.cfg["test"]["conditions"], ("train", "heldout", "all")
        ):
            rows = _select(
                self.gaussian_groups, condition=condition, combo_set=combo_set
            )
            _require(
                set(rows.snr) == set(self.snrs),
                f"Missing Gaussian condition {condition}/{combo_set}",
            )
        heldout_ids = self.cfg["stages"][self.stage]["heldout_combo_ids"]
        self.heldout_ids = (
            [c["combo_id"] for c in self.cfg["_heldout_combos"]]
            if heldout_ids == "all"
            else list(heldout_ids)
        )

    @property
    def seed_uncertainty(self):
        return (
            f"Mean +/- 1 sample SD across {len(self.seeds)} training seeds (not a CI)."
            if len(self.seeds) > 1
            else "One training seed: points only; training SD and t intervals are not estimable."
        )


class Figures:
    def __init__(self, data):
        self.data = data
        self.writer = FigureWriter(
            data.cfg["_workspace_root"],
            data.paths["figures"],
            f"phase2_{data.stage}",
            data.cfg["sampling_rate"],
            dpi=180,
            pdf=True,
        )
        self.fingerprints = {}

    def save(
        self,
        fig,
        axes,
        name,
        family,
        title,
        caption,
        tables,
        *,
        estimand,
        uncertainty,
        extra_sources=(),
        legend=True,
        colorbar=None,
        colorbar_label="",
    ):
        stage_note = (
            "TECHNICAL PILOT; not confirmatory. " if self.data.stage == "pilot" else ""
        )
        caption = stage_note + caption
        height = fig.get_figheight()
        footer = min(0.30, 1.6 / height)
        fig.suptitle(title, fontsize=13, y=1 - 0.12 / height)
        for ax in np.asarray(axes, dtype=object).ravel():
            for axis in (ax.xaxis, ax.yaxis):
                formatter = axis.get_major_formatter()
                if isinstance(formatter, ScalarFormatter):
                    formatter.set_useOffset(False)
                    formatter.set_scientific(False)
        fig.tight_layout(
            rect=(0, footer, 0.91 if colorbar is not None else 1, 1 - 0.6 / height)
        )
        if colorbar is not None:
            cax = fig.add_axes([0.925, 0.31, 0.012, 0.50])
            fig.colorbar(colorbar, cax=cax, label=colorbar_label)
        fig.text(
            0.02,
            0.5 / height,
            textwrap.fill(caption, width=max(95, int(fig.get_figwidth() * 14))),
            ha="left",
            va="bottom",
            fontsize=8,
        )
        if legend:
            _legend(fig, axes, ncol=4)
        paths = (
            [self.data.sources[t] for t in tables]
            + list(extra_sources)
            + self.data.protocol_paths
        )
        self.writer.save(fig, name, family, caption, paths)
        entry = self.writer.entries[-1]
        entry.update(estimand=estimand, uncertainty=uncertainty)
        for path in paths:
            key = str(path)
            if key not in self.fingerprints:
                self.fingerprints[key] = file_info(path)
        entry["source_files"] = [
            self.fingerprints[str(path)] for path in dict.fromkeys(paths)
        ]


def _strategy_curve(
    ax, rows, strategy, *, x="snr", y="mean", sd="std", count="n_seeds"
):
    rows = rows.sort_values(x, ascending=False)
    _require(not rows.empty, f"No curve rows for {strategy}")
    values = _finite(rows[y], f"{strategy}/{y}")
    errors = np.where(rows[count].to_numpy() > 1, rows[sd].to_numpy(), np.nan)
    _errorbar(
        ax,
        rows[x].to_numpy(),
        values,
        errors,
        color=COLORS[strategy],
        marker="o",
        markersize=4,
        linewidth=1.4,
        label=LABELS[strategy],
    )


def plot_clean(data, figures):
    for outcome, metrics in (
        ("absolute", list(METRICS)),
        ("clean_cost", ["macro_auroc", "macro_ap", "macro_f1"]),
    ):
        fig, axes = _panels(data.models, metrics, width=3.5)
        for i, model in enumerate(data.models):
            for j, metric in enumerate(metrics):
                ax = axes[i, j]
                for k, strategy in enumerate(STRATEGIES):
                    row = _one(
                        data.summary,
                        model=model,
                        strategy=strategy,
                        group_id="clean",
                        metric=metric,
                        outcome=outcome,
                    )
                    raw = _select(
                        data.seed,
                        model=model,
                        strategy=strategy,
                        group_id="clean",
                        metric=metric,
                        outcome=outcome,
                    ).sort_values("seed")
                    _require(
                        set(raw.seed) == set(data.seeds), "Missing clean seed/cost rows"
                    )
                    offset = (
                        np.linspace(-0.11, 0.11, len(raw))
                        if len(raw) > 1
                        else np.array([0.0])
                    )
                    ax.scatter(
                        k + offset,
                        raw.value,
                        color=COLORS[strategy],
                        alpha=0.55,
                        s=15,
                        zorder=3,
                    )
                    _errorbar(
                        ax,
                        [k],
                        [row["mean"]],
                        [row["std"] if row.n_seeds > 1 else np.nan],
                        color=COLORS[strategy],
                        marker="o",
                        label=LABELS[strategy],
                        markersize=5,
                    )
                    if metric == "macro_auroc":
                        patient = _one(
                            data.patient,
                            model=model,
                            strategy=strategy,
                            group_id="clean",
                            metric=metric,
                            outcome=outcome,
                            training_seed="mean_fixed_seeds",
                            rhs="",
                        )
                        _interval(
                            ax,
                            patient.point,
                            patient.ci95_low,
                            patient.ci95_high,
                            k + 0.20,
                            color=COLORS[strategy],
                            marker="s",
                        )
                ax.set_xticks(
                    range(4), ["Clean", "Ind. RMS", "Electrode", "Mixed"], rotation=25
                )
                ax.set_title(f"{model.upper()} | {METRICS[metric]}")
                ax.set_ylabel(
                    "Performance"
                    if outcome == "absolute"
                    else "Clean-only minus strategy"
                )
                if outcome == "clean_cost":
                    ax.axhline(0, color="#777777", linewidth=0.8)
                ax.grid(axis="y", alpha=0.18)
        caption = (
            (
                "Clean-test performance. "
                if outcome == "absolute"
                else "Positive clean cost means worse clean performance than same-seed clean-only; no clinical acceptability margin was prespecified. "
            )
            + data.seed_uncertainty
            + " Small dots are individual training seeds. AUROC squares show 95% patient-cluster percentile CIs conditional on the fixed checkpoints and noise; they do not include training variability."
        )
        figures.save(
            fig,
            axes,
            f"clean_{outcome}",
            "clean",
            (
                "Clean performance"
                if outcome == "absolute"
                else "Clean performance cost (descriptive)"
            ),
            caption,
            ["seed_summary", "group_seed_metrics", "patient_ci"],
            estimand="Clean record-level metric; clean cost = same-seed clean-only minus strategy",
            uncertainty="Training sample SD and individual seeds; AUROC conditional patient-cluster 95% CI separately",
        )


def plot_gaussian(data, figures):
    conditions = data.cfg["test"]["conditions"]
    row_keys = list(itertools.product(data.models, conditions))
    combo_sets = ("train", "heldout", "all")
    for outcome in ("absolute", "retention"):
        fig, axes = _panels(row_keys, combo_sets, height=2.8)
        for i, (model, condition) in enumerate(row_keys):
            for j, combo_set in enumerate(combo_sets):
                ax = axes[i, j]
                groups = _select(
                    data.gaussian_groups, condition=condition, combo_set=combo_set
                ).group_id
                for strategy in STRATEGIES:
                    rows = _select(
                        data.summary,
                        model=model,
                        strategy=strategy,
                        metric="macro_auroc",
                        outcome=outcome,
                    )
                    rows = rows[rows.group_id.isin(groups)]
                    _require(
                        set(rows.snr) == set(data.snrs), "Incomplete primary SNR curve"
                    )
                    _strategy_curve(ax, rows, strategy)
                combination_label = {
                    "train": "Training combinations",
                    "heldout": "Held-out combinations",
                    "all": "All electrodes (stress)",
                }[combo_set]
                ax.set_title(
                    f"{model.upper()} | {CONDITION_LABELS[condition]}\n{combination_label}"
                )
                ax.set_ylabel(f"{OUTCOMES[outcome]} macro AUROC")
                _snr_axis(ax, data.snrs)
                ax.grid(axis="y", alpha=0.18)
        figures.save(
            fig,
            axes,
            f"gaussian_macro_auroc_{outcome}",
            "gaussian_snr",
            f"Gaussian robustness | {OUTCOMES[outcome]} macro AUROC",
            "Equal-case condition metrics, then equal training-seed means. "
            + data.seed_uncertainty
            + " Rows cover both architectures and all three perturbation structures; columns separate training, held-out, and all-electrode combinations. 15/5 dB were unseen in training; 0 dB is pressure. Seen SNR does not imply a combination or perturbation structure was seen; clean-only has no noisy training exposure.",
            ["seed_summary"],
            estimand=f"Equal-condition mean macro AUROC, {outcome}",
            uncertainty="Training-seed sample SD only; fixed noise bases, not independent training replicates",
        )
    for name, metrics in (
        ("ap_f1", ("macro_ap", "macro_f1")),
        ("brier_ece", ("brier", "ece")),
    ):
        column_keys = list(itertools.product(data.models, metrics))
        fig, axes = _panels(conditions, column_keys, width=3.7)
        for i, condition in enumerate(conditions):
            groups = _select(
                data.gaussian_groups, condition=condition, combo_set="heldout"
            ).group_id
            for j, (model, metric) in enumerate(column_keys):
                ax = axes[i, j]
                for strategy in STRATEGIES:
                    rows = _select(
                        data.summary,
                        model=model,
                        strategy=strategy,
                        metric=metric,
                        outcome="absolute",
                    )
                    rows = rows[rows.group_id.isin(groups)]
                    _require(
                        set(rows.snr) == set(data.snrs),
                        "Incomplete supplemental SNR curve",
                    )
                    _strategy_curve(ax, rows, strategy)
                ax.set_title(f"{model.upper()} | {CONDITION_LABELS[condition]}")
                ax.set_ylabel(METRICS[metric])
                _snr_axis(ax, data.snrs)
                ax.grid(axis="y", alpha=0.18)
        figures.save(
            fig,
            axes,
            f"supplement_gaussian_heldout_{name}",
            "gaussian_snr",
            "Supplement | Gaussian held-out-combination performance",
            "Secondary metrics for held-out combinations, faceted by architecture and perturbation structure. "
            + data.seed_uncertainty
            + " 15/5 dB are unseen and 0 dB is pressure. F1 uses fixed clean-validation thresholds. Complete training/all-electrode secondary curves and all drop values remain in the saved CSVs; they are not omitted from analysis.",
            ["seed_summary"],
            estimand="Equal-condition secondary metric means on held-out combinations",
            uncertainty="Training-seed sample SD; no CI",
        )


def plot_heatmaps(data, figures):
    groups = data.groups[
        (data.groups.kind == "bandpass")
        & (data.groups.condition == "electrode")
        & data.groups.combo_id.isin(data.heldout_ids)
    ]
    for outcome in ("absolute", "retention"):
        rows = _select(data.summary, metric="macro_auroc", outcome=outcome)
        rows = rows[rows.group_id.isin(groups.group_id)]
        values = _finite(rows["mean"], "heatmap")
        _require(len(values) > 0, "Missing held-out heatmap data")
        if outcome == "absolute":
            lo, hi, cmap = 0.0, 1.0, "viridis"
        else:
            lo, hi, cmap = (
                min(0.0, float(values.min())),
                max(1.0, float(values.max())),
                "viridis",
            )
        fig, axes = _panels(
            data.models,
            STRATEGIES,
            width=3.8,
            height=max(3.2, 0.34 * len(data.heldout_ids)),
        )
        for i, model in enumerate(data.models):
            for j, strategy in enumerate(STRATEGIES):
                ax = axes[i, j]
                part = _select(rows, model=model, strategy=strategy)
                matrix = part.pivot(
                    index="combo_id", columns="snr", values="mean"
                ).reindex(
                    index=data.heldout_ids, columns=sorted(data.snrs, reverse=True)
                )
                array = _finite(matrix.to_numpy(), f"{model}/{strategy} heatmap")
                image = ax.imshow(array, vmin=lo, vmax=hi, cmap=cmap, aspect="auto")
                ax.set_yticks(
                    range(len(data.heldout_ids)),
                    [s.replace("_", "+") for s in data.heldout_ids],
                )
                ax.set_xticks(
                    range(len(data.snrs)),
                    [
                        f"{s}\n{'unseen' if s in (15, 5) else 'stress' if s == 0 else 'seen'}"
                        for s in sorted(data.snrs, reverse=True)
                    ],
                )
                ax.set_xlabel("Nominal SNR (dB)")
                ax.set_title(
                    f"{model.upper()} | {LABELS[strategy]}", color=COLORS[strategy]
                )
                for (r, c), value in np.ndenumerate(array):
                    rgba = image.cmap(image.norm(value))
                    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    ax.text(
                        c,
                        r,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if luminance > 0.5 else "white",
                    )
        figures.save(
            fig,
            axes,
            f"heldout_heatmap_{outcome}",
            "heldout_heatmaps",
            "Held-out electrode combinations | Gaussian electrode propagation",
            "Cell values average fixed noise bases then training seeds. Identical color scale in every panel; no uncertainty encoded. Every displayed electrode combination is held out; 15/5 dB are unseen and 0 dB is a pressure test. See seed and interval figures for uncertainty.",
            ["seed_summary"],
            estimand=f"Held-out combination mean macro AUROC {outcome}",
            uncertainty="None encoded; means only",
            legend=False,
            colorbar=image,
            colorbar_label=f"{OUTCOMES[outcome]} macro AUROC",
        )


def plot_primary(data, figures):
    pairs = data.cfg["statistics"]["primary_pairs"]
    fig, axes = _panels(
        data.models, ("retention", "absolute", "drop"), width=5.0, height=3.4
    )
    for i, model in enumerate(data.models):
        for j, outcome in enumerate(("retention", "absolute", "drop")):
            ax = axes[i, j]
            labels = []
            for k, (lhs, rhs) in enumerate(pairs):
                row = _one(
                    data.effects,
                    model=model,
                    lhs=lhs,
                    rhs=rhs,
                    group_id="primary_joint",
                    metric="macro_auroc",
                    outcome=outcome,
                )
                _interval(
                    ax,
                    row.point,
                    row.patient_ci95_low,
                    row.patient_ci95_high,
                    k - 0.12,
                    color=COLORS[lhs],
                    marker="s",
                    horizontal=True,
                    label="Patient-cluster 95% CI (fixed checkpoints)",
                )
                if row.n_seeds > 1:
                    _interval(
                        ax,
                        row.point,
                        row.ci95_low,
                        row.ci95_high,
                        k + 0.12,
                        color="#333333",
                        marker="o",
                        horizontal=True,
                        label="Paired training-seed 95% t CI",
                    )
                label = f"{SHORT_LABELS[lhs]} - {SHORT_LABELS[rhs]}"
                if outcome == "retention":
                    primary = _one(data.primary, model=model, lhs=lhs, rhs=rhs)
                    _require(
                        np.isclose(primary.point, row.point),
                        "Primary effect point differs from paired effect",
                    )
                    if data.stage == "full":
                        _require(
                            np.isfinite(primary.p_holm),
                            "Missing six-family Holm inference",
                        )
                        label += f"\nHolm p = {primary.p_holm:.3g}"
                labels.append(label)
            ax.axvline(0, color="#888888", linewidth=0.8)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_yticks(range(len(pairs)), labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(
                f"{model.upper()} | {OUTCOMES[outcome]}\n{'Confirmatory' if outcome == 'retention' and data.stage == 'full' else 'Descriptive / secondary'}"
            )
            ax.set_xlabel(
                "Paired strategy difference"
                + (
                    " (negative favors lhs)"
                    if outcome == "drop"
                    else " (positive favors lhs)"
                )
            )
            ax.grid(axis="x", alpha=0.18)
    figures.save(
        fig,
        axes,
        "primary_contrast_intervals",
        "primary_effects",
        "Joint unseen-SNR / held-out-combination contrast",
        "Equal-case AUROCs are transformed within checkpoint, then same-seed strategy differences are averaged. Squares: patient-cluster percentile 95% CIs conditional on fixed checkpoints/noise. Circles: paired training-seed t 95% CIs; these are different uncertainties, not a joint interval. "
        + (
            "Six retention tests share one Holm family; absolute/drop panels are secondary."
            if data.stage == "full"
            else "Only available pilot unseen SNRs enter; n=1, no training CI or confirmatory p value."
        ),
        ["paired_effects", "primary_comparisons"],
        estimand="Mean paired difference in joint-unseen macro AUROC retention (primary), absolute AUROC and drop (secondary)",
        uncertainty="Separate paired-training t CI and conditional paired-patient percentile CI",
    )


def plot_ranks_and_seeds(data, figures):
    for group in ("clean", "primary_joint"):
        fig, axes = _panels(data.models, ("value", "rank"), width=5.0)
        for i, model in enumerate(data.models):
            for j, field in enumerate(("value", "rank")):
                ax = axes[i, j]
                for seed_index, seed in enumerate(data.seeds):
                    rows = (
                        _select(
                            data.rank,
                            model=model,
                            seed=seed,
                            group_id=group,
                            metric="macro_auroc",
                        )
                        .set_index("strategy")
                        .reindex(STRATEGIES)
                    )
                    values = _finite(rows[field], f"rank {group}/{model}/{seed}")
                    marker = ("o", "s", "^", "D", "v")[seed_index % 5]
                    ax.plot(
                        range(4),
                        values,
                        color="#999999",
                        linewidth=0.8,
                        alpha=0.65,
                        marker=marker,
                        label=f"Training seed {seed}",
                    )
                    for k, strategy in enumerate(STRATEGIES):
                        ax.scatter(
                            k,
                            values[k],
                            color=COLORS[strategy],
                            marker=marker,
                            s=22,
                            zorder=3,
                        )
                if field == "rank":
                    means = [
                        _one(
                            data.rank_summary,
                            model=model,
                            strategy=s,
                            group_id=group,
                            metric="macro_auroc",
                        ).mean_rank
                        for s in STRATEGIES
                    ]
                    ax.plot(
                        range(4),
                        means,
                        "kD--",
                        markersize=4,
                        label="Mean rank (average ties)",
                    )
                    ax.set_ylim(4.3, 0.7)
                    ax.set_yticks([1, 2, 3, 4])
                ax.set_xticks(range(4), ["Clean", "Ind. RMS", "Electrode", "Mixed"])
                ax.set_title(
                    f"{model.upper()} | {'Macro AUROC' if field == 'value' else 'Rank (1 = best)'}"
                )
                ax.grid(axis="y", alpha=0.18)
        figures.save(
            fig,
            axes,
            f"ranks_individual_seeds_{group}",
            "ranks_seeds",
            f"Individual training seeds | {group}",
            "Every line connects the four strategies with the same training seed; no independent noise replicates are counted as training runs. Ranks use average ties. This is descriptive: a local rank change is not evidence of a universal ordering reversal. No uncertainty intervals are drawn.",
            ["rank_seed", "rank_summary"],
            estimand="Within-training-seed strategy AUROC and rank",
            uncertainty="Individual observed training seeds; no interval",
        )
    fig, axes = _panels(data.models, data.cfg["test"]["conditions"])
    for i, model in enumerate(data.models):
        for j, condition in enumerate(data.cfg["test"]["conditions"]):
            ax = axes[i, j]
            groups = _select(
                data.gaussian_groups, condition=condition, combo_set="heldout"
            )[["group_id", "snr"]]
            for strategy in STRATEGIES:
                rows = (
                    _select(
                        data.rank_summary,
                        model=model,
                        strategy=strategy,
                        metric="macro_auroc",
                    )
                    .merge(groups, on="group_id", validate="one_to_one")
                    .sort_values("snr", ascending=False)
                )
                _require(set(rows.snr) == set(data.snrs), "Missing mean-rank SNR cells")
                _errorbar(
                    ax,
                    rows.snr,
                    rows.mean_rank,
                    rows.std_rank if len(data.seeds) > 1 else None,
                    color=COLORS[strategy],
                    marker="o",
                    label=LABELS[strategy],
                )
            _snr_axis(ax, data.snrs)
            ax.set_title(f"{model.upper()} | {CONDITION_LABELS[condition]}")
            ax.set_ylabel("Mean rank (1 = best)")
            ax.set_ylim(4.9, 0.1)
            ax.set_yticks([1, 2, 3, 4])
    figures.save(
        fig,
        axes,
        "rank_snr_heldout",
        "ranks_seeds",
        "Descriptive strategy ranks | Held-out combinations",
        "Macro-AUROC ranks across four strategies, with average ties. "
        + data.seed_uncertainty
        + " Ranks at 15/5 dB are unseen-SNR tests; 0 dB is pressure. Rank movement at one SNR is not a general strategy-ordering claim. All training/all-electrode group ranks remain in the CSVs.",
        ["rank_summary"],
        estimand="Mean within-seed four-strategy rank on held-out combinations",
        uncertainty="Training-seed rank sample SD; none for one seed",
    )
    # Display the preregistered descriptive rank correlations, not a fitted trend.
    group_ids = ["primary_joint"] + data.gaussian_groups.sort_values(
        ["condition", "combo_set", "snr"], ascending=[True, True, False]
    ).group_id.tolist()
    correlations = data.correlations.copy()
    correlations["seed"] = correlations.seed.astype(str)
    fig, axes = _panels(
        data.models,
        ("spearman", "kendall"),
        width=5.0,
        height=max(4.0, 0.16 * len(group_ids)),
    )
    for i, model in enumerate(data.models):
        for j, metric in enumerate(("spearman", "kendall")):
            ax = axes[i, j]
            rows = _select(
                correlations, model=model, seed="mean_fixed_seeds", metric="macro_auroc"
            )
            labels, values = [], []
            for group in group_ids:
                pair = rows[
                    ((rows.group_lhs == "clean") & (rows.group_rhs == group))
                    | ((rows.group_rhs == "clean") & (rows.group_lhs == group))
                ]
                _require(
                    len(pair) == 1,
                    f"Missing/duplicate clean rank correlation: {model}/{group}",
                )
                labels.append(group.replace("bandpass__", "").replace("__", " / "))
                values.append(pair.iloc[0][metric])
            positions = np.arange(len(values))
            ax.axvline(0, color="#999999", linewidth=0.7)
            ax.scatter(values, positions, color="#333333", s=14)
            for y, value in enumerate(values):
                if not np.isfinite(value):
                    ax.text(0, y, "undefined (ties)", fontsize=7, va="center")
            ax.set_yticks(positions, labels, fontsize=6)
            ax.invert_yaxis()
            ax.set_xlim(-1.08, 1.08)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_title(f"{model.upper()} | {metric.title()}")
    figures.save(
        fig,
        [],
        "rank_correlations_clean_noisy",
        "ranks_seeds",
        "Clean-versus-noisy rank agreement (descriptive)",
        "Spearman and Kendall agreement over the four strategies, using mean fixed-training-seed metrics. No hypothesis test or confidence interval; ties can make correlation undefined and are labeled rather than replaced with zero. s15/s5 indicate unseen SNR; s0 is pressure.",
        ["rank_correlations"],
        estimand="Four-strategy clean/noisy rank correlation",
        uncertainty="None; descriptive four-strategy correlations",
        legend=False,
    )


def plot_calibration(data, figures):
    for model in data.models:
        fig, axes = _panels(
            ("clean", "primary_joint"), data.cfg["class_order"], width=3.2, height=3.0
        )
        for i, group in enumerate(("clean", "primary_joint")):
            for j, class_name in enumerate(data.cfg["class_order"]):
                ax = axes[i, j]
                ax.plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=1)
                for strategy in STRATEGIES:
                    rows = _select(
                        data.bins,
                        model=model,
                        strategy=strategy,
                        group_id=group,
                        class_name=class_name,
                    )
                    bins = rows.groupby("bin_id", sort=True)[
                        ["count", "probability_sum", "target_sum"]
                    ].sum()
                    bins = bins[bins["count"] > 0]
                    ax.plot(
                        bins.probability_sum / bins["count"],
                        bins.target_sum / bins["count"],
                        color=COLORS[strategy],
                        marker="o",
                        markersize=3,
                        linewidth=1.1,
                        label=LABELS[strategy],
                    )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_title(f"{group} | {class_name}")
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Observed positive fraction")
                ax.grid(alpha=0.15)
        figures.save(
            fig,
            axes,
            f"calibration_{model}",
            "calibration",
            f"{model.upper()} | Classwise reliability (descriptive)",
            "Saved 15-bin sufficient statistics pooled across fixed checkpoints and group cases. Repeated predictions are descriptive, not independent patients and not an ensemble. Empty bins are omitted, not assigned zero accuracy. No calibration fitting on test data, and no CI. Primary group = available joint unseen-SNR held-out combinations.",
            ["calibration_bins"],
            estimand="Prediction-level bin calibration across fixed group cases/checkpoints",
            uncertainty="None; repeated observations are not treated as independent patients",
        )


def plot_noise_stability(data, figures):
    outcomes = ("absolute", "retention", "drop")
    fig, axes = _panels(data.models, outcomes, width=4.0)
    for i, model in enumerate(data.models):
        for j, outcome in enumerate(outcomes):
            ax = axes[i, j]
            for k, strategy in enumerate(STRATEGIES):
                raw = _select(
                    data.noise,
                    model=model,
                    strategy=strategy,
                    group_id="primary_joint",
                    metric="macro_auroc",
                    outcome=outcome,
                )
                base = raw.set_index("noise_seed").value.reindex(data.noise_seeds)
                _finite(base, "noise realization primary")
                _require(
                    (raw.n_training_seeds == len(data.seeds)).all(),
                    "Incomplete training seeds for noise base",
                )
                offsets = (
                    np.linspace(-0.13, 0.13, len(base)) if len(base) > 1 else [0.0]
                )
                for n, (seed, value) in enumerate(base.items()):
                    ax.plot(
                        k + offsets[n],
                        value,
                        marker=("o", "s", "^", "D", "v")[n % 5],
                        color=COLORS[strategy],
                        markersize=4,
                        linestyle="none",
                        label=f"Noise base {seed}",
                    )
                row = _one(
                    data.replay,
                    model=model,
                    strategy=strategy,
                    group_id="primary_joint",
                    metric="macro_auroc",
                    outcome=outcome,
                )
                _require(
                    int(row.n_noise_seeds) == len(data.noise_seeds),
                    "Incomplete noise replay summary",
                )
                _errorbar(
                    ax,
                    [k + 0.24],
                    [row["mean"]],
                    [row["std"] if row.n_noise_seeds > 1 else np.nan],
                    color="#333333",
                    marker="_",
                    markersize=8,
                    label=(
                        "Noise-base mean +/- SD"
                        if row.n_noise_seeds > 1
                        else "Single noise-base value (no SD)"
                    ),
                )
            ax.set_xticks(range(len(STRATEGIES)), [SHORT_LABELS[s] for s in STRATEGIES])
            ax.set_title(f"{model.upper()} | {OUTCOMES[outcome]} macro AUROC")
            ax.set_ylabel("Training-seed mean per noise base")
            ax.grid(axis="y", alpha=0.18)
    figures.save(
        fig,
        axes,
        "noise_realization_stability",
        "noise_stability",
        "Noise realization stability | Joint unseen group",
        "Each marker is one fixed noise base after averaging paired training seeds. Bars are descriptive sample SD across noise bases, not training SD, not a confidence interval, and not extra independent training runs. "
        + (
            "Only one pilot noise base: no noise SD is estimable."
            if len(data.noise_seeds) == 1
            else "All registered noise bases are displayed."
        ),
        ["noise_seed_metrics", "noise_replay_summary"],
        estimand="Per-noise-base primary group AUROC, retention and drop after fixed-training-seed averaging",
        uncertainty="Descriptive noise-base sample SD; no CI",
    )
    fig, axes = _panels(data.models, ["primary"], width=7.0, height=3.4)
    for i, model in enumerate(data.models):
        ax = axes[i, 0]
        for pair_index, (lhs, rhs) in enumerate(
            data.cfg["statistics"]["primary_pairs"]
        ):
            rows = (
                _select(
                    data.noise_contrasts,
                    level="noise",
                    model=model,
                    lhs=lhs,
                    rhs=rhs,
                    group_id="primary_joint",
                    metric="macro_auroc",
                    outcome="retention",
                )
                .set_index("noise_seed")
                .reindex(data.noise_seeds)
            )
            _finite(rows.point, "per-noise primary contrast")
            ax.plot(
                range(len(data.noise_seeds)),
                rows.point,
                marker=("o", "s", "^")[pair_index],
                color=COLORS[lhs],
                linestyle="--" if rhs == "independent_rms" else "-",
                label=f"{SHORT_LABELS[lhs]} - {SHORT_LABELS[rhs]}",
            )
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.set_xticks(range(len(data.noise_seeds)), [str(s) for s in data.noise_seeds])
        ax.set_xlabel("Fixed test noise base (not a training replicate)")
        ax.set_ylabel("Paired retention difference")
        ax.set_title(model.upper())
        ax.grid(axis="y", alpha=0.18)
    figures.save(
        fig,
        axes,
        "noise_primary_contrast_stability",
        "noise_stability",
        "Direction stability of primary contrasts across noise bases",
        "Each point first averages same-training-seed paired strategy retention differences for one noise base. Lines only connect registered base identifiers; no fitted trend, CI, or extra hypothesis test. Pilot has a single base and cannot establish noise-realization stability.",
        ["noise_contrasts"],
        estimand="Per-noise-base mean paired retention difference",
        uncertainty="Observed noise realizations only; no uncertainty interval",
    )


def plot_nstdb(data, figures):
    if not data.cfg["stages"][data.stage]["nstdb"]:
        figures.writer.skip(
            "nstdb_sensitivity",
            "Explicit preregistration: NSTDB is not part of the technical pilot; mandatory in full stage.",
        )
        return
    row_keys = list(itertools.product(data.models, data.cfg["test"]["conditions"]))
    fig, axes = _panels(row_keys, data.cfg["test"]["nstdb_kinds"], height=2.8)
    for i, (model, condition) in enumerate(row_keys):
        for j, kind in enumerate(data.cfg["test"]["nstdb_kinds"]):
            ax = axes[i, j]
            for strategy in STRATEGIES:
                rows = _select(
                    data.summary,
                    model=model,
                    strategy=strategy,
                    kind=kind,
                    condition=condition,
                    combo_set="all",
                    combo_id="aggregate",
                    metric="macro_auroc",
                    outcome="absolute",
                )
                _require(
                    set(rows.snr) == set(data.cfg["test"]["nstdb_snrs"]),
                    f"Missing mandatory NSTDB cells: {model}/{strategy}/{kind}/{condition}",
                )
                _strategy_curve(ax, rows, strategy)
            _snr_axis(ax, data.cfg["test"]["nstdb_snrs"])
            ax.set_title(
                f"{model.upper()} | {CONDITION_LABELS[condition]}\n{KIND_LABELS[kind]}"
            )
            ax.set_ylabel("Macro AUROC")
            ax.grid(axis="y", alpha=0.18)
    figures.save(
        fig,
        axes,
        "exploratory_nstdb_sensitivity",
        "nstdb_sensitivity",
        "Supplement | EXPLORATORY / CONFOUNDED NSTDB sensitivity",
        "Lower-evidence sensitivity analysis, not a pure electrode-domain causal test: recorded lead-space NSTDB sources are repurposed as electrode disturbances, with source/recording mismatch. All noise types and perturbation structures are shown in the all-electrode stress configuration. "
        + data.seed_uncertainty
        + " No confirmatory inference; 0 dB is pressure. AP/F1, Brier/ECE, and other detailed metrics remain in the persisted CSVs.",
        ["seed_summary"],
        estimand="Exploratory all-electrode NSTDB macro AUROC mean",
        uncertainty="Training-seed sample SD; source-domain confounding not quantified",
    )


def plot_trajectories(data, figures):
    columns = [
        ("train_loss", "Training loss (augmented inputs)"),
        ("val_loss", "Clean-validation loss"),
        ("val_macro_auroc", "Clean-validation macro AUROC"),
    ]
    fig, axes = _panels(data.models, columns, width=4.3)
    for i, model in enumerate(data.models):
        for j, (field, label) in enumerate(columns):
            ax = axes[i, j]
            for strategy in STRATEGIES:
                runs = [
                    entry
                    for entry in data.trajectories
                    if entry[0] == strategy and entry[1] == model
                ]
                _require(
                    {entry[2] for entry in runs} == set(data.seeds),
                    "Incomplete trajectory seed grid",
                )
                values = []
                for _, _, seed, frame, _ in runs:
                    values.append(frame[field].to_numpy())
                    ax.plot(
                        frame.epoch,
                        frame[field],
                        color=COLORS[strategy],
                        linewidth=0.7,
                        alpha=0.28 if len(runs) > 1 else 0.7,
                    )
                ax.plot(
                    runs[0][3].epoch,
                    np.mean(values, axis=0),
                    color=COLORS[strategy],
                    linewidth=1.8,
                    label=LABELS[strategy],
                )
            ax.set_title(f"{model.upper()} | {label}")
            ax.set_xlabel("Completed epoch")
            ax.set_ylabel(label)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.grid(alpha=0.18)
    figures.save(
        fig,
        axes,
        "training_clean_validation_trajectories",
        "training_validation",
        "Training and clean-validation trajectories | All registered epochs",
        "Thin lines show individual training seeds and thick lines their arithmetic mean; no CI. Training losses involve strategy-specific augmented inputs and are not the checkpoint-selection endpoint. Checkpoints are selected by clean-validation macro AUROC, after all fixed epochs; no test-driven epoch or strategy selection.",
        [],
        estimand="Logged epoch-level training loss and clean-validation metrics",
        uncertainty="Individual training trajectories and arithmetic means, no uncertainty interval",
        extra_sources=[entry[4] for entry in data.trajectories],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("pilot", "full"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    require_preregistration(cfg)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    data = PlotInputs(cfg, args.stage)
    figures = Figures(data)
    for plot in (
        plot_clean,
        plot_gaussian,
        plot_heatmaps,
        plot_primary,
        plot_ranks_and_seeds,
        plot_calibration,
        plot_noise_stability,
        plot_nstdb,
        plot_trajectories,
    ):
        plot(data, figures)
    present = {entry["family"] for entry in figures.writer.entries}
    required = FAMILIES - ({"nstdb_sensitivity"} if args.stage == "pilot" else set())
    _require(
        required <= present, f"Missing mandatory figure families: {required - present}"
    )
    _require(
        (
            not figures.writer.skipped
            if args.stage == "full"
            else len(figures.writer.skipped) == 1
        ),
        "Unexpected silently skipped figure family",
    )
    manifest_path = figures.writer.finish(args.config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        status="completed",
        stage=args.stage,
        config_sha256=cfg["_config_sha256"],
        strategy_colors=COLORS,
        required_families=sorted(required),
        training_seeds=data.seeds,
        noise_seeds=data.noise_seeds,
    )
    save_json(manifest_path, manifest)
    print(
        f"Saved {len(figures.writer.entries)} PNG+PDF figures; manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
