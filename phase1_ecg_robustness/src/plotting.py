"""Scientific figures from persisted phase-1 metrics, predictions and noise audits.

Run ``python -m src.plotting --config configs/phase1_smoke.yaml [--pdf]``.
No simulations, model inference or inferential statistics are performed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import welch

from .datasets import CLASSES
from .lead_matrix import ELECTRODES, LEADS

CONDITIONS = ("independent", "independent_rms", "electrode", "covariance")
LABELS = {
    "independent": "Independent",
    "independent_rms": "RMS-matched independent",
    "electrode": "Electrode-propagated",
    "covariance": "Covariance-matched",
}
COLORS = dict(zip(CONDITIONS, ("#0072B2", "#E69F00", "#009E73", "#CC79A7")))
KIND_LABELS = {
    "bandpass": "Band-limited Gaussian",
    "bw": "Baseline wander",
    "ma": "Muscle artifact",
    "em": "Electrode motion",
}
KEYS = ["model", "kind", "snr", "condition", "active"]


def _mean_sd(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return (
        float(values.mean()) if len(values) else np.nan,
        float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        len(values),
    )


def _errorbar(ax, x, y, errors=None, **kwargs):
    """Never draw a zero-width/fabricated uncertainty interval for one seed."""
    x, y = np.asarray(x), np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if not finite.any():
        return
    color = kwargs.get("color", "#333333")
    ax.plot(x[finite], y[finite], **kwargs)
    if errors is not None:
        errors = np.asarray(errors, dtype=float)
        valid = finite & np.isfinite(errors) & (errors >= 0)
        if valid.any():
            ax.errorbar(
                x[valid],
                y[valid],
                yerr=errors[valid],
                fmt="none",
                color=color,
                capsize=3,
                linewidth=1,
                label="_nolegend_",
            )


class FigureWriter:
    def __init__(self, root, output, run, fs, dpi=180, pdf=False):
        self.root, self.output = Path(root), Path(output)
        self.run, self.fs = run, fs
        self.dpi, self.pdf = max(160, int(dpi)), pdf
        self.entries, self.skipped = [], []
        self.output.mkdir(parents=True, exist_ok=True)

    def source(self, path):
        path = Path(path).resolve()
        try:
            return path.relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def save(self, fig, name, family, caption, sources):
        files = []
        try:
            for extension in (["png", "pdf"] if self.pdf else ["png"]):
                path = self.output / f"{name}.{extension}"
                fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
                files.append(self.source(path))
            self.entries.append(
                {
                    "family": family,
                    "files": files,
                    "caption": caption,
                    "source_paths": list(
                        dict.fromkeys(self.source(p) for p in sources)
                    ),
                }
            )
        finally:
            plt.close(fig)

    def skip(self, family, reason):
        self.skipped.append({"family": family, "reason": reason})

    def finish(self, config_path):
        manifest = {
            "run_name": self.run,
            "sampling_rate_hz": self.fs,
            "dpi": self.dpi,
            "config_path": self.source(config_path),
            "figures": self.entries,
            "skipped": self.skipped,
            "interpretation": "Experimental robustness results, not clinical deployment evidence. Seed SD is descriptive; the inference unit is stated for confidence intervals.",
        }
        path = self.output / "figure_manifest.json"
        path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return path


def _legend(fig, axes, ncol=4):
    handles, labels = [], []
    for ax in np.asarray(axes, dtype=object).ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(ncol, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
            fontsize=8,
        )


def _panels(models, kinds, width=4.2, height=3.1):
    return plt.subplots(
        len(models),
        len(kinds),
        squeeze=False,
        figsize=(max(6, width * len(kinds)), height * len(models) + 1.1),
    )


def _title(model, kind):
    return f"{model} | {KIND_LABELS.get(kind, kind)}"


def plot_examples(writer, example_path):
    with np.load(example_path, allow_pickle=False) as saved:
        x, a = saved["x"], saved["A"]
        fs = float(saved["fs"])
        if x.ndim != 2 or x.shape[0] != 12 or a.shape != (12, 9):
            raise ValueError("Noise example must contain x(12,T) and A(12,9)")
        noises = {condition: saved[condition] for condition in CONDITIONS}
        covariances = {c: saved[f"covariance_{c}"] for c in CONDITIONS}
        correlations = {c: saved[f"correlation_{c}"] for c in CONDITIONS}
    if fs != writer.fs:
        raise ValueError("Noise example sampling rate differs from configuration")
    seconds = min(5, x.shape[1] / fs)
    length = min(x.shape[1], int(seconds * fs))
    time = np.arange(length) / fs
    selected = (0, 1, 5, 6, 8, 11)
    columns = ("clean", "independent", "independent_rms", "electrode", "covariance")
    fig, axes = plt.subplots(
        len(selected), len(columns), figsize=(17, 10), sharex=True, sharey="row"
    )
    for row, lead in enumerate(selected):
        for column, condition in enumerate(columns):
            ax = axes[row, column]
            if condition == "clean":
                signal, color = x[lead, :length], "#333333"
            else:
                signal, color = (
                    x[lead, :length] + noises[condition][lead, :length],
                    COLORS[condition],
                )
                ax.plot(time, x[lead, :length], color="#BBBBBB", lw=0.55, zorder=1)
            ax.plot(time, signal, color=color, lw=0.65, zorder=2)
            if row == 0:
                ax.set_title(
                    "Clean" if condition == "clean" else LABELS[condition], fontsize=10
                )
            if column == 0:
                ax.set_ylabel(f"{LEADS[lead]}\nAmplitude (mV)")
            if row == len(selected) - 1:
                ax.set_xlabel("Time (s)")
            ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        f"One saved ECG example: band-limited noise, 10 dB | {fs:g} Hz", fontsize=14
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    writer.save(
        fig,
        "01_waveforms",
        "waveforms",
        "Six explanatory leads from one saved ECG; identical clean ECG in all columns. Noisy traces are x plus the saved perturbation; gray overlays show clean signal. First five seconds, global target SNR 10 dB, band-limited Gaussian noise. Amplitudes are physical mV after per-lead DC removal, without model normalization; this example is not a population average.",
        [example_path],
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(a, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(9), ELECTRODES)
    ax.set_yticks(range(12), LEADS)
    ax.set(
        xlabel="Diagnostic electrode node (RL excluded)",
        ylabel="ECG lead",
        title=f"Ideal lead-formation matrix A | {fs:g} Hz experiment",
    )
    for row in range(12):
        for col in range(9):
            if a[row, col] != 0:
                ax.text(
                    col,
                    row,
                    f"{a[row, col]:.2g}",
                    ha="center",
                    va="center",
                    color="white" if abs(a[row, col]) > 0.7 else "black",
                    fontsize=9,
                )
    fig.colorbar(image, ax=ax, label="Signed coefficient")
    fig.tight_layout()
    writer.save(
        fig,
        "02_lead_matrix",
        "lead_matrix",
        "Saved 12 by 9 ideal diagnostic lead matrix. Chest leads use minus one third of each limb node as the Wilson central terminal reference. RL and acquisition-electronics effects are excluded; A is sampling-rate independent.",
        [example_path],
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    propagated = a @ np.eye(9)
    image = ax.imshow(propagated.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(12), LEADS)
    ax.set_yticks(range(9), ELECTRODES)
    ax.set(
        xlabel="Affected lead",
        ylabel="Single perturbed electrode",
        title=f"Unit-electrode propagation: A times each unit vector | {fs:g} Hz experiment",
    )
    for row in range(9):
        for col in range(12):
            value = propagated[col, row]
            if value:
                ax.text(
                    col,
                    row,
                    f"{value:.2g}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(value) > 0.7 else "black",
                )
    fig.colorbar(
        image, ax=ax, label="Signed lead response per unit electrode perturbation"
    )
    fig.tight_layout()
    writer.save(
        fig,
        "03_unit_electrode_propagation",
        "unit_electrode_propagation",
        "Deterministic propagation of each of the nine unit electrode vectors through the saved A matrix, without SNR rescaling. Values are signed lead responses per unit electrode amplitude, not estimated model sensitivity.",
        [example_path],
    )

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4), constrained_layout=True)
    covmax = max(float(np.max(np.abs(v))) for v in covariances.values())
    covmax = covmax if covmax > 0 else 1.0
    for col, condition in enumerate(CONDITIONS):
        for row, (matrices, bound) in enumerate(
            ((covariances, covmax), (correlations, 1.0))
        ):
            ax = axes[row, col]
            image = ax.imshow(
                matrices[condition], cmap="RdBu_r", vmin=-bound, vmax=bound
            )
            ax.set_xticks(range(12), LEADS, rotation=90, fontsize=7)
            ax.set_yticks(range(12), LEADS, fontsize=7)
            ax.set_title(LABELS[condition], fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    "Empirical covariance" if row == 0 else "Empirical correlation"
                )
        # Shared color scales are essential for comparing amplitudes.
    fig.colorbar(
        axes[0, -1].images[0],
        ax=list(axes[0]),
        shrink=0.75,
        label="Covariance (mV squared)",
    )
    fig.colorbar(
        axes[1, -1].images[0],
        ax=list(axes[1]),
        shrink=0.75,
        label="Pearson correlation",
    )
    fig.suptitle(f"Noise matrices from one finite-duration example, 10 dB | {fs:g} Hz")
    writer.save(
        fig,
        "04_noise_matrices",
        "covariance_correlation",
        "Empirical noise covariance and correlation from one saved band-limited 10 dB realization, not an aggregate over ECGs. Independent finite realizations can have nonzero cross-lead sample correlations. Each row uses a common color scale across all four conditions.",
        [example_path],
    )
    return noises, fs


def plot_psd(writer, psd_path, example_path, noises, fs):
    if psd_path.exists():
        data = pd.read_csv(psd_path)
        kinds = list(dict.fromkeys(data["kind"]))
        description = (
            "Audit-record and lead-averaged Welch PSD at 10 dB; all electrodes active"
        )
        sources = [psd_path]
    else:
        rows = []
        for condition, noise in noises.items():
            frequency, power = welch(
                noise, fs=fs, nperseg=min(256, noise.shape[-1]), axis=-1
            )
            rows.extend(
                {"kind": "bandpass", "condition": condition, "frequency": f, "power": p}
                for f, p in zip(frequency, power.mean(axis=0))
            )
        data, kinds = pd.DataFrame(rows), ["bandpass"]
        description = (
            "Lead-averaged Welch PSD of one saved 10 dB example (not audit aggregate)"
        )
        sources = [example_path]
    fig, axes = plt.subplots(
        1, len(kinds), figsize=(max(7, 4.3 * len(kinds)), 4.6), squeeze=False
    )
    for ax, kind in zip(axes[0], kinds):
        for condition in CONDITIONS:
            subset = data[
                (data.kind == kind) & (data.condition == condition)
            ].sort_values("frequency")
            valid = (subset.frequency > 0) & (subset.power > 0)
            ax.semilogy(
                subset.loc[valid, "frequency"],
                subset.loc[valid, "power"],
                color=COLORS[condition],
                label=LABELS[condition],
                lw=1.5,
            )
        ax.set(
            title=KIND_LABELS.get(kind, kind),
            xlabel="Frequency (Hz)",
            ylabel="Power spectral density (mV squared / Hz)",
            xlim=(0, fs / 2),
        )
        ax.grid(alpha=0.2)
    fig.suptitle(f"{description}\n{fs:g} Hz sampling", fontsize=12)
    _legend(fig, axes)
    fig.tight_layout(rect=(0, 0.10, 1, 0.89))
    writer.save(
        fig,
        "05_noise_psd",
        "welch_psd",
        description
        + ". Nonpositive power bins are omitted on the logarithmic power axis. Curves show actual saved power, not a PSD-matching assumption.",
        sources,
    )


def _summary(metrics, summary_path):
    if summary_path.exists():
        return pd.read_csv(summary_path), summary_path
    metric_names = [
        name
        for name in metrics
        if name.startswith(("macro_", "micro_", "auroc_", "ap_", "f1_"))
        or name in ("brier", "ece", "mean_loss")
    ]
    long = metrics.melt(
        id_vars=KEYS + ["seed"],
        value_vars=metric_names,
        var_name="metric",
        value_name="value",
    )
    result = (
        long.groupby(KEYS + ["metric"], dropna=False)["value"]
        .agg(mean="mean", std="std", n_seeds="count")
        .reset_index()
    )
    return result, None


def plot_performance(writer, metrics, summary, metrics_path, summary_path):
    noisy = summary[(summary.active == "all") & (summary.kind != "clean")]
    models, kinds = sorted(noisy.model.unique()), list(dict.fromkeys(noisy.kind))
    for metric, metric_label in (
        ("macro_auroc", "Macro AUROC"),
        ("macro_f1", "Macro F1"),
    ):
        if metric not in set(noisy.metric):
            continue
        fig, axes = _panels(models, kinds)
        for row, model in enumerate(models):
            clean = summary[
                (summary.model == model)
                & (summary.kind == "clean")
                & (summary.metric == metric)
            ]
            for col, kind in enumerate(kinds):
                ax = axes[row, col]
                group = noisy[
                    (noisy.model == model)
                    & (noisy.kind == kind)
                    & (noisy.metric == metric)
                ]
                snrs = sorted(group.snr.unique())
                for condition in CONDITIONS:
                    subset = group[group.condition == condition].sort_values("snr")
                    error = subset["std"].where(subset.n_seeds > 1).to_numpy()
                    _errorbar(
                        ax,
                        subset.snr,
                        subset["mean"],
                        error,
                        marker="o",
                        ms=4,
                        color=COLORS[condition],
                        label=LABELS[condition],
                        lw=1.5,
                    )
                if len(clean):
                    value = clean.iloc[0]
                    if np.isfinite(value["mean"]):
                        ax.axhline(
                            value["mean"],
                            color="#333333",
                            ls="--",
                            lw=1.1,
                            label="Clean reference",
                        )
                        if value.n_seeds > 1 and np.isfinite(value["std"]):
                            ax.axhspan(
                                value["mean"] - value["std"],
                                value["mean"] + value["std"],
                                color="#999999",
                                alpha=0.12,
                            )
                ax.set(
                    title=_title(model, kind),
                    xlabel="Global target SNR (dB; higher is cleaner)",
                    ylabel=metric_label,
                )
                ax.set_xticks(snrs)
                if len(snrs) == 1:
                    ax.set_xlim(snrs[0] - 2, snrs[0] + 2)
                ax.grid(alpha=0.2)
        fig.suptitle(
            f"{metric_label} under all-electrode noise | {writer.fs:g} Hz\nMean across training seeds; bars/band = 1 seed SD (none for one seed)",
            fontsize=12,
        )
        _legend(fig, axes, ncol=3)
        fig.tight_layout(rect=(0, 0.14, 1, 0.90))
        writer.save(
            fig,
            f"06_performance_{metric}",
            "performance_snr",
            f"{metric_label} for each model and noise kind across all four conditions, with numeric SNR increasing left to right. Clean is a separate horizontal reference, not an artificial 100 dB observation. Points are training-seed means; error bars and clean shading show one sample SD only when multiple finite seed values exist. This is seed variability, not a patient-level confidence interval, and three seeds remain a fragile uncertainty estimate.",
            [metrics_path] + ([summary_path] if summary_path else []),
        )


def _clean_drops(metrics, metric):
    clean = metrics[
        (metrics.kind == "clean")
        & (metrics.condition == "clean")
        & (metrics.active == "all")
    ][["model", "seed", metric]].rename(columns={metric: "clean_value"})
    noisy = metrics[metrics.kind != "clean"]
    paired = noisy.merge(
        clean, on=["model", "seed"], how="inner", validate="many_to_one"
    )
    paired["drop"] = paired.clean_value - paired[metric]
    return paired


def plot_electrodes(writer, metrics, metrics_path):
    data = _clean_drops(metrics, "macro_auroc")
    data = data[
        (data.snr == 10)
        & data.active.isin(ELECTRODES)
        & data.condition.isin(("electrode", "independent_rms"))
    ]
    if data.empty:
        writer.skip(
            "electrode_sensitivity",
            "No saved single-electrode 10 dB evaluations; no sensitivities inferred from all-electrode tests.",
        )
        return
    # Inner join enforces the same model/seed/kind/electrode in both conditions.
    paired = data[data.condition == "electrode"].merge(
        data[data.condition == "independent_rms"],
        on=["model", "seed", "kind", "snr", "active"],
        suffixes=("_physical", "_matched"),
        validate="one_to_one",
    )
    if paired.empty:
        writer.skip(
            "electrode_sensitivity",
            "No seed-paired electrode and RMS-matched independent evaluations at 10 dB.",
        )
        return
    models, kinds = sorted(paired.model.unique()), list(dict.fromkeys(paired.kind))
    electrodes = [e for e in ELECTRODES if e in set(paired.active)]
    fig, axes = _panels(models, kinds, width=max(4.5, 0.6 * len(electrodes)))
    positions = np.arange(len(electrodes))
    for row, model in enumerate(models):
        for col, kind in enumerate(kinds):
            ax = axes[row, col]
            subset = paired[(paired.model == model) & (paired.kind == kind)]
            for offset, condition, suffix in (
                (-0.12, "electrode", "physical"),
                (0.12, "independent_rms", "matched"),
            ):
                stats = [
                    _mean_sd(subset.loc[subset.active == electrode, f"drop_{suffix}"])
                    for electrode in electrodes
                ]
                _errorbar(
                    ax,
                    positions + offset,
                    [v[0] for v in stats],
                    [v[1] for v in stats],
                    ls="none",
                    marker="o",
                    color=COLORS[condition],
                    label=LABELS[condition],
                )
            ax.axhline(0, color="#777777", lw=0.8)
            ax.set_xticks(positions, electrodes, rotation=45)
            ax.set(
                title=_title(model, kind),
                xlabel="Single perturbed electrode",
                ylabel="Clean AUROC minus noisy AUROC",
            )
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"Single-electrode sensitivity at fixed 10 dB | {writer.fs:g} Hz\nPaired clean-to-noisy drops; mean and SD across matched training seeds",
        fontsize=12,
    )
    _legend(fig, axes, ncol=2)
    fig.tight_layout(rect=(0, 0.12, 1, 0.90))
    writer.save(
        fig,
        "07_electrode_sensitivity",
        "electrode_sensitivity",
        "Clean-minus-noisy macro AUROC at fixed global 10 dB for each actually evaluated electrode. Physical propagation and per-lead RMS-matched independent controls use the same seed and clean baseline; only seed-paired observations are included. Bars are one seed SD, absent for a single seed. Positive values indicate deterioration. Unsaved electrodes are not imputed.",
        [metrics_path],
    )


def _paired_seed_effects(metrics, metric):
    data = metrics[(metrics.active == "all") & (metrics.kind != "clean")]
    join = ["model", "seed", "kind", "snr", "active"]
    left = data[data.condition == "electrode"][join + [metric]]
    right = data[data.condition == "independent_rms"][join + [metric]]
    paired = left.merge(
        right, on=join, suffixes=("_physical", "_matched"), validate="one_to_one"
    )
    paired["difference"] = paired[f"{metric}_physical"] - paired[f"{metric}_matched"]
    records = []
    for key, group in paired.groupby(["model", "kind", "snr"], sort=False):
        mean, sd, n = _mean_sd(group.difference)
        records.append(
            dict(
                zip(["model", "kind", "snr"], key),
                mean_difference=mean,
                ci95_low=mean - sd,
                ci95_high=mean + sd,
                n_seeds=n,
            )
        )
    return pd.DataFrame(records)


def plot_effects(writer, metrics, metrics_path, effects_path):
    data, saved_inference = pd.DataFrame(), None
    if effects_path.exists():
        effects = pd.read_csv(effects_path)
        data = effects[
            (effects.comparison == "electrode_minus_independent_rms")
            & (effects.outcome == "macro_auroc")
            & (effects.active == "all")
        ].copy()
        if not data.empty and "inference_unit" in data:
            units = set(data.inference_unit)
            if units == {"training_seed"}:
                saved_inference = "training_seed"
            elif units == {"patient"}:
                saved_inference = "patient"
        if saved_inference is None:
            data = pd.DataFrame()
    if data.empty:
        data = _paired_seed_effects(metrics, "macro_auroc")
    if data.empty:
        writer.skip(
            "paired_effect_summary",
            "No paired electrode/RMS-matched independent macro AUROC values.",
        )
        return
    models = sorted(data.model.unique())
    kinds = list(dict.fromkeys(data.kind))
    order = [
        (kind, snr)
        for kind in kinds
        for snr in sorted(data.loc[data.kind == kind, "snr"].unique())
    ]
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(6 * len(models), max(4.8, 0.43 * len(order) + 2)),
        squeeze=False,
        sharey=True,
    )
    for ax, model in zip(axes[0], models):
        subset = data[data.model == model]
        for pos, (kind, snr) in enumerate(order):
            value = subset[(subset.kind == kind) & (subset.snr == snr)]
            if value.empty:
                continue
            value = value.iloc[0]
            mean, low, high = value.mean_difference, value.ci95_low, value.ci95_high
            if not np.isfinite(mean):
                continue
            # Draw endpoints directly: a bootstrap percentile interval need not contain the estimate.
            if np.isfinite(low) and np.isfinite(high):
                ax.hlines(pos, low, high, color=COLORS["electrode"], lw=1.7)
                ax.plot([low, high], [pos, pos], "|", color=COLORS["electrode"], ms=7)
            ax.plot(mean, pos, "o", color=COLORS["electrode"], ms=5)
        ax.axvline(0, ls="--", color="#555555", lw=0.9)
        ax.set(
            title=model, xlabel="Macro AUROC: electrode minus RMS-matched independent"
        )
        ax.set_yticks(
            range(len(order)), [f"{KIND_LABELS.get(k, k)} | {s:g} dB" for k, s in order]
        )
        ax.grid(axis="x", alpha=0.2)
    axes[0, 0].invert_yaxis()
    if saved_inference == "training_seed":
        uncertainty = (
            "95% paired training-seed Student-t CI; few seeds give fragile intervals"
        )
        interval_caption = "Intervals are the statistics module's paired training-seed Student-t 95% intervals, not patient-bootstrap intervals. Three training seeds give fragile inferential support; one-seed intervals are omitted. "
    elif saved_inference == "patient":
        uncertainty = "95% patient-cluster bootstrap CI"
        interval_caption = "Intervals are the statistics module's 95% patient-cluster bootstrap intervals; repeated ECGs within patients are not independent resampling units. "
    else:
        uncertainty = "1 SD of paired seed differences; no interval for one seed"
        interval_caption = "No explicitly labeled inferential statistics were available: intervals are descriptive SDs of same-seed differences, not confidence intervals, and omitted for one seed. "
    fig.suptitle(
        f"Paired robustness contrast | {writer.fs:g} Hz\n{uncertainty}", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    writer.save(
        fig,
        "08_paired_effect_summary",
        "paired_effect_summary",
        "Macro AUROC differences between electrode-propagated and RMS-matched independent noise, separated by model, noise kind and SNR. Positive means higher AUROC under electrode propagation. "
        + interval_caption
        + "Contrasts are experimental, not clinical causal effects.",
        [metrics_path] + ([effects_path] if saved_inference else []),
    )


def plot_class_drops(writer, metrics, metrics_path):
    classes = [name for name in CLASSES if f"auroc_{name}" in metrics]
    if not classes:
        writer.skip("per_class_drops", "No per-class AUROC columns in saved metrics.")
        return
    rows = []
    for name in classes:
        paired = _clean_drops(metrics, f"auroc_{name}")
        paired = paired[(paired.active == "all") & (paired.snr == 10)]
        rows.append(
            paired[["model", "kind", "condition", "seed", "drop"]].assign(target=name)
        )
    data = pd.concat(rows, ignore_index=True)
    if data.empty:
        writer.skip(
            "per_class_drops", "No saved all-electrode 10 dB per-class evaluations."
        )
        return
    models, kinds = sorted(data.model.unique()), list(dict.fromkeys(data.kind))
    fig, axes = _panels(models, kinds, width=4.6)
    x = np.arange(len(classes))
    for row, model in enumerate(models):
        for col, kind in enumerate(kinds):
            ax = axes[row, col]
            for index, condition in enumerate(CONDITIONS):
                subset = data[
                    (data.model == model)
                    & (data.kind == kind)
                    & (data.condition == condition)
                ]
                stats = [
                    _mean_sd(subset.loc[subset.target == name, "drop"])
                    for name in classes
                ]
                _errorbar(
                    ax,
                    x + (index - 1.5) * 0.16,
                    [v[0] for v in stats],
                    [v[1] for v in stats],
                    ls="none",
                    marker="o",
                    ms=4,
                    color=COLORS[condition],
                    label=LABELS[condition],
                )
            ax.axhline(0, color="#777777", lw=0.8)
            ax.set_xticks(x, classes, rotation=30)
            ax.set(
                title=_title(model, kind),
                xlabel="Diagnostic superclass",
                ylabel="Clean AUROC minus noisy AUROC",
            )
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"Per-class clean-to-noisy drops at 10 dB | {writer.fs:g} Hz\nSeed-paired mean and SD; no uncertainty bar for one seed",
        fontsize=12,
    )
    _legend(fig, axes)
    fig.tight_layout(rect=(0, 0.12, 1, 0.90))
    writer.save(
        fig,
        "09_per_class_drops",
        "per_class_drops",
        "Per-superclass AUROC decrease from each model/seed's clean result at all-electrode 10 dB noise. Means and sample SDs are computed from within-seed drops, not by subtracting unrelated summary intervals. Undefined class AUROCs are excluded without replacement. Positive values indicate deterioration.",
        [metrics_path],
    )


def plot_calibration(writer, metrics, metrics_path):
    if "prediction_path" not in metrics:
        writer.skip("calibration", "No persisted prediction paths in metrics.")
        return
    models = sorted(metrics.model.unique())
    fig, axes = plt.subplots(
        1, len(models), figsize=(5.3 * len(models), 5.3), squeeze=False
    )
    sources = [metrics_path]
    plotted = 0
    selections = []
    for ax, model in zip(axes[0], models):
        seed = sorted(metrics.loc[metrics.model == model, "seed"].unique())[0]
        model_rows = metrics[
            (metrics.model == model)
            & (metrics.seed == seed)
            & (metrics.active == "all")
        ]
        kinds = model_rows.loc[model_rows.kind != "clean", "kind"].unique()
        kind = "bandpass" if "bandpass" in kinds else (kinds[0] if len(kinds) else None)
        selected = model_rows[
            (model_rows.kind == "clean")
            | ((model_rows.kind == kind) & (model_rows.snr == 10))
        ]
        reference_ids, reference_y = None, None
        for condition in ("clean",) + CONDITIONS:
            match = selected[selected.condition == condition]
            if match.empty:
                continue
            path = writer.root / str(match.iloc[0].prediction_path)
            with np.load(path, allow_pickle=False) as prediction:
                p = np.asarray(prediction["p"], dtype=float)
                y = np.asarray(prediction["y"], dtype=float)
                ids = prediction["ids"]
                ordering = np.argsort(ids)
                p, y, ids = p[ordering], y[ordering], ids[ordering]
            if reference_ids is None:
                reference_ids, reference_y = ids, y
            elif not np.array_equal(reference_ids, ids) or not np.array_equal(
                reference_y, y
            ):
                raise ValueError(f"Calibration predictions are not paired: {path}")
            flat_p, flat_y = p.ravel(), y.ravel()
            bins = np.minimum((flat_p * 10).astype(int), 9)
            centers, fractions, counts = [], [], []
            for index in range(10):
                mask = bins == index
                if mask.any():
                    centers.append(float(flat_p[mask].mean()))
                    fractions.append(float(flat_y[mask].mean()))
                    counts.append(int(mask.sum()))
            color = "#333333" if condition == "clean" else COLORS[condition]
            label = "Clean" if condition == "clean" else LABELS[condition]
            ax.plot(centers, fractions, color=color, lw=1.2, label=label)
            ax.scatter(
                centers,
                fractions,
                s=12 + 60 * np.asarray(counts) / max(counts),
                color=color,
                zorder=3,
            )
            sources.append(path)
            plotted += 1
        selections.append(f"{model}: seed {seed}, {kind}")
        ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=0.8)
        ax.set(
            title=f"{model} | seed {seed}\n{KIND_LABELS.get(kind, kind)} at 10 dB",
            xlabel="Mean predicted probability",
            ylabel="Observed positive fraction",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.grid(alpha=0.2)
    if not plotted:
        plt.close(fig)
        writer.skip("calibration", "No eligible clean or 10 dB saved predictions.")
        return
    fig.suptitle(
        f"Descriptive pooled-label reliability | {writer.fs:g} Hz\nOne actual training seed per model; no independence-based confidence bands",
        fontsize=12,
    )
    _legend(fig, axes, ncol=3)
    fig.tight_layout(rect=(0, 0.17, 1, 0.86))
    writer.save(
        fig,
        "10_calibration",
        "calibration",
        "Descriptive reliability of saved probabilities across the five binary labels, pooled within one actual seed per model. Ten equal-width probability bins; empty bins omitted. Marker areas scale with within-curve bin counts. Labels and ECGs are dependent, so no binomial error bars are presented. This is not classwise calibration or a multi-seed uncertainty estimate. "
        + "; ".join(selections)
        + ".",
        sources,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--pdf", action="store_true", help="Also save vector PDF figures"
    )
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    results = Path(config.get("results_dir", "results"))
    if not results.is_absolute():
        results = root / results
    run = config["run_name"]
    metrics_dir, tables_dir = results / "metrics" / run, results / "tables" / run
    metrics_path = metrics_dir / "metrics.csv"
    example_path = metrics_dir / "noise_examples.npz"
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        raise ValueError(f"No saved evaluation rows in {metrics_path}")
    if metrics.duplicated(KEYS + ["seed"]).any():
        raise ValueError("Duplicate evaluation keys; plotting would double-count seeds")
    settings = config.get("plotting", {})
    writer = FigureWriter(
        root,
        results / "figures" / run,
        run,
        float(config.get("sampling_rate", 100)),
        settings.get("dpi", 180),
        args.pdf or settings.get("pdf", False),
    )
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
    summary, summary_path = _summary(metrics, tables_dir / "metrics_summary.csv")
    noises, fs = plot_examples(writer, example_path)
    plot_psd(writer, tables_dir / "noise_psd.csv", example_path, noises, fs)
    plot_performance(writer, metrics, summary, metrics_path, summary_path)
    plot_electrodes(writer, metrics, metrics_path)
    plot_effects(writer, metrics, metrics_path, tables_dir / "paired_effects.csv")
    plot_class_drops(writer, metrics, metrics_path)
    plot_calibration(writer, metrics, metrics_path)
    manifest = writer.finish(args.config)
    print(f"Saved {len(writer.entries)} scientific figures; manifest: {manifest}")


if __name__ == "__main__":
    main()
