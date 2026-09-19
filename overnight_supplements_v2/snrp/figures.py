"""Scientific static figures; fixed-condition distributions, not causal adjustment."""
from __future__ import annotations
from overnight_supplements_v2.shared.common import write_json, file_info
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .run import DEST, CONDITIONS, CLASSES

# Same scientific palette as signed_control_mechanism/plot.py and phase2 COLORS.
COLORS = dict(zip(CONDITIONS, ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7A5195"]))
LABELS = {"E": "Standard E", "I": "Independent I", **{f"S_{i:02d}": f"Signed {i}" for i in range(5)}}
MARKERS = dict(zip(CONDITIONS, ["o", "s", "^", "D", "v", "P", "X"]))


def finish(fig, stem, footer, files):
    fig.text(.5, .018, footer, ha="center", va="bottom", fontsize=8, linespacing=1.5)
    for suffix in ("pdf", "png", "svg"):
        path = DEST / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        files.append(path.relative_to(DEST.parent.parent).as_posix())
    plt.close(fig)


def legend(fig, y=.93):
    fig.legend(handles=[Line2D([], [], color=COLORS[c], marker=MARKERS[c], markersize=5, label=LABELS[c], linewidth=1.4) for c in CONDITIONS],
               loc="upper center", bbox_to_anchor=(.5, y), ncol=7, frameon=False, fontsize=9)


def finite(values):
    array = np.asarray(values, float)
    return array[np.isfinite(array)]


def nominal_geometry(ax, frame):
    offsets = np.linspace(-1.05, 1.05, 7)
    for position, condition in enumerate(CONDITIONS):
        for snr in (0, 5, 10):
            values = finite(frame[(frame.condition == condition) & (frame.snr_db == snr)].snr_p_db)
            if not len(values):
                continue
            q = np.quantile(values, [.05, .25, .5, .75, .95])
            x = snr+offsets[position]
            ax.vlines(x, q[0], q[4], color=COLORS[condition], linewidth=1, alpha=.7)
            ax.vlines(x, q[1], q[3], color=COLORS[condition], linewidth=4)
            ax.plot(x, q[2], marker=MARKERS[condition], color=COLORS[condition], markersize=4)
    ax.set(xticks=[0, 5, 10], xlim=(-1.7, 11.7), xlabel="Nominal total SNR (dB)", ylabel="Absolute projected SNR$_P$ (dB)")
    ax.grid(axis="y", alpha=.18)
    for snr in (0, 5, 10):
        ax.axvline(snr, color="#888888", linestyle=":", linewidth=.5, alpha=.4)


def make_figures(frame, cohort, overlay_complete):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 10, "axes.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    files = []
    n_nonfinite = int((~np.isfinite(frame.snr_p_db)).sum())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.9), sharey=True, sharex=True)
    fig.suptitle("Geometry diagnostic: absolute projected SNR", fontsize=15, fontweight="bold", y=.99)
    for ax, snr in zip(axes, (0, 5, 10)):
        for position, condition in enumerate(CONDITIONS):
            values = finite(frame[(frame.condition == condition) & (frame.snr_db == snr)].snr_p_db)
            if len(values) > 1 and np.ptp(values) > 0:
                violin = ax.violinplot([values], positions=[position], vert=False, widths=.77, showextrema=False, points=120)
                for body in violin["bodies"]:
                    body.set_facecolor(COLORS[condition]); body.set_edgecolor(COLORS[condition]); body.set_alpha(.28)
            if len(values):
                q = np.quantile(values, [.05, .25, .5, .75, .95])
                ax.hlines(position, q[0], q[4], color=COLORS[condition], linewidth=1)
                ax.hlines(position, q[1], q[3], color=COLORS[condition], linewidth=4)
                ax.plot(q[2], position, marker=MARKERS[condition], color=COLORS[condition], markersize=4)
        ax.set(title=f"Nominal total SNR: {snr} dB", yticks=range(7), yticklabels=[LABELS[c] for c in CONDITIONS], xlabel="Absolute projected SNR$_P$ (dB)")
        ax.grid(axis="x", alpha=.15)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=.10, right=.99, bottom=.20, top=.87, wspace=.15)
    finish(fig, "geometry_distribution", "Each distribution: 2,158 ECGs × six fixed noise bases; seven conditions kept separate.\nThin lines: 5th–95th percentiles; thick: IQR; marker: median (not confidence intervals). "
           + f"Nonfinite SNR values excluded from drawing only: {n_nonfinite}; all retained in tables.", files)
    if overlay_complete:
        curves = pd.read_csv(DEST / "auroc_curves.csv", keep_default_na=False)
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex="col", gridspec_kw={"height_ratios": [1, 1.05]})
        fig.suptitle("Performance and geometry aligned by nominal SNR", fontsize=15, fontweight="bold", y=.99)
        legend(fig, .96)
        for col, model in enumerate(("resnet", "tcn")):
            ax = axes[0, col]
            for condition in CONDITIONS:
                sub = curves[(curves.model == model) & (curves.condition == condition)].sort_values("snr_db")
                ax.plot(sub.snr_db, sub.macro_auroc, label=LABELS[condition], color=COLORS[condition], marker=MARKERS[condition], linewidth=1.6, markersize=5)
            ax.set(title={"resnet": "ResNet", "tcn": "TCN"}[model], ylabel="Absolute macro-AUROC (fraction)", xticks=[0, 5, 10], xlim=(-1.7, 11.7))
            ax.grid(alpha=.18)
            nominal_geometry(axes[1, col], frame)
            axes[1, col].set_title("Same input geometry (not extra observations)")
        fig.subplots_adjust(left=.08, right=.98, top=.86, bottom=.18, hspace=.27, wspace=.21)
        finish(fig, "nominal_snr_overlay", "Top: original-probability AUROC, equal mean across six fixed noises within each of three checkpoints, then across checkpoints.\nBottom: geometry median, IQR and 5th–95th percentiles; condition offsets are display-only around nominal 0/5/10 dB.\nThis is a parallel display—not equal exposure, causal mediation, or new patient uncertainty. E/I and modes retain original weights.", files)
        fig, axes = plt.subplots(5, 3, figsize=(15, 18), sharex="col", sharey=True)
        fig.suptitle("True-class positive and negative exposure distributions", fontsize=15, fontweight="bold", y=.995)
        legend(fig, .977)
        ids_to_row = {int(rid): i for i, rid in enumerate(cohort["ids"])}
        row_indices = np.asarray([ids_to_row[int(rid)] for rid in frame.record_id])
        for class_index, name in enumerate(CLASSES):
            labels = cohort["y"][row_indices, class_index]
            for col, snr in enumerate((0, 5, 10)):
                ax = axes[class_index, col]
                for condition in CONDITIONS:
                    base_mask = (frame.condition.to_numpy() == condition) & (frame.snr_db.to_numpy() == snr)
                    for label, style in ((1, "-"), (0, "--")):
                        values = np.sort(finite(frame.loc[base_mask & (labels == label), "snr_p_db"]))
                        if len(values):
                            # Exact ECDF; thinning only equal-height plotting vertices is unnecessary.
                            ax.plot(values, np.arange(1, len(values)+1)/len(values), color=COLORS[condition], linestyle=style, linewidth=1.05, alpha=.85, rasterized=False)
                positives = int(cohort["y"][:, class_index].sum())
                negatives = len(cohort["ids"])-positives
                ax.set_title(f"{name} | nominal {snr} dB\n{positives} positive / {negatives} negative ECGs", fontsize=9)
                ax.set_ylim(0, 1)
                ax.grid(alpha=.14)
                if col == 0:
                    ax.set_ylabel("Empirical cumulative fraction")
                if class_index == 4:
                    ax.set_xlabel("Absolute projected SNR$_P$ (dB)")
        fig.subplots_adjust(left=.075, right=.99, bottom=.08, top=.93, hspace=.42, wspace=.16)
        finish(fig, "class_exposure_distribution", "Solid: ground-truth positive; dashed: ground-truth negative (not predicted classifications).\nEach class retains six fixed noise exposures per ECG and seven separate conditions; no checkpoint replication, reweighting, or causal claim.\nAll original predictions were label/identity joined before this display. Finite values only on curves; null/+inf counts remain in audited summaries.", files)
    write_json(DEST / "figure_manifest.json", dict(files=[file_info(path, schema="scientific figure", source="audited geometry / original prediction overlay") for path in files],
        palette_source="signed_control_mechanism/plot.py MODE_COLORS; phase2/src/plot_phase2.py COLORS",
        visual_acceptance="pending Main visual inspection", statistical_status="Descriptive distributions and fixed-source absolute AUROC only; no new CI"))
    return files
