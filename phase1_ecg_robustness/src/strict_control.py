"""Paired circular-shift control: exact lead marginals and periodic spectra.

This is not independent lead noise: lagged shared-source dependence can remain.
Ordinary finite-window Welch estimates are measured, not claimed invariant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.signal import periodogram
import torch
import yaml

from .audit_noise import powers, record_seed
from .datasets import _data_identity, load_data
from .evaluate import classification_metrics, predict, sha256
from .lead_matrix import LEADS, matrix_provenance, validate_lead_matrix
from .models import build_model
from .noise_generators import _integrated_band, make_noise_triplet, noise_diagnostics
from .train import seed_everything


def apply_circular_shifts(noise, shifts):
    """Permute each lead without interpolation, normalization or dtype changes."""
    noise = np.asarray(noise)
    shifts = np.asarray(shifts)
    if noise.ndim != 2 or noise.shape[1] < 2 or not np.isfinite(noise).all():
        raise ValueError("Noise must be a finite lead-by-time array with T >= 2")
    if shifts.shape != (noise.shape[0],) or not np.issubdtype(shifts.dtype, np.integer):
        raise ValueError("One integer circular offset is required per lead")
    if np.any((shifts < 0) | (shifts >= noise.shape[1])):
        raise ValueError("Circular offsets must lie in [0, T)")
    result = np.empty_like(noise)
    for lead, offset in enumerate(shifts):
        offset = int(offset)
        if offset:
            result[lead, :offset] = noise[lead, -offset:]
            result[lead, offset:] = noise[lead, :-offset]
        else:
            result[lead] = noise[lead]
    return result


def marginal_shift(noise, seed):
    """Draw independent uniform offsets; zero/equal offsets are not rejected."""
    noise = np.asarray(noise)
    if noise.ndim != 2 or noise.shape[1] < 2:
        raise ValueError("Noise must be a lead-by-time array with T >= 2")
    offsets = np.random.default_rng(seed).integers(
        0, noise.shape[1], size=noise.shape[0], dtype=np.int32
    )
    return apply_circular_shifts(noise, offsets), offsets


def _relative_max(difference, reference):
    difference = np.asarray(difference)
    reference = np.asarray(reference)
    if np.any((reference == 0) & (difference != 0)):
        raise ValueError("Nonzero difference against a zero-power reference")
    ratio = np.divide(
        difference, reference, out=np.zeros_like(difference), where=reference > 0
    )
    return float(np.max(ratio))


def marginal_diagnostics(reference, shifted, signal, fs, band=(0.5, 40.0)):
    """Audit actual injected arrays; PSD and amplitude invariance are distinct."""
    a = noise_diagnostics(reference, signal, fs, band)
    b = noise_diagnostics(shifted, signal, fs, band)
    f, pa = periodogram(
        np.asarray(reference, dtype=np.float64),
        fs=fs,
        window="boxcar",
        detrend=False,
        scaling="density",
        axis=-1,
    )
    _, pb = periodogram(
        np.asarray(shifted, dtype=np.float64),
        fs=fs,
        window="boxcar",
        detrend=False,
        scaling="density",
        axis=-1,
    )
    periodic_denominator = pa.sum(axis=1)
    periodic_power = periodic_denominator * (f[1] - f[0])
    total_welch = a["psd"].sum(axis=1)
    result = {
        "empirical_marginals_exact": bool(
            np.array_equal(np.sort(reference, axis=1), np.sort(shifted, axis=1))
        ),
        "variance_max_relative_error": _relative_max(
            abs(a["per_lead_variance"] - b["per_lead_variance"]), a["per_lead_variance"]
        ),
        "rms_max_relative_error": _relative_max(
            abs(a["per_lead_rms"] - b["per_lead_rms"]), a["per_lead_rms"]
        ),
        "snr_absolute_difference_db": abs(a["actual_snr_db"] - b["actual_snr_db"]),
        "periodogram_l1_max_lead_relative_error": _relative_max(
            abs(pa - pb).sum(axis=1), periodic_denominator
        ),
        "welch_l1_max_lead_relative_error": _relative_max(
            abs(a["psd"] - b["psd"]).sum(axis=1), total_welch
        ),
        "welch_l1_all_leads_relative_error": float(
            abs(a["psd"] - b["psd"]).sum() / a["psd"].sum()
        ),
        "electrode_abs_offdiagonal_correlation": a["mean_abs_offdiagonal_correlation"],
        "shifted_abs_offdiagonal_correlation": b["mean_abs_offdiagonal_correlation"],
    }
    for name, lo, hi in (
        ("baseline", 0.0, 0.5),
        ("low", 0.5, 5.0),
        ("middle", 5.0, 15.0),
        ("high_ecg", 15.0, 40.0),
        ("above_ecg", 40.0, fs / 2),
    ):
        if lo >= hi:
            continue
        ea = _integrated_band(f, pa, lo, hi)
        eb = _integrated_band(f, pb, lo, hi)
        result[f"periodic_{name}_band_error_relative_to_total_max"] = _relative_max(
            abs(ea - eb), periodic_power
        )
        wa = _integrated_band(a["frequencies"], a["psd"], lo, hi)
        wb = _integrated_band(b["frequencies"], b["psd"], lo, hi)
        result[f"welch_{name}_band_error_relative_to_total_max"] = _relative_max(
            abs(wa - wb), a["welch_total_power"]
        )
    exact_fields = [
        "variance_max_relative_error",
        "rms_max_relative_error",
        "snr_absolute_difference_db",
        "periodogram_l1_max_lead_relative_error",
    ] + [key for key in result if key.startswith("periodic_")]
    if not result["empirical_marginals_exact"] or any(
        result[key] > 1e-10 for key in exact_fields
    ):
        raise RuntimeError(
            f"Strict marginal/periodic-spectrum control failed: {result}"
        )
    return result, a["frequencies"], a["psd"], b["psd"]


def write_figures(
    root, config_path, cfg, original, old_metrics, strict_metrics, audit, psd
):
    from .plotting import COLORS, LABELS, FigureWriter, _errorbar, _legend, plt

    output = root / cfg["results_dir"] / "figures" / cfg["run_name"] / "strict"
    writer = FigureWriter(
        root, output, cfg["run_name"], original["sampling_rate"], pdf=True
    )
    colors = {**COLORS, "marginal_shift": "#D55E00"}
    labels = {**LABELS, "marginal_shift": "Circular-shift matched marginals"}
    noise_base = int(original["noise"]["seed"])
    old = old_metrics[(old_metrics.kind == "bandpass") & (old_metrics.active == "all")]
    current = strict_metrics[strict_metrics.noise_seed == noise_base]
    values = pd.concat([old, current], ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for row, model in enumerate(original["models"]):
        for col, metric in enumerate(["macro_auroc", "macro_f1"]):
            ax = axes[row, col]
            for condition in labels:
                group = values[
                    (values.model == model) & (values.condition == condition)
                ]
                summary = group.groupby("snr")[metric].agg(["mean", "std"]).sort_index()
                _errorbar(
                    ax,
                    summary.index,
                    summary["mean"],
                    summary["std"],
                    color=colors[condition],
                    label=labels[condition],
                    marker="o",
                )
            clean = old_metrics[
                (old_metrics.model == model) & (old_metrics.condition == "clean")
            ][metric]
            ax.axhline(clean.mean(), color="#333333", linestyle="--", label="Clean")
            ax.set(
                title=f"{model} | {metric}",
                xlabel="Global target SNR (dB)",
                ylabel=metric,
            )
            ax.set_xticks(sorted(original["noise"]["snrs"]))
            ax.grid(alpha=0.2)
    fig.suptitle(
        "Strict marginal control | fixed primary Gaussian realization\nMean across three trained seeds; bars = one seed SD, not patient CI"
    )
    _legend(fig, axes, ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    writer.save(
        fig,
        "strict_control_performance",
        "strict_control_performance",
        "Original Gaussian controls and circular-shift surrogate at all SNR. Primary noise base8128; clean thresholds frozen. Mean and sample SD of three fixed trained seeds.",
        [
            root
            / original["results_dir"]
            / "metrics"
            / original["run_name"]
            / "metrics.csv",
            root
            / cfg["results_dir"]
            / "metrics"
            / cfg["run_name"]
            / "strict_metrics.csv",
        ],
    )

    selected = audit[(audit.noise_seed == noise_base) & (audit.snr == 0)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bins = np.linspace(
        0,
        max(
            selected.electrode_abs_offdiagonal_correlation.max(),
            selected.shifted_abs_offdiagonal_correlation.max(),
        )
        * 1.05,
        40,
    )
    for field, name, color in [
        ("electrode_abs_offdiagonal_correlation", "Electrode", colors["electrode"]),
        (
            "shifted_abs_offdiagonal_correlation",
            "Circular shift",
            colors["marginal_shift"],
        ),
    ]:
        axes[0].hist(
            selected[field], bins=bins, alpha=0.6, density=True, label=name, color=color
        )
    axes[0].set(
        xlabel="Mean absolute off-diagonal correlation per ECG",
        ylabel="Density",
        title="Cross-lead alignment changes",
    )
    axes[0].legend(frameon=False)
    axes[1].boxplot(
        [
            np.maximum(selected.periodogram_l1_max_lead_relative_error, 1e-18),
            np.maximum(selected.welch_l1_max_lead_relative_error, 1e-18),
        ],
        tick_labels=["Full-record periodogram", "Finite-window Welch"],
        whis=(5, 95),
        showfliers=False,
    )
    axes[1].set_yscale("log")
    axes[1].set(
        ylabel="Maximum per-lead relative PSD L1 error",
        title="Estimator distinction matters",
    )
    fig.suptitle("2158 actual test ECGs | Gaussian, 0 dB, base8128")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    writer.save(
        fig,
        "strict_control_noise_diagnostics",
        "strict_control_noise_diagnostics",
        "Record-level correlation distributions and per-lead PSD matching errors. Boxes show median/IQR, whiskers5th–95th percentiles across records, not confidence intervals. Exact zeros use1e-18 only for logarithmic display. Ordinary Welch is not shift invariant; stored data are not floored.",
        [
            root
            / cfg["results_dir"]
            / "tables"
            / cfg["run_name"]
            / "strict_marginal_audit.csv"
        ],
    )

    selected_psd = psd[(psd.noise_seed == noise_base) & (psd.snr == 0)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), squeeze=False)
    for ax, lead in zip(axes.ravel(), ["I", "II", "aVR", "V1", "V3", "V6"]):
        lead_data = selected_psd[selected_psd.lead == lead].sort_values("frequency_hz")
        ax.semilogy(
            lead_data.frequency_hz,
            lead_data.electrode_psd,
            color=colors["electrode"],
            label="Electrode",
        )
        ax.semilogy(
            lead_data.frequency_hz,
            lead_data.marginal_shift_psd,
            color=colors["marginal_shift"],
            linestyle="--",
            label="Circular shift",
        )
        ax.set(title=lead, xlabel="Frequency (Hz)", ylabel="PSD (mV²/Hz)")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Population-mean Welch PSD | 2158 test ECGs, Gaussian 0 dB\nPer-record periodic spectra are matched; finite-window Welch estimates may differ"
    )
    _legend(fig, axes, ncol=2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    writer.save(
        fig,
        "strict_control_welch_psd",
        "strict_control_welch_psd",
        "Six named leads' mean Welch spectra across the actual test cohort. This population average does not claim each record's Welch estimates coincide. nperseg256, overlap128, Hann window, per-window constant detrending.",
        [
            root
            / cfg["results_dir"]
            / "tables"
            / cfg["run_name"]
            / "strict_welch_psd.csv"
        ],
    )
    return writer.finish(config_path)


def run(config_path):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    original = yaml.safe_load((root / cfg["base_config"]).read_text(encoding="utf-8"))
    validate_lead_matrix()
    matrix = matrix_provenance()
    if matrix["sha256"] != cfg["matrix_sha256"]:
        raise ValueError(
            "Configured lead matrix fingerprint differs from current source"
        )
    spec = cfg["strict_control"]
    if spec["kind"] != "bandpass" or spec["condition"] != "marginal_shift":
        raise ValueError(
            "This strict control is defined only for periodic Gaussian noise"
        )
    noise_seeds = list(map(int, spec["noise_seeds"]))
    if len(set(noise_seeds)) != len(noise_seeds) or min(noise_seeds) < 0:
        raise ValueError("Noise bases must be unique nonnegative integers")
    if set(spec["snrs"]) != set(original["noise"]["snrs"]):
        raise ValueError("Strict control must cover all original SNR conditions")
    out = root / cfg["results_dir"] / "metrics" / cfg["run_name"]
    tables = root / cfg["results_dir"] / "tables" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    old_results = root / original["results_dir"]
    primary_dir = old_results / "metrics" / original["run_name"]
    old_protocol = json.loads(
        (primary_dir / "evaluation_protocol.json").read_text(encoding="utf-8")
    )
    gate = json.loads(
        (old_results / "tables" / original["run_name"] / "noise_gate.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        old_protocol.get("status") != "completed"
        or old_protocol.get("config") != original
    ):
        raise ValueError(
            "Original full evaluation is incomplete or has a different config"
        )
    if (
        not gate["passed"]
        or gate["config"] != original
        or not gate["gates"]["bandpass"]["strict_structure_control"]
    ):
        raise ValueError("Original Gaussian controls have not passed")
    original_metrics = pd.read_csv(primary_dir / "metrics.csv")
    replay_metrics_path = (
        old_results
        / "metrics"
        / (original["run_name"] + "_noise_repeats")
        / "metrics.csv"
    )
    replay_metrics = pd.read_csv(replay_metrics_path)
    data_dir = root / original["data_dir"]
    X, Y, meta = load_data(data_dir)
    identity = _data_identity(data_dir, meta, Y)
    torch.set_num_threads(4)
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states, checkpoint_hashes, clean_errors = [], {}, []
    indices = None
    for name in original["models"]:
        for seed in original["seeds"]:
            checkpoint = (
                old_results
                / "checkpoints"
                / original["run_name"]
                / name
                / f"seed_{seed}.pt"
            )
            c = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (
                c["config"] != original
                or not c.get("completed")
                or c.get("training_epochs_completed") != original["train"]["epochs"]
            ):
                raise ValueError(f"Unfinished or incompatible checkpoint: {checkpoint}")
            if (
                c.get("data_identity") != identity
                or c.get("model_name") != name
                or c.get("seed") != seed
            ):
                raise ValueError(f"Checkpoint identity mismatch: {checkpoint}")
            idx = np.asarray(c["test_indices"], dtype=np.int64)
            if indices is not None and not np.array_equal(indices, idx):
                raise ValueError("Checkpoints do not share ordered test indices")
            if indices is None:
                indices = idx
                x, y = np.asarray(X[idx], dtype=np.float32), np.asarray(
                    Y[idx], dtype=np.float32
                )
                ids = meta.iloc[idx].ecg_id.to_numpy(dtype=np.int64)
                patients = meta.iloc[idx].patient_id.to_numpy()
                if not np.all(meta.iloc[idx].strat_fold == 10):
                    raise ValueError("Strict evaluation is outside official test fold")
            model = build_model(c["model_name"], **c["model_kwargs"]).to(device).eval()
            model.load_state_dict(c["model_state"])
            clean_rows = original_metrics[
                (original_metrics.model == name)
                & (original_metrics.seed == seed)
                & (original_metrics.condition == "clean")
            ]
            if len(clean_rows) != 1:
                raise ValueError("Missing or duplicate clean reference")
            with np.load(
                root / clean_rows.iloc[0].prediction_path, allow_pickle=False
            ) as saved:
                for key, value in {
                    "ids": ids,
                    "patient_ids": patients,
                    "indices": idx,
                    "y": y,
                    "thresholds": np.asarray(c["thresholds"]),
                }.items():
                    if not np.array_equal(saved[key], value):
                        raise ValueError(f"Clean reference differs in {key}")
                clean_p = saved["p"].copy()
            replay_p = predict(
                model, x, c["scale_mv"], original["train"]["batch_size"], device
            )
            error = float(np.max(abs(replay_p - clean_p)))
            if error > cfg["audit"]["probability_tolerance"]:
                raise ValueError(
                    "Clean checkpoint replay differs from original evaluation"
                )
            clean_errors.append(
                {"model": name, "seed": seed, "max_probability_error": error}
            )
            states.append(
                (
                    name,
                    seed,
                    model,
                    float(c["scale_mv"]),
                    np.asarray(c["thresholds"]),
                    clean_p,
                )
            )
            checkpoint_hashes[checkpoint.as_posix()] = sha256(checkpoint)
    protocol = {
        "status": "running",
        "config": cfg,
        "source_config": original,
        "matrix": matrix,
        "checkpoint_sha256": checkpoint_hashes,
        "clean_replay_errors": clean_errors,
        "n_records": len(ids),
        "n_patients": len(np.unique(patients)),
        "data_identity": identity,
        "torch_execution": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "shift_rule": "SeedSequence([shift_seed,noise_base,ecg_id])->uint32; independent uniform PCG64 integer offsets in [0,T), including equal/zero offsets; shared across models/training seeds/SNR",
        "interpretation": "Exact empirical lead marginals and periodic spectrum; cross-lead phase/lag relationships change. Not independent noise, not a claim that only zero-lag covariance changes. Ordinary Welch estimates need not match.",
        "band_error_denominator": "Full per-lead spectral power, avoiding unstable division by near-zero out-of-band power; band integrals use the existing endpoint-interpolated trapezoid rule.",
        "source_code_sha256": {
            name: sha256(root / "src" / name)
            for name in ["strict_control.py", "noise_generators.py", "lead_matrix.py"]
        },
    }
    protocol_path = out / "strict_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows, audit_rows, psd_rows = [], [], []
    start = time.time()
    for noise_base in noise_seeds:
        mapped0, shifted0 = np.empty_like(x), np.empty_like(x)
        shifts = np.empty((len(x), x.shape[1]), dtype=np.int32)
        replay_seeds = np.asarray(
            [record_seed(noise_base, int(i), "bandpass") for i in ids], dtype=np.uint32
        )
        for i, signal in enumerate(x):
            original_noise = make_noise_triplet(
                signal,
                original["sampling_rate"],
                0,
                int(replay_seeds[i]),
                band=tuple(original["noise"]["band"]),
            )
            mapped0[i] = original_noise["electrode"]
            shift_seed = np.random.SeedSequence(
                [int(spec["shift_seed"]), noise_base, int(ids[i])]
            ).generate_state(1)[0]
            shifted0[i], shifts[i] = marginal_shift(mapped0[i], int(shift_seed))
        for snr in spec["snrs"]:
            multiplier = np.float32(10 ** (-snr / 20))
            mapped, shifted = mapped0 * multiplier, shifted0 * multiplier
            actual, rms, lead_snr = [], [], []
            original_rms = []
            aggregate_a = aggregate_b = None
            for i, signal in enumerate(x):
                diagnostics, frequencies, pa, pb = marginal_diagnostics(
                    mapped[i],
                    shifted[i],
                    signal,
                    original["sampling_rate"],
                    tuple(original["noise"]["band"]),
                )
                audit_rows.append(
                    {
                        "noise_seed": noise_base,
                        "snr": snr,
                        "ecg_id": int(ids[i]),
                        **diagnostics,
                    }
                )
                if aggregate_a is None:
                    aggregate_a, aggregate_b = np.zeros_like(pa), np.zeros_like(pb)
                aggregate_a += pa
                aggregate_b += pb
                measured = powers(signal, shifted[i])
                actual.append(measured[0])
                rms.append(measured[1])
                lead_snr.append(measured[2])
                original_rms.append(powers(signal, mapped[i])[1])
            actual, rms, lead_snr = (
                np.asarray(actual),
                np.asarray(rms),
                np.asarray(lead_snr),
            )
            if np.max(abs(actual - snr)) > 1e-4:
                raise RuntimeError("Strict-control actual SNR missed target")
            if noise_base == int(original["noise"]["seed"]):
                ref_rows = original_metrics[
                    (original_metrics.model == original["models"][0])
                    & (original_metrics.seed == original["seeds"][0])
                    & (original_metrics.kind == "bandpass")
                    & (original_metrics.active == "all")
                    & (original_metrics.snr == snr)
                    & (original_metrics.condition == "electrode")
                ]
            else:
                ref_rows = replay_metrics[
                    (replay_metrics.model == original["models"][0])
                    & (replay_metrics.seed == original["seeds"][0])
                    & (replay_metrics.noise_seed == noise_base)
                    & (replay_metrics.snr == snr)
                    & (replay_metrics.condition == "electrode")
                ]
            if len(ref_rows) != 1:
                raise ValueError("Missing paired original electrode realization")
            with np.load(
                root / ref_rows.iloc[0].prediction_path, allow_pickle=False
            ) as ref:
                for key, value in {
                    "ids": ids,
                    "y": y,
                    "patient_ids": patients,
                    "indices": indices,
                    "replay_seed": replay_seeds,
                }.items():
                    if not np.array_equal(ref[key], value):
                        raise ValueError(
                            f"Original electrode reference differs in {key}"
                        )
                if not np.array_equal(ref["noise_rms"], np.asarray(original_rms)):
                    raise ValueError(
                        "Rebuilt electrode noise differs from the original realization"
                    )
            for lead, name in enumerate(LEADS):
                for j, frequency in enumerate(frequencies):
                    psd_rows.append(
                        {
                            "noise_seed": noise_base,
                            "snr": snr,
                            "lead": name,
                            "frequency_hz": float(frequency),
                            "electrode_psd": float(aggregate_a[lead, j] / len(x)),
                            "marginal_shift_psd": float(aggregate_b[lead, j] / len(x)),
                        }
                    )
            noisy = x + shifted
            for name, seed, model, scale, thresholds, clean_p in states:
                p = predict(
                    model, noisy, scale, original["train"]["batch_size"], device
                )
                values, loss = classification_metrics(y, p, thresholds, clean_p)
                sub = out / "strict" / name / f"seed_{seed}"
                sub.mkdir(parents=True, exist_ok=True)
                path = sub / f"noise_{noise_base}_snr_{snr}_marginal_shift.npz"
                np.savez_compressed(
                    path,
                    p=p,
                    y=y,
                    ids=ids,
                    patient_ids=patients,
                    loss=loss,
                    thresholds=thresholds,
                    indices=indices,
                    actual_snr=actual,
                    noise_rms=rms,
                    lead_snr=lead_snr,
                    replay_seed=replay_seeds,
                    shifts=shifts,
                    matrix_sha256=np.asarray(matrix["sha256"]),
                )
                rows.append(
                    {
                        "model": name,
                        "seed": seed,
                        "noise_seed": noise_base,
                        "kind": "bandpass",
                        "snr": snr,
                        "condition": "marginal_shift",
                        "active": "all",
                        "prediction_path": path.relative_to(root).as_posix(),
                        **values,
                    }
                )
            pd.DataFrame(rows).to_csv(out / "strict_metrics.csv", index=False)
            pd.DataFrame(audit_rows).to_csv(
                tables / "strict_marginal_audit.csv", index=False
            )
            print(
                f"STRICT {noise_base=} {snr=} completed; {len(rows)} prediction groups",
                flush=True,
            )
    audit = pd.DataFrame(audit_rows)
    numeric = [c for c in audit.columns if c not in ["noise_seed", "snr", "ecg_id"]]
    summary = audit.groupby(["noise_seed", "snr"])[numeric].agg(
        ["mean", "std", "min", "max"]
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(tables / "strict_marginal_summary.csv", index=False)
    pd.DataFrame(psd_rows).to_csv(tables / "strict_welch_psd.csv", index=False)
    expected = len(states) * len(noise_seeds) * len(spec["snrs"])
    if (
        len(rows) != expected
        or not np.isfinite(pd.DataFrame(rows).select_dtypes("number")).all().all()
    ):
        raise RuntimeError("Strict-control evaluation is incomplete or nonfinite")
    protocol.update(
        status="completed",
        n_evaluations=len(rows),
        elapsed_seconds=time.time() - start,
        n_audit_rows=len(audit),
        all_empirical_marginals_exact=bool(audit.empirical_marginals_exact.all()),
    )
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    figure_manifest = write_figures(
        root,
        config_path,
        cfg,
        original,
        original_metrics,
        pd.DataFrame(rows),
        audit,
        pd.DataFrame(psd_rows),
    )
    protocol["figure_manifest"] = figure_manifest.relative_to(root).as_posix()
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(
        f"STRICT_CONTROL_COMPLETED {len(rows)} groups, {time.time()-start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_supplement.yaml")
    run(parser.parse_args().config)
