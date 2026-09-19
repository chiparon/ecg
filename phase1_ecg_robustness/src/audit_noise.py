"""Pre-training physical/statistical gates, measured on fixed validation ECGs."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.signal import welch
from .datasets import load_data, select_splits
from .lead_matrix import get_lead_matrix, validate_lead_matrix, LEADS, ELECTRODES
from .noise_generators import make_noise_triplet, nstdb_source_info
from .covariance_matching import covariance_factor

CONDITIONS = ("independent", "independent_rms", "electrode", "covariance")


def covariance(x):
    z = np.asarray(x, dtype=np.float64)
    z = z - z.mean(axis=-1, keepdims=True)
    return z @ z.T / z.shape[-1]


def correlation(cov):
    d = np.sqrt(np.maximum(np.diag(cov), 0))
    denom = d[:, None] * d[None, :]
    return np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-20)


def powers(x, n):
    px = np.mean(np.asarray(x, dtype=np.float64) ** 2, axis=-1)
    pn = np.mean(np.asarray(n, dtype=np.float64) ** 2, axis=-1)
    total_snr = 10 * np.log10(px.mean() / pn.mean())
    # Untouched leads have +inf SNR. Save these honestly in NPZ, not invalid JSON.
    lead_snr = np.full(12, np.inf)
    np.log10(
        np.divide(px, pn, out=np.ones(12), where=pn > 0), out=lead_snr, where=pn > 0
    )
    lead_snr[pn > 0] *= 10
    return float(total_snr), np.sqrt(pn), lead_snr


def record_seed(base, ecg_id, kind):
    # Explicit stable integer codes, independent of Python hash randomization/model seed.
    code = {"bandpass": 1, "bw": 2, "ma": 3, "em": 4}[kind]
    return int(
        np.random.SeedSequence([int(base), int(ecg_id), code]).generate_state(1)[0]
    )


def run(config):
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    out = Path(cfg["results_dir"])
    tables = out / "tables" / cfg["run_name"]
    metrics = out / "metrics" / cfg["run_name"]
    tables.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    X, _, meta = load_data(cfg["data_dir"])
    splits = select_splits(meta, cfg["train"])
    indices = splits["val"][: cfg["noise"].get("audit_records", 128)]
    if len(indices) < 16:
        raise ValueError("Noise audit needs at least 16 distinct validation records")
    spec = cfg["noise"]
    fs = cfg["sampling_rate"]
    summaries, leads, psds, bands, covrows = [], [], [], [], []
    factors, sources = [], {}
    examples = {"x": np.asarray(X[indices[0]]), "fs": fs, "A": get_lead_matrix()}
    gates = {}
    matrix = validate_lead_matrix()
    (tables / "matrix_validation.json").write_text(
        json.dumps(matrix, indent=2, default=lambda v: np.asarray(v).tolist()),
        encoding="utf-8",
    )
    for kind in spec["kinds"]:
        sources[kind] = (
            nstdb_source_info(spec["nstdb_dir"], kind, fs)
            if kind != "bandpass"
            else {
                "kind": "FFT-filtered Gaussian",
                "band_hz": spec["band"],
                "fs": fs,
                "boundary": "finite-record periodic FFT",
                "normalization": "global SNR after mapping; extra independent RMS control",
            }
        )
        stat = {c: [] for c in CONDITIONS}
        spectral = {c: [] for c in CONDITIONS}
        total_cov = {c: [] for c in CONDITIONS}
        for k, idx in enumerate(indices):
            x = np.asarray(X[idx], dtype=np.float64)
            noise = make_noise_triplet(
                x,
                fs,
                10,
                record_seed(spec["seed"], meta.iloc[idx].ecg_id, kind),
                kind=kind,
                band=tuple(spec["band"]),
                nstdb=spec.get("nstdb_dir"),
            )
            target = covariance(noise["electrode"])
            _, factor_info = covariance_factor(target)
            factors.append(
                dict(
                    kind=kind,
                    ecg_id=int(meta.iloc[idx].ecg_id),
                    rank=factor_info["rank"],
                    jitter=factor_info["jitter"],
                    min_eigenvalue=float(factor_info["eigenvalues"].min()),
                    eigenvalue_tolerance=factor_info["eigenvalue_tolerance"],
                    factor_residual=factor_info["factor_residual_frobenius"],
                )
            )
            target_rms = np.sqrt(np.mean(noise["electrode"] ** 2, axis=-1))
            for condition, n in noise.items():
                if condition not in CONDITIONS:
                    continue
                snr, rms, per_snr = powers(x, n)
                cov = covariance(n)
                corr = correlation(cov)
                off = np.mean(np.abs(corr[~np.eye(12, dtype=bool)]))
                err = np.linalg.norm(cov - target) / np.linalg.norm(target)
                rms_err = np.max(
                    np.abs(rms - target_rms) / np.maximum(target_rms, 1e-15)
                )
                f, psd = welch(n, fs=fs, nperseg=min(256, x.shape[-1]), axis=-1)
                spectral[condition].append(psd.mean(axis=0))
                total_cov[condition].append(cov)
                stat[condition].append(
                    (
                        snr,
                        off,
                        err,
                        rms_err,
                        rms,
                        per_snr,
                        np.mean(np.abs(corr - correlation(target))),
                    )
                )
                if k == 0 and kind == "bandpass":
                    examples[condition] = n
                    examples["covariance_" + condition] = cov
                    examples["correlation_" + condition] = corr
            if k % 32 == 0:
                print(f"audit {kind} {k + 1}/{len(indices)}", flush=True)
        averaged = {c: np.mean(spectral[c], axis=0) for c in CONDITIONS}
        psd_error = {
            c: float(
                np.sum(np.abs(averaged[c] - averaged["electrode"]))
                / np.sum(averaged["electrode"])
            )
            for c in CONDITIONS
        }
        for c in CONDITIONS:
            for freq, power in zip(f, averaged[c]):
                psds.append(dict(kind=kind, condition=c, frequency=freq, power=power))
            for lo, hi in [(0, 0.5), (0.5, 5), (5, 15), (15, 40), (40, fs / 2)]:
                mask = (f >= lo) & (f < hi)
                bands.append(
                    dict(
                        kind=kind,
                        condition=c,
                        low_hz=lo,
                        high_hz=hi,
                        mean_psd=float(averaged[c][mask].mean()) if mask.any() else 0.0,
                        integrated_power=float(
                            np.sum(averaged[c][mask]) * (f[1] - f[0])
                        ),
                    )
                )
            for i in range(12):
                for j in range(12):
                    covrows.append(
                        dict(
                            kind=kind,
                            condition=c,
                            lead_i=LEADS[i],
                            lead_j=LEADS[j],
                            covariance=float(np.mean(total_cov[c], axis=0)[i, j]),
                        )
                    )
            for snr in spec["snrs"]:
                scale = 10 ** ((10 - snr) / 20)
                a = stat[c]
                rms = np.array([v[4] for v in a]) * scale
                ls = np.array([v[5] for v in a]) + snr - 10
                summaries.append(
                    dict(
                        kind=kind,
                        snr=snr,
                        condition=c,
                        active="all",
                        n_records=len(indices),
                        actual_snr_mean=np.mean([v[0] for v in a]) + snr - 10,
                        actual_snr_max_error=max(abs(v[0] - 10) for v in a),
                        offdiag_abs_corr_mean=np.mean([v[1] for v in a]),
                        covariance_relative_error_mean=np.mean([v[2] for v in a]),
                        correlation_mae_mean=np.mean([v[6] for v in a]),
                        rms_relative_error_max=max(v[3] for v in a),
                        psd_band_relative_error=psd_error[c],
                    )
                )
                for l, name in enumerate(LEADS):
                    leads.append(
                        dict(
                            kind=kind,
                            snr=snr,
                            condition=c,
                            active="all",
                            lead=name,
                            rms_mean=rms[:, l].mean(),
                            variance_mean=(rms[:, l] ** 2).mean(),
                            snr_mean=ls[:, l].mean(),
                            snr_p05=np.quantile(ls[:, l], 0.05),
                            snr_p95=np.quantile(ls[:, l], 0.95),
                        )
                    )
        a = stat
        gates[kind] = {
            "max_snr_error_below_1e4": max(
                abs(v[0] - 10) for c in CONDITIONS for v in a[c]
            )
            < 1e-4,
            "rms_control_below_1e6": max(v[3] for v in a["independent_rms"]) < 1e-6,
            "covariance_match_below_1e5": max(v[2] for v in a["covariance"]) < 1e-5,
            "independent_corr_mean": float(np.mean([v[1] for v in a["independent"]])),
            "electrode_corr_mean": float(np.mean([v[1] for v in a["electrode"]])),
            "psd_relative_l1": psd_error,
            "strict_structure_control": kind == "bandpass",
        }
        if kind == "bandpass":
            gates[kind]["independent_corr_below_008"] = (
                gates[kind]["independent_corr_mean"] < 0.08
            )
            gates[kind]["mapped_corr_increase_above_005"] = (
                gates[kind]["electrode_corr_mean"]
                - gates[kind]["independent_corr_mean"]
                > 0.05
            )
            gates[kind]["aggregate_psd_error_below_015"] = (
                max(psd_error.values()) < 0.15
            )
    # Unit disturbances are propagated explicitly, including the Wilson reference.
    A = get_lead_matrix()
    unit = {
        e: [LEADS[i] for i in np.flatnonzero(A[:, j])] for j, e in enumerate(ELECTRODES)
    }
    (tables / "unit_electrode_propagation.json").write_text(
        json.dumps(unit, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(tables / "noise_summary.csv", index=False)
    pd.DataFrame(leads).to_csv(tables / "noise_lead_statistics.csv", index=False)
    pd.DataFrame(psds).to_csv(tables / "noise_psd.csv", index=False)
    pd.DataFrame(bands).to_csv(tables / "noise_bandpower.csv", index=False)
    pd.DataFrame(covrows).to_csv(tables / "noise_covariance.csv", index=False)
    pd.DataFrame(factors).to_csv(tables / "noise_covariance_factor.csv", index=False)
    (tables / "noise_sources.json").write_text(
        json.dumps(sources, indent=2), encoding="utf-8"
    )
    np.savez_compressed(metrics / "noise_examples.npz", **examples)
    failures = [
        (kind, key)
        for kind, values in gates.items()
        for key, val in values.items()
        if isinstance(val, (bool, np.bool_))
        and not val
        and key != "strict_structure_control"
    ]
    report = {
        "passed": not failures,
        "failures": failures,
        "gates": gates,
        "audit_split": "validation",
        "ecg_ids": meta.iloc[indices].ecg_id.astype(int).tolist(),
        "config": cfg,
        "notes": "NSTDB PSD/correlation recorded, not assumed matched. Covariance is empirical, sample-conditioned; no claim of identical higher-order law.",
    }
    (tables / "noise_gate.json").write_text(
        json.dumps(
            report, indent=2, default=lambda v: v.item() if hasattr(v, "item") else v
        ),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"Noise gates failed: {failures}")
    print(f'Noise gates PASS: {tables / "noise_gate.json"}', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)
