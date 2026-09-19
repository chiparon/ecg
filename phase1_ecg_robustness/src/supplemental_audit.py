"""Read-only legacy reconstruction and data/sampling/covariance review supplements.

Run from the project root. Only results/{tables,logs}/review_supplement are
written; no legacy prediction, cache, checkpoint, or preparation file is changed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import scipy
from scipy.integrate import trapezoid
from scipy.signal import resample_poly
import torch
import yaml

from .audit_noise import CONDITIONS, covariance, correlation, powers, record_seed
from .datasets import CLASSES, LEADS, _data_identity, load_data, select_splits
from .evaluate import predict, sha256
from .lead_matrix import (
    ELECTRODES,
    get_lead_matrix,
    matrix_provenance,
    validate_lead_matrix,
)
from .models import build_model
from .noise_generators import make_noise_triplet, noise_diagnostics, nstdb_source_info
from .prepare_ptbxl import map_superclasses
from .train import seed_everything

DEPENDENT = ("independent_rms", "electrode", "covariance")
EXPECTED_RUNS = {
    "smoke": 12,
    "pilot": 27,
    "full": 378,
    "pilot_noise_repeats": 135,
    "full_noise_repeats": 270,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_info(path):
    path = Path(path)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def equal(actual, expected, label):
    require(np.array_equal(actual, expected, equal_nan=True), f"Mismatch: {label}")


def pinned_matrix(cfg):
    result = matrix_provenance()
    require(
        result["sha256"] == cfg["matrix_sha256"],
        "Configured matrix fingerprint differs",
    )
    validation = validate_lead_matrix()
    require(
        validation["rank"] == 8
        and validation["lead_order"] == list(LEADS)
        and validation["electrode_order"] == list(ELECTRODES),
        "Matrix coordinate/rank mismatch",
    )
    equal(get_lead_matrix()[6:, :3], np.full((6, 3), -1 / 3), "negative WCT")
    return result


def sampling_audit(base_cfg, tables):
    """Inspect the imported implementation, not a generic SciPy description."""
    signature = inspect.signature(resample_poly)
    source = inspect.getsource(resample_poly)
    required = (
        "max_rate = max(up, down)",
        "f_c = 1. / max_rate",
        "half_len = 10 * max_rate",
        "firwin(2 * half_len + 1",
        "n_pre_pad = (down - half_len % down)",
        "h *= up",
        "y = upfirdn(",
    )
    require(
        all(s in source for s in required),
        "Installed resample_poly algorithm changed; inspect before reporting FIR parameters",
    )
    require(
        signature.parameters["window"].default == ("kaiser", 5.0)
        and signature.parameters["padtype"].default == "constant"
        and signature.parameters["cval"].default is None,
        "Installed resampling defaults changed",
    )
    sources = {}
    for kind in ("bw", "ma", "em"):
        info = nstdb_source_info(base_cfg["noise"]["nstdb_dir"], kind, 100)
        require(
            (
                info["source_fs"],
                info["target_fs"],
                info["resample_up"],
                info["resample_down"],
            )
            == (360, 100, 5, 18),
            "Unexpected NSTDB sample-rate provenance",
        )
        info["source_files"] = [
            file_info(Path(info["record_path"] + ext)) for ext in (".hea", ".dat")
        ]
        sources[kind] = info
    write_json(
        tables / "sampling_protocol.json",
        {
            "status": "completed",
            "ecg": {
                "sampling_rate_hz": 100,
                "samples": 1000,
                "duration_seconds": 10,
                "source": "official records100/filename_lr",
                "preprocessing": "physical mV conversion, lead reorder, per-record per-lead DC removal; no ECG resampling",
                "run_500hz": False,
                "limitation": "No 500 Hz experiment or cross-sampling-rate performance claim",
            },
            "scipy_version": scipy.__version__,
            "resample_poly_signature": str(signature),
            "implementation_file": file_info(inspect.getsourcefile(resample_poly)),
            "implementation_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "implementation_source": source,
            "nstdb": sources,
            "filter": {
                "up": 5,
                "down": 18,
                "window": "Kaiser",
                "beta": 5.0,
                "half_length": 180,
                "fir_taps_before_alignment_padding": 361,
                "cutoff_relative_to_upsampled_nyquist": 1 / 18,
                "upsampled_rate_hz": 1800,
                "cutoff_hz": 50,
                "gain_multiplier": 5,
                "filter_prepad_zeros": 18,
                "output_samples_removed_at_start": 11,
                "boundary": "constant zero outside original record (padtype=constant, cval=None -> 0)",
                "alignment": "symmetric odd FIR; prepad and crop compensate group delay; conditional postpadding ensures output length ceil(N*5/18)",
                "antialiasing": "linear-phase low-pass FIR before downsampling by 18; finite transition band, not ideal brick wall",
            },
        },
    )


def data_audit(base_cfg, tables):
    data_dir = Path(base_cfg["data_dir"])
    prep_path = data_dir / "preparation.json"
    prep = read_json(prep_path)
    raw = Path(prep["raw_dir"])
    metadata_path, mapping_path = raw / "ptbxl_database.csv", raw / "scp_statements.csv"
    official = pd.read_csv(metadata_path)
    mapping = pd.read_csv(mapping_path, index_col=0)
    require(
        len(official) == 21799 and not official.ecg_id.duplicated().any(),
        "Official metadata count/IDs differ",
    )
    require(mapping.index.is_unique, "Ambiguous SCP mapping")
    require(
        prep["status"] == "complete"
        and prep["requested_records"] == len(official)
        and prep["min_likelihood"] is None
        and not prep["errors"],
        "Full primary preparation contract differs",
    )
    require(
        prep["metadata_sha256"] == sha256(metadata_path)
        and prep["mapping_sha256"] == sha256(mapping_path),
        "Official source metadata hashes differ",
    )
    labels = np.stack(
        [map_superclasses(codes, mapping) for codes in official.scp_codes]
    )
    ledger_path = Path(base_cfg["results_dir"]) / "tables" / "dataset_exclusions.json"
    ledger = read_json(ledger_path)
    missing_ids = {int(r["ecg_id"]) for r in prep["missing_records"]}
    empty_ids = {int(r["ecg_id"]) for r in prep["empty_label_exclusions"]}
    require(
        missing_ids == {int(r["ecg_id"]) for r in ledger["unavailable"]}
        and empty_ids == {int(r["ecg_id"]) for r in ledger["empty_labels"]},
        "Original exclusion ledgers disagree",
    )
    actual_missing, rows = set(), []
    for i, row in official.iterrows():
        absent = [
            str(row.filename_lr) + ext
            for ext in (".hea", ".dat")
            if not (raw / (str(row.filename_lr) + ext)).is_file()
        ]
        if absent:
            actual_missing.add(int(row.ecg_id))
        reason = (
            "unavailable"
            if absent
            else "empty_diagnostic_labels" if not labels[i].any() else "included"
        )
        codes = ast.literal_eval(row.scp_codes)
        diagnostic = {
            k: v
            for k, v in codes.items()
            if k in mapping.index
            and float(mapping.loc[k].get("diagnostic", 0) or 0) == 1
        }
        rows.append(
            {
                "ecg_id": int(row.ecg_id),
                "patient_id": row.patient_id,
                "strat_fold": int(row.strat_fold),
                "included": reason == "included",
                "reason": reason,
                "source_filename": row.filename_lr,
                "filename_hr": row.filename_hr,
                "scp_codes": row.scp_codes,
                "diagnostic_codes": json.dumps(diagnostic, sort_keys=True),
                "diagnostic_superclasses": "|".join(
                    name for name, v in zip(CLASSES, labels[i]) if v
                ),
                "unknown_zero_diagnostic_codes": "|".join(
                    k for k, v in diagnostic.items() if float(v) == 0
                ),
                "missing_files": "|".join(absent),
                **{name: int(v) for name, v in zip(CLASSES, labels[i])},
            }
        )
    require(
        actual_missing == missing_ids,
        "Current waveform availability differs from preparation ledger",
    )
    audit = pd.DataFrame(rows)
    require(
        set(audit.loc[audit.reason.eq("empty_diagnostic_labels"), "ecg_id"])
        == empty_ids,
        "Independent all-code mapping differs from original empty-label ledger",
    )
    retained = audit.included.to_numpy()
    require(
        int(retained.sum()) == 21388 and int((~retained).sum()) == 411,
        "Retained/excluded reconciliation differs",
    )
    # Every cache is reconciled against its own ordered preparation IDs, not a new split.
    cache_checks = []
    for directory in sorted(Path("data").glob("processed*")):
        if not (directory / "preparation.json").is_file():
            continue
        cp = read_json(directory / "preparation.json")
        x, y, meta = load_data(directory)
        require(
            cp["metadata_sha256"] == prep["metadata_sha256"]
            and cp["mapping_sha256"] == prep["mapping_sha256"]
            and not cp["errors"],
            f"{directory} source metadata or preparation status differs",
        )
        requested = official.iloc[: cp["requested_records"]]
        requested_y = (
            labels[: len(requested)]
            if cp.get("min_likelihood") is None
            else np.stack(
                [
                    map_superclasses(c, mapping, cp["min_likelihood"])
                    for c in requested.scp_codes
                ]
            )
        )
        cache_missing = {int(r["ecg_id"]) for r in cp["missing_records"]}
        available_mask = ~requested.ecg_id.isin(cache_missing).to_numpy()
        nonempty_mask = requested_y.any(axis=1)
        require(
            set(requested.loc[available_mask & ~nonempty_mask, "ecg_id"])
            == {int(r["ecg_id"]) for r in cp["empty_label_exclusions"]},
            f"{directory} independently mapped exclusion ledger differs",
        )
        equal(
            meta.ecg_id.to_numpy(),
            requested.loc[available_mask & nonempty_mask, "ecg_id"].to_numpy(),
            f"{directory} independently retained ordered IDs",
        )
        positions = pd.Index(official.ecg_id).get_indexer(meta.ecg_id)
        require((positions >= 0).all(), f"Unknown cached ECG: {directory}")
        equal(
            meta.ecg_id.to_numpy(),
            np.array(cp["prepared_ecg_ids"]),
            f"{directory} ledger IDs",
        )
        for column in ("patient_id", "strat_fold"):
            equal(
                meta[column].to_numpy(),
                official.iloc[positions][column].to_numpy(),
                f"{directory} {column}",
            )
        expected_y = (
            labels[positions]
            if cp.get("min_likelihood") is None
            else np.stack(
                [
                    map_superclasses(c, mapping, cp["min_likelihood"])
                    for c in official.iloc[positions].scp_codes
                ]
            )
        )
        equal(y, expected_y, f"{directory} labels")
        require(
            meta.filename_lr.tolist() == official.iloc[positions].filename_lr.tolist(),
            f"{directory} filenames differ",
        )
        if directory.resolve() == data_dir.resolve():
            equal(
                meta.ecg_id.to_numpy(),
                official.loc[retained, "ecg_id"].to_numpy(),
                "full independently retained IDs",
            )
        cache_checks.append(
            {
                "path": str(directory),
                "records": len(meta),
                "shape": list(x.shape),
                "metadata": file_info(directory / "metadata.csv"),
                "labels": file_info(directory / "labels.npy"),
                "preparation": file_info(directory / "preparation.json"),
                "aligned": True,
            }
        )
    inventory_path = Path(prep["file_checks"]["inventory"])
    inventory = read_json(inventory_path)
    expected_files = {"ptbxl_database.csv", "scp_statements.csv"} | {
        str(name) + ext
        for name in official.loc[retained, "filename_lr"]
        for ext in (".hea", ".dat")
    }
    require(
        {r["path"] for r in inventory} == expected_files
        and len(inventory) == len(expected_files)
        and all(r.get("verified") for r in inventory),
        "Original verified file inventory differs",
    )
    audit.to_csv(tables / "data_record_audit.csv", index=False)
    write_json(
        tables / "data_audit.json",
        {
            "status": "completed",
            "official_records": len(official),
            "retained_records": int(retained.sum()),
            "excluded_records": int((~retained).sum()),
            "reason_counts": audit.reason.value_counts().to_dict(),
            "categories": {
                "empty_labels": len(empty_ids),
                "unavailable": len(missing_ids),
                "preprocessing_failures": 0,
                "duplicate_filtering": 0,
                "quality_filtering": 0,
                "additional_label_filters": 0,
            },
            "category_evidence": "Current preparation source has only missing-pair and empty-label exclusions. Duplicate IDs and waveform QC failures raise (no silent dropping); complete/error-free preparation plus exhaustive ordered cache/label/ledger reconciliation supports zero additional exclusions. No source-quality annotation filter is present. This is retrospective evidence, not a generation-time code fingerprint.",
            "label_policy": "Every diagnostic SCP code key including likelihood 0=unknown; no positive-likelihood threshold",
            "sources": [
                file_info(p)
                for p in (
                    metadata_path,
                    mapping_path,
                    prep_path,
                    ledger_path,
                    inventory_path,
                    Path("src/prepare_ptbxl.py"),
                    Path("src/datasets.py"),
                )
            ],
            "raw_waveform_verification": "Existence checked now; historical verified SHA256 inventory reconciled, not all waveform bytes rehashed",
            "caches": cache_checks,
        },
    )
    sampling_audit(base_cfg, tables)


def load_legacy_run(name, results):
    repeated = name.endswith("_noise_repeats")
    base_name = name.removesuffix("_noise_repeats")
    config_path = Path("configs") / f"phase1_{base_name}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    folder = results / "metrics" / name
    protocol_path = folder / (
        "protocol.json" if repeated else "evaluation_protocol.json"
    )
    protocol = read_json(protocol_path)
    require(
        protocol["status"] == "completed"
        and protocol["source_config" if repeated else "config"] == config,
        f"Incomplete or stale protocol: {name}",
    )
    rows = pd.read_csv(folder / "metrics.csv")
    require(
        len(rows) == protocol["n_evaluations"], f"Protocol row count differs: {name}"
    )
    if repeated:
        rows["kind"], rows["active"] = "bandpass", "all"
        require(
            protocol["kind"] == "bandpass"
            and protocol["noise_seeds"] == [101, 202, 303, 404, 505],
            "Unexpected replay noise bases",
        )
    else:
        rows["noise_seed"] = config["noise"]["seed"]
    keys = ["model", "seed", "noise_seed", "kind", "snr", "condition", "active"]
    require(
        not rows.duplicated(keys).any() and not rows.prediction_path.duplicated().any(),
        f"Duplicate legacy groups: {name}",
    )
    expected = set()
    for model in config["models"]:
        for seed in config["seeds"]:
            if not repeated:
                expected.add(
                    (model, seed, config["noise"]["seed"], "clean", 100, "clean", "all")
                )
            for base in (
                protocol["noise_seeds"] if repeated else [config["noise"]["seed"]]
            ):
                for kind in ["bandpass"] if repeated else config["noise"]["kinds"]:
                    for snr in config["noise"]["snrs"]:
                        for condition in CONDITIONS:
                            expected.add(
                                (model, seed, base, kind, snr, condition, "all")
                            )
                if not repeated:
                    for active in config["noise"].get("electrodes", []):
                        for condition in DEPENDENT:
                            expected.add(
                                (
                                    model,
                                    seed,
                                    base,
                                    "bandpass",
                                    config["noise"].get("electrode_snr", 10),
                                    condition,
                                    active,
                                )
                            )
    require(
        set(rows[keys].itertuples(index=False, name=None)) == expected,
        f"Incomplete/extra legacy grid: {name}",
    )
    recorded = {Path(p).resolve() for p in rows.prediction_path}
    require(
        recorded
        == {
            p.resolve() for p in folder.rglob("*.npz") if p.name != "noise_examples.npz"
        },
        f"Prediction file inventory differs from metrics: {name}",
    )
    require(
        int(rows.condition.isin(DEPENDENT).sum()) == EXPECTED_RUNS[name],
        f"Matrix-dependent scope differs: {name}",
    )
    return config, protocol, rows, config_path, protocol_path


def load_states(config, protocol, repeated, device):
    torch.set_num_threads(2 if repeated else 4)
    seed_everything(0)
    execution = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    require(
        execution == protocol["torch_execution"],
        "Original numeric execution policy differs",
    )
    if "device" in protocol:
        require(
            str(device) == protocol["device"],
            "Replay requires original execution device for numerical comparison",
        )
    X, Y, meta = load_data(config["data_dir"])
    identity = _data_identity(Path(config["data_dir"]), meta, Y)
    fingerprints = {
        Path(p).resolve(): v
        for p, v in protocol[
            "checkpoint_sha256" if repeated else "checkpoints_sha256"
        ].items()
    }
    states, indices = {}, None
    for model_name in config["models"]:
        for seed in config["seeds"]:
            path = (
                Path(config["results_dir"])
                / "checkpoints"
                / config["run_name"]
                / model_name
                / f"seed_{seed}.pt"
            )
            digest = sha256(path)
            require(
                fingerprints.get(path.resolve()) == digest,
                f"Checkpoint hash changed: {path}",
            )
            c = torch.load(path, map_location="cpu", weights_only=False)
            require(
                c["config"] == config
                and c.get("completed")
                and c["training_epochs_completed"] == config["train"]["epochs"]
                and c["data_identity"] == identity
                and c["model_name"] == model_name
                and c["seed"] == seed,
                f"Checkpoint completion/config/data/model identity differs: {path}",
            )
            idx = np.asarray(c["test_indices"], dtype=int)
            equal(
                idx,
                select_splits(meta, config["train"])["test"],
                "official configured test subset",
            )
            if indices is not None:
                equal(indices, idx, "paired checkpoint test indices")
            indices = idx
            model = build_model(model_name, **c["model_kwargs"]).to(device).eval()
            model.load_state_dict(c["model_state"])
            states[(model_name, seed)] = {
                "model": model,
                "scale": float(c["scale_mv"]),
                "thresholds": np.asarray(c["thresholds"]),
                "checkpoint": file_info(path),
            }
    require(np.all(meta.iloc[indices].strat_fold == 10), "Non-test fold in replay")
    return (
        states,
        np.asarray(X[indices], dtype=np.float32),
        {
            "y": np.asarray(Y[indices], dtype=np.float32),
            "ids": meta.iloc[indices].ecg_id.to_numpy(dtype=np.int64),
            "patient_ids": meta.iloc[indices].patient_id.to_numpy(),
            "indices": indices,
        },
        execution,
    )


def reconstruct(config, x, ids, base, kind, active, conditions=DEPENDENT):
    seeds = np.array(
        [record_seed(base, ecg_id, kind) for ecg_id in ids], dtype=np.uint32
    )
    noises = {c: np.empty_like(x) for c in conditions}
    for i, signal in enumerate(x):
        generated = make_noise_triplet(
            signal,
            config["sampling_rate"],
            0,
            int(seeds[i]),
            kind=kind,
            active=None if active == "all" else [active],
            nstdb=config["noise"].get("nstdb_dir"),
            band=tuple(config["noise"]["band"]),
        )
        for condition in conditions:
            noises[condition][i] = generated[condition]
    return noises, seeds


class CovarianceWriter:
    """Stream wide record diagnostics; keep only online summary moments."""

    def __init__(self, tables, matrix):
        self.tables, self.matrix = tables, matrix
        self.stream = (tables / "covariance_matching_records.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.writer, self.stats, self.records = None, {}, 0
        write_json(
            tables / "covariance_matching_protocol.json",
            {"status": "running", "matrix": matrix, "expected_records": 4 * 3 * 2158},
        )

    @staticmethod
    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator > 0 else float("nan")

    def add(self, config, x, metadata, kind, snr, electrode, control):
        for i, (signal, e, c) in enumerate(zip(x, electrode, control)):
            left = noise_diagnostics(
                e, signal, config["sampling_rate"], tuple(config["noise"]["band"])
            )
            right = noise_diagnostics(
                c, signal, config["sampling_rate"], tuple(config["noise"]["band"])
            )
            a, b = left["covariance"], right["covariance"]
            valid = np.isfinite(left["correlation"]) & np.isfinite(right["correlation"])
            variance = np.diag(a)
            variance_error = np.divide(
                abs(np.diag(b) - variance),
                variance,
                out=np.full(12, np.nan),
                where=variance > 0,
            )
            row = {
                "kind": kind,
                "snr": snr,
                "ecg_id": int(metadata["ids"][i]),
                "patient_id": metadata["patient_ids"][i],
                "covariance_relative_frobenius": self.ratio(
                    np.linalg.norm(b - a), np.linalg.norm(a)
                ),
                "correlation_mae": (
                    float(
                        np.mean(
                            abs(
                                right["correlation"][valid] - left["correlation"][valid]
                            )
                        )
                    )
                    if valid.any()
                    else float("nan")
                ),
                "correlation_undefined_cells": int((~valid).sum()),
                "variance_relative_error_max": (
                    float(np.max(variance_error))
                    if np.isfinite(variance_error).all()
                    else float("nan")
                ),
                "variance_zero_denominators": int((variance == 0).sum()),
                "actual_snr_abs_difference_db": abs(
                    powers(signal, e)[0] - powers(signal, c)[0]
                ),
            }
            f, ep, cp = left["frequencies"], left["psd"], right["psd"]
            for j, lead in enumerate((*LEADS, "aggregate")):
                pe = ep[j] if j < 12 else ep.mean(axis=0)
                pc = cp[j] if j < 12 else cp.mean(axis=0)
                total = float(trapezoid(pe, f))
                row[f"{lead}_welch_total_reference_mv2"] = total
                row[f"{lead}_welch_relative_l1"] = self.ratio(
                    trapezoid(abs(pc - pe), f), total
                )
                for band in left["band_powers"]:
                    eb = left["band_powers"][band]
                    cb = right["band_powers"][band]
                    ref, value = (
                        (float(eb[j]), float(cb[j]))
                        if j < 12
                        else (float(eb.mean()), float(cb.mean()))
                    )
                    prefix = f"{lead}_{band}"
                    row[prefix + "_reference_mv2"] = ref
                    row[prefix + "_abs_error_mv2"] = abs(value - ref)
                    row[prefix + "_relative_band_error"] = self.ratio(
                        abs(value - ref), ref
                    )
                    row[prefix + "_relative_total_error"] = self.ratio(
                        abs(value - ref), total
                    )
            if self.writer is None:
                self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
            self.records += 1
            for key, value in row.items():
                if key in ("kind", "snr", "ecg_id", "patient_id"):
                    continue
                stat = self.stats.setdefault(
                    (kind, snr, key), [0, 0.0, 0.0, -np.inf, 0]
                )
                if not np.isfinite(value):
                    stat[4] += 1
                    continue
                stat[0] += 1
                delta = value - stat[1]
                stat[1] += delta / stat[0]
                stat[2] += delta * (value - stat[1])
                stat[3] = max(stat[3], value)

    def finish(self):
        self.stream.close()
        require(
            self.records == 4 * 3 * 2158,
            "Covariance audit did not cover every full-primary record/group",
        )
        rows = []
        for (kind, snr, metric), (
            n,
            mean,
            m2,
            maximum,
            undefined,
        ) in self.stats.items():
            rows.append(
                {
                    "kind": kind,
                    "snr": snr,
                    "metric": metric,
                    "n": n,
                    "undefined": undefined,
                    "mean": mean if n else np.nan,
                    "sd": np.sqrt(max(0, m2) / (n - 1)) if n > 1 else np.nan,
                    "max": maximum if n else np.nan,
                }
            )
        pd.DataFrame(rows).to_csv(
            self.tables / "covariance_matching_summary.csv", index=False
        )
        write_json(
            self.tables / "covariance_matching_protocol.json",
            {
                "status": "completed",
                "records": self.records,
                "distinct_ecgs": 2158,
                "scope": "full primary, all four kinds x three SNR, all-active; shared realization across six trained models",
                "matrix": self.matrix,
                "noise": "actual injected buffer: float64 generator at 0dB -> float32 buffer -> float32 SNR multiplier; before float32 x+n (not ideal generator covariance)",
                "estimators": {
                    "covariance": "demeaned float64 accumulation of injected float32 samples; ddof=0",
                    "covariance_relative_frobenius": "||Ccontrol-Celectrode||F / ||Celectrode||F",
                    "correlation_mae": "mean absolute difference over all jointly defined matrix cells, including diagonal; undefined cell count saved",
                    "variance_relative_error_max": "max over leads abs(var_control-var_electrode)/var_electrode; undefined if any denominator zero",
                    "actual_snr_abs_difference_db": "absolute difference of 10log10(mean(x^2)/mean(n^2)), float64 power accumulation",
                    "welch": "fs=100, nperseg=256, noverlap=128, default periodic Hann, detrend=constant, one-sided density; noise_diagnostics",
                    "welch_relative_l1": "trapezoid(abs(PSDcontrol-PSDelectrode))/trapezoid(PSDelectrode); aggregate uses mean of 12 lead PSDs BEFORE absolute value",
                    "bands": "requested 0.5-40, baseline 0-0.5, ECG 0.5-40, high 40-50 Hz; linearly interpolate exact boundaries then trapezoid",
                    "band_error": "absolute integrated-power difference divided by reference band power AND reference total Welch power, both saved",
                    "denominators": "Reference band/total powers saved per record/lead. Exactly zero gives NaN (explicit undefined summary count); no epsilon floor. Very small positive bands remain reported and may produce large relative-band errors.",
                    "summary": "mean, sample SD(ddof=1), max over records separately per kind/SNR/metric; defined n and undefined counts explicit",
                },
                "limitations": "Covariance matching does not guarantee each finite-window Welch spectrum or temporal marginal distribution. Finite-window spectral mismatch is measured, not suppressed. No equivalence claim from small covariance error alone.",
            },
        )


def artifact_evidence(results, tables, matrix):
    current = validate_lead_matrix()
    checks, artifacts = [], {}

    def link(path):
        path = Path(path)
        key = path.resolve().as_posix()
        if key not in artifacts:
            artifacts[key] = file_info(path)
        return key

    for path in sorted((results / "tables").glob("*/matrix_validation.json")):
        saved = read_json(path)
        for key in (
            "shape",
            "rank",
            "lead_order",
            "electrode_order",
            "contributions",
            "unit_electrode_effects",
        ):
            require(
                saved[key] == current[key],
                f"Saved matrix validation differs: {path}/{key}",
            )
        require(
            max(saved["residuals"].values()) <= 1e-12,
            f"Saved matrix residual failed: {path}",
        )
        checks.append(
            {
                "artifact": link(path),
                "scope": "saved matrix coefficients, order, rank and residuals",
                "passed": True,
            }
        )
    for path in sorted((results / "metrics").glob("*/noise_examples.npz")):
        cfg_path = Path("configs") / f"phase1_{path.parent.name}.yaml"
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        X, _, meta = load_data(config["data_dir"])
        idx = select_splits(meta, config["train"])["val"][0]
        with np.load(path, allow_pickle=False) as z:
            equal(z["A"], get_lead_matrix(), f"{path} saved matrix")
            equal(z["x"], X[idx], f"{path} example input")
            require(
                float(z["fs"]) == config["sampling_rate"],
                f"Example sampling rate differs: {path}",
            )
            noises = make_noise_triplet(
                np.asarray(X[idx], dtype=np.float64),
                config["sampling_rate"],
                10,
                record_seed(config["noise"]["seed"], meta.iloc[idx].ecg_id, "bandpass"),
                kind="bandpass",
                band=tuple(config["noise"]["band"]),
                nstdb=config["noise"].get("nstdb_dir"),
            )
            error = 0.0
            for condition in CONDITIONS:
                cov = covariance(noises[condition])
                for key, value in (
                    (condition, noises[condition]),
                    ("covariance_" + condition, cov),
                    ("correlation_" + condition, correlation(cov)),
                ):
                    require(
                        np.allclose(z[key], value, rtol=0, atol=1e-10),
                        f"Saved example reconstruction differs: {path}/{key}",
                    )
                    error = max(error, float(np.max(abs(z[key] - value))))
        checks.append(
            {
                "artifact": link(path),
                "config": link(cfg_path),
                "scope": "all saved example noise/covariance/correlation arrays",
                "max_absolute_error": error,
                "passed": True,
            }
        )
    figures = []
    for path in sorted((results / "figures").glob("*/figure_manifest.json")):
        if path.parent.name == "review_supplement":
            continue
        saved = read_json(path)
        require(
            saved["sampling_rate_hz"] == 100,
            f"Figure sampling protocol differs: {path}",
        )
        entries = []
        for entry in saved["figures"]:
            entries.append(
                {
                    "family": entry["family"],
                    "files": [link(p) for p in entry["files"]],
                    "source_paths": [link(p) for p in entry["source_paths"]],
                }
            )
        figures.append(
            {
                "manifest": link(path),
                "config": link(saved["config_path"]),
                "entries": entries,
                "evidence_limit": "Current source and current input/file hashes linked retrospectively; image pixels do not establish generation-time provenance",
            }
        )
    for path in (*Path("src").glob("*.py"), Path("scripts/replay_noise.py")):
        link(path)
    archive_path = results / "logs" / "precision_before_fix.zip"
    correction_path = results / "logs" / "precision_correction.json"
    correction = read_json(correction_path)
    require(
        sha256(archive_path) == correction["archive_sha256"],
        "Historical precision archive hash differs",
    )
    with zipfile.ZipFile(archive_path) as archive:
        archive_files = [
            {"path": member.filename, "bytes": member.file_size, "crc32": member.CRC}
            for member in archive.infolist()
            if not member.is_dir()
        ]
    write_json(
        tables / "matrix_artifact_evidence.json",
        {
            "status": "completed",
            "provenance_type": "retrospective_verified_provenance",
            "generation_time_matrix_hash_available": False,
            "matrix": matrix,
            "checks": checks,
            "figures": figures,
            "artifacts": artifacts,
            "historical_precision_archive": {
                "archive": file_info(archive_path),
                "correction": file_info(correction_path),
                "label": "historical pre-correction numeric-policy artifacts; excluded from current 822-file replay, not current evidence",
                "members": archive_files,
            },
        },
    )


def replay_audit(cfg, tables, matrix, covariance_writer=None):
    results = Path(cfg["results_dir"])
    require(
        set(cfg["audit"]["legacy_runs"]) == set(EXPECTED_RUNS),
        "Exhaustive audit requires all five legacy runs",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = tables / "matrix_replay_files.csv"
    manifest = {
        "status": "running",
        "provenance_type": "retrospective_verified_provenance",
        "generation_time_matrix_hash_available": False,
        "matrix": matrix,
        "scope": "All current legacy independent_rms/electrode/covariance groups; clean and original independent are A-independent and excluded from forward replay",
        "tolerance": cfg["audit"]["probability_tolerance"],
        "runs": [],
        "n_replayed": 0,
        "n_excluded": 0,
        "failed_files": 0,
        "max_probability_error": 0.0,
    }
    protocol_path = tables / "matrix_replay_protocol.json"
    write_json(protocol_path, manifest)
    writer = None
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        for run in cfg["audit"]["legacy_runs"]:
            config, protocol, rows, config_path, source_protocol = load_legacy_run(
                run, results
            )
            states, x, metadata, execution = load_states(
                config, protocol, run.endswith("_noise_repeats"), device
            )
            excluded = rows.loc[~rows.condition.isin(DEPENDENT)]
            manifest["n_excluded"] += len(excluded)
            run_info = {
                "run": run,
                "source_config": file_info(config_path),
                "source_protocol": file_info(source_protocol),
                "source_metrics": file_info(results / "metrics" / run / "metrics.csv"),
                "checkpoints": [s["checkpoint"] for s in states.values()],
                "torch_execution": execution,
                "torch_num_threads": torch.get_num_threads(),
                "batch_size": config["train"]["batch_size"],
                "device": str(device),
                "records": len(x),
                "groups": 0,
                "excluded_A_independent": [
                    {"condition": r.condition, **file_info(r.prediction_path)}
                    for r in excluded.itertuples()
                ],
            }
            require(
                len(x) == protocol["n_records"]
                and len(np.unique(metadata["patient_ids"])) == protocol["n_patients"],
                "Saved replay cohort differs",
            )
            if "ids_sha256" in protocol:
                require(
                    hashlib.sha256(metadata["ids"].tobytes()).hexdigest()
                    == protocol["ids_sha256"],
                    "Ordered IDs fingerprint differs",
                )
            selected = rows.loc[rows.condition.isin(DEPENDENT)]
            for (base, kind, active), group in selected.groupby(
                ["noise_seed", "kind", "active"], sort=False
            ):
                noises, seeds = reconstruct(
                    config, x, metadata["ids"], int(base), kind, active
                )
                for snr, snr_rows in group.groupby("snr", sort=False):
                    scaled = {
                        condition: n * np.float32(10 ** (-float(snr) / 20))
                        for condition, n in noises.items()
                    }
                    if (
                        covariance_writer is not None
                        and run == "full"
                        and active == "all"
                    ):
                        covariance_writer.add(
                            config,
                            x,
                            metadata,
                            kind,
                            snr,
                            scaled["electrode"],
                            scaled["covariance"],
                        )
                    for condition, condition_rows in snr_rows.groupby(
                        "condition", sort=False
                    ):
                        n = scaled[condition]
                        intensity = [
                            powers(signal, noise) for signal, noise in zip(x, n)
                        ]
                        expected = {
                            **metadata,
                            "replay_seed": seeds,
                            "actual_snr": np.array([v[0] for v in intensity]),
                            "noise_rms": np.array([v[1] for v in intensity]),
                            "lead_snr": np.array([v[2] for v in intensity]),
                        }
                        require(
                            np.max(abs(expected["actual_snr"] - snr)) <= 1e-4,
                            "Reconstructed SNR calibration failed",
                        )
                        noisy = x + n
                        for row in condition_rows.itertuples():
                            state = states[(row.model, row.seed)]
                            path = Path(row.prediction_path)
                            p = predict(
                                state["model"],
                                noisy,
                                state["scale"],
                                config["train"]["batch_size"],
                                device,
                            )
                            with np.load(path, allow_pickle=False) as saved:
                                for key, value in {
                                    **expected,
                                    "thresholds": state["thresholds"],
                                }.items():
                                    equal(saved[key], value, f"{path}/{key}")
                                require(
                                    saved["p"].shape == p.shape
                                    and np.isfinite(saved["p"]).all()
                                    and np.isfinite(p).all(),
                                    f"Invalid probabilities: {path}",
                                )
                                error = float(np.max(abs(saved["p"] - p)))
                            passed = error <= cfg["audit"]["probability_tolerance"]
                            item = {
                                "run": run,
                                "model": row.model,
                                "seed": row.seed,
                                "noise_seed": int(base),
                                "kind": kind,
                                "snr": snr,
                                "condition": condition,
                                "active": active,
                                "prediction_path": path.as_posix(),
                                "sha256": sha256(path),
                                "records": len(x),
                                "probability_cells": int(p.size),
                                "max_probability_error": error,
                                "metadata_intensity_seed_thresholds_exact": True,
                                "passed": passed,
                                "matrix_sha256": matrix["sha256"],
                                "provenance_type": "retrospective_verified_provenance",
                            }
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(item))
                                writer.writeheader()
                            writer.writerow(item)
                            stream.flush()
                            manifest["n_replayed"] += 1
                            manifest["failed_files"] += int(not passed)
                            manifest["max_probability_error"] = max(
                                manifest["max_probability_error"], error
                            )
                            run_info["groups"] += 1
                    del scaled
                del noises
            require(
                run_info["groups"] == EXPECTED_RUNS[run], f"Incomplete replay: {run}"
            )
            manifest["runs"].append(run_info)
            write_json(protocol_path, manifest)
            del states, x
            print(f"Matrix replay {run}: {run_info['groups']} complete", flush=True)
    require(
        manifest["n_replayed"] == 822 and manifest["n_excluded"] == 229,
        "Exhaustive 1051-file scope not met",
    )
    manifest["status"] = (
        "completed" if not manifest["failed_files"] else "failed_probability_comparison"
    )
    manifest["per_file_results"] = file_info(output_path)
    manifest["reconstruction"] = (
        "record_seed -> make_noise_triplet at 0dB -> assignment to float32 -> float32 multiplier -> float32 x+n -> original model, training scale, batching; 4 primary / 2 repeated-noise Torch threads, deterministic FP32, TF32 disabled"
    )
    write_json(protocol_path, manifest)
    require(
        not manifest["failed_files"],
        "Legacy probability comparison failed; see exhaustive per-file errors",
    )


def covariance_audit(cfg, tables, writer):
    results = Path(cfg["results_dir"])
    config, protocol, _, _, _ = load_legacy_run("full", results)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states, x, metadata, _ = load_states(config, protocol, False, device)
    del states  # No model forward is needed for diagnostic B alone.
    for kind in config["noise"]["kinds"]:
        noises, _ = reconstruct(
            config,
            x,
            metadata["ids"],
            config["noise"]["seed"],
            kind,
            "all",
            ("electrode", "covariance"),
        )
        for snr in config["noise"]["snrs"]:
            multiplier = np.float32(10 ** (-snr / 20))
            writer.add(
                config,
                x,
                metadata,
                kind,
                snr,
                noises["electrode"] * multiplier,
                noises["covariance"] * multiplier,
            )
        del noises


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_supplement.yaml")
    parser.add_argument(
        "--stage", choices=("data", "matrix", "covariance", "all"), default="all"
    )
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    require(
        cfg["run_name"] == "review_supplement",
        "Audit output must remain separate from legacy runs",
    )
    base_cfg = yaml.safe_load(Path(cfg["base_config"]).read_text(encoding="utf-8"))
    tables = Path(cfg["results_dir"]) / "tables" / cfg["run_name"]
    logs = Path(cfg["results_dir"]) / "logs" / cfg["run_name"]
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    matrix = pinned_matrix(cfg)
    log_path = logs / f"audit_{args.stage}.json"
    log = {
        "status": "running",
        "stage": args.stage,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": file_info(args.config),
        "source": file_info(__file__),
        "matrix": matrix,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }
    write_json(log_path, log)
    covariance_writer = None
    try:
        if args.stage in ("data", "all"):
            data_audit(base_cfg, tables)
        if args.stage in ("covariance", "all"):
            covariance_writer = CovarianceWriter(tables, matrix)
        if args.stage in ("matrix", "all"):
            artifact_evidence(Path(cfg["results_dir"]), tables, matrix)
            replay_audit(cfg, tables, matrix, covariance_writer)
        elif args.stage == "covariance":
            covariance_audit(cfg, tables, covariance_writer)
        if covariance_writer is not None:
            covariance_writer.finish()
        log["status"] = "completed"
    except BaseException as error:
        log.update(status="failed", error=repr(error))
        raise
    finally:
        if covariance_writer is not None:
            covariance_writer.stream.close()
        log["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(log_path, log)


if __name__ == "__main__":
    main()
