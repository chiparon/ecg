"""Prepare all available 100 Hz official PTB-XL records in physical millivolts.

Default label policy follows the official example: every diagnostic code key is
included, even likelihood 0 (unknown, NOT absence). --min-likelihood is an
explicit sensitivity analysis excluding unknown likelihoods. No record-random
split, temporal filtering, amplitude normalization or resampling is performed.
"""

import argparse
import ast
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import wfdb

from .datasets import CLASSES, LEADS, select_splits, validate_patient_folds
from .download_ptbxl import checksums, sha256, write_json


def map_superclasses(codes, statements, min_likelihood=None):
    """Map diagnostic SCP keys; likelihood zero is unknown under the primary policy."""
    if isinstance(codes, str):
        codes = ast.literal_eval(codes)
    if not isinstance(codes, dict):
        raise ValueError("scp_codes must parse to a dictionary")
    if min_likelihood is not None and not 0 < min_likelihood <= 100:
        raise ValueError(
            "min_likelihood must be >0 and <=100; omit for official all-keys policy"
        )
    y = np.zeros(len(CLASSES), dtype=np.float32)
    for code, likelihood in codes.items():
        value = float(likelihood)
        if not np.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid likelihood for {code}: {likelihood}")
        if min_likelihood is not None and value < min_likelihood:
            continue
        if code not in statements.index:
            continue
        row = statements.loc[code]
        if (
            float(row.get("diagnostic", 0) or 0) == 1
            and row.diagnostic_class in CLASSES
        ):
            y[CLASSES.index(row.diagnostic_class)] = 1
    return y


def align_waveform(signal, fields):
    """Check source metadata, reorder leads, convert recognized units to mV, demean."""
    signal = np.asarray(signal)
    names = [str(name).strip().upper() for name in fields.get("sig_name", [])]
    expected = [name.upper() for name in LEADS]
    if len(names) != 12 or len(set(names)) != 12 or set(names) != set(expected):
        raise ValueError(f"Expected exactly the standard 12 named leads, got {names}")
    if float(fields.get("fs", 0)) != 100 or signal.shape != (1000, 12):
        raise ValueError(
            f"Expected 100 Hz, 1000x12; got fs={fields.get('fs')}, shape={signal.shape}"
        )
    if not np.isfinite(signal).all():
        raise ValueError("WFDB waveform contains non-finite samples")
    units = fields.get("units", [])
    if len(units) != 12:
        raise ValueError("Missing per-lead physical units")
    factors = {"mv": 1.0, "uv": 0.001, "μv": 0.001, "µv": 0.001, "v": 1000.0}
    try:
        scale = np.array([factors[str(unit).strip().lower()] for unit in units])
    except KeyError as error:
        raise ValueError(f"Unsupported physical waveform units: {units}") from error
    order = [names.index(name) for name in expected]
    mv = (signal * scale)[..., order].T
    means = mv.mean(axis=1, keepdims=True)
    mv = np.asarray(mv - means, dtype=np.float32)
    if not np.isfinite(mv).all():
        raise ValueError("Waveform overflow after physical unit conversion")
    qc = {
        "original_leads": fields["sig_name"],
        "original_units": units,
        "sampling_rate": float(fields["fs"]),
        "shape": list(mv.shape),
        "removed_dc_mv": means[:, 0].tolist(),
        "min_mv": float(mv.min()),
        "max_mv": float(mv.max()),
        "rms_mv_by_lead": np.sqrt(np.mean(mv.astype(np.float64) ** 2, axis=1)).tolist(),
        "max_abs_temporal_mean_mv": float(np.abs(mv.mean(axis=1)).max()),
        "finite": True,
    }
    return mv, qc


def counts(metadata, labels):
    splits = select_splits(metadata, {})
    result = {}
    for name, indices in {"all": np.arange(len(metadata)), **splits}.items():
        result[name] = {
            "records": len(indices),
            "patients": int(metadata.iloc[indices].patient_id.nunique()),
            "positive_counts": dict(
                zip(CLASSES, labels[indices].sum(axis=0).astype(int).tolist())
            ),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/ptb-xl-1.0.3"),
        help="Existing official metadata, mapping, and records100 directory",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--limit",
        type=int,
        help="Restrict requested metadata to first N rows; omitted requests all available records",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any requested waveform pair is missing instead of recording unavailable IDs",
    )
    parser.add_argument(
        "--min-likelihood",
        type=float,
        help="Sensitivity only: retain diagnostic codes with likelihood >= threshold (>0, <=100); default includes unknown likelihood 0",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip official raw-file SHA256 verification (clearly recorded; format checks still mandatory)",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    if args.min_likelihood is not None and not 0 < args.min_likelihood <= 100:
        parser.error("min-likelihood must be >0 and <=100")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = args.results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "version": "1.0.3",
        "source_url": "https://physionet.org/content/ptb-xl/1.0.3/",
        "raw_dir": str(args.raw_dir.resolve()),
        "lead_order": list(LEADS),
        "class_order": list(CLASSES),
        "sampling_rate": 100,
        "units": "mV",
        "label_policy": (
            "all diagnostic code keys including likelihood 0=unknown"
            if args.min_likelihood is None
            else f"sensitivity: likelihood >= {args.min_likelihood}; unknown 0 excluded"
        ),
        "min_likelihood": args.min_likelihood,
        "preprocessing": "WFDB physical unit conversion, lead reorder, per-record per-lead temporal DC removal only",
        "hash_check_requested": not args.skip_hash_check,
        "first_five_ecg_qc": [],
        "errors": [],
    }
    write_json(args.output_dir / "preparation.json", report)
    try:
        metadata_path = args.raw_dir / "ptbxl_database.csv"
        mapping_path = args.raw_dir / "scp_statements.csv"
        metadata = pd.read_csv(metadata_path)
        statements = pd.read_csv(mapping_path, index_col=0)
        if not statements.index.is_unique:
            raise ValueError("Duplicate SCP mapping keys")
        report["official_metadata_records"] = len(metadata)
        report["official_metadata_patients"] = int(metadata.patient_id.nunique())
        report["leakage"] = {"official_metadata": validate_patient_folds(metadata)}
        requested = (
            metadata.iloc[: args.limit].copy() if args.limit else metadata.copy()
        )
        report["requested_records"] = len(requested)
        report["requested_patients"] = int(requested.patient_id.nunique())
        checksum_path = args.raw_dir / "SHA256SUMS.txt"
        hashes = checksums(checksum_path) if checksum_path.exists() else {}
        if not hashes and not args.skip_hash_check:
            raise FileNotFoundError(
                "Official SHA256SUMS.txt required; download it or explicitly use --skip-hash-check"
            )
        file_checks = []

        def check_file(path, relative):
            row = {"path": relative, "bytes": path.stat().st_size}
            if not args.skip_hash_check:
                if relative not in hashes:
                    raise ValueError(f"No official SHA256 for {relative}")
                digest = sha256(path)
                if digest != hashes[relative]:
                    raise ValueError(f"SHA256 mismatch: {relative}")
                row.update(sha256=digest, verified=True)
            else:
                row["verified"] = False
            file_checks.append(row)

        check_file(metadata_path, "ptbxl_database.csv")
        check_file(mapping_path, "scp_statements.csv")
        available, labels, exclusions, missing = [], [], [], []
        for position, row in requested.iterrows():
            relative = str(row.filename_lr)
            if (
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not relative.startswith("records100/")
            ):
                raise ValueError(f"Unsafe/non-100Hz source path: {relative}")
            absent = [
                relative + ext
                for ext in (".hea", ".dat")
                if not (args.raw_dir / (relative + ext)).is_file()
            ]
            if absent:
                missing.append({"ecg_id": int(row.ecg_id), "missing_files": absent})
                continue
            y = map_superclasses(row.scp_codes, statements, args.min_likelihood)
            if not y.any():
                exclusions.append(
                    {
                        "ecg_id": int(row.ecg_id),
                        "reason": "empty diagnostic superclass labels under selected policy",
                    }
                )
                continue
            available.append(position)
            labels.append(y)
        report["missing_records"] = missing
        report["empty_label_exclusions"] = exclusions
        report["available_requested_records"] = len(available) + len(exclusions)
        if args.require_all and missing:
            raise FileNotFoundError(
                f"{len(missing)} requested records missing; see preparation.json"
            )
        if not available:
            raise ValueError(
                "No available records with nonempty diagnostic superclass labels"
            )
        retained = metadata.loc[available].reset_index(drop=True)
        y = np.stack(labels)
        report["leakage"]["retained"] = validate_patient_folds(retained)
        signal_tmp = args.output_dir / "signals.tmp.npy"
        signals = np.lib.format.open_memmap(
            signal_tmp, mode="w+", dtype=np.float32, shape=(len(retained), 12, 1000)
        )
        for output_row, row in retained.iterrows():
            relative = str(row.filename_lr)
            for ext in (".hea", ".dat"):
                check_file(args.raw_dir / (relative + ext), relative + ext)
            wave, fields = wfdb.rdsamp(str(args.raw_dir / relative))
            signals[output_row], qc = align_waveform(wave, fields)
            if output_row < 5:
                report["first_five_ecg_qc"].append({"ecg_id": int(row.ecg_id), **qc})
            if (output_row + 1) % 1000 == 0:
                print(f"Prepared {output_row + 1}/{len(retained)} records", flush=True)
        signals.flush()
        del signals
        np.save(args.output_dir / "labels.tmp.npy", y)
        retained.to_csv(args.output_dir / "metadata.tmp.csv", index=False)
        count_info = counts(retained, y)
        cooccurrence = (y.astype(np.int64).T @ y.astype(np.int64)).tolist()
        report.update(
            {
                "counts": count_info,
                "retained_records": len(retained),
                "retained_patients": int(retained.patient_id.nunique()),
                "cooccurrence": {"classes": list(CLASSES), "matrix": cooccurrence},
                "file_checks": {
                    "checked_files": len(file_checks),
                    "verified_files": sum(r["verified"] for r in file_checks),
                    "inventory": str(tables / "dataset_file_checks.json"),
                    "all_waveforms_finite": True,
                    "all_waveform_shapes": [12, 1000],
                    "all_sampling_rates": 100,
                },
                "metadata_sha256": sha256(metadata_path),
                "mapping_sha256": sha256(mapping_path),
            }
        )
        summary = [
            {
                "split": name,
                **{k: v for k, v in info.items() if k != "positive_counts"},
                **info["positive_counts"],
            }
            for name, info in count_info.items()
        ]
        pd.DataFrame(summary).to_csv(tables / "dataset_summary.csv", index=False)
        write_json(
            tables / "dataset_counts.json",
            {
                "official_records": len(metadata),
                "requested_records": len(requested),
                "missing_records": len(missing),
                "empty_label_exclusions": len(exclusions),
                "retained": count_info,
            },
        )
        write_json(tables / "dataset_cooccurrence.json", report["cooccurrence"])
        write_json(tables / "dataset_leakage.json", report["leakage"])
        write_json(
            tables / "dataset_exclusions.json",
            {"empty_labels": exclusions, "unavailable": missing},
        )
        write_json(tables / "dataset_file_checks.json", file_checks)
        for temporary, final in [
            ("signals.tmp.npy", "signals.npy"),
            ("labels.tmp.npy", "labels.npy"),
            ("metadata.tmp.csv", "metadata.csv"),
        ]:
            os.replace(args.output_dir / temporary, args.output_dir / final)
        report["prepared_ecg_ids"] = retained.ecg_id.astype(int).tolist()
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["errors"].append(repr(error))
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output_dir / "preparation.json", report)
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
