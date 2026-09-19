"""Freeze portable test inputs and copies of the six original checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
import shutil

from .common import (WORKSPACE, array_sha256, file_info, load_config, read_json,
                     resolve_path, save_json, sha256, stage_paths)
import numpy as np
import torch
from phase1_ecg_robustness.src.datasets import load_data, _data_identity
from phase1_ecg_robustness.src.supplemental_statistics import patient_draws

CORE_MODULES = ("common", "prepare", "bootstrap", "noise", "infer", "analyse_new", "analyse_existing")
BASELINE_MODULES = ("datasets", "models", "train", "evaluate", "lead_matrix", "noise_generators",
                    "covariance_matching", "audit_noise", "statistics", "supplemental_statistics")
PHASE2_MODULES = ("common", "generate_phase2_noise", "evaluate_phase2", "statistics_phase2")


def run(config, stage):
    cfg = load_config(config)
    paths = stage_paths(cfg, stage)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    if stage == "full":
        gate = read_json(stage_paths(cfg, "smoke")["logs"] / "smoke_verification.json")
        if gate.get("status") != "passed" or gate.get("config_sha256") != cfg["_config_sha256"]:
            raise ValueError("Full execution requires current-protocol smoke verification")
    baseline = WORKSPACE / "phase1_ecg_robustness"
    baseline_protocol_path = baseline / "results/metrics/full/evaluation_protocol.json"
    baseline_protocol = read_json(baseline_protocol_path)
    if baseline_protocol.get("status") != "completed":
        raise ValueError("The reusable phase-one experiment is incomplete")
    x, y, metadata = load_data(baseline / "data/processed")
    identity = _data_identity(baseline / "data/processed", metadata, y)
    indices = np.flatnonzero(metadata.strat_fold.to_numpy() == 10).astype(np.int64)
    if len(indices) != baseline_protocol["n_records"]:
        raise ValueError("Official test-fold size changed")
    limit = cfg["stages"][stage]["test_limit"]
    selected = indices if limit is None else indices[:int(limit)]
    ids = metadata.iloc[selected].ecg_id.to_numpy(dtype=np.int64)
    patients = metadata.iloc[selected].patient_id.to_numpy()
    unique_patients, inverse = np.unique(patients, return_inverse=True)
    reference = dict(y=np.asarray(y[selected], dtype=np.float32), ids=ids,
                     patient_ids=patients, indices=selected, unique_patients=unique_patients,
                     patient_inverse=inverse.astype(np.int64))
    if np.any(reference["y"].sum(axis=0) == 0) or np.any(reference["y"].sum(axis=0) == len(selected)):
        raise ValueError("Prepared cohort lacks positive or negative labels for a class")
    clean = np.asarray(x[selected], dtype=np.float32)
    if clean.shape != (len(selected), 12, 1000) or not np.isfinite(clean).all():
        raise ValueError("Input geometry or finite-value contract changed")
    if array_sha256(metadata.iloc[indices].ecg_id.to_numpy(dtype=np.int64)) != baseline_protocol["ids_sha256"]:
        raise ValueError("Original phase-one ECG order changed")

    input_root = paths["inputs"]
    np.save(input_root / "clean.npy", clean, allow_pickle=False)
    np.savez(input_root / "cohort.npz", **reference)
    bootstrap_source = WORKSPACE / "phase2/results/tables/full/patient_bootstrap"
    bootstrap_manifest_path = bootstrap_source / "manifest.json"
    bootstrap_manifest = read_json(bootstrap_manifest_path)
    for field in ("cohort", "draws"):
        source = resolve_path(bootstrap_manifest[field]["path"])
        if sha256(source) != bootstrap_manifest[field]["sha256"]:
            raise ValueError(f"Original phase-two bootstrap {field} changed")
    if stage == "full":
        with np.load(resolve_path(bootstrap_manifest["cohort"]["path"]), allow_pickle=False) as original:
            for key, value in reference.items():
                if key not in original or not np.array_equal(value, original[key]):
                    raise ValueError(f"Phase-one/two cohort mismatch: {key}")
        shutil.copy2(resolve_path(bootstrap_manifest["draws"]["path"]), input_root / "patient_draws.npy")
        draws = np.load(input_root / "patient_draws.npy", mmap_mode="r", allow_pickle=False)
        if draws.shape != (cfg["statistics"]["bootstrap_replicates"], len(unique_patients)):
            raise ValueError("Original patient draw count differs from frozen supplement")
    else:
        draws = patient_draws(len(unique_patients), cfg["stages"][stage]["bootstrap_replicates"],
                              cfg["statistics"]["bootstrap_seed"], cfg["statistics"]["bootstrap_batch_size"])
        np.save(input_root / "patient_draws.npy", draws, allow_pickle=False)

    checkpoint_dir = input_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    registry, original_checkpoints = [], []
    expected_hashes = {key.replace("\\", "/"): value for key, value in baseline_protocol["checkpoints_sha256"].items()}
    for model in cfg["models"]:
        for seed in cfg["phase1_training_seeds"]:
            source = baseline / f"results/checkpoints/full/{model}/seed_{seed}.pt"
            source_info = file_info(source)
            relative = source.relative_to(baseline).as_posix()
            if source_info["sha256"] != expected_hashes[relative]:
                raise ValueError(f"Original checkpoint changed: {source}")
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            if not checkpoint.get("completed") or checkpoint.get("training_epochs_completed") != baseline_protocol["config"]["train"]["epochs"]:
                raise ValueError("Cannot reuse an unfinished checkpoint")
            if checkpoint["config"] != baseline_protocol["config"] or checkpoint["data_identity"] != identity:
                raise ValueError("Frozen checkpoint configuration/data identity mismatch")
            if checkpoint["model_name"] != model or checkpoint["seed"] != seed or not np.array_equal(checkpoint["test_indices"], indices):
                raise ValueError("Checkpoint model/seed/test-order identity mismatch")
            destination = checkpoint_dir / f"{model}__seed_{seed}.pt"
            if not destination.exists() or sha256(destination) != source_info["sha256"]:
                shutil.copy2(source, destination)
            copied = file_info(destination)
            if copied["sha256"] != source_info["sha256"]:
                raise ValueError("Checkpoint copy is not byte-identical")
            thresholds = np.asarray(checkpoint["thresholds"])
            registry.append(dict(model=model, seed=seed, **copied,
                                 scale_mv=float(checkpoint["scale_mv"]), thresholds=thresholds.tolist(),
                                 thresholds_dtype=str(thresholds.dtype), model_kwargs=checkpoint["model_kwargs"],
                                 original_source=source_info))
            original_checkpoints.append(source_info)
    save_json(input_root / "checkpoints.json", dict(status="completed", stage=stage,
              config_sha256=cfg["_config_sha256"], checkpoints=registry))

    sources = [WORKSPACE / f"methodology_supplement/{name}.py" for name in CORE_MODULES]
    sources += [baseline / f"src/{name}.py" for name in BASELINE_MODULES]
    sources += [WORKSPACE / f"phase2/src/{name}.py" for name in PHASE2_MODULES]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise ValueError(f"Implementation incomplete before freeze: {missing}")
    protected = [baseline_protocol_path, baseline / "data/processed/preparation.json",
                 baseline / "data/processed/metadata.csv", baseline / "data/processed/labels.npy",
                 baseline / "reports/phase1_final_report.md",
                 WORKSPACE / "phase2/reports/phase2_final_report.md",
                 WORKSPACE / "phase2/results/logs/final_acceptance.json",
                 WORKSPACE / "phase2/results/logs/full/statistics_protocol.json",
                 WORKSPACE / "phase2/results/logs/full/evaluation_protocol.json",
                 WORKSPACE / "phase2/results/tables/full/metrics.csv",
                 WORKSPACE / "phase2/results/tables/full/group_seed_metrics.csv",
                 WORKSPACE / "phase2/results/tables/full/primary_comparisons.csv",
                 WORKSPACE / "phase2/results/test_inputs/full/manifest.json", bootstrap_manifest_path]
    source_files = [file_info(path) for path in sources]
    record = dict(status="completed", stage=stage, config_sha256=cfg["_config_sha256"],
                  config_file=file_info(cfg["_config_path"]),
                  frozen_at=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                  scope=cfg["included"], excluded=cfg["excluded"], no_training=True,
                  source_files=source_files,
                  protected_files=[file_info(path) for path in protected] + original_checkpoints + source_files,
                  data_identity=identity, n_records=len(selected), n_patients=len(unique_patients),
                  clean=file_info(input_root / "clean.npy"), cohort=file_info(input_root / "cohort.npz"),
                  draws=file_info(input_root / "patient_draws.npy"), checkpoints=file_info(input_root / "checkpoints.json"),
                  phase2_bootstrap_manifest=file_info(bootstrap_manifest_path),
                  phase2_draws_reused_byte_for_byte=stage == "full", array_hashes={key: array_sha256(value) for key, value in reference.items()})
    save_json(paths["logs"] / "freeze.json", record)
    print(json.dumps({"status": "completed", "stage": stage, "config_sha256": cfg["_config_sha256"],
                      "n_records": len(selected), "n_patients": len(unique_patients), "checkpoints": len(registry)}, indent=2), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
