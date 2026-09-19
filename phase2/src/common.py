"""Frozen phase-two configuration, unchanged phase-one cohorts, and artifact identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from phase1_ecg_robustness.src.datasets import (
    CLASSES,
    LEADS,
    _data_identity,
    load_data,
    select_splits,
)
from phase1_ecg_robustness.src.evaluate import sha256
from phase1_ecg_robustness.src.lead_matrix import ELECTRODES, matrix_provenance
from phase1_ecg_robustness.src.train import _atomic_json

WORKSPACE = Path(__file__).resolve().parents[2]
STRATEGY_CODES = {"clean_only": 0, "independent_rms": 1, "electrode": 2, "mixed": 3}
CATEGORIES = ("checkpoints", "predictions", "tables", "figures", "logs", "test_inputs")


def save_json(path, value):
    _atomic_json(Path(path), value)


def file_info(path):
    path = Path(path).resolve()
    return {
        "path": path.relative_to(WORKSPACE).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def resolve_path(cfg, path):
    value = Path(path)
    return value if value.is_absolute() else Path(cfg["_workspace_root"]) / value


def _canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _combos(path):
    values = json.loads(path.read_text(encoding="utf-8"))
    result, seen = [], set()
    for names in values:
        if (
            not names
            or len(names) != len(set(names))
            or not set(names) <= set(ELECTRODES)
        ):
            raise ValueError(f"Invalid electrode combination in {path}: {names}")
        canonical = tuple(name for name in ELECTRODES if name in names)
        if canonical in seen:
            raise ValueError(f"Duplicate electrode combination: {canonical}")
        seen.add(canonical)
        result.append({"combo_id": "_".join(canonical), "electrodes": list(canonical)})
    return result


def load_config(config_path):
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_combos = _combos(root / raw["train"]["combo_file"])
    heldout_combos = _combos(root / raw["test"]["combo_file"])
    if set(v["combo_id"] for v in train_combos) & set(
        v["combo_id"] for v in heldout_combos
    ):
        raise ValueError("Training and held-out electrode combinations overlap")
    if tuple(raw["class_order"]) != tuple(CLASSES) or tuple(raw["lead_order"]) != tuple(
        LEADS
    ):
        raise ValueError("Phase-one class/lead order must be preserved")
    if raw["matrix_sha256"] != matrix_provenance()["sha256"]:
        raise ValueError("WCT matrix fingerprint differs from the registered matrix")
    if raw["sampling_rate"] != 100 or raw["sequence_length"] != 1000:
        raise ValueError("This experiment uses the unchanged 100Hz/1000-sample cache")
    if set(raw["strategies"]) != set(STRATEGY_CODES):
        raise ValueError("Exactly the four registered main strategies are required")
    for name, probabilities in raw["strategies"].items():
        if set(probabilities) != {"clean", "independent_rms", "electrode"}:
            raise ValueError(f"Invalid strategy probability keys: {name}")
        values = np.asarray(list(probabilities.values()), dtype=float)
        if np.any(values < 0) or not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-12):
            raise ValueError(f"Invalid exclusive input-type probabilities: {name}")
    combo_probabilities = np.asarray(raw["train"]["combo_probabilities"], dtype=float)
    if len(combo_probabilities) != len(train_combos) or np.any(combo_probabilities < 0):
        raise ValueError(
            "Training combo probabilities do not match the enumerated support"
        )
    if not np.isclose(combo_probabilities.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Training combo probabilities must sum to one")
    if set(raw["train"]["snrs"]) != {20, 10}:
        raise ValueError("The registered training SNR support is exactly 20/10dB")
    if not np.isclose(sum(raw["train"]["snr_probabilities"]), 1.0):
        raise ValueError("SNR probabilities must sum to one")
    train_range = raw["train"]["noise_base_range"]
    test_range = raw["test"]["noise_seed_range"]
    if not train_range[0] <= raw["train"]["noise_base"] <= train_range[1]:
        raise ValueError("Training noise base outside the registered training range")
    if max(train_range[0], test_range[0]) <= min(train_range[1], test_range[1]):
        raise ValueError("Training and test noise-base ranges overlap")
    for stage, settings in raw["stages"].items():
        if not all(
            test_range[0] <= seed <= test_range[1]
            for seed in settings["test_noise_seeds"]
        ):
            raise ValueError(f"Test noise seed outside its isolated range: {stage}")
        selected = settings["heldout_combo_ids"]
        if selected != "all" and not set(selected) <= {
            v["combo_id"] for v in heldout_combos
        }:
            raise ValueError(f"Unknown held-out combination in stage {stage}")
    fingerprint = _canonical_hash(
        {"config": raw, "train_combos": train_combos, "heldout_combos": heldout_combos}
    )
    return {
        **raw,
        "_phase2_root": str(root),
        "_workspace_root": str(WORKSPACE),
        "_baseline_root": str((root / raw["baseline_root"]).resolve()),
        "_results_root": str((root / raw["results_dir"]).resolve()),
        "_config_path": str(config_path),
        "_config_sha256": fingerprint,
        "_train_combos": train_combos,
        "_heldout_combos": heldout_combos,
        "_strategy_codes": dict(STRATEGY_CODES),
    }


def stage_settings(cfg, stage):
    if stage not in cfg["stages"]:
        raise ValueError(f"Unknown phase-two stage: {stage}")
    return {**cfg["train"], **cfg["stages"][stage]}


def stage_paths(cfg, stage):
    if stage not in cfg["stages"]:
        raise ValueError(f"Unknown phase-two stage: {stage}")
    return {name: Path(cfg["_results_root"]) / name / stage for name in CATEGORIES}


def checkpoint_path(cfg, stage, strategy, model, seed, epoch=None):
    directory = (
        stage_paths(cfg, stage)["checkpoints"] / strategy / model / f"seed_{seed}"
    )
    return directory / ("best.pt" if epoch is None else f"epoch_{int(epoch):02d}.pt")


def run_directory(cfg, stage, strategy, model, seed):
    return stage_paths(cfg, stage)["logs"] / strategy / model / f"seed_{seed}"


def expected_runs(cfg, stage):
    return [
        (strategy, model, int(seed))
        for model in cfg["models"]
        for seed in cfg["stages"][stage]["seeds"]
        for strategy in cfg["strategies"]
    ]


def load_stage_data(cfg, stage):
    settings = stage_settings(cfg, stage)
    directory = Path(cfg["_baseline_root"]) / settings["data_dir"]
    x, y, metadata = load_data(directory)
    splits = select_splits(metadata, settings)
    for split, expected in settings["expected_records"].items():
        if len(splits[split]) != expected:
            raise ValueError(
                f"{stage}/{split} changed: {len(splits[split])} != {expected}"
            )
        positive = np.asarray(y[splits[split]]).sum(axis=0)
        if np.any(positive == 0) or np.any(positive == len(splits[split])):
            raise ValueError(f"All five classes need both labels in {stage}/{split}")
    scale_path = Path(cfg["_baseline_root"]) / cfg["scale_reference_checkpoint"]
    reference = torch.load(scale_path, map_location="cpu", weights_only=False)
    scale = float(reference["scale_mv"])
    if not np.isclose(scale, cfg["expected_scale_mv"], rtol=0, atol=1e-10):
        raise ValueError(
            "Full phase-one normalization scalar differs from the specification"
        )
    scale_source = file_info(scale_path)
    identity = {
        "phase1_cache": _data_identity(directory, metadata, y),
        "split_sha256": {
            name: hashlib.sha256(
                np.asarray(indices, dtype=np.int64).tobytes()
            ).hexdigest()
            for name, indices in splits.items()
        },
        "scale_mv": scale,
        "scale_source_sha256": scale_source["sha256"],
        "class_order": list(CLASSES),
        "lead_order": list(LEADS),
    }
    frozen_path = Path(cfg["_results_root"]) / "logs" / "preregistration.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if identity != frozen["datasets"][stage]:
            raise ValueError(
                "Cohort, preprocessing scale or cache identity changed after preregistration"
            )
    return {
        "x": x,
        "y": y,
        "metadata": metadata,
        "splits": splits,
        "scale_mv": scale,
        "identity": identity,
        "scale_source": scale_source,
    }


def _critical_sources(cfg):
    root = Path(cfg["_phase2_root"])
    baseline = Path(cfg["_baseline_root"])
    return [
        root / "src" / name
        for name in ("common.py", "generate_phase2_noise.py", "train_phase2.py")
    ] + [
        baseline / "src" / name
        for name in (
            "datasets.py",
            "models.py",
            "lead_matrix.py",
            "noise_generators.py",
            "covariance_matching.py",
            "train.py",
        )
    ]


def _protected_baseline_files(cfg):
    baseline = Path(cfg["_baseline_root"])
    files = {
        baseline / name
        for name in ("README.md", "data/README.md", "requirements.txt")
        if (baseline / name).is_file()
    }
    for relative in (
        "configs",
        "src",
        "scripts",
        "reports",
        "data/processed",
        "data/processed_pilot",
        "data/processed_smoke",
        "results/checkpoints",
        "results/metrics",
        "results/tables",
        "results/figures",
        "results/logs",
    ):
        directory = baseline / relative
        if directory.exists():
            files.update(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            )
    return sorted(files)


def preregister(cfg):
    directory = Path(cfg["_results_root"]) / "logs"
    destination = directory / "preregistration.json"
    if destination.exists():
        require_preregistration(cfg)
        return json.loads(destination.read_text(encoding="utf-8"))
    root = Path(cfg["_phase2_root"])
    specification = (root / cfg["specification"]).resolve()
    for stage in cfg["stages"]:
        if any(stage_paths(cfg, stage)["checkpoints"].rglob("*.pt")):
            raise ValueError(
                "Cannot retrospectively preregister after training checkpoints exist"
            )
    pilot = load_stage_data(cfg, "pilot")
    full = load_stage_data(cfg, "full")
    source_files = [file_info(path) for path in sorted((root / "src").glob("*.py"))]
    critical = [file_info(path) for path in _critical_sources(cfg)]
    protected = [file_info(path) for path in _protected_baseline_files(cfg)]
    save_json(
        directory / "phase1_preservation.json",
        {"status": "baseline_saved", "files": protected},
    )
    value = {
        "status": "frozen_before_pilot_and_full_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": cfg["_config_sha256"],
        "config": {key: value for key, value in cfg.items() if not key.startswith("_")},
        "configuration_files": [
            file_info(cfg["_config_path"]),
            file_info(root / cfg["train"]["combo_file"]),
            file_info(root / cfg["test"]["combo_file"]),
        ],
        "specification": file_info(specification),
        "matrix": matrix_provenance(),
        "critical_sources": critical,
        "initial_source_files": source_files,
        "train_combos": cfg["_train_combos"],
        "heldout_combos": cfg["_heldout_combos"],
        "combo_intersection": [],
        "noise_base_ranges_disjoint": True,
        "unseen_snr_intersection_with_training": [],
        "datasets": {"pilot": pilot["identity"], "full": full["identity"]},
        "stage_one_protected_files": len(protected),
        "counts": {
            "pilot_training_runs": len(expected_runs(cfg, "pilot")),
            "full_training_runs": len(expected_runs(cfg, "full")),
        },
        "interpretation": {
            "primary": "Equal-condition mean of per-checkpoint joint-unseen Macro-AUROC retention; six paired training-seed t tests with Holm. Patient intervals are separate fixed-model/fixed-noise conditional intervals.",
            "clean_cost": "No clinical utility margin is invented. Report magnitude, direction, uncertainty and tradeoffs rather than automatically declaring acceptability.",
            "cache": "Lossless factorized float32 representation; every SNR-specific normalized input is formed and hashed once during cache creation, then identically reconstructed and shared across all checkpoints.",
            "pilot": "Technical gate only; full settings are frozen before pilot test access. Single-seed pilot cannot select the winning strategy.",
            "optional": cfg["optional"],
        },
    }
    save_json(destination, value)
    return value


def require_preregistration(cfg):
    path = Path(cfg["_results_root"]) / "logs" / "preregistration.json"
    if not path.exists():
        raise ValueError("Run phase2.src.common --preregister before any experiment")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("config_sha256") != cfg["_config_sha256"]:
        raise ValueError(
            "Scientific configuration differs from the preregistered experiment"
        )
    for item in (
        saved["configuration_files"]
        + [saved["specification"]]
        + saved["critical_sources"]
    ):
        if sha256(resolve_path(cfg, item["path"])) != item["sha256"]:
            raise ValueError(f"Frozen input/source changed: {item['path']}")
    return saved


def verify_training_complete(cfg, stage):
    require_preregistration(cfg)
    completed = []
    for strategy, model, seed in expected_runs(cfg, stage):
        path = run_directory(cfg, stage, strategy, model, seed) / "summary.json"
        if not path.exists():
            raise ValueError(f"Training is not complete: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "strategy": strategy,
            "model": model,
            "seed": seed,
            "stage": stage,
            "config_sha256": cfg["_config_sha256"],
            "training_epochs_completed": cfg["stages"][stage]["epochs"],
        }
        if not summary.get("completed") or any(
            summary.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"Incomplete or mismatched training summary: {path}")
        checkpoint = checkpoint_path(cfg, stage, strategy, model, seed)
        info = file_info(checkpoint)
        if summary.get("checkpoint_sha256") != info["sha256"]:
            raise ValueError(
                f"Checkpoint fingerprint differs from completed training: {checkpoint}"
            )
        completed.append(
            {**summary, "checkpoint": info["path"], "checkpoint_sha256": info["sha256"]}
        )
    return completed


def verify_preservation(cfg):
    path = Path(cfg["_results_root"]) / "logs" / "phase1_preservation.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    for item in saved["files"]:
        current = file_info(resolve_path(cfg, item["path"]))
        if current != item:
            raise ValueError(f"Protected phase-one artifact changed: {item['path']}")
    result = {
        "status": "passed",
        "unchanged_files": len(saved["files"]),
        "baseline_manifest": file_info(path),
    }
    save_json(
        Path(cfg["_results_root"]) / "logs" / "phase1_preservation_verified.json",
        result,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="phase2/configs/phase2_main.yaml")
    parser.add_argument("--preregister", action="store_true")
    parser.add_argument("--verify-preservation", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.preregister:
        print(json.dumps(preregister(cfg)["counts"]), flush=True)
    elif args.verify_preservation:
        print(json.dumps(verify_preservation(cfg)), flush=True)
    else:
        print(
            json.dumps(
                {
                    "config_sha256": cfg["_config_sha256"],
                    "pilot_runs": len(expected_runs(cfg, "pilot")),
                    "full_runs": len(expected_runs(cfg, "full")),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
