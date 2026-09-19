"""Preregistered four-strategy training; selection and calibration use clean fold 9 only."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from phase1_ecg_robustness.src.models import build_model
from phase1_ecg_robustness.src.train import (
    CLASSES,
    _CleanDataset,
    _atomic_checkpoint,
    _environment,
    _rng_state,
    _seed_worker,
    _validate,
    _write_epochs,
    f1_thresholds,
    seed_everything,
    validation_metrics,
)
from .common import (
    checkpoint_path,
    expected_runs,
    file_info,
    load_config,
    load_stage_data,
    require_preregistration,
    resolve_path,
    run_directory,
    save_json,
    stage_settings,
)
from .generate_phase2_noise import training_batch

SELECTION = "clean_validation_macro_auroc"
AUDIT_DTYPES = {
    "input_type": np.uint8,
    "combo_index": np.int16,
    "snr_db": np.float32,
    "actual_snr_db": np.float64,
    "noise_seed_words": np.uint32,
    "rms_match_relative_error": np.float64,
}


class _IndexedMVData(_CleanDataset):
    """Retain baseline sample copying and loader semantics, but augment before scaling."""

    def __init__(self, signals, labels, indices):
        super().__init__(signals, labels, indices, scale_mv=1.0)

    def __getitem__(self, position):
        signal, label = super().__getitem__(position)
        return signal, label, position


def _array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _reference_orders(indices, seed, settings):
    """Replay the actual baseline DataLoader, including its iterator seed draws."""
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        np.arange(len(indices), dtype=np.int64),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    return [
        _array_hash(indices[np.concatenate([batch.numpy() for batch in loader])])
        for _ in range(int(settings["epochs"]))
    ]


def _execution_sources(cfg):
    baseline = Path(cfg["_baseline_root"]) / "src"
    package = Path(__file__).resolve().parent
    return [
        file_info(path)
        for path in (
            Path(__file__).resolve(),
            package / "common.py",
            package / "generate_phase2_noise.py",
            baseline / "train.py",
            baseline / "models.py",
            baseline / "datasets.py",
            baseline / "noise_generators.py",
            baseline / "lead_matrix.py",
            baseline / "audit_noise.py",
            baseline / "covariance_matching.py",
        )
    ]


def _check_artifact(cfg, descriptor):
    path = resolve_path(cfg, descriptor["path"])
    if not path.is_file() or file_info(path) != descriptor:
        raise ValueError(f"Missing or changed completed-run artifact: {path}")


def _reuse_completed(cfg, summary_path, best_path, expected, orders):
    if not summary_path.exists():
        if best_path.exists():
            raise ValueError(f"Checkpoint lacks its run summary: {best_path}")
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"Refusing stale run: {summary_path}; mismatch in {key}")
    if not summary.get("completed"):
        # Interrupted runs restart deterministically; no partial run is ever reused.
        return None
    epochs = int(expected["train_config"]["epochs"])
    if summary.get("training_epochs_completed") != epochs:
        raise ValueError("Completed summary has an incomplete epoch budget")
    if summary.get("epoch_order_sha256") != orders:
        raise ValueError("Completed summary has different data-order fingerprints")
    artifacts = summary.get("epoch_artifacts", [])
    if len(artifacts) != epochs:
        raise ValueError("Completed summary does not retain every epoch artifact")
    for epoch, artifact in enumerate(artifacts, 1):
        if artifact["epoch"] != epoch or artifact["order_sha256"] != orders[epoch - 1]:
            raise ValueError("Epoch artifact identity/order mismatch")
        for field in ("checkpoint", "augmentation"):
            _check_artifact(cfg, artifact[field])
    for field in ("validation_predictions", "epoch_log", "environment_artifact"):
        _check_artifact(cfg, summary[field])
    if summary.get("checkpoint") != file_info(best_path)["path"]:
        raise ValueError("Completed summary points to an unexpected checkpoint")
    if summary.get("checkpoint_sha256") != file_info(best_path)["sha256"]:
        raise ValueError("Completed best checkpoint checksum mismatch")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    for key, value in expected.items():
        checkpoint_key = "model_name" if key == "model" else key
        if best.get(checkpoint_key) != value:
            raise ValueError(f"Completed checkpoint mismatch in {checkpoint_key}")
    if (
        best.get("completed") is not True
        or best.get("training_epochs_completed") != epochs
        or best.get("calibration_pending") is not False
        or best.get("threshold_tuning_count") != 1
        or best.get("threshold_comparison") != ">="
        or best.get("best_epoch") != summary.get("best_epoch")
        or best.get("epoch") != summary.get("best_epoch")
        or best.get("epoch_order_sha256") != orders
        or best.get("thresholds") != summary.get("thresholds")
        or best.get("threshold_notes") != summary.get("threshold_notes")
        or best.get("validation_predictions") != summary["validation_predictions"]
    ):
        raise ValueError(
            "Best checkpoint is not the finalized, calibrated completed run"
        )
    thresholds = np.asarray(best["thresholds"], dtype=np.float64)
    if thresholds.shape != (len(CLASSES),) or not np.isfinite(thresholds).all():
        raise ValueError("Completed checkpoint thresholds are invalid")
    return summary


def _audit_batch(cfg, strategy, original, augmented, audit):
    n = len(original)
    if augmented.dtype != np.float32 or augmented.shape != original.shape:
        raise ValueError("Training augmentation must retain float32 ECG shape")
    if not np.isfinite(augmented).all():
        raise FloatingPointError("Nonfinite augmented training ECG")
    for name, dtype in AUDIT_DTYPES.items():
        values = np.asarray(audit[name])
        shape = (n, 4) if name == "noise_seed_words" else (n,)
        if values.shape != shape or values.dtype != np.dtype(dtype):
            raise ValueError(
                f"Training audit {name} must be {dtype} with shape {shape}"
            )
    types = audit["input_type"]
    if not np.isin(types, (0, 1, 2)).all():
        raise ValueError(
            "A record must activate exactly one clean/independent/electrode branch"
        )
    for code, name in enumerate(("clean", "independent_rms", "electrode")):
        if cfg["strategies"][strategy][name] == 0 and np.any(types == code):
            raise ValueError(f"Forbidden augmentation branch {name} for {strategy}")
    clean, noisy = types == 0, types != 0
    if not np.array_equal(augmented[clean], original[clean]):
        raise ValueError("Augmentation changed clean-branch ECGs")
    combos = audit["combo_index"]
    if not np.all(combos[clean] == -1) or not np.all(
        (combos[noisy] >= 0) & (combos[noisy] < len(cfg["_train_combos"]))
    ):
        raise ValueError("Training used a nonregistered electrode target combination")
    if (
        not np.isnan(audit["snr_db"][clean]).all()
        or not np.isin(audit["snr_db"][noisy], cfg["train"]["snrs"]).all()
    ):
        raise ValueError("Training used a nonregistered SNR")
    if (
        not np.isnan(audit["actual_snr_db"][clean]).all()
        or not np.isfinite(audit["actual_snr_db"][noisy]).all()
    ):
        raise ValueError("Missing actual-strength diagnostics")
    independent = types == 1
    errors = audit["rms_match_relative_error"]
    if (
        not np.isfinite(errors[independent]).all()
        or not np.isnan(errors[~independent]).all()
    ):
        raise ValueError(
            "RMS-match diagnostics do not identify independent-only inputs"
        )
    noise_base = int(audit["noise_base"])
    low, high = cfg["train"]["noise_base_range"]
    if noise_base != cfg["train"]["noise_base"] or not low <= noise_base <= high:
        raise ValueError("Training noise base violated the registered train-only range")


def _diagnostic_stats(values, prefix):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        f"{prefix}_count": len(finite),
        f"{prefix}_mean": float(finite.mean()) if len(finite) else None,
        f"{prefix}_std": float(finite.std()) if len(finite) else None,
        f"{prefix}_min": float(finite.min()) if len(finite) else None,
        f"{prefix}_max": float(finite.max()) if len(finite) else None,
    }


def _audit_row(cfg, audit):
    types = audit["input_type"]
    result = {}
    for code, name in enumerate(("clean", "independent", "electrode")):
        count = int(np.count_nonzero(types == code))
        result[f"{name}_count"] = count
        result[f"{name}_fraction"] = count / len(types)
    for snr in cfg["train"]["snrs"]:
        result[f"snr_{snr}_count"] = int(np.count_nonzero(audit["snr_db"] == snr))
    for index, combo in enumerate(cfg["_train_combos"]):
        result[f"combo_{combo['combo_id']}_count"] = int(
            np.count_nonzero(audit["combo_index"] == index)
        )
    result.update(_diagnostic_stats(audit["actual_snr_db"], "actual_snr_db"))
    result.update(
        _diagnostic_stats(
            audit["actual_snr_db"] - audit["snr_db"], "actual_snr_error_db"
        )
    )
    result.update(
        _diagnostic_stats(audit["rms_match_relative_error"], "rms_match_relative_error")
    )
    return result


def train_one(cfg, stage, strategy, model_name, seed, data, settings, device):
    """Run the full fixed epoch budget, then calibrate the single clean-selected model."""
    seed_everything(seed)
    model_kwargs = {"base_channels": int(settings["base_channels"])}
    if model_name == "tcn":
        model_kwargs["dropout"] = float(settings["tcn_dropout"])
    model = build_model(model_name, **model_kwargs)
    initialization = _model_hash(model)
    indices = {
        key: np.asarray(value, dtype=np.int64) for key, value in data["splits"].items()
    }
    index_fields = {f"{key}_indices": value.tolist() for key, value in indices.items()}
    train_indices, val_indices = indices["train"], indices["val"]
    n = len(train_indices)
    epochs = int(settings["epochs"])
    expected_orders = _reference_orders(train_indices, seed, settings)
    best_path = checkpoint_path(cfg, stage, strategy, model_name, seed)
    log_dir = run_directory(cfg, stage, strategy, model_name, seed)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.json"
    expected = {
        "config_sha256": cfg["_config_sha256"],
        "matrix_sha256": cfg["matrix_sha256"],
        "stage": stage,
        "strategy": strategy,
        "model": model_name,
        "seed": seed,
        "model_kwargs": model_kwargs,
        "train_config": settings,
        "data_identity": data["identity"],
        "scale_mv": float(data["scale_mv"]),
        "scale_source": data["scale_source"],
        "class_order": list(CLASSES),
        "model_initialization_sha256": initialization,
        "selection_metric": SELECTION,
        "execution_sources": _execution_sources(cfg),
        **index_fields,
    }
    reused = _reuse_completed(cfg, summary_path, best_path, expected, expected_orders)
    if reused is not None:
        print(
            f"Reusing verified completed {stage}/{strategy}/{model_name}/seed_{seed}",
            flush=True,
        )
        return reused
    started = time.perf_counter()
    environment = _environment(device, settings)
    environment["augmentation"] = {
        "strategy": strategy,
        "probabilities": cfg["strategies"][strategy],
        "rng": "record-keyed local generators, independent of training and loader streams",
    }
    save_json(log_dir / "environment.json", environment)
    summary = {
        **expected,
        "completed": False,
        "training_epochs_completed": 0,
        "checkpoint": best_path.resolve()
        .relative_to(Path(cfg["_workspace_root"]))
        .as_posix(),
        "best_epoch": None,
        "thresholds": None,
        "threshold_tuning_count": 0,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epoch_artifacts": [],
        "epoch_order_sha256": [],
        "environment": environment,
    }
    save_json(summary_path, summary)
    generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    loader_args = {
        "batch_size": int(settings["batch_size"]),
        "num_workers": int(settings["num_workers"]),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        _IndexedMVData(data["x"], data["y"], train_indices),
        shuffle=True,
        generator=generator,
        drop_last=False,
        **loader_args,
    )
    validation_loader = DataLoader(
        _CleanDataset(data["x"], data["y"], val_indices, data["scale_mv"]),
        shuffle=False,
        generator=validation_generator,
        **loader_args,
    )
    # Five-class AUROC is the registered selection rule; never silently switch to loss.
    val_y = np.asarray(data["y"][val_indices])
    positives = val_y.sum(axis=0)
    if np.any((positives == 0) | (positives == len(val_y))):
        raise ValueError(
            "Clean validation must contain both outcomes in every registered class"
        )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["lr"]),
        weight_decay=float(settings["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    ecg_ids = data["metadata"].ecg_id.to_numpy(dtype=np.int64)
    patient_ids = data["metadata"].patient_id.to_numpy()
    if patient_ids.dtype.hasobject:
        patient_ids = patient_ids.astype(str)
    rows, artifacts, order_hashes = [], [], []
    best_score, best_epoch = -float("inf"), None
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss, cursor = 0.0, 0
        exposure = np.zeros(n, dtype=np.uint8)
        order = np.empty(n, dtype=np.int64)
        audit_arrays = {
            name: np.empty((n, 4) if name == "noise_seed_words" else (n,), dtype=dtype)
            for name, dtype in AUDIT_DTYPES.items()
        }
        for batch_signals, batch_labels, positions in train_loader:
            positions = positions.numpy()
            batch_indices = train_indices[positions]
            if len(np.unique(positions)) != len(positions) or np.any(
                exposure[positions]
            ):
                raise ValueError(
                    "Training loader exposed a record more than once in an epoch"
                )
            exposure[positions] += 1
            size = len(positions)
            order[cursor : cursor + size] = batch_indices
            original = batch_signals.numpy()
            original.flags.writeable = False
            augmented, audit = training_batch(
                cfg, stage, strategy, seed, epoch, original, ecg_ids[batch_indices]
            )
            _audit_batch(cfg, strategy, original, augmented, audit)
            if not np.array_equal(batch_labels.numpy(), data["y"][batch_indices]):
                raise ValueError("Training labels differ from their original records")
            for name in AUDIT_DTYPES:
                audit_arrays[name][cursor : cursor + size] = audit[name]
            cursor += size
            # Out-of-place division owns its storage even when a clean callback returns its input.
            normalized = augmented / np.float32(data["scale_mv"])
            inputs = torch.from_numpy(normalized).to(
                device, non_blocking=device.type == "cuda"
            )
            targets = batch_labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            # Preserve the phase-one finite-gradient check without gradient clipping.
            squared_norm = torch.zeros((), device=device)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    squared_norm += parameter.grad.detach().float().square().sum()
            if not torch.isfinite(squared_norm):
                raise FloatingPointError(
                    f"Nonfinite training gradients at epoch {epoch}"
                )
            optimizer.step()
            total_loss += float(loss.item()) * size
        if cursor != n or not np.all(exposure == 1):
            raise ValueError(
                "Each original training record must appear exactly once per epoch"
            )
        order_hash = _array_hash(order)
        if order_hash != expected_orders[epoch - 1]:
            raise ValueError(
                "Training order diverged from the strategy-independent baseline loader"
            )
        order_hashes.append(order_hash)
        val_loss, labels, probabilities = _validate(
            model, validation_loader, criterion, device
        )
        if not np.array_equal(labels, val_y):
            raise ValueError("Clean validation labels/order changed")
        metrics = validation_metrics(labels, probabilities)
        score = metrics["macro_auroc"]
        if score is None or not np.isfinite(score):
            raise ValueError(
                "Registered five-class clean-validation AUROC is undefined"
            )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / n,
            "val_loss": val_loss,
            "val_macro_auroc": score,
            "val_macro_ap": metrics["macro_ap"],
            "val_macro_f1_at_0p5": float(
                f1_score(labels, probabilities >= 0.5, average="macro", zero_division=0)
            ),
            **_audit_row(cfg, audit_arrays),
            "exposure_count": n,
            "unique_records": n,
            "exposure_min": int(exposure.min()),
            "exposure_max": int(exposure.max()),
            "order_sha256": order_hash,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        audit_path = log_dir / f"augmentation_epoch_{epoch:02d}.npz"
        _atomic_npz(
            audit_path,
            **audit_arrays,
            indices=order,
            ids=ecg_ids[order],
            exposure_indices=train_indices,
            exposure_counts=exposure,
            epoch=np.asarray(epoch),
            noise_base=np.asarray(cfg["train"]["noise_base"]),
            config_sha256=np.asarray(cfg["_config_sha256"]),
            order_sha256=np.asarray(order_hash),
        )
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        checkpoint = {
            **expected,
            "model_name": model_name,
            "model_state": model.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "training_epochs_completed": epoch,
            "completed": False,
            "calibration_pending": True,
            "thresholds": None,
            "threshold_notes": None,
            "threshold_tuning_count": 0,
            "threshold_comparison": ">=",
            "validation_metrics": metrics,
            "validation_bce": val_loss,
            "validation_f1_at_0p5": row["val_macro_f1_at_0p5"],
            "optimizer_state": optimizer.state_dict(),
            "rng_state": _rng_state(generator),
            "validation_loader_rng_state": validation_generator.get_state(),
            "environment": environment,
            "epoch_order_sha256": list(order_hashes),
            "augmentation_audit": file_info(audit_path),
            "epoch_metrics": row,
        }
        epoch_path = checkpoint_path(
            cfg, stage, strategy, model_name, seed, epoch=epoch
        )
        _atomic_checkpoint(epoch_path, checkpoint)
        if improved:
            _atomic_checkpoint(best_path, checkpoint)
        artifacts.append(
            {
                "epoch": epoch,
                "order_sha256": order_hash,
                "checkpoint": file_info(epoch_path),
                "augmentation": checkpoint["augmentation_audit"],
            }
        )
        _write_epochs(log_dir / "epochs.csv", rows)
        summary.update(
            {
                "training_epochs_completed": epoch,
                "best_epoch": best_epoch,
                "best_selection_score": best_score,
                "last_epoch_metrics": row,
                "epoch_artifacts": artifacts,
                "epoch_order_sha256": list(order_hashes),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        save_json(summary_path, summary)
        print(
            f"{stage}/{strategy}/{model_name}/seed_{seed} epoch={epoch}/{epochs} "
            f"train_loss={row['train_loss']:.5f} val_loss={val_loss:.5f} "
            f"clean_val_auroc={score:.5f} elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state"])
    val_loss, labels, probabilities = _validate(
        model, validation_loader, criterion, device
    )
    thresholds, notes = f1_thresholds(labels, probabilities)
    validation_path = log_dir / "best_checkpoint_clean_validation.npz"
    _atomic_npz(
        validation_path,
        p=probabilities,
        y=labels,
        ids=ecg_ids[val_indices],
        patient_ids=patient_ids[val_indices],
        indices=val_indices,
        thresholds=np.asarray(thresholds, dtype=np.float64),
        config_sha256=np.asarray(cfg["_config_sha256"]),
        matrix_sha256=np.asarray(cfg["matrix_sha256"]),
        model_initialization_sha256=np.asarray(initialization),
        best_epoch=np.asarray(best_epoch),
    )
    best.update(
        {
            "completed": True,
            "training_epochs_completed": epochs,
            "calibration_pending": False,
            "thresholds": thresholds,
            "threshold_notes": notes,
            "threshold_tuning_count": 1,
            "threshold_source": "best_checkpoint_clean_validation_once",
            "validation_predictions": file_info(validation_path),
            "calibration_validation_bce": val_loss,
            "epoch_order_sha256": order_hashes,
            "calibration_validation_loader_rng_state": validation_generator.get_state(),
        }
    )
    _atomic_checkpoint(best_path, best)
    summary.update(
        {
            "completed": True,
            "training_epochs_completed": epochs,
            "thresholds": thresholds,
            "threshold_notes": notes,
            "threshold_tuning_count": 1,
            "checkpoint_sha256": file_info(best_path)["sha256"],
            "validation_predictions": best["validation_predictions"],
            "epoch_log": file_info(log_dir / "epochs.csv"),
            "environment_artifact": file_info(log_dir / "environment.json"),
            "best_validation_metrics": best["validation_metrics"],
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    save_json(summary_path, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument(
        "--strategy", choices=("clean_only", "independent_rms", "electrode", "mixed")
    )
    parser.add_argument("--model", choices=("resnet", "tcn"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    require_preregistration(cfg)
    settings = stage_settings(cfg, args.stage)
    runs = [
        (strategy, model, seed)
        for strategy, model, seed in expected_runs(cfg, args.stage)
        if (args.strategy is None or strategy == args.strategy)
        and (args.model is None or model == args.model)
        and (args.seed is None or seed == args.seed)
    ]
    if not runs:
        raise ValueError(
            "Selections must be members of the unchanged registered stage grid"
        )
    required = {
        "optimizer": "AdamW",
        "scheduler": "none",
        "gradient_clip_norm": None,
        "mixed_precision": False,
        "allow_tf32": False,
        "early_stopping": False,
        "deterministic": True,
        "selection_metric": SELECTION,
        "threshold_source": "best_checkpoint_clean_validation_once",
        "epoch_validation_f1_threshold": 0.5,
    }
    for key, value in required.items():
        if settings.get(key) != value:
            raise ValueError(f"Registered training semantics require {key}={value!r}")
    torch.set_num_threads(min(int(settings["torch_threads"]), os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    data = load_stage_data(cfg, args.stage)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for strategy, model, seed in runs:
        train_one(cfg, args.stage, strategy, model, int(seed), data, settings, device)


if __name__ == "__main__":
    main()
