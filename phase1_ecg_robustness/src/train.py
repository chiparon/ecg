"""Clean-only ECG training, validation-only selection, and durable epoch progress.

Run from the project root: python -m src.train --config configs/pilot.yaml
Incomplete runs restart deterministically; --resume-completed skips only verified
completed runs. Last checkpoints include optimizer and RNG state for inspection.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from .datasets import load_data, select_splits, _data_identity
from .models import build_model

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
DEFAULT_TRAIN = {
    "epochs": 12,
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "train_limit": None,
    "val_limit": None,
    "test_limit": None,
    "subset_seed": 2026,
    "num_workers": 0,
    "base_channels": 24,
    "torch_threads": 4,
}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_epochs(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            config, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.set_num_threads(1)


class _CleanDataset(Dataset):
    def __init__(self, signals, labels, indices, scale_mv: float):
        self.signals = signals
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.scale_mv = scale_mv

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        record = int(self.indices[position])
        # Private sample storage: never mutate prepared mV signals or mmap pages.
        signal = np.array(self.signals[record], dtype=np.float32, copy=True)
        signal /= self.scale_mv
        label = np.array(self.labels[record], dtype=np.float32, copy=True)
        return torch.from_numpy(signal), torch.from_numpy(label)


def training_scale_mv(signals, indices, chunk_size: int = 64) -> float:
    """Scalar RMS over selected training records only, with float64 accumulation."""
    total_squared = 0.0
    count = 0
    for start in range(0, len(indices), chunk_size):
        block = np.asarray(
            signals[np.asarray(indices[start : start + chunk_size])], dtype=np.float64
        )
        if not np.isfinite(block).all():
            raise ValueError("Training signals contain non-finite values")
        total_squared += float(np.einsum("ijk,ijk->", block, block))
        count += block.size
    scale = math.sqrt(total_squared / count) if count else 0.0
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Selected training data must have positive finite RMS in mV")
    return scale


def validation_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Undefined class AUROC/AP stays null; never replace it by chance performance."""
    aucs, aps, positives = [], [], []
    for column in range(len(CLASSES)):
        target = labels[:, column]
        positive = int(target.sum())
        positives.append(positive)
        aucs.append(
            float(roc_auc_score(target, probabilities[:, column]))
            if 0 < positive < len(target)
            else None
        )
        aps.append(
            float(average_precision_score(target, probabilities[:, column]))
            if positive > 0
            else None
        )
    defined_auc = [value for value in aucs if value is not None]
    defined_ap = [value for value in aps if value is not None]
    return {
        "macro_auroc": float(np.mean(aucs)) if len(defined_auc) == 5 else None,
        "macro_ap": float(np.mean(aps)) if len(defined_ap) == 5 else None,
        "macro_auroc_defined_classes": (
            float(np.mean(defined_auc)) if defined_auc else None
        ),
        "macro_ap_defined_classes": float(np.mean(defined_ap)) if defined_ap else None,
        "auroc_class_count": len(defined_auc),
        "ap_class_count": len(defined_ap),
        "per_class_auroc": dict(zip(CLASSES, aucs)),
        "per_class_ap": dict(zip(CLASSES, aps)),
        "validation_positives": dict(zip(CLASSES, positives)),
    }


def f1_thresholds(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[list[float], dict]:
    """Select each threshold on validation alone; decisions use probability >= t."""
    thresholds, notes = [], {}
    for column, name in enumerate(CLASSES):
        target = labels[:, column]
        if np.unique(target).size < 2:
            thresholds.append(0.5)
            notes[name] = "Untuned 0.5: validation lacks positive or negative examples"
            continue
        precision, recall, candidates = precision_recall_curve(
            target, probabilities[:, column]
        )
        scores = np.divide(
            2 * precision[:-1] * recall[:-1],
            precision[:-1] + recall[:-1],
            out=np.zeros_like(precision[:-1]),
            where=(precision[:-1] + recall[:-1]) > 0,
        )
        winners = np.flatnonzero(scores == scores.max())
        selected = winners[np.argmin(np.abs(candidates[winners] - 0.5))]
        thresholds.append(float(candidates[selected]))
        notes[name] = "Validation F1 maximum; ties closest to 0.5"
    return thresholds, notes


def _validate(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    targets, probabilities = [], []
    with torch.inference_mode():
        for signals, labels in loader:
            signals = signals.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            logits = model(signals)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss")
            total_loss += float(loss.item()) * len(labels)
            targets.append(labels.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(targets),
        np.concatenate(probabilities),
    )


def _environment(device, settings) -> dict:
    packages = {}
    for name in ("numpy", "scipy", "pandas", "torch", "wfdb", "PyYAML", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "tf32": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_threads": torch.get_num_threads(),
        "num_workers": int(settings["num_workers"]),
        "augmentation": "none; clean-only BCEWithLogitsLoss",
    }


def _rng_state(generator) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def train_one(
    config,
    model_name,
    seed,
    signals,
    labels,
    splits,
    scale_mv,
    identity,
    settings,
    device,
    resume_completed=False,
) -> dict:
    seed_everything(seed)
    run_name = str(config["run_name"])
    root = Path(config.get("results_dir", "results"))
    checkpoint_dir = root / "checkpoints" / run_name / model_name
    log_dir = root / "logs" / run_name / model_name / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"seed_{seed}.pt"
    last_path = checkpoint_dir / f"seed_{seed}.last.pt"
    summary_path = log_dir / "summary.json"
    fingerprint = _config_hash(config)
    model_kwargs = {"base_channels": int(settings["base_channels"])}
    if model_name == "tcn":
        model_kwargs["dropout"] = float(settings.get("tcn_dropout", 0.1))
    index_fields = {
        f"{key}_indices": np.asarray(value, dtype=np.int64).tolist()
        for key, value in splits.items()
    }
    for existing_path in (best_path, last_path):
        if not existing_path.exists():
            continue
        existing = torch.load(existing_path, map_location="cpu", weights_only=False)
        matches = (
            existing.get("config_sha256") == fingerprint
            and existing.get("data_identity") == identity
            and existing.get("model_name") == model_name
            and existing.get("model_kwargs") == model_kwargs
            and existing.get("seed") == seed
            and existing.get("train_config") == settings
            and all(existing.get(key) == value for key, value in index_fields.items())
        )
        if not matches:
            raise ValueError(
                f"Refusing stale/mismatched run {existing_path}; use a new run_name"
            )
        del existing
    if resume_completed and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("completed"):
            if (
                not best_path.exists()
                or not last_path.exists()
                or summary.get("config_sha256") != fingerprint
                or summary.get("data_identity") != identity
            ):
                raise ValueError(
                    f"Incomplete or mismatched completed-run artifacts: {summary_path}"
                )
            print(
                f"Skipping verified completed run: {model_name} seed={seed}", flush=True
            )
            return summary
    if best_path.exists() or last_path.exists():
        print(
            f"Restarting matching run from epoch 1: {model_name} seed={seed}",
            flush=True,
        )
    started = time.perf_counter()
    environment = _environment(device, settings)
    _atomic_json(log_dir / "environment.json", environment)
    summary = {
        "completed": False,
        "model": model_name,
        "seed": seed,
        "config_sha256": fingerprint,
        "config": config,
        "train_config": settings,
        "data_identity": identity,
        "scale_mv": scale_mv,
        "checkpoint": str(best_path),
        "environment": environment,
    }
    _atomic_json(summary_path, summary)
    generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    loader_args = {
        "batch_size": int(settings["batch_size"]),
        "num_workers": int(settings["num_workers"]),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        _CleanDataset(signals, labels, splits["train"], scale_mv),
        shuffle=True,
        generator=generator,
        **loader_args,
    )
    validation_loader = DataLoader(
        _CleanDataset(signals, labels, splits["val"], scale_mv),
        shuffle=False,
        generator=validation_generator,
        **loader_args,
    )
    model = build_model(model_name, **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["lr"]),
        weight_decay=float(settings["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    positives = np.asarray(labels[splits["val"]]).sum(axis=0)
    deficient = [
        name
        for name, count in zip(CLASSES, positives)
        if count == 0 or count == len(splits["val"])
    ]
    if deficient:
        warnings.warn(
            "Validation class coverage incomplete for "
            + ", ".join(deficient)
            + "; five-class macro AUROC is undefined. Selection uses available-class "
            "macro AUROC, or explicitly validation loss if no AUROC is defined."
        )
    rows, best_score, best_epoch, best_metrics = [], -math.inf, None, None
    epochs = int(settings["epochs"])
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        for batch_signals, batch_labels in train_loader:
            batch_signals = batch_signals.to(device, non_blocking=device.type == "cuda")
            batch_labels = batch_labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_signals)
            loss = criterion(logits, batch_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            # Detect failure before AdamW can corrupt parameters, without gradient clipping.
            squared_norm = torch.zeros((), device=device)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    squared_norm += parameter.grad.detach().float().square().sum()
            if not torch.isfinite(squared_norm):
                raise FloatingPointError(
                    f"Non-finite training gradients at epoch {epoch}"
                )
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_labels)
        val_loss, val_labels, val_probabilities = _validate(
            model, validation_loader, criterion, device
        )
        metrics = validation_metrics(val_labels, val_probabilities)
        thresholds, threshold_notes = f1_thresholds(val_labels, val_probabilities)
        score = metrics["macro_auroc_defined_classes"]
        selection = "macro_auroc" if not deficient else "macro_auroc_defined_classes"
        if score is None:
            score, selection = -val_loss, "negative_validation_loss_no_defined_auroc"
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader.dataset),
            "val_loss": val_loss,
            "val_macro_auroc": metrics["macro_auroc"],
            "val_macro_ap": metrics["macro_ap"],
            "val_macro_auroc_defined_classes": metrics["macro_auroc_defined_classes"],
            "val_macro_ap_defined_classes": metrics["macro_ap_defined_classes"],
            "auroc_class_count": metrics["auroc_class_count"],
            "selection_metric": selection,
            "selection_score": score,
            "epoch_compute_seconds": time.perf_counter() - epoch_started,
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        improved = score > best_score
        if improved:
            best_score, best_epoch, best_metrics = score, epoch, metrics
        checkpoint = {
            "model_state": model.state_dict(),
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "scale_mv": scale_mv,
            "thresholds": thresholds,
            "threshold_notes": threshold_notes,
            "threshold_comparison": ">=",
            "class_order": list(CLASSES),
            "seed": seed,
            "config": config,
            "train_config": settings,
            "config_sha256": fingerprint,
            "data_identity": identity,
            **index_fields,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "selection_metric": selection,
            "validation_metrics": metrics,
            "optimizer_state": optimizer.state_dict(),
            "rng_state": _rng_state(generator),
            "validation_loader_rng_state": validation_generator.get_state(),
            "environment": environment,
            "completed": epoch == epochs,
        }
        if improved:
            _atomic_checkpoint(best_path, checkpoint)
        _atomic_checkpoint(last_path, checkpoint)
        _write_epochs(log_dir / "epochs.csv", rows)
        summary.update(
            {
                "last_epoch": epoch,
                "best_epoch": best_epoch,
                "best_selection_score": best_score,
                "selection_metric": selection,
                "best_validation_metrics": best_metrics,
                "last_epoch_metrics": row,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        _atomic_json(summary_path, summary)
        print(
            f"{run_name}/{model_name}/seed_{seed} epoch={epoch}/{epochs} "
            f"train_loss={row['train_loss']:.5f} val_loss={val_loss:.5f} "
            f"{selection}={score:.5f} elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    best["completed"] = True
    best["training_epochs_completed"] = epochs
    _atomic_checkpoint(best_path, best)
    summary.update(
        {
            "completed": True,
            "elapsed_seconds": time.perf_counter() - started,
            "thresholds": best["thresholds"],
            "threshold_notes": best["threshold_notes"],
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
    )
    _atomic_json(summary_path, summary)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", choices=("resnet", "tcn"))
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Skip completed runs only after checking config/data/split identity",
    )
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not config.get("run_name"):
        raise ValueError("Config must contain a nonempty run_name")
    run_name = str(config["run_name"])
    if run_name in (".", "..") or "/" in run_name or "\\" in run_name:
        raise ValueError("run_name must be a single path component")
    settings = {**DEFAULT_TRAIN, **config.get("train", {})}
    for key in ("epochs", "batch_size", "base_channels", "torch_threads"):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"train.{key} must be a positive integer")
    if int(settings["num_workers"]) < 0:
        raise ValueError("train.num_workers cannot be negative")
    if float(settings["lr"]) <= 0 or float(settings["weight_decay"]) < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative")
    torch.set_num_threads(min(int(settings["torch_threads"]), os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    models = [args.model] if args.model else list(config.get("models", ["resnet"]))
    seeds = (
        [args.seed]
        if args.seed is not None
        else list(config.get("seeds", [17, 29, 43]))
    )
    if not models or any(name not in ("resnet", "tcn") for name in models):
        raise ValueError("models must contain resnet and/or tcn")
    if not seeds or any(
        not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds
    ):
        raise ValueError("seeds must be integers in [0, 2**32)")
    data_dir = Path(config.get("data_dir", "data/processed"))
    signals, labels, metadata = load_data(data_dir)
    splits = select_splits(metadata, settings)
    splits = {
        name: np.asarray(splits[name], dtype=np.int64)
        for name in ("train", "val", "test")
    }
    if any(len(indices) == 0 for indices in splits.values()):
        raise ValueError("Train, validation and fixed test splits must all be nonempty")
    if labels.shape != (len(signals), len(CLASSES)):
        raise ValueError(
            "Expected labels shape (records, 5) in NORM, MI, STTC, CD, HYP order"
        )
    if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be finite binary multi-label targets")
    scale_mv = training_scale_mv(signals, splits["train"])
    identity = _data_identity(data_dir, metadata, labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device={device}; train={len(splits['train'])}, val={len(splits['val'])}, "
        f"test={len(splits['test'])}; training-only scale_mv={scale_mv:.8g}",
        flush=True,
    )
    for model_name in models:
        for seed in seeds:
            train_one(
                config,
                model_name,
                int(seed),
                signals,
                labels,
                splits,
                scale_mv,
                identity,
                settings,
                device,
                args.resume_completed,
            )


if __name__ == "__main__":
    main()
