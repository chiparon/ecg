"""Small portable helpers for the selected, read-only-baseline supplement."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = WORKSPACE / "methodology_supplement/configs/minimal_five.json"


def resolve_path(path):
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE / value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    # Same strict, atomic convention as the frozen phase-one producer.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def file_info(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(WORKSPACE).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_config(path=None):
    path = resolve_path(path or DEFAULT_CONFIG).resolve()
    cfg = read_json(path)
    if cfg.get("experiment") != "methodology_minimal_five":
        raise ValueError("Not the user-selected minimum-five protocol")
    encoded = json.dumps(cfg, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    cfg["_config_sha256"] = hashlib.sha256(encoded).hexdigest()
    cfg["_config_path"] = str(path)
    return cfg


def stage_paths(cfg, stage):
    if stage not in cfg["stages"]:
        raise ValueError(f"Unknown supplement stage: {stage}")
    root = resolve_path(cfg["results_dir"])
    return {name: root / ("test_inputs" if name == "inputs" else name) / stage
            for name in ("inputs", "predictions", "tables", "figures", "logs", "reports", "bootstrap")}


def load_reference(cfg, stage):
    with np.load(stage_paths(cfg, stage)["inputs"] / "cohort.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_checkpoints(cfg, stage):
    manifest = read_json(stage_paths(cfg, stage)["inputs"] / "checkpoints.json")
    if manifest["config_sha256"] != cfg["_config_sha256"]:
        raise ValueError("Checkpoint registry belongs to a different supplement protocol")
    return manifest["checkpoints"]


def require_freeze(cfg, stage):
    record = read_json(stage_paths(cfg, stage)["logs"] / "freeze.json")
    if record.get("status") != "completed" or record.get("config_sha256") != cfg["_config_sha256"]:
        raise ValueError("Current supplement protocol is not frozen")
    for source in record["source_files"]:
        path = resolve_path(source["path"])
        if not path.is_file() or sha256(path) != source["sha256"]:
            raise ValueError(f"Frozen source changed: {source['path']}")
    return record


def summarize_distribution(seed_values, seed_distributions):
    values = np.asarray(seed_values, dtype=np.float64)
    distributions = np.asarray(seed_distributions, dtype=np.float64)
    if values.ndim != 1 or distributions.ndim != 2 or distributions.shape[0] != len(values) or distributions.shape[1] < 2:
        raise ValueError("Expected seed points and seed x (point + patient draws)")
    if not np.allclose(values, distributions[:, 0], atol=2e-7, rtol=0, equal_nan=True):
        raise ValueError("Bootstrap points disagree with seed estimates")
    draws = distributions[:, 1:].mean(axis=0)
    finite = draws[np.isfinite(draws)]
    lower, upper = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
    return {"estimate": float(values.mean()), "seed_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            "seed_n": len(values), "patient_ci_low": float(lower), "patient_ci_high": float(upper),
            "n_bootstrap": len(draws), "n_invalid": int(len(draws) - len(finite))}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if hasattr(rows, "to_csv"):
        rows.to_csv(temporary, index=False)
    else:
        rows = list(rows)
        if not rows:
            raise ValueError("An empty table needs an explicit DataFrame column schema")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)
