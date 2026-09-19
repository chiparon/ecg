"""Read-only baseline reuse and portable signed-control contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from methodology_supplement.common import (
    WORKSPACE, array_sha256, file_info, read_json, resolve_path,
    save_json, sha256, summarize_distribution, write_csv,
)

DEFAULT_CONFIG = WORKSPACE / "signed_control_mechanism/configs/signed_control_full.json"
METRICS = ("macro_auroc", "macro_f1", "ece")
CONTRASTS = ("E-S", "S-I", "E-I")


def load_config(path=None):
    path = resolve_path(path or DEFAULT_CONFIG).resolve()
    cfg = read_json(path)
    if cfg.get("experiment") != "signed_control_mechanism":
        raise ValueError("Not the frozen signed-control protocol")
    cfg["_config_sha256"] = hashlib.sha256(json.dumps(
        cfg, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    cfg["_config_path"] = str(path)
    return cfg


def stage_paths(cfg, stage):
    if stage not in cfg["stages"]:
        raise ValueError(f"Unknown stage: {stage}")
    root = resolve_path(cfg["results_dir"])
    paths = {key: root / key / stage for key in
             ("inputs", "predictions", "tables", "figures", "logs", "bootstrap")}
    paths["reports"] = root / "reports"
    return paths


def check_info(info):
    path = resolve_path(info["path"])
    if not path.is_file() or path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
        raise ValueError(f"Frozen file identity changed: {path}")
    return path


def require_freeze(cfg, stage):
    freeze = read_json(stage_paths(cfg, stage)["logs"] / "freeze.json")
    if freeze.get("status") != "completed" or freeze.get("config_sha256") != cfg["_config_sha256"]:
        raise ValueError("Stage does not have a completed current-protocol freeze")
    for info in freeze["source_files"]:
        check_info(info)
    for key in ("sign_controls", "matrices"):
        check_info(freeze[key])
    return freeze


def load_reference(cfg, stage):
    freeze = require_freeze(cfg, stage)
    with np.load(check_info(freeze["cohort"]), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def load_checkpoints(cfg, stage):
    return require_freeze(cfg, stage)["checkpoint_entries"]


def legacy_paths(cfg, stage):
    root = resolve_path(cfg["baseline_results_dir"])
    return {key: root / ("test_inputs" if key == "inputs" else key) / stage
            for key in ("inputs", "predictions", "tables", "logs", "bootstrap")}


def load_manifest(cfg, stage):
    manifest = read_json(stage_paths(cfg, stage)["inputs"] / "manifest.json")
    if manifest.get("status") != "completed" or manifest.get("config_sha256") != cfg["_config_sha256"] or manifest.get("stage") != stage:
        raise ValueError("Input manifest is missing, incomplete, or stale")
    return manifest


def case_grid(cfg, stage):
    freeze = require_freeze(cfg, stage)
    legacy = read_json(check_info(freeze["legacy_input_manifest"]))
    lookup = {(int(c["snr"]), int(c["noise_seed"]), c["condition"]): c
              for c in legacy["cases"] if c.get("source_id") == "standard"}
    controls = read_json(check_info(freeze["sign_controls"]))["controls"]
    signs = {item["mode"]: item["sign_vector"] for item in controls}
    result = []
    for snr in cfg["snrs"]:
        for noise_seed in cfg["phase1_noise_seeds"]:
            for condition in cfg["conditions"]:
                source_condition = "independent_rms" if condition == "I" else "electrode"
                original = lookup[(snr, noise_seed, source_condition)]
                result.append({
                    "case_id": f"{condition}__snr_{snr}__noise_{noise_seed}",
                    "condition": condition, "snr": snr, "noise_seed": noise_seed,
                    "mode": condition if condition.startswith("S_") else "",
                    "sign_vector": signs.get(condition, [1] * 12),
                    "noise_path": original["noise_path"], "noise_sha256": original["noise_sha256"],
                    "baseline_case_id": original["case_id"],
                    "baseline_input_sha256": original["input_sha256"],
                    "reuse_eligible": condition in ("E", "I"),
                })
    if len(result) != 126 or len({case["case_id"] for case in result}) != 126:
        raise ValueError("Signed-control grid must contain exactly 126 unique noisy cases")
    return result


def scaled_noise(base, case, out=None):
    """Sign the original base, then apply E's unmodified float32 SNR factor."""
    base = np.asarray(base)
    if base.dtype != np.float32 or base.ndim != 3 or base.shape[1:] != (12, 1000):
        raise ValueError("Expected frozen float32 record x lead x time noise")
    if out is None:
        out = np.empty_like(base)
    if case["condition"].startswith("S_"):
        signs = np.asarray(case["sign_vector"], dtype=np.float32)
        if signs.shape != (12,) or not np.isin(signs, (-1, 1)).all():
            raise ValueError("Invalid frozen sign vector")
        np.multiply(base, signs[None, :, None], out=out)
        np.multiply(out, np.float32(10 ** (-float(case["snr"]) / 20)), out=out)
    else:
        np.multiply(base, np.float32(10 ** (-float(case["snr"]) / 20)), out=out)
    return out


def materialize_input(clean, base, case, out=None):
    if clean.dtype != np.float32 or clean.shape != base.shape:
        raise ValueError("Clean/noise shape or precision mismatch")
    out = scaled_noise(base, case, out)
    np.add(clean, out, out=out)
    return out
