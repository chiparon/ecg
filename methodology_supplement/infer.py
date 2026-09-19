"""Frozen phase-one CUDA inference over deterministic, independently owned shards."""
from __future__ import annotations

import os

# This must precede importing torch, including its transitive imports.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import socket
import sys
import tempfile
import time

import numpy as np
import torch

from phase1_ecg_robustness.src.evaluate import classification_metrics, predict
from phase1_ecg_robustness.src.models import build_model
from phase1_ecg_robustness.src.train import seed_everything
from .common import (
    WORKSPACE, array_sha256, file_info, load_checkpoints, load_config,
    load_reference, read_json, require_freeze, resolve_path, save_json, sha256,
    stage_paths, write_csv,
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _verify_file(info):
    path = resolve_path(info["path"])
    _require(path.is_file(), f"Missing frozen input: {path}")
    _require(path.stat().st_size == int(info["bytes"]), f"Input size changed: {path}")
    _require(sha256(path) == info["sha256"], f"Input SHA256 changed: {path}")
    return path


def _verify_infos(value):
    """Diagnostics may be either a file_info or a mapping of file_infos."""
    if isinstance(value, dict) and {"path", "sha256", "bytes"} <= value.keys():
        _verify_file(value)
    elif isinstance(value, dict):
        for item in value.values():
            _verify_infos(item)
    elif isinstance(value, list):
        for item in value:
            _verify_infos(item)
    else:
        raise RuntimeError("Malformed frozen artifact fingerprint")


def _source_fingerprints():
    # Only executing inference dependencies belong in a prediction fingerprint.
    # Presentation/statistics edits must not invalidate already computed predictions.
    paths = {Path(__file__), Path(__file__).with_name("common.py"),
             Path(__file__).with_name("__init__.py")}
    paths.add(WORKSPACE / "methodology_supplement" / "implementation_contract.json")
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__name__", "")
        filename = getattr(module, "__file__", None)
        if filename and name.startswith("phase1_ecg_robustness.") and filename.endswith(".py"):
            paths.add(Path(filename).resolve())
    return [file_info(path) for path in sorted(paths)]


def _execution(cfg, model, shard_count, override):
    execution = cfg["execution"]
    _require(torch.cuda.is_available(), "CUDA is required; CPU fallback is forbidden")
    _require(execution["precision"] == "deterministic_fp32" and
             execution["amp"] is False and execution["tf32"] is False,
             "Only the frozen deterministic FP32 execution policy is supported")
    _require(shard_count == int(execution[f"{model}_workers"]) or override,
             f"{model} requires {execution[f'{model}_workers']} production workers")
    hostname = socket.gethostname()
    gpu = torch.cuda.get_device_name(0)
    expected = execution[f"{model}_hostname"]
    names = {hostname.lower(), socket.getfqdn().lower()}
    names |= {name.split(".")[0] for name in tuple(names)}
    assigned = expected.lower() in names and (model != "tcn" or "4070" in gpu)
    _require(assigned or override,
             f"Wrong host/device for {model}: expected {expected}, got {hostname}/{gpu}; "
             "--verification-only-host-override is nonproduction only")
    _require(os.environ["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8", ":16:8"),
             "CUBLAS_WORKSPACE_CONFIG must support deterministic CUDA")
    torch.set_num_threads(int(execution["torch_threads"]))
    torch.set_default_dtype(torch.float32)
    seed_everything(0)
    packages = {}
    for name in ("numpy", "scipy", "pandas", "torch", "scikit-learn", "PyYAML"):
        packages[name] = importlib.metadata.version(name)
    return {
        "host": hostname, "fqdn": socket.getfqdn(), "assigned_host": expected,
        "host_assignment_verified": assigned, "device": "cuda:0", "device_name": gpu,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "device_total_memory": int(torch.cuda.get_device_properties(0).total_memory),
        "software": {"python": sys.version, "platform": platform.platform(),
                     "packages": packages, "cuda_runtime": torch.version.cuda,
                     "cudnn_version": torch.backends.cudnn.version()},
        "fp32_settings": {
            "default_dtype": str(torch.get_default_dtype()), "amp": False,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "torch_threads": torch.get_num_threads(),
            "batch_size": int(execution["batch_size"]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def _atomic_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _probabilities(p, shape, label):
    _require(p.dtype == np.float32 and p.shape == shape and np.isfinite(p).all()
             and np.all((p >= 0) & (p <= 1)), f"Invalid FP32 probabilities: {label}")


def _resume(path, metadata, reference, thresholds):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            for key, value in metadata.items():
                _require(saved[key].shape == () and saved[key].item() == value,
                         f"Stale prediction metadata {key}: {path}")
            for key in ("y", "ids", "patient_ids", "indices"):
                _require(saved[key].dtype == reference[key].dtype and
                         np.array_equal(saved[key], reference[key]),
                         f"Stale prediction cohort {key}: {path}")
            _require(saved["thresholds"].dtype == thresholds.dtype and
                     np.array_equal(saved["thresholds"], thresholds),
                     f"Changed frozen thresholds: {path}")
            p = saved["p"]
            _probabilities(p, reference["y"].shape, str(path))
            _require(saved["p_sha256"].shape == () and
                     saved["p_sha256"].item() == array_sha256(p),
                     f"Prediction data hash mismatch: {path}")
            return p
    except Exception as exc:
        raise RuntimeError(f"Refusing unverified/stale resume output {path}: {exc}") from exc


def _load_models(entries, cfg, reference, stage):
    states = []
    identities = []
    expected_indices = reference["indices"]
    for entry in entries:
        path = resolve_path(entry["path"])
        _require(sha256(path) == entry["sha256"], f"Checkpoint changed: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        _require(checkpoint.get("completed") is True and
                 checkpoint.get("training_epochs_completed") == checkpoint["train_config"]["epochs"],
                 f"Checkpoint training is incomplete: {path}")
        for name, value in (("model_name", entry["model"]), ("seed", entry["seed"]),
                            ("model_kwargs", entry["model_kwargs"]),
                            ("class_order", cfg["class_order"]), ("threshold_comparison", ">=")):
            _require(checkpoint.get(name) == value, f"Checkpoint {name} mismatch: {path}")
        scale = float(checkpoint["scale_mv"])
        thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float64)
        _require(np.isfinite(scale) and scale > 0 and scale == float(entry["scale_mv"]),
                 f"Invalid frozen normalization: {path}")
        _require(thresholds.shape == (5,) and np.isfinite(thresholds).all() and
                 np.all((thresholds >= 0) & (thresholds <= 1)) and
                 np.array_equal(thresholds, np.asarray(entry["thresholds"], dtype=np.float64)),
                 f"Invalid frozen validation thresholds: {path}")
        test_indices = np.asarray(checkpoint["test_indices"], dtype=np.int64)
        if stage == "full":
            valid_indices = np.array_equal(expected_indices, test_indices)
        else:
            positions = {int(index): i for i, index in enumerate(test_indices)}
            chosen = [positions.get(int(index), -1) for index in expected_indices]
            valid_indices = all(i >= 0 for i in chosen) and all(a < b for a, b in zip(chosen, chosen[1:]))
        _require(valid_indices, f"Cohort is not the frozen checkpoint test cohort: {path}")
        identities.append(checkpoint["data_identity"])
        _require(identities[-1] == identities[0], "Checkpoint dataset identities differ")
        seed_everything(int(entry["seed"]))
        model = build_model(entry["model"], **entry["model_kwargs"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model = model.to(device="cuda:0", dtype=torch.float32).eval()
        model.requires_grad_(False)
        states.append({"model": model, "entry": entry, "scale": scale,
                       "thresholds": thresholds})
        del checkpoint
    return states


def run(args):
    started_at, started = _now(), time.perf_counter()
    print(f"worker-started model={args.model} shard={args.shard_index}/{args.shard_count} "
          f"pid={os.getpid()} host={socket.gethostname()} time={started_at}", flush=True)
    _require(args.shard_count > 0 and 0 <= args.shard_index < args.shard_count,
             "Invalid shard index/count")
    cfg = load_config(args.config)
    paths = stage_paths(cfg, args.stage)
    verification = args.case_ids is not None or args.verification_only_host_override
    case_filter = None
    if args.case_ids is not None:
        case_filter = [item.strip() for item in args.case_ids.split(",")]
        _require(all(case_filter) and len(set(case_filter)) == len(case_filter),
                 "--case-ids requires a nonempty, unique comma-list")
        case_filter = sorted(case_filter)
    suffix = f"{args.model}_{args.shard_index}"
    if verification:
        verification_id = _digest({"cases": case_filter, "override": args.verification_only_host_override,
                                   "shards": args.shard_count, "model": args.model,
                                   "shard": args.shard_index})[:16]
        suffix = f"verification_{suffix}_{verification_id}"
        prediction_root = paths["predictions"] / "verification" / verification_id
    else:
        prediction_root = paths["predictions"]
    for key in ("logs", "tables", "predictions"):
        paths[key].mkdir(parents=True, exist_ok=True)
    report_path = paths["logs"] / f"inference_{suffix}.json"
    ledger_path = paths["tables"] / f"evaluation_{suffix}.csv"
    report = {
        "status": "running", "stage": args.stage, "config_sha256": cfg["_config_sha256"],
        "model": args.model, "shard_index": args.shard_index, "shard_count": args.shard_count,
        "case_filter": case_filter, "verification_only": verification,
        "verification_only_host_override": args.verification_only_host_override,
        "pid": os.getpid(), "started_at": started_at, "finished_at": None,
        "expected_cells": None, "observed_cells": 0, "resumed_cells": 0,
        "internal_clean_prediction_hashes": {},
    }
    # A second process for the same shard must not replace the first one's outputs.
    lock_path = report_path.with_suffix(".lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(lock_fd)
    rows = []
    try:
        report.update(_execution(cfg, args.model, args.shard_count, args.verification_only_host_override))
        save_json(report_path, report)
        freeze = require_freeze(cfg, args.stage)
        manifest_path = paths["inputs"] / "manifest.json"
        manifest = read_json(manifest_path)
        _require(manifest["status"] == "completed" and manifest["stage"] == args.stage and
                 manifest["config_sha256"] == cfg["_config_sha256"], "Invalid input manifest")
        for key in ("cohort", "clean", "matrices", "diagnostics"):
            _verify_infos(manifest[key])
        _require(resolve_path(manifest["cohort"]["path"]) == (paths["inputs"] / "cohort.npz").resolve(),
                 "Manifest cohort path differs from the reference loader")
        reference = load_reference(cfg, args.stage)
        reference = {key: np.asarray(value) for key, value in reference.items()}
        n = int(manifest["n_records"])
        _require(reference["y"].shape == (n, 5) and reference["y"].dtype == np.float32,
                 "Invalid cohort labels")
        for key in ("ids", "patient_ids", "indices"):
            _require(reference[key].shape == (n,) and reference[key].dtype.kind != "O",
                     f"Invalid cohort {key}")
        _require(reference["ids"].dtype == np.int64 and reference["indices"].dtype == np.int64
                 and len(np.unique(reference["ids"])) == n, "Invalid ECG identities")
        _require(np.isin(reference["y"], [0, 1]).all(), "Nonbinary cohort labels")
        clean = np.load(resolve_path(manifest["clean"]["path"]), allow_pickle=False)
        _require(clean.dtype == np.float32 and clean.shape == (n, 12, cfg["sequence_length"])
                 and clean.flags.c_contiguous and np.isfinite(clean).all(), "Invalid clean input array")
        cases = manifest["cases"]
        case_ids = [case["case_id"] for case in cases]
        _require(len(cases) == manifest["n_cases"] and len(set(case_ids)) == len(cases)
                 and all(re.fullmatch(r"[A-Za-z0-9_.-]+", item) and item not in (".", "..")
                         for item in case_ids), "Invalid/duplicate case identities")
        _require(cases and cases[0]["kind"] == "clean" and
                 sum(case["kind"] == "clean" for case in cases) == 1, "Clean must be first and unique")
        if case_filter is not None:
            _require(set(case_filter) <= set(case_ids), "Unknown verification case ID")
        owned = [case for i, case in enumerate(cases) if i % args.shard_count == args.shard_index
                 and (case_filter is None or case["case_id"] in case_filter)]
        _require(bool(owned), "No selected cases belong to this shard")
        all_entries = load_checkpoints(cfg, args.stage)
        entries = [entry for entry in all_entries if entry["model"] == args.model]
        seeds = cfg["phase1_training_seeds"]
        _require(len(entries) == len(seeds) and sorted(entry["seed"] for entry in entries) == sorted(seeds),
                 "Checkpoint inventory must contain exactly all three training seeds")
        entries.sort(key=lambda entry: seeds.index(entry["seed"]))
        sources = _source_fingerprints()
        report.update({
            "expected_cells": len(owned) * len(entries), "owned_case_ids": [c["case_id"] for c in owned],
            "source_fingerprints": sources, "checkpoint_fingerprints": entries,
            "manifest_fingerprint": file_info(manifest_path),
            "freeze_fingerprint": file_info(paths["logs"] / "freeze.json"),
            "checkpoint_manifest_fingerprint": file_info(paths["inputs"] / "checkpoints.json"),
            "cohort_fingerprint": manifest["cohort"], "clean_fingerprint": manifest["clean"],
        })
        run_fingerprint = _digest({key: report[key] for key in (
            "stage", "config_sha256", "source_fingerprints", "manifest_fingerprint",
            "freeze_fingerprint", "checkpoint_manifest_fingerprint", "checkpoint_fingerprints",
            "software", "fp32_settings", "device_name", "verification_only")})
        report["run_fingerprint"] = run_fingerprint
        save_json(report_path, report)
        states = _load_models(entries, cfg, reference, args.stage)
        clean_hash = array_sha256(clean)
        _require(clean_hash == cases[0]["input_sha256"], "Actual clean input SHA256 mismatch")
        batch_size = int(cfg["execution"]["batch_size"])
        for state in states:
            state["clean_p"] = predict(state["model"], clean, state["scale"], batch_size, "cuda:0")
            _probabilities(state["clean_p"], reference["y"].shape, "internal clean")
            seed = state["entry"]["seed"]
            report["internal_clean_prediction_hashes"][str(seed)] = array_sha256(state["clean_p"])
            print(f"clean-reference model={args.model} seed={seed} "
                  f"p_sha256={report['internal_clean_prediction_hashes'][str(seed)]}", flush=True)
        save_json(report_path, report)
        buffer = np.empty(clean.shape, dtype=np.float32) if any(c["kind"] != "clean" for c in owned) else None
        noise_hashes = {}
        for case in owned:
            if case["kind"] == "clean":
                x = clean
            else:
                _require(case["kind"] == "bandpass" and
                         case["condition"] in ("electrode", "independent_rms"), "Unknown case kind/condition")
                noise_path = resolve_path(case["noise_path"])
                if noise_path not in noise_hashes:
                    noise_hashes[noise_path] = sha256(noise_path)
                _require(noise_hashes[noise_path] == case["noise_sha256"], f"Noise cache changed: {noise_path}")
                noise = np.load(noise_path, mmap_mode="r", allow_pickle=False)
                _require(noise.dtype == np.float32 and noise.shape == clean.shape and noise.flags.c_contiguous,
                         f"Invalid noise cache: {noise_path}")
                factor = np.float32(10 ** (-float(case["snr"]) / 20))
                np.multiply(noise, factor, out=buffer)
                np.add(clean, buffer, out=buffer)
                del noise
                x = buffer
            input_hash = array_sha256(x)
            _require(input_hash == case["input_sha256"] and np.isfinite(x).all(),
                     f"Actual final FP32 input SHA256 mismatch: {case['case_id']}")
            for state in states:
                entry, thresholds = state["entry"], state["thresholds"]
                path = prediction_root / args.model / f"seed_{entry['seed']}" / f"{case['case_id']}.npz"
                metadata = {
                    "model": args.model, "seed": int(entry["seed"]), "case_id": case["case_id"],
                    "config_sha256": cfg["_config_sha256"], "checkpoint_sha256": entry["sha256"],
                    "input_sha256": input_hash, "run_fingerprint": run_fingerprint,
                    "cohort_sha256": manifest["cohort"]["sha256"],
                    "manifest_sha256": report["manifest_fingerprint"]["sha256"],
                    "noise_sha256": case["noise_sha256"] or "",
                    "clean_p_sha256": report["internal_clean_prediction_hashes"][str(entry["seed"])],
                    "verification_only": verification,
                }
                p = _resume(path, metadata, reference, thresholds)
                resumed = p is not None
                if p is None:
                    p = state["clean_p"] if case["kind"] == "clean" else predict(
                        state["model"], x, state["scale"], batch_size, "cuda:0")
                    _probabilities(p, reference["y"].shape, case["case_id"])
                if case["kind"] == "clean":
                    _require(array_sha256(p) == metadata["clean_p_sha256"], "Resumed clean prediction differs")
                values, _ = classification_metrics(reference["y"], p, thresholds, state["clean_p"])
                if not resumed:
                    _atomic_npz(path, {**metadata, "p": p, "p_sha256": array_sha256(p),
                                      "thresholds": thresholds,
                                      **{key: reference[key] for key in ("y", "ids", "patient_ids", "indices")}})
                prediction_info = file_info(path)
                rows.append({"model": args.model, "seed": int(entry["seed"]), "case_id": case["case_id"],
                             "prediction_path": prediction_info["path"], "prediction_sha256": prediction_info["sha256"],
                             "checkpoint_sha256": entry["sha256"], "input_sha256": input_hash, **values})
                report["observed_cells"] += 1
                report["resumed_cells"] += int(resumed)
                print(f"cell-completed model={args.model} shard={args.shard_index}/{args.shard_count} "
                      f"seed={entry['seed']} case={case['case_id']} resumed={resumed} "
                      f"progress={report['observed_cells']}/{report['expected_cells']} "
                      f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)
            # Progress is durable, but completed is only published after every cell.
            save_json(report_path, report)
        _require(report["observed_cells"] == report["expected_cells"], "Incomplete shard")
        _require(_source_fingerprints() == sources, "Source changed while inference was running")
        _require(file_info(manifest_path) == report["manifest_fingerprint"], "Manifest changed during inference")
        _require(require_freeze(cfg, args.stage) == freeze, "Freeze changed during inference")
        write_csv(ledger_path, rows)
        report.update({"status": "completed", "finished_at": _now(),
                       "elapsed_seconds": time.perf_counter() - started, "ledger": file_info(ledger_path)})
        save_json(report_path, report)
        print(f"worker-completed report={report_path} cells={len(rows)}", flush=True)
        return report
    except BaseException as exc:
        report.update({"status": "failed", "finished_at": _now(),
                       "elapsed_seconds": time.perf_counter() - started,
                       "error": f"{type(exc).__name__}: {exc}"})
        save_json(report_path, report)
        raise
    finally:
        lock_path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--model", choices=("resnet", "tcn"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--case-ids", help="Comma-list for explicitly nonproduction verification only")
    parser.add_argument("--verification-only-host-override", action="store_true",
                        help="Bypass host/worker-count assignment only; CUDA and frozen inputs remain mandatory")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
