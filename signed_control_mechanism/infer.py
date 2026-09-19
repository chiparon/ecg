"""Independently owned, frozen-FP32 signed-control CUDA workers."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from pathlib import Path
import re
import time
import numpy as np

from methodology_supplement.infer import (
    _atomic_npz, _digest, _execution, _load_models, _now, _probabilities,
    _require, _resume, _source_fingerprints as _legacy_sources, _verify_infos,
)
from phase1_ecg_robustness.src.evaluate import classification_metrics, predict
from .common import (
    METRICS, array_sha256, case_grid, check_info, file_info, load_checkpoints,
    load_config, load_manifest, load_reference, materialize_input, read_json,
    require_freeze, resolve_path, save_json, sha256, stage_paths, write_csv,
)

COHORT_KEYS = ("y", "ids", "patient_ids", "indices")
RUN_KEYS = ("stage", "config_sha256", "source_fingerprints", "manifest_fingerprint",
            "freeze_fingerprint", "checkpoint_fingerprints", "software", "fp32_settings",
            "device_name", "model", "shard_index", "shard_count")


def source_fingerprints():
    sources = {info["path"]: info for info in _legacy_sources()}
    for name in ("infer.py", "common.py", "__init__.py"):
        info = file_info(Path(__file__).with_name(name))
        sources[info["path"]] = info
    return [sources[path] for path in sorted(sources)]


def require_smoke(cfg, stage):
    if stage == "full":
        path = stage_paths(cfg, "smoke")["logs"] / "smoke_verification.json"
        report = read_json(path)
        _require(report.get("status") in {"passed", "waived_by_user"} and
                 report.get("config_sha256") == cfg["_config_sha256"],
                 "Full inference/merge requires passed smoke or an explicit current-protocol user waiver")
        return file_info(path)
    return None


def validated_context(cfg, stage):
    freeze = require_freeze(cfg, stage)
    manifest = load_manifest(cfg, stage)
    for key in ("clean", "cohort", "draws", "sign_controls", "matrices"):
        _require(manifest[key] == freeze[key], f"Manifest/freeze mismatch: {key}")
        check_info(manifest[key])
    _verify_infos(manifest["audit"])
    expected = case_grid(cfg, stage)
    _require(manifest["n_cases"] == len(manifest["cases"]) == 126,
             "Input manifest must contain the entire 126-case grid")
    for actual, original in zip(manifest["cases"], expected):
        _require(all(actual.get(key) == value for key, value in original.items()),
                 "Input cases differ from frozen ordered grid")
        _require(re.fullmatch(r"[A-Za-z0-9_.-]+", actual["case_id"]) and
                 actual["case_id"] not in (".", "..") and
                 re.fullmatch(r"[0-9a-f]{64}", actual["input_sha256"]),
                 "Unsafe case ID or invalid input SHA")
    reference = load_reference(cfg, stage)
    n = manifest["n_records"]
    _require(n == freeze["n_records"] and reference["y"].shape == (n, 5) and
             reference["y"].dtype == np.float32 and np.isin(reference["y"], [0, 1]).all(),
             "Invalid frozen cohort labels")
    for key in ("ids", "patient_ids", "indices"):
        _require(reference[key].shape == (n,) and reference[key].dtype.kind != "O",
                 f"Invalid cohort {key}")
    _require(reference["ids"].dtype == np.int64 and reference["indices"].dtype == np.int64
             and len(np.unique(reference["ids"])) == n, "Invalid ECG identities")
    for key in COHORT_KEYS:
        _require(array_sha256(reference[key]) == freeze["array_hashes"][key],
                 f"Frozen cohort array hash changed: {key}")
    clean = np.load(check_info(manifest["clean"]), mmap_mode="r", allow_pickle=False)
    _require(clean.dtype == np.float32 and clean.shape == (n, 12, cfg["sequence_length"])
             and clean.flags.c_contiguous and np.isfinite(clean).all(), "Invalid clean inputs")
    legacy = read_json(check_info(freeze["legacy_input_manifest"]))
    clean_cases = [c for c in legacy["cases"] if c["case_id"] == "clean"]
    _require(len(clean_cases) == 1 and array_sha256(clean) == clean_cases[0]["input_sha256"],
             "Clean array differs from original immutable input")
    entries = load_checkpoints(cfg, stage)
    _require(len(entries) == 6 and {(e["model"], int(e["seed"])) for e in entries} ==
             {(m, s) for m in cfg["models"] for s in cfg["phase1_training_seeds"]},
             "Expected the exact six frozen checkpoints")
    for entry in entries:
        check_info(entry)
    return freeze, manifest, reference, clean, entries


def prediction_metadata(cfg, manifest, report, reference, case, entry, clean_hash):
    thresholds = np.asarray(entry["thresholds"], dtype=np.float64)
    return {
        "model": entry["model"], "seed": int(entry["seed"]), "case_id": case["case_id"],
        "config_sha256": cfg["_config_sha256"], "checkpoint_sha256": entry["sha256"],
        "input_sha256": case["input_sha256"], "run_fingerprint": report["run_fingerprint"],
        "cohort_sha256": manifest["cohort"]["sha256"],
        "manifest_sha256": report["manifest_fingerprint"]["sha256"],
        "freeze_sha256": report["freeze_fingerprint"]["sha256"],
        "source_sha256": _digest(report["source_fingerprints"]),
        "noise_sha256": case.get("noise_sha256", ""), "clean_p_sha256": clean_hash,
        "thresholds_sha256": array_sha256(thresholds), "verification_only": False,
        **{key + "_sha256": array_sha256(reference[key]) for key in COHORT_KEYS},
    }


def index_row(case, entry, info, metadata, p, metrics, origin="new", prediction_case_id=None):
    return {
        "model": entry["model"], "seed": int(entry["seed"]), "case_id": case["case_id"],
        "condition": case["condition"], "snr": case.get("snr", ""),
        "noise_seed": case.get("noise_seed", ""), "mode": case.get("mode", ""),
        "origin": origin, "prediction_path": info["path"], "prediction_sha256": info["sha256"],
        "prediction_case_id": prediction_case_id or case["case_id"],
        **{key: metadata[key] for key in (
            "checkpoint_sha256", "input_sha256", "cohort_sha256", "ids_sha256",
            "patient_ids_sha256", "indices_sha256", "y_sha256", "thresholds_sha256",
            "run_fingerprint", "config_sha256", "manifest_sha256", "freeze_sha256", "source_sha256")},
        "p_sha256": array_sha256(p), **{key: metrics[key] for key in METRICS},
    }


def run(args):
    started_at, started = _now(), time.perf_counter()
    cfg = load_config(args.config)
    _require(args.model in cfg["models"] and args.shard_count > 0 and
             0 <= args.shard_index < args.shard_count, "Invalid worker ownership")
    _require(cfg["execution"]["batch_size"] == 128, "Frozen batch size is 128")
    smoke_info = require_smoke(cfg, args.stage)
    paths = stage_paths(cfg, args.stage)
    for key in ("logs", "tables", "predictions"):
        paths[key].mkdir(parents=True, exist_ok=True)
    suffix = f"{args.model}_{args.shard_index}"
    report_path = paths["logs"] / f"inference_{suffix}.json"
    ledger_path = paths["tables"] / f"evaluation_{suffix}.csv"
    lock = report_path.with_suffix(".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    report = dict(status="running", stage=args.stage, model=args.model,
                  config_sha256=cfg["_config_sha256"], shard_index=args.shard_index,
                  shard_count=args.shard_count, pid=os.getpid(), started_at=started_at,
                  finished_at=None, observed_cells=0, resumed_cells=0,
                  internal_clean_prediction_hashes={}, internal_clean_predictions={},
                  smoke_verification=smoke_info, verification_only=False, case_filter=None)
    try:
        report.update(_execution(cfg, args.model, args.shard_count, False))
        save_json(report_path, report)
        freeze, manifest, reference, clean, all_entries = validated_context(cfg, args.stage)
        signed = [c for c in manifest["cases"] if c["condition"].startswith("S_")]
        _require(len(signed) == 90, "Expected 90 signed cases")
        owned = [c for i, c in enumerate(signed) if i % args.shard_count == args.shard_index]
        entries = sorted((e for e in all_entries if e["model"] == args.model), key=lambda e: e["seed"])
        sources = source_fingerprints()
        report.update(expected_cells=len(owned) * len(entries), owned_case_ids=[c["case_id"] for c in owned],
                      source_fingerprints=sources, checkpoint_fingerprints=entries,
                      manifest_fingerprint=file_info(paths["inputs"] / "manifest.json"),
                      freeze_fingerprint=file_info(paths["logs"] / "freeze.json"),
                      cohort_fingerprint=manifest["cohort"], clean_fingerprint=manifest["clean"])
        report["run_fingerprint"] = _digest({key: report[key] for key in RUN_KEYS})
        save_json(report_path, report)
        states = _load_models(entries, cfg, reference, args.stage)
        clean_case = {"case_id": "clean", "input_sha256": array_sha256(clean)}
        for state in states:
            seed = str(state["entry"]["seed"])
            p = predict(state["model"], clean, state["scale"], 128, "cuda:0")
            _probabilities(p, reference["y"].shape, "internal clean")
            state["clean_p"] = p
            report["internal_clean_prediction_hashes"][seed] = array_sha256(p)
            metadata = prediction_metadata(cfg, manifest, report, reference, clean_case,
                                           state["entry"], array_sha256(p))
            path = paths["predictions"] / f"worker_{suffix}" / "internal_clean" / f"seed_{seed}.npz"
            previous = _resume(path, metadata, reference, state["thresholds"])
            if previous is None:
                _atomic_npz(path, {**metadata, "p": p, "p_sha256": array_sha256(p),
                                  "thresholds": state["thresholds"],
                                  **{key: reference[key] for key in COHORT_KEYS}})
            else:
                _require(np.array_equal(previous, p), "Fresh internal clean differs from resume")
            report["internal_clean_predictions"][seed] = file_info(path)
        save_json(report_path, report)
        buffer = np.empty(clean.shape, dtype=np.float32)
        noise_hashes, rows = {}, []
        for case in owned:
            noise_path = resolve_path(case["noise_path"])
            if noise_path not in noise_hashes:
                noise_hashes[noise_path] = sha256(noise_path)
            _require(noise_hashes[noise_path] == case["noise_sha256"], "Frozen base noise changed")
            base = np.load(noise_path, mmap_mode="r", allow_pickle=False)
            materialize_input(clean, base, case, out=buffer)
            del base
            _require(np.isfinite(buffer).all() and array_sha256(buffer) == case["input_sha256"],
                     f"Final signed input SHA mismatch: {case['case_id']}")
            for state in states:
                entry = state["entry"]
                path = paths["predictions"] / f"worker_{suffix}" / f"seed_{entry['seed']}" / f"{case['case_id']}.npz"
                metadata = prediction_metadata(cfg, manifest, report, reference, case, entry,
                                               report["internal_clean_prediction_hashes"][str(entry["seed"])])
                p = _resume(path, metadata, reference, state["thresholds"])
                resumed = p is not None
                if not resumed:
                    p = predict(state["model"], buffer, state["scale"], 128, "cuda:0")
                    _probabilities(p, reference["y"].shape, case["case_id"])
                    _atomic_npz(path, {**metadata, "p": p, "p_sha256": array_sha256(p),
                                      "thresholds": state["thresholds"],
                                      **{key: reference[key] for key in COHORT_KEYS}})
                metrics, _ = classification_metrics(reference["y"], p, state["thresholds"], state["clean_p"])
                rows.append(index_row(case, entry, file_info(path), metadata, p, metrics))
                report["observed_cells"] += 1
                report["resumed_cells"] += int(resumed)
                print(f"cell-completed model={args.model} shard={args.shard_index} seed={entry['seed']} "
                      f"case={case['case_id']} progress={report['observed_cells']}/{report['expected_cells']}", flush=True)
            save_json(report_path, report)
        _require(report["observed_cells"] == report["expected_cells"], "Incomplete worker")
        _require(source_fingerprints() == sources, "Inference source changed during worker")
        _require(file_info(paths["inputs"] / "manifest.json") == report["manifest_fingerprint"]
                 and file_info(paths["logs"] / "freeze.json") == report["freeze_fingerprint"]
                 and require_freeze(cfg, args.stage) == freeze, "Frozen protocol changed during worker")
        write_csv(ledger_path, rows)
        report.update(status="completed", finished_at=_now(), elapsed_seconds=time.perf_counter() - started,
                      ledger=file_info(ledger_path))
        save_json(report_path, report)
        return report
    except BaseException as exc:
        report.update(status="failed", finished_at=_now(), error=f"{type(exc).__name__}: {exc}")
        save_json(report_path, report)
        raise
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--model", choices=("resnet", "tcn"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
