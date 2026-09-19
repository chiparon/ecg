"""Verify complete signed workers and strictly bridge immutable E/I/clean predictions."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import numpy as np
import pandas as pd

from methodology_supplement.infer import _digest, _now, _require, _resume
from phase1_ecg_robustness.src.evaluate import classification_metrics
from .common import (
    METRICS, array_sha256, check_info, file_info, load_config, materialize_input,
    read_json, resolve_path, save_json, sha256, stage_paths, write_csv,
)
from .infer import (
    COHORT_KEYS, RUN_KEYS, index_row, prediction_metadata, require_smoke,
    source_fingerprints, validated_context,
)

LEGACY_RUN_KEYS = ("stage", "config_sha256", "source_fingerprints", "manifest_fingerprint",
                   "freeze_fingerprint", "checkpoint_manifest_fingerprint", "checkpoint_fingerprints",
                   "software", "fp32_settings", "device_name", "verification_only")


def _same_row(observed, expected):
    for key, value in expected.items():
        actual = observed[key]
        if isinstance(value, float):
            _require(np.isfinite(float(actual)) and abs(float(actual) - value) <= 1e-12,
                     f"Prediction ledger metric mismatch: {key}")
        else:
            _require(actual == value, f"Prediction ledger identity mismatch: {key}")


def _production(report, cfg, stage, model, shard, count):
    _require(report["status"] == "completed" and report["stage"] == stage and
             report["model"] == model and report["shard_index"] == shard and
             report["shard_count"] == count and report["verification_only"] is False and
             report["case_filter"] is None and report["host_assignment_verified"] is True,
             "Invalid production worker report")
    expected_host = cfg["execution"][f"{model}_hostname"].lower()
    names = {report["host"].lower(), report["fqdn"].lower()}
    names |= {name.split(".")[0] for name in tuple(names)}
    _require(expected_host in names and report["assigned_host"].lower() == expected_host and
             report["device"] == "cuda:0" and (model != "tcn" or "4070" in report["device_name"]),
             "Worker was not on its assigned host/GPU")
    settings = report["fp32_settings"]
    expected = dict(default_dtype="torch.float32", amp=False, deterministic_algorithms=True,
                    cudnn_deterministic=True, cudnn_benchmark=False, cudnn_allow_tf32=False,
                    matmul_allow_tf32=False, batch_size=128)
    _require(all(settings[key] == value for key, value in expected.items()) and
             settings["cublas_workspace_config"] in (":4096:8", ":16:8"), "Unfrozen execution precision")


def _worker_rows(cfg, stage, paths, freeze, manifest, reference, clean, entries):
    cases = [c for c in manifest["cases"] if c["condition"].startswith("S_")]
    reports, rows, report_infos, clean_predictions = [], [], [], {}
    expected_sources = source_fingerprints()
    manifest_info = file_info(paths["inputs"] / "manifest.json")
    freeze_info = file_info(paths["logs"] / "freeze.json")
    for model in cfg["models"]:
        count = cfg["execution"][f"{model}_workers"]
        for shard in range(count):
            report_path = paths["logs"] / f"inference_{model}_{shard}.json"
            _require(not report_path.with_suffix(".lock").exists(), "Worker is still active")
            report = read_json(report_path)
            _production(report, cfg, stage, model, shard, count)
            model_entries = sorted((e for e in entries if e["model"] == model), key=lambda e: e["seed"])
            _require(report["config_sha256"] == cfg["_config_sha256"] and
                     report["manifest_fingerprint"] == manifest_info and report["freeze_fingerprint"] == freeze_info and
                     report["cohort_fingerprint"] == manifest["cohort"] and
                     report["clean_fingerprint"] == manifest["clean"] and
                     report["checkpoint_fingerprints"] == model_entries and
                     report["source_fingerprints"] == expected_sources and
                     report["run_fingerprint"] == _digest({key: report[key] for key in RUN_KEYS}),
                     "Worker protocol/source/run fingerprint mismatch")
            if stage == "full":
                _require(report["smoke_verification"] == require_smoke(cfg, stage), "Worker smoke gate changed")
            owned = [c for i, c in enumerate(cases) if i % count == shard]
            expected = {(model, int(e["seed"]), c["case_id"]) for e in model_entries for c in owned}
            frame = pd.read_csv(check_info(report["ledger"]), keep_default_na=False)
            observed = set(frame[["model", "seed", "case_id"]].itertuples(index=False, name=None))
            _require(observed == expected and len(frame) == len(expected) == report["observed_cells"] ==
                     report["expected_cells"] and report["owned_case_ids"] == [c["case_id"] for c in owned],
                     "Worker ledger is incomplete, duplicated, or outside its ownership")
            lookup = {(int(r["seed"]), r["case_id"]): r for r in frame.to_dict("records")}
            root = paths["predictions"] / f"worker_{model}_{shard}"
            for entry in model_entries:
                seed = str(entry["seed"])
                thresholds = np.asarray(entry["thresholds"], dtype=np.float64)
                clean_case = dict(case_id="clean", input_sha256=array_sha256(clean))
                clean_metadata = prediction_metadata(cfg, manifest, report, reference, clean_case, entry,
                                                     report["internal_clean_prediction_hashes"][seed])
                clean_info = report["internal_clean_predictions"][seed]
                clean_path = check_info(clean_info)
                _require(clean_path.resolve() == (root / "internal_clean" / f"seed_{seed}.npz").resolve(),
                         "Internal clean prediction escaped worker ownership")
                clean_p = _resume(clean_path, clean_metadata, reference, thresholds)
                _require(clean_p is not None and array_sha256(clean_p) == clean_metadata["clean_p_sha256"],
                         "Invalid internal clean reference")
                clean_predictions[(model, int(seed), shard)] = (clean_p, clean_info)
                for case in owned:
                    row = lookup[(int(seed), case["case_id"])]
                    path = resolve_path(row["prediction_path"])
                    _require(path.resolve() == (root / f"seed_{seed}" / f"{case['case_id']}.npz").resolve(),
                             "Prediction escaped worker ownership")
                    info = file_info(path)
                    _require(info["sha256"] == row["prediction_sha256"], "New prediction NPZ SHA mismatch")
                    metadata = prediction_metadata(cfg, manifest, report, reference, case, entry,
                                                   report["internal_clean_prediction_hashes"][seed])
                    p = _resume(path, metadata, reference, thresholds)
                    _require(p is not None, "Missing signed prediction")
                    metrics, _ = classification_metrics(reference["y"], p, thresholds, clean_p)
                    verified = index_row(case, entry, info, metadata, p, metrics)
                    _same_row(row, verified)
                    rows.append(verified)
            reports.append(report)
            report_infos.append(file_info(report_path))
    remote = [r for r in reports if r["model"] == "resnet"]
    _require(len(remote) == 2 and remote[0]["pid"] != remote[1]["pid"] and
             remote[0]["host"] == remote[1]["host"], "Two distinct DGX processes required")
    _require(remote[0]["internal_clean_prediction_hashes"] == remote[1]["internal_clean_prediction_hashes"],
             "DGX worker clean predictions are not identical")
    overlap = (min(datetime.fromisoformat(r["finished_at"]) for r in remote) -
               max(datetime.fromisoformat(r["started_at"]) for r in remote)).total_seconds()
    _require(overlap > 0, "Two DGX workers did not actually overlap")
    _require(len(rows) == 540, "Expected 540 new signed prediction units")
    return rows, reports, report_infos, clean_predictions, overlap


def _legacy_reports(cfg, stage, freeze, manifest, entries):
    old_freeze = read_json(check_info(freeze["legacy_freeze"]))
    old_manifest = read_json(check_info(freeze["legacy_input_manifest"]))
    _require(old_freeze["status"] == old_manifest["status"] == "completed" and
             old_freeze["stage"] == old_manifest["stage"] == stage and
             old_freeze["config_sha256"] == old_manifest["config_sha256"], "Invalid immutable baseline protocol")
    for key in ("cohort", "clean"):
        _require(old_manifest[key] == manifest[key], f"Baseline {key} identity changed")
    original_sources = {info["path"]: info for info in old_freeze["source_files"]}
    reports = []
    for info in freeze["legacy_worker_reports"]:
        report = read_json(check_info(info))
        model, shard, count = report["model"], report["shard_index"], report["shard_count"]
        _production(report, cfg, stage, model, shard, cfg["execution"][f"{model}_workers"])
        _require(count == cfg["execution"][f"{model}_workers"] and
                 report["config_sha256"] == old_manifest["config_sha256"] and
                 report["manifest_fingerprint"] == freeze["legacy_input_manifest"] and
                 report["freeze_fingerprint"] == freeze["legacy_freeze"] and
                 report["cohort_fingerprint"] == manifest["cohort"] and
                 report["clean_fingerprint"] == manifest["clean"] and
                 report["run_fingerprint"] == _digest({key: report[key] for key in LEGACY_RUN_KEYS}),
                 "Original worker provenance is unverifiable; rerun the affected conditions explicitly")
        check_info(report["checkpoint_manifest_fingerprint"])
        check_info(report["ledger"])
        for source in report["source_fingerprints"]:
            check_info(source)
            if source["path"] in original_sources:
                _require(source == original_sources[source["path"]], "Original model dependency freeze differs")
        required = {"phase1_ecg_robustness/src/models.py", "phase1_ecg_robustness/src/evaluate.py",
                    "phase1_ecg_robustness/src/train.py", "methodology_supplement/infer.py"}
        _require(required <= {s["path"] for s in report["source_fingerprints"]},
                 "Missing original model dependency fingerprints")
        expected_entries = sorted((e for e in entries if e["model"] == model), key=lambda e: e["seed"])
        _require(sorted(report["checkpoint_fingerprints"], key=lambda e: e["seed"]) == expected_entries,
                 "Original checkpoint configuration differs")
        reports.append(report)
    _require(len(reports) == 3 and {(r["model"], r["shard_index"]) for r in reports} ==
             {("resnet", 0), ("resnet", 1), ("tcn", 0)}, "Missing original worker evidence")
    return reports, old_manifest


def _reuse_rows(cfg, stage, freeze, manifest, reference, clean, entries, fresh_clean):
    reports, old_manifest = _legacy_reports(cfg, stage, freeze, manifest, entries)
    old_frame = pd.read_csv(check_info(freeze["legacy_prediction_index"]), keep_default_na=False)
    _require(not old_frame.duplicated(["model", "seed", "case_id"]).any(), "Original index has duplicate units")
    old_rows = {(r["model"], int(r["seed"]), r["case_id"]): r for r in old_frame.to_dict("records")}
    old_cases = {c["case_id"]: c for c in old_manifest["cases"]}
    clean_case = dict(case_id="clean", condition="clean", mode="", baseline_case_id="clean",
                      input_sha256=array_sha256(clean))
    cases = [clean_case] + [c for c in manifest["cases"] if c["condition"] in ("E", "I")]
    buffer, noise_hashes = np.empty(clean.shape, dtype=np.float32), {}
    for case in cases[1:]:
        original = old_cases[case["baseline_case_id"]]
        _require(case["input_sha256"] == original["input_sha256"] == case["baseline_input_sha256"] and
                 case["noise_sha256"] == original["noise_sha256"], "Reused condition input identity differs")
        path = resolve_path(case["noise_path"])
        if path not in noise_hashes:
            noise_hashes[path] = sha256(path)
        _require(noise_hashes[path] == case["noise_sha256"], "Reused base noise changed")
        base = np.load(path, mmap_mode="r", allow_pickle=False)
        materialize_input(clean, base, case, out=buffer)
        del base
        _require(array_sha256(buffer) == case["input_sha256"], "Reused final-input bytes differ")
    rows, bridge = [], []
    for entry in entries:
        model, seed = entry["model"], int(entry["seed"])
        thresholds = np.asarray(entry["thresholds"], dtype=np.float64)
        old_clean_p = None
        for case in cases:
            original_id = case["baseline_case_id"]
            key = (model, seed, original_id)
            _require(key in old_rows, f"No immutable baseline prediction: {key}; explicit rerun required")
            row = old_rows[key]
            owners = [r for r in reports if r["model"] == model and original_id in r["owned_case_ids"]]
            _require(len(owners) == 1, "Original prediction has no unique frozen owner")
            owner = owners[0]
            path = resolve_path(row["prediction_path"])
            info = file_info(path)
            _require(info["sha256"] == row["prediction_sha256"] and
                     row["checkpoint_sha256"] == entry["sha256"] and
                     row["input_sha256"] == case["input_sha256"], "Immutable prediction file/input/CP changed")
            original = old_cases[original_id]
            metadata = {
                "model": model, "seed": seed, "case_id": original_id,
                "config_sha256": old_manifest["config_sha256"], "checkpoint_sha256": entry["sha256"],
                "input_sha256": case["input_sha256"], "run_fingerprint": owner["run_fingerprint"],
                "cohort_sha256": manifest["cohort"]["sha256"],
                "manifest_sha256": freeze["legacy_input_manifest"]["sha256"],
                "noise_sha256": original.get("noise_sha256") or "",
                "clean_p_sha256": owner["internal_clean_prediction_hashes"][str(seed)],
                "verification_only": False,
            }
            p = _resume(path, metadata, reference, thresholds)
            _require(p is not None, f"Missing reused prediction: {path}")
            if original_id == "clean":
                old_clean_p = p
                _require(array_sha256(p) == metadata["clean_p_sha256"], "Original clean probability identity mismatch")
            _require(old_clean_p is not None and array_sha256(old_clean_p) == metadata["clean_p_sha256"],
                     "Original noisy prediction used a different clean reference")
            metrics, _ = classification_metrics(reference["y"], p, thresholds, old_clean_p)
            # Verify every original metric, including per-class and clean-relative metrics.
            for metric, value in metrics.items():
                tolerance = cfg["gates"]["legacy_f1_abs_error"] if "f1" in metric else cfg["gates"]["legacy_metric_abs_error"]
                _require(metric in row and np.isfinite(float(row[metric])) and
                         abs(value - float(row[metric])) <= tolerance,
                         f"Recomputed original metric mismatch: {key}: {metric}; reuse rejected")
            identity = {**metadata, "freeze_sha256": freeze["legacy_freeze"]["sha256"],
                        "source_sha256": _digest(owner["source_fingerprints"]),
                        "thresholds_sha256": array_sha256(thresholds),
                        **{name + "_sha256": array_sha256(reference[name]) for name in COHORT_KEYS}}
            rows.append(index_row(case, entry, info, identity, p, metrics, "reused", original_id))
            evidence = dict(model=model, seed=seed, case_id=case["case_id"], prediction_case_id=original_id,
                            condition=case["condition"], snr=case.get("snr", ""), noise_seed=case.get("noise_seed", ""),
                            status="verified_immutable_reuse", passed=True, reference_prediction_path=info["path"],
                            reference_prediction_sha256=info["sha256"], reference_p_sha256=array_sha256(p),
                            input_sha256=case["input_sha256"], checkpoint_sha256=entry["sha256"],
                            cohort_sha256=manifest["cohort"]["sha256"], thresholds_sha256=array_sha256(thresholds),
                            original_run_fingerprint=owner["run_fingerprint"], original_source_sha256=identity["source_sha256"],
                            all_original_metrics_recomputed=True,
                            maximum_original_metric_abs_error=max(abs(v - float(row[k])) for k, v in metrics.items()),
                            fresh_prediction_path="", fresh_prediction_sha256="", fresh_p_sha256="",
                            probabilities_max_abs=None, fixed_threshold_label_agreement=None,
                            fresh_macro_auroc_difference=None, fresh_macro_f1_difference=None, fresh_ece_difference=None,
                            numerical_comparison="not_applicable_immutable_file_reused_no_fresh_E_I_inference",
                            cross_device_bridge=False,
                            evidence_note="File SHA and stored probability SHA validate the immutable reference; no self-comparison is presented as a numerical bridge.")
            if original_id == "clean":
                fresh, fresh_info = fresh_clean[(model, seed, 0)]
                current, _ = classification_metrics(reference["y"], fresh, thresholds, fresh)
                differences = {metric: current[metric] - metrics[metric] for metric in METRICS}
                agreement = float(np.mean((fresh >= thresholds) == (p >= thresholds)))
                _require(abs(differences["macro_auroc"]) <= cfg["gates"]["legacy_metric_abs_error"] and
                         abs(differences["macro_f1"]) <= cfg["gates"]["legacy_f1_abs_error"] and
                         abs(differences["ece"]) <= cfg["gates"]["legacy_metric_abs_error"] and
                         agreement >= cfg["gates"]["legacy_label_agreement_min"],
                         "Fresh internal clean fails the frozen numerical bridge gates")
                evidence.update(fresh_prediction_path=fresh_info["path"], fresh_prediction_sha256=fresh_info["sha256"],
                                fresh_p_sha256=array_sha256(fresh), probabilities_max_abs=float(np.max(np.abs(fresh - p))),
                                fixed_threshold_label_agreement=agreement,
                                numerical_comparison="fresh_internal_clean_vs_immutable_original_clean_same_assigned_host",
                                evidence_note="Independent fresh clean computation, not a new condition; no claim of a cross-device comparison.",
                                **{"fresh_" + key + "_difference": value for key, value in differences.items()})
            bridge.append(evidence)
    _require(len(rows) == 222 and sum(r["condition"] == "clean" for r in rows) == 6,
             "Expected 216 E/I plus six clean immutable references")
    return rows, bridge


def run(config=None, stage="full"):
    cfg = load_config(config)
    smoke_info = require_smoke(cfg, stage)
    paths = stage_paths(cfg, stage)
    for key in ("tables", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    report_path = paths["logs"] / "evaluation_merge.json"
    lock = report_path.with_suffix(".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    result = dict(status="running", stage=stage, config_sha256=cfg["_config_sha256"], started_at=_now(),
                  smoke_verification=smoke_info)
    try:
        save_json(report_path, result)
        freeze, manifest, reference, clean, entries = validated_context(cfg, stage)
        new, reports, report_infos, fresh_clean, overlap = _worker_rows(
            cfg, stage, paths, freeze, manifest, reference, clean, entries)
        reused, bridge = _reuse_rows(cfg, stage, freeze, manifest, reference, clean, entries, fresh_clean)
        frame = pd.DataFrame(new + reused).sort_values(["model", "seed", "case_id"])
        expected = {(e["model"], int(e["seed"]), c) for e in entries
                    for c in ["clean"] + [case["case_id"] for case in manifest["cases"]]}
        observed = set(frame[["model", "seed", "case_id"]].itertuples(index=False, name=None))
        _require(observed == expected and len(frame) == len(expected) == 762 and
                 not frame.duplicated(["model", "seed", "case_id"]).any(), "Incomplete merged prediction grid")
        _require((frame.groupby("case_id").input_sha256.nunique() == 1).all(),
                 "Models/checkpoints did not use byte-identical inputs")
        for info in report_infos:
            check_info(info)
        prediction_path = paths["tables"] / "prediction_index.csv"
        bridge_path = paths["tables"] / "reuse_bridge.csv"
        write_csv(prediction_path, frame)
        write_csv(bridge_path, bridge)
        result.update(status="completed", finished_at=_now(), n_cases=126, n_checkpoints=6,
                      n_predictions=762, n_noisy_predictions=756, n_new_signed_predictions=540,
                      n_reused_E_I_predictions=216, n_reused_clean_predictions=6,
                      dgx_concurrent_processes=2, dgx_overlap_seconds=overlap,
                      dgx_worker_pids=[r["pid"] for r in reports if r["model"] == "resnet"],
                      spark_clean_predictions_identical=True, input_hashes_shared=True,
                      no_fresh_E_I_predictions_required=True, numerical_cross_device_bridge_performed=False,
                      immutable_reuse_evidence="Exact frozen NPZ, input, cohort, model dependencies, checkpoint, threshold and probability hashes; original metrics recomputed.",
                      worker_reports=report_infos, legacy_worker_reports=freeze["legacy_worker_reports"],
                      legacy_freeze=freeze["legacy_freeze"], legacy_prediction_index=freeze["legacy_prediction_index"],
                      source_files=[file_info(Path(__file__)), *source_fingerprints()],
                      freeze=file_info(paths["logs"] / "freeze.json"), manifest=file_info(paths["inputs"] / "manifest.json"),
                      outputs={"prediction_index": file_info(prediction_path), "reuse_bridge": file_info(bridge_path)})
        save_json(report_path, result)
        print(f"Merged 540 new signed units + 216 E/I + 6 clean references; DGX overlap={overlap:.3f}s", flush=True)
        return result
    except BaseException as exc:
        result.update(status="failed", finished_at=_now(), error=f"{type(exc).__name__}: {exc}")
        save_json(report_path, result)
        raise
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    args = parser.parse_args(argv)
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
