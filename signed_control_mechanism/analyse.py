"""Paired signed-control effects conditional on frozen checkpoints and patient draws."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import time

# Set these before importing NumPy, including in spawned Windows workers.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                  "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from methodology_supplement import bootstrap
from methodology_supplement.infer import _digest
from .common import (
    CONTRASTS, METRICS, WORKSPACE, array_sha256, case_grid, check_info,
    file_info, legacy_paths, load_config, load_manifest, load_reference,
    materialize_input, read_json, require_freeze, resolve_path, save_json,
    sha256, stage_paths, summarize_distribution, write_csv,
)

POINT_ATOL = 2e-7
CACHE_VERSION = 1
KEYS = ["model", "snr", "metric", "contrast"]
MODES = tuple(f"S_{index:02d}" for index in range(5))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _infos(value):
    """Traverse provenance objects, not arbitrary filesystem trees."""
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            yield {key: value[key] for key in ("path", "bytes", "sha256")}
        else:
            for item in value.values():
                yield from _infos(item)
    elif isinstance(value, list):
        for item in value:
            yield from _infos(item)


def _verify_infos(infos):
    unique = {}
    for info in infos:
        key = str(resolve_path(info["path"]).resolve())
        _require(key not in unique or unique[key] == info, f"Conflicting file identities: {key}")
        unique[key] = info
    for info in unique.values():
        check_info(info)
    return list(unique.values())


def _audit_point(values, record):
    expected = np.asarray([record[name] for name in METRICS], dtype=np.float64)
    _require(np.allclose(values[0], expected, rtol=0, atol=POINT_ATOL, equal_nan=True),
             f"Metric point mismatch: {record['model']}/{record['seed']}/{record['case_id']}")


def _prediction(record, checkpoint, case, reference, shared):
    path = resolve_path(record["prediction_path"])
    _require(sha256(path) == record["prediction_sha256"], f"Prediction hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        for name in ("y", "ids", "patient_ids", "indices"):
            _require(np.array_equal(saved[name], reference[name]), f"Prediction cohort mismatch: {path}: {name}")
        expected = dict(model=checkpoint["model"], seed=int(checkpoint["seed"]),
                        case_id=record["prediction_case_id"],
                        config_sha256=(shared["legacy_config_sha256"] if record["origin"] == "reused"
                                       else shared["config_sha256"]),
                        checkpoint_sha256=checkpoint["sha256"], input_sha256=case["input_sha256"],
                        cohort_sha256=shared["cohort_sha256"], run_fingerprint=record["run_fingerprint"],
                        manifest_sha256=(shared["legacy_manifest_sha256"] if record["origin"] == "reused"
                                         else shared["manifest_sha256"]),
                        noise_sha256=case.get("noise_sha256", ""),
                        clean_p_sha256=record["_clean_p_sha256"])
        for name, value in expected.items():
            _require(saved[name].shape == () and saved[name].item() == value,
                     f"Prediction provenance mismatch: {path}: {name}")
        for name in ("config_sha256", "manifest_sha256"):
            _require(record[name] == expected[name], f"Index provenance mismatch: {path}: {name}")
        freeze_hash = shared["legacy_freeze_sha256"] if record["origin"] == "reused" else shared["freeze_sha256"]
        _require(record["freeze_sha256"] == freeze_hash, f"Originating freeze mismatch: {path}")
        if record["origin"] == "new":
            for name in ("freeze_sha256", "source_sha256"):
                _require(saved[name].shape == () and saved[name].item() == record[name],
                         f"Signed prediction source mismatch: {path}: {name}")
        _require(not bool(saved["verification_only"].item()), f"Verification-only prediction: {path}")
        thresholds = saved["thresholds"]
        _require(thresholds.shape == (5,) and np.array_equal(
            thresholds, np.asarray(checkpoint["thresholds"], dtype=thresholds.dtype)),
            f"Original checkpoint thresholds changed: {path}")
        for name, array in (("ids", reference["ids"]), ("patient_ids", reference["patient_ids"]),
                            ("y", reference["y"]), ("indices", reference["indices"]), ("thresholds", thresholds)):
            digest = array_sha256(array)
            _require(record[f"{name}_sha256"] == digest, f"Index array hash mismatch: {path}: {name}")
            if f"{name}_sha256" in saved.files:
                _require(saved[f"{name}_sha256"].item() == digest, f"NPZ array hash mismatch: {path}: {name}")
        p = saved["p"]
        _require(p.shape == reference["y"].shape and p.dtype == np.float32
                 and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)),
                 f"Invalid FP32 probabilities: {path}")
        _require(array_sha256(p) == saved["p_sha256"].item() == record["p_sha256"],
                 f"Probability bytes changed: {path}")
    _require(sha256(path) == record["prediction_sha256"], f"Prediction changed while reading: {path}")
    return p, thresholds


def _invalid_auc(reference, draws):
    """Derive undefined draws from labels alone, retaining the common draw order."""
    inverse = reference["patient_inverse"]
    n_patients = draws.shape[1]
    positive = np.zeros((n_patients, 5), dtype=np.float64)
    np.add.at(positive, inverse, reference["y"])
    total = np.bincount(inverse, minlength=n_patients)[:, None]
    point = np.any((positive.sum(axis=0) == 0) | ((total - positive).sum(axis=0) == 0))
    positive_draws = draws @ positive
    negative_draws = draws @ (total - positive)
    return np.r_[point, np.any((positive_draws == 0) | (negative_draws == 0), axis=1)]


def _verify_inputs(cfg, stage, paths):
    freeze = require_freeze(cfg, stage)
    manifest_path = paths["inputs"] / "manifest.json"
    manifest = load_manifest(cfg, stage)
    index_path = paths["tables"] / "prediction_index.csv"
    merge_path = paths["logs"] / "evaluation_merge.json"
    merge = read_json(merge_path)
    _require(merge.get("status") == "completed" and merge.get("stage") == stage
             and merge.get("config_sha256") == cfg["_config_sha256"], "Missing or stale completed prediction merge")
    upstream = [file_info(manifest_path), file_info(index_path), file_info(merge_path),
                file_info(paths["logs"] / "freeze.json"), file_info(cfg["_config_path"])]
    upstream.extend(_infos(merge))
    upstream.extend(_infos(freeze))
    upstream.extend(_infos(manifest))
    origins = {}
    for info in [*merge["worker_reports"], *freeze["legacy_worker_reports"]]:
        report = read_json(check_info(info))
        _require(report["status"] == "completed" and report["stage"] == stage,
                 "Prediction producer is incomplete or from a different stage")
        identity = dict(clean_hashes=report["internal_clean_prediction_hashes"],
                        source_sha256=_digest(report["source_fingerprints"]))
        fingerprint = report["run_fingerprint"]
        _require(fingerprint not in origins or origins[fingerprint] == identity,
                 "Conflicting identities for a prediction run")
        origins[fingerprint] = identity
        upstream.extend(_infos(report))
    index_info = file_info(index_path)
    _require(index_info == merge["outputs"]["prediction_index"],
             "Merge does not authenticate current prediction index")
    _require(file_info(paths["tables"] / "reuse_bridge.csv") == merge["outputs"]["reuse_bridge"],
             "Merge does not authenticate current reuse bridge")
    for name in ("cohort", "draws", "clean", "sign_controls", "matrices"):
        _require(manifest[name] == freeze[name], f"Input/freeze identity mismatch: {name}")
    reference = load_reference(cfg, stage)
    n = len(reference["ids"])
    _require(n == manifest["n_records"] == freeze["n_records"] and reference["y"].shape == (n, 5)
             and np.isin(reference["y"], (0, 1)).all(), "Cohort labels or record count changed")
    unique, inverse = np.unique(reference["patient_ids"], return_inverse=True)
    _require(np.array_equal(unique, reference["unique_patients"])
             and np.array_equal(inverse, reference["patient_inverse"])
             and len(unique) == freeze["n_patients"], "Patient cluster mapping changed")
    _require(n == (100 if stage == "smoke" else 2158)
             and (stage != "full" or len(unique) == 1877), "Unexpected frozen cohort size")
    for name, digest in freeze["array_hashes"].items():
        if name in reference:
            _require(array_sha256(reference[name]) == digest, f"Frozen cohort array changed: {name}")
    draws = np.load(check_info(freeze["draws"]), mmap_mode="r", allow_pickle=False)
    _require(draws.dtype == np.int32 and draws.shape == (cfg["stages"][stage]["bootstrap_replicates"], len(unique))
             and len(draws) == (64 if stage == "smoke" else 2000)
             and np.all(draws >= 0) and np.all(draws.sum(axis=1) == len(unique)), "Invalid common patient draws")
    invalid_auc = _invalid_auc(reference, draws)
    clean = np.load(check_info(freeze["clean"]), mmap_mode="r", allow_pickle=False)
    _require(clean.dtype == np.float32 and clean.shape == (n, 12, 1000), "Clean array identity mismatch")
    cases = {}
    expected = {case["case_id"]: case for case in case_grid(cfg, stage)}
    noise_infos = {}
    for case in manifest["cases"]:
        name = case["case_id"]
        _require(name in expected and name not in cases, f"Unexpected or duplicate case: {name}")
        for key, value in expected[name].items():
            _require(case[key] == value, f"Case source identity mismatch: {name}: {key}")
        noise_path = resolve_path(case["noise_path"])
        if str(noise_path) not in noise_infos:
            noise_infos[str(noise_path)] = file_info(noise_path)
        _require(noise_infos[str(noise_path)]["sha256"] == case["noise_sha256"], f"Base noise changed: {name}")
        base = np.load(noise_path, mmap_mode="r", allow_pickle=False)
        _require(base.dtype == np.float32 and base.shape == clean.shape, f"Invalid base noise: {name}")
        digest = hashlib.sha256()
        for start in range(0, n, 128):
            block = materialize_input(clean[start:start + 128], base[start:start + 128], case)
            _require(np.isfinite(block).all(), f"Nonfinite final input: {name}")
            digest.update(memoryview(block).cast("B"))
        _require(digest.hexdigest() == case["input_sha256"], f"Final input bytes changed: {name}")
        if case["condition"] in ("E", "I"):
            _require(case["input_sha256"] == case["baseline_input_sha256"], f"Comparator bridge input mismatch: {name}")
        cases[name] = case
    _require(set(cases) == set(expected) and manifest["n_cases"] == 126, "Incomplete signed input grid")
    upstream.extend(noise_infos.values())
    legacy_manifest = read_json(check_info(freeze["legacy_input_manifest"]))
    old_cases = {case["case_id"]: case for case in legacy_manifest["cases"]}
    old_clean = [case for case in old_cases.values() if case["kind"] == "clean"]
    _require(len(old_clean) == 1, "Ambiguous legacy clean case")
    clean_hash = array_sha256(clean)
    _require(old_clean[0]["input_sha256"] == clean_hash, "Legacy clean input changed")
    cases["clean"] = dict(case_id="clean", condition="clean", mode="", input_sha256=clean_hash,
                          baseline_case_id=old_clean[0]["case_id"])
    del clean, draws
    checkpoints = {(cp["model"], int(cp["seed"])): cp for cp in freeze["checkpoint_entries"]}
    runs = {(model, seed) for model in cfg["models"] for seed in cfg["phase1_training_seeds"]}
    _require(len(freeze["checkpoint_entries"]) == len(checkpoints) == 6 and set(checkpoints) == runs,
             "Checkpoint grid changed")
    for checkpoint in checkpoints.values():
        info = file_info(resolve_path(checkpoint["path"]))
        _require(info["sha256"] == checkpoint["sha256"], "Checkpoint bytes changed")
        upstream.append(info)
    frame = pd.read_csv(index_path, keep_default_na=False)
    _require(not frame.duplicated(["model", "seed", "case_id"]).any()
             and set(frame[["model", "seed", "case_id"]].itertuples(index=False, name=None))
             == {(model, seed, case) for model, seed in runs for case in cases}, "Incomplete prediction grid")
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric].replace({"": np.nan, "nan": np.nan, "NaN": np.nan}), errors="raise")
    old_index = pd.read_csv(check_info(freeze["legacy_prediction_index"]), keep_default_na=False)
    old_rows = {(row["model"], int(row["seed"]), row["case_id"]): row for row in old_index.to_dict("records")}
    records = frame.to_dict("records")
    for record in records:
        case = cases[record["case_id"]]
        checkpoint = checkpoints[(record["model"], int(record["seed"]))]
        origin = origins[record["run_fingerprint"]]
        _require(record["source_sha256"] == origin["source_sha256"], "Prediction model-source fingerprint changed")
        record["_clean_p_sha256"] = origin["clean_hashes"][str(int(record["seed"]))]
        reused = case["condition"] in ("E", "I", "clean")
        _require(record["origin"] == ("reused" if reused else "new")
                 and record["condition"] == case["condition"] and record["mode"] == case["mode"]
                 and record["checkpoint_sha256"] == checkpoint["sha256"]
                 and record["input_sha256"] == case["input_sha256"]
                 and record["cohort_sha256"] == freeze["cohort"]["sha256"], "Merged prediction identity mismatch")
        if case["condition"] != "clean":
            _require(int(record["snr"]) == case["snr"] and int(record["noise_seed"]) == case["noise_seed"],
                     "Index noise pairing changed")
        _require(record["prediction_case_id"] == (case["baseline_case_id"] if reused else case["case_id"]),
                 "Saved/logical prediction case bridge mismatch")
        if reused:
            old = old_rows[(record["model"], int(record["seed"]), record["prediction_case_id"])]
            for key in ("prediction_sha256", "checkpoint_sha256", "input_sha256"):
                _require(record[key] == old[key], f"Legacy prediction bridge mismatch: {key}")
            _require(np.allclose([float(record[m]) for m in METRICS], [float(old[m]) for m in METRICS],
                                 rtol=0, atol=POINT_ATOL, equal_nan=True), "Legacy frozen metrics changed")
        info = file_info(resolve_path(record["prediction_path"]))
        _require(info["sha256"] == record["prediction_sha256"], "Index prediction hash mismatch")
        upstream.append(info)
    return cases, checkpoints, records, _verify_infos(upstream), freeze, legacy_manifest, invalid_auc


def _legacy_context(cfg, stage, freeze, legacy_manifest):
    """Cache-only incompatibility means recompute, never waive input provenance."""
    path = legacy_paths(cfg, stage)["logs"] / "new_statistics.json"
    try:
        log = read_json(path)
        _require(log.get("status") == "completed" and log["stage"] == stage
                 and log["config_sha256"] == legacy_manifest["config_sha256"], "Old analysis not completed/current")
        _require(log["software"]["numpy"] == np.__version__ and log["metrics"] == list(METRICS),
                 "Old NumPy version or metric order differs")
        for name in ("cohort", "draws"):
            _require(log["inputs"][name] == freeze[name], f"Old {name} identity differs")
        _require(log["inputs"]["manifest"] == freeze["legacy_input_manifest"]
                 and log["inputs"]["evaluation_index"] == freeze["legacy_prediction_index"],
                 "Old analysis input identities differ")
        old_freeze = read_json(check_info(log["inputs"]["freeze"]))
        dependencies = list(log["sources"]) + list(old_freeze["source_files"])
        _require(any(info["path"] == "phase1_ecg_robustness/src/supplemental_statistics.py"
                     for info in dependencies), "Missing original AUROC dependency fingerprint")
        inputs = _verify_infos([file_info(path), *log["inputs"].values(), *dependencies])
        expected_sources = ["methodology_supplement/analyse_new.py", "methodology_supplement/bootstrap.py",
                            "methodology_supplement/common.py", "phase1_ecg_robustness/src/evaluate.py",
                            "phase1_ecg_robustness/src/statistics.py", "phase2/src/statistics_phase2.py"]
        _require([info["path"] for info in log["sources"]] == expected_sources, "Unknown old cache source order")
        distribution_lookup = {(item["model"], int(item["seed"]), item["case_id"]): item["distribution"]
                               for item in log["distributions"]}
        shared = dict(cache_version=1, stage=stage, config_sha256=legacy_manifest["config_sha256"],
                      cohort_sha256=freeze["cohort"]["sha256"], draws_sha256=freeze["draws"]["sha256"],
                      source_sha256=[info["sha256"] for info in log["sources"]], metrics=list(METRICS),
                      point_absolute_tolerance=POINT_ATOL, numpy_version=np.__version__)
        return dict(compatible=True, reason="All original cache provenance matches", shared=shared,
                    inputs=inputs, distributions=distribution_lookup)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(compatible=False, reason=f"{type(exc).__name__}: {exc}", inputs=[])


def _check_distribution(values, record, invalid_auc):
    _require(values.shape == (len(invalid_auc), len(METRICS)) and values.dtype == np.float64,
             "Unexpected metric distribution shape/dtype")
    _require(np.array_equal(np.isnan(values[:, 0]), invalid_auc), "AUROC invalid draws changed or were dropped")
    _require(np.isfinite(values[:, 1:]).all() and not np.isinf(values).any(), "Nonfinite F1/ECE or infinite metric")
    finite = values[np.isfinite(values)]
    _require(np.all((finite >= 0) & (finite <= 1)), "Metric distribution outside raw 0-1 units")
    _audit_point(values, record)


def _save_array(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npy")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)
    return file_info(path)


def _checkpoint_distributions(task):
    cfg, stage, checkpoint, records, cases, shared, legacy, invalid_auc = task
    with threadpool_limits(limits=1):
        reference = load_reference(cfg, stage)
        freeze = require_freeze(cfg, stage)
        draws = np.load(check_info(freeze["draws"]), mmap_mode="r", allow_pickle=False)
        root = stage_paths(cfg, stage)["bootstrap"] / "metrics" / checkpoint["model"] / f"seed_{checkpoint['seed']}"
        root.mkdir(parents=True, exist_ok=True)
        outputs = []
        for record in records:
            started = time.perf_counter()
            case = cases[record["case_id"]]
            p, thresholds = _prediction(record, checkpoint, case, reference, shared)
            # Independently verify cached point values from the current probability bytes.
            point = bootstrap.metric_distribution(reference["y"], p, thresholds, reference["patient_inverse"],
                                                  draws[:0], batch_size=cfg["statistics"]["bootstrap_batch_size"])
            _audit_point(point, record)
            identity = dict(**shared, model=record["model"], seed=int(record["seed"]), case_id=case["case_id"],
                            prediction_case_id=record["prediction_case_id"], prediction_sha256=record["prediction_sha256"],
                            p_sha256=record["p_sha256"], input_sha256=case["input_sha256"],
                            checkpoint_sha256=checkpoint["sha256"], thresholds=checkpoint["thresholds"],
                            thresholds_sha256=record["thresholds_sha256"], run_fingerprint=record["run_fingerprint"])
            destination = root / f"{case['case_id']}.npy"
            sidecar = destination.with_suffix(".json")
            status, reason, info, cache_sources = "recomputed", "No compatible signed cache", None, []
            if sidecar.is_file():
                try:
                    saved = read_json(sidecar)
                    _require(saved["status"] == "completed" and saved["identity"] == identity, "Signed cache identity changed")
                    candidate = check_info(saved["distribution"])
                    _require(candidate.resolve() == destination.resolve(), "Signed cache points outside its owned destination")
                    values = np.load(candidate, mmap_mode="r", allow_pickle=False)
                    _check_distribution(values, record, invalid_auc)
                    info, status, reason = saved["distribution"], "resumed_new", "Verified same-identity signed-namespace cache"
                    cache_sources = [file_info(sidecar)]
                    del values
                except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
                    reason = f"Signed cache rejected: {type(exc).__name__}: {exc}"
            if info is None and record["origin"] == "reused":
                if legacy["compatible"]:
                    try:
                        key = (record["model"], int(record["seed"]), record["prediction_case_id"])
                        old_info = legacy["distributions"][key]
                        old_path = check_info(old_info)
                        old_sidecar = old_path.with_suffix(".json")
                        old_saved = read_json(old_sidecar)
                        expected = dict(**legacy["shared"], model=record["model"], seed=int(record["seed"]),
                                        case_id=record["prediction_case_id"], prediction_sha256=record["prediction_sha256"],
                                        input_sha256=case["input_sha256"], checkpoint_sha256=checkpoint["sha256"],
                                        thresholds=checkpoint["thresholds"])
                        _require(old_saved["status"] == "completed" and old_saved["identity"] == expected
                                 and old_saved["distribution"] == old_info, "Original cache identity mismatch")
                        values = np.load(old_path, mmap_mode="r", allow_pickle=False)
                        _check_distribution(values, record, invalid_auc)
                        info, status, reason = old_info, "reused_legacy", "Verified original distribution; referenced without copying"
                        cache_sources = [file_info(old_sidecar)]
                        del values
                    except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
                        reason = f"Legacy cache rejected: {type(exc).__name__}: {exc}"
                else:
                    reason = legacy["reason"]
            compute_seconds = 0.0
            if info is None:
                compute_started = time.perf_counter()
                values = bootstrap.metric_distribution(reference["y"], p, thresholds, reference["patient_inverse"], draws,
                                                       batch_size=cfg["statistics"]["bootstrap_batch_size"])
                compute_seconds = time.perf_counter() - compute_started
                _check_distribution(values, record, invalid_auc)
                info = _save_array(destination, values)
                save_json(sidecar, dict(status="completed", identity=identity, distribution=info))
                cache_sources = [file_info(sidecar)]
                del values
            outputs.append(dict(model=record["model"], seed=int(record["seed"]), case_id=case["case_id"],
                                cache_status=status, cache_reason=reason, distribution=info, cache_sources=cache_sources,
                                identity=identity, n_invalid=int(invalid_auc[1:].sum()),
                                compute_seconds=compute_seconds, elapsed_seconds=time.perf_counter() - started))
            del p
            print(f"statistics {record['model']} seed={record['seed']} {case['case_id']} {status}", flush=True)
        return outputs


def _tables(cfg, stage, cases, cache):
    seeds, noises = cfg["phase1_training_seeds"], cfg["phase1_noise_seeds"]
    paths = stage_paths(cfg, stage)
    lookup = {(item["model"], item["seed"], item["case_id"]): item["distribution"] for item in cache}
    logical = {(case["snr"], case["noise_seed"], case["condition"]): case["case_id"]
               for case in cases.values() if case["condition"] != "clean"}
    rows = {name: [] for name in ("signed_control_summary", "signed_control_seed_effects",
                                  "signed_control_noise_effects", "signed_mode_effects",
                                  "signed_mode_summary", "absolute_metrics")}
    paired = []

    def load(model, seed, case_id):
        return np.load(check_info(lookup[(model, seed, case_id)]), allow_pickle=False)

    def summarize(base, distributions, category, **extra):
        stats = summarize_distribution(distributions[:, 0], distributions)
        rows[category].append(dict(**base, **stats, **extra))
        name = "__".join(str(base.get(key, "")) for key in ("model", "snr", "metric", "contrast", "condition", "mode"))
        info = _save_array(paths["bootstrap"] / "paired" / category / f"{name}.npy", distributions)
        paired.append(dict(table=category, **base, distribution=info,
                           axes=["training_seed", "point_then_common_patient_draw"],
                           seed_order=list(seeds), n_invalid=stats["n_invalid"]))

    for model in cfg["models"]:
        clean = np.stack([load(model, seed, "clean") for seed in seeds])
        for metric_index, metric in enumerate(METRICS):
            summarize(dict(model=model, snr="", metric=metric, condition="clean", mode=""),
                      clean[:, :, metric_index], "absolute_metrics", noise_sd=0.0, signed_mode_sd=0.0)
        for snr in cfg["snrs"]:
            conditions = {}
            for condition in ("E", "I", *MODES):
                conditions[condition] = np.stack([
                    np.stack([load(model, seed, logical[(snr, noise, condition)]) for noise in noises])
                    for seed in seeds])
            # Axes: training seed, noise, signed mode, point+draw, metric.
            signed = np.stack([conditions[mode] for mode in MODES], axis=2)
            e, i = conditions["E"][:, :, None], conditions["I"][:, :, None]
            differences = {"E-S": e - signed, "S-I": signed - i,
                           "E-I": np.broadcast_to(e - i, signed.shape)}
            for contrast, per_mode in differences.items():
                per_noise = per_mode.mean(axis=2)
                per_seed = per_noise.mean(axis=1)
                mode_seed = per_mode.mean(axis=1)
                for metric_index, metric in enumerate(METRICS):
                    base = dict(model=model, snr=snr, metric=metric, contrast=contrast)
                    noise_points = per_noise[:, :, 0, metric_index].mean(axis=0)
                    mode_points = mode_seed[:, :, 0, metric_index].mean(axis=0)
                    summarize(base, per_seed[:, :, metric_index], "signed_control_summary",
                              noise_sd=float(noise_points.std(ddof=1)),
                              signed_mode_sd=float(mode_points.std(ddof=1)))
                    for seed_index, seed in enumerate(seeds):
                        rows["signed_control_seed_effects"].append(dict(
                            **base, seed=seed, estimate=float(per_seed[seed_index, 0, metric_index])))
                        for noise_index, noise in enumerate(noises):
                            rows["signed_control_noise_effects"].append(dict(
                                **base, seed=seed, noise_seed=noise,
                                estimate=float(per_noise[seed_index, noise_index, 0, metric_index])))
                        for mode_index, mode in enumerate(MODES):
                            rows["signed_mode_effects"].append(dict(
                                **base, seed=seed, mode=mode, redundant_contrast=contrast == "E-I",
                                estimate=float(mode_seed[seed_index, mode_index, 0, metric_index])))
                    for mode_index, mode in enumerate(MODES):
                        summarize(dict(**base, mode=mode), mode_seed[:, mode_index, :, metric_index],
                                  "signed_mode_summary", redundant_contrast=contrast == "E-I",
                                  noise_sd=float(per_mode[:, :, mode_index, 0, metric_index].mean(axis=0).std(ddof=1)))
            conditions["S"] = signed.mean(axis=2)
            for condition, values in conditions.items():
                for metric_index, metric in enumerate(METRICS):
                    mode_sd = (float(signed[:, :, :, 0, metric_index].mean(axis=1).mean(axis=0).std(ddof=1))
                               if condition == "S" else 0.0)
                    summarize(dict(model=model, snr=snr, metric=metric, condition=condition,
                                   mode=condition if condition in MODES else ""),
                              values.mean(axis=1)[:, :, metric_index], "absolute_metrics",
                              noise_sd=float(values[:, :, 0, metric_index].mean(axis=0).std(ddof=1)),
                              signed_mode_sd=mode_sd)
    expected = dict(signed_control_summary=54, signed_control_seed_effects=162,
                    signed_control_noise_effects=972, signed_mode_effects=810,
                    signed_mode_summary=270, absolute_metrics=150)
    keys = {"signed_control_summary": KEYS, "signed_control_seed_effects": [*KEYS, "seed"],
            "signed_control_noise_effects": [*KEYS, "seed", "noise_seed"],
            "signed_mode_effects": [*KEYS, "seed", "mode"], "signed_mode_summary": [*KEYS, "mode"],
            "absolute_metrics": ["model", "snr", "metric", "condition", "mode"]}
    frames = {}
    for name, items in rows.items():
        frame = pd.DataFrame(items)
        _require(len(frame) == expected[name] and not frame.duplicated(keys[name]).any(),
                 f"Incomplete or duplicate statistics table: {name}")
        frames[name] = frame.sort_values(keys[name], kind="stable").reset_index(drop=True)
    return frames, paired


def run(config_path=None, stage="full"):
    cfg = load_config(config_path)
    paths = stage_paths(cfg, stage)
    _require(resolve_path(cfg["results_dir"]).resolve() == (WORKSPACE / "signed_control_mechanism").resolve(),
             "Analysis outputs must stay in the new signed-control namespace")
    for category in ("tables", "logs", "bootstrap"):
        paths[category].mkdir(parents=True, exist_ok=True)
    log_path = paths["logs"] / "statistics.json"
    started = time.perf_counter()
    running = dict(status="running", stage=stage, config_sha256=cfg["_config_sha256"], started_at=_now())
    save_json(log_path, running)
    try:
        _require(tuple(bootstrap.METRICS) == METRICS and tuple(cfg["statistics"]["metrics"]) == METRICS
                 and tuple(cfg["contrasts"]) == CONTRASTS and tuple(cfg["conditions"][2:]) == MODES,
                 "Frozen metrics, contrasts or signed mode ordering changed")
        with threadpool_limits(limits=1):
            cases, checkpoints, records, upstream, freeze, legacy_manifest, invalid_auc = _verify_inputs(cfg, stage, paths)
        sources = [file_info(path) for path in (
            Path(__file__), Path(__file__).with_name("common.py"), Path(bootstrap.__file__),
            WORKSPACE / "methodology_supplement/common.py",
            WORKSPACE / "phase1_ecg_robustness/src/supplemental_statistics.py",
            WORKSPACE / "phase1_ecg_robustness/src/evaluate.py",
            WORKSPACE / "phase1_ecg_robustness/src/statistics.py",
            WORKSPACE / "phase2/src/statistics_phase2.py")]
        legacy = _legacy_context(cfg, stage, freeze, legacy_manifest)
        shared = dict(cache_version=CACHE_VERSION, stage=stage, config_sha256=cfg["_config_sha256"],
                      legacy_config_sha256=legacy_manifest["config_sha256"],
                      legacy_manifest_sha256=freeze["legacy_input_manifest"]["sha256"],
                      legacy_freeze_sha256=freeze["legacy_freeze"]["sha256"],
                      cohort_sha256=freeze["cohort"]["sha256"], draws_sha256=freeze["draws"]["sha256"],
                      manifest_sha256=file_info(paths["inputs"] / "manifest.json")["sha256"],
                      freeze_sha256=file_info(paths["logs"] / "freeze.json")["sha256"],
                      source_sha256=[source["sha256"] for source in sources],
                      metrics=list(METRICS), numpy_version=np.__version__, point_absolute_tolerance=POINT_ATOL)
        tasks = [(cfg, stage, checkpoint,
                  [row for row in records if (row["model"], int(row["seed"])) == key],
                  cases, shared, legacy, invalid_auc) for key, checkpoint in sorted(checkpoints.items())]
        workers = min(4, max(1, int(cfg["statistics"]["workers"])), len(tasks), os.cpu_count() or 1)
        cache = []
        computation_started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(_checkpoint_distributions, tasks):
                cache.extend(result)
        computation_seconds = time.perf_counter() - computation_started
        _require(len(cache) == 762 and len({(item["model"], item["seed"], item["case_id"]) for item in cache}) == 762,
                 "Distribution grid incomplete")
        frames, paired = _tables(cfg, stage, cases, cache)
        outputs = {}
        for name, frame in frames.items():
            path = paths["tables"] / f"{name}.csv"
            write_csv(path, frame)
            outputs[name] = dict(**file_info(path), rows=len(frame))
        source_manifest_path = paths["bootstrap"] / "distribution_manifest.json"
        save_json(source_manifest_path, dict(status="requires_completed_statistics_log",
                                            completion_log=str(log_path.relative_to(WORKSPACE).as_posix()),
                                            stage=stage, config_sha256=cfg["_config_sha256"],
                                            sources=sources, inputs=upstream, metrics=list(METRICS),
                                            metric_distributions=cache, paired_distributions=paired,
                                            draw_order="point at row zero; remaining rows are unchanged frozen draws",
                                            training_seed_order=cfg["phase1_training_seeds"]))
        outputs["distribution_manifest"] = file_info(source_manifest_path)
        # Recheck all original inputs, every prediction, caches and outputs immediately
        # before atomically publishing completion. Stale outputs never imply success.
        final_infos = [*upstream, *sources, *legacy["inputs"], *list(_infos(cache)),
                       *list(_infos(paired)), *list(_infos(outputs))]
        _verify_infos(final_infos)
        _require(require_freeze(cfg, stage) == freeze and load_config(config_path)["_config_sha256"] == cfg["_config_sha256"],
                 "Freeze/config changed during analysis")
        summary = frames["signed_control_summary"]
        result = dict(
            **running, finished_at=_now(), elapsed_seconds=time.perf_counter() - started,
            distribution_wall_seconds=computation_seconds, workers=workers, blas_threads=1,
            software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
            sources=sources, inputs=upstream, outputs=outputs, n_cases=126, n_checkpoints=6,
            n_records=freeze["n_records"], n_patients=freeze["n_patients"], n_distributions=len(cache),
            n_bootstrap=cfg["stages"][stage]["bootstrap_replicates"],
            cache_counts={status: sum(item["cache_status"] == status for item in cache)
                          for status in ("reused_legacy", "resumed_new", "recomputed")},
            cache_compute_seconds=sum(item["compute_seconds"] for item in cache),
            legacy_cache_compatible=legacy["compatible"], legacy_cache_reason=legacy["reason"],
            legacy_cache_sources=legacy["inputs"],
            n_invalid={metric: sorted(int(value) for value in summary.loc[summary.metric == metric, "n_invalid"].unique())
                       for metric in METRICS},
            n_invalid_common_auc_draws=int(invalid_auc[1:].sum()),
            metrics=list(METRICS), contrasts=list(CONTRASTS), primary=cfg["primary"],
            training_seeds=cfg["phase1_training_seeds"], noise_seeds=cfg["phase1_noise_seeds"], modes=list(MODES),
            row_counts={name: len(frame) for name, frame in frames.items()},
            uncertainty=dict(
                seed_sd="Sample SD ddof=1 across three training-seed paired estimates after equal mode/noise averaging.",
                patient_ci="95% percentile paired patient-cluster CI conditional on these checkpoints, noises and controls; not joint training/noise/mode/patient uncertainty. Ordinary means propagate missing-class AUROC NaNs; percentile endpoints use only valid aggregate draws, with invalid draws retained and counted, never redrawn.",
                noise_sd="Descriptive sample SD across six fixed noise estimates after averaging modes and training seeds.",
                signed_mode_sd="Descriptive sample SD across five fixed mode estimates after averaging noises and training seeds; not extra training repetitions or a population CI."),
            aggregation="Metric differences within identical checkpoint/noise/mode/patient draw; modes then noises within seed, then three seeds. Never a probability ensemble.",
            detail_tables=dict(
                signed_control_seed_effects="162 estimates after equal mode/noise averaging.",
                signed_control_noise_effects="972 seed/noise estimates after mode averaging.",
                signed_mode_effects="810 seed/mode estimates after noise averaging. E-I is identically repeated for all five modes, explicitly flagged redundant_contrast; never independent evidence.",
                signed_mode_summary="270 individually reportable mode effects with training seed SD and conditional patient CI; redundant E-I explicitly flagged.",
                absolute_metrics="150 rows: clean by model/metric, and E/I/five individual modes/S mean by model/SNR/metric. All in raw 0-1 units."),
            signs="Positive AUROC/F1 contrasts favor the first condition; positive ECE contrasts mean worse calibration for the first condition.",
            inference="Exploratory descriptive mechanism analysis only; no p-values, significance families, results-selected modes, or equivalence claims.",
        )
        result["status"] = "completed"
        save_json(log_path, result)
        return result
    except Exception as exc:
        save_json(log_path, dict(running, status="failed", finished_at=_now(),
                                 elapsed_seconds=time.perf_counter() - started,
                                 error=f"{type(exc).__name__}: {exc}"))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
