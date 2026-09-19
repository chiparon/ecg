"""Descriptive phase-one effects, with separately conditioned uncertainty levels."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd

from . import bootstrap
from .common import (
    WORKSPACE, array_sha256, file_info, load_checkpoints, load_config,
    load_reference, read_json, require_freeze, resolve_path, save_json,
    sha256, stage_paths, summarize_distribution, write_csv,
)

METRICS = ("macro_auroc", "macro_f1", "ece")
OUTCOMES = (
    "structure_effect", "electrode_absolute", "independent_rms_absolute",
    "electrode_drop", "independent_rms_drop",
)
KEYS = [
    "analysis", "phase", "model", "strategy", "source_id", "matrix_family",
    "band_id", "snr", "metric", "outcome",
]
SUMMARY_FIELDS = [
    *KEYS, "estimate", "seed_sd", "seed_n", "patient_ci_low", "patient_ci_high",
    "n_bootstrap", "n_invalid", "noise_sd", "random_matrix_sd",
]
POINT_ATOL = 2e-7
CACHE_VERSION = 1


def _now():
    return datetime.now(timezone.utc).isoformat()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_info(info):
    path = resolve_path(info["path"])
    _require(path.stat().st_size == info["bytes"] and sha256(path) == info["sha256"],
             f"Artifact fingerprint mismatch: {path}")
    return path


def _case_key(case):
    return (case["source_id"], int(case["snr"]), int(case["noise_seed"]), case["condition"])


def _expected_cases(cfg):
    expected = {}
    sources = [("standard", "standard", "legacy_0p5_40", cfg["snrs"])]
    sources += [(name, name, "legacy_0p5_40", cfg["ablation_snrs"])
                for name in ("limb_only", "precordial_only")]
    sources += [(f"random_{i:02d}", "randomized", "legacy_0p5_40", cfg["ablation_snrs"])
                for i in range(cfg["matrix"]["random_replicates"])]
    sources += [(band["id"], "standard", band["id"], cfg["ablation_snrs"])
                for band in cfg["bands"]]
    for source, family, band, snrs in sources:
        for snr in snrs:
            tags = {"band"} if band != "legacy_0p5_40" else {"matrix"}
            if source == "standard":
                tags = {"snr", "matrix"} if snr in cfg["ablation_snrs"] else {"snr"}
            for noise in cfg["phase1_noise_seeds"]:
                for condition in cfg["conditions"]:
                    expected[(source, snr, noise, condition)] = dict(
                        matrix_family=family, band_id=band,
                        matrix_id="standard" if band != "legacy_0p5_40" else source,
                        analysis_tags=tags,
                    )
    return expected


def _prediction(record, checkpoint, case, reference, config_hash):
    """Validate every saved identity, including raw probability bytes."""
    path = resolve_path(record["prediction_path"])
    before = sha256(path)
    _require(before == record["prediction_sha256"], f"Prediction checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        for key in ("y", "ids", "patient_ids", "indices"):
            _require(np.array_equal(saved[key], reference[key]), f"Cohort mismatch: {path}: {key}")
        for key, value in (
            ("model", checkpoint["model"]), ("seed", int(checkpoint["seed"])),
            ("case_id", case["case_id"]), ("config_sha256", config_hash),
            ("checkpoint_sha256", checkpoint["sha256"]),
            ("input_sha256", case["input_sha256"]),
        ):
            _require(saved[key].item() == value, f"Prediction provenance mismatch: {path}: {key}")
        thresholds = saved["thresholds"]
        _require(thresholds.shape == (5,) and np.array_equal(
            thresholds, np.asarray(checkpoint["thresholds"], dtype=thresholds.dtype)),
            f"Frozen threshold mismatch: {path}")
        p = saved["p"]
        _require(p.shape == reference["y"].shape and np.isfinite(p).all()
                 and not np.any((p < 0) | (p > 1)), f"Invalid probabilities: {path}")
        _require(array_sha256(p) == saved["p_sha256"].item(), f"Raw probability checksum mismatch: {path}")
    _require(sha256(path) == before, f"Prediction changed while reading: {path}")
    return p, thresholds


def _verify_inputs(cfg, stage, paths):
    freeze = require_freeze(cfg, stage)
    manifest_path = paths["inputs"] / "manifest.json"
    manifest = read_json(manifest_path)
    for key, value in (("status", "completed"), ("stage", stage),
                       ("config_sha256", cfg["_config_sha256"])):
        _require(manifest.get(key) == value, f"Input manifest mismatch: {key}")
    for key in ("cohort", "clean", "matrices"):
        _check_info(manifest[key])
    _require(resolve_path(manifest["cohort"]["path"]).resolve()
             == (paths["inputs"] / "cohort.npz").resolve(), "Unexpected cohort path")
    matrix_manifest = read_json(resolve_path(manifest["matrices"]["path"]))
    _require(matrix_manifest.get("status") == "completed"
             and matrix_manifest.get("config_sha256") == cfg["_config_sha256"],
             "Unfinalized or foreign matrix manifest")
    matrix_hashes = {}
    for matrix in matrix_manifest["matrices"]:
        values = np.asarray(matrix["values"], dtype=np.float64)
        _require(values.shape == (12, 9) and np.isfinite(values).all()
                 and matrix["id"] not in matrix_hashes, "Invalid or duplicate acquisition matrix")
        matrix_hashes[matrix["id"]] = array_sha256(values)
    reference = load_reference(cfg, stage)
    n = len(reference["ids"])
    _require(n == manifest["n_records"] and reference["y"].shape == (n, 5), "Invalid cohort shape")
    unique, inverse = np.unique(reference["patient_ids"], return_inverse=True)
    _require(np.array_equal(unique, reference["unique_patients"])
             and np.array_equal(inverse, reference["patient_inverse"]), "Patient cluster mapping changed")
    draws_path = paths["inputs"] / "patient_draws.npy"
    draws = np.load(draws_path, mmap_mode="r", allow_pickle=False)
    _require(draws.shape == (cfg["stages"][stage]["bootstrap_replicates"], len(unique))
             and draws.dtype == np.int32 and np.all(draws >= 0)
             and np.all(draws.sum(axis=1) == len(unique)), "Invalid shared patient multiplicities")
    clean = np.load(resolve_path(manifest["clean"]["path"]), mmap_mode="r", allow_pickle=False)
    _require(clean.shape == (n, 12, cfg["sequence_length"]) and clean.dtype == np.float32,
             "Invalid clean input array")
    expected = _expected_cases(cfg)
    cases = {}
    logical = {}
    clean_cases = []
    verified_noise = {}
    for case in manifest["cases"]:
        case_id = case["case_id"]
        _require(re.fullmatch(r"[A-Za-z0-9_.-]+", case_id) is not None
                 and case_id not in (".", "..") and case_id not in cases,
                 f"Unsafe or duplicated case ID: {case_id}")
        cases[case_id] = case
        if case["kind"] == "clean":
            _require(case["condition"] == "clean" and case["snr"] == 100
                     and case["noise_seed"] == 0 and case["noise_path"] is None
                     and case["noise_sha256"] is None, "Invalid clean case")
            _require(array_sha256(clean) == case["input_sha256"], "Clean input hash mismatch")
            clean_cases.append(case_id)
            continue
        key = _case_key(case)
        _require(case["kind"] == "bandpass" and key in expected and key not in logical,
                 f"Unexpected or duplicated scientific case: {key}")
        logical[key] = case_id
        for name, value in expected[key].items():
            actual = set(case[name]) if name == "analysis_tags" else case[name]
            _require(actual == value, f"Case metadata mismatch: {case_id}: {name}")
        _require(case["matrix_id"] in matrix_hashes
                 and case["matrix_sha256"] == matrix_hashes[case["matrix_id"]],
                 f"Acquisition matrix hash mismatch: {case_id}")
        noise_path = resolve_path(case["noise_path"])
        noise_key = str(noise_path.resolve())
        if noise_key not in verified_noise:
            verified_noise[noise_key] = sha256(noise_path)
        _require(verified_noise[noise_key] == case["noise_sha256"], f"Noise hash mismatch: {case_id}")
        noise = np.load(noise_path, mmap_mode="r", allow_pickle=False)
        _require(noise.shape == clean.shape and noise.dtype == np.float32, f"Noise shape/dtype mismatch: {case_id}")
        digest = hashlib.sha256()
        scale = np.float32(10 ** (-int(case["snr"]) / 20))
        # The canonical input uses two float32 operations, never a fused operation.
        for start in range(0, n, 128):
            block = np.multiply(noise[start:start + 128], scale, dtype=np.float32)
            np.add(clean[start:start + 128], block, out=block)
            _require(np.isfinite(block).all(), f"Nonfinite input: {case_id}")
            digest.update(memoryview(block).cast("B"))
        _require(digest.hexdigest() == case["input_sha256"], f"Actual noisy input hash mismatch: {case_id}")
        del noise
    _require(set(logical) == set(expected) and len(clean_cases) == 1
             and len(cases) == manifest["n_cases"] == len(expected) + 1, "Incomplete manifest case grid")
    del clean
    checkpoints = load_checkpoints(cfg, stage)
    expected_runs = {(model, seed) for model in cfg["models"] for seed in cfg["phase1_training_seeds"]}
    checkpoint_lookup = {(item["model"], int(item["seed"])): item for item in checkpoints}
    _require(len(checkpoint_lookup) == len(checkpoints) and set(checkpoint_lookup) == expected_runs,
             "Incomplete or duplicated checkpoint grid")
    for checkpoint in checkpoints:
        _require(sha256(resolve_path(checkpoint["path"])) == checkpoint["sha256"], "Checkpoint file hash mismatch")
        thresholds = np.asarray(checkpoint["thresholds"])
        _require(thresholds.shape == (5,) and np.isfinite(thresholds).all(), "Invalid frozen thresholds")
    index_path = paths["tables"] / "evaluation_index.csv"
    frame = pd.read_csv(index_path, keep_default_na=False, na_values=["", "nan", "NaN"])
    index_keys = ["model", "seed", "case_id"]
    _require(not frame.duplicated(index_keys).any(), "Duplicate merged evaluation rows")
    wanted = {(model, seed, case) for model, seed in expected_runs for case in cases}
    _require(set(frame[index_keys].itertuples(index=False, name=None)) == wanted,
             "Merged evaluation index is not the full checkpoint/case grid")
    for name in METRICS:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    records = frame.to_dict("records")
    for record in records:
        checkpoint = checkpoint_lookup[(record["model"], int(record["seed"]))]
        case = cases[record["case_id"]]
        _require(record["checkpoint_sha256"] == checkpoint["sha256"]
                 and record["input_sha256"] == case["input_sha256"], "Merged evaluation identity mismatch")
        _prediction(record, checkpoint, case, reference, cfg["_config_sha256"])
    fingerprints = dict(
        manifest=file_info(manifest_path), cohort=file_info(paths["inputs"] / "cohort.npz"),
        draws=file_info(draws_path), evaluation_index=file_info(index_path),
        checkpoints=file_info(paths["inputs"] / "checkpoints.json"),
        freeze=file_info(paths["logs"] / "freeze.json"),
    )
    return cases, logical, clean_cases[0], checkpoint_lookup, records, fingerprints, freeze


def _cache_identity(record, checkpoint, case, shared):
    return dict(
        **shared, model=record["model"], seed=int(record["seed"]), case_id=record["case_id"],
        prediction_sha256=record["prediction_sha256"], input_sha256=case["input_sha256"],
        checkpoint_sha256=checkpoint["sha256"], thresholds=checkpoint["thresholds"],
    )


def _audit_point(values, record):
    observed = np.asarray([record[name] for name in METRICS], dtype=np.float64)
    _require(np.allclose(values[0], observed, atol=POINT_ATOL, rtol=0, equal_nan=True),
             f"Bootstrap point disagrees with saved metrics: {record['model']}/{record['seed']}/{record['case_id']}")


def _checkpoint_distributions(task):
    cfg, stage, checkpoint, records, cases, shared = task
    paths = stage_paths(cfg, stage)
    reference = load_reference(cfg, stage)
    draws = np.load(paths["inputs"] / "patient_draws.npy", mmap_mode="r", allow_pickle=False)
    root = paths["bootstrap"] / "new" / checkpoint["model"] / f"seed_{checkpoint['seed']}"
    root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for record in records:
        case = cases[record["case_id"]]
        identity = _cache_identity(record, checkpoint, case, shared)
        destination = root / f"{case['case_id']}.npy"
        sidecar = destination.with_suffix(".json")
        resumed = False
        if destination.exists() and sidecar.exists():
            try:
                saved = read_json(sidecar)
                if (saved.get("status") == "completed" and saved.get("identity") == identity
                        and sha256(destination) == saved["distribution"]["sha256"]):
                    values = np.load(destination, mmap_mode="r", allow_pickle=False)
                    resumed = values.shape == (len(draws) + 1, 3) and values.dtype == np.float64
                    if resumed:
                        _audit_point(values, record)
                    del values
            except (ValueError, OSError, EOFError, KeyError):
                resumed = False
        if not resumed:
            p, thresholds = _prediction(record, checkpoint, case, reference, cfg["_config_sha256"])
            values = bootstrap.metric_distribution(
                reference["y"], p, thresholds, reference["patient_inverse"], draws,
                batch_size=cfg["statistics"]["bootstrap_batch_size"],
            )
            _require(values.shape == (len(draws) + 1, 3) and values.dtype == np.float64,
                     "Unexpected metric_distribution shape or dtype")
            _require(not np.isinf(values).any(), "Infinite bootstrap metric")
            _audit_point(values, record)
            temporary = destination.with_suffix(".partial.npy")
            with temporary.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            temporary.replace(destination)
            save_json(sidecar, dict(status="completed", identity=identity, distribution=file_info(destination)))
            del p, values
        outputs.append(dict(model=record["model"], seed=int(record["seed"]),
                            case_id=record["case_id"], resumed=resumed, distribution=file_info(destination)))
    return outputs


def _groups(cfg):
    for snr in cfg["snrs"]:
        yield ("snr", "standard", "standard", "legacy_0p5_40", snr, ["standard"])
    random_ids = [f"random_{i:02d}" for i in range(cfg["matrix"]["random_replicates"])]
    for family in cfg["matrix"]["families"]:
        for snr in cfg["ablation_snrs"]:
            yield ("matrix", family, family, "legacy_0p5_40", snr,
                   random_ids if family == "randomized" else [family])
    for band in cfg["bands"]:
        for snr in cfg["ablation_snrs"]:
            yield ("band", band["id"], "standard", band["id"], snr, [band["id"]])
    for source in random_ids:
        for snr in cfg["ablation_snrs"]:
            yield ("matrix_instance", source, "randomized", "legacy_0p5_40", snr, [source])


def _outcomes(electrode, independent, clean):
    return {
        "structure_effect": electrode - independent,
        "electrode_absolute": electrode,
        "independent_rms_absolute": independent,
        "electrode_drop": clean - electrode,
        "independent_rms_drop": clean - independent,
    }


def _tables(cfg, stage, logical, clean_case, cache):
    seeds = cfg["phase1_training_seeds"]
    noises = cfg["phase1_noise_seeds"]
    count = cfg["stages"][stage]["bootstrap_replicates"] + 1
    lookup = {(item["model"], item["seed"], item["case_id"]): resolve_path(item["distribution"]["path"])
              for item in cache}
    summary, seed_rows, noise_rows, matrix_rows = [], [], [], []
    for model in cfg["models"]:
        clean = np.stack([np.load(lookup[(model, seed, clean_case)], allow_pickle=False) for seed in seeds])
        for analysis, source, family, band, snr, members in _groups(cfg):
            conditions = {}
            for condition in cfg["conditions"]:
                values = np.zeros((len(seeds), len(noises), count, len(METRICS)), dtype=np.float64)
                for s, seed in enumerate(seeds):
                    for n, noise in enumerate(noises):
                        for member in members:
                            case_id = logical[(member, snr, noise, condition)]
                            part = np.load(lookup[(model, seed, case_id)], allow_pickle=False)
                            values[s, n] += part
                        values[s, n] /= len(members)
                conditions[condition] = values
            for outcome, per_noise in _outcomes(conditions["electrode"], conditions["independent_rms"], clean[:, None]).items():
                seed_distributions = per_noise.mean(axis=1)
                for metric_index, metric in enumerate(METRICS):
                    base = dict(analysis=analysis, phase="phase1", model=model, strategy="clean_only",
                                source_id=source, matrix_family=family, band_id=band, snr=snr,
                                metric=metric, outcome=outcome)
                    distributions = seed_distributions[:, :, metric_index]
                    stats = summarize_distribution(distributions[:, 0], distributions)
                    noise_points = per_noise[:, :, 0, metric_index].mean(axis=0)
                    summary.append({**base, **stats, "noise_sd": float(noise_points.std(ddof=1)),
                                    "random_matrix_sd": None})
                    for s, seed in enumerate(seeds):
                        seed_row = dict(**base, seed=seed, value=float(distributions[s, 0]),
                                        n_noise=len(noises), n_matrices=len(members))
                        seed_rows.append(seed_row)
                        if analysis == "matrix_instance":
                            matrix_rows.append(dict(**seed_row, matrix_id=source, replicate=int(source.rsplit("_", 1)[1])))
                        for n, noise in enumerate(noises):
                            noise_rows.append(dict(**base, seed=seed, noise_seed=noise,
                                                   value=float(per_noise[s, n, 0, metric_index]),
                                                   n_matrices=len(members)))
    instances = {}
    for row in summary:
        if row["analysis"] == "matrix_instance":
            key = (row["model"], row["snr"], row["metric"], row["outcome"])
            instances.setdefault(key, []).append(row["estimate"])
    for row in summary:
        if row["analysis"] == "matrix" and row["source_id"] == "randomized":
            values = np.asarray(instances[(row["model"], row["snr"], row["metric"], row["outcome"])] , dtype=float)
            _require(len(values) == cfg["matrix"]["random_replicates"], "Incomplete randomized matrix summary")
            row["random_matrix_sd"] = float(values.std(ddof=1))
    frames = {
        "new_summary": pd.DataFrame(summary, columns=SUMMARY_FIELDS),
        "new_seed_effects": pd.DataFrame(seed_rows),
        "new_noise_effects": pd.DataFrame(noise_rows),
        "random_matrix_effects": pd.DataFrame(matrix_rows),
    }
    unique_keys = {
        "new_summary": KEYS, "new_seed_effects": [*KEYS, "seed"],
        "new_noise_effects": [*KEYS, "seed", "noise_seed"],
        "random_matrix_effects": [*KEYS, "seed", "matrix_id"],
    }
    groups = list(_groups(cfg))
    expected_summary = len(groups) * len(cfg["models"]) * len(METRICS) * len(OUTCOMES)
    expected_counts = dict(new_summary=expected_summary,
                           new_seed_effects=expected_summary * len(seeds),
                           new_noise_effects=expected_summary * len(seeds) * len(noises),
                           random_matrix_effects=sum(g[0] == "matrix_instance" for g in groups)
                           * len(cfg["models"]) * len(METRICS) * len(OUTCOMES) * len(seeds))
    for name, frame in frames.items():
        _require(len(frame) == expected_counts[name] and not frame.duplicated(unique_keys[name]).any(),
                 f"Missing or duplicated output rows: {name}")
        frames[name] = frame.sort_values(unique_keys[name], kind="stable").reset_index(drop=True)
    return frames


def run(config_path=None, stage="full"):
    cfg = load_config(config_path)
    paths = stage_paths(cfg, stage)
    for category in ("tables", "logs", "bootstrap"):
        paths[category].mkdir(parents=True, exist_ok=True)
    log_path = paths["logs"] / "new_statistics.json"
    started = _now()
    running = dict(status="running", stage=stage, config_sha256=cfg["_config_sha256"], started_at=started)
    save_json(log_path, running)
    try:
        _require(tuple(bootstrap.METRICS) == METRICS, "Bootstrap metric order differs from analysis contract")
        cases, logical, clean_case, checkpoints, records, fingerprints, _ = _verify_inputs(cfg, stage, paths)
        sources = [Path(__file__), Path(bootstrap.__file__), Path(__file__).with_name("common.py"),
                   WORKSPACE / "phase1_ecg_robustness/src/evaluate.py",
                   WORKSPACE / "phase1_ecg_robustness/src/statistics.py",
                   WORKSPACE / "phase2/src/statistics_phase2.py"]
        source_info = [file_info(path) for path in sources]
        shared = dict(cache_version=CACHE_VERSION, stage=stage, config_sha256=cfg["_config_sha256"],
                      cohort_sha256=fingerprints["cohort"]["sha256"],
                      draws_sha256=fingerprints["draws"]["sha256"],
                      source_sha256=[info["sha256"] for info in source_info],
                      metrics=list(METRICS), point_absolute_tolerance=POINT_ATOL,
                      numpy_version=np.__version__)
        tasks = [(cfg, stage, checkpoint,
                  [row for row in records if (row["model"], int(row["seed"])) == key], cases, shared)
                 for key, checkpoint in sorted(checkpoints.items())]
        workers = max(1, min(int(cfg["statistics"]["workers"]), len(tasks), os.cpu_count() or 1))
        cache = []
        if workers == 1:
            for task in tasks:
                cache.extend(_checkpoint_distributions(task))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for result in executor.map(_checkpoint_distributions, tasks):
                    cache.extend(result)
        frames = _tables(cfg, stage, logical, clean_case, cache)
        # No completion stamp is published if any upstream identity changed mid-run.
        for info in fingerprints.values():
            _check_info(info)
        for info in source_info:
            _check_info(info)
        require_freeze(cfg, stage)
        outputs = {}
        for name, frame in frames.items():
            path = paths["tables"] / f"{name}.csv"
            write_csv(path, frame)
            outputs[name] = dict(**file_info(path), rows=len(frame))
        result = dict(
            **running, finished_at=_now(), workers=workers,
            software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
            sources=source_info, inputs=fingerprints, outputs=outputs,
            checkpoint_hashes=[dict(model=cp["model"], seed=int(cp["seed"]), sha256=cp["sha256"],
                                    thresholds=cp["thresholds"]) for cp in checkpoints.values()],
            n_cases=len(cases), n_checkpoints=len(checkpoints), n_distributions=len(cache),
            n_resumed=sum(item["resumed"] for item in cache), distributions=cache,
            metrics=list(METRICS), outcomes=list(OUTCOMES), summary_fields=SUMMARY_FIELDS,
            training_seeds=cfg["phase1_training_seeds"], noise_seeds=cfg["phase1_noise_seeds"],
            uncertainty=dict(
                seed_sd="Sample SD (ddof=1) of three paired training-seed estimates after equal noise/matrix averaging.",
                patient_ci="95% percentile paired patient-cluster CI of the mean over fixed checkpoints, noises and matrices. Ordinary means propagate undefined draws; invalid counts retained. Not joint training/patient uncertainty.",
                noise_sd="Descriptive sample SD across six fixed noise realizations after averaging training seeds and, for the randomized family, five matrices.",
                random_matrix_sd="Descriptive sample SD across five fixed matrix estimates after averaging seeds and noises. Present only on randomized-family rows; not independent training runs or a random-matrix population CI.",
            ),
            signs=dict(
                structure_effect="Electrode minus independent-RMS: positive AUROC/F1 is better; positive ECE is worse.",
                electrode_drop="Clean minus electrode: positive AUROC/F1 is degradation; positive ECE is improved calibration under noise, not degradation.",
                independent_rms_drop="Clean minus independent-RMS: positive AUROC/F1 is degradation; positive ECE is improved calibration under noise, not degradation.",
                absolute="Larger AUROC/F1 is better; larger ECE is worse. ECE is the raw classwise 15-bin metric, never negated.",
            ),
            aggregation="Metrics, not probabilities, are equally averaged. Five randomized matrices are averaged within each seed/noise/patient draw, then six fixed noises, then the three fixed training checkpoints.",
            detail_tables=dict(
                new_seed_effects="One row per summary key/training seed; noises and family matrices equally averaged.",
                new_noise_effects="One row per summary key/training seed/noise realization; family matrices equally averaged. Average training seeds before recomputing descriptive noise SD.",
                random_matrix_effects="Individual random-matrix estimates by training seed after equal noise averaging; average training seeds before recomputing descriptive matrix SD.",
            ),
            inference="Exploratory descriptive estimates only. No p-values, equality/equivalence claims or extra seed replicates.",
        )
        result["status"] = "completed"
        save_json(log_path, result)
        return result
    except Exception as exc:
        failed = dict(running, status="failed", finished_at=_now(), error=f"{type(exc).__name__}: {exc}")
        save_json(log_path, failed)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
