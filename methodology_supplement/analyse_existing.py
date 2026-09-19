"""Descriptive phase-two reuse: paired SNR effects, A4 plane, and Pareto.

Only the completed full phase-two experiment is supported. No inference, training,
new hypothesis tests, probability ensembles, or reads of patient_ci.csv occur.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_CONFIG,
    array_sha256,
    file_info,
    load_config,
    load_reference,
    read_json,
    require_freeze,
    resolve_path,
    save_json,
    sha256,
    stage_paths,
    summarize_distribution,
    write_csv,
)
from phase2.src.common import (
    load_config as load_phase2_config,
    require_preregistration,
    stage_paths as phase2_paths,
    verify_training_complete,
)
from phase2.src.evaluate_phase2 import CASE_FIELDS, _expected_case_keys
from phase2.src.generate_phase2_noise import _attach_groups, load_test_manifest

from .bootstrap import METRICS, metric_distribution

RUN_KEYS = ["strategy", "model", "seed"]
CELL_KEYS = [*RUN_KEYS, "case_id"]
SUMMARY_FIELDS = [
    "analysis", "phase", "model", "strategy", "source_id", "matrix_family",
    "band_id", "snr", "metric", "outcome", "estimate", "seed_sd", "seed_n",
    "patient_ci_low", "patient_ci_high", "n_bootstrap", "n_invalid",
    "noise_sd", "random_matrix_sd",
]
PLANE_FIELDS = [
    "model", "strategy", "kind", "combo_set", "combo_id", "snr", "metric",
    "x_mean", "x_seed_sd", "x_ci_low", "x_ci_high", "y_mean", "y_seed_sd",
    "y_ci_low", "y_ci_high", "n_seeds", "n_bootstrap", "n_invalid_x", "n_invalid_y",
]
PARETO_FIELDS = [
    "model", "strategy", "x_mean", "x_seed_sd", "x_ci_low", "x_ci_high",
    "y_mean", "y_seed_sd", "y_ci_low", "y_ci_high", "n_seeds", "n_bootstrap",
    "n_invalid_x", "n_invalid_y", "clean_auroc", "joint_unseen_auroc", "point_dominated",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _checked_source(item, checked):
    """Hash once per invocation, rejecting conflicting recorded identities."""
    path = resolve_path(item["path"])
    key = str(path.resolve())
    if key not in checked:
        checked[key] = file_info(path)
    actual = checked[key]
    _require(actual["sha256"] == item["sha256"] and actual["bytes"] == item["bytes"],
             f"Archived source fingerprint mismatch: {path}")
    return path


def _run_key(item):
    return item["strategy"], item["model"], int(item["seed"])


def _load_sources(cfg, paths):
    """Validate source registration and identities without loading ECG tensors."""
    old = load_phase2_config(resolve_path(cfg["phase2"]["root"]) / "configs/phase2_main.yaml")
    _require(cfg["phase2"]["stage"] == "full", "Reuse requires completed full phase two")
    _require(old["_config_sha256"] == cfg["phase2"]["config_sha256"], "Registered phase-two SHA changed")
    for actual, expected, name in (
        (old["class_order"], cfg["class_order"], "class order"),
        (old["models"], cfg["models"], "model order"),
        (list(old["strategies"]), cfg["strategies"], "strategy order"),
        (old["stages"]["full"]["seeds"], cfg["phase2_training_seeds"], "training seed order"),
        (old["stages"]["full"]["test_snrs"], cfg["snrs"], "SNR order"),
        (old["statistics"]["bootstrap_replicates"], cfg["statistics"]["bootstrap_replicates"], "draw count"),
        (old["statistics"]["bootstrap_seed"], cfg["statistics"]["bootstrap_seed"], "draw seed"),
    ):
        _require(actual == expected, f"Supplement/phase-two {name} mismatch")
    _require(list(METRICS) == ["macro_auroc", "macro_f1", "ece"], "Bootstrap metric order changed")
    registration = require_preregistration(old)
    training = {_run_key(item): item for item in verify_training_complete(old, "full")}
    source_paths = phase2_paths(old, "full")
    checked = {}
    evaluation_path = source_paths["logs"] / "evaluation_protocol.json"
    statistics_path = source_paths["logs"] / "statistics_protocol.json"
    evaluation, statistics = read_json(evaluation_path), read_json(statistics_path)
    manifest = load_test_manifest(old, "full")
    for name, protocol in (("evaluation", evaluation), ("statistics", statistics)):
        for key, value in (("status", "completed"), ("stage", "full"),
                           ("config_sha256", old["_config_sha256"]),
                           ("matrix_sha256", old["matrix_sha256"])):
            _require(protocol.get(key) == value, f"Incomplete/mismatched {name} protocol: {key}")
        for item in protocol["sources"]:
            _checked_source(item, checked)
    for item in evaluation["outputs"].values():
        _checked_source(item, checked)
    _checked_source(evaluation["manifest"], checked)
    _require(evaluation["data_identity"] == manifest["data_identity"] == registration["datasets"]["full"],
             "Evaluation/noise/registration cohort identity mismatch")
    _require(evaluation["data_identity"]["class_order"] == cfg["class_order"]
             and evaluation["data_identity"]["lead_order"] == old["lead_order"], "Class/lead ordering changed")
    _require(evaluation["thresholds_source"] == "best_checkpoint_clean_validation_once"
             and evaluation["threshold_comparison"] == ">=", "Frozen threshold convention changed")
    for flag in ("full_cartesian_coverage", "input_hash_invariance", "cohort_label_invariance", "threshold_invariance"):
        _require(evaluation.get(flag) is True, f"Incomplete evaluation invariant: {flag}")

    cases = {item["case_id"]: item for item in manifest["cases"]}
    _require(len(cases) == len(manifest["cases"]), "Duplicate archived case IDs")
    actual_keys = [tuple(item[key] for key in CASE_FIELDS) for item in manifest["cases"]]
    _require(len(actual_keys) == len(set(actual_keys)) and set(actual_keys) == _expected_case_keys(old, "full"),
             "Archived case grid differs from registration")
    groups = manifest["groups"]
    _require(groups == _attach_groups(old, "full", copy.deepcopy(manifest["cases"])),
             "Archived group membership/order differs from registration")
    group_ids = [group["group_id"] for group in groups]
    _require(len(group_ids) == len(set(group_ids)), "Duplicate archived group IDs")

    bootstrap_path = source_paths["tables"] / "patient_bootstrap/manifest.json"
    bootstrap = read_json(bootstrap_path)
    _require(bootstrap == statistics["bootstrap"], "Bootstrap manifest differs from completed statistics")
    _require(bootstrap["axis_order"] == ["registered_group", "point_then_common_patient_draw", "auc_metric"]
             and bootstrap["group_ids"] == group_ids
             and bootstrap["metrics"] == ["macro_auroc", *[f"auroc_{name}" for name in cfg["class_order"]]],
             "Stored distribution axis/order mismatch")
    cohort_path = _checked_source(bootstrap["cohort"], checked)
    with np.load(cohort_path, allow_pickle=False) as saved:
        reference = {key: saved[key] for key in saved.files}
    with np.load(_checked_source(manifest["cohort_file"], checked), allow_pickle=False) as saved:
        for key in ("y", "ids", "patient_ids", "indices"):
            _require(np.array_equal(saved[key], reference[key]), f"Bootstrap/input cohort mismatch: {key}")
            _require(array_sha256(saved[key]) == evaluation["cohort_sha256"][key],
                     f"Evaluation cohort hash mismatch: {key}")
    supplement_reference = load_reference(cfg, "full")
    for key in ("y", "ids", "patient_ids", "indices", "unique_patients", "patient_inverse"):
        _require(np.array_equal(reference[key], supplement_reference[key]), f"Supplement cohort mismatch: {key}")
    unique, inverse = np.unique(reference["patient_ids"], return_inverse=True)
    _require(np.array_equal(unique, reference["unique_patients"])
             and np.array_equal(inverse, reference["patient_inverse"]), "Patient mapping/order mismatch")
    draws_path = _checked_source(bootstrap["draws"], checked)
    _require(sha256(paths["inputs"] / "patient_draws.npy") == bootstrap["draws"]["sha256"],
             "Supplement must use byte-identical full phase-two patient draws")
    draws = np.load(draws_path, mmap_mode="r", allow_pickle=False)
    _require(draws.shape == (cfg["statistics"]["bootstrap_replicates"], len(unique))
             and draws.dtype.kind in "iu" and np.all(draws >= 0)
             and np.all(draws.sum(axis=1) == len(unique)), "Invalid patient multiplicity draws")

    checkpoints = {}
    for item in evaluation["checkpoints"]:
        key = _run_key(item)
        _require(key not in checkpoints and key in training, "Duplicate/unknown checkpoint")
        _checked_source(item["checkpoint"], checked)
        _checked_source(item["validation_predictions"], checked)
        _require(item["checkpoint"]["sha256"] == training[key]["checkpoint_sha256"]
                 and np.array_equal(item["thresholds"], training[key]["thresholds"]),
                 f"Training/evaluation checkpoint or thresholds changed: {key}")
        checkpoints[key] = item
    _require(set(checkpoints) == set(training), "Missing evaluated checkpoint")
    _require({_run_key(item): item["checkpoint_sha256"] for item in manifest["training_completed_before_generation"]}
             == {key: item["checkpoint"]["sha256"] for key, item in checkpoints.items()},
             "Noise generation/checkpoint identities changed")

    columns = [*CELL_KEYS, "stage", *CASE_FIELDS, "prediction_path", "prediction_sha256",
               "checkpoint_sha256", "input_sha256", *METRICS]
    metrics = pd.read_csv(source_paths["tables"] / "metrics.csv", usecols=columns,
                          keep_default_na=False, na_values=["", "nan", "NaN"])
    _require(not metrics.duplicated(CELL_KEYS).any() and metrics.stage.eq("full").all(),
             "Duplicate or wrong-stage evaluation rows")
    _require(len(metrics) == len(cases) * len(checkpoints), "Incomplete evaluation grid")
    for key, frame in metrics.groupby(RUN_KEYS, sort=False):
        _require(key in checkpoints and set(frame.case_id) == set(cases), f"Incomplete evaluation run: {key}")
        _require(frame.checkpoint_sha256.eq(checkpoints[key]["checkpoint"]["sha256"]).all(),
                 f"Evaluation checkpoint ledger mismatch: {key}")
    for field in [*CASE_FIELDS, "input_sha256"]:
        expected = metrics.case_id.map({key: value[field] for key, value in cases.items()})
        _require(metrics[field].eq(expected).all(), f"Case metadata changed: {field}")
    # The saved AUROC tensors record these exact archived prediction identities.
    provenance_item = next(item for item in statistics["outputs"]
                           if Path(item["path"]).name == "prediction_provenance.csv")
    provenance = pd.read_csv(_checked_source(provenance_item, checked),
                             usecols=[*CELL_KEYS, "path", "sha256"], keep_default_na=False)
    _require(not provenance.duplicated(CELL_KEYS).any(), "Duplicate bootstrap prediction provenance")
    ledger = metrics.set_index(CELL_KEYS).sort_index()
    provenance = provenance.set_index(CELL_KEYS).sort_index()
    _require(ledger.index.equals(provenance.index)
             and np.array_equal(ledger.prediction_path, provenance.path)
             and np.array_equal(ledger.prediction_sha256, provenance.sha256),
             "AUROC bootstrap and evaluation prediction identities differ")
    distributions = {}
    for item in bootstrap["distributions"]:
        key = tuple(item["run"])
        _require(key in checkpoints and key not in distributions, "Duplicate/unknown distribution run")
        distributions[key] = _checked_source(item["distribution"], checked)
    _require(set(distributions) == set(checkpoints), "Incomplete bootstrap run grid")
    for path in (evaluation_path, statistics_path, bootstrap_path,
                 source_paths["logs"].parent / "preregistration.json"):
        checked[str(path.resolve())] = file_info(path)
    return dict(old=old, cases=cases, groups=groups, bootstrap=bootstrap, reference=reference,
                draws=draws, checkpoints=checkpoints, ledger=ledger, distributions=distributions,
                checked=checked, evaluation=evaluation)


def _individual_groups(source):
    """'all' is archived as aggregate but contains one individual combination."""
    lookup = {}
    for index, group in enumerate(source["groups"]):
        if group["kind"] == "clean" or group["group_id"] == "primary_joint":
            continue
        if group["combo_id"] == "aggregate" and group["combo_set"] != "all":
            continue
        combo = "all" if group["combo_set"] == "all" else group["combo_id"]
        key = group["kind"], group["combo_set"], combo, int(group["snr"]), group["condition"]
        _require(key not in lookup, f"Duplicate individual group: {key}")
        members = [source["cases"][case_id] for case_id in group["case_ids"]]
        _require(len(members) == 5 and {int(case["noise_seed"]) for case in members}
                 == set(source["old"]["stages"]["full"]["test_noise_seeds"]),
                 f"Individual group does not have five paired noises: {key}")
        lookup[key] = index
    return lookup


def _load_auc(source, run):
    array = np.load(source["distributions"][run], mmap_mode="r", allow_pickle=False)
    _require(array.shape == (len(source["groups"]), len(source["draws"]) + 1, 6)
             and array.dtype == np.float64, f"Stored AUROC tensor shape/dtype mismatch: {run}")
    ledger = source["ledger"].loc[run]
    # Check point estimates against equal case-level averaging, not averaged probabilities.
    for index, group in enumerate(source["groups"]):
        point = ledger.loc[group["case_ids"], "macro_auroc"].to_numpy().mean()
        _require(np.isclose(array[index, 0, 0], point, rtol=0, atol=2e-7, equal_nan=True),
                 f"Stored AUROC group point mismatch: {run}/{group['group_id']}")
    return array


def _extra_distribution(cfg, paths, source, run, case_id, cache_stats):
    record = source["ledger"].loc[(*run, case_id)]
    prediction_path = resolve_path(record.prediction_path)
    prediction_info = file_info(prediction_path)
    _require(prediction_info["sha256"] == record.prediction_sha256,
             f"Archived prediction fingerprint mismatch: {prediction_path}")
    checkpoint = source["checkpoints"][run]
    identity = dict(
        schema="phase2_f1_ece_patient_v1", config_sha256=cfg["_config_sha256"],
        phase2_config_sha256=source["old"]["_config_sha256"], run=list(run), case_id=case_id,
        prediction=prediction_info, checkpoint_sha256=checkpoint["checkpoint"]["sha256"],
        input_sha256=record.input_sha256, thresholds=checkpoint["thresholds"],
        draws=source["bootstrap"]["draws"], cohort=source["bootstrap"]["cohort"],
        implementation=cache_stats["implementation"],
    )
    directory = paths["bootstrap"] / "phase2_extra" / run[0] / run[1] / f"seed_{run[2]}"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{case_id}.npy"
    metadata_path = directory / f"{case_id}.json"
    if destination.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata.get("identity") == identity and file_info(destination) == metadata.get("distribution"):
            values = np.load(destination, allow_pickle=False)
            if values.shape == (len(source["draws"]) + 1, 2) and np.isfinite(values).all():
                cache_stats["reused"] += 1
                return values
    reference = source["reference"]
    case = source["cases"][case_id]
    with np.load(prediction_path, allow_pickle=False) as saved:
        for key in ("y", "ids", "patient_ids", "indices"):
            _require(np.array_equal(saved[key], reference[key]), f"Prediction cohort mismatch: {case_id}/{key}")
        expected_scalars = dict(
            stage="full", strategy=run[0], model=run[1], seed=run[2], case_id=case_id,
            config_sha256=source["old"]["_config_sha256"],
            matrix_sha256=source["old"]["matrix_sha256"], input_sha256=record.input_sha256,
            checkpoint_sha256=checkpoint["checkpoint"]["sha256"],
            evaluation_fingerprint=source["evaluation"]["evaluation_fingerprint"],
            **{field: case[field] for field in CASE_FIELDS},
        )
        for key, value in expected_scalars.items():
            _require(saved[key].item() == value, f"Prediction identity mismatch: {case_id}/{key}")
        thresholds = saved["thresholds"]
        _require(np.array_equal(thresholds, np.asarray(checkpoint["thresholds"], dtype=thresholds.dtype)),
                 f"Prediction threshold mismatch: {case_id}")
        p = saved["p"]
        _require(p.shape == reference["y"].shape and np.isfinite(p).all()
                 and np.all((p >= 0) & (p <= 1)) and array_sha256(p) == saved["p_sha256"].item(),
                 f"Invalid archived probabilities: {case_id}")
        values = metric_distribution(reference["y"], p, thresholds, reference["patient_inverse"],
                                     source["draws"], batch_size=cfg["statistics"]["bootstrap_batch_size"],
                                     compute_auc=False)[:, 1:]
    _require(np.allclose(values[0], record[["macro_f1", "ece"]].to_numpy(dtype=float),
                        rtol=0, atol=2e-7, equal_nan=True), f"F1/ECE point disagrees with archive: {case_id}")
    _require(file_info(prediction_path) == prediction_info, f"Prediction changed while reading: {prediction_path}")
    temporary = destination.with_suffix(".partial.npy")
    np.save(temporary, values, allow_pickle=False)
    temporary.replace(destination)
    save_json(metadata_path, dict(identity=identity, distribution=file_info(destination)))
    cache_stats["computed"] += 1
    return values


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return summarize_distribution(values[:, 0], values)


def _axes(x, y):
    sx, sy = _summary(x), _summary(y)
    result = {f"{axis}_{name}": summary[key]
              for axis, summary in (("x", sx), ("y", sy))
              for name, key in (("mean", "estimate"), ("seed_sd", "seed_sd"),
                                ("ci_low", "patient_ci_low"), ("ci_high", "patient_ci_high"))}
    result.update(n_seeds=sx["seed_n"], n_bootstrap=sx["n_bootstrap"],
                  n_invalid_x=sx["n_invalid"], n_invalid_y=sy["n_invalid"])
    return result


def _snr_tables(cfg, paths, source, lookup, cache_stats):
    rows, seed_rows = [], []
    seeds = cfg["phase2_training_seeds"]
    noise_seeds = source["old"]["stages"]["full"]["test_noise_seeds"]
    for model in cfg["models"]:
        for strategy in cfg["strategies"]:
            # Only this strategy/model's five tiny SNR tensors stay resident.
            effects = np.empty((len(seeds), len(cfg["snrs"]), len(source["draws"]) + 1, 3))
            noise_effects = np.empty((len(seeds), len(cfg["snrs"]), len(noise_seeds), 3))
            for seed_index, seed in enumerate(seeds):
                run = strategy, model, seed
                auc = _load_auc(source, run)
                for snr_index, snr in enumerate(cfg["snrs"]):
                    condition_values, condition_points = {}, {}
                    for condition in cfg["conditions"]:
                        index = lookup[("bandpass", "all", "all", snr, condition)]
                        group = source["groups"][index]
                        distribution = np.zeros((len(source["draws"]) + 1, 3))
                        distribution[:, 0] = auc[index, :, 0]
                        points = []
                        by_noise = {source["cases"][case_id]["noise_seed"]: case_id for case_id in group["case_ids"]}
                        for noise_seed in noise_seeds:
                            case_id = by_noise[noise_seed]
                            distribution[:, 1:] += _extra_distribution(cfg, paths, source, run, case_id, cache_stats)
                            points.append(source["ledger"].loc[(*run, case_id), list(METRICS)].to_numpy(dtype=float))
                        distribution[:, 1:] /= len(noise_seeds)
                        condition_values[condition] = distribution
                        condition_points[condition] = np.asarray(points)
                    effects[seed_index, snr_index] = condition_values["electrode"] - condition_values["independent_rms"]
                    noise_effects[seed_index, snr_index] = condition_points["electrode"] - condition_points["independent_rms"]
                del auc
            for snr_index, snr in enumerate(cfg["snrs"]):
                for metric_index, metric in enumerate(METRICS):
                    values = effects[:, snr_index, :, metric_index]
                    descriptors = dict(analysis="snr", phase="phase2", model=model, strategy=strategy,
                                       source_id="standard", matrix_family="standard", band_id="legacy_0p5_40",
                                       snr=snr, metric=metric, outcome="structure_effect")
                    rows.append({**descriptors, **_summary(values),
                                 "noise_sd": float(noise_effects[:, snr_index, :, metric_index].mean(axis=0).std(ddof=1)),
                                 "random_matrix_sd": None})
                    for seed_index, seed in enumerate(seeds):
                        seed_rows.append({**descriptors, "seed": seed, "value": float(values[seed_index, 0]),
                                          "n_noise": len(noise_seeds)})
    return rows, seed_rows


def _plane_and_pareto(cfg, source, lookup):
    plane, plane_seeds, pareto, pareto_seeds = [], [], [], []
    seeds = cfg["phase2_training_seeds"]
    strata = [key[:-1] for key in lookup if key[-1] == "electrode"]
    # Explicitly group Gaussian main rows before confounded NSTDB sensitivity rows.
    kind_order = {"bandpass": 0, "bw": 1, "ma": 2, "em": 3}
    strata.sort(key=lambda key: (kind_order[key[0]], key[1], key[2], -key[3]))
    indices = {group["group_id"]: index for index, group in enumerate(source["groups"])}
    joint = source["groups"][indices["primary_joint"]]
    _require(joint["primary"] and joint["kind"] == "bandpass" and joint["condition"] == "electrode"
             and joint["combo_set"] == "heldout" and len(joint["case_ids"]) == 100,
             "Primary joint group differs from registered estimand")
    epsilon = cfg["phase2"]["retention_epsilon"]
    for model in cfg["models"]:
        # Keep only required macro-AUROC slices, not forty entire six-metric tensors.
        needed = [indices["clean"], indices["primary_joint"]]
        needed += [lookup[(*stratum, condition)] for stratum in strata for condition in cfg["conditions"]]
        needed = list(dict.fromkeys(needed))
        positions = {index: position for position, index in enumerate(needed)}
        baseline = []
        for seed in seeds:
            array = _load_auc(source, ("clean_only", model, seed))
            baseline.append(np.asarray(array[needed, :, 0]))
            del array
        baseline = np.asarray(baseline)
        clean_baseline = baseline[:, positions[indices["clean"]]]
        joint_baseline = baseline[:, positions[indices["primary_joint"]]]
        retention_baseline = joint_baseline / (clean_baseline + epsilon)
        for strategy in cfg["strategies"]:
            if strategy == "clean_only":
                augmented = baseline
            else:
                augmented = []
                for seed in seeds:
                    array = _load_auc(source, (strategy, model, seed))
                    augmented.append(np.asarray(array[needed, :, 0]))
                    del array
                augmented = np.asarray(augmented)
            clean = augmented[:, positions[indices["clean"]]]
            noisy = augmented[:, positions[indices["primary_joint"]]]
            retention = noisy / (clean + epsilon)
            x, y = clean_baseline - clean, retention - retention_baseline
            pareto.append(dict(model=model, strategy=strategy, **_axes(x, y),
                               clean_auroc=float(clean[:, 0].mean()), joint_unseen_auroc=float(noisy[:, 0].mean()),
                               point_dominated=False))
            for seed_index, seed in enumerate(seeds):
                pareto_seeds.append(dict(model=model, strategy=strategy, seed=seed,
                                         x=float(x[seed_index, 0]), y=float(y[seed_index, 0]),
                                         clean_auroc=float(clean[seed_index, 0]),
                                         joint_unseen_auroc=float(noisy[seed_index, 0]),
                                         retention=float(retention[seed_index, 0]),
                                         baseline_clean_auroc=float(clean_baseline[seed_index, 0]),
                                         baseline_joint_unseen_auroc=float(joint_baseline[seed_index, 0]),
                                         baseline_retention=float(retention_baseline[seed_index, 0])))
            if strategy == "clean_only":
                continue
            for stratum in strata:
                kind, combo_set, combo, snr = stratum
                electrode = positions[lookup[(*stratum, "electrode")]]
                independent = positions[lookup[(*stratum, "independent_rms")]]
                x = baseline[:, electrode] - baseline[:, independent]
                y = augmented[:, electrode] - baseline[:, electrode]
                descriptors = dict(model=model, strategy=strategy, kind=kind, combo_set=combo_set,
                                   combo_id=combo, snr=snr, metric="macro_auroc")
                plane.append({**descriptors, **_axes(x, y)})
                for seed_index, seed in enumerate(seeds):
                    plane_seeds.append({**descriptors, "seed": seed, "x": float(x[seed_index, 0]),
                                        "y": float(y[seed_index, 0]),
                                        "baseline_electrode_auroc": float(baseline[seed_index, electrode, 0]),
                                        "baseline_independent_rms_auroc": float(baseline[seed_index, independent, 0]),
                                        "strategy_electrode_auroc": float(augmented[seed_index, electrode, 0])})
            del augmented
        del baseline
    for row in pareto:
        row["point_dominated"] = any(
            other["model"] == row["model"]
            and other["x_mean"] <= row["x_mean"] and other["y_mean"] >= row["y_mean"]
            and (other["x_mean"] < row["x_mean"] or other["y_mean"] > row["y_mean"])
            for other in pareto
        )
    return plane, plane_seeds, pareto, pareto_seeds


def run(config_path=None, stage="full"):
    _require(stage == "full", "Archived analyses support full only; no smoke subset is claimed as complete")
    cfg = load_config(config_path)
    freeze = require_freeze(cfg, stage)
    paths = stage_paths(cfg, stage)
    for name in ("tables", "logs", "bootstrap"):
        paths[name].mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    log_path = paths["logs"] / "reused_statistics.json"
    save_json(log_path, dict(status="running", stage=stage, config_sha256=cfg["_config_sha256"], started_at=started))
    try:
        source = _load_sources(cfg, paths)
        lookup = _individual_groups(source)
        implementation = [file_info(Path(__file__)),
                          file_info(Path(__file__).with_name("bootstrap.py")),
                          file_info(Path(__file__).with_name("common.py"))]
        cache_stats = dict(computed=0, reused=0, implementation=implementation)
        snr, snr_seeds = _snr_tables(cfg, paths, source, lookup, cache_stats)
        plane, plane_seeds, pareto, pareto_seeds = _plane_and_pareto(cfg, source, lookup)
        tables = {
            "phase2_snr_summary": pd.DataFrame(snr, columns=SUMMARY_FIELDS),
            "phase2_snr_seed_effects": pd.DataFrame(snr_seeds),
            "effect_plane": pd.DataFrame(plane, columns=PLANE_FIELDS),
            "effect_plane_seed_values": pd.DataFrame(plane_seeds),
            "pareto": pd.DataFrame(pareto, columns=PARETO_FIELDS),
            "pareto_seed_values": pd.DataFrame(pareto_seeds),
        }
        expected_counts = dict(phase2_snr_summary=120, phase2_snr_seed_effects=600,
                               effect_plane=684, effect_plane_seed_values=3420,
                               pareto=8, pareto_seed_values=40)
        counts = {name: len(table) for name, table in tables.items()}
        _require(counts == expected_counts, f"Incomplete deterministic reused-analysis grid: {counts}")
        _require(cache_stats["computed"] + cache_stats["reused"] == 2000, "Incomplete F1/ECE prediction grid")
        for name, table in tables.items():
            write_csv(paths["tables"] / f"{name}.csv", table)
        protocol = dict(
            status="completed", stage=stage, partial=False, config_sha256=cfg["_config_sha256"],
            phase2_config_sha256=source["old"]["_config_sha256"], started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(), counts=counts,
            sources=list(source["checked"].values()), implementation=implementation,
            freeze_config_sha256=freeze["config_sha256"],
            outputs=[file_info(paths["tables"] / f"{name}.csv") for name in tables],
            cache={key: value for key, value in cache_stats.items() if key != "implementation"},
            coverage=dict(training_seeds=cfg["phase2_training_seeds"],
                          fixed_noise_seeds=source["old"]["stages"]["full"]["test_noise_seeds"],
                          gaussian_combinations=21, gaussian_plane_rows=630,
                          nstdb_sensitivity_rows=54, archived_checkpoints=40,
                          extra_prediction_distributions=2000, bootstrap_replicates=len(source["draws"])),
            estimands=dict(snr="Within-checkpoint electrode minus independent_rms; metrics then fixed-noise mean",
                           plane_x=cfg["phase2"]["effect_plane_x"], plane_y=cfg["phase2"]["effect_plane_y"],
                           pareto_x=cfg["phase2"]["pareto_x"], pareto_y=cfg["phase2"]["pareto_y"],
                           pareto_better_direction="upper_left", ece_positive="worse calibration"),
            uncertainty=dict(training=cfg["statistics"]["training_uncertainty"],
                             patient=cfg["statistics"]["patient_uncertainty"],
                             noise=cfg["statistics"]["noise_uncertainty"],
                             invalid=cfg["statistics"]["invalid_draws"], joint=False),
            limitations=[
                "Descriptive/exploratory only; no new p-values or change to the original six-comparison Holm family.",
                "Patient intervals condition on fixed checkpoints and five fixed noise realizations; not joint uncertainty.",
                "A4 Gaussian rows (kind=bandpass) and confounded NSTDB sensitivity rows (kind=bw/ma/em) must be displayed separately.",
                "A4 axes share a clean_only baseline; no regression or causal interpretation is warranted.",
                "Pareto dominance uses point estimates within architecture only, not statistically significant dominance or a clinical acceptability margin.",
                "No new inference or training; macro-AUROC tensors are reused and only per-case F1/ECE patient distributions are newly calculated.",
            ],
        )
        save_json(log_path, protocol)
        return protocol
    except Exception as error:
        save_json(log_path, dict(status="failed", stage=stage, config_sha256=cfg["_config_sha256"],
                                 started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                                 error=f"{type(error).__name__}: {error}"))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["full"], default="full",
                        help="Reuse is full-cohort only; this command performs no model inference")
    args = parser.parse_args(argv)
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
