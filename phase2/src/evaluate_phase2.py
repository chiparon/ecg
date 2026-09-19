"""Evaluate the complete preregistered matrix with immutable, case-major inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import pandas as pd
import sklearn
import torch

from phase1_ecg_robustness.src import evaluate as baseline_evaluate
from phase1_ecg_robustness.src import models as baseline_models
from phase1_ecg_robustness.src import train as baseline_train
from phase1_ecg_robustness.src.evaluate import CLASSES, classification_metrics
from phase1_ecg_robustness.src.models import build_model
from phase1_ecg_robustness.src.train import f1_thresholds, seed_everything

from . import common, generate_phase2_noise
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
    stage_paths,
    verify_training_complete,
)
from .generate_phase2_noise import load_case_inputs, load_test_manifest

CASE_FIELDS = ("kind", "condition", "combo_id", "combo_set", "snr", "noise_seed")
CALIBRATION_FIELDS = (
    "model",
    "strategy",
    "seed",
    "group_id",
    "class_name",
    "bin_id",
    "count",
    "probability_sum",
    "target_sum",
)


def array_sha256(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def extended_metrics(y, p, thresholds, clean_p=None):
    """Keep phase-one metrics; undefined predictive values remain explicitly NaN.

    Macro ratios target all five classes, not a changing subset: one undefined
    class makes that macro undefined. Confusion counts and denominator counts
    expose why; n_defined/n_undefined distinguish this from a missing result.
    """
    y, p, thresholds = np.asarray(y), np.asarray(p), np.asarray(thresholds)
    if y.ndim != 2 or y.shape != p.shape or y.shape[1] != len(CLASSES):
        raise ValueError("Expected matching record-by-five label/probability arrays")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Labels must remain binary multilabel targets")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if thresholds.shape != (len(CLASSES),) or not np.isfinite(thresholds).all():
        raise ValueError("Expected five finite clean-validation thresholds")
    if np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("Threshold outside [0, 1]")
    if clean_p is not None:
        if clean_p.shape != p.shape or not np.isfinite(clean_p).all():
            raise ValueError("Clean reference predictions do not match the cohort")
    values, loss = classification_metrics(y, p, thresholds, clean_p)
    target, prediction = y.astype(bool), p >= thresholds
    tp = np.count_nonzero(target & prediction, axis=0)
    tn = np.count_nonzero(~target & ~prediction, axis=0)
    fp = np.count_nonzero(~target & prediction, axis=0)
    fn = np.count_nonzero(target & ~prediction, axis=0)
    for name, counts in (("tp", tp), ("tn", tn), ("fp", fp), ("fn", fn)):
        values.update(
            {f"{name}_{label}": int(counts[j]) for j, label in enumerate(CLASSES)}
        )
    for metric, numerator, denominator in (
        ("sensitivity", tp, tp + fn),
        ("specificity", tn, tn + fp),
        ("ppv", tp, tp + fp),
        ("npv", tn, tn + fn),
    ):
        ratios = np.divide(
            numerator, denominator, out=np.full(5, np.nan), where=denominator != 0
        )
        for j, label in enumerate(CLASSES):
            values[f"{metric}_{label}"] = float(ratios[j])
            values[f"{metric}_denominator_{label}"] = int(denominator[j])
        n_defined = int(np.count_nonzero(denominator))
        values[f"macro_{metric}"] = float(ratios.mean())
        values[f"macro_{metric}_n_defined"] = n_defined
        values[f"macro_{metric}_n_undefined"] = len(CLASSES) - n_defined
    return values, loss


def calibration_bins(y, p):
    """All fifteen equal-width bins, including p=1 in bin14 and empty bins."""
    counts = np.zeros((len(CLASSES), 15), dtype=np.int64)
    probability_sums = np.zeros_like(counts, dtype=np.float64)
    target_sums = np.zeros_like(counts, dtype=np.float64)
    for j in range(len(CLASSES)):
        bins = np.minimum((p[:, j] * 15).astype(np.int64), 14)
        counts[j] = np.bincount(bins, minlength=15)
        probability_sums[j] = np.bincount(bins, weights=p[:, j], minlength=15)
        target_sums[j] = np.bincount(bins, weights=y[:, j], minlength=15)
    return counts, probability_sums, target_sums


def _cohort(data, split):
    indices = np.asarray(data["splits"][split], dtype=np.int64)
    records = data["metadata"].iloc[indices]
    return dict(
        indices=indices,
        y=np.asarray(data["y"][indices], dtype=np.float32),
        ids=records.ecg_id.to_numpy(dtype=np.int64),
        patient_ids=records.patient_id.to_numpy(dtype=np.int64),
    )


def _expected_case_keys(cfg, stage):
    settings = cfg["stages"][stage]
    heldout_ids = settings["heldout_combo_ids"]
    combos = [(c["combo_id"], "train") for c in cfg["_train_combos"]]
    combos += [
        (c["combo_id"], "heldout")
        for c in cfg["_heldout_combos"]
        if heldout_ids == "all" or c["combo_id"] in heldout_ids
    ]
    if cfg["test"]["include_all_electrodes"]:
        combos.append(("all", "all"))
    result = {("clean", "clean", "none", "clean", 100, 0)}
    for combo, combo_set in combos:
        for seed in settings["test_noise_seeds"]:
            for condition in cfg["test"]["conditions"]:
                for snr in settings["test_snrs"]:
                    result.add(("bandpass", condition, combo, combo_set, snr, seed))
    if settings["nstdb"]:
        for kind in cfg["test"]["nstdb_kinds"]:
            for seed in settings["test_noise_seeds"]:
                for condition in cfg["test"]["conditions"]:
                    for snr in cfg["test"]["nstdb_snrs"]:
                        result.add(
                            (
                                kind,
                                condition,
                                cfg["test"]["nstdb_combo"],
                                "all",
                                snr,
                                seed,
                            )
                        )
    return result


def _validate_manifest(cfg, stage, data, manifest, cohort):
    for key, expected in (
        ("status", "completed"),
        ("stage", stage),
        ("config_sha256", cfg["_config_sha256"]),
        ("matrix_sha256", cfg["matrix_sha256"]),
        ("data_identity", data["identity"]),
    ):
        require(manifest.get(key) == expected, f"Test manifest {key} mismatch")
    for key, array in (
        ("test_indices_sha256", cohort["indices"]),
        ("test_ids_sha256", cohort["ids"]),
        ("test_patient_ids_sha256", cohort["patient_ids"]),
        ("test_labels_sha256", cohort["y"]),
    ):
        require(
            manifest.get(key) == array_sha256(array), f"Test manifest {key} mismatch"
        )
    cases, groups = manifest["cases"], manifest["groups"]
    keys = [tuple(case[key] for key in CASE_FIELDS) for case in cases]
    require(len(keys) == len(set(keys)), "Duplicate test case descriptors")
    require(
        set(keys) == _expected_case_keys(cfg, stage),
        "Test cache omits or adds preregistered matrix cells",
    )
    by_id = {case["case_id"]: case for case in cases}
    require(
        len(by_id) == len(cases) and "clean" in by_id,
        "Nonunique case IDs or missing clean",
    )
    group_ids = [group["group_id"] for group in groups]
    require(len(set(group_ids)) == len(groups), "Duplicate test groups")
    require(
        {"clean", "primary_joint"}.issubset(group_ids), "Missing calibration groups"
    )
    for group in groups:
        members = group["case_ids"]
        require(
            bool(members) and len(members) == len(set(members)),
            "Empty or duplicate group membership",
        )
        require(set(members).issubset(by_id), "Unknown case in test group")
        for case in cases:
            require(
                (case["case_id"] in members) == (group["group_id"] in case["groups"]),
                "Case/group membership is inconsistent",
            )
    for case in cases:
        require(set(case["groups"]).issubset(group_ids), "Unknown group in case")
        require(
            Path(case["case_id"]).name == case["case_id"]
            and "/" not in case["case_id"],
            "Unsafe case ID",
        )
    primary = next(group for group in groups if group["group_id"] == "primary_joint")
    expected_primary = {
        case["case_id"]
        for case in cases
        if case["kind"] == cfg["statistics"]["primary_kind"]
        and case["condition"] == cfg["statistics"]["primary_condition"]
        and case["combo_set"] == cfg["statistics"]["primary_combo_set"]
        and case["snr"] in cfg["statistics"]["primary_snrs"]
    }
    require(
        set(primary["case_ids"]) == expected_primary,
        "Primary group differs from preregistration",
    )
    clean_group = next(group for group in groups if group["group_id"] == "clean")
    require(
        clean_group["case_ids"] == ["clean"],
        "Clean group must contain only clean input",
    )
    return [by_id["clean"]] + [case for case in cases if case["case_id"] != "clean"]


def _validate_validation(cfg, stage, data, checkpoint, summary, strategy, model, seed):
    path = (
        run_directory(cfg, stage, strategy, model, seed)
        / "best_checkpoint_clean_validation.npz"
    )
    info = file_info(path)
    require(
        summary["validation_predictions"]
        == info
        == checkpoint["validation_predictions"],
        "Clean-validation archive checksum mismatch",
    )
    expected = _cohort(data, "val")
    with np.load(path, allow_pickle=False) as saved:
        for key, array in expected.items():
            require(
                np.array_equal(saved[key], array), f"Clean-validation {key} mismatch"
            )
        for key, value in (
            ("config_sha256", cfg["_config_sha256"]),
            ("matrix_sha256", cfg["matrix_sha256"]),
            ("best_epoch", checkpoint["best_epoch"]),
            ("model_initialization_sha256", checkpoint["model_initialization_sha256"]),
        ):
            require(saved[key].item() == value, f"Clean-validation {key} mismatch")
        p = saved["p"]
        require(
            p.shape == expected["y"].shape
            and np.isfinite(p).all()
            and np.all((p >= 0) & (p <= 1)),
            "Invalid clean-validation probabilities",
        )
        # Verification only: never assign these thresholds to an evaluated model.
        thresholds, notes = f1_thresholds(expected["y"], p)
        require(
            np.array_equal(saved["thresholds"], checkpoint["thresholds"]),
            "Saved validation thresholds mismatch",
        )
        require(
            np.array_equal(thresholds, checkpoint["thresholds"]),
            "Thresholds not derived from best clean validation",
        )
        require(
            notes == checkpoint["threshold_notes"], "Threshold selection notes mismatch"
        )
    return info


def _load_models(cfg, stage, data, summaries, device):
    runs = expected_runs(cfg, stage)
    summary_map = {
        (row["strategy"], row["model"], int(row["seed"])): row for row in summaries
    }
    require(
        len(summary_map) == len(summaries) and set(summary_map) == set(runs),
        "Incomplete training summary grid",
    )
    states = []
    for strategy, model_name, seed in runs:
        summary = summary_map[strategy, model_name, seed]
        path = checkpoint_path(cfg, stage, strategy, model_name, seed)
        info = file_info(path)
        require(
            resolve_path(cfg, summary["checkpoint"]).resolve() == path.resolve(),
            "Unexpected completed checkpoint path",
        )
        require(
            info["sha256"] == summary["checkpoint_sha256"],
            "Checkpoint checksum changed after training",
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for key, value in (
            ("completed", True),
            ("config_sha256", cfg["_config_sha256"]),
            ("matrix_sha256", cfg["matrix_sha256"]),
            ("data_identity", data["identity"]),
            ("stage", stage),
            ("strategy", strategy),
            ("model_name", model_name),
            ("seed", seed),
            ("training_epochs_completed", cfg["stages"][stage]["epochs"]),
            ("selection_metric", "clean_validation_macro_auroc"),
            ("threshold_comparison", ">="),
            ("class_order", list(CLASSES)),
        ):
            require(checkpoint.get(key) == value, f"Checkpoint {key} mismatch: {path}")
        require(
            checkpoint["epoch"] == checkpoint["best_epoch"] == summary["best_epoch"],
            "Not the selected best checkpoint",
        )
        require(
            1 <= checkpoint["best_epoch"] <= cfg["stages"][stage]["epochs"],
            "Best epoch outside completed training",
        )
        require(
            float(checkpoint["scale_mv"])
            == float(data["scale_mv"])
            == float(summary["scale_mv"]),
            "Normalization scale mismatch",
        )
        for split in ("train", "val", "test"):
            require(
                np.array_equal(checkpoint[f"{split}_indices"], data["splits"][split]),
                f"Checkpoint {split} cohort mismatch",
            )
        thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float64)
        require(
            thresholds.shape == (5,)
            and np.isfinite(thresholds).all()
            and np.all((thresholds >= 0) & (thresholds <= 1)),
            "Uncalibrated checkpoint thresholds",
        )
        require(
            np.array_equal(thresholds, summary["thresholds"]),
            "Summary threshold mismatch",
        )
        validation = _validate_validation(
            cfg, stage, data, checkpoint, summary, strategy, model_name, seed
        )
        model = build_model(model_name, **checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval()
        model.requires_grad_(False)
        states.append(
            dict(
                strategy=strategy,
                model_name=model_name,
                seed=seed,
                model=model,
                thresholds=thresholds,
                checkpoint=info,
                validation=validation,
                calibration={
                    group: (
                        np.zeros((5, 15), dtype=np.int64),
                        np.zeros((5, 15), dtype=np.float64),
                        np.zeros((5, 15), dtype=np.float64),
                    )
                    for group in ("clean", "primary_joint")
                },
            )
        )
        del checkpoint
    return states


def _atomic_predictions(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _reuse_prediction(path, expected_arrays, expected_scalars):
    """A stale or corrupt archive is recomputed, never treated as a missing cell."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            if any(
                key not in saved or not np.array_equal(saved[key], value)
                for key, value in expected_arrays.items()
            ):
                return None
            if any(
                key not in saved or saved[key].item() != value
                for key, value in expected_scalars.items()
            ):
                return None
            p = saved["p"]
            if p.dtype != np.float32 or p.shape != expected_arrays["y"].shape:
                return None
            if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                return None
            if saved["p_sha256"].item() != array_sha256(p):
                return None
            safe = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
            y = expected_arrays["y"]
            loss = -np.mean(y * np.log(safe) + (1 - y) * np.log1p(-safe), axis=1)
            if not np.array_equal(saved["loss"], loss):
                return None
            return p
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


@torch.inference_mode()
def _predict_tensor(model, inputs, batch_size):
    probabilities = np.empty((len(inputs), len(CLASSES)), dtype=np.float32)
    for begin in range(0, len(inputs), batch_size):
        end = min(begin + batch_size, len(inputs))
        probabilities[begin:end] = torch.sigmoid(model(inputs[begin:end])).cpu().numpy()
    return probabilities


def _write_csv(path, rows, fields=None):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run(config, stage):
    cfg = load_config(config)
    require_preregistration(cfg)
    summaries = verify_training_complete(cfg, stage)  # Gate before all test access.
    data = load_stage_data(cfg, stage)
    paths = stage_paths(cfg, stage)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    manifest = load_test_manifest(cfg, stage)
    cohort = _cohort(data, "test")
    indices, y = cohort["indices"], cohort["y"]
    require(
        len(indices) == cfg["stages"][stage]["expected_records"]["test"],
        "Unexpected test cohort size",
    )
    require(len(set(cohort["ids"])) == len(indices), "Duplicate test ECG records")
    require(
        np.all(data["metadata"].iloc[indices].strat_fold.to_numpy() == 10),
        "Evaluation outside official fold10",
    )
    require(tuple(cfg["class_order"]) == CLASSES, "Changed label order")
    require(
        np.isin(y, (0, 1)).all()
        and np.all(y.sum(axis=0) > 0)
        and np.all(y.sum(axis=0) < len(y)),
        "Test classes need binary positive and negative labels",
    )
    cases = _validate_manifest(cfg, stage, data, manifest, cohort)
    torch.set_num_threads(cfg["train"]["torch_threads"])
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = _load_models(cfg, stage, data, summaries, device)
    sources = [
        file_info(Path(module.__file__))
        for module in (
            common,
            generate_phase2_noise,
            baseline_evaluate,
            baseline_models,
            baseline_train,
        )
    ]
    sources.append(file_info(Path(__file__)))
    execution = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        sklearn=sklearn.__version__,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=str(device),
        device_name=(
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        batch_size=cfg["train"]["batch_size"],
        threads=torch.get_num_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
    )
    require(
        execution["deterministic_algorithms"]
        and execution["cudnn_deterministic"]
        and not execution["cudnn_allow_tf32"]
        and not execution["matmul_allow_tf32"],
        "Evaluation must use deterministic FP32 policy",
    )
    manifest_info = file_info(paths["test_inputs"] / "manifest.json")
    fingerprint = canonical_sha256(
        dict(
            config_sha256=cfg["_config_sha256"],
            sources=sources,
            execution=execution,
            manifest=manifest_info,
        )
    )
    protocol_path = paths["logs"] / "evaluation_protocol.json"
    started = time.perf_counter()
    protocol = dict(
        status="running",
        stage=stage,
        config_sha256=cfg["_config_sha256"],
        matrix_sha256=cfg["matrix_sha256"],
        evaluation_fingerprint=fingerprint,
        manifest=manifest_info,
        sources=sources,
        execution=execution,
        data_identity=data["identity"],
        scale_mv=float(data["scale_mv"]),
        cohort_sha256={key: array_sha256(value) for key, value in cohort.items()},
        n_records=len(indices),
        n_patients=len(np.unique(cohort["patient_ids"])),
        n_cases=len(cases),
        n_checkpoints=len(states),
        expected_evaluations=len(cases) * len(states),
        case_major=True,
        same_normalized_float32_tensor_all_checkpoints=True,
        thresholds_source="best_checkpoint_clean_validation_once",
        threshold_comparison=">=",
        undefined_denominators="NaN; strict five-class macro; per-class denominator and undefined class counts saved",
        calibration_pooling="Descriptive repeated-condition bins, not independent patients or an ensemble",
        checkpoints=[
            dict(
                strategy=s["strategy"],
                model=s["model_name"],
                seed=s["seed"],
                checkpoint=s["checkpoint"],
                validation_predictions=s["validation"],
                thresholds=s["thresholds"].tolist(),
            )
            for s in states
        ],
    )
    save_json(protocol_path, protocol)
    rows, prediction_files, observed = [], [], set()
    metric_columns = None
    reused_count = 0
    try:
        for case in cases:
            x = load_case_inputs(cfg, data, case)
            require(
                x.dtype == np.float32
                and x.flags.c_contiguous
                and x.shape
                == (len(indices), len(cfg["lead_order"]), cfg["sequence_length"])
                and np.isfinite(x).all(),
                "Invalid normalized model inputs",
            )
            require(
                array_sha256(x) == case["input_sha256"],
                "Immutable test input checksum mismatch",
            )
            # Allocate/transfer once per case. Every model receives views of this
            # very same normalized float32 tensor, with no per-model scaling.
            inputs = torch.from_numpy(x).to(device)
            tensor_version = inputs._version
            for state in states:
                strategy, model_name, seed = (
                    state["strategy"],
                    state["model_name"],
                    state["seed"],
                )
                key = (strategy, model_name, seed, case["case_id"])
                require(key not in observed, "Repeated evaluation matrix cell")
                path = (
                    paths["predictions"]
                    / strategy
                    / model_name
                    / f"seed_{seed}"
                    / f"{case['case_id']}.npz"
                )
                arrays = dict(**cohort, thresholds=state["thresholds"])
                scalars = dict(
                    stage=stage,
                    strategy=strategy,
                    model=model_name,
                    seed=seed,
                    case_id=case["case_id"],
                    config_sha256=cfg["_config_sha256"],
                    matrix_sha256=cfg["matrix_sha256"],
                    input_sha256=case["input_sha256"],
                    checkpoint_sha256=state["checkpoint"]["sha256"],
                    evaluation_fingerprint=fingerprint,
                    **{field: case[field] for field in CASE_FIELDS},
                )
                p = _reuse_prediction(path, arrays, scalars)
                reused = p is not None
                if not reused:
                    p = _predict_tensor(
                        state["model"], inputs, cfg["train"]["batch_size"]
                    )
                require(
                    inputs._version == tensor_version,
                    "Model mutated shared test input tensor",
                )
                values, loss = extended_metrics(
                    y, p, state["thresholds"], state.get("clean_p")
                )
                if case["case_id"] == "clean":
                    state["clean_p"] = p
                if not reused:
                    _atomic_predictions(
                        path,
                        **arrays,
                        **scalars,
                        p=p,
                        loss=loss,
                        p_sha256=array_sha256(p),
                    )
                else:
                    reused_count += 1
                info = file_info(path)
                prediction_files.append(
                    dict(
                        strategy=strategy,
                        model=model_name,
                        seed=seed,
                        case_id=case["case_id"],
                        **info,
                    )
                )
                if metric_columns is None:
                    metric_columns = list(values)
                require(
                    list(values) == metric_columns,
                    "Metric columns changed within evaluation",
                )
                rows.append(
                    dict(
                        stage=stage,
                        model=model_name,
                        strategy=strategy,
                        seed=seed,
                        case_id=case["case_id"],
                        **{field: case[field] for field in CASE_FIELDS},
                        prediction_path=info["path"],
                        prediction_sha256=info["sha256"],
                        checkpoint_sha256=state["checkpoint"]["sha256"],
                        input_sha256=case["input_sha256"],
                        **values,
                    )
                )
                for group in ("clean", "primary_joint"):
                    if group in case["groups"]:
                        for total, update in zip(
                            state["calibration"][group], calibration_bins(y, p)
                        ):
                            total += update
                observed.add(key)
            require(
                array_sha256(x) == case["input_sha256"],
                "Shared host inputs mutated during evaluation",
            )
            del inputs, x
            print(
                f"eval {stage} {case['case_id']}: {len(observed)}/{protocol['expected_evaluations']} cells",
                flush=True,
            )
        expected = {
            (strategy, model, seed, case["case_id"])
            for strategy, model, seed in expected_runs(cfg, stage)
            for case in cases
        }
        require(
            observed == expected and len(rows) == len(expected),
            "Incomplete Cartesian evaluation grid",
        )
        table = pd.DataFrame(rows)
        require(
            not table.duplicated(["strategy", "model", "seed", "case_id"]).any(),
            "Duplicate saved metrics cells",
        )
        require(
            (table.groupby("case_id").input_sha256.nunique() == 1).all(),
            "Models saw different case inputs",
        )
        require(
            (
                table.groupby(["strategy", "model", "seed"]).checkpoint_sha256.nunique()
                == 1
            ).all(),
            "Checkpoint changed across cases",
        )
        bins_rows = []
        by_group = {group["group_id"]: group for group in manifest["groups"]}
        for state in states:
            for group, (counts, psums, ysums) in state["calibration"].items():
                require(
                    np.all(
                        counts.sum(axis=1)
                        == len(indices) * len(by_group[group]["case_ids"])
                    ),
                    "Incomplete calibration coverage",
                )
                for j, label in enumerate(CLASSES):
                    for bin_id in range(15):
                        bins_rows.append(
                            dict(
                                model=state["model_name"],
                                strategy=state["strategy"],
                                seed=state["seed"],
                                group_id=group,
                                class_name=label,
                                bin_id=bin_id,
                                count=int(counts[j, bin_id]),
                                probability_sum=float(psums[j, bin_id]),
                                target_sum=float(ysums[j, bin_id]),
                            )
                        )
        metrics_path = paths["tables"] / "metrics.csv"
        bins_path = paths["tables"] / "calibration_bins.csv"
        registry_path = paths["tables"] / "metrics_columns.json"
        _write_csv(metrics_path, rows)
        _write_csv(bins_path, bins_rows, CALIBRATION_FIELDS)
        save_json(registry_path, metric_columns)
        ledger_path = paths["logs"] / "prediction_manifest.json"
        save_json(
            ledger_path,
            dict(evaluation_fingerprint=fingerprint, predictions=prediction_files),
        )
        require(
            file_info(paths["test_inputs"] / "manifest.json") == manifest_info,
            "Test manifest changed during evaluation",
        )
        for state in states:
            require(
                file_info(resolve_path(cfg, state["checkpoint"]["path"]))
                == state["checkpoint"],
                "Checkpoint changed during evaluation",
            )
            require(
                file_info(resolve_path(cfg, state["validation"]["path"]))
                == state["validation"],
                "Clean-validation provenance changed during evaluation",
            )
        protocol.update(
            status="completed",
            elapsed_seconds=time.perf_counter() - started,
            n_evaluations=len(rows),
            reused_predictions=reused_count,
            n_undefined_metrics={
                column: int(table[column].isna().sum()) for column in metric_columns
            },
            outputs={
                "metrics": file_info(metrics_path),
                "metric_columns": file_info(registry_path),
                "calibration_bins": file_info(bins_path),
                "predictions": file_info(ledger_path),
            },
            full_cartesian_coverage=True,
            input_hash_invariance=True,
            cohort_label_invariance=True,
            threshold_invariance=True,
        )
        save_json(protocol_path, protocol)
    except BaseException as error:
        protocol.update(
            status="failed",
            elapsed_seconds=time.perf_counter() - started,
            n_evaluations=len(rows),
            error=f"{type(error).__name__}: {error}",
        )
        save_json(protocol_path, protocol)
        raise
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("pilot", "full"))
    args = parser.parse_args(argv)
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
