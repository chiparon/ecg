"""Independently audit saved experiment artifacts; never alter fitted models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import torch

from phase1_ecg_robustness.src.models import build_model
from phase1_ecg_robustness.src.train import f1_thresholds, seed_everything
from .common import (
    expected_runs,
    file_info,
    load_config,
    load_stage_data,
    require_preregistration,
    resolve_path,
    save_json,
    stage_paths,
    verify_preservation,
    verify_training_complete,
)
from .generate_phase2_noise import array_sha256, load_case_inputs, load_test_manifest
from .run_phase2 import technical_gate


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_file(cfg, recorded):
    current = file_info(resolve_path(cfg, recorded["path"]))
    require(current == recorded, f"Artifact fingerprint differs: {recorded['path']}")
    return current


def close(actual, expected, label, atol=2e-12):
    require(
        np.allclose(actual, expected, atol=atol, rtol=0, equal_nan=True),
        f"Independent recomputation differs: {label}",
    )


def audit_training(cfg, stage, data, summaries):
    rows, paired_aug = [], {}
    for summary in summaries:
        for field in ("epoch_log", "validation_predictions", "environment_artifact"):
            checked_file(cfg, summary[field])
        frame = pd.read_csv(resolve_path(cfg, summary["epoch_log"]["path"]))
        selected = int(frame.loc[frame.val_macro_auroc.idxmax(), "epoch"])
        require(
            selected == summary["best_epoch"],
            "Checkpoint selection was not earliest clean-validation AUROC maximum",
        )
        checkpoint = torch.load(
            resolve_path(cfg, summary["checkpoint"]),
            map_location="cpu",
            weights_only=False,
        )
        require(
            checkpoint["threshold_tuning_count"] == 1
            and checkpoint["threshold_source"]
            == "best_checkpoint_clean_validation_once",
            "Threshold source changed",
        )
        require(
            checkpoint["epoch"] == selected
            and checkpoint["training_epochs_completed"] == len(frame),
            "Final checkpoint epoch metadata differs",
        )
        with np.load(
            resolve_path(cfg, summary["validation_predictions"]["path"]),
            allow_pickle=False,
        ) as validation:
            expected_thresholds, _ = f1_thresholds(validation["y"], validation["p"])
            close(
                checkpoint["thresholds"],
                expected_thresholds,
                "clean-validation thresholds",
                atol=0,
            )
            close(
                roc_auc_score(validation["y"], validation["p"], average="macro"),
                frame.loc[frame.epoch.eq(selected), "val_macro_auroc"].item(),
                "selected clean-validation AUROC",
            )
            np.testing.assert_array_equal(validation["indices"], data["splits"]["val"])
            np.testing.assert_array_equal(
                validation["y"], data["y"][data["splits"]["val"]]
            )
        for artifact in summary["epoch_artifacts"]:
            checked_file(cfg, artifact["checkpoint"])
            checked_file(cfg, artifact["augmentation"])
            with np.load(
                resolve_path(cfg, artifact["augmentation"]["path"]), allow_pickle=False
            ) as audit:
                np.testing.assert_array_equal(
                    np.sort(audit["indices"]), data["splits"]["train"]
                )
                np.testing.assert_array_equal(
                    audit["exposure_indices"], data["splits"]["train"]
                )
                require(
                    np.all(audit["exposure_counts"] == 1),
                    "Record exposure differs from exactly once per epoch",
                )
                np.testing.assert_array_equal(
                    audit["ids"],
                    data["metadata"].iloc[audit["indices"]].ecg_id.to_numpy(),
                )
                noisy = audit["input_type"] != 0
                require(
                    set(audit["snr_db"][noisy]) <= set(cfg["train"]["snrs"]),
                    "Unregistered training SNR",
                )
                require(
                    np.all(audit["combo_index"][noisy] >= 0)
                    and np.all(audit["combo_index"][noisy] < len(cfg["_train_combos"])),
                    "Unregistered training electrode combination",
                )
                require(
                    int(audit["noise_base"]) == cfg["train"]["noise_base"],
                    "Wrong training noise domain",
                )
                if noisy.any():
                    require(
                        float(
                            np.max(
                                np.abs(
                                    audit["actual_snr_db"][noisy]
                                    - audit["snr_db"][noisy]
                                )
                            )
                        )
                        < 1e-4,
                        "Actual training SNR differs",
                    )
                fingerprints = tuple(
                    array_sha256(audit[field])
                    for field in (
                        "indices",
                        "input_type",
                        "combo_index",
                        "snr_db",
                        "noise_seed_words",
                    )
                )
                key = (summary["strategy"], summary["seed"], artifact["epoch"])
                paired_aug.setdefault(key, fingerprints)
                require(
                    paired_aug[key] == fingerprints,
                    "Model-dependent augmentation or record order",
                )
        rows.append(
            dict(
                strategy=summary["strategy"],
                model=summary["model"],
                seed=summary["seed"],
                epochs=len(frame),
                best_epoch=selected,
                parameter_count=summary["parameter_count"],
                checkpoint_sha256=summary["checkpoint_sha256"],
            )
        )
    return rows


def audit_predictions(cfg, stage, paths, manifest, summaries, device):
    metrics = pd.read_csv(
        paths["tables"] / "metrics.csv",
        keep_default_na=False,
        na_values=["", "nan", "NaN"],
    )
    primary_ids = next(
        group["case_ids"]
        for group in manifest["groups"]
        if group["group_id"] == "primary_joint"
    )
    expected_primary = 100 if stage == "full" else 2
    require(len(primary_ids) == expected_primary, "Primary case count differs")
    cases = {case["case_id"]: case for case in manifest["cases"]}
    expected_grid = {
        (*run, case_id) for run in expected_runs(cfg, stage) for case_id in cases
    }
    require(
        set(
            metrics[["strategy", "model", "seed", "case_id"]].itertuples(
                index=False, name=None
            )
        )
        == expected_grid,
        "Prediction matrix differs from registered grid",
    )
    require(
        not metrics.duplicated(["strategy", "model", "seed", "case_id"]).any(),
        "Duplicate prediction cell",
    )
    for row in metrics.itertuples(index=False):
        require(
            file_info(resolve_path(cfg, row.prediction_path))["sha256"]
            == row.prediction_sha256,
            "Prediction checksum differs",
        )
        require(
            row.input_sha256 == cases[row.case_id]["input_sha256"],
            "Unpaired test input hashes",
        )
    boot_root = paths["tables"] / "patient_bootstrap"
    boot_manifest = json.loads(
        (boot_root / "manifest.json").read_text(encoding="utf-8")
    )
    draws = np.load(boot_root / "patient_draws.npy", mmap_mode="r")
    with np.load(boot_root / "cohort.npz", allow_pickle=False) as reference:
        reference = {key: reference[key] for key in reference.files}
    n_patients = len(reference["unique_patients"])
    require(
        draws.shape == (cfg["statistics"]["bootstrap_replicates"], n_patients),
        "Bootstrap shape differs",
    )
    require(
        np.all(draws >= 0) and np.all(draws.sum(axis=1) == n_patients),
        "Invalid patient multiplicities",
    )
    expected_draws = np.random.default_rng(
        cfg["statistics"]["bootstrap_seed"]
    ).multinomial(n_patients, np.full(n_patients, 1 / n_patients), size=len(draws))
    np.testing.assert_array_equal(draws, expected_draws)
    checked_file(cfg, boot_manifest["draws"])
    checked_file(cfg, boot_manifest["cohort"])
    group_lookup = {
        name: index for index, name in enumerate(boot_manifest["group_ids"])
    }
    selected_draws = [0, len(draws) // 2, len(draws) - 1]
    expanded = [
        np.repeat(
            np.arange(len(reference["y"])), draws[i, reference["patient_inverse"]]
        )
        for i in selected_draws
    ]
    distributions, independent = {}, []
    macro_roc_error = macro_ap_error = macro_f1_error = replay_error = 0.0
    point_recomputations = expanded_recomputations = 0
    data = load_stage_data(cfg, stage)
    replay_cases = ["clean", primary_ids[0]]
    replay_inputs = {
        case_id: load_case_inputs(cfg, data, cases[case_id])[:128]
        for case_id in replay_cases
    }
    for summary in summaries:
        run = (summary["strategy"], summary["model"], summary["seed"])
        distribution_path = boot_root / f"{run[0]}__{run[1]}__seed_{run[2]}.npy"
        recorded = next(
            item["distribution"]
            for item in boot_manifest["distributions"]
            if tuple(item["run"]) == run
        )
        checked_file(cfg, recorded)
        distribution = np.load(distribution_path, mmap_mode="r")
        distributions[run] = distribution
        selected = metrics.loc[
            metrics.strategy.eq(run[0])
            & metrics.model.eq(run[1])
            & metrics.seed.eq(run[2])
        ].set_index("case_id")
        recomputed, expanded_values = {}, {}
        for case_id in ["clean", *primary_ids]:
            row = selected.loc[case_id]
            with np.load(
                resolve_path(cfg, row.prediction_path), allow_pickle=False
            ) as saved:
                y, p = saved["y"], saved["p"]
                for key in ("y", "ids", "patient_ids", "indices"):
                    np.testing.assert_array_equal(saved[key], reference[key])
                close(
                    saved["thresholds"],
                    summary["thresholds"],
                    "fixed test thresholds",
                    atol=0,
                )
                aucs = roc_auc_score(y, p, average=None)
                auc = float(aucs.mean())
                ap = float(average_precision_score(y, p, average="macro"))
                f1 = float(
                    f1_score(
                        y, p >= saved["thresholds"], average="macro", zero_division=0
                    )
                )
                close(
                    [auc, ap, f1],
                    [row.macro_auroc, row.macro_ap, row.macro_f1],
                    f"{run}/{case_id}",
                )
                macro_roc_error = max(macro_roc_error, abs(auc - row.macro_auroc))
                macro_ap_error = max(macro_ap_error, abs(ap - row.macro_ap))
                macro_f1_error = max(macro_f1_error, abs(f1 - row.macro_f1))
                point_recomputations += 1
                recomputed[case_id] = np.r_[auc, aucs]
                if run[2] == cfg["stages"][stage]["seeds"][0]:
                    expanded_values[case_id] = []
                    for indices in expanded:
                        class_auc = roc_auc_score(y[indices], p[indices], average=None)
                        expanded_values[case_id].append(
                            np.r_[class_auc.mean(), class_auc]
                        )
                        expanded_recomputations += 1
        clean = recomputed["clean"]
        primary = np.mean([recomputed[case_id] for case_id in primary_ids], axis=0)
        close(
            distribution[group_lookup["clean"], 0],
            clean,
            "stored clean bootstrap point",
        )
        close(
            distribution[group_lookup["primary_joint"], 0],
            primary,
            "stored primary bootstrap point",
        )
        independent.append(
            dict(
                strategy=run[0],
                model=run[1],
                seed=run[2],
                clean_auroc=float(clean[0]),
                primary_auroc=float(primary[0]),
                retention=float(primary[0] / (clean[0] + 1e-12)),
            )
        )
        if expanded_values:
            expanded_clean = np.asarray(expanded_values["clean"])
            expanded_primary = np.mean(
                [expanded_values[case_id] for case_id in primary_ids], axis=0
            )
            close(
                distribution[group_lookup["clean"], np.asarray(selected_draws) + 1],
                expanded_clean,
                "explicitly repeated patient ECGs: clean",
            )
            close(
                distribution[
                    group_lookup["primary_joint"], np.asarray(selected_draws) + 1
                ],
                expanded_primary,
                "explicitly repeated patient ECGs: primary",
            )
        checkpoint = torch.load(
            resolve_path(cfg, summary["checkpoint"]),
            map_location="cpu",
            weights_only=False,
        )
        model = build_model(run[1], **checkpoint["model_kwargs"]).to(device).eval()
        model.load_state_dict(checkpoint["model_state"])
        with torch.inference_mode():
            for case_id, x in replay_inputs.items():
                actual = (
                    torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy()
                )
                with np.load(
                    resolve_path(cfg, selected.loc[case_id, "prediction_path"]),
                    allow_pickle=False,
                ) as stored:
                    expected = stored["p"][: len(x)]
                error = float(np.max(np.abs(actual - expected)))
                require(
                    np.allclose(
                        actual,
                        expected,
                        rtol=1e-4 if device == "cpu" else 0,
                        atol=1e-5 if device == "cpu" else 1e-6,
                    ),
                    "Checkpoint inference cannot reproduce saved predictions",
                )
                replay_error = max(replay_error, error)
        del model, checkpoint
    result = dict(
        prediction_hashes_checked=len(metrics),
        independent_auc_ap_f1_cases=point_recomputations,
        max_macro_auroc_error=macro_roc_error,
        max_macro_ap_error=macro_ap_error,
        max_macro_f1_error=macro_f1_error,
        explicit_patient_repeat_cases=expanded_recomputations,
        explicit_draw_indices=selected_draws,
        bootstrap_replicates=len(draws),
        patients=n_patients,
        records=len(reference["y"]),
        inference_replay_device=device,
        inference_replay_checkpoints=len(summaries),
        inference_replay_cases_per_checkpoint=2,
        inference_replay_records_per_case=128,
        inference_replay_max_probability_error=replay_error,
    )
    return pd.DataFrame(independent), distributions, group_lookup, result


def audit_contrasts(cfg, stage, paths, independent, distributions, groups):
    primary = pd.read_csv(paths["tables"] / "primary_comparisons.csv")
    seeds = cfg["stages"][stage]["seeds"]
    pvalues, rows = [], []
    for model in cfg["models"]:
        wide = (
            independent[independent.model.eq(model)]
            .pivot(index="seed", columns="strategy", values="retention")
            .reindex(seeds)
        )
        for lhs, rhs in cfg["statistics"]["primary_pairs"]:
            row = primary.loc[
                primary.model.eq(model) & primary.lhs.eq(lhs) & primary.rhs.eq(rhs)
            ].iloc[0]
            differences = (wide[lhs] - wide[rhs]).to_numpy()
            close(row.point, differences.mean(), "primary paired retention difference")
            require(
                row.positive_seed_count == int((differences > 0).sum()),
                "Positive seed count differs",
            )
            paired_draws = []
            for seed in seeds:
                a, b = (
                    distributions[(lhs, model, seed)],
                    distributions[(rhs, model, seed)],
                )
                paired_draws.append(
                    a[groups["primary_joint"], :, 0]
                    / (a[groups["clean"], :, 0] + 1e-12)
                    - b[groups["primary_joint"], :, 0]
                    / (b[groups["clean"], :, 0] + 1e-12)
                )
            paired_draws = np.mean(paired_draws, axis=0)
            valid = paired_draws[1:][np.isfinite(paired_draws[1:])]
            interval = (
                np.quantile(valid, [0.025, 0.975]) if len(valid) else [np.nan, np.nan]
            )
            close(
                [row.patient_ci95_low, row.patient_ci95_high],
                interval,
                "patient percentile contrast interval",
            )
            require(
                row.patient_n_valid == len(valid), "Patient valid-draw count differs"
            )
            if stage == "full":
                t_result = stats.ttest_rel(wide[lhs], wide[rhs])
                close(
                    row.p_raw, t_result.pvalue, "independent scipy paired t probability"
                )
                close(row.dz, differences.mean() / differences.std(ddof=1), "paired dz")
                margin = (
                    stats.t.ppf(0.975, len(seeds) - 1)
                    * differences.std(ddof=1)
                    / np.sqrt(len(seeds))
                )
                close(
                    [row.ci95_low, row.ci95_high],
                    [differences.mean() - margin, differences.mean() + margin],
                    "training-seed t interval",
                )
                pvalues.append(t_result.pvalue)
            else:
                require(
                    pd.isna(row.p_raw) and pd.isna(row.p_holm) and not row.confirmatory,
                    "Pilot contains confirmatory inference",
                )
            rows.append(
                dict(
                    model=model,
                    lhs=lhs,
                    rhs=rhs,
                    independently_recomputed_retention_difference=float(
                        differences.mean()
                    ),
                    patient_valid_draws=len(valid),
                )
            )
    if stage == "full":
        order = np.argsort(pvalues)
        adjusted = np.minimum(
            1,
            np.maximum.accumulate(
                np.asarray(pvalues)[order] * np.arange(len(pvalues), 0, -1)
            ),
        )
        close(primary.p_holm.to_numpy()[order], adjusted, "one six-test Holm family")
    independent.to_csv(
        paths["tables"] / "independently_recomputed_primary_seed_metrics.csv",
        index=False,
    )
    return rows


def audit_figures(cfg, stage, paths):
    manifest_path = paths["figures"] / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest["status"] == "completed"
        and manifest["config_sha256"] == cfg["_config_sha256"],
        "Figure manifest incomplete or stale",
    )
    require(
        len(manifest["figures"]) == (19 if stage == "full" else 18),
        "Figure inventory incomplete",
    )
    require(
        {item["family"] for item in manifest["figures"]}
        == set(manifest["required_families"]),
        "Figure families differ",
    )
    checked, outputs = set(), []
    for figure in manifest["figures"]:
        for source in figure["source_files"]:
            if source["path"] not in checked:
                checked_file(cfg, source)
                checked.add(source["path"])
        require(
            {Path(path).suffix for path in figure["files"]} == {".png", ".pdf"},
            "Missing PNG/PDF figure pair",
        )
        for filename in figure["files"]:
            path = resolve_path(cfg, filename)
            if path.suffix == ".png":
                with Image.open(path) as image:
                    require(
                        image.format == "PNG"
                        and image.width >= 1000
                        and image.height >= 500,
                        "Invalid rendered PNG",
                    )
                    image.verify()
            else:
                with path.open("rb") as stream:
                    require(stream.read(5) == b"%PDF-", "Invalid PDF figure")
            outputs.append(file_info(path))
    return dict(
        manifest=file_info(manifest_path),
        figure_pairs=len(manifest["figures"]),
        source_files_checked=len(checked),
        outputs=outputs,
        visual_review="Separate human/model image inspection required; file decoding is not visual approval",
    )


def run(config_path, stage, device):
    cfg = load_config(config_path)
    require_preregistration(cfg)
    seed_everything(0)
    torch.set_num_threads(int(cfg["train"]["torch_threads"]))
    paths = stage_paths(cfg, stage)
    data = load_stage_data(cfg, stage)
    summaries = verify_training_complete(cfg, stage)
    technical_gate(cfg, stage)
    manifest = load_test_manifest(cfg, stage)
    for info in [
        manifest["cohort_file"],
        *manifest["base_files"],
        *manifest["diagnostic_files"],
    ]:
        checked_file(cfg, info)
    for name in ("evaluation_protocol", "statistics_protocol"):
        protocol = json.loads(
            (paths["logs"] / f"{name}.json").read_text(encoding="utf-8")
        )
        outputs = protocol["outputs"]
        for info in outputs.values() if isinstance(outputs, dict) else outputs:
            checked_file(cfg, info)
    training = audit_training(cfg, stage, data, summaries)
    print(
        f"verify {stage}: training, input cache and protocol fingerprints passed",
        flush=True,
    )
    independent, distributions, groups, predictions = audit_predictions(
        cfg, stage, paths, manifest, summaries, device
    )
    contrasts = audit_contrasts(cfg, stage, paths, independent, distributions, groups)
    result = dict(
        status="passed",
        stage=stage,
        time_utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=cfg["_config_sha256"],
        verifier=file_info(Path(__file__)),
        training=training,
        predictions=predictions,
        contrasts=contrasts,
        figures=audit_figures(cfg, stage, paths),
        first_stage_preservation=verify_preservation(cfg),
        scope="All checkpoint/epoch/input/prediction/output hashes; independent sklearn primary+clean metrics; explicit expanded patient draws for first training seed of every model/strategy; all primary contrast intervals and Holm; checkpoint inference replay; all figure source hashes, PNG decoding and PDF headers",
    )
    save_json(paths["logs"] / "independent_verification.json", result)
    print(
        json.dumps({"status": "passed", "stage": stage, "predictions": predictions}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="phase2/configs/phase2_main.yaml")
    parser.add_argument("--stage", required=True, choices=("pilot", "full"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run(args.config, args.stage, args.device)


if __name__ == "__main__":
    main()
