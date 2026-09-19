"""Preregistered, separately conditioned training-seed and patient statistics.

The saved bootstrap tensor contains condition-averaged AUROCs, not averaged
probabilities. Axis 1 starts with the unresampled point and then common draws.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from phase1_ecg_robustness.src.statistics import holm_adjust, seed_summary
from phase1_ecg_robustness.src.supplemental_statistics import (
    cluster_auc_distribution,
    mean_seed_auc,
    patient_draws,
)
from .common import (
    expected_runs,
    file_info,
    load_config,
    load_stage_data,
    require_preregistration,
    resolve_path,
    save_json,
    stage_paths,
    verify_training_complete,
)
from .generate_phase2_noise import load_test_manifest

DESCRIPTORS = ["group_id", "kind", "combo_set", "combo_id", "condition", "snr"]
SUMMARY_KEYS = ["model", "strategy", *DESCRIPTORS, "metric", "outcome"]
OUTCOMES = ("absolute", "drop", "retention", "clean_cost")
PAIRED_METRICS = ("macro_auroc", "macro_ap", "macro_f1")
CONDITIONING = (
    "95% percentile patient-cluster interval conditional on fixed trained "
    "checkpoints and fixed test-noise realizations; not joint uncertainty"
)


def performance_metric(name):
    """Only performance scores, not counts or calibration diagnostics, normalize."""
    families = ("auroc", "ap", "f1", "sensitivity", "specificity", "ppv", "npv")
    return any(
        name == f"{prefix}{family}"
        for family in families
        for prefix in ("macro_", "micro_")
    ) or any(
        name.startswith(f"{family}_")
        and name.split("_", 1)[1] in ("NORM", "MI", "STTC", "CD", "HYP")
        for family in families
    )


def outcome_value(value, clean, baseline_clean, outcome):
    """Apply matched-clean transformations within a seed or bootstrap replicate."""
    if outcome == "absolute":
        return value
    if outcome == "drop":
        return clean - value
    if outcome == "retention":
        return value / (clean + 1e-12)
    if outcome == "clean_cost":
        return baseline_clean - clean
    raise ValueError(f"Unknown outcome {outcome}")


def equal_condition_mean(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError("A registered group cannot have zero conditions")
    # Missing conditions/classes must not silently redefine the estimand.
    return values.mean(axis=0)


def interval(distribution):
    values = np.asarray(distribution, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Expected point followed by bootstrap replicates")
    valid = values[1:][np.isfinite(values[1:])]
    low, high = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
    return dict(
        point=float(values[0]),
        ci95_low=float(low),
        ci95_high=float(high),
        n_bootstrap=len(values) - 1,
        n_valid=len(valid),
        n_invalid=len(values) - 1 - len(valid),
    )


def paired_summary(lhs, rhs, expected=None, inferential=True):
    """Matched arrays: never pair finite subsets after independently dropping NaNs."""
    lhs, rhs = np.asarray(lhs, dtype=float), np.asarray(rhs, dtype=float)
    if lhs.shape != rhs.shape or lhs.ndim != 1:
        raise ValueError("Paired values must have identical one-dimensional shape")
    delta = lhs - rhs
    finite = delta[np.isfinite(delta)]
    summary = seed_summary(delta)
    result = dict(
        n_seeds=summary["n_seeds"],
        point=summary["mean"],
        sd=summary["std"],
        ci95_low=summary["ci95_low"],
        ci95_high=summary["ci95_high"],
        n_expected=len(delta) if expected is None else expected,
        n_undefined=int((~np.isfinite(delta)).sum()),
        positive_seed_count=int((finite > 0).sum()),
        negative_seed_count=int((finite < 0).sum()),
        zero_seed_count=int((finite == 0).sum()),
        dz=np.nan,
        t_stat=np.nan,
        p_raw=np.nan,
    )
    if len(finite) > 1:
        mean, sd = finite.mean(), finite.std(ddof=1)
        if sd > 0:
            dz = mean / sd
            t_stat = dz * np.sqrt(len(finite))
            p = 2 * stats.t.sf(abs(t_stat), len(finite) - 1)
        elif mean != 0:
            dz = t_stat = np.copysign(np.inf, mean)
            p = 0.0
        else:
            dz, t_stat, p = np.nan, 0.0, 1.0
        result["dz"] = float(dz)
        if inferential:
            result.update(t_stat=float(t_stat), p_raw=float(p))
    return result


def comparison_pairs(strategies):
    pairs = [
        (lhs, rhs)
        for rhs in ("clean_only", "independent_rms")
        for lhs in strategies
        if lhs != rhs
    ]
    return list(dict.fromkeys([*pairs, ("mixed", "electrode")]))


def _long_vectors(rows, metric_names):
    """Expand compact vector records only once, avoiding millions of dictionaries."""
    frame = pd.DataFrame(rows)
    values = np.stack(frame.pop("values"))
    undefined = np.stack(frame.pop("undefined")) if "undefined" in frame else None
    long = frame.loc[frame.index.repeat(len(metric_names))].reset_index(drop=True)
    long["metric"] = np.tile(metric_names, len(frame))
    long["value"] = values.reshape(-1)
    if undefined is not None:
        long["n_undefined"] = undefined.reshape(-1)
    return long


def group_tables(metrics, groups, metric_names):
    """Case means then seed means; all normalization precedes seed aggregation."""
    clean = metrics.loc[metrics.case_id.eq("clean")].set_index(
        ["model", "strategy", "seed"]
    )
    rows = {outcome: [] for outcome in OUTCOMES}
    noise_accumulated, primary_noise = {}, []
    performance = np.array(
        [i for i, name in enumerate(metric_names) if performance_metric(name)]
    )
    outcome_indices = {
        outcome: np.arange(len(metric_names)) if outcome == "absolute" else performance
        for outcome in OUTCOMES
    }
    for (model, strategy, seed), run in metrics.groupby(
        ["model", "strategy", "seed"], sort=True
    ):
        run = run.set_index("case_id")
        clean_values = clean.loc[(model, strategy, seed), metric_names].to_numpy(
            dtype=float
        )
        baseline = clean.loc[(model, "clean_only", seed), metric_names].to_numpy(
            dtype=float
        )
        for group in groups:
            cases = run.loc[group["case_ids"]]
            desc = {name: group[name] for name in DESCRIPTORS}
            base = dict(model=model, strategy=strategy, **desc)
            partitions = [(None, cases)]
            if group["group_id"] != "clean":
                partitions.extend(cases.groupby("noise_seed", sort=True))
            for noise_seed, subset in partitions:
                absolute = equal_condition_mean(
                    subset[metric_names].to_numpy(dtype=float)
                )
                for outcome, indices in outcome_indices.items():
                    values = np.asarray(
                        outcome_value(absolute, clean_values, baseline, outcome)
                    )[indices]
                    if noise_seed is None:
                        rows[outcome].append(
                            dict(
                                base,
                                seed=int(seed),
                                outcome=outcome,
                                values=values,
                                n_cases=len(subset),
                                n_noise_seeds=int(subset.noise_seed.nunique()),
                            )
                        )
                    else:
                        key = (
                            model,
                            strategy,
                            group["group_id"],
                            int(noise_seed),
                            outcome,
                        )
                        if key not in noise_accumulated:
                            noise_accumulated[key] = dict(
                                base,
                                noise_seed=int(noise_seed),
                                outcome=outcome,
                                values=np.zeros_like(values),
                                undefined=np.zeros(len(values), dtype=int),
                                n_training_seeds=0,
                                n_cases=len(subset),
                            )
                        accumulator = noise_accumulated[key]
                        accumulator["values"] += values
                        accumulator["undefined"] += ~np.isfinite(values)
                        accumulator["n_training_seeds"] += 1
                        if (
                            group["group_id"] == "primary_joint"
                            and outcome == "retention"
                        ):
                            index = list(indices).index(
                                metric_names.index("macro_auroc")
                            )
                            primary_noise.append(
                                dict(
                                    base,
                                    seed=int(seed),
                                    noise_seed=int(noise_seed),
                                    outcome=outcome,
                                    metric="macro_auroc",
                                    value=float(values[index]),
                                )
                            )
    raw, noise = [], []
    for outcome, indices in outcome_indices.items():
        names = [metric_names[i] for i in indices]
        raw.append(_long_vectors(rows[outcome], names))
        accumulated = [
            row for key, row in noise_accumulated.items() if key[-1] == outcome
        ]
        for row in accumulated:
            row["values"] /= row["n_training_seeds"]
        noise.append(_long_vectors(accumulated, names))
    return (
        pd.concat(raw, ignore_index=True),
        pd.concat(noise, ignore_index=True),
        pd.DataFrame(primary_noise),
    )


def summarize_seeds(frame):
    rows = []
    for keys, part in frame.groupby(SUMMARY_KEYS, sort=True, dropna=False):
        values = part.value.to_numpy(dtype=float)
        rows.append(
            dict(zip(SUMMARY_KEYS, keys))
            | seed_summary(values)
            | dict(
                n_expected=len(values), n_undefined=int((~np.isfinite(values)).sum())
            )
        )
    return pd.DataFrame(rows)


def noise_tables(by_noise):
    summaries = []
    for key, part in by_noise.groupby(SUMMARY_KEYS, sort=True, dropna=False):
        values = part.value.to_numpy(dtype=float)
        summaries.append(
            dict(zip(SUMMARY_KEYS, key))
            | dict(
                n_noise_seeds=len(values),
                mean=float(values.mean()),
                std=float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                min=float(values.min()),
                max=float(values.max()),
                n_undefined=int((~np.isfinite(values)).sum()),
            )
        )
    return pd.DataFrame(summaries)


def _scalar(archive, name):
    return str(archive[name].item())


def _checkpoint_bootstrap(task):
    """Bounded process worker, one read/condition and one saved base tensor/run."""
    (
        cfg,
        stage,
        run,
        records,
        groups,
        reference_path,
        draws_path,
        destination,
        checkpoint,
        batch_size,
        metric_names,
    ) = task
    with np.load(reference_path, allow_pickle=False) as ref:
        reference = {key: ref[key] for key in ref.files}
    draws = np.load(draws_path, mmap_mode="r", allow_pickle=False)
    shape = (len(groups), len(draws) + 1, len(metric_names))
    destination = Path(destination)
    temporary = destination.with_suffix(".partial.npy")
    accumulated = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float64, shape=shape
    )
    accumulated[:] = 0
    memberships = {}
    for index, group in enumerate(groups):
        for case in group["case_ids"]:
            memberships.setdefault(case, []).append(index)
    sources, counts = [], np.zeros(len(groups), dtype=int)
    for record in records:
        path = resolve_path(cfg, record["prediction_path"])
        before = file_info(path)
        if before["sha256"] != record["prediction_sha256"]:
            raise ValueError(
                f"Prediction checksum differs from evaluation ledger: {path}"
            )
        with np.load(path, allow_pickle=False) as saved:
            for key in ("y", "ids", "patient_ids", "indices"):
                if not np.array_equal(saved[key], reference[key]):
                    raise ValueError(f"Cohort mismatch: {path}: {key}")
            for key, expected in (
                ("config_sha256", cfg["_config_sha256"]),
                ("matrix_sha256", cfg["matrix_sha256"]),
                ("input_sha256", record["input_sha256"]),
                ("checkpoint_sha256", checkpoint["checkpoint_sha256"]),
            ):
                if _scalar(saved, key) != expected:
                    raise ValueError(f"Prediction provenance mismatch: {path}: {key}")
            if not np.array_equal(
                saved["thresholds"],
                np.asarray(checkpoint["thresholds"], dtype=saved["thresholds"].dtype),
            ):
                raise ValueError(f"Thresholds changed: {path}")
            p = saved["p"]
            if (
                p.shape != reference["y"].shape
                or not np.isfinite(p).all()
                or np.any((p < 0) | (p > 1))
            ):
                raise ValueError(f"Invalid probabilities: {path}")
            values = cluster_auc_distribution(
                reference["y"], p, reference["patient_inverse"], draws, batch_size
            )
        observed = np.asarray([record[name] for name in metric_names], dtype=float)
        if not np.allclose(values[0], observed, atol=2e-7, rtol=0, equal_nan=True):
            raise ValueError(f"Saved AUROC metrics disagree with predictions: {path}")
        after = file_info(path)
        if before != after:
            raise ValueError(f"Prediction changed while reading: {path}")
        sources.append(
            dict(
                model=run[1],
                strategy=run[0],
                seed=run[2],
                case_id=record["case_id"],
                **before,
            )
        )
        for index in memberships[record["case_id"]]:
            accumulated[index] += values
            counts[index] += 1
    for index, group in enumerate(groups):
        if counts[index] != len(group["case_ids"]):
            raise ValueError("Incomplete bootstrap condition coverage")
        accumulated[index] /= counts[index]
    accumulated.flush()
    del accumulated
    temporary.replace(destination)
    return dict(run=list(run), distribution=file_info(destination), predictions=sources)


def _distribution_paths(
    cfg, stage, groups, metric_names, metrics, checkpoints, reference, draws
):
    paths = stage_paths(cfg, stage)
    directory = paths["tables"] / "patient_bootstrap"
    directory.mkdir(parents=True, exist_ok=True)
    reference_path, draws_path = (
        directory / "cohort.npz",
        directory / "patient_draws.npy",
    )
    np.savez(reference_path, **reference)
    np.save(draws_path, draws, allow_pickle=False)
    by_run = {(s["strategy"], s["model"], int(s["seed"])): s for s in checkpoints}
    tasks = []
    for run in expected_runs(cfg, stage):
        strategy, model, seed = run
        selected = metrics.loc[
            metrics.strategy.eq(strategy)
            & metrics.model.eq(model)
            & metrics.seed.eq(seed)
        ]
        output = directory / f"{strategy}__{model}__seed_{seed}.npy"
        tasks.append(
            (
                cfg,
                stage,
                run,
                selected.to_dict("records"),
                groups,
                str(reference_path),
                str(draws_path),
                str(output),
                by_run[run],
                cfg["statistics"]["bootstrap_batch_size"],
                metric_names,
            )
        )
    workers = min(int(cfg["statistics"]["bootstrap_workers"]), len(tasks))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(_checkpoint_bootstrap, tasks))
    else:
        outputs = [_checkpoint_bootstrap(task) for task in tasks]
    sources = [source for output in outputs for source in output["predictions"]]
    pd.DataFrame(sources).to_csv(
        paths["tables"] / "prediction_provenance.csv", index=False
    )
    manifest = dict(
        axis_order=["registered_group", "point_then_common_patient_draw", "auc_metric"],
        group_ids=[g["group_id"] for g in groups],
        metrics=metric_names,
        draws=file_info(draws_path),
        cohort=file_info(reference_path),
        distributions=[
            {k: v for k, v in output.items() if k != "predictions"}
            for output in outputs
        ],
    )
    save_json(directory / "manifest.json", manifest)
    lookup = {
        tuple(out["run"]): np.load(
            resolve_path(cfg, out["distribution"]["path"]), mmap_mode="r"
        )
        for out in outputs
    }
    refs = {tuple(out["run"]): out["distribution"]["path"] for out in outputs}
    return lookup, refs, manifest


def patient_tables(cfg, stage, groups, metric_names, distributions, refs):
    seeds = cfg["stages"][stage]["seeds"]
    strategies = list(cfg["strategies"])
    rows, lookup = [], {}
    clean_index = next(i for i, g in enumerate(groups) if g["group_id"] == "clean")
    for model in cfg["models"]:
        for gi, group in enumerate(groups):
            desc = {name: group[name] for name in DESCRIPTORS}
            for outcome in OUTCOMES:
                transformed = {}
                for strategy in strategies:
                    values = []
                    expression_refs = []
                    for seed in seeds:
                        array = distributions[(strategy, model, seed)]
                        baseline = distributions[("clean_only", model, seed)]
                        value = outcome_value(
                            array[gi],
                            array[clean_index],
                            baseline[clean_index],
                            outcome,
                        )
                        transformed[(strategy, str(seed))] = value
                        values.append(value)
                        ref = dict(
                            base=refs[(strategy, model, seed)],
                            group_index=gi,
                            clean_index=clean_index,
                            baseline=refs[("clean_only", model, seed)],
                            outcome=outcome,
                        )
                        expression_refs.append(ref)
                    transformed[(strategy, "mean_fixed_seeds")] = mean_seed_auc(values)
                    for training_seed in [*map(str, seeds), "mean_fixed_seeds"]:
                        selected_refs = (
                            expression_refs
                            if training_seed == "mean_fixed_seeds"
                            else [expression_refs[seeds.index(int(training_seed))]]
                        )
                        for mi, metric in enumerate(metric_names):
                            result = interval(
                                transformed[(strategy, training_seed)][:, mi]
                            )
                            row = dict(
                                desc,
                                model=model,
                                strategy=strategy,
                                training_seed=training_seed,
                                metric=metric,
                                outcome=outcome,
                                lhs=strategy,
                                rhs="",
                                **result,
                                conditioning=CONDITIONING,
                                distribution_ref=json.dumps(
                                    dict(
                                        operation="equal_seed_mean",
                                        metric_index=mi,
                                        terms=selected_refs,
                                    ),
                                    separators=(",", ":"),
                                ),
                            )
                            rows.append(row)
                for lhs, rhs in comparison_pairs(strategies):
                    for training_seed in [*map(str, seeds), "mean_fixed_seeds"]:
                        difference = (
                            transformed[(lhs, training_seed)]
                            - transformed[(rhs, training_seed)]
                        )
                        for mi, metric in enumerate(metric_names):
                            result = interval(difference[:, mi])
                            row = dict(
                                desc,
                                model=model,
                                strategy=lhs,
                                training_seed=training_seed,
                                metric=metric,
                                outcome=outcome,
                                lhs=lhs,
                                rhs=rhs,
                                **result,
                                conditioning=CONDITIONING,
                                distribution_ref=json.dumps(
                                    dict(
                                        operation="paired_difference",
                                        model=model,
                                        lhs=lhs,
                                        rhs=rhs,
                                        training_seed=training_seed,
                                        group_index=gi,
                                        metric_index=mi,
                                        outcome=outcome,
                                        manifest="patient_bootstrap/manifest.json",
                                    ),
                                    separators=(",", ":"),
                                ),
                            )
                            rows.append(row)
                            if training_seed == "mean_fixed_seeds":
                                lookup[
                                    (
                                        model,
                                        group["group_id"],
                                        metric,
                                        outcome,
                                        lhs,
                                        rhs,
                                    )
                                ] = result
    return pd.DataFrame(rows), lookup


def paired_tables(cfg, stage, group_rows, patient_lookup):
    seeds = cfg["stages"][stage]["seeds"]
    rows = []
    selected = group_rows[group_rows.metric.isin(PAIRED_METRICS)]
    keys = ["model", *DESCRIPTORS, "metric", "outcome"]
    for key, part in selected.groupby(keys, sort=True, dropna=False):
        desc = dict(zip(keys, key))
        pivot = part.pivot(index="seed", columns="strategy", values="value").reindex(
            seeds
        )
        for lhs, rhs in comparison_pairs(list(cfg["strategies"])):
            summary = paired_summary(
                pivot[lhs], pivot[rhs], len(seeds), inferential=False
            )
            ci = patient_lookup.get(
                (
                    desc["model"],
                    desc["group_id"],
                    desc["metric"],
                    desc["outcome"],
                    lhs,
                    rhs,
                ),
                {},
            )
            rows.append(
                dict(
                    desc,
                    lhs=lhs,
                    rhs=rhs,
                    **summary,
                    patient_ci95_low=ci.get("ci95_low", np.nan),
                    patient_ci95_high=ci.get("ci95_high", np.nan),
                )
            )
    paired = pd.DataFrame(rows)
    primary = []
    for model in cfg["models"]:
        subset = selected.loc[
            selected.model.eq(model)
            & selected.group_id.eq("primary_joint")
            & selected.metric.eq("macro_auroc")
            & selected.outcome.eq("retention")
        ]
        pivot = subset.pivot(index="seed", columns="strategy", values="value").reindex(
            seeds
        )
        for lhs, rhs in cfg["statistics"]["primary_pairs"]:
            summary = paired_summary(
                pivot[lhs], pivot[rhs], len(seeds), inferential=stage == "full"
            )
            ci = patient_lookup[
                (model, "primary_joint", "macro_auroc", "retention", lhs, rhs)
            ]
            primary.append(
                dict(
                    model=model,
                    lhs=lhs,
                    rhs=rhs,
                    group_id="primary_joint",
                    metric="macro_auroc",
                    outcome="retention",
                    **summary,
                    p_holm=np.nan,
                    patient_ci95_low=ci["ci95_low"],
                    patient_ci95_high=ci["ci95_high"],
                    patient_n_valid=ci["n_valid"],
                    patient_n_invalid=ci["n_invalid"],
                    confirmatory=stage == "full",
                    stage=stage,
                )
            )
    primary = pd.DataFrame(primary)
    if stage == "full":
        if (
            len(primary) != 6
            or not primary.n_seeds.eq(5).all()
            or not np.isfinite(primary.p_raw).all()
        ):
            raise ValueError(
                "Confirmatory family requires all six valid five-seed contrasts"
            )
        primary["p_holm"] = holm_adjust(primary.p_raw.to_numpy())
    return paired, primary


def noise_contrasts(cfg, noise_rows):
    selected = noise_rows.loc[
        noise_rows.group_id.eq("primary_joint")
        & noise_rows.metric.eq("macro_auroc")
        & noise_rows.outcome.eq("retention")
    ]
    rows = []
    for model, part in selected.groupby("model", sort=True):
        pivot = part.pivot(
            index=["seed", "noise_seed"], columns="strategy", values="value"
        )
        for lhs, rhs in cfg["statistics"]["primary_pairs"]:
            differences = pivot[lhs] - pivot[rhs]
            base = dict(
                model=model,
                lhs=lhs,
                rhs=rhs,
                group_id="primary_joint",
                metric="macro_auroc",
                outcome="retention",
            )
            for (seed, noise_seed), value in differences.items():
                rows.append(
                    dict(
                        base,
                        level="seed_noise",
                        seed=seed,
                        noise_seed=noise_seed,
                        point=value,
                    )
                )
            by_noise = []
            for noise_seed, values in differences.groupby(level="noise_seed"):
                a = values.to_numpy(dtype=float)
                point = float(a.mean())
                by_noise.append(point)
                rows.append(
                    dict(
                        base,
                        level="noise",
                        seed=np.nan,
                        noise_seed=noise_seed,
                        point=point,
                        positive_seed_count=int((a > 0).sum()),
                        n_training_seeds=len(a),
                        n_undefined=int((~np.isfinite(a)).sum()),
                    )
                )
            a = np.asarray(by_noise)
            rows.append(
                dict(
                    base,
                    level="summary",
                    seed=np.nan,
                    noise_seed=np.nan,
                    point=float(a.mean()),
                    std=float(a.std(ddof=1)) if len(a) > 1 else np.nan,
                    min=float(a.min()),
                    max=float(a.max()),
                    n_noise_seeds=len(a),
                    positive_noise_count=int((a > 0).sum()),
                    negative_noise_count=int((a < 0).sum()),
                    zero_noise_count=int((a == 0).sum()),
                    n_undefined=int((~np.isfinite(a)).sum()),
                )
            )
    return pd.DataFrame(rows)


def ranking_tables(group_rows, groups, strategies):
    metrics = [*PAIRED_METRICS, "brier", "ece"]
    raw = group_rows.loc[
        group_rows.metric.isin(metrics) & group_rows.outcome.eq("absolute"),
        ["model", "seed", "group_id", "metric", "strategy", "value"],
    ].copy()
    raw["rank"] = np.nan
    for (_, _, _, metric), indices in raw.groupby(
        ["model", "seed", "group_id", "metric"]
    ).groups.items():
        values = raw.loc[indices, "value"]
        if len(values) == len(strategies) and np.isfinite(values).all():
            raw.loc[indices, "rank"] = values.rank(
                method="average", ascending=metric in ("brier", "ece")
            )
    summary = raw.groupby(
        ["model", "group_id", "metric", "strategy"], as_index=False
    ).agg(
        mean_rank=("rank", lambda x: np.asarray(x).mean()),
        std_rank=(
            "rank",
            lambda x: np.asarray(x).std(ddof=1) if len(x) > 1 else np.nan,
        ),
    )
    comparisons = [
        ("clean", g["group_id"], "clean_vs_noisy")
        for g in groups
        if g["group_id"] != "clean"
    ]
    metadata = pd.DataFrame([{k: g[k] for k in DESCRIPTORS} for g in groups])
    noisy = metadata[~metadata.group_id.isin(["clean", "primary_joint"])]
    for _, part in noisy.groupby(
        ["kind", "combo_set", "combo_id", "condition"], dropna=False
    ):
        ids = part.sort_values("snr", ascending=False).group_id.tolist()
        comparisons.extend((a, b, "adjacent_snr") for a, b in zip(ids, ids[1:]))
    aggregates = noisy[noisy.combo_id.eq("aggregate")]
    for _, part in aggregates.groupby(["kind", "condition", "snr"]):
        sets = part.set_index("combo_set").group_id.to_dict()
        if "train" in sets and "heldout" in sets:
            comparisons.append(
                (sets["train"], sets["heldout"], "seen_vs_heldout_combinations")
            )
    means = raw.groupby(
        ["model", "group_id", "metric", "strategy"], as_index=False
    ).agg(value=("value", lambda x: np.asarray(x).mean()))
    means["seed"] = "mean_fixed_seeds"
    all_values = pd.concat([raw.drop(columns="rank"), means], ignore_index=True)
    rows = []
    for (model, seed, metric), part in all_values.groupby(
        ["model", "seed", "metric"], sort=False
    ):
        wide = part.pivot(index="group_id", columns="strategy", values="value").reindex(
            columns=strategies
        )
        for lhs, rhs, comparison in comparisons:
            a, b = wide.loc[lhs].to_numpy(), wide.loc[rhs].to_numpy()
            valid = (
                np.isfinite(a).all()
                and np.isfinite(b).all()
                and np.ptp(a) > 0
                and np.ptp(b) > 0
            )
            rows.append(
                dict(
                    model=model,
                    seed=seed,
                    metric=metric,
                    group_lhs=lhs,
                    group_rhs=rhs,
                    comparison=comparison,
                    spearman=(
                        float(stats.spearmanr(a, b).statistic) if valid else np.nan
                    ),
                    kendall=(
                        float(stats.kendalltau(a, b).statistic) if valid else np.nan
                    ),
                    n_strategies=len(strategies),
                    descriptive=True,
                )
            )
    return raw, summary, pd.DataFrame(rows)


def _validate_inputs(cfg, stage, paths, manifest, checkpoints):
    protocol_path = paths["logs"] / "evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for key, value in (
        ("status", "completed"),
        ("stage", stage),
        ("config_sha256", cfg["_config_sha256"]),
        ("matrix_sha256", cfg["matrix_sha256"]),
    ):
        if protocol.get(key) != value:
            raise ValueError(f"Evaluation protocol mismatch: {key}")
    for name in ("metrics", "metric_columns", "predictions"):
        recorded = protocol["outputs"][name]
        if file_info(resolve_path(cfg, recorded["path"])) != recorded:
            raise ValueError(f"Evaluation output checksum mismatch: {name}")
    metrics_path = paths["tables"] / "metrics.csv"
    metrics = pd.read_csv(
        metrics_path, keep_default_na=False, na_values=["", "nan", "NaN"]
    )
    metric_names = json.loads(
        (paths["tables"] / "metrics_columns.json").read_text(encoding="utf-8")
    )
    cases = {case["case_id"]: case for case in manifest["cases"]}
    keys = ["strategy", "model", "seed", "case_id"]
    if metrics.duplicated(keys).any():
        raise ValueError("Duplicate evaluation rows")
    expected = {
        (*run, case_id) for run in expected_runs(cfg, stage) for case_id in cases
    }
    actual = set(metrics[keys].itertuples(index=False, name=None))
    if expected != actual or not metrics.stage.eq(stage).all():
        raise ValueError(
            "Evaluation grid differs from preregistered checkpoint x condition grid"
        )
    hashes = {
        (s["strategy"], s["model"], int(s["seed"])): s["checkpoint_sha256"]
        for s in checkpoints
    }
    for row in metrics.itertuples(index=False):
        case = cases[row.case_id]
        if row.checkpoint_sha256 != hashes[(row.strategy, row.model, row.seed)]:
            raise ValueError("Checkpoint hash differs from finalized training")
        for key in (
            "input_sha256",
            "kind",
            "condition",
            "combo_id",
            "combo_set",
            "snr",
            "noise_seed",
        ):
            if getattr(row, key) != case[key]:
                raise ValueError(
                    f"Evaluation case metadata mismatch: {row.case_id}: {key}"
                )
    group_ids = set()
    referenced = set()
    for group in manifest["groups"]:
        if (
            group["group_id"] in group_ids
            or not group["case_ids"]
            or len(set(group["case_ids"])) != len(group["case_ids"])
        ):
            raise ValueError("Invalid registered group")
        group_ids.add(group["group_id"])
        if not set(group["case_ids"]) <= cases.keys():
            raise ValueError("Group references absent cases")
        referenced.update(group["case_ids"])
    if referenced != cases.keys() or not {"clean", "primary_joint"} <= group_ids:
        raise ValueError("Registered groups do not cover the test matrix")
    for metric in metric_names:
        metrics[metric] = pd.to_numeric(metrics[metric], errors="raise")
    data = load_stage_data(cfg, stage)
    indices = data["splits"]["test"]
    cohort = data["metadata"].iloc[indices]
    ids = cohort["ecg_id"].to_numpy()
    patients = cohort["patient_id"].to_numpy()
    unique, inverse = np.unique(patients, return_inverse=True)
    reference = dict(
        y=np.asarray(data["y"][indices]),
        ids=ids,
        patient_ids=patients,
        indices=np.asarray(indices),
        patient_inverse=inverse,
        unique_patients=unique,
    )
    return metrics, metric_names, reference, protocol_path


def run(config_path, stage):
    cfg = load_config(config_path)
    require_preregistration(cfg)
    checkpoints = verify_training_complete(cfg, stage)
    paths = stage_paths(cfg, stage)
    manifest = load_test_manifest(cfg, stage)
    metrics, metric_names, reference, evaluation_protocol = _validate_inputs(
        cfg, stage, paths, manifest, checkpoints
    )
    groups = manifest["groups"]
    protocol_path = paths["logs"] / "statistics_protocol.json"
    save_json(
        protocol_path,
        dict(status="running", stage=stage, config_sha256=cfg["_config_sha256"]),
    )
    group_rows, noise_rows, primary_noise = group_tables(metrics, groups, metric_names)
    group_rows.to_csv(paths["tables"] / "group_seed_metrics.csv", index=False)
    summarize_seeds(group_rows).to_csv(
        paths["tables"] / "seed_summary.csv", index=False
    )
    noise_rows.to_csv(paths["tables"] / "noise_seed_metrics.csv", index=False)
    primary_noise.to_csv(
        paths["tables"] / "primary_seed_noise_metrics.csv", index=False
    )
    noise_tables(noise_rows).to_csv(
        paths["tables"] / "noise_replay_summary.csv", index=False
    )
    auc_names = ["macro_auroc", *[f"auroc_{name}" for name in cfg["class_order"]]]
    settings = cfg["statistics"]
    draws = patient_draws(
        len(reference["unique_patients"]),
        int(settings["bootstrap_replicates"]),
        int(settings["bootstrap_seed"]),
        int(settings["bootstrap_batch_size"]),
    )
    distributions, refs, bootstrap_manifest = _distribution_paths(
        cfg, stage, groups, auc_names, metrics, checkpoints, reference, draws
    )
    patient, patient_lookup = patient_tables(
        cfg, stage, groups, auc_names, distributions, refs
    )
    patient.to_csv(paths["tables"] / "patient_ci.csv", index=False)
    paired, primary = paired_tables(cfg, stage, group_rows, patient_lookup)
    paired.to_csv(paths["tables"] / "paired_effects.csv", index=False)
    primary.to_csv(paths["tables"] / "primary_comparisons.csv", index=False)
    noise_contrasts(cfg, primary_noise).to_csv(
        paths["tables"] / "noise_contrasts.csv", index=False
    )
    rank, rank_summary, correlations = ranking_tables(
        group_rows, groups, list(cfg["strategies"])
    )
    for name, table in (
        ("rank_seed", rank),
        ("rank_summary", rank_summary),
        ("rank_correlations", correlations),
    ):
        table.to_csv(paths["tables"] / f"{name}.csv", index=False)
    invalid = patient.groupby(
        ["training_seed", "metric", "outcome"], as_index=False
    ).agg(
        interval_rows=("n_invalid", "size"),
        rows_with_invalid=("n_invalid", lambda x: int((x > 0).sum())),
        max_invalid=("n_invalid", "max"),
    )
    invalid.to_csv(paths["tables"] / "bootstrap_invalid_summary.csv", index=False)
    source_paths = [
        Path(__file__),
        Path(cfg["_config_path"]),
        evaluation_protocol,
        paths["test_inputs"] / "manifest.json",
        paths["tables"] / "metrics.csv",
        paths["tables"] / "metrics_columns.json",
        Path(cfg["_baseline_root"]) / "src" / "supplemental_statistics.py",
        Path(cfg["_baseline_root"]) / "src" / "statistics.py",
    ]
    outputs = [
        "group_seed_metrics",
        "seed_summary",
        "noise_seed_metrics",
        "primary_seed_noise_metrics",
        "noise_replay_summary",
        "patient_ci",
        "paired_effects",
        "primary_comparisons",
        "noise_contrasts",
        "rank_seed",
        "rank_summary",
        "rank_correlations",
        "prediction_provenance",
        "bootstrap_invalid_summary",
    ]
    protocol = dict(
        status="completed",
        stage=stage,
        config_sha256=cfg["_config_sha256"],
        matrix_sha256=cfg["matrix_sha256"],
        sources=[file_info(p) for p in source_paths],
        outputs=[file_info(paths["tables"] / f"{name}.csv") for name in outputs],
        bootstrap=bootstrap_manifest,
        bootstrap_seed=settings["bootstrap_seed"],
        bootstrap_replicates=settings["bootstrap_replicates"],
        coverage=dict(
            checkpoints=len(checkpoints),
            cases=len(manifest["cases"]),
            groups=len(groups),
            prediction_rows=len(metrics),
            patient_interval_rows=len(patient),
            auc_metrics=auc_names,
        ),
        estimands=dict(
            group="equal arithmetic mean of condition-level metrics; no probability ensemble or pseudo-patient pooling",
            patient="record AUROC weighted by patient multiplicities, retaining all records of every selected patient",
            fixed_seed_mean="equal mean of fixed checkpoint metrics/ratios/differences within each shared patient draw",
            drop="matched checkpoint clean minus group metric",
            retention="group metric / (matched checkpoint clean metric + 1e-12), within each draw",
            clean_cost="same-seed clean_only clean minus strategy clean, repeated across groups for convenient joins",
            contrasts="lhs minus rhs, same training seeds, same patient draws, same fixed noises",
        ),
        uncertainty=dict(
            training="sample SD and Student t95 across independent training runs",
            patient=CONDITIONING,
            noise="descriptive SD/range across fixed noise bases AFTER averaging training seeds; not independent training repetitions",
            invalid="missing-class patient draws retained as NaN and counted, never redrawn; macro requires all five classes; percentile interval uses finite replicates",
            group_missing="ordinary equal mean propagates undefined constituent metrics; seed summaries additionally disclose expected and undefined counts",
            joint=False,
        ),
        confirmatory=dict(
            enabled=stage == "full",
            family="2 models x 3 preregistered retention contrasts",
            test="two-sided paired training-seed t",
            multiplicity="Holm across all six tests",
            n_seeds=len(cfg["stages"][stage]["seeds"]),
            pilot="technical descriptive only; no confirmatory p or effect test",
        ),
        descriptive_rank_comparisons="clean vs every noisy group, adjacent SNR within otherwise same group, and matched seen vs heldout aggregate groups; average tie ranks; undefined constant-vector correlations remain NaN",
        nstdb="exploratory confounded sensitivity; not isolated covariance evidence",
        clean_cost="descriptive; no clinical acceptability margin was preregistered",
    )
    save_json(protocol_path, protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="phase2/configs/phase2_main.yaml")
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
