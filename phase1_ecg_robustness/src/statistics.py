"""Seed-level summaries and paired, equal-patient ECG robustness inference.

Run from the project root: python -m src.statistics --config configs/phase1_pilot.yaml
CSV empty cells denote undefined quantities, never zero. All probability thresholds
are loaded solely to check provenance; this module never fits a test threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

PRIMARY = ("macro_auroc", "macro_ap", "macro_f1")
METRICS = (
    PRIMARY
    + (
        "micro_auroc",
        "micro_ap",
        "micro_f1",
        "brier",
        "ece",
        "mean_loss",
        "mean_confidence",
        "mean_entropy",
        "clean_agreement",
        "probability_shift",
    )
    + tuple(
        f"{metric}_{label}"
        for metric in ("auroc", "ap", "f1")
        for label in ("NORM", "MI", "STTC", "CD", "HYP")
    )
)
KEYS = ["model", "kind", "snr", "condition", "active"]
ROW_KEYS = ["model", "seed", "kind", "snr", "condition", "active"]
SUMMARY_COLUMNS = KEYS + ["metric", "n_seeds", "mean", "std", "ci95_low", "ci95_high"]
EFFECT_COLUMNS = [
    "model",
    "kind",
    "snr",
    "active",
    "comparison",
    "condition",
    "outcome",
    "mean_difference",
    "ci95_low",
    "ci95_high",
    "p_value",
    "p_holm",
    "standardized_effect",
    "n_patients",
    "n_records",
    "n_seeds",
    "inference_unit",
    "bootstrap_seed",
    "bootstrap_count",
    "permutation_seed",
    "permutation_count",
    "permutation_method",
    "holm_family",
    "status",
    "patient_median",
    "patient_fraction_positive",
    "mean_without_largest_1pct_abs",
    "mean_without_largest_5pct_abs",
]
SEED_COLUMNS = [
    "model",
    "kind",
    "snr",
    "active",
    "comparison",
    "condition",
    "metric",
    "seed",
    "left_value",
    "right_value",
    "difference",
    "n_seeds",
    "mean_difference",
    "std",
    "ci95_low",
    "ci95_high",
]
ROBUSTNESS_COLUMNS = ROW_KEYS + [
    "metric",
    "clean",
    "noisy",
    "clean_drop",
    "normalized_robustness",
]
AUDIT_COLUMNS = [
    "model",
    "kind",
    "snr",
    "active",
    "comparison",
    "condition",
    "ecg_id",
    "patient_id",
    "n_seeds",
    "loss_difference",
    "mean_probability_difference",
    "probability_shift_difference",
]
RANK_COLUMNS = KEYS + [
    "metric",
    "mean",
    "n_seeds",
    "rank",
    "n_models",
    "interpretation",
]
CORRELATION_COLUMNS = [
    "kind",
    "snr",
    "condition",
    "active",
    "metric",
    "n_models",
    "spearman",
    "kendall",
    "spearman_p_value",
    "kendall_p_value",
    "interpretation",
]
NOISE_PAIRS = (
    ("electrode", "independent_rms"),
    ("electrode", "independent"),
    ("covariance", "electrode"),
)
HOLM_FAMILY = "all_active_primary_loss_noise_contrasts_all_models_kinds_snrs"


def seed_summary(values):
    """Sample SD and two-sided Student-t CI over finite training-seed values."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    result = {
        "n_seeds": n,
        "mean": None,
        "std": None,
        "ci95_low": None,
        "ci95_high": None,
    }
    if n:
        result["mean"] = float(values.mean())
    if n > 1:
        sd = float(values.std(ddof=1))
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        result.update(
            std=sd, ci95_low=result["mean"] - half, ci95_high=result["mean"] + half
        )
    return result


def summarize(frame, metrics):
    rows = []
    for key, group in frame.groupby(KEYS, sort=True, dropna=False):
        base = dict(zip(KEYS, key))
        for metric in metrics:
            rows.append({**base, "metric": metric, **seed_summary(group[metric])})
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def derived_seed(base, *key):
    """Stable stream identity, unaffected by Python hash or table traversal order."""
    digest = hashlib.sha256(
        json.dumps([int(base), *key], default=str).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


def cluster_inference(
    patient_values, bootstrap, permutations, bootstrap_seed, permutation_seed
):
    """Resample/sign-flip patients, not records; every patient has equal weight."""
    values = np.asarray(patient_values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    result = dict(
        mean_difference=mean,
        ci95_low=None,
        ci95_high=None,
        p_value=None,
        standardized_effect=None,
        bootstrap_seed=str(bootstrap_seed),
        bootstrap_count=0,
        permutation_seed=str(permutation_seed),
        permutation_count=0,
        permutation_method="unavailable",
        status="insufficient_patients",
    )
    result.update(
        patient_median=float(np.median(values)),
        patient_fraction_positive=float(np.mean(values > 0)),
        mean_without_largest_1pct_abs=None,
        mean_without_largest_5pct_abs=None,
    )
    if n < 2:
        return result
    ordered = values[np.argsort(np.abs(values), kind="stable")]
    for fraction, name in (
        (0.01, "mean_without_largest_1pct_abs"),
        (0.05, "mean_without_largest_5pct_abs"),
    ):
        removed = max(1, int(np.ceil(n * fraction)))
        result[name] = float(ordered[:-removed].mean())
    sd = float(values.std(ddof=1))
    if sd > 0:
        result["standardized_effect"] = mean / sd
    rng = np.random.default_rng(bootstrap_seed)
    # Bound temporary index/sign matrices for full-test-set runs.
    chunk = max(1, min(256, 1_000_000 // n))
    draws = np.empty(bootstrap, dtype=np.float64)
    for start in range(0, bootstrap, chunk):
        count = min(chunk, bootstrap - start)
        indices = rng.integers(0, n, size=(count, n))
        draws[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    observed = abs(mean)
    tolerance = np.finfo(float).eps * max(float(np.abs(values).mean()), observed) * 32
    extreme = 0
    # Enumerate the exact sign distribution only when no larger than the request.
    exact = n <= 20 and 2**n <= permutations
    count_total = 2**n if exact else permutations
    rng = np.random.default_rng(permutation_seed)
    for start in range(0, count_total, chunk):
        count = min(chunk, count_total - start)
        if exact:
            codes = np.arange(start, start + count, dtype=np.uint64)[:, None]
            signs = (
                2 * ((codes >> np.arange(n, dtype=np.uint64)) & 1).astype(np.int8) - 1
            )
        else:
            signs = 2 * rng.integers(0, 2, size=(count, n), dtype=np.int8) - 1
        permuted = np.mean(signs * values, axis=1)
        extreme += int(np.count_nonzero(np.abs(permuted) >= observed - tolerance))
    p = extreme / count_total if exact else (extreme + 1) / (count_total + 1)
    result.update(
        ci95_low=float(low),
        ci95_high=float(high),
        p_value=float(p),
        bootstrap_count=bootstrap,
        permutation_count=count_total,
        permutation_method=(
            "exact_sign_flip" if exact else "monte_carlo_sign_flip_plus_one"
        ),
        status="ok" if sd > 0 else "constant_patient_differences_dz_undefined",
    )
    return result


def holm_adjust(p_values):
    """Holm step-down correction; the input is one explicitly defined family."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(order), dtype=float)
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate(p_values[order] * np.arange(len(order), 0, -1))
    )
    return adjusted


class PredictionLoader:
    """Read one group at a time while enforcing global test-sample alignment."""

    def __init__(self, root):
        self.root = root
        self.reference = None
        self.files_checked = 0

    def load(self, row, clean=None):
        path = Path(row.prediction_path)
        if not path.is_absolute():
            path = self.root / path
        with np.load(path, allow_pickle=False) as archive:
            names = ("p", "y", "ids", "patient_ids", "loss", "thresholds", "indices")
            missing = set(names) - set(archive.files)
            if missing:
                raise ValueError(f"{path}: missing prediction fields {sorted(missing)}")
            data = {name: archive[name] for name in names}
        n = len(data["ids"])
        expected = {
            "p": (n, 5),
            "y": (n, 5),
            "ids": (n,),
            "patient_ids": (n,),
            "loss": (n,),
            "thresholds": (5,),
            "indices": (n,),
        }
        for name, shape in expected.items():
            value = data[name]
            if value.shape != shape or not np.issubdtype(value.dtype, np.number):
                raise ValueError(f"{path}: invalid {name} shape/dtype")
            if not np.isfinite(value).all():
                raise ValueError(f"{path}: nonfinite {name}")
        if n == 0 or len(np.unique(data["ids"])) != n:
            raise ValueError(f"{path}: empty or duplicate ECG IDs")
        if not np.issubdtype(data["ids"].dtype, np.integer) or not np.issubdtype(
            data["indices"].dtype, np.integer
        ):
            raise ValueError(f"{path}: ECG IDs and dataset indices must be integers")
        if not np.isin(data["y"], [0, 1]).all() or np.any(
            (data["p"] < 0) | (data["p"] > 1)
        ):
            raise ValueError(f"{path}: labels or probabilities outside their domains")
        if np.any((data["thresholds"] < 0) | (data["thresholds"] > 1)):
            raise ValueError(f"{path}: classification thresholds outside [0,1]")
        mapping = ("ids", "y", "patient_ids", "indices")
        if self.reference is None:
            self.reference = {name: data[name].copy() for name in mapping}
        else:
            for name in mapping:
                if not np.array_equal(data[name], self.reference[name]):
                    raise ValueError(
                        f"{path}: test {name} differs in values/order from reference"
                    )
        if clean is not None and not np.array_equal(
            data["thresholds"], clean["thresholds"]
        ):
            raise ValueError(
                f"{path}: noisy thresholds differ from matched clean validation thresholds"
            )
        self.files_checked += 1
        return data


def validate_metrics(frame, config):
    required = set(ROW_KEYS + list(PRIMARY) + ["prediction_path"])
    if missing := required - set(frame.columns):
        raise ValueError(f"metrics.csv missing columns: {sorted(missing)}")
    if (
        frame.empty
        or frame[list(required)].isna().any().loc[ROW_KEYS + ["prediction_path"]].any()
    ):
        raise ValueError("Metrics are empty or have missing keys/prediction paths")
    if frame.duplicated(ROW_KEYS).any():
        raise ValueError("Duplicate model/seed/noise-condition metric rows")
    for field in ("seed", "snr"):
        values = pd.to_numeric(frame[field], errors="raise")
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError(f"{field} must contain finite integers")
        frame[field] = values.astype(np.int64)
    metric_columns = [metric for metric in METRICS if metric in frame]
    for metric in metric_columns:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
        if np.isinf(frame[metric]).any():
            raise ValueError(f"Infinite metric {metric}")
    clean_mask = frame.condition.eq("clean")
    if (
        not clean_mask.any()
        or not (
            frame.loc[clean_mask, "kind"].eq("clean")
            & frame.loc[clean_mask, "snr"].eq(100)
            & frame.loc[clean_mask, "active"].eq("all")
        ).all()
    ):
        raise ValueError("Clean rows must use kind=clean, snr=100, active=all")
    if frame.loc[~clean_mask, "kind"].eq("clean").any():
        raise ValueError("Non-clean condition cannot have kind=clean")
    expected_seeds = set(map(int, config["seeds"])) if "seeds" in config else None
    expected_models = set(config["models"]) if "models" in config else None
    if expected_models is not None and set(frame.model) != expected_models:
        raise ValueError("Metrics models do not match configured models")
    for model, group in frame.groupby("model", sort=True):
        clean = group[group.condition.eq("clean")]
        seeds = set(clean.seed)
        if expected_seeds is not None and seeds != expected_seeds:
            raise ValueError(f"{model}: clean training seeds do not match config")
        for key, block in group.groupby(KEYS[1:], sort=True):
            if set(block.seed) != seeds:
                raise ValueError(f"{model}/{key}: incomplete paired seed set")
    spec = config["noise"]
    expected = set()
    for model in expected_models:
        for seed in expected_seeds:
            expected.add((model, seed, "clean", 100, "clean", "all"))
            for kind in spec["kinds"]:
                for snr in spec["snrs"]:
                    for condition in (
                        "independent",
                        "independent_rms",
                        "electrode",
                        "covariance",
                    ):
                        expected.add((model, seed, kind, snr, condition, "all"))
            for electrode in spec.get("electrodes", []):
                for condition in ("independent_rms", "electrode", "covariance"):
                    expected.add(
                        (
                            model,
                            seed,
                            "bandpass",
                            spec.get("electrode_snr", 10),
                            condition,
                            electrode,
                        )
                    )
    actual = set(frame[ROW_KEYS].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(
            f"Metrics do not match configured evaluation grid: {len(expected-actual)} missing, {len(actual-expected)} unexpected"
        )
    return frame.sort_values(ROW_KEYS).reset_index(drop=True), metric_columns


def robustness_tables(frame, metrics):
    clean = frame[frame.condition.eq("clean")].set_index(["model", "seed"])
    rows = []
    selected = [
        m for m in metrics if m in PRIMARY or m.startswith(("auroc_", "ap_", "f1_"))
    ]
    for row in frame[~frame.condition.eq("clean")].itertuples(index=False):
        reference = clean.loc[(row.model, row.seed)]
        base = {key: getattr(row, key) for key in ROW_KEYS}
        for metric in selected:
            original, noisy = float(reference[metric]), float(getattr(row, metric))
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "clean": original,
                    "noisy": noisy,
                    "clean_drop": original - noisy,
                    "normalized_robustness": noisy / (original + 1e-12),
                }
            )
    detail = pd.DataFrame(rows, columns=ROBUSTNESS_COLUMNS)
    summary_rows = []
    for key, group in detail.groupby(KEYS + ["metric"], sort=True):
        base = dict(zip(KEYS, key[:-1]))
        for outcome in ("clean_drop", "normalized_robustness"):
            summary_rows.append(
                {
                    **base,
                    "metric": f"{key[-1]}_{outcome}",
                    **seed_summary(group[outcome]),
                }
            )
    return detail, pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)


def ranking_tables(summary):
    rows, correlations = [], []
    selected = summary[summary.metric.isin(PRIMARY)]
    clean = selected[selected.condition.eq("clean")]
    for key, group in selected.groupby(
        ["kind", "snr", "condition", "active", "metric"], sort=True
    ):
        group = group.copy()
        valid = group["mean"].notna()
        n_models = int(valid.sum())
        ranks = group["mean"].rank(method="average", ascending=False)
        interpretation = (
            "descriptive_unstable_two_architectures_no_meaningful_inference"
            if n_models == 2
            else (
                "descriptive_single_architecture"
                if n_models < 2
                else "descriptive_fixed_architecture_panel_no_population_inference"
            )
        )
        for index, row in group.iterrows():
            rows.append(
                {
                    **{
                        name: row[name] for name in KEYS + ["metric", "mean", "n_seeds"]
                    },
                    "rank": ranks.loc[index],
                    "n_models": n_models,
                    "interpretation": interpretation,
                }
            )
        baseline = clean[clean.metric.eq(key[-1])][["model", "mean"]]
        paired = (
            group[["model", "mean"]]
            .merge(
                baseline,
                on="model",
                suffixes=("_noisy", "_clean"),
                validate="one_to_one",
            )
            .dropna()
        )
        n = len(paired)
        rho = tau = None
        if (
            n >= 2
            and paired.mean_noisy.nunique() > 1
            and paired.mean_clean.nunique() > 1
        ):
            rho = float(stats.spearmanr(paired.mean_noisy, paired.mean_clean).statistic)
            tau = float(
                stats.kendalltau(
                    paired.mean_noisy, paired.mean_clean, variant="b"
                ).statistic
            )
        label = (
            interpretation
            if n == n_models
            else "descriptive_incomplete_architecture_panel"
        )
        if n >= 2 and (
            paired.mean_noisy.nunique() == 1 or paired.mean_clean.nunique() == 1
        ):
            label += ";undefined_constant_ranks"
        correlations.append(
            {
                **dict(zip(["kind", "snr", "condition", "active", "metric"], key)),
                "n_models": n,
                "spearman": rho,
                "kendall": tau,
                "spearman_p_value": None,
                "kendall_p_value": None,
                "interpretation": label,
            }
        )
    return (
        pd.DataFrame(rows, columns=RANK_COLUMNS),
        pd.DataFrame(correlations, columns=CORRELATION_COLUMNS),
    )


def paired_tables(frame, root, output, bootstrap, permutations, random_seed):
    effects, seed_rows = [], []
    loader = PredictionLoader(root)
    audit_path = output / "per_record_contrasts.csv"
    pd.DataFrame(columns=AUDIT_COLUMNS).to_csv(audit_path, index=False)
    for model, model_frame in frame.groupby("model", sort=True):
        clean_rows = model_frame[model_frame.condition.eq("clean")].set_index("seed")
        seeds = sorted(clean_rows.index)
        clean = {seed: loader.load(clean_rows.loc[seed]) for seed in seeds}
        noisy = model_frame[~model_frame.condition.eq("clean")]
        for key, group in noisy.groupby(["kind", "snr", "active"], sort=True):
            kind, snr, active = key
            lookup = group.set_index(["condition", "seed"])
            conditions = sorted(group.condition.unique())
            data = {
                (condition, seed): loader.load(
                    lookup.loc[(condition, seed)], clean[seed]
                )
                for condition in conditions
                for seed in seeds
            }
            pairs = [(condition, "clean") for condition in conditions]
            if active == "all":
                missing = {item for pair in NOISE_PAIRS for item in pair} - set(
                    conditions
                )
                if missing:
                    raise ValueError(
                        f"{model}/{kind}/{snr}: missing primary noise conditions {sorted(missing)}"
                    )
                pairs = list(NOISE_PAIRS) + pairs
            for left, right in pairs:
                comparison = (
                    f"{left}_minus_{right}" if right != "clean" else "noisy_minus_clean"
                )
                condition = left if right == "clean" else "all"
                base = dict(
                    model=model,
                    kind=kind,
                    snr=int(snr),
                    active=active,
                    comparison=comparison,
                    condition=condition,
                )
                n_records = len(clean[seeds[0]]["ids"])
                differences = np.zeros((n_records, 3), dtype=np.float64)
                for seed in seeds:
                    a = data[(left, seed)]
                    b = clean[seed] if right == "clean" else data[(right, seed)]
                    p_clean = clean[seed]["p"].astype(np.float64, copy=False)
                    p_a = a["p"].astype(np.float64, copy=False)
                    p_b = b["p"].astype(np.float64, copy=False)
                    differences[:, 0] += a["loss"].astype(np.float64) - b["loss"]
                    differences[:, 1] += np.mean(p_a - p_b, axis=1)
                    differences[:, 2] += np.mean(
                        np.abs(p_a - p_clean) - np.abs(p_b - p_clean), axis=1
                    )
                differences /= len(seeds)
                reference = clean[seeds[0]]
                patient_ids, inverse, counts = np.unique(
                    reference["patient_ids"], return_inverse=True, return_counts=True
                )
                audit = pd.DataFrame(
                    {
                        **base,
                        "ecg_id": reference["ids"],
                        "patient_id": reference["patient_ids"],
                        "n_seeds": len(seeds),
                        "loss_difference": differences[:, 0],
                        "mean_probability_difference": differences[:, 1],
                        "probability_shift_difference": differences[:, 2],
                    }
                )
                audit.to_csv(
                    audit_path,
                    mode="a",
                    index=False,
                    header=False,
                    columns=AUDIT_COLUMNS,
                )
                for index, outcome in enumerate(
                    ("loss", "mean_probability", "probability_shift")
                ):
                    patient_values = (
                        np.bincount(inverse, weights=differences[:, index]) / counts
                    )
                    stream_key = [
                        model,
                        kind,
                        int(snr),
                        active,
                        comparison,
                        condition,
                        outcome,
                    ]
                    bseed = derived_seed(random_seed, "bootstrap", *stream_key)
                    pseed = derived_seed(random_seed, "permutation", *stream_key)
                    family = (
                        HOLM_FAMILY
                        if outcome == "loss" and right != "clean" and active == "all"
                        else "exploratory_uncorrected"
                    )
                    result = cluster_inference(
                        patient_values, bootstrap, permutations, bseed, pseed
                    )
                    effects.append(
                        {
                            **base,
                            "outcome": outcome,
                            **result,
                            "p_holm": None,
                            "n_patients": len(patient_ids),
                            "n_records": n_records,
                            "n_seeds": len(seeds),
                            "inference_unit": "patient",
                            "holm_family": family,
                        }
                    )
                for metric in PRIMARY:
                    values = []
                    for seed in seeds:
                        a_row = lookup.loc[(left, seed)]
                        b_row = (
                            clean_rows.loc[seed]
                            if right == "clean"
                            else lookup.loc[(right, seed)]
                        )
                        a_value, b_value = float(a_row[metric]), float(b_row[metric])
                        values.append((seed, a_value, b_value, a_value - b_value))
                    summary = seed_summary([value[3] for value in values])
                    for seed, a_value, b_value, difference in values:
                        seed_rows.append(
                            {
                                **base,
                                "metric": metric,
                                "seed": seed,
                                "left_value": a_value,
                                "right_value": b_value,
                                "difference": difference,
                                "n_seeds": summary["n_seeds"],
                                "mean_difference": summary["mean"],
                                "std": summary["std"],
                                "ci95_low": summary["ci95_low"],
                                "ci95_high": summary["ci95_high"],
                            }
                        )
                    effects.append(
                        {
                            **base,
                            "outcome": metric,
                            "mean_difference": summary["mean"],
                            "ci95_low": summary["ci95_low"],
                            "ci95_high": summary["ci95_high"],
                            "n_patients": len(patient_ids),
                            "n_records": n_records,
                            "n_seeds": summary["n_seeds"],
                            "inference_unit": "training_seed",
                            "holm_family": "not_tested",
                            "bootstrap_count": 0,
                            "permutation_count": 0,
                            "permutation_method": "not_tested",
                            "status": (
                                "insufficient_seeds"
                                if summary["n_seeds"] < 2
                                else "seed_t_interval_fragile_with_few_seeds"
                            ),
                        }
                    )
    effects = pd.DataFrame(effects, columns=EFFECT_COLUMNS)
    family_mask = effects.holm_family.eq(HOLM_FAMILY) & effects.p_value.notna()
    effects.loc[family_mask, "p_holm"] = holm_adjust(
        effects.loc[family_mask, "p_value"]
    )
    return effects, pd.DataFrame(seed_rows, columns=SEED_COLUMNS), loader.files_checked


def run(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(__file__).resolve().parents[1]
    results = Path(config.get("results_dir", "results"))
    if not results.is_absolute():
        results = root / results
    run_name = str(config["run_name"])
    metrics_path = results / "metrics" / run_name / "metrics.csv"
    output = results / "tables" / run_name
    options = config.get("statistics", {})
    bootstrap = int(options.get("bootstrap", 2000))
    permutations = int(options.get("permutations", 10000))
    random_seed = int(options.get("seed", 20260916))
    if bootstrap < 1 or permutations < 1 or random_seed < 0:
        raise ValueError("bootstrap/permutations must be positive and seed nonnegative")
    evaluation = json.loads(
        (metrics_path.parent / "evaluation_protocol.json").read_text(encoding="utf-8")
    )
    if evaluation.get("status") != "completed":
        raise ValueError("Statistics require a completed evaluation")
    evaluation_config = {
        k: v
        for k, v in evaluation.get("config", {}).items()
        if k not in ("statistics", "plotting")
    }
    requested_config = {
        k: v for k, v in config.items() if k not in ("statistics", "plotting")
    }
    if evaluation_config != requested_config:
        raise ValueError("Evaluation configuration does not match requested statistics")
    frame, metrics = validate_metrics(pd.read_csv(metrics_path), config)
    if evaluation.get("n_evaluations") != len(frame):
        raise ValueError("Evaluation manifest row count does not match metrics")
    summary = summarize(frame, metrics)
    robustness, robustness_summary = robustness_tables(frame, metrics)
    rankings, correlations = ranking_tables(summary)
    output.mkdir(parents=True, exist_ok=True)
    effects, contrasts, files_checked = paired_tables(
        frame, root, output, bootstrap, permutations, random_seed
    )
    tables = {
        "metrics_summary": summary,
        "paired_effects": effects,
        "seed_contrasts": contrasts,
        "robustness": robustness,
        "robustness_summary": robustness_summary,
        "rankings": rankings,
        "rank_correlations": correlations,
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, na_rep="")
    undefined = {
        metric: int(frame[metric].isna().sum())
        for metric in metrics
        if frame[metric].isna().any()
    }
    protocol = {
        "run_name": run_name,
        "source_metrics": str(metrics_path),
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "config": config,
        "statistics_seed": random_seed,
        "bootstrap_requested": bootstrap,
        "permutations_requested": permutations,
        "random_streams": "SHA256 of base seed, method, model, noise kind, SNR, active, comparison, condition, outcome; first 8 bytes little-endian",
        "seed_summary": "Arithmetic mean, sample SD (ddof=1), Student-t 95% CI over finite training-seed metric values. n_seeds counts finite values; n=1 SD/CI undefined; no clipping to [0,1].",
        "undefined_metric_rows": undefined,
        "seed_contrast_ci": "Paired within-model training-seed metric differences, then Student-t CI; not patient bootstrap and not ensemble AUROC. No p-values claimed from few seeds.",
        "patient_estimand": "Average record-level left-minus-right contrasts over training seeds first; average records within each patient next; average patient means with equal patient weight. This differs from record-weighted metrics.csv losses.",
        "outcomes": {
            "loss": "Saved per-record multilabel loss, left minus right; positive means left is worse.",
            "mean_probability": "Mean over five classes of p_left - p_right; signed shift, not accuracy or magnitude.",
            "probability_shift": "Mean over five classes of abs(p_left-p_clean) - abs(p_right-p_clean); positive means left deviates more from the same-seed clean prediction.",
        },
        "bootstrap": "Percentile 95% CI from independently resampled patient means with replacement, preserving all records and seed-averaged contrasts within each patient. Conditional on trained seeds, not combined patient/training uncertainty.",
        "permutation": "Two-sided absolute equal-patient mean statistic. Independently flip whole patient contrasts; assumes independent patients and symmetric patient difference null. Exact enumeration when 2**n_patients <= requested permutations and n_patients<=20; otherwise Monte Carlo with (extreme+1)/(B+1).",
        "effect_size": "Paired patient Cohen dz = mean(patient differences) / sample SD(patient differences); null if fewer than 2 patients or zero SD; never infinite.",
        "influence_diagnostics": "Patient median and strict-positive fraction; descriptive mean after removing the largest absolute ceil(1%*n) or ceil(5%*n) patient contrasts (at least one, n>=2; stable input-order ties). These are sensitivity estimands, not alternative significance tests or changes to primary bootstrap/permutation results.",
        "holm_family": {
            "name": HOLM_FAMILY,
            "definition": "Every available primary patient-loss test for electrode-minus-independent_rms, electrode-minus-independent, covariance-minus-electrode, active=all, across all models, noise kinds and SNRs in this run, pooled as one family.",
            "n_tested": int(
                (effects.holm_family.eq(HOLM_FAMILY) & effects.p_value.notna()).sum()
            ),
            "other_outcomes": "Probability outcomes and noisy-minus-clean losses are exploratory, raw p only; p_holm blank. Seed metrics not hypothesis-tested.",
        },
        "comparison_labels": "Noise contrasts encode both conditions in comparison; noisy_minus_clean uses condition to identify the noisy condition; condition=all for the three noise-vs-noise contrasts.",
        "robustness": "clean_drop=clean-noisy; normalized_robustness=noisy/(clean+1e-12), paired model and training seed, for macro and per-class AUROC/AP/F1. Values at zero clean metric can be large; no clipping or removal.",
        "alignment": "Every NPZ verified for exact ordered IDs, labels, patient mappings and indices across models/seeds/conditions; within model/seed noisy thresholds must equal saved clean thresholds. No fitting or modification of thresholds.",
        "prediction_files_checked": files_checked,
        "rankings": "Descending seed-mean primary metric; ties get average ranks. Spearman and Kendall tau-b against clean on common finite architectures. Correlations are descriptive; p-values omitted for fixed architecture panel, exactly two architectures explicitly unstable/no meaningful inference; constant ranks undefined.",
        "null_encoding": "Undefined CSV cells are empty; JSON null, never NaN or Infinity.",
        "limitations": [
            "Few training seeds give fragile t intervals; one seed gives none.",
            "Patient bootstrap does not quantify retraining or noise-replay uncertainty.",
            "Noise-control audits must pass before interpreting causal structural contrasts.",
            "Raw exploratory p-values do not control familywise error; failure to reject is not equivalence.",
            "Synthetic/benchmark results do not support clinical deployment claims.",
        ],
        "schemas": {
            **{name + ".csv": list(table.columns) for name, table in tables.items()},
            "per_record_contrasts.csv": AUDIT_COLUMNS,
        },
    }
    with (output / "statistics_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        f"Statistics: {len(frame)} metric rows, {len(effects)} paired effects -> {output}"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
