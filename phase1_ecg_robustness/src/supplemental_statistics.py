"""Fixed-model patient-cluster AUROC CIs and separate seed/noise descriptions.

Run only after the primary, five-noise-replay and strict-control producers finish.
No fitting, probability ensembles, p-values or joint uncertainty intervals are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .evaluate import CLASSES, sha256
from .lead_matrix import matrix_provenance
from .statistics import PRIMARY, PredictionLoader, seed_summary, validate_metrics

OLD_CONDITIONS = ("independent", "independent_rms", "electrode", "covariance")
CONTRASTS = (
    ("electrode", "independent_rms"),
    ("covariance", "electrode"),
    ("electrode", "marginal_shift"),
)
AUC_METRICS = ("macro_auroc",) + tuple(f"auroc_{name}" for name in CLASSES)
GROUP = ["model", "kind", "snr", "condition", "active"]
CONDITIONING = (
    "95% percentile patient-cluster CI; fixed trained models, fixed test-noise "
    "realization (base 8128); not joint patient/training/noise uncertainty"
)


class _PreparedAUC:
    """Sort once; sum positive/negative multiplicities within exact score ties."""

    def __init__(self, y, scores):
        y, scores = np.asarray(y), np.asarray(scores)
        if y.ndim != 1 or y.shape != scores.shape or not len(y):
            raise ValueError("AUC needs nonempty aligned one-dimensional inputs")
        if not np.isin(y, (0, 1)).all() or not np.isfinite(scores).all():
            raise ValueError("AUC needs binary labels and finite scores")
        self.order = np.argsort(scores, kind="stable")
        ordered = scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        self.positive = y[self.order].astype(np.float64)
        self.negative = 1.0 - self.positive

    def evaluate(self, weights):
        ordered = weights[:, self.order]
        positive = np.add.reduceat(ordered * self.positive, self.starts, axis=1)
        negative = np.add.reduceat(ordered * self.negative, self.starts, axis=1)
        denominator = positive.sum(axis=1) * negative.sum(axis=1)
        numerator = np.sum(
            positive * (np.cumsum(negative, axis=1) - 0.5 * negative), axis=1
        )
        result = np.full(len(weights), np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=result, where=denominator > 0)
        return result


def weighted_auc(y, scores, weights):
    """Tie-correct record AUROC; zero class weight returns NaN, not a new draw.

    ``weights`` is one record-weight vector or a batch of vectors. Integer
    patient multiplicities are exactly equivalent to replicating all records.
    """
    weights = np.asarray(weights, dtype=np.float64)
    scalar = weights.ndim == 1
    weights = np.atleast_2d(weights)
    if weights.ndim != 2 or weights.shape[1] != len(y):
        raise ValueError("Weights must align with records")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Weights must be finite and nonnegative")
    result = _PreparedAUC(y, scores).evaluate(weights)
    return float(result[0]) if scalar else result


def patient_draws(n_patients, repetitions, seed, batch_size):
    """Uniform cluster bootstrap represented by patient selection multiplicities."""
    if min(n_patients, repetitions, batch_size) < 1 or seed < 0:
        raise ValueError(
            "Positive patients/repetitions/batch size and nonnegative seed required"
        )
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, n_patients), dtype=np.int32)
    probability = np.full(n_patients, 1.0 / n_patients)
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        draws[start:stop] = rng.multinomial(n_patients, probability, size=stop - start)
    return draws


def cluster_auc_distribution(y, p, patient_inverse, multiplicities, batch_size=128):
    """Return point then draws, columns macro followed by every input class.

    Macro requires every class to be valid: ordinary mean intentionally
    propagates NaN rather than silently changing the class set.
    """
    y, p = np.asarray(y), np.asarray(p)
    inverse = np.asarray(patient_inverse)
    multiplicities = np.asarray(multiplicities)
    if y.ndim != 2 or y.shape != p.shape or inverse.shape != (len(y),):
        raise ValueError(
            "Labels, probabilities and record-to-patient mapping must align"
        )
    if not np.issubdtype(inverse.dtype, np.integer) or np.any(inverse < 0):
        raise ValueError("Patient mapping must contain nonnegative integer indices")
    if (
        multiplicities.ndim != 2
        or not np.issubdtype(multiplicities.dtype, np.integer)
        or np.any(multiplicities < 0)
        or not len(inverse)
        or inverse.max() >= multiplicities.shape[1]
        or not np.all(multiplicities.sum(axis=1) == multiplicities.shape[1])
        or batch_size < 1
    ):
        raise ValueError("Each bootstrap draw must select exactly the patient count")
    prepared = [_PreparedAUC(y[:, j], p[:, j]) for j in range(y.shape[1])]
    result = np.empty((len(multiplicities) + 1, y.shape[1] + 1))
    unit = np.ones((1, len(y)), dtype=np.float64)
    for j, auc in enumerate(prepared, start=1):
        result[0, j] = auc.evaluate(unit)[0]
    for start in range(0, len(multiplicities), batch_size):
        stop = min(start + batch_size, len(multiplicities))
        weights = multiplicities[start:stop, inverse].astype(np.float64)
        for j, auc in enumerate(prepared, start=1):
            result[start + 1 : stop + 1, j] = auc.evaluate(weights)
    result[:, 0] = result[:, 1:].mean(axis=1)
    return result


def mean_seed_auc(values):
    """Average seed-specific AUROCs/contrasts, never probabilities or valid subsets."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or not len(values):
        raise ValueError("Seed-specific estimates require a nonempty seed axis")
    return values.mean(axis=0)


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _check_grid(frame, expected, label):
    keys = ["model", "seed", "noise_seed", "kind", "snr", "condition", "active"]
    required = keys + ["prediction_path"] + list(AUC_METRICS)
    if set(required) - set(frame):
        raise ValueError(
            f"{label}: missing columns {sorted(set(required) - set(frame))}"
        )
    if frame[required].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{label}: missing values or duplicate evaluation keys")
    for field in ("seed", "noise_seed", "snr"):
        values = pd.to_numeric(frame[field], errors="raise")
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError(f"{label}: {field} must be finite integers")
        frame[field] = values.astype(np.int64)
    actual = set(frame[keys].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(
            f"{label}: configured grid mismatch: {len(expected-actual)} missing, "
            f"{len(actual-expected)} unexpected"
        )


def _sources(root, config, results):
    base_path = _resolve(root, config["base_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    primary_dir = results / "metrics" / base["run_name"]
    replay_dir = results / "metrics" / f"{base['run_name']}_noise_repeats"
    strict_dir = results / "metrics" / config["run_name"]
    paths = {
        "base_config": base_path,
        "primary_metrics": primary_dir / "metrics.csv",
        "primary_protocol": primary_dir / "evaluation_protocol.json",
        "replay_metrics": replay_dir / "metrics.csv",
        "replay_protocol": replay_dir / "protocol.json",
        "strict_metrics": strict_dir / "strict_metrics.csv",
        "strict_protocol": strict_dir / "strict_protocol.json",
    }
    protocols = {
        name: _load_json(paths[f"{name}_protocol"])
        for name in ("primary", "replay", "strict")
    }
    for name, protocol in protocols.items():
        if protocol.get("status") != "completed":
            raise ValueError(f"{name}: completed evaluation protocol required")
    if protocols["primary"].get("config") != base:
        raise ValueError(
            "Primary protocol configuration differs from the full source config"
        )
    for name in ("replay", "strict"):
        if protocols[name].get("source_config") != base:
            raise ValueError(f"{name}: source configuration differs from the primary")
    if protocols["strict"].get("config") != config:
        raise ValueError("Strict protocol differs from the supplement configuration")
    matrix = matrix_provenance()
    if (
        matrix["sha256"] != config["matrix_sha256"]
        or protocols["strict"].get("matrix") != matrix
    ):
        raise ValueError(
            "Strict protocol/config canonical lead matrix differs from current matrix"
        )
    checkpoints = protocols["primary"].get("checkpoints_sha256", {})
    if len(checkpoints) != len(base["models"]) * len(base["seeds"]):
        raise ValueError("Primary checkpoint hash coverage is incomplete")
    canonical_checkpoints = {
        _resolve(root, path).resolve(): digest for path, digest in checkpoints.items()
    }
    if len(canonical_checkpoints) != len(checkpoints):
        raise ValueError("Primary checkpoint paths alias the same file")
    for name in ("replay", "strict"):
        recorded = protocols[name].get("checkpoint_sha256", {})
        canonical_recorded = {
            _resolve(root, path).resolve(): digest for path, digest in recorded.items()
        }
        if (
            len(canonical_recorded) != len(recorded)
            or canonical_recorded != canonical_checkpoints
        ):
            raise ValueError(
                f"{name}: checkpoints do not match the primary fixed models"
            )
    for path, digest in checkpoints.items():
        if sha256(_resolve(root, path)) != digest:
            raise ValueError(f"Checkpoint changed: {path}")
    primary, metrics = validate_metrics(pd.read_csv(paths["primary_metrics"]), base)
    replay = pd.read_csv(paths["replay_metrics"])
    replay["kind"], replay["active"] = "bandpass", "all"
    strict = pd.read_csv(paths["strict_metrics"])
    control = config["strict_control"]
    primary_seed = int(base["noise"]["seed"])
    noise_seeds = list(map(int, control["noise_seeds"]))
    replay_seeds = [seed for seed in noise_seeds if seed != primary_seed]
    if (
        primary_seed != 8128
        or len(base["models"]) != 2
        or len(base["seeds"]) != 3
        or len(set(base["seeds"])) != 3
        or len(set(noise_seeds)) != 6
        or primary_seed not in noise_seeds
        or len(replay_seeds) != 5
        or set(base["noise"]["snrs"]) != {20, 10, 0}
        or set(control["snrs"]) != {20, 10, 0}
        or control["condition"] != "marginal_shift"
        or control["kind"] != "bandpass"
    ):
        raise ValueError(
            "Supplement requires two models, three seeds, 20/10/0 dB and base8128 plus five replays"
        )
    if (
        protocols["replay"].get("noise_seeds") != replay_seeds
        or protocols["replay"].get("kind") != "bandpass"
    ):
        raise ValueError(
            "Replay protocol noise bases/kind differ from configured coverage"
        )
    _check_grid(
        replay,
        set(
            product(
                base["models"],
                base["seeds"],
                replay_seeds,
                ["bandpass"],
                control["snrs"],
                OLD_CONDITIONS,
                ["all"],
            )
        ),
        "replay",
    )
    _check_grid(
        strict,
        set(
            product(
                base["models"],
                base["seeds"],
                noise_seeds,
                ["bandpass"],
                control["snrs"],
                ["marginal_shift"],
                ["all"],
            )
        ),
        "strict",
    )
    for name, frame in (("primary", primary), ("replay", replay), ("strict", strict)):
        if protocols[name].get("n_evaluations") != len(frame):
            raise ValueError(f"{name}: protocol evaluation count differs from table")
        for metric in metrics:
            if (
                metric not in frame
                or not np.isfinite(pd.to_numeric(frame[metric], errors="raise")).all()
            ):
                raise ValueError(f"{name}: missing or nonfinite metric {metric}")
    primary["noise_seed"] = primary_seed
    return base, protocols, paths, primary, replay, strict, metrics, replay_seeds


def _validate_predictions(root, frames, primary, protocols, matrix_hash):
    loader = PredictionLoader(root)
    clean = {}
    retained = {}
    sources = []
    clean_rows = primary[primary.condition.eq("clean")]
    ordered = [("primary", row) for row in clean_rows.itertuples(index=False)]
    for name, frame in frames:
        ordered.extend(
            (name, row)
            for row in frame[~frame.condition.eq("clean")].itertuples(index=False)
        )
    for name, row in ordered:
        key = (row.model, int(row.seed))
        reference = None if row.condition == "clean" else clean[key]
        data = loader.load(row, reference)
        if row.condition == "clean":
            clean[key] = data
        path = _resolve(root, row.prediction_path)
        if name == "strict":
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "matrix_sha256",
                    "actual_snr",
                    "noise_rms",
                    "lead_snr",
                    "replay_seed",
                    "shifts",
                }
                if required - set(archive.files):
                    raise ValueError(f"{path}: missing strict-control provenance")
                if str(archive["matrix_sha256"].item()) != matrix_hash:
                    raise ValueError(f"{path}: strict matrix hash mismatch")
        observed = np.array(
            [
                weighted_auc(
                    data["y"][:, j], data["p"][:, j], np.ones(len(data["ids"]))
                )
                for j in range(len(CLASSES))
            ]
        )
        observed = np.r_[observed.mean(), observed]
        reported = np.array([getattr(row, metric) for metric in AUC_METRICS])
        if not np.allclose(observed, reported, rtol=0, atol=1e-10):
            raise ValueError(
                f"{path}: saved AUROC metrics differ from aligned probabilities"
            )
        sources.append(
            {
                "source": name,
                "model": row.model,
                "seed": row.seed,
                "noise_seed": row.noise_seed,
                "kind": row.kind,
                "snr": row.snr,
                "condition": row.condition,
                "active": row.active,
                "prediction_path": str(row.prediction_path),
                "sha256": sha256(path),
            }
        )
        if row.active == "all" and row.noise_seed == 8128:
            retained[
                (row.model, int(row.seed), row.kind, int(row.snr), row.condition)
            ] = data["p"]
    reference = loader.reference
    ids_hash = hashlib.sha256(reference["ids"].astype(np.int64).tobytes()).hexdigest()
    if protocols["primary"].get("ids_sha256") != ids_hash:
        raise ValueError(
            "Aligned test ECG order differs from the original protocol hash"
        )
    n_records, n_patients = len(reference["ids"]), len(
        np.unique(reference["patient_ids"])
    )
    for name, protocol in protocols.items():
        for field, actual in (("n_records", n_records), ("n_patients", n_patients)):
            if field in protocol and protocol[field] != actual:
                raise ValueError(
                    f"{name}: {field} differs from aligned prediction metadata"
                )
    if (n_records, n_patients) != (2158, 1877):
        raise ValueError(
            "Full supplement requires all 2158 test ECGs and 1877 patients"
        )
    return retained, reference, pd.DataFrame(sources), clean


def _descriptive_tables(primary, replay, strict, metrics, seeds):
    # Existing normalized-robustness convention: noisy/(matched clean + 1e-12).
    clean = primary[primary.condition.eq("clean")].set_index(["model", "seed"])
    primary_all = primary[primary.active.eq("all")]
    combined = pd.concat([primary_all, replay, strict], ignore_index=True)
    detail = []
    for row in combined.itertuples(index=False):
        base = {key: getattr(row, key) for key in GROUP + ["seed", "noise_seed"]}
        for metric in metrics:
            value = float(getattr(row, metric))
            clean_value = float(clean.loc[(row.model, row.seed), metric])
            outcomes = [("absolute", value)]
            if metric in PRIMARY or metric.startswith(("auroc_", "ap_", "f1_")):
                outcomes.extend(
                    [
                        ("clean_drop", clean_value - value),
                        ("normalized_robustness", value / (clean_value + 1e-12)),
                    ]
                )
            for outcome, estimate in outcomes:
                detail.append(
                    {**base, "metric": metric, "outcome": outcome, "value": estimate}
                )
    detail = pd.DataFrame(detail)
    primary_detail = detail[detail.noise_seed.eq(8128)]
    summary = []
    keys = GROUP + ["metric", "outcome"]
    for key, group in primary_detail.groupby(keys, sort=True):
        if set(group.seed) != set(seeds) or len(group) != len(seeds):
            raise ValueError(f"Incomplete primary training-seed group {key}")
        summary.append(
            {
                **dict(zip(keys, key)),
                "noise_seed": 8128,
                "uncertainty": "training_seed_student_t_95",
                **seed_summary(group.value),
            }
        )
    # Clean is a single reused reference, not five new clean-noise experiments.
    noise_detail = detail[~detail.noise_seed.eq(8128)]
    per_noise = []
    for key, group in noise_detail.groupby(keys + ["noise_seed"], sort=True):
        if set(group.seed) != set(seeds) or len(group) != len(seeds):
            raise ValueError(f"Incomplete replay training-seed group {key}")
        per_noise.append(
            {
                **dict(zip(keys + ["noise_seed"], key)),
                "n_training_seeds": len(seeds),
                "mean_over_training_seeds": float(group.value.mean()),
            }
        )
    per_noise = pd.DataFrame(per_noise)
    replay_summary = []
    for key, group in per_noise.groupby(keys, sort=True):
        if len(group) != 5:
            raise ValueError(f"Incomplete five-noise-replay group {key}")
        replay_summary.append(
            {
                **dict(zip(keys, key)),
                "n_noise_replays": 5,
                "n_fixed_training_seeds": len(seeds),
                "mean": float(group.mean_over_training_seeds.mean()),
                "std": float(group.mean_over_training_seeds.std(ddof=1)),
                "uncertainty": "descriptive_noise_replay_sd_after_training_seed_average_no_CI",
            }
        )
    return {
        "all_snr_seed_metric_detail.csv": detail,
        "primary_three_seed_mean_sd_tci.csv": pd.DataFrame(summary),
        "five_noise_replay_seed_averaged_detail.csv": per_noise,
        "five_noise_replay_mean_sd.csv": pd.DataFrame(replay_summary),
    }


def _bootstrap_tables(probabilities, reference, base, draws, batch_size):
    patients, inverse = np.unique(reference["patient_ids"], return_inverse=True)
    distributions = {}
    for key, p in probabilities.items():
        distributions[key] = cluster_auc_distribution(
            reference["y"], p, inverse, draws, batch_size
        )
    rows, columns = [], []
    seeds = list(map(int, base["seeds"]))

    def append(model, kind, snr, outcome, lhs, rhs, values):
        for seed, value in list(zip(seeds, values)) + [
            ("mean_fixed_seeds", mean_seed_auc(values))
        ]:
            for j, metric in enumerate(AUC_METRICS):
                samples = value[1:, j]
                finite = np.isfinite(samples)
                n_valid = int(finite.sum())
                low, high = (
                    np.percentile(samples[finite], [2.5, 97.5])
                    if n_valid
                    else (np.nan, np.nan)
                )
                rows.append(
                    {
                        "distribution_index": len(columns),
                        "model": model,
                        "kind": kind,
                        "snr": snr,
                        "active": "all",
                        "noise_seed": 8128,
                        "training_seed": seed,
                        "n_fixed_training_seeds": (
                            len(seeds) if isinstance(seed, str) else 1
                        ),
                        "outcome": outcome,
                        "lhs": lhs,
                        "rhs": rhs,
                        "metric": metric,
                        "point": float(value[0, j]),
                        "ci95_low": float(low),
                        "ci95_high": float(high),
                        "n_bootstrap": len(draws),
                        "n_valid": n_valid,
                        "n_invalid": int(len(draws) - n_valid),
                        "ci_method": "patient_cluster_percentile_95",
                        "conditioning": CONDITIONING,
                    }
                )
                columns.append(samples)

    for model in base["models"]:
        clean = np.stack(
            [distributions[(model, seed, "clean", 100, "clean")] for seed in seeds]
        )
        append(model, "clean", 100, "absolute", "clean", "", clean)
        for kind in base["noise"]["kinds"]:
            conditions = OLD_CONDITIONS + (
                ("marginal_shift",) if kind == "bandpass" else ()
            )
            for snr in base["noise"]["snrs"]:
                estimates = {}
                for condition in conditions:
                    value = np.stack(
                        [
                            distributions[(model, seed, kind, int(snr), condition)]
                            for seed in seeds
                        ]
                    )
                    estimates[condition] = value
                    append(model, kind, snr, "absolute", condition, "", value)
                    append(
                        model,
                        kind,
                        snr,
                        "clean_drop",
                        "clean",
                        condition,
                        clean - value,
                    )
                for lhs, rhs in CONTRASTS:
                    if rhs in conditions:
                        append(
                            model,
                            kind,
                            snr,
                            "paired_contrast",
                            lhs,
                            rhs,
                            estimates[lhs] - estimates[rhs],
                        )
    return pd.DataFrame(rows), np.stack(columns, axis=1), patients, inverse


def _figures(table, distributions, directory, base, config_path):
    from matplotlib.ticker import MaxNLocator

    from .plotting import COLORS, FigureWriter, _legend, plt

    root = Path(__file__).resolve().parents[1]
    writer = FigureWriter(
        root, directory, directory.name, base["sampling_rate"], pdf=True
    )
    source_tables = directory.parent.parent / "tables" / directory.name
    selected = table[
        table.training_seed.eq("mean_fixed_seeds") & table.outcome.eq("paired_contrast")
    ]
    colors = (COLORS["independent_rms"], COLORS["covariance"], "#D55E00")
    for kind in base["noise"]["kinds"]:
        pairs = [
            pair
            for pair in CONTRASTS
            if pair[1] != "marginal_shift" or kind == "bandpass"
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        classfig, classaxes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
        for i, model in enumerate(base["models"]):
            for j, snr in enumerate(base["noise"]["snrs"]):
                subset = selected[
                    selected.model.eq(model)
                    & selected.kind.eq(kind)
                    & selected.snr.eq(snr)
                ]
                ax, cax = axes[i, j], classaxes[i, j]
                for k, (lhs, rhs) in enumerate(pairs):
                    block = subset[subset.lhs.eq(lhs) & subset.rhs.eq(rhs)]
                    macro = block[block.metric.eq("macro_auroc")].iloc[0]
                    samples = distributions[:, int(macro.distribution_index)]
                    valid = samples[np.isfinite(samples)]
                    label = f"{lhs} − {rhs} (valid {len(valid)}/{len(samples)})"
                    if len(valid):
                        ax.hist(
                            valid,
                            bins=40,
                            histtype="step",
                            linewidth=1.5,
                            color=colors[k],
                            label=label,
                        )
                        ax.axvline(macro.point, color=colors[k], linewidth=1)
                        ax.axvspan(
                            macro.ci95_low, macro.ci95_high, color=colors[k], alpha=0.06
                        )
                    else:
                        ax.plot([], [], color=colors[k], label=label + "; CI undefined")
                    classes = block.set_index("metric").loc[list(AUC_METRICS[1:])]
                    y = np.arange(len(CLASSES)) + (k - (len(pairs) - 1) / 2) * 0.2
                    # Draw endpoints directly: percentile intervals need not contain the point.
                    cax.hlines(
                        y,
                        classes.ci95_low,
                        classes.ci95_high,
                        color=colors[k],
                        linewidth=1.7,
                    )
                    cax.scatter(
                        classes.point, y, color=colors[k], s=19, label=f"{lhs} − {rhs}"
                    )
                    for cy, row in zip(y, classes.itertuples()):
                        if row.n_invalid:
                            cax.annotate(
                                f"invalid {row.n_invalid}/{row.n_bootstrap}",
                                (row.point, cy),
                                fontsize=6,
                            )
                for axis in (ax, cax):
                    axis.axvline(0, color="0.5", linewidth=0.7, linestyle="--")
                    axis.set_title(f"{model} · {snr} dB")
                    axis.spines[["top", "right"]].set_visible(False)
                    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
                    axis.ticklabel_format(
                        axis="x", style="sci", scilimits=(-3, 3), useMathText=True
                    )
                ax.set_xlabel("Paired macro AUROC difference")
                ax.set_ylabel("Patient-bootstrap draws")
                cax.set_yticks(np.arange(len(CLASSES)), CLASSES)
                cax.invert_yaxis()
                cax.set_xlabel("Paired class AUROC difference")
        _legend(fig, axes, ncol=len(pairs))
        _legend(classfig, classaxes, ncol=len(pairs))
        conditioning_note = (
            "Patient clusters resampled; all records retained. Fixed models and noise base 8128.\n"
            "Not an ensemble, training-seed CI, noise-replay CI, or joint CI."
        )
        if kind != "bandpass":
            conditioning_note += "\nConfounded NSTDB sensitivity replay; not isolated structural evidence."
        for figure, stem, description in (
            (
                fig,
                f"patient_bootstrap_distributions_{kind}",
                "Macro contrasts; lines=points, shaded bands=95% percentile CIs",
            ),
            (
                classfig,
                f"patient_bootstrap_classwise_paired_ci_{kind}",
                "All five class contrasts; points and 95% percentile CIs",
            ),
        ):
            figure.suptitle(
                f"{kind}: mean of three fixed-seed AUROC differences\n{description}",
                fontsize=12,
            )
            figure.text(
                0.5,
                0.055,
                conditioning_note,
                ha="center",
                fontsize=9,
                va="bottom",
            )
            figure.tight_layout(rect=(0, 0.14, 1, 0.96))
            writer.save(
                figure,
                stem,
                stem,
                f"{kind}: {description}. {CONDITIONING} {conditioning_note}",
                [
                    source_tables / "patient_bootstrap_auroc_ci.csv",
                    source_tables / "patient_bootstrap_distributions.npz",
                ],
            )
    path = writer.finish(config_path)
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}


def run(config_path):
    root = Path(__file__).resolve().parents[1]
    config_path = _resolve(root, config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results = _resolve(root, config.get("results_dir", "results"))
    output = results / "tables" / config["run_name"]
    logs = results / "logs" / config["run_name"]
    figures = results / "figures" / config["run_name"]
    options = config["bootstrap"]
    repetitions, seed, batch_size = (
        int(options[key]) for key in ("repetitions", "seed", "batch_size")
    )
    if repetitions != 2000 or seed != 20260917 or batch_size < 1:
        raise ValueError(
            "Review protocol requires 2000 bootstrap draws, seed20260917 and positive batch size"
        )
    base, protocols, paths, primary, replay, strict, metrics, replay_seeds = _sources(
        root, config, results
    )
    paths["supplement_config"] = config_path
    probabilities, reference, sources, clean = _validate_predictions(
        root,
        [("primary", primary), ("replay", replay), ("strict", strict)],
        primary,
        protocols,
        config["matrix_sha256"],
    )
    patients = np.unique(reference["patient_ids"])
    draws = patient_draws(len(patients), repetitions, seed, batch_size)
    tables = _descriptive_tables(primary, replay, strict, metrics, base["seeds"])
    table, distributions, patients, inverse = _bootstrap_tables(
        probabilities, reference, base, draws, batch_size
    )
    tables["patient_bootstrap_auroc_ci.csv"] = table
    tables["prediction_source_checksums.csv"] = sources
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    draw_path = output / "patient_bootstrap_draws.npz"
    distribution_path = output / "patient_bootstrap_distributions.npz"
    np.savez_compressed(
        draw_path,
        patient_multiplicities=draws,
        patient_order=patients,
        record_to_patient=inverse,
        ids=reference["ids"],
        patient_ids=reference["patient_ids"],
        indices=reference["indices"],
        y=reference["y"],
        seed=np.array(seed),
        repetitions=np.array(repetitions),
        batch_size=np.array(batch_size),
        replicate_index=np.arange(repetitions),
    )
    ordered_clean = [
        clean[(model, int(training_seed))]
        for model in base["models"]
        for training_seed in base["seeds"]
    ]
    np.savez_compressed(
        distribution_path,
        auroc_estimates=distributions,
        distribution_index=table.distribution_index.to_numpy(),
        replicate_index=np.arange(repetitions),
        models=np.asarray(base["models"]),
        training_seeds=np.asarray(base["seeds"]),
        class_order=np.asarray(CLASSES),
        clean_thresholds=np.stack([item["thresholds"] for item in ordered_clean]),
    )
    figure_manifest = _figures(table, distributions, figures, base, config_path)
    manifest = {
        "status": "completed",
        "config": config,
        "source_config": base,
        "matrix": matrix_provenance(),
        "matrix_provenance_scope": "Strict artifacts store their generation hash; legacy sources are identified by checksums, not backdated matrix hashes.",
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "checkpoint_sha256": protocols["primary"]["checkpoints_sha256"],
        "prediction_checksums": "prediction_source_checksums.csv",
        "n_prediction_files_checked": len(sources),
        "n_records": len(reference["ids"]),
        "n_patients": len(patients),
        "models": base["models"],
        "training_seeds": base["seeds"],
        "noise_seed": 8128,
        "noise_replay_seeds": replay_seeds,
        "class_order": list(CLASSES),
        "bootstrap": options,
        "draw_method": "NumPy default_rng PCG64 multinomial(n_patients, uniform patient probabilities); shared patient multiplicities for every model/seed/class/condition/SNR",
        "numpy_version": np.__version__,
        "conditioning": CONDITIONING,
        "estimands": {
            "individual_seed": "Record-level AUROC after retaining all records of sampled patients with cluster multiplicity",
            "mean_fixed_seeds": "Arithmetic mean of three seed-specific AUROCs or paired differences inside each draw; never probability ensembling",
            "macro": "Arithmetic mean of all five class AUROCs; undefined if any class is undefined",
            "clean_drop": "Matched seed clean AUROC minus noisy AUROC using the same patient draw",
            "normalized_robustness": "Noisy metric / (matched clean metric + 1e-12), following src.statistics; descriptive only",
            "training_seed_interval": "Student-t 95% CI over the three original training seeds; SD uses ddof=1",
            "noise_replay": "Average three fixed training-seed metrics (or paired clean drops/ratios) first, then descriptive mean and sample SD across five noise bases; no CI and no claim of 15 trainings",
        },
        "degeneracy": "A missing positive or negative class has NaN AUROC; no redraw and no class deletion. Percentile endpoints use finite draws; n_valid/n_invalid explicitly accompany every interval. Macro/fixed-seed means propagate invalid components.",
        "undefined_counts": {
            "point_estimates": int(table.point.isna().sum()),
            "intervals": int(table.ci95_low.isna().sum()),
            "rows_with_invalid_draws": int(table.n_invalid.gt(0).sum()),
        },
        "intentional_scope": [
            "Patient bootstrap uses only primary noise base8128.",
            "Strict marginal_shift and five-noise replays exist only for bandpass/all; primary original conditions include every configured kind and SNR.",
            "Single-electrode legacy rows are alignment/checksum validated but excluded from these all-active supplement estimands.",
            "Clean is a reused reference, not five new clean-noise observations.",
            "No p-values, equivalence decisions, probability ensemble, or joint uncertainty interval.",
        ],
        "distribution_schema": {
            "shape": list(distributions.shape),
            "rows": "zero-based replicate_index shared with patient_bootstrap_draws.npz",
            "columns": "distribution_index in patient_bootstrap_auroc_ci.csv",
            "values": "AUROC or AUROC difference, including explicit NaN invalid draws",
            "clean_thresholds_order": "model-major, then configured training_seeds; five class columns",
        },
        "outputs": {
            name: {"path": str(output / name), "sha256": sha256(output / name)}
            for name in list(tables) + [draw_path.name, distribution_path.name]
        },
        "figures": figure_manifest,
    }
    (logs / "supplemental_statistics_protocol.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Completed supplemental statistics: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_supplement.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
