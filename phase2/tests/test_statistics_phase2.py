"""Scientific edge cases: cluster weights, group estimands and paired ratios."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from phase1_ecg_robustness.src.supplemental_statistics import cluster_auc_distribution
from phase2.src.statistics_phase2 import (
    equal_condition_mean,
    group_tables,
    interval,
    noise_tables,
    outcome_value,
    paired_summary,
    paired_tables,
    patient_tables,
    ranking_tables,
)


def _group(group_id, case_ids, snr=-1):
    return dict(
        group_id=group_id,
        case_ids=case_ids,
        kind="clean" if group_id == "clean" else "bandpass",
        condition="clean" if group_id == "clean" else "electrode",
        combo_set="clean" if group_id == "clean" else "heldout",
        combo_id="none" if group_id == "clean" else "aggregate",
        snr=snr,
    )


def test_cluster_weights_keep_all_records_and_score_ties():
    # Patient zero owns two records, so sampling it twice repeats BOTH records.
    y = np.array([[0], [1], [1], [0], [1]])
    p = np.array([[0.5], [0.5], [0.8], [0.2], [0.2]])
    inverse = np.array([0, 0, 1, 2, 2])
    draws = np.array([[2, 0, 1], [0, 2, 1]], dtype=np.int32)
    actual = cluster_auc_distribution(y, p, inverse, draws, batch_size=1)
    expected = [roc_auc_score(y, p)]
    for draw in draws:
        records = np.repeat(np.arange(len(y)), draw[inverse])
        expected.append(roc_auc_score(y[records], p[records]))
    np.testing.assert_allclose(actual[:, 0], expected)
    np.testing.assert_allclose(actual[:, 1], expected)


def test_missing_class_draw_is_counted_and_macro_does_not_change_classes():
    y = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    p = np.array([[0.1, 0.1], [0.9, 0.3], [0.3, 0.8], [0.8, 0.9]])
    inverse = np.array([0, 0, 1, 1])
    # First patient alone contains both classes of target0, but no target1 positive.
    draws = np.array([[2, 0], [1, 1], [0, 2]], dtype=np.int32)
    distribution = cluster_auc_distribution(y, p, inverse, draws)
    assert distribution[1, 1] == 1
    assert np.isnan(distribution[1, 2])
    assert np.isnan(distribution[1, 0])
    result = interval(distribution[:, 0])
    assert result["n_bootstrap"] == 3
    assert result["n_valid"] == 1
    assert result["n_invalid"] == 2
    assert result["ci95_low"] == result["ci95_high"] == 1


def test_group_auroc_is_neither_ensemble_nor_pooled_conditions():
    y = np.array([[0], [1], [0], [1]])
    inverse = np.arange(4)
    draws = np.ones((1, 4), dtype=np.int32)
    a = np.array([[0.1], [0.2], [0.3], [0.4]])
    b = np.array([[0.8], [0.9], [0.6], [0.7]])
    distributions = [cluster_auc_distribution(y, p, inverse, draws) for p in (a, b)]
    grouped = equal_condition_mean(distributions)
    assert grouped[0, 0] == 0.75
    assert roc_auc_score(y, (a + b) / 2) == 1
    assert roc_auc_score(np.tile(y, (2, 1)), np.concatenate([a, b])) == 0.625
    distributions[1][1, 0] = np.nan
    assert np.isnan(equal_condition_mean(distributions)[1, 0])


def test_group_and_noise_retention_average_seed_ratios_not_ratio_of_means():
    groups = [_group("clean", ["clean"], 100), _group("primary_joint", ["a", "b"])]
    rows = []
    for strategy in ("clean_only", "electrode"):
        for seed, clean, noisy in ((17, 0.8, (0.6, 0.4)), (29, 0.4, (0.4, 0.2))):
            for case, value, noise in (
                ("clean", clean, 0),
                ("a", noisy[0], 20001),
                ("b", noisy[1], 20002),
            ):
                rows.append(
                    dict(
                        model="m",
                        strategy=strategy,
                        seed=seed,
                        case_id=case,
                        noise_seed=noise,
                        macro_auroc=value,
                        macro_ppv_n_defined=5,
                    )
                )
    raw, noise, primary = group_tables(
        pd.DataFrame(rows), groups, ["macro_auroc", "macro_ppv_n_defined"]
    )
    group_values = raw.loc[
        (raw.group_id == "primary_joint")
        & (raw.strategy == "electrode")
        & (raw.outcome == "retention"),
        "value",
    ]
    np.testing.assert_allclose(group_values, [0.5 / (0.8 + 1e-12), 0.3 / (0.4 + 1e-12)])
    base = noise.loc[
        (noise.strategy == "electrode")
        & (noise.noise_seed == 20001)
        & (noise.outcome == "retention")
    ].iloc[0]
    assert base.value == pytest.approx((0.6 / 0.8 + 0.4 / 0.4) / 2)
    assert base.value != pytest.approx(((0.6 + 0.4) / 2) / ((0.8 + 0.4) / 2))
    summary = noise_tables(noise)
    summary = summary.loc[
        (summary.strategy == "electrode") & (summary.outcome == "retention")
    ].iloc[0]
    assert summary.n_noise_seeds == 2
    assert summary["std"] == pytest.approx(np.std([0.875, 0.5], ddof=1))
    assert set(raw.loc[raw.metric == "macro_ppv_n_defined", "outcome"]) == {"absolute"}
    assert primary.loc[
        primary.strategy == "electrode", "value"
    ].tolist() == pytest.approx([0.75, 0.5, 1, 0.5])


def test_patient_ratio_and_paired_difference_are_formed_in_common_draws():
    strategies = ["clean_only", "independent_rms", "electrode", "mixed"]
    cfg = dict(
        models=["m"],
        strategies=dict.fromkeys(strategies),
        stages={"full": {"seeds": [17, 29]}},
    )
    groups = [_group("clean", ["clean"], 100), _group("primary_joint", ["a"])]
    arrays, refs = {}, {}
    for strategy in strategies:
        for seed in (17, 29):
            clean = np.array([0.8, 0.4, 0.9, 0.5])
            noisy = np.array([0.4, 0.3, 0.3, 0.3])
            if strategy == "electrode":
                noisy = np.array([0.6, 0.2, 0.8, 0.2])
            if seed == 29:
                clean = clean * 0.9
            arrays[(strategy, "m", seed)] = np.stack([clean, noisy])[..., None]
            refs[(strategy, "m", seed)] = f"{strategy}_{seed}.npy"
    table, lookup = patient_tables(cfg, "full", groups, ["macro_auroc"], arrays, refs)
    expected = np.mean(
        [
            outcome_value(
                arrays[("electrode", "m", seed)][1, :, 0],
                arrays[("electrode", "m", seed)][0, :, 0],
                0,
                "retention",
            )
            - outcome_value(
                arrays[("clean_only", "m", seed)][1, :, 0],
                arrays[("clean_only", "m", seed)][0, :, 0],
                0,
                "retention",
            )
            for seed in (17, 29)
        ],
        axis=0,
    )
    actual = lookup[
        ("m", "primary_joint", "macro_auroc", "retention", "electrode", "clean_only")
    ]
    np.testing.assert_allclose(
        [actual["ci95_low"], actual["ci95_high"]],
        np.quantile(expected[1:], [0.025, 0.975]),
    )
    assert actual["point"] == pytest.approx(expected[0])
    # The class of uncertainty is fixed checkpoint/mean-fixed-seed, never seed bootstrap.
    assert set(table.training_seed) == {"17", "29", "mean_fixed_seeds"}


def test_pairwise_missingness_does_not_shift_seed_alignment_and_pilot_has_no_ci():
    result = paired_summary([0.8, np.nan, 0.6, 0.9], [0.5, 0.4, np.nan, 0.5])
    assert result["n_seeds"] == 2
    assert result["n_undefined"] == 2
    assert result["point"] == pytest.approx(0.35)
    assert result["sd"] == pytest.approx(np.std([0.3, 0.4], ddof=1))
    pilot = paired_summary([0.8], [0.7], inferential=False)
    assert (
        pilot["sd"] is None and pilot["ci95_low"] is None and pilot["ci95_high"] is None
    )
    assert np.isnan(pilot["p_raw"]) and np.isnan(pilot["t_stat"])


def test_six_primary_tests_share_one_holm_family():
    seeds = [17, 29, 43, 101, 202]
    strategies = ["clean_only", "independent_rms", "electrode", "mixed"]
    pairs = [
        ["electrode", "clean_only"],
        ["electrode", "independent_rms"],
        ["mixed", "electrode"],
    ]
    cfg = dict(
        models=["resnet", "tcn"],
        strategies=dict.fromkeys(strategies),
        stages={"full": {"seeds": seeds}},
        statistics={"primary_pairs": pairs},
    )
    rows, lookup = [], {}
    for mi, model in enumerate(cfg["models"]):
        for strategy_index, strategy in enumerate(strategies):
            for si, seed in enumerate(seeds):
                value = (
                    0.6
                    + strategy_index * 0.013
                    + (strategy_index + 1) * (si - 2) * (mi + 1) * 0.004
                )
                rows.append(
                    dict(
                        model=model,
                        strategy=strategy,
                        seed=seed,
                        **{
                            k: v
                            for k, v in _group("primary_joint", ["a"]).items()
                            if k != "case_ids"
                        },
                        metric="macro_auroc",
                        outcome="retention",
                        value=value,
                    )
                )
        for lhs, rhs in pairs:
            lookup[(model, "primary_joint", "macro_auroc", "retention", lhs, rhs)] = (
                dict(ci95_low=-0.1, ci95_high=0.1, n_valid=20, n_invalid=0)
            )
    frame = pd.DataFrame(rows)
    _, primary = paired_tables(cfg, "full", frame, lookup)
    expected_p = []
    for model in cfg["models"]:
        pivot = (
            frame.loc[frame.model == model]
            .pivot(index="seed", columns="strategy", values="value")
            .reindex(seeds)
        )
        for lhs, rhs in pairs:
            expected_p.append(stats.ttest_rel(pivot[lhs], pivot[rhs]).pvalue)
    np.testing.assert_allclose(primary.p_raw, expected_p)
    order = np.argsort(expected_p)
    adjusted = np.minimum(
        1, np.maximum.accumulate(np.asarray(expected_p)[order] * np.arange(6, 0, -1))
    )
    np.testing.assert_allclose(primary.p_holm.to_numpy()[order], adjusted)
    assert primary.positive_seed_count.min() > 0


def test_ranks_use_average_ties_and_calibration_has_lower_is_better_orientation():
    groups = [_group("clean", ["clean"], 100), _group("primary_joint", ["a"])]
    strategies = ["clean_only", "independent_rms", "electrode", "mixed"]
    rows = []
    for group in groups:
        for strategy, value in zip(strategies, [0.4, 0.4, 0.8, 0.9]):
            for metric in ("macro_auroc", "brier"):
                rows.append(
                    dict(
                        model="m",
                        seed=17,
                        strategy=strategy,
                        group_id=group["group_id"],
                        metric=metric,
                        outcome="absolute",
                        value=value,
                    )
                )
    rank, summary, correlation = ranking_tables(pd.DataFrame(rows), groups, strategies)
    auc = rank.loc[
        (rank.group_id == "clean") & (rank.metric == "macro_auroc")
    ].set_index("strategy")
    assert auc.loc["clean_only", "rank"] == 3.5
    brier = rank.loc[(rank.group_id == "clean") & (rank.metric == "brier")].set_index(
        "strategy"
    )
    assert brier.loc["clean_only", "rank"] == 1.5
    assert summary.std_rank.isna().all()
    np.testing.assert_allclose(correlation.spearman, 1)
