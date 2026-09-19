"""Defend record-level clustered AUROC and fixed-seed (not ensemble) estimands."""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.supplemental_statistics import (
    cluster_auc_distribution,
    mean_seed_auc,
    patient_draws,
    weighted_auc,
)


def test_weighted_auc_ties_and_zero_weights_match_explicit_record_replication():
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.2, 0.2, 0.7, 0.7, 0.9, 0.9])
    weights = np.array([[2, 1, 0, 3, 1, 0], [0, 2, 4, 1, 0, 3]])
    expected = []
    for draw in weights:
        records = np.repeat(np.arange(len(y)), draw)
        expected.append(roc_auc_score(y[records], p[records]))
    np.testing.assert_allclose(weighted_auc(y, p, weights), expected, atol=1e-15)
    assert np.isnan(weighted_auc(y, p, np.zeros(len(y))))
    assert np.isnan(weighted_auc(y, p, y))


def test_cluster_draw_retains_every_record_in_unequal_sized_patients():
    # Two negative ECGs for patient0, one positive for patient1, two for patient2.
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
    inverse = np.array([0, 0, 1, 2, 2])
    draws = np.array([[2, 0, 1], [1, 1, 1], [0, 0, 3]])
    result = cluster_auc_distribution(
        y[:, None], p[:, None], inverse, draws, batch_size=2
    )
    for index, counts in enumerate(draws[:2], start=1):
        records = np.repeat(np.arange(len(y)), counts[inverse])
        expected = roc_auc_score(y[records], p[records])
        np.testing.assert_allclose(result[index], [expected, expected])
    # Patient-averaged probabilities would give 0, not the required record AUC 0.5.
    averaged_patient_auc = roc_auc_score([0, 0, 1], [0.5, 0.5, 0.45])
    assert result[1, 0] == 0.5
    assert result[1, 0] != averaged_patient_auc
    assert np.isnan(result[3]).all()  # No replacement draw for positive-only sample.
    generated = patient_draws(3, 20, seed=20260917, batch_size=7)
    np.testing.assert_array_equal(generated.sum(axis=1), np.full(20, 3))
    # Patient multiplicity applies identically to its two ECGs, not record sampling.
    record_weights = generated[:, inverse]
    np.testing.assert_array_equal(record_weights[:, 0], record_weights[:, 1])
    np.testing.assert_array_equal(record_weights[:, 3], record_weights[:, 4])


def test_fixed_seed_mean_averages_auroc_not_probabilities():
    y = np.array([0, 0, 1, 1])[:, None]
    seed_probabilities = np.array(
        [
            [0.1, 0.8, 0.2, 0.9],
            [0.7, 0.1, 0.8, 0.2],
            [0.9, 0.8, 0.1, 0.2],
        ]
    )
    draws = np.array([[1, 1, 1, 1]])
    estimates = np.stack(
        [
            cluster_auc_distribution(y, p[:, None], np.arange(4), draws)
            for p in seed_probabilities
        ]
    )
    mean = mean_seed_auc(estimates)
    expected = np.mean([roc_auc_score(y[:, 0], p) for p in seed_probabilities])
    np.testing.assert_allclose(mean, expected)
    assert expected == 0.5
    assert mean[1, 0] != roc_auc_score(y[:, 0], seed_probabilities.mean(axis=0))


def test_undefined_class_invalidates_macro_without_discarding_other_class():
    y = np.array([[0, 0], [1, 0], [1, 1]])
    p = np.array([[0.1, 0.1], [0.8, 0.2], [0.9, 0.9]])
    result = cluster_auc_distribution(y, p, np.arange(3), np.array([[0, 2, 1]]))
    assert np.isfinite(result[0]).all()
    assert np.isnan(result[1, 0])
    assert np.isnan(result[1, 1])
    assert result[1, 2] == 1.0
    # A fixed-seed mean also cannot quietly omit an invalid seed contribution.
    assert np.isnan(mean_seed_auc(np.array([[0.8, np.nan], [0.9, 0.7], [0.7, 0.6]]))[1])
