"""Protect undefined predictive values and fixed-threshold/calibration boundaries."""

import numpy as np

from phase2.src.evaluate_phase2 import calibration_bins, extended_metrics


def test_absent_predicted_class_remains_undefined_in_strict_macro():
    y = np.tile(np.array([[0], [1]], dtype=np.float32), (1, 5))
    p = np.tile(np.array([[0.1], [0.9]], dtype=np.float32), (1, 5))
    p[:, 0] = 0.1  # No predicted positives in NORM, not a zero-valued PPV.
    p[:, 1] = 0.9  # No predicted negatives in MI, not a zero-valued NPV.
    metrics, _ = extended_metrics(y, p, np.full(5, 0.5))
    assert np.isnan(metrics["ppv_NORM"])
    assert metrics["ppv_denominator_NORM"] == 0
    assert np.isnan(metrics["npv_MI"])
    assert metrics["npv_denominator_MI"] == 0
    assert np.isnan(metrics["macro_ppv"])
    assert np.isnan(metrics["macro_npv"])
    assert metrics["macro_ppv_n_defined"] == 4
    assert metrics["macro_npv_n_undefined"] == 1
    assert metrics["sensitivity_NORM"] == 0
    assert metrics["specificity_MI"] == 0


def test_threshold_ties_are_positive_and_preserve_multilabel_counts():
    y = np.tile(np.array([[1], [0], [1], [0]], dtype=np.float32), (1, 5))
    p = np.tile(np.array([[0.5], [0.5], [0.2], [0.1]], dtype=np.float32), (1, 5))
    metrics, _ = extended_metrics(y, p, np.full(5, 0.5))
    for name in ("NORM", "MI", "STTC", "CD", "HYP"):
        for count in ("tp", "tn", "fp", "fn"):
            assert metrics[f"{count}_{name}"] == 1
        for ratio in ("sensitivity", "specificity", "ppv", "npv"):
            assert metrics[f"{ratio}_{name}"] == 0.5
    assert metrics["macro_ppv"] == 0.5
    assert metrics["macro_npv"] == 0.5


def test_calibration_includes_zero_one_and_empty_bins_without_dropping_records():
    p = np.tile(np.array([[0], [0.5], [1]], dtype=np.float32), (1, 5))
    y = np.tile(np.array([[0], [1], [1]], dtype=np.float32), (1, 5))
    counts, probability_sum, target_sum = calibration_bins(y, p)
    expected_counts = np.zeros((5, 15), dtype=np.int64)
    expected_counts[:, [0, 7, 14]] = 1
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_array_equal(probability_sum[:, 14], np.ones(5))
    np.testing.assert_array_equal(probability_sum[:, 7], np.full(5, 0.5))
    np.testing.assert_array_equal(target_sum.sum(axis=1), np.full(5, 2))
