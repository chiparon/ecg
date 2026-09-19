"""Patient-cluster metric draws; preserve fixed thresholds and all record repeats."""
from __future__ import annotations

import numpy as np
from phase1_ecg_robustness.src.supplemental_statistics import cluster_auc_distribution

METRICS = ("macro_auroc", "macro_f1", "ece")


def metric_distribution(y, p, thresholds, patient_inverse, draws, batch_size=64, compute_auc=True):
    y, p = np.asarray(y), np.asarray(p)
    thresholds = np.asarray(thresholds)
    inverse, draws = np.asarray(patient_inverse), np.asarray(draws)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 5 or thresholds.shape != (5,):
        raise ValueError("Expected aligned five-class predictions and frozen thresholds")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid labels or probabilities")
    if not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("Invalid frozen thresholds")
    if inverse.shape != (len(y),) or not np.issubdtype(inverse.dtype, np.integer) or np.any(inverse < 0):
        raise ValueError("Patient mapping does not align with records")
    if draws.ndim != 2 or not np.issubdtype(draws.dtype, np.integer) or np.any(draws < 0) or batch_size < 1:
        raise ValueError("Expected nonnegative patient multiplicities")
    n_patients = draws.shape[1]
    if not len(inverse) or inverse.max() >= n_patients or not np.all(draws.sum(axis=1) == n_patients):
        raise ValueError("Each draw must resample the full patient population")
    result = np.full((len(draws) + 1, len(METRICS)), np.nan, dtype=np.float64)
    if compute_auc:
        result[:, 0] = cluster_auc_distribution(y, p, inverse, draws, batch_size)[:, 0]

    # F1: five TP/FP/FN totals. ECE: signed residual total per class/bin.
    # ECE = sum(abs(probability_sum - target_sum))/(5 * weighted record count).
    # This avoids a record-sized weighted copy for every bootstrap draw.
    sufficient = np.zeros((n_patients, 1 + 15 + 75), dtype=np.float64)
    sufficient[:, 0] = np.bincount(inverse, minlength=n_patients)
    truth, positive = y.astype(bool), p >= thresholds
    for j in range(5):
        for offset, mask in enumerate((truth[:, j] & positive[:, j], ~truth[:, j] & positive[:, j], truth[:, j] & ~positive[:, j])):
            sufficient[:, 1 + j * 3 + offset] = np.bincount(inverse, weights=mask.astype(np.float64), minlength=n_patients)
        # Keep float32 bin arithmetic when p is float32, exactly as the baseline.
        bins = np.minimum((p[:, j] * 15).astype(np.int64), 14)
        flattened = inverse * 15 + bins
        residual = p[:, j].astype(np.float64) - y[:, j]
        sufficient[:, 16 + j * 15:16 + (j + 1) * 15] = np.bincount(flattened, weights=residual, minlength=n_patients * 15).reshape(n_patients, 15)

    def reduce_totals(totals):
        confusion = totals[:, 1:16].reshape(-1, 5, 3)
        tp, fp, fn = confusion[:, :, 0], confusion[:, :, 1], confusion[:, :, 2]
        denominator = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0).mean(axis=1)
        ece = np.abs(totals[:, 16:]).sum(axis=1) / (5 * totals[:, 0])
        return np.column_stack((f1, ece))

    result[0, 1:] = reduce_totals(sufficient.sum(axis=0, keepdims=True))[0]
    for start in range(0, len(draws), batch_size):
        stop = min(start + batch_size, len(draws))
        totals = draws[start:stop].astype(np.float64) @ sufficient
        result[start + 1:stop + 1, 1:] = reduce_totals(totals)
    return result
