"""Rank-aware empirical covariance controls without diagonal jitter.

Exact empirical matching conditions fresh temporal draws on their sample
covariance (ddof=0). Consequently the result is not an unconditioned IID Gaussian
sample, and its finite-sample PSD need not equal the mapped realization's PSD.
Only this covariance control is whitened; independent conditions never are.
"""

from __future__ import annotations

import numpy as np


def empirical_covariance(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] < 2:
        raise ValueError(
            "Covariance requires nonempty channels and at least two time samples"
        )
    if not np.isfinite(samples).all():
        raise ValueError("Covariance samples must be finite")
    centered = samples - samples.mean(axis=1, keepdims=True)
    return centered @ centered.T / samples.shape[1]


def covariance_to_correlation(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return correlation and active mask; zero-variance correlations are NaN.

    NaNs explicitly mean undefined, not zero correlation. Consumers must use the
    returned active-channel mask when computing summaries or plotting subsets.
    """
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance must be square")
    if not np.isfinite(covariance).all() or np.any(np.diag(covariance) < 0):
        raise ValueError("Covariance must be finite with nonnegative diagonal")
    variance = np.diag(covariance)
    active = variance > 0
    correlation = np.full_like(covariance, np.nan)
    ix = np.ix_(active, active)
    standard_deviation = np.sqrt(variance[active])
    correlation[ix] = covariance[ix] / np.outer(standard_deviation, standard_deviation)
    return correlation, active


def covariance_factor(covariance: np.ndarray) -> tuple[np.ndarray, dict]:
    """Factor a PSD matrix; discard only roundoff-scale eigenvalues.

    Tolerance is 128 * dimension * machine epsilon * spectral radius. Significant
    negative eigenvalues are errors, not repaired targets. Rank loss is expected for
    ideal ECG formation. No diagonal jitter is applied (reported jitter is zero).
    """
    covariance = np.asarray(covariance, dtype=np.float64)
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not covariance.size
    ):
        raise ValueError("Target covariance must be a nonempty square matrix")
    if not np.isfinite(covariance).all():
        raise ValueError("Target covariance must be finite")
    scale = float(np.max(np.abs(covariance)))
    symmetry_tolerance = 128 * len(covariance) * np.finfo(np.float64).eps * scale
    symmetry_residual = float(np.max(np.abs(covariance - covariance.T)))
    if symmetry_residual > symmetry_tolerance:
        raise ValueError("Target covariance is not symmetric within roundoff tolerance")
    symmetric = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance = (
        128
        * len(covariance)
        * np.finfo(np.float64).eps
        * float(np.max(np.abs(eigenvalues)))
    )
    if np.any(eigenvalues < -tolerance):
        raise ValueError(
            f"Target covariance is not PSD: minimum eigenvalue {eigenvalues[0]:.6g}"
        )
    keep = eigenvalues > tolerance
    clipped = np.where(keep, eigenvalues, 0.0)
    factor = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
    # Preserve mathematically inactive channels exactly, including in rank-one cases.
    inactive = np.diag(symmetric) == 0
    if np.any(symmetric[inactive] != 0):
        raise ValueError("A zero-variance covariance row must be exactly zero")
    factor[inactive] = 0.0
    diagnostics = {
        "rank": int(keep.sum()),
        "jitter": 0.0,
        "eigenvalues": eigenvalues,
        "clipped_eigenvalues": clipped,
        "eigenvalue_tolerance": tolerance,
        "clipped_count": int((~keep).sum()),
        "symmetry_residual": symmetry_residual,
        "factor_residual_frobenius": float(
            np.linalg.norm(factor @ factor.T - symmetric)
        ),
    }
    return factor, diagnostics


def covariance_errors(target: np.ndarray, actual: np.ndarray) -> dict:
    """Compare ddof=0 covariances; correlation MAE excludes undefined pairs."""
    target = np.asarray(target, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if target.shape != actual.shape:
        raise ValueError("Covariance shapes differ")
    norm = float(np.linalg.norm(target, ord="fro"))
    difference = float(np.linalg.norm(actual - target, ord="fro"))
    target_corr, target_active = covariance_to_correlation(target)
    actual_corr, actual_active = covariance_to_correlation(actual)
    active = target_active & actual_active
    mask = np.outer(active, active)
    return {
        "covariance_relative_frobenius": (
            difference / norm if norm else (0.0 if difference == 0 else float("inf"))
        ),
        "correlation_mae": (
            float(np.mean(np.abs(target_corr[mask] - actual_corr[mask])))
            if mask.any()
            else None
        ),
        "correlation_pair_count": int(mask.sum()),
        "active_mask_equal": bool(np.array_equal(target_active, actual_active)),
    }


def match_covariance(
    target: np.ndarray, fresh_source: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Color fresh independent lead-space draws to exact empirical covariance.

    fresh_source has (channels, time) shape and MUST be a new realization, not the
    mapped electrode noise. The leading rank(target) temporal draws are empirically
    whitened, then colored by the target's eigenfactor. This is finite-sample
    conditioning, not a claim of identical realizations or pointwise matching PSDs.
    """
    factor, diagnostics = covariance_factor(target)
    fresh_source = np.asarray(fresh_source, dtype=np.float64)
    if (
        fresh_source.ndim != 2
        or fresh_source.shape[1] < 2
        or not np.isfinite(fresh_source).all()
    ):
        raise ValueError("Fresh sources must be finite channels-by-time samples")
    rank = diagnostics["rank"]
    if fresh_source.shape[0] < rank or fresh_source.shape[1] <= rank:
        raise ValueError(
            "Insufficient independent temporal degrees of freedom for covariance rank"
        )
    if rank == 0:
        result = np.zeros((factor.shape[0], fresh_source.shape[1]), dtype=np.float64)
        diagnostics.update({"source_rank": 0, "source_condition_number": None})
    else:
        source = fresh_source[:rank] - fresh_source[:rank].mean(axis=1, keepdims=True)
        source_cov = empirical_covariance(source)
        values, vectors = np.linalg.eigh(source_cov)
        tolerance = 128 * rank * np.finfo(np.float64).eps * float(values[-1])
        source_rank = int(np.count_nonzero(values > tolerance))
        if source_rank != rank:
            raise ValueError(
                "Fresh temporal source is rank deficient; enlarge duration or spectral support"
            )
        whitened = (vectors / np.sqrt(values)).T @ source
        result = factor @ whitened
        result -= result.mean(axis=1, keepdims=True)
        diagnostics.update(
            {
                "source_rank": source_rank,
                "source_condition_number": float(values[-1] / values[0]),
            }
        )
    diagnostics.update(covariance_errors(target, empirical_covariance(result)))
    diagnostics["finite_sample_conditioned"] = True
    return result, diagnostics
