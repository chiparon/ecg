import numpy as np
import pytest

from src.strict_control import (
    apply_circular_shifts,
    marginal_diagnostics,
    marginal_shift,
)


def test_shift_preserves_marginals_and_periodic_spectrum_with_wraparound():
    original = np.array(
        [[0, 2, -3, 2, 8, -1, 4], [5, -2, 4, 0, 1, -6, 3], [2, 0, 0, 4, -2, 7, 1]],
        dtype=np.float32,
    )
    offsets = np.array([0, 1, 6])
    shifted = apply_circular_shifts(original, offsets)
    assert shifted.dtype == original.dtype
    np.testing.assert_array_equal(shifted[0], original[0])
    np.testing.assert_array_equal(shifted[1], [3, 5, -2, 4, 0, 1, -6])
    np.testing.assert_array_equal(shifted[2], [0, 0, 4, -2, 7, 1, 2])
    np.testing.assert_array_equal(np.sort(shifted, axis=1), np.sort(original, axis=1))
    # Match the audit's float64 spectral accumulation of unchanged float32 noise.
    np.testing.assert_allclose(
        abs(np.fft.rfft(shifted.astype(np.float64))) ** 2,
        abs(np.fft.rfft(original.astype(np.float64))) ** 2,
        rtol=1e-12,
        atol=1e-12,
    )
    assert not np.shares_memory(original, shifted)


def test_distinct_shifts_change_cross_lead_alignment_without_changing_variances():
    waveform = np.random.default_rng(45).normal(size=4096)
    original = np.tile(waveform, (12, 1))
    shifted = apply_circular_shifts(original, np.arange(12) * 271)
    before = np.corrcoef(original)
    after = np.corrcoef(shifted)
    off_diagonal = ~np.eye(12, dtype=bool)
    assert np.mean(abs(before[off_diagonal])) > 0.99
    assert np.mean(abs(after[off_diagonal])) < 0.1
    np.testing.assert_allclose(
        np.var(original, axis=1), np.var(shifted, axis=1), rtol=1e-12
    )


def test_periodic_matching_does_not_claim_welch_matching():
    reference = np.zeros((12, 1000), dtype=np.float32)
    reference[:, :20] = np.arange(1, 13)[:, None]
    shifted = apply_circular_shifts(reference, np.full(12, 450))
    clean = np.random.default_rng(1).normal(size=reference.shape)
    measured, _, _, _ = marginal_diagnostics(reference, shifted, clean, 100)
    assert measured["empirical_marginals_exact"]
    assert measured["periodogram_l1_max_lead_relative_error"] < 1e-12
    assert measured["welch_l1_all_leads_relative_error"] > 0.1


def test_saved_offsets_reconstruct_deterministic_control_and_invalid_offsets_fail():
    original = np.random.default_rng(8).normal(size=(12, 127)).astype(np.float32)
    a, offsets = marginal_shift(original, 39)
    b, repeated = marginal_shift(original, 39)
    np.testing.assert_array_equal(offsets, repeated)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, apply_circular_shifts(original, offsets))
    with pytest.raises(ValueError, match="offsets"):
        apply_circular_shifts(original, np.full(12, 127))
