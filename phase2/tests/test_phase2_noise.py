"""Behavioral guards for paired augmentation and the immutable input representation."""

from pathlib import Path

import numpy as np
import pytest

from phase1_ecg_robustness.src.noise_generators import make_noise_triplet
from phase2.src.common import load_config
from phase2.src.generate_phase2_noise import (
    _base_noises,
    _seed_integer,
    _sources,
    array_sha256,
    load_case_inputs,
    seed_words,
    training_batch,
)


@pytest.fixture
def cfg():
    return load_config(
        Path(__file__).resolve().parents[1] / "configs" / "phase2_main.yaml"
    )


def signals(count):
    x = np.random.default_rng(713).normal(0, 0.2, (count, 12, 1000)).astype(np.float32)
    x -= x.mean(axis=2, keepdims=True)
    return x


def test_training_realization_is_invariant_to_batch_order_and_unrelated_rng(cfg):
    x = signals(24)
    original = x.copy()
    ids = np.arange(600, 624)
    first, _ = training_batch(cfg, "pilot", "mixed", 17, 3, x, ids)
    np.random.default_rng(112).normal(size=10000)
    reversed_inputs, _ = training_batch(
        cfg, "pilot", "mixed", 17, 3, x[::-1], ids[::-1]
    )
    np.testing.assert_array_equal(first, reversed_inputs[::-1])
    np.testing.assert_array_equal(x, original)


def test_chest_only_controls_preserve_inactive_leads_and_target_energy(cfg):
    x = signals(1)[0]
    words = seed_words(cfg, "full", "test", 20001, 612)
    sources = _sources(
        cfg, words, "bandpass", 1000, need_independent=True, need_covariance=True
    )
    noises = _base_noises(
        x, sources, ["V2"], ("electrode", "independent_rms", "covariance")
    )
    target = np.mean(np.square(noises["electrode"], dtype=np.float64), axis=1)
    for noise in noises.values():
        np.testing.assert_array_equal(np.delete(noise, 7, axis=0), 0.0)
        np.testing.assert_allclose(
            np.mean(np.square(noise, dtype=np.float64), axis=1),
            target,
            rtol=1e-6,
            atol=0,
        )
        actual = 10 * np.log10(
            np.mean(np.square(x, dtype=np.float64))
            / np.mean(np.square(noise, dtype=np.float64))
        )
        assert abs(actual) < 1e-4


def test_optimized_source_reuse_preserves_phase_one_waveform_protocol(cfg):
    x = signals(1)[0]
    words = seed_words(cfg, "full", "test", 20003, 800)
    sources = _sources(
        cfg, words, "bandpass", 1000, need_independent=True, need_covariance=True
    )
    combo = ["RA", "LA", "V1", "V6"]
    actual = _base_noises(
        x, sources, combo, ("electrode", "independent_rms", "covariance")
    )
    expected = make_noise_triplet(x, 100, 0, _seed_integer(words), active=combo)
    for condition, noise in actual.items():
        np.testing.assert_array_equal(noise, expected[condition].astype(np.float32))


def test_factorized_cache_preserves_exact_float32_input_and_rejects_changes(
    cfg, tmp_path
):
    x = signals(3)
    noise = np.random.default_rng(54).normal(0, 0.2, x.shape).astype(np.float32)
    path = tmp_path / "noise.npy"
    np.save(path, noise)
    scale = 0.2254905872
    expected = x + noise * np.float32(10 ** (-15 / 20))
    expected /= scale
    data = {"x": x, "splits": {"test": np.arange(3)}, "scale_mv": scale}
    case = {
        "kind": "bandpass",
        "snr": 15,
        "base_noise_path": str(path),
        "case_id": "fixed_case",
        "input_sha256": array_sha256(expected),
    }
    np.testing.assert_array_equal(load_case_inputs(cfg, data, case), expected)
    noise[0, 0, 0] += 0.125
    np.save(path, noise)
    with pytest.raises(ValueError, match="differs from frozen input"):
        load_case_inputs(cfg, data, case)
