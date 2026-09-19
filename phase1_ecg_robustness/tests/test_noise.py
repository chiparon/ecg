"""Scientific-contract regression tests; run with unittest discovery or pytest."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from src.covariance_matching import (
    covariance_errors,
    covariance_factor,
    covariance_to_correlation,
    empirical_covariance,
    match_covariance,
)
from src.lead_matrix import (
    ELECTRODES,
    LEADS,
    get_lead_matrix,
    unit_electrode_effects,
    validate_lead_matrix,
)
from src.noise_generators import (
    make_noise_triplet,
    match_lead_rms,
    noise_diagnostics,
    nstdb_source_info,
    scale_to_snr,
)


def signal(length: int = 2000) -> np.ndarray:
    time = np.arange(length) / 100
    return (np.arange(1, 13)[:, None] / 12) * (
        np.sin(2 * np.pi * 1.3 * time) + 0.2 * np.sin(2 * np.pi * 8 * time)
    )


class LeadFormationTests(unittest.TestCase):
    def test_standard_identities_and_common_mode(self):
        a = get_lead_matrix()
        self.assertEqual(a.shape, (12, 9))
        self.assertEqual(np.linalg.matrix_rank(a), 8)
        np.testing.assert_allclose(a @ np.ones(9), 0, atol=1e-15)
        rng = np.random.default_rng(99)
        potentials = rng.normal(size=(9, 81))
        leads = a @ potentials
        np.testing.assert_allclose(leads[2], leads[1] - leads[0], atol=1e-15)
        np.testing.assert_allclose(leads[3], -(leads[0] + leads[1]) / 2, atol=1e-15)
        np.testing.assert_allclose(leads[4], leads[0] - leads[1] / 2, atol=1e-15)
        np.testing.assert_allclose(leads[5], leads[1] - leads[0] / 2, atol=1e-15)
        np.testing.assert_allclose(
            leads[6:], potentials[3:] - potentials[:3].mean(axis=0), atol=1e-15
        )
        np.testing.assert_array_equal(get_lead_matrix(True)[:, -1], 0)
        np.testing.assert_allclose(get_lead_matrix(True) @ np.ones(10), 0, atol=1e-15)
        self.assertEqual(validate_lead_matrix()["rank"], 8)

    def test_unit_ra_and_chest_propagation(self):
        a = get_lead_matrix()
        expected_ra = np.array([-1, -1, 0, 1, -0.5, -0.5] + [-1 / 3] * 6)
        np.testing.assert_array_equal(a[:, 0], expected_ra)
        expected_v1 = np.zeros(12)
        expected_v1[6] = 1
        np.testing.assert_array_equal(a[:, ELECTRODES.index("V1")], expected_v1)
        effects = unit_electrode_effects(True)
        self.assertEqual(set(effects["RA"]), set(LEADS) - {"III"})
        self.assertEqual(effects["V1"], {"V1": 1.0})
        self.assertEqual(effects["RL"], {})


class NoiseControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = signal()
        cls.noises = make_noise_triplet(cls.x, 100, 10, 781)

    def test_snr_zero_dc_and_measured_band_power(self):
        for condition, noise in self.noises.items():
            with self.subTest(condition=condition):
                diagnostics = noise_diagnostics(noise, self.x, 100)
                self.assertAlmostEqual(diagnostics["actual_snr_db"], 10, places=10)
                np.testing.assert_allclose(noise.mean(axis=1), 0, atol=1e-15)
                self.assertGreater(min(diagnostics["band_power_fraction"]), 0.94)

    def test_rms_control_preserves_physical_reference(self):
        physical = self.noises["electrode"]
        before = physical.copy()
        result = match_lead_rms(self.noises["independent"], physical)
        np.testing.assert_array_equal(physical, before)
        np.testing.assert_allclose(
            np.mean(result**2, axis=1), np.mean(physical**2, axis=1), rtol=1e-12
        )
        np.testing.assert_allclose(
            result, self.noises["independent_rms"], rtol=1e-12, atol=1e-15
        )
        original_corr, _ = covariance_to_correlation(
            empirical_covariance(self.noises["independent"])
        )
        matched_corr, _ = covariance_to_correlation(empirical_covariance(result))
        np.testing.assert_allclose(original_corr, matched_corr, atol=1e-14)
        self.assertGreater(np.max(np.abs(original_corr - np.eye(12))), 1e-4)

    def test_structures_and_exact_empirical_covariance(self):
        independent = noise_diagnostics(self.noises["independent"], self.x, 100)
        physical = noise_diagnostics(self.noises["electrode"], self.x, 100)
        matched = noise_diagnostics(self.noises["covariance"], self.x, 100)
        self.assertLess(independent["mean_abs_offdiagonal_correlation"], 0.1)
        self.assertGreater(
            physical["mean_abs_offdiagonal_correlation"],
            independent["mean_abs_offdiagonal_correlation"] + 0.1,
        )
        error = covariance_errors(physical["covariance"], matched["covariance"])
        self.assertLess(error["covariance_relative_frobenius"], 1e-11)
        self.assertLess(error["correlation_mae"], 1e-11)
        self.assertFalse(
            np.allclose(self.noises["electrode"], self.noises["covariance"])
        )
        _, factor_info = covariance_factor(physical["covariance"])
        self.assertEqual(factor_info["rank"], 8)
        self.assertEqual(factor_info["jitter"], 0)
        self.assertTrue(np.all(factor_info["clipped_eigenvalues"] >= 0))

    def test_seeds_and_snr_pairing(self):
        repeat = make_noise_triplet(self.x, 100, 10, 781)
        different = make_noise_triplet(self.x, 100, 10, 782)
        louder = make_noise_triplet(self.x, 100, 0, 781)
        for condition in self.noises:
            np.testing.assert_array_equal(repeat[condition], self.noises[condition])
            self.assertFalse(
                np.array_equal(different[condition], self.noises[condition])
            )
            # Eigenvectors in a repeated eigenspace have sign/order ambiguity,
            # so temporal SNR pairing is asserted for the non-factorized sources.
            if condition != "covariance":
                np.testing.assert_allclose(
                    louder[condition], np.sqrt(10) * self.noises[condition], atol=1e-14
                )

    def test_single_and_multi_electrode_untouched_leads(self):
        for active in (["RA"], ["V1"], ["RA", "LA", "LL"], ["LA", "V3", "V6"]):
            with self.subTest(active=active):
                noises = make_noise_triplet(self.x, 100, 20, 888, active=active)
                columns = [ELECTRODES.index(name) for name in active]
                untouched = ~np.any(get_lead_matrix()[:, columns] != 0, axis=1)
                for condition in ("electrode", "independent_rms", "covariance"):
                    np.testing.assert_array_equal(noises[condition][untouched], 0)
                    self.assertAlmostEqual(
                        noise_diagnostics(noises[condition], self.x, 100)[
                            "actual_snr_db"
                        ],
                        20,
                        places=10,
                    )
                np.testing.assert_allclose(
                    np.mean(noises["independent_rms"] ** 2, axis=1),
                    np.mean(noises["electrode"] ** 2, axis=1),
                    atol=1e-15,
                )
                if active == ["RA"]:
                    physical = noises["electrode"]
                    np.testing.assert_allclose(
                        physical, get_lead_matrix()[:, :1] * -physical[0], atol=1e-15
                    )

    def test_zero_power_and_invalid_inputs_are_explicit(self):
        for kwargs in (
            {"active": []},
            {"active": ["RL"]},
            {"active": ["RA", "RA"]},
            {"active": ["unknown"]},
            {"kind": "unknown"},
            {"kind": "bw"},
            {"band": (40, 60)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_noise_triplet(self.x, 100, 10, 4, **kwargs)
        with self.assertRaises(ValueError):
            make_noise_triplet(np.zeros_like(self.x), 100, 10, 4)
        with self.assertRaises(ValueError):
            scale_to_snr(np.zeros_like(self.x), self.x, 10)
        with self.assertRaises(ValueError):
            match_lead_rms(np.zeros_like(self.x), self.noises["electrode"])
        contaminated = self.x.copy()
        contaminated[0, 0] = np.nan
        with self.assertRaises(ValueError):
            make_noise_triplet(contaminated, 100, 10, 4)

    def test_zero_covariance_and_inactive_diagnostics(self):
        source = np.random.default_rng(45).normal(size=(12, 100))
        result, info = match_covariance(np.zeros((12, 12)), source)
        np.testing.assert_array_equal(result, 0)
        self.assertEqual(info["rank"], 0)
        self.assertEqual(info["covariance_relative_frobenius"], 0)
        self.assertIsNone(info["correlation_mae"])
        x = signal(100)
        x[0] = 0
        diagnostics = noise_diagnostics(result, x, 100)
        self.assertFalse(diagnostics["active_leads"].any())
        self.assertTrue(np.isnan(diagnostics["correlation"]).all())
        self.assertIsNone(diagnostics["per_lead_snr_db"][0])
        self.assertEqual(diagnostics["per_lead_snr_db"][1], float("inf"))
        self.assertIsNone(diagnostics["mean_abs_offdiagonal_correlation"])


class CovarianceBoundaryTests(unittest.TestCase):
    def test_roundoff_clipping_versus_genuinely_indefinite_target(self):
        factor, info = covariance_factor(np.diag([1, 0.5, -1e-15]))
        self.assertEqual(info["rank"], 2)
        np.testing.assert_allclose(factor @ factor.T, np.diag([1, 0.5, 0]), atol=1e-14)
        with self.assertRaises(ValueError):
            covariance_factor(np.diag([1, -0.01]))
        with self.assertRaises(ValueError):
            covariance_factor(np.array([[1, 0.8], [0.2, 1]]))

    def test_insufficient_temporal_rank_cannot_fake_a_match(self):
        with self.assertRaises(ValueError):
            match_covariance(np.eye(3), np.ones((3, 100)))
        with self.assertRaises(ValueError):
            match_covariance(np.eye(3), np.ones((3, 3)))

    def test_fresh_realizations_match_rank_deficient_target(self):
        rng = np.random.default_rng(456)
        directions = rng.normal(size=(12, 4))
        target = directions @ directions.T
        first, info = match_covariance(target, rng.normal(size=(12, 700)))
        second, _ = match_covariance(target, rng.normal(size=(12, 700)))
        self.assertLess(info["covariance_relative_frobenius"], 1e-12)
        self.assertTrue(info["finite_sample_conditioned"])
        self.assertFalse(np.allclose(first, second))
        np.testing.assert_allclose(empirical_covariance(second), target, atol=1e-12)


class NSTDBReplayTests(unittest.TestCase):
    def test_polyphase_replay_keeps_physical_frequency_and_explicit_provenance(self):
        # A sinusoidal source is deliberately rank one. Single V1 perturbation
        # isolates temporal resampling from lead mixing and global calibration.
        time = np.arange(360 * 25) / 360
        samples = (1000 * np.sin(2 * np.pi * 7 * time))[:, None]
        record = types.SimpleNamespace(p_signal=samples, fs=360, units=["uV"])
        fake_wfdb = types.SimpleNamespace(rdrecord=lambda path, physical: record)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules, {"wfdb": fake_wfdb}
        ):
            noises = make_noise_triplet(
                signal(1000), 100, 10, 94, kind="bw", active=["V1"], nstdb=directory
            )
            again = make_noise_triplet(
                signal(1000), 100, 10, 94, kind="bw", active=["V1"], nstdb=directory
            )
            provenance = nstdb_source_info(directory, "bw", 100)
        physical = noises["electrode"][6]
        frequency = np.fft.rfftfreq(physical.size, 1 / 100)
        self.assertAlmostEqual(
            frequency[np.argmax(np.abs(np.fft.rfft(physical)))], 7, places=1
        )
        np.testing.assert_array_equal(noises["electrode"], again["electrode"])
        self.assertEqual(
            (provenance["resample_up"], provenance["resample_down"]), (5, 18)
        )
        self.assertEqual(provenance["resampled_samples"], 2500)
        self.assertEqual(provenance["output_units"], "mV")
        np.testing.assert_array_equal(noises["electrode"][:6], 0)
        comparison = covariance_errors(
            empirical_covariance(noises["electrode"]),
            empirical_covariance(noises["covariance"]),
        )
        self.assertLess(comparison["covariance_relative_frobenius"], 1e-12)


if __name__ == "__main__":
    unittest.main()
