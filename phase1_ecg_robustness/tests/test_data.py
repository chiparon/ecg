"""Scientific data contracts: unknown likelihood, patient isolation, physical units."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.datasets import (
    CLASSES,
    LEADS,
    load_data,
    select_splits,
    validate_patient_folds,
)
from src.prepare_ptbxl import align_waveform, map_superclasses


class DataContracts(unittest.TestCase):
    def test_unknown_likelihood_is_included_but_sensitivity_excludes_it(self):
        statements = pd.DataFrame(
            {"diagnostic": [1, 1, 0], "diagnostic_class": ["CD", "MI", "NORM"]},
            index=["LAFB", "IMI", "SR"],
        )
        codes = "{'LAFB': 0, 'IMI': 80, 'SR': 100}"
        primary = map_superclasses(codes, statements)
        np.testing.assert_array_equal(primary, [0, 1, 0, 1, 0])
        np.testing.assert_array_equal(
            map_superclasses(codes, statements, 50), [0, 1, 0, 0, 0]
        )
        self.assertEqual(map_superclasses("{'SR': 100}", statements).sum(), 0)

    def test_rejects_patient_cross_fold_even_before_subsampling(self):
        metadata = pd.DataFrame(
            {"ecg_id": [1, 2, 3], "patient_id": [10, 10, 20], "strat_fold": [1, 10, 9]}
        )
        with self.assertRaises(ValueError):
            select_splits(metadata, {"train_limit": 1, "test_limit": 1})
        metadata.loc[1, "strat_fold"] = 2
        with self.assertRaises(ValueError):
            validate_patient_folds(metadata)

    def test_fixed_subsets_preserve_official_partitions(self):
        metadata = pd.DataFrame(
            {
                "ecg_id": np.arange(60),
                "patient_id": np.arange(60),
                "strat_fold": np.repeat([1, 9, 10], 20),
            }
        )
        config = {
            "train_limit": 7,
            "val_limit": 5,
            "test_limit": 6,
            "subset_seed": 2026,
        }
        first = select_splits(metadata, {**config, "seed": 17})
        second = select_splits(metadata, {**config, "seed": 43})
        for name, fold in [("train", 1), ("val", 9), ("test", 10)]:
            np.testing.assert_array_equal(first[name], second[name])
            self.assertTrue((metadata.iloc[first[name]].strat_fold == fold).all())
            self.assertEqual(len(first[name]), config[f"{name}_limit"])
        self.assertFalse(set(first["train"]) & set(first["test"]))

    def test_lead_alignment_preserves_mv_and_removes_only_dc(self):
        order = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6]
        time = np.sin(2 * np.pi * np.arange(1000) / 100)
        canonical = time[:, None] * np.arange(1, 13)[None, :] + 40
        fields = {
            "fs": 100,
            "sig_name": [LEADS[i].upper() for i in order],
            "units": ["uV"] * 12,
        }
        result, qc = align_waveform(canonical[:, order] * 1000, fields)
        np.testing.assert_allclose(result, (canonical - 40).T, atol=1e-6)
        np.testing.assert_allclose(result.mean(axis=1), 0, atol=1e-6)
        self.assertTrue(qc["finite"])
        with self.assertRaises(ValueError):
            align_waveform(canonical[:999], fields)
        invalid = canonical.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            align_waveform(invalid, fields)

    def test_mmap_loader_rejects_metadata_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "signals.npy", np.zeros((2, 12, 1000), np.float32))
            labels = np.zeros((2, len(CLASSES)), np.float32)
            labels[0, 0], labels[1, 1] = 1, 1
            np.save(root / "labels.npy", labels)
            meta = pd.DataFrame(
                {"ecg_id": [123, 456], "patient_id": [1, 2], "strat_fold": [1, 10]}
            )
            meta.to_csv(root / "metadata.csv", index=False)
            (root / "preparation.json").write_text(
                json.dumps({"status": "complete", "prepared_ecg_ids": [123, 456]})
            )
            x, y, loaded = load_data(root)
            try:
                self.assertIsInstance(x, np.memmap)
                self.assertFalse(x.flags.writeable)
                np.testing.assert_array_equal(y, labels)
                self.assertEqual(loaded.ecg_id.tolist(), [123, 456])
            finally:
                x._mmap.close()
                y._mmap.close()
            meta.iloc[::-1].to_csv(root / "metadata.csv", index=False)
            with self.assertRaises(ValueError):
                load_data(root)


if __name__ == "__main__":
    unittest.main()
