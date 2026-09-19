"""Unequal records per patient and missing-class draws are scientific boundaries."""
import unittest
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from methodology_supplement.bootstrap import metric_distribution


class PatientMetricBoundaries(unittest.TestCase):
    def setUp(self):
        self.y = np.array([[1,0,1,0,1], [0,1,0,1,0], [1,1,0,0,1], [0,0,1,1,0],
                           [1,0,0,1,1], [0,1,1,0,0], [1,1,1,0,0], [0,0,0,1,1]], dtype=np.float32)
        self.p = np.array([[1,0,.5,.2,.8], [.2,1,.5,.8,0], [.5,.8,.1,0,.8], [0,.2,1,1,.2],
                           [.8,0,.2,.8,.5], [.2,.8,.8,0,.1], [.5,1,.8,.2,0], [0,.2,0,1,1]], dtype=np.float32)
        self.thresholds = np.array([.4,.6,.5,.3,.7])
        self.inverse = np.array([0,0,1,2,3,3,3,4])
        self.draws = np.array([[2,0,1,1,1], [0,3,0,0,2], [0,0,5,0,0]], dtype=np.int32)

    def expanded_reference(self, counts):
        selection = np.repeat(np.arange(len(self.y)), counts[self.inverse])
        y, p = self.y[selection], self.p[selection]
        auc = np.nan if np.any(y.sum(axis=0) == 0) or np.any(y.sum(axis=0) == len(y)) else roc_auc_score(y, p, average="macro")
        f1 = f1_score(y, p >= self.thresholds, average="macro", zero_division=0)
        ece = 0.0
        for column in range(5):
            bins = np.minimum((p[:, column] * 15).astype(int), 14)
            for index in range(15):
                selected = bins == index
                if selected.any():
                    ece += selected.mean() * abs(p[selected, column].mean() - y[selected, column].mean()) / 5
        return np.array([auc, f1, ece])

    def test_patient_draw_matches_repeating_every_selected_record(self):
        observed = metric_distribution(self.y, self.p, self.thresholds, self.inverse, self.draws, batch_size=2)
        expected = np.stack([self.expanded_reference(counts) for counts in np.vstack([np.ones(5, dtype=int), self.draws])])
        np.testing.assert_allclose(observed, expected, atol=2e-7, rtol=0, equal_nan=True)
        self.assertEqual(observed.shape, (4, 3))

    def test_missing_class_draw_is_retained_without_losing_f1_or_ece(self):
        observed = metric_distribution(self.y, self.p, self.thresholds, self.inverse, self.draws[-1:], batch_size=1)
        self.assertTrue(np.isnan(observed[1, 0]))
        np.testing.assert_allclose(observed[1, 1:], self.expanded_reference(self.draws[-1])[1:], atol=2e-7, rtol=0)
        self.assertTrue(np.isfinite(observed[1, 1:]).all())


if __name__ == "__main__":
    unittest.main()
