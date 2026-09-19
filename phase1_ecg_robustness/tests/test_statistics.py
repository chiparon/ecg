"""Never infer a smaller multiple-comparison family from interrupted evaluation."""

import pandas as pd
import pytest
from src.statistics import validate_metrics


def test_missing_whole_snr_group_is_rejected_even_when_seeds_are_complete():
    config = {
        "models": ["resnet"],
        "seeds": [17, 29, 43],
        "noise": {"kinds": ["bandpass"], "snrs": [20, 10, 0], "electrodes": []},
    }
    rows = []
    for seed in config["seeds"]:
        for kind, snr, condition in [("clean", 100, "clean")] + [
            ("bandpass", 20, c)
            for c in ("independent", "independent_rms", "electrode", "covariance")
        ]:
            rows.append(
                dict(
                    model="resnet",
                    seed=seed,
                    kind=kind,
                    snr=snr,
                    condition=condition,
                    active="all",
                    prediction_path="unused.npz",
                    macro_auroc=0.7,
                    macro_ap=0.6,
                    macro_f1=0.5,
                )
            )
    with pytest.raises(ValueError, match="configured evaluation grid"):
        validate_metrics(pd.DataFrame(rows), config)
