"""Ideal diagnostic lead formation, not a model of acquisition electronics.

The nine diagnostic nodes exclude RL. Optional RL is a zero column: driven-right-
leg feedback and finite common-mode rejection are deliberately not simulated.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
ELECTRODES = ("RA", "LA", "LL", "V1", "V2", "V3", "V4", "V5", "V6")

# Chest leads are V_i - WCT, so every limb-node coefficient is NEGATIVE.
_A = np.array(
    [
        [-1, 1, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0, 0, 0],
        [1, -0.5, -0.5, 0, 0, 0, 0, 0, 0],
        [-0.5, 1, -0.5, 0, 0, 0, 0, 0, 0],
        [-0.5, -0.5, 1, 0, 0, 0, 0, 0, 0],
        [-1 / 3, -1 / 3, -1 / 3, 1, 0, 0, 0, 0, 0],
        [-1 / 3, -1 / 3, -1 / 3, 0, 1, 0, 0, 0, 0],
        [-1 / 3, -1 / 3, -1 / 3, 0, 0, 1, 0, 0, 0],
        [-1 / 3, -1 / 3, -1 / 3, 0, 0, 0, 1, 0, 0],
        [-1 / 3, -1 / 3, -1 / 3, 0, 0, 0, 0, 1, 0],
        [-1 / 3, -1 / 3, -1 / 3, 0, 0, 0, 0, 0, 1],
    ],
    dtype=np.float64,
)
_A.setflags(write=False)


def get_lead_matrix(include_rl: bool = False) -> np.ndarray:
    """Return an independent array in LEADS/ELECTRODES order; RL is appended."""
    if include_rl:
        return np.column_stack((_A, np.zeros(len(LEADS))))
    return _A.copy()


def matrix_provenance() -> dict:
    """Fingerprint coefficients and coordinate order with canonical JSON."""
    payload = {
        "schema": "ideal_wct_12x9_v1",
        "lead_order": list(LEADS),
        "electrode_order": list(ELECTRODES),
        "values": get_lead_matrix().tolist(),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        **payload,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "hash_encoding": "UTF-8 JSON of schema/orders/values; sorted keys; compact separators",
    }


def lead_contributions(include_rl: bool = False) -> dict[str, dict[str, float]]:
    """Named, nonzero coefficients in each lead's electrode decomposition."""
    names = ELECTRODES + (("RL",) if include_rl else ())
    return {
        lead: {name: float(value) for name, value in zip(names, row) if value != 0}
        for lead, row in zip(LEADS, get_lead_matrix(include_rl))
    }


def unit_electrode_effects(include_rl: bool = False) -> dict[str, dict[str, float]]:
    """Affected leads and signed mV response to a unit-mV electrode change."""
    names = ELECTRODES + (("RL",) if include_rl else ())
    return {
        name: {lead: float(value) for lead, value in zip(LEADS, column) if value != 0}
        for name, column in zip(names, get_lead_matrix(include_rl).T)
    }


def validate_lead_matrix() -> dict:
    """Numerical residuals; raise if the ideal lead identities are violated."""
    a = get_lead_matrix()
    residuals = {
        "einthoven": float(np.max(np.abs(a[2] - a[1] + a[0]))),
        "avr": float(np.max(np.abs(a[3] + (a[0] + a[1]) / 2))),
        "avl": float(np.max(np.abs(a[4] - a[0] + a[1] / 2))),
        "avf": float(np.max(np.abs(a[5] - a[1] + a[0] / 2))),
        "augmented_sum": float(np.max(np.abs(a[3:6].sum(axis=0)))),
        "wct": float(np.max(np.abs(a[6:, :3] + 1 / 3))),
        "chest_identity": float(np.max(np.abs(a[6:, 3:] - np.eye(6)))),
        "common_mode": float(np.max(np.abs(a @ np.ones(9)))),
        "rl": float(np.max(np.abs(get_lead_matrix(True)[:, -1]))),
    }
    rank = int(np.linalg.matrix_rank(a))
    if a.shape != (12, 9) or rank != 8 or max(residuals.values()) > 1e-12:
        raise AssertionError(
            f"Invalid diagnostic lead matrix: rank={rank}, {residuals}"
        )
    return {
        "shape": list(a.shape),
        "rank": rank,
        "residuals": residuals,
        "lead_order": list(LEADS),
        "electrode_order": list(ELECTRODES),
        "contributions": lead_contributions(),
        "unit_electrode_effects": unit_electrode_effects(True),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(validate_lead_matrix(), indent=2))
