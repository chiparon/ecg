"""Aligned, memory-mapped PTB-XL arrays and official patient-level splits."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def validate_patient_folds(metadata):
    """Reject missing identifiers, invalid folds and any cross-fold patient reuse."""
    required = {"ecg_id", "patient_id", "strat_fold"}
    if not required.issubset(metadata.columns):
        raise ValueError(
            f"Missing metadata columns: {required - set(metadata.columns)}"
        )
    if metadata[list(required)].isna().any().any():
        raise ValueError("Missing ECG/patient identifiers or official fold")
    if metadata.ecg_id.duplicated().any():
        raise ValueError("Duplicate ecg_id; waveform/label alignment is ambiguous")
    folds = pd.to_numeric(metadata.strat_fold, errors="raise")
    if not folds.isin(range(1, 11)).all():
        raise ValueError("strat_fold must be an integer from 1 through 10")
    leakage = metadata.groupby("patient_id").strat_fold.nunique()
    offenders = leakage[leakage > 1].index.tolist()
    if offenders:
        raise ValueError(f"Patient leakage across official folds: {offenders[:20]}")
    return {
        "passed": True,
        "patients": int(metadata.patient_id.nunique()),
        "cross_fold_patients": 0,
        "rule": "each patient occupies exactly one official fold",
    }


def select_splits(metadata, train_config):
    """Fixed within-fold subsets, independent of model/training random seeds.

    Limits select records, not a new split. Sorting selected row indices preserves
    metadata alignment. Patients are never moved between official partitions.
    """
    validate_patient_folds(metadata)
    folds = metadata.strat_fold.to_numpy(dtype=int)
    masks = {"train": folds <= 8, "val": folds == 9, "test": folds == 10}
    seed = int(train_config.get("subset_seed", 2026))
    result = {}
    for offset, (name, mask) in enumerate(masks.items()):
        indices = np.flatnonzero(mask)
        limit = train_config.get(f"{name}_limit")
        if limit is not None:
            if isinstance(limit, bool) or int(limit) != limit or limit <= 0:
                raise ValueError(f"{name}_limit must be a positive integer or null")
            if len(indices) > limit:
                rng = np.random.default_rng(np.random.SeedSequence([seed, offset]))
                indices = np.sort(rng.choice(indices, int(limit), replace=False))
        result[name] = indices
    return result


def load_data(path):
    """Return (signals mmap, labels mmap, metadata); never materialize all signals."""
    root = Path(path)
    metadata = pd.read_csv(root / "metadata.csv")
    provenance = root / "preparation.json"
    if provenance.exists():
        prepared = json.loads(provenance.read_text(encoding="utf-8"))
        if prepared.get("status") != "complete":
            raise ValueError(
                "Preparation did not finish; refusing incomplete/stale arrays"
            )
        if prepared.get("prepared_ecg_ids") != metadata.ecg_id.astype(int).tolist():
            raise ValueError("Metadata row order does not match preparation ECG IDs")
    x = np.load(root / "signals.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False)
    if x.shape != (len(metadata), 12, 1000) or x.dtype != np.float32:
        raise ValueError(f"Expected float32 (N,12,1000), got {x.dtype} {x.shape}")
    if y.shape != (len(metadata), 5) or y.dtype != np.float32:
        raise ValueError(f"Expected float32 labels (N,5), got {y.dtype} {y.shape}")
    if (
        not np.isfinite(y).all()
        or not np.isin(y, [0, 1]).all()
        or np.any(y.sum(axis=1) == 0)
    ):
        raise ValueError("Labels must be finite, binary and nonempty")
    validate_patient_folds(metadata)
    return x, y, metadata


def _data_identity(data_dir: Path, metadata, labels) -> dict:
    # Full small metadata/labels hash, plus large-array identity and preparation provenance.
    digest = hashlib.sha256(metadata.to_csv(index=False).encode())
    digest.update(np.asarray(labels, dtype=np.float32).tobytes())
    preparation = data_dir / "preparation.json"
    if preparation.exists():
        digest.update(preparation.read_bytes())
    signals = data_dir / "signals.npy"
    stat = signals.stat()
    return {
        "metadata_labels_provenance_sha256": digest.hexdigest(),
        "signals_bytes": stat.st_size,
        "signals_mtime_ns": stat.st_mtime_ns,
        "data_dir": str(data_dir.resolve()),
    }
