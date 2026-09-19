"""Read-only timing of four preselected operations; no model imports."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "overnight_supplements_v2"


def info(path):
    path = ROOT / path
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def save(name, value):
    value.update(training_invocations=0, inference_invocations=0)
    (OUT / "freeze" / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    started = time.perf_counter()
    prediction = "phase2/results/predictions/full/electrode/resnet/seed_17/bandpass_n20001_cLA_electrode_s5.npz"
    cohort_path = "phase2/results/tables/full/patient_bootstrap/cohort.npz"
    draws_path = "phase2/results/tables/full/patient_bootstrap/patient_draws.npy"
    frozen = json.loads((ROOT / "signed_control_mechanism/logs/full/freeze.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "signed_control_mechanism/inputs/full/manifest.json").read_text(encoding="utf-8"))
    case = sorted(manifest["cases"], key=lambda c: (c["condition"], c["snr"], c["noise_seed"]))[0]
    protocol = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "prediction": info(prediction), "cohort": info(cohort_path), "draws": info(draws_path),
        "clean": info(frozen["clean"]["path"]), "noise": info(case["noise_path"]),
        "matrix": info("signed_control_mechanism/inputs/matrices.json"),
        "prediction_unit": {"architecture": "resnet", "snr_db": 5, "train_seed": 17, "test_structure": "electrode", "combo_id": "LA", "noise_seed": 20001},
        "draw_id": 0, "geometry_case": case,
        "timing_repetitions": 1, "cache_state": "OS cache uncontrolled; no cold-cache claim",
        "blas_threads": 1, "selection_rule": "First typed lexicographic P0 unit, electrode training; first signed-control case in condition/SNR/noise order; every record in that case",
    }
    save("benchmark_selection.json", protocol)
    t = time.perf_counter()
    with np.load(ROOT / prediction, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    read_seconds = time.perf_counter() - t
    t = time.perf_counter()
    point = roc_auc_score(data["y"], data["p"], average="macro")
    point_seconds = time.perf_counter() - t
    with np.load(ROOT / cohort_path, allow_pickle=False) as saved:
        reference = {key: saved[key] for key in saved.files}
    draws = np.load(ROOT / draws_path, mmap_mode="r", allow_pickle=False)
    assert draws.shape == (2000, len(reference["unique_patients"])) if "unique_patients" in reference else draws.shape[0] == 2000
    weights = draws[0, reference["patient_inverse"]]
    t = time.perf_counter()
    draw_auc = roc_auc_score(data["y"], data["p"], average="macro", sample_weight=weights)
    draw_seconds = time.perf_counter() - t
    matrix = json.loads((ROOT / "signed_control_mechanism/inputs/matrices.json").read_text(encoding="utf-8"))
    A = np.asarray(matrix["standard"]["matrix"], dtype=np.float64)
    P = A @ np.linalg.pinv(A, rcond=1e-12)
    t = time.perf_counter()
    clean = np.load(ROOT / frozen["clean"]["path"], mmap_mode="r", allow_pickle=False)
    base = np.load(ROOT / case["noise_path"], mmap_mode="r", allow_pickle=False)
    final_hasher = hashlib.sha256()
    count = 0
    projected_clean_sum = projected_noise_sum = 0.0
    for start in range(0, len(clean), 64):
        x32 = np.asarray(clean[start:start + 64])
        n32 = np.asarray(base[start:start + 64]) * np.float32(10 ** (-float(case["snr"]) / 20))
        z32 = x32 + n32
        final_hasher.update(memoryview(np.ascontiguousarray(z32)).cast("B"))
        x = x32.astype(np.float64)
        n = z32.astype(np.float64) - x
        px, pn = P @ x, P @ n
        projected_clean_sum += float(np.sum(px * px))
        projected_noise_sum += float(np.sum(pn * pn))
        count += len(x)
    projection_seconds = time.perf_counter() - t
    assert final_hasher.hexdigest() == case["input_sha256"]
    result = {
        "status": "completed", "frozen_selection": info("overnight_supplements_v2/freeze/benchmark_selection.json"),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "file_read_seconds": read_seconds, "macro_auroc_seconds": point_seconds,
        "one_patient_draw_seconds": draw_seconds, "projection_scan_seconds": projection_seconds,
        "projection_records": count, "projection_records_per_second": count / projection_seconds,
        "point_macro_auroc": float(point), "draw_macro_auroc": float(draw_auc),
        "verified_final_input_sha256": final_hasher.hexdigest(),
        "projected_clean_energy_sum": projected_clean_sum, "projected_noise_energy_sum": projected_noise_sum,
        "prediction_schema": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in data.items()},
        "cohort_schema": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in reference.items()},
        "draws_schema": {"shape": list(draws.shape), "dtype": str(draws.dtype), "row_sum_min": int(draws.sum(axis=1).min()), "row_sum_max": int(draws.sum(axis=1).max())},
        "pyarrow_available": importlib.util.find_spec("pyarrow") is not None,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {"logical_cpus": os.cpu_count(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
    }
    save("throughput_benchmark.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
