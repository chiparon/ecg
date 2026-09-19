"""Export immutable benchmark inputs without changing either scientific experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from phase1_ecg_robustness.src.datasets import load_data, select_splits
from phase1_ecg_robustness.src.noise_generators import make_noise_triplet
from phase1_ecg_robustness.src.audit_noise import record_seed
from phase1_ecg_robustness.src.lead_matrix import matrix_provenance


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = Path(__file__).resolve().parents[1] / "phase1_ecg_robustness"
    manifest = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sampling_rate": 100,
                "snr_db": 10, "noise_seed": 20001, "active": "all", "smoke_only": args.smoke,
                "matrix": matrix_provenance(), "files": {}, "sources": {}, "counts": {}}
    full = None
    for stage, data_name in (("pilot", "processed_pilot"), ("full", "processed")):
        if args.smoke and stage == "pilot":
            continue
        data_dir = baseline / "data" / data_name
        x, y, meta = load_data(data_dir)
        settings = {"subset_seed": 2026}
        if stage == "pilot":
            settings.update(train_limit=4000, val_limit=1000, test_limit=1000)
        splits = select_splits(meta, settings)
        manifest["counts"][stage] = {key: len(value) for key, value in splits.items()}
        for file in ("preparation.json", "metadata.csv", "labels.npy", "signals.npy"):
            manifest["sources"][str((data_dir/file).relative_to(baseline.parent))] = sha(data_dir/file)
        if not args.smoke:
            for split in ("train", "val"):
                indices = splits[split]
                np.savez(args.output / f"{split}_{stage}.npz", x=x[indices], y=y[indices],
                         ecg_id=meta.iloc[indices].ecg_id.to_numpy(dtype=np.int64),
                         patient_id=meta.iloc[indices].patient_id.to_numpy(dtype=np.int64))
        if stage == "full":
            full = (x, y, meta, splits["test"])
    x, y, meta, indices = full
    if args.smoke:
        indices = indices[:100]
    clean = np.array(x[indices], dtype=np.float32)
    ids = meta.iloc[indices].ecg_id.to_numpy(dtype=np.int64)
    arrays = {"clean": clean, "y": np.array(y[indices]), "ecg_id": ids,
              "patient_id": meta.iloc[indices].patient_id.to_numpy(dtype=np.int64)}
    for condition in ("independent_rms", "electrode", "covariance"):
        arrays[condition] = np.empty_like(clean)
    for i, ecg_id in enumerate(ids):
        noises = make_noise_triplet(clean[i], 100, 10, record_seed(20001, ecg_id, "bandpass"))
        for condition in ("independent_rms", "electrode", "covariance"):
            arrays[condition][i] = clean[i].astype(np.float64) + noises[condition]
        if (i+1) % 100 == 0:
            print(f"prepared fixed noise {i+1}/{len(ids)}", flush=True)
    np.savez(args.output / "test_full.npz", **arrays)
    for model in ("resnet", "tcn"):
        source = baseline / "results" / "checkpoints" / "full" / model / "seed_17.pt"
        shutil.copyfile(source, args.output / f"{model}.pt")
        manifest["sources"][str(source.relative_to(baseline.parent))] = sha(source)
    manifest["test_records"] = len(ids)
    manifest["input_array_sha256"] = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()}
    for path in sorted(args.output.iterdir()):
        manifest["files"][path.name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["status"] = "complete"
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(args.output),"records":len(ids),"status":"complete"}),flush=True)


if __name__ == "__main__":
    main()
