"""Descriptive crossed training/noise-seed sensitivity, without retraining or pooling seeds as patients."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml
from src.audit_noise import CONDITIONS, powers, record_seed
from src.datasets import load_data, _data_identity
from src.evaluate import classification_metrics, predict, sha256
from src.models import build_model
from src.noise_generators import make_noise_triplet
from src.train import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_pilot.yaml")
    parser.add_argument(
        "--noise-seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505]
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = Path(cfg["results_dir"])
    out = base / "metrics" / (cfg["run_name"] + "_noise_repeats")
    tables = base / "tables" / (cfg["run_name"] + "_noise_repeats")
    out.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    gate = json.loads(
        (base / "tables" / cfg["run_name"] / "noise_gate.json").read_text()
    )
    if (
        not gate["passed"]
        or gate["config"] != cfg
        or not gate["gates"].get("bandpass", {}).get("strict_structure_control")
    ):
        raise ValueError("Original bandpass noise controls must pass first")
    X, Y, meta = load_data(cfg["data_dir"])
    identity = _data_identity(Path(cfg["data_dir"]), meta, Y)
    torch.set_num_threads(2)
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, fingerprints, indices = [], {}, None
    for name in cfg["models"]:
        for seed in cfg["seeds"]:
            path = base / "checkpoints" / cfg["run_name"] / name / f"seed_{seed}.pt"
            c = torch.load(path, map_location="cpu", weights_only=False)
            if (
                c["config"] != cfg
                or not c["completed"]
                or c["training_epochs_completed"] != cfg["train"]["epochs"]
                or c["data_identity"] != identity
            ):
                raise ValueError("Unfinished or stale checkpoint")
            if c["model_name"] != name or c["seed"] != seed:
                raise ValueError("Checkpoint model/seed identity mismatch")
            idx = np.array(c["test_indices"])
            if indices is not None and not np.array_equal(indices, idx):
                raise ValueError("Unpaired test cohort")
            if indices is None:
                indices = idx
                x, y = np.array(X[indices]), np.array(Y[indices])
            model = build_model(c["model_name"], **c["model_kwargs"]).to(device).eval()
            model.load_state_dict(c["model_state"])
            clean = predict(model, x, c["scale_mv"], cfg["train"]["batch_size"], device)
            models.append(
                (name, seed, model, c["scale_mv"], np.array(c["thresholds"]), clean)
            )
            fingerprints[str(path)] = sha256(path)
    ids, patients = (
        meta.iloc[indices].ecg_id.to_numpy(),
        meta.iloc[indices].patient_id.to_numpy(),
    )
    protocol = dict(
        source_config=cfg,
        noise_seeds=args.noise_seeds,
        checkpoint_sha256=fingerprints,
        status="running",
        kind="bandpass",
        design="Same trained checkpoints, same test ECGs, fresh record-keyed Gaussian streams. Crossed model training seeds and noise seeds; purely descriptive, no seed pseudoreplication or additional significance tests.",
    )
    protocol["torch_execution"] = dict(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows = []
    start = time.time()
    for noise_seed in args.noise_seeds:
        noise = {c: np.empty_like(x) for c in CONDITIONS}
        replay = np.array(
            [record_seed(noise_seed, i, "bandpass") for i in ids], dtype=np.uint32
        )
        for i, signal in enumerate(x):
            samples = make_noise_triplet(
                signal,
                cfg["sampling_rate"],
                0,
                int(replay[i]),
                band=tuple(cfg["noise"]["band"]),
            )
            for c in CONDITIONS:
                noise[c][i] = samples[c]
        for snr in cfg["noise"]["snrs"]:
            for condition in CONDITIONS:
                n = noise[condition] * np.float32(10 ** (-snr / 20))
                intensity = [powers(a, b) for a, b in zip(x, n)]
                actual = np.asarray([value[0] for value in intensity])
                noise_rms = np.stack([value[1] for value in intensity])
                lead_snr = np.stack([value[2] for value in intensity])
                if np.max(abs(actual - snr)) > 1e-4:
                    raise ValueError("Replay SNR failed")
                noisy = x + n
                for name, seed, model, scale, threshold, clean in models:
                    p = predict(model, noisy, scale, cfg["train"]["batch_size"], device)
                    values, loss = classification_metrics(y, p, threshold, clean)
                    file = (
                        out / f"{name}_{seed}_noise{noise_seed}_{snr}_{condition}.npz"
                    )
                    np.savez_compressed(
                        file,
                        p=p,
                        y=y,
                        ids=ids,
                        patient_ids=patients,
                        loss=loss,
                        thresholds=threshold,
                        indices=indices,
                        replay_seed=replay,
                        actual_snr=actual,
                        noise_rms=noise_rms,
                        lead_snr=lead_snr,
                    )
                    rows.append(
                        dict(
                            model=name,
                            seed=seed,
                            noise_seed=noise_seed,
                            snr=snr,
                            condition=condition,
                            prediction_path=file.as_posix(),
                            **values,
                        )
                    )
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        print(f"Completed independent noise seed {noise_seed}", flush=True)
    frame = pd.DataFrame(rows)
    index = ["model", "seed", "noise_seed", "snr"]
    left = frame[frame.condition.eq("electrode")].set_index(index)
    right = frame[frame.condition.eq("independent_rms")].set_index(index)
    contrasts = (
        left[["macro_auroc", "macro_ap", "macro_f1", "mean_loss"]]
        - right[["macro_auroc", "macro_ap", "macro_f1", "mean_loss"]]
    )
    contrasts.reset_index().to_csv(tables / "crossed_seed_contrasts.csv", index=False)
    mean_by_replay = contrasts.groupby(level=["model", "noise_seed", "snr"]).mean()
    mean_by_replay.reset_index().to_csv(
        tables / "mean_across_training_seeds.csv", index=False
    )
    summary = mean_by_replay.groupby(level=["model", "snr"]).agg(
        ["mean", "std", "min", "max"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(
        tables / "descriptive_noise_seed_summary.csv", index=False
    )
    protocol.update(
        status="completed",
        n_evaluations=len(rows),
        n_records=len(ids),
        n_patients=len(np.unique(patients)),
        elapsed_seconds=time.time() - start,
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("NOISE_REPLAY_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
