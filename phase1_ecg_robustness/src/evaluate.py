"""Evaluate saved clean models using identical ECGs and record-keyed perturbations."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from .audit_noise import CONDITIONS, powers, record_seed
from .datasets import load_data, _data_identity
from .models import build_model
from .noise_generators import make_noise_triplet
from .train import seed_everything

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def classification_metrics(y, p, thresholds, clean_p=None):
    if y.shape != p.shape or p.shape[1] != 5 or not np.isfinite(p).all():
        raise ValueError("Invalid prediction shape or nonfinite probabilities")
    if np.any(y.sum(axis=0) == 0) or np.any(y.sum(axis=0) == len(y)):
        raise ValueError(
            "Evaluation subset must contain positive and negative examples of all five classes"
        )
    pred = p >= thresholds
    auroc = roc_auc_score(y, p, average=None)
    ap = average_precision_score(y, p, average=None)
    f1 = f1_score(y, pred, average=None, zero_division=0)
    safe = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    loss = -np.mean(y * np.log(safe) + (1 - y) * np.log1p(-safe), axis=1)
    entropy = -(safe * np.log(safe) + (1 - safe) * np.log1p(-safe))
    # Classwise equal-width ECE of positive-class probability, 15 bins, then macro mean.
    ece = 0.0
    for j in range(5):
        ids = np.minimum((p[:, j] * 15).astype(int), 14)
        for b in range(15):
            m = ids == b
            if m.any():
                ece += m.mean() * abs(p[m, j].mean() - y[m, j].mean()) / 5
    result = dict(
        macro_auroc=float(auroc.mean()),
        macro_ap=float(ap.mean()),
        macro_f1=float(f1.mean()),
        micro_auroc=float(roc_auc_score(y, p, average="micro")),
        micro_ap=float(average_precision_score(y, p, average="micro")),
        micro_f1=float(f1_score(y, pred, average="micro", zero_division=0)),
        brier=float(np.mean((p - y) ** 2)),
        ece=float(ece),
        mean_loss=float(loss.mean()),
        mean_confidence=float(np.maximum(p, 1 - p).mean()),
        mean_entropy=float(entropy.mean()),
        clean_agreement=(
            float(np.mean(pred == (clean_p >= thresholds)))
            if clean_p is not None
            else 1.0
        ),
        probability_shift=(
            float(np.abs(p - clean_p).mean()) if clean_p is not None else 0.0
        ),
    )
    for j, name in enumerate(CLASSES):
        result.update(
            {
                f"auroc_{name}": float(auroc[j]),
                f"ap_{name}": float(ap[j]),
                f"f1_{name}": float(f1[j]),
            }
        )
    return result, loss


@torch.inference_mode()
def predict(model, x, scale_mv, batch_size, device):
    result = []
    for begin in range(0, len(x), batch_size):
        batch = np.ascontiguousarray(x[begin : begin + batch_size], dtype=np.float32)
        tensor = torch.from_numpy(batch).to(device) / scale_mv
        result.append(torch.sigmoid(model(tensor)).cpu().numpy())
    return np.concatenate(result)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def run(config):
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    base = Path(cfg["results_dir"])
    out = base / "metrics" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    gate = json.loads(
        (base / "tables" / cfg["run_name"] / "noise_gate.json").read_text(
            encoding="utf-8"
        )
    )
    if not gate["passed"] or gate["config"] != cfg:
        raise RuntimeError(
            "Current configuration has not passed the pre-training noise gate"
        )
    torch.set_num_threads(4)
    seed_everything(0)  # Same deterministic FP32 policy as training and validation.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y, meta = load_data(cfg["data_dir"])
    identity = _data_identity(Path(cfg["data_dir"]), meta, Y)
    states = []
    indices = None
    ckpt_hashes = {}
    for name in cfg["models"]:
        for seed in cfg["seeds"]:
            checkpoint = (
                base / "checkpoints" / cfg["run_name"] / name / f"seed_{seed}.pt"
            )
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if ckpt["config"] != cfg:
                raise RuntimeError(f"Checkpoint config mismatch: {checkpoint}")
            if (
                not ckpt.get("completed")
                or ckpt.get("training_epochs_completed") != cfg["train"]["epochs"]
            ):
                raise RuntimeError(f"Checkpoint training is unfinished: {checkpoint}")
            if ckpt.get("data_identity") != identity:
                raise RuntimeError(
                    f"Checkpoint dataset identity mismatch: {checkpoint}"
                )
            if ckpt.get("model_name") != name or ckpt.get("seed") != seed:
                raise RuntimeError(
                    f"Checkpoint model/seed identity mismatch: {checkpoint}"
                )
            this_indices = np.asarray(ckpt["test_indices"], dtype=int)
            if indices is not None and not np.array_equal(indices, this_indices):
                raise RuntimeError(
                    "Model checkpoints do not share identical test indices"
                )
            indices = this_indices
            model = (
                build_model(ckpt["model_name"], **ckpt["model_kwargs"])
                .to(device)
                .eval()
            )
            model.load_state_dict(ckpt["model_state"])
            states.append(
                dict(
                    model=model,
                    name=name,
                    seed=seed,
                    scale=float(ckpt["scale_mv"]),
                    thresholds=np.asarray(ckpt["thresholds"]),
                )
            )
            ckpt_hashes[str(checkpoint)] = sha256(checkpoint)
    x = np.asarray(X[indices], dtype=np.float32)
    y = np.asarray(Y[indices], dtype=np.float32)
    ids = meta.iloc[indices].ecg_id.to_numpy(dtype=np.int64)
    patients = meta.iloc[indices].patient_id.to_numpy()
    if not np.all(meta.iloc[indices].strat_fold == 10):
        raise RuntimeError("Evaluation outside official test fold")
    rows = []
    spec = cfg["noise"]
    start = time.time()
    provenance = dict(
        config=cfg,
        checkpoints_sha256=ckpt_hashes,
        ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        n_records=len(ids),
        n_patients=len(np.unique(patients)),
        noise_seed_rule="SeedSequence([noise.seed,ecg_id,kind_code]); independent of training seed and model; same realization scaled across SNR",
        noise_unit="mV before training-only scalar normalization",
        clean_snr_sentinel=100,
        device=str(device),
        status="running",
    )
    provenance["torch_execution"] = dict(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
    )
    (out / "evaluation_protocol.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    def save(state, p, kind, snr, condition, active, diagnostics=None):
        sub = out / state["name"] / f"seed_{state['seed']}"
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"{kind}_{snr}_{condition}_{active}.npz"
        values, loss = classification_metrics(
            y, p, state["thresholds"], state.get("clean_p")
        )
        data = dict(
            p=p,
            y=y,
            ids=ids,
            patient_ids=patients,
            loss=loss,
            thresholds=state["thresholds"],
            indices=indices,
        )
        if diagnostics is not None:
            data.update(diagnostics)
        np.savez_compressed(path, **data)
        rows.append(
            dict(
                model=state["name"],
                seed=state["seed"],
                kind=kind,
                snr=snr,
                condition=condition,
                active=active,
                prediction_path=path.as_posix(),
                **values,
            )
        )
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        print(
            f"eval {state['name']}/{state['seed']} {kind}/{snr}/{condition}/{active} AUROC={values['macro_auroc']:.5f}",
            flush=True,
        )

    for state in states:
        p = predict(
            state["model"], x, state["scale"], cfg["train"]["batch_size"], device
        )
        state["clean_p"] = p
        save(state, p, "clean", 100, "clean", "all")
    groups = [(kind, "all", spec["snrs"], CONDITIONS) for kind in spec["kinds"]]
    groups += [
        (
            "bandpass",
            e,
            [spec.get("electrode_snr", 10)],
            ("independent_rms", "electrode", "covariance"),
        )
        for e in spec.get("electrodes", [])
    ]
    for kind, active, snrs, conditions in groups:
        # Generate once per source/subset at 0dB; positive SNR is scalar rescaling.
        noises = {c: np.empty_like(x) for c in conditions}
        seeds = np.array(
            [record_seed(spec["seed"], ecg_id, kind) for ecg_id in ids], dtype=np.uint32
        )
        for i, (signal, seed) in enumerate(zip(x, seeds)):
            triplet = make_noise_triplet(
                signal,
                cfg["sampling_rate"],
                0,
                int(seed),
                kind=kind,
                active=None if active == "all" else [active],
                nstdb=spec.get("nstdb_dir"),
                band=tuple(spec["band"]),
            )
            for c in conditions:
                noises[c][i] = triplet[c]
            if (i + 1) % 500 == 0:
                print(f"noise {kind}/{active} {i+1}/{len(x)}", flush=True)
        for snr in snrs:
            multiplier = np.float32(10 ** (-snr / 20))
            for condition, n0 in noises.items():
                n = n0 * multiplier
                d = [powers(signal, noise) for signal, noise in zip(x, n)]
                actual_snr = np.array([a[0] for a in d])
                noise_rms = np.array([a[1] for a in d])
                lead_snr = np.array([a[2] for a in d])
                if np.max(np.abs(actual_snr - snr)) > 1e-4:
                    raise RuntimeError(
                        "Per-record SNR calibration failed at evaluation"
                    )
                noisy = x + n
                if noisy.shape != x.shape or not np.isfinite(noisy).all():
                    raise RuntimeError("Invalid perturbed inputs")
                diagnostics = dict(
                    actual_snr=actual_snr,
                    noise_rms=noise_rms,
                    lead_snr=lead_snr,
                    replay_seed=seeds,
                )
                for state in states:
                    p = predict(
                        state["model"],
                        noisy,
                        state["scale"],
                        cfg["train"]["batch_size"],
                        device,
                    )
                    save(state, p, kind, snr, condition, active, diagnostics)
        del noises
    provenance.update(
        status="completed",
        elapsed_seconds=time.time() - start,
        n_evaluations=len(rows),
        input_shape=list(x.shape),
        all_metrics_finite=bool(
            np.isfinite(pd.DataFrame(rows).select_dtypes("number")).all().all()
        ),
    )
    (out / "evaluation_protocol.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(
        f"Evaluation complete: {len(rows)} conditions/models, {time.time()-start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)
