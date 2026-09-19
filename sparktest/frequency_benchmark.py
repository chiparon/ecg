"""Plan G: real PTB-XL 100/500 Hz, two epochs, ResNet/TCN, clean/electrode.

Requires immutable prepare_inputs outputs, official ptbxl_database.csv and
SHA256SUMS.txt. Only selected records500 .hea/.dat files are downloaded. Raw
cache survives failures/restarts; every invocation requires a NEW output dir.
No interpolation, synthetic ECG, CPU fallback, or scientific-result claim.
"""
from __future__ import annotations

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import wfdb
from torch.utils.data import DataLoader, RandomSampler

from phase1_ecg_robustness.src.audit_noise import record_seed
from phase1_ecg_robustness.src.datasets import LEADS
from phase1_ecg_robustness.src.download_ptbxl import (
    PTB_URL, ZIP_URL, checksums, download_file, extract_ranges, inventory,
    sha256, write_json,
)
from phase1_ecg_robustness.src.models import build_model
from phase1_ecg_robustness.src.noise_generators import make_noise_triplet
from phase1_ecg_robustness.src.train import _CleanDataset, seed_everything, validation_metrics
from sparktest.train_benchmark import MemoryMonitor, environment

COUNTS = {"train": 4000, "val": 1000, "test": 1000}
BATCHES = (128, 64, 32, 16)


def utc():
    return datetime.now(timezone.utc).isoformat()


def append_json(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def array_hash(array):
    value = hashlib.sha256()
    # Hash contiguous records without copying the entire 500 Hz corpus.
    for row in array:
        value.update(np.ascontiguousarray(row).tobytes())
    return value.hexdigest()


def array_info(path):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {"path": str(path), "sha256": sha256(path),
            "array_sha256": array_hash(array), "shape": list(array.shape),
            "dtype": str(array.dtype), "bytes": path.stat().st_size}


def save_array(path, array):
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)


def align_true500(signal, fields):
    """Native-500 counterpart of align_waveform: identical physical transforms.

    Deliberately does not call the 100-only helper with falsified fs or reshape
    its input. No filter, amplitude normalization, or resampling is performed.
    """
    signal = np.asarray(signal)
    names = [str(name).strip().upper() for name in fields.get("sig_name", [])]
    expected = [name.upper() for name in LEADS]
    if len(names) != 12 or len(set(names)) != 12 or set(names) != set(expected):
        raise ValueError(f"Expected exactly standard 12 leads, got {names}")
    if float(fields.get("fs", 0)) != 500 or signal.shape != (5000, 12):
        raise ValueError(f"Require actual fs500 shape(5000,12), got {fields.get('fs')}, {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError("Non-finite physical WFDB samples")
    units = fields.get("units", [])
    if len(units) != 12:
        raise ValueError("Missing per-lead physical units")
    factors = {"mv": 1.0, "uv": 0.001, "μv": 0.001, "µv": 0.001, "v": 1000.0}
    try:
        scale = np.array([factors[str(unit).strip().lower()] for unit in units])
    except KeyError as error:
        raise ValueError(f"Unsupported physical units: {units}") from error
    order = [names.index(name) for name in expected]
    mv = (signal * scale)[..., order].T
    means = mv.mean(axis=1, keepdims=True)
    mv = np.asarray(mv - means, dtype=np.float32)
    if not np.isfinite(mv).all():
        raise ValueError("Non-finite converted waveform")
    return mv, {"original_leads": fields["sig_name"], "original_units": units,
                "sampling_rate": float(fields["fs"]), "shape": list(mv.shape),
                "removed_dc_mv": means[:, 0].tolist(), "finite": True}


def fixed_inputs(args, output, summary):
    manifest_path = args.inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("smoke_only"):
        raise ValueError("Require complete non-smoke fixed input manifest")
    summary["input_hashes"] = {"manifest.json": sha256(manifest_path)}
    for name in ("train_pilot.npz", "val_pilot.npz", "test_full.npz", "resnet.pt", "tcn.pt"):
        actual = sha256(args.inputs / name)
        expected = manifest["files"][name]["sha256"]
        if actual != expected:
            raise ValueError(f"Immutable fixed input SHA256 mismatch: {name}")
        summary["input_hashes"][name] = actual
    splits = {}
    rate = output / "inputs_100"
    rate.mkdir()
    start = time.perf_counter()
    for split, name in (("train", "train_pilot"), ("val", "val_pilot"), ("test", "test_full")):
        with np.load(args.inputs / f"{name}.npz", allow_pickle=False) as archive:
            n = COUNTS[split]
            ids = np.asarray(archive["ecg_id"][:n], dtype=np.int64)
            patients = np.asarray(archive["patient_id"][:n], dtype=np.int64)
            labels = np.asarray(archive["y"][:n], dtype=np.float32)
            x = np.asarray(archive["clean" if split == "test" else "x"][:n], dtype=np.float32)
            if len(ids) != n or len(set(ids.tolist())) != n or x.shape != (n, 12, 1000):
                raise ValueError(f"Incorrect fixed {split} count/shape or repeated IDs")
            if labels.shape != (n, 5) or not np.isfinite(x).all() or not np.isin(labels, (0, 1)).all():
                raise ValueError(f"Invalid fixed {split} labels/signals")
            if split != "test" and len(archive["ecg_id"]) != n:
                raise ValueError(f"Require exact {n}-record fixed {split} pilot")
            save_array(rate / f"{split}_clean.npy", x)
            splits[split] = {"ecg_id": ids, "patient_id": patients, "y": labels}
            for key, values in splits[split].items():
                save_array(rate / f"{split}_{key}.npy", values)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for key in ("ecg_id", "patient_id"):
            if set(splits[left][key]) & set(splits[right][key]):
                raise ValueError(f"Fixed input {key} overlap: {left}/{right}")
    summary["selection"] = {split: {"count": len(values["ecg_id"]),
        "ecg_ids": values["ecg_id"].tolist(), "patient_ids": values["patient_id"].tolist()}
        for split, values in splits.items()}
    summary["rates"]["100"] = {"status": "preparing", "directory": str(rate),
        "clean_preparation_seconds": time.perf_counter() - start,
        "provenance": "immutable physical 100 Hz fixed input arrays; no reprocessing"}
    return splits


def download_true500(args, splits, output, summary):
    started = time.perf_counter()
    args.raw_cache.mkdir(parents=True, exist_ok=True)
    log = output / "download_inventory.jsonl"
    persistent_log = args.raw_cache / "frequency_download_history.jsonl"
    result = {"status": "running", "started_at": utc(), "failures": [],
              "inventory": str(log), "persistent_history": str(persistent_log),
              "official_url": PTB_URL, "archive_url": ZIP_URL,
              "raw_cache": str(args.raw_cache), "verified_files": 0}
    summary["download"] = result

    def record(row):
        row = {**row, "run_output": str(output), "at": utc()}
        append_json(log, row)
        append_json(persistent_log, row)
        result["verified_files"] += int(row.get("verified", False))

    def failure(stage, error, path=None):
        row = {"at": utc(), "stage": stage, "path": path, "error": repr(error)}
        result["failures"].append(row)
        record({"status": "failed", **row})
        write_json(output / "summary.json", summary)

    try:
        if not args.metadata.is_file() or not args.checksums.is_file():
            raise FileNotFoundError("Provide official PTB-XL 1.0.3 ptbxl_database.csv and SHA256SUMS.txt via --metadata/--checksums")
        hashes = checksums(args.checksums)
        metadata_hash = sha256(args.metadata)
        if hashes.get("ptbxl_database.csv") != metadata_hash:
            raise ValueError("Metadata does not match ptbxl_database.csv in official supplied SHA256SUMS.txt")
        result["metadata_sha256"] = metadata_hash
        result["checksums_sha256"] = sha256(args.checksums)
        result["checksum_source"] = PTB_URL + "SHA256SUMS.txt (supplied immutable file)"
        with args.metadata.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        metadata = {int(row["ecg_id"]): row for row in rows}
        if len(metadata) != len(rows):
            raise ValueError("Duplicate official metadata ecg_id")
        selected = {}
        for split in splits.values():
            for ecg_id, patient_id in zip(split["ecg_id"], split["patient_id"]):
                row = metadata[int(ecg_id)]
                if int(float(row["patient_id"])) != int(patient_id):
                    raise ValueError(f"Patient mismatch for fixed ECG {ecg_id}")
                name = row["filename_hr"]
                pure = PurePosixPath(name)
                if (pure.is_absolute() or ".." in pure.parts or "\\" in name
                        or len(pure.parts) != 3 or pure.parts[0] != "records500"):
                    raise ValueError(f"Unsafe/non-native filename_hr: {name}")
                selected[int(ecg_id)] = name
        paths = [name + ext for name in selected.values() for ext in (".hea", ".dat")]
        absent = sorted(set(paths) - hashes.keys())
        if absent:
            result["missing_checksum_entries"] = absent
            raise ValueError(f"Official SHA256 entries absent for {len(absent)} raw500 files")
        result["requested_files"] = len(paths)
        result["requested_records"] = len(selected)
        missing = []
        for relative in paths:
            target = args.raw_cache / relative
            if target.is_file() and sha256(target) == hashes[relative]:
                record(inventory(target, relative, PTB_URL + relative, hashes[relative], "existing", True))
            else:
                missing.append(relative)
        if missing:
            try:
                extract_ranges(args.raw_cache, missing, hashes, record, result)
            except Exception as error:
                failure("zip-range", error)
                # Helpers preserve complete verified members and retry only incomplete
                # small files. Never request a complete archive as a fallback.
                remaining = [p for p in missing if not (args.raw_cache / p).is_file()
                             or sha256(args.raw_cache / p) != hashes[p]]
                with ThreadPoolExecutor(max_workers=6) as pool:
                    pending = {pool.submit(download_file, args.raw_cache, p, PTB_URL, hashes[p]): p
                               for p in remaining}
                    for future in as_completed(pending):
                        relative = pending[future]
                        try:
                            record(future.result())
                        except Exception as direct_error:
                            failure("direct", direct_error, relative)
        missing = [p for p in paths if not (args.raw_cache / p).is_file()
                   or sha256(args.raw_cache / p) != hashes[p]]
        result["missing_files"] = missing
        if missing:
            raise OSError(f"{len(missing)} selected raw500 files remain absent or unverified")
        result["status"] = "complete"
        return selected
    except Exception as error:
        failure("raw500-prerequisite", error)
        result.update(status="blocked", reason=str(error), required_prerequisite=
            "All selected metadata filename_hr .hea/.dat records from official PTB-XL 1.0.3, matching supplied official SHA256SUMS.txt; reachable range/direct PhysioNet endpoints or prefilled verified --raw-cache")
        return None
    finally:
        result["seconds"] = time.perf_counter() - started
        result["finished_at"] = utc()
        write_json(output / "summary.json", summary)


def prepare_rate(rate_hz, args, splits, selected, output, summary):
    started = time.perf_counter()
    directory = output / f"inputs_{rate_hz}"
    rate = summary["rates"].setdefault(str(rate_hz), {"status": "preparing", "directory": str(directory)})
    if rate_hz == 500:
        directory.mkdir()
        qc_path = output / "raw500_waveform_qc.jsonl"
        rate["waveform_qc"] = str(qc_path)
        rate["provenance"] = "actual physical WFDB records500: named-lead reorder, unit-to-mV, per-lead DC removal only"
        for split, values in splits.items():
            path = directory / f"{split}_clean.npy"
            x = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                         shape=(COUNTS[split], 12, 5000))
            for index, ecg_id in enumerate(values["ecg_id"]):
                name = selected[int(ecg_id)]
                signal, fields = wfdb.rdsamp(str(args.raw_cache / name))
                x[index], qc = align_true500(signal, fields)
                append_json(qc_path, {"ecg_id": int(ecg_id), "record": name, **qc})
            x.flush()
            del x
            for key, array in values.items():
                save_array(directory / f"{split}_{key}.npy", array)
    clean_seconds = time.perf_counter() - started
    rate["noise"] = {"kind": "bandpass", "band_hz": [0.5, 40.0], "active": "all",
        "snr_db": 10, "base_seed": 20001, "record_seed": "record_seed(20001, ecg_id, 'bandpass')",
        "implementation": "original make_noise_triplet(...)[electrode]", "fixed_across_epochs": True}
    noise_started = time.perf_counter()
    for split, values in splits.items():
        clean = np.load(directory / f"{split}_clean.npy", mmap_mode="r")
        noisy = np.lib.format.open_memmap(directory / f"{split}_electrode.npy", mode="w+",
                                         dtype=np.float32, shape=clean.shape)
        for index, ecg_id in enumerate(values["ecg_id"]):
            noise = make_noise_triplet(clean[index], rate_hz, 10,
                record_seed(20001, int(ecg_id), "bandpass"), kind="bandpass", active=None)["electrode"]
            noisy[index] = clean[index].astype(np.float64) + noise
        noisy.flush()
        del noisy, clean
    rate["noise_generation_seconds"] = time.perf_counter() - noise_started
    if rate_hz == 500:
        rate["clean_preparation_seconds"] = clean_seconds
    rate["arrays"] = {p.name: array_info(p) for p in sorted(directory.glob("*.npy"))}
    rate["input_generation_seconds"] = (time.perf_counter() - started
        + (rate["clean_preparation_seconds"] if rate_hz == 100 else 0))
    rate["status"] = "complete"
    rate["finished_at"] = utc()
    write_json(output / "summary.json", summary)


def load_dataset(directory, split, condition, scale):
    x = np.load(directory / f"{split}_{condition}.npy", mmap_mode="r")
    y = np.load(directory / f"{split}_y.npy", mmap_mode="r")
    return _CleanDataset(x, y, np.arange(len(y)), scale)


def model_hash(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def evaluate(model, loader, device):
    model.eval()
    predictions, labels = [], []
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Non-finite inference/validation logits or BCE")
            total_loss += float(loss.item()) * len(y)
            predictions.append(logits.sigmoid().cpu().numpy())
            labels.append(y.cpu().numpy())
    torch.cuda.synchronize(device)
    labels, probabilities = np.concatenate(labels), np.concatenate(predictions)
    return total_loss / len(loader.dataset), probabilities, labels


def train_attempt(config, summary, output):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    seed_everything(17)
    torch.cuda.reset_peak_memory_stats(device)
    summary["environment"].update(gpu=torch.cuda.get_device_name(device),
        gpu_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
    reference_path = Path(config["inputs"]) / f"{config['model']}.pt"
    if sha256(reference_path) != config["checkpoint_source_sha256"]:
        raise ValueError("Reference checkpoint changed after input verification")
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    if reference["model_name"] != config["model"]:
        raise ValueError("Reference model name mismatch")
    scale = float(reference["scale_mv"])
    thresholds = np.asarray(reference["thresholds"], dtype=np.float32)
    kwargs = reference["model_kwargs"]
    if not math.isfinite(scale) or scale <= 0 or thresholds.shape != (5,) or not np.isfinite(thresholds).all():
        raise ValueError("Invalid original scale or thresholds")
    del reference
    model = build_model(config["model"], **kwargs).to(device)
    summary.update(scale_mv=scale, model_kwargs=kwargs, initial_model_sha256=model_hash(model),
        initialization="fresh seed17, original model_kwargs; never reference trained weights",
        precision="deterministic FP32; no AMP/TF32", optimizer={"name": "AdamW", "lr": 0.001, "weight_decay": 0.0001})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()
    directory = Path(config["data_directory"])
    datasets = {split: load_dataset(directory, split, config["condition"], scale) for split in COUNTS}
    loaders = {}
    for index, (split, data) in enumerate(datasets.items()):
        sampler = RandomSampler(data, generator=torch.Generator().manual_seed(17)) if split == "train" else None
        loaders[split] = DataLoader(data, batch_size=config["batch_size"], sampler=sampler,
            shuffle=False, num_workers=0, pin_memory=True,
            generator=torch.Generator().manual_seed(18 + index))
    for epoch in (1, 2):
        torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, steps = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch} step {steps}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * len(y)
            steps += 1
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_start
        validation_start = time.perf_counter()
        val_loss, probabilities, labels = evaluate(model, loaders["val"], device)
        validation_seconds = time.perf_counter() - validation_start
        metrics = validation_metrics(labels, probabilities)
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("Non-finite trained parameters")
        if any(not bool(torch.isfinite(state[key]).all()) for state in optimizer.state.values()
               for key in ("exp_avg", "exp_avg_sq")):
            raise FloatingPointError("Non-finite AdamW moments")
        checkpoint = output / f"epoch_{epoch:03d}.pt"
        save_started = time.perf_counter()
        with checkpoint.open("xb") as stream:
            torch.save({"model_name": config["model"], "model_kwargs": kwargs,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scale_mv": scale, "thresholds": thresholds.tolist(), "epoch": epoch,
                "seed": 17, "config": config, "validation_metrics": metrics,
                "benchmark_only": True, "completed": epoch == 2}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        row = {"epoch": epoch, "train_loss": loss_sum / COUNTS["train"],
            "val_loss": val_loss, "validation": metrics, "steps": steps,
            "train_epoch_seconds": train_seconds, "samples_per_sec": COUNTS["train"] / train_seconds,
            "validation_seconds": validation_seconds, "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint), "checkpoint_seconds": time.perf_counter() - save_started,
            "epoch_seconds": time.perf_counter() - epoch_start}
        summary["epochs"].append(row)
        write_json(output / "summary.json", summary)
        print(json.dumps(row), flush=True)
    with (output / "validation_predictions.npz").open("xb") as stream:
        np.savez(stream, probabilities=probabilities, y=labels,
            ecg_id=np.load(directory / "val_ecg_id.npy"), patient_id=np.load(directory / "val_patient_id.npy"))
    # Warm-up is inference-only after training; no optimizer/BatchNorm updates.
    model.eval()
    with torch.inference_mode():
        for index, (x, _) in enumerate(loaders["test"]):
            model(x.to(device, non_blocking=True))
            if index == 2:
                break
    torch.cuda.synchronize(device)
    inference_start = time.perf_counter()
    test_loss, probabilities, labels = evaluate(model, loaders["test"], device)
    inference_seconds = time.perf_counter() - inference_start
    prediction_path = output / "test_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez(stream, probabilities=probabilities, y=labels,
            ecg_id=np.load(directory / "test_ecg_id.npy"), patient_id=np.load(directory / "test_patient_id.npy"))
    summary["inference"] = {"seconds": inference_seconds, "samples_per_sec": COUNTS["test"] / inference_seconds,
        "records": COUNTS["test"], "batch_size": config["batch_size"], "warmup_batches": 3,
        "scope": "end-to-end data loading, mV normalization, H2D, model, BCE, sigmoid and D2H; excludes warmup/serialization/metrics",
        "bce": test_loss, "metrics": validation_metrics(labels, probabilities),
        "predictions": str(prediction_path), "predictions_sha256": sha256(prediction_path)}
    summary["train_seconds"] = sum(row["train_epoch_seconds"] for row in summary["epochs"])
    summary["samples_per_sec"] = 2 * COUNTS["train"] / summary["train_seconds"]
    summary["checkpoint"] = summary["epochs"][-1]["checkpoint"]
    summary["checkpoint_sha256"] = summary["epochs"][-1]["checkpoint_sha256"]


def worker(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = config_path.parent
    summary = {"status": "running", "started_at": utc(), "config": config,
               "epochs": [], "environment": environment()}
    write_json(output / "summary.json", summary)
    monitor = MemoryMonitor(output)
    started = time.perf_counter()
    monitor.start()
    exit_code = 1
    try:
        train_attempt(config, summary, output)
        summary["status"] = "complete"
        exit_code = 0
    except torch.cuda.OutOfMemoryError as error:
        summary.update(status="oom", error=repr(error), traceback=traceback.format_exc(), failed_at=utc())
        exit_code = 75
    except Exception as error:
        summary.update(status="failed", error=repr(error), traceback=traceback.format_exc(), failed_at=utc())
    finally:
        summary["memory"] = monitor.finish()
        if torch.cuda.is_initialized():
            summary["memory"].update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        summary.update(seconds=time.perf_counter() - started, finished_at=utc(), exit_code=exit_code)
        write_json(output / "summary.json", summary)
    return exit_code


def run_jobs(args, output, summary):
    for rate_hz in (100, 500):
        for model in ("resnet", "tcn"):
            for condition in ("clean", "electrode"):
                job = {"rate_hz": rate_hz, "model": model, "condition": condition, "attempts": []}
                summary["jobs"].append(job)
                if summary["rates"].get(str(rate_hz), {}).get("status") != "complete":
                    job.update(status="blocked", reason="Rate input preparation is not complete")
                    write_json(output / "summary.json", summary)
                    continue
                for batch in BATCHES:
                    directory = output / f"{rate_hz}hz_{model}_{condition}" / f"batch_{batch}"
                    directory.mkdir(parents=True)
                    config = {"rate_hz": rate_hz, "model": model, "condition": condition,
                        "batch_size": batch, "seed": 17, "epochs": 2, "inputs": str(args.inputs),
                        "checkpoint_source_sha256": summary["input_hashes"][f"{model}.pt"],
                        "data_directory": str(output / f"inputs_{rate_hz}"),
                        "source_hashes": summary["source_hashes"], "input_hashes": summary["input_hashes"]}
                    config_path = directory / "config.json"
                    write_json(config_path, config)
                    attempt = {"batch_size": batch, "started_at": utc(), "summary": str(directory / "summary.json"),
                               "fresh_process_and_seed17_weights": True}
                    job["attempts"].append(attempt)
                    job["status"] = "running"
                    write_json(output / "summary.json", summary)
                    with (directory / "stdout.log").open("x", encoding="utf-8") as stream:
                        process = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                            "--worker", str(config_path)], stdout=stream, stderr=subprocess.STDOUT, check=False)
                    result_path = directory / "summary.json"
                    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
                    attempt.update(finished_at=utc(), exit_code=process.returncode,
                        status=result.get("status", "failed"), error=result.get("error"))
                    if process.returncode == 0 and result.get("status") == "complete":
                        job.update(status="complete", batch_size=batch, successful_attempt=str(directory / "summary.json"),
                            samples_per_sec=result["samples_per_sec"], epochs=result["epochs"],
                            inference=result["inference"], memory=result["memory"], checkpoint=result["checkpoint"])
                        break
                    if process.returncode != 75 or result.get("status") != "oom":
                        job.update(status="failed", reason="Non-OOM failure; no batch-size fallback")
                        break
                    job.update(status="oom", reason="Real CUDA OOM; restart fresh process/weights at next smaller batch")
                write_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, help="Immutable prepare_inputs directory")
    parser.add_argument("--output", type=Path, help="NEW run directory; existing paths are refused")
    parser.add_argument("--metadata", type=Path, help="Official PTB-XL 1.0.3 ptbxl_database.csv")
    parser.add_argument("--checksums", type=Path, help="Official PTB-XL 1.0.3 SHA256SUMS.txt")
    parser.add_argument("--raw-cache", type=Path, help="Persistent cache root containing records500/")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    for key in ("inputs", "output", "metadata", "checksums", "raw_cache"):
        if getattr(args, key) is None:
            parser.error(f"--{key.replace('_', '-')} is required")
        setattr(args, key, getattr(args, key).resolve())
    output = args.output
    # Prevent accidental writes under immutable inputs or raw/source metadata.
    if output == args.inputs or args.inputs in output.parents:
        parser.error("Output must not be inside immutable --inputs")
    if args.raw_cache == args.inputs or args.inputs in args.raw_cache.parents:
        parser.error("Raw cache must not be inside immutable --inputs")
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "sparktest/train_benchmark.py"] + [
        ROOT / "phase1_ecg_robustness/src" / name for name in (
            "download_ptbxl.py", "prepare_ptbxl.py", "models.py", "train.py", "datasets.py",
            "noise_generators.py", "audit_noise.py", "lead_matrix.py", "covariance_matching.py")]
    summary = {"status": "running", "started_at": utc(), "benchmark": "G",
        "benchmark_only": True, "scientific_claims": False, "rates": {}, "jobs": [],
        "environment": environment(), "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "counts": COUNTS, "seed": 17, "epochs": 2, "batch_attempt_order": list(BATCHES),
        "condition_protocol": "Each clean/electrode condition is independently trained and validated for two epochs, then inferred on matching test condition. Same fixed record IDs at both native rates.",
        "comparison_caveat": "Benchmark only: same sample-domain architectures have different time-domain receptive fields at 100/500 Hz; batch fallback can alter optimizer trajectory; unified system RAM includes unrelated workloads."}
    started = time.perf_counter()
    exit_code = 1
    write_json(output / "summary.json", summary)
    try:
        splits = fixed_inputs(args, output, summary)
        prepare_rate(100, args, splits, None, output, summary)
        selected = download_true500(args, splits, output, summary)
        if selected is None:
            summary["rates"]["500"] = {"status": "blocked", "reason": summary["download"]["reason"],
                "required_prerequisite": summary["download"]["required_prerequisite"]}
        else:
            try:
                prepare_rate(500, args, splits, selected, output, summary)
            except Exception as error:
                summary["rates"]["500"].update(status="failed", error=repr(error), failed_at=utc(),
                    traceback=traceback.format_exc())
        del splits
        gc.collect()
        run_jobs(args, output, summary)
        completed = sum(job["status"] == "complete" for job in summary["jobs"])
        summary["status"] = ("complete" if completed == 8 else "partial" if completed
            else "failed" if any(job["status"] in ("failed", "oom") for job in summary["jobs"]) else "blocked")
        exit_code = 0 if completed == 8 else 2
    except Exception as error:
        summary.update(status="failed", error=repr(error), traceback=traceback.format_exc(), failed_at=utc())
    finally:
        summary["coverage"] = {"expected_jobs": 8,
            "completed_jobs": sum(job.get("status") == "complete" for job in summary["jobs"]),
            "true500_completed_jobs": sum(job.get("status") == "complete" and job["rate_hz"] == 500 for job in summary["jobs"]),
            "not_completed": [{"rate_hz": rate, "model": model, "condition": condition}
                for rate in (100, 500) for model in ("resnet", "tcn") for condition in ("clean", "electrode")
                if not any(job["rate_hz"] == rate and job["model"] == model and job["condition"] == condition
                    and job.get("status") == "complete" for job in summary["jobs"])]}
        summary.update(finished_at=utc(), seconds=time.perf_counter() - started, exit_code=exit_code)
        write_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), "status": summary["status"], "coverage": summary["coverage"]}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
