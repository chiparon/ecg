"""Standalone A/B/E/F training benchmarks using unchanged phase-one models."""
from __future__ import annotations

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import contextlib
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler

from phase1_ecg_robustness.src.models import build_model
from phase1_ecg_robustness.src.train import _CleanDataset, _seed_worker, validation_metrics

PRECISIONS = ("fp32", "fp32_nondeterministic", "amp_fp16", "amp_bf16", "tf32")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_write(path, value):
    # Exclusive creation protects prior runs, even when their summary is incomplete.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_hashes():
    paths = [Path(__file__), ROOT / "phase1_ecg_robustness/src/models.py",
             ROOT / "phase1_ecg_robustness/src/train.py"]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def environment():
    return {"host": platform.node(), "platform": platform.platform(),
            "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "cpu_count": os.cpu_count(),
            "interference_note": "Existing ComfyUI and other processes are not stopped; concurrent workloads may affect results."}


class MemoryMonitor:
    """Sample process trees/system RAM and optional portable nvidia-smi telemetry."""

    def __init__(self, directory):
        self.directory = directory
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.peaks = {"process_tree_rss_bytes": 0, "system_used_bytes": 0,
                      "system_total_bytes": psutil.virtual_memory().total,
                      "sample_interval_sec": 1.0, "samples": 0}
        self.smi = shutil.which("nvidia-smi")
        self.errors = []

    def sample(self):
        rss = 0
        process = psutil.Process()
        for item in [process] + process.children(recursive=True):
            try:
                rss += item.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        used = psutil.virtual_memory().used
        self.peaks["process_tree_rss_bytes"] = max(self.peaks["process_tree_rss_bytes"], rss)
        self.peaks["system_used_bytes"] = max(self.peaks["system_used_bytes"], used)
        self.peaks["samples"] += 1
        row = {"timestamp": timestamp(), "process_tree_rss_bytes": rss, "system_used_bytes": used}
        if self.smi:
            result = subprocess.run([self.smi,
                "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            row["nvidia_smi_exit_code"] = result.returncode
            row["nvidia_smi_csv"] = result.stdout.strip()
            if result.stderr.strip():
                row["nvidia_smi_stderr"] = result.stderr.strip()
        return row

    def run(self):
        with (self.directory / "memory_samples.jsonl").open("x", encoding="utf-8") as stream:
            while True:
                try:
                    row = self.sample()
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                except Exception as exc:
                    self.errors.append(f"{type(exc).__name__}: {exc}")
                if self.stop_event.wait(1.0):
                    break

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join()
        return {**self.peaks, "nvidia_smi_available": bool(self.smi),
                "telemetry_errors": self.errors,
                "rss_note": "Sampled process-tree RSS may double-count shared mmap pages; system-used RAM includes unrelated processes. CUDA allocator peaks are reported separately."}


class MappedCleanDataset(_CleanDataset):
    """Reuse phase-one sample normalization, reopen mmap rather than pickle ECG arrays."""

    def __init__(self, x_path, y_path, scale_mv):
        self.x_path, self.y_path = str(x_path), str(y_path)
        signals = np.load(self.x_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(self.y_path, mmap_mode="r", allow_pickle=False)
        super().__init__(signals, labels, np.arange(len(labels)), scale_mv)

    def __getstate__(self):
        return self.x_path, self.y_path, self.scale_mv

    def __setstate__(self, state):
        self.__init__(*state)


def prepare_dataset(inputs, output, name, scale):
    source = inputs / f"{name}.npz"
    with np.load(source, allow_pickle=False) as data:
        x, y = data["x"], data["y"]
        if x.dtype != np.float32 or x.shape != (len(y), 12, 1000):
            raise ValueError(f"Invalid ECG shape/dtype in {source}")
        if y.dtype != np.float32 or y.shape != (len(x), 5) or not len(x):
            raise ValueError(f"Invalid labels in {source}")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError(f"Invalid non-finite or non-binary input in {source}")
        if data["ecg_id"].shape != (len(x),) or data["patient_id"].shape != (len(x),):
            raise ValueError(f"Invalid identifiers in {source}")
        for key, array in (("x", x), ("y", y)):
            with (output / f"{name}_{key}.npy").open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
    return MappedCleanDataset(output / f"{name}_x.npy", output / f"{name}_y.npy", scale)


def configure(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is not a benchmark result")
    requested = torch.device(args.device)
    device = torch.device("cuda", torch.cuda.current_device() if requested.index is None else requested.index)
    torch.cuda.set_device(device)
    if args.precision == "amp_bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("AMP BF16 is unsupported on this CUDA device")
    deterministic = args.precision == "fp32"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = args.precision == "tf32"
    torch.backends.cudnn.allow_tf32 = args.precision == "tf32"
    torch.cuda.reset_peak_memory_stats(device)
    return device


def autocast_context(precision):
    if precision in ("amp_fp16", "amp_bf16"):
        return torch.autocast("cuda", dtype=torch.float16 if precision == "amp_fp16" else torch.bfloat16)
    return contextlib.nullcontext()


def scaler_for(precision):
    enabled = precision == "amp_fp16"
    # Both current torch.amp and the older torch.cuda.amp API use the same scale semantics.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_task(args, output, summary):
    device = configure(args)
    summary["environment"].update({"gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory})
    inputs = Path(args.inputs).resolve()
    checkpoint_path = inputs / f"{args.model}.pt"
    identity_paths = [inputs / "manifest.json", checkpoint_path,
                      inputs / f"train_{args.dataset}.npz", inputs / f"val_{args.dataset}.npz"]
    summary["input_hashes"] = {path.name: digest(path) for path in identity_paths}
    reference = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if reference["model_name"] != args.model:
        raise ValueError("Checkpoint model name disagrees with requested model")
    model_kwargs = reference["model_kwargs"]
    scale = float(reference["scale_mv"])
    thresholds = np.asarray(reference["thresholds"], dtype=np.float32)
    if not math.isfinite(scale) or scale <= 0 or thresholds.shape != (5,) or not np.isfinite(thresholds).all():
        raise ValueError("Invalid reference normalization or thresholds")
    del reference
    # Architectures and scale come from the reference; weights are freshly seed-initialized.
    model = build_model(args.model, **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = scaler_for(args.precision)
    summary.update({"scale_mv": scale, "model_kwargs": model_kwargs,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "initialization": "fresh seeded weights, not checkpoint fine-tuning",
                    "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.0001},
                    "precision_settings": {"deterministic": args.precision == "fp32",
                        "tf32": args.precision == "tf32", "cudnn_benchmark": False,
                        "grad_scaler_enabled": scaler.is_enabled()}})
    cache = output / "dataset_cache"
    cache.mkdir()
    prepare_start = time.perf_counter()
    train_data = prepare_dataset(inputs, cache, f"train_{args.dataset}", scale)
    val_data = prepare_dataset(inputs, cache, f"val_{args.dataset}", scale)
    summary["input_preparation_seconds"] = time.perf_counter() - prepare_start
    summary["records"] = {"train": len(train_data), "validation": len(val_data)}
    # Separate generators make the shuffle order independent of worker lifecycle/persistence.
    sampler = RandomSampler(train_data, generator=torch.Generator().manual_seed(args.seed))
    loader_options = {"batch_size": args.batch_size, "num_workers": args.num_workers,
        "pin_memory": args.pin_memory, "persistent_workers": args.persistent_workers,
        "worker_init_fn": _seed_worker}
    if args.num_workers:
        loader_options["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(train_data, sampler=sampler,
        generator=torch.Generator().manual_seed(args.seed + 1), **loader_options)
    val_loader = DataLoader(val_data, shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 2), **loader_options)
    rows = summary["epochs"]
    last_probabilities = None
    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        wait_seconds = step_seconds = 0.0
        steps = skipped_steps = 0
        iterator_start = time.perf_counter()
        iterator = iter(train_loader)
        wait_seconds += time.perf_counter() - iterator_start
        while True:
            wait_start = time.perf_counter()
            try:
                x, y = next(iterator)
            except StopIteration:
                wait_seconds += time.perf_counter() - wait_start
                break
            wait_seconds += time.perf_counter() - wait_start
            step_start = time.perf_counter()
            x, y = x.to(device, non_blocking=args.pin_memory), y.to(device, non_blocking=args.pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args.precision):
                logits = model(x)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, step {steps}")
            previous_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            skipped_steps += int(scaler.get_scale() < previous_scale)
            torch.cuda.synchronize(device)
            step_seconds += time.perf_counter() - step_start
            train_loss += float(loss.detach().item()) * len(y)
            steps += 1
        train_seconds = time.perf_counter() - epoch_start
        validation_start = time.perf_counter()
        model.eval()
        labels, probabilities = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=args.pin_memory), y.to(device, non_blocking=args.pin_memory)
                with autocast_context(args.precision):
                    logits = model(x)
                    loss = criterion(logits, y)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite validation output")
                val_loss += float(loss.item()) * len(y)
                labels.append(y.cpu().numpy())
                probabilities.append(logits.float().sigmoid().cpu().numpy())
        torch.cuda.synchronize(device)
        validation_seconds = time.perf_counter() - validation_start
        labels, probabilities = np.concatenate(labels), np.concatenate(probabilities)
        metrics = validation_metrics(labels, probabilities)
        if metrics["macro_auroc"] is None or not math.isfinite(metrics["macro_auroc"]):
            raise FloatingPointError("Five-class validation AUROC is undefined")
        metrics["macro_f1_reference_thresholds"] = float(f1_score(labels, probabilities >= thresholds,
            average="macro", zero_division=0))
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("Non-finite model parameters after optimizer steps")
        optimizer_steps = [int(state["step"].item()) for state in optimizer.state.values() if "step" in state]
        expected_updates = sum(r["steps"] - r["amp_skipped_steps"] for r in rows) + steps - skipped_steps
        if expected_updates <= 0 or len(optimizer_steps) != len(list(model.parameters())) or any(
            count != expected_updates for count in optimizer_steps
        ):
            raise RuntimeError("AdamW update counts do not match successful unscaled/scaled steps")
        for state in optimizer.state.values():
            if not torch.isfinite(state["exp_avg"]).all() or not torch.isfinite(state["exp_avg_sq"]).all():
                raise FloatingPointError("Non-finite AdamW moment state")
        row = {"epoch": epoch, "train_loss": train_loss / len(train_data),
            "val_loss": val_loss / len(val_data), "validation": metrics,
            "data_wait_seconds": wait_seconds, "train_step_seconds": step_seconds,
            "mean_step_seconds": step_seconds / steps, "train_epoch_seconds": train_seconds,
            "validation_seconds": validation_seconds, "steps": steps,
            "amp_skipped_steps": skipped_steps, "optimizer_step_min": min(optimizer_steps),
            "optimizer_step_max": max(optimizer_steps),
            "samples_per_sec": len(train_data) / train_seconds,
            "data_wait_fraction": wait_seconds / train_seconds}
        checkpoint_start = time.perf_counter()
        checkpoint_path = output / f"epoch_{epoch:03d}.pt"
        payload = {"model_name": args.model, "model_kwargs": model_kwargs,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "grad_scaler_state": scaler.state_dict(), "scale_mv": scale,
            "thresholds": thresholds.tolist(), "threshold_origin": "fixed original reference checkpoint",
            "epoch": epoch, "seed": args.seed, "config": summary["config"],
            "validation_metrics": metrics, "input_hashes": summary["input_hashes"],
            "source_hashes": summary["source_hashes"], "completed": epoch == args.epochs}
        with checkpoint_path.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_state = model.state_dict()
        if saved["epoch"] != epoch or set(saved["model_state"]) != set(expected_state):
            raise RuntimeError("Saved checkpoint metadata/state validation failed")
        for key, tensor in expected_state.items():
            if not torch.equal(tensor.detach().cpu(), saved["model_state"][key]):
                raise RuntimeError(f"Saved checkpoint tensor mismatch: {key}")
        del saved, payload
        row.update({"checkpoint": checkpoint_path.name, "checkpoint_sha256": digest(checkpoint_path),
                    "checkpoint_validated": True,
                    "checkpoint_seconds": time.perf_counter() - checkpoint_start,
                    "epoch_seconds": time.perf_counter() - epoch_start})
        json_write(output / f"epoch_{epoch:03d}.json", row)
        rows.append(row)
        last_probabilities = probabilities
        print(f"{args.model} {args.dataset} seed={args.seed} epoch={epoch} loss={row['train_loss']:.6f} AUROC={metrics['macro_auroc']:.6f}", flush=True)
    with (output / "validation_predictions.npz").open("xb") as stream:
        np.savez(stream, probabilities=last_probabilities, y=labels, thresholds=thresholds)
        stream.flush()
        os.fsync(stream.fileno())
    summary["validation_predictions_sha256"] = digest(output / "validation_predictions.npz")
    summary["validation_probability_array_sha256"] = hashlib.sha256(
        np.ascontiguousarray(last_probabilities).tobytes()
    ).hexdigest()
    total_train = sum(row["train_epoch_seconds"] for row in rows)
    summary.update({"train_seconds": total_train,
        "samples_per_sec": len(train_data) * args.epochs / total_train,
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in rows])),
        "mean_step_seconds": sum(row["train_step_seconds"] for row in rows) / sum(row["steps"] for row in rows),
        "final_validation": rows[-1]["validation"], "checkpoint": rows[-1]["checkpoint"],
        "checkpoint_sha256": rows[-1]["checkpoint_sha256"], "checkpoint_validated": True,
        "amp_skipped_steps": sum(row["amp_skipped_steps"] for row in rows)})
    # These mappings are benchmark-owned staging only, never shared source inputs.
    del iterator, train_loader, val_loader, sampler, train_data, val_data
    shutil.rmtree(cache)


def single(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config = {key: value for key, value in vars(args).items() if key not in ("output", "inputs")}
    summary = {"kind": "training_task", "status": "failed", "exit_code": 1,
        "started_at": timestamp(), "config": config, "environment": environment(),
        "source_hashes": source_hashes(), "epochs": []}
    summary["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    json_write(output / "started.json", summary)
    monitor = MemoryMonitor(output)
    monitor.start()
    try:
        train_task(args, output, summary)
        summary.update(status="passed", exit_code=0)
    except BaseException as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        summary["status"] = "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower() else "failed"
        json_write(output / "error.json", summary["error"])
        print(summary["error"]["traceback"], file=sys.stderr, flush=True)
    finally:
        summary["memory"] = monitor.finish()
        if torch.cuda.is_initialized():
            summary["memory"].update({"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device)})
        summary["finished_at"] = timestamp()
        summary["wall_time_seconds"] = time.perf_counter() - started
        json_write(output / "summary.json", summary)
    return summary["exit_code"]


def task_config(model, dataset="pilot", epochs=3, **kwargs):
    return {"model": model, "dataset": dataset, "epochs": epochs, "batch_size": 128,
            "seed": 17, "num_workers": 0, "pin_memory": True,
            "persistent_workers": False, "precision": "fp32", **kwargs}


def run_group(args, root, name, tasks, concurrency):
    directory = root / name
    directory.mkdir()
    started = time.perf_counter()
    began = timestamp()
    pending = list(enumerate(tasks))
    active, jobs = [], []
    monitor = MemoryMonitor(directory)
    monitor.start()
    try:
        while pending or active:
            while pending and len(active) < concurrency:
                index, config = pending.pop(0)
                job_name = f"task_{index:02d}_{config['model']}_seed{config['seed']}"
                job_dir = directory / job_name
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--single",
                    "--inputs", str(Path(args.inputs).resolve()), "--output", str(job_dir),
                    "--device", args.device, "--threads", str(args.threads)]
                for key, value in config.items():
                    flag = "--" + key.replace("_", "-")
                    command.extend([flag, str(value).lower() if isinstance(value, bool) else str(value)])
                log = (directory / f"{job_name}.log").open("x", encoding="utf-8")
                try:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                except BaseException:
                    log.close()
                    raise
                active.append((process, log, job_dir, config, time.perf_counter(), command))
            for item in active[:]:
                process, log, job_dir, config, job_start, command = item
                code = process.poll()
                if code is None:
                    continue
                log.close()
                result = {"status": "failed", "error": "Child exited without a summary"}
                result_path = job_dir / "summary.json"
                if result_path.exists():
                    try:
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        result = {"status": "failed", "error": f"Unreadable child summary: {exc}"}
                if code != 0 and result.get("status") == "passed":
                    result["status"] = "failed"
                passed = result.get("status") == "passed"
                if passed and (not result.get("checkpoint_validated") or len(result.get("epochs", [])) != config["epochs"]):
                    result.update(status="failed", error="Incomplete child checkpoint/epoch verification")
                job = {"directory": str(job_dir.relative_to(root)), "config": config,
                       "process_exit_code": code, "process_wall_seconds": time.perf_counter() - job_start,
                       "command": command, "result": result}
                json_write(directory / f"{job_dir.name}_result.json", job)
                jobs.append(job)
                active.remove(item)
            if active:
                time.sleep(0.1)
    finally:
        for process, log, *_ in active:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()
        memory = monitor.finish()
    wall = time.perf_counter() - started
    passed = len(jobs) == len(tasks) and all(job["result"]["status"] == "passed" for job in jobs)
    records = sum(job["result"].get("records", {}).get("train", 0) * len(job["result"].get("epochs", [])) for job in jobs)
    result = {"name": name, "concurrency": concurrency, "started_at": began,
        "finished_at": timestamp(), "status": "passed" if passed else "failed",
        "wall_time_seconds": wall, "total_training_samples": records,
        "samples_per_sec": records / wall if passed else None, "jobs": jobs, "memory": memory}
    json_write(directory / "summary.json", result)
    return result


def bool_arg(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("Expected true or false")
    return value.lower() == "true"


def suite(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {"kind": "training_suite", "suite": args.suite, "started_at": timestamp(),
        "status": "failed", "exit_code": 1, "environment": environment(),
        "source_hashes": source_hashes(), "input_manifest_sha256": None,
        "groups": [], "skipped": [], "comparisons": [],
        "timing_note": "Training steps synchronize CUDA; wall times include startup, input staging, validation and durable checkpoints. Telemetry sampling adds overhead."}
    json_write(root / "started.json", result)
    groups = result["groups"]

    def execute(name, configs, concurrency=1):
        group = run_group(args, root, name, configs, concurrency)
        groups.append(group)
        return group

    try:
        result["input_manifest_sha256"] = digest(Path(args.inputs) / "manifest.json")
        if args.suite == "baseline":
            for model in args.models:
                for dataset, epochs in (("pilot", 3), ("full", 5)):
                    for repeat in (1, 2):
                        execute(f"{model}_{dataset}_repeat{repeat}", [task_config(model, dataset, epochs)])
        elif args.suite == "batch":
            for model in args.models:
                oom_batch = None
                for batch in (128, 256, 512, 1024, 32, 64):
                    if oom_batch is not None and batch > oom_batch:
                        entry = {"model": model, "batch_size": batch, "status": "skipped",
                                 "reason": f"Larger than observed OOM batch {oom_batch}"}
                        result["skipped"].append(entry)
                        json_write(root / f"{model}_batch{batch}_skipped.json", entry)
                        continue
                    group = execute(f"{model}_batch{batch}", [task_config(model, batch_size=batch)])
                    child = group["jobs"][0]["result"]
                    if child["status"] == "oom":
                        oom_batch = batch if oom_batch is None else min(batch, oom_batch)
                rows = [{"batch_size": g["jobs"][0]["config"]["batch_size"],
                         "status": g["jobs"][0]["result"]["status"],
                         "samples_per_sec": g["jobs"][0]["result"].get("samples_per_sec"),
                         "macro_auroc": g["jobs"][0]["result"].get("final_validation", {}).get("macro_auroc")}
                        for g in groups if g["jobs"][0]["config"]["model"] == model]
                result["comparisons"].append({"model": model, "batch_results": rows,
                    "interpretation": "Descriptive only: changing batch size changes optimization, not merely numerical precision."})
        elif args.suite == "loader":
            for model in args.models:
                candidates = []
                for workers in (0, 2, 4, 8):
                    group = execute(f"{model}_workers{workers}_pintrue_persistentfalse",
                        [task_config(model, num_workers=workers)])
                    if group["status"] == "passed":
                        candidates.append(group)
                if not candidates:
                    result["skipped"].append({"model": model, "reason": "No successful worker configuration for second-stage loader comparison"})
                    continue
                best = max(candidates, key=lambda g: g["jobs"][0]["result"]["samples_per_sec"])
                workers = best["jobs"][0]["config"]["num_workers"]
                result["comparisons"].append({"model": model, "best_workers": workers,
                    "selection": "highest measured train samples/sec with pin=true, persistent=false"})
                for pin in (False, True):
                    for persistent in ((False, True) if workers else (False,)):
                        if pin and not persistent:
                            continue  # Already measured in the worker sweep.
                        execute(f"{model}_workers{workers}_pin{str(pin).lower()}_persistent{str(persistent).lower()}",
                            [task_config(model, num_workers=workers, pin_memory=pin, persistent_workers=persistent)])
        elif args.suite == "parallel":
            experiments = []
            if "resnet" in args.models:
                tasks = [task_config("resnet", seed=seed) for seed in (17, 29, 43)]
                sequential = execute("pilot_S1", tasks)
                for count in (2, 3):
                    concurrent = execute(f"pilot_P{count}", tasks, count)
                    experiments.append((sequential, concurrent, count))
            if {"resnet", "tcn"}.issubset(args.models):
                tasks = [task_config("resnet", "full", 2, seed=17), task_config("tcn", "full", 2, seed=29)]
                sequential = execute("full_S2", tasks)
                concurrent = execute("full_P2", tasks, 2)
                experiments.append((sequential, concurrent, 2))
            else:
                result["skipped"].append({"reason": "Full mixed-model parallel experiment requires --models resnet tcn"})
            for sequential, concurrent, count in experiments:
                valid = sequential["status"] == concurrent["status"] == "passed"
                speedup = sequential["wall_time_seconds"] / concurrent["wall_time_seconds"] if valid else None
                comparisons = []
                for job in concurrent["jobs"]:
                    reference = next(j for j in sequential["jobs"] if j["config"] == job["config"])
                    if job["result"]["status"] == reference["result"]["status"] == "passed":
                        comparisons.append({"model": job["config"]["model"], "seed": job["config"]["seed"],
                            "train_loss_delta": job["result"]["epochs"][-1]["train_loss"] - reference["result"]["epochs"][-1]["train_loss"],
                            "macro_auroc_delta": job["result"]["final_validation"]["macro_auroc"] - reference["result"]["final_validation"]["macro_auroc"],
                            "prediction_hash_equal": job["result"]["validation_probability_array_sha256"] == reference["result"]["validation_probability_array_sha256"]})
                result["comparisons"].append({"sequential": sequential["name"], "parallel": concurrent["name"],
                    "valid": valid, "speedup": speedup, "efficiency": speedup / count if valid else None,
                    "numerical_comparison": comparisons})
        elif args.suite == "precision":
            for model in args.models:
                reference = None
                for precision in PRECISIONS:
                    group = execute(f"{model}_{precision}", [task_config(model, precision=precision)])
                    if group["status"] != "passed":
                        continue
                    job = group["jobs"][0]
                    with np.load(root / job["directory"] / "validation_predictions.npz") as data:
                        predictions = data["probabilities"].copy()
                    if precision == "fp32":
                        reference = (predictions, job["result"]["final_validation"])
                    if reference is not None:
                        result["comparisons"].append({"model": model, "precision": precision,
                            "max_probability_difference_vs_fp32": float(np.max(np.abs(predictions - reference[0]))),
                            "macro_auroc_delta_vs_fp32": job["result"]["final_validation"]["macro_auroc"] - reference[1]["macro_auroc"],
                            "macro_f1_delta_vs_fp32": job["result"]["final_validation"]["macro_f1_reference_thresholds"] - reference[1]["macro_f1_reference_thresholds"],
                            "interpretation": "Exploratory fresh training comparison; deterministic FP32 remains confirmatory."})
            result["skipped"].append({"mode": "torch.compile", "reason": "Optional compiler experiment omitted; eager execution isolates precision and concurrency without compilation amortization."})
        passed = bool(groups) and all(g["status"] == "passed" for g in groups)
        result.update(status="passed" if passed else "failed", exit_code=0 if passed else 1)
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        json_write(root / "error.json", result["error"])
    result.update(finished_at=timestamp(), wall_time_seconds=time.perf_counter() - started)
    json_write(root / "summary.json", result)
    return result["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True, help="New directory; existing directories are never reused")
    parser.add_argument("--suite", choices=("baseline", "batch", "loader", "parallel", "precision"), default="baseline")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", choices=("resnet", "tcn"), default=["resnet", "tcn"])
    parser.add_argument("--model", choices=("resnet", "tcn"), default="resnet")
    parser.add_argument("--dataset", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", type=bool_arg, nargs="?", const=True, default=True)
    parser.add_argument("--persistent-workers", type=bool_arg, nargs="?", const=True, default=False)
    parser.add_argument("--precision", choices=PRECISIONS, default="fp32")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.threads < 1:
        parser.error("epochs/batch-size/threads must be positive; num-workers must be nonnegative")
    if args.persistent_workers and args.num_workers == 0:
        parser.error("persistent-workers requires num-workers > 0")
    if len(args.models) != len(set(args.models)):
        parser.error("models must not contain duplicates")
    return single(args) if args.single else suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
