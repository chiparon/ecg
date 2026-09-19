"""Plan D: phase1 CPU noise/statistics, exploratory CUDA, and real ResNet pipeline.

Run from the workspace root with --inputs DIR --output NEW_DIR. Optional
--predictions DIR accepts aligned first1000/full-test NPZ predictions, never probe100.
All CPU worker-count comparisons use the exact phase1 float64 numerical routines.
The CUDA translation is explicitly exploratory: different RNG, same band and
rank-aware covariance construction; it is not a bitwise replay of phase1.
"""
from __future__ import annotations

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# One numerical thread per worker makes the worker sweep interpretable.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import socket
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import torch
from scipy.signal import welch

from phase1_ecg_robustness.src.covariance_matching import empirical_covariance, match_covariance
from phase1_ecg_robustness.src.evaluate import classification_metrics
from phase1_ecg_robustness.src.lead_matrix import get_lead_matrix, matrix_provenance
from phase1_ecg_robustness.src.models import build_model
from phase1_ecg_robustness.src.noise_generators import _source_channels, noise_diagnostics, scale_to_snr
from phase1_ecg_robustness.src.statistics import derived_seed

COUNT = 1000
FS = 100.0
SNR = 10.0
BAND = (0.5, 40.0)
WORKERS = (1, 2, 4, 8, 16)
NOISE_OPS = ("independent", "electrode", "covariance_matched")
_STATE = {}


def utc():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def ordered_update(digest, index, value):
    array = np.ascontiguousarray(value)
    digest.update(json.dumps([index, array.dtype.str, array.shape]).encode())
    digest.update(array.tobytes())


class Resources:
    """Sample benchmark process tree and system, not just Python allocations."""

    def __init__(self):
        self.stop = threading.Event()
        self.tree_rss_peak = 0
        self.system_used_peak = 0
        self.available_min = psutil.virtual_memory().available
        self.system_cpu = []
        self.tree_cpu_peak = 0.0
        self.tree_cpu_seconds = 0.0
        self.previous = {}
        self.last_time = time.perf_counter()
        self.error = None
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def sample(self):
        now = time.perf_counter()
        elapsed = now - self.last_time
        processes = [psutil.Process()]
        processes += processes[0].children(recursive=True)
        rss, delta = 0, 0.0
        current = {}
        for process in processes:
            try:
                rss += process.memory_info().rss
                cpu = process.cpu_times()
                seconds = cpu.user + cpu.system
                current[process.pid] = seconds
                delta += max(0.0, seconds - self.previous.get(process.pid, seconds))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        memory = psutil.virtual_memory()
        self.tree_rss_peak = max(self.tree_rss_peak, rss)
        self.system_used_peak = max(self.system_used_peak, memory.total - memory.available)
        self.available_min = min(self.available_min, memory.available)
        self.system_cpu.append(psutil.cpu_percent())
        self.tree_cpu_peak = max(self.tree_cpu_peak, 100 * delta / max(elapsed, 1e-9))
        self.tree_cpu_seconds += delta
        self.previous, self.last_time = current, now

    def watch(self):
        try:
            while not self.stop.wait(0.05):
                self.sample()
        except Exception:
            self.error = traceback.format_exc()

    def __enter__(self):
        self.sample()
        self.started = time.perf_counter()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()
        self.elapsed = time.perf_counter() - self.started

    def result(self):
        return dict(process_tree_rss_peak_bytes=self.tree_rss_peak,
                    system_used_peak_bytes=self.system_used_peak,
                    system_available_min_bytes=self.available_min,
                    process_tree_cpu_peak_percent=self.tree_cpu_peak,
                    process_tree_cpu_mean_percent=100 * self.tree_cpu_seconds / self.elapsed,
                    system_cpu_peak_percent=max(self.system_cpu, default=0),
                    system_cpu_mean_percent=float(np.mean(self.system_cpu)),
                    sampling_interval_seconds=0.05, sampler_error=self.error,
                    note="RSS sums may double-count shared pages; sampled CPU misses short-lived processes. System memory includes other workloads.")


def initialize(prepared, seed, prediction=None):
    global _STATE
    torch.set_num_threads(1)
    folder = Path(prepared)
    _STATE = {"x": np.load(folder / "clean.npy", mmap_mode="r"),
              "noise": np.load(folder / "scaling_source.npy", mmap_mode="r"),
              "seed": seed, "a": get_lead_matrix()}
    if prediction:
        with np.load(prediction, allow_pickle=False) as archive:
            _STATE.update({key: archive[key] for key in archive.files})
        patients, inverse = np.unique(_STATE["patient_id"], return_inverse=True)
        _STATE["groups"] = [np.flatnonzero(inverse == i) for i in range(len(patients))]
        _, loss = classification_metrics(_STATE["y"], _STATE["p"], _STATE["thresholds"])
        _STATE["patient_loss"] = np.array([loss[group].mean() for group in _STATE["groups"]])


def phase1_noise(index, operation):
    x = _STATE["x"][index]
    streams = np.random.SeedSequence(derived_seed(_STATE["seed"], "noise", index)).spawn(3)
    if operation == "independent":
        source = _source_channels(12, 1000, FS, streams[0], "bandpass", None, BAND)
        return scale_to_snr(source, x, SNR)
    source = _source_channels(9, 1000, FS, streams[1], "bandpass", None, BAND)
    electrode = scale_to_snr(_STATE["a"] @ source, x, SNR)
    if operation == "electrode":
        return electrode
    fresh = _source_channels(12, 1000, FS, streams[2], "bandpass", None, BAND)
    return match_covariance(empirical_covariance(electrode), fresh)[0]


def work(task):
    operation, index = task
    started = time.perf_counter()
    if operation in NOISE_OPS:
        value = phase1_noise(index, operation)
    elif operation == "snr_scaling":
        value = scale_to_snr(_STATE["noise"][index], _STATE["x"][index], SNR)
    elif operation == "covariance_12x12":
        value = empirical_covariance(_STATE["noise"][index])
    elif operation == "welch_psd":
        frequency, psd = welch(_STATE["noise"][index], fs=FS, nperseg=256,
                               noverlap=128, detrend="constant", scaling="density", axis=1)
        value = np.vstack((frequency, psd))
    elif operation == "patient_bootstrap":
        groups = _STATE["groups"]
        rng = np.random.default_rng(derived_seed(_STATE["seed"], "patient_bootstrap", index))
        selected = rng.integers(0, len(groups), size=len(groups))
        # Repeat every record of each sampled patient, including repeated clusters.
        indices = np.concatenate([groups[i] for i in selected])
        metrics, _ = classification_metrics(_STATE["y"][indices], _STATE["p"][indices], _STATE["thresholds"])
        value = np.array([metrics[key] for key in ("macro_auroc", "macro_ap", "macro_f1")])
        value = np.append(value, _STATE["patient_loss"][selected].mean())
    else:
        raise ValueError(operation)
    return index, value, time.perf_counter() - started


def cpu_sweep(operation, workers, prepared, seed, prediction=None, retain=False):
    digest = hashlib.sha256()
    values = []
    kernel_seconds = 0.0
    started = time.perf_counter()
    if workers == 1:
        initialize(prepared, seed, prediction)
        results = map(work, ((operation, i) for i in range(COUNT)))
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                       initializer=initialize, initargs=(str(prepared), seed, prediction))
        results = executor.map(work, ((operation, i) for i in range(COUNT)), chunksize=8)
    try:
        for expected, (index, value, duration) in enumerate(results):
            if index != expected or not np.isfinite(value).all():
                raise ValueError("Nonfinite or unordered worker result")
            ordered_update(digest, index, value)
            kernel_seconds += duration
            if retain:
                values.append(value)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    seconds = time.perf_counter() - started
    result = dict(workers=workers, items=COUNT, wall_seconds=seconds,
                  samples_per_second=COUNT / seconds, ordered_sha256=digest.hexdigest(),
                  sum_worker_compute_seconds=kernel_seconds,
                  timing_scope="Includes initializer, spawn/imports, IPC, ordered hashing, pool shutdown; excludes input preparation and JSON writing",
                  units="bootstrap replicates/s over fixed 1000 ECGs" if operation == "patient_bootstrap" else "records/s")
    return result, np.stack(values) if retain else None


def prediction_archive(directory, prepared, inputs, checkpoint):
    """Accept phase1 or benchmark aliases, with exact ID/label/patient alignment."""
    if directory is None:
        return None, {"status": "blocked", "reason": "No --predictions directory supplied; no synthetic probabilities generated"}
    rejected = []
    for path in sorted(Path(directory).rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                def take(*names):
                    for name in names:
                        if name in archive.files:
                            return archive[name]
                    raise ValueError(f"missing any of {names}")
                p = take("p", "probabilities")
                ids = take("ecg_id", "ids", "ecg_ids")
                patients = take("patient_id", "patient_ids")
                y = take("y", "labels")
                thresholds = archive["thresholds"] if "thresholds" in archive.files else np.asarray(checkpoint["thresholds"])
            if len(p) < COUNT:
                raise ValueError(f"Only {len(p)} predictions: first1000 required (probe100 is not sufficient)")
            for label, actual, wanted in (("ecg_id", ids, inputs["ecg_id"]),
                                           ("patient_id", patients, inputs["patient_id"]), ("y", y, inputs["y"])):
                if not np.array_equal(actual[:COUNT], wanted):
                    raise ValueError(f"{label} differs from ordered fixed first1000")
            p = p[:COUNT]
            if p.shape != (COUNT, 5) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError("invalid probabilities")
            if thresholds.shape != (5,) or not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
                raise ValueError("invalid thresholds")
            target = prepared / "bootstrap_predictions.npz"
            np.savez(target, p=p, y=inputs["y"], patient_id=inputs["patient_id"], thresholds=thresholds)
            return str(target), dict(status="ready", source=str(path.resolve()), source_sha256=file_hash(path),
                                    rejected=rejected, selection="First lexicographically sorted compatible NPZ; source/condition not pooled",
                                    patients=int(len(np.unique(inputs["patient_id"]))), records=COUNT,
                                    semantics="1000 patient-cluster bootstrap replicates; macro metrics on all ECGs of sampled patients; BCE first averaged within each patient then equally across sampled patients. No record bootstrap; fixed thresholds.",
                                    metrics=["macro_auroc", "macro_ap", "macro_f1", "equal_patient_bce"])
        except Exception as error:
            rejected.append({"path": str(path), "reason": str(error)})
    return None, dict(status="blocked", reason="No compatible fixed first1000 prediction archive", rejected=rejected)


def cuda_noise(x, operation, seed):
    """Exploratory float64 port of the read, hashed phase1 routines.

    Native torch RNG is not PCG64/SeedSequence. Float64 preserves the phase1
    eigenvalue cutoff; no jitter, no full-rank replacement of the rank-8 target.
    Per-record seeds remain independent of scheduling and batch size.
    """
    a = torch.as_tensor(get_lead_matrix(), device="cuda", dtype=torch.float64)
    frequency = torch.fft.rfftfreq(1000, d=1 / FS, device="cuda")
    mask = (frequency >= BAND[0]) & (frequency <= BAND[1]) & (frequency > 0)
    output, targets = [], []
    for index in range(COUNT):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(derived_seed(seed, "noise", index))

        def source(channels):
            raw = torch.randn((channels, 1000), dtype=torch.float64, device="cuda", generator=generator)
            spectrum = torch.fft.rfft(raw)
            spectrum[:, ~mask] = 0
            value = torch.fft.irfft(spectrum, n=1000)
            return value - value.mean(dim=1, keepdim=True)

        def scale(value):
            centered = value - value.mean(dim=1, keepdim=True)
            return centered * torch.sqrt(x[index].square().mean() * 0.1 / centered.square().mean())

        if operation == "independent":
            result = scale(source(12))
            target = None
        else:
            electrode = scale(a @ source(9))
            target = electrode @ electrode.T / 1000
            if operation == "electrode":
                result = electrode
            else:
                fresh = source(12)
                eigenvalues, vectors = torch.linalg.eigh((target + target.T) / 2)
                tolerance = 128 * 12 * torch.finfo(torch.float64).eps * eigenvalues.abs().max()
                if bool((eigenvalues < -tolerance).any()):
                    raise ValueError("CUDA covariance target has negative eigenvalue")
                keep = eigenvalues > tolerance
                factor = vectors[:, keep] * eigenvalues[keep].sqrt()
                rank = factor.shape[1]
                temporal = fresh[:rank] - fresh[:rank].mean(dim=1, keepdim=True)
                covariance = temporal @ temporal.T / 1000
                values, directions = torch.linalg.eigh(covariance)
                if bool((values <= 128 * rank * torch.finfo(torch.float64).eps * values[-1]).any()):
                    raise ValueError("CUDA fresh source is rank deficient")
                whitened = (directions / values.sqrt()).T @ temporal
                result = factor @ whitened
                result -= result.mean(dim=1, keepdim=True)
        output.append(result)
        if target is not None:
            targets.append(target)
    return torch.stack(output), torch.stack(targets) if targets else None


def gpu_benchmark(clean, operation, seed, reference):
    torch.cuda.synchronize()
    upload_started = time.perf_counter()
    x = torch.as_tensor(clean, dtype=torch.float64, device="cuda")
    torch.cuda.synchronize()
    upload_seconds = time.perf_counter() - upload_started
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start_event.record()
    generated, targets = cuda_noise(x, operation, seed)
    end_event.record()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    download_started = time.perf_counter()
    values = generated.cpu().numpy()
    target_values = targets.cpu().numpy() if targets is not None else None
    torch.cuda.synchronize()
    download_seconds = time.perf_counter() - download_started
    # Validity work is deliberately outside the reported generation timing.
    validation_started = time.perf_counter()
    digest = hashlib.sha256()
    snr_errors, covariance_errors, rms_errors, psd_ratios, means, leakage = [], [], [], [], [], []
    pooled_cpu, pooled_gpu = None, None
    for index, value in enumerate(values):
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite GPU noise")
        ordered_update(digest, index, value)
        actual = noise_diagnostics(value, clean[index], FS, BAND)
        original = noise_diagnostics(reference[index], clean[index], FS, BAND)
        snr_errors.append(abs(actual["actual_snr_db"] - SNR))
        means.append(float(np.max(np.abs(actual["per_lead_mean"]))))
        rms_errors.append(abs(float(np.mean(value ** 2)) / (float(np.mean(clean[index].astype(np.float64) ** 2)) * 0.1) - 1))
        psd_ratios.append(float(np.mean(actual["welch_total_power"]) / actual["noise_power_mv2"]))
        spectrum = np.fft.rfft(value)
        frequency = np.fft.rfftfreq(1000, d=1 / FS)
        outside = (frequency < BAND[0]) | (frequency > BAND[1])
        leakage.append(float(np.sum(abs(spectrum[:, outside]) ** 2) / np.sum(abs(spectrum) ** 2)))
        if target_values is not None:
            covariance_errors.append(float(np.linalg.norm(actual["covariance"] - target_values[index]) / np.linalg.norm(target_values[index])))
        # Pool unit-power PSDs: waveform draws and exact individual PSDs must differ.
        c = original["psd"].mean(axis=0) / original["noise_power_mv2"]
        g = actual["psd"].mean(axis=0) / actual["noise_power_mv2"]
        pooled_cpu = c.copy() if pooled_cpu is None else pooled_cpu + c
        pooled_gpu = g.copy() if pooled_gpu is None else pooled_gpu + g
    psd_relative_l1 = float(np.sum(abs(pooled_cpu - pooled_gpu)) / np.sum(abs(pooled_cpu)))
    passed = (max(snr_errors) < 1e-8 and max(rms_errors) < 1e-8 and max(means) < 1e-10
              and max(leakage) < 1e-15 and max(covariance_errors, default=0) < 1e-8
              and min(psd_ratios) > 0.5 and max(psd_ratios) < 1.5 and psd_relative_l1 < 0.15)
    validation = dict(passed=passed, snr_max_abs_error_db=max(snr_errors),
                      total_power_max_relative_error=max(rms_errors), per_lead_mean_max_abs_mv=max(means),
                      covariance_target_max_relative_frobenius=max(covariance_errors) if covariance_errors else None,
                      out_of_band_fft_power_fraction_max=max(leakage),
                      welch_integral_to_rms_power_ratio_min=min(psd_ratios),
                      welch_integral_to_rms_power_ratio_max=max(psd_ratios),
                      pooled_unit_power_welch_cpu_gpu_relative_l1=psd_relative_l1,
                      limits={"snr_abs_db": 1e-8, "power_relative": 1e-8, "mean_abs_mv": 1e-10,
                              "covariance_relative": 1e-8, "fft_leakage": 1e-15, "welch_power_ratio": [0.5, 1.5],
                              "pooled_psd_relative_l1": 0.15},
                      note="Distributional spectral check, not pointwise PSD equivalence. Independent noise has no prescribed cross-lead covariance target; electrode/covariance are compared with their GPU electrode target.",
                      validation_seconds=time.perf_counter() - validation_started)
    return dict(status="ok" if passed else "invalid_numerics", operation=operation,
                classification="exploratory_non_equivalent_rng", dtype="float64 noise; FP32 is reserved for model inference",
                samples=COUNT, wall_seconds=seconds, samples_per_second=COUNT / seconds,
                cuda_event_seconds=start_event.elapsed_time(end_event) / 1000,
                upload_seconds=upload_seconds, download_seconds=download_seconds,
                end_to_end_generation_seconds=upload_seconds + seconds + download_seconds,
                end_to_end_samples_per_second=COUNT / (upload_seconds + seconds + download_seconds),
                ordered_sha256=digest.hexdigest(), hash_comparison="Not comparable to CPU PCG64 results",
                warmup="None; cold operation including first-use FFT/eigensolver effects",
                numerical_validation=validation)


def pipeline(prepared, data, checkpoint, operation, workers, seed, batch_size, output):
    clean = data["clean"]
    model = build_model(checkpoint["model_name"], **checkpoint["model_kwargs"]).cuda().eval()
    model.load_state_dict(checkpoint["model_state"])
    scale = float(checkpoint["scale_mv"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Checkpoint scale_mv must be finite and positive")
    # Real warmup, excluded from pipeline timing.
    with torch.inference_mode():
        model(torch.from_numpy(np.array(clean[:batch_size], copy=True)).cuda() / scale)
    torch.cuda.synchronize()
    generated_hash, logits_hash = hashlib.sha256(), hashlib.sha256()
    predictions, logits_saved, pending, kernel_events = [], [], [], []
    cpu_work = 0.0
    started = time.perf_counter()

    def consume(batch):
        indices = [item[0] for item in batch]
        # Pinned staging supports actual CPU-production/GPU-consumption overlap.
        staged = torch.empty((len(batch), 12, 1000), dtype=torch.float32, pin_memory=True)
        for row, (index, noise, _) in enumerate(batch):
            ordered_update(generated_hash, index, noise)
            np.copyto(staged[row].numpy(), clean[index].astype(np.float64) + noise, casting="unsafe")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            tensor = staged.cuda(non_blocking=True) / scale
            logits = model(tensor)
            probability = torch.sigmoid(logits)
        end.record()
        logits_cpu = logits.cpu().numpy()
        probability_cpu = probability.cpu().numpy()
        if logits_cpu.shape != (len(batch), 5) or not np.isfinite(logits_cpu).all():
            raise ValueError("Invalid pipeline logits")
        for index, row in zip(indices, logits_cpu):
            ordered_update(logits_hash, index, row)
        predictions.append(probability_cpu)
        logits_saved.append(logits_cpu)
        kernel_events.append((start, end))

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize, initargs=(str(prepared), seed, None)) as executor:
        # Bounded in-flight records avoid accidentally pre-generating all records.
        futures = {}
        window = max(workers * 2, batch_size * 2)
        for index in range(min(window, COUNT)):
            futures[index] = executor.submit(work, (operation, index))
        for index in range(COUNT):
            item = futures.pop(index).result()
            cpu_work += item[2]
            pending.append(item)
            upcoming = index + window
            if upcoming < COUNT:
                futures[upcoming] = executor.submit(work, (operation, upcoming))
            if len(pending) == batch_size or index == COUNT - 1:
                consume(pending)
                pending = []
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    np.savez(output / "predictions.npz", p=np.concatenate(predictions), logits=np.concatenate(logits_saved),
             thresholds=np.asarray(checkpoint["thresholds"]), y=data["y"],
             ecg_id=data["ecg_id"], patient_id=data["patient_id"])
    return dict(workers=workers, batch_size=batch_size, samples=COUNT, wall_seconds=seconds,
                samples_per_second=COUNT / seconds,
                gpu_batch_event_seconds=sum(start.elapsed_time(end) for start, end in kernel_events) / 1000,
                sum_worker_compute_seconds=cpu_work, noise_ordered_sha256=generated_hash.hexdigest(),
                logits_ordered_sha256=logits_hash.hexdigest(),
                timing_scope="CPU spawn, generation, bounded IPC, FP32 cast/normalization, H2D, ResNet inference, D2H, ordered hashing and shutdown; model load/warmup and NPZ writing excluded",
                comparison_note="Pipeline includes real inference; generation-only CPU/GPU throughput is not an equal workload comparison")


def run(args, output):
    summary = dict(started_utc=utc(), status="running", jobs=[], seed=args.seed,
                   config=dict(records=COUNT, workers=list(WORKERS), fs_hz=FS, snr_db=SNR, band_hz=list(BAND),
                               noise_kind="bandpass", active_electrodes="all original nine", tf32=False,
                               deterministic=True, cudnn_benchmark=False, model_dtype="float32", cpu_noise_dtype="float64",
                               blas_threads_per_process=1, pipeline_batch_size=args.batch_size),
                   interference="Existing ComfyUI/other workloads are not stopped; system memory/CPU and GPU peaks can be affected.")
    failed = False
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    summary["environment"] = dict(host=socket.gethostname(), platform=platform.platform(), python=sys.version,
                                  torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                                  cpu_logical=psutil.cpu_count(), cpu_physical=psutil.cpu_count(logical=False),
                                  system_memory_bytes=psutil.virtual_memory().total,
                                  cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"])
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for plan D CPU/GPU comparison and real pipeline")
        summary["environment"].update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)))
        inputs = Path(args.inputs).resolve()
        summary["hashes"] = {"manifest_sha256": file_hash(inputs / "manifest.json"),
                              "test_full_sha256": file_hash(inputs / "test_full.npz"),
                              "resnet_checkpoint_sha256": file_hash(inputs / "resnet.pt"),
                              "source": {str(path.relative_to(ROOT)): file_hash(path) for path in [Path(__file__).resolve(),
                                  *[ROOT / "phase1_ecg_robustness" / "src" / name for name in
                                    ("noise_generators.py", "covariance_matching.py", "lead_matrix.py", "statistics.py", "evaluate.py", "models.py")]]}}
        summary["lead_matrix"] = matrix_provenance()
        with np.load(inputs / "test_full.npz", allow_pickle=False) as archive:
            data = {key: archive[key][:COUNT].copy() for key in ("clean", "y", "ecg_id", "patient_id")}
        clean = data["clean"]
        if clean.shape != (COUNT, 12, 1000) or clean.dtype != np.float32 or not np.isfinite(clean).all():
            raise ValueError("Expected finite float32 first1000 clean mV records with shape (1000,12,1000)")
        if data["y"].shape != (COUNT, 5) or not np.isin(data["y"], [0, 1]).all():
            raise ValueError("Expected binary labels with shape (1000,5)")
        if data["ecg_id"].shape != (COUNT,) or data["patient_id"].shape != (COUNT,) or len(np.unique(data["ecg_id"])) != COUNT:
            raise ValueError("Expected one patient ID and unique ECG ID per record")
        checkpoint = torch.load(inputs / "resnet.pt", map_location="cpu", weights_only=False)
        prepared = output / "prepared"
        prepared.mkdir()
        np.save(prepared / "clean.npy", clean)
        # Common, precomputed unscaled lead-space source for SNR/covariance/PSD microbenchmarks.
        source = np.lib.format.open_memmap(prepared / "scaling_source.npy", mode="w+", dtype=np.float64, shape=clean.shape)
        preparation_started = time.perf_counter()
        for index in range(COUNT):
            stream = np.random.SeedSequence(derived_seed(args.seed, "noise", index)).spawn(3)[0]
            source[index] = _source_channels(12, 1000, FS, stream, "bandpass", None, BAND)
        source.flush()
        del source
        summary["preparation_seconds"] = time.perf_counter() - preparation_started
        summary["operation_semantics"] = dict(independent="12 phase1 PCG64 bandpass streams + global SNR scaling (not independent-RMS)",
            electrode="9 phase1 PCG64 bandpass streams + original A + global SNR scaling",
            covariance_matched="electrode generation + fresh 12 lead streams + original empirical rank-aware covariance matching",
            snr_scaling="original scale_to_snr on precomputed unscaled independent bandpass source",
            covariance_12x12="original empirical_covariance ddof=0 on common unscaled source",
            welch_psd="phase1 diagnostic Welch settings on common unscaled source; frequency included in ordered hash")
        prediction, summary["bootstrap"] = prediction_archive(args.predictions, prepared, data, checkpoint)
        json_write(output / "run_config.json", {key: value for key, value in summary.items() if key != "jobs"})

        def job(name, function, gpu=False):
            nonlocal failed
            folder = output / name
            folder.mkdir()
            entry = dict(name=name, started_utc=utc(), status="running")
            result = None
            try:
                if gpu:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                with Resources() as monitor:
                    result = function(folder)
                    if gpu:
                        torch.cuda.synchronize()
                entry.update(result[0] if isinstance(result, tuple) else result)
                entry["resources"] = monitor.result()
                if entry["status"] == "running":
                    entry["status"] = "ok"
                if entry["status"] != "ok":
                    failed = True
            except Exception:
                failed = True
                entry.update(status="error", error=traceback.format_exc())
                with (folder / "error.txt").open("x", encoding="utf-8") as handle:
                    handle.write(entry["error"])
                    handle.flush()
                    os.fsync(handle.fileno())
                if "monitor" in locals() and hasattr(monitor, "elapsed"):
                    entry["resources"] = monitor.result()
            if gpu:
                entry["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                entry["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            entry["ended_utc"] = utc()
            entry["exit_code"] = 0 if entry["status"] == "ok" else 1
            json_write(folder / "result.json", entry)
            with (output / "jobs.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            summary["jobs"].append(entry)
            return entry, result[1] if isinstance(result, tuple) else None

        hashes, timings = {}, {}
        costly, costly_seconds, reference = None, -1.0, None
        operations = [*NOISE_OPS, "snr_scaling", "covariance_12x12", "welch_psd"]
        if prediction:
            operations.append("patient_bootstrap")
        for operation in operations:
            for workers in WORKERS:
                retain = workers == 1 and (operation in NOISE_OPS or operation == "patient_bootstrap")

                def cpu_job(folder, operation=operation, workers=workers, retain=retain):
                    result, values = cpu_sweep(operation, workers, prepared, args.seed,
                                                prediction if operation == "patient_bootstrap" else None, retain)
                    result["operation"] = operation
                    if workers == 1:
                        hashes[operation] = result["ordered_sha256"]
                    result["hash_matches_serial"] = result["ordered_sha256"] == hashes.get(operation)
                    if not result["hash_matches_serial"]:
                        result["status"] = "hash_mismatch"
                    if operation == "patient_bootstrap" and values is not None:
                        np.save(folder / "bootstrap_metrics.npy", values)
                        result["ci95_percentiles"] = np.quantile(values, [0.025, 0.975], axis=0).tolist()
                    return result, values

                entry, values = job(f"cpu_{operation}_w{workers}", cpu_job)
                if entry["status"] == "ok":
                    timings[(operation, workers)] = entry["wall_seconds"]
                    if workers == 1 and operation in NOISE_OPS and entry["wall_seconds"] > costly_seconds:
                        costly, costly_seconds, reference = operation, entry["wall_seconds"], values
        summary["cpu_hashes_match_all_workers"] = all(
            entry.get("hash_matches_serial", False) for entry in summary["jobs"] if entry["name"].startswith("cpu_"))
        if costly is None:
            raise RuntimeError("No successful serial noise benchmark; cannot select most costly operation")
        mp_options = [(duration, workers) for (operation, workers), duration in timings.items() if operation == costly and workers > 1]
        if not mp_options:
            raise RuntimeError("No successful multiprocessing noise benchmark; cannot select pipeline workers")
        best_mp = min(mp_options)[1]
        summary["selected_noise_comparison"] = dict(operation=costly, selection="Largest measured serial end-to-end generation wall time", best_multiprocess_workers=best_mp,
            cpu_serial_seconds=timings[(costly, 1)], cpu_best_multiprocess_seconds=timings[(costly, best_mp)])
        job("gpu_torch", lambda _: gpu_benchmark(clean, costly, args.seed, reference), gpu=True)

        def pipeline_job(folder):
            result = pipeline(prepared, data, checkpoint, costly, best_mp, args.seed, args.batch_size, folder)
            result["hash_matches_cpu_noise"] = result["noise_ordered_sha256"] == hashes[costly]
            if not result["hash_matches_cpu_noise"]:
                result["status"] = "hash_mismatch"
            return result

        job("cpu_gpu_pipeline", pipeline_job, gpu=True)
        summary["status"] = "failed" if failed else "completed_with_bootstrap_blocked" if not prediction else "completed"
        summary["exit_code"] = 1 if failed else 0
    except Exception:
        summary.update(status="failed", exit_code=1, error=traceback.format_exc())
        with (output / "error.txt").open("x", encoding="utf-8") as handle:
            handle.write(summary["error"])
            handle.flush()
            os.fsync(handle.fileno())
    summary["ended_utc"] = utc()
    json_write(output / "summary.json", summary)
    return summary["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.seed < 0 or args.seed >= 2**32 or args.batch_size <= 0:
        parser.error("seed must be in [0,2**32), batch-size must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    raise SystemExit(run(args, args.output.resolve()))


if __name__ == "__main__":
    mp.freeze_support()
    main()
