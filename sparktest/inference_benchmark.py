"""Fixed-input CUDA inference, migration probes, precision and cross-host reports."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from phase1_ecg_robustness.src.models import build_model

CONDITIONS = ("clean", "independent_rms", "electrode", "covariance")
MODELS = ("resnet", "tcn")
MODES = ("deterministic_fp32", "nondeterministic_fp32", "amp_fp16", "amp_bf16", "tf32")
METRICS = ("macro_auroc", "macro_ap", "macro_f1")
SEED = 17


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    # Exclusive creation preserves previous experiments, including partial runs.
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_predictions(path, logits, probabilities, y, thresholds, ecg_id, patient_id):
    with open(path, "xb") as handle:
        np.savez_compressed(handle, logits=logits, probabilities=probabilities,
                            y=y, thresholds=thresholds, ecg_id=ecg_id, patient_id=patient_id)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256(path)


class Memory:
    """Sample host RSS/system memory; CUDA peaks are allocator measurements."""
    def __init__(self):
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.peak_rss = 0
        self.peak_system_used = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.peak_system_used = max(self.peak_system_used, psutil.virtual_memory().used)

    def run(self):
        while not self.stop.wait(0.02):
            self.sample()

    def __enter__(self):
        self.sample()
        torch.cuda.reset_peak_memory_stats()
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join()
        self.sample()

    def result(self):
        free, total = torch.cuda.mem_get_info()
        return {"peak_process_rss_bytes": self.peak_rss,
                "peak_system_used_bytes": self.peak_system_used,
                "host_sampling_interval_seconds": 0.02,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                "cuda_free_bytes_at_end": free, "cuda_total_bytes": total,
                "unified_memory_note": "Host/system/CUDA counters can overlap; do not sum them."}


def configure(mode):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    deterministic = mode == "deterministic_fp32"
    tf32 = mode == "tf32"
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    return {"seed": SEED, "precision": mode, "deterministic": deterministic,
            "cudnn_benchmark": False, "tf32": tf32,
            "amp": mode in ("amp_fp16", "amp_bf16"),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "input_dtype": "float32", "workers": 0}


def metrics(y, probabilities, thresholds):
    if y.shape != probabilities.shape or y.ndim != 2 or y.shape[1] != 5:
        raise ValueError("Predictions and labels must have matching (N, 5) shape")
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite probabilities")
    if np.any(y.sum(axis=0) == 0) or np.any(y.sum(axis=0) == len(y)):
        raise ValueError("Each of the five classes needs positive and negative labels")
    return {"macro_auroc": float(roc_auc_score(y, probabilities, average="macro")),
            "macro_ap": float(average_precision_score(y, probabilities, average="macro")),
            "macro_f1": float(f1_score(y, probabilities >= thresholds,
                                        average="macro", zero_division=0))}


def environment():
    info = {"hostname": platform.node(), "platform": platform.platform(),
            "machine": platform.machine(), "python": sys.version,
            "executable": sys.executable, "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "system_memory_bytes": psutil.virtual_memory().total,
            "interference_warning": "Other GPU jobs, including existing ComfyUI, were not stopped and may affect timings."}
    if info["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, capability=list(torch.cuda.get_device_capability(0)),
                    gpu_total_memory_bytes=p.total_memory)
    return info


def identities(inputs):
    files = ("manifest.json", "test_full.npz", "resnet.pt", "tcn.pt")
    sources = (Path(__file__), ROOT / "phase1_ecg_robustness/src/models.py")
    return {"inputs": {name: sha256(inputs / name) for name in files},
            "sources": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sources}}


def load_inputs(inputs, minimum_count):
    with np.load(inputs / "test_full.npz", allow_pickle=False) as packed:
        data = {k: packed[k] for k in (*CONDITIONS, "y", "ecg_id", "patient_id")}
    count = len(data["y"])
    if count < minimum_count:
        raise ValueError(f"Suite requires at least {minimum_count} fixed rows, got {count}")
    for condition in CONDITIONS:
        x = data[condition]
        if x.shape != (count, 12, 1000) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f"Invalid fixed input {condition}: {x.shape}, {x.dtype}")
    y = data["y"]
    if y.shape != (count, 5) or y.dtype != np.float32 or not np.isin(y, [0, 1]).all():
        raise ValueError(f"Expected binary float32 y ({count}, 5)")
    if data["ecg_id"].shape != (count,) or data["ecg_id"].dtype != np.int64:
        raise ValueError(f"Expected int64 ecg_id ({count},)")
    if data["patient_id"].shape != (count,):
        raise ValueError(f"Expected patient_id ({count},)")
    return data


def load_model(inputs, name):
    start = time.perf_counter()
    checkpoint = torch.load(inputs / f"{name}.pt", map_location="cpu", weights_only=False)
    if checkpoint["model_name"] != name:
        raise ValueError(f"Checkpoint model name does not match {name}")
    model = build_model(checkpoint["model_name"], **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    scale = float(checkpoint["scale_mv"])
    thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float32)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid scale_mv")
    if thresholds.shape != (5,) or not np.isfinite(thresholds).all():
        raise ValueError("Expected five finite checkpoint thresholds")
    model = model.cuda().eval()
    torch.cuda.synchronize()
    return model, scale, thresholds, time.perf_counter() - start


@torch.inference_mode()
def predict(model, x, scale, batch_size, mode):
    count = len(x)
    logits = np.empty((count, 5), dtype=np.float32)
    probabilities = np.empty_like(logits)
    times = dict(cpu_preparation_seconds=0.0, h2d_seconds=0.0,
                 forward_seconds=0.0, sigmoid_seconds=0.0, d2h_seconds=0.0,
                 cuda_event_h2d_seconds=0.0, cuda_event_forward_seconds=0.0,
                 cuda_event_sigmoid_seconds=0.0, cuda_event_d2h_seconds=0.0,
                 host_cuda_synchronize_wait_seconds=0.0)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    amp = mode in ("amp_fp16", "amp_bf16")
    dtype = torch.bfloat16 if mode == "amp_bf16" else torch.float16

    def timed(stage, action):
        begin = time.perf_counter()
        events[0].record()
        value = action()
        events[1].record()
        wait_start = time.perf_counter()
        torch.cuda.synchronize()
        times["host_cuda_synchronize_wait_seconds"] += time.perf_counter() - wait_start
        times[f"{stage}_seconds"] += time.perf_counter() - begin
        times[f"cuda_event_{stage}_seconds"] += events[0].elapsed_time(events[1]) / 1000
        return value

    torch.cuda.synchronize()
    start = time.perf_counter()
    with Memory() as memory:
        for begin in range(0, count, batch_size):
            end = min(begin + batch_size, count)
            preparation = time.perf_counter()
            # The fixed mV inputs are normalized once on CPU, identically on both hosts.
            cpu = torch.from_numpy(np.ascontiguousarray(x[begin:end] / np.float32(scale)))
            times["cpu_preparation_seconds"] += time.perf_counter() - preparation
            batch = timed("h2d", lambda: cpu.to("cuda"))
            with torch.autocast("cuda", dtype=dtype, enabled=amp):
                output = timed("forward", lambda: model(batch))
            if output.shape != (end - begin, 5):
                raise ValueError(f"Wrong model output shape: {tuple(output.shape)}")
            output = output.float()
            prob = timed("sigmoid", lambda: torch.sigmoid(output))
            out_cpu, prob_cpu = timed("d2h", lambda: (output.cpu(), prob.cpu()))
            logits[begin:end] = out_cpu.numpy()
            probabilities[begin:end] = prob_cpu.numpy()
            del batch, output, prob, out_cpu, prob_cpu, cpu
    elapsed = time.perf_counter() - start
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise ValueError("NaN/Inf in forward logits or probabilities")
    times.update(end_to_end_seconds=elapsed, samples_per_second=count / elapsed,
                 batches_per_second=((count + batch_size - 1) // batch_size) / elapsed,
                 forward_samples_per_second=count / times["cuda_event_forward_seconds"],
                 memory=memory.result(),
                 timing_note="Each GPU stage synchronized; end-to-end includes normalization, transfers, forward, sigmoid, host output copies and sampling. Excludes input/model loading, CUDA context init, metrics and artifact writes. Synchronize wait overlaps GPU-stage time; do not add it again.")
    return logits, probabilities, times


def backward_probe(model, x, y, scale):
    configure("deterministic_fp32")
    # Deep copy the loaded model, including buffers: no checkpoint/model contamination.
    clone = copy.deepcopy(model).train()
    before = [p.detach().clone() for p in clone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(clone.parameters(), lr=1e-4, weight_decay=1e-2)
    losses = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    with Memory() as memory:
        for begin in range(0, len(x), 16):
            bx = torch.from_numpy(np.ascontiguousarray(x[begin:begin + 16] / np.float32(scale))).cuda()
            by = torch.from_numpy(np.ascontiguousarray(y[begin:begin + 16])).cuda()
            optimizer.zero_grad(set_to_none=True)
            logits = clone(bx)
            if logits.shape != by.shape or not torch.isfinite(logits).all().item():
                raise ValueError("Invalid backward-probe forward output")
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, by)
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite backward-probe loss")
            loss.backward()
            for name, parameter in clone.named_parameters():
                if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all().item()):
                    raise ValueError(f"Missing or nonfinite gradient: {name}")
            optimizer.step()
            if any(not torch.isfinite(p).all().item() for p in clone.parameters()):
                raise ValueError("Nonfinite parameters after AdamW step")
            losses.append(float(loss.detach().cpu()))
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    changed = any(not torch.equal(old, new.detach()) for old, new in zip(
        before, (p for p in clone.parameters() if p.requires_grad)))
    if not changed:
        raise ValueError("AdamW produced no parameter change")
    return {"passed": True, "finite_losses": True, "finite_gradients": True,
            "parameters_changed": changed, "losses": losses, "batch_size": 16,
            "samples": len(x), "optimizer": "AdamW", "learning_rate": 1e-4,
            "weight_decay": 1e-2, "elapsed_seconds": elapsed,
            "samples_per_second": len(x) / elapsed, "memory": memory.result()}


def noise_effect(rows):
    return {metric: rows["electrode"][metric] - rows["independent_rms"][metric]
            for metric in METRICS}


def prediction_difference(reference, candidate):
    for key in ("y", "thresholds", "ecg_id", "patient_id"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Prediction identity mismatch: {key}")
    for key in ("logits", "probabilities"):
        if reference[key].shape != candidate[key].shape or not np.isfinite(candidate[key]).all() or not np.isfinite(reference[key]).all():
            raise ValueError(f"Invalid compared predictions: {key}")
    rm = metrics(reference["y"], reference["probabilities"], reference["thresholds"])
    cm = metrics(candidate["y"], candidate["probabilities"], candidate["thresholds"])
    diffs = {k: cm[k] - rm[k] for k in METRICS}
    agreement = float(np.mean((reference["probabilities"] >= reference["thresholds"]) ==
                             (candidate["probabilities"] >= candidate["thresholds"])))
    acceptable = abs(diffs["macro_auroc"]) < 1e-4 and abs(diffs["macro_ap"]) < 1e-4 and abs(diffs["macro_f1"]) < 1e-3 and agreement > 0.999
    return {"logits_maxabs": float(np.max(np.abs(candidate["logits"].astype(np.float64) - reference["logits"]))),
            "probabilities_maxabs": float(np.max(np.abs(candidate["probabilities"].astype(np.float64) - reference["probabilities"]))),
            "reference_metrics": rm, "candidate_metrics": cm, "metric_differences_candidate_minus_reference": diffs,
            "fixed_threshold_label_agreement": agreement, "passed": acceptable,
            "ideal": abs(diffs["macro_auroc"]) < 1e-5 and abs(diffs["macro_ap"]) < 1e-5 and abs(diffs["macro_f1"]) < 1e-4 and agreement > 0.999}


def load_prediction(output, job):
    path = output / job["directory"] / "predictions.npz"
    if sha256(path) != job["predictions_sha256"]:
        raise ValueError(f"Prediction artifact hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def compare(args, summary):
    reference = json.loads((args.reference / "summary.json").read_text(encoding="utf-8"))
    candidate = json.loads((args.candidate / "summary.json").read_text(encoding="utf-8"))
    diagnostics = []
    for label, result in (("reference", reference), ("candidate", candidate)):
        if result.get("suite") != "probe" or not result.get("passed"):
            diagnostics.append(f"{label} must be a completed, passing probe suite")
    if reference["identity"]["inputs"] != candidate["identity"]["inputs"]:
        diagnostics.append("Input manifest/test/checkpoint hashes differ")
    if summary["identity"]["inputs"] != reference["identity"]["inputs"]:
        diagnostics.append("--inputs differs from the reference input bundle")
    if reference["identity"]["sources"] != candidate["identity"]["sources"]:
        diagnostics.append("Source hashes differ; inspect before accepting cross-host consistency")
    summary["tolerances"] = {"macro_auroc": {"ideal_abs_lt": 1e-5, "acceptable_abs_lt": 1e-4},
                             "macro_ap": {"ideal_abs_lt": 1e-5, "acceptable_abs_lt": 1e-4},
                             "macro_f1": {"ideal_abs_lt": 1e-4, "acceptable_abs_lt": 1e-3},
                             "label_agreement_gt": 0.999,
                             "logits_probabilities": "Record maximum absolute differences; no absolute gate in plan",
                             "noise_effect": "Exact sign agreement for electrode minus independent-RMS for all three macro metrics"}
    summary["reference"] = {"path": str(args.reference), "summary_sha256": sha256(args.reference / "summary.json"), "environment": reference["environment"]}
    summary["candidate"] = {"path": str(args.candidate), "summary_sha256": sha256(args.candidate / "summary.json"), "environment": candidate["environment"]}
    comparisons = []
    effects = {}
    for model in MODELS:
        rmetrics, cmetrics = {}, {}
        for condition in CONDITIONS:
            try:
                rjobs = [j for j in reference["jobs"] if j["model"] == model and j["condition"] == condition]
                cjobs = [j for j in candidate["jobs"] if j["model"] == model and j["condition"] == condition]
                if len(rjobs) != 1 or len(cjobs) != 1:
                    raise ValueError("Expected exactly one probe job per model/condition")
                rjob, cjob = rjobs[0], cjobs[0]
                if rjob["config"] != cjob["config"]:
                    raise ValueError("Probe execution configurations differ")
                diff = prediction_difference(load_prediction(args.reference, rjob), load_prediction(args.candidate, cjob))
                rmetrics[condition], cmetrics[condition] = diff["reference_metrics"], diff["candidate_metrics"]
                comparisons.append(dict(model=model, condition=condition, **diff))
                if not diff["passed"]:
                    diagnostics.append(f"Numerical acceptance failed: {model}/{condition}")
            except Exception as error:
                diagnostics.append(f"{model}/{condition}: {error}")
        if all(c in rmetrics and c in cmetrics for c in ("electrode", "independent_rms")):
            reffect, ceffect = noise_effect(rmetrics), noise_effect(cmetrics)
            signs = {k: bool(np.sign(reffect[k]) == np.sign(ceffect[k])) for k in METRICS}
            effects[model] = {"reference": reffect, "candidate": ceffect, "exact_sign_agreement": signs}
            if not all(signs.values()):
                diagnostics.append(f"Noise effect direction differs: {model}")
    summary.update(comparisons=comparisons, noise_effects=effects, diagnostics=diagnostics,
                   passed=not diagnostics,
                   troubleshooting=["Check TF32/AMP, deterministic settings, PyTorch/CUDA versions, input dtype, checkpoint loading, batch size and accidental noise regeneration before interpreting a failure as hardware unsuitability."])


def execute_job(args, data, model_name, condition, count, batch_size, mode, pass_names):
    name = f"{model_name}_{condition}_n{count}_b{batch_size}_{mode}"
    directory = args.output / name
    directory.mkdir(exist_ok=False)
    job = {"directory": name, "model": model_name, "condition": condition,
           "samples": count, "batch_size": batch_size, "mode": mode,
           "started_at": utc(), "passed": False}
    job_start = time.perf_counter()
    try:
        job["config"] = dict(configure(mode), samples=count, batch_size=batch_size,
                             snr_db=10, noise_generation="none; fixed original-matrix input bundle")
        model, scale, thresholds, load_seconds = load_model(args.inputs, model_name)
        job.update(model_load_seconds=load_seconds, scale_mv=scale, thresholds=thresholds.tolist())
        x, y = data[condition][:count], data["y"][:count]
        job["passes"] = []
        for index, label in enumerate(pass_names):
            if label == "warm":
                _, _, warmup_time = predict(model, x, scale, batch_size, mode)
                job["untallied_warmup"] = warmup_time
            logits, probabilities, timing = predict(model, x, scale, batch_size, mode)
            job["passes"].append(dict(name=label, **timing))
        job["metrics"] = metrics(y, probabilities, thresholds)
        job["forward_passed"] = True
        job["predictions_sha256"] = save_predictions(directory / "predictions.npz", logits, probabilities, y, thresholds,
                                                     data["ecg_id"][:count], data["patient_id"][:count])
        job["predictions_pass"] = pass_names[-1]
        if args.suite == "probe":
            job["backward"] = backward_probe(model, x, y, scale)
            job["backward_passed"] = True
        job["passed"] = True
        del model
    except Exception as error:
        job.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        save_json(directory / "error.json", {k: job[k] for k in ("error_type", "error", "traceback")})
    finally:
        job.update(finished_at=utc(), job_wall_seconds=time.perf_counter() - job_start,
                   exit_code=0 if job["passed"] else 1)
        save_json(directory / "result.json", job)
        torch.cuda.empty_cache()
    return job


def run_gpu(args, summary):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is prohibited")
    configure("deterministic_fp32")
    init_start = time.perf_counter()
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    summary["cuda_initialization_seconds"] = time.perf_counter() - init_start
    input_start = time.perf_counter()
    minimum_count = {"probe": 100, "precision": 1000, "inference": 2158}[args.suite]
    data = load_inputs(args.inputs, minimum_count)
    summary["input_loading_validation_seconds"] = time.perf_counter() - input_start
    summary["cold_definition"] = "First inference pass after loading a fresh model for the configuration; CUDA context already initialized. No device/cache reset. Warm follows one additional full untallied warm-up pass; steady follows warm."
    summary["jobs"] = []
    modes = MODES if args.suite == "precision" else ("deterministic_fp32",)
    sizes = (100,) if args.suite == "probe" else ((1000, 2158) if args.suite == "inference" else (1000,))
    batches = (128, 256, 512, 1024) if args.suite == "inference" else ((16,) if args.suite == "probe" else (128,))
    summary["skipped_modes"] = []
    for mode in modes:
        if mode == "amp_bf16" and not torch.cuda.is_bf16_supported():
            summary["skipped_modes"].append({"mode": mode, "reason": "CUDA device/PyTorch reports BF16 unsupported"})
            continue
        if mode == "tf32" and torch.cuda.get_device_capability(0)[0] < 8:
            summary["skipped_modes"].append({"mode": mode, "reason": "TF32 requires compute capability >= 8"})
            continue
        for model in MODELS:
            for condition in CONDITIONS:
                for count in sizes:
                    for batch in batches:
                        names = ("probe",) if args.suite == "probe" else ("cold", "warm", "steady")
                        job = execute_job(args, data, model, condition, count, batch, mode, names)
                        summary["jobs"].append(job)
                        print(f"{job['directory']}: {'passed' if job['passed'] else 'FAILED'}", flush=True)
    summary["passed"] = all(j["passed"] for j in summary["jobs"])
    summary["noise_effects"] = {}
    if args.suite in ("probe", "precision"):
        for model in MODELS:
            for mode in modes:
                rows = {j["condition"]: j["metrics"] for j in summary["jobs"]
                        if j["model"] == model and j["mode"] == mode and j["passed"]}
                if "electrode" in rows and "independent_rms" in rows:
                    summary["noise_effects"][f"{model}/{mode}"] = noise_effect(rows)
    if args.suite == "precision":
        summary["precision_comparisons"] = []
        summary["precision_effect_signs"] = {}
        for model in MODELS:
            base_effect = summary["noise_effects"].get(f"{model}/deterministic_fp32")
            for mode in modes[1:]:
                effect = summary["noise_effects"].get(f"{model}/{mode}")
                if base_effect is not None and effect is not None:
                    summary["precision_effect_signs"][f"{model}/{mode}"] = {
                        k: bool(np.sign(base_effect[k]) == np.sign(effect[k])) for k in METRICS}
                for condition in CONDITIONS:
                    jobs = [j for j in summary["jobs"] if j["model"] == model and j["condition"] == condition and j["passed"]]
                    base = next((j for j in jobs if j["mode"] == "deterministic_fp32"), None)
                    other = next((j for j in jobs if j["mode"] == mode), None)
                    if base is not None and other is not None:
                        diff = prediction_difference(load_prediction(args.output, base), load_prediction(args.output, other))
                        summary["precision_comparisons"].append(dict(model=model, condition=condition, mode=mode, **diff))
        summary["precision_policy"] = "Passing suite means execution succeeded, not that exploratory precision is scientifically equivalent. Comparison passed flags apply numerical tolerances; exact effect signs are reported separately. Confirmatory results must use deterministic FP32 regardless."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--suite", required=True, choices=("probe", "inference", "precision", "compare"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    if args.suite == "compare" and (args.reference is None or args.candidate is None):
        parser.error("compare requires --reference and --candidate probe directories")
    if args.suite != "compare" and (args.reference is not None or args.candidate is not None):
        parser.error("--reference and --candidate apply only to compare")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    summary = {"schema_version": 1, "suite": args.suite, "started_at": utc(),
               "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "passed": False}
    try:
        summary["environment"] = environment()
        summary["identity"] = identities(args.inputs)
        summary["input_manifest"] = json.loads((args.inputs / "manifest.json").read_text(encoding="utf-8"))
        if args.suite == "compare":
            compare(args, summary)
        else:
            run_gpu(args, summary)
    except Exception as error:
        summary.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        save_json(args.output / "error.json", {k: summary[k] for k in ("error_type", "error", "traceback")})
    summary.update(finished_at=utc(), wall_seconds=time.perf_counter() - start,
                   exit_code=0 if summary["passed"] else 1)
    save_json(args.output / "summary.json", summary)
    print(json.dumps({"summary": str(args.output / "summary.json"), "passed": summary["passed"]}), flush=True)
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
