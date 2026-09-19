"""Gate 0 only: inspect public text, metadata and device inventory; never import a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

from overnight_supplements_v2.shared.common import ROOT, OUT, file_info, now, read_json, write_json

COMMIT = "bbb61982741d466bfa81669edc0e17f1971980af"
BASE = f"https://raw.githubusercontent.com/timeseriesAI/tsai/{COMMIT}"


def remote_text(relative):
    url = f"{BASE}/{relative}"
    request = urllib.request.Request(url, headers={"User-Agent": "ecg-read-only-inventory/2"})
    with urllib.request.urlopen(request, timeout=45) as response:
        content = response.read()
    return content.decode("utf-8"), {"path": url, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "schema": "UTF-8 text; inspected only", "source": f"Public tsai repository at immutable commit {COMMIT}"}


def main():
    source, source_info = remote_text("tsai/models/InceptionTime.py")
    license_text, license_info = remote_text("LICENSE")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("Pinned public license differs from inspected Apache-2.0")
    for expected in ("class InceptionTime", "nf=32", "depth=6", "ks=40", "k - 1", "d % 3 == 2", "GAP1d(1)"):
        if expected not in source:
            raise ValueError(f"Pinned source topology differs: {expected}")
    original = read_json("phase2/results/test_inputs/full/manifest.json")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=True)
    paths = ["phase1_ecg_robustness/data/processed/signals.npy", "phase1_ecg_robustness/data/processed/labels.npy", "phase1_ecg_robustness/data/processed/metadata.csv", "phase2/results/test_inputs/full/cohort.npz", "phase2/results/tables/full/patient_bootstrap/patient_draws.npy"]
    result = {
        "status": "DEFERRED", "statement": "DEFERRED — inventory only; no model execution performed.",
        "recorded_at": now(), "architecture_candidate": "tsai InceptionTime (one 1-D network, not an ensemble)",
        "public_repository": "https://github.com/timeseriesAI/tsai", "public_commit": COMMIT,
        "implementation_attribution": "Unofficial PyTorch implementation by Ignacio Oguiza, based on Fawaz et al.; no code imported or vendored", "license": "Apache-2.0",
        "public_sources": [source_info, license_info],
        "planned_topology": {"input_channels": 12, "output_logits": 5, "inception_depth": 6, "filters_per_branch": 32, "bottleneck_channels": 32, "kernel_argument": 40, "effective_odd_kernel_sizes": [39, 19, 9], "fourth_branch": "max-pool kernel3 stride1 padding1, then 1x1 convolution", "concatenated_channels": 128, "residual_every_modules": 3, "head": "global average pooling then linear128-to-5; no softmax for multilabel task"},
        "planned_input": {"shape": ["batch", 12, 1000], "dtype": "float32", "sampling_rate_hz": 100, "class_order": original["data_identity"]["class_order"], "lead_order": original["data_identity"]["lead_order"], "preprocessing": "Would reuse immutable phase2 preprocessing/split; not executed"},
        "planned_patient_split": {"source": "phase2/results/test_inputs/full/manifest.json", "split_sha256": original["data_identity"]["split_sha256"], "expected_records": {"train": 17084, "validation": 2146, "test": 2158}, "new_split_created": False, "patient_disjointness_reexecuted": False},
        "gpu_inventory_command": "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader", "gpu_inventory": gpu.stdout.strip(),
        "local_paths": [{"path": p, "exists": (ROOT / p).is_file(), "bytes": (ROOT / p).stat().st_size if (ROOT / p).is_file() else None} for p in paths],
        "remote_gpu": {"host": "dgx6", "checked_this_task": False, "execution": "not_requested_not_used"},
        "unverified": ["dependency installation and model import", "model instantiation", "forward and backward correctness", "GPU memory requirement", "epoch duration", "training or generalization performance"],
        "forward_invocations": 0, "backward_invocations": 0, "model_imports": 0, "gpu_smoke_invocations": 0,
        "sources": [file_info("phase2/results/test_inputs/full/manifest.json", schema="phase2 immutable input and split metadata", source="Historical phase2 full run"), file_info("phase2/configs/phase2_main.yaml", schema="phase2_generalization_v1 YAML", source="Historical planned record counts")],
    }
    write_json(OUT / "third_architecture/gate0_manifest.json", result)
    text = f"""# Third-architecture gate 0\n\nDEFERRED — inventory only; no model execution performed.\n\n- Public candidate: [tsai InceptionTime](https://github.com/timeseriesAI/tsai/blob/{COMMIT}/tsai/models/InceptionTime.py), commit `{COMMIT}`; Apache-2.0. This is the repository author's unofficial PyTorch implementation, not a claim of official InceptionTime implementation.\n- Planned input: float32 batch × 12 × 1000, 100 Hz; 5 multilabel logits. Six modules, 32 filters per branch, effective kernels 39/19/9, residual links every three modules, global average pooling. Static source inspection only.\n- Planned split: reuse the exact phase2 train/validation/test identities recorded in the manifest; no new split or training.\n- GPU inventory: `{gpu.stdout.strip()}`. Device listing is not model execution or a CUDA compatibility test.\n- Data paths and immutable public-source hashes are in `gate0_manifest.json`. Remote GPU was not checked or used for this CPU-only reanalysis.\n- Dependency/model import, forward/backward, GPU smoke, memory/epoch timing, and training remain unverified and require a separately approved task.\n\ntraining_invocations = 0\ninference_invocations = 0\nmodel_imports = 0\n"""
    (OUT / "third_architecture/gate0_status.md").write_text(text, encoding="utf-8")
    print(json.dumps({"status": "DEFERRED", "public_commit": COMMIT, "license": "Apache-2.0", "training_invocations": 0, "inference_invocations": 0, "model_imports": 0}, indent=2))


if __name__ == "__main__":
    main()
