"""Save actual runtime, packages and local hardware; exercise CUDA when present."""

import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import psutil
import torch


def main():
    root = Path(__file__).resolve().parents[1]
    for folder in [
        "data/processed",
        "results/logs",
        "results/tables",
        "results/figures",
        "results/checkpoints",
        "results/metrics",
        "reports",
        "scripts",
        "tests",
    ]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    log = root / "results/logs"
    report = dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        cpu_count=psutil.cpu_count(),
        memory=psutil.virtual_memory()._asdict(),
        disk=shutil.disk_usage(root)._asdict(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        packages={
            p: importlib.metadata.version(p)
            for p in [
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "matplotlib",
                "seaborn",
                "torch",
                "wfdb",
                "PyYAML",
                "pytest",
                "psutil",
                "requests",
            ]
        },
    )
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        a = torch.arange(8, device="cuda", dtype=torch.float32, requires_grad=True)
        (a.square().sum()).backward()
        assert np.array_equal(a.grad.cpu().numpy(), np.arange(8) * 2)
        report["cuda_backward_probe"] = "passed"
    (log / "environment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    (root / "requirements-lock.txt").write_text(freeze.stdout, encoding="utf-8")
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    (log / "nvidia-smi.txt").write_text(smi.stdout + smi.stderr, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
