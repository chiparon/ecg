"""Read-only host probe plus real CUDA FP32 forward/backward execution."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import traceback


def main():
    result = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node(),
              "platform": platform.platform(), "python": sys.version, "executable": sys.executable,
              "commands": {}, "packages": {}, "checks": [], "status": "failed"}
    for command in (["uname", "-a"], ["lscpu"], ["free", "-h"], ["df", "-h", "."], ["nvidia-smi"]):
        try:
            p = subprocess.run(command, capture_output=True, text=True, timeout=20)
            result["commands"][" ".join(command)] = {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        except Exception as exc:
            result["commands"][" ".join(command)] = {"error": str(exc)}
    for package in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "PyYAML", "wfdb", "psutil"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    try:
        import torch
        result["torch_cuda"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if not result["cuda_available"]:
            raise RuntimeError("CUDA unavailable")
        result["gpu"] = torch.cuda.get_device_name(0)
        result["capability"] = torch.cuda.get_device_capability(0)
        result["arch_list"] = torch.cuda.get_arch_list()
        result["total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        torch.set_num_threads(4)
        torch.manual_seed(17)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        x = torch.randn(16, 12, 1000).cuda()
        y = (x * x).sum()
        torch.cuda.synchronize()
        result["checks"].append({"name": "cuda_elementwise", "finite": bool(torch.isfinite(y))})
        net = torch.nn.Sequential(torch.nn.Conv1d(12, 24, 7, padding=3), torch.nn.ReLU(),
                                  torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(), torch.nn.Linear(24, 5)).cuda()
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        prediction = net(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, torch.zeros_like(prediction))
        loss.backward()
        assert prediction.shape == (16, 5) and torch.isfinite(prediction).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        optimizer.step()
        torch.cuda.synchronize()
        result["checks"].append({"name": "conv1d_forward_backward_adamw", "loss": loss.item(), "shape": list(prediction.shape)})
        result["status"] = "passed"
    except Exception:
        result["error"] = traceback.format_exc()
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
