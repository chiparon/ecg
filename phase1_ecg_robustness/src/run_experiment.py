"""Run the mandatory gate -> clean training -> paired evaluation -> statistics -> figures."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    logdir = Path(cfg["results_dir"]) / "logs" / cfg["run_name"]
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    steps = ["audit_noise"]
    if not args.skip_training:
        steps.append("train")
    if not args.skip_evaluation:
        steps.append("evaluate")
    steps += ["statistics", "plotting"]
    events = []
    print("EXPERIMENT_STARTED " + cfg["run_name"], flush=True)
    for module in steps:
        command = [sys.executable, "-u", "-m", "src." + module, "--config", args.config]
        if module == "train":
            command.append("--resume-completed")
        start = time.time()
        with (logdir / (module + ".log")).open("a", encoding="utf-8") as stream:
            stream.write("\nCOMMAND: " + subprocess.list2cmdline(command) + "\n")
            stream.flush()
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "OMP_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "4",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                },
            )
            for line in proc.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = proc.wait()
        events.append(
            dict(
                module=module,
                command=command,
                exit_code=code,
                elapsed_seconds=time.time() - start,
            )
        )
        (logdir / "run_status.json").write_text(
            json.dumps(events, indent=2), encoding="utf-8"
        )
        if code:
            raise SystemExit(code)
    print("EXPERIMENT_COMPLETED " + cfg["run_name"], flush=True)


if __name__ == "__main__":
    main()
