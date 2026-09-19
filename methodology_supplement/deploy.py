"""Finite source/input/result transfers; GPU workers are supervised separately."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time
import zipfile

from .common import WORKSPACE, file_info, load_config, read_json, save_json, stage_paths

REMOTE_ROOT = "/home/chiparon/ecg_methodology_minimal_five"
REMOTE_PYTHON = "/home/chiparon/ecg_benchmark/20260919_134009/venv/bin/python"


def _run(command):
    subprocess.run(command, cwd=WORKSPACE, check=True)


def _ssh(host, arguments):
    _run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, shlex.join(arguments)])


def _scp(arguments):
    _run(["scp", "-B", "-o", "ConnectTimeout=15", *arguments])


def _archive_results(cfg, stage):
    paths = stage_paths(cfg, stage)
    archive = paths["logs"] / "resnet_results.zip"
    files = list((paths["predictions"] / "resnet").rglob("*.npz"))
    files += [paths["logs"] / f"inference_resnet_{i}.json" for i in (0, 1)]
    files += [paths["tables"] / f"evaluation_resnet_{i}.csv" for i in (0, 1)]
    if not files or any(not path.is_file() for path in files):
        raise ValueError("Spark results are incomplete")
    for index in (0, 1):
        if read_json(paths["logs"] / f"inference_resnet_{index}.json")["status"] != "completed":
            raise ValueError("Cannot archive an unfinished Spark worker")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        for path in sorted(files):
            output.write(path, path.relative_to(WORKSPACE).as_posix())
    return file_info(archive)


def run(args):
    start = time.perf_counter()
    cfg = load_config(args.config)
    paths = stage_paths(cfg, args.stage)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    print(f"deployment-start direction={args.direction} stage={args.stage}", flush=True)
    remote = args.remote_root.rstrip("/")
    remote_categories = {name: f"{remote}/{path.relative_to(WORKSPACE).as_posix()}" for name, path in paths.items()}
    if args.direction == "archive":
        result = _archive_results(cfg, args.stage)
        print(f"deployment-completed archive={result['path']}", flush=True)
        return result
    if args.direction == "push":
        freeze = read_json(paths["logs"] / "freeze.json")
        files = {WORKSPACE / item["path"] for item in freeze["source_files"]}
        files.update((WORKSPACE / "methodology_supplement").glob("*.py"))
        files.update([Path(cfg["_config_path"]), WORKSPACE / "methodology_supplement/implementation_contract.json",
                      WORKSPACE / "phase1_ecg_robustness/src/__init__.py", WORKSPACE / "phase2/src/__init__.py",
                      WORKSPACE / "phase2/__init__.py"])
        bundle = paths["logs"] / "source_bundle.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(files):
                archive.write(path, path.relative_to(WORKSPACE).as_posix())
        _ssh(args.host, ["mkdir", "-p", remote, str(PurePosixPath(remote_categories["inputs"]).parent),
                         remote_categories["tables"], remote_categories["logs"]])
        _scp([str(bundle.relative_to(WORKSPACE)), f"{args.host}:{remote}/source_bundle.zip"])
        extract = "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])"
        _ssh(args.host, [args.remote_python, "-c", extract, f"{remote}/source_bundle.zip", remote])
        _scp(["-r", str(paths["inputs"].relative_to(WORKSPACE)),
              f"{args.host}:{PurePosixPath(remote_categories['inputs']).parent}/"])
        _scp([str((paths["logs"] / "freeze.json").relative_to(WORKSPACE)), f"{args.host}:{remote_categories['logs']}/"])
        _scp([str((paths["tables"] / "noise_diagnostics.csv").relative_to(WORKSPACE)),
              str((paths["tables"] / "noise_summary.csv").relative_to(WORKSPACE)), f"{args.host}:{remote_categories['tables']}/"])
        result = dict(source_bundle=file_info(bundle), canonical_input_directory=paths["inputs"].relative_to(WORKSPACE).as_posix())
    else:
        _ssh(args.host, ["env", f"PYTHONPATH={remote}", "PYTHONDONTWRITEBYTECODE=1", args.remote_python,
                         "-u", "-m", "methodology_supplement.deploy", "--direction", "archive", "--stage", args.stage])
        local_archive = paths["logs"] / "resnet_results.zip"
        _scp([f"{args.host}:{remote_categories['logs']}/resnet_results.zip", str(local_archive.relative_to(WORKSPACE))])
        allowed_prefix = (paths["predictions"] / "resnet").relative_to(WORKSPACE).as_posix() + "/"
        allowed_files = {(paths["logs"] / f"inference_resnet_{i}.json").relative_to(WORKSPACE).as_posix() for i in (0, 1)}
        allowed_files |= {(paths["tables"] / f"evaluation_resnet_{i}.csv").relative_to(WORKSPACE).as_posix() for i in (0, 1)}
        with zipfile.ZipFile(local_archive) as archive:
            for member in archive.namelist():
                parsed = PurePosixPath(member)
                if parsed.is_absolute() or ".." in parsed.parts or not (member.startswith(allowed_prefix) or member in allowed_files):
                    raise ValueError(f"Unexpected archive member: {member}")
            archive.extractall(WORKSPACE)
        result = dict(results_archive=file_info(local_archive))
    result.update(status="completed", stage=args.stage, direction=args.direction, host=args.host,
                  remote_root=remote, remote_python=args.remote_python,
                  config_sha256=cfg["_config_sha256"], completed_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.perf_counter() - start)
    save_json(paths["logs"] / f"deployment_{args.direction}.json", result)
    print(f"deployment-completed direction={args.direction} elapsed_s={result['elapsed_seconds']:.1f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--direction", choices=("push", "pull", "archive"), required=True)
    parser.add_argument("--host", default="dgx6")
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    parser.add_argument("--remote-python", default=REMOTE_PYTHON)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
