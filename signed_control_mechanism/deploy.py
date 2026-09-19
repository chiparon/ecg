"""Transfer new code/audits only; verify existing DGX waveforms in place."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import time
import zipfile

from methodology_supplement.deploy import _ssh, _scp
from .common import WORKSPACE, file_info, load_config, load_manifest, read_json, resolve_path, save_json, stage_paths


PREFIX = "signed_control_mechanism/"


def _archive_results(cfg, stage):
    paths = stage_paths(cfg, stage)
    files = set()
    for index in (0, 1):
        report_path = paths["logs"] / f"inference_resnet_{index}.json"
        report = read_json(report_path)
        if report.get("status") != "completed" or report.get("config_sha256") != cfg["_config_sha256"]:
            raise ValueError("Refuse to archive an incomplete/stale DGX worker")
        files.add(report_path)
        files.add(resolve_path(report["ledger"]["path"]))
        files.update((paths["predictions"] / f"worker_resnet_{index}").rglob("*.npz"))
    if len([path for path in files if path.suffix == ".npz"]) != 276:
        raise ValueError("Expected 270 signed predictions plus six internal clean references on DGX")
    archive_path = paths["logs"] / "resnet_results.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(WORKSPACE).as_posix())
    return file_info(archive_path)


def run(config=None, stage="smoke", direction="push"):
    started = time.perf_counter()
    cfg = load_config(config)
    paths = stage_paths(cfg, stage)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    host = cfg["execution"]["resnet_host"]
    remote = cfg["execution"]["remote_root"].rstrip("/")
    python = cfg["execution"]["remote_python"]
    remote_log = remote + "/" + paths["logs"].relative_to(WORKSPACE).as_posix()
    print(f"deployment-start direction={direction} stage={stage}", flush=True)
    if direction == "archive":
        return _archive_results(cfg, stage)
    if direction == "push":
        manifest = load_manifest(cfg, stage)
        freeze_path = paths["logs"] / "freeze.json"
        freeze = read_json(freeze_path)
        files = set(resolve_path(cfg["results_dir"]).glob("*.py"))
        files.update([Path(cfg["_config_path"]), freeze_path, paths["inputs"] / "manifest.json",
                      resolve_path(freeze["sign_controls"]["path"]), resolve_path(freeze["matrices"]["path"]),
                      resolve_path(manifest["audit"]["path"])])
        if stage == "full":
            files.add(stage_paths(cfg, "smoke")["logs"] / "smoke_verification.json")
            files.add(paths["logs"] / "smoke_verification.json")
        bundle_path = paths["logs"] / "deployment_bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in sorted(files):
                name = path.relative_to(WORKSPACE).as_posix()
                if not name.startswith(PREFIX) or path.suffix == ".npy":
                    raise ValueError("Only new namespace code/metadata/audits may be transferred")
                archive.write(path, name)
        _ssh(host, ["mkdir", "-p", remote_log])
        remote_bundle = remote_log + "/deployment_bundle.zip"
        _scp([str(bundle_path.relative_to(WORKSPACE)), f"{host}:{remote_bundle}"])
        extract = (
            "import sys,zipfile; from pathlib import PurePosixPath; "
            "z=zipfile.ZipFile(sys.argv[1]); "
            "assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts "
            "and n.startswith('signed_control_mechanism/') for n in z.namelist()); "
            "z.extractall(sys.argv[2])"
        )
        _ssh(host, [python, "-c", extract, remote_bundle, remote])
        verify = (
            "import sys; from signed_control_mechanism.common import load_config,require_freeze,check_info; "
            "c=load_config(); f=require_freeze(c,sys.argv[1]); "
            "items=[f[k] for k in ('clean','cohort','draws','legacy_input_manifest')]+f['noise_bases']+f['checkpoint_entries']; "
            "[check_info(x) for x in items]; print('existing-DGX-data-verified files='+str(len(items)))"
        )
        _ssh(host, ["env", f"PYTHONPATH={remote}", "PYTHONDONTWRITEBYTECODE=1", "OPENBLAS_NUM_THREADS=1",
                    "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", python, "-c", verify, stage])
        result = dict(bundle=file_info(bundle_path), new_raw_or_clean_waveform_files_transferred=0,
                      reused_existing_dgx_files=len(freeze["noise_bases"]) + len(freeze["checkpoint_entries"]) + 4)
    elif direction == "pull":
        _ssh(host, ["env", f"PYTHONPATH={remote}", "PYTHONDONTWRITEBYTECODE=1", python, "-u", "-m",
                    "signed_control_mechanism.deploy", "--stage", stage, "--direction", "archive"])
        archive_path = paths["logs"] / "resnet_results.zip"
        _scp([f"{host}:{remote_log}/resnet_results.zip", str(archive_path.relative_to(WORKSPACE))])
        allowed_roots = [(paths["predictions"] / f"worker_resnet_{index}").relative_to(WORKSPACE).as_posix() + "/" for index in (0, 1)]
        allowed_files = {(paths["logs"] / f"inference_resnet_{index}.json").relative_to(WORKSPACE).as_posix() for index in (0, 1)}
        allowed_files |= {(paths["tables"] / f"evaluation_resnet_{index}.csv").relative_to(WORKSPACE).as_posix() for index in (0, 1)}
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or not (name in allowed_files or any(name.startswith(root) for root in allowed_roots)):
                    raise ValueError(f"Unexpected result archive member: {name}")
            archive.extractall(WORKSPACE)
        result = dict(archive=file_info(archive_path))
    else:
        raise ValueError("Unknown deployment direction")
    result.update(status="completed", stage=stage, direction=direction, config_sha256=cfg["_config_sha256"],
                  host=host, remote_root=remote, remote_python=python,
                  completed_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started)
    save_json(paths["logs"] / f"deployment_{direction}.json", result)
    print(f"deployment-completed direction={direction} seconds={result['elapsed_seconds']:.2f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--direction", choices=("push", "pull", "archive"), required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(args.config, args.stage, args.direction)


if __name__ == "__main__":
    main()
