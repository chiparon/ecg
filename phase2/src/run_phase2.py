"""Run the registered pilot and full experiment without altering scientific settings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .common import (
    expected_runs,
    file_info,
    load_config,
    preregister,
    run_directory,
    save_json,
    stage_paths,
    verify_preservation,
    verify_training_complete,
)


def technical_gate(cfg, stage):
    """Judge execution integrity, never choose strategies/hyperparameters from test scores."""
    summaries = verify_training_complete(cfg, stage)
    paths = stage_paths(cfg, stage)
    orders, initializations = {}, {}
    run_checks = []
    for summary in summaries:
        strategy, model, seed = summary["strategy"], summary["model"], summary["seed"]
        key = (model, seed)
        initialization = summary["model_initialization_sha256"]
        initializations.setdefault(key, initialization)
        if initializations[key] != initialization:
            raise ValueError(
                "Strategies did not start from identical paired model weights"
            )
        epoch_orders = summary["epoch_order_sha256"]
        orders.setdefault(seed, epoch_orders)
        if orders[seed] != epoch_orders:
            raise ValueError(
                "Strategies/models consumed different per-epoch record orders"
            )
        epochs_path = run_directory(cfg, stage, strategy, model, seed) / "epochs.csv"
        frame = pd.read_csv(epochs_path)
        expected_epochs = cfg["stages"][stage]["epochs"]
        expected_records = cfg["stages"][stage]["expected_records"]["train"]
        if frame.epoch.tolist() != list(range(1, expected_epochs + 1)):
            raise ValueError("Incomplete fixed epoch budget")
        if not (
            (frame.exposure_count == expected_records)
            & (frame.unique_records == expected_records)
            & (frame.exposure_min == 1)
            & (frame.exposure_max == 1)
        ).all():
            raise ValueError("Unequal or repeated training exposure")
        total = expected_epochs * expected_records
        probabilities = cfg["strategies"][strategy]
        counts = {}
        for name, column in (
            ("clean", "clean_count"),
            ("independent_rms", "independent_count"),
            ("electrode", "electrode_count"),
        ):
            observed = int(frame[column].sum())
            probability = probabilities[name]
            tolerance = 6.0 * np.sqrt(total * probability * (1.0 - probability)) + 1.0
            if abs(observed - total * probability) > tolerance:
                raise ValueError(
                    f"Observed augmentation frequency inconsistent with registered Bernoulli schedule: {strategy}/{name}"
                )
            counts[name] = observed
        if sum(counts.values()) != total:
            raise ValueError(
                "Mixed augmentation did not partition each exposure into one branch"
            )
        finite_errors = frame[
            ["actual_snr_error_db_min", "actual_snr_error_db_max"]
        ].to_numpy(dtype=float)
        if np.isfinite(finite_errors).any() and np.nanmax(np.abs(finite_errors)) > 1e-4:
            raise ValueError("Training actual SNR misses registered strength")
        if summary["threshold_tuning_count"] != 1:
            raise ValueError(
                "Final thresholds must be tuned exactly once on clean validation"
            )
        run_checks.append(
            {
                "strategy": strategy,
                "model": model,
                "seed": seed,
                "epochs": expected_epochs,
                "exposures": total,
                "augmentation_counts": counts,
                "epoch_log": file_info(epochs_path),
            }
        )
    for filename in ("evaluation_protocol.json", "statistics_protocol.json"):
        value = json.loads((paths["logs"] / filename).read_text(encoding="utf-8"))
        if (
            value.get("status") != "completed"
            or value.get("config_sha256") != cfg["_config_sha256"]
        ):
            raise ValueError(f"Incomplete or mismatched technical stage: {filename}")
    manifest = json.loads(
        (paths["test_inputs"] / "manifest.json").read_text(encoding="utf-8")
    )
    train_combos = {entry["combo_id"] for entry in cfg["_train_combos"]}
    heldout_combos = {
        case["combo_id"] for case in manifest["cases"] if case["combo_set"] == "heldout"
    }
    if train_combos & heldout_combos:
        raise ValueError("Held-out electrode combinations appeared in training support")
    unseen_snrs = {
        case["snr"]
        for case in manifest["cases"]
        if case["kind"] == "bandpass" and not case["snr_seen_in_training"]
    }
    if unseen_snrs & set(cfg["train"]["snrs"]):
        raise ValueError("An unseen-labeled SNR was present in training")
    if any(
        case["noise_seed"] == cfg["train"]["noise_base"]
        for case in manifest["cases"]
        if case["kind"] != "clean"
    ):
        raise ValueError("Training and test noise bases overlap")
    result = {
        "status": "passed",
        "stage": stage,
        "config_sha256": cfg["_config_sha256"],
        "expected_runs": len(expected_runs(cfg, stage)),
        "run_checks": run_checks,
        "paired_initializations_identical": True,
        "paired_epoch_orders_identical": True,
        "combination_intersection": [],
        "unseen_snr_intersection": [],
        "noise_base_domains_disjoint": True,
        "performance_selection": "none: this gate checks execution, exposure, strength and provenance, not which strategy wins",
    }
    save_json(paths["logs"] / "technical_gate.json", result)
    return result


def run(config_path, stages):
    cfg = load_config(config_path)
    preregister(cfg)
    log_root = Path(cfg["_results_root"]) / "logs"
    status_path = log_root / "run_status.json"
    history = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else []
    )
    environment = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "4",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for stage in stages:
        if stage == "full":
            pilot_gate = log_root / "pilot" / "technical_gate.json"
            if not pilot_gate.exists():
                raise ValueError(
                    "Complete the preregistered pilot technical gate before full training"
                )
            gate = json.loads(pilot_gate.read_text(encoding="utf-8"))
            if (
                gate.get("status") != "passed"
                or gate.get("config_sha256") != cfg["_config_sha256"]
            ):
                raise ValueError(
                    "Pilot gate does not match the fixed full configuration"
                )
        for module in (
            "train_phase2",
            "generate_phase2_noise",
            "evaluate_phase2",
            "statistics_phase2",
            "plot_phase2",
        ):
            number = 1 + sum(
                row["stage"] == stage and row["module"] == module for row in history
            )
            command = [
                sys.executable,
                "-u",
                "-m",
                f"phase2.src.{module}",
                "--config",
                cfg["_config_path"],
                "--stage",
                stage,
            ]
            logfile = log_root / stage / f"{module}_attempt_{number:02d}.log"
            logfile.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "stage": stage,
                "module": module,
                "attempt": number,
                "command": command,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "running",
            }
            history.append(row)
            save_json(status_path, history)
            start = time.perf_counter()
            print(f"phase2: starting {stage}/{module} attempt={number}", flush=True)
            with logfile.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=cfg["_workspace_root"],
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line.rstrip(), flush=True)
                code = process.wait()
            row.update(
                status="completed" if code == 0 else "failed",
                exit_code=code,
                elapsed_seconds=time.perf_counter() - start,
                log=file_info(logfile),
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            save_json(status_path, history)
            if code:
                raise subprocess.CalledProcessError(code, command)
        technical_gate(cfg, stage)
        print(
            f"phase2: {stage} technical gate passed; no test-driven configuration changes",
            flush=True,
        )
    verify_preservation(cfg)
    print(
        "phase2: requested stages completed and protected phase-one files unchanged",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="phase2/configs/phase2_main.yaml")
    parser.add_argument("--stage", choices=("pilot", "full", "all"), default="all")
    args = parser.parse_args()
    run(args.config, ["pilot", "full"] if args.stage == "all" else [args.stage])


if __name__ == "__main__":
    main()
