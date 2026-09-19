"""Verify and merge the two Spark shards and one local shard, never loose files."""
from __future__ import annotations

import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from .common import (array_sha256, file_info, load_checkpoints, load_config, load_reference,
                     read_json, require_freeze, resolve_path, save_json, sha256, stage_paths, write_csv)


def run(config=None, stage="full"):
    cfg = load_config(config)
    freeze = require_freeze(cfg, stage)
    paths = stage_paths(cfg, stage)
    manifest = read_json(paths["inputs"] / "manifest.json")
    if manifest["status"] != "completed" or manifest["config_sha256"] != cfg["_config_sha256"]:
        raise ValueError("Noise cache is not complete for this protocol")
    reference = load_reference(cfg, stage)
    checkpoints = {(item["model"], int(item["seed"])): item for item in load_checkpoints(cfg, stage)}
    cases = {item["case_id"]: item for item in manifest["cases"]}
    reports, frames = [], []
    for model in cfg["models"]:
        count = cfg["execution"][f"{model}_workers"]
        for shard in range(count):
            report_path = paths["logs"] / f"inference_{model}_{shard}.json"
            report = read_json(report_path)
            if (report["status"] != "completed" or report["verification_only"] or report["case_filter"] is not None
                    or report["config_sha256"] != cfg["_config_sha256"] or report["stage"] != stage
                    or report["model"] != model or report["shard_index"] != shard or report["shard_count"] != count
                    or not report["host_assignment_verified"]):
                raise ValueError(f"Invalid production worker report: {report_path}")
            ledger = resolve_path(report["ledger"]["path"])
            if sha256(ledger) != report["ledger"]["sha256"]:
                raise ValueError("Worker ledger changed")
            frame = pd.read_csv(ledger)
            owned = [case["case_id"] for i, case in enumerate(manifest["cases"]) if i % count == shard]
            expected = {(model, int(seed), case) for seed in cfg["phase1_training_seeds"] for case in owned}
            observed = set(frame[["model", "seed", "case_id"]].itertuples(index=False, name=None))
            if observed != expected or len(frame) != len(expected) or report["observed_cells"] != len(expected):
                raise ValueError("Worker did not cover exactly its owned cases and all seeds")
            if report["manifest_fingerprint"] != file_info(paths["inputs"] / "manifest.json"):
                raise ValueError("Workers used a different case cache")
            if report["freeze_fingerprint"]["sha256"] != sha256(paths["logs"] / "freeze.json"):
                raise ValueError("Workers used a different freeze")
            for row in frame.to_dict("records"):
                checkpoint = checkpoints[(model, int(row["seed"]))]
                case = cases[row["case_id"]]
                path = resolve_path(row["prediction_path"])
                if sha256(path) != row["prediction_sha256"]:
                    raise ValueError(f"Prediction file changed: {path}")
                with np.load(path, allow_pickle=False) as saved:
                    for key in ("y", "ids", "patient_ids", "indices"):
                        if not np.array_equal(saved[key], reference[key]):
                            raise ValueError(f"Prediction cohort mismatch: {path}: {key}")
                    for key, value in (("config_sha256", cfg["_config_sha256"]), ("case_id", case["case_id"]),
                                       ("checkpoint_sha256", checkpoint["sha256"]), ("input_sha256", case["input_sha256"])):
                        if str(saved[key].item()) != value:
                            raise ValueError(f"Prediction identity mismatch: {path}: {key}")
                    if not np.array_equal(saved["thresholds"], np.asarray(checkpoint["thresholds"])):
                        raise ValueError("Frozen thresholds changed")
                    p = saved["p"]
                    if p.dtype != np.float32 or p.shape != reference["y"].shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                        raise ValueError("Invalid saved prediction probabilities")
                    if array_sha256(p) != str(saved["p_sha256"].item()) or bool(saved["verification_only"].item()):
                        raise ValueError("Prediction is corrupt or nonproduction")
                    if str(saved["clean_p_sha256"].item()) != report["internal_clean_prediction_hashes"][str(row["seed"])]:
                        raise ValueError("Unpaired clean reference")
            frames.append(frame)
            reports.append(report)
    merged = pd.concat(frames, ignore_index=True).sort_values(["model", "seed", "case_id"])
    if merged.duplicated(["model", "seed", "case_id"]).any() or len(merged) != len(cases) * len(checkpoints):
        raise ValueError("Shards overlap or the Cartesian grid is incomplete")
    if not (merged.groupby("case_id").input_sha256.nunique() == 1).all():
        raise ValueError("Models did not use byte-identical inputs")
    remote = [report for report in reports if report["model"] == "resnet"]
    if len(remote) != 2 or remote[0]["pid"] == remote[1]["pid"]:
        raise ValueError("Two distinct Spark worker processes were required")
    if remote[0]["internal_clean_prediction_hashes"] != remote[1]["internal_clean_prediction_hashes"]:
        raise ValueError("Concurrent Spark workers produced different deterministic clean predictions")
    overlap = (min(datetime.fromisoformat(r["finished_at"]) for r in remote)
               - max(datetime.fromisoformat(r["started_at"]) for r in remote)).total_seconds()
    if overlap <= 0:
        raise ValueError("No actual overlap between the two requested Spark workers")
    output = paths["tables"] / "evaluation_index.csv"
    write_csv(output, merged)
    result = dict(status="completed", stage=stage, config_sha256=cfg["_config_sha256"],
                  n_cases=len(cases), n_checkpoints=len(checkpoints), n_predictions=len(merged),
                  expected_predictions=len(cases) * len(checkpoints), dgx_concurrent_processes=2,
                  dgx_worker_pids=[r["pid"] for r in remote], dgx_overlap_seconds=overlap,
                  spark_clean_predictions_identical=True, input_hashes_shared=True,
                  ledger=file_info(output), worker_reports=[file_info(paths["logs"] / f"inference_{r['model']}_{r['shard_index']}.json") for r in reports],
                  freeze_sha256=sha256(paths["logs"] / "freeze.json"))
    save_json(paths["logs"] / "evaluation_merge.json", result)
    print(f"Merged {len(merged)} verified predictions; Spark overlap {overlap:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
