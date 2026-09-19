"""Freeze controls before reading predictions; reference existing legal data only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import numpy as np

from phase1_ecg_robustness.src.lead_matrix import get_lead_matrix, LEADS, ELECTRODES
from .common import (
    WORKSPACE, check_info, file_info, array_sha256, load_config, legacy_paths,
    read_json, resolve_path, save_json, sha256, stage_paths, write_csv,
)

CORE_MODULES = ("__init__", "common", "prepare", "inputs", "infer", "merge", "analyse")
SPEC = WORKSPACE / "Agent Task：保秩保谱 Signed-Control 机制实验.md"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _base_revision(cfg):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True).strip()
    if revision != cfg["base_commit"]:
        raise ValueError(f"Expected base commit {cfg['base_commit']}, found {revision}")
    result = subprocess.run(["git", "diff", "--quiet", cfg["base_commit"], "--",
                             "phase1_ecg_robustness", "phase2", "methodology_supplement"], cwd=WORKSPACE)
    if result.returncode:
        raise ValueError("Tracked protected baseline differs from the requested frozen commit")
    return revision


def freeze_controls(cfg):
    revision = _base_revision(cfg)
    root = resolve_path(cfg["results_dir"]) / "inputs"
    root.mkdir(parents=True, exist_ok=True)
    control_path, matrix_path = root / "sign_controls.json", root / "matrices.json"
    if control_path.exists() or matrix_path.exists():
        if not (control_path.exists() and matrix_path.exists()):
            raise ValueError("Incomplete control freeze; investigate instead of replacing modes")
        controls, matrices = read_json(control_path), read_json(matrix_path)
        if controls["config_sha256"] != cfg["_config_sha256"] or matrices["config_sha256"] != cfg["_config_sha256"]:
            raise ValueError("Existing controls belong to a different protocol; never overwrite them")
        if controls["matrices_sha256"] != sha256(matrix_path):
            raise ValueError("Frozen matrix artifact changed")
        return controls, matrices
    A = np.asarray(get_lead_matrix(), dtype=np.float64)
    if list(LEADS) != cfg["lead_order"] or A.shape != (12, 9):
        raise ValueError("Frozen Standard matrix geometry/order changed")
    C = A @ A.T
    singular = np.linalg.svd(A, compute_uv=False)
    eigenvalues = np.linalg.eigvalsh(C)
    rank = int(np.linalg.matrix_rank(A))
    P = A @ np.linalg.pinv(A, rcond=cfg["subspace"]["pseudoinverse_rcond"])
    Q = np.eye(12, dtype=np.float64) - P
    symmetry = float(np.linalg.norm(P - P.T, "fro"))
    idempotence = float(np.linalg.norm(P @ P - P, "fro"))
    if max(symmetry, idempotence) >= cfg["subspace"]["projection_tolerance"]:
        raise ValueError("Standard subspace projector failed its algebraic gate")
    rng = np.random.Generator(np.random.PCG64(cfg["selection"]["seed"]))
    controls, matrices, candidates, validations = [], [], [], []
    accepted = set()
    low, high = cfg["selection"]["negative_count_inclusive"]
    while len(controls) < cfg["selection"]["count"]:
        signs = np.concatenate((np.array([1], dtype=np.int64), rng.choice(np.array([-1, 1]), size=11)))
        candidate_index = len(candidates) + 1
        negative_count = int((signs < 0).sum())
        signed_A = signs[:, None] * A
        signed_C = signed_A @ signed_A.T
        difference = float(np.linalg.norm(signed_C - C, "fro") / np.linalg.norm(C, "fro"))
        reason = ("negative_count_outside_4_to_7" if not low <= negative_count <= high
                  else "all_positive" if negative_count == 0
                  else "duplicate" if tuple(signs) in accepted
                  else "unchanged_covariance" if difference <= cfg["selection"]["covariance_change_min"]
                  else "accepted_first_eligible_in_rng_order")
        candidate = dict(candidate_index=candidate_index, sign_vector=signs.tolist(),
                         negative_count=negative_count, covariance_relative_change=difference,
                         accepted=reason.startswith("accepted"), reason=reason)
        candidates.append(candidate)
        if not candidate["accepted"]:
            continue
        mode = f"S_{len(controls):02d}"
        accepted.add(tuple(signs))
        signed_singular = np.linalg.svd(signed_A, compute_uv=False)
        signed_eigenvalues = np.linalg.eigvalsh(signed_C)
        offdiagonal = ~np.eye(12, dtype=bool)
        active = offdiagonal & (np.abs(C) > 1e-14)
        row = dict(mode=mode, candidate_index=candidate_index,
                   sign_vector=json.dumps(signs.tolist(), separators=(",", ":")), negative_count=negative_count,
                   rank_standard=rank, rank_signed=int(np.linalg.matrix_rank(signed_A)),
                   singular_max_abs_error=float(np.max(np.abs(singular - signed_singular))),
                   eigenvalues_max_abs_error=float(np.max(np.abs(eigenvalues - signed_eigenvalues))),
                   covariance_diagonal_max_abs_error=float(np.max(np.abs(np.diag(C) - np.diag(signed_C)))),
                   unit_source_rms_max_abs_error=float(np.max(np.abs(np.sqrt(np.diag(C)) - np.sqrt(np.diag(signed_C))))),
                   covariance_relative_change=difference,
                   offdiagonal_sign_differences=int(np.sum(np.sign(C[active]) != np.sign(signed_C[active]))),
                   different_from_standard=not np.array_equal(A, signed_A),
                   matrix_array_sha256=array_sha256(signed_A))
        tolerance = cfg["gates"]["matrix_absolute_tolerance"]
        row["passed"] = bool(row["rank_standard"] == row["rank_signed"] and row["different_from_standard"]
                             and row["offdiagonal_sign_differences"] > 0
                             and max(row["singular_max_abs_error"], row["eigenvalues_max_abs_error"],
                                     row["covariance_diagonal_max_abs_error"], row["unit_source_rms_max_abs_error"]) <= tolerance)
        if not row["passed"]:
            raise ValueError(f"Accepted candidate failed a matrix invariant: {mode}; do not substitute another candidate")
        validations.append(row)
        controls.append(dict(mode=mode, **candidate, matrix_array_sha256=row["matrix_array_sha256"]))
        signed_P = signs[:, None] * P * signs[None, :]
        matrices.append(dict(mode=mode, matrix=signed_A.tolist(), covariance=signed_C.tolist(),
                             singular_values=signed_singular.tolist(), eigenvalues=signed_eigenvalues.tolist(),
                             projection=signed_P.tolist(), rank=row["rank_signed"],
                             array_sha256=row["matrix_array_sha256"],
                             unit_covariance_q_reference=float(np.trace(Q @ signed_C) / np.trace(signed_C))))
    matrix_record = dict(status="frozen", config_sha256=cfg["_config_sha256"],
                         lead_order=list(LEADS), electrode_order=list(ELECTRODES),
                         standard=dict(matrix=A.tolist(), covariance=C.tolist(), singular_values=singular.tolist(),
                                       eigenvalues=eigenvalues.tolist(), rank=rank, array_sha256=array_sha256(A)),
                         controls=matrices, validation=validations,
                         projection=dict(P=P.tolist(), Q=Q.tolist(), symmetry_fro=symmetry,
                                         idempotence_fro=idempotence, rank=rank,
                                         standard_annihilation_fro=float(np.linalg.norm(Q @ A, "fro"))))
    save_json(matrix_path, matrix_record)
    control_record = dict(status="frozen_before_predictions", frozen_at=_now(), base_commit=revision,
                          config_sha256=cfg["_config_sha256"], config_file=file_info(cfg["_config_path"]),
                          task_specification=file_info(SPEC), selection_seed=cfg["selection"]["seed"],
                          generator=cfg["selection"]["generator"], numpy_version=np.__version__,
                          lead_order=list(LEADS), controls=controls, candidate_history=candidates,
                          matrices_sha256=sha256(matrix_path), selection_used_classification_results=False)
    save_json(control_path, control_record)
    print(json.dumps({"status":"controls_frozen_before_predictions", "config_sha256":cfg["_config_sha256"],
                      "controls":[dict(mode=c["mode"], candidate_index=c["candidate_index"], sign_vector=c["sign_vector"]) for c in controls]}, indent=2), flush=True)
    return control_record, matrix_record


def run(config=None, stage="smoke", controls_only=False):
    cfg = load_config(config)
    controls, matrices = freeze_controls(cfg)
    if controls_only:
        return controls
    paths = stage_paths(cfg, stage)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    if stage == "full":
        smoke_path = stage_paths(cfg, "smoke")["logs"] / "smoke_verification.json"
        smoke = read_json(smoke_path)
        if smoke.get("status") not in {"passed", "waived_by_user"} or smoke.get("config_sha256") != cfg["_config_sha256"]:
            raise ValueError("Full execution requires passed smoke or an explicit current-protocol user waiver")
        save_json(paths["logs"] / "smoke_verification.json", smoke)
    old = legacy_paths(cfg, stage)
    old_freeze_path = old["logs"] / "freeze.json"
    old_freeze = read_json(old_freeze_path)
    old_manifest_path = old["inputs"] / "manifest.json"
    old_manifest = read_json(old_manifest_path)
    if old_freeze["status"] != "completed" or old_manifest["status"] != "completed" or old_freeze["config_sha256"] != old_manifest["config_sha256"]:
        raise ValueError("Existing cached experiment is incomplete or inconsistent")
    for info in old_freeze["source_files"]:
        check_info(info)
    for key in ("clean", "cohort", "draws", "checkpoints"):
        check_info(old_freeze[key])
    with np.load(resolve_path(old_freeze["cohort"]["path"]), allow_pickle=False) as archive:
        reference = {key: archive[key] for key in archive.files}
    n = len(reference["ids"])
    expected_n = 2158 if stage == "full" else 100
    if n != expected_n or reference["y"].shape != (n, 5):
        raise ValueError("Existing cohort differs from the prescribed sample")
    for key, expected in old_freeze["array_hashes"].items():
        if array_sha256(reference[key]) != expected:
            raise ValueError(f"Existing cohort array hash changed: {key}")
    draws = np.load(resolve_path(old_freeze["draws"]["path"]), mmap_mode="r", allow_pickle=False)
    if draws.shape != (cfg["stages"][stage]["bootstrap_replicates"], len(reference["unique_patients"])):
        raise ValueError("Existing common patient draws have the wrong geometry")
    registry = read_json(resolve_path(old_freeze["checkpoints"]["path"]))
    entries = registry["checkpoints"]
    if {(entry["model"], entry["seed"]) for entry in entries} != {(model, seed) for model in cfg["models"] for seed in cfg["phase1_training_seeds"]}:
        raise ValueError("The checkpoint registry is not the prescribed six models")
    for entry in entries:
        check_info(entry)
    source_paths = [resolve_path(cfg["results_dir"]) / f"{module}.py" for module in CORE_MODULES]
    source_infos = [file_info(path) for path in source_paths] + old_freeze["source_files"]
    index_path = old["tables"] / "evaluation_index.csv"
    check_info(file_info(index_path))
    selected_cases = [case for case in old_manifest["cases"] if case.get("source_id") == "standard" and int(case["snr"]) in cfg["snrs"]]
    if len(selected_cases) != 36:
        raise ValueError("Frozen E/I reference must contain 36 conditions")
    base_paths = sorted({case["noise_path"] for case in selected_cases})
    noise_infos = [file_info(resolve_path(path)) for path in base_paths]
    hashes = {info["path"]: info["sha256"] for info in noise_infos}
    if any(hashes[case["noise_path"]] != case["noise_sha256"] for case in selected_cases):
        raise ValueError("Existing Standard base arrays changed")
    dependency_reports = []
    for name in ("inference_resnet_0.json", "inference_resnet_1.json", "inference_tcn_0.json"):
        report_path = old["logs"] / name
        report = read_json(report_path)
        if report["status"] != "completed":
            raise ValueError("The old prediction producer did not complete")
        for info in report["source_fingerprints"]:
            check_info(info)
        dependency_reports.append(file_info(report_path))
    shared = resolve_path(cfg["results_dir"]) / "inputs"
    protected = {info["path"]: info for info in old_freeze["protected_files"]}
    extra = [file_info(old_freeze_path), file_info(old_manifest_path), file_info(index_path),
             old_freeze["clean"], old_freeze["cohort"], old_freeze["draws"], old_freeze["checkpoints"],
             *noise_infos, *dependency_reports,
             file_info(WORKSPACE / "methodology_supplement/results/logs/full/final_acceptance.json")]
    for info in extra:
        protected[info["path"]] = info
    record = dict(status="completed", stage=stage, frozen_at=_now(), host=platform.node(),
                  base_commit=cfg["base_commit"], config_sha256=cfg["_config_sha256"],
                  config_file=file_info(cfg["_config_path"]), task_specification=file_info(SPEC),
                  no_training=True, new_raw_waveform_copies=0,
                  sign_controls=file_info(shared / "sign_controls.json"), matrices=file_info(shared / "matrices.json"),
                  clean=old_freeze["clean"], cohort=old_freeze["cohort"], draws=old_freeze["draws"],
                  checkpoint_entries=entries, checkpoints=old_freeze["checkpoints"],
                  n_records=n, n_patients=len(reference["unique_patients"]), array_hashes=old_freeze["array_hashes"],
                  legacy_config_sha256=old_freeze["config_sha256"], legacy_input_manifest=file_info(old_manifest_path),
                  legacy_prediction_index=file_info(index_path), legacy_worker_reports=dependency_reports,
                  legacy_freeze=file_info(old_freeze_path), noise_bases=noise_infos,
                  data_identity=old_freeze["data_identity"], source_files=source_infos,
                  protected_files=list(protected.values()), draws_reused_byte_for_byte=True,
                  shift_noise=dict(
                      status="not_newly_generated",
                      reason="The existing strict_control.py producer and marginal_shift predictions retain offsets, seeds and diagnostics, but not shifted/noisy waveform arrays or final-input SHA-256. No directly reusable shift input cache was found. This experiment does not reconstruct or generate an equivalent shift branch.",
                      existing_implementation=file_info(WORKSPACE / "phase1_ecg_robustness/src/strict_control.py"),
                      existing_protocol=file_info(WORKSPACE / "phase1_ecg_robustness/results/metrics/review_supplement/strict_protocol.json"),
                      prediction_example=file_info(WORKSPACE / "phase1_ecg_robustness/results/metrics/review_supplement/strict/tcn/seed_17/noise_8128_snr_0_marginal_shift.npz")))
    destination = paths["logs"] / "freeze.json"
    if destination.exists():
        existing = read_json(destination)
        for key in ("frozen_at",):
            record[key] = existing[key]
        if existing != record:
            raise ValueError("Existing stage freeze differs; refuse a silent replacement")
    else:
        save_json(destination, record)
    write_csv(paths["tables"] / "matrix_validation.csv", matrices["validation"])
    save_json(paths["inputs"] / "patient_draws_provenance.json", dict(
        stage=stage, config_sha256=cfg["_config_sha256"], draws=record["draws"], cohort=record["cohort"],
        array_hashes=record["array_hashes"], source_freeze=record["legacy_freeze"], byte_identical=True))
    if stage == "full":
        save_json(shared / "manifest.json", dict(status="protocol_and_sources_frozen", config_sha256=cfg["_config_sha256"],
                  sign_controls=record["sign_controls"], matrices=record["matrices"], freeze=file_info(destination),
                  clean=record["clean"], cohort=record["cohort"], draws=record["draws"], noise_bases=noise_infos,
                  final_input_completion_marker="signed_control_mechanism/inputs/full/manifest.json"))
    print(json.dumps({"status":"completed", "stage":stage, "n_records":n, "n_patients":record["n_patients"],
                      "config_sha256":cfg["_config_sha256"], "existing_base_files":len(noise_infos)}, indent=2), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--freeze-controls-only", action="store_true")
    args = parser.parse_args()
    run(args.config, args.stage, args.freeze_controls_only)


if __name__ == "__main__":
    main()
