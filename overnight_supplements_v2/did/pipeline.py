"""Read-only heldout DiD. Run with --prepare-only or --run (prepare + compute)."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from overnight_supplements_v2.shared.common import ROOT, OUT, resolve, read_json, file_info, write_json, now, config
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import re
import time
import traceback
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = OUT / "did"
INPUT = "phase2/results/test_inputs/full/manifest.json"
EVALUATION = "phase2/results/logs/full/evaluation_protocol.json"
LEDGER = "phase2/results/logs/full/prediction_manifest.json"
BOOT = "phase2/results/tables/full/patient_bootstrap/manifest.json"
METRICS = "phase2/results/tables/full/metrics.csv"
GROUP_METRICS = "phase2/results/tables/full/group_seed_metrics.csv"
POINT_VERIFICATION = "phase2/results/logs/full/point_statistics_verification.json"
REPORT = "phase2/reports/phase2_final_report.md"
PUBLICATION = "signed_control_mechanism/logs/full/publication_verification.json"
AUC_SOURCE = "phase1_ecg_robustness/src/supplemental_statistics.py"
STRATEGIES = ("electrode", "independent_rms")
ARMS = ("M_EE", "M_IE", "M_EI", "M_II")
FIELDS = (*ARMS, "g_E", "g_I", "gamma")
UNIT_FIELDS = ("architecture", "snr_db", "train_seed", "test_structure", "combo_id", "noise_seed")
TOL = 1e-12


def require(value, message):
    if not value:
        raise ValueError(message)


def array_hash(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def schema(path):
    path = resolve(path)
    if path.suffix == ".npz":
        result = {}
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
                    result[member.removesuffix(".npy")] = {"shape": list(shape), "dtype": str(dtype), "fortran_order": fortran}
        return result
    if path.suffix == ".npy":
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        return {"shape": list(shape), "dtype": str(dtype), "fortran_order": fortran}
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return {"columns": next(csv.reader(handle))}
    if path.suffix == ".json":
        value = read_json(path)
        return {"type": type(value).__name__, "keys": list(value) if isinstance(value, dict) else None}
    return {"format": path.suffix, "interpretation": "opaque bytes; checkpoints never deserialized" if path.suffix == ".pt" else "read-only source text"}


def load_auc_class():
    # Importing the original module would transitively import evaluate/models.
    # Compile exactly the pure original class, not a rewritten metric operator.
    tree = ast.parse(resolve(AUC_SOURCE).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_PreparedAUC"]
    require(len(nodes) == 1, "Original _PreparedAUC definition is not unique")
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), AUC_SOURCE, "exec"), namespace)
    return namespace["_PreparedAUC"]


def values(arms):
    arms = np.asarray(arms, dtype=np.float64)
    ge, gi = arms[..., 0] - arms[..., 1], arms[..., 2] - arms[..., 3]
    return np.concatenate((arms, ge[..., None], gi[..., None], (ge - gi)[..., None]), axis=-1)


def json_values(row):
    return {key: float(value) if np.isfinite(value) else None for key, value in zip(FIELDS, row)}


def relative(path):
    return resolve(path).relative_to(ROOT).as_posix()


def prediction_key(strategy, architecture, seed, case_id):
    return strategy, architecture, int(seed), case_id


def read_selected_metrics(path, selected, groups=False):
    columns = ["model", "strategy", "seed", "group_id", "kind", "combo_set", "combo_id", "condition", "snr", "outcome", "n_cases", "n_noise_seeds", "metric", "value"] if groups else ["stage", "model", "strategy", "seed", "case_id", "kind", "condition", "combo_id", "combo_set", "snr", "noise_seed", "prediction_path", "prediction_sha256", "checkpoint_sha256", "input_sha256", "macro_auroc"]
    frames = []
    for chunk in pd.read_csv(resolve(path), usecols=columns, chunksize=100000, float_precision="round_trip"):
        mask = chunk.strategy.isin(STRATEGIES) & chunk.model.isin(["resnet", "tcn"]) & chunk.seed.isin([17, 29, 43, 101, 202])
        if groups:
            mask &= chunk.group_id.isin(selected) & chunk.metric.eq("macro_auroc") & chunk.outcome.eq("absolute")
        else:
            mask &= chunk.case_id.isin(selected)
        if mask.any():
            frames.append(chunk.loc[mask])
    require(bool(frames), f"No legal source rows in {path}")
    return pd.concat(frames, ignore_index=True)


def enumerate_design(manifest, cfg):
    scope = cfg["scope"]["p0"]
    require(scope["combo_set"] == "heldout" and scope["kind"] == "bandpass" and scope["family"] == "Gaussian", "Frozen scope conflict")
    require(set(scope["snr_db"]) == {5, 15} and scope["train_seeds"] == [17, 29, 43, 101, 202], "Frozen strengths/seeds conflict")
    require(scope["architectures"] == ["resnet", "tcn"] and scope["train_strategies"] == list(STRATEGIES) and scope["test_structures"] == list(STRATEGIES), "Frozen strategy/model conflict")
    cases = [case for case in manifest["cases"] if case["kind"] == "bandpass" and case["combo_set"] == "heldout" and case["snr"] in (5, 15) and case["condition"] in STRATEGIES]
    combos = sorted({case["combo_id"] for case in cases})
    require(len(combos) == 10 and len(cases) == 200, "Incomplete registered heldout case grid")
    case_map = {(c["snr"], c["condition"], c["combo_id"], c["noise_seed"]): c for c in cases}
    require(len(case_map) == 200, "Duplicate registered cases")
    expected = set(product([5, 15], STRATEGIES, combos, scope["noise_seeds"]))
    require(set(case_map) == expected, "Frozen case Cartesian coverage conflict")
    groups = {}
    for snr, condition in product([5, 15], STRATEGIES):
        gid = f"bandpass__heldout__{condition}__s{snr}"
        found = [g for g in manifest["groups"] if g["group_id"] == gid]
        require(len(found) == 1, f"Missing/duplicate group {gid}")
        group = found[0]
        wanted = {c["case_id"] for c in cases if c["snr"] == snr and c["condition"] == condition}
        require(len(group["case_ids"]) == 50 and set(group["case_ids"]) == wanted, f"Group case membership conflict: {gid}")
        require(group["combo_set"] == "heldout" and group["kind"] == "bandpass" and group["condition"] == condition and group["snr"] == snr, f"Group descriptor conflict: {gid}")
        groups[gid] = group
    units = sorted((a, int(c["snr"]), int(seed), c["condition"], c["combo_id"], int(c["noise_seed"])) for a, seed, c in product(scope["architectures"], scope["train_seeds"], cases))
    require(len(units) == 2000 and len(set(units)) == 2000, "Legal unit coverage conflict")
    return cases, case_map, combos, groups, units


def prepare():
    began = time.perf_counter()
    HERE.mkdir(parents=True, exist_ok=True)
    cfg = config()
    manifest, evaluation, ledger, boot = [read_json(p) for p in (INPUT, EVALUATION, LEDGER, BOOT)]
    require(read_json(PUBLICATION)["status"] == "passed", "Signed publication verification has not passed")
    require(manifest["status"] == evaluation["status"] == "completed", "Original input/evaluation is incomplete")
    cases, case_map, combos, groups, units = enumerate_design(manifest, cfg)
    clean = next(c for c in manifest["cases"] if c["case_id"] == "clean")
    analysis = {
        "status": "preparing", "frozen_at": now(), "scope": cfg["scope"]["p0"],
        "qa_unit_A": dict(zip(UNIT_FIELDS, units[0])), "qa_unit_B": dict(zip(UNIT_FIELDS, units[-1])),
        "qa_draw_ids": [0, 1999], "point_draw_id": -1, "class_order": cfg["class_order"],
        "n_legal_units": len(units), "expected_prediction_count": 4000,
        "cases": cases, "groups": groups, "heldout_combinations": combos,
        "source_paths": {"input_manifest": INPUT, "prediction_manifest": LEDGER, "evaluation_protocol": EVALUATION, "bootstrap_manifest": BOOT, "cohort": boot["cohort"]["path"], "patient_draws": boot["draws"]["path"], "group_metrics": GROUP_METRICS, "case_metrics": METRICS, "original_report": REPORT, "auc_implementation": AUC_SOURCE},
        "tolerance": TOL,
        "metric_operator": "Exact original _PreparedAUC AST class only; stable sorting, exact ties, half-credit tie U statistic; arithmetic macro over all five classes, NaN propagates",
        "weighting": "At each matched combo/noise case first subtract train strategies per test structure; equal average over 10 combos x 5 noise bases; within fixed seed difference across tests; equal average across five fixed seeds",
        "uncertainty": "ddof=1 training seed SD separate from 2.5/97.5 common-patient percentile CI; checkpoints and noise fixed; no new hypothesis tests",
        "identity_rules": {"common": "identical ordered IDs, patients, indices, labels, classes, clean tensor identity, config and draw map", "within_test": "same whole-case and derived per-record final normalized float32 noisy input across training strategies", "within_training": "same checkpoint across test structures; strategy checkpoints must differ", "record_hashes": "Diagnostics contain no record hashes. Reconstruct immutable float32 cache; verify whole tensor hash before deriving normalized raw-byte record hashes; no waveform output"},
        "schema": {"four_arm_alignment": "one selected prediction per row; 4000 rows", "record_input_identity": "one row per input case/record, independent of strategy/model/seed; 200 x 2158 rows", "did_long": "one row per architecture/SNR/train_seed/combo/noise; 1000 rows", "did_draw_audit": "all 2000 draws x (5 seeds + aggregate seed=-1) x 4 architecture/SNR groups, including NaNs and valid flag"},
    }
    for key in ("qa_unit_A", "qa_unit_B"):
        unit = analysis[key]
        unit["case_id"] = case_map[(unit["snr_db"], unit["test_structure"], unit["combo_id"], unit["noise_seed"])]["case_id"]
        unit["paired_case_ids"] = {t: case_map[(unit["snr_db"], t, unit["combo_id"], unit["noise_seed"])]["case_id"] for t in STRATEGIES}
    write_json(HERE / "did_analysis.json", analysis)
    # All historical bytes and schemas are frozen BEFORE any derived numerical value.
    sources = {}
    def add(path, source, expected=None):
        path = relative(path)
        if path not in sources:
            sources[path] = file_info(path, schema=schema(path), source=source)
        info = sources[path]
        if expected is not None:
            require(info["sha256"] == expected["sha256"] and info["bytes"] == expected["bytes"], f"Historical ledger hash conflict: {path}")
        return info
    for path in (PUBLICATION, INPUT, EVALUATION, LEDGER, BOOT, POINT_VERIFICATION, REPORT, AUC_SOURCE, "phase2/src/generate_phase2_noise.py", "phase2/src/statistics_phase2.py", "phase2/src/evaluate_phase2.py", "phase2/src/common.py", "phase1_ecg_robustness/src/datasets.py"):
        add(path, "Frozen protocol, pure metric definition, or provenance of historical producer")
    add(INPUT, "Original input manifest", evaluation["manifest"])
    add(LEDGER, "Complete original prediction inventory", evaluation["outputs"]["predictions"])
    add(METRICS, "Unrounded original per-prediction metrics and identity links", evaluation["outputs"]["metrics"])
    add(GROUP_METRICS, "Original unrounded group absolute macro AUROC", read_json(POINT_VERIFICATION)["group_seed_metrics"])
    report_ledger = "phase2/results/logs/full/report_manifest.json"
    add(report_ledger, "Original report content ledger")
    add(REPORT, "Original report verified against its historical content ledger", read_json(report_ledger)["report"])
    for entry in (boot["draws"], boot["cohort"], manifest["cohort_file"]):
        add(entry["path"], "Original common patients/draws and input cohort", entry)
    signals = relative(Path(manifest["data_identity"]["phase1_cache"]["data_dir"]) / "signals.npy")
    add(signals, "Original immutable float32 clean waveform mmap; reconstruct hash only, no inference")
    base_index = {entry["path"]: entry for entry in manifest["base_files"]}
    diag_index = {entry["path"]: entry for entry in manifest["diagnostic_files"]}
    for case in cases:
        add(case["base_noise_path"], "Original lossless float32 zero-dB noise basis", base_index[case["base_noise_path"]])
        add(case["diagnostics_path"], "Original per-record diagnostic cohort and noise base identifiers", diag_index[case["diagnostics_path"]])
    selected_cp = {}
    for cp in evaluation["checkpoints"]:
        if cp["strategy"] in STRATEGIES and cp["model"] in cfg["scope"]["p0"]["architectures"] and cp["seed"] in cfg["scope"]["p0"]["train_seeds"]:
            key = (cp["strategy"], cp["model"], int(cp["seed"]))
            require(key not in selected_cp, "Duplicate checkpoint ledger key")
            selected_cp[key] = cp
            add(cp["checkpoint"]["path"], "Original selected checkpoint; SHA256 bytes only, never loaded", cp["checkpoint"])
    require(len(selected_cp) == 20, "Selected checkpoint grid incomplete")
    for architecture, seed in product(["resnet", "tcn"], cfg["scope"]["p0"]["train_seeds"]):
        require(selected_cp[(STRATEGIES[0], architecture, seed)]["checkpoint"]["sha256"] != selected_cp[(STRATEGIES[1], architecture, seed)]["checkpoint"]["sha256"], "Different training strategies share checkpoint")
    selected_case_ids = {c["case_id"] for c in cases} | {"clean"}
    pred_index = {}
    for entry in ledger["predictions"]:
        if (entry["strategy"], entry["model"], int(entry["seed"])) in selected_cp and entry["case_id"] in selected_case_ids:
            key = prediction_key(entry["strategy"], entry["model"], entry["seed"], entry["case_id"])
            require(key not in pred_index, "Duplicate prediction ledger key")
            pred_index[key] = entry
            add(entry["path"], "Original selected probabilities, IDs, labels, input/model/config identity", entry)
    require(len(pred_index) == 4020, "Probability availability gap: require 4000 noisy + 20 clean files")
    write_json(HERE / "sources.json", {"frozen_at": now(), "before_deriving_values": True, "sources": list(sources.values())})
    write_json(HERE / "source_freeze_audit.json", {"source_count": len(sources), "source_bytes": sum(s["bytes"] for s in sources.values()), "frozen_before_derivation": True})
    with np.load(resolve(boot["cohort"]["path"]), allow_pickle=False) as z:
        cohort = {key: z[key] for key in z.files}
    with np.load(resolve(manifest["cohort_file"]["path"]), allow_pickle=False) as z:
        for field in ("ids", "patient_ids", "indices", "y"):
            require(np.array_equal(z[field], cohort[field]), f"Input/bootstrap cohort conflict {field}")
    require(manifest["data_identity"]["class_order"] == evaluation["data_identity"]["class_order"] == cfg["class_order"], "Class order conflict")
    require(manifest["data_identity"] == evaluation["data_identity"], "Clean data provenance conflict")
    require(np.array_equal(cohort["unique_patients"][cohort["patient_inverse"]], cohort["patient_ids"]), "Patient inverse map conflict")
    require(len(np.unique(cohort["ids"])) == len(cohort["ids"]) == 2158 and len(cohort["unique_patients"]) == 1877, "Original cohort cardinality conflict")
    draws = np.load(resolve(boot["draws"]["path"]), mmap_mode="r", allow_pickle=False)
    require(draws.shape == (2000, 1877) and np.issubdtype(draws.dtype, np.integer) and np.all(draws >= 0) and np.all(draws.sum(axis=1) == 1877), "Original draw multiplicity conflict")
    analysis["n_records"], analysis["n_patients"] = 2158, 1877
    return finish_prepare(analysis, sources, manifest, evaluation, ledger, boot, cases, case_map, groups, selected_cp, pred_index, cohort, draws, signals, began)


def finish_prepare(analysis, sources, manifest, evaluation, ledger, boot, cases, case_map, groups, checkpoints, predictions, cohort, draws, signals, began):
    analysis["source_paths"]["signals"] = signals
    n = len(cohort["ids"])
    x = np.load(resolve(signals), mmap_mode="r", allow_pickle=False)
    require(x.dtype == np.float32 and x.shape[1:] == (12, 1000), "Clean waveform schema conflict")
    clean_case = next(c for c in manifest["cases"] if c["case_id"] == "clean")
    clean_hashes = []
    clean_whole = hashlib.sha256()
    for start in range(0, n, 64):
        batch = np.array(x[cohort["indices"][start:start + 64]], dtype=np.float32, copy=True)
        batch /= float(manifest["scale_mv"])
        clean_whole.update(memoryview(np.ascontiguousarray(batch)).cast("B"))
        clean_hashes.extend(array_hash(row) for row in batch)
    require(clean_whole.hexdigest() == clean_case["input_sha256"], "Reconstructed clean normalized input hash conflict")
    identity_path = HERE / "record_input_identity.parquet"
    writer = None
    seed_words_by_base = {}
    try:
        for case in sorted(cases, key=lambda c: c["case_id"]):
            with np.load(resolve(case["diagnostics_path"]), allow_pickle=False) as z:
                for field in ("ids", "patient_ids", "indices"):
                    require(np.array_equal(z[field], cohort[field]), f"Diagnostic identity conflict {case['case_id']}/{field}")
                require(z["noise_seed_words"].shape == (n, 4), "Missing original per-record noise keys")
                words = z["noise_seed_words"]
                if case["noise_seed"] in seed_words_by_base:
                    require(np.array_equal(words, seed_words_by_base[case["noise_seed"]]), "Original noise-base patient keys differ across structures/combinations")
                else:
                    seed_words_by_base[case["noise_seed"]] = words
            base = np.load(resolve(case["base_noise_path"]), mmap_mode="r", allow_pickle=False)
            require(base.shape == (n, 12, 1000) and base.dtype == np.float32, "Noise basis shape/dtype conflict")
            hashes, whole = [], hashlib.sha256()
            for start in range(0, n, 64):
                end = min(start + 64, n)
                batch = np.array(x[cohort["indices"][start:end]], dtype=np.float32, copy=True)
                batch += np.asarray(base[start:end], dtype=np.float32) * np.float32(10.0 ** (-float(case["snr"]) / 20.0))
                batch /= float(manifest["scale_mv"])
                require(np.isfinite(batch).all(), "Nonfinite reconstructed final input")
                whole.update(memoryview(np.ascontiguousarray(batch)).cast("B"))
                hashes.extend(array_hash(row) for row in batch)
            require(whole.hexdigest() == case["input_sha256"], f"Reconstructed final noisy input hash conflict: {case['case_id']}")
            table = pa.Table.from_pydict({"case_id": [case["case_id"]] * n, "record_id": cohort["ids"], "patient_id": cohort["patient_ids"].astype(np.int64), "record_index": cohort["indices"], "clean_input_sha256": clean_hashes, "noisy_input_sha256": hashes})
            if writer is None:
                writer = pq.ParquetWriter(identity_path, table.schema, compression="zstd", use_dictionary=True)
            writer.write_table(table)
            del base
    finally:
        if writer is not None:
            writer.close()
    del x
    selected_ids = {c["case_id"] for c in cases} | {"clean"}
    original = read_selected_metrics(METRICS, selected_ids)
    require(len(original) == 4020 and not original.duplicated(["strategy", "model", "seed", "case_id"]).any(), "Original case metric coverage conflict")
    original = original.set_index(["strategy", "model", "seed", "case_id"])
    AUC = load_auc_class()
    point_checks, alignment = [], []
    for key, entry in sorted(predictions.items()):
        strategy, architecture, seed, case_id = key
        case = clean_case if case_id == "clean" else next(c for c in cases if c["case_id"] == case_id)
        cp = checkpoints[(strategy, architecture, seed)]
        with np.load(resolve(entry["path"]), allow_pickle=False) as z:
            for field in ("ids", "patient_ids", "indices", "y"):
                require(np.array_equal(z[field], cohort[field]), f"Prediction cohort conflict: {entry['path']}/{field}")
            scalars = {"stage": "full", "strategy": strategy, "model": architecture, "seed": seed, "case_id": case_id, "config_sha256": manifest["config_sha256"], "matrix_sha256": manifest["matrix_sha256"], "input_sha256": case["input_sha256"], "checkpoint_sha256": cp["checkpoint"]["sha256"], "evaluation_fingerprint": ledger["evaluation_fingerprint"], **{field: case[field] for field in ("kind", "condition", "combo_id", "combo_set", "snr", "noise_seed")}}
            for field, expected in scalars.items():
                require(z[field].item() == expected, f"Prediction scalar conflict: {entry['path']}/{field}")
            require(np.array_equal(z["thresholds"], np.asarray(cp["thresholds"])), "Frozen threshold identity conflict; no threshold reselection")
            p = z["p"]
            require(p.shape == cohort["y"].shape and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), "Invalid original probability array")
            require(array_hash(p) == z["p_sha256"].item(), "Probability payload hash conflict")
            point = float(np.mean([AUC(cohort["y"][:, j], p[:, j]).evaluate(np.ones((1, n), dtype=np.float64))[0] for j in range(5)]))
        old = original.loc[key]
        for field, expected in (("prediction_sha256", entry["sha256"]), ("checkpoint_sha256", cp["checkpoint"]["sha256"]), ("input_sha256", case["input_sha256"]), ("prediction_path", entry["path"])):
            require(old[field] == expected, f"Case metric ledger identity conflict: {key}/{field}")
        for field in ("kind", "condition", "combo_id", "combo_set", "snr", "noise_seed"):
            require(old[field] == case[field], f"Case metric descriptor conflict: {key}/{field}")
        error = abs(point - float(old["macro_auroc"]))
        require(error <= TOL, f"Original point AUROC mismatch {key}: {error}")
        row = {"architecture": architecture, "snr_db": int(case["snr"]), "train_seed": seed, "train_strategy": strategy, "test_structure": case["condition"], "combo_id": case["combo_id"], "noise_seed": int(case["noise_seed"]), "case_id": case_id, "prediction_path": entry["path"], "prediction_sha256": entry["sha256"], "checkpoint_path": cp["checkpoint"]["path"], "checkpoint_sha256": cp["checkpoint"]["sha256"], "clean_input_sha256": clean_case["input_sha256"], "noisy_input_sha256": case["input_sha256"], "config_sha256": manifest["config_sha256"], "matrix_sha256": manifest["matrix_sha256"], "record_identity_path": relative(identity_path), "diagnostics_path": case["diagnostics_path"], "cohort_path": boot["cohort"]["path"], "patient_draws_path": boot["draws"]["path"], "n_records": n, "n_patients": 1877, "class_order": json.dumps(analysis["class_order"]), "unit": "fraction"}
        point_checks.append({**row, "recomputed_macro_auroc": point, "original_macro_auroc": float(old["macro_auroc"]), "absolute_error": error, "passed": True})
        if case_id != "clean":
            alignment.append(row)
    checks = pd.DataFrame(point_checks)
    checks.to_csv(HERE / "point_checks.csv", index=False)
    aligned = pd.DataFrame(alignment)
    require(len(aligned) == 4000, "Four-arm prediction coverage conflict")
    aligned.to_parquet(HERE / "four_arm_alignment.parquet", index=False)
    # Recompute registered absolute groups from exact per-case point values.
    selected_groups = dict(groups)
    primary = [g for g in manifest["groups"] if g["group_id"] == "primary_joint"]
    require(len(primary) == 1 and len(primary[0]["case_ids"]) == 100 and set(primary[0]["case_ids"]) == {c["case_id"] for c in cases if c["condition"] == "electrode"}, "Original report primary group membership conflict")
    selected_groups["primary_joint"] = primary[0]
    oldgroups = read_selected_metrics(GROUP_METRICS, selected_groups, groups=True)
    require(len(oldgroups) == 100 and not oldgroups.duplicated(["model", "strategy", "seed", "group_id"]).any(), "Unrounded absolute group metric coverage conflict")
    group_checks = []
    for row in oldgroups.to_dict("records"):
        g = selected_groups[row["group_id"]]
        for field in ("kind", "combo_set", "combo_id", "condition", "snr"):
            require(row[field] == g[field], f"Group metric descriptor conflict {row['group_id']}/{field}")
        require(row["n_cases"] == len(g["case_ids"]) and row["n_noise_seeds"] == 5, "Group metric weights/count conflict")
        selected = checks[(checks.architecture == row["model"]) & (checks.train_strategy == row["strategy"]) & (checks.train_seed == row["seed"]) & checks.case_id.isin(g["case_ids"])]
        require(len(selected) == len(g["case_ids"]), "Group point membership coverage conflict")
        value = float(selected.recomputed_macro_auroc.mean())
        error = abs(value - row["value"])
        require(error <= TOL, f"Original group AUROC mismatch: {row['group_id']}")
        group_checks.append({**row, "recomputed_value": value, "absolute_error": error, "passed": True})
    group_frame = pd.DataFrame(group_checks)
    group_frame.to_csv(HERE / "group_point_checks.csv", index=False)
    report = resolve(REPORT).read_text(encoding="utf-8")
    report_checks = []
    for architecture, strategy in product(["resnet", "tcn"], STRATEGIES):
        seed_points = group_frame[(group_frame.model == architecture) & (group_frame.strategy == strategy) & (group_frame.group_id == "primary_joint")].sort_values("seed").recomputed_value.to_numpy()
        label = "independent-RMS" if strategy == "independent_rms" else strategy
        matching = re.findall(r"^\| " + re.escape(architecture) + r" \| " + re.escape(label) + r" \| ([0-9.]+) ± ([0-9.]+) \|", report, flags=re.MULTILINE)
        require(len(matching) >= 1, "Original report primary numeric cells not found")
        shown_mean, shown_sd = matching[0]
        digits_mean = len(shown_mean.split(".")[1])
        digits_sd = len(shown_sd.split(".")[1])
        mean, sd = float(seed_points.mean()), float(seed_points.std(ddof=1))
        require(f"{mean:.{digits_mean}f}" == shown_mean and f"{sd:.{digits_sd}f}" == shown_sd, f"Original report display precision mismatch {architecture}/{strategy}")
        report_checks.append({"architecture": architecture, "strategy": strategy, "source_group": "primary_joint", "recomputed_mean": mean, "recomputed_seed_sd": sd, "report_mean": shown_mean, "report_sd": shown_sd, "passed": True})
    for unit_name in ("qa_unit_A", "qa_unit_B"):
        unit = analysis[unit_name]
        unit["sources"] = {}
        for arm, (test, train) in zip(ARMS, product(STRATEGIES, STRATEGIES)):
            entry = predictions[(train, unit["architecture"], unit["train_seed"], unit["paired_case_ids"][test])]
            unit["sources"][arm] = {"prediction_path": entry["path"], "prediction_sha256": entry["sha256"], "checkpoint_sha256": checkpoints[(train, unit["architecture"], unit["train_seed"])]["checkpoint"]["sha256"]}
    analysis.update(status="prepared", source_status="P0-COMPLETE", point_checks={"count": len(checks), "selected_noisy": 4000, "clean_controls": 20, "max_error": float(checks.absolute_error.max()), "group_checks": len(group_checks), "max_group_error": float(group_frame.absolute_error.max())}, original_report_checks=report_checks, coverage={"expected_predictions": 4000, "observed_predictions": len(aligned), "ratio": 1.0, "record_input_rows": 200 * n, "four_arm_common_identity_passed": True, "within_test_noisy_identity_passed": True, "within_strategy_checkpoint_identity_passed": True, "different_strategy_checkpoints_distinct": True}, prepare_elapsed_seconds=time.perf_counter() - began)
    write_json(HERE / "did_analysis.json", analysis)
    write_json(HERE / "status.json", {"status": "P0-COMPLETE", "stage": "prepared", "availability_decided_at": now(), "availability_elapsed_seconds": time.perf_counter() - began, "identity_verified": True, "probability_coverage": 1.0, "patient_draws_available": 2000, "analysis_complete": False, "independent_qa": "pending", "note": "All required original data verified; bootstrap production not yet executed"})
    return analysis


def shard(job):
    architecture, snr, seed, analysis = job
    AUC = load_auc_class()
    with np.load(resolve(analysis["source_paths"]["cohort"]), allow_pickle=False) as z:
        y, inverse = z["y"], z["patient_inverse"]
    draws = np.load(resolve(analysis["source_paths"]["patient_draws"]), mmap_mode="r", allow_pickle=False)
    alignment = pd.read_parquet(HERE / "four_arm_alignment.parquet")
    alignment = alignment[(alignment.architecture == architecture) & (alignment.snr_db == snr) & (alignment.train_seed == seed)]
    links = {(r.test_structure, r.train_strategy, r.combo_id, r.noise_seed): r for r in alignment.itertuples(index=False)}
    sums = np.zeros((2001, 7), dtype=np.float64)
    long_rows, qa = [], {}
    ones = np.ones((1, len(y)), dtype=np.float64)
    n_cases = 0
    for combo, noise in product(analysis["heldout_combinations"], analysis["scope"]["noise_seeds"]):
        prepared = []
        for test, train in product(STRATEGIES, STRATEGIES):
            row = links[(test, train, combo, noise)]
            require(file_info(row.prediction_path)["sha256"] == row.prediction_sha256, "Probability source changed after preparation")
            with np.load(resolve(row.prediction_path), allow_pickle=False) as z:
                p = z["p"]
            prepared.append([AUC(y[:, j], p[:, j]) for j in range(5)])
        arm = np.empty((2001, 4), dtype=np.float64)
        for j in range(4):
            arm[0, j] = np.mean([a.evaluate(ones)[0] for a in prepared[j]])
        for start in range(0, 2000, 128):
            stop = min(start + 128, 2000)
            weights = draws[start:stop, inverse].astype(np.float64)
            for j in range(4):
                arm[start + 1:stop + 1, j] = np.mean([a.evaluate(weights) for a in prepared[j]], axis=0)
        # Differences exist at the lowest matched case, before any case mean.
        case_values = values(arm)
        sums += case_values
        n_cases += 1
        long_rows.append({"architecture": architecture, "snr_db": snr, "train_seed": seed, "combo_id": combo, "noise_seed": noise, "combo_set": "heldout", "metric": "absolute macro-AUROC", "unit": "fraction", **dict(zip(FIELDS, case_values[0]))})
        for unit_name in ("qa_unit_A", "qa_unit_B"):
            unit = analysis[unit_name]
            if (architecture, snr, seed, combo, noise) == (unit["architecture"], unit["snr_db"], unit["train_seed"], unit["combo_id"], unit["noise_seed"]):
                qa[unit_name] = [{"draw_id": d, **json_values(case_values[d + 1]), "valid": bool(np.isfinite(case_values[d + 1]).all())} for d in (-1, 0, 1999)]
    require(n_cases == 50, "Seed shard case count conflict")
    result = sums / 50.0
    shard_path = HERE / "shards" / f"{architecture}_s{snr}_seed{seed}.parquet"
    shard_path.parent.mkdir(exist_ok=True)
    frame = pd.DataFrame(result, columns=FIELDS)
    frame.insert(0, "draw_id", np.arange(-1, 2000))
    frame["architecture"], frame["snr_db"], frame["train_seed"] = architecture, snr, seed
    frame["valid"] = np.isfinite(result).all(axis=1)
    frame.to_parquet(shard_path, index=False)
    return architecture, snr, seed, result, long_rows, qa


def run(analysis):
    start = time.perf_counter()
    require(analysis["status"] == "prepared" and analysis["source_status"] == "P0-COMPLETE", "Full calculation requires completed identity preparation")
    require(config()["resources"]["p0_workers"] == 4 and config()["resources"]["p0_bootstrap_batch_size"] == 128, "Frozen resource contract changed")
    results, long_rows, qa = {}, [], {}
    jobs = [(a, d, s, analysis) for a, d, s in product(["resnet", "tcn"], [5, 15], analysis["scope"]["train_seeds"])]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for architecture, snr, seed, result, rows, checks in pool.map(shard, jobs, chunksize=1):
            results[(architecture, snr, seed)] = result
            long_rows.extend(rows)
            qa.update(checks)
    summaries, audit_rows, seed_rows = [], [], []
    for architecture, snr in product(["resnet", "tcn"], [5, 15]):
        seeds = analysis["scope"]["train_seeds"]
        stack = np.stack([results[(architecture, snr, s)] for s in seeds])
        aggregate = stack.mean(axis=0)
        # Ordinary means propagate any invalid class/case/seed. No finite-subset averaging.
        valid = np.isfinite(stack[:, 1:, :]).all(axis=(0, 2))
        gamma_draws = aggregate[1:, -1][valid]
        low, high = np.percentile(gamma_draws, [2.5, 97.5]) if len(gamma_draws) else (np.nan, np.nan)
        seed_gamma = stack[:, 0, -1]
        summaries.append({"architecture": architecture, "snr_db": snr, "combo_set": "heldout", "metric": "absolute macro-AUROC", "unit": "fraction", "n_records": 2158, "n_patients": 1877, "n_heldout_combinations": 10, "n_noise_bases": 5, **dict(zip(FIELDS, aggregate[0])), "seed_values": json.dumps({str(s): float(v) for s, v in zip(seeds, seed_gamma)}, allow_nan=False), "seed_sd": float(seed_gamma.std(ddof=1)), "patient_ci_low": low, "patient_ci_high": high, "n_draws": 2000, "n_valid_draws": int(valid.sum()), "n_invalid_draws": int((~valid).sum()), "source_status": "P0-COMPLETE"})
        for seed, distribution in [*(zip(seeds, stack)), (-1, aggregate)]:
            if seed != -1:
                seed_rows.append({"architecture": architecture, "snr_db": snr, "train_seed": seed, "unit": "fraction", **dict(zip(FIELDS, distribution[0]))})
            frame = pd.DataFrame(distribution[1:], columns=FIELDS)
            frame["architecture"], frame["snr_db"], frame["train_seed"] = architecture, snr, seed
            frame["draw_id"] = np.arange(2000)
            frame["valid"] = np.isfinite(distribution[1:]).all(axis=1)
            frame["unit"] = "fraction"
            audit_rows.append(frame)
    pd.DataFrame(long_rows).sort_values(["architecture", "snr_db", "train_seed", "combo_id", "noise_seed"]).to_csv(HERE / "did_long.csv", index=False)
    pd.DataFrame(summaries).to_csv(HERE / "did_summary.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(HERE / "did_seed_values.csv", index=False)
    audit = pd.concat(audit_rows, ignore_index=True)
    require(len(audit) == 48000 and len(long_rows) == 1000 and len(summaries) == 4, "Production output coverage conflict")
    audit.to_parquet(HERE / "did_draw_audit.parquet", index=False)
    analysis.update(status="completed", qa_case_checks=qa, bootstrap_elapsed_seconds=time.perf_counter() - start, completed_at=now(), production={"workers": 4, "batch_size": 128, "draw_count": 2000, "summary_rows": 4, "seed_values": 20, "audit_rows": len(audit), "long_rows": len(long_rows), "fresh_case_level_computation": True, "no_old_marginal_bootstrap_arrays_consumed": True})
    write_json(HERE / "did_analysis.json", analysis)
    status = read_json(HERE / "status.json")
    status.update(stage="completed", analysis_complete=True, completed_at=now(), bootstrap_elapsed_seconds=time.perf_counter() - start, independent_qa="pending", note="Identity preparation and fresh case-level bootstrap production completed; independent acceptance is recorded separately in qa/did_recompute.json")
    write_json(HERE / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true", help="Freeze all sources, input identities and complete point/group/report checks; no bootstrap")
    mode.add_argument("--run", action="store_true", help="Prepare and then compute complete 2000-draw DiD")
    args = parser.parse_args()
    try:
        analysis = prepare()
        if args.run:
            run(analysis)
    except Exception as exc:
        # Identity/metric conflicts prohibit all new DiD outputs, including stale runs.
        for name in ("did_long.csv", "did_summary.csv", "did_draw_audit.parquet", "did_seed_values.csv"):
            path = HERE / name
            if path.exists():
                path.unlink()
        HERE.mkdir(parents=True, exist_ok=True)
        write_json(HERE / "status.json", {"status": "P0-NO-GO", "stage": "blocked", "error": str(exc), "exception": type(exc).__name__, "recorded_at": now(), "analysis_complete": False, "new_did_values_released": False})
        write_json(HERE / "failure.json", {"error": str(exc), "traceback": traceback.format_exc(), "no_new_did_values": True})
        (HERE / "no_go_gap_report.md").write_text("# P0 gap / correction record\n\nP0-NO-GO. No new DiD values are released.\n\n" + str(exc) + "\n\nHistorical sources are unchanged. A source identity or metric mismatch pauses citation of the affected result pending correction; a missing artifact requires recovery of the immutable original, not new inference.\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
