"""Audit frozen signed-control inputs without persisting duplicate waveforms.

Each CPU task owns one (SNR, noise-seed) shard. Final tables and the completed
manifest are published only after all shards and frozen identities pass.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .common import (
    WORKSPACE, array_sha256, case_grid, check_info, file_info, load_config,
    materialize_input, read_json, require_freeze, resolve_path, save_json,
    scaled_noise, sha256, stage_paths, write_csv,
)


AUDIT_FIELDS = (
    "case_id", "condition", "snr", "noise_seed", "mode", "record_index", "ecg_id",
    "patient_id", "input_sha256", "noise_path", "noise_sha256", "baseline_case_id",
    "baseline_input_sha256", "clean_sha256", "sign_controls_sha256",
    "achieved_snr_db", "snr_abs_error_db", "designed_snr_db", "designed_snr_abs_error_db",
    "max_lead_rms_relative_error", "max_designed_lead_rms_relative_error",
    "max_periodogram_peak_relative_error", "max_designed_periodogram_peak_relative_error",
    "max_base_sign_absolute_error", "max_scaled_sign_absolute_error",
    "zero_clean_power", "zero_actual_noise_power", "zero_designed_noise_power",
    "zero_reference_rms_leads", "zero_designed_reference_rms_leads",
    "zero_reference_psd_leads", "zero_designed_reference_psd_leads",
    "snr_pass", "rms_pass", "psd_applicable", "psd_pass", "base_sign_applicable",
    "base_sign_pass", "all_pass",
)
Q_FIELDS = (
    "object", "case_id", "condition", "snr", "noise_seed", "mode", "record_index",
    "ecg_id", "patient_id", "q", "zero_norm", "input_sha256", "noise_path", "noise_sha256",
)
MAX_FIELDS = (
    "snr_abs_error_db", "designed_snr_abs_error_db", "max_lead_rms_relative_error",
    "max_designed_lead_rms_relative_error", "max_periodogram_peak_relative_error",
    "max_designed_periodogram_peak_relative_error", "max_base_sign_absolute_error",
    "max_scaled_sign_absolute_error",
)


def _nullable(value):
    return float(value) if np.isfinite(value) else None


def _energy(value):
    return np.einsum("nlt,nlt->n", value, value, dtype=np.float64, optimize=False)


def _q(value, projection):
    value = np.asarray(value, dtype=np.float64)
    denominator = _energy(value)
    projected = projection @ value
    result = np.full(len(value), np.nan, dtype=np.float64)
    np.divide(_energy(projected), denominator, out=result, where=denominator > 0)
    return result, denominator == 0


def _snr(clean_power, noise):
    power = _energy(noise) / (noise.shape[1] * noise.shape[2])
    valid = (clean_power > 0) & (power > 0)
    result = np.full(len(noise), np.nan, dtype=np.float64)
    result[valid] = 10 * np.log10(clean_power[valid] / power[valid])
    return result, power == 0


def _rms(value):
    value = np.asarray(value, dtype=np.float64)
    return np.sqrt(np.einsum("nlt,nlt->nl", value, value, optimize=False) / value.shape[-1])


def _relative_error(candidate, reference):
    """Per-record worst relative error; zero references require exact zeros."""
    zero = reference == 0
    difference = np.abs(candidate - reference)
    errors = np.zeros_like(reference, dtype=np.float64)
    np.divide(difference, reference, out=errors, where=~zero)
    errors[zero & (difference != 0)] = np.inf
    return errors.max(axis=1), zero.sum(axis=1)


def _periodogram(value, fs):
    # Same rectangular, full-record, one-sided spectrum as the legacy audit,
    # but retain each lead and every bin (including DC) for the signed gate.
    spectrum = np.fft.rfft(value, axis=-1)
    power = np.abs(spectrum) ** 2
    power /= float(fs) * value.shape[-1]
    power[..., 1:-1 if value.shape[-1] % 2 == 0 else None] *= 2
    return power


def _psd_error(candidate, reference):
    peak = reference.max(axis=-1)
    difference = np.abs(candidate - reference).max(axis=-1)
    zero = peak == 0
    errors = np.zeros_like(peak)
    np.divide(difference, peak, out=errors, where=~zero)
    errors[zero & (difference != 0)] = np.inf
    return errors.max(axis=1), zero.sum(axis=1)


def _summary(values, metadata):
    finite = values[np.isfinite(values)]
    row = dict(metadata, n_valid=len(finite), n_null=len(values) - len(finite))
    row.update(dict.fromkeys(("mean", "sd", "median", "q1", "q3", "iqr", "min", "max")))
    if len(finite):
        q1, median, q3 = np.percentile(finite, [25, 50, 75])
        row.update(mean=float(finite.mean()), sd=float(finite.std(ddof=1)) if len(finite) > 1 else None,
                   median=float(median), q1=float(q1), q3=float(q3), iqr=float(q3 - q1),
                   min=float(finite.min()), max=float(finite.max()))
    return row


def _sync(handle):
    handle.flush()
    os.fsync(handle.fileno())


def _close_maps(*arrays):
    for array in arrays:
        if isinstance(array, np.memmap):
            array._mmap.close()


def _write_q(writer, values, zero, case, object_name, start, ids, patients, hashes):
    for offset, value in enumerate(values):
        index = start + offset
        writer.writerow({
            "object": object_name, "case_id": case["case_id"], "condition": case["condition"],
            "snr": case.get("snr"), "noise_seed": case.get("noise_seed"), "mode": case.get("mode", ""),
            "record_index": index, "ecg_id": int(ids[index]), "patient_id": int(patients[index]),
            "q": _nullable(value), "zero_norm": bool(zero[offset]), "input_sha256": hashes[offset],
            "noise_path": case.get("noise_path"), "noise_sha256": case.get("noise_sha256"),
        })


def _group_job(job):
    started = time.perf_counter()
    cfg, cases = job["cfg"], job["cases"]
    folder = Path(job["folder"])
    audit_path = folder / "input_validation.csv"
    q_path = folder / "subspace_diagnostics_per_record.csv"
    folder.mkdir()
    with np.load(job["cohort_path"], allow_pickle=False) as cohort:
        ids, patients = cohort["ids"], cohort["patient_ids"]
    clean = np.load(job["clean_path"], mmap_mode="r", allow_pickle=False)
    e_case = next(case for case in cases if case["condition"] == "E")
    i_case = next(case for case in cases if case["condition"] == "I")
    e_base = np.load(resolve_path(e_case["noise_path"]), mmap_mode="r", allow_pickle=False)
    i_base = np.load(resolve_path(i_case["noise_path"]), mmap_mode="r", allow_pickle=False)
    hashes = {case["case_id"]: hashlib.sha256() for case in cases}
    q_values = {}
    for case in cases:
        for suffix in ("noise", "input"):
            q_values[(case["case_id"], suffix)] = np.empty(len(ids), dtype=np.float64)
    maxima = dict.fromkeys(MAX_FIELDS, 0.0)
    gates, fs = cfg["gates"], cfg["sampling_rate"]
    n_rows = 0
    try:
        if any(base.dtype != np.float32 or base.shape != clean.shape for base in (e_base, i_base)):
            raise ValueError("Frozen noise precision/shape differs from clean")
        with audit_path.open("w", encoding="utf-8", newline="") as audit_handle, q_path.open(
                "w", encoding="utf-8", newline="") as q_handle:
            audit = csv.DictWriter(audit_handle, fieldnames=AUDIT_FIELDS)
            q_writer = csv.DictWriter(q_handle, fieldnames=Q_FIELDS)
            audit.writeheader()
            q_writer.writeheader()
            for start in range(0, len(ids), cfg["execution"]["input_batch_size"]):
                stop = min(start + cfg["execution"]["input_batch_size"], len(ids))
                signal = np.asarray(clean[start:stop])
                signal64 = signal.astype(np.float64)
                standard = np.asarray(e_base[start:stop])
                independent = np.asarray(i_base[start:stop])
                if not all(np.isfinite(array).all() for array in (signal, standard, independent)):
                    raise ValueError("Nonfinite frozen clean/base input")
                clean_power = _energy(signal64) / (12 * cfg["sequence_length"])
                e_designed = scaled_noise(standard, e_case)
                e_input = materialize_input(signal, standard, e_case)
                e_residual = e_input.astype(np.float64) - signal64
                e_rms, e_designed_rms = _rms(e_residual), _rms(e_designed)
                e_psd, e_designed_psd = _periodogram(e_residual, fs), _periodogram(e_designed, fs)
                for case in cases:
                    signed = case["condition"].startswith("S_")
                    base = independent if case["condition"] == "I" else standard
                    designed = e_designed if case["condition"] == "E" else scaled_noise(base, case)
                    final = e_input if case["condition"] == "E" else materialize_input(signal, base, case)
                    residual = e_residual if case["condition"] == "E" else final.astype(np.float64) - signal64
                    if not np.isfinite(final).all():
                        raise ValueError(f"Nonfinite final input: {case['case_id']}")
                    hashes[case["case_id"]].update(memoryview(np.ascontiguousarray(final)).cast("B"))
                    record_hashes = [array_sha256(record) for record in final]
                    achieved, zero_actual = _snr(clean_power, residual)
                    designed_snr, zero_designed = _snr(clean_power, designed)
                    snr_error = np.abs(achieved - float(case["snr"]))
                    designed_snr_error = np.abs(designed_snr - float(case["snr"]))
                    rms_error, zero_rms = _relative_error(_rms(residual), e_rms)
                    designed_rms_error, zero_designed_rms = _relative_error(_rms(designed), e_designed_rms)
                    psd_applicable = case["condition"] != "I"
                    psd_error = designed_psd_error = np.full(len(signal), np.nan)
                    zero_psd = zero_designed_psd = [None] * len(signal)
                    if psd_applicable:
                        psd_error, zero_psd = _psd_error(_periodogram(residual, fs), e_psd)
                        designed_psd_error, zero_designed_psd = _psd_error(_periodogram(designed, fs), e_designed_psd)
                    base_error = scaled_error = np.full(len(signal), np.nan)
                    if signed:
                        signs = np.asarray(case["sign_vector"], dtype=np.float32)[None, :, None]
                        signed_base = np.multiply(standard, signs, dtype=np.float32)
                        # Inverse signs prove elementwise correspondence to the original base.
                        base_error = np.max(np.abs(signed_base * signs - standard), axis=(1, 2))
                        scaled_error = np.max(np.abs(designed - e_designed * signs), axis=(1, 2))
                        expected = np.multiply(signed_base, np.float32(10 ** (-float(case["snr"]) / 20)), dtype=np.float32)
                        if not np.array_equal(expected, designed):
                            raise RuntimeError("Signed scaling differs from the frozen Standard scaling")
                    snr_pass = np.isfinite(snr_error) & np.isfinite(designed_snr_error)
                    snr_pass &= (snr_error <= gates["snr_abs_error_db"]) & (designed_snr_error <= gates["snr_abs_error_db"])
                    rms_pass = (rms_error <= gates["lead_rms_relative_error"]) & (designed_rms_error <= gates["lead_rms_relative_error"])
                    psd_pass = (psd_error <= gates["periodogram_tolerance"]) & (designed_psd_error == 0) if psd_applicable else np.ones(len(signal), dtype=bool)
                    base_pass = (base_error == 0) & (scaled_error == 0) if signed else np.ones(len(signal), dtype=bool)
                    passed = snr_pass & rms_pass & psd_pass & base_pass
                    observed = {
                        "snr_abs_error_db": snr_error, "designed_snr_abs_error_db": designed_snr_error,
                        "max_lead_rms_relative_error": rms_error, "max_designed_lead_rms_relative_error": designed_rms_error,
                        "max_periodogram_peak_relative_error": psd_error,
                        "max_designed_periodogram_peak_relative_error": designed_psd_error,
                        "max_base_sign_absolute_error": base_error, "max_scaled_sign_absolute_error": scaled_error,
                    }
                    for field, values in observed.items():
                        finite = values[np.isfinite(values)]
                        if len(finite):
                            maxima[field] = max(maxima[field], float(finite.max()))
                    for offset in range(len(signal)):
                        index = start + offset
                        row = {key: case[key] for key in (
                            "case_id", "condition", "snr", "noise_seed", "mode", "noise_path", "noise_sha256",
                            "baseline_case_id", "baseline_input_sha256")}
                        row.update(record_index=index, ecg_id=int(ids[index]), patient_id=int(patients[index]),
                                   input_sha256=record_hashes[offset], clean_sha256=job["clean_sha256"],
                                   sign_controls_sha256=job["sign_controls_sha256"],
                                   achieved_snr_db=_nullable(achieved[offset]), designed_snr_db=_nullable(designed_snr[offset]),
                                   zero_clean_power=bool(clean_power[offset] == 0), zero_actual_noise_power=bool(zero_actual[offset]),
                                   zero_designed_noise_power=bool(zero_designed[offset]),
                                   zero_reference_rms_leads=int(zero_rms[offset]), zero_designed_reference_rms_leads=int(zero_designed_rms[offset]),
                                   zero_reference_psd_leads=int(zero_psd[offset]) if psd_applicable else None,
                                   zero_designed_reference_psd_leads=int(zero_designed_psd[offset]) if psd_applicable else None,
                                   snr_pass=bool(snr_pass[offset]), rms_pass=bool(rms_pass[offset]),
                                   psd_applicable=psd_applicable, psd_pass=bool(psd_pass[offset]),
                                   base_sign_applicable=signed, base_sign_pass=bool(base_pass[offset]), all_pass=bool(passed[offset]))
                        row.update({field: _nullable(values[offset]) for field, values in observed.items()})
                        audit.writerow(row)
                    n_rows += len(signal)
                    if not passed.all():
                        first = int(np.flatnonzero(~passed)[0])
                        detail = {key: _nullable(value[first]) for key, value in observed.items()}
                        raise RuntimeError(f"Input gate failed: {case['case_id']}, ECG={ids[start + first]}, observed={detail}")
                    prefix = "S" if signed else case["condition"]
                    for suffix, value in (("noise", residual), ("input", final)):
                        q, zero = _q(value, job["Q"])
                        q_values[(case["case_id"], suffix)][start:stop] = q
                        _write_q(q_writer, q, zero, case, f"{prefix}_{suffix}", start, ids, patients, record_hashes)
            _sync(audit_handle)
            _sync(q_handle)
    finally:
        _close_maps(clean, e_base, i_base)
    summaries, completed_cases = [], []
    for case in cases:
        digest = hashes[case["case_id"]].hexdigest()
        if case["reuse_eligible"] and digest != case["baseline_input_sha256"]:
            raise RuntimeError(f"Legacy complete input identity mismatch: {case['case_id']}")
        completed_cases.append(dict(case, input_sha256=digest,
                                    legacy_input_hash_verified=bool(case["reuse_eligible"])))
        for suffix in ("noise", "input"):
            prefix = "S" if case["condition"].startswith("S_") else case["condition"]
            summaries.append(_summary(q_values[(case["case_id"], suffix)], {
                "object": f"{prefix}_{suffix}", "snr": case["snr"], "noise_seed": case["noise_seed"], "mode": case["mode"],
            }))
    return {"cases": completed_cases, "summaries": summaries, "maxima": maxima,
            "audit_path": str(audit_path), "q_path": str(q_path), "n_rows": n_rows,
            "seconds": time.perf_counter() - started}


def _merge_csv(destination, sources, fields):
    """Concatenate bounded-memory shards; each header is checked before copying."""
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(fields)
        for path in sources:
            with Path(path).open("r", encoding="utf-8", newline="") as source:
                if next(csv.reader(source)) != list(fields):
                    raise RuntimeError(f"Shard schema mismatch: {path}")
                shutil.copyfileobj(source, output, length=1024 * 1024)
        _sync(output)


def _clean_diagnostics(clean_path, cohort_path, q_matrix, batch_size, destination):
    with np.load(cohort_path, allow_pickle=False) as cohort:
        ids, patients = cohort["ids"], cohort["patient_ids"]
    clean = np.load(clean_path, mmap_mode="r", allow_pickle=False)
    values = np.empty(len(ids), dtype=np.float64)
    digest = hashlib.sha256()
    try:
        if clean.dtype != np.float32 or clean.shape != (len(ids), 12, 1000) or not len(ids):
            raise ValueError("Frozen clean must be nonempty float32[N,12,1000]")
        if len(set(ids.tolist())) != len(ids) or len(patients) != len(ids):
            raise ValueError("Frozen cohort identity/order is invalid")
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=Q_FIELDS)
            writer.writeheader()
            for start in range(0, len(ids), batch_size):
                batch = np.asarray(clean[start:start + batch_size])
                if not np.isfinite(batch).all():
                    raise ValueError("Nonfinite clean input")
                digest.update(memoryview(np.ascontiguousarray(batch)).cast("B"))
                q, zero = _q(batch, q_matrix)
                values[start:start + len(batch)] = q
                _write_q(writer, q, zero, {"case_id": "clean", "condition": "clean"}, "clean", start,
                         ids, patients, [array_sha256(record) for record in batch])
            _sync(handle)
    finally:
        _close_maps(clean)
    return _summary(values, {"object": "clean", "snr": None, "noise_seed": None, "mode": ""}), digest.hexdigest()


def _require_smoke(cfg):
    path = stage_paths(cfg, "smoke")["logs"] / "smoke_verification.json"
    report = read_json(path)
    if report.get("status") not in {"passed", "waived_by_user"} or report.get("config_sha256") != cfg["_config_sha256"]:
        raise RuntimeError("Full input generation requires passed smoke or an explicit current-protocol user waiver")
    if report["status"] == "passed":
        require_freeze(cfg, "smoke")
    return file_info(path)


def _shift_diagnostics(policy, clean_path, cohort_path, q_matrix, batch_size, folder):
    """Only frozen, directly readable full-input arrays qualify; never regenerate shifts."""
    if policy.get("status") == "not_newly_generated":
        if not policy.get("reason"):
            raise ValueError("The frozen omission of shifts needs a reason")
        return [], [], []
    if policy.get("status") != "reuse_verified" or not policy.get("cases"):
        raise ValueError("Missing or unsupported frozen circular-shift policy")
    with np.load(cohort_path, allow_pickle=False) as cohort:
        ids, patients = cohort["ids"], cohort["patient_ids"]
    clean = np.load(clean_path, mmap_mode="r", allow_pickle=False)
    summaries, paths, sources = [], [], []
    try:
        for ordinal, case in enumerate(policy["cases"]):
            if case["cohort_sha256"] != sha256(cohort_path):
                raise ValueError("Shift input cohort is not the frozen cohort")
            info = case["input"]
            shifted = np.load(check_info(info), mmap_mode="r", allow_pickle=False)
            path = folder / f"shift_{ordinal:03d}.csv"
            values, digest = np.empty(len(ids)), hashlib.sha256()
            try:
                if shifted.dtype != np.float32 or shifted.shape != clean.shape:
                    raise ValueError("Verified shift input must be a complete float32 cohort array")
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=Q_FIELDS)
                    writer.writeheader()
                    for start in range(0, len(ids), batch_size):
                        final = np.asarray(shifted[start:start + batch_size])
                        if not np.isfinite(final).all():
                            raise ValueError("Nonfinite frozen shift input")
                        digest.update(memoryview(np.ascontiguousarray(final)).cast("B"))
                        residual = final.astype(np.float64) - clean[start:start + len(final)].astype(np.float64)
                        q, zero = _q(residual, q_matrix)
                        values[start:start + len(final)] = q
                        _write_q(writer, q, zero, dict(case, condition="shift", noise_path=info["path"], noise_sha256=info["sha256"]),
                                 "shift_noise", start, ids, patients, [array_sha256(record) for record in final])
                    _sync(handle)
                if digest.hexdigest() != case["input_sha256"]:
                    raise ValueError("Frozen shift complete input identity mismatch")
            finally:
                _close_maps(shifted)
            summaries.append(_summary(values, {"object": "shift_noise", "snr": case["snr"], "noise_seed": case["noise_seed"], "mode": ""}))
            paths.append(path)
            sources.append(info)
    finally:
        _close_maps(clean)
    return summaries, paths, sources


def run(config=None, stage="full"):
    started = time.perf_counter()
    cfg = load_config(config)
    paths = stage_paths(cfg, stage)
    freeze = require_freeze(cfg, stage)
    smoke = _require_smoke(cfg) if stage == "full" else None
    for name in ("inputs", "tables", "logs"):
        paths[name].mkdir(parents=True, exist_ok=True)
    lock = paths["inputs"] / ".input-generation.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    report_path = paths["logs"] / "input_generation.json"
    manifest_path = paths["inputs"] / "manifest.json"
    provenance = {}
    try:
        # Invalidate a previous completion marker before a new attempt, including
        # attempts that fail before publication. Old scientific tables stay intact.
        manifest_path.unlink(missing_ok=True)
        required = ("clean", "cohort", "draws", "sign_controls", "matrices", "legacy_input_manifest")
        verified = {key: check_info(freeze[key]) for key in required}
        legacy_manifest = read_json(verified["legacy_input_manifest"])
        legacy_cases = {case["case_id"]: case for case in legacy_manifest["cases"]}
        sources = [file_info(Path(__file__)), file_info(Path(__file__).with_name("common.py")),
                   file_info(WORKSPACE / "methodology_supplement/common.py"),
                   file_info(WORKSPACE / "methodology_supplement/noise.py")]
        provenance = {"freeze": file_info(paths["logs"] / "freeze.json"),
                      "source_files": sources, "config_file": file_info(cfg["_config_path"]),
                      **{key: freeze[key] for key in required}, "smoke_verification": smoke}
        matrices = read_json(verified["matrices"])
        p = np.asarray(matrices["projection"]["P"], dtype=np.float64)
        q = np.asarray(matrices["projection"]["Q"], dtype=np.float64)
        tolerance = cfg["subspace"]["projection_tolerance"]
        if p.shape != (12, 12) or q.shape != (12, 12) or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("Frozen P/Q must be finite float64 12x12 projections")
        projection_errors = {"symmetry_frobenius": float(np.linalg.norm(p - p.T)),
                             "idempotence_frobenius": float(np.linalg.norm(p @ p - p)),
                             "complement_frobenius": float(np.linalg.norm(q - (np.eye(12) - p)))}
        if any(value >= tolerance for value in projection_errors.values()):
            raise ValueError(f"Frozen projection validation failed: {projection_errors}")
        cases = case_grid(cfg, stage)
        controls = read_json(verified["sign_controls"])["controls"]
        modes = [f"S_{index:02d}" for index in range(5)]
        if [control["mode"] for control in controls] != modes:
            raise ValueError("The five frozen controls changed or are out of order")
        expected = {(snr, seed, condition) for snr in (0, 5, 10) for seed in (8128, 101, 202, 303, 404, 505)
                    for condition in ("E", "I", *modes)}
        if {(case["snr"], case["noise_seed"], case["condition"]) for case in cases} != expected or len(cases) != 126:
            raise ValueError("Input cases differ from the complete fixed grid")
        base_files = {}
        for case in cases:
            case["baseline_source"] = legacy_cases[case["baseline_case_id"]]
            path = resolve_path(case["noise_path"])
            if case["noise_path"] not in base_files:
                info = file_info(path)
                if info["sha256"] != case["noise_sha256"]:
                    raise ValueError(f"Frozen Standard/independent base changed: {path}")
                base_files[case["noise_path"]] = info
            if base_files[case["noise_path"]]["sha256"] != case["noise_sha256"]:
                raise ValueError("Inconsistent base identity across frozen cases")
        provenance["base_files"] = list(base_files.values())
        batch_size = int(cfg["execution"]["input_batch_size"])
        workers = int(cfg["execution"].get("input_workers", 1))
        if batch_size < 1 or workers < 1:
            raise ValueError("Input batch size and worker count must be positive")
        with np.load(verified["cohort"], allow_pickle=False) as cohort:
            n_records = len(cohort["ids"])
            n_patients = len(np.unique(cohort["patient_ids"]))
        if n_records != freeze["n_records"] or n_patients != freeze["n_patients"]:
            raise ValueError("Frozen cohort counts changed")
        if stage == "full" and (n_records, n_patients) != (2158, 1877):
            raise ValueError("Full cohort must contain 2158 ECGs and 1877 patients")
        if stage == "smoke" and n_records != 100:
            raise ValueError("Smoke requires the existing first 100-record cohort")
        preflight_seconds = time.perf_counter() - started
        with tempfile.TemporaryDirectory(prefix=".input-audit-", dir=paths["inputs"]) as temporary:
            staging = Path(temporary)
            clean_q_path = staging / "clean_q.csv"
            clean_summary, clean_hash = _clean_diagnostics(verified["clean"], verified["cohort"], q, batch_size, clean_q_path)
            legacy = read_json(verified["legacy_input_manifest"])
            clean_cases = [case for case in legacy["cases"] if case["case_id"] == "clean"]
            if len(clean_cases) != 1 or clean_hash != clean_cases[0]["input_sha256"]:
                raise ValueError("Clean complete input identity differs from the frozen baseline")
            groups = {}
            for case in cases:
                groups.setdefault((case["snr"], case["noise_seed"]), []).append(case)
            jobs = [{"cfg": cfg, "cases": group, "folder": str(staging / f"group_{index:02d}"),
                     "cohort_path": str(verified["cohort"]), "clean_path": str(verified["clean"]), "Q": q,
                     "clean_sha256": freeze["clean"]["sha256"], "sign_controls_sha256": freeze["sign_controls"]["sha256"]}
                    for index, group in enumerate(groups.values())]
            if workers == 1:
                results = [_group_job(job) for job in jobs]
            else:
                with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
                    results = list(executor.map(_group_job, jobs))
            policy = freeze.get("shift_noise", {})
            shift_summaries, shift_paths, shift_sources = _shift_diagnostics(
                policy, verified["clean"], verified["cohort"], q, batch_size, staging)
            provenance["shift_sources"] = shift_sources
            summaries = [clean_summary, *[row for result in results for row in result["summaries"]], *shift_summaries]
            completed_cases = [case for result in results for case in result["cases"]]
            if len(completed_cases) != 126 or len({case["case_id"] for case in completed_cases}) != 126:
                raise RuntimeError("Incomplete or duplicate final input grid")
            if sum(result["n_rows"] for result in results) != 126 * n_records:
                raise RuntimeError("Incomplete per-record input audit")
            _merge_csv(staging / "input_validation.csv", [result["audit_path"] for result in results], AUDIT_FIELDS)
            _merge_csv(staging / "subspace_diagnostics_per_record.csv",
                       [clean_q_path, *[result["q_path"] for result in results], *shift_paths], Q_FIELDS)
            write_csv(staging / "subspace_diagnostics_summary.csv", summaries)
            # Recheck every mutable source before any published result is replaced.
            if require_freeze(cfg, stage) != freeze:
                raise RuntimeError("Freeze changed during input validation")
            for info in [provenance["freeze"], provenance["config_file"], *sources,
                         *[freeze[key] for key in required], *base_files.values(), *shift_sources]:
                check_info(info)
            if smoke is not None:
                check_info(smoke)
                _require_smoke(cfg)
            maxima = {field: max(result["maxima"][field] for result in results) for field in MAX_FIELDS}
            # No completed marker exists until all final tables are replaced.
            table_infos = {}
            for name in ("input_validation", "subspace_diagnostics_per_record", "subspace_diagnostics_summary"):
                destination = paths["tables"] / f"{name}.csv"
                os.replace(staging / f"{name}.csv", destination)
                table_infos[name] = file_info(destination)
            manifest = {
                "status": "completed", "stage": stage, "config_sha256": cfg["_config_sha256"],
                "n_records": n_records, "n_patients": n_patients, "n_cases": 126,
                "n_validation_rows": 126 * n_records, "n_subspace_rows": (1 + 2 * 126 + len(shift_sources)) * n_records,
                "cases": completed_cases, "clean_input_sha256": clean_hash,
                **{key: freeze[key] for key in ("clean", "cohort", "draws", "sign_controls", "matrices")},
                "audit": table_infos["input_validation"], "tables": table_infos,
                "diagnostics": {"per_record": table_infos["subspace_diagnostics_per_record"],
                                "summary": table_infos["subspace_diagnostics_summary"]},
                "provenance": provenance, "shift_policy": policy,
                "gates": {"status": "passed", **maxima, "limits": cfg["gates"],
                          "legacy_input_hashes_verified": 36, "projection_errors": projection_errors},
                "definitions": {
                    "actual_noise": "float64(final_float32_input) - float64(clean_float32); includes addition rounding",
                    "designed_noise": "frozen float32 base, signed before the unchanged Standard float32 SNR scale",
                    "psd": "Rectangular full-record one-sided per-lead periodogram, doubled interior bins; all bins including DC. Per-lead maximum bin difference / Standard lead peak, then maximum over leads. Exact zero reference requires exact zero.",
                    "q": "Float64 ||Q v||_F^2 / ||v||_F^2; zero denominator is null (empty CSV cell), never regularized. Noise q uses actual residuals; full noisy inputs also included.",
                    "summary": "Equal-record descriptive statistics per object/SNR/noise/mode; SD uses ddof=1; Q1/Q3 use NumPy linear percentiles; nulls counted and excluded, never replaced.",
                    "null_audit": "Blank signed/PSD fields on nonapplicable comparisons; explicit applicability and zero-denominator counts. Undefined SNR fails its gate.",
                    "identities": "input_sha256 hashes contiguous float32 C-order data bytes, without an NPY header; source SHA-256 hashes whole files.",
                },
            }
            report = {
                "status": "completed", "stage": stage, "config_sha256": cfg["_config_sha256"],
                "completed_at": datetime.now(timezone.utc).isoformat(), "provenance": provenance,
                "source_files": sources, "workers": min(workers, len(jobs)), "batch_size": batch_size,
                "n_records": n_records, "n_cases": 126, "n_validation_rows": 126 * n_records,
                "gates": manifest["gates"], "tables": table_infos, "shift_policy": policy,
                "timings": {"preflight_seconds": preflight_seconds,
                            "shards": [{"snr": result["cases"][0]["snr"], "noise_seed": result["cases"][0]["noise_seed"],
                                        "seconds": result["seconds"]} for result in results],
                            "total_seconds": time.perf_counter() - started},
            }
            save_json(report_path, report)
            manifest["input_generation"] = file_info(report_path)
            save_json(manifest_path, manifest)
        return manifest
    except BaseException as error:
        save_json(report_path, {"status": "failed", "stage": stage, "config_sha256": cfg["_config_sha256"],
                               "provenance": provenance, "error": f"{type(error).__name__}: {error}",
                               "seconds": time.perf_counter() - started})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
