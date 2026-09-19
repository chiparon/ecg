"""Read-only direct projected-SNR reconstruction, provenance, and legal overlays."""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
from overnight_supplements_v2.shared.common import ROOT, OUT, resolve, read_json, write_json, file_info, now
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd

DEST = OUT / "snrp"
SIGNED = "signed_control_mechanism"
PATHS = {
    "manifest": SIGNED + "/inputs/full/manifest.json",
    "freeze": SIGNED + "/logs/full/freeze.json",
    "matrices": SIGNED + "/inputs/matrices.json",
    "input_validation": SIGNED + "/tables/full/input_validation.csv",
    "q_diagnostics": SIGNED + "/tables/full/subspace_diagnostics_per_record.csv",
    "prediction_index": SIGNED + "/tables/full/prediction_index.csv",
}
TOL = dict(projection_frobenius=1e-10, pseudoinverse_rcond=1e-12,
           energy_negative_absolute=1e-12, q_bound_absolute=1e-12,
           q_crosscheck_absolute=1e-12, snr_db_absolute=1e-9, identity_db_absolute=1e-10, energy_relative=1e-12)
CONDITIONS = ["E", "I", "S_00", "S_01", "S_02", "S_03", "S_04"]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def raw_hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest, freeze = read_json(PATHS["manifest"]), read_json(PATHS["freeze"])
    paths = dict(PATHS, clean=freeze["clean"]["path"], cohort=freeze["cohort"]["path"])
    with np.load(resolve(paths["cohort"]), allow_pickle=False) as data:
        ids = data["ids"]
        require(len(ids) == 2158 and len(np.unique(ids)) == len(ids), "Invalid frozen cohort IDs")
        qa = sorted(int(x) for x in ids)[:5]
    # Written before waveform energies or any classification probabilities are inspected.
    analysis = dict(status="prepared", prepared_at=now(), qa_record_ids=qa,
        qa_selection="Five numerically smallest unique immutable cohort record IDs; fixed before any energies",
        tolerances=TOL, paths=paths, class_order=CLASSES, conditions=CONDITIONS,
        case_mapping=manifest["cases"], expected_geometry_rows=271908, expected_cases=126,
        expected_noisy_prediction_sources=756, expected_clean_prediction_sources=6,
        reconstruction="mmap immutable float32 clean/base; multiply signed base by float32 sign first, then float32(10**(-snr/20)); add float32 clean in place; never save waveforms",
        projection="float64 P=A@pinv(A,rcond=1e-12); Q=I-P; actual noise=float64(final_float32)-float64(clean_float32)",
        energy_definition="sum of squares across all lead/time samples in float64, without normalization",
        q_definition="||Qv||^2/||v||^2 by direct Q projection; null for zero total energy; no clipping",
        snr_definition="10log10(clean_projected_energy/noise_projected_energy); zero numerator has priority and is null; positive numerator and zero denominator is +inf",
        identity_definition="actual_total_snr_db+10log10((1-q_clean)/(1-q_noise)); independent consistency diagnostic, not primary estimate",
        hash_definition="clean_input_sha256 and noisy_input_sha256 are per-record SHA256 of C-contiguous raw float32 bytes; whole-case hashes separately audited; file sha256 includes .npy headers",
        primary_key=["case_id", "record_id"], anomaly_policy="No clipping, epsilon, silent zeroing or record deletion. Every anomaly has a code in anomalies.csv and joined anomaly_code; invalid geometry prevents publication, boundary zero codes do not.",
        aggregation="Every condition/mode kept separate. Distribution pools equally sized fixed noise cases without implying independence. Six-noise per-record means are descriptive only. No mode-as-random-replicate, patient CI, p value, reweighting or causal mediation.",
        boundary_inf_json_encoding="+inf string; null means undefined", source_manifest="overnight_supplements_v2/snrp/sources.json")
    write_json(DEST / "snrp_analysis.json", analysis)
    infos = {}
    expected_conflicts = []
    missing_provenance = []
    def add(path, schema, source, expected=None, optional=False):
        path = resolve(path)
        key = path.relative_to(ROOT).as_posix()
        if optional and not path.is_file():
            missing_provenance.append(key)
            return None
        if key not in infos:
            infos[key] = file_info(path, schema=schema, source=source)
        if expected and infos[key]["sha256"] != expected:
            expected_conflicts.append(dict(path=key, expected_sha256=expected, observed_sha256=infos[key]["sha256"]))
        return infos[key]
    for key, path in paths.items():
        add(path, "CSV header defines columns" if path.endswith(".csv") else "frozen JSON or NumPy array", "signed-control " + key, optional=key == "prediction_index")
    for key in ("clean", "cohort", "matrices", "sign_controls", "legacy_input_manifest", "legacy_freeze"):
        info = freeze[key]
        add(info["path"], "frozen historical source", "signed freeze " + key, info["sha256"], optional=key.startswith("legacy"))
    for case in manifest["cases"]:
        add(case["noise_path"], {"dtype": "float32", "shape": [2158, 12, 1000]}, "immutable original noise base", case["noise_sha256"])
    reports = []
    report_paths = [info["path"] for info in freeze["legacy_worker_reports"]]
    report_paths += [SIGNED + "/logs/full/inference_" + worker + ".json" for worker in ("resnet_0", "resnet_1", "tcn_0")]
    for path in report_paths:
        if add(path, "worker provenance JSON", "historical inference provenance (read only)", optional=True) is None:
            continue
        report = read_json(path)
        reports.append(dict(path=path, report=report))
        for info in report["source_fingerprints"]:
            add(info["path"], "historical source code (not imported)", "worker source fingerprint", info["sha256"], optional=True)
        for key in ("manifest_fingerprint", "freeze_fingerprint", "checkpoint_manifest_fingerprint"):
            if key in report:
                info = report[key]
                add(info["path"], "frozen provenance JSON", "worker " + key, info["sha256"], optional=True)
    for path in [SIGNED + "/logs/full/publication_verification.json", SIGNED + "/configs/signed_control_full.json", SIGNED + "/plot.py", SIGNED + "/common.py", SIGNED + "/inputs.py", SIGNED + "/merge.py", "phase2/src/plot_phase2.py", "methodology_supplement/plot.py"]:
        add(path, "historical configuration/source/verification", "protocol and plotting/reconstruction conventions")
    index = pd.read_csv(resolve(paths["prediction_index"]), keep_default_na=False) if resolve(paths["prediction_index"]).is_file() else pd.DataFrame()
    missing = []
    for row in index.to_dict("records"):
        if resolve(row["prediction_path"]).is_file():
            add(row["prediction_path"], {"format": "npz", "arrays": ["ids", "patient_ids", "indices", "y", "p", "thresholds"], "n_records": 2158, "classes": CLASSES}, "original signed/reused prediction", row["prediction_sha256"])
        else:
            missing.append(row["prediction_path"])
    write_json(DEST / "sources.json", dict(frozen_at=now(), sources=list(infos.values()), missing_prediction_paths=missing, missing_prediction_provenance=missing_provenance, expected_hash_conflicts=expected_conflicts))
    # Metadata only, never model imports or checkpoint loading/hashing.
    write_json(DEST / "worker_provenance.json", dict(reports=reports, checkpoint_validation="NPZ/index/worker/frozen checkpoint SHA256 metadata; checkpoint files never opened"))
    return analysis, infos, expected_conflicts


def energy(value):
    return np.einsum("nlt,nlt->n", value, value, dtype=np.float64, optimize=False)


def ratio_db(numerator, denominator):
    result = np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan)
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    valid = (numerator > 0) & (denominator > 0) & np.isfinite(numerator) & np.isfinite(denominator)
    result[valid] = 10 * np.log10(numerator[valid] / denominator[valid])
    result[(numerator > 0) & np.isfinite(numerator) & (denominator == 0)] = np.inf
    return result


def scalar_result(row):
    codes = []
    if row.get("record_id") is None or not row.get("case_id"):
        codes.append("MISSING_PRIMARY_KEY")
    energies = [row[x] for x in ("clean_total_energy", "noise_total_energy", "clean_projected_energy", "noise_projected_energy")]
    if any(not np.isfinite(x) for x in energies):
        codes.append("NONFINITE_ENERGY")
    if any(x < 0 for x in energies):
        codes.append("NEGATIVE_ENERGY")
    for key in ("q_clean", "q_noise"):
        value = row[key]
        if not np.isfinite(value):
            codes.append("NONFINITE_Q")
        elif value < -TOL["q_bound_absolute"] or value > 1 + TOL["q_bound_absolute"]:
            codes.append("Q_OUT_OF_BOUNDS")
    a, b = row["clean_projected_energy"], row["noise_projected_energy"]
    if a == 0:
        codes.append("ZERO_PROJECTED_CLEAN")
    elif b == 0:
        codes.append("ZERO_PROJECTED_NOISE")
    value = float(ratio_db(np.asarray(a), np.asarray(b)))
    if any(code not in {"ZERO_PROJECTED_CLEAN", "ZERO_PROJECTED_NOISE"} for code in codes):
        value = np.nan
    return dict(snr_p_db=None if np.isnan(value) else "+inf" if np.isposinf(value) else value,
                anomaly_code=";".join(sorted(set(codes))))


def boundaries():
    base = dict(record_id=1, case_id="synthetic", clean_total_energy=4., noise_total_energy=2., clean_projected_energy=2., noise_projected_energy=1., q_clean=.5, q_noise=.5)
    variants = [("regular", {}), ("zero_numerator", dict(clean_projected_energy=0., q_clean=1.)),
        ("zero_denominator", dict(noise_projected_energy=0., q_noise=1.)),
        ("both_zero", dict(clean_projected_energy=0., noise_projected_energy=0., q_clean=1., q_noise=1.)),
        ("tiny_positive_denominator", dict(noise_projected_energy=1e-30, q_noise=1.)),
        ("tiny_positive_numerator", dict(clean_projected_energy=1e-30, q_clean=1.)),
        ("negative_energy", dict(noise_projected_energy=-1.)), ("q_out_of_bounds", dict(q_noise=1.1)),
        ("missing_key", dict(record_id=None))]
    cases = []
    for name, change in variants:
        row = dict(base, **change)
        cases.append(dict(name=name, input=row, output=scalar_result(row)))
    write_json(DEST / "boundary_cases.json", dict(cases=cases))


def summarize(frame, by, metrics):
    rows = []
    for keys, group in frame.groupby(by, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(by, keys))
        for metric in metrics:
            values = group[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            row = dict(identity, metric=metric, n_rows=len(group), n_records=group.record_id.nunique(),
                       n_patients=group.patient_id.nunique(), n_noise_bases=group.noise_seed.nunique(),
                       n_finite=len(finite), n_null=int(np.isnan(values).sum()),
                       n_positive_inf=int(np.isposinf(values).sum()), n_negative_inf=int(np.isneginf(values).sum()),
                       n_anomalous=int(group.anomaly_code.ne("").sum()), unit="dB" if metric.endswith("db") else "fraction")
            for name, q in [("min", 0), ("p05", .05), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95), ("max", 1)]:
                row[name] = float(np.quantile(finite, q)) if len(finite) else np.nan
            row["mean"] = float(finite.mean()) if len(finite) else np.nan
            row["sd"] = float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def geometry(analysis, infos, conflicts):
    manifest = read_json(analysis["paths"]["manifest"])
    freeze = read_json(analysis["paths"]["freeze"])
    with np.load(resolve(analysis["paths"]["cohort"]), allow_pickle=False) as f:
        cohort = {k: f[k] for k in ("ids", "patient_ids", "indices", "y")}
    clean = np.load(resolve(analysis["paths"]["clean"]), mmap_mode="r", allow_pickle=False)
    require(clean.dtype == np.float32 and clean.shape == (2158, 12, 1000), "Clean shape/dtype mismatch")
    require(manifest["clean"] == freeze["clean"] and manifest["cohort"] == freeze["cohort"], "Manifest/freeze clean or cohort identity differs")
    require(np.isfinite(cohort["patient_ids"]).all() and np.isfinite(cohort["ids"]).all(), "Missing cohort primary keys")
    require(np.array_equal(cohort["patient_ids"], cohort["patient_ids"].astype(np.int64)), "Patient identity cannot be represented as integer")
    matrix = np.asarray(read_json(analysis["paths"]["matrices"])["standard"]["matrix"], dtype=np.float64)
    P = matrix @ np.linalg.pinv(matrix, rcond=TOL["pseudoinverse_rcond"])
    Q = np.eye(12) - P
    symmetry, idempotence = float(np.linalg.norm(P-P.T)), float(np.linalg.norm(P@P-P))
    require(symmetry < TOL["projection_frobenius"] and idempotence < TOL["projection_frobenius"], "Invalid projector")
    require(len(manifest["cases"]) == 126 and len({c["case_id"] for c in manifest["cases"]}) == 126, "Invalid case grid")
    n = len(clean)
    anomalies, audits, frames = [], [], []
    def anomaly(case_id, index, code, detail=""):
        anomalies.append(dict(case_id=case_id, record_id=int(cohort["ids"][index]) if index is not None else None,
                              record_index=index, anomaly_code=code, detail=str(detail)))
    geometry_sources = {analysis["paths"][key] for key in ("clean", "cohort", "matrices", "manifest", "input_validation", "q_diagnostics")}
    geometry_sources.update(case["noise_path"] for case in manifest["cases"])
    geometry_sources.add(freeze["sign_controls"]["path"])
    for item in conflicts:
        # Missing predictions/fingerprint failures cannot block valid waveform geometry.
        if item["path"] in geometry_sources:
            anomaly("source", None, "SOURCE_HASH_MISMATCH", item)
    clean_hashes = np.array([raw_hash(x) for x in clean])
    clean_total, clean_proj, q_clean = (np.empty(n) for _ in range(3))
    for start in range(0, n, 64):
        end = min(n, start+64)
        x = clean[start:end].astype(np.float64)
        clean_total[start:end] = energy(x)
        clean_proj[start:end] = energy(P @ x)
        q_clean[start:end] = np.divide(energy(Q @ x), clean_total[start:end], out=np.full(end-start, np.nan), where=clean_total[start:end] > 0)
    if raw_hash(clean) != manifest["clean_input_sha256"]:
        anomaly("clean", None, "CLEAN_ARRAY_HASH_MISMATCH")
    audit_columns = ["case_id", "condition", "snr", "noise_seed", "mode", "record_index", "ecg_id", "patient_id", "input_sha256", "noise_sha256", "clean_sha256", "achieved_snr_db"]
    old_inputs = pd.read_csv(resolve(analysis["paths"]["input_validation"]), usecols=audit_columns, keep_default_na=False)
    old_q = pd.read_csv(resolve(analysis["paths"]["q_diagnostics"]), keep_default_na=False,
                        dtype={"snr": "string", "noise_seed": "string"})
    old_q["q"] = pd.to_numeric(old_q.q, errors="coerce")
    require(len(old_inputs) == 271908 and len(old_q) == 545974, "Historical diagnostic grid count differs")
    old_clean = old_q[old_q.object == "clean"].sort_values("record_index")
    clean_ok = len(old_clean) == n and np.array_equal(old_clean.ecg_id.to_numpy(), cohort["ids"]) and np.array_equal(old_clean.patient_id.to_numpy(), cohort["patient_ids"])
    require(clean_ok, "Historical clean q identity mismatch")
    for index in range(n):
        if old_clean.iloc[index].input_sha256 != clean_hashes[index]:
            anomaly("clean", index, "CLEAN_RECORD_HASH_MISMATCH")
        if not np.isclose(old_clean.iloc[index].q, q_clean[index], atol=TOL["q_crosscheck_absolute"], rtol=0, equal_nan=True):
            anomaly("clean", index, "OLD_Q_CLEAN_MISMATCH")
    groups_input = {key: value.sort_values("record_index") for key, value in old_inputs.groupby("case_id", sort=False)}
    groups_q = {(key[0], key[1]): value.sort_values("record_index") for key, value in old_q[old_q.object != "clean"].groupby(["case_id", "object"], sort=False)}
    for case in manifest["cases"]:
        cid = case["case_id"]
        base = np.load(resolve(case["noise_path"]), mmap_mode="r", allow_pickle=False)
        require(base.shape == clean.shape and base.dtype == np.float32, "Noise base shape/dtype mismatch")
        total_noise, proj_noise, q_noise, q_noisy = (np.empty(n) for _ in range(4))
        record_hashes, final_digest = [], hashlib.sha256()
        for start in range(0, n, 64):
            end = min(n, start+64)
            block = np.asarray(base[start:end])
            if case["condition"].startswith("S_"):
                final = np.multiply(block, np.asarray(case["sign_vector"], dtype=np.float32)[None, :, None])
                np.multiply(final, np.float32(10**(-float(case["snr"])/20)), out=final)
            else:
                final = np.multiply(block, np.float32(10**(-float(case["snr"])/20)))
            np.add(clean[start:end], final, out=final)
            final_digest.update(memoryview(final).cast("B"))
            record_hashes.extend(raw_hash(x) for x in final)
            z = final.astype(np.float64)
            residual = z - clean[start:end].astype(np.float64)
            nt, pn = energy(residual), energy(P @ residual)
            total_noise[start:end], proj_noise[start:end] = nt, pn
            q_noise[start:end] = np.divide(energy(Q @ residual), nt, out=np.full(end-start, np.nan), where=nt > 0)
            zt = energy(z)
            q_noisy[start:end] = np.divide(energy(Q @ z), zt, out=np.full(end-start, np.nan), where=zt > 0)
        del base
        actual = ratio_db(clean_total, total_noise)
        direct = ratio_db(clean_proj, proj_noise)
        delta = ratio_db(1-q_clean, 1-q_noise)
        identity = actual + delta
        frame = pd.DataFrame(dict(record_id=cohort["ids"], patient_id=cohort["patient_ids"].astype(np.int64),
            record_index=np.arange(n), condition=case["condition"], mode=case["mode"], snr_db=case["snr"],
            noise_seed=case["noise_seed"], case_id=cid, clean_input_sha256=clean_hashes,
            noisy_input_sha256=record_hashes, clean_total_energy=clean_total, noise_total_energy=total_noise,
            clean_projected_energy=clean_proj, noise_projected_energy=proj_noise, q_clean=q_clean,
            q_noise=q_noise, q_noisy_input=q_noisy, actual_total_snr_db=actual, snr_p_db=direct,
            delta_p_db=delta, identity_snr_p_db=identity))
        final_hash = final_digest.hexdigest()
        if final_hash != case["input_sha256"]:
            anomaly(cid, None, "FINAL_INPUT_HASH_MISMATCH", final_hash)
        old = groups_input.get(cid)
        require(old is not None and len(old) == n, "Missing original per-record input diagnostics: " + cid)
        prefix = "S" if case["mode"] else case["condition"]
        qn, qz = groups_q[(cid, prefix+"_noise")], groups_q[(cid, prefix+"_input")]
        for table in (old, qn, qz):
            require(len(table) == n and not table.record_index.duplicated().any(), "Duplicate/missing historical diagnostic keys: " + cid)
            identity_ok = (table.record_index.to_numpy() == np.arange(n)) & (table.ecg_id.to_numpy() == cohort["ids"]) & (table.patient_id.to_numpy() == cohort["patient_ids"])
            for i in np.flatnonzero(~identity_ok):
                anomaly(cid, int(i), "OLD_DIAGNOSTIC_KEY_MISMATCH")
            metadata_ok = (table.condition.to_numpy() == case["condition"]) & (pd.to_numeric(table.snr).to_numpy() == case["snr"]) & (pd.to_numeric(table.noise_seed).to_numpy() == case["noise_seed"]) & (table["mode"].to_numpy() == case["mode"])
            for i in np.flatnonzero(~metadata_ok):
                anomaly(cid, int(i), "OLD_DIAGNOSTIC_METADATA_MISMATCH")
            hashes_ok = table.input_sha256.to_numpy() == np.asarray(record_hashes)
            for i in np.flatnonzero(~hashes_ok):
                anomaly(cid, int(i), "OLD_INPUT_RECORD_HASH_MISMATCH")
            for i in np.flatnonzero(table.noise_sha256.to_numpy() != case["noise_sha256"]):
                anomaly(cid, int(i), "OLD_NOISE_BASE_HASH_MISMATCH")
        for i in np.flatnonzero(old.clean_sha256.to_numpy() != freeze["clean"]["sha256"]):
            anomaly(cid, int(i), "OLD_CLEAN_FILE_HASH_MISMATCH")
        comparisons = [("OLD_Q_NOISE_MISMATCH", q_noise, qn.q.to_numpy(float), TOL["q_crosscheck_absolute"]),
                       ("OLD_Q_INPUT_MISMATCH", q_noisy, qz.q.to_numpy(float), TOL["q_crosscheck_absolute"]),
                       ("OLD_TOTAL_SNR_MISMATCH", actual, pd.to_numeric(old.achieved_snr_db, errors="coerce").to_numpy(float), TOL["snr_db_absolute"]),
                       ("IDENTITY_MISMATCH", direct, identity, TOL["identity_db_absolute"])]
        errors = {}
        for code, current, historical, tolerance in comparisons:
            ok = np.isclose(current, historical, rtol=0, atol=tolerance, equal_nan=True)
            for i in np.flatnonzero(~ok):
                anomaly(cid, int(i), code, f"current={current[i]}; reference={historical[i]}")
            finite = np.isfinite(current) & np.isfinite(historical)
            errors[code.lower()+"_max_abs_error"] = float(np.max(np.abs(current[finite]-historical[finite]))) if finite.any() else None
        for i, row in enumerate(frame.to_dict("records")):
            scalar = scalar_result(row)
            codes = scalar["anomaly_code"]
            if scalar["snr_p_db"] is None:
                frame.at[i, "snr_p_db"] = np.nan
            for code in codes.split(";") if codes else []:
                anomaly(cid, i, code)
        audits.append(dict(case_id=cid, condition=case["condition"], mode=case["mode"], snr_db=case["snr"], noise_seed=case["noise_seed"],
            n_records=n, final_input_sha256=final_hash, expected_input_sha256=case["input_sha256"], final_hash_verified=final_hash == case["input_sha256"],
            clean_array_sha256=manifest["clean_input_sha256"], noise_file_sha256=infos[case["noise_path"]]["sha256"], **errors))
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    require(len(result) == 271908 and not result.duplicated(["case_id", "record_id"]).any(), "Output primary key grid mismatch")
    anomalies_df = pd.DataFrame(anomalies, columns=["case_id", "record_id", "record_index", "anomaly_code", "detail"])
    code_map = {}
    for row in anomalies:
        code_map.setdefault((row["case_id"], row["record_id"]), set()).add(row["anomaly_code"])
    result["anomaly_code"] = [";".join(sorted(code_map.get((cid, int(rid)), set()) | code_map.get((cid, None), set()) | code_map.get(("clean", int(rid)), set()) | code_map.get(("clean", None), set()) | code_map.get(("source", None), set()))) for cid, rid in zip(result.case_id, result.record_id)]
    result.to_parquet(DEST / "snr_p_per_record.parquet", index=False)
    anomalies_df.to_csv(DEST / "anomalies.csv", index=False)
    pd.DataFrame(audits).to_csv(DEST / "geometry_input_audit.csv", index=False)
    metrics = ["snr_p_db", "delta_p_db", "actual_total_snr_db", "q_clean", "q_noise"]
    summarize(result, ["condition", "mode", "snr_db", "noise_seed"], metrics).to_csv(DEST / "geometry_by_noise.csv", index=False)
    summarize(result, ["condition", "mode", "snr_db"], metrics).to_csv(DEST / "geometry_summary.csv", index=False)
    boundaries()
    benign = {"ZERO_PROJECTED_CLEAN", "ZERO_PROJECTED_NOISE"}
    fatal = [r for r in anomalies if r["anomaly_code"] not in benign]
    audit = dict(geometry_status="P1-GEO-DIRECT" if not fatal else "P1-GEO-INVALID", n_records=n,
        n_cases=126, n_rows=len(result), n_patients=len(np.unique(cohort["patient_ids"])),
        projector_symmetry_frobenius=symmetry, projector_idempotence_frobenius=idempotence,
        whole_case_hash_matches=sum(x["final_hash_verified"] for x in audits), old_q_rows_crosschecked=len(old_q),
        old_input_rows_crosschecked=len(old_inputs), anomaly_count=len(anomalies), anomalous_geometry_rows=int(result.anomaly_code.ne("").sum()),
        anomaly_counts=anomalies_df.anomaly_code.value_counts().to_dict(), fatal_anomaly_count=len(fatal),
        independent_verification="pending qa/snrp_recompute.py", finished_at=now())
    write_json(DEST / "geometry_audit.json", audit)
    return result, cohort, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    analysis, infos, conflicts = prepare()
    if args.prepare_only:
        return
    try:
        frame, cohort, audit = geometry(analysis, infos, conflicts)
    except (ValueError, KeyError, OSError) as error:
        row = dict(case_id="unresolved", record_id=None, record_index=None,
                   anomaly_code="GEOMETRY_SOURCE_OR_PRIMARY_KEY_FAILURE", detail=str(error))
        pd.DataFrame([row]).to_csv(DEST / "anomalies.csv", index=False)
        write_json(DEST / "status.json", dict(geometry_status="P1-GEO-INVALID",
            performance_overlay_status="P1-OVERLAY-NOT-EVALUATED", anomaly_count=1, error=str(error), finished_at=now()))
        raise
    status = dict(audit, performance_overlay_status="P1-OVERLAY-PENDING")
    write_json(DEST / "status.json", status)
    from .overlay import run_overlay
    try:
        overlay = run_overlay(analysis, infos, conflicts, frame, cohort)
    except (ValueError, KeyError, OSError) as error:
        overlay = dict(performance_overlay_status="P1-OVERLAY-UNAVAILABLE", overlay_error=str(error))
        write_json(DEST / "overlay_failure.json", overlay)
    status.update(overlay)
    from .figures import make_figures
    if audit["geometry_status"] == "P1-GEO-DIRECT":
        figures = make_figures(frame, cohort, overlay["performance_overlay_status"] == "P1-OVERLAY")
    else:
        figures = []
    status.update(elapsed_seconds=time.perf_counter()-started, finished_at=now(), figure_files=figures)
    analysis.update(status="completed", result_status=status, field_meanings={
        "record_index": "Zero-based row in frozen clean/cohort arrays, not original dataset index",
        "q_noisy_input": "Direct ||Qz||^2/||z||^2; old full input-q crosscheck only",
        "delta_p_db": "Projection-relative-to-total SNR increment, distinct from absolute snr_p_db",
        "source_prediction_id": "Original prediction file SHA256; one source row only, no E/I replication"})
    write_json(DEST / "snrp_analysis.json", analysis)
    write_json(DEST / "status.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
