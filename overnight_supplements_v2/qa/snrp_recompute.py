"""Independent P1 acceptance; reads immutable artifacts, never producer geometry code.

Run after P1: python -B -m overnight_supplements_v2.qa.snrp_recompute
Only the two named QA JSON files are written. No waveform/probability copies.
"""
from __future__ import annotations

import os
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from overnight_supplements_v2.shared.common import ROOT, OUT, resolve, read_json, write_json, file_info, now, config

import hashlib
import json
import math
import traceback
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = OUT / "snrp"
HISTORY = "signed_control_mechanism/"
GEOMETRY_COLUMNS = [
    "record_id", "patient_id", "record_index", "condition", "mode", "snr_db",
    "noise_seed", "case_id", "clean_input_sha256", "noisy_input_sha256",
    "clean_total_energy", "noise_total_energy", "clean_projected_energy",
    "noise_projected_energy", "q_clean", "q_noise", "actual_total_snr_db",
    "snr_p_db", "delta_p_db", "identity_snr_p_db", "anomaly_code",
]


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("+inf" if value > 0 else "-inf")
    return value


def raw_hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def numeric(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def equal_number(left, right, atol=0.0, rtol=0.0):
    left, right = numeric(left), numeric(right)
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    if math.isinf(left) or math.isinf(right):
        return left == right
    return abs(left - right) <= atol + rtol * max(abs(left), abs(right))


def ratio_db(numerator, denominator):
    # Numerator-zero has priority over denominator-zero, including 0/0.
    if numerator == 0:
        return float("nan")
    if denominator == 0:
        return float("inf")
    if numerator < 0 or denominator < 0:
        return float("nan")
    return 10.0 * math.log10(numerator / denominator)


def code_set(value):
    if value is None or value == "":
        return set()
    if isinstance(value, list):
        return set(value)
    return set(str(value).split(";"))


class Audit:
    def __init__(self):
        self.result = {
            "started_at": now(), "status": "running", "checks": [], "failures": [],
            "sources": [], "training_invocations": 0, "inference_invocations": 0,
            "independence": "SVD column-space projection and independent rank-sum AUROC; no P1 calculation function imports",
            "scope": "5 frozen records x all 126 cases; full geometry/hash/join grid; representative original-NPZ overlay metrics; no new CI",
        }
        self.hashed = {}

    def check(self, name, passed, **evidence):
        entry = native({"name": name, "passed": bool(passed), **evidence})
        self.result["checks"].append(entry)
        if not passed:
            self.result["failures"].append(entry)
        return bool(passed)

    def source(self, path, schema, expected=None):
        key = resolve(path).relative_to(ROOT).as_posix()
        if key not in self.hashed:
            item = file_info(path, schema=schema, source="Independent P1 acceptance read-only input")
            self.hashed[key] = item
            self.result["sources"].append(item)
        item = self.hashed[key]
        if expected is not None:
            self.check("source_sha256:" + key, item["sha256"] == expected,
                       expected=expected, observed=item["sha256"])
        return item

    def phase(self, name, function):
        try:
            function()
        except Exception as exc:
            self.check(name + ":exception", False, error=repr(exc), traceback=traceback.format_exc())

    def finish(self):
        self.result["finished_at"] = now()
        self.result["status"] = "passed" if not self.result["failures"] else "failed"
        self.result["n_checks"] = len(self.result["checks"])
        self.result["n_failures"] = len(self.result["failures"])
        result = native(self.result)
        write_json(OUT / "qa/snrp_recompute.json", result)
        write_json(HERE / "qa.json", result)
        return 0 if result["status"] == "passed" else 1


class IndependentP1:
    def __init__(self, audit):
        self.audit = audit

    def prepare(self):
        a = self.audit
        self.analysis = read_json(HERE / "snrp_analysis.json")
        self.status = read_json(HERE / "status.json")
        self.cfg = config()
        self.p1 = self.cfg["scope"]["p1"]
        a.source(HERE / "snrp_analysis.json", "frozen P1 analysis contract")
        a.source(HERE / "status.json", "separate geometry and performance states")
        a.source(OUT / "freeze/run_config.json", "frozen selection and numerical tolerances")
        a.check("producer_zero_model_invocations", all(
            value.get(counter) == 0 for value in (self.analysis, self.status)
            for counter in ("training_invocations", "inference_invocations")))
        a.check("published_class_order_matches_frozen", self.analysis["class_order"] == self.cfg["class_order"],
                observed=self.analysis["class_order"], expected=self.cfg["class_order"])
        self.manifest = read_json(HISTORY + "inputs/full/manifest.json")
        self.freeze = read_json(HISTORY + "logs/full/freeze.json")
        a.source(HISTORY + "inputs/full/manifest.json", "126 immutable signed-control cases")
        a.source(HISTORY + "logs/full/freeze.json", "immutable source identity registry")
        self.cases = sorted(self.manifest["cases"], key=lambda item: item["case_id"])
        self.case_map = {case["case_id"]: case for case in self.cases}
        a.check("case_grid_126", len(self.cases) == len(self.case_map) == 126, observed=len(self.cases))
        a.check("analysis_case_mapping_matches_original",
                sorted(self.analysis["case_mapping"], key=lambda row: row["case_id"]) == self.cases)
        for name in ("cohort", "clean", "matrices"):
            item = self.freeze[name]
            a.source(item["path"], name + " immutable artifact", item["sha256"])
        with np.load(resolve(self.freeze["cohort"]["path"]), allow_pickle=False) as cohort:
            self.ids = cohort["ids"]
            self.patients = cohort["patient_ids"]
            self.indices = cohort["indices"]
            self.y = cohort["y"]
        self.fixed_ids = list(map(int, sorted(self.ids)[:5]))
        # These values must be taken from the analysis frozen before any energy calculation.
        self.published_ids = list(map(int, self.analysis["qa_record_ids"]))
        a.check("five_records_frozen_before_calculation", self.published_ids == self.fixed_ids,
                expected=self.fixed_ids, observed=self.published_ids)
        if self.published_ids != self.fixed_ids:
            raise ValueError("Frozen record selection differs; refusing a replacement QA selection")
        self.positions = np.array([np.flatnonzero(self.ids == rid).item() for rid in self.fixed_ids])
        a.check("cohort_identity", len(self.ids) == 2158 and len(set(self.ids)) == 2158
                and len(set(self.patients)) == 1877 and self.y.shape == (2158, 5),
                records=len(self.ids), patients=len(set(self.patients)), labels_shape=self.y.shape)
        self.clean = np.load(resolve(self.freeze["clean"]["path"]), mmap_mode="r", allow_pickle=False)
        a.check("clean_shape_dtype", self.clean.shape == (2158, 12, 1000) and self.clean.dtype == np.float32,
                shape=self.clean.shape, dtype=str(self.clean.dtype))
        self.matrix = np.asarray(read_json(self.freeze["matrices"]["path"])["standard"]["matrix"], dtype=np.float64)
        self.prediction_index_path = HISTORY + "tables/full/prediction_index.csv"
        a.source(self.prediction_index_path, "one row per immutable prediction source")
        self.pred_index = pd.read_csv(resolve(self.prediction_index_path), keep_default_na=False)
        self.all_pred_index = self.pred_index.copy()
        self.pred_index = self.pred_index[self.pred_index.case_id.isin(self.case_map)].copy()
        self.pred_index["seed"] = self.pred_index.seed.astype(int)
        # Freeze representative endpoints before opening any probability NPZ.
        ordered = self.pred_index.sort_values(["model", "seed", "case_id"])
        self.overlay_units = pd.concat([
            group.iloc[[0, -1]] for _, group in ordered.groupby(["condition", "snr"], sort=True)
        ]).drop_duplicates(["model", "seed", "case_id"])
        a.result["frozen_overlay_units"] = self.overlay_units[["model", "seed", "case_id", "prediction_path"]].to_dict("records")
        a.result["overlay_selection_rule"] = "First and last (model, integer train seed, case_id) within each condition x nominal SNR, before any probability read"
        a.result["qa_record_ids"] = self.fixed_ids
        a.result["analysis_sha256_before_computation"] = a.hashed[(HERE / "snrp_analysis.json").relative_to(ROOT).as_posix()]["sha256"]

    def geometry_grid(self):
        a = self.audit
        path = HERE / "snr_p_per_record.parquet"
        a.source(path, "unique case_id x record_id geometry and identities")
        table = pq.read_table(path)
        a.check("geometry_required_schema", set(GEOMETRY_COLUMNS).issubset(table.column_names),
                missing=sorted(set(GEOMETRY_COLUMNS) - set(table.column_names)), schema=str(table.schema))
        self.frame = table.to_pandas()
        del table
        self.frame["mode"] = self.frame["mode"].fillna("")
        self.frame["anomaly_code"] = self.frame["anomaly_code"].fillna("")
        f = self.frame
        a.check("geometry_exact_rowcount", len(f) == 2158 * 126, expected=271908, observed=len(f))
        duplicate = f.duplicated(["case_id", "record_id"], keep=False)
        a.check("geometry_unique_keys", not duplicate.any(), duplicates=f.loc[duplicate, ["case_id", "record_id"]].to_dict("records"))
        a.check("geometry_no_missing_keys", not f[["case_id", "record_id", "patient_id", "record_index"]].isna().any().any())
        a.check("geometry_exact_cases", set(f.case_id) == set(self.case_map),
                missing=sorted(set(self.case_map) - set(f.case_id)), extra=sorted(set(f.case_id) - set(self.case_map)))
        failures = []
        for case_id, group in f.groupby("case_id", sort=False):
            case = self.case_map[case_id]
            group = group.sort_values("record_index")
            ok = (len(group) == len(self.ids)
                  and np.array_equal(group.record_index, np.arange(len(self.ids)))
                  and np.array_equal(group.record_id, self.ids)
                  and np.array_equal(group.patient_id, self.patients)
                  and group.condition.eq(case["condition"]).all()
                  and group["mode"].eq(case["mode"]).all()
                  and group.snr_db.eq(case["snr"]).all()
                  and group.noise_seed.eq(case["noise_seed"]).all())
            if not ok:
                failures.append(case_id)
        a.check("geometry_full_case_record_grid", not failures, failed_cases=failures)
        self.keyed = f.set_index(["case_id", "record_id"], verify_integrity=True)
        # Compare every published per-record input hash to the original signed audit.
        validation_path = HISTORY + "tables/full/input_validation.csv"
        info = self.manifest["tables"]["input_validation"]
        a.source(validation_path, "original final float32 hashes and achieved total SNR", info["sha256"])
        columns = ["case_id", "ecg_id", "patient_id", "record_index", "input_sha256", "achieved_snr_db"]
        original = pd.read_csv(resolve(validation_path), usecols=columns).rename(columns={"ecg_id": "record_id"})
        original = original.set_index(["case_id", "record_id"], verify_integrity=True)
        self.original_validation = original
        exact_keys = self.keyed.index.sort_values().equals(original.index.sort_values())
        a.check("full_original_hash_key_coverage", exact_keys, historical_rows=len(original), produced_rows=len(f))
        joined = self.keyed.join(original, how="outer", rsuffix="_original")
        bad = joined.noisy_input_sha256.ne(joined.input_sha256)
        a.check("full_record_noisy_hash_matches", not bad.any(), checked_rows=len(joined),
                mismatches=joined.loc[bad, ["noisy_input_sha256", "input_sha256"]].reset_index().to_dict("records"))
        self.numeric_compare("full_actual_total_snr_vs_original", joined.actual_total_snr_db,
                             joined.achieved_snr_db, self.p1["identity_snr_atol_db"], keys=joined.index)
        for field in ("record_index", "patient_id"):
            bad = joined[field].ne(joined[field + "_original"])
            a.check("full_original_" + field, not bad.any(), failed_keys=list(joined.index[bad]))

    def numeric_compare(self, name, observed, expected, atol, rtol=0.0, keys=None):
        observed = np.asarray(observed, dtype=float)
        expected = np.asarray(expected, dtype=float)
        good = np.isclose(observed, expected, atol=atol, rtol=rtol, equal_nan=True)
        finite = np.isfinite(observed) & np.isfinite(expected)
        errors = np.abs(observed[finite] - expected[finite])
        bad_positions = np.flatnonzero(~good)
        mismatch = [{"key": (keys[i] if keys is not None else int(i)), "observed": observed[i], "expected": expected[i]}
                    for i in bad_positions]
        self.audit.check(name, good.all(), checked=int(observed.size), atol=atol, rtol=rtol,
                         max_absolute_error=float(errors.max()) if errors.size else None,
                         mismatches=mismatch)

    def direct_recompute(self):
        a = self.audit
        u, singular, _ = np.linalg.svd(self.matrix, full_matrices=False)
        basis = u[:, singular > singular[0] * self.p1["rcond"]]
        projector = basis @ basis.T
        pinv_projector = self.matrix @ np.linalg.pinv(self.matrix, rcond=self.p1["rcond"])
        evidence = {
            "rank": basis.shape[1], "svd_vs_pinv_frobenius": np.linalg.norm(projector - pinv_projector),
            "symmetry_frobenius": np.linalg.norm(projector - projector.T),
            "idempotence_frobenius": np.linalg.norm(projector @ projector - projector),
            "pinv_symmetry_frobenius": np.linalg.norm(pinv_projector - pinv_projector.T),
            "pinv_idempotence_frobenius": np.linalg.norm(pinv_projector @ pinv_projector - pinv_projector),
        }
        a.check("independent_projector_invariants", evidence["rank"] == 8 and
                all(v < self.p1["projector_tolerance"] for k, v in evidence.items() if k != "rank"), **evidence)
        q_path = HISTORY + "tables/full/subspace_diagnostics_per_record.csv"
        a.source(q_path, "historical clean and actual-residual q and input identities",
                 self.manifest["tables"]["subspace_diagnostics_per_record"]["sha256"])
        pieces = []
        clean_hashes = {}
        for chunk in pd.read_csv(resolve(q_path), chunksize=50000):
            clean_rows = chunk[chunk["object"].eq("clean")]
            clean_hashes.update(zip(clean_rows.ecg_id.astype(int), clean_rows.input_sha256))
            pieces.append(chunk[chunk.ecg_id.isin(self.fixed_ids) &
                                (chunk["object"].eq("clean") | chunk["object"].str.endswith("_noise"))])
        oldq = pd.concat(pieces, ignore_index=True)
        old_clean = oldq[oldq["object"].eq("clean")].set_index("ecg_id", verify_integrity=True)
        old_noise = oldq[oldq["object"].str.endswith("_noise")].set_index(["case_id", "ecg_id"], verify_integrity=True)
        a.check("old_q_fixed_grid", len(old_clean) == 5 and len(old_noise) == 630,
                clean_rows=len(old_clean), noise_rows=len(old_noise))
        f = self.frame
        expected_clean_hashes = f.record_id.map(clean_hashes)
        bad = f.clean_input_sha256.ne(expected_clean_hashes)
        a.check("full_clean_hash_matches_original", not bad.any(), checked_rows=len(f),
                mismatches=f.loc[bad, ["case_id", "record_id", "clean_input_sha256"]].to_dict("records"))
        bases = {}
        for case in self.cases:
            if case["noise_path"] not in bases:
                a.source(case["noise_path"], "immutable float32 0-dB noise base", case["noise_sha256"])
                bases[case["noise_path"]] = np.load(resolve(case["noise_path"]), mmap_mode="r", allow_pickle=False)
        computed = []
        hash_failures = []
        for case in self.cases:
            base = bases[case["noise_path"]]
            for position in self.positions:
                rid = int(self.ids[position])
                x32 = self.clean[position]
                # Independent spelling of historical float32 order: signs, scale, addition.
                z32 = np.array(base[position], dtype=np.float32, copy=True)
                if case["condition"].startswith("S_"):
                    z32 *= np.asarray(case["sign_vector"], dtype=np.float32)[:, None]
                z32 *= np.float32(10.0 ** (-float(case["snr"]) / 20.0))
                z32 += x32
                x = x32.astype(np.float64)
                noise = z32.astype(np.float64) - x
                px_coordinates = basis.T @ x
                pn_coordinates = basis.T @ noise
                x_energy = float(np.sum(x * x))
                n_energy = float(np.sum(noise * noise))
                px_energy = float(np.sum(px_coordinates * px_coordinates))
                pn_energy = float(np.sum(pn_coordinates * pn_coordinates))
                residual_x = x - basis @ px_coordinates
                residual_n = noise - basis @ pn_coordinates
                qx = float(np.sum(residual_x * residual_x) / x_energy) if x_energy else float("nan")
                qn = float(np.sum(residual_n * residual_n) / n_energy) if n_energy else float("nan")
                total = ratio_db(x_energy, n_energy)
                direct = ratio_db(px_energy, pn_energy)
                delta = ratio_db(1.0 - qx, 1.0 - qn)
                identity = total + delta
                key = (case["case_id"], rid)
                row = self.keyed.loc[key]
                old = self.original_validation.loc[key]
                zhash, xhash = raw_hash(z32), raw_hash(x32)
                if zhash != old.input_sha256 or zhash != row.noisy_input_sha256 or xhash != row.clean_input_sha256:
                    hash_failures.append({"case_id": key[0], "record_id": rid, "reconstructed": zhash,
                                          "original": old.input_sha256, "published": row.noisy_input_sha256,
                                          "clean_reconstructed": xhash, "clean_published": row.clean_input_sha256})
                computed.append({
                    "case_id": key[0], "record_id": rid, "clean_total_energy": x_energy,
                    "noise_total_energy": n_energy, "clean_projected_energy": px_energy,
                    "noise_projected_energy": pn_energy, "q_clean": qx, "q_noise": qn,
                    "actual_total_snr_db": total, "snr_p_db": direct, "delta_p_db": delta,
                    "identity_snr_p_db": identity,
                    "old_q_clean": old_clean.loc[rid, "q"], "old_q_noise": old_noise.loc[key, "q"],
                })
        independent = pd.DataFrame(computed).set_index(["case_id", "record_id"], verify_integrity=True)
        a.check("independent_630_input_reconstructions", len(independent) == 630 and not hash_failures,
                checked=len(independent), original_hash_mismatches=hash_failures,
                arithmetic="float32(base*sign); float32(*float32(10**(-snr/20))); float32(+clean); actual residual=float64(final)-float64(clean)")
        published = self.keyed.loc[independent.index]
        for field in ("clean_total_energy", "noise_total_energy", "clean_projected_energy", "noise_projected_energy"):
            self.numeric_compare("svd_630_" + field, published[field], independent[field], 0.0,
                                 self.p1["energy_relative_tolerance"], independent.index)
        for field in ("q_clean", "q_noise"):
            self.numeric_compare("svd_630_" + field, published[field], independent[field],
                                 self.p1["q_tolerance"], keys=independent.index)
            self.numeric_compare("old_630_" + field, independent[field], independent["old_" + field],
                                 self.p1["q_tolerance"], keys=independent.index)
        for field in ("actual_total_snr_db", "snr_p_db", "delta_p_db", "identity_snr_p_db"):
            self.numeric_compare("svd_630_" + field, published[field], independent[field],
                                 self.p1["identity_snr_atol_db"], keys=independent.index)
        self.numeric_compare("independent_direct_vs_identity_630", independent.snr_p_db,
                             independent.identity_snr_p_db, self.p1["identity_snr_atol_db"], keys=independent.index)
        a.result["independent_scalar_evidence"] = independent.reset_index().to_dict("records")
        a.check("geometry_direct_route_claim", self.status.get("geometry_status") == "P1-GEO-DIRECT",
                observed=self.status.get("geometry_status"), verified_route="independent SVD projection of hash-matched reconstructed final float32 inputs")

    def anomalies(self):
        a = self.audit
        a.source(HERE / "boundary_cases.json", "production boundary function outputs on analytical fixtures")
        boundary = read_json(HERE / "boundary_cases.json")
        fixtures = boundary["cases"]
        categories = set()
        evidence = []
        failures = []
        tol = self.analysis["tolerances"]
        a.check("tolerances_match_frozen_contract",
                tol["projection_frobenius"] == self.p1["projector_tolerance"]
                and tol["pseudoinverse_rcond"] == self.p1["rcond"]
                and tol["q_crosscheck_absolute"] == self.p1["q_tolerance"]
                and tol["identity_db_absolute"] == self.p1["identity_snr_atol_db"]
                and tol["energy_relative"] == self.p1["energy_relative_tolerance"],
                analysis_tolerances=tol, frozen_tolerances=self.p1)
        for fixture in fixtures:
            values = fixture["input"]
            output = fixture["output"]
            codes = set()
            energies = [numeric(values.get(name)) for name in (
                "clean_total_energy", "noise_total_energy", "clean_projected_energy", "noise_projected_energy")]
            qvalues = [numeric(values.get("q_clean")), numeric(values.get("q_noise"))]
            missing = any(values.get(key) is None or values.get(key) == "" for key in ("case_id", "record_id"))
            if missing:
                codes.add("MISSING_PRIMARY_KEY")
                categories.add("missing_keys")
            if any(math.isfinite(value) and value < -tol["energy_negative_absolute"] for value in energies):
                codes.add("NEGATIVE_ENERGY")
                categories.add("invalid_negative_energy")
            if any(not math.isfinite(value) for value in energies):
                codes.add("NONFINITE_ENERGY")
                categories.add("nonfinite_energy")
            if any(math.isfinite(value) and (value < -tol["q_bound_absolute"]
                                            or value > 1 + tol["q_bound_absolute"]) for value in qvalues):
                codes.add("Q_OUT_OF_BOUNDS")
                categories.add("q_out_of_bounds")
            for total, qvalue in zip(energies[:2], qvalues):
                if not math.isfinite(qvalue) and total != 0:
                    codes.add("NONFINITE_Q")
            invalid = bool(codes)
            numerator, denominator = energies[2:]
            result = float("nan")
            if not invalid:
                if numerator == 0:
                    codes.add("ZERO_PROJECTED_CLEAN")
                    categories.add("both_zero" if denominator == 0 else "zero_numerator")
                elif denominator == 0:
                    codes.add("ZERO_PROJECTED_NOISE")
                    categories.add("zero_denominator")
                    result = float("inf")
                else:
                    result = ratio_db(numerator, denominator)
                    categories.add("finite_nonzero")
                    if 0 < numerator < np.finfo(np.float64).eps:
                        categories.add("tiny_positive_numerator")
                    if 0 < denominator < np.finfo(np.float64).eps:
                        categories.add("tiny_positive_denominator")
            actual_codes = code_set(output["anomaly_code"])
            good = equal_number(result, output["snr_p_db"], atol=self.p1["identity_snr_atol_db"]) and actual_codes == codes
            item = {"name": fixture["name"], "input": values, "published_output": output,
                    "independent_snr_p_db": result, "independent_codes": sorted(codes), "passed": good}
            evidence.append(item)
            if not good:
                failures.append(item)
        required = {"zero_numerator", "zero_denominator", "both_zero", "invalid_negative_energy",
                    "q_out_of_bounds", "missing_keys", "finite_nonzero",
                    "tiny_positive_numerator", "tiny_positive_denominator"}
        a.check("synthetic_boundary_category_coverage", required.issubset(categories),
                required=sorted(required), observed=sorted(categories), missing=sorted(required - categories))
        a.check("synthetic_boundary_independent_oracle", not failures, fixture_count=len(fixtures), failures=failures)
        a.result["synthetic_boundary_evidence"] = evidence
        a.source(HERE / "anomalies.csv", "one real anomaly row per case/record/code")
        anomalous = pd.read_csv(HERE / "anomalies.csv", keep_default_na=False)
        columns = {"case_id", "record_id", "record_index", "anomaly_code", "detail"}
        a.check("real_anomaly_table_schema", columns.issubset(anomalous.columns),
                missing=sorted(columns - set(anomalous.columns)))
        duplicate = anomalous.duplicated(["case_id", "record_id", "anomaly_code"], keep=False)
        a.check("real_anomaly_unique_codes", not duplicate.any(), duplicates=anomalous[duplicate].to_dict("records"))
        declared = set()
        known_codes = {
            "ZERO_PROJECTED_CLEAN", "ZERO_PROJECTED_NOISE", "NEGATIVE_ENERGY", "NONFINITE_ENERGY",
            "Q_OUT_OF_BOUNDS", "NONFINITE_Q", "MISSING_PRIMARY_KEY", "IDENTITY_MISMATCH",
        }
        code_counts = Counter()
        numeric_errors = []
        for row in self.frame.itertuples(index=False):
            codes = code_set(row.anomaly_code)
            for code in codes:
                declared.add((row.case_id, int(row.record_id), int(row.record_index), code))
                code_counts[code] += 1
            energies = (row.clean_total_energy, row.noise_total_energy,
                        row.clean_projected_energy, row.noise_projected_energy)
            required_codes = set()
            if any(not math.isfinite(value) for value in energies):
                required_codes.add("NONFINITE_ENERGY")
            if any(value < -tol["energy_negative_absolute"] for value in energies):
                required_codes.add("NEGATIVE_ENERGY")
            for total, qvalue in ((row.clean_total_energy, row.q_clean),
                                  (row.noise_total_energy, row.q_noise)):
                if math.isfinite(qvalue):
                    if qvalue < -tol["q_bound_absolute"] or qvalue > 1 + tol["q_bound_absolute"]:
                        required_codes.add("Q_OUT_OF_BOUNDS")
                elif total != 0:
                    required_codes.add("NONFINITE_Q")
            invalid = bool(required_codes)
            if not invalid and row.clean_projected_energy == 0:
                required_codes.add("ZERO_PROJECTED_CLEAN")
            elif not invalid and row.noise_projected_energy == 0:
                required_codes.add("ZERO_PROJECTED_NOISE")
            expected = float("nan") if invalid else ratio_db(row.clean_projected_energy, row.noise_projected_energy)
            if (not required_codes.issubset(codes)
                    or not equal_number(expected, row.snr_p_db, atol=self.p1["identity_snr_atol_db"])
                    or ("ZERO_PROJECTED_CLEAN" in codes and "ZERO_PROJECTED_NOISE" in codes)):
                numeric_errors.append({"case_id": row.case_id, "record_id": row.record_id,
                                       "codes": sorted(codes), "required_codes": sorted(required_codes),
                                       "observed_snr_p_db": row.snr_p_db, "expected_snr_p_db": expected})
        recorded = set()
        unmatched_anomaly_rows = []
        for row in anomalous.itertuples(index=False):
            scope = self.frame
            if row.case_id not in {"clean", "source"}:
                scope = scope[scope.case_id.eq(row.case_id)]
            if row.record_id != "":
                scope = scope[scope.record_id.eq(int(row.record_id))]
            if row.record_index != "":
                scope = scope[scope.record_index.eq(int(row.record_index))]
            if scope.empty:
                unmatched_anomaly_rows.append(row._asdict())
            recorded.update((item.case_id, int(item.record_id), int(item.record_index), row.anomaly_code)
                            for item in scope[["case_id", "record_id", "record_index"]].itertuples(index=False))
        a.check("real_anomaly_scope_keys", not unmatched_anomaly_rows, unmatched_rows=unmatched_anomaly_rows)
        a.check("real_anomaly_table_exact_consistency", recorded == declared,
                missing_rows=sorted(declared - recorded), extra_rows=sorted(recorded - declared),
                table_rows=len(anomalous), per_record_code_rows=len(declared), counts=dict(code_counts))
        a.check("real_anomaly_numerical_contract", not numeric_errors, failures=numeric_errors)
        # Boundary infinities/nulls are valid documented outputs; identity/hash/q conflicts are not.
        invalid_real = {code: count for code, count in code_counts.items()
                        if code not in {"ZERO_PROJECTED_CLEAN", "ZERO_PROJECTED_NOISE"}}
        a.check("no_unresolved_real_anomalies", not invalid_real, unresolved_code_counts=invalid_real,
                unknown_codes=sorted(set(code_counts) - known_codes))
        audit_path = HERE / "geometry_input_audit.csv"
        a.source(audit_path, "producer full-cohort final input hash coverage per case")
        full = pd.read_csv(audit_path, keep_default_na=False)
        a.check("producer_full_input_hash_case_coverage",
                len(full) == 126 and not full.case_id.duplicated().any()
                and set(full.case_id) == set(self.case_map),
                cases=len(full), missing=sorted(set(self.case_map) - set(full.case_id)))
        failures = []
        for row in full.itertuples(index=False):
            expected = self.case_map[row.case_id]["input_sha256"]
            good = (row.final_input_sha256 == expected == row.expected_input_sha256
                    and str(row.final_hash_verified).lower() in {"true", "1"} and int(row.n_records) == len(self.ids))
            if not good:
                failures.append(row._asdict())
        a.check("producer_full_case_hash_matches_original", not failures, cases_checked=len(full), failures=failures,
                scope="Producer full-cohort digest audit is cross-checked against immutable manifest; independent raw reconstruction is exactly 630 records, not all 271908")

    def overlay(self):
        a = self.audit
        classes = self.cfg["class_order"]
        path = HERE / "auroc_overlay.csv"
        a.source(path, "one unweighted absolute AUROC row per existing prediction source")
        metrics = pd.read_csv(path, keep_default_na=False)
        required = {"source_prediction_id", "model", "train_seed", "case_id", "condition",
                    "mode", "snr_db", "noise_seed", "macro_auroc", "n_records", "n_patients",
                    "unit", "aggregation_weight",
                    *("auroc_" + name for name in classes),
                    *("n_positive_" + name for name in classes),
                    *("n_negative_" + name for name in classes)}
        a.check("overlay_metric_schema", required.issubset(metrics.columns),
                missing=sorted(required - set(metrics.columns)))
        a.check("overlay_fraction_and_no_reweighting",
                metrics.unit.eq("fraction").all() and metrics.aggregation_weight.eq(1).all())
        original = self.all_pred_index.set_index("prediction_sha256", verify_integrity=True)
        a.check("overlay_original_source_grid", len(original) == 762 and len(self.pred_index) == 756,
                all_sources=len(original), noisy_sources=len(self.pred_index))
        unique = not metrics.source_prediction_id.duplicated().any()
        a.check("overlay_metric_unique_sources", unique, duplicate_sources=metrics.loc[
                metrics.source_prediction_id.duplicated(keep=False), "source_prediction_id"].tolist())
        produced = metrics.set_index("source_prediction_id", verify_integrity=True)
        a.check("overlay_metric_source_coverage", set(produced.index) == set(original.index),
                missing=sorted(set(original.index) - set(produced.index)),
                extra=sorted(set(produced.index) - set(original.index)))
        metadata_errors = []
        for source_id, row in produced.iterrows():
            ref = original.loc[source_id]
            fields = {"model": "model", "train_seed": "seed", "case_id": "case_id",
                      "condition": "condition", "mode": "mode"}
            for field, old_field in fields.items():
                if str(row[field]) != str(ref[old_field]):
                    metadata_errors.append({"source": source_id, "field": field,
                                            "observed": row[field], "expected": ref[old_field]})
            for field, old_field in (("snr_db", "snr"), ("noise_seed", "noise_seed")):
                if not equal_number(row[field], ref[old_field]):
                    metadata_errors.append({"source": source_id, "field": field,
                                            "observed": row[field], "expected": ref[old_field]})
        a.check("overlay_all_source_metadata", not metadata_errors, mismatches=metadata_errors)
        a.check("overlay_record_patient_counts", metrics.n_records.eq(len(self.ids)).all()
                and metrics.n_patients.eq(len(set(self.patients))).all(),
                expected_records=len(self.ids), expected_patients=len(set(self.patients)))
        for column, name in enumerate(classes):
            positives = int(self.y[:, column].sum())
            negatives = len(self.ids) - positives
            a.check("overlay_all_class_counts:" + name,
                    metrics["n_positive_" + name].eq(positives).all()
                    and metrics["n_negative_" + name].eq(negatives).all(),
                    expected_positive=positives, expected_negative=negatives)
        self.numeric_compare("overlay_all_original_macro_auroc",
                             produced.loc[original.index, "macro_auroc"].astype(float),
                             original.macro_auroc.astype(float), 1e-12, keys=original.index)
        alignment_path = HERE / "overlay_alignment.parquet"
        a.source(alignment_path, "auditable original prediction-row links, no probabilities")
        parquet = pq.ParquetFile(alignment_path)
        alignment_columns = {
            "source_prediction_id", "prediction_path", "prediction_row_index", "model",
            "train_seed", "case_id", "condition", "mode", "snr_db", "noise_seed",
            "record_id", "patient_id", "record_index", *("label_" + name for name in classes),
            "dataset_index", "clean_input_sha256", "noisy_input_sha256", "checkpoint_sha256",
        }
        a.check("overlay_alignment_schema", alignment_columns.issubset(parquet.schema_arrow.names),
                missing=sorted(alignment_columns - set(parquet.schema_arrow.names)),
                schema=str(parquet.schema_arrow))
        a.check("overlay_no_probability_copy", not any(
            column == "p" or column.startswith(("probability", "prob_", "p_"))
            for column in parquet.schema_arrow.names))
        coverage = {source: np.zeros(len(self.ids), dtype=np.int32)
                    for source in self.pred_index.prediction_sha256}
        hash_links = {}
        for case_id, rows in self.frame.groupby("case_id", sort=False):
            rows = rows.sort_values("record_index")
            hash_links[case_id] = (rows.clean_input_sha256.to_numpy(), rows.noisy_input_sha256.to_numpy())
        link_errors = []
        unknown_sources = Counter()
        count = 0
        for batch in parquet.iter_batches(batch_size=65536, columns=sorted(alignment_columns)):
            frame = batch.to_pandas()
            frame["mode"] = frame["mode"].fillna("")
            count += len(frame)
            for source_id, group in frame.groupby("source_prediction_id", sort=False):
                if source_id not in coverage:
                    unknown_sources[source_id] += len(group)
                    continue
                ref = original.loc[source_id]
                positions_raw = group.prediction_row_index.to_numpy(dtype=float)
                valid = (np.isfinite(positions_raw) & (positions_raw >= 0)
                         & (positions_raw < len(self.ids)) & (positions_raw == np.floor(positions_raw)))
                if not valid.all():
                    link_errors.append({"source": source_id, "invalid_prediction_rows": positions_raw[~valid]})
                    continue
                positions = positions_raw.astype(np.int64)
                coverage[source_id] += np.bincount(positions, minlength=len(self.ids)).astype(np.int32)
                expected_clean, expected_noisy = hash_links[ref.case_id]
                matching = {
                    "record_id": np.array_equal(group.record_id, self.ids[positions]),
                    "patient_id": np.array_equal(group.patient_id, self.patients[positions]),
                    "record_index": np.array_equal(group.record_index, positions),
                    "dataset_index": np.array_equal(group.dataset_index, self.indices[positions]),
                    "clean_input_sha256": np.array_equal(group.clean_input_sha256, expected_clean[positions]),
                    "noisy_input_sha256": np.array_equal(group.noisy_input_sha256, expected_noisy[positions]),
                    "checkpoint_sha256": group.checkpoint_sha256.eq(ref.checkpoint_sha256).all(),
                    "prediction_path": group.prediction_path.eq(ref.prediction_path).all(),
                    "model": group.model.eq(ref.model).all(),
                    "train_seed": group.train_seed.eq(int(ref.seed)).all(),
                    "case_id": group.case_id.eq(ref.case_id).all(),
                    "condition": group.condition.eq(ref.condition).all(),
                    "mode": group["mode"].eq(ref["mode"]).all(),
                    "labels": np.array_equal(group[["label_" + name for name in classes]], self.y[positions]),
                }
                for field, old_field in (("snr_db", "snr"), ("noise_seed", "noise_seed")):
                    matching[field] = all(equal_number(value, ref[old_field]) for value in group[field].unique())
                case = self.case_map[ref.case_id]
                matching["geometry_case"] = (ref.condition == case["condition"] and ref["mode"] == case["mode"])
                failed = [key for key, value in matching.items() if not value]
                if failed:
                    link_errors.append({"source": source_id, "failed_fields": failed,
                                        "rows": group.prediction_row_index.tolist()})
        bad_coverage = [{"source": source_id, "missing_rows": np.flatnonzero(values == 0).tolist(),
                         "duplicate_rows": np.flatnonzero(values > 1).tolist()}
                        for source_id, values in coverage.items() if not (values == 1).all()]
        a.check("overlay_full_exact_source_record_grid", count == len(self.pred_index) * len(self.ids)
                and not bad_coverage and not unknown_sources,
                rows=count, expected_rows=len(self.pred_index) * len(self.ids),
                noisy_sources=len(self.pred_index), clean_geometry_cases=0,
                source_coverage_errors=bad_coverage, unknown_sources=dict(unknown_sources),
                scope="756 noisy sources x 2158 records only; six clean AUROC references audited separately, never given fabricated noisy geometry")
        a.check("overlay_exact_record_patient_label_metadata_join", not link_errors, mismatches=link_errors)
        # This check prevents treating E/I display replication as extra statistical observations.
        a.check("overlay_no_EI_fivefold_replication", unique and not bad_coverage,
                E_sources=int(original.condition.eq("E").sum()), I_sources=int(original.condition.eq("I").sum()),
                signed_sources=int(original.condition.str.startswith("S_").sum()),
                interpretation="one metric/source and one linked row/source/ECG; signed modes remain distinct")
        fixed = pd.concat([self.overlay_units,
                           self.all_pred_index[self.all_pred_index.condition.eq("clean")]]).drop_duplicates("prediction_sha256")
        numeric_evidence = []
        errors = []
        for _, source in fixed.iterrows():
            a.source(source.prediction_path, "immutable probabilities read only for fixed independent AUROC",
                     source.prediction_sha256)
            with np.load(resolve(source.prediction_path), allow_pickle=False) as prediction:
                ids, patients, indices = prediction["ids"], prediction["patient_ids"], prediction["indices"]
                labels, scores = prediction["y"], prediction["p"]
                keys_match = (np.array_equal(ids, self.ids) and np.array_equal(patients, self.patients)
                              and np.array_equal(indices, self.indices) and np.array_equal(labels, self.y))
                hashes_match = (str(prediction["input_sha256"].item()) == source.input_sha256
                                and str(prediction["checkpoint_sha256"].item()) == source.checkpoint_sha256
                                and raw_hash(scores) == source.p_sha256)
                if not keys_match or not hashes_match:
                    errors.append({"source": source.prediction_sha256, "keys_match": keys_match,
                                   "input_checkpoint_match": hashes_match})
                if labels.shape != scores.shape or not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all():
                    raise ValueError("Invalid original probability/label array for " + source.prediction_path)
                aucs = []
                for column in range(labels.shape[1]):
                    truth = labels[:, column].astype(np.int64)
                    order = np.argsort(scores[:, column], kind="mergesort")
                    sorted_scores = scores[order, column]
                    sorted_truth = truth[order]
                    starts = np.r_[0, 1 + np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])]
                    ends = np.r_[starts[1:], len(truth)]
                    average_ranks = (starts + 1 + ends) / 2.0
                    ranks = np.repeat(average_ranks, ends - starts)
                    n_positive = int(truth.sum())
                    denominator = n_positive * (len(truth) - n_positive)
                    rank_sum = float(ranks[sorted_truth == 1].sum())
                    aucs.append((rank_sum - n_positive * (n_positive + 1) / 2.0) / denominator
                                if denominator else float("nan"))
                row = produced.loc[source.prediction_sha256]
                macro = float(np.mean(aucs))
                expected_auc = [float(row["auroc_" + name]) for name in classes]
                count_positive = labels.sum(axis=0).astype(int)
                count_negative = len(labels) - count_positive
                for i, name in enumerate(classes):
                    if (int(row["n_positive_" + name]) != int(count_positive[i])
                            or int(row["n_negative_" + name]) != int(count_negative[i])):
                        errors.append({"source": source.prediction_sha256, "class": name,
                                       "positive_count": int(count_positive[i]), "negative_count": int(count_negative[i]),
                                       "published_positive_count": row["n_positive_" + name],
                                       "published_negative_count": row["n_negative_" + name]})
                    if not equal_number(aucs[i], expected_auc[i], atol=1e-12):
                        errors.append({"source": source.prediction_sha256, "class": name,
                                       "observed": expected_auc[i], "independent": aucs[i]})
                if not equal_number(macro, row.macro_auroc, atol=1e-12):
                    errors.append({"source": source.prediction_sha256, "metric": "macro_auroc",
                                   "observed": row.macro_auroc, "independent": macro})
                numeric_evidence.append({
                    "model": source.model, "train_seed": int(source.seed), "case_id": source.case_id,
                    "source_prediction_id": source.prediction_sha256, "auroc": dict(zip(classes, aucs)),
                    "positive_counts": dict(zip(classes, count_positive)),
                    "negative_counts": dict(zip(classes, count_negative)), "macro_auroc": macro,
                    "published_macro_auroc": float(row.macro_auroc),
                    "max_class_absolute_error": float(np.max(np.abs(np.asarray(aucs) - expected_auc))),
                    "macro_absolute_error": abs(macro - float(row.macro_auroc)),
                })
        a.check("overlay_fixed_original_NPZ_recomputation", not errors, n_sources=len(fixed),
                n_noisy_sources=len(self.overlay_units), n_clean_sources=len(fixed) - len(self.overlay_units),
                errors=errors, atol=1e-12, algorithm="independent average-rank Mann-Whitney, macro of five classes")
        a.result["overlay_numeric_evidence"] = numeric_evidence
        a.check("overlay_status_claim", self.status.get("performance_overlay_status") == "P1-OVERLAY",
                observed=self.status.get("performance_overlay_status"))

    def finish_contract(self):
        # Detect changes to the pre-computation selection/contract, rather than accepting a refreeze.
        before = self.audit.result["analysis_sha256_before_computation"]
        after = file_info(HERE / "snrp_analysis.json")["sha256"]
        self.audit.check("analysis_unchanged_during_qa", before == after, before=before, after=after)


def main():
    audit = Audit()
    checker = IndependentP1(audit)
    audit.phase("preparation", checker.prepare)
    if hasattr(checker, "positions") and hasattr(checker, "pred_index"):
        audit.phase("geometry_grid", checker.geometry_grid)
        if hasattr(checker, "keyed") and hasattr(checker, "original_validation"):
            audit.phase("direct_recompute", checker.direct_recompute)
        audit.phase("anomaly_contract", checker.anomalies)
        audit.phase("overlay", checker.overlay)
        audit.phase("contract_stability", checker.finish_contract)
    return audit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
