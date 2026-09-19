"""Independent, read-only P0 acceptance; run only after the producer finishes."""
from __future__ import annotations

import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from overnight_supplements_v2.shared.common import OUT, ROOT, config, file_info, now, read_json, resolve, write_json

import hashlib
import itertools
import json
import time
import traceback

import numpy as np
import pandas as pd


METRICS = ["M_EE", "M_IE", "M_EI", "M_II", "g_E", "g_I", "gamma"]
ARMS = [("electrode", "electrode"), ("independent_rms", "electrode"),
        ("electrode", "independent_rms"), ("independent_rms", "independent_rms")]
KEY = ["architecture", "snr_db", "train_seed", "combo_id", "noise_seed"]
UNIT_KEY = ["architecture", "snr_db", "train_seed", "test_structure", "combo_id", "noise_seed"]
TOL = 1e-12
DRAWS = [0, 1999]
DID = OUT / "did"
INPUT_MANIFEST = "phase2/results/test_inputs/full/manifest.json"
BOOT_MANIFEST = "phase2/results/tables/full/patient_bootstrap/manifest.json"
GROUP_METRICS = "phase2/results/tables/full/group_seed_metrics.csv"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def raw_hash(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def literal_weighted_auc(y, probability, weights):
    """Weighted P(score+>score-)+0.5 P(tie), counting negative masses.

    This does not use production rank/aggregation helpers or marginal CIs.
    Weights has three rows: the full cohort and the two frozen patient draws.
    """
    values = np.empty((weights.shape[0], y.shape[1]), dtype=np.float64)
    for label in range(y.shape[1]):
        positive = y[:, label] == 1
        negative = ~positive
        pos_scores = probability[positive, label]
        neg_scores = probability[negative, label]
        order = np.argsort(neg_scores, kind="stable")
        ordered_scores = neg_scores[order]
        neg_weights = weights[:, negative][:, order]
        prefix = np.concatenate((np.zeros((len(weights), 1)), np.cumsum(neg_weights, axis=1)), axis=1)
        lower = np.searchsorted(ordered_scores, pos_scores, side="left")
        upper = np.searchsorted(ordered_scores, pos_scores, side="right")
        pos_weights = weights[:, positive]
        numerator = np.sum(pos_weights * (prefix[:, lower] + 0.5 * (prefix[:, upper] - prefix[:, lower])), axis=1)
        denominator = pos_weights.sum(axis=1) * neg_weights.sum(axis=1)
        values[:, label] = np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)
    return values.mean(axis=1)


def with_contrasts(arms):
    g_e = arms[..., 0] - arms[..., 1]
    g_i = arms[..., 2] - arms[..., 3]
    return np.concatenate((arms, g_e[..., None], g_i[..., None], (g_e - g_i)[..., None]), axis=-1)


class Audit:
    def __init__(self):
        self.result = dict(started_at=now(), status="FAIL", release_pass=False,
                           tolerance=TOL, unit="fraction", failures=[], comparison_count=0,
                           max_absolute_error=0.0, source_evidence=[], qa_units={}, summary_groups=[],
                           coverage=dict(frozen_qa_units=2, summary_groups=4,
                                         independently_recomputed_draw_ids=DRAWS,
                                         all_2000_draws_recomputed_from_probabilities=False),
                           implementation="literal weighted positive-negative counting; half-credit ties; five-class ordinary macro mean; no production imports",
                           not_executed=["training", "inference", "p-values", "raw-probability recomputation of the other 1998 draws"],
                           training_invocations=0, inference_invocations=0)
        self.sources = {}
        self.verified = {}

    def compare(self, name, expected, observed):
        expected = np.asarray(expected, dtype=float)
        observed = np.asarray(observed, dtype=float)
        require(expected.shape == observed.shape, f"Shape mismatch at {name}: {expected.shape} != {observed.shape}")
        same_nan = np.isnan(expected) & np.isnan(observed)
        finite = np.isfinite(expected) & np.isfinite(observed)
        errors = np.full(expected.shape, np.inf)
        errors[same_nan] = 0.0
        errors[finite] = np.abs(expected[finite] - observed[finite])
        maximum = float(errors.max()) if errors.size else 0.0
        self.result["comparison_count"] += int(expected.size)
        self.result["max_absolute_error"] = max(self.result["max_absolute_error"], maximum)
        if maximum > TOL:
            self.result["failures"].append(dict(check=name, max_absolute_error=maximum))
        return maximum

    def source(self, path, historical=None):
        name = resolve(path).relative_to(ROOT).as_posix()
        require(name in self.sources, f"Historical source not frozen in producer sources.json: {name}")
        if name not in self.verified:
            frozen = self.sources[name]
            require(all(k in frozen for k in ("bytes", "sha256", "schema", "source")), f"Incomplete source metadata: {name}")
            current = file_info(name, schema=frozen["schema"], source=frozen["source"])
            require(current["sha256"] == frozen["sha256"] and current["bytes"] == frozen["bytes"], f"Source changed: {name}")
            self.verified[name] = current
            self.result["source_evidence"].append(current)
        current = self.verified[name]
        if historical is not None:
            require(current["sha256"] == historical["sha256"] and current["bytes"] == historical["bytes"], f"Historical manifest conflict: {name}")
        return current

    def run(self):
        cfg = config()
        scope = cfg["scope"]["p0"]
        require(scope["qa_draw_ids"] == DRAWS and scope["qa_unit_order"] == UNIT_KEY, "Frozen QA ordering/draws changed")
        require(scope["combo_set"] == "heldout" and scope["unit"] == "fraction", "Incorrect frozen scope/units")
        require(sorted(scope["train_seeds"]) == [17, 29, 43, 101, 202], "Incorrect seed roster")
        require(sorted(scope["snr_db"]) == [5, 15] and sorted(scope["architectures"]) == ["resnet", "tcn"], "Incorrect primary scope")
        analysis = read_json(DID / "did_analysis.json")
        status = read_json(DID / "status.json")
        require("P0-COMPLETE" in (status.get("status"), status.get("source_status")), "Independent complete acceptance requires P0-COMPLETE")
        frozen_sources = read_json(DID / "sources.json")["sources"]
        self.sources = {s["path"]: s for s in frozen_sources}
        require(len(self.sources) == len(frozen_sources), "Duplicate source manifest paths")
        for path in self.sources:
            self.source(path)
        self.result["producer_output_evidence"] = [file_info(DID / p) for p in (
            "sources.json", "status.json", "did_analysis.json", "did_long.csv", "did_summary.csv",
            "did_draw_audit.parquet", "four_arm_alignment.parquet", "record_input_identity.parquet")]
        self.result["config_evidence"] = file_info(OUT / "freeze/run_config.json")
        self.result["code_evidence"] = file_info(__file__)
        self.source(INPUT_MANIFEST)
        self.source(BOOT_MANIFEST)
        manifest = read_json(INPUT_MANIFEST)
        bootstrap = read_json(BOOT_MANIFEST)
        classes = manifest["data_identity"]["class_order"]
        require(classes == cfg["class_order"] == ["NORM", "MI", "STTC", "CD", "HYP"], "Class order conflict")
        self.result["class_order"] = classes
        self.source(bootstrap["cohort"]["path"], bootstrap["cohort"])
        self.source(bootstrap["draws"]["path"], bootstrap["draws"])
        self.source(manifest["cohort_file"]["path"], manifest["cohort_file"])
        with np.load(resolve(bootstrap["cohort"]["path"]), allow_pickle=False) as saved:
            cohort = {k: saved[k] for k in saved.files}
        with np.load(resolve(manifest["cohort_file"]["path"]), allow_pickle=False) as saved:
            for field in ("ids", "patient_ids", "indices", "y"):
                require(np.array_equal(saved[field], cohort[field]), f"Input/bootstrap cohort mismatch: {field}")
        unique, inverse = np.unique(cohort["patient_ids"], return_inverse=True)
        require(np.array_equal(unique, cohort["unique_patients"]) and np.array_equal(inverse, cohort["patient_inverse"]), "Patient multiplicity map mismatch")
        y = cohort["y"]
        n = len(y)
        require(y.shape == (2158, 5) and len(unique) == 1877 and np.isin(y, [0, 1]).all(), "Unexpected labels/cohort size")
        require(len(np.unique(cohort["ids"])) == n and len(np.unique(cohort["indices"])) == n, "Duplicate record identities")
        draws = np.load(resolve(bootstrap["draws"]["path"]), mmap_mode="r", allow_pickle=False)
        require(draws.shape == (2000, len(unique)) and np.issubdtype(draws.dtype, np.integer), "Patient draws schema mismatch")
        require((draws >= 0).all() and (draws.sum(axis=1) == len(unique)).all(), "Invalid patient draw multiplicities")
        weights = np.stack((np.ones(n), draws[0, inverse], draws[1999, inverse])).astype(np.float64)
        patient_pos = np.stack([np.bincount(inverse, weights=y[:, k], minlength=len(unique)) for k in range(5)], axis=1)
        patient_neg = np.bincount(inverse, minlength=len(unique))[:, None] - patient_pos
        expected_valid = np.empty(2000, dtype=bool)
        for start in range(0, 2000, 128):
            block = np.asarray(draws[start:start + 128], dtype=np.float64)
            expected_valid[start:start + 128] = ((block @ patient_pos > 0) & (block @ patient_neg > 0)).all(axis=1)
        self.result["patient_draw_validity"] = dict(valid=int(expected_valid.sum()), invalid=int((~expected_valid).sum()),
                                                  source=bootstrap["draws"], mapping=bootstrap["cohort"])
        cases = {c["case_id"]: c for c in manifest["cases"]}
        require(len(cases) == len(manifest["cases"]), "Duplicate original case IDs")
        groups = {g["group_id"]: g for g in manifest["groups"]}
        paired = {}
        for snr, condition in itertools.product(sorted(scope["snr_db"]), scope["test_structures"]):
            group = groups[f"bandpass__heldout__{condition}__s{snr}"]
            mapping = {}
            for case_id in group["case_ids"]:
                case = cases[case_id]
                require(case["kind"] == "bandpass" and case["combo_set"] == "heldout" and case["condition"] == condition and int(case["snr"]) == snr, "Original group descriptor mismatch")
                key = (case["combo_id"], int(case["noise_seed"]))
                require(key not in mapping, "Duplicate combo/noise unit")
                mapping[key] = case
            combos = sorted({k[0] for k in mapping})
            require(len(combos) == 10 and set(mapping) == set(itertools.product(combos, scope["noise_seeds"])), "Incomplete 10-by-5 original case grid")
            paired[(snr, condition)] = mapping
        for snr in scope["snr_db"]:
            require(paired[(snr, "electrode")].keys() == paired[(snr, "independent_rms")].keys(), "Unpaired E/I cases")
        units = sorted((a, d, s, t, c, z) for a, d, s, t in itertools.product(scope["architectures"], scope["snr_db"], scope["train_seeds"], scope["test_structures"]) for c, z in paired[(d, t)])
        frozen_units = {"qa_unit_A": units[0], "qa_unit_B": units[-1]}
        for name, unit in frozen_units.items():
            require(tuple(analysis[name][k] for k in UNIT_KEY) == unit, f"Frozen {name} is not the typed lexicographic endpoint")
            require(analysis[name]["case_id"] == paired[(unit[1], unit[3])][(unit[4], unit[5])]["case_id"], f"Incorrect {name} case ID")
        long = pd.read_csv(DID / "did_long.csv", float_precision="round_trip")
        summary = pd.read_csv(DID / "did_summary.csv", float_precision="round_trip")
        draw_table = pd.read_parquet(DID / "did_draw_audit.parquet")
        alignment = pd.read_parquet(DID / "four_arm_alignment.parquet")
        require(len(long) == 1000 and not long.duplicated(KEY).any(), "Long table primary-key coverage mismatch")
        require(long["unit"].eq("fraction").all() and summary["unit"].eq("fraction").all(), "Fraction/percentage unit mismatch")
        require(len(summary) == 4 and not summary.duplicated(["architecture", "snr_db"]).any(), "Summary group coverage mismatch")
        require(len(alignment) == 4000 and not alignment.duplicated(KEY + ["train_strategy", "test_structure"]).any(), "Four-arm prediction alignment mismatch")
        long = long.set_index(KEY)
        summary = summary.set_index(["architecture", "snr_db"])
        align_key = KEY + ["train_strategy", "test_structure"]
        alignment = alignment.set_index(align_key)
        checkpoints = {(r["model"], int(r["seed"]), r["strategy"]): r["checkpoint_sha256"] for r in manifest["training_completed_before_generation"]}
        clean_hash = cases["clean"]["input_sha256"]
        seed_results = {}
        prediction_count = 0
        for architecture, snr, seed in itertools.product(sorted(scope["architectures"]), sorted(scope["snr_db"]), sorted(scope["train_seeds"])):
            per_case = []
            for combo, noise in sorted(paired[(snr, "electrode")]):
                key = (architecture, snr, seed, combo, noise)
                arm_values = np.empty((3, 4))
                source_paths = {}
                model_hashes = []
                for arm_i, (strategy, condition) in enumerate(ARMS):
                    case = paired[(snr, condition)][(combo, noise)]
                    path = f"phase2/results/predictions/full/{strategy}/{architecture}/seed_{seed}/{case['case_id']}.npz"
                    info = self.source(path)
                    source_paths[METRICS[arm_i]] = info
                    row = alignment.loc[key + (strategy, condition)]
                    expected_checkpoint = checkpoints[(architecture, seed, strategy)]
                    for field, expected in dict(case_id=case["case_id"], prediction_path=path,
                                                prediction_sha256=info["sha256"], checkpoint_sha256=expected_checkpoint,
                                                clean_input_sha256=clean_hash, noisy_input_sha256=case["input_sha256"]).items():
                        require(row[field] == expected, f"Alignment mismatch {path}: {field}")
                    checkpoint_info = self.source(row["checkpoint_path"])
                    require(checkpoint_info["sha256"] == expected_checkpoint, f"Checkpoint bytes mismatch: {path}")
                    require(row["diagnostics_path"] == case["diagnostics_path"], f"Diagnostic source link mismatch: {path}")
                    self.source(case["diagnostics_path"])
                    with np.load(resolve(path), allow_pickle=False) as saved:
                        for field in ("ids", "patient_ids", "indices", "y"):
                            require(np.array_equal(saved[field], cohort[field]), f"Prediction cohort mismatch {path}: {field}")
                        expected_meta = dict(strategy=strategy, model=architecture, seed=seed, case_id=case["case_id"],
                                             condition=condition, combo_id=combo, combo_set="heldout", snr=snr, noise_seed=noise,
                                             kind="bandpass", stage="full", checkpoint_sha256=expected_checkpoint,
                                             input_sha256=case["input_sha256"], config_sha256=manifest["config_sha256"], matrix_sha256=manifest["matrix_sha256"])
                        for field, expected in expected_meta.items():
                            require(saved[field].item() == expected, f"Prediction metadata mismatch {path}: {field}")
                        probability = saved["p"]
                        require(probability.shape == y.shape and np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(), f"Invalid probabilities: {path}")
                        require(saved["p_sha256"].item() == raw_hash(probability), f"Stored probability checksum mismatch: {path}")
                        arm_values[:, arm_i] = literal_weighted_auc(y, probability, weights)
                    model_hashes.append(expected_checkpoint)
                    prediction_count += 1
                require(model_hashes[0] == model_hashes[2] and model_hashes[1] == model_hashes[3] and model_hashes[0] != model_hashes[1], "Four-arm checkpoint relationship failure")
                metrics = with_contrasts(arm_values)
                self.compare(f"long:{key}", metrics[0], long.loc[key, METRICS].to_numpy(dtype=float))
                per_case.append(metrics)
                for name, unit in frozen_units.items():
                    if key == (unit[0], unit[1], unit[2], unit[4], unit[5]):
                        unit_sources = analysis[name]["sources"]
                        for arm, evidence in source_paths.items():
                            require(unit_sources[arm]["prediction_path"] == evidence["path"] and unit_sources[arm]["prediction_sha256"] == evidence["sha256"], f"QA source mapping conflict: {name}/{arm}")
                            arm_index = METRICS.index(arm)
                            require(unit_sources[arm]["checkpoint_sha256"] == model_hashes[arm_index], f"QA checkpoint mapping conflict: {name}/{arm}")
                        expected_pairs = {t: paired[(snr, t)][(combo, noise)]["case_id"] for t in scope["test_structures"]}
                        require(analysis[name]["paired_case_ids"] == expected_pairs, f"QA paired case mapping conflict: {name}")
                        records = analysis["qa_case_checks"][name]
                        records_by_draw = {r["draw_id"]: r for r in records}
                        require(set(records_by_draw) == {-1, 0, 1999} and len(records) == 3, f"QA point/draw coverage mismatch: {name}")
                        comparisons = []
                        for i, draw_id in enumerate([-1, 0, 1999]):
                            observed = [records_by_draw[draw_id][m] for m in METRICS]
                            error = self.compare(f"{name}:draw={draw_id}", metrics[i], observed)
                            comparisons.append(dict(draw_id=draw_id, recomputed=dict(zip(METRICS, metrics[i])), producer=dict(zip(METRICS, observed)), max_absolute_error=error))
                        self.result["qa_units"][name] = dict(unit=dict(zip(UNIT_KEY, unit)), sources=source_paths, comparisons=comparisons)
            seed_results[(architecture, snr, seed)] = np.stack(per_case).mean(axis=0)
        require(prediction_count == 4000 and len(self.result["qa_units"]) == 2, "Incomplete independent probability coverage")
        self.result["coverage"].update(prediction_npz_files=prediction_count, paired_case_sets=1000, cases_per_seed_group=50, train_seeds_per_group=5)
        self.check_record_identity(cohort, paired, alignment)
        self.check_draws_and_summary(scope, seed_results, summary, draw_table, expected_valid)
        self.check_historical_group_metrics(scope, seed_results)
        self.result["source_hashes_verified"] = len(self.verified)
        self.result["status"] = "PASS" if not self.result["failures"] else "FAIL"
        self.result["release_pass"] = self.result["status"] == "PASS"

    def check_record_identity(self, cohort, paired, alignment):
        records = pd.read_parquet(DID / "record_input_identity.parquet")
        expected_cases = {c["case_id"] for mapping in paired.values() for c in mapping.values()}
        require(set(records["case_id"]) == expected_cases and len(records) == len(expected_cases) * len(cohort["ids"]), "Record-input identity coverage mismatch")
        require(not records.duplicated(["case_id", "record_id"]).any(), "Duplicate per-record input identities")
        for case_id, part in records.groupby("case_id", sort=False):
            part = part.set_index("record_id").loc[cohort["ids"]]
            require(np.array_equal(part["patient_id"], cohort["patient_ids"]) and np.array_equal(part["record_index"], cohort["indices"]), f"Per-record identity mapping conflict: {case_id}")
            for column in ("clean_input_sha256", "noisy_input_sha256"):
                require(part[column].astype(str).str.fullmatch("[0-9a-f]{64}").all(), f"Missing record input hash: {case_id}/{column}")
        require(alignment["record_identity_path"].eq("overnight_supplements_v2/did/record_input_identity.parquet").all(), "Alignment record-identity link mismatch")
        self.result["record_identity_audit"] = dict(rows=len(records), cases=len(expected_cases),
            verified="exact record/patient/index joins and complete SHA256 fields; case tensor identities validated against original NPZ/manifest",
            limitation="Per-record waveform hashes are producer evidence; this independent probability audit does not reconstruct waveforms.")

    def check_draws_and_summary(self, scope, seed_results, summary, table, expected_valid):
        draw_key = ["architecture", "snr_db", "draw_id", "train_seed"]
        expected_index = pd.MultiIndex.from_product([sorted(scope["architectures"]), sorted(scope["snr_db"]), range(2000), [-1] + sorted(scope["train_seeds"])], names=draw_key)
        require(len(table) == len(expected_index) and not table.duplicated(draw_key).any(), "Draw audit missing/duplicate rows")
        table = table.set_index(draw_key)
        require(table.index.difference(expected_index).empty and expected_index.difference(table.index).empty, "Draw audit primary-key mismatch")
        for architecture, snr in itertools.product(sorted(scope["architectures"]), sorted(scope["snr_db"])):
            seeds = sorted(scope["train_seeds"])
            independently_computed = np.stack([seed_results[(architecture, snr, seed)] for seed in seeds])
            aggregate = independently_computed.mean(axis=0)
            row = summary.loc[(architecture, snr)]
            require(row["combo_set"] == "heldout" and row["source_status"] == "P0-COMPLETE", "Summary scope/status mismatch")
            require(row["metric"] in ("macro_auroc", "absolute macro-AUROC", "absolute_macro_auroc"), "Summary metric is not absolute macro AUROC")
            for field, expected in dict(n_records=2158, n_patients=1877, n_heldout_combinations=10, n_noise_bases=5, n_draws=2000, n_valid_draws=int(expected_valid.sum()), n_invalid_draws=int((~expected_valid).sum())).items():
                require(int(row[field]) == expected, f"Summary count mismatch: {architecture}/{snr}/{field}")
            errors = {"point": self.compare(f"summary:{architecture}/{snr}", aggregate[0], row[METRICS].to_numpy(dtype=float))}
            seed_values = json.loads(row["seed_values"])
            if isinstance(seed_values, dict):
                seed_values = [seed_values[str(s)] for s in seeds]
            self.compare(f"seed-values:{architecture}/{snr}", independently_computed[:, 0, 6], seed_values)
            self.compare(f"seed-sd:{architecture}/{snr}", independently_computed[:, 0, 6].std(ddof=1), float(row["seed_sd"]))
            all_seed_draws = []
            for seed in seeds + [-1]:
                sub = table.xs((architecture, snr), level=("architecture", "snr_db")).xs(seed, level="train_seed").sort_index()
                require(np.array_equal(sub.index.to_numpy(), np.arange(2000)), "Draw IDs absent")
                require(sub["valid"].dtype == bool and np.array_equal(sub["valid"].to_numpy(), expected_valid), "Invalid class draw silently omitted or validity altered")
                numbers = sub[METRICS].to_numpy(dtype=float)
                require(np.isfinite(numbers[expected_valid]).all() and np.isnan(numbers[~expected_valid]).all(), "Invalid draw NaN filling or nonfinite valid metric")
                self.compare(f"draw-contrast-algebra:{architecture}/{snr}/{seed}", with_contrasts(numbers[:, :4]), numbers)
                expected = aggregate if seed == -1 else seed_results[(architecture, snr, seed)]
                for i, draw_id in enumerate(DRAWS, start=1):
                    errors[f"seed={seed}/draw={draw_id}"] = self.compare(f"raw-draw:{architecture}/{snr}/{seed}/{draw_id}", expected[i], numbers[draw_id])
                if seed != -1:
                    all_seed_draws.append(numbers)
                else:
                    aggregate_draws = numbers
            self.compare(f"all-draw-fixed-seed-mean:{architecture}/{snr}", np.stack(all_seed_draws).mean(axis=0), aggregate_draws)
            interval = np.percentile(aggregate_draws[expected_valid, 6], [2.5, 97.5]) if expected_valid.any() else np.full(2, np.nan)
            self.compare(f"CI-from-audit:{architecture}/{snr}", interval, [row["patient_ci_low"], row["patient_ci_high"]])
            self.result["summary_groups"].append(dict(architecture=architecture, snr_db=snr,
                point=dict(zip(METRICS, aggregate[0])), seed_values=dict(zip(seeds, independently_computed[:, 0, 6])),
                seed_sd=float(independently_computed[:, 0, 6].std(ddof=1)),
                frozen_aggregate_draws={str(draw): dict(zip(METRICS, aggregate[i])) for i, draw in enumerate(DRAWS, start=1)},
                errors=errors, patient_ci_from_producer_draw_audit=interval,
                interval_coverage="Recomputed percentile of all 2000 producer aggregate draws; raw probabilities independently recomputed only for draws 0 and 1999"))

    def check_historical_group_metrics(self, scope, seed_results):
        self.source(GROUP_METRICS)
        wanted = {f"bandpass__heldout__{t}__s{d}" for t in scope["test_structures"] for d in scope["snr_db"]}
        selected = []
        for chunk in pd.read_csv(resolve(GROUP_METRICS), chunksize=150000, float_precision="round_trip"):
            mask = chunk["group_id"].isin(wanted) & chunk["strategy"].isin(scope["train_strategies"]) & chunk["model"].isin(scope["architectures"]) & chunk["seed"].isin(scope["train_seeds"]) & chunk["metric"].eq("macro_auroc") & chunk["outcome"].eq("absolute")
            selected.extend(chunk.loc[mask].to_dict("records"))
        require(len(selected) == 80, "Original absolute group metric coverage mismatch")
        seen = set()
        comparisons = []
        for row in selected:
            key = (row["model"], int(row["snr"]), int(row["seed"]))
            arm = ARMS.index((row["strategy"], row["condition"]))
            unique_key = key + (arm,)
            require(unique_key not in seen and row["combo_set"] == "heldout" and row["combo_id"] == "aggregate" and int(row["n_cases"]) == 50 and int(row["n_noise_seeds"]) == 5, "Historical metric source definition conflict")
            seen.add(unique_key)
            expected = seed_results[key][0, arm]
            error = self.compare(f"historical-source:{unique_key}", expected, float(row["value"]))
            comparisons.append(dict(architecture=key[0], snr_db=key[1], train_seed=key[2], arm=METRICS[arm], recomputed=expected, historical=float(row["value"]), max_absolute_error=error, source_path=GROUP_METRICS, group_id=row["group_id"]))
        self.result["historical_group_metric_comparisons"] = comparisons


def main():
    started = time.perf_counter()
    audit = Audit()
    try:
        audit.run()
    except Exception as exc:
        audit.result["failures"].append(dict(check="fatal_source_or_schema_inconsistency", error=str(exc), traceback=traceback.format_exc()))
        audit.result["status"] = "FAIL"
        audit.result["release_pass"] = False
    audit.result["finished_at"] = now()
    audit.result["elapsed_seconds"] = time.perf_counter() - started
    result = finite_json(audit.result)
    write_json(OUT / "qa/did_recompute.json", result)
    write_json(DID / "independent_recompute.json", result)
    print(json.dumps({k: result[k] for k in ("status", "release_pass", "comparison_count", "max_absolute_error", "elapsed_seconds")}, allow_nan=False))
    return 0 if result["release_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
