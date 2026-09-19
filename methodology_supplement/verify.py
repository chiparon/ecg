"""Independent acceptance checks for actual predictions, estimands and provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
from PIL import Image
import numpy as np
import pandas as pd
from phase1_ecg_robustness.src.audit_noise import record_seed
from phase1_ecg_robustness.src.evaluate import classification_metrics
from phase1_ecg_robustness.src.noise_generators import make_noise_triplet
from .common import (WORKSPACE, array_sha256, file_info, load_config, load_reference, read_json,
                     require_freeze, resolve_path, save_json, sha256, stage_paths, write_csv)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _legacy_bridge(cfg, stage, paths, manifest, reference):
    ledger = pd.read_csv(paths["tables"] / "evaluation_index.csv")
    cases = {item["case_id"]: item for item in manifest["cases"]}
    rows = []
    baseline = WORKSPACE / "phase1_ecg_robustness/results/metrics"
    for row in ledger.to_dict("records"):
        case = cases[row["case_id"]]
        if case["kind"] != "clean" and (case["source_id"] != "standard" or case["snr"] not in (20, 10, 0)):
            continue
        model, seed = row["model"], int(row["seed"])
        if case["kind"] == "clean":
            old_path = baseline / "full" / model / f"seed_{seed}" / "clean_100_clean_all.npz"
        elif case["noise_seed"] == 8128:
            old_path = baseline / "full" / model / f"seed_{seed}" / f"bandpass_{case['snr']}_{case['condition']}_all.npz"
        else:
            old_path = baseline / "full_noise_repeats" / f"{model}_{seed}_noise{case['noise_seed']}_{case['snr']}_{case['condition']}.npz"
        with np.load(old_path, allow_pickle=False) as old, np.load(resolve_path(row["prediction_path"]), allow_pickle=False) as new:
            count = len(reference["ids"])
            for key in ("ids", "patient_ids", "y"):
                _check(np.array_equal(old[key][:count], reference[key]), f"Legacy bridge cohort mismatch: {old_path}: {key}")
            _check(np.array_equal(old["thresholds"], new["thresholds"]), "Legacy bridge changed thresholds")
            previous, current = old["p"][:count], new["p"]
            old_metrics, _ = classification_metrics(reference["y"], previous, old["thresholds"])
            new_metrics, _ = classification_metrics(reference["y"], current, new["thresholds"])
            differences = {metric: float(new_metrics[metric] - old_metrics[metric]) for metric in ("macro_auroc", "macro_ap", "macro_f1")}
            agreement = float(np.mean((current >= new["thresholds"]) == (previous >= old["thresholds"])))
            passed = (abs(differences["macro_auroc"]) <= cfg["gates"]["legacy_metric_abs_error"]
                      and abs(differences["macro_ap"]) <= cfg["gates"]["legacy_metric_abs_error"]
                      and abs(differences["macro_f1"]) <= cfg["gates"]["legacy_f1_abs_error"]
                      and agreement >= cfg["gates"]["legacy_label_agreement_min"])
            rows.append(dict(model=model, seed=seed, case_id=case["case_id"], n_records=count,
                             old_prediction_path=old_path.relative_to(WORKSPACE).as_posix(), old_prediction_sha256=sha256(old_path),
                             probabilities_max_abs=float(np.max(np.abs(current - previous))),
                             fixed_threshold_label_agreement=agreement, passed=passed, **differences))
    _check(len(rows) == 222, f"Legacy bridge coverage is not 222 cells: {len(rows)}")
    write_csv(paths["tables"] / "legacy_bridge.csv", rows)
    _check(all(row["passed"] for row in rows), "Legacy numerical bridge exceeds frozen tolerances; see legacy_bridge.csv")
    return dict(cells=len(rows), records_per_cell=len(reference["ids"]), all_passed=True,
                maximum_probability_difference=max(row["probabilities_max_abs"] for row in rows),
                minimum_label_agreement=min(row["fixed_threshold_label_agreement"] for row in rows),
                table=file_info(paths["tables"] / "legacy_bridge.csv"))


def _noise_and_matrices(cfg, paths, manifest, reference):
    matrices = read_json(resolve_path(manifest["matrices"]["path"]))["matrices"]
    _check(len(matrices) == 8, "Expected three physical matrices and five fixed randomized matrices")
    standard = np.asarray(next(item for item in matrices if item["id"] == "standard")["values"], dtype=np.float64)
    for item in matrices:
        matrix = np.asarray(item["values"], dtype=np.float64)
        _check(array_sha256(matrix) == item["matrix_sha256"], "Matrix fingerprint mismatch")
        rank = int(np.linalg.matrix_rank(matrix))
        _check((rank, 9 - rank, 12 - rank) == (item["diagnostics"]["rank"], item["diagnostics"]["right_nullity"], item["diagnostics"]["left_nullity"]), "Incorrect matrix nullity/rank")
        if item["family"] == "randomized":
            for axis in (0, 1):
                _check(np.array_equal(np.count_nonzero(matrix, axis=axis), np.count_nonzero(standard, axis=axis)), "Random matrix changed degree")
                _check(np.allclose(np.sum(matrix ** 2, axis=axis), np.sum(standard ** 2, axis=axis), atol=1e-14, rtol=0), "Random matrix changed marginal energy")
    diagnostics = pd.read_csv(resolve_path(manifest["diagnostics"]["per_record"]["path"]))
    _check(len(diagnostics) == 144 * len(reference["ids"]), "Incomplete actual-noise diagnostics")
    max_snr = float(diagnostics.max_snr_abs_error_db.max())
    max_rms = float(diagnostics.max_lead_rms_relative_error.max())
    _check(max_snr <= cfg["gates"]["snr_abs_error_db"] and max_rms <= cfg["gates"]["lead_rms_relative_error"], "Final injected residual failed SNR/RMS gates")
    low = diagnostics[diagnostics.source_id.eq("band_0p05_5")]
    _check(np.allclose(low.source_fft_df_hz, .01) and np.allclose(low.source_lowest_retained_hz, .05), "Low-band source does not include 0.05 Hz")
    x = np.load(paths["inputs"] / "clean.npy", mmap_mode="r", allow_pickle=False)
    selected = sorted(set([0, 1, 2, len(x) // 2, len(x) - 2, len(x) - 1]))
    array_count = 0
    for noise_seed in cfg["phase1_noise_seeds"]:
        files = {condition: np.load(resolve_path(next(case["noise_path"] for case in manifest["cases"]
                     if case["source_id"] == "standard" and case["noise_seed"] == noise_seed and case["condition"] == condition)),
                     mmap_mode="r", allow_pickle=False) for condition in cfg["conditions"]}
        for index in selected:
            actual_seed = record_seed(noise_seed, int(reference["ids"][index]), "bandpass")
            expected = make_noise_triplet(x[index], 100, 0, actual_seed, band=(.5, 40.0))
            for condition, values in files.items():
                _check(np.array_equal(values[index], expected[condition].astype(np.float32)), "Canonical source stream differs from original phase-one generator")
                array_count += 1
    return dict(diagnostic_rows=len(diagnostics), max_final_residual_snr_error_db=max_snr,
                max_final_residual_lead_rms_relative_error=max_rms, matrices=8,
                original_generator_records_checked=selected, original_generator_noise_seeds=cfg["phase1_noise_seeds"],
                original_generator_arrays_exact=array_count, source_fft_df_hz=.01,
                lower_band_source_frequency_hz=.05, ten_second_observed_fft_df_hz=.1)


def _summary_points(cfg, paths, manifest):
    summary = pd.read_csv(paths["tables"] / "new_summary.csv")
    index = pd.read_csv(paths["tables"] / "evaluation_index.csv")
    descriptors = pd.DataFrame(manifest["cases"])[["case_id", "source_id", "condition", "snr"]]
    data = index.merge(descriptors, on="case_id", validate="many_to_one")
    _check(len(summary) == 930, "New statistical summary grid incomplete")
    bands = ("band_0p05_5", "band_5_15", "band_15_40", "band_0p5_40")
    groups = {
        "snr": [("standard", snr) for snr in cfg["snrs"]],
        "matrix": [(source, snr) for source in ("standard", "limb_only", "precordial_only", "randomized") for snr in cfg["ablation_snrs"]],
        "matrix_instance": [(f"random_{i:02d}", snr) for i in range(5) for snr in cfg["ablation_snrs"]],
        "band": [(source, snr) for source in bands for snr in cfg["ablation_snrs"]],
    }
    outcomes = ("structure_effect", "electrode_absolute", "independent_rms_absolute", "electrode_drop", "independent_rms_drop")
    expected = {(analysis, model, source, snr, metric, outcome) for analysis, settings in groups.items()
                for model in cfg["models"] for source, snr in settings
                for metric in ("macro_auroc", "macro_f1", "ece") for outcome in outcomes}
    actual = set(summary[["analysis", "model", "source_id", "snr", "metric", "outcome"]].itertuples(index=False, name=None))
    _check(actual == expected and len(actual) == len(summary), "A prespecified summary cell is missing, duplicated or unexpected")
    detail_counts = {"new_seed_effects": 2790, "new_noise_effects": 16740, "random_matrix_effects": 900}
    for name, count in detail_counts.items():
        _check(len(pd.read_csv(paths["tables"] / f"{name}.csv")) == count, f"Incomplete variability table: {name}")
    checked, maximum_error = 0, 0.0
    for row in summary.itertuples(index=False):
        sources = [f"random_{i:02d}" for i in range(5)] if row.source_id == "randomized" else [row.source_id]
        frame = data[data.model.eq(row.model) & data.source_id.isin(sources) & data.snr.eq(row.snr)]
        e = frame[frame.condition.eq("electrode")].groupby("seed")[row.metric].mean()
        i = frame[frame.condition.eq("independent_rms")].groupby("seed")[row.metric].mean()
        clean = data[data.model.eq(row.model) & data.condition.eq("clean")].set_index("seed")[row.metric]
        values = {"structure_effect": e - i, "electrode_absolute": e, "independent_rms_absolute": i,
                  "electrode_drop": clean - e, "independent_rms_drop": clean - i}[row.outcome]
        _check(len(values) == 3 and values.notna().all(), "Point-check seed pairing incomplete")
        error = max(abs(float(values.mean()) - row.estimate), abs(float(values.std(ddof=1)) - row.seed_sd))
        _check(error <= 2e-7, f"Summary disagrees with independent metric-ledger aggregation: {row}")
        _check(row.seed_n == 3 and row.n_invalid < row.n_bootstrap and row.patient_ci_low <= row.patient_ci_high, "Invalid uncertainty metadata")
        maximum_error = max(maximum_error, error)
        checked += 1
    return dict(rows_checked=checked, maximum_point_or_sd_error=maximum_error,
                exact_prespecified_grid=True, detailed_variability_rows=detail_counts)


def _existing_tables(cfg, paths):
    report = read_json(paths["logs"] / "reused_statistics.json")
    _check(report["status"] == "completed" and report["config_sha256"] == cfg["_config_sha256"], "Reused analyses are incomplete")
    required = {"phase2_snr_summary": 120, "phase2_snr_seed_effects": 600, "effect_plane": 684,
                "effect_plane_seed_values": 3420, "pareto": 8, "pareto_seed_values": 40}
    for name, count in required.items():
        _check(len(pd.read_csv(paths["tables"] / f"{name}.csv")) == count, f"Incomplete {name}")
    bootstrap_root = WORKSPACE / "phase2/results/tables/full/patient_bootstrap"
    old_manifest = read_json(bootstrap_root / "manifest.json")
    group_indices = {name: i for i, name in enumerate(old_manifest["group_ids"])}
    old_groups = read_json(WORKSPACE / "phase2/results/test_inputs/full/manifest.json")["groups"]
    lookup = {}
    for group in old_groups:
        if group["kind"] == "clean" or group["condition"] not in cfg["conditions"]:
            continue
        if group["combo_set"] == "all":
            combo = "all"
        elif group["combo_id"] != "aggregate":
            combo = group["combo_id"]
        else:
            continue
        lookup[(group["kind"], combo, int(group["snr"]), group["condition"])] = group_indices[group["group_id"]]
    arrays = {(strategy, model, seed): np.load(bootstrap_root / f"{strategy}__{model}__seed_{seed}.npy", mmap_mode="r", allow_pickle=False)
              for strategy in cfg["strategies"] for model in cfg["models"] for seed in cfg["phase2_training_seeds"]}
    def check_axis(row, name, values):
        estimate = values[:, 0].mean()
        sd = values[:, 0].std(ddof=1)
        draws = values[:, 1:].mean(axis=0)
        finite = draws[np.isfinite(draws)]
        interval = np.quantile(finite, [.025, .975])
        actual = np.array([row[f"{name}_mean"], row[f"{name}_seed_sd"], row[f"{name}_ci_low"], row[f"{name}_ci_high"]])
        _check(np.allclose(actual, [estimate, sd, *interval], rtol=0, atol=2e-12), f"Archived bootstrap axis differs: {name}")
    plane = pd.read_csv(paths["tables"] / "effect_plane.csv")
    for row in plane.to_dict("records"):
        group_e = lookup[(row["kind"], row["combo_id"], int(row["snr"]), "electrode")]
        group_i = lookup[(row["kind"], row["combo_id"], int(row["snr"]), "independent_rms")]
        x, y = [], []
        for seed in cfg["phase2_training_seeds"]:
            clean = arrays[("clean_only", row["model"], seed)]
            strategy = arrays[(row["strategy"], row["model"], seed)]
            x.append(clean[group_e, :, 0] - clean[group_i, :, 0])
            y.append(strategy[group_e, :, 0] - clean[group_e, :, 0])
        check_axis(row, "x", np.stack(x))
        check_axis(row, "y", np.stack(y))
    pareto = pd.read_csv(paths["tables"] / "pareto.csv")
    clean_index, joint_index = group_indices["clean"], group_indices["primary_joint"]
    for row in pareto.to_dict("records"):
        x, y = [], []
        for seed in cfg["phase2_training_seeds"]:
            clean = arrays[("clean_only", row["model"], seed)]
            strategy = arrays[(row["strategy"], row["model"], seed)]
            x.append(clean[clean_index, :, 0] - strategy[clean_index, :, 0])
            y.append(strategy[joint_index, :, 0] / (strategy[clean_index, :, 0] + 1e-12)
                     - clean[joint_index, :, 0] / (clean[clean_index, :, 0] + 1e-12))
        check_axis(row, "x", np.stack(x))
        check_axis(row, "y", np.stack(y))
    metrics = ("macro_auroc", "macro_f1", "ece")
    ledger = pd.read_csv(WORKSPACE / "phase2/results/tables/full/metrics.csv",
                         usecols=["model", "strategy", "seed", "kind", "combo_id", "condition", "snr", *metrics])
    ledger = ledger[ledger.kind.eq("bandpass") & ledger.combo_id.eq("all") & ledger.condition.isin(cfg["conditions"])]
    curves = pd.read_csv(paths["tables"] / "phase2_snr_summary.csv")
    curve_error = 0.0
    auc_intervals = 0
    for row in curves.to_dict("records"):
        selected = ledger[ledger.model.eq(row["model"]) & ledger.strategy.eq(row["strategy"]) & ledger.snr.eq(row["snr"])]
        _check(len(selected) == 50, "Phase-two curve did not use exactly five seeds and five noises per structure")
        by_seed = selected.groupby(["seed", "condition"])[row["metric"]].mean().unstack("condition")
        values = by_seed.electrode - by_seed.independent_rms
        error = max(abs(values.mean() - row["estimate"]), abs(values.std(ddof=1) - row["seed_sd"]))
        _check(error < 2e-7, "Phase-two curve disagrees with original metric ledger")
        curve_error = max(curve_error, float(error))
        if row["metric"] == "macro_auroc":
            e = lookup[("bandpass", "all", int(row["snr"]), "electrode")]
            i = lookup[("bandpass", "all", int(row["snr"]), "independent_rms")]
            distributions = np.stack([arrays[(row["strategy"], row["model"], seed)][e, :, 0]
                                      - arrays[(row["strategy"], row["model"], seed)][i, :, 0]
                                      for seed in cfg["phase2_training_seeds"]])
            interval = np.quantile(distributions[:, 1:].mean(axis=0), [.025, .975])
            _check(np.allclose(interval, [row["patient_ci_low"], row["patient_ci_high"]], atol=2e-12, rtol=0),
                   "Phase-two AUROC curve lost paired bootstrap alignment")
            auc_intervals += 1
    return dict(counts=required, plane_paired_axes_independently_checked=len(plane) * 2,
                pareto_paired_axes_independently_checked=len(pareto) * 2, better_direction="upper_left",
                curve_rows_independently_checked=len(curves), curve_maximum_point_or_sd_error=curve_error,
                curve_auc_intervals_independently_checked=auc_intervals)


def run(config=None, stage="full"):
    cfg = load_config(config)
    freeze = require_freeze(cfg, stage)
    paths = stage_paths(cfg, stage)
    merged = read_json(paths["logs"] / "evaluation_merge.json")
    stats = read_json(paths["logs"] / "new_statistics.json")
    _check(merged["status"] == "completed" and merged["n_predictions"] == 1950 and merged["dgx_overlap_seconds"] > 0, "Execution/concurrency proof is missing")
    _check(stats["status"] == "completed" and stats["n_distributions"] == 1950, "New patient statistics are incomplete")
    manifest = read_json(paths["inputs"] / "manifest.json")
    reference = load_reference(cfg, stage)
    result = dict(status="running", stage=stage, config_sha256=cfg["_config_sha256"])
    destination = paths["logs"] / ("smoke_verification.json" if stage == "smoke" else "numeric_verification.json")
    save_json(destination, result)
    try:
        result["noise"] = _noise_and_matrices(cfg, paths, manifest, reference)
        result["bridge"] = _legacy_bridge(cfg, stage, paths, manifest, reference)
        result["new_summary"] = _summary_points(cfg, paths, manifest)
        if stage == "full":
            result["reused_summary"] = _existing_tables(cfg, paths)
        for item in freeze["protected_files"]:
            _check(sha256(resolve_path(item["path"])) == item["sha256"], f"Protected artifact changed: {item['path']}")
        result.update(status="passed", checked_at=datetime.now(timezone.utc).isoformat(),
                      protected_files_unchanged=len(freeze["protected_files"]),
                      dgx_overlap_seconds=merged["dgx_overlap_seconds"], technical_only=stage == "smoke")
        save_json(destination, result)
        print(f"{stage} numerical/provenance verification passed", flush=True)
        return result
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(destination, result)
        raise


class _ReportDOM(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.identifiers = set()
        self.links = []
        self.images = []
        self.numbers = {}
        self.active_number = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        identifier = attrs.get("id")
        if identifier:
            _check(identifier not in self.identifiers, f"Duplicate report element ID: {identifier}")
            self.identifiers.add(identifier)
        if tag == "a":
            self.links.append(attrs["href"])
        elif tag == "img":
            self.images.append(attrs["src"])
        elif tag == "span" and identifier and identifier.startswith("number-"):
            self.active_number = identifier
            self.numbers[identifier] = ""

    def handle_data(self, data):
        if self.active_number is not None:
            self.numbers[self.active_number] += data

    def handle_endtag(self, tag):
        if tag == "span":
            self.active_number = None


def verify_publication(config=None):
    cfg = load_config(config)
    require_freeze(cfg, "full")
    paths = stage_paths(cfg, "full")
    numerical = read_json(paths["logs"] / "numeric_verification.json")
    figures = read_json(paths["figures"] / "manifest.json")
    report = read_json(paths["reports"] / "manifest.json")
    _check(numerical["status"] == "passed", "Numerical acceptance is missing")
    for item in (numerical, figures, report):
        _check(item["config_sha256"] == cfg["_config_sha256"], "Publication configuration mismatch")
    _check(figures["status"] == report["status"] == "completed", "Publication package is incomplete")

    checked_paths = set()
    def check_file(item):
        if item["path"] in checked_paths:
            return
        path = resolve_path(item["path"])
        _check(path.is_file() and path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"],
               f"Publication artifact is missing, changed or stale: {item['path']}")
        checked_paths.add(item["path"])

    expected_figures = {
        "matrix_heatmaps", "matrix_bipartite", "matrix_unit_propagation", "matrix_singular_values",
        "matrix_structure_effects", "matrix_correlation_drop", "snr_macro_auroc", "snr_macro_f1",
        "snr_ece", "band_drop_heatmap", "band_structure_effects", "band_power_effect",
        "evaluation_training_plane", "evaluation_training_nstdb", "pareto",
    }
    _check({item["id"] for item in figures["figures"]} == expected_figures and len(figures["figures"]) == 15,
           "Five-deliverable figure inventory is incomplete")
    for item in figures["figures"]:
        _check(set(item["files"]) == {"png", "pdf", "svg"}, "Missing publication figure format")
        for value in item["files"].values():
            check_file(value)
        for value in item["sources"]:
            check_file(value)
        with Image.open(resolve_path(item["files"]["png"]["path"])) as image:
            _check(image.format == "PNG", "Invalid figure image format")
            image.load()
    check_file(figures["implementation"])
    for key in ("report", "implementation", "figures_manifest", "numeric_verification"):
        check_file(report[key])
    for value in report["sources"].values():
        check_file(value)
    report_path = resolve_path(report["report"]["path"])
    dom = _ReportDOM()
    dom.feed(report_path.read_text(encoding="utf-8"))
    _check(len(dom.images) == 15, "The report does not embed all fifteen actual figures")
    sources = {name: pd.read_csv(resolve_path(value["path"])) for name, value in report["sources"].items()}
    for item in report["numeric_cells"]:
        value = float(sources[item["source"]].loc[item["row"], item["column"]]) * item["scale"]
        expected = format(value, f"{'+' if item['signed'] else ''}.{item['digits']}f")
        _check(dom.numbers.get(item["id"]) == expected, f"Rendered report number disagrees with CSV: {item['id']}")
    _check(len(dom.numbers) == len(report["numeric_cells"]), "Untracked numeric report spans")
    for link in dom.links + dom.images:
        parsed = urlsplit(link)
        _check(not parsed.scheme and not parsed.netloc, "Unexpected external report dependency")
        if parsed.path:
            _check((report_path.parent / unquote(parsed.path)).resolve().is_file(), f"Broken local publication link: {link}")
        elif parsed.fragment:
            _check(unquote(parsed.fragment) in dom.identifiers, f"Broken report section link: {link}")
    result = dict(status="passed", config_sha256=cfg["_config_sha256"], checked_at=datetime.now(timezone.utc).isoformat(),
                  figures=15, figure_files=45, report_numeric_cells=len(dom.numbers),
                  local_links_and_images_checked=len(dom.links) + len(dom.images), unique_hashes_checked=len(checked_paths),
                  report=file_info(report_path), figures_manifest=file_info(paths["figures"] / "manifest.json"),
                  visual_review="Separate actual browser and figure inspection is required; this check does not claim visual acceptance.")
    save_json(paths["logs"] / "publication_verification.json", result)
    print(f"Publication data/provenance passed: 15 figures, 45 files, {len(dom.numbers)} numeric report cells", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--publication", action="store_true")
    args = parser.parse_args()
    if args.publication:
        _check(args.stage == "full", "Publication acceptance requires the full cohort")
        verify_publication(args.config)
    else:
        run(args.config, args.stage)


if __name__ == "__main__":
    main()
