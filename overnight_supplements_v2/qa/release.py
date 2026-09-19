"""Verify current publication artifacts and emit the task's release checklist."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re

from overnight_supplements_v2.shared.common import OUT, ROOT, config, file_info, now, read_json, resolve, write_json
from PIL import Image
import pyarrow.parquet as pq

REQUIRED = [
    "freeze/source_manifest.json", "freeze/run_config.json", "freeze/throughput_benchmark.json", "freeze/code_manifest.json",
    "did/status.json", "did/did_analysis.json", "did/four_arm_alignment.parquet", "did/did_long.csv", "did/did_summary.csv", "did/did_draw_audit.parquet", "did/independent_recompute.json", "did/record_input_identity.parquet",
    "snrp/status.json", "snrp/snrp_analysis.json", "snrp/snr_p_per_record.parquet", "snrp/geometry_summary.csv", "snrp/auroc_overlay.csv", "snrp/overlay_alignment.parquet", "snrp/qa.json",
    "qa/did_recompute.json", "qa/snrp_recompute.json", "qa/source_protection.json", "qa/visual_acceptance.json", "qa/p1_correction_record.json",
    "qa/lossless_alignment_pack.json",
    "third_architecture/gate0_manifest.json", "third_architecture/gate0_status.md", "reports/overnight_supplement_final_report.md", "reports/manifest.json",
]


def require(value, message):
    if not value:
        raise ValueError(message)


def csv_rows(relative):
    with (OUT / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_tokens(value):
    return [float(number) for number in re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", value)]


def main():
    cfg = config()
    did_qa = read_json(OUT / "qa/did_recompute.json")
    snrp_qa = read_json(OUT / "qa/snrp_recompute.json")
    protected = read_json(OUT / "qa/source_protection.json")
    visual = read_json(OUT / "qa/visual_acceptance.json")
    require(did_qa.get("release_pass") is True, "P0 independent acceptance failed")
    require(snrp_qa["status"] == "passed" and snrp_qa["n_failures"] == 0, "P1 independent acceptance failed")
    p1_errors = {item["name"]: item for item in snrp_qa["checks"] if "max_absolute_error" in item}
    for entry in snrp_qa["sources"]:
        if entry["path"].startswith("overnight_supplements_v2/"):
            require(file_info(entry["path"])["sha256"] == entry["sha256"], "P1 independent QA refers to a stale derived artifact")
    require(protected["status"] == "passed" and not protected["failures"], "Historical source protection failed")
    require(visual["status"] == "passed", "Actual figure visual acceptance missing")
    require(read_json(OUT / "qa/p1_correction_record.json")["status"] == "resolved_and_independently_verified", "Overlay correction is unresolved")
    packed = read_json(OUT / "qa/lossless_alignment_pack.json")
    require(packed["status"] == "passed" and packed["arrow_table_equality_with_metadata"], "Lossless alignment conservation failed")
    require(file_info(OUT / "snrp/overlay_alignment.parquet") == packed["after"], "Packed alignment bytes changed")
    frozen_code = read_json(OUT / "freeze/code_manifest.json")
    require(protected["code_manifest"] == file_info(OUT / "freeze/code_manifest.json"), "Code protection refers to stale manifest")
    for entry in frozen_code["scripts"] + [frozen_code["run_config"]]:
        require(file_info(entry["path"]) == entry, "Frozen code or configuration changed")
    for relative in REQUIRED:
        require((OUT / relative).is_file() and (OUT / relative).stat().st_size > 0, "Required deliverable missing: " + relative)
        require((OUT / relative).stat().st_size < 100*1024*1024, "Required artifact exceeds GitHub single-blob limit: " + relative)
    did_status, p1_status = read_json(OUT / "did/status.json"), read_json(OUT / "snrp/status.json")
    require(did_status["status"] == "P0-COMPLETE" and did_status["analysis_complete"], "P0 result is incomplete")
    require(p1_status["geometry_status"] == "P1-GEO-DIRECT" and p1_status["performance_overlay_status"] == "P1-OVERLAY", "P1 release tiers differ")
    require(p1_status["anomaly_count"] == p1_status["overlay_anomaly_count"] == 0, "Unadjudicated real-data anomalies")
    counts = {"did/four_arm_alignment.parquet": 4000, "did/record_input_identity.parquet": 431600, "did/did_draw_audit.parquet": 48000, "snrp/snr_p_per_record.parquet": 271908, "snrp/overlay_alignment.parquet": 1631448}
    for relative, expected in counts.items():
        metadata = pq.ParquetFile(OUT / relative)
        require(metadata.metadata.num_rows == expected, f"Wrong release row count: {relative}")
        names = set(metadata.schema_arrow.names)
        require(not names.intersection({"p", "probabilities", "raw_probabilities", "waveform", "raw_waveform", "signals"}), "Prohibited copied raw values in " + relative)
    did = {(row["architecture"], int(row["snr_db"])): row for row in csv_rows("did/did_summary.csv")}
    require(set(did) == {(a, d) for a in ("resnet", "tcn") for d in (15, 5)}, "Wrong primary scope")
    for row in did.values():
        require(row["combo_set"] == "heldout" and row["unit"] == "fraction", "Wrong scope or unit")
        require((int(row["n_records"]), int(row["n_patients"]), int(row["n_heldout_combinations"]), int(row["n_noise_bases"])) == (2158, 1877, 10, 5), "Wrong P0 sample definition")
        require(int(row["n_valid_draws"]) + int(row["n_invalid_draws"]) == int(row["n_draws"]) == 2000, "Invalid draw accounting")
    geometry = {(int(row["snr_db"]), row["condition"]): row for row in csv_rows("snrp/geometry_summary.csv") if row["metric"] == "snr_p_db"}
    require(len(geometry) == 21, "Missing geometry conditions")
    report_path = OUT / "reports/overnight_supplement_final_report.md"
    report = report_path.read_text(encoding="utf-8")
    numeric_cells, primary_rows, seed_rows, geometry_rows = 0, 0, 0, 0
    max_rounding_error = 0.0
    for line in report.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells[0] in ("resnet", "tcn"):
            row = did[(cells[0], int(cells[1]))]
            if len(cells) == 12:
                expected = [100 * float(row[name]) for name in ("M_EE", "M_IE", "M_EI", "M_II", "g_E", "g_I", "gamma", "seed_sd", "patient_ci_low", "patient_ci_high")]
                displayed = [float(cell) for cell in cells[2:10]] + numeric_tokens(cells[10])
                require(len(displayed) == len(expected), "Malformed primary display")
                for shown, raw in zip(displayed, expected):
                    error = abs(shown - raw)
                    require(error <= 5.000001e-5, "Primary report differs beyond four-decimal rounding")
                    max_rounding_error = max(max_rounding_error, error)
                require(cells[11] == f"{row['n_valid_draws']}/{row['n_draws']}", "Draw count display differs")
                primary_rows += 1
                numeric_cells += len(expected)
            elif len(cells) == 7:
                values = json.loads(row["seed_values"])
                for cell, seed in zip(cells[2:], cfg["scope"]["p0"]["train_seeds"]):
                    require(abs(float(cell) - 100 * values[str(seed)]) <= 5.000001e-7, "Seed display differs")
                seed_rows += 1
                numeric_cells += 5
        elif len(cells) == 10 and cells[0] in ("0", "5", "10") and cells[1] in cfg["scope"]["p1"]["conditions"]:
            row = geometry[(int(cells[0]), cells[1])]
            displayed = [float(cells[2]), float(cells[3]), *numeric_tokens(cells[4])]
            expected = [float(row[name]) for name in ("mean", "median", "p05", "p95")]
            require(len(displayed) == 4 and all(abs(a-b) <= 5.000001e-7 for a, b in zip(displayed, expected)), "Geometry display differs")
            require(cells[5:] == [row[name] for name in ("n_rows", "n_records", "n_patients", "n_noise_bases", "n_anomalous")], "Geometry sample counts differ")
            geometry_rows += 1
            numeric_cells += 9
    require((primary_rows, seed_rows, geometry_rows) == (4, 4, 21), "Incomplete numeric report tables")
    links = []
    for target in re.findall(r"\]\(([^)]+)\)", report):
        if target.startswith(("https://", "http://", "#")):
            continue
        destination = (report_path.parent / target.split("#", 1)[0]).resolve()
        if destination == (OUT / "qa/release_checklist.md").resolve():
            continue  # Written below only after every preceding acceptance check passes.
        require(destination.is_file(), "Broken report link: " + target)
        links.append(target)
    figures = read_json(OUT / "snrp/figure_manifest.json")["files"]
    require(len(figures) == 9, "Missing requested scientific figure formats")
    visual_files = {entry["path"]: entry for entry in visual["files"]}
    for entry in figures:
        actual = file_info(entry["path"])
        require(actual["sha256"] == entry["sha256"] and actual["bytes"] == entry["bytes"], "Stale figure manifest")
        require(entry["path"] in visual_files and actual["sha256"] == visual_files[entry["path"]]["sha256"], "Figure changed since visual acceptance")
        if entry["path"].endswith(".png"):
            with Image.open(resolve(entry["path"])) as image:
                image.verify()
    presentation = read_json(OUT / "qa/presentation_corrections.json")
    for relative, expected in presentation["numeric_artifact_identities_before"].items():
        require(file_info(OUT / relative) == expected, "Presentation refresh changed accepted numerical data")
    source_info = file_info(OUT / "freeze/source_manifest.json")
    code_info = file_info(OUT / "freeze/code_manifest.json")
    require(protected["source_manifest"]["sha256"] == source_info["sha256"], "Source protection refers to stale manifest")
    manifest = read_json(OUT / "reports/manifest.json")
    require(manifest["report"] == file_info(report_path), "Report bytes differ from render manifest")
    require(manifest["source_manifest"]["sha256"] == source_info["sha256"] and manifest["code_manifest"]["sha256"] == code_info["sha256"], "Report provenance is stale")
    require(manifest["paper_insertion"] in report, "Paper insert differs from accepted render payload")
    unexpected_raw = [path.relative_to(OUT).as_posix() for path in OUT.rglob("*") if path.is_file() and "vendor" not in path.parts and path.suffix.lower() in (".npy", ".npz", ".pt", ".pth", ".dat", ".hea")]
    require(not unexpected_raw, "Raw array/checkpoint copies in release namespace")
    checklist = [
        "# Release checklist — overnight supplements v2", "", f"Verified at {now()}. Final scientific release: **PASS**. Git commit/push identity is reported separately after this immutable release is committed.", "",
        "training_invocations = 0", "inference_invocations = 0", "",
        f"- [x] Source paths, sizes, SHA-256 and source schema: `freeze/source_manifest.json`, SHA-256 `{source_info['sha256']}`; {protected['historical_local_sources_checked']} local consumed historical files rehashed, no changes.",
        f"- [x] Code and configuration versions: `freeze/code_manifest.json`, SHA-256 `{code_info['sha256']}`; frozen baseline `{cfg['baseline_commit']}`.",
        "- [x] P0 sole primary scope: Gaussian, heldout combinations,15/5dB,ResNet/TCN,five seeds. No train-combination calculation or selection by result.",
        "- [x] P0 four-arm common identities; same-test-structure input equality; within-training-strategy checkpoint equality; separate clean/noisy/checkpoint/prediction hashes. E inputs never required to equal I inputs.",
        "- [x] P0 complete4000noisy probability sources,4020point checks including20clean references;431600unique case-record input identities;1000paired case rows;20seed values;4summary rows;48000draw audit rows.",
        "- [x] Lowest-case pairing before equal condition/noise aggregation; within-seed DiD before five-seed mean;2000original multiplicity draws;NaN preservation and ddof1seed SD separate from conditional patient CI.",
        f"- [x] P0 independent raw-probability/weight recomputation: QA units A/B, point and draws0/1999, all4groups/all20seeds; {did_qa['comparison_count']} comparisons; max error `{did_qa['max_absolute_error']}` versus1e-12. Remaining1998draws audited for consistency, not claimed independently recomputed from raw probabilities.",
        "- [x] P1 GEO-DIRECT: all126final input hashes,271908record rows; old545974q rows and271908input diagnostics crosschecked. P symmetry/idempotence gates passed; residual defined from finalfloat32 minus cleanfloat32 in float64.",
        "- [x] P1 independent five fixedIDs×126cases=630SVD geometry checks; nine boundary fixtures; noepsilon/clipping. Real geometry anomalies0; overlay anomalies0.",
        "- [x] P1 overlay:756noisy×2158=1631448exact record joins;762source metrics including6cleanreferences; independent42noisy+6clean metric checks; E/Iweights not replicated. Five-class positive/negative exposure distributions retained.",
        f"- [x] Publication: {numeric_cells} displayed numeric cells/counts checked against accepted CSVs; {len(links)} local links;3visually inspectedPNGfigures with9hashedPDF/PNG/SVGfiles. Full-resolution text/legends/axes/captions checked. Ground-truth labels not described as prediction true-positives.",
        "- [x] Initial54overlay rejections retained in qa/attempts with root-cause/correction record; repairedone-to-manyworkerprovenance, then fullproductionandindependentQApassed. No historical scientific artifact correction was needed.",
        "- [x] Caption and completed-status-note corrections changed no accepted numeric artifacts;10artifact identities retained in qa/presentation_corrections.json.",
        f"- [x] Lossless alignment packaging: {packed['before']['bytes']}→{packed['after']['bytes']} bytes; {packed['rows']} rows×{packed['columns']} columns exactly equal, including schema/nulls/order. Actual production writer rerun; independent P1 QA repeated on final bytes. No sample or column omission.",
        f"- [x] P1 independent maximum errors: absolute projected SNR `{p1_errors['svd_630_snr_p_db']['max_absolute_error']}` dB (absolute tolerance1e-10); noise q `{p1_errors['svd_630_q_noise']['max_absolute_error']}` (absolute tolerance1e-12); largest of four energy absolute errors `{max(p1_errors['svd_630_'+name]['max_absolute_error'] for name in ('clean_total_energy','noise_total_energy','clean_projected_energy','noise_projected_energy'))}` (each comparison passed relative tolerance1e-12, not an absolute-energy1e-12 gate).",
        "- [x] P3 DEFERRED: publiccommit/license/topology/path/deviceinventoryonly; zero modelimports/forward/backward/epoch timing.",
        "- [x] Unexecuted by design: training,inference,checkpoint/thresholdselection,pvalues/newtestingfamilies,binningstandardization,reweightedAUROC,propensity,equalexposurecomparisons,mediation,thirdarchitectureexecution.",
        "- [x] No raw waveforms,input caches,original probability arrays or checkpoints copied into the release namespace. Isolated packagevendor excluded from Git. Audited scalar/linkParquet files may be committed.",
        "- [x] Producer status files are timestamped production snapshots; final independent acceptance is the separate qa evidence and this checklist, not retroactively inferred from a producer's pending-QA label.", "",
        "## Exact evidence paths", "", "- `qa/did_recompute.json` and `did/independent_recompute.json`", "- `qa/snrp_recompute.json` and `snrp/qa.json`", "- `qa/source_protection.json`, `qa/visual_acceptance.json`, `qa/publication_verification.json`", "- `reports/overnight_supplement_final_report.md`, `reports/manifest.json`", "",
    ]
    checklist_path = OUT / "qa/release_checklist.md"
    checklist_path.write_text("\n".join(checklist), encoding="utf-8")
    result = {"status": "passed", "verified_at": now(), "p0_status": "P0-COMPLETE", "p1_geometry_status": "P1-GEO-DIRECT", "p1_performance_overlay_status": "P1-OVERLAY", "numeric_cells_checked": numeric_cells, "numeric_table_rows": {"primary": primary_rows, "seeds": seed_rows, "geometry": geometry_rows}, "largest_primary_rounding_error_pp": max_rounding_error, "local_report_links_checked": len(links)+1, "figure_files_checked": len(figures), "parquet_rows_checked": counts, "source_manifest": source_info, "code_manifest": code_info, "report": file_info(report_path), "checklist": file_info(checklist_path), "required_deliverables": [file_info(OUT / relative) for relative in REQUIRED]}
    write_json(OUT / "qa/publication_verification.json", result)
    print(json.dumps({key: result[key] for key in ("status", "numeric_cells_checked", "figure_files_checked", "local_report_links_checked")}, indent=2))


if __name__ == "__main__":
    main()
