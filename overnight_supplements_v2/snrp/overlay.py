"""Original probabilities only: validated per-source AUROCs and record linkage."""
from __future__ import annotations
from overnight_supplements_v2.shared.common import resolve, read_json, write_json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .run import DEST, CLASSES, CONDITIONS, raw_hash, digest, require, summarize

SIGNED_RUN_KEYS = ("stage", "config_sha256", "source_fingerprints", "manifest_fingerprint", "freeze_fingerprint", "checkpoint_fingerprints", "software", "fp32_settings", "device_name", "model", "shard_index", "shard_count")
LEGACY_RUN_KEYS = ("stage", "config_sha256", "source_fingerprints", "manifest_fingerprint", "freeze_fingerprint", "checkpoint_manifest_fingerprint", "checkpoint_fingerprints", "software", "fp32_settings", "device_name", "verification_only")


def auc(y, probabilities):
    """Exact empirical probability-of-superiority AUROC with half credit for ties."""
    values = []
    for column in range(y.shape[1]):
        labels, scores = y[:, column], probabilities[:, column]
        order = np.argsort(scores, kind="stable")
        scores, labels = scores[order], labels[order]
        starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1])+1]
        positive = np.add.reduceat(labels.astype(np.float64), starts)
        sizes = np.diff(np.r_[starts, len(scores)])
        negative = sizes-positive
        npos, nneg = positive.sum(), negative.sum()
        values.append(float(np.sum(positive*(np.cumsum(negative)-.5*negative))/(npos*nneg)) if npos > 0 and nneg > 0 else np.nan)
    return np.asarray(values)


def run_overlay(analysis, infos, conflicts, geometry, cohort):
    index = pd.read_csv(resolve(analysis["paths"]["prediction_index"]), keep_default_na=False)
    freeze = read_json(analysis["paths"]["freeze"])
    manifest = read_json(analysis["paths"]["manifest"])
    workers = read_json(DEST / "worker_provenance.json")["reports"]
    cases = {x["case_id"]: x for x in manifest["cases"]}
    cases["clean"] = dict(case_id="clean", condition="clean", mode="", snr=None, noise_seed=None,
                          input_sha256=manifest["clean_input_sha256"], noise_sha256="")
    checkpoints = {(x["model"], int(x["seed"])): x for x in freeze["checkpoint_entries"]}
    exceptions, metrics, audits = [], [], []
    writer = None
    pending_tables = []
    expected = {(model, seed, cid) for model in ("resnet", "tcn") for seed in (17, 29, 43) for cid in cases}
    actual = {(r.model, int(r.seed), r.case_id) for r in index.itertuples()}
    if index.duplicated(["model", "seed", "case_id"]).any() or index.prediction_sha256.duplicated().any() or index.prediction_path.duplicated().any() or actual != expected:
        exceptions.append(dict(source_prediction_id="", prediction_path="", case_id="", anomaly_code="PREDICTION_GRID_MISMATCH", detail=str(dict(missing=sorted(expected-actual), extra=sorted(actual-expected)))))
    source_conflicts = {x["path"]: x for x in conflicts}
    geometry_cases = {key: value.sort_values("record_index") for key, value in geometry.groupby("case_id", sort=False)}
    reports_by_fp = {}
    for item in workers:
        report = item["report"]
        signed = item["path"].startswith("signed_control_mechanism/")
        keys = SIGNED_RUN_KEYS if signed else LEGACY_RUN_KEYS
        fp = report["run_fingerprint"]
        reports_by_fp.setdefault(fp, []).append(item)
        try:
            require(report["status"] == "completed" and report["stage"] == "full", "Incomplete historical inference worker")
            require(fp == digest({key: report[key] for key in keys}), "Worker run fingerprint mismatch")
            for info in report["source_fingerprints"]:
                require(info["path"] in infos and infos[info["path"]]["sha256"] == info["sha256"], "Historical worker source missing/changed: " + info["path"])
            for key in ("manifest_fingerprint", "freeze_fingerprint", "checkpoint_manifest_fingerprint"):
                if key in report:
                    info = report[key]
                    require(infos[info["path"]]["sha256"] == info["sha256"], "Historical worker provenance changed")
        except (ValueError, KeyError) as error:
            exceptions.append(dict(source_prediction_id="", prediction_path=item["path"], case_id="", anomaly_code="WORKER_FINGERPRINT_MISMATCH", detail=str(error)))
    try:
        for row in index.to_dict("records"):
            sid = row["prediction_sha256"]
            path = row["prediction_path"]
            cid = row["case_id"]
            try:
                require(path in infos and path not in source_conflicts, "Missing or changed immutable prediction")
                require(infos[path]["sha256"] == sid, "Prediction file hash mismatch")
                case = cases[cid]
                cp = checkpoints[(row["model"], int(row["seed"]))]
                require(row["condition"] == case["condition"] and row["mode"] == case["mode"], "Index condition/mode mismatch")
                if cid != "clean":
                    require(int(row["snr"]) == case["snr"] and int(row["noise_seed"]) == case["noise_seed"], "Index SNR/noise mismatch")
                require(row["input_sha256"] == case["input_sha256"] and row["checkpoint_sha256"] == cp["sha256"], "Frozen input/checkpoint mismatch")
                owners = [item for item in reports_by_fp[row["run_fingerprint"]]
                          if item["report"]["model"] == row["model"]
                          and row["prediction_case_id"] in item["report"]["owned_case_ids"]]
                require(len(owners) == 1, "Prediction must resolve to exactly one worker owner")
                owner = owners[0]
                report = owner["report"]
                require(row["config_sha256"] == report["config_sha256"], "Index config fingerprint mismatch")
                for field, key in (("manifest_sha256", "manifest_fingerprint"), ("freeze_sha256", "freeze_fingerprint")):
                    require(row[field] == report[key]["sha256"], "Index provenance mismatch: " + field)
                require(row["source_sha256"] == digest(report["source_fingerprints"]), "Index source fingerprint mismatch")
                worker_cp = [x for x in report["checkpoint_fingerprints"] if x["model"] == row["model"] and int(x["seed"]) == int(row["seed"])]
                require(len(worker_cp) == 1 and worker_cp[0]["sha256"] == cp["sha256"], "Worker checkpoint fingerprint mismatch")
                with np.load(resolve(path), allow_pickle=False) as data:
                    for field in ("model", "seed", "config_sha256", "checkpoint_sha256", "input_sha256", "run_fingerprint", "cohort_sha256", "manifest_sha256"):
                        expected_value = int(row[field]) if field == "seed" else row[field]
                        require(data[field].item() == expected_value, "Prediction metadata mismatch: " + field)
                    for field in ("freeze_sha256", "source_sha256", "thresholds_sha256"):
                        if field in data:
                            require(data[field].item() == row[field], "Prediction optional fingerprint mismatch: " + field)
                    require(data["case_id"].item() == row["prediction_case_id"], "Prediction case ID mismatch")
                    require(not bool(data["verification_only"].item()), "Verification-only prediction not legal overlay")
                    require(data["noise_sha256"].item() == case["noise_sha256"], "Prediction noise base hash mismatch")
                    require(row["cohort_sha256"] == freeze["cohort"]["sha256"], "Prediction cohort fingerprint mismatch")
                    for key in ("ids", "patient_ids", "indices", "y"):
                        require(np.array_equal(data[key], cohort[key]), "Prediction identity or labels mismatch: " + key)
                        require(raw_hash(data[key]) == row[key+"_sha256"], "Prediction identity array hash mismatch: " + key)
                    thresholds = data["thresholds"]
                    require(np.array_equal(thresholds, np.asarray(cp["thresholds"], dtype=np.float64)) and raw_hash(thresholds) == row["thresholds_sha256"], "Frozen threshold fingerprint mismatch (not reselected)")
                    y, p = data["y"], data["p"]
                    require(y.shape == p.shape == (2158, 5), "Prediction shape mismatch")
                    require(np.isin(y, [0, 1]).all(), "Nonbinary labels")
                    require(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), "Invalid original probabilities")
                    require(raw_hash(p) == row["p_sha256"] == data["p_sha256"].item(), "Prediction probability hash mismatch")
                    class_aucs = auc(y, p)
                    macro = float(np.mean(class_aucs))
                    require(abs(macro-float(row["macro_auroc"])) <= 1e-12, "Original unrounded AUROC does not reproduce")
                    metric = dict(model=row["model"], train_seed=int(row["seed"]), case_id=cid,
                        condition=case["condition"], mode=case["mode"], snr_db=case["snr"], noise_seed=case["noise_seed"],
                        source_prediction_id=sid, prediction_path=path, prediction_sha256=sid,
                        checkpoint_sha256=row["checkpoint_sha256"], input_sha256=row["input_sha256"],
                        macro_auroc=macro, original_macro_auroc=float(row["macro_auroc"]), unit="fraction", n_records=len(y),
                        n_patients=len(np.unique(cohort["patient_ids"])), aggregation_weight=1, source_status="original_probabilities_verified")
                    for j, name in enumerate(CLASSES):
                        metric["auroc_"+name] = class_aucs[j]
                        metric["n_positive_"+name] = int(y[:, j].sum())
                        metric["n_negative_"+name] = int((1-y[:, j]).sum())
                    metrics.append(metric)
                    if cid != "clean":
                        geo = geometry_cases[cid]
                        require(np.array_equal(geo.record_id.to_numpy(), data["ids"]) and np.array_equal(geo.patient_id.to_numpy(), data["patient_ids"]), "Geometry prediction linkage mismatch")
                        clean_hashes, noisy_hashes = geo.clean_input_sha256.to_numpy(), geo.noisy_input_sha256.to_numpy()
                        links = pd.DataFrame(dict(record_id=data["ids"], patient_id=data["patient_ids"].astype(np.int64), record_index=np.arange(len(y)),
                            dataset_index=data["indices"], prediction_row_index=np.arange(len(y)), model=row["model"], train_seed=int(row["seed"]),
                            case_id=cid, condition=case["condition"], mode=case["mode"], snr_db=int(case["snr"]),
                            noise_seed=int(case["noise_seed"]), source_prediction_id=sid, prediction_path=path,
                            clean_input_sha256=clean_hashes, noisy_input_sha256=noisy_hashes, checkpoint_sha256=row["checkpoint_sha256"],
                            **{"label_"+name: y[:, j].astype(np.int8) for j, name in enumerate(CLASSES)}))
                        table = pa.Table.from_pandas(links, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(DEST / "overlay_alignment.parquet", table.schema, compression="zstd", compression_level=6, dictionary_pagesize_limit=16*1024*1024)
                        # Share repeated record identities across bounded, zero-copy Arrow batches.
                        pending_tables.append(table)
                        if len(pending_tables) == 64:
                            writer.write_table(pa.concat_tables(pending_tables), row_group_size=64*len(cohort["patient_ids"]))
                            pending_tables.clear()
                    audits.append(dict(source_prediction_id=sid, prediction_path=path, model=row["model"], train_seed=int(row["seed"]), case_id=cid,
                        prediction_case_id=row["prediction_case_id"], worker_report=owner["path"], run_fingerprint=row["run_fingerprint"],
                        source_sha256=row["source_sha256"], n_aligned_records=len(y) if cid != "clean" else 0, input_verified=True, checkpoint_verified=True,
                        absolute_macro_auroc_error=abs(macro-float(row["macro_auroc"]))))
            except (ValueError, KeyError, OSError) as error:
                exceptions.append(dict(source_prediction_id=sid, prediction_path=path, case_id=cid, anomaly_code="PREDICTION_ALIGNMENT_FAILURE", detail=str(error)))
    finally:
        if writer is not None:
            if pending_tables:
                writer.write_table(pa.concat_tables(pending_tables), row_group_size=64*len(cohort["patient_ids"]))
            writer.close()
    pd.DataFrame(exceptions, columns=["source_prediction_id", "prediction_path", "case_id", "anomaly_code", "detail"]).to_csv(DEST / "overlay_anomalies.csv", index=False)
    pd.DataFrame(audits).to_csv(DEST / "overlay_source_audit.csv", index=False)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(DEST / "auroc_overlay.csv", index=False)
    complete = not exceptions and len(metrics) == 762
    if complete:
        # First mean six fixed noises within each checkpoint; three checkpoints then equal mean.
        per_seed = metrics_df[metrics_df.condition != "clean"].groupby(["model", "train_seed", "condition", "mode", "snr_db"], dropna=False).agg(macro_auroc=("macro_auroc", "mean"), n_sources=("source_prediction_id", "nunique")).reset_index()
        require((per_seed.n_sources == 6).all(), "Incomplete fixed-noise curve cells")
        per_seed.to_csv(DEST / "auroc_curves_by_seed.csv", index=False)
        curves = per_seed.groupby(["model", "condition", "mode", "snr_db"], dropna=False).agg(macro_auroc=("macro_auroc", "mean"), seed_sd=("macro_auroc", "std"), n_train_seeds=("train_seed", "nunique")).reset_index()
        curves["n_noise_bases"] = 6
        curves["unit"] = "fraction"
        curves.to_csv(DEST / "auroc_curves.csv", index=False)
        # Geometry has one record/case, never six checkpoint copies, and E/I are not fivefold replicated.
        label_lookup = pd.DataFrame(dict(record_id=cohort["ids"], **{"label_"+name: cohort["y"][:, j].astype(np.int8) for j, name in enumerate(CLASSES)}))
        exposures = geometry.merge(label_lookup, on="record_id", validate="many_to_one")
        distributions = []
        for name in CLASSES:
            for label in (0, 1):
                subset = exposures[exposures["label_"+name] == label]
                summary = summarize(subset, ["condition", "mode", "snr_db", "noise_seed"], ["snr_p_db", "delta_p_db", "q_noise"])
                summary["class_name"], summary["true_label"] = name, label
                summary["stratum"] = "positive" if label else "negative"
                distributions.append(summary)
        pd.concat(distributions, ignore_index=True).to_csv(DEST / "class_exposure_summary.csv", index=False)
    result = dict(performance_overlay_status="P1-OVERLAY" if complete else "P1-OVERLAY-UNAVAILABLE",
        overlay_verified_sources=len(metrics), overlay_verified_noisy_sources=sum(x["condition"] != "clean" for x in metrics),
        overlay_verified_clean_sources=sum(x["condition"] == "clean" for x in metrics), overlay_alignment_rows=sum(x["condition"] != "clean" for x in metrics)*2158,
        overlay_anomaly_count=len(exceptions), overlay_source_weight_policy="Each immutable source exactly once; E/I never fivefold replicated; no probability averaging",
        performance_interpretation="Nominal-SNR aligned parallel display only; no equal-exposure or causal claim; no new CI")
    write_json(DEST / "overlay_audit.json", result)
    return result
