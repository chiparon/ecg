"""Counter-keyed training noise and losslessly factorized, shared test inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from phase1_ecg_robustness.src.covariance_matching import (
    empirical_covariance,
    match_covariance,
)
from phase1_ecg_robustness.src.lead_matrix import ELECTRODES, get_lead_matrix
from phase1_ecg_robustness.src.noise_generators import (
    _source_channels,
    match_lead_rms,
    nstdb_source_info,
    scale_to_snr,
)

from .common import (
    file_info,
    load_config,
    load_stage_data,
    require_preregistration,
    resolve_path,
    save_json,
    stage_paths,
    verify_training_complete,
)

KINDS = {"bandpass": 0, "bw": 1, "ma": 2, "em": 3}
INPUT_TYPES = ("clean", "independent_rms", "electrode")
SNR_TOLERANCE_DB = 1e-4
_LEAD_MATRIX = get_lead_matrix()
_LEAD_MATRIX.flags.writeable = False


def array_sha256(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def seed_words(
    cfg,
    stage,
    domain,
    noise_base,
    ecg_id,
    *,
    training_seed=0,
    epoch=0,
    strategy_code=0,
    kind="bandpass",
):
    """Disjoint domain/base keys; no model, SNR or test-combination dependence."""
    key = [
        int(noise_base),
        int(cfg["noise"]["domain_codes"][domain]),
        int(cfg["noise"]["stage_codes"][stage]),
        int(training_seed),
        int(epoch),
        int(ecg_id),
        int(strategy_code),
        KINDS[kind],
    ]
    return np.random.SeedSequence(key).generate_state(4, dtype=np.uint32)


def _seed_integer(words):
    return int.from_bytes(np.asarray(words, dtype="<u4").tobytes(), "little")


def _sources(cfg, words, kind, length, *, need_independent, need_covariance):
    children = np.random.SeedSequence(_seed_integer(words)).spawn(3)
    fs = float(cfg["sampling_rate"])
    band = tuple(cfg["noise"]["band"])
    nstdb = Path(cfg["_baseline_root"]) / cfg["noise"]["nstdb_dir"]
    electrode = _source_channels(9, length, fs, children[1], kind, nstdb, band)
    independent = (
        _source_channels(12, length, fs, children[0], kind, nstdb, band)
        if need_independent
        else None
    )
    fresh = (
        _source_channels(12, length, fs, children[2], kind, nstdb, band)
        if need_covariance
        else None
    )
    return electrode, independent, fresh


def _base_noises(x_mv, sources, electrodes, conditions):
    """Reuse phase-one physical/RMS/covariance transforms, without unused branches."""
    node_source, independent_source, fresh_source = sources
    active = np.asarray([name in electrodes for name in ELECTRODES])
    if not active.any():
        raise ValueError("An empty electrode target cannot attain finite global SNR")
    masked = np.array(node_source, copy=True)
    masked[~active] = 0.0
    electrode = scale_to_snr(_LEAD_MATRIX @ masked, x_mv, 0.0)
    result = {}
    if "electrode" in conditions:
        result["electrode"] = electrode.astype(np.float32)
    if "independent_rms" in conditions:
        if independent_source is None:
            raise ValueError("Independent source stream was not generated")
        result["independent_rms"] = match_lead_rms(
            independent_source, electrode
        ).astype(np.float32)
    if "covariance" in conditions:
        if fresh_source is None:
            raise ValueError("Covariance source stream was not generated")
        covariance, _ = match_covariance(empirical_covariance(electrode), fresh_source)
        result["covariance"] = covariance.astype(np.float32)
    return result


def _snr(x, noise):
    signal_power = float(np.mean(np.square(x, dtype=np.float64)))
    noise_power = float(np.mean(np.square(noise, dtype=np.float64)))
    if signal_power <= 0 or noise_power <= 0:
        raise ValueError("Signal and added noise must have positive finite power")
    return float(10.0 * np.log10(signal_power / noise_power))


def training_batch(cfg, stage, strategy, training_seed, epoch, x_mv, ecg_ids):
    x_mv = np.asarray(x_mv, dtype=np.float32)
    ecg_ids = np.asarray(ecg_ids, dtype=np.int64)
    if (
        x_mv.ndim != 3
        or x_mv.shape[1:] != (12, cfg["sequence_length"])
        or len(x_mv) != len(ecg_ids)
    ):
        raise ValueError("Training inputs/ECG IDs must be aligned (N,12,1000) arrays")
    if not np.isfinite(x_mv).all() or len(np.unique(ecg_ids)) != len(ecg_ids):
        raise ValueError("Training batch has nonfinite signals or duplicate ECG IDs")
    count = len(x_mv)
    audit = {
        "input_type": np.zeros(count, dtype=np.uint8),
        "combo_index": np.full(count, -1, dtype=np.int16),
        "snr_db": np.full(count, np.nan, dtype=np.float32),
        "actual_snr_db": np.full(count, np.nan, dtype=np.float64),
        "noise_seed_words": np.zeros((count, 4), dtype=np.uint32),
        "rms_match_relative_error": np.full(count, np.nan, dtype=np.float64),
        "noise_base": int(cfg["train"]["noise_base"]),
    }
    if strategy == "clean_only":
        return x_mv, audit
    if strategy not in cfg["strategies"]:
        raise ValueError(f"Unknown strategy: {strategy}")
    strategy_code = cfg["_strategy_codes"][strategy]
    probabilities = np.asarray(
        [cfg["strategies"][strategy][name] for name in INPUT_TYPES]
    )
    boundaries = np.cumsum(probabilities)
    output = None
    for row, ecg_id in enumerate(ecg_ids):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    int(cfg["train"]["augmentation_seed"]),
                    int(cfg["noise"]["stage_codes"][stage]),
                    int(training_seed),
                    int(epoch),
                    int(ecg_id),
                    int(strategy_code),
                ]
            )
        )
        input_type = min(
            int(np.searchsorted(boundaries, rng.random(), side="right")), 2
        )
        audit["input_type"][row] = input_type
        if input_type == 0:
            continue
        combo_index = int(
            rng.choice(len(cfg["_train_combos"]), p=cfg["train"]["combo_probabilities"])
        )
        nominal_snr = float(
            rng.choice(cfg["train"]["snrs"], p=cfg["train"]["snr_probabilities"])
        )
        words = seed_words(
            cfg,
            stage,
            "train",
            cfg["train"]["noise_base"],
            ecg_id,
            training_seed=training_seed,
            epoch=epoch,
            strategy_code=strategy_code,
        )
        condition = INPUT_TYPES[input_type]
        sources = _sources(
            cfg,
            words,
            "bandpass",
            x_mv.shape[2],
            need_independent=condition == "independent_rms",
            need_covariance=False,
        )
        needed = (
            ("electrode", "independent_rms")
            if condition == "independent_rms"
            else ("electrode",)
        )
        bases = _base_noises(
            x_mv[row], sources, cfg["_train_combos"][combo_index]["electrodes"], needed
        )
        multiplier = np.float32(10.0 ** (-nominal_snr / 20.0))
        noise = bases[condition] * multiplier
        actual = _snr(x_mv[row], noise)
        if abs(actual - nominal_snr) > SNR_TOLERANCE_DB:
            raise ValueError(
                "Actual training noise strength differs from requested SNR"
            )
        if output is None:
            output = np.array(x_mv, copy=True)
        output[row] += noise
        audit["combo_index"][row] = combo_index
        audit["snr_db"][row] = nominal_snr
        audit["actual_snr_db"][row] = actual
        audit["noise_seed_words"][row] = words
        if condition == "independent_rms":
            reference = bases["electrode"] * multiplier
            target = np.sqrt(np.mean(np.square(reference, dtype=np.float64), axis=1))
            measured = np.sqrt(np.mean(np.square(noise, dtype=np.float64), axis=1))
            active = target > 0
            if np.any(measured[~active] != 0):
                raise ValueError("RMS matching added noise to an inactive target lead")
            error = float(
                np.max(np.abs(measured[active] - target[active]) / target[active])
            )
            if error > 1e-6:
                raise ValueError(
                    "Actual float32 independent noise no longer matches target lead RMS"
                )
            audit["rms_match_relative_error"][row] = error
    return x_mv if output is None else output, audit


def _cohort(data):
    indices = np.ascontiguousarray(data["splits"]["test"], dtype=np.int64)
    selected = data["metadata"].iloc[indices]
    return {
        "indices": indices,
        "ids": np.ascontiguousarray(selected.ecg_id.to_numpy(), dtype=np.int64),
        "patient_ids": np.ascontiguousarray(
            selected.patient_id.to_numpy(), dtype=np.int64
        ),
        "y": np.ascontiguousarray(data["y"][indices], dtype=np.float32),
    }


def _combination_targets(cfg, stage):
    result = [{**value, "combo_set": "train"} for value in cfg["_train_combos"]]
    selected = cfg["stages"][stage]["heldout_combo_ids"]
    result += [
        {**value, "combo_set": "heldout"}
        for value in cfg["_heldout_combos"]
        if selected == "all" or value["combo_id"] in selected
    ]
    if cfg["test"]["include_all_electrodes"]:
        result.append(
            {"combo_id": "all", "electrodes": list(ELECTRODES), "combo_set": "all"}
        )
    return result


def _case_id(kind, base, combo, condition, snr):
    return f"{kind}_n{base}_c{combo}_{condition}_s{int(snr)}"


def _normalized_inputs(x_mv, noise0, snr, scale_mv):
    result = np.array(x_mv, dtype=np.float32, copy=True)
    if noise0 is not None:
        result += np.asarray(noise0, dtype=np.float32) * np.float32(
            10.0 ** (-float(snr) / 20.0)
        )
    result /= float(scale_mv)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite cached/reconstructed model input")
    return result


def load_case_inputs(cfg, data, case):
    x_mv = data["x"][data["splits"]["test"]]
    noise0 = None
    if case["kind"] != "clean":
        noise0 = np.load(
            resolve_path(cfg, case["base_noise_path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        if noise0.shape != x_mv.shape or noise0.dtype != np.float32:
            raise ValueError(
                "Cached base noise shape/dtype differs from the fixed cohort"
            )
    result = _normalized_inputs(x_mv, noise0, case["snr"], data["scale_mv"])
    if array_sha256(result) != case["input_sha256"]:
        raise ValueError(
            f"Reconstructed test input differs from frozen input: {case['case_id']}"
        )
    return result


def _diagnostics(x_mv, noise0, snr, words, cohort):
    noise = np.asarray(noise0) * np.float32(10.0 ** (-float(snr) / 20.0))
    signal_power = np.mean(np.square(x_mv, dtype=np.float64), axis=(1, 2))
    lead_signal_power = np.mean(np.square(x_mv, dtype=np.float64), axis=2)
    lead_noise_power = np.mean(np.square(noise, dtype=np.float64), axis=2)
    noise_power = lead_noise_power.mean(axis=1)
    if np.any(signal_power <= 0) or np.any(noise_power <= 0):
        raise ValueError("Cannot measure finite per-record global SNR")
    actual_snr = 10 * np.log10(signal_power / noise_power)
    if np.max(np.abs(actual_snr - snr)) > SNR_TOLERANCE_DB:
        raise ValueError("Cached float32 noise misses registered SNR tolerance")
    lead_snr = np.full_like(lead_noise_power, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log10(
            np.divide(
                lead_signal_power,
                lead_noise_power,
                out=np.ones_like(lead_noise_power),
                where=lead_noise_power > 0,
            ),
            out=lead_snr,
            where=lead_noise_power > 0,
        )
    lead_snr[lead_noise_power > 0] *= 10
    return {
        "ids": cohort["ids"],
        "patient_ids": cohort["patient_ids"],
        "indices": cohort["indices"],
        "actual_snr": actual_snr,
        "noise_rms": np.sqrt(lead_noise_power),
        "lead_snr": lead_snr,
        "noise_seed_words": words,
    }


def _attach_groups(cfg, stage, cases):
    groups = {}

    def add(case, group_id, *, combo_id=None, snr=None, primary=False):
        if group_id not in groups:
            groups[group_id] = {
                "group_id": group_id,
                "kind": case["kind"],
                "condition": case["condition"],
                "combo_set": case["combo_set"],
                "combo_id": case["combo_id"] if combo_id is None else combo_id,
                "snr": case["snr"] if snr is None else snr,
                "primary": primary,
                "case_ids": [],
            }
        groups[group_id]["case_ids"].append(case["case_id"])
        case["groups"].append(group_id)

    for case in cases:
        case["groups"] = []
        if case["kind"] == "clean":
            add(case, "clean")
            continue
        group = (
            f"{case['kind']}__{case['combo_set']}__{case['condition']}__s{case['snr']}"
        )
        add(case, group, combo_id="aggregate")
        if case["combo_set"] in ("train", "heldout"):
            group = f"{case['kind']}__combo_{case['combo_id']}__{case['condition']}__s{case['snr']}"
            add(case, group)
        if (
            case["kind"] == cfg["statistics"]["primary_kind"]
            and case["condition"] == cfg["statistics"]["primary_condition"]
            and case["combo_set"] == cfg["statistics"]["primary_combo_set"]
            and case["snr"] in cfg["statistics"]["primary_snrs"]
        ):
            add(
                case,
                "primary_joint",
                combo_id="aggregate",
                snr=-1,
                primary=stage == "full",
            )
    if not groups.get("primary_joint", {}).get("case_ids"):
        raise ValueError("Registered joint-unseen test group is empty")
    if stage == "full" and len(groups["primary_joint"]["case_ids"]) != 100:
        raise ValueError(
            "Full primary endpoint must contain exactly ten combos x two SNR x five realizations"
        )
    return list(groups.values())


def load_test_manifest(cfg, stage):
    path = stage_paths(cfg, stage)["test_inputs"] / "manifest.json"
    if not path.exists():
        raise ValueError(
            "Generate and freeze test inputs after all stage models finish training"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "completed"
        or value.get("config_sha256") != cfg["_config_sha256"]
        or value.get("matrix_sha256") != cfg["matrix_sha256"]
        or value.get("stage") != stage
    ):
        raise ValueError("Incomplete or mismatched fixed test cache")
    return value


def run(config_path, stage):
    cfg = load_config(config_path)
    prereg = require_preregistration(cfg)
    training = verify_training_complete(cfg, stage)
    data = load_stage_data(cfg, stage)
    if data["identity"] != prereg["datasets"][stage]:
        raise ValueError("Stage dataset differs from the frozen pre-training cohort")
    paths = stage_paths(cfg, stage)
    cache = paths["test_inputs"]
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != cfg["_config_sha256"]:
            raise ValueError(
                "Refusing to overwrite a different experiment's input cache"
            )
        if existing.get("status") == "completed":
            for info in existing["base_files"] + existing["diagnostic_files"]:
                if file_info(resolve_path(cfg, info["path"])) != info:
                    raise ValueError(f"Frozen cache file changed: {info['path']}")
            print(
                f"noise-cache: verified completed {stage} cache, {len(existing['cases'])} cases",
                flush=True,
            )
            return existing
    cohort = _cohort(data)
    x_mv = np.asarray(data["x"][cohort["indices"]], dtype=np.float32)
    np.savez_compressed(cache / "cohort.npz", **cohort)
    start = time.perf_counter()
    protocol = {
        "status": "building",
        "stage": stage,
        "config_sha256": cfg["_config_sha256"],
        "matrix_sha256": cfg["matrix_sha256"],
        "data_identity": data["identity"],
        "scale_mv": data["scale_mv"],
        "n_records": len(cohort["ids"]),
        "n_patients": int(len(np.unique(cohort["patient_ids"]))),
        "test_indices_sha256": array_sha256(cohort["indices"]),
        "test_ids_sha256": array_sha256(cohort["ids"]),
        "test_patient_ids_sha256": array_sha256(cohort["patient_ids"]),
        "test_labels_sha256": array_sha256(cohort["y"]),
        "cohort_file": file_info(cache / "cohort.npz"),
        "training_completed_before_generation": [
            {
                "strategy": row["strategy"],
                "model": row["model"],
                "seed": row["seed"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
            for row in training
        ],
        "representation": cfg["test"]["cache_representation"],
        "composition": "float32((X_mv + float32(noise0 * float32(10**(-snr/20)))) / scale_mv)",
        "seed_key": "SeedSequence([noise_base,test_domain,stage_code,0,0,ecg_id,0,kind_code]).generate_state(4,uint32), little-endian 128-bit integer; no strategy/model/training_seed/combo/SNR in test key",
        "base_files": [],
        "diagnostic_files": [],
        "cases": [],
        "groups": [],
        "nstdb_interpretation": "confounded exploratory sensitivity, not pure covariance evidence",
        "nstdb_sources": {},
    }
    save_json(manifest_path, protocol)
    clean = {
        "case_id": "clean",
        "kind": "clean",
        "condition": "clean",
        "combo_id": "none",
        "combo_set": "clean",
        "snr": 100,
        "noise_seed": 0,
        "base_noise_path": None,
        "diagnostics_path": None,
        "input_sha256": array_sha256(
            _normalized_inputs(x_mv, None, 100, data["scale_mv"])
        ),
    }
    protocol["cases"].append(clean)
    targets = _combination_targets(cfg, stage)
    settings = cfg["stages"][stage]
    kinds = ["bandpass"] + (cfg["test"]["nstdb_kinds"] if settings["nstdb"] else [])
    for kind in kinds:
        if kind != "bandpass":
            protocol["nstdb_sources"][kind] = nstdb_source_info(
                Path(cfg["_baseline_root"]) / cfg["noise"]["nstdb_dir"],
                kind,
                cfg["sampling_rate"],
            )
        kind_targets = (
            targets
            if kind == "bandpass"
            else [target for target in targets if target["combo_id"] == "all"]
        )
        snrs = (
            settings["test_snrs"] if kind == "bandpass" else cfg["test"]["nstdb_snrs"]
        )
        for noise_base in settings["test_noise_seeds"]:
            words = np.stack(
                [
                    seed_words(cfg, stage, "test", noise_base, ecg_id, kind=kind)
                    for ecg_id in cohort["ids"]
                ]
            )
            # Each base's 33 source channels are drawn once, then shared across combo masks.
            source_bank = np.empty(
                (len(x_mv), 33, cfg["sequence_length"]), dtype=np.float64
            )
            for record in range(len(x_mv)):
                node, independent, fresh = _sources(
                    cfg,
                    words[record],
                    kind,
                    cfg["sequence_length"],
                    need_independent=True,
                    need_covariance=True,
                )
                source_bank[record, :9] = node
                source_bank[record, 9:21] = independent
                source_bank[record, 21:] = fresh
            for target in kind_targets:
                buffers, filenames = {}, {}
                for condition in cfg["test"]["conditions"]:
                    filename = (
                        cache
                        / "noise_0db"
                        / f"{kind}_n{noise_base}_c{target['combo_id']}_{condition}.npy"
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    filenames[condition] = filename
                    buffers[condition] = np.lib.format.open_memmap(
                        filename, mode="w+", dtype=np.float32, shape=x_mv.shape
                    )
                covariance_errors, rms_errors = [], []
                for record in range(len(x_mv)):
                    sources = (
                        source_bank[record, :9],
                        source_bank[record, 9:21],
                        source_bank[record, 21:],
                    )
                    noises = _base_noises(
                        x_mv[record],
                        sources,
                        target["electrodes"],
                        cfg["test"]["conditions"],
                    )
                    for condition, noise in noises.items():
                        buffers[condition][record] = noise
                    reference_cov = empirical_covariance(noises["electrode"])
                    covariance_errors.append(
                        float(
                            np.linalg.norm(
                                empirical_covariance(noises["covariance"])
                                - reference_cov
                            )
                            / np.linalg.norm(reference_cov)
                        )
                    )
                    reference_rms = np.sqrt(
                        np.mean(
                            np.square(noises["electrode"], dtype=np.float64), axis=1
                        )
                    )
                    matched_rms = np.sqrt(
                        np.mean(
                            np.square(noises["independent_rms"], dtype=np.float64),
                            axis=1,
                        )
                    )
                    active = reference_rms > 0
                    if np.any(matched_rms[~active] != 0):
                        raise ValueError(
                            "Test RMS control corrupts inactive target leads"
                        )
                    rms_errors.append(
                        float(
                            np.max(
                                abs(matched_rms[active] - reference_rms[active])
                                / reference_rms[active]
                            )
                        )
                    )
                if max(covariance_errors) > 1e-5 or max(rms_errors) > 1e-6:
                    raise ValueError(
                        "Actual float32 test controls miss covariance/RMS matching gates"
                    )
                for condition, buffer in buffers.items():
                    buffer.flush()
                    base_info = file_info(filenames[condition])
                    protocol["base_files"].append(base_info)
                    for snr in snrs:
                        case_id = _case_id(
                            kind, noise_base, target["combo_id"], condition, snr
                        )
                        diagnostics = _diagnostics(x_mv, buffer, snr, words, cohort)
                        diagnostic_path = cache / "diagnostics" / f"{case_id}.npz"
                        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(diagnostic_path, **diagnostics)
                        info = file_info(diagnostic_path)
                        protocol["diagnostic_files"].append(info)
                        case = {
                            "case_id": case_id,
                            "kind": kind,
                            "condition": condition,
                            "combo_id": target["combo_id"],
                            "combo_set": target["combo_set"],
                            "electrodes": target["electrodes"],
                            "snr": int(snr),
                            "noise_seed": int(noise_base),
                            "base_noise_path": base_info["path"],
                            "diagnostics_path": info["path"],
                            "input_sha256": array_sha256(
                                _normalized_inputs(x_mv, buffer, snr, data["scale_mv"])
                            ),
                            "actual_snr_max_target_error_db": float(
                                np.max(abs(diagnostics["actual_snr"] - snr))
                            ),
                            "base_covariance_relative_frobenius_max": max(
                                covariance_errors
                            ),
                            "base_lead_rms_relative_error_max": max(rms_errors),
                            "snr_seen_in_training": int(snr) in cfg["train"]["snrs"],
                            "combo_seen_in_training": target["combo_set"] == "train",
                        }
                        protocol["cases"].append(case)
                del buffer, buffers
                print(
                    f"noise-cache: {stage} {kind} base={noise_base} combo={target['combo_id']} cases={len(protocol['cases'])} elapsed={time.perf_counter()-start:.1f}s",
                    flush=True,
                )
            del source_bank
            save_json(manifest_path, protocol)
    protocol["groups"] = _attach_groups(cfg, stage, protocol["cases"])
    expected = 1 + len(targets) * len(settings["test_noise_seeds"]) * len(
        settings["test_snrs"]
    ) * len(cfg["test"]["conditions"])
    if settings["nstdb"]:
        expected += (
            len(cfg["test"]["nstdb_kinds"])
            * len(settings["test_noise_seeds"])
            * len(cfg["test"]["nstdb_snrs"])
            * len(cfg["test"]["conditions"])
        )
    if (
        len(protocol["cases"]) != expected
        or len({case["case_id"] for case in protocol["cases"]}) != expected
    ):
        raise ValueError("Incomplete or duplicate registered test-input grid")
    protocol["elapsed_seconds"] = time.perf_counter() - start
    protocol["status"] = "completed"
    save_json(manifest_path, protocol)
    save_json(
        paths["logs"] / "noise_cache_protocol.json",
        {
            "status": "completed",
            "stage": stage,
            "config_sha256": cfg["_config_sha256"],
            "matrix_sha256": cfg["matrix_sha256"],
            "case_count": len(protocol["cases"]),
            "group_count": len(protocol["groups"]),
            "manifest": file_info(manifest_path),
            "elapsed_seconds": protocol["elapsed_seconds"],
        },
    )
    print(
        f"noise-cache: completed {stage}: {len(protocol['cases'])} cases, {len(protocol['groups'])} groups",
        flush=True,
    )
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="phase2/configs/phase2_main.yaml")
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
