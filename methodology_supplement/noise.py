"""Canonical CPU-only noise cache, with audits of the float32 model inputs.

The manifest is the completion marker. A failed generation never publishes a
complete manifest; a cache hit requires all input, source and output fingerprints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np

from phase1_ecg_robustness.src.audit_noise import record_seed
from phase1_ecg_robustness.src.lead_matrix import ELECTRODES, LEADS, get_lead_matrix
from phase1_ecg_robustness.src.noise_generators import (
    _source_channels,
    match_lead_rms,
    scale_to_snr,
)
from .common import (
    DEFAULT_CONFIG,
    WORKSPACE,
    array_sha256,
    file_info,
    load_config,
    load_reference,
    read_json,
    require_freeze,
    resolve_path,
    save_json,
    sha256,
    stage_paths,
    write_csv,
)

CONDITIONS = ("electrode", "independent_rms")
POWER_FIELDS = (
    "power_fraction_0_5", "power_fraction_5_15",
    "power_fraction_15_40", "power_fraction_40_50",
)
MEASURE_FIELDS = (
    "achieved_snr_0", "max_snr_abs_error_db",
    "max_scaled_noise_snr_abs_error_db", "max_lead_rms_relative_error",
    "max_scaled_lead_rms_relative_error", "effective_cross_lead_correlation",
    "active_lead_count", *POWER_FIELDS, "crop_mean_removed_rms",
    "crop_to_long_source_power_ratio", "crop_demeaning_retained_power_fraction",
    "source_fft_df_hz", "source_lowest_retained_hz",
)
DIAGNOSTIC_FIELDS = (
    "source_id", "matrix_id", "family", "band_id", "noise_seed", "condition",
    "ecg_id", *MEASURE_FIELDS,
)


def _portable(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()


def _nullable(array: np.ndarray) -> list:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        return [float(v) if np.isfinite(v) else None for v in values]
    return [_nullable(row) for row in values]


def matrix_diagnostics(matrix: np.ndarray) -> dict:
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    covariance = matrix @ matrix.T
    gram = matrix.T @ matrix
    norms = np.sqrt(np.diag(gram))
    denominator = norms[:, None] * norms[None, :]
    cosine = np.full_like(gram, np.nan)
    np.divide(gram, denominator, out=cosine, where=denominator > 0)
    return {
        "rank": rank,
        "right_nullity": int(matrix.shape[1] - rank),
        "left_nullity": int(matrix.shape[0] - rank),
        "row_nonzero_counts": np.count_nonzero(matrix, axis=1).tolist(),
        "column_nonzero_counts": np.count_nonzero(matrix, axis=0).tolist(),
        "row_squared_norms": np.sum(matrix ** 2, axis=1).tolist(),
        "column_squared_norms": np.sum(matrix ** 2, axis=0).tolist(),
        "singular_values": singular.tolist(),
        "mean_abs_offdiagonal_covariance": float(
            np.abs(covariance[~np.eye(matrix.shape[0], dtype=bool)]).mean()
        ),
        "row_covariance": covariance.tolist(),
        "column_gram": gram.tolist(),
        "column_cosine": _nullable(cosine),
    }


def build_matrices(cfg: dict) -> list[dict]:
    standard = get_lead_matrix()
    limb = standard.copy()
    limb[:, 3:] = 0
    precordial = standard.copy()
    precordial[:, :3] = 0
    values = [("standard", "standard", None, standard),
              ("limb_only", "limb_only", None, limb),
              ("precordial_only", "precordial_only", None, precordial)]
    rng = np.random.Generator(np.random.PCG64(cfg["matrix"]["random_seed"]))
    groups = [np.asarray(group, dtype=np.int64) for group in cfg["matrix"]["row_groups"]]
    if sorted(np.concatenate(groups).tolist()) != list(range(12)):
        raise ValueError("Random matrix row groups must partition all twelve leads")
    for group in groups:
        if not np.allclose(np.sum(standard[group] ** 2, axis=1),
                           np.sum(standard[group[0]] ** 2), rtol=0, atol=1e-14):
            raise ValueError("Randomization requires equal-row-energy groups")
        if len(set(np.count_nonzero(standard[group], axis=1).tolist())) != 1:
            raise ValueError("Randomization requires equal-row-sparsity groups")
    seen = {array_sha256(standard)}
    covariance = standard @ standard.T
    while len(values) < 3 + int(cfg["matrix"]["random_replicates"]):
        candidate = standard.copy()
        for group in groups:
            candidate[group] = standard[rng.permutation(group)]
        active = candidate != 0
        candidate[active] *= rng.choice(np.array([-1.0, 1.0]), size=int(active.sum()))
        digest = array_sha256(candidate)
        if digest in seen or np.array_equal(candidate @ candidate.T, covariance):
            continue
        for axis in (0, 1):
            if not np.array_equal(np.count_nonzero(candidate, axis=axis),
                                  np.count_nonzero(standard, axis=axis)):
                raise RuntimeError("Randomization changed nonzero counts")
            if not np.allclose(np.sum(candidate ** 2, axis=axis),
                               np.sum(standard ** 2, axis=axis), rtol=0, atol=1e-14):
                raise RuntimeError("Randomization changed coefficient power")
        replicate = len(values) - 3
        seen.add(digest)
        values.append((f"random_{replicate:02d}", "randomized", replicate, candidate))
    return [
        {"id": identifier, "family": family, "replicate": replicate,
         "values": matrix.tolist(), "matrix_sha256": array_sha256(matrix),
         "diagnostics": matrix_diagnostics(matrix)}
        for identifier, family, replicate, matrix in values
    ]


def _sources(cfg: dict, matrices: list[dict]) -> list[dict]:
    result = []
    for matrix in matrices:
        result.append({
            "source_id": matrix["id"], "matrix_id": matrix["id"],
            "matrix_family": matrix["family"], "matrix_sha256": matrix["matrix_sha256"],
            "band_id": "legacy_0p5_40", "band": tuple(cfg["legacy_generation"]["band"]),
            "long_source": False,
            "snrs": list(cfg["snrs"] if matrix["id"] == "standard" else cfg["ablation_snrs"]),
        })
    for band in cfg["bands"]:
        result.append({
            "source_id": band["id"], "matrix_id": "standard", "matrix_family": "standard",
            "matrix_sha256": matrices[0]["matrix_sha256"], "band_id": band["id"],
            "band": (float(band["low"]), float(band["high"])), "long_source": True,
            "snrs": list(cfg["ablation_snrs"]),
        })
    return result


def _case(source: dict, seed: int, condition: str, snr: int, cfg: dict) -> dict:
    if source["long_source"]:
        tags = ["band"]
    elif source["source_id"] == "standard":
        tags = ["snr"] + (["matrix"] if snr in cfg["ablation_snrs"] else [])
    else:
        tags = ["matrix"]
    return {
        "case_id": f'{source["source_id"]}__n{seed}__{condition}__snr{snr}',
        "kind": "bandpass", "condition": condition,
        "source_id": source["source_id"], "matrix_id": source["matrix_id"],
        "matrix_family": source["matrix_family"], "band_id": source["band_id"],
        "snr": int(snr), "noise_seed": int(seed), "analysis_tags": tags,
        "matrix_sha256": source["matrix_sha256"],
    }


def _white_spectra(seed: np.random.SeedSequence, channels: int, length: int) -> np.ndarray:
    # One forward FFT per white channel, shared by every long-source band.
    spectrum = np.empty((channels, length // 2 + 1), dtype=np.complex128)
    for channel, child in enumerate(seed.spawn(channels)):
        rng = np.random.Generator(np.random.PCG64(child))
        spectrum[channel] = np.fft.rfft(rng.standard_normal(length))
    spectrum[:, 0] = 0
    return spectrum


def _long_band(spectra: np.ndarray, keep: np.ndarray, length: int) -> np.ndarray:
    filtered = spectra.copy()
    filtered[:, ~keep] = 0
    return np.fft.irfft(filtered, n=length, axis=1)


def _crop_diagnostics(raw: np.ndarray, long_power: float) -> dict:
    power = float(np.mean(raw ** 2))
    means = raw.mean(axis=1, keepdims=True)
    centered_power = float(np.mean((raw - means) ** 2))
    return {
        "crop_mean_removed_rms": float(np.sqrt(np.mean(means ** 2))),
        "crop_to_long_source_power_ratio": power / long_power if long_power > 0 else None,
        "crop_demeaning_retained_power_fraction": centered_power / power if power > 0 else None,
    }


def _record_sources(cfg: dict, sources: list[dict], matrices: dict, seed: int, ecg_id: int):
    fs = float(cfg["sampling_rate"])
    length = int(cfg["sequence_length"])
    key = record_seed(seed, ecg_id, "bandpass")
    streams = np.random.SeedSequence(key).spawn(3)
    band = tuple(cfg["legacy_generation"]["band"])
    independent = _source_channels(12, length, fs, streams[0], "bandpass", None, band)
    electrode = _source_channels(9, length, fs, streams[1], "bandpass", None, band)
    for source in sources:
        if source["long_source"]:
            continue
        physical = matrices[source["matrix_id"]] @ electrode
        yield source, physical, independent, {
            "electrode": _crop_diagnostics(physical, float(np.mean(physical ** 2))),
            "independent_rms": _crop_diagnostics(independent, float(np.mean(independent ** 2))),
        }, fs / length, float(np.fft.rfftfreq(length, d=1 / fs)[
            (np.fft.rfftfreq(length, d=1 / fs) >= band[0]) &
            (np.fft.rfftfreq(length, d=1 / fs) <= band[1]) &
            (np.fft.rfftfreq(length, d=1 / fs) > 0)
        ][0])
    generation = cfg["band_generation"]
    full_length = int(generation["source_length"])
    start = int(generation["crop_start"])
    stop = start + int(generation["crop_length"])
    if stop > full_length or start < 0 or stop - start != length:
        raise ValueError("Long-source crop is inconsistent with prepared ECG duration")
    streams = np.random.SeedSequence(key).spawn(3)
    independent_fft = _white_spectra(streams[0], 12, full_length)
    electrode_fft = _white_spectra(streams[1], 9, full_length)
    frequencies = np.fft.rfftfreq(full_length, d=1 / fs)
    for source in sources:
        if not source["long_source"]:
            continue
        low, high = source["band"]
        keep = (frequencies >= low) & (frequencies <= high) & (frequencies > 0)
        if not keep.any():
            raise ValueError("Band has no retained non-DC Fourier bins")
        independent_full = _long_band(independent_fft, keep, full_length)
        electrode_full = _long_band(electrode_fft, keep, full_length)
        # Mapping the long waveform is needed for measured crop attenuation;
        # all these arrays are one record only, never a cohort-sized allocation.
        physical_full = matrices[source["matrix_id"]] @ electrode_full
        physical_crop = physical_full[:, start:stop]
        independent_crop = independent_full[:, start:stop]
        metadata = {
            "electrode": _crop_diagnostics(physical_crop, float(np.mean(physical_full ** 2))),
            "independent_rms": _crop_diagnostics(independent_crop, float(np.mean(independent_full ** 2))),
        }
        yield source, physical_crop, independent_crop, metadata, fs / full_length, float(frequencies[keep][0])


def _rms_error(left: np.ndarray, right: np.ndarray) -> float:
    left_rms = np.sqrt(np.mean(np.asarray(left, dtype=np.float64) ** 2, axis=1))
    right_rms = np.sqrt(np.mean(np.asarray(right, dtype=np.float64) ** 2, axis=1))
    active = right_rms > 0
    if np.any(left_rms[~active] != 0):
        raise RuntimeError("Independent comparator activates a reference-inactive lead")
    return float(np.max(np.abs(left_rms[active] - right_rms[active]) / right_rms[active]))


def _snr(signal_power: float, noise: np.ndarray) -> float:
    noise_power = float(np.mean(np.asarray(noise, dtype=np.float64) ** 2))
    if not np.isfinite(noise_power) or noise_power <= 0:
        raise RuntimeError("Actual injected noise has zero or nonfinite power")
    return float(10 * np.log10(signal_power / noise_power))


def _final_diagnostics(noise: np.ndarray, fs: float, bins: list) -> dict:
    centered = noise - noise.mean(axis=1, keepdims=True)
    covariance = centered @ centered.T / centered.shape[1]
    variance = np.diag(covariance)
    active = variance > 0
    count = int(active.sum())
    correlation = None
    if count > 1:
        sub = covariance[np.ix_(active, active)]
        denominator = np.sqrt(variance[active, None] * variance[None, active])
        correlation = float(np.abs((sub / denominator)[~np.eye(count, dtype=bool)]).mean())
    # Rectangular-window, full-record, one-sided periodogram. Common scaling
    # cancels in fractions; double non-DC/non-Nyquist bins for Parseval power.
    spectrum = np.fft.rfft(noise, axis=1)
    power = np.sum(np.abs(spectrum) ** 2, axis=0)
    if noise.shape[1] % 2 == 0:
        power[1:-1] *= 2
    else:
        power[1:] *= 2
    power[0] = 0
    frequency = np.fft.rfftfreq(noise.shape[1], d=1 / fs)
    total = float(power.sum())
    result = {"effective_cross_lead_correlation": correlation, "active_lead_count": count}
    for i, ((low, high), name) in enumerate(zip(bins, POWER_FIELDS)):
        mask = (frequency >= low) & (frequency > 0)
        mask &= frequency <= high if i == len(bins) - 1 else frequency < high
        result[name] = float(power[mask].sum() / total) if total > 0 else None
    return result


def _update_summary(aggregate: dict, key: tuple, row: dict) -> None:
    summary = aggregate.setdefault(key, {"n_records": 0, "fields": {}})
    summary["n_records"] += 1
    for field in MEASURE_FIELDS:
        value = row[field]
        if value is None:
            continue
        value = float(value)
        count, total, minimum, maximum = summary["fields"].get(field, (0, 0.0, value, value))
        summary["fields"][field] = (count + 1, total + value, min(minimum, value), max(maximum, value))


def _summary_rows(aggregate: dict) -> list[dict]:
    rows = []
    for key, summary in aggregate.items():
        source, seed, condition, family, band = key
        row = {"scope": "overall" if source == "__overall__" else "source_seed_condition",
               "source_id": source, "noise_seed": seed, "condition": condition,
               "family": family, "band_id": band, "n_records": summary["n_records"]}
        for field in MEASURE_FIELDS:
            count, total, minimum, maximum = summary["fields"].get(field, (0, 0.0, None, None))
            row[field] = total / count if count else None
            row[field + "_min"] = minimum
            row[field + "_max"] = maximum
            row[field + "_n_defined"] = count
        rows.append(row)
    return rows


def _fingerprints(cfg: dict, paths: dict) -> dict:
    source_paths = [Path(__file__), Path(__file__).with_name("common.py"),
                    Path(__file__).with_name("implementation_contract.json")]
    for name in ("audit_noise.py", "noise_generators.py", "lead_matrix.py"):
        source_paths.append(WORKSPACE / "phase1_ecg_robustness" / "src" / name)
    return {
        "config_sha256": cfg["_config_sha256"],
        "cohort": file_info(paths["inputs"] / "cohort.npz"),
        "clean": file_info(paths["inputs"] / "clean.npy"),
        "freeze": file_info(paths["logs"] / "freeze.json"),
        "sources": [file_info(path) for path in source_paths],
    }


def _complete_cache(manifest_path: Path, fingerprints: dict, expected_cases: int,
                    expected_bases: int, stage: str) -> dict | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = read_json(manifest_path)
        if (manifest.get("status") != "completed" or manifest.get("stage") != stage
                or manifest.get("provenance") != fingerprints
                or manifest.get("n_cases") != expected_cases
                or len(manifest.get("cases", [])) != expected_cases
                or len(manifest.get("base_files", [])) != expected_bases):
            return None
        infos = [manifest["matrices"], *manifest["diagnostics"].values(), *manifest["base_files"]]
        for info in infos:
            path = resolve_path(info["path"])
            if not path.is_file() or path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
                return None
        if len({case["case_id"] for case in manifest["cases"]}) != expected_cases:
            return None
        base_by_path = {info["path"]: info for info in manifest["base_files"]}
        if len(base_by_path) != expected_bases or manifest["cases"][0]["kind"] != "clean":
            return None
        for case in manifest["cases"]:
            if case["kind"] != "clean" and base_by_path[case["noise_path"]]["sha256"] != case["noise_sha256"]:
                return None
            if len(case["input_sha256"]) != 64:
                return None
        return manifest
    except (OSError, KeyError, TypeError, ValueError):
        return None


def run(config: str | Path | None = None, stage: str = "full") -> dict:
    cfg = load_config(config)
    require_freeze(cfg, stage)
    paths = stage_paths(cfg, stage)
    for category in ("inputs", "tables"):
        paths[category].mkdir(parents=True, exist_ok=True)
    fingerprints = _fingerprints(cfg, paths)
    matrices = build_matrices(cfg)
    sources = _sources(cfg, matrices)
    seeds = [int(seed) for seed in cfg["phase1_noise_seeds"]]
    expected_bases = len(sources) * len(seeds) * len(CONDITIONS)
    expected_cases = 1 + sum(len(source["snrs"]) for source in sources) * len(seeds) * len(CONDITIONS)
    manifest_path = paths["inputs"] / "manifest.json"
    cached = _complete_cache(manifest_path, fingerprints, expected_cases, expected_bases, stage)
    if cached is not None:
        print(f'Noise cache verified: {cached["n_cases"]} cases, {expected_bases} base files', flush=True)
        return cached
    reference = load_reference(cfg, stage)
    ids = np.asarray(reference["ids"])
    clean = np.load(paths["inputs"] / "clean.npy", mmap_mode="r", allow_pickle=False)
    expected_shape = (len(ids), len(LEADS), int(cfg["sequence_length"]))
    if clean.dtype != np.float32 or clean.shape != expected_shape or not len(ids):
        raise ValueError("Prepared clean.npy must be nonempty float32[N,12,sequence_length]")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Prepared ECG ids must be unique")
    clean_hash = hashlib.sha256()
    for signal in clean:
        if not np.isfinite(signal).all() or not np.any(signal):
            raise ValueError("Clean cohort contains zero-power or nonfinite ECGs")
        clean_hash.update(memoryview(np.ascontiguousarray(signal)).cast("B"))
    cases = [{
        "case_id": "clean", "kind": "clean", "condition": "clean",
        "source_id": "clean", "matrix_id": "clean", "matrix_family": "clean",
        "band_id": "clean", "snr": 100, "noise_seed": 0, "analysis_tags": ["clean"],
        "noise_path": None, "noise_sha256": None, "matrix_sha256": None,
        "input_sha256": clean_hash.hexdigest(),
    }]
    save_json(manifest_path, {"status": "generating", "stage": stage,
                              "config_sha256": cfg["_config_sha256"], "provenance": fingerprints})
    matrix_arrays = {item["id"]: np.asarray(item["values"], dtype=np.float64) for item in matrices}
    summaries = {}
    noise_dir = paths["inputs"] / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)
    base_infos = []
    gate_snr = 0.0
    gate_rms = 0.0
    with tempfile.TemporaryDirectory(prefix=".noise-build-", dir=paths["inputs"]) as temporary:
        staging = Path(temporary)
        diagnostics_path = staging / "noise_diagnostics.csv"
        with diagnostics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_FIELDS)
            writer.writeheader()
            for seed in seeds:
                buffers = {}
                hashes = {}
                filenames = {}
                seed_cases = {}
                for source in sources:
                    identifier = source["source_id"]
                    for condition in CONDITIONS:
                        key = (identifier, condition)
                        filename = f"{identifier}__n{seed}__{condition}__0db.npy"
                        filenames[key] = filename
                        buffers[key] = np.lib.format.open_memmap(
                            staging / filename, mode="w+", dtype=np.float32, shape=clean.shape)
                        for snr in source["snrs"]:
                            case = _case(source, seed, condition, int(snr), cfg)
                            seed_cases[(identifier, condition, int(snr))] = case
                            hashes[(identifier, condition, int(snr))] = hashlib.sha256()
                try:
                    for index, (ecg_id, signal) in enumerate(zip(ids, clean)):
                        signal64 = np.asarray(signal, dtype=np.float64)
                        signal_power = float(np.mean(signal64 ** 2))
                        for source, physical, independent, crop, df, lowest in _record_sources(
                                cfg, sources, matrix_arrays, seed, int(ecg_id)):
                            identifier = source["source_id"]
                            physical_scaled = scale_to_snr(physical, signal, 0)
                            base = {
                                "electrode": physical_scaled.astype(np.float32),
                                "independent_rms": match_lead_rms(independent, physical_scaled).astype(np.float32),
                            }
                            for condition in CONDITIONS:
                                buffers[(identifier, condition)][index] = base[condition]
                            maxima = {condition: {"max_snr_abs_error_db": 0.0,
                                                  "max_scaled_noise_snr_abs_error_db": 0.0}
                                      for condition in CONDITIONS}
                            rms_error = 0.0
                            scaled_rms_error = 0.0
                            zero_residuals = {}
                            zero_snr = {}
                            for snr in source["snrs"]:
                                multiplier = np.float32(10 ** (-float(snr) / 20))
                                scaled = {}
                                residuals = {}
                                for condition in CONDITIONS:
                                    # These two explicit float32 operations are the inference contract.
                                    scaled[condition] = np.multiply(base[condition], multiplier, dtype=np.float32)
                                    final = np.add(signal, scaled[condition], dtype=np.float32)
                                    hashes[(identifier, condition, int(snr))].update(memoryview(final).cast("B"))
                                    residuals[condition] = final.astype(np.float64) - signal64
                                    achieved = _snr(signal_power, residuals[condition])
                                    actual_error = abs(achieved - snr)
                                    scaled_error = abs(_snr(signal_power, scaled[condition]) - snr)
                                    maxima[condition]["max_snr_abs_error_db"] = max(
                                        maxima[condition]["max_snr_abs_error_db"], actual_error)
                                    maxima[condition]["max_scaled_noise_snr_abs_error_db"] = max(
                                        maxima[condition]["max_scaled_noise_snr_abs_error_db"], scaled_error)
                                    if snr == 0:
                                        zero_residuals[condition] = residuals[condition]
                                        zero_snr[condition] = achieved
                                    gate_snr = max(gate_snr, actual_error, scaled_error)
                                rms_error = max(rms_error, _rms_error(residuals["independent_rms"], residuals["electrode"]))
                                scaled_rms_error = max(scaled_rms_error, _rms_error(scaled["independent_rms"], scaled["electrode"]))
                            gate_rms = max(gate_rms, rms_error, scaled_rms_error)
                            if gate_snr > cfg["gates"]["snr_abs_error_db"]:
                                raise RuntimeError(f"Float32 SNR gate failed: {identifier}, seed={seed}, ecg={ecg_id}, max={gate_snr}")
                            if gate_rms > cfg["gates"]["lead_rms_relative_error"]:
                                raise RuntimeError(f"Float32 matched-RMS gate failed: {identifier}, seed={seed}, ecg={ecg_id}, max={gate_rms}")
                            for condition in CONDITIONS:
                                row = {
                                    "source_id": identifier, "matrix_id": source["matrix_id"],
                                    "family": source["matrix_family"], "band_id": source["band_id"],
                                    "noise_seed": seed, "condition": condition, "ecg_id": int(ecg_id),
                                    "achieved_snr_0": zero_snr[condition],
                                    "max_lead_rms_relative_error": rms_error,
                                    "max_scaled_lead_rms_relative_error": scaled_rms_error,
                                    "source_fft_df_hz": df, "source_lowest_retained_hz": lowest,
                                    **maxima[condition], **crop[condition],
                                    **_final_diagnostics(zero_residuals[condition], float(cfg["sampling_rate"]),
                                                         cfg["band_generation"]["power_bins"]),
                                }
                                writer.writerow(row)
                                _update_summary(summaries, (identifier, seed, condition, source["matrix_family"], source["band_id"]), row)
                                _update_summary(summaries, ("__overall__", None, condition, "all", "all"), row)
                    for buffer in buffers.values():
                        buffer.flush()
                finally:
                    # Explicitly close mappings before rename, including on Windows.
                    for buffer in buffers.values():
                        buffer._mmap.close()
                    buffers.clear()
                for key, filename in filenames.items():
                    destination = noise_dir / filename
                    os.replace(staging / filename, destination)
                    info = file_info(destination)
                    base_infos.append(info)
                    identifier, condition = key
                    for case_key, case in seed_cases.items():
                        if case_key[:2] == key:
                            case["noise_path"] = info["path"]
                            case["noise_sha256"] = info["sha256"]
                            case["input_sha256"] = hashes[case_key].hexdigest()
                            cases.append(case)
                handle.flush()
                print(f"Noise seed {seed}: {len(ids)} records, {len(filenames)} base files completed", flush=True)
            os.fsync(handle.fileno())
        os.replace(diagnostics_path, paths["tables"] / "noise_diagnostics.csv")
    write_csv(paths["tables"] / "noise_summary.csv", _summary_rows(summaries))
    matrix_path = paths["inputs"] / "matrices.json"
    save_json(matrix_path, {
        "status": "completed", "config_sha256": cfg["_config_sha256"],
        "lead_order": list(LEADS), "electrode_order": list(ELECTRODES), "matrices": matrices,
        "randomization": cfg["matrix"]["randomization"],
        "interpretation": cfg["matrix"]["interpretation"],
        "mask_definition": "limb_only retains RA/LA/LL columns; precordial_only retains V1..V6 columns; neither masks output lead rows",
        "covariance_definition": "A A^T for unit-variance independent electrodes; column_gram is signed A^T A, cosine is its column-norm normalization",
        "rank_changes": {item["id"]: item["diagnostics"]["rank"] - matrices[0]["diagnostics"]["rank"] for item in matrices},
        "randomization_preserves": ["row/column nonzero counts", "row/column squared norms"],
        "randomization_does_not_preserve": ["rank", "signed Gram", "common-mode nullspace", "lead covariance topology"],
    })
    cases = [cases[0], *sorted(cases[1:], key=lambda case: case["case_id"])]
    if len(cases) != expected_cases or len(base_infos) != expected_bases:
        raise RuntimeError("Incomplete noise case/base-file grid")
    if len({case["case_id"] for case in cases}) != expected_cases:
        raise RuntimeError("Duplicate noise cases")
    # Detect a changed freeze, source, clean file or cohort during a long run.
    if _fingerprints(cfg, paths) != fingerprints:
        raise RuntimeError("Noise inputs or source fingerprints changed during generation")
    manifest = {
        "status": "completed", "stage": stage, "config_sha256": cfg["_config_sha256"],
        "provenance": fingerprints, "cohort": fingerprints["cohort"], "clean": fingerprints["clean"],
        "matrices": file_info(matrix_path),
        "diagnostics": {"per_record": file_info(paths["tables"] / "noise_diagnostics.csv"),
                        "summary": file_info(paths["tables"] / "noise_summary.csv")},
        "n_records": len(ids), "n_cases": len(cases), "n_base_files": len(base_infos),
        "base_files": base_infos, "cases": cases,
        "gates": {"status": "passed", "max_snr_abs_error_db": gate_snr,
                  "snr_abs_error_db_limit": cfg["gates"]["snr_abs_error_db"],
                  "max_lead_rms_relative_error": gate_rms,
                  "lead_rms_relative_error_limit": cfg["gates"]["lead_rms_relative_error"]},
        "diagnostic_definitions": {
            "final_noise": "actual float32 noisy input converted to float64 minus float32 clean converted to float64; includes float32 addition rounding",
            "base_noise": "float32 0-dB .npy shared across SNR; multiply in float32 then add clean in float32",
            "achieved_snr_0": "whole-record whole-12-lead clean power divided by actual injected residual power, at 0dB",
            "max_errors": "maximum across every configured SNR for this base file; scaled variants audit float32 noise before addition; other variants audit actual post-addition residual",
            "lead_rms": "paired independent_rms versus electrode on reference-active leads; inactive leads must remain exactly zero",
            "effective_cross_lead_correlation": "mean absolute off-diagonal Pearson correlation among positive-variance residual leads only, at 0dB; undefined when fewer than two active leads",
            "power_fractions": "actual 0-dB injected residual, rectangular full-1000-sample one-sided periodogram; doubled interior bins, no DC; summed across all leads; fractions, not Welch integrals",
            "power_bin_rule": cfg["band_generation"]["power_bin_rule"],
            "crop_mean_removed_rms": "RMS across twelve lead means of the unscaled crop, before scale_to_snr or match_lead_rms; source units, not final mV",
            "crop_to_long_source_power_ratio": "uncentered crop mean power / full source mean power after lead mapping but before SNR/RMS scaling; independent condition uses raw independent-lead sources",
            "crop_demeaning_retained_power_fraction": "demeaned crop power / raw crop power, before SNR/RMS scaling",
            "source_fft_df_hz": "generation grid: 0.1Hz legacy periodic, 0.01Hz long-source branches; final ten-second diagnostic grid is 0.1Hz",
            "source_lowest_retained_hz": "lowest included positive source FFT bin, not a claim that the cropped record resolves this frequency",
            "low_frequency_limit": "0.05Hz exists on the 100-second source grid but is not resolved in a ten-second periodogram; central cropping and demeaning attenuate slowly varying content and introduce spectral leakage",
            "band_sharing": "same independent/electrode 100-second white draws and precomputed forward FFTs shared across all four bands; inclusive low/high source endpoints",
            "legacy_sharing": "same phase-one per-record and per-channel SeedSequence sources across all matrix variants; only first two of three streams are generated, no covariance control",
            "aggregation": "equal records within each source/seed/condition; overall separately for each condition gives equal weight to every source/seed/record, not an inferential replicate pool",
            "host": "canonical CPU NumPy generation only; downstream machines consume these exact files",
        },
    }
    save_json(manifest_path, manifest)
    print(f"Noise generation completed: {len(cases)} cases, {len(base_infos)} base files", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
