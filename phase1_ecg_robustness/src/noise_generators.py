"""Reproducible mV-domain noise controls for a single 12-lead ECG.

Synthetic sources use a rectangular real-FFT bandpass on Gaussian draws, with
periodic finite-record boundaries. Welch leakage is measured, never assumed
absent. NSTDB sources are contiguous independently selected channel/offset
segments after polyphase resampling; they are structured replay approximations,
NOT measured nine-electrode or twelve-lead recordings. Their native spectra are
retained (the synthetic band argument does not filter NSTDB).
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import resample_poly, welch

from .covariance_matching import (
    covariance_to_correlation,
    empirical_covariance,
    match_covariance,
)
from .lead_matrix import ELECTRODES, LEADS, get_lead_matrix


def _record_array(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != 12 or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape (12, T), T >= 2")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite mV values")
    return array


def _positive_signal_power(x: np.ndarray) -> float:
    power = float(np.mean(np.square(x)))
    if not np.isfinite(power) or power <= 0:
        raise ValueError(
            "ECG must have finite, positive total signal power for SNR calibration"
        )
    return power


def scale_to_snr(noise: np.ndarray, x: np.ndarray, snr_db: float) -> np.ndarray:
    """One global scale factor over all 12 leads, after noise DC removal."""
    x = _record_array(x, "ECG")
    noise = _record_array(noise, "Noise")
    if x.shape != noise.shape:
        raise ValueError("ECG and noise shapes differ")
    if not np.isfinite(snr_db):
        raise ValueError("Requested SNR must be finite")
    signal_power = _positive_signal_power(x)
    centered = noise - noise.mean(axis=1, keepdims=True)
    noise_power = float(np.mean(np.square(centered)))
    if not np.isfinite(noise_power) or noise_power <= 0:
        raise ValueError(
            "Cannot calibrate zero or nonfinite noise power to a finite SNR"
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        desired_power = signal_power * np.power(10.0, -float(snr_db) / 10)
    if not np.isfinite(desired_power) or desired_power <= 0:
        raise ValueError("Requested SNR exceeds representable nonzero power range")
    result = centered * np.sqrt(desired_power / noise_power)
    if not np.isfinite(result).all() or not np.any(result):
        raise ValueError("SNR scaling exceeds representable sample range")
    return result


def match_lead_rms(independent: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale independent leads; NEVER rescale the physical reference.

    Reference-inactive channels remain exactly zero. Natural accidental sample
    correlations of independent draws are retained, not whitened away.
    """
    independent = _record_array(independent, "Independent noise")
    reference = _record_array(reference, "Reference noise")
    if independent.shape != reference.shape:
        raise ValueError("Noise shapes differ")
    centered = independent - independent.mean(axis=1, keepdims=True)
    source_rms = np.sqrt(np.mean(centered**2, axis=1))
    target_rms = np.sqrt(np.mean(reference**2, axis=1))
    active = target_rms > 0
    if np.any(source_rms[active] == 0):
        raise ValueError("Cannot RMS-match an active target to a zero-power source")
    result = np.zeros_like(centered)
    result[active] = (
        centered[active] * (target_rms[active] / source_rms[active])[:, None]
    )
    return result


def _check_band(fs: float, band: tuple[float, float]) -> tuple[float, float]:
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be positive and finite")
    if len(band) != 2:
        raise ValueError("Band needs lower and upper frequency bounds")
    low, high = map(float, band)
    if not np.isfinite([low, high]).all() or not 0 <= low < high <= fs / 2:
        raise ValueError("Band must satisfy 0 <= low < high <= Nyquist")
    return low, high


def bandpass_gaussian(
    length: int,
    fs: float,
    rng: np.random.Generator,
    band: tuple[float, float] = (0.5, 40.0),
) -> np.ndarray:
    """Zero-DC finite-record Gaussian realization filtered in the FFT domain."""
    low, high = _check_band(fs, band)
    if length < 2:
        raise ValueError("At least two temporal samples are required")
    frequency = np.fft.rfftfreq(length, d=1 / fs)
    keep = (frequency >= low) & (frequency <= high) & (frequency > 0)
    if not keep.any():
        raise ValueError(
            "Selected band contains no non-DC Fourier modes at this duration"
        )
    spectrum = np.fft.rfft(rng.standard_normal(length))
    spectrum[~keep] = 0
    result = np.fft.irfft(spectrum, n=length)
    result -= result.mean()
    return result


@lru_cache(maxsize=12)
def _load_nstdb(record_path: str, fs: float) -> tuple[np.ndarray, dict]:
    """Cache full resampled physical waveform once per resolved record/fs pair."""
    import wfdb

    record = wfdb.rdrecord(record_path, physical=True)
    data = np.asarray(record.p_signal, dtype=np.float64)
    source_fs = float(record.fs)
    if source_fs <= 0 or not np.isfinite(source_fs):
        raise ValueError("NSTDB record has invalid sample rate")
    if data.ndim != 2 or data.shape[0] < 2 or not np.isfinite(data).all():
        raise ValueError("NSTDB physical samples must be finite time-by-channel values")
    unit_factors = {"mv": 1.0, "uv": 0.001, "µv": 0.001, "μv": 0.001, "v": 1000.0}
    units = list(record.units)
    if len(units) != data.shape[1]:
        raise ValueError("NSTDB channel unit metadata is missing")
    factors = []
    for unit in units:
        key = str(unit).strip().lower()
        if key not in unit_factors:
            raise ValueError(
                f"Unsupported NSTDB physical unit {unit!r}; cannot infer mV"
            )
        factors.append(unit_factors[key])
    data = data * np.asarray(factors)[None, :]
    ratio = Fraction(str(fs)) / Fraction(str(source_fs))
    ratio = ratio.limit_denominator(100000)
    if not np.isclose(source_fs * ratio.numerator / ratio.denominator, fs, rtol=1e-12):
        raise ValueError("Sampling rate ratio cannot be represented accurately")
    if ratio.numerator != ratio.denominator:
        data = resample_poly(data, ratio.numerator, ratio.denominator, axis=0)
    data = np.ascontiguousarray(data.T)
    data.setflags(write=False)
    provenance = {
        "record_path": record_path,
        "source_fs": source_fs,
        "target_fs": float(fs),
        "resample_up": ratio.numerator,
        "resample_down": ratio.denominator,
        "resample_method": "scipy.signal.resample_poly, default Kaiser window",
        "source_samples": int(len(record.p_signal)),
        "resampled_samples": int(data.shape[1]),
        "channels": int(data.shape[0]),
        "source_units": units,
        "output_units": "mV",
        "segment_policy": "independent channel and contiguous offset per draw; per-segment DC removed",
        "normalization": "global SNR after lead formation; independent_rms adds per-lead RMS control",
        "interpretation": "structured replay approximation, not original 12-lead electrode recordings",
    }
    return data, provenance


def nstdb_source_info(nstdb: str | Path, kind: str, fs: float = 100.0) -> dict:
    if kind not in ("bw", "ma", "em"):
        raise ValueError("NSTDB kind must be bw, ma or em")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be positive and finite")
    path = str((Path(nstdb) / kind).resolve())
    _, provenance = _load_nstdb(path, float(fs))
    return dict(provenance)


def _source_channels(
    channels: int,
    length: int,
    fs: float,
    seed: np.random.SeedSequence,
    kind: str,
    nstdb: str | Path | None,
    band: tuple[float, float],
) -> np.ndarray:
    data = None
    if kind in ("bw", "ma", "em"):
        if nstdb is None:
            raise ValueError(
                "NSTDB directory is required for bw/ma/em; no silent Gaussian fallback"
            )
        data, _ = _load_nstdb(str((Path(nstdb) / kind).resolve()), float(fs))
        if data.shape[1] < length:
            raise ValueError(
                "NSTDB record shorter than requested segment; repeating it is not permitted"
            )
    elif kind != "bandpass":
        raise ValueError(f"Unknown noise kind {kind!r}; use bandpass, bw, ma or em")
    result = np.empty((channels, length), dtype=np.float64)
    for channel, child in enumerate(seed.spawn(channels)):
        rng = np.random.Generator(np.random.PCG64(child))
        if data is None:
            result[channel] = bandpass_gaussian(length, fs, rng, band)
        else:
            source_channel = int(rng.integers(data.shape[0]))
            start = int(rng.integers(data.shape[1] - length + 1))
            result[channel] = data[source_channel, start : start + length]
            result[channel] -= result[channel].mean()
            if not np.any(result[channel]):
                raise ValueError(
                    f"NSTDB sampled a constant segment: {kind}, channel={source_channel}, start={start}"
                )
    return result


def make_noise_triplet(
    x: np.ndarray,
    fs: float,
    snr_db: float,
    seed: int,
    kind: str = "bandpass",
    active: list[str] | tuple[str, ...] | None = None,
    nstdb: str | Path | None = None,
    band: tuple[float, float] = (0.5, 40.0),
) -> dict[str, np.ndarray]:
    """Return four noise arrays (NOT corrupted ECGs), each (12,T), in mV.

    A record seed is split using SeedSequence into independent lead, electrode and
    fresh covariance-control streams, then independent per-channel PCG64 streams.
    Independent-RMS uses the same temporal lead draws as independent, changing only
        amplitude. Changing SNR preserves the non-factorized temporal realizations;
        covariance-control eigenvectors can change sign or rotate in degenerate
        eigenspaces, so cross-SNR waveform equality is not guaranteed. All nine
    source streams are drawn before active selection, preserving each electrode's
    realization across subset sensitivity experiments. RL-only/empty active sets
    are rejected because a finite global SNR is impossible with zero mapped power.
    """
    x = _record_array(x, "ECG")
    _positive_signal_power(x)
    _check_band(fs, band)
    if not np.isfinite(snr_db):
        raise ValueError("Requested SNR must be finite")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Record seed must be a nonnegative integer")
    if active is None:
        selected = ELECTRODES
    else:
        selected = (active,) if isinstance(active, str) else tuple(active)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError(
                "Active electrode set must be nonempty and contain no duplicates"
            )
        unknown = set(selected) - set(ELECTRODES)
        if unknown:
            raise ValueError(
                f"Unknown/inactive diagnostic electrodes: {sorted(unknown)}; RL has zero ideal mapping"
            )
    streams = np.random.SeedSequence(int(seed)).spawn(3)
    independent_source = _source_channels(
        12, x.shape[1], fs, streams[0], kind, nstdb, band
    )
    electrode_source = _source_channels(
        9, x.shape[1], fs, streams[1], kind, nstdb, band
    )
    electrode_source[[name not in selected for name in ELECTRODES]] = 0.0
    electrode = scale_to_snr(get_lead_matrix() @ electrode_source, x, snr_db)
    independent = scale_to_snr(independent_source, x, snr_db)
    independent_rms = match_lead_rms(independent_source, electrode)
    fresh_source = _source_channels(12, x.shape[1], fs, streams[2], kind, nstdb, band)
    covariance, _ = match_covariance(empirical_covariance(electrode), fresh_source)
    return {
        "independent": independent,
        "independent_rms": independent_rms,
        "electrode": electrode,
        "covariance": covariance,
    }


def _power_snr(signal_power: float, noise_power: float) -> float | None:
    if noise_power == 0:
        return float("inf") if signal_power > 0 else None
    if signal_power == 0:
        return float("-inf")
    return float(10 * np.log10(signal_power / noise_power))


def _integrated_band(
    frequencies: np.ndarray, psd: np.ndarray, low: float, high: float
) -> np.ndarray:
    inside = (frequencies > low) & (frequencies < high)
    selected_frequencies = np.concatenate(([low], frequencies[inside], [high]))
    selected_psd = np.column_stack(
        (
            [np.interp(low, frequencies, row) for row in psd],
            psd[:, inside],
            [np.interp(high, frequencies, row) for row in psd],
        )
    )
    return trapezoid(selected_psd, selected_frequencies, axis=1)


def noise_diagnostics(
    noise: np.ndarray, x: np.ndarray, fs: float, band: tuple[float, float] = (0.5, 40.0)
) -> dict:
    """Measured temporal, covariance and Welch spectral diagnostics.

    Undefined inactive-lead correlations are explicit NaN with active_leads mask.
    Zero/zero lead SNR is None; positive/zero is +inf; zero/positive is -inf. Mean
    absolute off-diagonal correlation excludes inactive leads; with fewer than two
    active leads it is None. No undefined quantity is silently changed to zero.
    """
    noise = _record_array(noise, "Noise")
    x = _record_array(x, "ECG")
    if x.shape != noise.shape:
        raise ValueError("ECG and noise shapes differ")
    signal_power = _positive_signal_power(x)
    low, high = _check_band(fs, band)
    covariance = empirical_covariance(noise)
    correlation, active = covariance_to_correlation(covariance)
    off_diagonal = np.outer(active, active) & ~np.eye(12, dtype=bool)
    lead_noise_power = np.mean(noise**2, axis=1)
    lead_signal_power = np.mean(x**2, axis=1)
    frequencies, psd = welch(
        noise,
        fs=fs,
        nperseg=min(256, noise.shape[1]),
        noverlap=min(256, noise.shape[1]) // 2,
        detrend="constant",
        scaling="density",
        axis=1,
    )
    band_power = _integrated_band(frequencies, psd, low, high)
    total_power = trapezoid(psd, frequencies, axis=1)
    fraction = [
        float(value / total) if total > 0 else None
        for value, total in zip(band_power, total_power)
    ]
    bands = {"requested": band_power}
    for name, lower, upper in (
        ("baseline_0_0.5", 0, 0.5),
        ("ecg_0.5_40", 0.5, 40),
        ("high_40_nyquist", 40, fs / 2),
    ):
        upper = min(upper, fs / 2)
        if lower < upper:
            bands[name] = _integrated_band(frequencies, psd, lower, upper)
    return {
        "actual_snr_db": _power_snr(signal_power, float(lead_noise_power.mean())),
        "signal_power_mv2": signal_power,
        "noise_power_mv2": float(lead_noise_power.mean()),
        "per_lead_rms": np.sqrt(lead_noise_power),
        "per_lead_variance": np.diag(covariance).copy(),
        "per_lead_snr_db": [
            _power_snr(float(s), float(n))
            for s, n in zip(lead_signal_power, lead_noise_power)
        ],
        "per_lead_mean": noise.mean(axis=1),
        "covariance": covariance,
        "correlation": correlation,
        "active_leads": active,
        "lead_order": list(LEADS),
        "mean_abs_offdiagonal_correlation": (
            float(np.mean(np.abs(correlation[off_diagonal])))
            if off_diagonal.any()
            else None
        ),
        "frequencies": frequencies,
        "psd": psd,
        "band": [low, high],
        "band_power": band_power,
        "band_power_fraction": fraction,
        "band_powers": bands,
        "welch_total_power": total_power,
        "covariance_ddof": 0,
    }
