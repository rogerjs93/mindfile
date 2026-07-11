"""ECG heart-rate + heart-rate-variability features.

Band-power on a raw ECG is a poor proxy; the clinically meaningful signal is the
rhythm. This detects R-peaks (QRS emphasis + peak-finding) and computes
time-domain HRV — HR, SDNN, RMSSD, pNN50 — the standard bedside/research metrics.
Dependency-light (numpy + scipy). Frequency-domain HRV (LF/HF) is a later add.
"""

from __future__ import annotations

import numpy as np

HRV_LABELS = ["hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50", "mean_rr_s", "n_beats"]


def rpeaks(sig: np.ndarray, sfreq: float) -> np.ndarray:
    """R-peak sample indices via QRS energy (derivative -> square -> smooth)."""
    from scipy.signal import find_peaks
    x = np.asarray(sig, dtype=np.float64)
    x = x - np.median(x)
    e = np.diff(x, prepend=x[:1]) ** 2                 # emphasize sharp QRS slopes
    w = max(1, int(round(0.08 * sfreq)))
    ma = np.convolve(e, np.ones(w) / w, mode="same")
    thr = ma.mean() + 0.5 * ma.std()
    peaks, _ = find_peaks(ma, height=thr, distance=max(1, int(round(0.3 * sfreq))))
    return peaks


def hrv_features(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Return [HR, SDNN(ms), RMSSD(ms), pNN50, mean_RR(s), n_beats]."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    lead = data[int(np.argmax(data.var(axis=1)))]       # lead with most QRS energy
    pk = rpeaks(lead, sfreq)
    if len(pk) < 3:
        return np.zeros(len(HRV_LABELS), dtype="float32")
    rr = np.diff(pk) / float(sfreq)                      # R-R intervals, seconds
    drr = np.diff(rr)
    hr = 60.0 / rr.mean()
    sdnn = rr.std() * 1000.0
    rmssd = float(np.sqrt((drr ** 2).mean()) * 1000.0) if drr.size else 0.0
    pnn50 = float((np.abs(drr) > 0.05).mean()) if drr.size else 0.0
    return np.array([hr, sdnn, rmssd, pnn50, rr.mean(), float(len(pk))], dtype="float32")
