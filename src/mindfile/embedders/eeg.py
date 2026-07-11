"""EEG band-power embedder — turns an EDF recording into a fixed-length vector.

Features (dim = 2 x n_bands, default 10):
    * relative power in delta/theta/alpha/beta/gamma, averaged across channels
    * the per-band standard deviation across channels (spatial heterogeneity)

Relative powers are scale-free (they sum to ~1 per channel), which makes them a
reasonable semantic fingerprint of a recording for clustering. Deterministic and
model-free: no downloads, consistent with the existing MNE stack.

``mne`` is imported lazily so importing mindfile never requires it.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


class EEGEmbedder:
    def __init__(
        self,
        bands: dict[str, tuple[float, float]] | None = None,
        crop_seconds: float | None = 60.0,
        id: str = "eeg-bandpower-v1",
    ):
        self.bands = bands or _BANDS
        self.crop_seconds = crop_seconds
        self.id = id
        self.dim = 2 * len(self.bands)

    def _features_from_array(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """data: (n_channels, n_samples) -> (2*n_bands,) feature vector."""
        from scipy.signal import welch

        trapz = getattr(np, "trapezoid", np.trapz)  # numpy>=2 renamed trapz
        nperseg = int(min(data.shape[1], max(256, sfreq * 2)))
        freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=1)  # (n_ch, n_freq)

        lo = min(b[0] for b in self.bands.values())
        hi = max(b[1] for b in self.bands.values())
        total_mask = (freqs >= lo) & (freqs <= hi)
        total = trapz(psd[:, total_mask], freqs[total_mask], axis=1)
        total[total == 0] = 1.0

        rel = np.empty((data.shape[0], len(self.bands)), dtype=np.float64)
        for bi, (f0, f1) in enumerate(self.bands.values()):
            m = (freqs >= f0) & (freqs < f1)
            rel[:, bi] = trapz(psd[:, m], freqs[m], axis=1) / total

        return np.concatenate([rel.mean(axis=0), rel.std(axis=0)]).astype(np.float32)

    def embed(self, payloads: Sequence) -> np.ndarray:
        import mne

        out = np.empty((len(payloads), self.dim), dtype=np.float32)
        for i, path in enumerate(payloads):
            raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
            if self.crop_seconds:
                raw.crop(tmax=min(self.crop_seconds, raw.times[-1]))
            out[i] = self._features_from_array(raw.get_data(), raw.info["sfreq"])
        return out


def hemisphere(name: str) -> str | None:
    """Left/right from a 10-20 channel name: odd trailing digit = left, even =
    right, none / 'z' midline = ``None`` (excluded). Portable across montages."""
    m = re.search(r"(\d+)$", str(name).strip())
    if not m:
        return None
    return "L" if int(m.group(1)) % 2 == 1 else "R"


class AsymmetryEmbedder:
    """Directed hemispheric band-power asymmetry (a Brain-Symmetry-Index style
    feature, dim = n_bands, default 5).

    Per band: ``(P_left - P_right) / (P_left + P_right)`` over the mean absolute
    band power of left vs right channels, in [-1, 1]. Positive = left hemisphere
    carries more power in that band. The affected (lesioned) hemisphere tends to
    show increased slow-wave (delta/theta) power, so the sign of the delta
    component tracks lesion laterality. Needs channel names.
    """

    def __init__(self, bands=None, crop_seconds: float | None = 60.0,
                 id: str = "eeg-asymmetry-v1", channels=None):
        self.bands = bands or _BANDS
        self.crop_seconds = crop_seconds
        self.id = id
        self.dim = len(self.bands)
        # optional whitelist (e.g. sensorimotor C3/C4..) to avoid diluting focal
        # asymmetries with channels that carry no signal of interest
        self.channels = {c.upper() for c in channels} if channels else None

    def _features_from_array(self, data: np.ndarray, sfreq: float,
                             ch_names=None) -> np.ndarray:
        if ch_names is None:
            return np.zeros(self.dim, dtype=np.float32)

        def keep(n):
            return self.channels is None or str(n).upper() in self.channels
        left = [i for i, n in enumerate(ch_names) if hemisphere(n) == "L" and keep(n)]
        right = [i for i, n in enumerate(ch_names) if hemisphere(n) == "R" and keep(n)]
        if not left or not right:
            return np.zeros(self.dim, dtype=np.float32)
        from scipy.signal import welch

        trapz = getattr(np, "trapezoid", np.trapz)
        nperseg = int(min(data.shape[1], max(256, sfreq * 2)))
        freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=1)
        out = np.empty(self.dim, dtype=np.float64)
        for bi, (f0, f1) in enumerate(self.bands.values()):
            m = (freqs >= f0) & (freqs < f1)
            bp = trapz(psd[:, m], freqs[m], axis=1)
            l, r = float(bp[left].mean()), float(bp[right].mean())
            out[bi] = (l - r) / (l + r) if (l + r) > 0 else 0.0
        return out.astype(np.float32)

    def embed(self, payloads: Sequence) -> np.ndarray:
        import mne

        out = np.empty((len(payloads), self.dim), dtype=np.float32)
        for i, path in enumerate(payloads):
            raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
            if self.crop_seconds:
                raw.crop(tmax=min(self.crop_seconds, raw.times[-1]))
            out[i] = self._features_from_array(raw.get_data(), raw.info["sfreq"],
                                               raw.ch_names)
        return out


class ConnectivityEmbedder:
    """Per-band inter-channel connectivity — mean absolute Pearson correlation
    between channels after band-pass filtering (dim = n_bands, default 5).

    Channel-count invariant (summarizes the pairwise matrix), model-free, and
    computed from the stored waveform. Loss of inter-channel/inter-hemispheric
    connectivity is a clinically meaningful signal (e.g. after stroke).
    """

    def __init__(self, bands=None, crop_seconds: float | None = 60.0,
                 id: str = "eeg-connectivity-v1"):
        self.bands = bands or _BANDS
        self.crop_seconds = crop_seconds
        self.id = id
        self.dim = len(self.bands)

    def _features_from_array(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        n_ch, n = data.shape
        if n_ch < 2 or n < 2:
            return np.zeros(self.dim, dtype=np.float32)
        data = data - data.mean(axis=1, keepdims=True)
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
        spec = np.fft.rfft(data, axis=1)
        iu = np.triu_indices(n_ch, k=1)
        out = np.empty(self.dim, dtype=np.float64)
        for bi, (f0, f1) in enumerate(self.bands.values()):
            band = np.where((freqs >= f0) & (freqs < f1), spec, 0.0)
            xb = np.fft.irfft(band, n=n, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.corrcoef(xb)               # nan for a zero-variance channel
            vals = np.abs(c[iu])
            vals = vals[np.isfinite(vals)]
            out[bi] = float(vals.mean()) if vals.size else 0.0
        return out.astype(np.float32)

    def embed(self, payloads: Sequence) -> np.ndarray:
        import mne

        out = np.empty((len(payloads), self.dim), dtype=np.float32)
        for i, path in enumerate(payloads):
            raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
            if self.crop_seconds:
                raw.crop(tmax=min(self.crop_seconds, raw.times[-1]))
            out[i] = self._features_from_array(raw.get_data(), raw.info["sfreq"])
        return out
