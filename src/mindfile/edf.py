"""Dependency-free EDF/EDF+ *digital* reader.

MNE returns physical values as float64; storing those loses the fact that an EDF
holds **int16 digital samples** plus a per-channel linear calibration
(``physical = digital * gain + offset``). Reading the digital integers directly
is both more faithful to the source file and ~2x smaller before compression
(int16 vs float32) — and it compresses much harder under the signal codec's
delta+shuffle path. Physical (SI volt) values reconstruct exactly on call.

Only standard EDF/EDF+ is handled; the "EDF Annotations" channel is dropped from
the signal matrix. If the ordinary signal channels don't share one sample rate
we raise, and the caller falls back to the MNE float path.
"""

from __future__ import annotations

import numpy as np

# physical-dimension string -> scale to SI volts
_UNIT_SI = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "µv": 1e-6, "nv": 1e-9}


def _si_scale(unit: str) -> float:
    return _UNIT_SI.get(unit.strip().lower(), 1.0)


def read_edf_digital(path: str) -> dict:
    """Read an EDF's raw int16 samples + SI calibration for the signal channels.

    Returns ``{digital(int16 ch x time), gain, offset, unit, sfreq, ch_names}``
    where ``digital * gain[:,None] + offset[:,None]`` is the physical waveform in
    **SI volts** (matching ``mne.get_data``). Raises ``ValueError`` if the file
    isn't uniform-rate standard EDF.
    """
    with open(path, "rb") as f:
        hdr = f.read(256)
        if len(hdr) < 256:
            raise ValueError("not an EDF file (truncated header)")
        n_records = int(hdr[236:244])
        rec_dur = float(hdr[244:252])
        ns = int(hdr[252:256])
        if ns <= 0 or n_records <= 0 or rec_dur <= 0:
            raise ValueError("EDF header has non-positive record count/duration/signals")

        def block(width: int) -> list[str]:
            return [f.read(width).decode("latin-1") for _ in range(ns)]

        labels = [s.strip() for s in block(16)]
        f.read(80 * ns)                                   # transducer
        units = [s.strip() for s in block(8)]             # physical dimension
        pmin = np.array([float(x) for x in block(8)])
        pmax = np.array([float(x) for x in block(8)])
        dmin = np.array([float(x) for x in block(8)])
        dmax = np.array([float(x) for x in block(8)])
        f.read(80 * ns)                                   # prefiltering
        nsamp = np.array([int(x) for x in block(8)])
        f.read(32 * ns)                                   # reserved
        data = np.fromfile(f, dtype="<i2")

    per_rec = int(nsamp.sum())
    data = data[:n_records * per_rec].reshape(n_records, per_rec)
    offs = np.concatenate([[0], np.cumsum(nsamp)])

    keep = [i for i, lab in enumerate(labels) if "annotation" not in lab.lower()]
    if not keep:
        raise ValueError("EDF has no ordinary signal channels")
    if len({nsamp[i] for i in keep}) != 1:
        raise ValueError("EDF signal channels have differing sample rates")

    span = dmax - dmin
    if np.any(span == 0):
        raise ValueError("EDF has a channel with zero digital span")
    gain_phys = (pmax - pmin) / span
    off_phys = pmin - dmin * gain_phys
    scale = np.array([_si_scale(units[i]) for i in range(ns)])

    digital = np.stack([data[:, offs[i]:offs[i + 1]].reshape(-1) for i in keep]).astype(np.int16)
    gain = (gain_phys * scale)[keep]
    offset = (off_phys * scale)[keep]
    return {
        "digital": digital,
        "gain": gain.astype(np.float64),
        "offset": offset.astype(np.float64),
        "unit": "V",
        "sfreq": float(nsamp[keep[0]] / rec_dur),
        "ch_names": [labels[i] for i in keep],
    }
