"""Event/epoch support and baseline-normalised ERD for stored signals.

A raw waveform is a static snapshot; task EEG is about *event-locked dynamics* —
e.g. event-related desynchronisation (ERD), the band-power drop over the cortex
engaged by a task, measured **relative to a pre-event baseline**. A static
whole-recording feature cannot see this (it averages the dynamics away), which is
why a plain hemispheric-asymmetry feature fails to recover motor-imagery
lateralisation. This module stores the event markers with a recording and
computes ERD from the stored waveform on call.

ERD(%) = 100 * (P_active - P_baseline) / P_baseline, per channel, in a band.
Negative = desynchronisation (power drop). Contralateral motor cortex
desynchronises during hand motor imagery, so the *lateralisation* of ERD tracks
the imagined side.
"""

from __future__ import annotations

import numpy as np

from .embedders.eeg import hemisphere
from .file import MindFile, utcnow

_BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
          "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}


# -- store / load ----------------------------------------------------------
def add_events(mf: MindFile, item_id: int, events) -> int:
    """Attach event markers to a recording. ``events`` is an iterable of dicts
    (``onset`` [s] required; optional ``duration``, ``condition``, ``value``,
    ``label``) or ``(onset, duration, condition)`` tuples. Returns the count."""
    rows = []
    for e in events:
        if isinstance(e, dict):
            rows.append((item_id, float(e["onset"]), float(e.get("duration", 0.0)),
                         e.get("condition"), (None if e.get("value") is None else str(e["value"])),
                         e.get("label")))
        else:
            o, d, cond = (list(e) + [0.0, None])[:3]
            rows.append((item_id, float(o), float(d), cond, None, None))
    mf.conn.executemany(
        "INSERT INTO events(item_id, onset, duration, condition, value, label) "
        "VALUES(?,?,?,?,?,?)", rows)
    mf.record("events", {"item_id": item_id, "n": len(rows)})
    mf.conn.commit()
    return len(rows)


def load_events(mf: MindFile, item_id: int, *, condition=None, value=None) -> list[dict]:
    sql = "SELECT onset, duration, condition, value, label FROM events WHERE item_id=?"
    params: list = [item_id]
    if condition is not None:
        sql += " AND condition=?"; params.append(condition)
    if value is not None:
        sql += " AND value=?"; params.append(str(value))
    sql += " ORDER BY onset"
    return [dict(r) for r in mf.conn.execute(sql, params)]


# -- ERD -------------------------------------------------------------------
def _band_power(win: np.ndarray, sfreq: float, band) -> np.ndarray:
    """Per-channel power in ``band`` for a (n_channels, n_samples) window."""
    n = win.shape[1]
    if n < 4:
        return np.zeros(win.shape[0])
    win = win - win.mean(axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    psd = np.abs(np.fft.rfft(win, axis=1)) ** 2
    m = (freqs >= band[0]) & (freqs < band[1])
    return psd[:, m].sum(axis=1) / max(n, 1)


def erd(mf: MindFile, item_id: int, *, band: str = "alpha", condition=None,
        active=(0.0, None), baseline_len: float = 2.0) -> np.ndarray:
    """Per-channel ERD(%) averaged over the epochs of ``condition``.

    For each event, ``baseline`` = the ``baseline_len`` seconds before onset and
    ``active`` = ``(start, stop)`` seconds relative to onset (``stop=None`` uses
    the event's duration). Returns a per-channel array of ERD percentages.
    """
    from . import signal as _signal
    data, sfreq, ch = _signal.load_signal(mf, item_id)
    b = _BANDS[band]
    evs = load_events(mf, item_id, condition=condition)
    n = data.shape[1]
    # Average power across epochs FIRST, then take the ratio once — the standard
    # ERD definition; robust to individual epochs with a near-zero baseline
    # (per-epoch ratios blow up on tiny denominators).
    pa_sum = pb_sum = None
    cnt = 0
    for e in evs:
        onset = e["onset"]
        a0 = int((onset + active[0]) * sfreq)
        a1 = int((onset + (active[1] if active[1] is not None else e["duration"])) * sfreq)
        b0 = int((onset - baseline_len) * sfreq)
        b1 = int(onset * sfreq)
        if b0 < 0 or a1 > n or a1 <= a0 or b1 <= b0:
            continue
        pa = _band_power(data[:, a0:a1], sfreq, b)
        pb = _band_power(data[:, b0:b1], sfreq, b)
        pa_sum = pa if pa_sum is None else pa_sum + pa
        pb_sum = pb if pb_sum is None else pb_sum + pb
        cnt += 1
    if not cnt:
        return np.full(data.shape[0], np.nan)
    pb_mean = np.where(pb_sum <= 0, np.nan, pb_sum / cnt)
    return 100.0 * (pa_sum / cnt - pb_mean) / pb_mean


def erd_lateralization(mf: MindFile, item_id: int, *, band: str = "alpha",
                       condition=None, channels=None, **kw) -> dict:
    """Hemispheric ERD lateralisation: mean ERD over right vs left channels.

    ``li = ERD_right - ERD_left``. Contralateral desynchronisation makes ``li``
    more negative for left-hand and more positive for right-hand motor imagery.
    Power is *pooled across a hemisphere's channels before dividing* (a sum of
    many channels can't be near-zero), so the ratio is stable. Returns
    ``{left, right, li, n_left, n_right}``.
    """
    from . import signal as _signal
    data, sfreq, ch = _signal.load_signal(mf, item_id)
    b = _BANDS[band]
    active = kw.get("active", (0.0, None))
    baseline_len = kw.get("baseline_len", 2.0)
    evs = load_events(mf, item_id, condition=condition)
    n = data.shape[1]
    keep = (lambda name: True) if channels is None else (lambda name: str(name).upper() in
                                                         {c.upper() for c in channels})
    left = [i for i, nm in enumerate(ch or []) if hemisphere(nm) == "L" and keep(nm)]
    right = [i for i, nm in enumerate(ch or []) if hemisphere(nm) == "R" and keep(nm)]

    def pooled(idx):
        if not idx:
            return float("nan")
        pa = pb = 0.0
        for e in evs:
            onset = e["onset"]
            a0 = int((onset + active[0]) * sfreq)
            a1 = int((onset + (active[1] if active[1] is not None else e["duration"])) * sfreq)
            b0, b1 = int((onset - baseline_len) * sfreq), int(onset * sfreq)
            if b0 < 0 or a1 > n or a1 <= a0 or b1 <= b0:
                continue
            pa += float(_band_power(data[idx][:, a0:a1], sfreq, b).sum())
            pb += float(_band_power(data[idx][:, b0:b1], sfreq, b).sum())
        return 100.0 * (pa - pb) / pb if pb > 0 else float("nan")

    ml, mr = pooled(left), pooled(right)
    return {"left": ml, "right": mr, "li": mr - ml, "n_left": len(left), "n_right": len(right)}
