import numpy as np

from mindfile import MindFile, modality, ecg_features, render


def _synthetic_ecg(bpm=60, dur=20, sfreq=250, seed=0):
    """QRS-like spikes at a fixed rate + light noise."""
    rng = np.random.default_rng(seed)
    n = int(dur * sfreq)
    sig = np.zeros(n)
    step = sfreq * 60.0 / bpm
    c = 0.1 * sfreq
    while c < n:
        idx = np.arange(int(c) - 8, int(c) + 8)
        idx = idx[(idx >= 0) & (idx < n)]
        sig[idx] += np.exp(-0.5 * ((idx - c) / 2.0) ** 2)
        c += step
    return (sig + 0.01 * rng.normal(size=n)).astype("float32")


def test_hr_detected_from_synthetic_ecg():
    sig = _synthetic_ecg(bpm=60, sfreq=250)
    feats = ecg_features.hrv_features(sig, 250.0)
    assert 55 <= feats[0] <= 65 and feats[5] >= 15   # HR ~60 bpm, ~20 beats
    # a faster rhythm reads faster
    fast = ecg_features.hrv_features(_synthetic_ecg(bpm=100, sfreq=250), 250.0)
    assert fast[0] > feats[0] + 20


def test_ecg_processor_adds_hrv_space(mind_path):
    mf = MindFile.create(mind_path, modality="ecg")
    item = modality.ingest(mf, "ecg", _synthetic_ecg(bpm=75)[None, :], sfreq=250.0, subject_id="P1")
    dims = {r["name"]: r["dim"] for r in mf.conn.execute(
        "SELECT s.name, v.dim FROM vectors v JOIN spaces s ON s.id=v.space_id WHERE v.item_id=?",
        (item,))}
    assert dims.get("ecg_hrv") == 6
    card = render.card(mf, item)
    assert "HR" in card and "bpm" in card
    mf.close()
