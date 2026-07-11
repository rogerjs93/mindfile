import numpy as np

from mindfile import MindFile, events, signal


def _erd_recording():
    """2 channels (C3 left, C4 right), 8 trials. Each trial = 2s baseline (strong
    alpha both channels) + 2s active. In 'left' trials the RIGHT channel's alpha
    drops (contralateral ERD); in 'right' trials the LEFT channel's drops."""
    sfreq, trial_s = 128.0, 4.0
    conds = ["left", "right"] * 4
    t_tr = np.arange(int(trial_s * sfreq)) / sfreq
    carrier = np.sin(2 * np.pi * 10 * t_tr)          # 10 Hz = alpha
    half = len(t_tr) // 2
    rng = np.random.default_rng(0)
    c3, c4, evs = [], [], []
    for i, cond in enumerate(conds):
        aL = np.full(len(t_tr), 3.0); aR = np.full(len(t_tr), 3.0)
        drop = aR if cond == "left" else aL          # contralateral channel
        drop[half:] = 0.4                             # alpha ERD during active half
        c3.append(aL * carrier + 0.05 * rng.normal(size=len(t_tr)))
        c4.append(aR * carrier + 0.05 * rng.normal(size=len(t_tr)))
        evs.append({"onset": i * trial_s + trial_s / 2, "duration": 2.0, "condition": cond})
    data = np.stack([np.concatenate(c3), np.concatenate(c4)]).astype("float32")
    return data, sfreq, ["C3", "C4"], evs


def test_events_store_and_load(mind_path):
    data, sfreq, ch, evs = _erd_recording()
    mf = MindFile.create(mind_path, modality="eeg")
    iid, _ = signal.add_recording(mf, data, sfreq=sfreq, ch_names=ch)
    assert events.add_events(mf, iid, evs) == 8
    assert len(events.load_events(mf, iid)) == 8
    assert len(events.load_events(mf, iid, condition="left")) == 4
    mf.close()


def test_erd_detects_contralateral_desync(mind_path):
    data, sfreq, ch, evs = _erd_recording()
    mf = MindFile.create(mind_path, modality="eeg")
    iid, _ = signal.add_recording(mf, data, sfreq=sfreq, ch_names=ch)
    events.add_events(mf, iid, evs)

    left = events.erd(mf, iid, band="alpha", condition="left")     # [C3, C4]
    assert left[1] < -50 and left[0] > -20                          # C4(right) desync, C3 ~flat

    # lateralization index: negative for left-hand, positive for right-hand
    liL = events.erd_lateralization(mf, iid, band="alpha", condition="left")["li"]
    liR = events.erd_lateralization(mf, iid, band="alpha", condition="right")["li"]
    assert liL < 0 < liR
    mf.close()
