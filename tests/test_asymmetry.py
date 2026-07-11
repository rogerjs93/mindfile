import numpy as np

from mindfile import MindFile, lazy
from mindfile.embedders.eeg import hemisphere
from mindfile.modality import EEGProcessor


def test_hemisphere_from_10_20_names():
    assert hemisphere("C3") == "L" and hemisphere("C4") == "R"
    assert hemisphere("FP1") == "L" and hemisphere("FP2") == "R"
    assert hemisphere("Cz") is None and hemisphere("Fz") is None      # midline
    assert hemisphere("HEOL") is None                                  # EOG excluded


def test_asymmetry_detects_lateralized_slowing(mind_path):
    sfreq, T = 128.0, 1024
    t = np.arange(T) / sfreq
    rng = np.random.default_rng(0)
    delta = np.sin(2 * np.pi * 2 * t)                 # 2 Hz = delta band
    ch = ["C3", "F3", "C4", "F4"]                      # 2 left, 2 right
    # strong delta on LEFT channels, weak on right -> positive delta asymmetry
    data = np.stack([
        3.0 * delta + 0.1 * rng.normal(size=T),       # C3 left
        3.0 * delta + 0.1 * rng.normal(size=T),       # F3 left
        0.1 * rng.normal(size=T),                     # C4 right
        0.1 * rng.normal(size=T),                     # F4 right
    ]).astype("float32")

    proc = EEGProcessor(crop_seconds=None)
    assert proc.spaces == ["bandpower", "connectivity", "asymmetry"]
    mf = MindFile.create(mind_path, modality="eeg")
    iid = proc.ingest(mf, data, sfreq=sfreq, ch_names=ch, subject_id="S")

    a = lazy.resolve(mf, iid, "asymmetry")
    assert a.shape == (5,) and -1.0 <= a[0] <= 1.0
    assert a[0] > 0.5                                  # delta dominant on the left

    # flip lateralization -> sign flips
    data2 = data[[2, 3, 0, 1]]                          # swap L/R content
    iid2 = proc.ingest(mf, data2, sfreq=sfreq, ch_names=ch, subject_id="S2")
    assert lazy.resolve(mf, iid2, "asymmetry")[0] < -0.5
    mf.close()
