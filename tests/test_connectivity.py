import numpy as np

from mindfile import MindFile, lazy
from mindfile.modality import EEGProcessor


def test_connectivity_separates_correlated_from_independent(mind_path):
    sfreq, T, nch = 128.0, 512, 6
    t = np.arange(T) / sfreq
    rng = np.random.default_rng(0)
    shared = np.sin(2 * np.pi * 10 * t)                              # 10 Hz = alpha band
    corr = np.stack([shared + 0.05 * rng.normal(size=T) for _ in range(nch)]).astype("float32")
    indep = rng.normal(size=(nch, T)).astype("float32")

    proc = EEGProcessor(crop_seconds=None)
    assert proc.spaces == ["bandpower", "connectivity", "asymmetry"]
    mf = MindFile.create(mind_path, modality="eeg")
    i_corr = proc.ingest(mf, corr, sfreq=sfreq, subject_id="C")
    i_indep = proc.ingest(mf, indep, sfreq=sfreq, subject_id="I")

    c1 = lazy.resolve(mf, i_corr, "connectivity")     # computed on call from the waveform
    c2 = lazy.resolve(mf, i_indep, "connectivity")
    assert c1.shape == (5,) and c2.shape == (5,)
    assert np.all(c1 >= 0) and np.all(c1 <= 1.0001)   # correlations are bounded
    alpha = 2
    assert c1[alpha] > 0.8 and c2[alpha] < 0.5 and c1[alpha] > c2[alpha]

    # lazy: no vectors row exists until materialized; then it resolves identically
    assert mf.conn.execute("SELECT COUNT(*) FROM vectors v JOIN spaces s ON s.id=v.space_id "
                           "WHERE s.name='connectivity'").fetchone()[0] == 0
    lazy.materialize_space(mf, "connectivity")
    assert np.allclose(lazy.resolve(mf, i_corr, "connectivity"), c1)
    mf.close()
