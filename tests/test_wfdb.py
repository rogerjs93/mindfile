import numpy as np

from mindfile import MindFile, modality, lazy, signal, render


def test_wfdb_reuses_waveform_store(mind_path):
    mf = MindFile.create(mind_path, modality="wfdb")
    data = np.random.default_rng(0).normal(0, 1, (3, 1500)).astype("float32")
    item = modality.ingest(mf, "wfdb", data, sfreq=250.0,
                           ch_names=["II", "V", "PLETH"], subject_id="P1")
    assert signal.has_signal(mf, item)                       # lossless waveform stored
    assert lazy.resolve(mf, item, "wfdb_spectral").shape[0] == 6   # 3 bands x (mean,std)
    card = render.card(mf, item)
    assert "signals II,V,PLETH" in card and "@250Hz" in card and "rel power" in card
    mf.close()


def test_wfdb_in_digest(mind_path):
    mf = MindFile.create(mind_path, modality="wfdb")
    modality.ingest(mf, "wfdb", np.random.default_rng(1).normal(0, 1, (1, 600)).astype("float32"),
                    sfreq=125.0, ch_names=["PLETH"], subject_id="P2")
    lazy.materialize_space(mf, "wfdb_spectral")
    assert "[wfdb]" in render.digest(mf)
    mf.close()
