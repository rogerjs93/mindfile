import numpy as np

from mindfile import MindFile, modality, lazy, signal


def test_eeg_processor_stores_waveform_and_lazy_features(mind_path):
    mf = MindFile.create(mind_path, modality="eeg")
    data = np.random.default_rng(0).normal(0, 1, (8, 1000)).astype("float32")
    item = modality.ingest(mf, "eeg", data, sfreq=500.0, subject_id="P1", session_id="t0")
    # waveform stored losslessly; band-power vector is a lazy recipe from it
    assert signal.has_signal(mf, item)
    assert lazy.is_virtual(mf, item, "bandpower")
    feat = lazy.resolve(mf, item, "bandpower")
    assert feat.shape[0] == 10
    mf.close()


def test_tabular_processor_ingests_a_row(mind_path):
    mf = MindFile.create(mind_path, modality="labs")
    proc = modality.register(modality.TabularProcessor(["hb", "wbc", "crp"], modality="labs"))
    item = proc.ingest(mf, {"hb": 13.5, "wbc": 6.2, "crp": 4.0}, subject_id="P1")
    r = mf.conn.execute("SELECT modality, payload FROM items WHERE id=?", (item,)).fetchone()
    assert r["modality"] == "labs" and b"hb" in r["payload"]
    v = mf.conn.execute(
        "SELECT v.dim FROM vectors v JOIN spaces s ON s.id=v.space_id WHERE s.name='labs'"
    ).fetchone()
    assert v["dim"] == 3
    mf.close()


def test_registry_describes_protocols(mind_path):
    desc = {d["modality"]: d for d in modality.describe()}
    assert {"eeg", "ecg", "text", "mri", "genomics"} <= set(desc)
    assert desc["eeg"]["spaces"] == ["bandpower", "connectivity", "asymmetry"]


def test_ecg_reuses_waveform_store(mind_path):
    mf = MindFile.create(mind_path, modality="ecg")
    data = np.random.default_rng(0).normal(0, 1, (2, 1000)).astype("float32")
    item = modality.ingest(mf, "ecg", data, sfreq=250.0, subject_id="P1")
    assert signal.has_signal(mf, item)
    assert lazy.resolve(mf, item, "ecg_bandpower").shape[0] == 6   # 3 bands x (mean,std)
    mf.close()


def test_text_processor(mind_path):
    mf = MindFile.create(mind_path, modality="text")
    item = modality.ingest(mf, "text", "patient reports chest pain and dyspnea", subject_id="P1")
    r = mf.conn.execute("SELECT modality, payload FROM items WHERE id=?", (item,)).fetchone()
    assert r["modality"] == "text" and b"dyspnea" in r["payload"]
    assert mf.conn.execute(
        "SELECT v.dim FROM vectors v JOIN spaces s ON s.id=v.space_id WHERE s.name='text'"
    ).fetchone()["dim"] == 256
    mf.close()


def test_mri_processor_sets_xyz_axes(mind_path):
    from mindfile import axes
    mf = MindFile.create(mind_path, modality="mri")
    vol = np.random.default_rng(0).random((8, 8, 8)).astype("float32")
    item = modality.ingest(mf, "mri", vol, subject_id="P1")
    assert set(axes.get_axes(mf, item)) == {"x", "y", "z"}    # feeds constellations
    assert mf.conn.execute(
        "SELECT v.dim FROM vectors v JOIN spaces s ON s.id=v.space_id WHERE s.name='radiomics'"
    ).fetchone()["dim"] == 21    # 15 first-order + 6 GLCM texture
    mf.close()


def test_genomics_processor_hashes_variants(mind_path):
    mf = MindFile.create(mind_path, modality="genomics")
    variants = ["1:100:A:G", "2:200:C:T", "7:55:G:A"]
    item = modality.ingest(mf, "genomics", variants, subject_id="P1")
    row = mf.conn.execute(
        "SELECT v.dim, v.embedding FROM vectors v JOIN spaces s ON s.id=v.space_id "
        "WHERE s.name='variants'").fetchone()
    assert row["dim"] == 512
    assert np.isclose(np.linalg.norm(np.frombuffer(row["embedding"], dtype="<f4")), 1.0, atol=1e-5)
    mf.close()
