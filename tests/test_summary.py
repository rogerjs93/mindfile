import numpy as np

from mindfile import MindFile, ingest, fuse, lazy, signal, cluster, summary


def _rich(mind_path):
    """A file with lazy fused vectors + a stored waveform — i.e. lots that COULD unfold."""
    mf = MindFile.create(mind_path, modality="eeg")
    rng = np.random.default_rng(0)
    for i, p in enumerate(["P0", "P1", "P2", "P3"]):
        ingest.add_vectors(mf, rng.normal(i, 0.3, (1, 5)).astype("float32"), space="bandpower",
                           subject_ids=p, session_ids="t0", modalities="eeg", external_ids=f"{p}-eeg")
        ingest.add_vectors(mf, rng.normal(i, 0.3, (1, 8)).astype("float32"), space="mri",
                           subject_ids=p, session_ids="t0", modalities="mri", external_ids=f"{p}-mri")
    cluster.run_clustering(mf, "bandpower", k=2)
    fuse.fuse(mf, ["bandpower", "mri"], lazy=True)        # virtual fused vectors
    item = ingest.add_vectors(mf, np.zeros((1, 1), "float32"), space="ph", subject_ids="P0")[0]
    signal.store_signal(mf, item, rng.normal(0, 1, (8, 1000)).astype("float32"), sfreq=500.0)
    return mf


def test_summary_unfolds_nothing(mind_path):
    mf = _rich(mind_path)
    before_vectors = mf.counts()["vectors"]
    fused_virtual_before = lazy.is_virtual(mf, mf.conn.execute(
        "SELECT id FROM items WHERE origin='derived' LIMIT 1").fetchone()["id"], "fused")

    s = summary.summary(mf)
    txt = summary.summary_text(mf)

    # nothing got materialized: no new vector rows, fused stays virtual
    assert mf.counts()["vectors"] == before_vectors
    assert fused_virtual_before is True
    assert lazy.is_virtual(mf, mf.conn.execute(
        "SELECT id FROM items WHERE origin='derived' LIMIT 1").fetchone()["id"], "fused")

    # the summary carries the useful shape without the bulk
    assert s["counts"]["items"] >= 8
    assert s["subjects"] == 4
    assert any(sp["name"] == "bandpower" and sp["clusters"] for sp in s["spaces"])
    assert s["signals"]["recordings"] == 1
    assert isinstance(txt, str) and "bandpower" in txt
    mf.close()


def test_file_vector_is_cheap_centroid_summary(mind_path):
    mf = _rich(mind_path)
    fv = summary.file_vector(mf, "bandpower")
    assert fv["space"] == "bandpower" and fv["dim"] == 5
    assert len(fv["vector"]) == 5
    mf.close()
