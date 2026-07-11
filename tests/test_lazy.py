import numpy as np

from mindfile import MindFile, ingest, axes, fuse, constellation, lazy


def _two_modalities(mind_path):
    """3 patients with EEG(5d)+MRI(8d) at t0, EEG items carry x,y,z,t."""
    mf = MindFile.create(mind_path, modality="eeg")
    rng = np.random.default_rng(0)
    eeg = []
    for i, p in enumerate(["P0", "P1", "P2"]):
        eeg += ingest.add_vectors(mf, rng.normal(i, 0.3, (1, 5)).astype("float32"), space="bandpower",
                                  subject_ids=p, session_ids="t0", modalities="eeg", external_ids=f"{p}-eeg")
        ingest.add_vectors(mf, rng.normal(i, 0.3, (1, 8)).astype("float32"), space="mri",
                           subject_ids=p, session_ids="t0", modalities="mri", external_ids=f"{p}-mri")
    axes.set_axes_bulk(mf, eeg, rng.normal(0, 1, (len(eeg), 4)).astype("float32"), ["x", "y", "z", "t"])
    return mf


def test_lazy_fusion_resolves_identically_to_eager(tmp_path):
    # eager file
    eager = _two_modalities(tmp_path / "eager.mind")
    fuse.fuse(eager, ["bandpower", "mri"])
    eager_vecs = {r["external_id"]: np.frombuffer(r["embedding"], dtype="<f4")
                  for r in eager.conn.execute(
                      "SELECT i.external_id, v.embedding FROM items i JOIN vectors v ON v.item_id=i.id "
                      "JOIN spaces s ON s.id=v.space_id WHERE s.name='fused'")}
    # lazy file (same inputs)
    mf = _two_modalities(tmp_path / "lazy.mind")
    fuse.fuse(mf, ["bandpower", "mri"], lazy=True)
    # nothing materialized in 'fused' yet
    assert mf.counts()["vector_defs"] == 3
    sid = __import__("mindfile").spaces.get_space_id(mf, "fused")
    assert mf.conn.execute("SELECT COUNT(*) FROM vectors WHERE space_id=?", (sid,)).fetchone()[0] == 0

    for r in mf.conn.execute("SELECT id, external_id FROM items WHERE origin='derived'"):
        assert lazy.is_virtual(mf, r["id"], "fused")
        resolved = lazy.resolve(mf, r["id"], "fused")
        np.testing.assert_allclose(resolved, eager_vecs[r["external_id"]], rtol=1e-6, atol=1e-6)
    eager.close(); mf.close()


def test_materialize_then_dematerialize_roundtrips(tmp_path):
    mf = _two_modalities(tmp_path / "m.mind")
    fuse.fuse(mf, ["bandpower", "mri"], lazy=True)
    sid = __import__("mindfile").spaces.get_space_id(mf, "fused")

    n = lazy.materialize_space(mf, "fused")
    assert n == 3
    assert mf.conn.execute("SELECT COUNT(*) FROM vectors WHERE space_id=?", (sid,)).fetchone()[0] == 3
    item = mf.conn.execute("SELECT id FROM items WHERE origin='derived' LIMIT 1").fetchone()["id"]
    assert not lazy.is_virtual(mf, item, "fused")          # now concrete
    before = lazy.resolve(mf, item, "fused").copy()

    lazy.dematerialize(mf, item, "fused")
    assert lazy.is_virtual(mf, item, "fused")              # back to virtual, recipe intact
    np.testing.assert_array_equal(lazy.resolve(mf, item, "fused"), before)
    mf.close()


def test_verify_defs_detects_source_tamper(tmp_path):
    mf = _two_modalities(tmp_path / "v.mind")
    fuse.fuse(mf, ["bandpower", "mri"], lazy=True)
    assert lazy.verify_defs(mf)["status"] == "ok"
    # tamper a source (P0 bandpower) -> the recipe's inputs no longer match
    blob = mf.conn.execute(
        "SELECT v.embedding FROM vectors v JOIN items i ON i.id=v.item_id "
        "JOIN spaces s ON s.id=v.space_id WHERE i.external_id='P0-eeg' AND s.name='bandpower'"
    ).fetchone()[0]
    arr = np.frombuffer(blob, dtype="<f4").copy(); arr[0] += 5.0
    mf.conn.execute(
        "UPDATE vectors SET embedding=?, vec_sha256=? WHERE item_id="
        "(SELECT id FROM items WHERE external_id='P0-eeg') AND space_id="
        "(SELECT id FROM spaces WHERE name='bandpower')",
        (arr.tobytes(), __import__("mindfile")._vec.sha256_bytes(arr.tobytes())))
    mf.conn.commit()
    assert lazy.verify_defs(mf)["status"] == "corrupted"
    mf.close()


def test_lazy_constellation_matches_eager(tmp_path):
    eager = _two_modalities(tmp_path / "ce.mind")
    constellation.build_constellation_space(eager, "bandpower")
    eager_vecs = {r["item_id"]: np.frombuffer(r["embedding"], dtype="<f4")
                  for r in eager.conn.execute(
                      "SELECT v.item_id, v.embedding FROM vectors v JOIN spaces s ON s.id=v.space_id "
                      "WHERE s.name='constellation'")}
    mf = _two_modalities(tmp_path / "cl.mind")
    constellation.build_constellation_space(mf, "bandpower", lazy=True)
    for item_id, ev in eager_vecs.items():
        np.testing.assert_allclose(lazy.resolve(mf, item_id, "constellation"), ev, rtol=1e-6, atol=1e-6)
    eager.close(); mf.close()
