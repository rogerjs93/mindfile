import numpy as np

from mindfile import (MindFile, ingest, cluster, checkpoint, validate, fuse)


def _patients(mind_path):
    """Two patients, each with an EEG (bandpower) + MRI item at session t0,
    plus one EEG-only patient (incomplete for fusion)."""
    mf = MindFile.create(mind_path, modality="eeg")
    rng = np.random.default_rng(0)
    for p in ("P1", "P2"):
        ingest.add_vectors(mf, rng.normal(0, 1, (1, 5)).astype("float32"), space="bandpower",
                           subject_ids=p, session_ids="t0", modalities="eeg",
                           acquired_ats="2021", external_ids=f"{p}-eeg")
        ingest.add_vectors(mf, rng.normal(0, 1, (1, 8)).astype("float32"), space="mri",
                           subject_ids=p, session_ids="t0", modalities="mri",
                           acquired_ats="2021", external_ids=f"{p}-mri")
    # P3 has EEG only -> should be skipped by fusion.
    ingest.add_vectors(mf, rng.normal(0, 1, (1, 5)).astype("float32"), space="bandpower",
                       subject_ids="P3", session_ids="t0", modalities="eeg", external_ids="P3-eeg")
    return mf


def test_fusion_is_additive_and_conserves_originals(mind_path):
    mf = _patients(mind_path)
    primaries_before = mf.conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    cp = checkpoint.create_checkpoint(mf, "primary")   # save point over originals only

    res = fuse.fuse(mf, ["bandpower", "mri"])
    assert len(res["created"]) == 2                    # P1, P2 fused; P3 skipped
    assert any(s["reason"] == "missing" for s in res["skipped"])

    # Derived items are NEW rows; originals untouched -> old save point still ok.
    assert mf.conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == primaries_before + 2
    derived = mf.conn.execute("SELECT id, modality, origin FROM items WHERE origin='derived'").fetchall()
    assert len(derived) == 2 and all(d["modality"] == "fused" for d in derived)
    fused_dim = mf.conn.execute(
        "SELECT v.dim FROM vectors v JOIN spaces s ON s.id=v.space_id WHERE s.name='fused' LIMIT 1"
    ).fetchone()["dim"]
    assert fused_dim == 5 + 8                           # concat of the two source dims

    assert checkpoint.verify_checkpoint(mf, "primary")["status"] == "ok"
    assert validate.validate(mf)["status"] == "ok"
    mf.close()


def test_verify_derivations_detects_source_tamper(mind_path):
    mf = _patients(mind_path)
    fuse.fuse(mf, ["bandpower", "mri"])
    assert fuse.verify_derivations(mf)["status"] == "ok"

    # Tamper a SOURCE vector (P1's bandpower) -> the fused point is no longer
    # backed by intact originals.
    blob = mf.conn.execute(
        "SELECT v.embedding FROM vectors v JOIN items i ON i.id=v.item_id "
        "JOIN spaces s ON s.id=v.space_id WHERE i.external_id='P1-eeg' AND s.name='bandpower'"
    ).fetchone()[0]
    arr = np.frombuffer(blob, dtype="<f4").copy(); arr[0] += 9.0
    mf.conn.execute(
        "UPDATE vectors SET embedding=? WHERE item_id=("
        "SELECT id FROM items WHERE external_id='P1-eeg') AND space_id=("
        "SELECT id FROM spaces WHERE name='bandpower')", (arr.tobytes(),))
    mf.conn.commit()

    rep = fuse.verify_derivations(mf)
    assert rep["status"] == "corrupted" and rep["drifted"]
    mf.close()


def test_fused_space_is_clusterable(mind_path):
    mf = _patients(mind_path)
    # add a couple more complete patients so we have >=2 fused points to cluster
    rng = np.random.default_rng(5)
    for p in ("P4", "P5"):
        ingest.add_vectors(mf, rng.normal(0, 1, (1, 5)).astype("float32"), space="bandpower",
                           subject_ids=p, session_ids="t0", modalities="eeg", external_ids=f"{p}-eeg")
        ingest.add_vectors(mf, rng.normal(0, 1, (1, 8)).astype("float32"), space="mri",
                           subject_ids=p, session_ids="t0", modalities="mri", external_ids=f"{p}-mri")
    fuse.fuse(mf, ["bandpower", "mri"])
    cid = cluster.run_clustering(mf, "fused", k=2)
    # exactly the 4 fused points were clustered (primaries excluded -> no vector in 'fused')
    assert len(cluster.assignments(mf, cid)) == 4
    mf.close()
