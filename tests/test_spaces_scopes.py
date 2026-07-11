import numpy as np
import pytest

from mindfile import (MindFile, ingest, cluster, checkpoint, validate, analyze,
                      spaces, collections, graph)


def test_multiple_spaces_per_item(mind_path, three_blobs):
    mf = MindFile.create(mind_path, modality="eeg")
    ids = ingest.add_vectors(mf, three_blobs, space="bandpower",
                             subject_ids=[f"p{i}" for i in range(len(three_blobs))])
    # Attach a SECOND embedding view to the same items.
    other = np.random.default_rng(1).standard_normal((len(ids), 5)).astype("float32")
    ingest.add_space_vectors(mf, ids, other, space="connectivity")

    names = {s["name"]: s for s in spaces.list_spaces(mf)}
    assert names["bandpower"]["dim"] == 2 and names["connectivity"]["dim"] == 5
    assert names["bandpower"]["n_vectors"] == names["connectivity"]["n_vectors"] == len(ids)

    # Each space clusters independently.
    c1 = cluster.run_clustering(mf, "bandpower", k=3)
    c2 = cluster.run_clustering(mf, "connectivity", k=2)
    assert len(cluster.clusters_of(mf, c1)) == 3
    assert len(cluster.clusters_of(mf, c2)) == 2

    # A checkpoint covers vectors in BOTH spaces and verifies clean.
    cp = checkpoint.create_checkpoint(mf, "orig")
    assert cp["vector_count"] == 2 * len(ids)
    assert checkpoint.verify_checkpoint(mf, "orig")["status"] == "ok"
    mf.close()


def test_write_once_space_vector(mind_path, three_blobs):
    mf = MindFile.create(mind_path)
    ids = ingest.add_vectors(mf, three_blobs, space="a")
    with pytest.raises(ValueError):  # already has a vector in space 'a'
        ingest.add_space_vectors(mf, ids, three_blobs, space="a")
    mf.close()


def test_group_study_scoped_clustering(mind_path):
    mf = MindFile.create(mind_path, modality="eeg")
    rng = np.random.default_rng(0)
    s1 = rng.normal((0, 0), 0.3, (10, 2)).astype("float32")
    s2 = rng.normal((9, 9), 0.3, (10, 2)).astype("float32")
    ingest.add_vectors(mf, s1, study_ids="S1", modalities="eeg")
    ingest.add_vectors(mf, s2, study_ids="S2", modalities="eeg")
    res = analyze.group_study(mf, "cohortS1", where={"study_id": "S1"}, k=2)
    assert collections.collection_id(mf, "cohortS1") is not None
    # Clustering ranged only over S1's 10 items.
    asg = cluster.assignments(mf, res["clustering_id"])
    assert len(asg) == 10
    mf.close()


def test_longitudinal_trajectory_and_edges(mind_path):
    mf = MindFile.create(mind_path, modality="mri")
    # One subject, 4 timepoints: stable profile then a sharp directional shift.
    vecs = np.array([[1, 0], [0.98, 0.02], [0.96, 0.05], [0.0, 1.0]], dtype="float32")
    ids = ingest.add_vectors(
        mf, vecs, subject_ids="P1",
        session_ids=["t0", "t1", "t2", "t3"],
        acquired_ats=["2020", "2021", "2022", "2023"])
    res = analyze.longitudinal(mf, "P1", change_threshold=0.5)
    assert res["order"] == ids                      # ordered by acquired_at
    assert len(res["steps"]) == 3
    assert res["flagged"] and res["flagged"][-1]["to"] == ids[-1]   # big jump flagged
    # temporal edges were written between consecutive sessions
    assert graph.neighbors(mf, ids[0], kind="temporal")
    mf.close()


def test_multimodal_links(mind_path):
    mf = MindFile.create(mind_path)
    eeg = ingest.add_vectors(mf, np.zeros((1, 2), "float32"), subject_ids="P1", modalities="eeg")
    mri = ingest.add_vectors(mf, np.zeros((1, 2), "float32"), subject_ids="P1", modalities="mri")
    res = analyze.multimodal(mf, "P1")
    assert set(res["modalities"]) == {"eeg", "mri"} and res["links"] == 1
    assert any(e["kind"] == "same_subject" for e in graph.neighbors(mf, eeg[0]))
    mf.close()


def test_validate_flags_problems(mind_path):
    mf = MindFile.create(mind_path, modality="eeg")
    ingest.add_vectors(mf, np.zeros((2, 2), "float32"),
                       subject_ids="P1", session_ids="t0", modalities="eeg")  # duplicate identity
    clean = validate.validate(mf)
    codes = {i["code"] for i in clean["issues"]}
    assert "duplicate_identity" in codes

    # Inject a NaN vector and re-validate.
    bad = np.array([np.nan, 1.0], dtype="<f4").tobytes()
    from mindfile._vec import sha256_bytes
    mf.conn.execute("UPDATE vectors SET embedding=?, vec_sha256=? WHERE item_id=1",
                    (bad, sha256_bytes(bad)))
    mf.conn.commit()
    rep = validate.validate(mf)
    assert rep["errors"] >= 1
    assert "non_finite_vector" in {i["code"] for i in rep["issues"]}
    mf.close()
