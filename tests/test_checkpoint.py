import numpy as np
import pytest

from mindfile import MindFile, ingest, cluster, evolve, checkpoint


def _seed(mind_path, blobs):
    mf = MindFile.create(mind_path, modality="synthetic")
    ingest.add_vectors(mf, blobs, payloads=[f"s{i}" for i in range(len(blobs))])
    cluster.run_clustering(mf, k=2)
    return mf


def test_checkpoint_survives_growth_and_recluster(mind_path, three_blobs):
    mf = _seed(mind_path, three_blobs)
    checkpoint.create_checkpoint(mf, "original")
    # Add more data + re-cluster: must NOT corrupt the original save point.
    extra = np.random.default_rng(1).normal((4, 0), 0.4, (15, 2)).astype("float32")
    ingest.add_vectors(mf, extra, payloads=[f"new{i}" for i in range(15)])
    evolve.recluster(mf, k=3)
    rep = checkpoint.verify_checkpoint(mf, "original")
    assert rep["status"] == "ok"
    assert rep["changed"] == [] and rep["missing"] == [] and rep["changed_vectors"] == []
    mf.close()


def test_tamper_is_detected(mind_path, three_blobs):
    mf = _seed(mind_path, three_blobs)
    checkpoint.create_checkpoint(mf, "original")
    # Corrupt one original vector's bytes in place.
    blob = mf.conn.execute("SELECT embedding FROM vectors WHERE item_id=1").fetchone()[0]
    arr = np.frombuffer(blob, dtype="<f4").copy()
    arr[0] += 123.0
    mf.conn.execute("UPDATE vectors SET embedding=? WHERE item_id=1", (arr.tobytes(),))
    mf.conn.commit()

    rep = checkpoint.verify_checkpoint(mf, "original")
    assert rep["status"] == "corrupted"
    assert any(cv["item_id"] == 1 for cv in rep["changed_vectors"])
    ig = checkpoint.verify_integrity(mf)
    assert ig["status"] == "corrupted"
    assert any(b["item_id"] == 1 for b in ig["bad_vector"])
    mf.close()


def test_duplicate_checkpoint_name_rejected(mind_path, three_blobs):
    mf = _seed(mind_path, three_blobs)
    checkpoint.create_checkpoint(mf, "original")
    with pytest.raises(ValueError):
        checkpoint.create_checkpoint(mf, "original")
    mf.close()
