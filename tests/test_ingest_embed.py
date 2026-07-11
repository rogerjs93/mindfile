import numpy as np
import pytest

from mindfile import MindFile, HashEmbedder, ingest, index
from mindfile._vec import sha256_bytes, pack


def test_embed_self_retrieval(mind_path):
    emb = HashEmbedder(dim=16)
    mf = MindFile.create(mind_path, modality="text", embedder=emb)
    texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
    ingest.add_items(mf, texts, external_ids=texts)
    # A payload's own embedding must retrieve itself first (deterministic embedder).
    qvec = emb.embed(["gamma"])[0]
    top = index.search(mf, qvec, k=1)[0]
    ext = mf.conn.execute("SELECT external_id FROM items WHERE id=?", (top[0],)).fetchone()[0]
    assert ext == "gamma"
    assert top[1] == pytest.approx(1.0, abs=1e-5)
    mf.close()


def test_add_vectors_sets_correct_hashes(mind_path):
    mf = MindFile.create(mind_path, modality="synthetic")
    vecs = np.random.default_rng(0).standard_normal((4, 8)).astype("float32")
    ids = ingest.add_vectors(mf, vecs, payloads=[b"a", b"b", b"c", b"d"])
    for i, item_id in enumerate(ids):
        r = mf.conn.execute(
            "SELECT i.payload, i.payload_sha256, v.embedding, v.vec_sha256 "
            "FROM items i JOIN vectors v ON v.item_id=i.id WHERE i.id=?", (item_id,)
        ).fetchone()
        assert r["payload_sha256"] == sha256_bytes(r["payload"])
        assert r["vec_sha256"] == sha256_bytes(r["embedding"])
        assert r["vec_sha256"] == sha256_bytes(pack(vecs[i]))
    mf.close()


def test_dim_mismatch_rejected(mind_path):
    mf = MindFile.create(mind_path, modality="synthetic")
    ingest.add_vectors(mf, np.zeros((2, 8), dtype="float32"))
    with pytest.raises(ValueError):
        ingest.add_vectors(mf, np.zeros((2, 16), dtype="float32"))
    mf.close()
