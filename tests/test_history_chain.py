import numpy as np

from mindfile import MindFile, ingest, cluster, audit


def _active(mind_path):
    mf = MindFile.create(mind_path, modality="eeg")
    rng = np.random.default_rng(0)
    ingest.add_vectors(mf, rng.normal(0, 1, (10, 5)).astype("float32"),
                       external_ids=[f"x{i}" for i in range(10)])
    cluster.run_clustering(mf, k=2)
    return mf


def test_chain_is_valid_after_operations(mind_path):
    mf = _active(mind_path)
    rep = audit.verify_history(mf)
    assert rep["status"] == "ok" and rep["entries"] >= 2 and rep["head_ok"]
    # head in meta matches the last entry
    last = mf.conn.execute("SELECT entry_hash FROM history ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert mf.meta_get("history_head") == last
    mf.close()


def test_editing_an_entry_breaks_the_chain(mind_path):
    mf = _active(mind_path)
    mid = mf.conn.execute("SELECT id FROM history ORDER BY id LIMIT 1").fetchone()["id"]
    mf.conn.execute("UPDATE history SET detail='{\"tampered\":true}' WHERE id=?", (mid,))
    mf.conn.commit()
    rep = audit.verify_history(mf)
    assert rep["status"] == "corrupted"
    assert any(b["issue"] == "content_tampered" for b in rep["broken"])
    mf.close()


def test_deleting_an_entry_breaks_the_chain(mind_path):
    mf = _active(mind_path)
    ids = [r["id"] for r in mf.conn.execute("SELECT id FROM history ORDER BY id")]
    mf.conn.execute("DELETE FROM history WHERE id=?", (ids[0],))   # remove genesis-linked entry
    mf.conn.commit()
    rep = audit.verify_history(mf)
    assert rep["status"] == "corrupted"
    assert any(b["issue"] == "chain_break" for b in rep["broken"])
    mf.close()


def test_audit_includes_history(mind_path):
    mf = _active(mind_path)
    assert audit.audit(mf)["status"] == "ok"
    mf.conn.execute("UPDATE history SET event='forged' WHERE id="
                    "(SELECT id FROM history ORDER BY id LIMIT 1)")
    mf.conn.commit()
    rep = audit.audit(mf)
    assert rep["status"] == "corrupted" and "history" in rep["corrupted"]
    mf.close()
