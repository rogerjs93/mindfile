import numpy as np
import pytest

from mindfile import MindFile, ingest, fuse, lazy, checkpoint, spaces


def _fused(mind_path):
    mf = MindFile.create(mind_path, modality="eeg")
    rng = np.random.default_rng(0)
    for i, p in enumerate(["P0", "P1", "P2"]):
        ingest.add_vectors(mf, rng.normal(i, 0.3, (1, 5)).astype("float32"), space="bandpower",
                           subject_ids=p, session_ids="t0", modalities="eeg", external_ids=f"{p}-eeg")
        ingest.add_vectors(mf, rng.normal(i, 0.3, (1, 8)).astype("float32"), space="mri",
                           subject_ids=p, session_ids="t0", modalities="mri", external_ids=f"{p}-mri")
    fuse.fuse(mf, ["bandpower", "mri"])          # eager, materialized
    return mf


def test_compaction_is_lossless_and_reconstructs_exactly(mind_path):
    mf = _fused(mind_path)
    fsid = spaces.get_space_id(mf, "fused")
    before = {r["item_id"]: bytes(r["embedding"]) for r in mf.conn.execute(
        "SELECT item_id, embedding FROM vectors WHERE space_id=?", (fsid,))}

    n = lazy.compact_space(mf, "fused")
    assert n == 3
    # raw rows gone, codes stored, vectors now virtual
    assert mf.conn.execute("SELECT COUNT(*) FROM vectors WHERE space_id=?", (fsid,)).fetchone()[0] == 0
    assert mf.conn.execute("SELECT COUNT(*) FROM vector_codes WHERE space_id=?", (fsid,)).fetchone()[0] == 3
    for item_id, raw in before.items():
        assert lazy.is_virtual(mf, item_id, "fused")
        # byte-for-byte reconstruction
        assert lazy.resolve(mf, item_id, "fused").astype("<f4").tobytes() == raw
    assert lazy.verify_defs(mf)["status"] == "ok"
    mf.close()


def test_compaction_refuses_primary_space(mind_path):
    mf = _fused(mind_path)
    with pytest.raises(ValueError):
        lazy.compact_space(mf, "bandpower")      # primary -> refused (would touch originals)
    # nothing happened to primaries
    assert mf.conn.execute(
        "SELECT COUNT(*) FROM vectors v JOIN spaces s ON s.id=v.space_id "
        "WHERE s.name='bandpower'").fetchone()[0] == 3
    mf.close()


def test_compact_then_materialize_restores_bytes(mind_path):
    mf = _fused(mind_path)
    item = mf.conn.execute("SELECT id FROM items WHERE origin='derived' LIMIT 1").fetchone()["id"]
    original = lazy.resolve(mf, item, "fused").astype("<f4").tobytes()
    lazy.compact_space(mf, "fused")
    lazy.materialize(mf, item, "fused")
    row = mf.conn.execute(
        "SELECT embedding FROM vectors v JOIN spaces s ON s.id=v.space_id "
        "WHERE v.item_id=? AND s.name='fused'", (item,)).fetchone()
    assert bytes(row["embedding"]) == original
    mf.close()
