import numpy as np
import pytest

from mindfile import MindFile, ingest, fuse, signal, checkpoint, audit
from mindfile._vec import sha256_bytes


def _full(mind_path):
    """A file exercising every integrity surface: vectors, lazy fusion, a stored
    waveform, and a save point."""
    mf = MindFile.create(mind_path, modality="eeg")
    rng = np.random.default_rng(0)
    eeg = []
    for i, p in enumerate(["P0", "P1", "P2"]):
        eeg += ingest.add_vectors(mf, rng.normal(i, 0.3, (1, 5)).astype("float32"), space="bandpower",
                                  subject_ids=p, session_ids="t0", modalities="eeg", external_ids=f"{p}-eeg")
        ingest.add_vectors(mf, rng.normal(i, 0.3, (1, 8)).astype("float32"), space="mri",
                           subject_ids=p, session_ids="t0", modalities="mri", external_ids=f"{p}-mri")
    fuse.fuse(mf, ["bandpower", "mri"], lazy=True)
    signal.store_signal(mf, eeg[0], rng.normal(0, 1, (8, 1000)).astype("float32"), sfreq=500.0)
    checkpoint.create_checkpoint(mf, "baseline")
    return mf, eeg


def test_audit_ok_on_sound_file(mind_path):
    mf, _ = _full(mind_path)
    rep = audit.audit(mf, deep=True)
    assert rep["status"] == "ok" and not rep["corrupted"]
    mf.close()


def test_audit_catches_vector_tamper(mind_path):
    mf, _ = _full(mind_path)
    blob = mf.conn.execute("SELECT embedding FROM vectors WHERE item_id=1 AND space_id="
                           "(SELECT id FROM spaces WHERE name='bandpower')").fetchone()[0]
    arr = np.frombuffer(blob, dtype="<f4").copy(); arr[0] += 9.0
    mf.conn.execute("UPDATE vectors SET embedding=? WHERE item_id=1 AND space_id="
                    "(SELECT id FROM spaces WHERE name='bandpower')", (arr.tobytes(),))
    mf.conn.commit()
    rep = audit.audit(mf)
    assert rep["status"] == "corrupted"
    # the same source feeds the save point, the fused recipe and live integrity
    assert {"integrity", "defs", "checkpoint:baseline"} <= set(rep["corrupted"])
    mf.close()


def test_audit_catches_waveform_tamper(mind_path):
    mf, eeg = _full(mind_path)
    code = mf.conn.execute("SELECT code FROM signals WHERE item_id=?", (eeg[0],)).fetchone()[0]
    mf.conn.execute("UPDATE signals SET code=? WHERE item_id=?", (code + b"\x01", eeg[0]))
    mf.conn.commit()
    rep = audit.audit(mf)
    assert rep["status"] == "corrupted" and "signals" in rep["corrupted"]
    mf.close()
