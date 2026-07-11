"""A save point now pins the WHOLE file: waveforms and virtual vectors too."""

import numpy as np

from mindfile import MindFile, ingest, fuse, signal, checkpoint


def _file(mind_path):
    mf = MindFile.create(mind_path, modality="eeg")
    rng = np.random.default_rng(0)
    eeg = []
    for i, p in enumerate(["P0", "P1", "P2"]):
        eeg += ingest.add_vectors(mf, rng.normal(i, 0.3, (1, 5)).astype("float32"), space="bandpower",
                                  subject_ids=p, session_ids="t0", modalities="eeg", external_ids=f"{p}-eeg")
        ingest.add_vectors(mf, rng.normal(i, 0.3, (1, 8)).astype("float32"), space="mri",
                           subject_ids=p, session_ids="t0", modalities="mri", external_ids=f"{p}-mri")
    fuse.fuse(mf, ["bandpower", "mri"], lazy=True)        # virtual fused vectors
    signal.store_signal(mf, eeg[0], rng.normal(0, 1, (8, 1000)).astype("float32"), sfreq=500.0)
    return mf, eeg


def test_save_point_pins_signals_and_defs(mind_path):
    mf, eeg = _file(mind_path)
    cp = checkpoint.create_checkpoint(mf, "complete")
    assert cp["signal_count"] == 1 and cp["def_count"] == 3   # waveform + 3 lazy fused vectors
    assert checkpoint.verify_checkpoint(mf, "complete")["status"] == "ok"
    mf.close()


def test_save_point_detects_waveform_substitution(mind_path):
    mf, eeg = _file(mind_path)
    checkpoint.create_checkpoint(mf, "complete")
    # swap the waveform for a DIFFERENT valid one (hashes updated) -> previously
    # undetectable by a save point; now caught.
    signal.store_signal(mf, eeg[0], np.random.default_rng(9).normal(0, 1, (8, 1000)).astype("float32"),
                        sfreq=500.0)
    rep = checkpoint.verify_checkpoint(mf, "complete")
    assert rep["status"] == "corrupted"
    assert any(s["item_id"] == eeg[0] for s in rep["changed_signals"])
    mf.close()


def test_save_point_detects_virtual_vector_drift(mind_path):
    mf, _ = _file(mind_path)
    checkpoint.create_checkpoint(mf, "complete")
    # tamper a source vector that a virtual fused vector depends on
    blob = mf.conn.execute("SELECT embedding FROM vectors WHERE item_id=1 AND space_id="
                           "(SELECT id FROM spaces WHERE name='bandpower')").fetchone()[0]
    arr = np.frombuffer(blob, dtype="<f4").copy(); arr[0] += 7.0
    mf.conn.execute("UPDATE vectors SET embedding=? WHERE item_id=1 AND space_id="
                    "(SELECT id FROM spaces WHERE name='bandpower')", (arr.tobytes(),))
    mf.conn.commit()
    rep = checkpoint.verify_checkpoint(mf, "complete")
    assert rep["status"] == "corrupted"
    assert rep["changed_defs"]            # the fused recipe's source drifted
    mf.close()
