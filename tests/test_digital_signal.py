import numpy as np

from mindfile import MindFile, audit, checkpoint, signal


def _digital_recording():
    rng = np.random.default_rng(3)
    # smooth-ish int16 digital samples, 4 channels x 2000 samples
    dig = np.cumsum(rng.integers(-8, 9, size=(4, 2000)), axis=1).astype(np.int16)
    gain = np.array([0.25e-6, 0.5e-6, 0.25e-6, 1.0e-6])       # digital->SI volts
    offset = np.array([0.0, 1e-6, -2e-6, 0.0])
    cal = {"gain": gain.tolist(), "offset": offset.tolist(), "unit": "V"}
    return dig, gain, offset, cal


def test_digital_roundtrip_and_physical(mind_path):
    dig, gain, offset, cal = _digital_recording()
    mf = MindFile.create(mind_path, modality="eeg")
    iid, info = signal.add_recording(mf, dig, sfreq=250.0, ch_names=list("ABCD"),
                                     calibration=cal)
    data, sfreq, ch = signal.load_signal(mf, iid, verify=True)
    # physical reconstruction is exact
    assert np.allclose(data, dig.astype(np.float64) * gain[:, None] + offset[:, None])
    assert sfreq == 250.0 and ch == list("ABCD")
    # int16 stored, not float32: ~2x smaller raw + it still compresses
    assert info["raw_bytes"] == dig.size * 2 and info["ratio"] > 1.0
    assert signal.verify_signals(mf)["status"] == "ok"
    mf.close()


def test_calibration_is_integrity_covered(mind_path):
    dig, *_1, cal = _digital_recording()
    mf = MindFile.create(mind_path, modality="eeg")
    iid, _ = signal.add_recording(mf, dig, sfreq=250.0, calibration=cal)
    checkpoint.create_checkpoint(mf, "cp")
    assert audit.audit(mf, deep=True)["status"] == "ok"

    # tamper ONLY the embedded calibration (flip a byte inside `code`); because
    # calibration lives in the hashed code blob, both verifiers must catch it.
    code = bytearray(mf.conn.execute("SELECT code FROM signals WHERE item_id=?",
                                     (iid,)).fetchone()["code"])
    code[10] ^= 0x01
    mf.conn.execute("UPDATE signals SET code=? WHERE item_id=?", (bytes(code), iid))
    mf.conn.commit()
    assert signal.verify_signals(mf)["status"] == "corrupted"
    assert checkpoint.verify_checkpoint(mf, "cp")["status"] != "ok"
    mf.close()
