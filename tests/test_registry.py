import numpy as np
import pytest

from mindfile import MindFile, ingest, cluster, sign, registry


def _patient(path, sign_it=True):
    mf = MindFile.create(path, modality="eeg", overwrite=True)
    ingest.add_vectors(mf, np.random.default_rng(0).normal(0, 1, (8, 5)).astype("float32"))
    cluster.run_clustering(mf, k=2)
    key_id = None
    if sign_it:
        info = sign.sign_state(mf, sign.Ed25519Signer())
        key_id = info["key_id"]
    mf.close()
    return key_id


def _registry_over(tmp_path, patient_name="patient.mind", **link_kw):
    patient = tmp_path / patient_name
    reg = MindFile.create(tmp_path / "study.mind", modality="study")
    registry.link_file(reg, patient_name, role="patient", **link_kw)   # relative path
    return reg


def test_verify_ok_for_unchanged_signed_file(tmp_path):
    _patient(tmp_path / "patient.mind")
    reg = _registry_over(tmp_path)
    rep = registry.verify_refs(reg)
    assert rep["status"] == "ok" and rep["refs"][0]["status"] == "ok"
    reg.close()


def test_detects_state_drift(tmp_path):
    _patient(tmp_path / "patient.mind")
    reg = _registry_over(tmp_path)
    # patient does more legitimate work -> its state_root advances
    p = MindFile.open(tmp_path / "patient.mind")
    ingest.add_vectors(p, np.random.default_rng(1).normal(0, 1, (3, 5)).astype("float32"))
    p.close()
    assert registry.verify_refs(reg)["refs"][0]["status"] == "state_changed"
    reg.close()


def test_detects_substitution(tmp_path):
    _patient(tmp_path / "patient.mind")
    reg = _registry_over(tmp_path)
    _patient(tmp_path / "patient.mind")          # different file (new uuid) at same path
    assert registry.verify_refs(reg)["refs"][0]["status"] == "uuid_mismatch"
    reg.close()


def test_detects_missing(tmp_path):
    _patient(tmp_path / "patient.mind")
    reg = _registry_over(tmp_path)
    (tmp_path / "patient.mind").unlink()
    assert registry.verify_refs(reg)["refs"][0]["status"] == "missing"
    reg.close()


def test_detects_signature_tamper_when_key_pinned(tmp_path):
    pytest.importorskip("cryptography")
    _patient(tmp_path / "patient.mind")
    reg = _registry_over(tmp_path)               # pins the patient's signing key
    # forge the patient's signature (state_root unchanged, but the sig is now invalid)
    p = MindFile.open(tmp_path / "patient.mind")
    p.conn.execute("UPDATE signatures SET signature=? WHERE id=1", (b"\x00" * 64,))
    p.conn.commit(); p.close()
    assert registry.verify_refs(reg)["refs"][0]["status"] == "signature_invalid"
    reg.close()


def test_require_signed_rejects_unsigned(tmp_path):
    _patient(tmp_path / "patient.mind", sign_it=False)
    reg = MindFile.create(tmp_path / "study.mind", modality="study")
    with pytest.raises(ValueError):
        registry.link_file(reg, "patient.mind", require_signed=True)
    reg.close()
