import numpy as np
import pytest

from mindfile import MindFile, ingest, cluster, sign, audit


def _signed_file(mind_path, signer):
    mf = MindFile.create(mind_path, modality="eeg")
    ingest.add_vectors(mf, np.random.default_rng(0).normal(0, 1, (8, 5)).astype("float32"))
    cluster.run_clustering(mf, k=2)
    info = sign.sign_state(mf, signer)
    return mf, info


def test_hmac_roundtrip(mind_path):
    secret = b"shared-secret"
    mf, info = _signed_file(mind_path, sign.HmacSigner(secret))
    rep = sign.verify_signatures(mf, hmac_secret=secret)
    s = rep["signatures"][0]
    assert s["valid"] is True and s["matches_current"] is True and rep["status"] == "ok"
    # wrong secret -> invalid
    bad = sign.verify_signatures(mf, hmac_secret=b"wrong")
    assert bad["signatures"][0]["valid"] is False and bad["status"] == "corrupted"
    mf.close()


def test_ed25519_public_verification(mind_path):
    pytest.importorskip("cryptography")
    signer = sign.Ed25519Signer()
    mf, info = _signed_file(mind_path, signer)
    # Verifiable with NO secret (public key is stored in the file)
    rep = sign.verify_signatures(mf)
    s = rep["signatures"][0]
    assert s["valid"] is True and s["matches_current"] is True
    assert audit.audit(mf)["status"] == "ok"
    mf.close()


def test_signature_survives_growth_but_flags_state_advanced(mind_path):
    pytest.importorskip("cryptography")
    mf, _ = _signed_file(mind_path, sign.Ed25519Signer())
    # legitimate new work advances the (hash-chained) state
    ingest.add_vectors(mf, np.random.default_rng(1).normal(0, 1, (3, 5)).astype("float32"))
    rep = sign.verify_signatures(mf)
    s = rep["signatures"][0]
    assert s["valid"] is True            # still authentic for what it signed
    assert s["matches_current"] is False  # but the file has moved on since
    assert rep["status"] == "ok"         # not corrupted — just advanced
    mf.close()


def test_forged_signature_is_caught(mind_path):
    pytest.importorskip("cryptography")
    mf, _ = _signed_file(mind_path, sign.Ed25519Signer())
    mf.conn.execute("UPDATE signatures SET signature=? WHERE id=1", (b"\x00" * 64,))
    mf.conn.commit()
    rep = sign.verify_signatures(mf)
    assert rep["signatures"][0]["valid"] is False and rep["status"] == "corrupted"
    assert "signatures" in audit.audit(mf)["corrupted"]
    mf.close()
