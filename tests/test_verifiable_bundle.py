"""Regression-protects the flagship's core claim: every realistic tamper class
is caught by one audit() call, and legitimate growth is NOT flagged as tampering."""

import hashlib
import shutil
import sqlite3

import numpy as np
import pytest

from mindfile import MindFile, audit, checkpoint, lazy, modality, sign


def _sealed_bundle(path):
    mf = MindFile.create(path, modality="patient", overwrite=True)
    kw = dict(subject_id="PT-001", session_id="v1")
    modality.ingest(mf, "eeg", np.random.default_rng(0).normal(size=(6, 800)).astype("float32"),
                    sfreq=200.0, ch_names=["FP1", "FP2", "C3", "C4", "O1", "O2"], **kw)
    modality.ingest(mf, "labs", {"crp": 9.0, "wbc": 12.0}, **kw)
    modality.ingest(mf, "text", "acute stroke, right hemiparesis", **kw)
    for (sp,) in mf.conn.execute("SELECT name FROM spaces"):
        lazy.materialize_space(mf, sp)
    checkpoint.create_checkpoint(mf, "issued")
    sign.sign_state(mf, sign.Ed25519Signer())
    mf.close()


def _audit_after(src, tmp_path, mutate):
    dst = tmp_path / "attacked.gnos"
    shutil.copy(src, dst)
    con = sqlite3.connect(dst); con.row_factory = sqlite3.Row
    mutate(con); con.commit(); con.close()
    mf = MindFile.open(dst, readonly=True)
    rep = audit.audit(mf, deep=True)
    mf.close()
    return rep


def _flip(con, table, col, where=""):
    r = con.execute(f"SELECT rowid AS rid, {col} FROM {table} {where} LIMIT 1").fetchone()
    b = bytearray(r[col] if isinstance(r[col], (bytes, bytearray)) else str(r[col]).encode())
    b[len(b) // 2] ^= 0x01
    con.execute(f"UPDATE {table} SET {col}=? WHERE rowid=?", (bytes(b), r["rid"]))


ATTACKS = {
    "payload": (lambda c: _flip(c, "items", "payload", "WHERE payload IS NOT NULL"), "integrity"),
    "waveform": (lambda c: _flip(c, "signals", "code"), "signals"),
    "vector": (lambda c: _flip(c, "vectors", "embedding"), "integrity"),
    "history": (lambda c: c.execute("DELETE FROM history WHERE id IN "
                "(SELECT id FROM history ORDER BY id LIMIT 1 OFFSET 2)"), "history"),
    "signature": (lambda c: _flip(c, "signatures", "signature"), "signatures"),
}


@pytest.mark.parametrize("name", list(ATTACKS))
def test_each_tamper_class_is_caught_and_localized(tmp_path, name):
    src = tmp_path / "bundle.gnos"
    _sealed_bundle(src)
    mutate, expect = ATTACKS[name]
    rep = _audit_after(src, tmp_path, mutate)
    assert rep["status"] == "corrupted", f"{name} not caught"
    assert expect in rep["corrupted"], f"{name}: expected {expect} in {rep['corrupted']}"


def test_forged_inline_hash_still_caught_by_signed_checkpoint(tmp_path):
    # attacker edits a payload AND recomputes payload_sha256 so live integrity passes;
    # the signed checkpoint froze the original hash, so it is still caught.
    src = tmp_path / "bundle.gnos"
    _sealed_bundle(src)

    def forge(con):
        r = con.execute("SELECT id, payload FROM items WHERE payload IS NOT NULL LIMIT 1").fetchone()
        p = r["payload"] if isinstance(r["payload"], (bytes, bytearray)) else str(r["payload"]).encode()
        b = bytearray(p); b[len(b) // 2] ^= 0x01
        con.execute("UPDATE items SET payload=?, payload_sha256=? WHERE id=?",
                    (bytes(b), hashlib.sha256(bytes(b)).hexdigest(), r["id"]))

    rep = _audit_after(src, tmp_path, forge)
    assert rep["status"] == "corrupted"
    assert any(c.startswith("checkpoint:") for c in rep["corrupted"])


def test_pristine_bundle_audits_clean(tmp_path):
    src = tmp_path / "bundle.gnos"
    _sealed_bundle(src)
    mf = MindFile.open(src, readonly=True)
    rep = audit.audit(mf, deep=True)
    sigs = sign.verify_signatures(mf)
    mf.close()
    assert rep["status"] == "ok" and rep["corrupted"] == []
    assert sigs["signatures"][0]["valid"] and sigs["signatures"][0]["matches_current"]


def test_authorized_growth_is_not_corruption(tmp_path):
    src = tmp_path / "bundle.gnos"
    _sealed_bundle(src)
    mf = MindFile.open(src)
    modality.ingest(mf, "text", "follow-up note: improving", subject_id="PT-001", session_id="v2")
    rep = audit.audit(mf, deep=True)
    sigs = sign.verify_signatures(mf)
    mf.close()
    assert rep["corrupted"] == []                       # growth is NOT tampering
    assert sigs["signatures"][0]["valid"] is True        # signature still valid
    assert sigs["signatures"][0]["matches_current"] is False  # but state moved past the seal
