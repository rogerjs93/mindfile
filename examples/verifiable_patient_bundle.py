"""FLAGSHIP: a Verifiable Patient Bundle.

The differentiated, proven strength of a `.gnos` file is *integrity + provenance*:
one self-contained, signed, LLM-readable multimodal record that detects any
tampering. This demo proves it end to end:

  1. build a multimodal patient bundle (EEG waveform + labs + FHIR + note + DICOM
     + genomics), all under one identity
  2. SEAL it: content-hashed checkpoint + Ed25519 signature over the whole state,
     with the signing key pinned as trusted
  3. show the LLM-readable view (what an AI reads) and the authenticity summary
  4. run an ADVERSARY SUITE: for each realistic attack (payload edit, hash-forged
     edit, waveform bit-flip, vector edit, history deletion, forged signature,
     file substitution) show it is CAUGHT and LOCALIZED by one call to audit()
  5. show that *legitimate* growth is NOT flagged as tampering
  6. confirm the pristine bundle still audits clean

Portable: uses only synthetic-but-realistic data, so it runs anywhere in seconds.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import (MindFile, audit, checkpoint, lazy, modality, registry,
                      render, sign)

TMP = Path(tempfile.gettempdir())
SUBJECT = "PT-001"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_bundle(path: Path, *, subject=SUBJECT, seed=0) -> None:
    """A compact but realistic multimodal patient in one .gnos."""
    rng = np.random.default_rng(seed)
    mf = MindFile.create(path, modality="patient", overwrite=True)
    kw = dict(subject_id=subject, session_id="visit1")
    modality.ingest(mf, "eeg", rng.normal(size=(8, 1500)).astype("float32"),
                    sfreq=250.0, ch_names=["FP1", "FP2", "C3", "C4", "P3", "P4", "O1", "O2"], **kw)
    modality.ingest(mf, "labs",
                    {"crp": 9.1, "wbc": 12.0, "sodium": 139.0, "glucose": 142.0, "creatinine": 1.0}, **kw)
    modality.ingest(mf, "fhir", {"resourceType": "Condition", "subject": {"reference": f"Patient/{subject}"},
                                 "code": {"text": "Ischemic stroke, left pons"}}, **kw)
    modality.ingest(mf, "text", "68yo male, acute ischemic stroke, right hemiparesis; NIHSS 8.", **kw)
    modality.ingest(mf, "dicom", {"PatientID": subject, "StudyInstanceUID": "S1", "Modality": "CT",
                                  "BodyPartExamined": "HEAD", "Rows": 512, "Columns": 512,
                                  "SeriesDescription": "Axial 5mm", "StudyDate": "20240103"}, **kw)
    modality.ingest(mf, "genomics", ["19:45411941:T:C", "20:14370:G:A"], **kw)
    for (sp,) in mf.conn.execute("SELECT name FROM spaces"):
        lazy.materialize_space(mf, sp)
    mf.close()


def seal(path: Path):
    """Content-hash checkpoint + Ed25519 sign the state + pin the key as trusted."""
    mf = MindFile.open(path)
    checkpoint.create_checkpoint(mf, "issued")
    signer = sign.Ed25519Signer()
    sign.sign_state(mf, signer)
    sign.trust_key(mf, signer.public_key, label="issuing hospital")
    mf.close()
    return signer


# -- adversary helpers ------------------------------------------------------
def _audit_copy(src: Path, mutate) -> tuple[bool, str]:
    """Copy the sealed file, apply a raw-SQL mutation, then re-audit it."""
    dst = TMP / "attacked.gnos"
    shutil.copy(src, dst)
    con = sqlite3.connect(dst)
    con.row_factory = sqlite3.Row
    mutate(con)
    con.commit()
    con.close()
    mf = MindFile.open(dst, readonly=True)
    rep = audit.audit(mf, deep=True)
    mf.close()
    caught = rep["status"] != "ok"
    where = ", ".join(rep["corrupted"]) if rep["corrupted"] else rep["status"]
    return caught, where


def _flip_blob(con, table, col, where=""):
    r = con.execute(f"SELECT rowid AS rid, {col} FROM {table} {where} LIMIT 1").fetchone()
    b = bytearray(r[col] if isinstance(r[col], (bytes, bytearray)) else str(r[col]).encode())
    b[len(b) // 2] ^= 0x01
    con.execute(f"UPDATE {table} SET {col}=? WHERE rowid=?", (bytes(b), r["rid"]))
    return bytes(b)


def main():
    patient = TMP / "verifiable_patient.gnos"
    build_bundle(patient)
    signer = seal(patient)

    mf = MindFile.open(patient, readonly=True)
    counts = mf.counts()
    nsp = mf.conn.execute("SELECT COUNT(*) FROM spaces").fetchone()[0]
    root = sign.state_root(mf)
    vs = sign.verify_signatures(mf)
    print("=" * 74)
    print(f"VERIFIABLE PATIENT BUNDLE  ({patient.name})")
    print("=" * 74)
    print(f"[1] one self-contained file: {counts['items']} records across {nsp} embedding "
          f"spaces, {counts['signals']} lossless waveform(s)")
    print(f"[2] authenticity: signed by key {signer.key_id[:16]}.. | trusted={vs['trusted_attested']} "
          f"| matches current state={vs['signatures'][0]['matches_current']}")
    print(f"    state_root = {root[:32]}..  (one hash pinning the ENTIRE bundle)")
    print("[3] what an LLM reads (raw bytes stay stored + verifiable):")
    for line in render.digest(mf, subject_id=SUBJECT).splitlines()[:9]:
        print(f"      {line}")
    mf.close()

    print("\n[4] ADVERSARY SUITE - each attack on a copy, caught by one audit() call:")
    attacks = [
        ("edit a lab/report payload byte",
         lambda c: _flip_blob(c, "items", "payload", "WHERE payload IS NOT NULL")),
        ("edit a payload AND recompute its inline hash (forge)",
         _forge_payload),
        ("flip one bit inside a stored EEG waveform",
         lambda c: _flip_blob(c, "signals", "code")),
        ("tamper an embedding vector",
         lambda c: _flip_blob(c, "vectors", "embedding")),
        ("delete a provenance (history) row",
         lambda c: c.execute("DELETE FROM history WHERE id IN "
                             "(SELECT id FROM history ORDER BY id LIMIT 1 OFFSET 2)")),
        ("forge / corrupt the signature",
         lambda c: _flip_blob(c, "signatures", "signature")),
    ]
    all_caught = True
    for label, mutate in attacks:
        caught, where = _audit_copy(patient, mutate)
        all_caught &= caught
        print(f"    {'CAUGHT ' if caught else 'MISSED!'} | {label:52} -> {where}")

    print("\n[5] legitimate growth is NOT tampering (add a follow-up record):")
    grown = TMP / "verifiable_grown.gnos"
    shutil.copy(patient, grown)
    g = MindFile.open(grown)
    modality.ingest(g, "text", "90-day follow-up: improving, NIHSS 4, independent ambulation.",
                    subject_id=SUBJECT, session_id="visit2")
    rep = audit.audit(g, deep=True)
    vs2 = sign.verify_signatures(g)
    g.close()
    print(f"    tamper check: corrupted = {rep['corrupted']} (empty - growth is NOT corruption)")
    print(f"    signature still cryptographically valid = {vs2['signatures'][0]['valid']}  |  "
          f"still matches the SEALED state = {vs2['signatures'][0]['matches_current']} "
          "(so any change since issuance is visible)")
    print("    -> authorised growth is recorded in the hash-chained history and can be re-sealed;")
    print("       a raw edit breaks the seal with no valid history event. The file tells them apart.")

    print("\n[6] CROSS-FILE PROVENANCE - a study that references this patient file:")
    _cross_file_scene(patient)

    print("\n[7] the pristine bundle still verifies clean:")
    mf = MindFile.open(patient, readonly=True)
    rep = audit.audit(mf, deep=True)
    mf.close()
    print(f"    deep audit = {rep['status']}  corrupted={rep['corrupted']}")
    print("\n" + ("ALL ATTACKS CAUGHT." if all_caught else "!! an attack slipped through !!"))


def _forge_payload(con):
    r = con.execute("SELECT id, payload FROM items WHERE payload IS NOT NULL LIMIT 1").fetchone()
    p = r["payload"] if isinstance(r["payload"], (bytes, bytearray)) else str(r["payload"]).encode()
    b = bytearray(p); b[len(b) // 2] ^= 0x01
    # attacker also updates the inline content hash so live-integrity would pass...
    con.execute("UPDATE items SET payload=?, payload_sha256=? WHERE id=?",
                (bytes(b), _sha(bytes(b)), r["id"]))
    # ...but the signed checkpoint froze the original hash, so it is still caught.


def _cross_file_scene(patient: Path):
    study = TMP / "study_registry.gnos"
    sm = MindFile.create(study, modality="study", overwrite=True)
    registry.link_file(sm, str(patient), role="enrolled patient")     # pins uuid + state_root + key
    print(f"    linked patient -> verify_refs: {registry.verify_refs(sm)['refs'][0]['status']}")

    backup = TMP / "patient_backup.gnos"
    shutil.copy(patient, backup)                                       # keep the real one
    other = TMP / "other_patient.gnos"
    build_bundle(other, subject="PT-999", seed=99)
    seal(other)
    shutil.copy(other, patient)                                        # substitute a different file
    print(f"    swap in a different file -> verify_refs: {registry.verify_refs(sm)['refs'][0]['status']}")
    shutil.copy(backup, patient)                                       # restore for step [7]
    sm.close()


if __name__ == "__main__":
    main()
