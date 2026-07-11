"""End-to-end multimodal integration: every subsystem composing on one cohort."""

import numpy as np

from mindfile import (MindFile, modality, lazy, fuse, cluster, checkpoint, sign,
                      audit)

SPACES = ["bandpower", "ecg_bandpower", "radiomics", "text", "variants"]
NOTES = {0: "ischemic stroke aphasia", 1: "healthy control normal"}
VARIANTS = {0: ["1:100:A:G", "7:55:G:A"], 1: ["2:200:C:T", "9:88:T:A"]}


def _cohort(mind_path, n=4):
    mf = MindFile.create(mind_path, modality="multimodal")
    rng = np.random.default_rng(0)
    for i in range(n):
        pid, sub = f"P{i}", i % 2
        s = sub * 3.0
        modality.ingest(mf, "eeg", rng.normal(s, 0.3, (4, 500)).astype("float32"),
                        sfreq=250.0, subject_id=pid, session_id="t0")
        modality.ingest(mf, "ecg", rng.normal(s, 0.3, (1, 500)).astype("float32"),
                        sfreq=250.0, subject_id=pid, session_id="t0")
        modality.ingest(mf, "mri", rng.normal(s, 0.2, (6, 6, 6)).astype("float32"),
                        subject_id=pid, session_id="t0")
        modality.ingest(mf, "text", NOTES[sub], subject_id=pid, session_id="t0")
        modality.ingest(mf, "genomics", VARIANTS[sub], subject_id=pid, session_id="t0")
    return mf


def test_multimodal_fusion_and_subtyping(mind_path):
    mf = _cohort(mind_path, n=4)
    for sp in ("bandpower", "ecg_bandpower"):
        lazy.materialize_space(mf, sp)

    res = fuse.fuse(mf, SPACES, key=("subject_id", "session_id"), out_space="patient")
    assert len(res["created"]) == 4 and not res["skipped"]   # one unified vector per patient

    cid = cluster.run_clustering(mf, "patient", k=2)
    asg = cluster.assignments(mf, cid)
    sub_of = {f"P{i}": i % 2 for i in range(4)}
    groups = {}
    for item_id, cl in asg.items():
        pid = mf.conn.execute("SELECT external_id FROM items WHERE id=?", (item_id,)).fetchone()[0].split(":")[1]
        groups.setdefault(cl, []).append(sub_of[pid])
    # each cluster is a single subtype -> fusion carried cross-modal signal
    assert all(len(set(s)) == 1 for s in groups.values())
    mf.close()


def test_full_stack_integrity_holds(mind_path):
    mf = _cohort(mind_path, n=4)
    for sp in ("bandpower", "ecg_bandpower"):
        lazy.materialize_space(mf, sp)
    fuse.fuse(mf, SPACES, key=("subject_id", "session_id"), out_space="patient")
    cluster.run_clustering(mf, "patient", k=2)

    checkpoint.create_checkpoint(mf, "baseline")
    signer = sign.Ed25519Signer()
    sign.sign_state(mf, signer)
    sign.trust_key(mf, signer.public_key)

    rep = audit.audit(mf, deep=True)               # every layer, including deep resolve
    assert rep["status"] == "ok" and not rep["corrupted"]
    assert sign.verify_signatures(mf)["trusted_attested"] is True
    assert fuse.verify_derivations(mf)["status"] == "ok"   # fused patients backed by intact sources
    assert checkpoint.verify_checkpoint(mf, "baseline")["status"] == "ok"
    mf.close()
