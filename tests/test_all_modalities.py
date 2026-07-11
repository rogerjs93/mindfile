"""Consolidation test: every registered file type ingests, renders a meaningful
LLM card, and composes into one integrity-sealed .gnos."""

import numpy as np
import pytest

from mindfile import MindFile, audit, checkpoint, lazy, modality, render, sign


def _samples():
    rng = np.random.default_rng(0)
    return {
        "eeg": (rng.normal(size=(8, 2000)).astype("float32"),
                dict(sfreq=250.0, ch_names=["FP1", "FP2", "C3", "C4", "P3", "P4", "O1", "O2"])),
        "ecg": (rng.normal(size=(2, 2000)).astype("float32"), dict(sfreq=250.0, ch_names=["II", "V"])),
        "wfdb": (rng.normal(size=(2, 1500)).astype("float32"), dict(sfreq=250.0, ch_names=["II", "PLETH"])),
        "mri": (rng.random((16, 16, 16)).astype("float32"), {}),
        "dicom": ({"PatientID": "P1", "StudyInstanceUID": "S1", "Modality": "CT",
                   "BodyPartExamined": "HEAD", "Rows": 512, "Columns": 512,
                   "SeriesDescription": "Axial 5mm", "StudyDate": "20240103"}, {}),
        "fhir": ({"resourceType": "Observation", "subject": {"reference": "Patient/P1"},
                  "code": {"text": "C-reactive protein"},
                  "valueQuantity": {"value": 12, "unit": "mg/L"}}, {}),
        "text": ("Patient reports chest pain.", {}),
        "genomics": (["chr1:123:A>G", "chr7:55:C>T"], {}),
        "labs": ({"crp": 12.0, "sodium": 140.0}, {}),
        "tabular": ({"age": 64.0, "bmi": 27.5}, {}),
    }


def test_every_registered_modality_has_metadata_and_card():
    cat = {r["modality"]: r for r in modality.catalog()}
    from mindfile.render import _RENDERERS
    # registry, catalog, and renderers agree — no orphans in either direction
    assert set(cat) == set(modality.processors())
    for m, r in cat.items():
        assert r["card"] and m in _RENDERERS, f"{m} has no card renderer"
        assert r["spaces"] and r["extensions"], f"{m} missing spaces/extensions metadata"


def test_all_modalities_compose_in_one_file(mind_path):
    samples = _samples()
    assert set(samples) == set(modality.processors())      # covers EVERY registered type

    mf = MindFile.create(mind_path, modality="multi")
    items = {}
    for mod, (source, kw) in samples.items():
        items[mod] = modality.ingest(mf, mod, source, subject_id="P1",
                                     session_id="v1", **kw)

    for (sp,) in mf.conn.execute("SELECT name FROM spaces"):
        lazy.materialize_space(mf, sp)

    for mod, iid in items.items():
        c = render.card(mf, iid).lower()
        assert f"[{mod}]" in c and len(c) > 12             # a real, modality-tagged card
        assert "unknown" not in c

    checkpoint.create_checkpoint(mf, "all")
    signer = sign.Ed25519Signer()
    sign.sign_state(mf, signer)
    sign.trust_key(mf, signer.public_key)
    rep = audit.audit(mf, deep=True)
    assert rep["status"] == "ok" and rep["corrupted"] == []
    assert sign.verify_signatures(mf)["trusted_attested"] is True
    mf.close()
