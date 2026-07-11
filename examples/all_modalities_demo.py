"""Consolidation demo: ingest ONE of every registered file type into a single
.gnos, render each as an LLM card, discover findings, and seal + deep-audit —
proving the whole multimodal surface composes in one live, verifiable file.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import MindFile, audit, checkpoint, findings, lazy, modality, render, sign


def samples(rng):
    """One minimal, representative source per registered modality."""
    return {
        "eeg": (rng.normal(size=(8, 2000)).astype("float32"),
                dict(sfreq=250.0, ch_names=["FP1", "FP2", "C3", "C4", "P3", "P4", "O1", "O2"])),
        "ecg": (rng.normal(size=(2, 2000)).astype("float32"),
                dict(sfreq=250.0, ch_names=["II", "V"])),
        "wfdb": (rng.normal(size=(2, 1500)).astype("float32"),
                 dict(sfreq=250.0, ch_names=["II", "PLETH"])),
        "mri": (rng.random((16, 16, 16)).astype("float32"), {}),
        "dicom": ({"PatientID": "P1", "StudyInstanceUID": "S1", "Modality": "CT",
                   "BodyPartExamined": "HEAD", "Rows": 512, "Columns": 512,
                   "SeriesDescription": "Axial 5mm", "StudyDate": "20240103"}, {}),
        "fhir": ({"resourceType": "Observation", "subject": {"reference": "Patient/P1"},
                  "code": {"text": "C-reactive protein"},
                  "valueQuantity": {"value": 12, "unit": "mg/L"},
                  "effectiveDateTime": "2024-01-03"}, {}),
        "text": ("Patient reports intermittent chest pain and dyspnea on exertion.", {}),
        "genomics": (["chr1:123456:A>G", "chr7:5555:C>T", "chr17:41276045:G>A"], {}),
        "labs": ({"crp": 12.0, "troponin": 0.5, "sodium": 140.0, "creatinine": 1.1}, {}),
        "tabular": ({"age": 64.0, "bmi": 27.5, "sbp": 150.0}, {}),
    }


def main():
    rng = np.random.default_rng(0)
    out = Path(tempfile.gettempdir()) / "all_modalities.gnos"
    mf = MindFile.create(out, modality="multi", overwrite=True)

    print(f"[1] Ingesting one of every registered file type "
          f"({len(modality.catalog())} modalities) into ONE .gnos:")
    items = {}
    for mod, (source, kw) in samples(rng).items():
        items[mod] = modality.ingest(mf, mod, source, subject_id="P1",
                                     session_id="visit1", **kw)
        print(f"    {mod:9} -> item {items[mod]}  spaces={modality.get(mod).spaces}")

    for (sp,) in mf.conn.execute("SELECT name FROM spaces"):     # collapse lazy features
        lazy.materialize_space(mf, sp)

    print("\n[2] LLM card per record (what an external LLM reads; raw stays stored):")
    for iid in items.values():
        print(f"    {render.card(mf, iid)}")

    print("\n[3] Findings + seal + verify:")
    findings.discover(mf)
    checkpoint.create_checkpoint(mf, "all")
    signer = sign.Ed25519Signer()
    sign.sign_state(mf, signer)
    sign.trust_key(mf, signer.public_key)
    rep = audit.audit(mf, deep=True)
    nspaces = mf.conn.execute("SELECT COUNT(*) FROM spaces").fetchone()[0]
    print(f"    {mf.counts()['items']} items across {nspaces} embedding spaces; "
          f"AUDIT={rep['status']} corrupted={rep['corrupted']}; "
          f"signature trusted+current={sign.verify_signatures(mf)['trusted_attested']}")
    mf.close()
    print(f"\nWrote {out} ({out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
