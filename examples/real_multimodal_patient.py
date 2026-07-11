"""Real multimodal test: build ONE .gnos for a single patient from REAL files of
every supported type, then evaluate it.

Anchor patient = stroke_qeeg **sub-01** (a real stroke patient). Provenance is
labelled honestly per record:
  [PATIENT]        genuinely this patient's data (real EEG recording + the real
                   clinical row from participants.tsv; FHIR/text derived from it)
  [REAL-PUBLIC]    a real public sample FILE used as a representative stand-in for
                   a scan this patient doesn't have on disk (downloaded)
  [REPRESENTATIVE] a plausible constructed panel (labs), clearly not measured

Evaluation: render every record as an LLM card (the whole patient as text),
surface findings (reference-range flags), then checkpoint + Ed25519-sign +
deep-audit to prove the whole multimodal file is byte-verifiable.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import (MindFile, audit, checkpoint, findings, lazy, modality,
                      render, sign, summary)

STROKE = Path(r"E:/Roger/projects data/stroke_qeeg/data")
REAL = Path(r"C:/Users/roger/AppData/Local/Temp/claude/E--Roger-projects-data/"
            r"67a9e55b-0870-446c-8876-659d8587bf53/scratchpad/realdata")
SUB = "sub-01"
OUT = REAL.parent / "real_multimodal_patient.gnos"


def clinical_row():
    with open(STROKE / "participants.tsv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["Participant_ID"] == SUB:
                return r
    raise SystemExit("sub-01 not found")


def fhir_bundle(c):
    ref = {"reference": f"Patient/{SUB}"}
    return {"resourceType": "Bundle", "type": "collection", "entry": [
        {"resource": {"resourceType": "Patient", "id": SUB, "gender": c["Gender"]}},
        {"resource": {"resourceType": "Condition", "subject": ref,
                      "code": {"text": f"Ischemic stroke, {c['StrokeLocation']}"}}},
        {"resource": {"resourceType": "Observation", "subject": ref,
                      "code": {"text": "NIHSS stroke severity"},
                      "valueQuantity": {"value": float(c["NIHSS"]), "unit": "score"}}},
    ]}


def main():
    c = clinical_row()
    edf = next((STROKE / "edf" / "edffile" / SUB / "eeg").glob("*.edf"))
    prov, kw = {}, dict(subject_id=SUB, session_id="visit1")
    mf = MindFile.create(OUT, modality="multi", overwrite=True)

    def ingest(tag, mod, source, label, **extra):
        r = modality.ingest(mf, mod, source, **kw, **extra)
        for iid in (r if isinstance(r, list) else [r]):
            prov[iid] = (tag, label)

    print(f"Building one .gnos for patient {SUB} from real files of every type...\n")
    # --- genuinely this patient -------------------------------------------
    ingest("PATIENT", "eeg", str(edf), "real EEG recording (motor-imagery EDF)")
    ingest("PATIENT", "tabular", {"age": float(c["Age"]), "nihss": float(c["NIHSS"]),
           "mrs": float(c["mRS"]), "mbi": float(c["MBI"]), "duration_months": float(c["Duration"])},
           "real clinical scores (participants.tsv)")
    ingest("PATIENT", "fhir", fhir_bundle(c), "FHIR bundle built from this patient's record")
    ingest("PATIENT", "text",
           f"{c['Age']}-year-old {c['Handedness']}-handed {c['Gender']}. First-ever ischemic "
           f"stroke in the {c['StrokeLocation']}. {c['ParalysisSide']}-sided hemiparesis. "
           f"Admission NIHSS {c['NIHSS']}, mRS {c['mRS']}, Barthel {c['MBI']}.",
           "clinical note derived from this patient's record")
    # --- real public sample files (representative scans/records) -----------
    ingest("REAL-PUBLIC", "dicom", str(REAL / "mr_brain.dcm"),
           "real public MR DICOM (pydicom sample) - representative brain scan")
    import nibabel as nib
    vol = np.asarray(nib.load(str(REAL / "brain.nii.gz")).dataobj)
    ingest("REAL-PUBLIC", "mri", (vol[..., 0] if vol.ndim == 4 else vol).astype("float32"),
           "real public NIfTI volume (nibabel sample) - representative MRI")
    ingest("REAL-PUBLIC", "wfdb", str(REAL / "rec_1"),
           "real public ECG record (PhysioNet ecgiddb) - representative ECG")
    ingest("REAL-PUBLIC", "genomics", (REAL / "sample.vcf").read_text(),
           "real public VCF (vcflib sample) - representative variants")
    # --- representative constructed ---------------------------------------
    ingest("REPRESENTATIVE", "labs",
           {"crp": 8.2, "wbc": 11.5, "glucose": 140.0, "sodium": 139.0, "creatinine": 1.0, "ldl": 130.0},
           "plausible acute-stroke labs (constructed, not measured)")

    for (sp,) in mf.conn.execute("SELECT name FROM spaces"):
        lazy.materialize_space(mf, sp)

    print("[records ingested] provenance | card")
    for iid, (tag, label) in prov.items():
        print(f"  [{tag:14}] {render.card(mf, iid)}")
        print(f"  {'':16}   ^ {label}")

    print("\n[findings the file surfaces]")
    for fnd in findings.discover(mf):
        print(f"  - {fnd['statement']}")

    print("\n[seal + verify the whole multimodal file]")
    checkpoint.create_checkpoint(mf, f"patient-{SUB}")
    signer = sign.Ed25519Signer()
    sign.sign_state(mf, signer)
    sign.trust_key(mf, signer.public_key)
    rep = audit.audit(mf, deep=True)
    s = summary.summary(mf)
    nsp = mf.conn.execute("SELECT COUNT(*) FROM spaces").fetchone()[0]
    print(f"  {mf.counts()['items']} records, {nsp} embedding spaces, "
          f"{mf.counts()['signals']} lossless waveforms")
    print(f"  deep AUDIT = {rep['status']}  corrupted={rep['corrupted']}")
    print(f"  signature trusted+current = {sign.verify_signatures(mf)['trusted_attested']}")
    print(f"  file = {OUT.stat().st_size/1e6:.1f} MB")
    mf.close()
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
