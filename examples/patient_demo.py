"""One aggregate .gnos, three analysis scopes, two embedding spaces.

Demonstrates the v2 capabilities on synthetic patient data:
  * several EMBEDDING SPACES per item (bandpower + connectivity)
  * GROUP study   - cluster a cohort across patients
  * LONGITUDINAL  - one patient's sessions over time -> trajectory + change
  * MULTI-MODAL   - one patient's EEG + MRI linked as a bundle
  * validation + a save point that verifies across all spaces

Run:  python examples/patient_demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import (MindFile, analyze, checkpoint, cluster, ingest, spaces,
                      validate)

PATIENTS = ["P1", "P2", "P3", "P4", "P5", "P6"]
YEARS = ["2021", "2022", "2023"]


def main() -> None:
    out = Path(tempfile.gettempdir()) / "patients.gnos"
    rng = np.random.default_rng(3)
    mf = MindFile.create(out, modality="eeg", overwrite=True)

    print("Ingest: 6 patients x 3 EEG sessions (bandpower space) + 1 MRI each (mri space)")
    eeg_item_ids: list[int] = []
    for p in PATIENTS:
        # Each patient has a stable spectral 'profile'; P1 shifts at the last session.
        profile = rng.normal(0, 1, 5).astype("float32")
        for ti, year in enumerate(YEARS):
            v = profile + rng.normal(0, 0.05, 5).astype("float32")
            if p == "P1" and ti == 2:
                v = rng.normal(0, 1, 5).astype("float32")  # sharp change
            ids = ingest.add_vectors(
                mf, v[None, :], space="bandpower",
                subject_ids=p, session_ids=f"t{ti}", modalities="eeg", acquired_ats=year,
                external_ids=f"{p}-eeg-t{ti}")
            eeg_item_ids += ids
        # One MRI item per patient in its own space.
        ingest.add_vectors(
            mf, rng.normal(0, 1, 8)[None, :].astype("float32"), space="mri",
            subject_ids=p, session_ids="t0", modalities="mri", acquired_ats=YEARS[0],
            external_ids=f"{p}-mri")

    # A second embedding VIEW for the EEG items (e.g. connectivity features).
    conn_vecs = rng.normal(0, 1, (len(eeg_item_ids), 6)).astype("float32")
    ingest.add_space_vectors(mf, eeg_item_ids, conn_vecs, space="connectivity")
    print("spaces:", {s["name"]: f"dim{s['dim']}/{s['n_vectors']}vec" for s in spaces.list_spaces(mf)})

    print("\nGROUP study - cluster the EEG cohort (bandpower space):")
    g = analyze.group_study(mf, "eeg-cohort", where={"modality": "eeg"}, space="bandpower", k=3)
    for c in g["clusters"]:
        print(f"   {c['label']}: size={c['size']}")

    print("\nLONGITUDINAL - P1 trajectory over time (bandpower):")
    lt = analyze.longitudinal(mf, "P1", space="bandpower", change_threshold=0.4)
    for s in lt["steps"]:
        flag = "  <-- CHANGE" if s["distance"] > 0.4 else ""
        print(f"   {s['from_session']}->{s['to_session']}  dist={s['distance']:.2f}{flag}")

    print("\nMULTI-MODAL - link P1's EEG + MRI bundle:")
    mm = analyze.multimodal(mf, "P1")
    print(f"   modalities={mm['modalities']}  same_subject_links={mm['links']}")

    print("\nVALIDATE (semantic/structural):")
    rep = validate.validate(mf)
    print(f"   status={rep['status']} errors={rep['errors']} warnings={rep['warnings']}")

    print("\nSAVE POINT across all spaces + verify:")
    cp = checkpoint.create_checkpoint(mf, "baseline", note="6 patients, 2 spaces")
    v = checkpoint.verify_checkpoint(mf, "baseline")
    print(f"   {cp['item_count']} items / {cp['vector_count']} vectors  -> verify status={v['status']}")

    mf.close()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
