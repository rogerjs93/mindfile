"""Capstone: a multimodal patient cohort, end to end.

For each patient we ingest FIVE modalities through their per-type protocols
(EEG, ECG, MRI, clinical text, genomics), then:
  1. materialize the lazy waveform features,
  2. FUSE all modality spaces per patient -> one unified 'patient' vector,
  3. cluster the cohort by it (cross-modal subtyping),
  4. checkpoint + Ed25519-sign + audit the whole file.

The cohort has two latent subtypes; clustering the fused vector should recover
them from the combination of modalities. Run:  python examples/multimodal_demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import (MindFile, audit, checkpoint, cluster, fuse, lazy,
                      modality, sign)

MODALITY_SPACES = ["bandpower", "ecg_bandpower", "radiomics", "text", "variants"]
NOTES = {0: "ischemic stroke left mca aphasia hemiparesis",
         1: "healthy control normal scan no deficit"}
VARIANTS = {0: ["1:100:A:G", "7:55:G:A", "X:12:C:T"],
            1: ["2:200:C:T", "9:88:T:A"]}


def ingest_patient(mf, pid, subtype, rng):
    s = subtype * 2.0  # latent offset that separates the two subtypes
    modality.ingest(mf, "eeg", rng.normal(s, 0.4, (8, 1000)).astype("float32"),
                    sfreq=500.0, subject_id=pid, session_id="t0")
    modality.ingest(mf, "ecg", rng.normal(s, 0.4, (2, 1000)).astype("float32"),
                    sfreq=250.0, subject_id=pid, session_id="t0")
    modality.ingest(mf, "mri", rng.normal(s, 0.3, (8, 8, 8)).astype("float32"),
                    subject_id=pid, session_id="t0")
    modality.ingest(mf, "text", NOTES[subtype], subject_id=pid, session_id="t0")
    modality.ingest(mf, "genomics", VARIANTS[subtype], subject_id=pid, session_id="t0")


def main() -> None:
    out = Path(tempfile.gettempdir()) / "cohort.gnos"
    rng = np.random.default_rng(0)
    mf = MindFile.create(out, modality="multimodal", overwrite=True)

    patients = [(f"P{i}", i % 2) for i in range(8)]   # 8 patients, 2 subtypes
    print("Ingest 8 patients x 5 modalities (per-type protocols)...")
    for pid, sub in patients:
        ingest_patient(mf, pid, sub, rng)
    print(f"  items: {mf.counts()['items']}  spaces: "
          f"{[s['name'] for s in __import__('mindfile').spaces.list_spaces(mf)]}")

    # Lazy waveform features must be concrete before they can be fused.
    for sp in ("bandpower", "ecg_bandpower"):
        lazy.materialize_space(mf, sp)

    print("\nFuse all 5 modality spaces per patient -> one 'patient' vector:")
    res = fuse.fuse(mf, MODALITY_SPACES, key=("subject_id", "session_id"), out_space="patient")
    print(f"  created {len(res['created'])} unified patient vectors, skipped {len(res['skipped'])}")

    print("\nCluster the cohort by the fused multimodal vector:")
    cid = cluster.run_clustering(mf, "patient", k=2)
    asg = cluster.assignments(mf, cid)
    # purity vs latent subtype
    sub_of = {f"P{i}": i % 2 for i in range(8)}
    by_cluster = {}
    for item_id, cl in asg.items():
        ext = mf.conn.execute("SELECT external_id FROM items WHERE id=?", (item_id,)).fetchone()[0]
        # fused item external_id looks like "fused:P3:t0"
        pid = ext.split(":")[1]
        by_cluster.setdefault(cl, []).append(sub_of[pid])
    for cl, subs in by_cluster.items():
        maj = max(set(subs), key=subs.count)
        print(f"  cluster {cl}: {len(subs)} patients, dominant subtype {maj} "
              f"({subs.count(maj)}/{len(subs)} pure)")

    print("\nSeal it: checkpoint + sign + audit")
    checkpoint.create_checkpoint(mf, "cohort-baseline")
    signer = sign.Ed25519Signer()
    sign.sign_state(mf, signer)
    sign.trust_key(mf, signer.public_key)
    rep = audit.audit(mf, deep=True)
    print(f"  audit: status={rep['status']}  corrupted={rep['corrupted']}")
    print(f"  signature trusted+current: {sign.verify_signatures(mf)['trusted_attested']}")

    mf.close()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
