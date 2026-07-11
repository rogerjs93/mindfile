"""Cross-file registry: a study file that verifies many signed patient files.

Builds several signed patient .gnos files + a study registry that references
them, then shows the registry detecting drift, substitution, and a missing file.

Run:  python examples/registry_demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import MindFile, cluster, ingest, registry, sign


def make_patient(path, seed):
    mf = MindFile.create(path, modality="eeg", overwrite=True)
    ingest.add_vectors(mf, np.random.default_rng(seed).normal(seed, 0.5, (8, 5)).astype("float32"),
                       subject_ids=path.stem)
    cluster.run_clustering(mf, k=2)
    sign.sign_state(mf, sign.Ed25519Signer())     # each patient lab-signs its file
    mf.close()


def main() -> None:
    d = Path(tempfile.mkdtemp(prefix="mind_registry_"))
    patients = ["P1", "P2", "P3"]
    for i, p in enumerate(patients):
        make_patient(d / f"{p}.gnos", seed=i)

    study = MindFile.create(d / "study.gnos", modality="study", overwrite=True)
    for p in patients:
        info = registry.link_file(study, f"{p}.gnos", role="patient", require_signed=True)
        print(f"linked {p}: uuid {info['uuid'][:8]} state {info['state_root'][:12]}... key {info['key_id']}")

    print("\nverify all referenced files:")
    for r in registry.verify_refs(study)["refs"]:
        print(f"  {r['path']}: {r['status']}")

    print("\nNow: P2 does more work (drift), P3 is replaced (substitution), P1 is deleted (missing)")
    p2 = MindFile.open(d / "P2.gnos")
    ingest.add_vectors(p2, np.random.default_rng(99).normal(0, 1, (3, 5)).astype("float32"))
    p2.close()
    make_patient(d / "P3.gnos", seed=999)          # new uuid at same path
    (d / "P1.gnos").unlink()

    rep = registry.verify_refs(study)
    print(f"registry status: {rep['status']}")
    for r in rep["refs"]:
        print(f"  {r['path']}: {r['status']}")
    study.close()
    print(f"\nFiles in {d}")


if __name__ == "__main__":
    main()
