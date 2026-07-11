"""Constellations + the in-file Planner -> Reviewer -> Renderer allocator.

Builds a small cohort with two modalities and x,y,z,t coordinates, then lets the
file MANAGE ITSELF: the allocator plans what to materialize (cluster, kNN, fuse,
position-aware constellation space, correlations), reviews it under guards, and
renders it - logging every step. Run twice to watch it self-converge.

Run:  python examples/allocate_demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import MindFile, allocate, axes, checkpoint, ingest, spaces, stats


def main() -> None:
    out = Path(tempfile.gettempdir()) / "allocate.gnos"
    rng = np.random.default_rng(1)
    mf = MindFile.create(out, modality="eeg", overwrite=True)

    print("Ingest 8 patients: EEG (bandpower 5d) + MRI (8d), with x,y,z,t on the EEG items")
    eeg = []
    for i, p in enumerate([f"P{j}" for j in range(8)]):
        eeg += ingest.add_vectors(mf, rng.normal(i, 0.5, (1, 5)).astype("float32"), space="bandpower",
                                  subject_ids=p, session_ids="t0", modalities="eeg", external_ids=f"{p}-eeg")
        ingest.add_vectors(mf, rng.normal(i, 0.5, (1, 8)).astype("float32"), space="mri",
                           subject_ids=p, session_ids="t0", modalities="mri", external_ids=f"{p}-mri")
    axes.set_axes_bulk(mf, eeg, rng.normal(0, 1, (len(eeg), 4)).astype("float32"), ["x", "y", "z", "t"])

    checkpoint.create_checkpoint(mf, "primary", note="raw cohort")
    print(f"\nSpaces before: {[s['name'] for s in spaces.list_spaces(mf)]}")

    for run in (1, 2):
        print(f"\n=== allocate pass {run} (Planner -> Reviewer -> Renderer) ===")
        res = allocate.orchestrate(mf)
        print(f"proposed {res['proposed']}, approved {res['approved']}, rendered {res['rendered']}")
        for a in res["actions"]:
            mark = {"done": "OK", "skipped": "--", "failed": "!!"}[a["render_status"]]
            print(f"  [{mark}] {a['action']:<13} {a['review_reason']}")

    print(f"\nSpaces after: {[s['name'] for s in spaces.list_spaces(mf)]}")
    print("\nCorrelations computed in-file:")
    for c in stats.list_correlations(mf):
        v = f"{c['value']:.3f}" if c["value"] is not None else "n/a"
        print(f"  {c['kind']:<24} {c['a_ref']} vs {c['b_ref']} = {v} (n={c['n']})")

    rep = checkpoint.verify_checkpoint(mf, "primary")
    print(f"\nOriginals conserved through all allocation: verify={rep['status']}")
    mf.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
