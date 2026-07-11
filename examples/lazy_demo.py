"""Lazy / virtual derived vectors — store a recipe, resolve on call.

Fuses two modalities WITHOUT materializing the fused vectors: only recipes are
stored. The fused points exist as a "dynamic state" until called, then resolve
to bytes identical to the eager computation. Materialize to cache, dematerialize
to reclaim space — the recipe (and its verifiability) stays.

Run:  python examples/lazy_demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import MindFile, fuse, ingest, lazy, spaces


def main() -> None:
    out = Path(tempfile.gettempdir()) / "lazy.gnos"
    rng = np.random.default_rng(0)
    mf = MindFile.create(out, modality="eeg", overwrite=True)
    for i, p in enumerate(["P0", "P1", "P2", "P3"]):
        ingest.add_vectors(mf, rng.normal(i, 0.3, (1, 5)).astype("float32"), space="bandpower",
                           subject_ids=p, session_ids="t0", modalities="eeg", external_ids=f"{p}-eeg")
        ingest.add_vectors(mf, rng.normal(i, 0.3, (1, 8)).astype("float32"), space="mri",
                           subject_ids=p, session_ids="t0", modalities="mri", external_ids=f"{p}-mri")

    print("Fuse bandpower+mri LAZILY (recipes only, no fused bytes stored):")
    fuse.fuse(mf, ["bandpower", "mri"], lazy=True)
    derived = [r["id"] for r in mf.conn.execute("SELECT id FROM items WHERE origin='derived'")]
    fused_vecs = mf.conn.execute(
        "SELECT COUNT(*) FROM vectors v JOIN spaces s ON s.id=v.space_id WHERE s.name='fused'").fetchone()[0]
    print(f"  derived items: {len(derived)}   recipes: {mf.counts()['vector_defs']}   "
          f"materialized fused vectors: {fused_vecs}")

    item = derived[0]
    print(f"\nitem {item} virtual? {lazy.is_virtual(mf, item, 'fused')}")
    v = lazy.resolve(mf, item, "fused")
    print(f"resolve on call -> vector of dim {v.shape[0]} (computed from sources, not stored)")

    print("\nMaterialize the whole fused space (cache the bytes):")
    n = lazy.materialize_space(mf, "fused")
    fused_vecs = mf.conn.execute(
        "SELECT COUNT(*) FROM vectors v JOIN spaces s ON s.id=v.space_id WHERE s.name='fused'").fetchone()[0]
    print(f"  materialized {n}  ->  fused vectors now stored: {fused_vecs}, virtual? "
          f"{lazy.is_virtual(mf, item, 'fused')}")

    print("\nDematerialize one (reclaim space; recipe + verifiability stay):")
    lazy.dematerialize(mf, item, "fused")
    print(f"  item {item} virtual again? {lazy.is_virtual(mf, item, 'fused')}   "
          f"verify_defs: {lazy.verify_defs(mf)['status']}")

    mf.close()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
