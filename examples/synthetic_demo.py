"""End-to-end .gnos demo on synthetic data — no external data needed.

Tells the medical narrative from the design sketch:
  Day 1:  "healthy aging" + "tumor-like"          (k=2)
  Later:  a save point is taken, more subjects arrive, and re-clustering
          SPLITS "healthy aging" into "normal aging" + "early decline" (k=3),
          while the original save point still verifies byte-for-byte.

Run:  python examples/synthetic_demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import MindFile, agents, checkpoint, cluster, evolve, graph, ingest, query


def main() -> None:
    out = Path(tempfile.gettempdir()) / "synthetic_scans.gnos"
    rng = np.random.default_rng(7)

    # Two sub-cohorts of "healthy" (near each other) + a distinct "tumor-like" group.
    normal_aging = rng.normal((0, 0), 0.45, (30, 2))
    early_decline = rng.normal((4, 0), 0.45, (30, 2))
    tumor = rng.normal((30, 30), 0.45, (30, 2))
    healthy = np.vstack([normal_aging, early_decline])

    mf = MindFile.create(out, modality="synthetic-mri", overwrite=True)

    print("Day 1 - ingest 90 scans, cluster into 2 groups")
    ingest.add_vectors(mf, np.vstack([healthy, tumor]).astype("float32"),
                       external_ids=[f"scan_{i:03d}" for i in range(90)])
    v1 = cluster.run_clustering(mf, k=2, labels={0: "healthy aging", 1: "tumor-like"})
    for c in cluster.clusters_of(mf, v1):
        print(f"   [{c['local_index']}] {c['label']:<14} size={c['size']}")

    # Register a self-update policy and freeze an immutable save point.
    agents.register_policy(mf, "auto-recluster", purpose="re-cluster as cohort grows",
                           trigger="threshold", action="recluster",
                           params={"new_items_since_cluster": 20})
    cp = checkpoint.create_checkpoint(mf, "day1-original", note="90-scan baseline")
    print(f"\nSave point 'day1-original' frozen - manifest root {cp['manifest_sha256'][:16]}...")

    print("\nLater - 30 new scans arrive")
    ingest.add_vectors(mf, rng.normal((4, 0), 0.45, (30, 2)).astype("float32"),
                       external_ids=[f"scan_{i:03d}" for i in range(90, 120)])
    print("   self-update policy due now?", bool(agents.due(mf)))

    print("Re-cluster into 3 groups and detect structural change:")
    res = evolve.recluster(mf, k=3, note="more data + research update")
    for s in res["splits"]:
        into = ", ".join(f"{i['label']}({i['count']})" for i in s["into"])
        print(f"   SPLIT  {s['parent_label']}  ->  {into}")

    print("\nVerify the original save point survived the growth + re-cluster:")
    rep = checkpoint.verify_checkpoint(mf, "day1-original")
    print(f"   status={rep['status']}  checked_items={rep['checked_items']}  "
          f"changed={rep['changed']}  missing={rep['missing']}")

    graph.build_knn_edges(mf, k=5)
    print("\nexplain(scan_000):")
    ex = query.explain(mf, 1)
    print(f"   cluster={ex['cluster']['label']}  neighbors={len(ex['neighbors'])}  "
          f"evolution_events={len(ex['evolution'])}")

    mf.close()
    print(f"\nWrote {out}  (open it in any SQLite browser - it is a real DB)")


if __name__ == "__main__":
    main()
