"""Evolution: re-cluster as data/understanding changes, and *record how the
structure changed* — the differentiating feature of the format.

``recluster`` runs a new clustering version whose ``parent_id`` is the previous
run, then diffs the two by item-set overlap to:

* link each new cluster to the parent cluster it mostly came from
  (``clusters.parent_cluster_id``), and
* detect **splits** (one parent cluster spread across several new clusters) and
  **merges** (several parent clusters folded into one), appending them to the
  append-only ``history`` log.

So "healthy aging -> {normal aging, early cognitive decline}" is a recorded,
queryable event, not a silent overwrite.
"""

from __future__ import annotations

from collections import defaultdict

from .cluster import (assignments, clusters_of, latest_clustering_id,
                      run_clustering)
from .file import MindFile
from .spaces import DEFAULT_SPACE


def _overlap(parent_asg: dict[int, int], new_asg: dict[int, int]):
    """Contingency counts: (parent_cluster_id, new_cluster_id) -> shared items."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for item_id, ncid in new_asg.items():
        pcid = parent_asg.get(item_id)
        if pcid is not None:
            counts[(pcid, ncid)] += 1
    return counts


def _label_map(mf: MindFile, clustering_id: int) -> dict[int, str]:
    return {c["id"]: c["label"] for c in clusters_of(mf, clustering_id)}


def _size_map(mf: MindFile, clustering_id: int) -> dict[int, int]:
    return {c["id"]: c["size"] for c in clusters_of(mf, clustering_id)}


def recluster(
    mf: MindFile,
    space: str = DEFAULT_SPACE,
    k: int | None = None,
    *,
    collection: str | None = None,
    seed: int = 0,
    note: str | None = None,
    labels: dict[int, str] | None = None,
    min_share: float = 0.15,
    min_count: int = 2,
) -> dict:
    """Re-cluster ``space`` with lineage + split/merge detection vs the previous
    version (same space + collection).

    A parent/new pair is "significant" when their overlap is at least
    ``min_count`` items and at least ``min_share`` of the relevant cluster.
    """
    parent_id = latest_clustering_id(mf, space, collection)
    if parent_id is None:
        raise ValueError(f"no prior clustering in space {space!r} to evolve from")

    new_id = run_clustering(mf, space, k=k, collection=collection, seed=seed,
                            parent_id=parent_id, note=note, labels=labels)

    parent_asg = assignments(mf, parent_id)
    new_asg = assignments(mf, new_id)
    counts = _overlap(parent_asg, new_asg)

    plabel, psize = _label_map(mf, parent_id), _size_map(mf, parent_id)
    nlabel, nsize = _label_map(mf, new_id), _size_map(mf, new_id)

    # By-parent and by-new groupings of the overlaps.
    by_parent: dict[int, list[tuple[int, int]]] = defaultdict(list)  # pcid -> [(ncid,cnt)]
    by_new: dict[int, list[tuple[int, int]]] = defaultdict(list)     # ncid -> [(pcid,cnt)]
    for (pcid, ncid), cnt in counts.items():
        by_parent[pcid].append((ncid, cnt))
        by_new[ncid].append((pcid, cnt))

    # Link each new cluster to its dominant parent (most shared items).
    for ncid, parents in by_new.items():
        dominant = max(parents, key=lambda t: (t[1], -t[0]))[0]
        mf.conn.execute(
            "UPDATE clusters SET parent_cluster_id = ? WHERE id = ?",
            (dominant, ncid),
        )

    def _significant(cnt: int, denom: int) -> bool:
        return cnt >= min_count and cnt >= min_share * max(denom, 1)

    splits, merges = [], []

    # Split: a parent cluster significantly feeds >= 2 new clusters.
    for pcid, children in by_parent.items():
        sig = [(ncid, cnt) for ncid, cnt in children if _significant(cnt, psize.get(pcid, 0))]
        if len(sig) >= 2:
            into = [{"cluster_id": ncid, "label": nlabel.get(ncid), "count": cnt}
                    for ncid, cnt in sorted(sig, key=lambda t: -t[1])]
            ev = {"parent_cluster_id": pcid, "parent_label": plabel.get(pcid),
                  "parent_clustering_id": parent_id, "new_clustering_id": new_id,
                  "into": into}
            splits.append(ev)
            mf.record("split", ev)

    # Merge: a new cluster significantly draws from >= 2 parent clusters.
    for ncid, parents in by_new.items():
        sig = [(pcid, cnt) for pcid, cnt in parents if _significant(cnt, nsize.get(ncid, 0))]
        if len(sig) >= 2:
            frm = [{"cluster_id": pcid, "label": plabel.get(pcid), "count": cnt}
                   for pcid, cnt in sorted(sig, key=lambda t: -t[1])]
            ev = {"into_cluster_id": ncid, "into_label": nlabel.get(ncid),
                  "parent_clustering_id": parent_id, "new_clustering_id": new_id,
                  "from": frm}
            merges.append(ev)
            mf.record("merge", ev)

    mf.conn.commit()
    return {"clustering_id": new_id, "parent_id": parent_id,
            "splits": splits, "merges": merges}


def lineage(mf: MindFile, clustering_id: int) -> list[dict]:
    """For each cluster in a run, its label and the parent cluster it descends from."""
    rows = mf.conn.execute(
        "SELECT c.id, c.label, c.parent_cluster_id, p.label AS parent_label "
        "FROM clusters c LEFT JOIN clusters p ON p.id = c.parent_cluster_id "
        "WHERE c.clustering_id = ? ORDER BY c.local_index",
        (clustering_id,),
    ).fetchall()
    return [dict(r) for r in rows]
