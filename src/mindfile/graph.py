"""Knowledge block: build and query the relationship graph.

``build_knn_edges`` connects each item to its k nearest neighbours within a
space. Edges carry the ``space_id`` they came from, so different embedding views
yield different graphs. ``link`` records typed relations (e.g. ``temporal`` for a
longitudinal series, ``same_subject`` for a multi-modal bundle).
"""

from __future__ import annotations

import json

import numpy as np

from .file import MindFile
from .index import _l2_normalize, load_all_vectors
from .spaces import DEFAULT_SPACE, get_space_id


def build_knn_edges(
    mf: MindFile, k: int = 5, space: str = DEFAULT_SPACE, kind: str = "knn",
    item_ids: list[int] | None = None,
) -> int:
    """(Re)build within-file kNN edges for one space. Returns edges written.

    If ``item_ids`` is given, edges are built only among that subset (e.g. one
    patient's points -> a per-patient constellation graph).
    """
    sid = get_space_id(mf, space) or 0
    ids, mat = load_all_vectors(mf, space, item_ids=item_ids)
    n = ids.shape[0]
    if n < 2:
        return 0
    k = min(k, n - 1)
    norm = _l2_normalize(mat)
    sims = norm @ norm.T
    np.fill_diagonal(sims, -np.inf)

    written = 0
    for i in range(n):
        nbr = np.argpartition(-sims[i], k - 1)[:k]
        nbr = nbr[np.argsort(-sims[i][nbr])]
        for j in nbr:
            cur = mf.conn.execute(
                "INSERT OR IGNORE INTO edges(src_item_id, dst_item_id, dst_file, space_id, "
                "kind, weight) VALUES(?,?,?,?,?,?)",
                (int(ids[i]), int(ids[j]), "", sid, kind, float(sims[i][j])),
            )
            written += cur.rowcount
    mf.record("link", {"kind": kind, "space": space, "k": k, "edges_written": written})
    mf.conn.commit()
    return written


def link(
    mf: MindFile,
    src_item_id: int,
    dst_item_id: int,
    *,
    kind: str = "manual",
    space: str | None = None,
    weight: float | None = None,
    meta: dict | None = None,
) -> None:
    """Record a typed within-file edge (e.g. temporal, same_subject, manual)."""
    sid = get_space_id(mf, space) if space else 0
    mf.conn.execute(
        "INSERT OR IGNORE INTO edges(src_item_id, dst_item_id, dst_file, space_id, kind, "
        "weight, meta) VALUES(?,?,?,?,?,?,?)",
        (int(src_item_id), int(dst_item_id), "", sid or 0, kind, weight,
         json.dumps(meta) if meta else None),
    )
    mf.conn.commit()


def neighbors(mf: MindFile, item_id: int, kind: str | None = None, limit: int | None = None) -> list[dict]:
    """Outgoing edges from an item, strongest first (optionally filtered by kind)."""
    sql = ("SELECT dst_item_id, dst_file, space_id, kind, weight FROM edges "
           "WHERE src_item_id = ?")
    params: list = [item_id]
    if kind is not None:
        sql += " AND kind = ?"
        params.append(kind)
    sql += " ORDER BY weight DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [dict(r) for r in mf.conn.execute(sql, params)]
