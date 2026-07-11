"""Constellations — position-aware vectorization + per-patient point graphs.

A constellation enriches each item's semantic vector with its coordinate axes
(x, y, z, t) to produce a *position-aware* embedding, then connects a patient's
points into a graph across space and time. This is additive: it writes a NEW
space on existing items and NEW edges; the originals are untouched and stay
verifiable.

The constellation vector = ``[ L2(semantic)  ⊕  coord_weight · zscore(x,y,z,t) ]``.
Semantic parts are L2-normalized (directional); coordinates are standardized
per-axis across the dataset (so position/scale is preserved, not collapsed).
"""

from __future__ import annotations

import numpy as np

from . import graph as _graph
from .axes import axes_matrix
from .collections import collection_id as _collection_id
from .collections import create_collection
from .file import MindFile
from .index import _l2_normalize, load_all_vectors
from .ingest import add_space_vectors
from .spaces import get_space_id

DEFAULT_AXES = ("x", "y", "z", "t")


def build_constellation_space(
    mf: MindFile,
    semantic_space: str,
    *,
    axes=DEFAULT_AXES,
    out_space: str = "constellation",
    coord_weight: float = 1.0,
    lazy: bool = False,
) -> dict:
    """Write position-aware vectors for every item that has ``semantic_space`` +
    all ``axes``. With ``lazy=True`` store recipes (resolved on access) instead of
    materialized vectors. Idempotent: items already present are skipped."""
    axes = tuple(axes)
    sem_ids, sem_mat = load_all_vectors(mf, semantic_space)
    if sem_ids.size == 0:
        return {"created": 0, "space": out_space}

    kept_ids, coords = axes_matrix(mf, sem_ids.tolist(), axes)
    if kept_ids.size == 0:
        return {"created": 0, "space": out_space, "reason": "no items have all axes"}

    row_of = {int(sem_ids[i]): i for i in range(sem_ids.size)}
    sem = np.stack([sem_mat[row_of[int(i)]] for i in kept_ids])
    sem_n = _l2_normalize(sem)

    mean, std = coords.mean(axis=0), coords.std(axis=0)
    std[std == 0] = 1.0
    coords_n = (coords - mean) / std * float(coord_weight)
    const_all = np.concatenate([sem_n, coords_n], axis=1).astype(np.float32)

    existing = set()
    out_sid = get_space_id(mf, out_space)
    if out_sid is not None:
        existing = {r[0] for r in mf.conn.execute(
            "SELECT item_id FROM vectors WHERE space_id = ?", (out_sid,))}
    new = [i for i, iid in enumerate(kept_ids) if int(iid) not in existing]
    if not new:
        return {"created": 0, "space": out_space, "dim": const_all.shape[1]}

    if lazy:
        from . import lazy as _lazy
        recipe = {"method": "constellation-concat-v1", "semantic_space": semantic_space,
                  "axes": list(axes), "coord_weight": float(coord_weight),
                  "mean": mean.tolist(), "std": std.tolist()}
        for i in new:
            _lazy.define(mf, int(kept_ids[i]), out_space, "constellation", recipe,
                         dim=const_all.shape[1], embedder_id="constellation-concat-v1", derived=True)
    else:
        add_space_vectors(mf, [int(kept_ids[i]) for i in new], const_all[new],
                          space=out_space, embedder_id="constellation-concat-v1", derived=True)
    mf.record("constellation", {"space": out_space, "from": semantic_space, "axes": list(axes),
                                "created": len(new), "lazy": lazy})
    return {"created": len(new), "space": out_space, "dim": const_all.shape[1], "lazy": lazy}


def build_constellation(
    mf: MindFile, subject_id: str, *, space: str = "constellation", k: int = 4
) -> dict:
    """Form one patient's constellation: a collection of their points + a kNN
    graph among them in the position-aware ``space``."""
    sid = get_space_id(mf, space)
    if sid is None:
        raise ValueError(f"no space {space!r}; build the constellation space first")
    members = [r["id"] for r in mf.conn.execute(
        "SELECT i.id FROM items i JOIN vectors v ON v.item_id = i.id "
        "WHERE i.subject_id = ? AND v.space_id = ? ORDER BY i.id", (subject_id, sid))]
    name = f"constellation:{subject_id}"
    if members and _collection_id(mf, name) is None:
        create_collection(mf, name, kind="constellation", item_ids=members)
    edges = 0
    if len(members) >= 2:
        edges = _graph.build_knn_edges(mf, k=min(k, len(members) - 1), space=space,
                                       kind="constellation", item_ids=members)
    return {"subject_id": subject_id, "points": len(members), "edges": edges}


def subjects_with_axes(mf: MindFile) -> list[str]:
    """Subjects that have at least one item carrying coordinate axes."""
    return [r["subject_id"] for r in mf.conn.execute(
        "SELECT DISTINCT i.subject_id FROM items i JOIN item_axes a ON a.item_id = i.id "
        "WHERE i.subject_id IS NOT NULL ORDER BY i.subject_id")]
