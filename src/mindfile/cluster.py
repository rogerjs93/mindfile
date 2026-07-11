"""Clustering: group item vectors *within a space* (and optional collection),
persisting each run as an immutable, versioned ``clusterings`` row.

Versions increment within a space, so each embedding view evolves independently.
Prior runs are never modified — that immutability is what lets understanding
*evolve* (see ``evolve.py``) while the history of how it changed stays intact.
"""

from __future__ import annotations

import json
import math

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from . import _vec
from .collections import collection_id as _collection_id
from .collections import members as _members
from .file import MindFile, utcnow
from .index import load_all_vectors
from .spaces import DEFAULT_SPACE, get_space_id


def _default_k(n: int) -> int:
    return max(2, min(8, int(round(math.sqrt(n / 2))) or 2))


def _count_vectors(mf: MindFile, sid: int, member_ids=None) -> int:
    sql = "SELECT COUNT(*) FROM vectors WHERE space_id = ?"
    params: list = [sid]
    if member_ids is not None:
        if not member_ids:
            return 0
        sql += f" AND item_id IN ({','.join('?' * len(member_ids))})"
        params += list(member_ids)
    return mf.conn.execute(sql, params).fetchone()[0]


def _fit_in_ram(mf, space, member_ids, k, seed):
    """Exact KMeans over the whole space held in RAM. Yields centroids + a list
    of (item_id, local_label, distance)."""
    ids, mat = load_all_vectors(mf, space, item_ids=member_ids)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    lab = km.fit_predict(mat)
    cents = km.cluster_centers_.astype(np.float32)
    assign = [(int(ids[i]), int(lab[i]), float(np.linalg.norm(mat[i] - cents[lab[i]])))
              for i in range(ids.shape[0])]
    return cents, assign


def _space_item_ids(mf, sid, member_ids):
    sql = "SELECT item_id FROM vectors WHERE space_id = ?"
    params: list = [sid]
    if member_ids is not None:
        sql += f" AND item_id IN ({','.join('?' * len(member_ids))})"
        params += list(member_ids)
    return [r[0] for r in mf.conn.execute(sql, params)]


def _fit_streaming(mf, space, sid, member_ids, k, seed, batch, epochs):
    """Out-of-core MiniBatchKMeans: never holds more than ``batch`` vectors in
    RAM (only the lightweight id list). Batches are drawn in *shuffled* id order
    each epoch so mini-batches stay well mixed even when the store is sorted by
    subject/cluster; a final streaming pass assigns each item. Approximate."""
    ids_all = _space_item_ids(mf, sid, member_ids)
    order = np.array(ids_all, dtype=np.int64)
    rng = np.random.default_rng(seed)
    mbk = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                          batch_size=max(batch, k * 8))
    for _ in range(epochs):
        rng.shuffle(order)
        for s in range(0, len(order), batch):
            _ids, mat = load_all_vectors(mf, space, item_ids=order[s:s + batch].tolist())
            if mat.shape[0]:
                mbk.partial_fit(mat)
    cents = mbk.cluster_centers_.astype(np.float32)
    assign = []
    for s in range(0, len(ids_all), batch):
        _ids, mat = load_all_vectors(mf, space, item_ids=ids_all[s:s + batch])
        lab = mbk.predict(mat)
        d = np.linalg.norm(mat - cents[lab], axis=1)
        assign += [(int(_ids[i]), int(lab[i]), float(d[i])) for i in range(_ids.shape[0])]
    return cents, assign


def _resolve_collection(mf: MindFile, collection: str | None):
    if collection is None:
        return None, None
    cid = _collection_id(mf, collection)
    if cid is None:
        raise ValueError(f"no collection named {collection!r}")
    return cid, _members(mf, collection)


def latest_clustering_id(
    mf: MindFile, space: str = DEFAULT_SPACE, collection: str | None = None
) -> int | None:
    sid = get_space_id(mf, space)
    if sid is None:
        return None
    cid, _ = _resolve_collection(mf, collection)
    cond = "collection_id IS NULL" if cid is None else "collection_id = ?"
    params = [sid] if cid is None else [sid, cid]
    row = mf.conn.execute(
        f"SELECT id FROM clusterings WHERE space_id = ? AND {cond} "
        "ORDER BY version DESC, id DESC LIMIT 1", params,
    ).fetchone()
    return row["id"] if row else None


def clustering_versions(mf: MindFile, space: str | None = None) -> list[dict]:
    sql = ("SELECT cl.id, cl.version, cl.algo, cl.parent_id, cl.created_at, cl.note, "
           "s.name AS space, cl.collection_id "
           "FROM clusterings cl JOIN spaces s ON s.id = cl.space_id")
    params: list = []
    if space is not None:
        sql += " WHERE s.name = ?"
        params.append(space)
    sql += " ORDER BY s.name, cl.version"
    return [dict(r) for r in mf.conn.execute(sql, params)]


def assignments(mf: MindFile, clustering_id: int) -> dict[int, int]:
    rows = mf.conn.execute(
        "SELECT item_id, cluster_id FROM cluster_assignments WHERE clustering_id = ?",
        (clustering_id,),
    ).fetchall()
    return {r["item_id"]: r["cluster_id"] for r in rows}


def clusters_of(mf: MindFile, clustering_id: int) -> list[dict]:
    rows = mf.conn.execute(
        "SELECT id, local_index, label, size, parent_cluster_id "
        "FROM clusters WHERE clustering_id = ? ORDER BY local_index",
        (clustering_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def run_clustering(
    mf: MindFile,
    space: str = DEFAULT_SPACE,
    k: int | None = None,
    *,
    collection: str | None = None,
    algo: str = "kmeans",
    seed: int = 0,
    parent_id: int | None = None,
    note: str | None = None,
    labels: dict[int, str] | None = None,
    batch: int = 8192,
    epochs: int = 5,
) -> int:
    """Cluster vectors in ``space`` (optionally restricted to ``collection``).

    ``algo='kmeans'`` (default) is exact but loads the whole space into RAM.
    ``algo='minibatch'`` is out-of-core: it streams the space in ``batch``-sized
    chunks (memory bounded regardless of corpus size), at the cost of mini-batch
    approximation. Returns the new ``clustering_id``.
    """
    sid = get_space_id(mf, space)
    if sid is None:
        raise ValueError(f"no space named {space!r}")
    coll_id, member_ids = _resolve_collection(mf, collection)

    n = _count_vectors(mf, sid, member_ids)
    if n < 2:
        raise ValueError(f"need at least 2 vectors in space {space!r} to cluster (got {n})")
    k = min(int(k) if k else _default_k(n), n)

    if algo == "kmeans":
        centroids, assign = _fit_in_ram(mf, space, member_ids, k, seed)
    elif algo == "minibatch":
        centroids, assign = _fit_streaming(mf, space, sid, member_ids, k, seed, batch, epochs)
    else:
        raise ValueError(f"unsupported algo {algo!r} (use 'kmeans' or 'minibatch')")

    version = mf.conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM clusterings WHERE space_id = ?", (sid,)
    ).fetchone()[0]
    embedder_id = mf.conn.execute(
        "SELECT embedder_id FROM spaces WHERE id = ?", (sid,)
    ).fetchone()["embedder_id"]
    cur = mf.conn.execute(
        "INSERT INTO clusterings(space_id, collection_id, version, algo, params, "
        "embedder_id, created_at, parent_id, note) VALUES(?,?,?,?,?,?,?,?,?)",
        (sid, coll_id, version, algo, json.dumps({"k": k, "seed": seed}), embedder_id,
         utcnow(), parent_id, note),
    )
    clustering_id = cur.lastrowid

    sizes = np.bincount([a[1] for a in assign], minlength=k)
    local_to_clusterid: dict[int, int] = {}
    for ci in range(k):
        label = (labels or {}).get(ci) or f"cluster {ci}"
        c = mf.conn.execute(
            "INSERT INTO clusters(clustering_id, local_index, label, centroid, size) "
            "VALUES(?,?,?,?,?)",
            (clustering_id, ci, label, _vec.pack(centroids[ci]), int(sizes[ci])),
        )
        local_to_clusterid[ci] = c.lastrowid

    mf.conn.executemany(
        "INSERT INTO cluster_assignments(clustering_id, item_id, cluster_id, distance) "
        "VALUES(?,?,?,?)",
        [(clustering_id, iid, local_to_clusterid[ci], dist) for iid, ci, dist in assign])

    mf.record("cluster", {"clustering_id": clustering_id, "space": space, "version": version,
                          "k": k, "n": n, "collection": collection, "parent_id": parent_id,
                          "algo": algo})
    mf.conn.commit()
    return clustering_id
