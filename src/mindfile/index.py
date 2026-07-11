"""Vector index / similarity search over a ``.mind`` file, scoped to a space.

Default backend is an exact NumPy brute-force cosine search. The schema stores
embeddings as float32 BLOBs, which a libSQL build can later index with
``libsql_vector_idx`` for ANN without changing the file format.
"""

from __future__ import annotations

import numpy as np

from . import _vec
from .file import MindFile
from .spaces import DEFAULT_SPACE, get_space_id


def load_all_vectors(
    mf: MindFile, space: str = DEFAULT_SPACE, item_ids: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (item_ids[int64 (n,)], matrix[float32 (n, dim)]) for one space.

    If ``item_ids`` is given, restrict to that subset (e.g. a collection).
    """
    sid = get_space_id(mf, space)
    if sid is None:
        return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    sql = "SELECT item_id, dim, embedding FROM vectors WHERE space_id = ?"
    params: list = [sid]
    if item_ids is not None:
        if not item_ids:
            return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        sql += f" AND item_id IN ({','.join('?' * len(item_ids))})"
        params += list(item_ids)
    sql += " ORDER BY item_id"
    rows = mf.conn.execute(sql, params).fetchall()
    if not rows:
        return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    ids = np.array([r["item_id"] for r in rows], dtype=np.int64)
    dim = rows[0]["dim"]
    mat = np.empty((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        mat[i] = _vec.unpack(r["embedding"], dim)
    return ids, mat


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def iter_vectors(mf: MindFile, space: str = DEFAULT_SPACE, *, item_ids=None, batch: int = 8192):
    """Yield (ids, matrix) batches for a space, streaming from the DB cursor so
    the whole space is never held in memory at once."""
    sid = get_space_id(mf, space)
    if sid is None:
        return
    sql = "SELECT item_id, dim, embedding FROM vectors WHERE space_id = ?"
    params: list = [sid]
    if item_ids is not None:
        if not item_ids:
            return
        sql += f" AND item_id IN ({','.join('?' * len(item_ids))})"
        params += list(item_ids)
    sql += " ORDER BY item_id"
    cur = mf.conn.execute(sql, params)
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break
        dim = rows[0]["dim"]
        ids = np.array([r["item_id"] for r in rows], dtype=np.int64)
        mat = np.empty((len(rows), dim), dtype=np.float32)
        for i, r in enumerate(rows):
            mat[i] = _vec.unpack(r["embedding"], dim)
        yield ids, mat


def search(mf: MindFile, query, k: int = 5, space: str = DEFAULT_SPACE,
           batch: int = 8192) -> list[tuple[int, float]]:
    """Exact k-NN cosine search, **streamed**: memory is bounded to ``batch``
    vectors regardless of how many the space holds. Returns (item_id, sim), best
    first. (Bulk ops like clustering still load the full matrix — that path is
    the remaining scale frontier.)"""
    import heapq
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    qn = q / (np.linalg.norm(q) or 1.0)
    heap: list = []                                  # min-heap of (sim, item_id)
    for ids, mat in iter_vectors(mf, space, batch=batch):
        if mat.shape[1] != qn.shape[0]:
            raise ValueError(f"query dim {qn.shape[0]} != space {space!r} dim {mat.shape[1]}")
        sims = _l2_normalize(mat) @ qn
        kk = min(k, sims.shape[0])
        for i in np.argpartition(-sims, kk - 1)[:kk]:   # only the batch's top-k touch the heap
            s, iid = float(sims[i]), int(ids[i])
            if len(heap) < k:
                heapq.heappush(heap, (s, iid))
            elif s > heap[0][0]:
                heapq.heapreplace(heap, (s, iid))
    return [(iid, s) for s, iid in sorted(heap, key=lambda t: -t[0])]


def search_ann(mf: MindFile, query, k: int = 5, space: str = DEFAULT_SPACE,
               nprobe: int = 2) -> list[tuple[int, float]]:
    """Approximate search using the space's latest clustering as an IVF index:
    probe the ``nprobe`` nearest cells and brute-force only their members.

    Faster than full scan when there are many clusters; falls back to exact
    ``search`` if the space isn't clustered. (A libSQL build with native vector
    support can replace this with an HNSW backend without changing callers.)
    """
    sid = get_space_id(mf, space)
    if sid is None:
        return []
    run = mf.conn.execute(
        "SELECT id FROM clusterings WHERE space_id=? ORDER BY version DESC, id DESC LIMIT 1",
        (sid,)).fetchone()
    if run is None:
        return search(mf, query, k=k, space=space)  # no index yet -> exact
    clustering_id = run["id"]
    dim = mf.conn.execute("SELECT dim FROM spaces WHERE id=?", (sid,)).fetchone()["dim"]

    cells = mf.conn.execute(
        "SELECT id, centroid FROM clusters WHERE clustering_id=? AND centroid IS NOT NULL",
        (clustering_id,)).fetchall()
    if not cells:
        return search(mf, query, k=k, space=space)
    cids = [c["id"] for c in cells]
    cmat = _l2_normalize(np.stack([_vec.unpack(c["centroid"], dim) for c in cells]))
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    qn = q / (np.linalg.norm(q) or 1.0)
    nprobe = min(nprobe, len(cids))
    probe = np.argsort(-(cmat @ qn))[:nprobe]
    probe_cids = [cids[i] for i in probe]

    members = [r["item_id"] for r in mf.conn.execute(
        f"SELECT item_id FROM cluster_assignments WHERE clustering_id=? AND cluster_id IN "
        f"({','.join('?' * len(probe_cids))})", [clustering_id, *probe_cids])]
    ids, mat = load_all_vectors(mf, space, item_ids=members)
    if ids.shape[0] == 0:
        return []
    sims = _l2_normalize(mat) @ qn
    k = min(k, sims.shape[0])
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(ids[i]), float(sims[i])) for i in top]
