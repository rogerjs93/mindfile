"""In-file correlation statistics, stored in the ``correlations`` table.

Four kinds:
* ``clustering_ari``          — Adjusted Rand Index between two clusterings
                                (do two views tell the same story?).
* ``cross_space``             — Pearson r between the pairwise distance matrices
                                of two spaces over shared items.
* ``feature_corr``            — correlation across the dimensions of one space.
* ``subject_cross_modality``  — like cross_space, matched one-vector-per-subject.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import adjusted_rand_score

from . import _vec
from .cluster import assignments
from .file import MindFile, utcnow
from .index import _l2_normalize, load_all_vectors
from .spaces import get_space_id


def _record(mf, kind, a_ref, b_ref, method, value, n, detail=None):
    mf.conn.execute(
        "INSERT INTO correlations(kind, a_ref, b_ref, method, value, n, detail, created_at) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (kind, a_ref, b_ref, method,
         float(value) if value is not None and np.isfinite(value) else None,
         int(n), json.dumps(detail or {}), utcnow()))
    mf.conn.commit()


def _distance_pearson(A: np.ndarray, B: np.ndarray) -> float:
    Da, Db = 1.0 - A @ A.T, 1.0 - B @ B.T
    iu = np.triu_indices(A.shape[0], 1)
    return float(np.corrcoef(Da[iu], Db[iu])[0, 1])


def clustering_agreement(mf: MindFile, clustering_a: int, clustering_b: int,
                         a_ref: str | None = None, b_ref: str | None = None) -> dict | None:
    a, b = assignments(mf, clustering_a), assignments(mf, clustering_b)
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return None
    ari = adjusted_rand_score([a[i] for i in shared], [b[i] for i in shared])
    _record(mf, "clustering_ari", a_ref or f"clustering:{clustering_a}",
            b_ref or f"clustering:{clustering_b}", "adjusted_rand_index", ari, len(shared),
            {"clustering_a": clustering_a, "clustering_b": clustering_b})
    return {"ari": float(ari), "n": len(shared)}


def cross_space_agreement(mf: MindFile, space_a: str, space_b: str) -> dict | None:
    ida, ma = load_all_vectors(mf, space_a)
    idb, mb = load_all_vectors(mf, space_b)
    shared = sorted(set(ida.tolist()) & set(idb.tolist()))
    if len(shared) < 3:
        return None
    ra = {int(ida[i]): i for i in range(ida.size)}
    rb = {int(idb[i]): i for i in range(idb.size)}
    A = _l2_normalize(np.stack([ma[ra[i]] for i in shared]))
    B = _l2_normalize(np.stack([mb[rb[i]] for i in shared]))
    r = _distance_pearson(A, B)
    _record(mf, "cross_space", f"space:{space_a}", f"space:{space_b}",
            "distance_pearson", r, len(shared))
    return {"r": r, "n": len(shared)}


def feature_correlation(mf: MindFile, space: str) -> dict | None:
    _, mat = load_all_vectors(mf, space)
    if mat.shape[0] < 3 or mat.shape[1] < 2:
        return None
    C = np.corrcoef(mat.T)
    iu = np.triu_indices(C.shape[0], 1)
    mean_abs = float(np.nanmean(np.abs(C[iu])))
    _record(mf, "feature_corr", f"space:{space}", None, "pearson_matrix",
            mean_abs, mat.shape[0], {"dim": int(mat.shape[1])})
    return {"mean_abs_offdiag": mean_abs, "dim": int(mat.shape[1])}


def subject_cross_modality(mf: MindFile, space_a: str, space_b: str) -> dict | None:
    def subj_vectors(space):
        sid = get_space_id(mf, space)
        out = {}
        for r in mf.conn.execute(
            "SELECT i.subject_id, v.embedding, v.dim FROM items i "
            "JOIN vectors v ON v.item_id = i.id WHERE v.space_id = ? "
            "AND i.subject_id IS NOT NULL ORDER BY i.id", (sid,)):
            out.setdefault(r["subject_id"], _vec.unpack(r["embedding"], r["dim"]))
        return out

    va, vb = subj_vectors(space_a), subj_vectors(space_b)
    shared = sorted(set(va) & set(vb))
    if len(shared) < 3:
        return None
    A = _l2_normalize(np.stack([va[s] for s in shared]))
    B = _l2_normalize(np.stack([vb[s] for s in shared]))
    r = _distance_pearson(A, B)
    _record(mf, "subject_cross_modality", f"space:{space_a}", f"space:{space_b}",
            "subject_distance_pearson", r, len(shared))
    return {"r": r, "subjects": len(shared)}


def list_correlations(mf: MindFile) -> list[dict]:
    return [dict(r) for r in mf.conn.execute(
        "SELECT kind, a_ref, b_ref, method, value, n FROM correlations ORDER BY id")]
