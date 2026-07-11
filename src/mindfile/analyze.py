"""Analysis scopes over the same items: cohort, longitudinal, multi-modal.

These are thin compositions of the primitives (collections + spaces + clustering
+ graph). The same recordings can be viewed three ways depending on the scope:

* ``group_study``  — cluster a cohort (cross-sectional) in one space.
* ``longitudinal`` — one subject's sessions over time -> trajectory + change +
                     temporal edges.
* ``multimodal``   — one patient across modalities -> same_subject links (and a
                     per-subject view spanning spaces).
"""

from __future__ import annotations

import numpy as np

from . import collections as _collections
from . import graph as _graph
from .cluster import clusters_of, run_clustering
from .file import MindFile
from .index import load_all_vectors
from .spaces import DEFAULT_SPACE


# -- group / cohort --------------------------------------------------------
def group_study(
    mf: MindFile,
    name: str,
    *,
    where: dict,
    space: str = DEFAULT_SPACE,
    k: int | None = None,
    note: str | None = None,
) -> dict:
    """Materialise a cohort from ``where`` (e.g. {study_id, modality}) and cluster it."""
    if _collections.collection_id(mf, name) is None:
        _collections.create_collection(mf, name, kind="cohort", where=where)
    cid = run_clustering(mf, space, k=k, collection=name, note=note)
    return {"collection": name, "clustering_id": cid, "space": space,
            "clusters": clusters_of(mf, cid)}


# -- longitudinal ----------------------------------------------------------
def _subject_items_ordered(mf: MindFile, subject_id: str, modality: str | None) -> list[dict]:
    sql = ("SELECT id, session_id, acquired_at, modality FROM items "
           "WHERE subject_id = ?")
    params: list = [subject_id]
    if modality is not None:
        sql += " AND modality = ?"
        params.append(modality)
    sql += " ORDER BY (acquired_at IS NULL), acquired_at, session_id, id"
    return [dict(r) for r in mf.conn.execute(sql, params)]


def longitudinal(
    mf: MindFile,
    subject_id: str,
    *,
    space: str = DEFAULT_SPACE,
    modality: str | None = None,
    link: bool = True,
    change_threshold: float | None = None,
) -> dict:
    """Trajectory of one subject's sessions over time in ``space``.

    Returns ordered items and the cosine distance between consecutive sessions
    (a simple change signal). Optionally writes ``temporal`` edges and flags steps
    whose distance exceeds ``change_threshold``.
    """
    items = _subject_items_ordered(mf, subject_id, modality)
    ids, mat = load_all_vectors(mf, space, item_ids=[i["id"] for i in items])
    vec = {int(ids[j]): mat[j] for j in range(len(ids))}
    # A trajectory in a space ranges only over items embedded in that space, in
    # time order — otherwise items from other modalities fragment the chain.
    items = [i for i in items if i["id"] in vec]
    if len(items) < 2:
        return {"subject_id": subject_id, "order": [i["id"] for i in items], "steps": []}

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na and nb else 0.0

    steps, flagged = [], []
    name = f"longitudinal:{subject_id}"
    if _collections.collection_id(mf, name) is None:
        _collections.create_collection(mf, name, kind="longitudinal",
                                       item_ids=[i["id"] for i in items])
    for a, b in zip(items, items[1:]):
        sim = cos(vec[a["id"]], vec[b["id"]])
        dist = 1.0 - sim
        step = {"from": a["id"], "to": b["id"], "from_session": a["session_id"],
                "to_session": b["session_id"], "similarity": sim, "distance": dist}
        steps.append(step)
        if change_threshold is not None and dist > change_threshold:
            flagged.append(step)
        if link:
            _graph.link(mf, a["id"], b["id"], kind="temporal", space=space, weight=sim)

    mf.record("trajectory", {"subject_id": subject_id, "space": space,
                             "n_sessions": len(items), "flagged": len(flagged)})
    mf.conn.commit()
    return {"subject_id": subject_id, "space": space,
            "order": [i["id"] for i in items], "steps": steps, "flagged": flagged}


# -- multi-modal -----------------------------------------------------------
def multimodal(mf: MindFile, subject_id: str, *, link: bool = True) -> dict:
    """Link one patient's items across modalities (same_subject edges).

    A fused cross-modality embedding is intentionally out of scope here (Phase 2);
    this establishes the patient-level bundle and its links.
    """
    items = [dict(r) for r in mf.conn.execute(
        "SELECT id, modality, session_id FROM items WHERE subject_id = ? ORDER BY id",
        (subject_id,))]
    name = f"multimodal:{subject_id}"
    if _collections.collection_id(mf, name) is None and items:
        _collections.create_collection(mf, name, kind="multimodal",
                                       item_ids=[i["id"] for i in items])

    links = 0
    if link:
        for a in items:
            for b in items:
                if a["id"] < b["id"] and a["modality"] != b["modality"]:
                    _graph.link(mf, a["id"], b["id"], kind="same_subject")
                    _graph.link(mf, b["id"], a["id"], kind="same_subject")
                    links += 1
        mf.conn.commit()

    modalities = sorted({i["modality"] for i in items if i["modality"]})
    return {"subject_id": subject_id, "items": items, "modalities": modalities, "links": links}
