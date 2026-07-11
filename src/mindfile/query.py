"""Query: search the file and explain what it knows about an item.

Read-only helpers that stitch the blocks together — vector search (per space) +
cluster membership + graph neighbours + evolution history — into the "small
knowledge universe" view the format is meant to expose.
"""

from __future__ import annotations

from .cluster import latest_clustering_id
from .file import MindFile
from .graph import neighbors
from .index import search as _vector_search
from .spaces import DEFAULT_SPACE


def _item_brief(mf: MindFile, item_id: int) -> dict:
    r = mf.conn.execute(
        "SELECT id, external_id, subject_id, session_id, modality FROM items WHERE id = ?",
        (item_id,),
    ).fetchone()
    return dict(r) if r else {"id": item_id}


def cluster_of(
    mf: MindFile, item_id: int, space: str = DEFAULT_SPACE, clustering_id: int | None = None
) -> dict | None:
    """The cluster an item belongs to in ``space`` (latest run by default)."""
    clustering_id = clustering_id or latest_clustering_id(mf, space)
    if clustering_id is None:
        return None
    r = mf.conn.execute(
        "SELECT c.id, c.label, c.local_index, a.distance "
        "FROM cluster_assignments a JOIN clusters c ON c.id = a.cluster_id "
        "WHERE a.clustering_id = ? AND a.item_id = ?",
        (clustering_id, item_id),
    ).fetchone()
    return dict(r) if r else None


def search(mf: MindFile, query, k: int = 5, space: str = DEFAULT_SPACE) -> list[dict]:
    """Vector search in ``space``, enriched with identity + current cluster label."""
    out = []
    for item_id, score in _vector_search(mf, query, k=k, space=space):
        brief = _item_brief(mf, item_id)
        cl = cluster_of(mf, item_id, space)
        out.append({**brief, "score": score, "cluster": cl["label"] if cl else None})
    return out


def search_text(mf: MindFile, text: str, k: int = 5, space: str = DEFAULT_SPACE) -> list[dict]:
    if mf.embedder is None:
        raise RuntimeError("attach an embedder to search by text")
    qvec = mf.embedder.embed([text])[0]
    return search(mf, qvec, k=k, space=space)


def explain(mf: MindFile, item_id: int, k: int = 5, space: str = DEFAULT_SPACE) -> dict:
    """A compact 'what this file understands about this item' report."""
    brief = _item_brief(mf, item_id)
    cl = cluster_of(mf, item_id, space)

    related_events = []
    if cl:
        for h in mf.history():
            if h["event"] not in ("split", "merge"):
                continue
            d = h["detail"]
            ids = {d.get("parent_cluster_id"), d.get("into_cluster_id")}
            ids |= {x.get("cluster_id") for x in d.get("into", [])}
            ids |= {x.get("cluster_id") for x in d.get("from", [])}
            if cl["id"] in ids:
                related_events.append({"event": h["event"], "ts": h["ts"], "detail": d})

    nbrs = []
    for e in neighbors(mf, item_id, limit=k):
        nb = _item_brief(mf, e["dst_item_id"])
        nbrs.append({**nb, "kind": e["kind"], "weight": e["weight"]})

    # Which spaces does this item have a vector in?
    spaces = [r["name"] for r in mf.conn.execute(
        "SELECT s.name FROM vectors v JOIN spaces s ON s.id = v.space_id "
        "WHERE v.item_id = ? ORDER BY s.name", (item_id,))]

    return {"item": brief, "spaces": spaces, "cluster": cl,
            "neighbors": nbrs, "evolution": related_events}


def evolution_report(mf: MindFile) -> list[dict]:
    return [h for h in mf.history() if h["event"] in ("split", "merge")]
