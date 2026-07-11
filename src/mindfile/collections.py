"""Collections — named subsets of items that scope an analysis.

A collection materialises the set of items an analysis ranges over, e.g. a
cohort (``study_id=S, modality=eeg``), a longitudinal series (``subject_id=P``),
or a multi-modal patient bundle. Members are stored explicitly; the originating
filter is kept in ``definition`` so the set can be re-materialised.
"""

from __future__ import annotations

import json

from .file import MindFile, utcnow

# Item columns a collection filter may constrain.
_FILTERABLE = {"subject_id", "session_id", "study_id", "modality", "external_id"}


def _where(filt: dict) -> tuple[str, list]:
    bad = set(filt) - _FILTERABLE
    if bad:
        raise ValueError(f"unsupported filter columns: {sorted(bad)}")
    clauses = [f"{col} = ?" for col in filt]
    return (" AND ".join(clauses) if clauses else "1", list(filt.values()))


def members_matching(mf: MindFile, filt: dict) -> list[int]:
    where, params = _where(filt)
    rows = mf.conn.execute(
        f"SELECT id FROM items WHERE {where} ORDER BY id", params
    ).fetchall()
    return [r["id"] for r in rows]


def create_collection(
    mf: MindFile,
    name: str,
    *,
    kind: str = "custom",
    where: dict | None = None,
    item_ids: list[int] | None = None,
) -> int:
    """Create collection ``name`` from a filter (``where``) or explicit ``item_ids``."""
    if (where is None) == (item_ids is None):
        raise ValueError("provide exactly one of where= or item_ids=")
    ids = members_matching(mf, where) if where is not None else list(item_ids)

    cur = mf.conn.execute(
        "INSERT INTO collections(name, kind, definition, created_at) VALUES(?,?,?,?)",
        (name, kind, json.dumps(where) if where else None, utcnow()),
    )
    cid = cur.lastrowid
    mf.conn.executemany(
        "INSERT INTO collection_items(collection_id, item_id) VALUES(?,?)",
        [(cid, i) for i in ids],
    )
    mf.record("collection", {"name": name, "kind": kind, "size": len(ids)})
    mf.conn.commit()
    return cid


def collection_id(mf: MindFile, name: str) -> int | None:
    r = mf.conn.execute("SELECT id FROM collections WHERE name = ?", (name,)).fetchone()
    return r["id"] if r else None


def members(mf: MindFile, name: str) -> list[int]:
    cid = collection_id(mf, name)
    if cid is None:
        raise ValueError(f"no collection named {name!r}")
    rows = mf.conn.execute(
        "SELECT item_id FROM collection_items WHERE collection_id = ? ORDER BY item_id",
        (cid,),
    ).fetchall()
    return [r["item_id"] for r in rows]


def list_collections(mf: MindFile) -> list[dict]:
    rows = mf.conn.execute(
        "SELECT c.id, c.name, c.kind, "
        "(SELECT COUNT(*) FROM collection_items ci WHERE ci.collection_id = c.id) AS size "
        "FROM collections c ORDER BY c.id"
    ).fetchall()
    return [dict(r) for r in rows]
