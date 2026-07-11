"""Embedding spaces — the "several ways of embedding" an item.

Each space is a named vector view (e.g. ``bandpower``, ``connectivity``,
``encoder``) with its own dimension and embedder. An item may carry a vector in
several spaces; clusterings and similarity edges are scoped to one space.
"""

from __future__ import annotations

from .file import MindFile, utcnow

DEFAULT_SPACE = "default"


def get_space(mf: MindFile, name: str) -> dict | None:
    r = mf.conn.execute(
        "SELECT id, name, embedder_id, dim, modality FROM spaces WHERE name = ?", (name,)
    ).fetchone()
    return dict(r) if r else None


def get_space_id(mf: MindFile, name: str) -> int | None:
    r = mf.conn.execute("SELECT id FROM spaces WHERE name = ?", (name,)).fetchone()
    return r["id"] if r else None


def get_or_create_space(
    mf: MindFile,
    name: str,
    dim: int,
    *,
    embedder_id: str | None = None,
    modality: str | None = None,
    derived: bool = False,
) -> int:
    """Return the id of space ``name``, creating it with ``dim`` if absent.

    Raises if the space exists with a different dimension (a space is fixed-dim).
    ``derived=True`` marks a computed space (fused/constellation) that may be
    losslessly compacted; primary spaces (derived=0) are never compacted.
    """
    existing = get_space(mf, name)
    if existing is not None:
        if existing["dim"] != dim:
            raise ValueError(
                f"space {name!r} has dim {existing['dim']}, got {dim}"
            )
        return existing["id"]
    cur = mf.conn.execute(
        "INSERT INTO spaces(name, embedder_id, dim, modality, derived, created_at) "
        "VALUES(?,?,?,?,?,?)",
        (name, embedder_id, int(dim), modality, 1 if derived else 0, utcnow()),
    )
    return cur.lastrowid


def list_spaces(mf: MindFile) -> list[dict]:
    rows = mf.conn.execute(
        "SELECT s.id, s.name, s.embedder_id, s.dim, s.modality, "
        "(SELECT COUNT(*) FROM vectors v WHERE v.space_id = s.id) AS n_vectors "
        "FROM spaces s ORDER BY s.id"
    ).fetchall()
    return [dict(r) for r in rows]
