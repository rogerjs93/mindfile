"""Lazy / virtual derived vectors — store a recipe, resolve on call.

A *virtual* vector is a ``vector_defs`` recipe with no materialized row in
``vectors``. It is computed only when accessed (``resolve``), and may be cached
(``materialize``) or dropped again (``dematerialize``) to trade compute for
storage — the "dynamic value until it's called" idea.

Recipe kinds:
* ``fused`` / ``constellation`` — recompute from sources (deterministic, exact).
* ``compressed``                — a lossless code (zlib of the float bytes) in
  ``vector_codes``; reconstructs byte-identically. Only *derived* spaces may be
  compacted, so primary originals stay exact.

Recipes resolve to bytes identical to the eager computation, which keeps lazy
values verifiable; ``source_sha256`` fingerprints the inputs so drift is
detectable (``verify_defs``) even though the output was never stored.
"""

from __future__ import annotations

import hashlib
import json
import zlib

import numpy as np

from . import _vec
from .file import MindFile, utcnow
from .spaces import get_or_create_space, get_space_id


# -- source fingerprinting (provenance, not output bytes) ------------------
def _source_fingerprint(mf: MindFile, kind: str, item_id: int, space_id: int, recipe: dict) -> str:
    h = hashlib.sha256()
    # Source hashes are recomputed from the actual bytes (not the stored hash
    # columns) so byte tampering that leaves a column stale is still caught.
    if kind == "fused":
        for src in recipe["sources"]:
            sid = get_space_id(mf, src["space"])
            r = mf.conn.execute(
                "SELECT embedding FROM vectors WHERE item_id=? AND space_id=?",
                (src["item_id"], sid)).fetchone()
            v = _vec.sha256_bytes(r["embedding"]) if r else "MISSING"
            h.update(f"{src['item_id']}\t{sid}\t{v}\n".encode())
    elif kind == "constellation":
        sid = get_space_id(mf, recipe["semantic_space"])
        r = mf.conn.execute(
            "SELECT embedding FROM vectors WHERE item_id=? AND space_id=?",
            (item_id, sid)).fetchone()
        v = _vec.sha256_bytes(r["embedding"]) if r else "MISSING"
        h.update(f"sem\t{item_id}\t{sid}\t{v}\n".encode())
        ax = {a["axis"]: a["value"] for a in mf.conn.execute(
            "SELECT axis, value FROM item_axes WHERE item_id=?", (item_id,))}
        for a in recipe["axes"]:
            h.update(f"{a}={ax.get(a)}\n".encode())
    elif kind == "compressed":
        r = mf.conn.execute(
            "SELECT code FROM vector_codes WHERE item_id=? AND space_id=?",
            (item_id, space_id)).fetchone()
        h.update((_vec.sha256_bytes(r["code"]) if r else "MISSING").encode())
    elif kind == "signal_feature":
        r = mf.conn.execute(
            "SELECT raw_sha256 FROM signals WHERE item_id=?", (item_id,)).fetchone()
        h.update((r["raw_sha256"] if r else "MISSING").encode())
    return h.hexdigest()


# -- resolvers (deterministic; must match the eager computation) -----------
def _resolve_fused(mf, item_id, space_id, recipe):
    from .fuse import ConcatFusion
    fusion = ConcatFusion(weights=recipe.get("params", {}).get("weights"),
                          normalize=recipe.get("params", {}).get("normalize", "l2"),
                          id=recipe["method"])
    parts = [(src["space"], resolve(mf, src["item_id"], src["space"])) for src in recipe["sources"]]
    return fusion.combine(parts)


def _resolve_constellation(mf, item_id, space_id, recipe):
    sem = resolve(mf, item_id, recipe["semantic_space"])
    sem_n = sem / (np.linalg.norm(sem) or 1.0)
    ax = {a["axis"]: a["value"] for a in mf.conn.execute(
        "SELECT axis, value FROM item_axes WHERE item_id=?", (item_id,))}
    coords = np.array([ax[a] for a in recipe["axes"]], dtype=np.float32)
    mean = np.array(recipe["mean"], dtype=np.float32)
    std = np.array(recipe["std"], dtype=np.float32)
    coords_n = (coords - mean) / std * float(recipe["coord_weight"])
    return np.concatenate([sem_n, coords_n]).astype(np.float32)


def _resolve_compressed(mf, item_id, space_id, recipe):
    r = mf.conn.execute(
        "SELECT code, raw_dim FROM vector_codes WHERE item_id=? AND space_id=?",
        (item_id, space_id)).fetchone()
    if r is None:
        raise KeyError(f"no code for item {item_id} in space_id {space_id}")
    return _vec.unpack(zlib.decompress(r["code"]), r["raw_dim"])


_RESOLVERS = {"fused": _resolve_fused, "constellation": _resolve_constellation,
              "compressed": _resolve_compressed}


def register_resolver(kind: str, fn) -> None:
    """Register an extra resolver ``fn(mf, item_id, space_id, recipe)`` for ``kind``.

    Lets other modules (e.g. signal.py) add recipe kinds without lazy.py importing
    them, avoiding import cycles.
    """
    _RESOLVERS[kind] = fn


# -- public API ------------------------------------------------------------
def define(mf, item_id, space, kind, recipe, *, dim, embedder_id=None,
           deterministic=True, derived=False):
    """Register a virtual vector (a recipe) for ``item_id`` in ``space``."""
    sid = get_or_create_space(mf, space, dim, embedder_id=embedder_id or kind, derived=derived)
    fp = _source_fingerprint(mf, kind, item_id, sid, recipe)
    mf.conn.execute(
        "INSERT INTO vector_defs(item_id, space_id, kind, recipe, source_sha256, "
        "deterministic, created_at) VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(item_id, space_id) DO UPDATE SET kind=excluded.kind, "
        "recipe=excluded.recipe, source_sha256=excluded.source_sha256",
        (item_id, sid, kind, json.dumps(recipe), fp, 1 if deterministic else 0, utcnow()))
    mf.conn.commit()
    return sid


def is_virtual(mf, item_id, space):
    sid = get_space_id(mf, space)
    if sid is None:
        return False
    has_def = mf.conn.execute(
        "SELECT 1 FROM vector_defs WHERE item_id=? AND space_id=?", (item_id, sid)).fetchone()
    has_vec = mf.conn.execute(
        "SELECT 1 FROM vectors WHERE item_id=? AND space_id=?", (item_id, sid)).fetchone()
    return has_def is not None and has_vec is None


def resolve(mf, item_id, space, *, materialize=False):
    """Return the vector for ``item_id`` in ``space``: concrete if present, else
    computed from its recipe (optionally caching it)."""
    sid = get_space_id(mf, space)
    if sid is None:
        raise ValueError(f"no space {space!r}")
    row = mf.conn.execute(
        "SELECT embedding, dim FROM vectors WHERE item_id=? AND space_id=?",
        (item_id, sid)).fetchone()
    if row is not None:
        return _vec.unpack(row["embedding"], row["dim"])

    d = mf.conn.execute(
        "SELECT kind, recipe FROM vector_defs WHERE item_id=? AND space_id=?",
        (item_id, sid)).fetchone()
    if d is None:
        raise KeyError(f"no concrete vector or recipe for item {item_id} in space {space!r}")
    vec = _RESOLVERS[d["kind"]](mf, item_id, sid, json.loads(d["recipe"]))
    if materialize:
        vbytes = _vec.pack(vec)
        mf.conn.execute(
            "INSERT OR REPLACE INTO vectors(item_id, space_id, embedder_id, dim, "
            "embedding, vec_sha256) VALUES(?,?,?,?,?,?)",
            (item_id, sid, d["kind"], len(vec), vbytes, _vec.sha256_bytes(vbytes)))
        mf.conn.commit()
    return vec


def materialize(mf, item_id, space):
    return resolve(mf, item_id, space, materialize=True)


def materialize_space(mf, space):
    """Resolve+cache every virtual vector in ``space``. Returns count materialized."""
    sid = get_space_id(mf, space)
    if sid is None:
        return 0
    pending = [r[0] for r in mf.conn.execute(
        "SELECT item_id FROM vector_defs WHERE space_id=? AND item_id NOT IN "
        "(SELECT item_id FROM vectors WHERE space_id=?)", (sid, sid))]
    for it in pending:
        materialize(mf, it, space)
    if pending:
        mf.record("materialize", {"space": space, "count": len(pending)})
        mf.conn.commit()
    return len(pending)


def dematerialize(mf, item_id, space):
    """Drop the cached bytes but keep the recipe (reclaim space, stay resolvable)."""
    sid = get_space_id(mf, space)
    if not mf.conn.execute(
            "SELECT 1 FROM vector_defs WHERE item_id=? AND space_id=?", (item_id, sid)).fetchone():
        raise ValueError("refusing to dematerialize: no recipe to recompute from")
    mf.conn.execute("DELETE FROM vectors WHERE item_id=? AND space_id=?", (item_id, sid))
    mf.conn.commit()


# -- lossless compaction (derived spaces only) -----------------------------
def compact(mf, item_id, space):
    """Replace a materialized derived vector with a lossless compressed code.

    Refuses primary spaces, so originals are never compacted. The vector becomes
    virtual (kind 'compressed') and reconstructs byte-identically on access.
    """
    sid = get_space_id(mf, space)
    if sid is None:
        raise ValueError(f"no space {space!r}")
    drow = mf.conn.execute("SELECT derived FROM spaces WHERE id=?", (sid,)).fetchone()
    if not drow or not drow["derived"]:
        raise ValueError(f"refusing to compact non-derived space {space!r} (keep primaries exact)")
    v = mf.conn.execute(
        "SELECT embedding, dim FROM vectors WHERE item_id=? AND space_id=?",
        (item_id, sid)).fetchone()
    if v is None:
        return False
    raw = bytes(v["embedding"])
    code = zlib.compress(raw, 9)
    mf.conn.execute(
        "INSERT OR REPLACE INTO vector_codes(item_id, space_id, codec, code, raw_dim, "
        "raw_sha256, code_sha256, created_at) VALUES(?,?,?,?,?,?,?,?)",
        (item_id, sid, "zlib", code, v["dim"], _vec.sha256_bytes(raw),
         _vec.sha256_bytes(code), utcnow()))
    define(mf, item_id, space, "compressed", {"codec": "zlib"}, dim=v["dim"])
    mf.conn.execute("DELETE FROM vectors WHERE item_id=? AND space_id=?", (item_id, sid))
    mf.conn.commit()
    return True


def compact_space(mf, space):
    """Compact every materialized vector in a derived ``space``. Returns count."""
    sid = get_space_id(mf, space)
    if sid is None:
        return 0
    items = [r[0] for r in mf.conn.execute(
        "SELECT item_id FROM vectors WHERE space_id=?", (sid,))]
    n = sum(1 for it in items if compact(mf, it, space))
    if n:
        mf.record("compact", {"space": space, "count": n})
        mf.conn.commit()
    return n


def list_defs(mf):
    return [dict(r) for r in mf.conn.execute(
        "SELECT vd.item_id, s.name AS space, vd.kind, "
        "(vd.item_id IN (SELECT item_id FROM vectors v WHERE v.space_id=vd.space_id)) AS materialized "
        "FROM vector_defs vd JOIN spaces s ON s.id=vd.space_id ORDER BY vd.item_id")]


def verify_defs(mf):
    """Confirm each recipe's inputs are still byte-identical to def time."""
    drifted = []
    for r in mf.conn.execute(
            "SELECT item_id, space_id, kind, recipe, source_sha256 FROM vector_defs"):
        cur = _source_fingerprint(mf, r["kind"], r["item_id"], r["space_id"], json.loads(r["recipe"]))
        if cur != r["source_sha256"]:
            drifted.append({"item_id": r["item_id"], "kind": r["kind"]})
    n = mf.conn.execute("SELECT COUNT(*) FROM vector_defs").fetchone()[0]
    return {"status": "ok" if not drifted else "corrupted", "checked": n, "drifted": drifted}
