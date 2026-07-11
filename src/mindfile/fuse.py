"""Multi-modal fusion — combine per-modality vectors into a fused view, while
**conserving the originals**.

Fusion is purely additive: it creates NEW *derived* items (``origin='derived'``)
whose vector lives in a fused space, with provenance back to the source items.
Primary items and their per-modality cluster assignments are never modified, so
the original data points keep living in the same file and stay verifiable.

Each fused item records exactly which source ``(item, space)`` vectors and their
hashes produced it (``derivations`` / ``derivation_sources``), so
``verify_derivations`` can prove a fused point still corresponds to intact,
byte-identical originals.
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np

from . import _vec
from . import graph as _graph
from .file import MindFile, utcnow
from .spaces import get_or_create_space, get_space

_KEY_COLUMNS = {"subject_id", "session_id", "study_id", "external_id"}


class ConcatFusion:
    """Default fusion: L2-normalize each source vector, optionally weight per
    space, then concatenate. Fused dim = sum of source dims."""

    def __init__(self, weights: dict[str, float] | None = None,
                 normalize: str = "l2", id: str = "concat-l2-v1"):
        self.weights = weights or {}
        self.normalize = normalize
        self.id = id

    def combine(self, parts: list[tuple[str, np.ndarray]]) -> np.ndarray:
        out = []
        for name, v in parts:
            v = np.asarray(v, dtype=np.float32)
            if self.normalize == "l2":
                n = np.linalg.norm(v)
                if n:
                    v = v / n
            out.append(v * float(self.weights.get(name, 1.0)))
        return np.concatenate(out).astype(np.float32)

    def params(self) -> dict:
        return {"normalize": self.normalize, "weights": self.weights}


def _fingerprint(rows: list[tuple[int, int, str | None]]) -> str:
    h = hashlib.sha256()
    for item_id, space_id, vsha in sorted(rows, key=lambda r: (r[0], r[1])):
        h.update(f"{item_id}\t{space_id}\t{vsha or ''}\n".encode("utf-8"))
    return h.hexdigest()


def fuse(
    mf: MindFile,
    spaces: Sequence[str],
    *,
    key: Sequence[str] = ("subject_id", "session_id"),
    method: ConcatFusion | None = None,
    out_space: str = "fused",
    modality: str = "fused",
    lazy: bool = False,
) -> dict:
    """Fuse ``spaces`` into ``out_space`` for each group sharing ``key``.

    Only groups that have exactly one vector in every required space are fused;
    incomplete or ambiguous groups are skipped and reported. With ``lazy=True``
    the fused vector is stored as a recipe (resolved on access) instead of being
    materialized. Returns ``{created, skipped, out_space}``.
    """
    method = method or ConcatFusion()
    bad = set(key) - _KEY_COLUMNS
    if bad:
        raise ValueError(f"unsupported key columns: {sorted(bad)}")
    spaces = list(spaces)

    space_meta = {}
    for sname in spaces:
        sp = get_space(mf, sname)
        if sp is None:
            raise ValueError(f"no space named {sname!r}")
        space_meta[sname] = sp  # has id, dim

    keycols = ", ".join(f"i.{c}" for c in key)
    # Collect, per space, the primary items + their key values + vectors.
    groups: dict[tuple, dict[str, list]] = {}
    for sname in spaces:
        rows = mf.conn.execute(
            f"SELECT i.id AS item_id, {keycols}, v.embedding, v.vec_sha256 "
            "FROM items i JOIN vectors v ON v.item_id = i.id "
            "WHERE v.space_id = ? AND i.origin = 'primary'",
            (space_meta[sname]["id"],),
        ).fetchall()
        for r in rows:
            keytuple = tuple(r[c] for c in key)
            if any(k is None for k in keytuple):
                continue  # cannot group without a complete key
            vec = _vec.unpack(r["embedding"], space_meta[sname]["dim"])
            groups.setdefault(keytuple, {}).setdefault(sname, []).append(
                (r["item_id"], vec, r["vec_sha256"]))

    created, skipped = [], []
    for keytuple, by_space in sorted(groups.items()):
        parts, sources, ok = [], [], True
        for sname in spaces:
            entries = by_space.get(sname, [])
            if len(entries) != 1:
                ok = False
                skipped.append({"key": dict(zip(key, keytuple)), "space": sname,
                                "reason": "missing" if not entries else "ambiguous"})
                break
            item_id, vec, vsha = entries[0]
            parts.append((sname, vec))
            sources.append((item_id, space_meta[sname]["id"], vsha))
        if not ok:
            continue

        fused_vec = method.combine(parts)
        src_ids = [s[0] for s in sources]
        ident = _source_identity(mf, src_ids, key, keytuple)
        ext = "fused:" + ":".join(str(v) for v in keytuple)

        cur = mf.conn.execute(
            "INSERT INTO items(external_id, origin, subject_id, session_id, study_id, "
            "modality, acquired_at, payload_kind, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (ext, "derived", ident["subject_id"], ident["session_id"], ident["study_id"],
             modality, ident["acquired_at"], "none", utcnow()),
        )
        derived_id = cur.lastrowid

        if lazy:
            from . import lazy as _lazy
            recipe = {"method": method.id, "params": method.params(),
                      "sources": [{"item_id": sources[i][0], "space": spaces[i]}
                                  for i in range(len(spaces))]}
            out_sid = _lazy.define(mf, derived_id, out_space, "fused", recipe,
                                   dim=len(fused_vec), embedder_id=method.id, derived=True)
        else:
            out_sid = get_or_create_space(mf, out_space, len(fused_vec),
                                          embedder_id=method.id, modality=modality, derived=True)
            vbytes = _vec.pack(fused_vec)
            mf.conn.execute(
                "INSERT INTO vectors(item_id, space_id, embedder_id, dim, embedding, vec_sha256) "
                "VALUES(?,?,?,?,?,?)",
                (derived_id, out_sid, method.id, len(fused_vec), vbytes, _vec.sha256_bytes(vbytes)),
            )

        fp = _fingerprint(sources)
        dcur = mf.conn.execute(
            "INSERT INTO derivations(derived_item_id, out_space_id, method, params, "
            "fusion_key, source_sha256, created_at) VALUES(?,?,?,?,?,?,?)",
            (derived_id, out_sid, method.id,
             json.dumps({"spaces": spaces, **method.params()}),
             json.dumps(dict(zip(key, keytuple))), fp, utcnow()),
        )
        deriv_id = dcur.lastrowid
        mf.conn.executemany(
            "INSERT INTO derivation_sources(derivation_id, source_item_id, space_id, vec_sha256) "
            "VALUES(?,?,?,?)",
            [(deriv_id, s[0], s[1], s[2]) for s in sources])
        for sid in src_ids:
            _graph.link(mf, derived_id, sid, kind="fused_from")
        created.append(derived_id)

    mf.record("fuse", {"out_space": out_space, "spaces": spaces, "method": method.id,
                       "created": len(created), "skipped": len(skipped)})
    mf.conn.commit()
    return {"created": created, "skipped": skipped, "out_space": out_space}


def _source_identity(mf: MindFile, src_ids: list[int], key, keytuple) -> dict:
    """Identity for a derived item: key values where present, else inherited from
    sources; acquired_at = earliest source time so fused longitudinal works."""
    rows = [dict(r) for r in mf.conn.execute(
        f"SELECT subject_id, session_id, study_id, acquired_at FROM items "
        f"WHERE id IN ({','.join('?' * len(src_ids))})", src_ids)]
    keymap = dict(zip(key, keytuple))

    def inherit(col):
        if col in keymap:
            return keymap[col]
        vals = {r[col] for r in rows if r[col] is not None}
        return next(iter(vals)) if len(vals) == 1 else None

    times = sorted(r["acquired_at"] for r in rows if r["acquired_at"])
    return {"subject_id": inherit("subject_id"), "session_id": inherit("session_id"),
            "study_id": inherit("study_id"), "acquired_at": times[0] if times else None}


def list_derivations(mf: MindFile) -> list[dict]:
    rows = mf.conn.execute(
        "SELECT d.id, d.derived_item_id, i.external_id, d.method, d.fusion_key, "
        "d.source_sha256 FROM derivations d JOIN items i ON i.id = d.derived_item_id "
        "ORDER BY d.id"
    ).fetchall()
    return [dict(r) for r in rows]


def verify_derivations(mf: MindFile) -> dict:
    """Prove every fused item is still backed by intact, identical source vectors.

    Recomputes each source's current hash and compares to the value frozen at
    fusion time. A mismatch/missing means an original changed under a fused point.
    """
    drifted = []
    for d in mf.conn.execute(
        "SELECT id, derived_item_id, source_sha256 FROM derivations"
    ).fetchall():
        srcs = mf.conn.execute(
            "SELECT source_item_id, space_id, vec_sha256 FROM derivation_sources "
            "WHERE derivation_id = ?", (d["id"],)).fetchall()
        issues = []
        for s in srcs:
            row = mf.conn.execute(
                "SELECT embedding FROM vectors WHERE item_id = ? AND space_id = ?",
                (s["source_item_id"], s["space_id"])).fetchone()
            if row is None:
                issues.append({"source_item_id": s["source_item_id"], "issue": "missing"})
            elif _vec.sha256_bytes(row["embedding"]) != s["vec_sha256"]:
                # re-hash the CURRENT bytes vs the value frozen at fusion time
                issues.append({"source_item_id": s["source_item_id"], "issue": "changed"})
        if issues:
            drifted.append({"derived_item_id": d["derived_item_id"], "issues": issues})

    return {"status": "ok" if not drifted else "corrupted",
            "checked": mf.conn.execute("SELECT COUNT(*) FROM derivations").fetchone()[0],
            "drifted": drifted}
