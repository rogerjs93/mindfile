"""Checkpoints — immutable save points + content verification (multi-space).

``create_checkpoint`` freezes a manifest of every item's payload hash AND every
per-space vector hash, plus a Merkle-style root over the whole manifest. Because
``items``/``vectors`` are write-once, ``verify_checkpoint`` can later re-hash the
live data and prove that adding data, attaching new embedding spaces, or
re-clustering left the originals byte-identical.
"""

from __future__ import annotations

import hashlib
import json

from . import _vec
from . import lazy as _lazy
from .file import MindFile, utcnow


def _space_names(mf: MindFile) -> dict[int, str]:
    return {r["id"]: r["name"] for r in mf.conn.execute("SELECT id, name FROM spaces")}


def _manifest_root(item_rows, vector_rows, signal_rows=(), def_rows=()) -> str:
    """Root hash over the complete frozen state: items, vectors, signals, and
    virtual-vector recipes (sorted, prefixed by type)."""
    h = hashlib.sha256()
    for item_id, psha in sorted(item_rows, key=lambda r: r[0]):
        h.update(f"I\t{item_id}\t{psha or ''}\n".encode("utf-8"))
    for item_id, space_id, vsha in sorted(vector_rows, key=lambda r: (r[0], r[1])):
        h.update(f"V\t{item_id}\t{space_id}\t{vsha or ''}\n".encode("utf-8"))
    for item_id, csha in sorted(signal_rows, key=lambda r: r[0]):
        h.update(f"S\t{item_id}\t{csha or ''}\n".encode("utf-8"))
    for item_id, space_id, ssha in sorted(def_rows, key=lambda r: (r[0], r[1])):
        h.update(f"D\t{item_id}\t{space_id}\t{ssha or ''}\n".encode("utf-8"))
    return h.hexdigest()


def create_checkpoint(
    mf: MindFile, name: str, *, note: str | None = None, clustering_id: int | None = None
) -> dict:
    """Freeze a save point named ``name`` over all items and all per-space vectors."""
    if mf.conn.execute("SELECT 1 FROM checkpoints WHERE name = ?", (name,)).fetchone():
        raise ValueError(f"checkpoint {name!r} already exists (save points are immutable)")

    item_rows = [(r["id"], r["payload_sha256"])
                 for r in mf.conn.execute("SELECT id, payload_sha256 FROM items ORDER BY id")]
    vector_rows = [(r["item_id"], r["space_id"], r["vec_sha256"]) for r in mf.conn.execute(
        "SELECT item_id, space_id, vec_sha256 FROM vectors ORDER BY item_id, space_id")]
    signal_rows = [(r["item_id"], r["raw_sha256"], r["code_sha256"]) for r in mf.conn.execute(
        "SELECT item_id, raw_sha256, code_sha256 FROM signals ORDER BY item_id")]
    def_rows = [(r["item_id"], r["space_id"], r["kind"], r["source_sha256"])
                for r in mf.conn.execute(
        "SELECT item_id, space_id, kind, source_sha256 FROM vector_defs "
        "ORDER BY item_id, space_id")]
    root = _manifest_root(item_rows, vector_rows,
                          [(s[0], s[2]) for s in signal_rows],
                          [(d[0], d[1], d[3]) for d in def_rows])

    cur = mf.conn.execute(
        "INSERT INTO checkpoints(name, created_at, item_count, vector_count, "
        "clustering_id, manifest_sha256, note) VALUES(?,?,?,?,?,?,?)",
        (name, utcnow(), len(item_rows), len(vector_rows), clustering_id, root, note),
    )
    cp_id = cur.lastrowid
    mf.conn.executemany(
        "INSERT INTO checkpoint_items(checkpoint_id, item_id, payload_sha256) VALUES(?,?,?)",
        [(cp_id, iid, psha) for iid, psha in item_rows],
    )
    mf.conn.executemany(
        "INSERT INTO checkpoint_vectors(checkpoint_id, item_id, space_id, vec_sha256) "
        "VALUES(?,?,?,?)",
        [(cp_id, iid, sid, vsha) for iid, sid, vsha in vector_rows],
    )
    mf.conn.executemany(
        "INSERT INTO checkpoint_signals(checkpoint_id, item_id, raw_sha256, code_sha256) "
        "VALUES(?,?,?,?)",
        [(cp_id, iid, rsha, csha) for iid, rsha, csha in signal_rows],
    )
    mf.conn.executemany(
        "INSERT INTO checkpoint_defs(checkpoint_id, item_id, space_id, kind, source_sha256) "
        "VALUES(?,?,?,?,?)",
        [(cp_id, iid, sid, kind, ssha) for iid, sid, kind, ssha in def_rows],
    )
    mf.record("checkpoint", {"name": name, "item_count": len(item_rows),
                             "vector_count": len(vector_rows), "signal_count": len(signal_rows),
                             "def_count": len(def_rows), "manifest_sha256": root})
    mf.conn.commit()
    return {"checkpoint_id": cp_id, "name": name, "item_count": len(item_rows),
            "vector_count": len(vector_rows), "signal_count": len(signal_rows),
            "def_count": len(def_rows), "manifest_sha256": root}


def list_checkpoints(mf: MindFile) -> list[dict]:
    rows = mf.conn.execute(
        "SELECT name, created_at, item_count, vector_count, manifest_sha256, note "
        "FROM checkpoints ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def verify_checkpoint(mf: MindFile, name: str) -> dict:
    """Recall a save point and verify live data still matches it byte-for-byte."""
    cp = mf.conn.execute(
        "SELECT id, manifest_sha256 FROM checkpoints WHERE name = ?", (name,)
    ).fetchone()
    if cp is None:
        raise ValueError(f"no checkpoint named {name!r}")

    frozen_items = mf.conn.execute(
        "SELECT item_id, payload_sha256 FROM checkpoint_items WHERE checkpoint_id = ?",
        (cp["id"],)).fetchall()
    frozen_vecs = mf.conn.execute(
        "SELECT item_id, space_id, vec_sha256 FROM checkpoint_vectors WHERE checkpoint_id = ?",
        (cp["id"],)).fetchall()
    frozen_signals = mf.conn.execute(
        "SELECT item_id, raw_sha256, code_sha256 FROM checkpoint_signals WHERE checkpoint_id = ?",
        (cp["id"],)).fetchall()
    frozen_defs = mf.conn.execute(
        "SELECT item_id, space_id, kind, source_sha256 FROM checkpoint_defs WHERE checkpoint_id = ?",
        (cp["id"],)).fetchall()
    names = _space_names(mf)

    changed: list[int] = []
    missing: list[int] = []
    changed_vectors: list[dict] = []
    changed_signals: list[dict] = []
    changed_defs: list[dict] = []

    for r in frozen_items:
        row = mf.conn.execute("SELECT payload FROM items WHERE id = ?", (r["item_id"],)).fetchone()
        if row is None:
            missing.append(r["item_id"])
            continue
        cur = _vec.sha256_bytes(row["payload"]) if row["payload"] is not None else None
        if cur != r["payload_sha256"]:
            changed.append(r["item_id"])

    for r in frozen_vecs:
        row = mf.conn.execute(
            "SELECT embedding FROM vectors WHERE item_id = ? AND space_id = ?",
            (r["item_id"], r["space_id"])).fetchone()
        if row is None:
            changed_vectors.append({"item_id": r["item_id"],
                                    "space": names.get(r["space_id"]), "issue": "missing"})
            continue
        if _vec.sha256_bytes(row["embedding"]) != r["vec_sha256"]:
            changed_vectors.append({"item_id": r["item_id"],
                                    "space": names.get(r["space_id"]), "issue": "changed"})

    # Waveforms: any change to the stored code (tamper or substitution) flips its hash.
    for r in frozen_signals:
        row = mf.conn.execute("SELECT code FROM signals WHERE item_id = ?", (r["item_id"],)).fetchone()
        if row is None:
            changed_signals.append({"item_id": r["item_id"], "issue": "missing"})
        elif _vec.sha256_bytes(row["code"]) != r["code_sha256"]:
            changed_signals.append({"item_id": r["item_id"], "issue": "changed"})

    # Virtual vectors: recompute the recipe's source fingerprint vs the frozen one.
    for r in frozen_defs:
        d = mf.conn.execute(
            "SELECT kind, recipe FROM vector_defs WHERE item_id=? AND space_id=?",
            (r["item_id"], r["space_id"])).fetchone()
        if d is None:
            changed_defs.append({"item_id": r["item_id"],
                                 "space": names.get(r["space_id"]), "issue": "missing"})
            continue
        cur_fp = _lazy._source_fingerprint(mf, d["kind"], r["item_id"], r["space_id"],
                                           json.loads(d["recipe"]))
        if cur_fp != r["source_sha256"]:
            changed_defs.append({"item_id": r["item_id"],
                                 "space": names.get(r["space_id"]), "issue": "changed"})

    recomputed = _manifest_root(
        [(r["item_id"], r["payload_sha256"]) for r in frozen_items],
        [(r["item_id"], r["space_id"], r["vec_sha256"]) for r in frozen_vecs],
        [(r["item_id"], r["code_sha256"]) for r in frozen_signals],
        [(r["item_id"], r["space_id"], r["source_sha256"]) for r in frozen_defs])
    manifest_ok = recomputed == cp["manifest_sha256"]

    ok = (not changed and not missing and not changed_vectors and not changed_signals
          and not changed_defs and manifest_ok)
    return {"name": name, "status": "ok" if ok else "corrupted",
            "checked_items": len(frozen_items), "checked_vectors": len(frozen_vecs),
            "checked_signals": len(frozen_signals), "checked_defs": len(frozen_defs),
            "changed": changed, "missing": missing, "changed_vectors": changed_vectors,
            "changed_signals": changed_signals, "changed_defs": changed_defs,
            "manifest_ok": manifest_ok, "manifest_sha256": cp["manifest_sha256"]}


def verify_integrity(mf: MindFile) -> dict:
    """Verify every live row against its stored hash columns (in-place corruption)."""
    names = _space_names(mf)
    bad_payload: list[int] = []
    bad_vector: list[dict] = []

    for r in mf.conn.execute("SELECT id, payload, payload_sha256 FROM items"):
        if r["payload"] is not None and _vec.sha256_bytes(r["payload"]) != r["payload_sha256"]:
            bad_payload.append(r["id"])

    item_ids = {r["id"] for r in mf.conn.execute("SELECT id FROM items")}
    have_vec = set()
    for r in mf.conn.execute("SELECT item_id, space_id, embedding, vec_sha256 FROM vectors"):
        have_vec.add(r["item_id"])
        if _vec.sha256_bytes(r["embedding"]) != r["vec_sha256"]:
            bad_vector.append({"item_id": r["item_id"], "space": names.get(r["space_id"])})

    no_vector = sorted(item_ids - have_vec)
    status = "ok" if not (bad_payload or bad_vector) else "corrupted"
    return {"status": status, "bad_payload": bad_payload, "bad_vector": bad_vector,
            "items_without_vectors": no_vector}
