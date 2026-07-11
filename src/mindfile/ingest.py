"""Ingest: add data, identity, and vectors to a ``.mind`` file.

* ``add_items``        — embed payloads with an embedder into a space.
* ``add_vectors``      — store precomputed vectors into a space.
* ``add_space_vectors``— attach a *second* embedding (another space) to items
                         that already exist (the "several ways of embedding").

All three INSERT only; existing rows are never mutated. Every payload and vector
is sha256-hashed at write time. Identity columns (subject/session/study/modality/
acquired_at) let the same items later be analysed as a cohort, a longitudinal
series, or a multi-modal patient bundle.
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from . import _vec
from .file import MindFile, utcnow
from .spaces import DEFAULT_SPACE, get_or_create_space


def _broadcast(val, n: int) -> list:
    """A scalar -> repeated n times; a sequence -> validated length n; None -> [None]*n."""
    if val is None:
        return [None] * n
    if isinstance(val, (str, bytes)) or not isinstance(val, (list, tuple, np.ndarray)):
        return [val] * n
    if len(val) != n:
        raise ValueError(f"expected {n} values, got {len(val)}")
    return list(val)


def _normalize_payload(payload, payload_kind: str) -> tuple[bytes | None, str]:
    if payload is None:
        return None, payload_kind
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload), payload_kind
    return str(payload).encode("utf-8"), payload_kind


def _insert(
    mf: MindFile,
    *,
    vectors: np.ndarray,
    space: str,
    embedder_id: str,
    payloads: Sequence | None,
    external_ids,
    subject_ids,
    session_ids,
    study_ids,
    modalities,
    acquired_ats,
    metas,
    payload_kind: str,
) -> list[int]:
    vectors = np.asarray(vectors, dtype=np.float32)
    n, dim = vectors.shape
    space_id = get_or_create_space(mf, space, dim, embedder_id=embedder_id)
    now = utcnow()

    ext = _broadcast(external_ids, n)
    subj = _broadcast(subject_ids, n)
    sess = _broadcast(session_ids, n)
    study = _broadcast(study_ids, n)
    mods = _broadcast(modalities if modalities is not None else (mf.modality or "generic"), n)
    acq = _broadcast(acquired_ats, n)
    meta_list = _broadcast(metas, n)
    pay = payloads if payloads is not None else [None] * n

    new_ids: list[int] = []
    for i in range(n):
        blob, kind = _normalize_payload(pay[i], payload_kind)
        psha = _vec.sha256_bytes(blob) if blob is not None else None
        cur = mf.conn.execute(
            "INSERT INTO items(external_id, subject_id, session_id, study_id, modality, "
            "acquired_at, payload, payload_kind, payload_sha256, meta, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (ext[i], subj[i], sess[i], study[i], mods[i], acq[i], blob, kind, psha,
             json.dumps(meta_list[i]) if meta_list[i] is not None else None, now),
        )
        item_id = cur.lastrowid
        vbytes = _vec.pack(vectors[i])
        mf.conn.execute(
            "INSERT INTO vectors(item_id, space_id, embedder_id, dim, embedding, vec_sha256) "
            "VALUES(?,?,?,?,?,?)",
            (item_id, space_id, embedder_id, dim, vbytes, _vec.sha256_bytes(vbytes)),
        )
        new_ids.append(item_id)

    mf.record("ingest", {"added": n, "space": space, "embedder_id": embedder_id, "dim": dim})
    mf.conn.commit()
    return new_ids


def add_items(
    mf: MindFile,
    payloads: Sequence,
    *,
    space: str = DEFAULT_SPACE,
    embedder=None,
    external_ids=None,
    subject_ids=None,
    session_ids=None,
    study_ids=None,
    modalities=None,
    acquired_ats=None,
    metas=None,
    payload_kind: str = "inline",
) -> list[int]:
    """Embed ``payloads`` into ``space`` using ``embedder`` (or ``mf.embedder``)."""
    embedder = embedder or mf.embedder
    if embedder is None:
        raise RuntimeError("no embedder; pass embedder= or use add_vectors() for precomputed")
    vectors = embedder.embed(payloads)
    return _insert(
        mf, vectors=vectors, space=space, embedder_id=embedder.id, payloads=payloads,
        external_ids=external_ids, subject_ids=subject_ids, session_ids=session_ids,
        study_ids=study_ids, modalities=modalities, acquired_ats=acquired_ats,
        metas=metas, payload_kind=payload_kind,
    )


def add_vectors(
    mf: MindFile,
    vectors: np.ndarray,
    *,
    space: str = DEFAULT_SPACE,
    embedder_id: str = "precomputed",
    payloads=None,
    external_ids=None,
    subject_ids=None,
    session_ids=None,
    study_ids=None,
    modalities=None,
    acquired_ats=None,
    metas=None,
    payload_kind: str = "inline",
) -> list[int]:
    """Store precomputed ``vectors`` into ``space``."""
    return _insert(
        mf, vectors=vectors, space=space, embedder_id=embedder_id, payloads=payloads,
        external_ids=external_ids, subject_ids=subject_ids, session_ids=session_ids,
        study_ids=study_ids, modalities=modalities, acquired_ats=acquired_ats,
        metas=metas, payload_kind=payload_kind,
    )


def add_space_vectors(
    mf: MindFile,
    item_ids: Sequence[int],
    vectors: np.ndarray,
    *,
    space: str,
    embedder_id: str = "precomputed",
    derived: bool = False,
) -> None:
    """Attach a *new embedding space* to items that already exist (write-once).

    This is how an item gets a second view, e.g. a connectivity vector alongside
    its band-power vector. Use ``add_items``/``add_vectors`` with ``embedder=`` to
    compute them, then this to attach precomputed ones.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape[0] != len(item_ids):
        raise ValueError("item_ids and vectors length mismatch")
    space_id = get_or_create_space(mf, space, vectors.shape[1], embedder_id=embedder_id,
                                   derived=derived)
    for item_id, vec in zip(item_ids, vectors):
        if mf.conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone() is None:
            raise ValueError(f"item {item_id} does not exist")
        if mf.conn.execute(
            "SELECT 1 FROM vectors WHERE item_id = ? AND space_id = ?", (item_id, space_id)
        ).fetchone():
            raise ValueError(f"item {item_id} already has a vector in space {space!r} (write-once)")
        vbytes = _vec.pack(vec)
        mf.conn.execute(
            "INSERT INTO vectors(item_id, space_id, embedder_id, dim, embedding, vec_sha256) "
            "VALUES(?,?,?,?,?,?)",
            (int(item_id), space_id, embedder_id, vectors.shape[1], vbytes,
             _vec.sha256_bytes(vbytes)),
        )
    mf.record("embed", {"space": space, "added": len(item_ids), "embedder_id": embedder_id})
    mf.conn.commit()
