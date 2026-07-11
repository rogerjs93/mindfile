"""Cross-file registry — a study file that references and verifies other ``.mind``
files (e.g. one registry over many signed patient files).

``link_file`` pins a referenced file's identity (``file_uuid``), its exact state
(``state_root``) and optionally its signing key. ``verify_refs`` re-opens each
referenced file **read-only** (never mutating it) and reports whether it is still
the same file, in the same state, signed by the same key — catching a file that
went missing, was substituted, drifted since it was registered, or lost its valid
signature.
"""

from __future__ import annotations

from pathlib import Path

from . import sign as _sign
from .file import MindFile, utcnow


def _resolve(mf: MindFile, path: str, base_dir=None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(base_dir) if base_dir else mf.path.parent) / p


def _latest_signature(ref: MindFile):
    row = ref.conn.execute(
        "SELECT key_id, public_key FROM signatures ORDER BY id DESC LIMIT 1").fetchone()
    return (row["key_id"], row["public_key"]) if row else (None, None)


def link_file(mf: MindFile, path: str, *, role: str | None = None,
              require_signed: bool = False, expect_key_id: str | None = None,
              note: str | None = None, base_dir=None) -> dict:
    """Reference another ``.mind`` file, pinning its identity, state, and key."""
    target = _resolve(mf, path, base_dir)
    ref = MindFile.open(target, readonly=True)
    try:
        uuid = ref.meta_get("file_uuid")
        root = _sign.state_root(ref)
        key_id, pub = _latest_signature(ref)
        if require_signed and key_id is None:
            raise ValueError(f"{target} is not signed")
        if expect_key_id is not None and key_id != expect_key_id:
            raise ValueError(f"{target} signed by {key_id!r}, expected {expect_key_id!r}")
    finally:
        ref.close()

    mf.conn.execute(
        "INSERT INTO file_refs(path, file_uuid, state_root, key_id, public_key, role, "
        "added_at, note) VALUES(?,?,?,?,?,?,?,?)",
        (str(path), uuid, root, key_id, pub, role, utcnow(), note))
    mf.record("link_file", {"path": str(path), "uuid": uuid, "state_root": root, "role": role})
    mf.conn.commit()
    return {"path": str(path), "uuid": uuid, "state_root": root, "key_id": key_id, "role": role}


def verify_refs(mf: MindFile, *, base_dir=None) -> dict:
    """Re-open every referenced file (read-only) and verify it matches its pin."""
    out = []
    for r in mf.conn.execute(
            "SELECT id, path, file_uuid, state_root, key_id, role FROM file_refs ORDER BY id"):
        item = {"id": r["id"], "path": r["path"], "role": r["role"]}
        target = _resolve(mf, r["path"], base_dir)
        if not target.exists():
            item["status"] = "missing"
            out.append(item); continue
        try:
            ref = MindFile.open(target, readonly=True)
        except Exception as e:
            item["status"] = "unreadable"; item["error"] = str(e)
            out.append(item); continue
        try:
            if ref.meta_get("file_uuid") != r["file_uuid"]:
                item["status"] = "uuid_mismatch"           # a different file at this path
            elif _sign.state_root(ref) != r["state_root"]:
                item["status"] = "state_changed"           # drifted since registered
            elif r["key_id"]:
                rep = _sign.verify_signatures(ref)
                ok = any(s["key_id"] == r["key_id"] and s["valid"] and s["matches_current"]
                         for s in rep["signatures"])
                item["status"] = "ok" if ok else "signature_invalid"
            else:
                item["status"] = "ok"
        finally:
            ref.close()
        out.append(item)
    status = "ok" if all(i["status"] == "ok" for i in out) else "issues"
    return {"status": status, "refs": out}


def list_refs(mf: MindFile) -> list[dict]:
    return [dict(r) for r in mf.conn.execute(
        "SELECT id, path, file_uuid, state_root, key_id, role, added_at FROM file_refs ORDER BY id")]
