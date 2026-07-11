"""``MindFile`` — open/create a ``.mind`` file and manage its connection,
header metadata, and append-only history log.

This is the low-level handle. Higher-level operations (ingest, cluster, evolve,
graph, query, checkpoint) live in their own modules and take a ``MindFile``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import schema as _schema
from .version import (APPLICATION_ID, EXTENSION, FORMAT_NAME, FORMAT_VERSION,
                      SCHEMA_VERSION)


def utcnow() -> str:
    """ISO-8601 UTC timestamp used for all created_at / ts fields."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MindFile:
    """A handle to a single ``.mind`` database file."""

    def __init__(self, path: str | Path, conn: sqlite3.Connection):
        self.path = Path(path)
        self.conn = conn
        self.embedder = None  # optional, attachable for re-embedding new data

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def create(
        cls,
        path: str | Path,
        modality: str = "generic",
        embedder: Any | None = None,
        overwrite: bool = False,
    ) -> "MindFile":
        """Create a new ``.mind`` file and write its header metadata."""
        path = Path(path)
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"{path} already exists (pass overwrite=True)")
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)

        conn = _connect(path)
        # Stamp the in-content magic number so the file self-identifies as a
        # mindfile regardless of its extension (SQLite header fields).
        conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        _schema.apply_schema(conn)

        mf = cls(path, conn)
        mf.embedder = embedder
        meta = {
            "format_name": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "schema_version": str(SCHEMA_VERSION),
            "file_uuid": str(uuid.uuid4()),
            "created_at": utcnow(),
            "modality": modality,
        }
        for k, v in meta.items():
            mf.meta_set(k, v)
        mf.conn.commit()
        return mf

    @classmethod
    def open(cls, path: str | Path, embedder: Any | None = None,
             readonly: bool = False) -> "MindFile":
        """Open an existing ``.mind`` file (schema-applied + version-checked).

        ``readonly=True`` opens with a read-only connection and skips schema
        application, so verifying another party's file never mutates it (used by
        the cross-file registry).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        conn = _connect(path, readonly=readonly)
        _schema.check_compatible(conn)
        mf = cls(path, conn)
        mf.embedder = embedder
        if not readonly:
            stored = _schema.stored_schema_version(conn)
            if stored is None or stored < SCHEMA_VERSION:
                changes = _schema.migrate(conn)     # create missing tables + columns
                mf.meta_set("schema_version", str(SCHEMA_VERSION))
                if stored is not None:
                    mf.record("migrate", {"from": stored, "to": SCHEMA_VERSION,
                                          "added": changes})
                mf.conn.commit()
            else:
                _schema.apply_schema(conn)          # current: ensure tables (idempotent)
        return mf

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "MindFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.conn.rollback()
            self.conn.close()

    # -- header / meta -----------------------------------------------------
    def meta_set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO mind_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def meta_get(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM mind_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def meta_all(self) -> dict[str, str]:
        return {
            r["key"]: r["value"]
            for r in self.conn.execute("SELECT key, value FROM mind_meta")
        }

    @property
    def modality(self) -> str | None:
        return self.meta_get("modality")

    # -- history (append-only, hash-chained) ------------------------------
    def _genesis_hash(self) -> str:
        """Chain seed unique to this file, so chains aren't transplantable."""
        return hashlib.sha256(
            ("mindfile-genesis:" + (self.meta_get("file_uuid") or "")).encode("utf-8")
        ).hexdigest()

    def record(self, event: str, detail: dict | None = None) -> None:
        ts = utcnow()
        detail_json = json.dumps(detail or {})
        last = self.conn.execute(
            "SELECT entry_hash FROM history ORDER BY id DESC LIMIT 1").fetchone()
        prev = last["entry_hash"] if last and last["entry_hash"] else self._genesis_hash()
        entry = hashlib.sha256(
            f"{prev}\t{ts}\t{event}\t{detail_json}".encode("utf-8")).hexdigest()
        self.conn.execute(
            "INSERT INTO history(ts, event, detail, prev_hash, entry_hash) VALUES(?,?,?,?,?)",
            (ts, event, detail_json, prev, entry),
        )
        self.meta_set("history_head", entry)

    def history(self, limit: int | None = None) -> list[dict]:
        sql = "SELECT id, ts, event, detail FROM history ORDER BY id"
        if limit:
            sql += f" DESC LIMIT {int(limit)}"
        rows = self.conn.execute(sql).fetchall()
        out = [
            {"id": r["id"], "ts": r["ts"], "event": r["event"],
             "detail": json.loads(r["detail"] or "{}")}
            for r in rows
        ]
        return list(reversed(out)) if limit else out

    # -- misc --------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        """Row counts per table — used by `mind inspect`."""
        tables = [
            "items", "vectors", "spaces", "item_axes", "collections",
            "collection_items", "clusterings", "clusters", "cluster_assignments",
            "edges", "derivations", "derivation_sources", "vector_defs",
            "vector_codes", "signals", "plans", "plan_actions", "correlations",
            "findings", "agents", "history", "checkpoints", "checkpoint_items",
            "checkpoint_vectors", "checkpoint_signals", "checkpoint_defs",
            "signatures", "file_refs", "trusted_keys",
        ]
        out: dict[str, int] = {}
        for t in tables:
            out[t] = self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        return out

    def __repr__(self) -> str:
        return f"<MindFile {self.path.name} modality={self.modality!r}>"


def is_mindfile(path: str | Path) -> bool:
    """True if ``path`` is a mindfile — detected by the in-content magic number
    (``application_id``), independent of the file's extension."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        conn = sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True)
        aid = conn.execute("PRAGMA application_id").fetchone()[0]
        conn.close()
        return aid == APPLICATION_ID
    except Exception:
        return False


def _connect(path: Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn  # no write pragmas on a read-only connection
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # Keep the artifact a single file (no -wal/-shm sidecars).
    conn.execute("PRAGMA journal_mode = DELETE")
    return conn


__all__ = ["MindFile", "utcnow", "EXTENSION"]
