"""Schema application and version checks for ``.mind`` files."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .version import SCHEMA_VERSION

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def schema_sql() -> str:
    """Return the canonical schema DDL."""
    return _SCHEMA_PATH.read_text(encoding="utf-8")


def apply_schema(conn: sqlite3.Connection) -> None:
    """Create all tables/indexes if missing (idempotent)."""
    conn.executescript(schema_sql())


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Bring an older file up to the current schema and return the changes.

    Additive reconciliation: create any missing tables, then add any missing
    columns (with their declared default). This is complete and safe *because
    the format is additive-only* — no version ever drops, renames, or retypes a
    column — so the canonical ``schema.sql`` is the only source of truth needed.
    Existing rows get each new column's default automatically.
    """
    changes: list[str] = []
    ref = sqlite3.connect(":memory:")
    try:
        ref.executescript(schema_sql())
        ref_tables = [r[0] for r in ref.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        existing = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        # Add missing columns to tables that already exist FIRST, so the indexes
        # created by executescript below don't reference a not-yet-added column.
        for table in ref_tables:
            if table not in existing:
                continue
            have = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
            for _cid, name, ctype, notnull, dflt, _pk in ref.execute(
                    f'PRAGMA table_info("{table}")'):
                if name in have:
                    continue
                coldef = f'"{name}" {ctype}'
                if dflt is not None:
                    coldef += f" DEFAULT {dflt}"
                if notnull and dflt is not None:   # NOT NULL needs a default to backfill
                    coldef += " NOT NULL"
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN {coldef}')
                changes.append(f"{table}.{name}")
    finally:
        ref.close()
    conn.executescript(schema_sql())                 # create missing tables + indexes
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return changes


def stored_schema_version(conn: sqlite3.Connection) -> int | None:
    """Read the schema_version recorded in mind_meta, or None if absent."""
    try:
        row = conn.execute(
            "SELECT value FROM mind_meta WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return int(row[0]) if row and row[0] is not None else None


def check_compatible(conn: sqlite3.Connection) -> None:
    """Raise if the file's schema is newer than this library understands."""
    found = stored_schema_version(conn)
    if found is None:
        return
    if found > SCHEMA_VERSION:
        raise RuntimeError(
            f"file schema_version {found} is newer than supported {SCHEMA_VERSION}; "
            "upgrade the mindfile library"
        )
