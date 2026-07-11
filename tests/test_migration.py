import sqlite3

import numpy as np

from mindfile import MindFile, ingest
from mindfile.version import APPLICATION_ID, SCHEMA_VERSION


def _make_old_file(path):
    """A v2-style file: items missing all the columns added since (origin,
    subject_id, session_id, study_id, acquired_at) and most tables absent."""
    c = sqlite3.connect(str(path))
    c.execute(f"PRAGMA application_id = {APPLICATION_ID}")
    c.execute("CREATE TABLE mind_meta(key TEXT PRIMARY KEY, value TEXT)")
    c.execute("CREATE TABLE items(id INTEGER PRIMARY KEY AUTOINCREMENT, external_id TEXT, "
              "modality TEXT, payload BLOB, payload_kind TEXT DEFAULT 'inline', "
              "payload_sha256 TEXT, meta TEXT, created_at TEXT)")
    c.execute("INSERT INTO mind_meta VALUES('schema_version', '2')")
    c.execute("INSERT INTO mind_meta VALUES('file_uuid', 'old-uuid-123')")
    c.execute("INSERT INTO items(id, modality, created_at) VALUES(1, 'eeg', '2020-01-01')")
    c.commit()
    c.close()


def test_open_migrates_old_file(tmp_path):
    p = tmp_path / "old.gnos"
    _make_old_file(p)
    mf = MindFile.open(p)

    # missing columns added, existing row backfilled with the column default
    cols = {r[1] for r in mf.conn.execute("PRAGMA table_info(items)")}
    assert {"origin", "subject_id", "session_id", "study_id", "acquired_at"} <= cols
    assert mf.conn.execute("SELECT origin FROM items WHERE id=1").fetchone()[0] == "primary"

    # missing tables created; version stamped
    tables = {r[0] for r in mf.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"spaces", "vectors", "signals", "findings", "vector_defs", "history"} <= tables
    assert mf.meta_get("schema_version") == str(SCHEMA_VERSION)
    assert mf.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert any(h["event"] == "migrate" for h in mf.history())

    # and the migrated file is fully functional
    ids = ingest.add_vectors(mf, np.zeros((1, 4), "float32"), subject_ids="P1")
    assert mf.conn.execute("SELECT subject_id FROM items WHERE id=?", (ids[0],)).fetchone()[0] == "P1"
    mf.close()


def test_current_file_opens_without_migrating(tmp_path):
    MindFile.create(tmp_path / "new.gnos", modality="eeg").close()
    mf = MindFile.open(tmp_path / "new.gnos")
    assert not any(h["event"] == "migrate" for h in mf.history())
    assert mf.meta_get("schema_version") == str(SCHEMA_VERSION)
    mf.close()
