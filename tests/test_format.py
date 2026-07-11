import sqlite3

import pytest

from mindfile import MindFile, SCHEMA_VERSION


def test_create_open_roundtrip(mind_path):
    mf = MindFile.create(mind_path, modality="text")
    uuid = mf.meta_get("file_uuid")
    mf.record("ingest", {"n": 1})
    mf.close()

    mf2 = MindFile.open(mind_path)
    assert mf2.meta_get("file_uuid") == uuid
    assert mf2.modality == "text"
    assert mf2.meta_get("format_name") == "mindfile"
    assert [h["event"] for h in mf2.history()] == ["ingest"]
    mf2.close()


def test_is_plain_sqlite(mind_path):
    MindFile.create(mind_path).close()
    conn = sqlite3.connect(str(mind_path))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"items", "vectors", "clusterings", "checkpoints", "history"} <= tables


def test_future_schema_rejected(mind_path):
    mf = MindFile.create(mind_path)
    mf.meta_set("schema_version", str(SCHEMA_VERSION + 1))
    mf.conn.commit()
    mf.close()
    with pytest.raises(RuntimeError):
        MindFile.open(mind_path)
