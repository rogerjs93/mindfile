import sqlite3

from mindfile import MindFile, is_mindfile
from mindfile.version import APPLICATION_ID


def test_created_file_carries_magic_number(mind_path):
    MindFile.create(mind_path).close()
    # detected by content, regardless of the file's name/extension
    assert is_mindfile(mind_path) is True
    conn = sqlite3.connect(str(mind_path))
    assert conn.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 1
    conn.close()


def test_detection_is_extension_independent(tmp_path):
    # same content under a different extension is still recognized
    p = tmp_path / "scan.weirdext"
    MindFile.create(p).close()
    assert is_mindfile(p) is True


def test_plain_sqlite_is_not_a_mindfile(tmp_path):
    p = tmp_path / "other.db"
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE t(x)")
    conn.commit(); conn.close()
    assert is_mindfile(p) is False
    assert is_mindfile(tmp_path / "does-not-exist.gnos") is False
