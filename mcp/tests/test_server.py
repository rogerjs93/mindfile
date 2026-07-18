"""Tests for the mindfile MCP tools (embed the mindfile library; no server needed)."""
import json

from mindfile_mcp import server


def test_create_fill_summarize_search_checkpoint_audit(tmp_path):
    p = str(tmp_path / "demo.gnos")

    r = server.create_mindfile(p, modality="generic")
    assert "error" not in r, r
    assert r["created"] == p

    # a couple of "technical files" to fold into the container
    (tmp_path / "metrics.json").write_text(json.dumps({"auc": 0.83, "n": 57, "task": "rest"}))
    (tmp_path / "notes.txt").write_text("resting-state functional connectivity summary and QC")
    r = server.add_files(p, [str(tmp_path / "metrics.json"), str(tmp_path / "notes.txt")])
    assert "error" not in r, r
    assert r["added"] == 2
    assert set(r["files"]) == {"metrics.json", "notes.txt"}

    s = server.summarize(p)
    assert "error" not in s, s
    assert isinstance(s["text"], str) and s["text"]

    res = server.search(p, "connectivity", k=3)
    assert "error" not in res, res
    assert isinstance(res["results"], list)

    cp = server.checkpoint(p, "v1")
    assert "error" not in cp, cp

    a = server.audit(p)
    assert "error" not in a, a

    i = server.info(p)
    assert i.get("is_mindfile") is True
    assert i["counts"]


def test_add_files_reports_missing(tmp_path):
    p = str(tmp_path / "demo2.gnos")
    server.create_mindfile(p)
    r = server.add_files(p, [str(tmp_path / "nope.json")])
    assert "error" in r


def test_info_rejects_non_mindfile(tmp_path):
    f = tmp_path / "plain.txt"
    f.write_text("not a mindfile")
    assert "error" in server.info(str(f))
