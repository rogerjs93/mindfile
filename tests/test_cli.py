"""Smoke + behaviour tests for the `mind` CLI (previously untested)."""

import pytest

from mindfile import cli


def run(capsys, *argv):
    """Invoke the CLI; return (stdout, exit_code). Commands may sys.exit()."""
    code = 0
    try:
        cli.main(list(argv))
    except SystemExit as e:
        code = e.code or 0
    return capsys.readouterr().out, code


@pytest.fixture
def built(tmp_path, capsys):
    f = str(tmp_path / "t.gnos")
    lines = tmp_path / "lines.txt"
    lines.write_text("\n".join(f"clinical note {i} on patient recovery and status" for i in range(10)),
                     encoding="utf-8")
    out, code = run(capsys, "create", f, "--modality", "text")
    assert code == 0 and "created" in out
    out, code = run(capsys, "ingest", f, "--text", str(lines), "--dim", "16")
    assert code == 0 and "ingested 10 items" in out
    out, code = run(capsys, "cluster", f, "--k", "3")
    assert code == 0 and "clustering in space 'default'" in out
    return f


def test_modalities_lists_every_type(capsys):
    out, code = run(capsys, "modalities")
    assert code == 0 and "11 supported file types" in out
    for m in ("eeg", "dicom", "fhir", "labs", "genomics", "wfdb", "tabular"):
        assert m in out


def test_build_and_inspect(built, capsys):
    out, _ = run(capsys, "inspect", built)
    assert "10" in out                       # 10 items reported somewhere in the dump


def test_query_and_explain(built, capsys):
    out, code = run(capsys, "query", built, "--text", "patient recovery", "--k", "3")
    assert code == 0 and out.strip()          # returns some ranked results
    out, code = run(capsys, "explain", built, "--item", "1")
    assert code == 0


@pytest.mark.parametrize("cmd", ["summary", "validate", "audit", "findings", "history", "digest"])
def test_readonly_commands_run(built, capsys, cmd):
    out, code = run(capsys, cmd, built)
    assert code == 0 and out.strip()


def test_card(built, capsys):
    out, code = run(capsys, "card", built, "--item", "1")
    assert code == 0 and "clinical note" in out


def test_checkpoint_then_verify_ok(built, capsys):
    out, code = run(capsys, "checkpoint", built, "cp1")
    assert code == 0 and "checkpoint 'cp1' saved" in out
    out, code = run(capsys, "verify", built, "cp1")     # exits 0 when integrity ok
    assert code == 0 and '"status"' in out and "ok" in out


def test_unknown_command_errors(capsys):
    with pytest.raises(SystemExit):
        cli.main(["definitely-not-a-command"])
