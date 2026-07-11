import json

from mindfile import MindFile, findings, labs, modality, render
from mindfile.modality import TabularProcessor


def test_default_labs_processor_is_stateless(tmp_path):
    # regression: the shared registered labs processor must NOT lock feature_names
    # to the first file's columns (that silently corrupted other files' rows).
    a = MindFile.create(tmp_path / "a.gnos", modality="labs")
    modality.ingest(a, "labs", {"crp": 9.0, "wbc": 12.0}, subject_id="A")
    a.close()

    b = MindFile.create(tmp_path / "b.gnos", modality="labs")
    ib = modality.ingest(b, "labs", {"sodium": 140.0, "glucose": 99.0, "crp": 3.0}, subject_id="B")
    row = json.loads(b.conn.execute("SELECT payload FROM items WHERE id=?", (ib,)).fetchone()[0])
    b.close()
    assert set(row) == {"sodium", "glucose", "crp"}          # B keeps its OWN columns
    assert "wbc" not in row                                    # not forced into A's layout
    assert all(v == v for v in row.values())                  # no NaN from a locked schema


def test_reference_flags():
    assert labs.flag("crp", 12) == "high"
    assert labs.flag("crp", 2) == "normal"
    assert labs.flag("egfr", 40) == "low"          # one-sided low marker
    assert labs.flag("troponin_t", 0.5) == "high"  # alias resolves
    assert labs.flag("unknown_marker", 5) is None   # unknown analyte
    assert labs.flag("crp", float("nan")) is None    # non-finite


def test_labs_card_and_findings(mind_path):
    mf = MindFile.create(mind_path, modality="labs")
    proc = TabularProcessor(["crp", "troponin", "sodium"])
    i1 = proc.ingest(mf, {"crp": 12, "troponin": 0.5, "sodium": 140}, subject_id="P1")
    proc.ingest(mf, {"crp": 30, "troponin": 0.01, "sodium": 150}, subject_id="P2")

    c = render.card(mf, i1)
    assert "crp 12 [HIGH]" in c and "sodium 140" in c and "[HIGH]" in c.split("sodium")[0]

    stmts = findings.labs_abnormalities(mf)
    assert "2/2 subjects have high crp" in stmts          # both abnormal
    assert "1/2 subjects have high troponin" in stmts      # only P1
    assert "1/2 subjects have high sodium" in stmts        # only P2 (150>145)
    # recorded and deduped
    kinds = {f["kind"] for f in findings.list_findings(mf)}
    assert "labs_abnormality" in kinds
    n1 = len(findings.list_findings(mf))
    findings.labs_abnormalities(mf)
    assert len(findings.list_findings(mf)) == n1          # idempotent
    mf.close()
