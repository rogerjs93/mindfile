import numpy as np

from mindfile import MindFile, modality, lazy, render


def _patient(mind_path):
    mf = MindFile.create(mind_path, modality="multimodal")
    rng = np.random.default_rng(0)
    modality.ingest(mf, "eeg", rng.normal(0, 1, (8, 1000)).astype("float32"),
                    sfreq=500.0, subject_id="P1", session_id="t0")
    modality.ingest(mf, "text", "patient reports chest pain and dyspnea",
                    subject_id="P1", session_id="t0")
    modality.ingest(mf, "genomics", ["1:100:A:G", "7:55:G:A"], subject_id="P1")
    proc = modality.register(modality.TabularProcessor(["hb", "wbc"], modality="labs"))
    proc.ingest(mf, {"hb": 13.5, "wbc": 6.2}, subject_id="P1")
    lazy.materialize_space(mf, "bandpower")
    return mf


def test_cards_are_llm_readable(mind_path):
    mf = _patient(mind_path)
    cards = {r["modality"]: render.card(mf, r["id"]) for r in mf.conn.execute(
        "SELECT id, modality FROM items")}
    assert "rel power" in cards["eeg"] and "@500Hz" in cards["eeg"]
    assert "chest pain" in cards["text"]
    assert "variants" in cards["genomics"]
    assert "hb 13.5" in cards["labs"]
    # every card names its modality and subject
    assert all(c.startswith("[") and "P1" in c for c in cards.values())
    mf.close()


def test_digest_reads_whole_file(mind_path):
    mf = _patient(mind_path)
    text = render.digest(mf)
    assert "## records" in text
    for m in ("eeg", "text", "genomics", "labs"):
        assert f"[{m}]" in text
    mf.close()
