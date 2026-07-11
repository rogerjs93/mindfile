import numpy as np

from mindfile import (MindFile, modality, ingest, cluster, findings, retrieve)


def _cohort(mind_path, n=6):
    """n patients, 2 subtypes, with EEG bandpower + labs that both separate by subtype."""
    mf = MindFile.create(mind_path, modality="multimodal")
    rng = np.random.default_rng(0)
    labs = modality.register(modality.TabularProcessor(["crp", "troponin"], modality="labs"))
    for i in range(n):
        sub = i % 2
        ingest.add_vectors(mf, rng.normal(sub * 3, 0.3, (1, 10)).astype("float32"),
                           space="bandpower", subject_ids=f"P{i}", modalities="eeg")
        labs.ingest(mf, {"crp": 2.0 + sub * 20, "troponin": 0.1 + sub * 2}, subject_id=f"P{i}")
    cluster.run_clustering(mf, "bandpower", k=2)
    cluster.run_clustering(mf, "labs", k=2)
    return mf


def test_cross_modal_agreement_finding(mind_path):
    mf = _cohort(mind_path)
    found = findings.discover(mf)
    kinds = {f["kind"] for f in found}
    assert "cross_modal_agreement" in kinds and "cluster_signature" in kinds
    # the two modalities separate by the same subtype -> they should agree
    agree = [f for f in found if f["kind"] == "cross_modal_agreement"][0]
    assert agree["value"] > 0.5 and "subjects" in agree["statement"]
    mf.close()


def test_cluster_signature_uses_feature_names(mind_path):
    mf = _cohort(mind_path)
    sigs = findings.cluster_signature(mf, "labs")
    assert sigs and any("crp" in s or "troponin" in s for s in sigs)   # named labs features
    mf.close()


def test_retrieve_returns_context_with_cards_and_findings(tmp_path):
    from mindfile.embedders import CallableEmbedder
    mf = MindFile.create(tmp_path / "t.gnos", modality="ehr")
    # a tiny deterministic "semantic" embedder: presence of keywords
    vocab = ["stroke", "cardiac", "normal", "headache"]
    fn = lambda s: [float(w in s.lower()) for w in vocab]
    emb = CallableEmbedder(fn, dim=len(vocab), id="kw")
    txt = modality.TextProcessor(embedder=emb)
    txt.ingest(mf, "acute ischemic stroke with aphasia", subject_id="P1")
    txt.ingest(mf, "cardiac arrest, troponin elevated", subject_id="P2")
    txt.ingest(mf, "normal study, no acute findings", subject_id="P3")

    hits = retrieve.retrieve(mf, "stroke", emb, space="text", k=1)
    assert hits and "stroke" in hits[0]["card"]
    ctx = retrieve.context(mf, "stroke", emb, space="text", k=2)
    assert "# context for query: stroke" in ctx and "relevant records" in ctx
    mf.close()
