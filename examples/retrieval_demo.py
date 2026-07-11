"""The 'diamond' vision: separate structured data types held intact in one file,
with CONNECTIONS created between them (nothing is altered), plus real semantic
retrieval for an external LLM.

Each patient contributes three already-structured data types (EEG band-power,
labs, a clinical note). The file:
  * embeds the notes with a REAL model (sentence-transformers),
  * clusters each modality,
  * DISCOVERS correlations across them (findings) without changing the data,
  * answers a semantic query with an LLM-ready context bundle.

Run:  python examples/retrieval_demo.py   (downloads a small model on first run)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import MindFile, cluster, findings, ingest, modality, retrieve

# subtype 0 = cardioembolic stroke, subtype 1 = healthy-ish control
NOTES = {
    0: ["acute ischemic stroke, left MCA, dysarthria and right arm weakness",
        "cardioembolic infarct, atrial fibrillation, elevated troponin"],
    1: ["routine follow-up, no acute complaints, normal neurological exam",
        "healthy volunteer, unremarkable study, no deficits"],
}


def main() -> None:
    try:
        from mindfile.embedders import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"sentence-transformers unavailable ({e}); install `[.auth]`-style extra or run offline-cached.")
        return

    out = Path(tempfile.gettempdir()) / "diamond.gnos"
    rng = np.random.default_rng(0)
    mf = MindFile.create(out, modality="multimodal", overwrite=True)
    labs = modality.TabularProcessor(["crp", "troponin"], modality="labs")
    text = modality.TextProcessor(embedder=emb)   # REAL semantic embeddings

    print("Ingest 8 patients: EEG band-power + labs + a clinical note (3 structured types each)")
    for i in range(8):
        sub = i % 2
        ingest.add_vectors(mf, rng.normal(sub * 3, 0.3, (1, 10)).astype("float32"),
                           space="bandpower", subject_ids=f"P{i}", modalities="eeg")
        labs.ingest(mf, {"crp": 2 + sub * 25, "troponin": 0.1 + sub * 3}, subject_id=f"P{i}")
        text.ingest(mf, NOTES[sub][i // 2 % 2], subject_id=f"P{i}")

    cluster.run_clustering(mf, "bandpower", k=2)
    cluster.run_clustering(mf, "labs", k=2)

    print("\nThe file DISCOVERS connections across the structured data (data unchanged):")
    for f in findings.discover(mf):
        print(f"  - {f['statement']}")

    print("\nSemantic query -> LLM context bundle:")
    print("  query: 'cardiac source of stroke'")
    for h in retrieve.retrieve(mf, "cardiac source of stroke", emb, space="text", k=3):
        print(f"    ({h['score']:.2f}) {h['card']}")

    mf.close()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
