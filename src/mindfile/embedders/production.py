"""Production embedders — real semantic vectors, pluggable into the protocol.

The LLM/model lives OUTSIDE the file; only the vectors it produces are stored.

* ``CallableEmbedder`` — wrap any ``text -> vector`` function (a local model, an
  API client, …). Zero dependencies; you bring the model.
* ``SentenceTransformerEmbedder`` — real sentence-transformers embeddings for
  clinical text / FHIR / DICOM cards (optional ``sentence-transformers`` extra).

Pass either to a modality processor (``TextProcessor(embedder=…)`` etc.) or to
``retrieve`` for query embedding.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


class CallableEmbedder:
    def __init__(self, fn: Callable[[str], Sequence[float]], dim: int, id: str = "callable-v1"):
        self.fn = fn
        self.dim = int(dim)
        self.id = id

    def embed(self, payloads) -> np.ndarray:
        return np.array([np.asarray(self.fn(str(p)), dtype=np.float32) for p in payloads],
                        dtype=np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", id: str | None = None,
                 normalize: bool = True):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.id = id or f"st:{model_name}"
        self.normalize = normalize

    def embed(self, payloads) -> np.ndarray:
        v = self._model.encode([str(p) for p in payloads],
                               normalize_embeddings=self.normalize, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)
