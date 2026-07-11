"""The ``Embedder`` protocol: anything that turns payloads into vectors.

Keeping this a structural ``Protocol`` (not a base class) means new modalities
plug in without importing mindfile internals — the format stays
modality-agnostic. An embedder must expose a stable ``id`` (recorded in the
file so vectors are reproducible/auditable) and a fixed ``dim``.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    id: str
    dim: int

    def embed(self, payloads: Sequence) -> np.ndarray:
        """Return an (n, dim) float32 array, one row per payload."""
        ...
