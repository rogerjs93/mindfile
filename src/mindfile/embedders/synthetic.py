"""A deterministic, dependency-free embedder for tests and demos.

Maps each payload to a reproducible pseudo-random vector via a seed derived
from its sha256. Same payload -> same vector, always, so it exercises the
embed -> store -> search -> hash round trip without any model download.

(For demos that need *clustered* structure with a controllable split, generate
vectors directly and use ``ingest.add_vectors`` instead — see examples/.)
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


class HashEmbedder:
    def __init__(self, dim: int = 16, id: str = "synthetic-hash-v1"):
        self.dim = int(dim)
        self.id = id

    def embed(self, payloads: Sequence) -> np.ndarray:
        out = np.empty((len(payloads), self.dim), dtype=np.float32)
        for i, p in enumerate(payloads):
            b = p if isinstance(p, (bytes, bytearray)) else str(p).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(b).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(self.dim).astype(np.float32)
        return out
