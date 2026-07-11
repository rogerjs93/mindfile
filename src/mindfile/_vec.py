"""Vector (de)serialization and content hashing helpers.

Embeddings live in the file as little-endian float32 BLOBs. These helpers are
the single chokepoint for converting between numpy arrays and those bytes, and
for hashing bytes so integrity checks are consistent everywhere.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Canonical on-disk dtype for embeddings and centroids.
_DTYPE = np.dtype("<f4")  # little-endian float32


def pack(vec) -> bytes:
    """Serialize a 1-D vector to canonical little-endian float32 bytes."""
    arr = np.ascontiguousarray(np.asarray(vec, dtype=_DTYPE))
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got shape {arr.shape}")
    return arr.tobytes()


def unpack(blob: bytes, dim: int | None = None) -> np.ndarray:
    """Deserialize float32 bytes back into a 1-D numpy array."""
    arr = np.frombuffer(blob, dtype=_DTYPE)
    if dim is not None and arr.shape[0] != dim:
        raise ValueError(f"expected dim {dim}, blob holds {arr.shape[0]}")
    return arr


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of raw bytes (used for payloads and vectors)."""
    return hashlib.sha256(data).hexdigest()


def sha256_vec(vec) -> str:
    """Hex sha256 of a vector's canonical byte representation."""
    return sha256_bytes(pack(vec))
