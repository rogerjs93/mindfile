"""mindfile — an AI-native, self-clustering, content-verifiable file format.

A ``.mind`` file is a single SQLite (libSQL/Turso-compatible) database that
carries its own raw data *and* an evolving semantic map of itself: vectors,
clusters, a relationship graph, append-only history, and immutable save points
for integrity verification.
"""

from __future__ import annotations

from . import (agents, allocate, analyze, audit, axes, checkpoint, cluster,
               collections, constellation, ecg_features, events, evolve, findings,
               fuse, graph, index,
               ingest, labs, lazy, modality, query, radiomics, reconall, registry,
               render, retrieve, sign, signal, spaces, stats, summary, validate)
from .embedders import (CallableEmbedder, Embedder, HashEmbedder,
                        SentenceTransformerEmbedder)
from .file import MindFile, is_mindfile
from .version import EXTENSION, FORMAT_NAME, FORMAT_VERSION, SCHEMA_VERSION

__all__ = [
    "MindFile",
    "is_mindfile",
    "Embedder",
    "HashEmbedder",
    "CallableEmbedder",
    "SentenceTransformerEmbedder",
    "findings",
    "retrieve",
    "ingest",
    "index",
    "spaces",
    "collections",
    "cluster",
    "evolve",
    "graph",
    "query",
    "checkpoint",
    "validate",
    "audit",
    "analyze",
    "modality",
    "reconall",
    "fuse",
    "axes",
    "constellation",
    "stats",
    "allocate",
    "lazy",
    "signal",
    "sign",
    "registry",
    "summary",
    "render",
    "agents",
    "EXTENSION",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "SCHEMA_VERSION",
]
