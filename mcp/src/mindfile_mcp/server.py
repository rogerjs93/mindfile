"""FastMCP server for the mindfile (.gnos) file format.

Wraps the mindfile library in-process, so an MCP client can create a .mind/.gnos file, fill it
with technical files (JSON, text, ...) so the file becomes a self-contained, searchable,
summarizable container of them, then checkpoint (immutable save point) and audit (tamper check).

Uses the dependency-free HashEmbedder by default, so no ML model download is required.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from mindfile import HashEmbedder, MindFile, is_mindfile
from mindfile import audit as audit_mod
from mindfile import checkpoint as ckpt_mod
from mindfile import ingest as ingest_mod
from mindfile import query as query_mod
from mindfile import summary as summary_mod

mcp = FastMCP("mindfile")


def _emb() -> Any:
    return HashEmbedder()


def _space_kw(space: Optional[str]) -> dict:
    return {"space": space} if space else {}


# ---------------------------------------------------------------- create / fill

@mcp.tool()
def create_mindfile(path: str, modality: str = "generic", overwrite: bool = False) -> dict:
    """Create a new mindfile (.gnos) at `path`: a single self-contained, content-verifiable file
    (a SQLite/libSQL database) that will carry its own data plus an evolving semantic map.

    - modality: a label for the file's domain (e.g. 'generic', 'clinical', 'eeg').
    - overwrite: replace an existing file if present.
    """
    try:
        mf = MindFile.create(path, modality=modality, overwrite=overwrite)
        meta = mf.meta_all()
        mf.close()
        return {"created": path, "meta": meta}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@mcp.tool()
def add_files(path: str, files: list[str], space: Optional[str] = None) -> dict:
    """Add technical files (JSON, text, CSV, ...) into a mindfile as items, turning it into a
    self-contained, searchable, summarizable CONTAINER of those files.

    Each file's text content is embedded and stored with its filename and kind. Use this to fold
    a pile of `.json` (and other) technical files into one verifiable `.gnos` bundle.
    """
    try:
        payloads: list[str] = []
        ext_ids: list[str] = []
        metas: list[dict] = []
        missing: list[str] = []
        for f in files:
            p = Path(f)
            if not p.is_file():
                missing.append(f)
                continue
            payloads.append(p.read_text(encoding="utf-8", errors="replace"))
            ext_ids.append(p.name)
            metas.append({"source": str(p), "kind": (p.suffix.lstrip(".") or "text").lower()})
        if not payloads:
            return {"error": f"no readable files (missing: {missing})"}
        mf = MindFile.open(path, embedder=_emb())
        ids = ingest_mod.add_items(mf, payloads, external_ids=ext_ids, metas=metas, **_space_kw(space))
        mf.record("mcp_add_files", {"files": ext_ids})
        mf.close()
        return {"added": len(ids), "files": ext_ids, "missing": missing, "item_ids": ids}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@mcp.tool()
def add_texts(path: str, texts: list[str], external_ids: Optional[list[str]] = None,
              space: Optional[str] = None) -> dict:
    """Add raw text snippets to a mindfile as items (embedded + stored + clustered)."""
    try:
        mf = MindFile.open(path, embedder=_emb())
        ids = ingest_mod.add_items(mf, texts, external_ids=external_ids, **_space_kw(space))
        mf.record("mcp_add_texts", {"n": len(ids)})
        mf.close()
        return {"added": len(ids), "item_ids": ids}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# ------------------------------------------------------------------ read / search

@mcp.tool()
def search(path: str, query_text: str, k: int = 5, space: Optional[str] = None) -> dict:
    """Semantic search inside a mindfile: return the top-k items most similar to `query_text`."""
    try:
        mf = MindFile.open(path, embedder=_emb())
        results = query_mod.search_text(mf, query_text, k=k, **_space_kw(space))
        mf.close()
        return {"query": query_text, "results": results}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@mcp.tool()
def summarize(path: str) -> dict:
    """Summarize a mindfile: a human-readable overview plus a structured summary (counts, spaces,
    clusters). This is how the file serves as a summary of the technical files it contains."""
    try:
        mf = MindFile.open(path)
        text = summary_mod.summary_text(mf)
        structured = summary_mod.summary(mf)
        mf.close()
        return {"text": text, "summary": structured}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@mcp.tool()
def info(path: str) -> dict:
    """Basic info about a mindfile: whether it is one, item/vector counts, header metadata, and the
    last few history events."""
    try:
        if not is_mindfile(path):
            return {"error": f"{path} is not a mindfile"}
        mf = MindFile.open(path, readonly=True)
        out = {
            "is_mindfile": True,
            "counts": mf.counts(),
            "meta": mf.meta_all(),
            "history_tail": mf.history(limit=5),
        }
        mf.close()
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# ----------------------------------------------------------- integrity / save points

@mcp.tool()
def checkpoint(path: str, name: str, note: Optional[str] = None) -> dict:
    """Create an immutable save point named `name` over the file's current state (content-hashed).
    Save points are the basis for later tamper detection."""
    try:
        mf = MindFile.open(path)
        result = ckpt_mod.create_checkpoint(mf, name, note=note)
        mf.close()
        return result
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@mcp.tool()
def audit(path: str, deep: bool = False) -> dict:
    """Verify a mindfile's integrity and chain of custody: catches tampering of payloads,
    embeddings, history, and (when present) signatures. `deep=True` also checks stored waveforms."""
    try:
        mf = MindFile.open(path, readonly=True)
        result = audit_mod.audit(mf, deep=deep)
        mf.close()
        return result
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# ----------------------------------------------------------------- bridge / one-shot

@mcp.tool()
def seal_bundle(
    mind_path: str,
    files: Optional[list[str]] = None,
    from_dir: Optional[str] = None,
    pattern: str = "*.json",
    modality: str = "generic",
    checkpoint_name: Optional[str] = None,
    space: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    """Seal a set of technical output files into a verifiable .gnos bundle, in one call.

    Ideal for capturing another tool's run (e.g. a neuro-orchestrator `process_dataset` run: its
    group-stats and per-subject JSON outputs). Creates (or appends to) the mindfile, folds in the
    files as a searchable container, adds an immutable checkpoint (the "seal"), and audits it.

    Provide `files` (explicit paths) and/or `from_dir` + `pattern` (e.g. an outputs/ folder of
    JSONs, searched recursively). Returns the created path, files added, checkpoint, audit verdict,
    and a summary.
    """
    try:
        collected: list[str] = list(files or [])
        if from_dir:
            base = Path(from_dir)
            if base.is_dir():
                collected += [str(p) for p in sorted(base.rglob(pattern)) if p.is_file()]
        collected = [f for f in dict.fromkeys(collected) if Path(f).is_file()]
        if not collected:
            return {"error": "no files to seal (pass files=[...] and/or from_dir=... with a matching pattern)"}

        if overwrite or not Path(mind_path).exists():
            created = create_mindfile(mind_path, modality=modality, overwrite=overwrite)
            if "error" in created:
                return created

        added = add_files(mind_path, collected, space=space)
        if "error" in added:
            return added

        name = checkpoint_name or f"sealed-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
        cp = checkpoint(mind_path, name)
        aud = audit(mind_path)
        summ = summarize(mind_path)
        return {
            "mind_path": mind_path,
            "sealed": "error" not in cp,
            "files_added": added.get("added"),
            "files": added.get("files"),
            "checkpoint": name if "error" not in cp else cp,
            "audit": aud.get("status", aud),
            "summary_text": summ.get("text"),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def main() -> None:
    """Console entry point: run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
