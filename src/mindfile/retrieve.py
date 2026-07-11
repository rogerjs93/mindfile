"""Retrieval — turn a question into an LLM context bundle. The LLM stays outside
the file; this hands it the relevant records + findings to reason over.

``retrieve`` semantic-searches a text-bearing space; ``context`` assembles the
summary + top records + discovered findings into one text block you pass to an
LLM (RAG over a `.gnos` file, no model embedded).
"""

from __future__ import annotations

from .index import search as _search
from .spaces import get_space, list_spaces

_TEXT_SPACES = ("text", "fhir", "dicom")


def _text_space(mf):
    names = {s["name"] for s in list_spaces(mf)}
    for n in _TEXT_SPACES:
        if n in names:
            return n
    return None


def embedder_for(mf, space):
    """Reconstruct the embedder a space was built with (so a query matches it)."""
    sp = get_space(mf, space)
    if sp is None:
        return None
    eid = sp["embedder_id"] or ""
    if eid.startswith("st:"):
        from .embedders import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(eid[3:], id=eid)
    if "hashing" in eid:
        from .modality import HashingTextEmbedder
        return HashingTextEmbedder(dim=sp["dim"], id=eid)
    return None


def retrieve(mf, query, embedder=None, *, space=None, k=5):
    """Rank items in ``space`` by semantic similarity to ``query``."""
    from . import render as _render
    space = space or _text_space(mf)
    if space is None:
        raise ValueError("no text-bearing space to search; pass space=")
    embedder = embedder or embedder_for(mf, space)
    if embedder is None:
        raise ValueError(f"could not reconstruct an embedder for space {space!r}; pass embedder=")
    qv = embedder.embed([query])[0]
    return [{"item_id": i, "score": float(s), "space": space, "card": _render.card(mf, i)}
            for i, s in _search(mf, qv, k=k, space=space)]


def context(mf, query, embedder=None, *, space=None, k=5, max_findings=10) -> str:
    """A text bundle for an external LLM: summary + relevant records + findings."""
    from . import findings as _findings
    from . import summary as _summary
    hits = retrieve(mf, query, embedder, space=space, k=k)
    lines = [f"# context for query: {query}", "", _summary.summary_text(mf), "",
             "## relevant records"]
    lines += [f"- ({h['score']:.2f}) {h['card']}" for h in hits] or ["- (none)"]
    f = _findings.list_findings(mf)
    if f:
        lines += ["", "## findings"] + [f"- {x['statement']}" for x in f[:max_findings]]
    return "\n".join(lines)
