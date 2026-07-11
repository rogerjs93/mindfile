"""Fast, resident file summary — what an AI reads first, without unfolding.

Everything here is assembled from small, always-materialized tables: meta,
counts, spaces, cluster centroids/labels/sizes (the resident summary produced by
clustering), collections, correlations, evolution, save points, and signal
stats. It NEVER resolves a lazy vector, decompresses a waveform, or scans the
full item set — so an AI agent can grasp the file's content cheaply.

``file_vector`` returns a single size-weighted centroid-of-centroids so another
system can place/compare the whole file with one vector (e.g. index many `.mind`
files for retrieval) without opening them up.
"""

from __future__ import annotations

import numpy as np

from . import _vec
from .cluster import clusters_of, latest_clustering_id
from .file import MindFile
from .spaces import list_spaces


def _latest_clusters(mf: MindFile, space: str, with_centroids=False) -> list[dict]:
    cid = latest_clustering_id(mf, space)
    if cid is None:
        return []
    out = []
    rows = mf.conn.execute(
        "SELECT local_index, label, size, centroid FROM clusters WHERE clustering_id=? "
        "ORDER BY local_index", (cid,)).fetchall()
    dim = mf.conn.execute("SELECT dim FROM spaces WHERE name=?", (space,)).fetchone()["dim"]
    for r in rows:
        c = {"label": r["label"], "size": r["size"]}
        if with_centroids and r["centroid"] is not None:
            c["centroid"] = _vec.unpack(r["centroid"], dim).tolist()
        out.append(c)
    return out


def file_vector(mf: MindFile, space: str | None = None) -> dict | None:
    """A single resident vector summarizing the file: the size-weighted mean of a
    space's cluster centroids. Cheap — reads only ``clusters.centroid``."""
    spaces = [s for s in list_spaces(mf) if latest_clustering_id(mf, s["name"])]
    if not spaces:
        return None
    if space is None:  # prefer a non-derived (primary-ish) space
        space = next((s["name"] for s in spaces if not _is_derived(mf, s["name"])),
                     spaces[0]["name"])
    cid = latest_clustering_id(mf, space)
    dim = mf.conn.execute("SELECT dim FROM spaces WHERE name=?", (space,)).fetchone()["dim"]
    rows = mf.conn.execute(
        "SELECT size, centroid FROM clusters WHERE clustering_id=? AND centroid IS NOT NULL",
        (cid,)).fetchall()
    if not rows:
        return None
    cents = np.stack([_vec.unpack(r["centroid"], dim) for r in rows])
    weights = np.array([max(r["size"], 1) for r in rows], dtype=np.float64)
    vec = np.average(cents, axis=0, weights=weights).astype(np.float32)
    return {"space": space, "dim": int(dim), "vector": vec.tolist()}


def _is_derived(mf: MindFile, space: str) -> bool:
    r = mf.conn.execute("SELECT derived FROM spaces WHERE name=?", (space,)).fetchone()
    return bool(r["derived"]) if r else False


def summary(mf: MindFile, *, with_centroids: bool = False) -> dict:
    """A compact, resident overview of the file. Unfolds nothing."""
    meta = mf.meta_all()
    counts = mf.counts()

    # virtual vector counts per space (cheap: vector_defs vs vectors).
    virtual = {r["name"]: r["v"] for r in mf.conn.execute(
        "SELECT s.name, COUNT(*) AS v FROM vector_defs vd JOIN spaces s ON s.id=vd.space_id "
        "WHERE vd.item_id NOT IN (SELECT item_id FROM vectors v WHERE v.space_id=vd.space_id) "
        "GROUP BY s.name")}

    spaces = []
    for s in list_spaces(mf):
        spaces.append({"name": s["name"], "dim": s["dim"], "vectors": s["n_vectors"],
                       "virtual": virtual.get(s["name"], 0), "derived": _is_derived(mf, s["name"]),
                       "clusters": _latest_clusters(mf, s["name"], with_centroids)})

    collections = [dict(r) for r in mf.conn.execute(
        "SELECT name, kind, (SELECT COUNT(*) FROM collection_items ci WHERE ci.collection_id=c.id) "
        "AS size FROM collections c ORDER BY id")]
    correlations = [dict(r) for r in mf.conn.execute(
        "SELECT kind, a_ref, b_ref, value, n FROM correlations ORDER BY id")]
    evolution = [{"event": h["event"], "detail": __import__("json").loads(h["detail"] or "{}")}
                 for h in mf.history() if h["event"] in ("split", "merge")]
    checkpoints = [dict(r) for r in mf.conn.execute(
        "SELECT name, item_count, vector_count, manifest_sha256 FROM checkpoints ORDER BY id")]

    sig = mf.conn.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(n_channels),0) ch, COALESCE(SUM(n_samples),0) sm, "
        "COALESCE(SUM(LENGTH(code)),0) bytes FROM signals").fetchone()

    return {
        "file": {"uuid": meta.get("file_uuid"), "format_version": meta.get("format_version"),
                 "modality": meta.get("modality"), "created_at": meta.get("created_at")},
        "counts": counts,
        "subjects": mf.conn.execute(
            "SELECT COUNT(DISTINCT subject_id) FROM items WHERE subject_id IS NOT NULL").fetchone()[0],
        "spaces": spaces,
        "collections": collections,
        "correlations": correlations,
        "evolution": evolution,
        "checkpoints": checkpoints,
        "signals": {"recordings": sig["n"], "code_bytes": sig["bytes"]},
        "file_vector": file_vector(mf),
    }


def summary_text(mf: MindFile) -> str:
    """A short human/AI-readable digest (a few lines)."""
    s = summary(mf)
    f = s["file"]
    lines = [f"{f['modality']} .gnos ({f['format_version']}) - {s['counts']['items']} items, "
             f"{s['subjects']} subjects, {s['signals']['recordings']} waveforms"]
    for sp in s["spaces"]:
        cl = ", ".join(f"{c['label']} x{c['size']}" for c in sp["clusters"]) or "no clusters"
        v = f" (+{sp['virtual']} virtual)" if sp["virtual"] else ""
        lines.append(f"  space {sp['name']} dim{sp['dim']} [{sp['vectors']}{v}]: {cl}")
    for c in s["correlations"]:
        val = f"{c['value']:.2f}" if c["value"] is not None else "n/a"
        lines.append(f"  corr {c['kind']} {c['a_ref']}~{c['b_ref']}={val}")
    for e in s["evolution"]:
        d = e["detail"]
        if e["event"] == "split":
            into = ", ".join(i["label"] for i in d.get("into", []))
            lines.append(f"  SPLIT {d.get('parent_label')} -> {into}")
    return "\n".join(lines)
