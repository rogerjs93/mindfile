"""Findings — the living correlation surface an external LLM analyses.

``discover`` scans the file for relationships across its data structures and
records them as plain‑English **statements** (plus the metric). No LLM runs here;
this is what the LLM *reads*:

* **cross‑modal agreement** — do two modalities' clusterings agree across the
  same subjects? (subject‑aligned Adjusted Rand Index).
* **cluster signature** — what distinguishes each cluster (features that deviate
  most from the global mean), named where the space has interpretable features.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import adjusted_rand_score

from . import _vec
from .cluster import assignments, latest_clustering_id
from .file import MindFile, utcnow
from .index import load_all_vectors
from .spaces import list_spaces

# Interpretable feature labels (mean part; std part appended as <name>_sd).
_BANDS = {"bandpower": ["delta", "theta", "alpha", "beta", "gamma"],
          "ecg_bandpower": ["low", "mid", "high"],
          "wfdb_spectral": ["low", "mid", "high"]}
# spaces whose vector is one interpretable value per band (no std half)
_FLAT = {"connectivity": ["delta", "theta", "alpha", "beta", "gamma"],
         "asymmetry": ["delta", "theta", "alpha", "beta", "gamma"]}


def _record(mf, kind, statement, value=None, n=None, refs=None):
    if mf.conn.execute("SELECT 1 FROM findings WHERE kind=? AND statement=? LIMIT 1",
                       (kind, statement)).fetchone():
        return
    mf.conn.execute(
        "INSERT INTO findings(kind, statement, value, n, refs, detail, created_at) "
        "VALUES(?,?,?,?,?,?,?)",
        (kind, statement, float(value) if value is not None else None,
         int(n) if n is not None else None, json.dumps(refs or {}), "{}", utcnow()))
    mf.conn.commit()


def list_findings(mf: MindFile) -> list[dict]:
    return [dict(r) for r in mf.conn.execute(
        "SELECT kind, statement, value, n FROM findings ORDER BY id")]


def _subject_clusters(mf, space) -> dict:
    cid = latest_clustering_id(mf, space)
    if cid is None:
        return {}
    out = {}
    for item_id, cl in assignments(mf, cid).items():
        r = mf.conn.execute("SELECT subject_id FROM items WHERE id=?", (item_id,)).fetchone()
        if r and r["subject_id"] is not None and r["subject_id"] not in out:
            out[r["subject_id"]] = cl
    return out


def cross_modal_agreement(mf: MindFile, space_a: str, space_b: str) -> dict | None:
    """Do two modalities' clusterings agree across the same subjects?"""
    a, b = _subject_clusters(mf, space_a), _subject_clusters(mf, space_b)
    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return None
    ari = float(adjusted_rand_score([a[s] for s in shared], [b[s] for s in shared]))
    verb = "agree" if ari > 0.3 else ("partly agree" if ari > 0.1 else "diverge")
    stmt = (f"{space_a} and {space_b} cluster structures {verb} across "
            f"{len(shared)} subjects (ARI={ari:.2f})")
    _record(mf, "cross_modal_agreement", stmt, ari, len(shared),
            {"space_a": space_a, "space_b": space_b})
    return {"ari": ari, "n": len(shared), "statement": stmt}


def _labels(mf, space, dim) -> list[str]:
    if space in _BANDS:
        return _BANDS[space] + [f"{x}_sd" for x in _BANDS[space]]
    if space in _FLAT:
        return _FLAT[space]
    if space == "labs":
        r = mf.conn.execute(
            "SELECT i.payload FROM items i JOIN vectors v ON v.item_id=i.id "
            "JOIN spaces s ON s.id=v.space_id WHERE s.name=? LIMIT 1", (space,)).fetchone()
        if r and r["payload"]:
            try:
                return list(json.loads(bytes(r["payload"]).decode("utf-8")).keys())
            except Exception:
                pass
    return [f"dim{i}" for i in range(dim)]


def cluster_signature(mf: MindFile, space: str, top: int = 2) -> list[str]:
    """What distinguishes each cluster in ``space`` (deviation from global mean)."""
    cid = latest_clustering_id(mf, space)
    if cid is None:
        return []
    _, mat = load_all_vectors(mf, space)
    if mat.shape[0] < 3:
        return []
    mean, std = mat.mean(0), mat.std(0)
    std[std == 0] = 1.0
    labels = _labels(mf, space, mat.shape[1])
    out = []
    for c in mf.conn.execute(
            "SELECT local_index, label, centroid FROM clusters WHERE clustering_id=? "
            "AND centroid IS NOT NULL ORDER BY local_index", (cid,)):
        z = (_vec.unpack(c["centroid"], mat.shape[1]) - mean) / std
        idx = np.argsort(-np.abs(z))[:top]
        feats = ", ".join(f"{'high' if z[i] > 0 else 'low'} {labels[i]}" for i in idx)
        stmt = f"{space} {c['label']}: {feats}"
        _record(mf, "cluster_signature", stmt, refs={"space": space, "cluster": c["local_index"]})
        out.append(stmt)
    return out


def labs_abnormalities(mf: MindFile, space: str = "labs") -> list[str]:
    """Surface out-of-reference-range lab values across subjects as findings."""
    from . import labs as _labs
    rows = mf.conn.execute(
        "SELECT DISTINCT i.id, i.subject_id, i.payload FROM items i "
        "JOIN vectors v ON v.item_id=i.id JOIN spaces s ON s.id=v.space_id "
        "WHERE s.name=?", (space,)).fetchall()
    if not rows:
        return []
    subjects, per_marker = set(), {}
    for r in rows:
        sid = r["subject_id"] if r["subject_id"] is not None else f"item:{r['id']}"
        subjects.add(sid)
        try:
            row = json.loads(bytes(r["payload"]).decode("utf-8"))
        except Exception:
            continue
        for analyte, direction in _labs.abnormal(row):
            per_marker.setdefault((analyte, direction), set()).add(sid)
    m = len(subjects)
    out = []
    for (analyte, direction), subs in sorted(per_marker.items(), key=lambda kv: -len(kv[1])):
        k = len(subs)
        stmt = f"{k}/{m} subjects have {direction} {analyte}"
        _record(mf, "labs_abnormality", stmt, value=k / m, n=k,
                refs={"analyte": analyte, "direction": direction})
        out.append(stmt)
    return out


def discover(mf: MindFile) -> list[dict]:
    """Run all available correlation analyses and record the findings."""
    spaces = [s["name"] for s in list_spaces(mf) if latest_clustering_id(mf, s["name"])]
    for sp in spaces:
        cluster_signature(mf, sp)
    for i in range(len(spaces)):
        for j in range(i + 1, len(spaces)):
            cross_modal_agreement(mf, spaces[i], spaces[j])
    labs_abnormalities(mf)
    found = list_findings(mf)
    mf.record("findings", {"count": len(found)})
    mf.conn.commit()
    return found
