"""The in-file allocator: a deterministic Planner -> Reviewer -> Renderer loop
(RecursiveMAS Tier-1 "Sequential"), driven from the file's ``agents`` registry.

* PLANNER  inspects the file and proposes an ordered list of actions
           (cluster / knn / fuse / constellation / correlate / vacuum).
* REVIEWER approves or rejects each action under guards (write-once safety,
           min-vectors, already-done, a size budget, and a clean ``validate()``).
* RENDERER executes the approved actions and logs every step.

"Allocate" here means LOGICAL materialization — deciding which artifacts the
file should contain for the clusters that matter — plus VACUUM to reclaim space.
A physical/page strategy can hook in later at the Renderer.
"""

from __future__ import annotations

import json

from . import agents as _agents
from . import constellation as _con
from . import findings as _findings
from . import fuse as _fuse
from . import graph as _graph
from . import stats as _stats
from .cluster import clusters_of, latest_clustering_id, run_clustering
from .constellation import DEFAULT_AXES
from .file import MindFile, utcnow
from .spaces import get_space_id, list_spaces
from .validate import validate

_ROLES = [
    ("planner", "propose allocation actions from file state", "plan", 1, "reviewer"),
    ("reviewer", "approve/reject proposed actions under guards", "review", 2, "renderer"),
    ("renderer", "execute approved actions and log them", "render", 3, None),
]


def register_roles(mf: MindFile) -> None:
    for name, purpose, action, order, nxt in _ROLES:
        _agents.register_policy(mf, name, purpose=purpose, trigger="on_demand",
                                action=action, params={"role": name, "order": order, "next": nxt})


# -- small state probes ----------------------------------------------------
def _n_vectors(mf, space):
    sid = get_space_id(mf, space)
    return mf.conn.execute("SELECT COUNT(*) FROM vectors WHERE space_id = ?", (sid,)).fetchone()[0] if sid else 0


def _has_knn(mf, space):
    sid = get_space_id(mf, space) or 0
    return mf.conn.execute(
        "SELECT 1 FROM edges WHERE space_id = ? AND kind = 'knn' LIMIT 1", (sid,)).fetchone() is not None


def _fuse_candidates(mf):
    return [r[0] for r in mf.conn.execute(
        "SELECT DISTINCT s.name FROM spaces s JOIN vectors v ON v.space_id = s.id "
        "JOIN items i ON i.id = v.item_id WHERE i.origin = 'primary' "
        "AND s.name NOT IN ('constellation', 'fused') ORDER BY s.name")]


def _axis_item_count(mf):
    return mf.conn.execute("SELECT COUNT(DISTINCT item_id) FROM item_axes").fetchone()[0]


def _pick_semantic_space(mf):
    best, best_n = None, 0
    for s in list_spaces(mf):
        if s["name"] in ("constellation", "fused"):
            continue
        n = mf.conn.execute(
            "SELECT COUNT(*) FROM vectors v WHERE v.space_id = ? AND v.item_id IN "
            "(SELECT item_id FROM item_axes WHERE axis IN ('x','y','z','t') "
            " GROUP BY item_id HAVING COUNT(*) = 4)", (s["id"],)).fetchone()[0]
        if n > best_n:
            best, best_n = s["name"], n
    return best, best_n


def _freelist(mf):
    return mf.conn.execute("PRAGMA freelist_count").fetchone()[0]


def _existing_corr(mf, kind, a_ref, b_ref=None):
    if b_ref is None:
        return mf.conn.execute(
            "SELECT 1 FROM correlations WHERE kind=? AND a_ref=? AND b_ref IS NULL LIMIT 1",
            (kind, a_ref)).fetchone() is not None
    return mf.conn.execute(
        "SELECT 1 FROM correlations WHERE kind=? AND ((a_ref=? AND b_ref=?) OR "
        "(a_ref=? AND b_ref=?)) LIMIT 1", (kind, a_ref, b_ref, b_ref, a_ref)).fetchone() is not None


def _item_shared(mf, sid_a, sid_b):
    return mf.conn.execute(
        "SELECT COUNT(*) FROM (SELECT item_id FROM vectors WHERE space_id=? "
        "INTERSECT SELECT item_id FROM vectors WHERE space_id=?)", (sid_a, sid_b)).fetchone()[0]


def _subject_shared(mf, sid_a, sid_b):
    q = ("SELECT DISTINCT i.subject_id FROM items i JOIN vectors v ON v.item_id=i.id "
         "WHERE v.space_id=? AND i.subject_id IS NOT NULL")
    return mf.conn.execute(
        f"SELECT COUNT(*) FROM ({q} INTERSECT {q})", (sid_a, sid_b)).fetchone()[0]


def _correlation_actions(mf, spaces):
    """Propose only correlations that have something to compare: within-space
    feature correlations, cross-space (shared items), or cross-modality (shared
    subjects, different items). De-duplicated against what's already recorded."""
    acts = []
    for s in spaces:
        if s["n_vectors"] >= 3 and s["dim"] >= 2 and not _existing_corr(mf, "feature_corr", f"space:{s['name']}"):
            acts.append({"action": "correlate", "args": {"kind": "feature_corr", "space": s["name"]},
                         "reason": f"feature correlations within {s['name']}"})
    for i in range(len(spaces)):
        for j in range(i + 1, len(spaces)):
            a, b = spaces[i], spaces[j]
            ar, br = f"space:{a['name']}", f"space:{b['name']}"
            if _item_shared(mf, a["id"], b["id"]) >= 3:
                if not _existing_corr(mf, "cross_space", ar, br):
                    acts.append({"action": "correlate",
                                 "args": {"kind": "cross_space", "space_a": a["name"], "space_b": b["name"]},
                                 "reason": f"shared-item agreement {a['name']} vs {b['name']}"})
                if not _existing_corr(mf, "clustering_ari", ar, br):
                    acts.append({"action": "correlate",
                                 "args": {"kind": "clustering_ari", "space_a": a["name"], "space_b": b["name"]},
                                 "reason": f"cluster agreement {a['name']} vs {b['name']}"})
            elif _subject_shared(mf, a["id"], b["id"]) >= 3:
                if not _existing_corr(mf, "subject_cross_modality", ar, br):
                    acts.append({"action": "correlate",
                                 "args": {"kind": "subject_cross_modality", "space_a": a["name"], "space_b": b["name"]},
                                 "reason": f"cross-modality (same subjects) {a['name']} vs {b['name']}"})
    return acts


# -- PLANNER ---------------------------------------------------------------
def plan(mf: MindFile) -> list[dict]:
    actions = []
    spaces = [s for s in list_spaces(mf) if s["n_vectors"] >= 2]

    for s in spaces:
        if latest_clustering_id(mf, s["name"]) is None:
            actions.append({"action": "cluster", "args": {"space": s["name"]},
                            "reason": f"{s['n_vectors']} vectors, not yet clustered"})
    for s in spaces:
        if not _has_knn(mf, s["name"]):
            actions.append({"action": "knn", "args": {"space": s["name"]},
                            "reason": "no kNN graph yet"})

    cand = _fuse_candidates(mf)
    if len(cand) >= 2 and get_space_id(mf, "fused") is None:
        actions.append({"action": "fuse", "args": {"spaces": cand},
                        "reason": f"{len(cand)} modality spaces present, not fused"})

    if _axis_item_count(mf) and get_space_id(mf, "constellation") is None:
        sem, n = _pick_semantic_space(mf)
        if sem:
            actions.append({"action": "constellation", "args": {"semantic_space": sem},
                            "reason": f"{n} items carry x,y,z,t axes; build position-aware space"})

    actions += _correlation_actions(mf, spaces)

    # Live findings: once anything is clustered, discover cross-structure correlations.
    if any(latest_clustering_id(mf, s["name"]) for s in spaces):
        actions.append({"action": "findings", "args": {},
                        "reason": "discover cross-structure correlations for LLM analysis"})

    if _freelist(mf) > 64:
        actions.append({"action": "vacuum", "args": {}, "reason": "reclaim free pages"})
    return actions


# -- REVIEWER --------------------------------------------------------------
def _estimate_new_vectors(mf, action, args):
    if action == "constellation":
        sid = get_space_id(mf, args["semantic_space"])
        return mf.conn.execute(
            "SELECT COUNT(*) FROM vectors v WHERE v.space_id = ? AND v.item_id IN "
            "(SELECT item_id FROM item_axes WHERE axis IN ('x','y','z','t') "
            " GROUP BY item_id HAVING COUNT(*) = 4)", (sid,)).fetchone()[0] if sid else 0
    return 0


def review(mf: MindFile, actions: list[dict], budget: dict | None = None) -> list[dict]:
    budget = budget or {}
    max_new = budget.get("max_new_vectors")
    spent = 0
    file_ok = validate(mf)["errors"] == 0
    out = []
    for a in actions:
        act, args = a["action"], a["args"]
        status, reason = "approved", "ok"
        if not file_ok:
            status, reason = "rejected", "file fails validation"
        elif act == "cluster":
            if _n_vectors(mf, args["space"]) < 2:
                status, reason = "rejected", "fewer than 2 vectors"
            elif latest_clustering_id(mf, args["space"]) is not None:
                status, reason = "rejected", "already clustered"
        elif act == "knn" and _has_knn(mf, args["space"]):
            status, reason = "rejected", "kNN already built"
        elif act == "fuse" and get_space_id(mf, "fused") is not None:
            status, reason = "rejected", "already fused"
        elif act == "constellation":
            est = _estimate_new_vectors(mf, act, args)
            if est == 0:
                status, reason = "rejected", "no eligible axis-bearing items"
            elif max_new is not None and spent + est > max_new:
                status, reason = "rejected", f"exceeds size budget ({spent}+{est}>{max_new})"
            else:
                spent += est
        out.append({**a, "review_status": status, "review_reason": reason})
    return out


# -- RENDERER --------------------------------------------------------------
def _render_cluster(mf, args):
    cid = run_clustering(mf, args["space"])
    return {"clustering_id": cid, "k": len(clusters_of(mf, cid))}


def _render_constellation(mf, args):
    r = _con.build_constellation_space(mf, args["semantic_space"])
    subs = _con.subjects_with_axes(mf)
    built = [_con.build_constellation(mf, s) for s in subs]
    return {"vectors": r.get("created", 0), "constellations": len(built)}


def _render_correlate(mf, args):
    k = args["kind"]
    if k == "feature_corr":
        return _stats.feature_correlation(mf, args["space"])
    if k == "cross_space":
        return _stats.cross_space_agreement(mf, args["space_a"], args["space_b"])
    if k == "subject_cross_modality":
        return _stats.subject_cross_modality(mf, args["space_a"], args["space_b"])
    if k == "clustering_ari":
        a = latest_clustering_id(mf, args["space_a"])
        b = latest_clustering_id(mf, args["space_b"])
        if a and b:
            return _stats.clustering_agreement(
                mf, a, b, a_ref=f"space:{args['space_a']}", b_ref=f"space:{args['space_b']}")
        return {"skipped": "missing clusterings"}
    return None


def render(mf: MindFile, reviewed: list[dict]) -> list[dict]:
    results = []
    for a in reviewed:
        if a["review_status"] != "approved":
            results.append({**a, "render_status": "skipped", "render_detail": {}})
            continue
        act, args = a["action"], a["args"]
        try:
            if act == "cluster":
                d = _render_cluster(mf, args)
            elif act == "knn":
                d = {"edges": _graph.build_knn_edges(mf, space=args["space"])}
            elif act == "fuse":
                r = _fuse.fuse(mf, args["spaces"])
                d = {"created": len(r["created"]), "skipped": len(r["skipped"])}
            elif act == "constellation":
                d = _render_constellation(mf, args)
            elif act == "correlate":
                d = _render_correlate(mf, args)
            elif act == "findings":
                d = {"found": len(_findings.discover(mf))}
            elif act == "vacuum":
                mf.conn.execute("VACUUM")
                d = {"vacuumed": True}
            else:
                d = {}
            results.append({**a, "render_status": "done", "render_detail": d})
        except Exception as e:  # a failed action must not abort the whole render
            results.append({**a, "render_status": "failed", "render_detail": {"error": str(e)}})
    return results


# -- ORCHESTRATION (Sequential pipeline) -----------------------------------
def orchestrate(mf: MindFile, budget: dict | None = None) -> dict:
    """Run one Planner -> Reviewer -> Renderer pass, logging the plan in-file."""
    register_roles(mf)
    proposed = plan(mf)
    cur = mf.conn.execute(
        "INSERT INTO plans(created_at, status, summary) VALUES(?,?,?)",
        (utcnow(), "planned", f"{len(proposed)} actions proposed"))
    plan_id = cur.lastrowid

    reviewed = review(mf, proposed, budget)
    rendered = render(mf, reviewed)

    for seq, a in enumerate(rendered):
        mf.conn.execute(
            "INSERT INTO plan_actions(plan_id, seq, action, args, plan_reason, "
            "review_status, review_reason, render_status, render_detail) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (plan_id, seq, a["action"], json.dumps(a["args"]), a["reason"],
             a["review_status"], a["review_reason"], a["render_status"],
             json.dumps(a.get("render_detail", {}))))

    approved = sum(1 for a in reviewed if a["review_status"] == "approved")
    done = sum(1 for a in rendered if a["render_status"] == "done")
    mf.conn.execute("UPDATE plans SET status = 'rendered', summary = ? WHERE id = ?",
                    (f"proposed {len(proposed)}, approved {approved}, rendered {done}", plan_id))
    mf.record("allocate", {"plan_id": plan_id, "proposed": len(proposed),
                           "approved": approved, "rendered": done})
    mf.conn.commit()
    return {"plan_id": plan_id, "proposed": len(proposed), "approved": approved,
            "rendered": done, "actions": rendered}
