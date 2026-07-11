"""Agent block: self-update policies stored *inside* the file.

This is the Turso-registry pattern applied to the file itself — rows describing
how the file should maintain its own semantic map (e.g. "re-cluster when N new
items have been added"). The MVP records and reports policies and provides a
``due`` check; wiring policies to an autonomous runner is Phase 2.
"""

from __future__ import annotations

import json

from .file import MindFile


def register_policy(
    mf: MindFile,
    name: str,
    *,
    purpose: str,
    trigger: str,          # on_insert | on_demand | threshold
    action: str,           # recluster | reembed | relink
    params: dict | None = None,
    status: str = "enabled",
) -> None:
    mf.conn.execute(
        "INSERT INTO agents(name, purpose, trigger, action, params, status) "
        "VALUES(?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
        "purpose=excluded.purpose, trigger=excluded.trigger, action=excluded.action, "
        "params=excluded.params, status=excluded.status",
        (name, purpose, trigger, action, json.dumps(params or {}), status),
    )
    mf.conn.commit()


def list_policies(mf: MindFile) -> list[dict]:
    rows = mf.conn.execute(
        "SELECT name, purpose, trigger, action, params, status FROM agents ORDER BY name"
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["params"] = json.loads(d["params"] or "{}")
        out.append(d)
    return out


def due(mf: MindFile) -> list[dict]:
    """Return enabled 'threshold' policies whose condition is currently met.

    Supported condition (MVP): params {"new_items_since_cluster": N} fires when
    at least N items exist beyond those covered by the latest clustering run.
    """
    from .cluster import assignments, latest_clustering_id

    cid = latest_clustering_id(mf)
    clustered = len(assignments(mf, cid)) if cid else 0
    total = mf.conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    pending = total - clustered

    fired = []
    for p in list_policies(mf):
        if p["status"] != "enabled" or p["trigger"] != "threshold":
            continue
        n = p["params"].get("new_items_since_cluster")
        if n is not None and pending >= n:
            fired.append({**p, "pending_items": pending})
    return fired
