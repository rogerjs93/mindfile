"""Semantic / structural validation — distinct from byte-corruption verification.

``verify_*`` (checkpoint.py) answers "did the bytes change?". ``validate`` answers
"is this file internally well-formed and analysis-ready?": dimensions consistent,
no NaN/Inf, every item embedded, identity present where an analysis needs it, no
duplicate records, referential integrity intact.

Returns ``{"status": "ok" | "issues", "errors": n, "warnings": n, "issues": [...]}``
where each issue is ``{severity, code, message, ids?}``.
"""

from __future__ import annotations

import numpy as np

from .file import MindFile
from .index import load_all_vectors
from .spaces import list_spaces


def _issue(severity, code, message, ids=None):
    d = {"severity": severity, "code": code, "message": message}
    if ids:
        d["ids"] = ids
    return d


def validate(mf: MindFile) -> dict:
    issues: list[dict] = []

    # 1/2. Per space: dimension consistency + finite values.
    for sp in list_spaces(mf):
        ids, mat = load_all_vectors(mf, sp["name"])
        if mat.size == 0:
            continue
        if mat.shape[1] != sp["dim"]:
            issues.append(_issue("error", "space_dim_mismatch",
                                 f"space {sp['name']!r} declares dim {sp['dim']} but vectors are {mat.shape[1]}"))
        finite = np.isfinite(mat).all(axis=1)
        if not finite.all():
            bad = [int(i) for i in ids[~finite]]
            issues.append(_issue("error", "non_finite_vector",
                                 f"space {sp['name']!r} has NaN/Inf vectors", bad))

    # 3. Items with no vector in any space.
    no_vec = [r["id"] for r in mf.conn.execute(
        "SELECT id FROM items WHERE id NOT IN (SELECT item_id FROM vectors)")]
    if no_vec:
        issues.append(_issue("warning", "item_without_vector",
                             f"{len(no_vec)} item(s) have no vector in any space", no_vec))

    # 4. Duplicate identity (same subject + session + modality).
    dups = mf.conn.execute(
        "SELECT subject_id, session_id, modality, COUNT(*) c FROM items "
        "WHERE subject_id IS NOT NULL GROUP BY subject_id, session_id, modality HAVING c > 1"
    ).fetchall()
    if dups:
        issues.append(_issue("warning", "duplicate_identity",
                             f"{len(dups)} (subject, session, modality) combination(s) appear more than once",
                             [f"{d['subject_id']}/{d['session_id']}/{d['modality']}" for d in dups]))

    # 5. Referential integrity (FKs should hold, but verify explicitly).
    orphan_assign = mf.conn.execute(
        "SELECT COUNT(*) FROM cluster_assignments a "
        "WHERE NOT EXISTS (SELECT 1 FROM items i WHERE i.id = a.item_id) "
        "OR NOT EXISTS (SELECT 1 FROM clusters c WHERE c.id = a.cluster_id)"
    ).fetchone()[0]
    if orphan_assign:
        issues.append(_issue("error", "orphan_assignment",
                             f"{orphan_assign} cluster assignment(s) reference missing items/clusters"))

    # 6. Longitudinal collections need subject_id + acquired_at, single subject.
    for coll in mf.conn.execute("SELECT id, name FROM collections WHERE kind = 'longitudinal'"):
        rows = mf.conn.execute(
            "SELECT i.id, i.subject_id, i.acquired_at FROM collection_items ci "
            "JOIN items i ON i.id = ci.item_id WHERE ci.collection_id = ?", (coll["id"],)
        ).fetchall()
        missing_time = [r["id"] for r in rows if not r["acquired_at"]]
        if missing_time:
            issues.append(_issue("warning", "longitudinal_missing_time",
                                 f"longitudinal collection {coll['name']!r} has items without acquired_at",
                                 missing_time))
        subjects = {r["subject_id"] for r in rows}
        if len(subjects) > 1:
            issues.append(_issue("warning", "longitudinal_multi_subject",
                                 f"longitudinal collection {coll['name']!r} spans {len(subjects)} subjects"))

    # 7. Derived items must carry provenance (a derivation).
    orphan_derived = [r["id"] for r in mf.conn.execute(
        "SELECT id FROM items WHERE origin = 'derived' "
        "AND id NOT IN (SELECT derived_item_id FROM derivations)")]
    if orphan_derived:
        issues.append(_issue("error", "derived_without_provenance",
                             f"{len(orphan_derived)} derived item(s) have no derivation record",
                             orphan_derived))

    # 8b. Lossy signals are not byte-exact originals — flag them explicitly.
    lossy = [r["item_id"] for r in mf.conn.execute(
        "SELECT item_id FROM signals WHERE lossy = 1")]
    if lossy:
        issues.append(_issue("info", "lossy_signal",
                             f"{len(lossy)} waveform(s) stored with a lossy codec (not byte-exact)",
                             lossy))

    # 8. Authenticity policy: if trusted keys are pinned, require a current
    #    valid signature from one of them.
    if mf.conn.execute("SELECT 1 FROM trusted_keys LIMIT 1").fetchone():
        from . import sign as _sign
        if not _sign.verify_signatures(mf)["trusted_attested"]:
            issues.append(_issue("warning", "untrusted_state",
                                 "trusted keys are pinned but the current state has no valid "
                                 "signature from one of them"))

    errors = sum(1 for i in issues if i["severity"] == "error")
    warnings = sum(1 for i in issues if i["severity"] == "warning")
    return {"status": "ok" if not issues else "issues",
            "errors": errors, "warnings": warnings, "issues": issues}
