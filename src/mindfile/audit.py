"""One authoritative integrity + validation pass over the whole file.

Integrity is the format's #1 guarantee, so it should take a single call to ask
"is this file sound?" — not six. ``audit`` runs every verifier and the semantic
validator and rolls them into one verdict:

* ``ok``        — everything verifies and validates;
* ``issues``    — bytes are intact but structure/semantics have warnings/errors;
* ``corrupted`` — at least one content-integrity check failed.

It is exact and cheap by default (hash comparisons over stored data). ``deep=True``
additionally resolves every virtual vector and reconstructs every waveform to
prove the recipes/codecs actually produce well-formed output (this unfolds data).
"""

from __future__ import annotations

import hashlib

from . import checkpoint as _cp
from . import fuse as _fuse
from . import lazy as _lazy
from . import sign as _sign
from . import signal as _signal
from . import validate as _validate
from .file import MindFile
from .spaces import get_space_id


def _deep_resolve(mf: MindFile) -> dict:
    """Resolve every virtual vector + reconstruct every signal; confirm shapes."""
    bad = []
    for r in mf.conn.execute(
            "SELECT vd.item_id, s.name AS space, s.dim FROM vector_defs vd "
            "JOIN spaces s ON s.id = vd.space_id"):
        try:
            vec = _lazy.resolve(mf, r["item_id"], r["space"])
            if vec.shape[0] != r["dim"]:
                bad.append({"item_id": r["item_id"], "space": r["space"], "issue": "dim"})
        except Exception as e:  # a recipe that won't run is an integrity failure
            bad.append({"item_id": r["item_id"], "space": r["space"], "issue": str(e)})
    for r in mf.conn.execute("SELECT item_id FROM signals"):
        try:
            _signal.load_signal(mf, r["item_id"], verify=True)
        except Exception as e:
            bad.append({"item_id": r["item_id"], "space": "<signal>", "issue": str(e)})
    return {"status": "ok" if not bad else "corrupted", "bad": bad}


def verify_history(mf: MindFile) -> dict:
    """Walk the history hash-chain; detect any edit/deletion/insertion/reorder.

    Each entry must (a) hash to its stored ``entry_hash`` and (b) link to the
    previous entry's hash. The head must match ``mind_meta('history_head')``.
    """
    rows = mf.conn.execute(
        "SELECT id, ts, event, detail, prev_hash, entry_hash FROM history ORDER BY id"
    ).fetchall()
    prev = mf._genesis_hash()
    broken = []
    for r in rows:
        recomputed = hashlib.sha256(
            f"{r['prev_hash']}\t{r['ts']}\t{r['event']}\t{r['detail']}".encode("utf-8")).hexdigest()
        if r["prev_hash"] != prev:
            broken.append({"id": r["id"], "issue": "chain_break"})
        if recomputed != r["entry_hash"]:
            broken.append({"id": r["id"], "issue": "content_tampered"})
        prev = r["entry_hash"]
    head_ok = True if not rows else (rows[-1]["entry_hash"] == mf.meta_get("history_head"))
    status = "ok" if (not broken and head_ok) else "corrupted"
    return {"status": status, "entries": len(rows), "broken": broken, "head_ok": head_ok}


def audit(mf: MindFile, *, deep: bool = False) -> dict:
    """Run all integrity checks + semantic validation; return one verdict."""
    checks = {
        "integrity": _cp.verify_integrity(mf),     # live vectors/payloads vs stored hashes
        "derivations": _fuse.verify_derivations(mf),  # fused points vs source bytes
        "defs": _lazy.verify_defs(mf),             # lazy recipes vs their inputs
        "signals": _signal.verify_signals(mf),     # waveforms reconstruct exactly
        "history": verify_history(mf),             # provenance chain unbroken
        "signatures": _sign.verify_signatures(mf),  # authenticity (Ed25519 self-verifies)
        "validation": _validate.validate(mf),      # structure/semantics
    }
    checkpoints = {cp["name"]: _cp.verify_checkpoint(mf, cp["name"])
                   for cp in _cp.list_checkpoints(mf)}
    checks["checkpoints"] = checkpoints
    if deep:
        checks["deep"] = _deep_resolve(mf)

    corrupted = [k for k in ("integrity", "derivations", "defs", "signals", "history",
                             "signatures")
                 if checks[k]["status"] != "ok"]
    corrupted += [f"checkpoint:{n}" for n, r in checkpoints.items() if r["status"] != "ok"]
    if deep and checks["deep"]["status"] != "ok":
        corrupted.append("deep")

    val_errors = checks["validation"]["errors"]
    val_warnings = checks["validation"]["warnings"]
    status = "corrupted" if corrupted else ("issues" if val_errors else "ok")

    return {"status": status, "corrupted": corrupted,
            "validation_errors": val_errors, "validation_warnings": val_warnings,
            "checks": checks}
