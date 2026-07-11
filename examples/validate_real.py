"""End-to-end validation on REAL EEG data (the stroke_qeeg EDF cohort).

Exercises the FULL current stack on real recordings — modality ingest -> lossless
waveform store -> materialize band-power -> cluster -> auto-discover findings ->
self-allocator -> checkpoint -> Ed25519 sign -> deep audit — and reports real
observations. Requires `mne` and a folder of .edf files.

Run:  python examples/validate_real.py --root "E:/Roger/projects data/stroke_qeeg/data/edf/edffile" --limit 12
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import (MindFile, allocate, audit, checkpoint, cluster, findings,
                      lazy, modality, render, sign)

DEFAULT_ROOT = r"E:/Roger/projects data/stroke_qeeg/data/edf/edffile"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    try:
        import mne  # noqa: F401
    except ImportError:
        print("mne not installed — `pip install mne` to run real validation."); return
    edfs = sorted(Path(args.root).rglob("*.edf"))[:args.limit]
    if not edfs:
        print(f"no .edf files under {args.root}"); return

    out = Path(tempfile.gettempdir()) / "validate_real.gnos"
    mf = MindFile.create(out, modality="eeg", overwrite=True)

    print(f"[1] Ingesting {len(edfs)} REAL EEG recordings via the eeg modality processor...")
    for p in edfs:
        modality.ingest(mf, "eeg", str(p), subject_id=p.stem, external_id=p.stem)
    rows = mf.conn.execute("SELECT dtype, n_channels, n_samples, LENGTH(code) AS c "
                           "FROM signals").fetchall()
    itemsize = {"<i2": 2, "<f4": 4}
    dig = sum(r["n_channels"] * r["n_samples"] * itemsize.get(r["dtype"], 4) for r in rows)
    f32 = sum(r["n_channels"] * r["n_samples"] * 4 for r in rows)
    code = sum(r["c"] for r in rows)
    n = lazy.materialize_space(mf, "bandpower")
    print(f"    {mf.counts()['items']} recordings; {mf.counts()['signals']} waveforms stored "
          f"lossless int16 {dig/1e6:.1f}MB -> {code/1e6:.1f}MB ({dig/max(code,1):.2f}x codec); "
          f"vs float32 physical {f32/1e6:.1f}MB = {f32/max(code,1):.2f}x smaller file")
    print(f"    materialized {n} band-power vectors (recomputed from the stored waveforms)")

    print("[2] Cluster the cohort by band-power:")
    cid = cluster.run_clustering(mf, "bandpower", k=min(3, len(edfs)))
    for c in cluster.clusters_of(mf, cid):
        print(f"    {c['label']}: size {c['size']}")

    print("[3] Auto-discovered findings (the file finding relationships, unchanged data):")
    for f in findings.discover(mf):
        print(f"    - {f['statement']}")

    print("[4] Self-allocator pass + seal (checkpoint + sign + deep audit):")
    allocate.orchestrate(mf)
    checkpoint.create_checkpoint(mf, "real-cohort")
    signer = sign.Ed25519Signer()
    sign.sign_state(mf, signer)
    sign.trust_key(mf, signer.public_key)
    rep = audit.audit(mf, deep=True)
    print(f"    AUDIT: status={rep['status']}  corrupted={rep['corrupted']}")
    print(f"    signature trusted+current: {sign.verify_signatures(mf)['trusted_attested']}")

    print("[5] Sample LLM card (what an external LLM reads):")
    print(f"    {render.card(mf, 1)}")

    mf.close()
    print(f"\nWrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
