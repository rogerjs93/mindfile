"""End-to-end .gnos demo on REAL EEG data (EDF files), waveform-first.

Stores each recording's WAVEFORM losslessly inside the .gnos (channels x time),
then treats band-power vectors as lazy recipes recomputed from the waveform on
call. Clusters the (materialized) features and verifies the signals reconstruct
byte-exact. Requires `mne` and a folder of .edf files.

Run:
  python examples/eeg_demo.py --root "E:/Roger/projects data/stroke_qeeg/data/edf/edffile" --limit 12
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import MindFile, checkpoint, cluster, lazy, signal
from mindfile.embedders import EEGEmbedder

DEFAULT_ROOT = r"E:/Roger/projects data/stroke_qeeg/data/edf/edffile"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()

    try:
        import mne
    except ImportError:
        print("mne not installed - `pip install mne` to run the EEG demo."); return

    edfs = sorted(Path(args.root).rglob("*.edf"))[:args.limit]
    if not edfs:
        print(f"no .edf files under {args.root}"); return
    print(f"found {len(edfs)} EDF files; storing waveforms losslessly + lazy features...")

    out = Path(tempfile.gettempdir()) / "eeg_signals.gnos"
    mf = MindFile.create(out, modality="eeg", overwrite=True)
    emb = EEGEmbedder()

    raw_total = code_total = 0
    item_ids = []
    for p in edfs:
        raw = mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR")
        data = raw.get_data().astype("float32")          # (channels, time)
        item_id, info = signal.add_recording(
            mf, data, sfreq=raw.info["sfreq"], ch_names=raw.ch_names,
            subject_id=p.stem, external_id=p.stem, embedder=emb)
        item_ids.append(item_id)
        raw_total += info["raw_bytes"]; code_total += info["code_bytes"]

    print(f"stored {len(item_ids)} waveforms: {raw_total/1e6:.1f} MB raw -> "
          f"{code_total/1e6:.1f} MB compressed ({raw_total/max(code_total,1):.2f}x, lossless)")

    # Band-power vectors are virtual until called; materialize to cluster.
    print("band-power vector of first recording is virtual?",
          lazy.is_virtual(mf, item_ids[0], "bandpower"))
    n = lazy.materialize_space(mf, "bandpower")
    print(f"materialized {n} band-power vectors (recomputed from the stored waveforms)")

    cid = cluster.run_clustering(mf, "bandpower", k=min(args.k, len(item_ids)))
    for c in cluster.clusters_of(mf, cid):
        print(f"   {c['label']}: size={c['size']}")

    checkpoint.create_checkpoint(mf, "eeg-original")
    print("verify waveforms reconstruct exactly:", signal.verify_signals(mf)["status"])
    print("verify lazy feature recipes vs signals:", lazy.verify_defs(mf)["status"])
    mf.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
