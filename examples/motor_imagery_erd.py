"""Re-test motor-imagery laterality with ERD (the right tool).

The static hemispheric-asymmetry feature failed to recover imagery side (a clean
null) because ERD is a *baseline-relative dynamic* a static snapshot can't see.
Here we store each recording's events and compute baseline-normalised ERD from
the stored waveform, then test the contralateral signature:

  left-hand imagery  -> RIGHT hemisphere desync -> li = ERD_right - ERD_left < 0
  right-hand imagery -> LEFT  hemisphere desync -> li > 0

So li(right-hand) should exceed li(left-hand) within each subject.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import MindFile, events, modality

ROOT = Path(r"E:/Roger/projects data/stroke_qeeg/data")


def imagery_events():
    out = []
    with open(ROOT / "events.tsv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("value") == "2":                         # motor-imagery onset
                out.append({"onset": int(r["onset"]) / 1000.0,
                            "duration": int(r["duration"]) / 1000.0,
                            "condition": "left" if r["trial_type"] == "1" else "right"})
    return out


def main():
    evs = imagery_events()
    edfs = sorted((ROOT / "edf" / "edffile").rglob("*.edf"))
    mf = MindFile.create(Path(tempfile.gettempdir()) / "mi_erd.gnos", modality="eeg",
                         overwrite=True)

    print(f"[1] Storing {len(edfs)} recordings + their {len(evs)} imagery events, "
          "computing ERD from the stored waveform...")
    rows = {"alpha": [], "beta": []}
    for p in edfs:
        sub = p.stem.split("_")[0]
        iid = modality.ingest(mf, "eeg", str(p), subject_id=sub)
        events.add_events(mf, iid, evs)
        for band in ("alpha", "beta"):
            liL = events.erd_lateralization(mf, iid, band=band, condition="left",
                                            active=(0.0, 4.0), baseline_len=2.0)["li"]
            liR = events.erd_lateralization(mf, iid, band=band, condition="right",
                                            active=(0.0, 4.0), baseline_len=2.0)["li"]
            if np.isfinite(liL) and np.isfinite(liR):
                rows[band].append((liL, liR))

    print("\n[2] Contralateral ERD test (expect li(right-hand) > li(left-hand)):")
    for band in ("alpha", "beta"):
        pairs = rows[band]
        liL = np.array([a for a, _ in pairs])
        liR = np.array([b for _, b in pairs])
        diff = liR - liL
        acc = float((diff > 0).mean())
        _, p = wilcoxon(diff, alternative="greater")   # rank-based: outlier-immune
        star = "*" if p < 0.05 else " "
        # report medians (a few subjects have artifact-inflated li; the Wilcoxon is
        # the valid test since it uses ranks, not magnitudes)
        print(f"     {band:5}: median li left-hand {np.median(liL):+6.1f}%  right-hand "
              f"{np.median(liR):+6.1f}%  | {acc*100:.0f}% subjects correct | Wilcoxon p={p:.4f}{star}"
              f"  (n={len(pairs)})")
    print("\n  compare: static hemispheric-asymmetry gave alpha 60% / p=0.076, beta null.")
    mf.close()


if __name__ == "__main__":
    main()
