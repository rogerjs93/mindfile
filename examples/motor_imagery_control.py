"""Positive control: does the file's hemispheric-asymmetry feature detect the
KNOWN lateralization of motor imagery?

Motor imagery of the LEFT hand desynchronizes the RIGHT (contralateral)
sensorimotor cortex -> mu/beta power drops on the right -> (L-R)/(L+R) asymmetry
becomes MORE POSITIVE than during RIGHT-hand imagery. This is established
physiology, so it's a ground-truth test of the feature on real EEG.

Per subject we build two condition-averaged recordings (left- vs right-hand
imagery epochs), ingest both into one .gnos, materialize the asymmetry space, and
compare beta/alpha asymmetry between conditions with a paired test across
subjects. A positive result here also explains the lesion-laterality null: the
task itself imposes a lateralization that a whole-recording feature can't
separate from lesion effects.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import MindFile, index, lazy
from mindfile.embedders.eeg import AsymmetryEmbedder
from mindfile.modality import EEGProcessor

SENSORIMOTOR = ["C3", "C4", "FC3", "FC4", "CP3", "CP4"]   # motor-imagery montage

ROOT = Path(r"E:/Roger/projects data/stroke_qeeg/data")
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def imagery_events():
    """(onset_ms, dur_ms, hand) for each motor-imagery period; hand 1=left 2=right."""
    ev = []
    with open(ROOT / "events.tsv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("value") == "2":                      # 'motor imagery beginning'
                ev.append((int(r["onset"]), int(r["duration"]), int(r["trial_type"])))
    return ev


def condition_arrays(raw_data, sfreq, events):
    """Concatenate imagery epochs into (left_data, right_data)."""
    n = raw_data.shape[1]
    left, right = [], []
    for onset_ms, dur_ms, hand in events:
        a = int(onset_ms / 1000 * sfreq)
        b = min(a + int(dur_ms / 1000 * sfreq), n)
        if b - a < int(0.5 * sfreq):
            continue
        (left if hand == 1 else right).append(raw_data[:, a:b])
    if not left or not right:
        return None, None
    return np.hstack(left), np.hstack(right)


def main():
    import mne

    events = imagery_events()
    edfs = sorted((ROOT / "edf" / "edffile").rglob("*.edf"))
    out = Path(tempfile.gettempdir()) / "motor_imagery_control.gnos"
    mf = MindFile.create(out, modality="eeg", overwrite=True)
    proc = EEGProcessor(crop_seconds=None, connectivity=False, asymmetry=True)

    sm_emb = AsymmetryEmbedder(crop_seconds=None, channels=SENSORIMOTOR)
    sm = {}                                            # subject -> {cond: sensorimotor asym}

    print(f"[1] Epoching {len(edfs)} subjects into left/right motor-imagery recordings "
          f"({len(events)} imagery events/subject) ...")
    kept = 0
    for p in edfs:
        sub = p.stem.split("_")[0]
        raw = mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR")
        ch, sfreq = raw.ch_names, raw.info["sfreq"]
        ld, rd = condition_arrays(raw.get_data().astype("float32"), sfreq, events)
        if ld is None:
            continue
        proc.ingest(mf, ld, sfreq=sfreq, ch_names=ch, subject_id=sub, session_id="left")
        proc.ingest(mf, rd, sfreq=sfreq, ch_names=ch, subject_id=sub, session_id="right")
        sm[sub] = {"left": sm_emb._features_from_array(ld.astype(np.float64), sfreq, ch),
                   "right": sm_emb._features_from_array(rd.astype(np.float64), sfreq, ch)}
        kept += 1
    lazy.materialize_space(mf, "asymmetry")
    print(f"    {kept} subjects x 2 conditions ingested; asymmetry materialized")

    # gather per-subject asymmetry by condition
    meta = {r["id"]: (r["subject_id"], r["session_id"])
            for r in mf.conn.execute("SELECT id, subject_id, session_id FROM items")}
    ids, asym = index.load_all_vectors(mf, "asymmetry")
    by_sub = {}
    for i, v in zip(ids, asym):
        sub, cond = meta[int(i)]
        by_sub.setdefault(sub, {})[cond] = v
    paired = [(d["left"], d["right"]) for d in by_sub.values() if "left" in d and "right" in d]

    sm_paired = [(d["left"], d["right"]) for d in sm.values() if "left" in d and "right" in d]

    def report(pairs, label):
        print(f"\n{label} (n={len(pairs)}; expect LEFT-hand asym > RIGHT-hand, diff > 0):")
        for band in ("alpha", "beta"):
            bi = BANDS.index(band)
            dl = np.array([l[bi] for l, _ in pairs])
            dr = np.array([r[bi] for _, r in pairs])
            diff = dl - dr
            acc = float((diff > 0).mean())
            _, p = wilcoxon(diff, alternative="greater")
            star = "*" if p < 0.05 else " "
            print(f"     {band:5} asym: left {dl.mean():+.4f} vs right {dr.mean():+.4f} "
                  f"| median diff {np.median(diff):+.4f} | {acc*100:.0f}% correct "
                  f"| Wilcoxon p={p:.4f}{star}")

    report(paired, "[2] WHOLE-HEAD asymmetry (all L vs all R channels)")
    report(sm_paired, "[3] SENSORIMOTOR asymmetry (C3/C4/FC3/FC4/CP3/CP4 only)")
    mf.close()
    print(f"\nWrote {out} ({out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
