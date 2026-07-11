"""Real cohort analysis on the stroke_qeeg dataset (50 subjects) — does the
structure the .gnos file discovers match clinical ground truth?

Literature-grounded hypothesis: qEEG *slowing* (Delta/Alpha Ratio and the
(delta+theta)/(alpha+beta) ratio, DTABR) increases with stroke severity. So the
band-power the file computes should correlate with NIHSS (severity), mRS
(disability, higher=worse) and MBI (function, higher=better -> negative corr).

Pipeline: ingest 50 real EDFs (lossless int16) -> materialize band-power +
connectivity -> per-subject slowing ratios -> Spearman vs clinical scores ->
unsupervised 2-cluster split tested against severity -> in-file findings ->
checkpoint + sign + audit. Reports REAL numbers, including nulls.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindfile import (MindFile, audit, checkpoint, findings, index, lazy, sign)
from mindfile.modality import EEGProcessor

ROOT = Path(r"E:/Roger/projects data/stroke_qeeg/data")
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


_CORTICAL_KW = ("cortex", "cortical", "frontal", "parietal", "temporal",
                "occipital", "lobe", "insula", "watershed")


def _lesion_type(loc: str) -> str:
    t = loc.lower()
    if "subcortical" in t:                 # "subcortical cerebral hemisphere" etc.
        return "subcortical"
    return "cortical" if any(k in t for k in _CORTICAL_KW) else "subcortical"


def load_labels():
    out = {}
    with open(ROOT / "participants.tsv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[r["Participant_ID"]] = {
                "paralysis": r["ParalysisSide"].strip().lower(),
                "nihss": _num(r["NIHSS"]), "mrs": _num(r["mRS"]), "mbi": _num(r["MBI"]),
                "cortical": _lesion_type(r["StrokeLocation"]) == "cortical"}
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    rho, p = spearmanr(x[m], y[m])
    return rho, p, int(m.sum())


def main():
    labels = load_labels()
    edfs = sorted((ROOT / "edf" / "edffile").rglob("*.edf"))
    out = Path(tempfile.gettempdir()) / "stroke_cohort.gnos"
    mf = MindFile.create(out, modality="eeg", overwrite=True)
    proc = EEGProcessor(crop_seconds=None)             # use full recordings for features

    print(f"[1] Ingesting {len(edfs)} real EEG recordings (lossless int16) ...")
    for p in edfs:
        sub = p.stem.split("_")[0]
        if sub in labels:
            proc.ingest(mf, str(p), subject_id=sub, external_id=sub)
    lazy.materialize_space(mf, "bandpower")
    lazy.materialize_space(mf, "connectivity")
    lazy.materialize_space(mf, "asymmetry")
    print(f"    {mf.counts()['items']} recordings; band-power + connectivity + asymmetry materialized")

    # per-subject features from the file's stored vectors
    sub_of = {r["id"]: r["subject_id"]
              for r in mf.conn.execute("SELECT id, subject_id FROM items")}
    ids, bp = index.load_all_vectors(mf, "bandpower")
    _, conn = index.load_all_vectors(mf, "connectivity")
    _, asym = index.load_all_vectors(mf, "asymmetry")
    subs = [sub_of[int(i)] for i in ids]
    delta, theta, alpha, beta = bp[:, 0], bp[:, 1], bp[:, 2], bp[:, 3]
    dar = delta / np.clip(alpha, 1e-9, None)
    dtabr = (delta + theta) / np.clip(alpha + beta, 1e-9, None)
    nihss = [labels[s]["nihss"] for s in subs]
    mrs = [labels[s]["mrs"] for s in subs]
    mbi = [labels[s]["mbi"] for s in subs]

    print("\n[2] qEEG slowing vs clinical scores (Spearman rho, p, n):")
    print(f"    {'':14}{'NIHSS(sev)':>16}{'mRS(disab)':>16}{'MBI(func)':>16}")
    for name, x in [("DAR", dar), ("DTABR", dtabr)]:
        cells = []
        for y in (nihss, mrs, mbi):
            rho, p, n = spearman(x, y)
            star = "*" if p < 0.05 else " "
            cells.append(f"{rho:+.2f} p={p:.3f}{star}")
        print(f"    {name:14}{cells[0]:>16}{cells[1]:>16}{cells[2]:>16}")

    print("\n[2b] Hemispheric asymmetry vs paralysis side (lesion laterality):")
    print("     hypothesis: right paralysis = LEFT lesion -> more left slow power -> +delta asym")
    paralysis = np.array([labels[s]["paralysis"] for s in subs])
    valid = np.isin(paralysis, ["left", "right"])
    y_right = (paralysis[valid] == "right").astype(int)   # 1 = right paralysis (left lesion)
    delta_asym, theta_asym = asym[:, 0], asym[:, 1]
    slow_asym = (delta_asym + theta_asym) / 2
    print(f"     ({valid.sum()} subjects: {int((paralysis=='left').sum())} left / "
          f"{int((paralysis=='right').sum())} right paralysis)")
    def _test(mask, x, y):
        xv, yv = x[mask], y[mask]
        if len(set(yv.tolist())) < 2:
            return "  (one class only)"
        auc = roc_auc_score(yv, xv)
        _, p = mannwhitneyu(xv[yv == 1], xv[yv == 0], alternative="greater")
        return f"AUC={auc:.2f} MWU(1-sided) p={p:.3f}{'*' if p < 0.05 else ''}"

    cortical = np.array([labels[s]["cortical"] for s in subs])[valid]
    dv = delta_asym[valid]
    print(f"     all subjects           n={valid.sum():2d}  {_test(np.ones_like(y_right, bool), dv, y_right)}")
    print(f"     CORTICAL lesions only  n={int(cortical.sum()):2d}  {_test(cortical, dv, y_right)}")
    print(f"     subcortical/brainstem  n={int((~cortical).sum()):2d}  {_test(~cortical, dv, y_right)}")
    print("     (delta hemispheric asymmetry predicting right-paralysis / left-lesion)")

    print("\n[3] Unsupervised 2-cluster split (band-power + connectivity), "
          "tested vs severity:")
    X = np.hstack([bp, conn]).astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    lab = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X)
    n0 = np.array(nihss)[lab == 0]
    n1 = np.array(nihss)[lab == 1]
    n0, n1 = n0[~np.isnan(n0)], n1[~np.isnan(n1)]
    u, p = mannwhitneyu(n0, n1, alternative="two-sided")
    print(f"    cluster A (n={len(n0)}) median NIHSS {np.median(n0):.1f}  vs  "
          f"cluster B (n={len(n1)}) median NIHSS {np.median(n1):.1f}")
    print(f"    Mann-Whitney U: p={p:.3f} {'(significant)' if p < 0.05 else '(n.s.)'}")

    print("\n[4] In-file findings (what the file surfaces to an LLM):")
    from mindfile import cluster as _cl
    _cl.run_clustering(mf, "bandpower", k=3)
    _cl.run_clustering(mf, "connectivity", k=3)
    for fnd in findings.discover(mf)[:6]:
        print(f"    - {fnd['statement']}")

    print("\n[5] Seal + verify:")
    checkpoint.create_checkpoint(mf, "stroke-cohort")
    signer = sign.Ed25519Signer()
    sign.sign_state(mf, signer)
    sign.trust_key(mf, signer.public_key)
    rep = audit.audit(mf)
    print(f"    AUDIT={rep['status']}  signature trusted+current="
          f"{sign.verify_signatures(mf)['trusted_attested']}  file={out.stat().st_size/1e6:.0f}MB")
    mf.close()


if __name__ == "__main__":
    main()
