import numpy as np

from mindfile import (MindFile, ingest, axes, constellation, allocate, stats,
                      cluster, checkpoint)


def _cohort(mind_path, n=6):
    """n patients, each EEG (5d) + MRI (8d) at t0, with x,y,z,t axes on EEG items."""
    mf = MindFile.create(mind_path, modality="eeg")
    rng = np.random.default_rng(0)
    eeg = []
    for i, p in enumerate([f"P{j}" for j in range(n)]):
        eeg += ingest.add_vectors(mf, rng.normal(i, 0.4, (1, 5)).astype("float32"), space="bandpower",
                                  subject_ids=p, session_ids="t0", modalities="eeg", external_ids=f"{p}-eeg")
        ingest.add_vectors(mf, rng.normal(i, 0.4, (1, 8)).astype("float32"), space="mri",
                           subject_ids=p, session_ids="t0", modalities="mri", external_ids=f"{p}-mri")
    axes.set_axes_bulk(mf, eeg, rng.normal(0, 1, (len(eeg), 4)).astype("float32"), ["x", "y", "z", "t"])
    return mf, eeg


def test_constellation_space_is_additive(mind_path):
    mf, eeg = _cohort(mind_path)
    cp = checkpoint.create_checkpoint(mf, "primary")
    r = constellation.build_constellation_space(mf, "bandpower")
    assert r["created"] == len(eeg)
    assert r["dim"] == 5 + 4                       # semantic 5d + x,y,z,t
    # originals untouched
    assert checkpoint.verify_checkpoint(mf, "primary")["status"] == "ok"
    # per-patient constellation = collection + (here single point) graph
    out = constellation.build_constellation(mf, "P0")
    assert out["points"] == 1
    mf.close()


def test_clustering_ari_self_is_one(mind_path):
    mf, _ = _cohort(mind_path)
    cid = cluster.run_clustering(mf, "bandpower", k=3)
    res = stats.clustering_agreement(mf, cid, cid)
    assert res["ari"] == 1.0
    mf.close()


def test_allocator_plans_reviews_renders(mind_path):
    mf, _ = _cohort(mind_path)
    cp = checkpoint.create_checkpoint(mf, "primary")
    res = allocate.orchestrate(mf)
    assert res["proposed"] >= 5 and res["rendered"] == res["approved"]
    # the plan was logged in-file
    assert mf.counts()["plans"] == 1
    assert mf.counts()["plan_actions"] == res["proposed"]
    # artifacts materialized: clusterings, a fused space, a constellation space
    names = {s["name"] for s in __import__("mindfile").spaces.list_spaces(mf)}
    assert {"bandpower", "mri", "fused", "constellation"} <= names
    assert mf.counts()["clusterings"] >= 2
    # originals conserved through the whole allocation
    assert checkpoint.verify_checkpoint(mf, "primary")["status"] == "ok"
    mf.close()


def test_allocator_size_budget_blocks_constellation(mind_path):
    mf, _ = _cohort(mind_path)
    allocate.orchestrate(mf, budget={"max_new_vectors": 0})
    names = {s["name"] for s in __import__("mindfile").spaces.list_spaces(mf)}
    assert "constellation" not in names          # rejected by budget
    # the rejection was recorded with a reason
    reasons = [r[0] for r in mf.conn.execute(
        "SELECT review_reason FROM plan_actions WHERE action='constellation'")]
    assert any("budget" in (r or "") for r in reasons)
    mf.close()


def test_allocator_records_correlations(mind_path):
    mf, _ = _cohort(mind_path)
    allocate.orchestrate(mf)
    allocate.orchestrate(mf)                      # second pass: file self-converges
    kinds = {c["kind"] for c in stats.list_correlations(mf)}
    assert {"feature_corr", "subject_cross_modality", "cross_space", "clustering_ari"} <= kinds
    mf.close()
