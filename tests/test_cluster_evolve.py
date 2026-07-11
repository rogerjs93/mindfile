from mindfile import MindFile, ingest, cluster, evolve


def _build(mind_path, blobs):
    mf = MindFile.create(mind_path, modality="synthetic")
    ingest.add_vectors(mf, blobs, external_ids=[f"p{i}" for i in range(len(blobs))])
    return mf


def test_cluster_v1(mind_path, three_blobs):
    mf = _build(mind_path, three_blobs)
    cid = cluster.run_clustering(mf, k=3)
    sizes = sorted(c["size"] for c in cluster.clusters_of(mf, cid))
    assert sizes == [20, 20, 20]
    assert "cluster" in [h["event"] for h in mf.history()]
    mf.close()


def test_split_detected_and_recorded(mind_path, three_blobs):
    mf = _build(mind_path, three_blobs)
    cluster.run_clustering(mf, k=2, labels={0: "healthy", 1: "tumor"})
    res = evolve.recluster(mf, k=3)
    assert len(res["splits"]) >= 1
    # The split's two children both descend from the same parent cluster.
    lin = {l["id"]: l["parent_cluster_id"] for l in evolve.lineage(mf, res["clustering_id"])}
    parents = [p for p in lin.values() if p is not None]
    assert len(parents) > len(set(parents))  # at least one parent reused -> a split
    assert "split" in [h["event"] for h in mf.history()]
    mf.close()


def test_prior_version_is_immutable(mind_path, three_blobs):
    mf = _build(mind_path, three_blobs)
    v1 = cluster.run_clustering(mf, k=2)
    before = cluster.assignments(mf, v1)
    evolve.recluster(mf, k=3)
    after = cluster.assignments(mf, v1)
    assert before == after  # re-clustering never mutates an earlier run
    mf.close()
