import numpy as np

from mindfile import MindFile, cluster, ingest


def _three_blobs(n=180, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[5, 0, 0, 0, 0, 0], [0, 5, 0, 0, 0, 0], [0, 0, 5, 0, 0, 0]],
                       dtype="float32")[:, :dim]
    X = np.vstack([c + 0.3 * rng.normal(size=(n // 3, dim)) for c in centers]).astype("float32")
    truth = np.repeat([0, 1, 2], n // 3)
    return X, truth


def test_minibatch_is_out_of_core_and_recovers_blobs(mind_path):
    X, truth = _three_blobs()
    mf = MindFile.create(mind_path, modality="synthetic")
    ids = ingest.add_vectors(mf, X)

    # tiny batch => streamed in ~9 chunks; whole matrix never resident
    cid = cluster.run_clustering(mf, k=3, algo="minibatch", batch=20, epochs=8)
    asn = cluster.assignments(mf, cid)

    # every item assigned exactly once; sizes sum to n
    assert len(asn) == len(ids)
    sizes = [c["size"] for c in cluster.clusters_of(mf, cid)]
    assert sum(sizes) == len(ids)

    # clustering recovers the 3 blobs: each true blob lands in one dominant cluster
    lab = np.array([asn[int(i)] for i in ids])
    for t in (0, 1, 2):
        counts = np.bincount(lab[truth == t])
        assert counts.max() / counts.sum() > 0.9
    mf.close()


def test_minibatch_matches_kmeans_partition(mind_path):
    X, _ = _three_blobs(seed=1)
    mf = MindFile.create(mind_path, modality="synthetic")
    ids = ingest.add_vectors(mf, X)
    exact = cluster.assignments(mf, cluster.run_clustering(mf, k=3, algo="kmeans"))
    ooc = cluster.assignments(mf, cluster.run_clustering(mf, k=3, algo="minibatch",
                                                         batch=32, epochs=8))
    # same co-membership structure (labels may be permuted): pairs together in
    # exact stay together in streamed, for well-separated blobs
    il = list(ids)[:60]
    agree = sum((exact[int(a)] == exact[int(b)]) == (ooc[int(a)] == ooc[int(b)])
                for a in il for b in il)
    assert agree / (len(il) ** 2) > 0.95
    mf.close()
