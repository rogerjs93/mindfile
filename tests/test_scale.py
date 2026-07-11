import numpy as np

from mindfile import MindFile, ingest, index


def test_streamed_search_is_exact(mind_path):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (50, 8)).astype("float32")
    mf = MindFile.create(mind_path, modality="synthetic")
    ids = ingest.add_vectors(mf, X)

    q = X[7]
    qn = q / np.linalg.norm(q)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    ref = [int(ids[i]) for i in np.argsort(-(Xn @ qn))[:5]]

    # tiny batch => 8 streamed passes over 50 vectors; must equal brute-force
    got = [i for i, _ in index.search(mf, q, k=5, batch=7)]
    assert got == ref and got[0] == int(ids[7])
    mf.close()


def test_iter_vectors_bounds_batch_size(mind_path):
    mf = MindFile.create(mind_path, modality="synthetic")
    ingest.add_vectors(mf, np.zeros((20, 4), "float32"))
    sizes = [len(b_ids) for b_ids, _ in index.iter_vectors(mf, batch=6)]
    assert sizes == [6, 6, 6, 2]        # never more than `batch` vectors in RAM at once
    mf.close()
