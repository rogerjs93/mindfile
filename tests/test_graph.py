from mindfile import MindFile, ingest, cluster, graph, query, index


def test_knn_edges_and_neighbors(mind_path, three_blobs):
    mf = MindFile.create(mind_path, modality="synthetic")
    ingest.add_vectors(mf, three_blobs, external_ids=[f"p{i}" for i in range(len(three_blobs))])
    n = graph.build_knn_edges(mf, k=4)
    assert n > 0
    nb = graph.neighbors(mf, 1)
    assert nb and all(e["dst_item_id"] != 1 for e in nb)
    weights = [e["weight"] for e in nb]
    assert weights == sorted(weights, reverse=True)
    mf.close()


def test_search_enriched_with_cluster(mind_path, three_blobs):
    mf = MindFile.create(mind_path, modality="synthetic")
    ingest.add_vectors(mf, three_blobs, external_ids=[f"p{i}" for i in range(len(three_blobs))])
    cluster.run_clustering(mf, k=3)
    _, mat = index.load_all_vectors(mf)
    res = query.search(mf, mat[0], k=3)
    assert res[0]["external_id"] == "p0"
    assert res[0]["cluster"] is not None
    mf.close()
