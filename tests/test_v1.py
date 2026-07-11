import numpy as np
import pytest

from mindfile import MindFile, ingest, cluster, index, sign, signal, validate


# -- trusted-key pinning ----------------------------------------------------
def test_trusted_attestation(mind_path):
    pytest.importorskip("cryptography")
    mf = MindFile.create(mind_path, modality="eeg")
    ingest.add_vectors(mf, np.random.default_rng(0).normal(0, 1, (6, 5)).astype("float32"))
    cluster.run_clustering(mf, k=2)
    signer = sign.Ed25519Signer()
    sign.sign_state(mf, signer)
    sign.trust_key(mf, signer.public_key)                 # pin the expected signer
    rep = sign.verify_signatures(mf)
    assert rep["trusted_attested"] is True and rep["signatures"][0]["trusted"] is True
    assert validate.validate(mf)["status"] == "ok"
    mf.close()


def test_untrusted_signer_flagged(mind_path):
    pytest.importorskip("cryptography")
    mf = MindFile.create(mind_path, modality="eeg")
    ingest.add_vectors(mf, np.random.default_rng(0).normal(0, 1, (6, 5)).astype("float32"))
    sign.sign_state(mf, sign.Ed25519Signer())             # signed by key A
    sign.trust_key(mf, sign.Ed25519Signer().public_key)   # but we trust key B
    rep = sign.verify_signatures(mf)
    assert rep["trusted_attested"] is False
    codes = {i["code"] for i in validate.validate(mf)["issues"]}
    assert "untrusted_state" in codes
    mf.close()


# -- IVF ANN search ---------------------------------------------------------
def test_ann_search_finds_self(mind_path):
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=200, centers=8, cluster_std=0.5, random_state=0)
    mf = MindFile.create(mind_path, modality="synthetic")
    ids = ingest.add_vectors(mf, X.astype("float32"))
    cluster.run_clustering(mf, k=8)
    _, mat = index.load_all_vectors(mf)
    # query with an existing point -> ANN (probing a couple cells) returns it #1
    res = index.search_ann(mf, mat[0], k=3, nprobe=2)
    assert res and res[0][0] == int(ids[0])
    # ANN top result agrees with exact search
    assert index.search(mf, mat[0], k=1)[0][0] == res[0][0]
    mf.close()


def test_ann_falls_back_without_clustering(mind_path):
    mf = MindFile.create(mind_path, modality="synthetic")
    ingest.add_vectors(mf, np.random.default_rng(0).normal(0, 1, (5, 4)).astype("float32"))
    res = index.search_ann(mf, np.zeros(4, "float32"), k=2)   # no clustering -> exact fallback
    assert len(res) == 2
    mf.close()


# -- lossy spectral waveform codec (opt-in) ---------------------------------
def test_spectral_codec_is_smaller_and_reconstructs_approximately(mind_path):
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4, 2000)
    base = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)
    data = np.stack([base + rng.normal(0, 0.02, 2000) for _ in range(8)]).astype("float32")

    mf = MindFile.create(mind_path, modality="eeg")
    item = ingest.add_vectors(mf, np.zeros((1, 1), "float32"), space="ph")[0]
    info = signal.store_signal(mf, item, data, sfreq=500.0, codec="spectral", keep_fraction=0.15)
    assert info["lossy"] is True and info["error"] < 0.1     # smooth signal -> low error
    assert info["code_bytes"] < info["raw_bytes"]            # and smaller

    recon, _, _ = signal.load_signal(mf, item)
    assert recon.shape == data.shape
    assert np.corrcoef(recon.ravel(), data.ravel())[0, 1] > 0.99
    # deterministic reconstruction is integrity-verifiable; validate flags it lossy
    assert signal.verify_signals(mf)["status"] == "ok"
    assert "lossy_signal" in {i["code"] for i in validate.validate(mf)["issues"]}
    mf.close()
