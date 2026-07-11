import numpy as np
import pytest

from mindfile import MindFile, ingest, signal, lazy
from mindfile.embedders import EEGEmbedder


def _smooth_eeg(n_ch=8, n_samp=2000, seed=0):
    """Synthetic band-limited multichannel signal (smooth in time, correlated channels)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4, n_samp)
    base = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)
    sig = np.stack([base * (1 + 0.1 * c) + rng.normal(0, 0.05, n_samp) for c in range(n_ch)])
    return sig.astype("float32")


@pytest.mark.parametrize("dtype", ["float32", "int16"])
def test_codec_roundtrip_is_exact(dtype):
    data = _smooth_eeg()
    if dtype == "int16":
        data = (data * 1000).astype("int16")
    code, meta = signal.encode(data)
    out = signal.decode(code, meta)
    assert out.dtype == data.dtype
    np.testing.assert_array_equal(out, data)
    # and it actually shrinks a smooth signal
    assert len(code) < data.nbytes


def test_store_and_load_exact(mind_path):
    mf = MindFile.create(mind_path, modality="eeg")
    item = ingest.add_vectors(mf, np.zeros((1, 1), "float32"), space="placeholder",
                              subject_ids="P0", modalities="eeg")[0]
    data = _smooth_eeg()
    info = signal.store_signal(mf, item, data, sfreq=500.0, ch_names=[f"c{i}" for i in range(8)])
    assert info["ratio"] > 1.0
    back, sf, ch = signal.load_signal(mf, item, verify=True)
    np.testing.assert_array_equal(back, data)
    assert sf == 500.0 and ch[0] == "c0"
    assert signal.verify_signals(mf)["status"] == "ok"
    mf.close()


def test_verify_signals_detects_tamper(mind_path):
    mf = MindFile.create(mind_path, modality="eeg")
    item = ingest.add_vectors(mf, np.zeros((1, 1), "float32"), space="placeholder")[0]
    signal.store_signal(mf, item, _smooth_eeg(), sfreq=500.0)
    bad = mf.conn.execute("SELECT code FROM signals WHERE item_id=?", (item,)).fetchone()[0]
    mf.conn.execute("UPDATE signals SET code=? WHERE item_id=?", (bad + b"\x00", item))
    mf.conn.commit()
    assert signal.verify_signals(mf)["status"] == "corrupted"
    mf.close()


def test_lazy_feature_recomputes_from_waveform(mind_path):
    mf = MindFile.create(mind_path, modality="eeg")
    item = ingest.add_vectors(mf, np.zeros((1, 1), "float32"), space="placeholder")[0]
    data = _smooth_eeg()
    signal.store_signal(mf, item, data, sfreq=500.0)

    emb = EEGEmbedder(crop_seconds=None)          # use full signal
    signal.define_feature_vector(mf, item, "bandpower", emb)
    # the feature vector is virtual until called
    assert lazy.is_virtual(mf, item, "bandpower")
    feat = lazy.resolve(mf, item, "bandpower")
    expected = emb._features_from_array(data.astype("float64"), 500.0)
    np.testing.assert_allclose(feat, expected, rtol=1e-6, atol=1e-6)
    assert feat.shape[0] == emb.dim
    # recipe integrity tracks the waveform
    assert lazy.verify_defs(mf)["status"] == "ok"
    mf.close()
