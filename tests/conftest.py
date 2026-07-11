import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _isolate_modality_registry():
    """Snapshot and restore the global modality registry around every test, so a
    test that registers/replaces a processor can't leak into later tests."""
    from mindfile import modality
    saved = dict(modality._REGISTRY)
    try:
        yield
    finally:
        modality._REGISTRY.clear()
        modality._REGISTRY.update(saved)


@pytest.fixture
def mind_path(tmp_path):
    return tmp_path / "test.mind"


@pytest.fixture
def three_blobs():
    """Two nearby sub-groups (A,B) + one far group (C): forces a clean k=2->k=3 split."""
    rng = np.random.default_rng(0)
    a = rng.normal((0, 0), 0.4, (20, 2))
    b = rng.normal((4, 0), 0.4, (20, 2))
    c = rng.normal((30, 30), 0.4, (20, 2))
    return np.vstack([a, b, c]).astype("float32")
