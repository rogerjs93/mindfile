import numpy as np
import pytest

from mindfile import MindFile, lazy, modality


def test_fif_ingest_via_eeg_processor(mind_path, tmp_path):
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(0)
    data = rng.normal(size=(4, 512)) * 1e-6
    info = mne.create_info(["a", "b", "c", "d"], 128.0, "eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    path = tmp_path / "sub_raw.fif"                 # MNE requires a *raw.fif suffix
    raw.save(str(path), verbose="ERROR")

    mf = MindFile.create(mind_path, modality="eeg")
    iid = modality.ingest(mf, "eeg", str(path), subject_id="S")
    # FIF has no int16 digital form -> stored as float32 (no calibration)
    assert mf.conn.execute("SELECT dtype FROM signals WHERE item_id=?",
                           (iid,)).fetchone()[0] == "<f4"
    assert lazy.resolve(mf, iid, "bandpower").shape == (10,)
    assert lazy.resolve(mf, iid, "connectivity").shape == (5,)
    mf.close()
