import numpy as np

from mindfile import MindFile, modality, radiomics


def test_radiomics_dim_and_determinism():
    vol = np.random.default_rng(0).random((8, 8, 8)).astype("float32")
    f1 = radiomics.radiomics_features(vol)
    f2 = radiomics.radiomics_features(vol)
    assert f1.shape[0] == 21 and np.array_equal(f1, f2)   # 15 first-order + 6 texture
    assert len(radiomics.RADIOMICS_LABELS) == 21


def test_texture_distinguishes_smooth_from_noisy():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 16)
    smooth = np.stack([np.outer(x, x)] * 16).astype("float32")        # low texture
    noisy = rng.random((16, 16, 16)).astype("float32")               # high texture
    # GLCM contrast (texture idx 0) is much higher for noise than a smooth gradient
    c_smooth = radiomics.texture_features(radiomics.glcm(smooth))[0]
    c_noisy = radiomics.texture_features(radiomics.glcm(noisy))[0]
    assert c_noisy > c_smooth * 2


def test_dicom_pixel_features_as_second_space(mind_path):
    mf = MindFile.create(mind_path, modality="imaging")
    tags = {"PatientID": "P1", "Modality": "CT", "StudyDate": "20240103",
            "Rows": "16", "Columns": "16", "StudyInstanceUID": "1.2.3"}
    px = np.random.default_rng(0).random((16, 16)).astype("float32")
    item = modality.ingest(mf, "dicom", tags, pixels=px)
    # metadata card space + pixel radiomics space, both on the same item
    dims = {r["name"]: r["dim"] for r in mf.conn.execute(
        "SELECT s.name, v.dim FROM vectors v JOIN spaces s ON s.id=v.space_id WHERE v.item_id=?",
        (item,))}
    assert dims.get("dicom") == 256 and dims.get("dicom_pixels") == 21
    from mindfile import axes
    assert set(axes.get_axes(mf, item)) == {"x", "y", "z"}   # centroid from pixels
    mf.close()
