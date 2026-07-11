from mindfile import MindFile, modality, render

CT = {"PatientID": "P1", "PatientSex": "M", "StudyInstanceUID": "1.2.3",
      "StudyDate": "20240103", "StudyDescription": "HEAD",
      "SeriesDescription": "Axial 5mm", "Modality": "CT",
      "BodyPartExamined": "HEAD", "Rows": "512", "Columns": "512",
      "AccessionNumber": "ACC9"}
MR = {"PatientID": "P1", "StudyInstanceUID": "1.2.4", "StudyDate": "20240105",
      "SeriesDescription": "T1 MPRAGE", "Modality": "MR", "BodyPartExamined": "BRAIN",
      "Rows": "256", "Columns": "256"}


def test_dicom_card_from_tags(mind_path):
    mf = MindFile.create(mind_path, modality="imaging")
    cid = modality.ingest(mf, "dicom", CT)
    mid = modality.ingest(mf, "dicom", MR)
    assert "CT HEAD, 'Axial 5mm', 512x512, study 2024-01-03" in render.card(mf, cid)
    assert "MR BRAIN, 'T1 MPRAGE', 256x256, study 2024-01-05" in render.card(mf, mid)
    mf.close()


def test_dicom_identity_mapping(mind_path):
    mf = MindFile.create(mind_path, modality="imaging")
    modality.ingest(mf, "dicom", CT)
    r = mf.conn.execute("SELECT subject_id, session_id, acquired_at, modality FROM items").fetchone()
    assert (r["subject_id"], r["session_id"], r["acquired_at"], r["modality"]) == \
        ("P1", "1.2.3", "2024-01-03", "dicom")
    mf.close()


def test_dicom_in_digest(mind_path):
    mf = MindFile.create(mind_path, modality="imaging")
    modality.ingest(mf, "dicom", CT)
    modality.ingest(mf, "dicom", MR)
    text = render.digest(mf)
    assert "[dicom]" in text and "T1 MPRAGE" in text
    mf.close()
