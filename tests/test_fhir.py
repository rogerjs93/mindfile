from mindfile import MindFile, modality, render


PATIENT = {"resourceType": "Patient", "id": "P1", "gender": "male",
           "birthDate": "1966-04-12", "name": [{"family": "Doe", "given": ["John"]}]}
HB = {"resourceType": "Observation", "subject": {"reference": "Patient/P1"},
      "effectiveDateTime": "2024-01-03",
      "code": {"text": "Hemoglobin"}, "valueQuantity": {"value": 13.5, "unit": "g/dL"}}
STROKE = {"resourceType": "Condition", "subject": {"reference": "Patient/P1"},
          "onsetDateTime": "2024-01-02",
          "clinicalStatus": {"coding": [{"code": "active"}]},
          "code": {"text": "Ischemic stroke"}}


def test_fhir_cards_are_normalized_text(mind_path):
    mf = MindFile.create(mind_path, modality="ehr")
    pid = modality.ingest(mf, "fhir", PATIENT)
    oid = modality.ingest(mf, "fhir", HB)
    cid = modality.ingest(mf, "fhir", STROKE)
    assert render.card(mf, pid).startswith("[fhir]") and "Patient John Doe, male, born 1966" in render.card(mf, pid)
    assert "Hemoglobin: 13.5 g/dL (2024-01-03)" in render.card(mf, oid)
    assert "Ischemic stroke (active)" in render.card(mf, cid)
    mf.close()


def test_fhir_identity_links_to_subject(mind_path):
    mf = MindFile.create(mind_path, modality="ehr")
    modality.ingest(mf, "fhir", HB)         # subject derived from "Patient/P1" reference
    r = mf.conn.execute("SELECT subject_id, modality, acquired_at FROM items").fetchone()
    assert r["subject_id"] == "P1" and r["modality"] == "fhir" and r["acquired_at"] == "2024-01-03"
    mf.close()


def test_fhir_bundle_ingests_all_entries(mind_path):
    mf = MindFile.create(mind_path, modality="ehr")
    bundle = {"resourceType": "Bundle", "entry": [{"resource": PATIENT},
                                                  {"resource": HB}, {"resource": STROKE}]}
    ids = modality.ingest(mf, "fhir", bundle)
    assert isinstance(ids, list) and len(ids) == 3
    assert mf.conn.execute("SELECT COUNT(*) FROM items WHERE modality='fhir'").fetchone()[0] == 3
    # all readable in the digest
    text = render.digest(mf)
    assert "Hemoglobin" in text and "Ischemic stroke" in text
    mf.close()
