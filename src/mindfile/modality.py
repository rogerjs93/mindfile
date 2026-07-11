"""Per-modality processing protocols.

A ``ModalityProcessor`` encapsulates the *right* pipeline for one health-data
type: how to ingest it, how to store it (waveform / volume / tabular), and which
embedding ``spaces`` it produces — so multimodal ingest uses each data type to
its strengths instead of expecting pre-made vectors. Register processors and
dispatch by modality; fusion/constellations/correlations then compose across the
spaces they produce.

Two reference processors ship here (EEG + tabular/labs). Heavier ones (MRI/NIfTI,
DICOM, ECG, genomics, clinical text) plug in the same way.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModalityProcessor(Protocol):
    modality: str
    spaces: list[str]

    def ingest(self, mf, source, **identity) -> int:
        """Store one record and produce its vectors; return the new item id."""
        ...


_REGISTRY: dict[str, "ModalityProcessor"] = {}


def register(proc: "ModalityProcessor") -> "ModalityProcessor":
    _REGISTRY[proc.modality] = proc
    return proc


def get(modality: str) -> "ModalityProcessor":
    if modality not in _REGISTRY:
        raise KeyError(f"no processor registered for modality {modality!r} "
                       f"(have {sorted(_REGISTRY)})")
    return _REGISTRY[modality]


def processors() -> dict[str, "ModalityProcessor"]:
    return dict(_REGISTRY)


def describe() -> list[dict]:
    """What protocol each registered modality uses (introspection)."""
    return [{"modality": p.modality, "spaces": list(p.spaces),
             "processor": type(p).__name__} for p in _REGISTRY.values()]


# Static per-modality annotations, merged with the live registry by ``catalog()``.
# Extensions/deps are documentation; ``spaces`` and ``card`` come from live state.
_CATALOG_META = {
    "eeg": {"exts": [".edf", ".bdf", ".fif", "array"], "storage": "lossless waveform "
            "(int16 digital for EDF)", "deps": ["mne (non-EDF paths)", "scipy"]},
    "ecg": {"exts": ["array", ".edf"], "storage": "lossless waveform",
            "deps": ["scipy"]},
    "wfdb": {"exts": [".dat/.hea", "array"], "storage": "lossless waveform",
             "deps": ["wfdb (record paths)"]},
    "mri": {"exts": [".nii", ".nii.gz", "3D array"], "storage": "radiomics vector "
            "(volume not inlined)", "deps": ["nibabel (NIfTI paths)"]},
    "dicom": {"exts": [".dcm", "tag dict"], "storage": "tags JSON (+optional pixel "
              "radiomics)", "deps": ["pydicom (paths/pixels)"]},
    "fhir": {"exts": [".json", "dict", "Bundle"], "storage": "resource JSON", "deps": []},
    "text": {"exts": [".txt", "str"], "storage": "note text",
             "deps": ["sentence-transformers (optional upgrade)"]},
    "genomics": {"exts": [".vcf", "variant list"], "storage": "variant list", "deps": []},
    "labs": {"exts": [".csv", "dict"], "storage": "row JSON + reference-range flags",
             "deps": []},
    "tabular": {"exts": [".csv", "dict", "array"], "storage": "row JSON", "deps": []},
}


def catalog() -> list[dict]:
    """Authoritative consolidation of every supported file type: one row per
    registered modality, merging the live registry (spaces, card availability)
    with static metadata (extensions, storage, optional deps)."""
    from .render import _RENDERERS
    out = []
    for m, p in sorted(_REGISTRY.items()):
        meta = _CATALOG_META.get(m, {})
        out.append({"modality": m, "processor": type(p).__name__,
                    "spaces": list(p.spaces), "card": m in _RENDERERS,
                    "extensions": meta.get("exts", []), "storage": meta.get("storage", ""),
                    "deps": meta.get("deps", [])})
    return out


def ingest(mf, modality: str, source, **identity) -> int:
    """Dispatch ingest of ``source`` to the processor for ``modality``."""
    return get(modality).ingest(mf, source, **identity)


# -- reference processor: EEG (waveform store + lazy spectral features) ------
class EEGProcessor:
    modality = "eeg"

    def __init__(self, embedder=None, crop_seconds: float | None = 60.0,
                 connectivity: bool = True, asymmetry: bool = True):
        from .embedders.eeg import AsymmetryEmbedder, ConnectivityEmbedder, EEGEmbedder
        self.embedder = embedder or EEGEmbedder(crop_seconds=crop_seconds)
        self.conn_embedder = (ConnectivityEmbedder(crop_seconds=crop_seconds)
                              if connectivity else None)
        self.asym_embedder = (AsymmetryEmbedder(crop_seconds=crop_seconds)
                              if asymmetry else None)
        self.spaces = (["bandpower"] + (["connectivity"] if connectivity else [])
                       + (["asymmetry"] if asymmetry else []))

    def ingest(self, mf, source, *, sfreq: float | None = None, ch_names=None,
               **identity) -> int:
        from . import signal
        calibration = None
        if isinstance(source, (str, Path)):           # an EDF/BDF/FIF file path
            data, sfreq, ch_names, calibration = self._read_path(str(source))
        else:                                          # an array (channels, time)
            data = np.asarray(source, dtype="float32")
        if sfreq is None:
            raise ValueError("EEG ingest needs sfreq (or an EDF path)")
        item_id, _ = signal.add_recording(mf, data, sfreq=sfreq, ch_names=ch_names,
                                          modality="eeg", embedder=self.embedder,
                                          feature_space="bandpower",
                                          calibration=calibration, **identity)
        if self.conn_embedder is not None:            # lazy connectivity from the same waveform
            signal.define_connectivity_vector(mf, item_id, "connectivity", self.conn_embedder)
        if self.asym_embedder is not None:            # lazy hemispheric asymmetry
            signal.define_asymmetry_vector(mf, item_id, "asymmetry", self.asym_embedder)
        if self.conn_embedder is not None or self.asym_embedder is not None:
            mf.conn.commit()
        return item_id

    @staticmethod
    def _read_path(path: str):
        """Read a biosignal file. For ``.edf`` prefer the dependency-free int16
        *digital* reader (lossless + compact); for FIF/BDF/other MNE-readable
        formats fall back to MNE float64 (no digital calibration)."""
        from . import edf
        if path.lower().endswith(".edf"):
            try:
                e = edf.read_edf_digital(path)
                cal = {"gain": e["gain"].tolist(), "offset": e["offset"].tolist(),
                       "unit": e["unit"]}
                return e["digital"], e["sfreq"], e["ch_names"], cal
            except (ValueError, OSError):
                pass                                   # fall through to MNE
        import mne
        raw = mne.io.read_raw(path, preload=True, verbose="ERROR")  # fif/bdf/edf/...
        return (raw.get_data().astype("float32"), raw.info["sfreq"], raw.ch_names, None)


# -- reference processor: tabular / labs (row payload + feature vector) -------
class TabularProcessor:
    def __init__(self, feature_names=None, *, modality: str = "labs", space: str = "labs",
                 normalize: dict | None = None):
        # feature_names may be omitted: it locks to the keys of the first dict
        # ingested, so the default processor works via modality.ingest(mf,"labs",{...}).
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.modality = modality
        self.space = space
        self.spaces = [space]
        self.normalize = normalize  # name -> (mean, std) for optional z-scoring

    def ingest(self, mf, source, **identity) -> int:
        from . import ingest as _ingest
        names = self.feature_names
        if names is None:
            # Inferred PER INGEST and NOT persisted — the registered default is a
            # shared singleton, so locking it to one file's columns would corrupt
            # other files' rows. Construct TabularProcessor(feature_names=[...]) for
            # a fixed cohort layout.
            if not isinstance(source, dict):
                raise ValueError("tabular ingest needs feature_names, or a dict source "
                                 "to infer them from")
            names = sorted(source)
        if isinstance(source, dict):
            row = {n: float(source.get(n, np.nan)) for n in names}
        else:
            row = {n: float(v) for n, v in zip(names, np.asarray(source).ravel())}
        vec = np.array([row[n] for n in names], dtype="float32")
        if self.normalize:
            vec = np.array([(row[n] - self.normalize[n][0]) / (self.normalize[n][1] or 1.0)
                            for n in names], dtype="float32")
        ids = _ingest.add_vectors(
            mf, vec[None, :], space=self.space, payloads=[json.dumps(row)],
            modalities=self.modality, embedder_id=f"tabular:{self.modality}",
            subject_ids=identity.get("subject_id"), session_ids=identity.get("session_id"),
            study_ids=identity.get("study_id"), acquired_ats=identity.get("acquired_at"),
            external_ids=identity.get("external_id"))
        return ids[0]


def _identity(identity: dict) -> dict:
    """Map singular identity kwargs to the plural sequence kwargs add_vectors uses."""
    return {f"{k}s": identity.get(k) for k in
            ("subject_id", "session_id", "study_id", "acquired_at", "external_id")}


# -- ECG / EMG (reuses the waveform store; spectral features, ECG bands) ------
ECG_BANDS = {"low": (0.5, 5.0), "qrs": (5.0, 15.0), "high": (15.0, 40.0)}


class ECGProcessor:
    modality = "ecg"
    spaces = ["ecg_bandpower", "ecg_hrv"]

    def __init__(self, crop_seconds: float | None = None):
        from .embedders.eeg import EEGEmbedder
        self.embedder = EEGEmbedder(bands=ECG_BANDS, crop_seconds=crop_seconds,
                                    id="ecg-bandpower-v1")

    def ingest(self, mf, source, *, sfreq: float | None = None, ch_names=None, **identity) -> int:
        from . import ecg_features
        from . import ingest as _ingest
        from . import signal
        data = np.asarray(source, dtype="float32")
        if data.ndim == 1:
            data = data[None, :]
        if sfreq is None:
            raise ValueError("ECG ingest needs sfreq")
        item_id, _ = signal.add_recording(mf, data, sfreq=sfreq, ch_names=ch_names,
                                          modality="ecg", embedder=self.embedder,
                                          feature_space="ecg_bandpower", **identity)
        # HR/HRV — the clinically meaningful features — as a second space.
        hrv = ecg_features.hrv_features(data, sfreq)
        _ingest.add_space_vectors(mf, [item_id], hrv[None, :], space="ecg_hrv",
                                  embedder_id="ecg-hrv-v1")
        return item_id


# -- clinical text / notes (dependency-free hashing embedder) ----------------
class HashingTextEmbedder:
    def __init__(self, dim: int = 256, id: str = "text-hashing-v1"):
        self.dim = int(dim)
        self.id = id

    def embed(self, payloads) -> np.ndarray:
        out = np.zeros((len(payloads), self.dim), dtype=np.float32)
        for i, p in enumerate(payloads):
            for tok in str(p).lower().split():
                b = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                out[i, int.from_bytes(b, "big") % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class TextProcessor:
    modality = "text"
    spaces = ["text"]

    def __init__(self, embedder=None):
        # Optional upgrade: pass a sentence-transformers wrapper as `embedder`.
        self.embedder = embedder or HashingTextEmbedder()

    def ingest(self, mf, source, **identity) -> int:
        from . import ingest as _ingest
        ids = _ingest.add_items(mf, [str(source)], space="text", embedder=self.embedder,
                                modalities="text", **_identity(identity))
        return ids[0]


# -- MRI / CT volumes (first-order + GLCM texture radiomics) ------------------
class MRIProcessor:
    modality = "mri"
    spaces = ["radiomics"]

    def ingest(self, mf, source, **identity) -> int:
        from . import axes as _axes
        from . import ingest as _ingest
        from . import radiomics as _rad
        arr, ref = self._load(source)
        feats = _rad.radiomics_features(arr)   # first-order + GLCM texture
        ids = _ingest.add_vectors(mf, feats[None, :], space="radiomics",
                                  payloads=[ref], payload_kind="path",
                                  modalities="mri", **_identity(identity))
        item = ids[0]
        cx, cy, cz = _rad.centroid(arr)        # x,y,z feed the constellation space
        _axes.set_axes(mf, item, {"x": cx, "y": cy, "z": cz})
        return item

    def _load(self, source):
        if isinstance(source, (str, Path)):
            try:
                import nibabel as nib
            except ImportError as e:
                raise RuntimeError("MRI from a NIfTI path needs `nibabel`; "
                                   "or pass a numpy 3-D array directly") from e
            return np.asarray(nib.load(str(source)).get_fdata(), dtype=np.float32), str(source)
        return np.asarray(source, dtype=np.float32), "array"


# -- genomics / variants (hashed bag-of-variants) ----------------------------
class GenomicsProcessor:
    modality = "genomics"
    spaces = ["variants"]

    def __init__(self, dim: int = 512):
        self.dim = int(dim)

    def ingest(self, mf, source, **identity) -> int:
        from . import ingest as _ingest
        variants = self._variants(source)
        vec = np.zeros(self.dim, dtype=np.float32)
        for v in variants:
            b = hashlib.blake2b(v.encode("utf-8"), digest_size=8).digest()
            vec[int.from_bytes(b, "big") % self.dim] += 1.0
        n = np.linalg.norm(vec)
        if n:
            vec /= n
        ids = _ingest.add_vectors(mf, vec[None, :], space="variants",
                                  payloads=[json.dumps(variants)], modalities="genomics",
                                  **_identity(identity))
        return ids[0]

    def _variants(self, source) -> list[str]:
        # Accept a VCF file path, raw VCF text, a single variant string, or a
        # list of variant strings. (A bare string is NOT iterated char-by-char.)
        text = None
        if isinstance(source, (str, Path)):
            p = Path(source) if len(str(source)) < 260 else None
            if p is not None and p.exists():
                text = p.read_text(encoding="utf-8")
            elif isinstance(source, str) and ("\n" in source or source.startswith("#")):
                text = source                              # VCF text passed directly
            else:
                return [str(source)]                        # a single variant token
        if text is not None:
            out = []
            for line in text.splitlines():
                if not line or line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) >= 5:
                    out.append(f"{f[0]}:{f[1]}:{f[3]}:{f[4]}")  # CHROM:POS:REF:ALT
            return out
        return [str(v) for v in source]                     # a list of variant tokens


# -- FHIR (EHR resources — already structured, so the card is a normalization) -
def _codeable(cc) -> str:
    if not cc:
        return ""
    if isinstance(cc, list):
        cc = cc[0] if cc else {}
    if cc.get("text"):
        return cc["text"]
    for c in cc.get("coding", []):
        if c.get("display"):
            return c["display"]
        if c.get("code"):
            return c["code"]
    return ""


def _ref_id(ref):
    if not ref:
        return None
    r = ref.get("reference") if isinstance(ref, dict) else str(ref)
    return r.split("/")[-1] if r else None


def _fhir_subject(resource) -> str | None:
    if resource.get("resourceType") == "Patient":
        return resource.get("id")
    return _ref_id(resource.get("subject") or resource.get("patient"))


def _fhir_date(resource):
    for k in ("effectiveDateTime", "issued", "recordedDate", "onsetDateTime", "authoredOn", "date"):
        if resource.get(k):
            return resource[k]
    p = resource.get("effectivePeriod") or resource.get("period")
    return p.get("start") if isinstance(p, dict) else None


def _obs_value(resource) -> str:
    if "valueQuantity" in resource:
        q = resource["valueQuantity"]
        return f"{q.get('value', '')} {q.get('unit', '')}".strip()
    for k in ("valueString", "valueBoolean", "valueInteger"):
        if k in resource:
            return str(resource[k])
    if "valueCodeableConcept" in resource:
        return _codeable(resource["valueCodeableConcept"])
    if "component" in resource:  # e.g. blood pressure
        return "; ".join(f"{_codeable(c.get('code'))} {_obs_value(c)}".strip()
                         for c in resource["component"])
    return ""


def _patient_name(resource) -> str:
    n = resource.get("name")
    if isinstance(n, list) and n:
        n0 = n[0]
        if n0.get("text"):
            return n0["text"]
        return f"{' '.join(n0.get('given', []))} {n0.get('family', '')}".strip()
    return resource.get("id", "")


def fhir_card(resource: dict) -> str:
    """Render a FHIR resource as one LLM-readable line."""
    t = resource.get("resourceType", "Resource")
    if t == "Patient":
        return f"Patient {_patient_name(resource)}, {resource.get('gender', '?')}, " \
               f"born {resource.get('birthDate', '?')}"
    if t == "Observation":
        d = _fhir_date(resource)
        s = f"{_codeable(resource.get('code'))}: {_obs_value(resource)}".strip(": ")
        return s + (f" ({d})" if d else "")
    if t == "Condition":
        return f"{_codeable(resource.get('code'))} " \
               f"({_codeable(resource.get('clinicalStatus')) or 'condition'})"
    if t == "MedicationRequest":
        med = _codeable(resource.get("medicationCodeableConcept")) or _ref_id(resource.get("medicationReference"))
        return f"Medication: {med}"
    if t == "Procedure":
        return f"Procedure: {_codeable(resource.get('code'))}"
    if t == "AllergyIntolerance":
        return f"Allergy: {_codeable(resource.get('code'))}"
    if t == "DiagnosticReport":
        return f"{_codeable(resource.get('code'))}: {resource.get('conclusion', '')}".strip(": ")
    if t == "Encounter":
        cls = resource.get("class", {})
        return f"Encounter {cls.get('code', '') if isinstance(cls, dict) else cls}".strip()
    return f"{t}: {_codeable(resource.get('code')) or resource.get('id', '')}"


class FHIRProcessor:
    modality = "fhir"
    spaces = ["fhir"]

    def __init__(self, embedder=None):
        self.embedder = embedder or HashingTextEmbedder(dim=256, id="fhir-hashing-v1")

    def ingest(self, mf, source, **identity):
        res = self._load(source)
        if res.get("resourceType") == "Bundle":
            return [self._one(mf, e["resource"], identity)
                    for e in res.get("entry", []) if e.get("resource")]
        return self._one(mf, res, identity)

    def _one(self, mf, resource, identity) -> int:
        from . import ingest as _ingest
        card = fhir_card(resource)
        rtype = resource.get("resourceType", "Resource")
        ids = _ingest.add_vectors(
            mf, self.embedder.embed([card]), space="fhir", embedder_id=self.embedder.id,
            payloads=[json.dumps(resource)], modalities="fhir",
            subject_ids=identity.get("subject_id") or _fhir_subject(resource),
            session_ids=identity.get("session_id") or _ref_id(resource.get("encounter")),
            study_ids=identity.get("study_id"),
            acquired_ats=identity.get("acquired_at") or _fhir_date(resource),
            external_ids=identity.get("external_id") or f"{rtype}/{resource.get('id', '')}",
            metas=[{"resourceType": rtype}])
        return ids[0]

    def _load(self, source) -> dict:
        if isinstance(source, dict):
            return source
        s = str(source)
        if isinstance(source, Path) or (len(s) < 260 and s.endswith(".json") and Path(s).exists()):
            with open(source, encoding="utf-8") as f:
                return json.load(f)
        return json.loads(s)


# -- DICOM (imaging — the metadata tags are as valuable as the pixels) --------
_DICOM_TAGS = ["PatientID", "PatientName", "PatientBirthDate", "PatientSex",
               "StudyInstanceUID", "StudyDate", "StudyDescription", "AccessionNumber",
               "SeriesInstanceUID", "SeriesNumber", "SeriesDescription", "Modality",
               "BodyPartExamined", "Rows", "Columns", "SliceThickness", "PixelSpacing",
               "Manufacturer", "ProtocolName", "ContrastBolusAgent"]


def _dicom_date(s):
    s = str(s or "")
    return f"{s[:4]}-{s[4:6]}-{s[6:]}" if len(s) == 8 and s.isdigit() else (s or None)


def _dicom_tags(source) -> dict:
    if isinstance(source, dict):
        return {t: source[t] for t in _DICOM_TAGS if source.get(t) not in (None, "")}
    if isinstance(source, (str, Path)):
        try:
            import pydicom
        except ImportError as e:
            raise RuntimeError("reading .dcm needs `pydicom`; or pass a dict of tags") from e
        source = pydicom.dcmread(str(source), stop_before_pixels=True, force=True)
    out = {}
    for t in _DICOM_TAGS:                       # a pydicom Dataset (attribute access)
        v = getattr(source, t, None)
        if v not in (None, ""):
            out[t] = str(v)
    return out


def dicom_card(tags: dict) -> str:
    """Render DICOM metadata as one LLM-readable line."""
    head = f"{tags.get('Modality', 'IMG')} {tags.get('BodyPartExamined', '')}".strip()
    desc = tags.get("SeriesDescription") or tags.get("StudyDescription")
    dims = f"{tags['Rows']}x{tags['Columns']}" if tags.get("Rows") and tags.get("Columns") else None
    date = _dicom_date(tags.get("StudyDate"))
    parts = [head] + [p for p in (f"'{desc}'" if desc else None, dims,
                                  f"study {date}" if date else None) if p]
    return ", ".join(parts)


class DICOMProcessor:
    modality = "dicom"
    spaces = ["dicom", "dicom_pixels"]

    def __init__(self, embedder=None):
        self.embedder = embedder or HashingTextEmbedder(dim=256, id="dicom-hashing-v1")

    def ingest(self, mf, source, *, pixels=None, **identity) -> int:
        from . import ingest as _ingest
        tags = _dicom_tags(source)
        ids = _ingest.add_vectors(
            mf, self.embedder.embed([dicom_card(tags)]), space="dicom",
            embedder_id=self.embedder.id, payloads=[json.dumps(tags)], modalities="dicom",
            subject_ids=identity.get("subject_id") or tags.get("PatientID"),
            session_ids=identity.get("session_id") or tags.get("StudyInstanceUID") or tags.get("AccessionNumber"),
            study_ids=identity.get("study_id"),
            acquired_ats=identity.get("acquired_at") or _dicom_date(tags.get("StudyDate")),
            external_ids=identity.get("external_id") or tags.get("SeriesInstanceUID"),
            metas=[{"dicom_modality": tags.get("Modality")}])
        item = ids[0]
        # Optional pixel-level radiomics as a second view on the same item.
        px = self._pixels(source, pixels)
        if px is not None:
            from . import axes as _axes
            from . import radiomics as _rad
            arr = np.asarray(px, dtype="float32")
            _ingest.add_space_vectors(mf, [item], _rad.radiomics_features(arr)[None, :],
                                      space="dicom_pixels", embedder_id="radiomics-v1")
            cx, cy, cz = _rad.centroid(arr)
            _axes.set_axes(mf, item, {"x": cx, "y": cy, "z": cz})
        return item

    def _pixels(self, source, pixels):
        if pixels is not None:
            return pixels
        if isinstance(source, dict):
            return source.get("pixel_array")
        try:
            if isinstance(source, (str, Path)):
                import pydicom
                return pydicom.dcmread(str(source), force=True).pixel_array
            return source.pixel_array          # a pydicom Dataset
        except Exception:
            return None


# -- WFDB (PhysioNet signals — just another waveform source) -----------------
WFDB_BANDS = {"low": (0.5, 5.0), "mid": (5.0, 15.0), "high": (15.0, 40.0)}


class WFDBProcessor:
    modality = "wfdb"
    spaces = ["wfdb_spectral"]

    def __init__(self, crop_seconds: float | None = None, bands=None):
        from .embedders.eeg import EEGEmbedder
        self.embedder = EEGEmbedder(bands=bands or WFDB_BANDS, crop_seconds=crop_seconds,
                                    id="wfdb-spectral-v1")

    def ingest(self, mf, source, *, sfreq: float | None = None, ch_names=None, **identity) -> int:
        from . import signal
        data, sfreq, ch_names = self._read(source, sfreq, ch_names)
        item_id, _ = signal.add_recording(mf, data, sfreq=sfreq, ch_names=ch_names,
                                          modality="wfdb", embedder=self.embedder,
                                          feature_space="wfdb_spectral", **identity)
        return item_id

    def _read(self, source, sfreq, ch_names):
        if isinstance(source, (str, Path)):
            try:
                import wfdb
            except ImportError as e:
                raise RuntimeError("reading WFDB records needs `wfdb`; or pass "
                                   "(array, sfreq=, ch_names=)") from e
            rec = wfdb.rdrecord(str(source).rsplit(".", 1)[0])
            return rec.p_signal.T.astype("float32"), float(rec.fs), list(rec.sig_name)
        data = np.asarray(source, dtype="float32")
        if data.ndim == 1:
            data = data[None, :]
        if sfreq is None:
            raise ValueError("WFDB array ingest needs sfreq")
        return data, sfreq, ch_names


for _p in (EEGProcessor(), ECGProcessor(), TextProcessor(), MRIProcessor(),
           GenomicsProcessor(), FHIRProcessor(), DICOMProcessor(), WFDBProcessor(),
           TabularProcessor(modality="labs", space="labs"),
           TabularProcessor(modality="tabular", space="tabular")):
    register(_p)
