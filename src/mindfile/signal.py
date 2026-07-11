"""Lossless waveform store for channels x time signals (EEG/MEG), with
collapse-on-call features.

A waveform compresses well because it's a regular grid: samples are smooth in
time and channels are correlated. The codec exploits both, losslessly:

  * **byte-shuffle** (group the i-th byte of every sample together) so zlib sees
    long runs of similar bytes;
  * **integer delta along time** for integer dtypes (EDF is int16) — consecutive
    samples differ little, so deltas compress hard; modular arithmetic keeps it
    exact.

Reconstruction is byte-identical (verified by ``raw_sha256``). The waveform
becomes the compact source of truth; a band-power vector is then a lazy
``signal_feature`` recipe that recomputes from it on call.
"""

from __future__ import annotations

import json
import struct
import zlib

import numpy as np

from . import _vec
from . import lazy as _lazy
from .file import MindFile, utcnow


# -- codec (exact, lossless) -----------------------------------------------
def _shuffle(raw: bytes, itemsize: int) -> bytes:
    if itemsize == 1:
        return raw
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, itemsize).T.tobytes()


def _unshuffle(shuf: bytes, itemsize: int) -> bytes:
    if itemsize == 1:
        return shuf
    return np.frombuffer(shuf, dtype=np.uint8).reshape(itemsize, -1).T.tobytes()


def encode(arr: np.ndarray) -> tuple[bytes, dict]:
    """Compress a (n_channels, n_samples) array losslessly. Returns (code, meta)."""
    arr = np.ascontiguousarray(arr)
    if arr.ndim != 2:
        raise ValueError("signal must be 2-D (n_channels, n_samples)")
    dt = arr.dtype
    delta = bool(np.issubdtype(dt, np.integer))
    work = arr
    if delta:  # modular delta along time (exact for integers)
        work = arr.copy()
        work[:, 1:] = (arr[:, 1:].astype(np.int64) - arr[:, :-1].astype(np.int64)).astype(dt)
    code = zlib.compress(_shuffle(np.ascontiguousarray(work).tobytes(), dt.itemsize), 9)
    return code, {"dtype": dt.str, "n_channels": arr.shape[0], "n_samples": arr.shape[1],
                  "delta": delta}


def decode(code: bytes, meta: dict) -> np.ndarray:
    dt = np.dtype(meta["dtype"])
    c, t = meta["n_channels"], meta["n_samples"]
    raw = _unshuffle(zlib.decompress(code), dt.itemsize)
    work = np.frombuffer(raw, dtype=dt).reshape(c, t)
    if meta["delta"]:  # invert modular delta via cumulative sum
        return np.cumsum(work.astype(np.int64), axis=1).astype(dt)
    return np.array(work)


# -- digital codec: int16 samples + per-channel SI calibration --------------
# The calibration is embedded INSIDE `code`, so `code_sha256` (frozen by every
# checkpoint and re-checked by verify_signals) already covers it — no extra
# column or hash is needed. `raw_sha256` stays the hash of the exact int16
# samples; physical (SI) values reconstruct on call as digital*gain + offset.
def encode_digital(digital: np.ndarray, cal: dict) -> tuple[bytes, dict]:
    if not np.issubdtype(np.asarray(digital).dtype, np.integer):
        raise ValueError("digital calibration requires integer samples")
    core, meta = encode(digital)
    cal_json = json.dumps(cal, separators=(",", ":"), sort_keys=True).encode()
    return struct.pack("<I", len(cal_json)) + cal_json + core, meta


def _split_digital(code: bytes) -> tuple[dict, bytes]:
    (n,) = struct.unpack("<I", code[:4])
    return json.loads(code[4:4 + n]), code[4 + n:]


# -- optional LOSSY spectral codec (opt-in; not byte-exact) -----------------
def encode_spectral(data: np.ndarray, keep_fraction: float = 0.1):
    """Keep the top per-channel rFFT coefficients. Returns (code, meta, recon).

    Lossy: the waveform is approximated by its strongest spectral components.
    ``meta['error']`` is the relative reconstruction error.
    """
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    n_ch, n_samp = data.shape
    spec = np.fft.rfft(data, axis=1)
    n_freq = spec.shape[1]
    k = max(1, int(keep_fraction * n_freq))
    idx = np.argsort(-np.abs(spec), axis=1)[:, :k]
    # Round coeffs to the stored precision (complex64) BEFORE reconstructing, so
    # encode's reconstruction is bit-identical to decode's (hash-verifiable).
    coeff = np.take_along_axis(spec, idx, axis=1).astype("<c8")
    recon_spec = np.zeros_like(spec)
    np.put_along_axis(recon_spec, idx, coeff.astype(np.complex128), axis=1)
    recon = np.fft.irfft(recon_spec, n=n_samp, axis=1).astype(np.float32)
    err = float(np.linalg.norm(recon.astype(np.float64) - data) / (np.linalg.norm(data) or 1.0))
    header = np.array([n_ch, n_samp, k, n_freq], dtype=np.int64).tobytes()
    body = idx.astype("<i4").tobytes() + coeff.tobytes()
    return zlib.compress(header + body, 9), {"n_channels": n_ch, "n_samples": n_samp,
                                             "error": err}, recon


def decode_spectral(code: bytes) -> np.ndarray:
    raw = zlib.decompress(code)
    n_ch, n_samp, k, n_freq = (int(x) for x in np.frombuffer(raw[:32], dtype=np.int64))
    off = 32
    idx = np.frombuffer(raw[off:off + n_ch * k * 4], dtype="<i4").reshape(n_ch, k)
    off += n_ch * k * 4
    coeff = np.frombuffer(raw[off:off + n_ch * k * 8], dtype="<c8").reshape(n_ch, k)
    spec = np.zeros((n_ch, n_freq), dtype=np.complex128)
    np.put_along_axis(spec, idx.astype(np.int64), coeff.astype(np.complex128), axis=1)
    return np.fft.irfft(spec, n=n_samp, axis=1).astype(np.float32)


# -- store / load ----------------------------------------------------------
def store_signal(mf: MindFile, item_id: int, data: np.ndarray, *, sfreq: float,
                 ch_names=None, dtype=None, codec: str = "lossless",
                 keep_fraction: float = 0.1, calibration: dict | None = None) -> dict:
    """Compress and store a waveform for ``item_id``.

    ``codec='lossless'`` (default) reconstructs byte-exact. If ``calibration``
    (``{gain, offset, unit}``) is given, ``data`` must be the integer *digital*
    samples: they're stored byte-exact and physical (SI) values reconstruct on
    call — smaller and more faithful than storing pre-scaled floats.
    ``codec='spectral'`` is an opt-in LOSSY representation (smaller, approximate).
    """
    data = np.asarray(data)
    if calibration is not None:                        # int16 digital + calibration
        data = np.ascontiguousarray(data)
        code, meta = encode_digital(data, calibration)
        codec_name = "digital-cal"
        stored, lossy, error = data, 0, None
        dtype_str, delta, n_ch, n_samp = meta["dtype"], int(meta["delta"]), \
            meta["n_channels"], meta["n_samples"]
    elif codec == "lossless":
        if dtype is not None:
            data = data.astype(dtype)
        data = np.ascontiguousarray(data)
        code, meta = encode(data)
        codec_name = "delta-shuffle-zlib" if meta["delta"] else "shuffle-zlib"
        stored, lossy, error = data, 0, None
        dtype_str, delta, n_ch, n_samp = meta["dtype"], int(meta["delta"]), \
            meta["n_channels"], meta["n_samples"]
    elif codec == "spectral":
        code, meta, stored = encode_spectral(data, keep_fraction)
        codec_name, lossy, error = "spectral-zlib", 1, meta["error"]
        dtype_str, delta, n_ch, n_samp = "<f4", 0, meta["n_channels"], meta["n_samples"]
    else:
        raise ValueError(f"unknown codec {codec!r} (use 'lossless' or 'spectral')")

    raw_bytes = stored.tobytes()
    mf.conn.execute(
        "INSERT OR REPLACE INTO signals(item_id, codec, dtype, n_channels, n_samples, "
        "sfreq, ch_names, delta, code, raw_sha256, code_sha256, lossy, error, created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (item_id, codec_name, dtype_str, n_ch, n_samp, sfreq,
         json.dumps(list(ch_names) if ch_names is not None else None), delta,
         code, _vec.sha256_bytes(raw_bytes), _vec.sha256_bytes(code), lossy, error, utcnow()))
    mf.record("signal", {"item_id": item_id, "shape": [n_ch, n_samp], "lossy": bool(lossy),
                         "raw_bytes": len(raw_bytes), "code_bytes": len(code)})
    mf.conn.commit()
    return {"raw_bytes": len(raw_bytes), "code_bytes": len(code),
            "ratio": len(raw_bytes) / max(len(code), 1), "lossy": bool(lossy), "error": error}


def _decode_stored(r) -> tuple[np.ndarray, dict | None]:
    """Return the byte-exact stored array (the hashed artifact) and any
    calibration. For the digital codec the array is the int16 samples."""
    if r["codec"] == "spectral-zlib":
        return decode_spectral(r["code"]), None
    meta = {"dtype": r["dtype"], "n_channels": r["n_channels"],
            "n_samples": r["n_samples"], "delta": bool(r["delta"])}
    if r["codec"] == "digital-cal":
        cal, core = _split_digital(r["code"])
        return decode(core, meta), cal
    return decode(r["code"], meta), None


def load_signal(mf: MindFile, item_id: int, *, verify: bool = False):
    """Reconstruct the waveform. Returns (data, sfreq, ch_names); for a digital
    recording ``data`` is the physical (SI) signal reconstructed from calibration."""
    r = mf.conn.execute(
        "SELECT codec, dtype, n_channels, n_samples, sfreq, ch_names, delta, code, raw_sha256 "
        "FROM signals WHERE item_id=?", (item_id,)).fetchone()
    if r is None:
        raise KeyError(f"no signal stored for item {item_id}")
    stored, cal = _decode_stored(r)
    if verify and _vec.sha256_bytes(stored.tobytes()) != r["raw_sha256"]:
        raise ValueError(f"signal for item {item_id} failed integrity check")
    if cal is not None:
        gain = np.asarray(cal["gain"], dtype=np.float64)[:, None]
        offset = np.asarray(cal["offset"], dtype=np.float64)[:, None]
        data = stored.astype(np.float64) * gain + offset
    else:
        data = stored
    ch = json.loads(r["ch_names"]) if r["ch_names"] else None
    return data, r["sfreq"], ch


def has_signal(mf: MindFile, item_id: int) -> bool:
    return mf.conn.execute("SELECT 1 FROM signals WHERE item_id=?", (item_id,)).fetchone() is not None


def verify_signals(mf: MindFile) -> dict:
    """Recompute each waveform's hashes and report any drift/corruption."""
    bad = []
    for r in mf.conn.execute(
            "SELECT item_id, codec, dtype, n_channels, n_samples, delta, code, "
            "raw_sha256, code_sha256 FROM signals"):
        if _vec.sha256_bytes(r["code"]) != r["code_sha256"]:
            bad.append({"item_id": r["item_id"], "issue": "code"})
            continue
        stored, _ = _decode_stored(r)              # the byte-exact stored artifact
        if _vec.sha256_bytes(stored.tobytes()) != r["raw_sha256"]:
            bad.append({"item_id": r["item_id"], "issue": "reconstruction"})
    n = mf.conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    return {"status": "ok" if not bad else "corrupted", "checked": n, "bad": bad}


# -- lazy band-power features from the stored waveform ---------------------
def define_feature_vector(mf: MindFile, item_id: int, space: str, embedder) -> int:
    """Register a lazy feature vector recomputed from the item's stored waveform.

    ``embedder`` is an EEGEmbedder-like object exposing ``id``, ``dim``, ``bands``,
    ``crop_seconds`` and ``_features_from_array(data, sfreq)``.
    """
    recipe = {"method": embedder.id,
              "params": {"crop_seconds": embedder.crop_seconds, "bands": embedder.bands}}
    return _lazy.define(mf, item_id, space, "signal_feature", recipe,
                        dim=embedder.dim, embedder_id=embedder.id, derived=True)


def add_recording(mf: MindFile, data: np.ndarray, *, sfreq: float, ch_names=None,
                  subject_id=None, session_id=None, study_id=None, modality: str = "eeg",
                  external_id=None, acquired_at=None, dtype="float32",
                  codec: str = "lossless", keep_fraction: float = 0.1,
                  calibration: dict | None = None,
                  embedder=None, feature_space: str = "bandpower") -> tuple[int, dict]:
    """Waveform-first ingest: create an item, store its waveform losslessly, and
    (optionally) register a lazy band-power feature recomputed from it on call.

    If ``calibration`` is given, ``data`` is the integer digital signal and is
    stored byte-exact (physical values reconstruct on call). Returns
    ``(item_id, store_info)``.
    """
    cur = mf.conn.execute(
        "INSERT INTO items(external_id, origin, subject_id, session_id, study_id, "
        "modality, acquired_at, payload_kind, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (external_id, "primary", subject_id, session_id, study_id, modality,
         acquired_at, "signal", utcnow()))
    item_id = cur.lastrowid
    info = store_signal(mf, item_id, data, sfreq=sfreq, ch_names=ch_names,
                        dtype=None if calibration is not None else dtype,
                        codec=codec, keep_fraction=keep_fraction, calibration=calibration)
    if embedder is not None:
        define_feature_vector(mf, item_id, feature_space, embedder)
    mf.conn.commit()
    return item_id, info


def define_connectivity_vector(mf: MindFile, item_id: int, space: str, embedder) -> int:
    """Register a lazy inter-channel connectivity vector from the item's waveform.

    ``embedder`` is a ConnectivityEmbedder-like object (``id``, ``dim``, ``bands``,
    ``crop_seconds``, ``_features_from_array``)."""
    recipe = {"method": embedder.id,
              "params": {"crop_seconds": embedder.crop_seconds, "bands": embedder.bands}}
    return _lazy.define(mf, item_id, space, "signal_connectivity", recipe,
                        dim=embedder.dim, embedder_id=embedder.id, derived=True)


def _crop(data, sfreq, crop_seconds):
    if crop_seconds:
        return data[:, :int(min(crop_seconds * sfreq, data.shape[1]))]
    return data


def _resolve_signal_feature(mf, item_id, space_id, recipe):
    from .embedders.eeg import EEGEmbedder
    data, sfreq, _ = load_signal(mf, item_id)
    p = recipe.get("params", {})
    data = _crop(data, sfreq, p.get("crop_seconds"))
    emb = EEGEmbedder(bands=p.get("bands"), crop_seconds=p.get("crop_seconds"),
                      id=recipe["method"])
    return emb._features_from_array(data.astype(np.float64), sfreq)


def _resolve_signal_connectivity(mf, item_id, space_id, recipe):
    from .embedders.eeg import ConnectivityEmbedder
    data, sfreq, _ = load_signal(mf, item_id)
    p = recipe.get("params", {})
    data = _crop(data, sfreq, p.get("crop_seconds"))
    emb = ConnectivityEmbedder(bands=p.get("bands"), crop_seconds=p.get("crop_seconds"),
                               id=recipe["method"])
    return emb._features_from_array(data.astype(np.float64), sfreq)


def define_asymmetry_vector(mf: MindFile, item_id: int, space: str, embedder) -> int:
    """Register a lazy directed hemispheric-asymmetry vector from the waveform.

    ``embedder`` is an AsymmetryEmbedder-like object; resolution passes channel
    names (needed to split left/right)."""
    recipe = {"method": embedder.id,
              "params": {"crop_seconds": embedder.crop_seconds, "bands": embedder.bands}}
    return _lazy.define(mf, item_id, space, "signal_asymmetry", recipe,
                        dim=embedder.dim, embedder_id=embedder.id, derived=True)


def _resolve_signal_asymmetry(mf, item_id, space_id, recipe):
    from .embedders.eeg import AsymmetryEmbedder
    data, sfreq, ch = load_signal(mf, item_id)
    p = recipe.get("params", {})
    data = _crop(data, sfreq, p.get("crop_seconds"))
    emb = AsymmetryEmbedder(bands=p.get("bands"), crop_seconds=p.get("crop_seconds"),
                            id=recipe["method"])
    return emb._features_from_array(data.astype(np.float64), sfreq, ch)


_lazy.register_resolver("signal_feature", _resolve_signal_feature)
_lazy.register_resolver("signal_connectivity", _resolve_signal_connectivity)
_lazy.register_resolver("signal_asymmetry", _resolve_signal_asymmetry)
