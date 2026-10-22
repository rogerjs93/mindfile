"""Lossless container for a FreeSurfer ``recon-all`` subject tree.

A recon-all subject is ~0.5-1 GB spread over hundreds of files, and most of that
bulk is *structured and mutually redundant*: volumes are gzipped MGH grids that
re-compress poorly with generic tools, the surface files of one hemisphere all
share the same topology and nearly the same coordinates, and the label/stats
files are text. This module folds the whole tree into one ``.gnos`` file and
compresses each part with a codec that understands its structure:

* **gzip containers** (``.mgz``, ``.nii.gz``, ``.m3z``) are *unwrapped* — the
  deflate stream is replaced by a much stronger code, and the parameters needed
  to re-emit the original gzip stream **byte-for-byte** are stored alongside;
* **MGH volumes** are predicted from their already-decoded neighbours (raster
  ``x`` delta, or a median-edge predictor over the y/z neighbours) and the
  residual is LZMA-coded — all integer/modular, so exact for any dtype;
* **volumes that are near-copies of another volume in the same tree**
  (``brain`` vs ``brain.finalsurfs``, ``T1`` vs ``nu``) are stored as a delta
  against it, chosen by a cheap strided-subsample similarity probe;
* **surfaces** are split into coordinates/faces and delta-coded against another
  surface with the same topology (``?h.white`` vs ``?h.pial`` vs ``?h.orig``);
* everything else falls back to LZMA, then to raw bytes.

Integrity comes first: *no codec is trusted*. Every candidate encoding is
decoded again at store time and its SHA-256 compared with the original file's,
and the first candidate that reproduces the file exactly wins. A parser bug can
therefore only cost compression ratio, never fidelity. ``restore_subject``
rebuilds a byte-identical tree; ``verify_files`` re-checks every stored file.

Two profiles:

* ``lossless`` (default) — every file in the tree, restorable bit-for-bit;
* ``analysis``  — an explicit keep-list of the non-regenerable, analysis-bearing
  outputs. Omitted files are *still listed in the manifest with their SHA-256*,
  so a bundle records exactly what was dropped and can verify an external copy.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import lzma
import struct
import zlib
from pathlib import Path

import numpy as np

from . import _vec
from .file import MindFile, utcnow

# -- MGH / FreeSurfer format constants --------------------------------------
MGH_HEADER_BYTES = 284
_MGH_DTYPES = {0: ">u1", 1: ">i4", 3: ">f4", 4: ">i2"}
_SURFACE_MAGIC = 0xFFFFFE
_CURV_MAGIC = 0xFFFFFF

_PRESET = 6


# ---------------------------------------------------------------------------
# low-level LZMA helpers
# ---------------------------------------------------------------------------
def _filters(delta: int | None, preset: int, dict_size: int | None) -> list[dict]:
    filters = []
    if delta and delta > 1:
        filters.append({"id": lzma.FILTER_DELTA, "dist": delta})
    spec = {"id": lzma.FILTER_LZMA2, "preset": preset}
    if dict_size:
        spec["dict_size"] = dict_size
    filters.append(spec)
    return filters


def _xz(raw: bytes, *, delta: int | None = None, preset: int = _PRESET,
        dict_size: int | None = None) -> bytes:
    return lzma.compress(raw, format=lzma.FORMAT_RAW,
                         filters=_filters(delta, preset, dict_size))


def _unxz(code: bytes, *, delta: int | None = None, preset: int = _PRESET,
          dict_size: int | None = None) -> bytes:
    # A raw LZMA2 stream carries no header, so the decoder must be handed the
    # SAME dict_size the encoder used — hence blocks record theirs.
    return lzma.decompress(code, format=lzma.FORMAT_RAW,
                           filters=_filters(delta, preset, dict_size))


def _shuffle(raw: bytes, itemsize: int) -> bytes:
    if itemsize <= 1:
        return raw
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, itemsize).T.tobytes()


def _unshuffle(raw: bytes, itemsize: int) -> bytes:
    if itemsize <= 1:
        return raw
    return np.frombuffer(raw, dtype=np.uint8).reshape(itemsize, -1).T.tobytes()


# ---------------------------------------------------------------------------
# gzip container: unwrap, and prove we can re-emit the identical stream
# ---------------------------------------------------------------------------
def _gzip_split(data: bytes) -> tuple[bytes, bytes, bytes] | None:
    """Split a single-member gzip file into (header, deflate body, trailer)."""
    if len(data) < 18 or data[:2] != b"\x1f\x8b" or data[2] != 8:
        return None
    flg = data[3]
    off = 10
    try:
        if flg & 4:                                   # FEXTRA
            (xlen,) = struct.unpack("<H", data[off:off + 2])
            off += 2 + xlen
        for bit in (8, 16):                           # FNAME, FCOMMENT
            if flg & bit:
                off = data.index(b"\x00", off) + 1
        if flg & 2:                                   # FHCRC
            off += 2
    except (ValueError, struct.error):
        return None
    if off >= len(data) - 8:
        return None
    return data[:off], data[off:-8], data[-8:]


#: Bytes of payload used to reject a candidate zlib setting before committing to
#: a full recompression. Deflate is a streaming codec, so whatever a setting
#: emits for a prefix of the input is a prefix of its final stream — if that
#: doesn't match, no amount of further input will make it match.
_GZIP_PROBE_BYTES = 1 << 20


def _gzip_params(body: bytes, payload: bytes) -> dict | None:
    """Find zlib settings that reproduce this exact deflate stream, or None."""
    probe = payload[:_GZIP_PROBE_BYTES]
    for level in (6, 9, 1, 5, 4, 3, 2, 7, 8):
        for memlevel in (8, 9):
            for strategy in (zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED):
                if len(payload) > _GZIP_PROBE_BYTES:
                    co = zlib.compressobj(level, zlib.DEFLATED, -15, memlevel, strategy)
                    head = co.compress(probe)
                    if head and not body.startswith(head):
                        continue
                co = zlib.compressobj(level, zlib.DEFLATED, -15, memlevel, strategy)
                if co.compress(payload) + co.flush() == body:
                    return {"level": level, "memlevel": memlevel, "strategy": strategy}
    return None


def _gzip_rebuild(header: bytes, trailer: bytes, payload: bytes, params: dict) -> bytes:
    co = zlib.compressobj(params["level"], zlib.DEFLATED, -15,
                          params["memlevel"], params["strategy"])
    return header + co.compress(payload) + co.flush() + trailer


# ---------------------------------------------------------------------------
# MGH volumes
# ---------------------------------------------------------------------------
def _mgh_split(payload: bytes):
    """Split an uncompressed MGH payload into (header, array, tail), or None."""
    if len(payload) < MGH_HEADER_BYTES:
        return None
    version, width, height, depth, nframes, typ = struct.unpack(">6i", payload[:24])
    if version != 1 or typ not in _MGH_DTYPES:
        return None
    if min(width, height, depth, nframes) <= 0:
        return None
    dt = np.dtype(_MGH_DTYPES[typ])
    n = width * height * depth * nframes
    end = MGH_HEADER_BYTES + n * dt.itemsize
    if end > len(payload):
        return None
    arr = np.frombuffer(payload[MGH_HEADER_BYTES:end], dtype=dt).reshape(
        nframes, depth, height, width)
    return payload[:MGH_HEADER_BYTES], arr, payload[end:]


def _mgh_join(header: bytes, arr: np.ndarray, tail: bytes) -> bytes:
    return header + np.ascontiguousarray(arr).tobytes() + tail


def _uint_dtype(dt: np.dtype) -> np.dtype:
    """The unsigned integer dtype with the same width *and byte order* as ``dt``.

    Byte order matters: MGH data is big-endian, and reattaching the wrong order
    after decoding silently byte-swaps every value, so this is derived from the
    dtype rather than assumed. Single-byte dtypes have no byte order ('|').
    """
    it = dt.itemsize
    if it == 1:
        return np.dtype("u1")
    return np.dtype(f"{'>' if dt.byteorder == '>' else '<'}u{it}")


def _as_uint_view(arr: np.ndarray) -> tuple[np.ndarray, int, int]:
    """View any dtype as unsigned integers of the same width (exact, no copy)."""
    it = arr.dtype.itemsize
    return arr.view(_uint_dtype(arr.dtype)), it, 1 << (8 * it)


def _predict(view64: np.ndarray, mode: str) -> np.ndarray:
    """Prediction from already-decoded neighbours (no x dependency for 'yzmed')."""
    if mode == "x":
        p = np.zeros_like(view64)
        p[..., 1:] = view64[..., :-1]
        return p
    if mode == "y":
        p = np.zeros_like(view64)
        p[:, :, 1:, :] = view64[:, :, :-1, :]
        return p
    if mode == "z":
        p = np.zeros_like(view64)
        p[:, 1:, :, :] = view64[:, :-1, :, :]
        return p
    if mode == "yzmed":
        up = np.zeros_like(view64); up[:, :, 1:, :] = view64[:, :, :-1, :]
        pz = np.zeros_like(view64); pz[:, 1:, :, :] = view64[:, :-1, :, :]
        dg = np.zeros_like(view64); dg[:, 1:, 1:, :] = view64[:, :-1, :-1, :]
        return np.where(dg >= np.maximum(up, pz), np.minimum(up, pz),
                        np.where(dg <= np.minimum(up, pz), np.maximum(up, pz), up + pz - dg))
    raise ValueError(f"unknown predictor {mode!r}")


def _encode_residual(arr: np.ndarray, mode: str, shuf: bool,
                     preset: int = _PRESET) -> bytes:
    uview, it, mod = _as_uint_view(arr)
    v = uview.astype(np.int64)
    resid = ((v - _predict(v, mode)) % mod).astype(f"<u{it}")
    raw = np.ascontiguousarray(resid).tobytes()
    return _xz(_shuffle(raw, it) if shuf else raw, preset=preset)


def _decode_residual(code: bytes, shape, dtype: str, mode: str, shuf: bool) -> np.ndarray:
    dt = np.dtype(dtype)
    it = dt.itemsize
    mod = 1 << (8 * it)
    raw = _unxz(code)
    if shuf:
        raw = _unshuffle(raw, it)
    resid = np.frombuffer(raw, dtype=f"<u{it}").reshape(shape).astype(np.int64)
    if mode == "x":                       # running sum along the fastest axis
        out = np.cumsum(resid, axis=3) % mod
    elif mode == "y":
        out = np.cumsum(resid, axis=2) % mod
    elif mode == "z":
        out = np.cumsum(resid, axis=1) % mod
    elif mode == "yzmed":
        # No x dependency: each (frame, slice, row) decodes as one vector op
        # against rows already reconstructed above it / behind it.
        out = np.zeros(shape, dtype=np.int64)
        nf, nd, nh, _ = shape
        zeros = np.zeros(shape[3], dtype=np.int64)
        for f in range(nf):
            for z in range(nd):
                for y in range(nh):
                    up = out[f, z, y - 1] if y else zeros
                    pz = out[f, z - 1, y] if z else zeros
                    dg = out[f, z - 1, y - 1] if (z and y) else zeros
                    med = np.where(dg >= np.maximum(up, pz), np.minimum(up, pz),
                                   np.where(dg <= np.minimum(up, pz), np.maximum(up, pz),
                                            up + pz - dg))
                    out[f, z, y] = (resid[f, z, y] + med) % mod
    else:
        raise ValueError(f"unknown predictor {mode!r}")
    return out.astype(_uint_dtype(dt)).view(dt)


_SIG_SAMPLES = 8192
_SIG_STRIDE = 2654435761        # Knuth's multiplier: coprime with any power of two


def _volume_signature(arr: np.ndarray) -> np.ndarray:
    """Scattered sample of a volume, used to pick a delta reference.

    The positions are deterministic but *not* a plain stride: a stride that
    divides the slice width lands on the same (row, column) of every slice, which
    for a conformed brain volume means sampling only the empty background — every
    pair of volumes then looks identical and the reference choice is meaningless.
    """
    flat = np.ascontiguousarray(arr).reshape(-1)
    n = min(_SIG_SAMPLES, flat.size)
    idx = (np.arange(n, dtype=np.int64) * _SIG_STRIDE) % flat.size
    return flat[idx].astype(np.float64)


# ---------------------------------------------------------------------------
# FreeSurfer surface / curv files
# ---------------------------------------------------------------------------
def _surface_split(data: bytes):
    """Split a FreeSurfer triangular surface into head/vertices/faces/tail."""
    if len(data) < 20 or int.from_bytes(data[:3], "big") != _SURFACE_MAGIC:
        return None
    try:
        i = data.index(b"\n", data.index(b"\n") + 1) + 1   # two comment lines
        nv, nf = struct.unpack(">2i", data[i:i + 8])
    except (ValueError, struct.error):
        return None
    if nv <= 0 or nf < 0:
        return None
    off = i + 8
    end_v = off + nv * 12
    end_f = end_v + nf * 12
    if end_f > len(data):
        return None
    verts = np.frombuffer(data[off:end_v], dtype=">u4").reshape(nv, 3)
    faces = np.frombuffer(data[end_v:end_f], dtype=">u4").reshape(nf, 3)
    return data[:off], verts, faces, data[end_f:]


def _curv_split(data: bytes):
    """Split a per-vertex curv/thickness file into head/values."""
    if len(data) < 15 or int.from_bytes(data[:3], "big") != _CURV_MAGIC:
        return None
    nv, _nf, vals = struct.unpack(">3i", data[3:15])
    if nv <= 0 or vals <= 0:
        return None
    end = 15 + nv * vals * 4
    if end != len(data):
        return None
    return data[:15], np.frombuffer(data[15:end], dtype=">u4")


# ---------------------------------------------------------------------------
# candidate encodings (each is verified before it can win)
# ---------------------------------------------------------------------------
class _Context:
    """Per-subject encoding state: what is already stored and reusable as a ref."""

    #: Above this size, candidate codecs are ranked on a strided subsample
    #: instead of by compressing the whole volume once per candidate.
    PROBE_BYTES = 4_000_000

    #: Ceiling on reference data held in memory. A subject can carry hundreds of
    #: megabytes of volumes; keeping them all resident to serve as delta
    #: references would cost more RAM than the tree itself. Oldest are dropped
    #: first — only the ratio suffers, never correctness.
    REF_BUDGET_BYTES = 512_000_000

    def __init__(self, effort: str = "balanced", ref_budget: int | None = None):
        self.effort = effort
        self.ref_budget = self.REF_BUDGET_BYTES if ref_budget is None else ref_budget
        self.volumes: list[dict] = []    # {file_id, sig, shape, dtype, arr}
        self.surfaces: list[dict] = []   # {file_id, nv, nf, verts, faces}
        self._by_id: dict[int, object] = {}
        self._held = 0

    def _evict(self) -> None:
        while self._held > self.ref_budget and (self.volumes or self.surfaces):
            pool = self.volumes if self.volumes else self.surfaces
            old = pool.pop(0)
            self._held -= old["bytes"]
            self._by_id.pop(old["file_id"], None)

    def remember_volume(self, file_id: int, arr: np.ndarray) -> None:
        self.volumes.append({"file_id": file_id, "sig": _volume_signature(arr),
                             "shape": arr.shape, "dtype": arr.dtype.str, "arr": arr,
                             "bytes": arr.nbytes})
        self._by_id[file_id] = arr
        self._held += arr.nbytes
        self._evict()

    def remember_surface(self, file_id: int, verts, faces) -> None:
        n = verts.nbytes + faces.nbytes
        self.surfaces.append({"file_id": file_id, "nv": int(verts.shape[0]),
                              "nf": int(faces.shape[0]), "verts": verts, "faces": faces,
                              "bytes": n})
        self._by_id[file_id] = (verts, faces)
        self._held += n
        self._evict()

    def resolve(self, file_id):
        """Reference resolver used to *verify* a delta encoding at store time."""
        if file_id not in self._by_id:
            raise KeyError(f"reference {file_id} not in encoding context")
        return self._by_id[file_id]

    def volume_refs(self, arr: np.ndarray, k: int = 2) -> list[tuple[dict, float]]:
        """The ``k`` stored volumes most similar to ``arr`` (same shape+dtype)."""
        sig = _volume_signature(arr)
        scored = [(cand, float((cand["sig"] == sig).mean())) for cand in self.volumes
                  if cand["shape"] == arr.shape and cand["dtype"] == arr.dtype.str]
        scored.sort(key=lambda cs: -cs[1])
        return scored[:k]

    def surface_ref(self, nv: int, nf: int):
        for cand in self.surfaces:
            if cand["nv"] == nv and cand["nf"] == nf:
                return cand
        return None


def _array_specs(arr: np.ndarray, ctx: "_Context") -> list[dict]:
    """Candidate codecs for a voxel array, chosen by dtype.

    Measured on a real subject: neighbour prediction pays for integer volumes
    (intensity/label grids are piecewise smooth) but *costs* on float32 fields
    (warp maps, parameter maps), where plain LZMA wins — so floats are not
    predicted unless explicitly asked for.
    """
    is_float = arr.dtype.kind == "f"
    if is_float:
        specs = [{"kind": "plain", "shuf": False}, {"kind": "plain", "shuf": True}]
    else:
        specs = [{"kind": "pred", "mode": "x", "shuf": False},
                 {"kind": "pred", "mode": "yzmed", "shuf": False},
                 {"kind": "plain", "shuf": True}]
    if ctx.effort == "fast":
        specs = specs[:1]
    elif ctx.effort == "max":
        specs = [{"kind": "plain", "shuf": False}, {"kind": "plain", "shuf": True},
                 {"kind": "pred", "mode": "x", "shuf": False},
                 {"kind": "pred", "mode": "y", "shuf": False},
                 {"kind": "pred", "mode": "z", "shuf": False},
                 {"kind": "pred", "mode": "yzmed", "shuf": False},
                 {"kind": "pred", "mode": "yzmed", "shuf": True}]
    for ref, score in ctx.volume_refs(arr):
        if score > 0.25:                          # only worth it when they overlap
            specs.append({"kind": "delta", "ref": ref["file_id"], "_arr": ref["arr"]})
    return specs


def _encode_spec(arr: np.ndarray, spec: dict, *, preset: int = _PRESET,
                 ref: np.ndarray | None = None) -> bytes:
    it = arr.dtype.itemsize
    if spec["kind"] == "pred":
        return _encode_residual(arr, spec["mode"], bool(spec.get("shuf")), preset)
    if spec["kind"] == "delta":
        uview, _, mod = _as_uint_view(arr)
        rview, _, _ = _as_uint_view(ref)
        raw = np.ascontiguousarray(
            ((uview.astype(np.int64) - rview.astype(np.int64)) % mod).astype(
                f"<u{it}")).tobytes()
    else:                                          # plain
        raw = np.ascontiguousarray(arr).tobytes()
    return _xz(_shuffle(raw, it) if spec.get("shuf") else raw, preset=preset)


def _spec_meta(arr: np.ndarray, spec: dict) -> tuple[str, dict]:
    base = {"shape": list(arr.shape), "dtype": arr.dtype.str, "shuf": bool(spec.get("shuf"))}
    if spec["kind"] == "pred":
        return f"pred:{spec['mode']}{':shuf' if spec.get('shuf') else ''}", \
            {**base, "pred": spec["mode"]}
    if spec["kind"] == "delta":
        return "delta", {**base, "ref": spec["ref"]}
    return f"plain{':shuf' if spec.get('shuf') else ''}", base


def _subsample(arr: np.ndarray) -> np.ndarray:
    """Every 4th slice — enough to rank codecs at a quarter of the cost."""
    return np.ascontiguousarray(arr[:, ::4]) if arr.shape[1] >= 8 else arr


def _array_options(arr: np.ndarray, ctx: "_Context"):
    """Yield (codec_suffix, meta, code) for the array, ranked cheaply if large."""
    specs = _array_specs(arr, ctx)
    if len(specs) > 1 and arr.nbytes > ctx.PROBE_BYTES and ctx.effort == "balanced":
        sub = _subsample(arr)
        scored = []
        for spec in specs:
            ref = spec.get("_arr")
            scored.append((len(_encode_spec(sub, spec, preset=1,
                                            ref=_subsample(ref) if ref is not None else None)),
                           spec))
        specs = [min(scored, key=lambda s: s[0])[1]]
    for spec in specs:
        name, meta = _spec_meta(arr, spec)
        yield name, meta, _encode_spec(arr, spec, ref=spec.get("_arr"))


def _decode_array(meta: dict, code: bytes, resolve_ref) -> np.ndarray:
    dtype = meta["dtype"]
    shape = tuple(meta["shape"])
    dt = np.dtype(dtype)
    it = dt.itemsize
    if "pred" in meta:
        return _decode_residual(code, shape, dtype, meta["pred"], meta["shuf"])
    raw = _unxz(code)
    if meta.get("shuf"):
        raw = _unshuffle(raw, it)
    if "ref" not in meta:                          # plain
        return np.frombuffer(raw, dtype=dt).reshape(shape)
    mod = 1 << (8 * it)
    diff = np.frombuffer(raw, dtype=f"<u{it}").reshape(shape).astype(np.int64)
    rview, _, _ = _as_uint_view(resolve_ref(meta["ref"]))
    out = ((diff + rview.astype(np.int64)) % mod).astype(_uint_dtype(dt))
    return out.view(dt)


# ---------------------------------------------------------------------------
# file-level encode / decode
# ---------------------------------------------------------------------------
def _encode_file(data: bytes, ctx: _Context, relpath: str):
    """Return (codec, meta, code, extras) for the smallest *verified* encoding.

    Nothing is taken on trust: every candidate is decoded again here and must
    reproduce ``data`` byte-for-byte before it can be chosen, so a parser bug
    can only cost ratio, never fidelity. Structure-aware codecs are tried first;
    the generic ones are computed only if none of them works.
    """
    extras: dict = {}

    gz = _gzip_split(data)
    payload, gz_meta = data, None
    if gz is not None:
        header, body, trailer = gz
        try:
            inner = zlib.decompress(body, -15)
        except zlib.error:
            inner = None
        if inner is not None:
            params = _gzip_params(body, inner)
            if params is not None:          # we can re-emit this gzip stream exactly
                payload = inner
                gz_meta = {"header": header.hex(), "trailer": trailer.hex(), **params}

    prefix = "gz+" if gz_meta else ""
    options: list[tuple[str, dict, bytes]] = []

    mgh = _mgh_split(payload) if (gz_meta or payload[:4] == b"\x00\x00\x00\x01") else None
    if mgh is not None:
        head, arr, tail = mgh
        extras["volume"] = arr
        for suffix, meta, code in _array_options(arr, ctx):
            options.append((f"{prefix}mgh:{suffix}",
                            {"gz": gz_meta, "head": head.hex(), "tail": tail.hex(), **meta},
                            code))

    surf = _surface_split(payload) if not gz_meta else None
    if surf is not None:
        head, verts, faces, tail = surf
        extras["surface"] = (verts, faces)
        base = {"head": head.hex(), "tail": tail.hex(), "gz": gz_meta,
                "nv": int(verts.shape[0]), "nf": int(faces.shape[0])}
        vb = np.ascontiguousarray(verts.astype("<u4")).tobytes()
        fb = np.ascontiguousarray(faces.astype("<u4")).tobytes()
        options.append((f"{prefix}surf", dict(base),
                        _pack(_xz(vb, delta=4), _xz(fb, delta=4))))
        ref = ctx.surface_ref(verts.shape[0], faces.shape[0])
        if ref is not None:     # same topology: coordinates differ only a little
            dv = ((verts.astype(np.int64) - ref["verts"].astype(np.int64)) % (1 << 32)
                  ).astype("<u4")
            df = ((faces.astype(np.int64) - ref["faces"].astype(np.int64)) % (1 << 32)
                  ).astype("<u4")
            options.append((f"{prefix}surf:delta", {**base, "ref": ref["file_id"]},
                            _pack(_xz(np.ascontiguousarray(dv).tobytes(), delta=4),
                                  _xz(np.ascontiguousarray(df).tobytes(), delta=4))))

    curv = _curv_split(payload) if not gz_meta else None
    if curv is not None:
        head, vals = curv
        options.append((f"{prefix}curv", {"head": head.hex(), "gz": gz_meta,
                                          "n": int(vals.size)},
                        _xz(_shuffle(np.ascontiguousarray(vals.astype("<u4")).tobytes(), 4))))

    options.sort(key=lambda o: len(o[2]))
    for codec, meta, code in options:
        try:
            if _decode_file(codec, meta, code, ctx.resolve) == data:
                return codec, meta, code, extras
        except Exception:
            continue

    # Generic fallbacks: always available and trivially exact.
    if len(data) > 512:
        codec, meta, code = f"{prefix}xz", {"gz": gz_meta}, _xz(payload)
        try:
            if len(code) < len(data) and _decode_file(codec, meta, code, ctx.resolve) == data:
                return codec, meta, code, extras
        except Exception:
            pass
    return "store", {}, data, extras          # last resort: verbatim bytes


def _pack(*parts: bytes) -> bytes:
    out = [struct.pack("<I", len(parts))]
    out += [struct.pack("<I", len(p)) for p in parts]
    out += list(parts)
    return b"".join(out)


def _unpack(blob: bytes) -> list[bytes]:
    (n,) = struct.unpack("<I", blob[:4])
    lens = [struct.unpack("<I", blob[4 + 4 * i:8 + 4 * i])[0] for i in range(n)]
    off = 4 + 4 * n
    parts = []
    for ln in lens:
        parts.append(blob[off:off + ln])
        off += ln
    return parts


def _decode_file(codec: str, meta: dict, code: bytes, resolve_ref) -> bytes:
    """Reconstruct the original file bytes from a stored code."""
    if codec == "store":
        return code
    body = codec[3:] if codec.startswith("gz+") else codec

    if body.startswith("mgh:"):
        arr = _decode_array(meta, code, resolve_ref)
        payload = _mgh_join(bytes.fromhex(meta["head"]),
                            arr.reshape(tuple(meta["shape"])), bytes.fromhex(meta["tail"]))
    elif body.startswith("surf"):
        vpart, fpart = _unpack(code)
        nv, nf = meta["nv"], meta["nf"]
        verts = np.frombuffer(_unxz(vpart, delta=4), dtype="<u4").reshape(nv, 3)
        faces = np.frombuffer(_unxz(fpart, delta=4), dtype="<u4").reshape(nf, 3)
        if body == "surf:delta":
            ref_v, ref_f = resolve_ref(meta["ref"])
            verts = ((verts.astype(np.int64) + ref_v.astype(np.int64)) % (1 << 32)).astype("<u4")
            faces = ((faces.astype(np.int64) + ref_f.astype(np.int64)) % (1 << 32)).astype("<u4")
        payload = (bytes.fromhex(meta["head"])
                   + np.ascontiguousarray(verts.astype(">u4")).tobytes()
                   + np.ascontiguousarray(faces.astype(">u4")).tobytes()
                   + bytes.fromhex(meta["tail"]))
    elif body == "curv":
        vals = np.frombuffer(_unshuffle(_unxz(code), 4), dtype="<u4")
        payload = bytes.fromhex(meta["head"]) + \
            np.ascontiguousarray(vals.astype(">u4")).tobytes()
    elif body == "xz":
        payload = _unxz(code)
    else:
        raise ValueError(f"unknown codec {codec!r}")

    gz = meta.get("gz")
    if gz:
        return _gzip_rebuild(bytes.fromhex(gz["header"]), bytes.fromhex(gz["trailer"]),
                             payload, gz)
    return payload


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------
#: Files kept by the ``analysis`` profile — the non-regenerable, analysis-bearing
#: outputs. Everything else is omitted (but still listed, with its hash, in the
#: manifest). Patterns match the subject-relative POSIX path.
ANALYSIS_KEEP = [
    "stats/*",
    "scripts/recon-all.log", "scripts/build-stamp.txt", "scripts/patchdir.txt",
    "label/*.annot", "label/*.label",
    "surf/?h.white", "surf/?h.pial", "surf/?h.orig", "surf/?h.inflated",
    "surf/?h.sphere.reg", "surf/?h.thickness", "surf/?h.curv", "surf/?h.sulc",
    "surf/?h.area", "surf/?h.area.pial", "surf/?h.volume", "surf/?h.curv.pial",
    "mri/T1.mgz", "mri/orig.mgz", "mri/norm.mgz", "mri/brain.mgz",
    "mri/brainmask.mgz", "mri/aseg.mgz", "mri/aparc+aseg.mgz",
    "mri/aparc.a2009s+aseg.mgz", "mri/wm.mgz", "mri/ribbon.mgz",
    "mri/transforms/talairach.xfm", "mri/transforms/talairach.lta",
]

PROFILES = {"lossless": None, "analysis": ANALYSIS_KEEP}


def _selected(relpath: str, keep: list[str] | None) -> bool:
    if keep is None:
        return True
    return any(fnmatch.fnmatch(relpath, pat) for pat in keep)


# ---------------------------------------------------------------------------
# store / restore
# ---------------------------------------------------------------------------
def _tree_root(entries) -> str:
    """Merkle-style root over (relpath, sha256) so the file *set* is pinned too."""
    h = hashlib.sha256()
    for rel, sha in sorted(entries):
        h.update(f"F\t{rel}\t{sha}\n".encode("utf-8"))
    return h.hexdigest()


def store_subject(mf: MindFile, subject_dir, *, subject_id: str | None = None,
                  session_id: str | None = None, study_id: str | None = None,
                  profile: str = "lossless", effort: str = "balanced",
                  item_id: int | None = None, progress=None) -> dict:
    """Fold a recon-all subject tree into ``mf``; return a report.

    ``profile='lossless'`` stores every file (restorable bit-for-bit).
    ``profile='analysis'`` stores only :data:`ANALYSIS_KEEP`; omitted files are
    recorded in the manifest with their size and SHA-256.
    """
    root = Path(subject_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r} (have {sorted(PROFILES)})")
    keep = PROFILES[profile]
    subject_id = subject_id or root.name

    files = sorted((p for p in root.rglob("*") if p.is_file()),
                   key=lambda p: p.relative_to(root).as_posix())
    # Directories holding no files (recon-all leaves tmp/, trash/, src/ empty)
    # would otherwise vanish on restore, so record them explicitly.
    empty_dirs = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                        if p.is_dir() and not any(q.is_file() for q in p.rglob("*")))
    if item_id is None:
        cur = mf.conn.execute(
            "INSERT INTO items(external_id, origin, subject_id, session_id, study_id, "
            "modality, payload_kind, created_at) VALUES(?,?,?,?,?,?,?,?)",
            (subject_id, "primary", subject_id, session_id, study_id,
             "reconall", "json", utcnow()))
        item_id = cur.lastrowid
    elif mf.conn.execute("SELECT 1 FROM files WHERE item_id=?", (item_id,)).fetchone():
        # Re-storing would replace rows and hand them new ids, orphaning any
        # delta that referenced them. Refuse rather than corrupt.
        raise ValueError(f"item {item_id} already holds a stored tree; "
                         "store into a new item instead")

    ctx = _Context(effort=effort)
    entries, omitted = [], []
    block_candidates: list[dict] = []
    total_bytes = stored_bytes = 0
    codecs: dict[str, int] = {}

    for n, path in enumerate(files, 1):
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        sha = _vec.sha256_bytes(data)
        total_bytes += len(data)
        entries.append((rel, sha))
        if not _selected(rel, keep):
            omitted.append({"path": rel, "size": len(data), "sha256": sha})
            continue

        codec, meta, code, extras = _encode_file(data, ctx, rel)
        cur = mf.conn.execute(
            "INSERT OR REPLACE INTO files(item_id, relpath, size, codec, meta, code, "
            "code_bytes, raw_sha256, code_sha256, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (item_id, rel, len(data), codec,
             json.dumps(meta, separators=(",", ":"), sort_keys=True), code, len(code),
             sha, _vec.sha256_bytes(code), utcnow()))
        file_id = cur.lastrowid
        stored_bytes += len(code)
        codecs[codec] = codecs.get(codec, 0) + 1
        # Only non-delta files become references, so delta chains stay depth-1
        # and a restore never has to unwind a cascade.
        if "volume" in extras and not codec.endswith("delta"):
            ctx.remember_volume(file_id, extras["volume"])
        if "surface" in extras and not codec.endswith("delta"):
            v, f = extras["surface"]
            ctx.remember_surface(file_id, v, f)
        if codec in GENERIC_CODECS:
            # No structural codec fit this one; it may still compress better
            # alongside its peers. Re-read from disk in the blocking pass rather
            # than holding every candidate's bytes in memory.
            block_candidates.append(
                {"file_id": file_id, "relpath": rel, "size": len(data),
                 "code_bytes": len(code), "raw_sha256": sha, "path": path,
                 "data": (lambda p=path: p.read_bytes())})
        if progress:
            progress(n, len(files), rel, len(data), len(code))

    block_info = _store_blocks(mf, item_id, block_candidates)
    if block_info["blocks"]:
        stored_bytes -= block_info["saved_bytes"]

    manifest = {
        "kind": "freesurfer-reconall", "subject": subject_id, "profile": profile,
        "source": str(root), "n_files": len(files), "n_stored": len(files) - len(omitted),
        "total_bytes": total_bytes, "stored_bytes": stored_bytes,
        "tree_sha256": _tree_root(entries), "omitted": omitted,
        "empty_dirs": empty_dirs, "codecs": codecs,
        "blocks": block_info["blocks"], "block_saved_bytes": block_info["saved_bytes"],
        "created_at": utcnow(),
    }
    payload = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    mf.conn.execute("UPDATE items SET payload=?, payload_sha256=? WHERE id=?",
                    (payload, _vec.sha256_bytes(payload), item_id))
    mf.record("reconall", {"item_id": item_id, "subject": subject_id, "profile": profile,
                           "n_files": len(files), "n_stored": manifest["n_stored"],
                           "total_bytes": total_bytes, "stored_bytes": stored_bytes,
                           "tree_sha256": manifest["tree_sha256"]})
    mf.conn.commit()
    if block_info["blocks"]:
        # Blocking replaces per-file codes with one shared code, freeing pages
        # SQLite keeps on its freelist. Without reclaiming them the *file* grows
        # even though its contents shrank, which would defeat the whole point.
        _reclaim(mf)
    return {"item_id": item_id, "subject": subject_id, "profile": profile,
            "n_files": len(files), "n_stored": manifest["n_stored"],
            "n_omitted": len(omitted), "total_bytes": total_bytes,
            "stored_bytes": stored_bytes,
            "ratio": total_bytes / max(stored_bytes, 1), "codecs": codecs,
            "tree_sha256": manifest["tree_sha256"]}


# ---------------------------------------------------------------------------
# solid blocks: cross-file redundancy for files no structural codec fits
# ---------------------------------------------------------------------------
#: Codecs that mean "nothing structural applied" — these are the block candidates.
GENERIC_CODECS = {"xz", "gz+xz", "store"}

#: A file at or above this size is only ever paired (2-member block), never
#: pooled with many others, so one corrupt block can't take out a whole group.
BLOCK_LARGE_BYTES = 4_000_000
#: Cap on the uncompressed bytes pooled into one small-file block.
BLOCK_MAX_BYTES = 16_000_000
#: Cap on members in a small-file block (bounds the blast radius by count).
BLOCK_MAX_MEMBERS = 24
#: A pair whose members together exceed this is left alone: a block is coded
#: with a dictionary spanning it, and that gets slow and memory-hungry fast.
BLOCK_PAIR_MAX_BYTES = 128_000_000
#: Hard ceiling on a block's LZMA dictionary.
BLOCK_MAX_DICT = 256 << 20


def _solid_dict_size(total_bytes: int) -> int:
    """A dictionary that spans the whole block.

    This is the difference between blocking working and doing nothing: with the
    default 8 MB window, two 18 MB near-duplicates coded together saved 0.01 MB;
    with a window covering both, the same pair saved 7.5 MB. The match simply
    has to be reachable.
    """
    need = max(total_bytes, 1 << 23)                 # never below LZMA2 preset-6
    return min(1 << (need - 1).bit_length(), BLOCK_MAX_DICT)


def _solid_encode(blobs: list[bytes], preset: int = _PRESET) -> tuple[bytes, int]:
    joined = b"".join(blobs)
    dict_size = _solid_dict_size(len(joined))
    return _xz(joined, preset=preset, dict_size=dict_size), dict_size


#: Bytes of each large file sampled when probing whether two of them pair well.
PAIR_PROBE_BYTES = 2_000_000


def _read_head(path, n: int = PAIR_PROBE_BYTES) -> bytes:
    """Read only the first ``n`` bytes — pairing probes must not slurp the file."""
    with open(path, "rb") as fh:
        return fh.read(n)


def _plan_blocks(candidates: list[dict]) -> list[list[dict]]:
    """Group generic-coded files into bounded solid blocks.

    Small files are pooled by (top-level directory, suffix) — that is where many
    little files share structure. Large files are only ever paired with their
    most promising peer, keeping a corrupt block to two files.
    """
    groups: list[list[dict]] = []
    small = [c for c in candidates if c["size"] < BLOCK_LARGE_BYTES]
    large = [c for c in candidates if c["size"] >= BLOCK_LARGE_BYTES]

    keyed: dict[tuple, list[dict]] = {}
    for c in small:
        rel = c["relpath"]
        key = (rel.split("/")[0] if "/" in rel else ".", rel.rsplit(".", 1)[-1])
        keyed.setdefault(key, []).append(c)
    for members in keyed.values():
        cur: list[dict] = []
        total = 0
        for c in members:
            if cur and (total + c["size"] > BLOCK_MAX_BYTES
                        or len(cur) >= BLOCK_MAX_MEMBERS):
                groups.append(cur)
                cur, total = [], 0
            cur.append(c)
            total += c["size"]
        if cur:
            groups.append(cur)

    # Large files are only ever PAIRED, so one corrupt block costs two files.
    # Probe on cached heads: reading whole multi-hundred-MB files to rank
    # candidates would cost far more than the compression it saves.
    heads = {c["file_id"]: _read_head(c["path"]) for c in large}
    # The probe must use a window spanning BOTH heads; preset 1's 1 MB default
    # cannot reach across a 2 MB head, so it would score every pair as useless.
    probe_dict = _solid_dict_size(2 * PAIR_PROBE_BYTES)
    solo_head = {fid: len(_xz(h, preset=1, dict_size=probe_dict))
                 for fid, h in heads.items()}
    remaining = list(large)
    while len(remaining) > 1:
        a = remaining.pop(0)
        best, best_gain = None, 0
        for b in remaining:
            if a["size"] + b["size"] > BLOCK_PAIR_MAX_BYTES:
                continue
            together = len(_xz(heads[a["file_id"]] + heads[b["file_id"]], preset=1,
                               dict_size=probe_dict))
            gain = solo_head[a["file_id"]] + solo_head[b["file_id"]] - together
            if gain > best_gain:
                best, best_gain = b, gain
        if best is not None:
            remaining.remove(best)
            groups.append([a, best])
    return [g for g in groups if len(g) > 1]


def _store_blocks(mf: MindFile, item_id: int, candidates: list[dict],
                  progress=None) -> dict:
    """Re-code groups of generic files as solid blocks where that is smaller.

    Each block is decoded again and every member re-hashed before it replaces
    the independent encodings, so blocking can never cost fidelity.
    """
    saved = 0
    blocks = 0
    for group in _plan_blocks(candidates):
        blobs = [c["data"]() for c in group]
        solo = sum(c["code_bytes"] for c in group)
        code, dict_size = _solid_encode(blobs)
        if len(code) >= solo:
            continue                                   # blocking didn't pay
        offsets, off = [], 0
        for b in blobs:
            offsets.append((off, len(b)))
            off += len(b)
        joined = _unxz(code, dict_size=dict_size)      # verify before committing
        if len(joined) != off or any(
                _vec.sha256_bytes(joined[o:o + n]) != c["raw_sha256"]
                for (o, n), c in zip(offsets, group)):
            continue
        cur = mf.conn.execute(
            "INSERT INTO file_blocks(item_id, codec, n_files, raw_bytes, code, "
            "code_bytes, code_sha256, dict_size, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (item_id, "solid:xz", len(group), off, code, len(code),
             _vec.sha256_bytes(code), dict_size, utcnow()))
        bid = cur.lastrowid
        for (o, n), c in zip(offsets, group):
            mf.conn.execute(
                "UPDATE files SET code=?, code_bytes=0, code_sha256='', codec=?, "
                "block_id=?, block_offset=?, block_length=? WHERE id=?",
                (b"", "block", bid, o, n, c["file_id"]))
        saved += solo - len(code)
        blocks += 1
        if progress:
            progress(blocks, len(group), solo - len(code))
    mf.conn.commit()
    return {"blocks": blocks, "saved_bytes": saved}


def _reclaim(mf: MindFile) -> int:
    """VACUUM so pages freed by blocking are actually returned to the filesystem.

    Returns the bytes reclaimed. VACUUM cannot run inside a transaction, so the
    connection is briefly put into autocommit.
    """
    before = mf.path.stat().st_size if mf.path.exists() else 0
    mf.conn.commit()
    prev = mf.conn.isolation_level
    try:
        mf.conn.isolation_level = None
        mf.conn.execute("VACUUM")
    finally:
        mf.conn.isolation_level = prev
    after = mf.path.stat().st_size if mf.path.exists() else 0
    return max(0, before - after)


def _block_bytes(mf: MindFile, block_id: int) -> bytes:
    r = mf.conn.execute("SELECT code, dict_size FROM file_blocks WHERE id=?",
                        (block_id,)).fetchone()
    if r is None:
        raise KeyError(f"solid block {block_id} is missing")
    return _unxz(r["code"], dict_size=r["dict_size"])


def _ref_resolver(mf: MindFile, budget: int = 512_000_000):
    """Decode a referenced file and hand back the array/surface it contributes.

    Decoded references are cached under a byte budget: a reference is typically
    reused by a handful of files, but holding every one would defeat the point
    of not unpacking the tree.
    """
    cache: dict[int, object] = {}
    held = [0]

    def _size(obj) -> int:
        if isinstance(obj, tuple):
            return sum(a.nbytes for a in obj)
        return obj.nbytes if hasattr(obj, "nbytes") else len(obj)

    def resolve(file_id):
        if file_id in cache:
            return cache[file_id]
        r = mf.conn.execute(
            "SELECT codec, meta, code FROM files WHERE id=?", (file_id,)).fetchone()
        if r is None:
            raise KeyError(f"delta reference file id {file_id} is missing")
        meta = json.loads(r["meta"])
        data = _decode_file(r["codec"], meta, r["code"], resolve)
        body = r["codec"][3:] if r["codec"].startswith("gz+") else r["codec"]
        payload = data
        if meta.get("gz"):
            payload = zlib.decompress(_gzip_split(data)[1], -15)
        if body.startswith("mgh"):
            out = _mgh_split(payload)[1]
        elif body.startswith("surf"):
            _h, v, f, _t = _surface_split(payload)
            out = (v, f)
        else:
            out = payload
        while cache and held[0] + _size(out) > budget:
            held[0] -= _size(cache.pop(next(iter(cache))))
        cache[file_id] = out
        held[0] += _size(out)
        return out

    return resolve


def load_file(mf: MindFile, item_id: int, relpath: str) -> bytes:
    """Reconstruct one stored file's original bytes."""
    r = mf.conn.execute(
        "SELECT codec, meta, code, block_id, block_offset, block_length FROM files "
        "WHERE item_id=? AND relpath=?", (item_id, relpath)).fetchone()
    if r is None:
        raise KeyError(f"{relpath!r} is not stored for item {item_id}")
    if r["block_id"] is not None:      # solid member: decode its block, then slice
        blob = _block_bytes(mf, r["block_id"])
        return blob[r["block_offset"]:r["block_offset"] + r["block_length"]]
    return _decode_file(r["codec"], json.loads(r["meta"]), r["code"], _ref_resolver(mf))


def list_files(mf: MindFile, item_id: int) -> list[dict]:
    return [dict(r) for r in mf.conn.execute(
        "SELECT id, relpath, size, code_bytes, codec, raw_sha256 FROM files "
        "WHERE item_id=? ORDER BY relpath", (item_id,))]


def manifest(mf: MindFile, item_id: int) -> dict:
    r = mf.conn.execute("SELECT payload FROM items WHERE id=?", (item_id,)).fetchone()
    if r is None or r["payload"] is None:
        raise KeyError(f"item {item_id} has no recon-all manifest")
    return json.loads(r["payload"])


def restore_subject(mf: MindFile, item_id: int, dest, *, verify: bool = True,
                    progress=None) -> dict:
    """Write the stored tree back to ``dest``; returns a per-file report."""
    dest = Path(dest)
    resolve = _ref_resolver(mf)
    rows = mf.conn.execute(
        "SELECT relpath, codec, meta, code, raw_sha256, block_id, block_offset, "
        "block_length FROM files WHERE item_id=? ORDER BY block_id, id",
        (item_id,)).fetchall()
    try:                                   # recreate directories that hold no files
        for rel in manifest(mf, item_id).get("empty_dirs", []):
            (dest / rel).mkdir(parents=True, exist_ok=True)
    except KeyError:
        pass
    written, bad = 0, []
    block_cache: dict[int, bytes] = {}
    for n, r in enumerate(rows, 1):
        if r["block_id"] is not None:      # decode each block once, serve all members
            if r["block_id"] not in block_cache:
                block_cache = {r["block_id"]: _block_bytes(mf, r["block_id"])}
            blob = block_cache[r["block_id"]]
            data = blob[r["block_offset"]:r["block_offset"] + r["block_length"]]
        else:
            data = _decode_file(r["codec"], json.loads(r["meta"]), r["code"], resolve)
        if verify and _vec.sha256_bytes(data) != r["raw_sha256"]:
            bad.append(r["relpath"])
            continue
        out = dest / r["relpath"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        written += 1
        if progress:
            progress(n, len(rows), r["relpath"])
    return {"status": "ok" if not bad else "corrupted", "restored": written,
            "failed": bad, "dest": str(dest)}


def verify_files(mf: MindFile, *, deep: bool = False) -> dict:
    """Re-check every stored file.

    The cheap pass re-hashes each stored code (catching in-place corruption of
    hundreds of megabytes for the cost of a hash). ``deep=True`` additionally
    decodes every file and compares against the *original* file's hash, proving
    the codecs still reconstruct the tree exactly.
    """
    resolve = _ref_resolver(mf)
    bad = []
    n = 0
    # A solid block is hashed once; if it fails, every member it carries is
    # reported — that shared fate is the cost of blocking, made explicit.
    broken_blocks = {r["id"] for r in mf.conn.execute(
        "SELECT id, code, code_sha256 FROM file_blocks")
        if _vec.sha256_bytes(r["code"]) != r["code_sha256"]}

    for r in mf.conn.execute(
            "SELECT id, item_id, relpath, codec, meta, code, raw_sha256, code_sha256, "
            "block_id, block_offset, block_length FROM files ORDER BY id"):
        n += 1
        if r["block_id"] is not None:
            if r["block_id"] in broken_blocks:
                bad.append({"item_id": r["item_id"], "path": r["relpath"],
                            "issue": "block", "block_id": r["block_id"]})
                continue
        elif _vec.sha256_bytes(r["code"]) != r["code_sha256"]:
            bad.append({"item_id": r["item_id"], "path": r["relpath"], "issue": "code"})
            continue
        if not deep:
            continue
        try:
            if r["block_id"] is not None:
                blob = _block_bytes(mf, r["block_id"])
                data = blob[r["block_offset"]:r["block_offset"] + r["block_length"]]
            else:
                data = _decode_file(r["codec"], json.loads(r["meta"]), r["code"], resolve)
        except Exception as e:
            bad.append({"item_id": r["item_id"], "path": r["relpath"], "issue": str(e)})
            continue
        if _vec.sha256_bytes(data) != r["raw_sha256"]:
            bad.append({"item_id": r["item_id"], "path": r["relpath"],
                        "issue": "reconstruction"})
    return {"status": "ok" if not bad else "corrupted", "checked": n,
            "deep": deep, "bad": bad}


# ---------------------------------------------------------------------------
# morphometry: the numbers recon-all actually produced
# ---------------------------------------------------------------------------
#: Whole-brain measures from the ``# Measure`` header lines of aseg.stats.
ASEG_MEASURES = [
    "CortexVol", "lhCortexVol", "rhCortexVol", "CorticalWhiteMatterVol",
    "lhCorticalWhiteMatterVol", "rhCorticalWhiteMatterVol", "SubCortGrayVol",
    "TotalGrayVol", "SupraTentorialVol", "BrainSegVol", "eTIV", "ICV",
]

#: Subcortical structures from the aseg.stats table (fixed order -> fixed dim).
ASEG_STRUCTURES = [
    "Left-Lateral-Ventricle", "Right-Lateral-Ventricle", "Left-Inf-Lat-Vent",
    "Right-Inf-Lat-Vent", "Left-Cerebellum-White-Matter",
    "Right-Cerebellum-White-Matter", "Left-Cerebellum-Cortex",
    "Right-Cerebellum-Cortex", "Left-Thalamus-Proper", "Right-Thalamus-Proper",
    "Left-Caudate", "Right-Caudate", "Left-Putamen", "Right-Putamen",
    "Left-Pallidum", "Right-Pallidum", "3rd-Ventricle", "4th-Ventricle",
    "Brain-Stem", "Left-Hippocampus", "Right-Hippocampus", "Left-Amygdala",
    "Right-Amygdala", "CSF", "Left-Accumbens-area", "Right-Accumbens-area",
    "Left-VentralDC", "Right-VentralDC", "CC_Posterior", "CC_Mid_Posterior",
    "CC_Central", "CC_Mid_Anterior", "CC_Anterior",
]

#: Desikan-Killiany cortical regions (?h.aparc.stats), per hemisphere.
DK_REGIONS = [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus",
    "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
    "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
    "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
    "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
    "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
    "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
    "temporalpole", "transversetemporal", "insula",
]

#: Dimension of :func:`morphometry_vector` — fixed, so subjects are comparable.
MORPHOMETRY_DIM = len(ASEG_MEASURES) + len(ASEG_STRUCTURES) + 2 * len(DK_REGIONS)


def parse_aseg_stats(text: str) -> dict:
    """Parse ``stats/aseg.stats`` into ``{measures: {...}, structures: {...}}``."""
    measures, structures = {}, {}
    for line in text.splitlines():
        if line.startswith("# Measure"):
            parts = [p.strip() for p in line[len("# Measure"):].split(",")]
            if len(parts) >= 4:
                try:
                    measures[parts[1]] = float(parts[-2])
                except ValueError:
                    pass
        elif line and not line.startswith("#"):
            cols = line.split()
            if len(cols) >= 5:
                try:
                    structures[cols[4]] = float(cols[3])   # volume in mm^3
                except (ValueError, IndexError):
                    pass
    return {"measures": measures, "structures": structures}


def parse_aparc_stats(text: str) -> dict:
    """Parse ``stats/?h.aparc.stats`` into per-region thickness/area/volume."""
    measures, regions = {}, {}
    for line in text.splitlines():
        if line.startswith("# Measure"):
            parts = [p.strip() for p in line[len("# Measure"):].split(",")]
            if len(parts) >= 4:
                try:
                    measures[parts[1]] = float(parts[-2])
                except ValueError:
                    pass
        elif line and not line.startswith("#"):
            cols = line.split()
            if len(cols) >= 5:
                try:
                    regions[cols[0]] = {"area": float(cols[2]), "volume": float(cols[3]),
                                        "thickness": float(cols[4])}
                except (ValueError, IndexError):
                    pass
    return {"measures": measures, "regions": regions}


def read_stats(mf: MindFile, item_id: int) -> dict:
    """Read whatever stats tables the stored tree contains."""
    out: dict = {"aseg": None, "lh": None, "rh": None}
    have = {r["relpath"] for r in mf.conn.execute(
        "SELECT relpath FROM files WHERE item_id=?", (item_id,))}
    if "stats/aseg.stats" in have:
        out["aseg"] = parse_aseg_stats(
            load_file(mf, item_id, "stats/aseg.stats").decode("utf-8", "replace"))
    for hemi in ("lh", "rh"):
        rel = f"stats/{hemi}.aparc.stats"
        if rel in have:
            out[hemi] = parse_aparc_stats(
                load_file(mf, item_id, rel).decode("utf-8", "replace"))
    return out


def morphometry_vector(stats: dict) -> np.ndarray:
    """Fixed-dimension morphometry vector: volumes normalised by eTIV + thickness.

    Volumes are divided by the subject's intracranial volume so head size does
    not dominate; cortical thickness is already size-normalised and used as-is.
    Missing entries are 0, so a partial tree still yields a comparable vector.
    """
    vec = np.zeros(MORPHOMETRY_DIM, dtype=np.float32)
    aseg = stats.get("aseg") or {"measures": {}, "structures": {}}
    measures, structures = aseg["measures"], aseg["structures"]
    etiv = measures.get("eTIV") or measures.get("ICV") or 0.0
    scale = 1.0 / etiv if etiv else 0.0
    i = 0
    for key in ASEG_MEASURES:
        vec[i] = measures.get(key, 0.0) * scale
        i += 1
    for key in ASEG_STRUCTURES:
        vec[i] = structures.get(key, 0.0) * scale
        i += 1
    for hemi in ("lh", "rh"):
        regions = (stats.get(hemi) or {}).get("regions", {})
        for name in DK_REGIONS:
            vec[i] = regions.get(name, {}).get("thickness", 0.0)
            i += 1
    return vec


def verify_tree(mf: MindFile, item_id: int) -> dict:
    """Check the stored file *set* against the manifest's Merkle root."""
    man = manifest(mf, item_id)
    stored = {r["relpath"]: r["raw_sha256"] for r in mf.conn.execute(
        "SELECT relpath, raw_sha256 FROM files WHERE item_id=?", (item_id,))}
    entries = list(stored.items()) + [(o["path"], o["sha256"]) for o in man["omitted"]]
    root = _tree_root(entries)
    return {"status": "ok" if root == man["tree_sha256"] else "corrupted",
            "tree_sha256": root, "expected": man["tree_sha256"],
            "n_stored": len(stored), "n_omitted": len(man["omitted"])}
