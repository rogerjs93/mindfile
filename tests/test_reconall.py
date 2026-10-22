"""recon-all container: codecs are exact, the tree restores, tamper is caught."""

from __future__ import annotations

import gzip
import struct
import zlib

import numpy as np
import pytest

from mindfile import MindFile, _vec, audit, checkpoint, modality, reconall, render


# -- synthetic fixtures shaped like the real formats -------------------------
def _mgh_bytes(arr: np.ndarray, *, gz: bool = True) -> bytes:
    """A minimal but valid MGH/MGZ file around ``arr`` (nframes, d, h, w)."""
    typ = {("u", 1): 0, ("i", 4): 1, ("f", 4): 3, ("i", 2): 4}[
        (arr.dtype.kind, arr.dtype.itemsize)]
    nf, d, h, w = arr.shape
    header = struct.pack(">6i", 1, w, h, d, nf, typ) + b"\x00" * (284 - 24)
    payload = header + np.ascontiguousarray(arr).tobytes() + b"\x00" * 194
    return gzip.compress(payload, 6) if gz else payload


def _surface_bytes(verts: np.ndarray, faces: np.ndarray) -> bytes:
    head = b"\xff\xff\xfe" + b"created by test\n\n" + struct.pack(
        ">2i", verts.shape[0], faces.shape[0])
    return (head + np.ascontiguousarray(verts.astype(">f4")).tobytes()
            + np.ascontiguousarray(faces.astype(">i4")).tobytes())


def _curv_bytes(vals: np.ndarray) -> bytes:
    return (b"\xff\xff\xff" + struct.pack(">3i", vals.size, 0, 1)
            + np.ascontiguousarray(vals.astype(">f4")).tobytes())


ASEG_STATS = """\
# Title Segmentation Statistics
# Measure Cortex, CortexVol, Total cortical gray matter volume, 500228.329498, mm^3
# Measure IntraCranialVol, ICV, Intracranial Volume, 1703159.366507, mm^3
# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1700000.0, mm^3
# ColHeaders  Index SegId NVoxels Volume_mm3 StructName
  1   4     11330    11330.0  Left-Lateral-Ventricle            34.5709
  2  17      4200     4200.0  Left-Hippocampus                  60.1000
  3  53      4310     4310.0  Right-Hippocampus                 61.2000
"""

APARC_STATS = """\
# Measure Cortex, NumVert, Number of Vertices, 147147, unitless
# Measure Cortex, MeanThickness, Mean Thickness, 2.40686, mm
# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg
bankssts                                 1604   1133   2427  2.203 0.440
cuneus                                   3810   2446   6756  2.539 0.509
"""


@pytest.fixture()
def subject(tmp_path):
    """A small synthetic recon-all-shaped subject tree."""
    root = tmp_path / "subj"
    rng = np.random.default_rng(0)

    # Two nearly-identical uint8 volumes (like brain / brain.finalsurfs) and a
    # smooth float32 field (like a warp map).
    base = np.zeros((1, 16, 32, 32), dtype=">u1")
    base[0, 4:12, 8:24, 8:24] = rng.integers(60, 200, (8, 16, 16)).astype(">u1")
    twin = base.copy()
    field = (np.linspace(0, 1, 16 * 32 * 32, dtype=">f4")
             .reshape(1, 16, 32, 32).astype(">f4"))

    (root / "mri" / "transforms").mkdir(parents=True)
    (root / "mri" / "T1.mgz").write_bytes(_mgh_bytes(base))
    (root / "mri" / "brain.mgz").write_bytes(_mgh_bytes(twin))
    (root / "mri" / "warp.mgz").write_bytes(_mgh_bytes(field))
    (root / "mri" / "uncompressed.mgh").write_bytes(_mgh_bytes(base, gz=False))
    (root / "mri" / "transforms" / "talairach.xfm").write_text("MNI Transform File\n")

    # Two surfaces sharing topology (like ?h.white / ?h.pial) + a curv file.
    nv = 500
    verts = rng.normal(0, 10, (nv, 3)).astype(">f4")
    faces = rng.integers(0, nv, (900, 3)).astype(">i4")
    (root / "surf").mkdir()
    (root / "surf" / "lh.white").write_bytes(_surface_bytes(verts, faces))
    (root / "surf" / "lh.pial").write_bytes(
        _surface_bytes((verts.astype(np.float32) + 0.5).astype(">f4"), faces))
    (root / "surf" / "lh.thickness").write_bytes(
        _curv_bytes(rng.normal(2.5, 0.4, nv).astype(">f4")))

    (root / "stats").mkdir()
    (root / "stats" / "aseg.stats").write_text(ASEG_STATS)
    (root / "stats" / "lh.aparc.stats").write_text(APARC_STATS)
    (root / "stats" / "rh.aparc.stats").write_text(APARC_STATS)

    (root / "scripts").mkdir()
    (root / "scripts" / "recon-all.log").write_text("recon-all\n" * 500)
    (root / "touch").mkdir()
    (root / "touch" / "done.touch").write_text("")
    return root


@pytest.fixture()
def mf(tmp_path):
    f = MindFile.create(tmp_path / "recon.gnos", modality="reconall")
    yield f
    f.close()


def _tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


# -- codec unit tests --------------------------------------------------------
@pytest.mark.parametrize("dtype", [">u1", ">i2", ">i4", ">f4"])
@pytest.mark.parametrize("mode", ["x", "y", "z", "yzmed"])
def test_predictors_are_exact_for_every_dtype(dtype, mode):
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 200, (2, 5, 7, 9)).astype(dtype)
    code = reconall._encode_residual(arr, mode, False)
    back = reconall._decode_residual(code, arr.shape, dtype, mode, False)
    assert back.tobytes() == arr.tobytes()


def test_gzip_stream_is_reproduced_byte_for_byte():
    payload = b"freesurfer" * 5000
    original = gzip.compress(payload, 6)
    header, body, trailer = reconall._gzip_split(original)
    params = reconall._gzip_params(body, zlib.decompress(body, -15))
    assert params is not None
    assert reconall._gzip_rebuild(header, trailer, payload, params) == original


def test_unrebuildable_gzip_still_round_trips(mf, tmp_path):
    """A gzip stream we cannot re-emit must fall back to an exact codec."""
    root = tmp_path / "odd"
    root.mkdir()
    # zlib level 6 output with a bespoke header the fast path won't match.
    weird = b"\x1f\x8b\x08\x08" + b"\x00" * 6 + b"name\x00"
    co = zlib.compressobj(9, zlib.DEFLATED, -15, 9, zlib.Z_HUFFMAN_ONLY)
    body = co.compress(b"abc" * 10000) + co.flush()
    (root / "weird.gz").write_bytes(weird + body + b"\x00" * 8)
    rep = reconall.store_subject(mf, root)
    assert reconall.load_file(mf, rep["item_id"], "weird.gz") == (root / "weird.gz").read_bytes()


# -- store / restore ---------------------------------------------------------
def test_store_and_restore_is_byte_identical(mf, subject, tmp_path):
    (subject / "trash").mkdir()                      # recon-all leaves dirs empty
    rep = reconall.store_subject(mf, subject, subject_id="s1")
    assert rep["n_stored"] == rep["n_files"]

    dest = tmp_path / "restored"
    res = reconall.restore_subject(mf, rep["item_id"], dest)
    assert res["status"] == "ok"
    assert _tree(dest) == _tree(subject)
    assert (dest / "trash").is_dir()                 # empty dirs survive too


def test_restoring_into_an_existing_item_is_refused(mf, subject):
    rep = reconall.store_subject(mf, subject)
    with pytest.raises(ValueError, match="already holds a stored tree"):
        reconall.store_subject(mf, subject, item_id=rep["item_id"])


def test_near_identical_volume_uses_a_delta(mf, subject):
    rep = reconall.store_subject(mf, subject)
    rows = {r["relpath"]: r for r in reconall.list_files(mf, rep["item_id"])}
    assert rows["mri/brain.mgz"]["codec"].endswith("delta")
    # the delta must be far smaller than the volume it mirrors
    assert rows["mri/brain.mgz"]["code_bytes"] < rows["mri/T1.mgz"]["code_bytes"] / 4


def test_surface_with_shared_topology_uses_a_delta(mf, subject):
    """Whichever surface is stored second delta-codes against the first."""
    rep = reconall.store_subject(mf, subject)
    rows = {r["relpath"]: r for r in reconall.list_files(mf, rep["item_id"])}
    anchor, delta = rows["surf/lh.pial"], rows["surf/lh.white"]   # stored in sorted order
    assert anchor["codec"] == "surf"
    assert delta["codec"] == "surf:delta"
    assert delta["code_bytes"] < anchor["code_bytes"]


def test_uncompressed_mgh_and_curv_use_structure_aware_codecs(mf, subject):
    rep = reconall.store_subject(mf, subject)
    rows = {r["relpath"]: r for r in reconall.list_files(mf, rep["item_id"])}
    assert rows["mri/uncompressed.mgh"]["codec"].startswith("mgh:")
    assert rows["surf/lh.thickness"]["codec"] == "curv"


def test_analysis_profile_omits_but_still_records(mf, subject):
    rep = reconall.store_subject(mf, subject, profile="analysis")
    assert rep["n_stored"] < rep["n_files"]
    man = reconall.manifest(mf, rep["item_id"])
    omitted = {o["path"] for o in man["omitted"]}
    assert "touch/done.touch" in omitted
    assert "stats/aseg.stats" not in omitted
    # every omitted file is still pinned by its hash
    assert all(len(o["sha256"]) == 64 for o in man["omitted"])
    assert reconall.verify_tree(mf, rep["item_id"])["status"] == "ok"


# -- solid blocks ------------------------------------------------------------
@pytest.fixture()
def blocky(tmp_path):
    """A tree of many small, mutually-similar text files (labels)."""
    root = tmp_path / "blocky"
    (root / "label").mkdir(parents=True)
    for i in range(reconall.BLOCK_MAX_MEMBERS + 8):     # enough to span >1 block
        # same structure, different numbers — compresses far better together
        rows = "\n".join(f"{j + i}  -12.34  45.67  -8.90  0.0000000000"
                         for j in range(4000))
        (root / "label" / f"lh.region{i:02d}.label").write_text(f"#!ascii label\n{rows}\n")
    return root


def test_many_similar_small_files_are_pooled_into_blocks(mf, blocky):
    rep = reconall.store_subject(mf, blocky)
    man = reconall.manifest(mf, rep["item_id"])
    assert man["blocks"] >= 1 and man["block_saved_bytes"] > 0

    rows = reconall.list_files(mf, rep["item_id"])
    assert any(r["codec"] == "block" for r in rows)
    # every member still reconstructs exactly, one file at a time
    for r in rows:
        data = reconall.load_file(mf, rep["item_id"], r["relpath"])
        assert _vec.sha256_bytes(data) == r["raw_sha256"]


def test_blocks_stay_bounded(mf, blocky):
    reconall.store_subject(mf, blocky)
    for b in mf.conn.execute("SELECT n_files, raw_bytes FROM file_blocks"):
        assert b["n_files"] <= reconall.BLOCK_MAX_MEMBERS
        assert b["raw_bytes"] <= reconall.BLOCK_MAX_BYTES + reconall.BLOCK_LARGE_BYTES


def test_blocked_tree_still_restores_byte_identically(mf, blocky, tmp_path):
    rep = reconall.store_subject(mf, blocky)
    assert reconall.manifest(mf, rep["item_id"])["blocks"] >= 1
    dest = tmp_path / "out"
    assert reconall.restore_subject(mf, rep["item_id"], dest)["status"] == "ok"
    assert _tree(dest) == _tree(blocky)


def test_corrupt_block_flags_every_member_it_carries(mf, blocky):
    """The cost of blocking, made explicit: shared fate within one block."""
    rep = reconall.store_subject(mf, blocky)
    b = mf.conn.execute(
        "SELECT id, code, n_files FROM file_blocks ORDER BY n_files DESC LIMIT 1").fetchone()
    blob = bytearray(b["code"])
    blob[len(blob) // 2] ^= 0x01
    mf.conn.execute("UPDATE file_blocks SET code=? WHERE id=?", (bytes(blob), b["id"]))
    mf.conn.commit()

    assert mf.conn.execute("SELECT COUNT(*) FROM file_blocks").fetchone()[0] >= 2
    report = reconall.verify_files(mf)
    assert report["status"] == "corrupted"
    assert len(report["bad"]) == b["n_files"]          # the whole block, no more
    assert all(x["issue"] == "block" for x in report["bad"])
    total = mf.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    assert len(report["bad"]) < total                  # other blocks are untouched


def test_block_tamper_breaks_the_checkpoint(mf, blocky):
    rep = reconall.store_subject(mf, blocky)
    checkpoint.create_checkpoint(mf, "sealed")
    assert checkpoint.verify_checkpoint(mf, "sealed")["status"] == "ok"

    b = mf.conn.execute("SELECT id, code FROM file_blocks LIMIT 1").fetchone()
    blob = bytearray(b["code"])
    blob[len(blob) // 2] ^= 0xFF
    mf.conn.execute(
        "UPDATE file_blocks SET code=?, code_sha256=? WHERE id=?",
        (bytes(blob), _vec.sha256_bytes(bytes(blob)), b["id"]))   # also fix the hash
    mf.conn.commit()

    cp = checkpoint.verify_checkpoint(mf, "sealed")
    assert cp["status"] == "corrupted" and cp["changed_files"]


def test_large_near_duplicates_are_paired_with_a_spanning_dictionary(mf, tmp_path):
    """Regression: the win needs a window covering BOTH files.

    With LZMA's default 8MB window (and a 1MB window in the pairing probe) the
    cross-file match is simply out of reach, and blocking silently saves nothing.
    """
    root = tmp_path / "dup"
    root.mkdir()
    rng = np.random.default_rng(7)
    big = rng.integers(0, 256, 5_000_000, dtype=np.uint8).tobytes()   # incompressible
    (root / "a.bin").write_bytes(big)
    (root / "b.bin").write_bytes(big[:-16] + b"DIFFERENTTAILXX")      # near-duplicate

    rep = reconall.store_subject(mf, root)
    rows = {r["relpath"]: r for r in reconall.list_files(mf, rep["item_id"])}
    assert rows["a.bin"]["codec"] == "block" and rows["b.bin"]["codec"] == "block"

    b = mf.conn.execute("SELECT n_files, raw_bytes, code_bytes, dict_size "
                        "FROM file_blocks").fetchone()
    assert b["n_files"] == 2                       # large files pair, never pool
    assert b["dict_size"] >= b["raw_bytes"]        # window spans the whole block
    # the duplicate should cost almost nothing on top of the original
    assert b["code_bytes"] < 0.6 * len(big) * 2

    for rel in ("a.bin", "b.bin"):
        assert reconall.load_file(mf, rep["item_id"], rel) == (root / rel).read_bytes()


def test_blocking_never_makes_a_file_bigger(mf, subject):
    """A block is only adopted when it beats independent coding."""
    rep = reconall.store_subject(mf, subject)
    man = reconall.manifest(mf, rep["item_id"])
    assert man["block_saved_bytes"] >= 0


def test_blocking_shrinks_the_file_on_disk(tmp_path, blocky):
    """Regression: replaced codes free SQLite pages that must be reclaimed,
    or the .gnos grows on disk even though its contents shrank."""
    a, b = tmp_path / "noblock.gnos", tmp_path / "block.gnos"
    with MindFile.create(a, modality="reconall") as m:
        reconall.store_subject(m, blocky)
        # undo blocking's benefit by measuring a store with blocking disabled
    with MindFile.create(b, modality="reconall") as m:
        rep = reconall.store_subject(m, blocky)
        saved = reconall.manifest(m, rep["item_id"])["block_saved_bytes"]
        assert saved > 0
        logical = m.conn.execute(
            "SELECT COALESCE(SUM(code_bytes),0) FROM files").fetchone()[0] + \
            m.conn.execute("SELECT COALESCE(SUM(code_bytes),0) FROM file_blocks").fetchone()[0]
    # the file should sit close to its logical content, not carry freed pages
    assert b.stat().st_size < logical * 1.5 + 1_000_000


# -- integrity ---------------------------------------------------------------
def test_verify_and_audit_pass_on_a_sound_file(mf, subject):
    reconall.store_subject(mf, subject)
    assert reconall.verify_files(mf)["status"] == "ok"
    assert reconall.verify_files(mf, deep=True)["status"] == "ok"
    assert audit.audit(mf, deep=True)["status"] == "ok"


def test_single_bit_tamper_in_a_stored_file_is_caught(mf, subject):
    rep = reconall.store_subject(mf, subject)
    row = mf.conn.execute(
        "SELECT id, code FROM files WHERE relpath='mri/T1.mgz' AND item_id=?",
        (rep["item_id"],)).fetchone()
    blob = bytearray(row["code"])
    blob[len(blob) // 2] ^= 0x01
    mf.conn.execute("UPDATE files SET code=? WHERE id=?", (bytes(blob), row["id"]))
    mf.conn.commit()

    report = reconall.verify_files(mf)
    assert report["status"] == "corrupted"
    assert report["bad"][0]["path"] == "mri/T1.mgz"
    assert "files" in audit.audit(mf)["corrupted"]


def test_tamper_with_updated_hash_is_still_caught_by_a_checkpoint(mf, subject):
    """Rewriting the code *and* its hash defeats the local check, not the save point."""
    rep = reconall.store_subject(mf, subject)
    checkpoint.create_checkpoint(mf, "sealed")
    assert checkpoint.verify_checkpoint(mf, "sealed")["status"] == "ok"

    from mindfile import _vec
    row = mf.conn.execute(
        "SELECT id, code FROM files WHERE relpath='mri/T1.mgz' AND item_id=?",
        (rep["item_id"],)).fetchone()
    blob = bytearray(row["code"])
    blob[len(blob) // 2] ^= 0xFF
    mf.conn.execute("UPDATE files SET code=?, code_sha256=? WHERE id=?",
                    (bytes(blob), _vec.sha256_bytes(bytes(blob)), row["id"]))
    mf.conn.commit()

    assert reconall.verify_files(mf)["status"] == "ok"          # local hashes agree
    cp = checkpoint.verify_checkpoint(mf, "sealed")             # the save point does not
    assert cp["status"] == "corrupted"
    assert cp["changed_files"][0]["path"] == "mri/T1.mgz"
    assert "checkpoint:sealed" in audit.audit(mf)["corrupted"]


def test_deleting_a_stored_file_is_caught(mf, subject):
    rep = reconall.store_subject(mf, subject)
    checkpoint.create_checkpoint(mf, "sealed")
    mf.conn.execute("DELETE FROM files WHERE relpath='surf/lh.white' AND item_id=?",
                    (rep["item_id"],))
    mf.conn.commit()
    cp = checkpoint.verify_checkpoint(mf, "sealed")
    assert cp["status"] == "corrupted"
    assert cp["changed_files"][0]["issue"] == "missing"
    assert reconall.verify_tree(mf, rep["item_id"])["status"] == "corrupted"


def test_growth_does_not_invalidate_an_earlier_checkpoint(mf, subject, tmp_path):
    reconall.store_subject(mf, subject, subject_id="s1")
    checkpoint.create_checkpoint(mf, "first")
    other = tmp_path / "s2"
    other.mkdir()
    (other / "notes.txt").write_text("second subject")
    reconall.store_subject(mf, other, subject_id="s2")
    assert checkpoint.verify_checkpoint(mf, "first")["status"] == "ok"
    assert audit.audit(mf)["status"] == "ok"


# -- analysis layer ----------------------------------------------------------
def test_stats_parse_into_a_fixed_dimension_morphometry_vector(mf, subject):
    rep = reconall.store_subject(mf, subject)
    stats = reconall.read_stats(mf, rep["item_id"])
    assert stats["aseg"]["measures"]["eTIV"] == 1700000.0
    assert stats["aseg"]["structures"]["Left-Hippocampus"] == 4200.0
    assert stats["lh"]["regions"]["cuneus"]["thickness"] == 2.539

    vec = reconall.morphometry_vector(stats)
    assert vec.shape == (reconall.MORPHOMETRY_DIM,)
    # volumes normalised by eTIV land in a sane range; thickness stays in mm
    assert 0 < vec[reconall.ASEG_MEASURES.index("CortexVol")] < 1
    lh_cuneus = (len(reconall.ASEG_MEASURES) + len(reconall.ASEG_STRUCTURES)
                 + reconall.DK_REGIONS.index("cuneus"))
    assert vec[lh_cuneus] == pytest.approx(2.539, abs=1e-3)


def test_partial_tree_still_yields_a_comparable_vector():
    vec = reconall.morphometry_vector({"aseg": None, "lh": None, "rh": None})
    assert vec.shape == (reconall.MORPHOMETRY_DIM,)
    assert not vec.any()


def test_modality_ingest_registers_the_subject_and_a_card(mf, subject):
    item = modality.ingest(mf, "reconall", subject, subject_id="s1")
    assert mf.conn.execute(
        "SELECT modality FROM items WHERE id=?", (item,)).fetchone()[0] == "reconall"
    vec = mf.conn.execute(
        "SELECT 1 FROM vectors v JOIN spaces s ON s.id=v.space_id "
        "WHERE v.item_id=? AND s.name='morphometry'", (item,)).fetchone()
    assert vec is not None
    text = render.card(mf, item)
    assert "recon-all" in text and "eTIV" in text


def test_reconall_is_in_the_catalog():
    cat = {c["modality"]: c for c in modality.catalog()}
    assert "reconall" in cat
    assert cat["reconall"]["spaces"] == ["morphometry"]
    assert cat["reconall"]["card"] is True
