# Storing a FreeSurfer `recon-all` subject in one `.gnos`

A `recon-all` subject directory is a few hundred megabytes to a gigabyte spread
over hundreds of files. Generic archivers do poorly on it, because most of the
bulk is *already gzipped* (`.mgz`) — recompressing compressed bytes gains almost
nothing. The win comes from understanding what those bytes are.

```python
from mindfile import MindFile, modality, reconall

mf = MindFile.create("subject.gnos", modality="reconall")
item = modality.ingest(mf, "reconall", "/subjects/sub-01", subject_id="sub-01")

reconall.restore_subject(mf, item, "/tmp/sub-01")     # byte-identical tree
reconall.verify_files(mf, deep=True)                  # every file still exact
```

```bash
mind recon-store   subject.gnos /subjects/sub-01
mind recon-restore subject.gnos /tmp/sub-01 --item 1
mind recon-verify  subject.gnos --item 1 --deep
```

## The rule: no codec is trusted

Every candidate encoding is **decoded again at store time** and its SHA-256
compared with the original file's. Only a candidate that reproduces the file
byte-for-byte may be chosen; otherwise the next one is tried, ending at raw
bytes. A bug in a parser can therefore cost compression ratio and nothing else.

Two hashes are kept per file: `raw_sha256` (the original file's bytes) and
`code_sha256` (what is actually stored). The cheap integrity pass re-hashes
`code` — catching in-place corruption of a gigabyte for the price of a hash —
and `deep=True` decodes everything and checks against `raw_sha256`.

## The codecs

| Codec | Applies to | What it does |
|---|---|---|
| `gz+…` | `.mgz`, `.nii.gz`, `.m3z` | Unwraps the gzip member and stores the *payload* under one of the codecs below, plus the zlib settings and header bytes needed to re-emit the original stream exactly. Falls back if the stream can't be reproduced |
| `mgh:pred:x` / `:yzmed` | integer MGH volumes | Predicts each voxel from already-decoded neighbours and LZMA-codes the residual. `x` is a raster delta; `yzmed` is a median-edge predictor over the y/z neighbours — deliberately *not* using the x neighbour, so decoding stays a vector operation per row instead of a Python loop per voxel |
| `mgh:plain[:shuf]` | float MGH volumes | Plain LZMA, optionally byte-plane shuffled. Measured: prediction *hurts* on float32 warp/parameter maps, so floats are not predicted |
| `mgh:delta` | volumes mirroring another volume | Modular difference against the most similar already-stored volume of the same shape/dtype |
| `surf` / `surf:delta` | `?h.white`, `?h.pial`, … | Splits coordinates from faces; a surface sharing another's topology is stored as a coordinate delta against it |
| `curv` | `?h.thickness`, `?h.curv`, … | Byte-plane shuffle + LZMA over the per-vertex float32 values |
| `xz` / `store` | everything else | Plain LZMA, or verbatim bytes |

Deltas are **depth-1**: only a file stored under a non-delta codec can be a
reference, so a restore never unwinds a chain.

## Solid blocks — cross-file redundancy where no structural codec fits

Some files only compress well *together*. After the pass above, any file that
ended up on a generic codec (`xz`, `gz+xz`, `store`) becomes a candidate for a
**solid block**: several files coded as one LZMA stream, each member recording
its `(offset, length)` inside the decompressed block.

This is what a solid `.7z` does, and it is where 7-Zip beat per-file coding.
Measured on the real subject:

| group | per-file | solid | gain |
|---|---:|---:|---:|
| `label/` (37 files) | 4.104 MB | 3.248 MB | 1.26x |
| `bem/` small (15 files, 2.0 MB) | 0.688 MB | 0.323 MB | 0.365 MB |
| one large near-duplicate pair | 16.411 MB | 8.910 MB | **7.5 MB** |
| `stats/`, `touch/` | 0.028 MB | 0.013 MB | 1.8–4.0x |
| `surf/` curv files | 21.97 MB | 22.32 MB | **worse — not blocked** |

### The dictionary has to span the block

That 7.5 MB pair gain is **entirely conditional on the LZMA window**. The same
two files, coded together:

| dictionary | solo | together | solidity saves |
|---|---:|---:|---:|
| 8 MB (LZMA preset 6 default) | 16.424 MB | 16.414 MB | **0.010 MB** |
| 64 MB | 16.411 MB | 8.910 MB | **7.501 MB** |

They are ~18 MB each, so with an 8 MB window the second copy's match is simply
out of reach and blocking accomplishes nothing. Blocks therefore compress with
`_solid_dict_size()` — a power-of-two window covering the whole block, capped at
256 MB — and record it in `file_blocks.dict_size`, because a raw LZMA2 stream
has no header and the decoder must be handed the same value. The same trap
applies to the pairing probe: at preset 1's 1 MB window it scored this pair as
*negative* and rejected it.

Raising the window for ordinary single files is **not** worth it: on the largest
file in this tree it bought 2.17 MB for 394 s. Only blocks get the big window.

Two more lessons drove the grouping policy. The gain is not where it looks —
pooling many small files helps, but one pair of large near-duplicates outweighs
every small file in `bem/` combined. And solidity does nothing for per-vertex
float arrays (curv files got *bigger*), so they are left alone.

Blocks are therefore **bounded, and bounded differently by size**:

* files under `BLOCK_LARGE_BYTES` (4 MB) are pooled by (top-level directory,
  suffix), capped at `BLOCK_MAX_BYTES` (16 MB) and `BLOCK_MAX_MEMBERS` (24);
* files at or above that size are only ever **paired** — a 2-member block —
  chosen by probing the compression gain on their heads.

A block is adopted only when it is measurably smaller than coding its members
independently, and only after the block is decoded again and *every* member's
SHA-256 re-checked. As everywhere else here, blocking can cost ratio but never
fidelity.

### What blocking costs

**Corruption localisation.** Detection is unaffected — each member keeps its
`raw_sha256`, and the block's `code_sha256` catches any tamper. But the blast
radius grows: a corrupt block reports *all* of its members as bad, and
`verify_files` says so explicitly (`issue: "block"`). This is the same exposure a
depth-1 delta already carries (corrupt the reference and its dependent fails
too), which is why blocks are kept small rather than per-directory.

**Random access.** Reading one member decompresses its whole block. With the
caps above that is a few hundred milliseconds, not a whole directory.
`restore_subject` decodes each block once and serves all its members from it.

**A VACUUM at the end.** Blocking replaces per-file codes with one shared code,
which frees SQLite pages but does not return them to the filesystem — the first
working block run shrank the stored content yet grew the *file* by 0.8 MB. When
any block is formed, `store_subject` reclaims the freed pages before returning.

Checkpoints handle members correctly: the frozen `code_sha256` for a block
member is the *block's* hash, so tamper inside a block breaks the save point for
every file it carries.

### How a delta reference is chosen

Trying every stored volume as a reference would be far too slow, so each volume
carries a scattered 8192-voxel signature and the two most similar candidates are
probed. The sample positions are deliberately *not* a plain stride: a stride
that divides the slice width lands on the same (row, column) of every slice,
which in a conformed brain volume means sampling only the empty background —
every pair of volumes then scores as identical and the choice is meaningless.

For volumes above 4 MB the candidate codecs are ranked on a strided subsample at
a fast LZMA preset, and only the winner is run over the full volume. This is a
size/time trade: a subsample can occasionally mis-rank two close candidates, and
`effort="max"` compresses every candidate in full instead (`"fast"` skips the
ranking entirely).

## Profiles

`profile="lossless"` (default) stores every file; the tree restores bit-for-bit.

`profile="analysis"` keeps only the non-regenerable, analysis-bearing outputs
(`reconall.ANALYSIS_KEEP`: stats, labels, the main surfaces and per-vertex
metrics, the segmentation and intensity volumes, the talairach transform) and
drops the rest. **Omitted files are still listed in the manifest with their size
and SHA-256**, so the bundle records exactly what was dropped, and the tree
Merkle root still covers them — an external copy can be checked against it
later. The restore of an `analysis` bundle is partial by construction.

## What it produces besides bytes

The stats tables recon-all already computed are parsed into a fixed-dimension
`morphometry` vector (aseg volumes normalised by eTIV, plus Desikan-Killiany
cortical thickness per hemisphere), so subjects are comparable and searchable
without unpacking a single volume. `render.card()` renders the subject as one
line of clinical text (eTIV, cortical grey volume, hippocampal volumes, mean
thickness per hemisphere).

## Integrity

`files` participates in the same guarantees as the rest of the format:

* `verify_files(mf)` / `verify_files(mf, deep=True)` — stored codes and full
  reconstruction;
* `verify_tree(mf, item)` — the file *set* against the manifest's Merkle root,
  so a deleted or added file is caught, not just a modified one;
* `checkpoint_files` — a save point freezes every file's `raw_sha256` and
  `code_sha256`, so tamper that also rewrites the local hash is still caught;
* `audit(mf)` includes `files` in its verdict and `audit(mf, deep=True)` decodes
  the whole tree.

## Measured on a real subject

`examples/reconall_demo.py` run against the FreeSurfer subject shipped with
MNE's sample dataset (274 files, 1059.0 MB — a real recon-all tree plus the
BEM/FLASH artifacts MNE adds to it):

| Directory | Original | Stored | Ratio |
|---|---:|---:|---:|
| `bem/` (MNE-added, not recon-all) | 521.2 MB | 265.1 MB | 1.97x |
| `mri/` | 341.4 MB | 219.1 MB | 1.56x |
| `surf/` | 169.2 MB | 63.9 MB | 2.65x |
| `label/` | 26.4 MB | 4.1 MB | 6.44x |
| `scripts/` | 0.7 MB | 0.1 MB | 6.91x |
| `stats/` | 0.1 MB | 0.0 MB | 4.47x |
| **Total (.gnos on disk)** | **1059.0 MB** | **539.1 MB** | **1.96x** |

18 solid blocks account for 15.4 MB of that. The `analysis` profile keeps 84 of
the 274 files and lands at **41.0 MB — 25.8x** — with the 190 omitted files still
listed with their hashes.

For comparison, on the same tree: `zip -mx5` 720.6 MB (1.47x), `zip`/LZMA
663.2 MB (1.60x), `7z -mx=9` solid 614.8 MB (1.72x), `rar -m5 -s` 616.1 MB
(1.72x). The `.gnos` is ~12% smaller than the best of them — but 7-Zip gets to
1.72x in 175 s, where this takes ~16 min. Recompressing the finished `.gnos`
with 7z gains 0.6%: its contents are already LZMA-coded.

Storing took 16.2 min; restoring all 274 files took 130 s; `audit(deep=True)`
over the whole tree took 133 s. Every file was restored and re-hashed
independently of the library: **identical**. A single flipped bit in the largest
stored blob was caught by `audit` and pinpointed to its exact path.

Where the ratio comes from is uneven, and worth knowing before you predict a
number for your own data:

* near-duplicate volumes collapse almost completely — `rawavg.mgz` 11.41 MB → 0.00 MB
  (it duplicates `orig/001.mgz`), `flash30.mgz` 11.73 MB → 0.02 MB, `brain.mgz`
  1.47 MB → 0.02 MB. 14 of the 37 volumes ended up as deltas;
* surfaces delta-code well — 25 of 32 used `surf:delta`, e.g. `?h.orig.nofix`
  5.68 MB → 0.71 MB;
* text labels compress ~6x;
* **float32 parameter and warp maps barely move** — `sse.mgz` 44.54 → 40.08 MB,
  `PD.mgz` 40.95 → 34.41 MB, `talairach.m3z` 32.05 → 30.23 MB. These dominate
  `mri/` here and are what hold the total near 1.9x.

## Honest limits

* The ratio depends heavily on what the tree contains. Integer intensity and
  label volumes compress well; float32 parameter maps and non-FreeSurfer
  artifacts sitting in the subject directory barely move.
* Storing is much slower than reading — it is an archival step (LZMA at preset
  6, plus verification of each candidate), not something to put in a hot loop.
* `mgh:pred:yzmed` decodes with a Python loop over rows. It is bounded and
  vectorised across each row, but restoring a large tree is dominated by it.
* Re-emitting a gzip stream exactly requires that it was produced by zlib with
  settings the encoder searches. FreeSurfer's own `.mgz` files match (level 6);
  a stream from another deflate implementation falls back to a generic codec,
  costing ratio only.
* File *contents* are preserved exactly; POSIX metadata (mtime, permissions,
  symlinks) is not. Empty directories are recorded and recreated, but a symlink
  in the tree is followed and stored as a regular file.
* One tree per item. Storing a second tree into an item that already holds one
  is refused, because replacing rows would renumber them and orphan any delta
  that referenced them.
