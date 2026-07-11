# `.gnos` — a verifiable, AI-native, multimodal file format

**A `.gnos` file is one self-contained, signed, LLM-readable multimodal record
that detects any tampering and carries its own chain of custody.** Seal it once
(content-hashed checkpoint + Ed25519 signature over the whole state), and a single
`audit()` call afterwards catches *and localizes* every realistic attack:

```
$ python examples/verifiable_patient_bundle.py
[1] one self-contained file: 6 records across 8 embedding spaces, 1 lossless waveform
[2] authenticity: signed by key 598e623f.. | trusted=True | matches current state=True
[4] ADVERSARY SUITE - each attack on a copy, caught by one audit() call:
    CAUGHT | edit a lab/report payload byte              -> integrity, checkpoint:issued
    CAUGHT | edit a payload AND recompute its inline hash -> checkpoint:issued
    CAUGHT | flip one bit inside a stored EEG waveform    -> signals, checkpoint:issued, deep
    CAUGHT | tamper an embedding vector                   -> integrity, checkpoint:issued
    CAUGHT | delete a provenance (history) row            -> history
    CAUGHT | forge / corrupt the signature               -> signatures
[6] substitute a referenced patient file -> verify_refs: uuid_mismatch
ALL ATTACKS CAUGHT.
```

This is the proven, differentiated capability — see **[docs/INTEGRITY.md](docs/INTEGRITY.md)**
for the guarantees, threat model, and honest limits. On top of that integrity
core, the file *also* carries **an evolving semantic map of itself** — embeddings
(in several views), clusters, a relationship graph, an append-only history of how
its understanding changed, and immutable save points.

It is **inspired by Turso**: the file *is* a [libSQL](https://github.com/tursodatabase/libsql)/SQLite
database, so you get ACID durability, SQL queries, and (optionally) native vector
search for free — and you can open a `.gnos` file in any SQLite tool.

```
normal file                         AI-native .gnos file
-----------                         --------------------
brain_scan.nii.gz                   patient_cohort.gnos
   |                                   |-- data + identity      (items: subject/session/modality/time)
   +-- voxels only                     |-- embeddings, N VIEWS  (vectors per space)
                                        |-- clusters per space   (clusterings/clusters)
                                        |-- relationships        (edges: knn/temporal/same_subject)
                                        |-- history of meaning   (history)
                                        |-- save points          (checkpoints)
                                        +-- self-update policy    (agents)
```

## Several ways of embedding — spaces

An item can carry **multiple embeddings at once**, each in a named *space* with
its own dimension and embedder (e.g. EEG `bandpower` + `connectivity` + a learned
`encoder`). Each space clusters and graphs independently.

```python
ids = ingest.add_vectors(mf, bandpower_vecs, space="bandpower", subject_ids=subjects)
ingest.add_space_vectors(mf, ids, connectivity_vecs, space="connectivity")  # second view
cluster.run_clustering(mf, "bandpower", k=3)     # versions are per-space
cluster.run_clustering(mf, "connectivity", k=2)
```

## Per-modality processing protocols

Each health-data type has a `ModalityProcessor` that owns its pipeline — how to
ingest it, how to store it, and which embedding space(s) it produces — so
multimodal ingest uses each type to its strengths and fusion composes across them:

```python
modality.ingest(mf, "eeg",      edf_or_array, sfreq=500, subject_id="P1")  # waveform + lazy band-power
modality.ingest(mf, "ecg",      ecg_array,    sfreq=250, subject_id="P1")  # waveform + spectral (ECG bands)
modality.ingest(mf, "mri",      nifti_or_vol,            subject_id="P1")  # radiomics features + x,y,z axes
modality.ingest(mf, "text",     "clinical note...",      subject_id="P1")  # text embedding
modality.ingest(mf, "genomics", ["1:100:A:G", ...],      subject_id="P1")  # hashed variant vector
modality.ingest(mf, "fhir",     observation_json,        subject_id="P1")  # FHIR resource
modality.ingest(mf, "labs",     {"crp": 12, ...},        subject_id="P1")  # values + reference-range flags
modality.catalog()    # every supported file type: spaces, card, extensions, deps
```

**10 file types**, one uniform contract — `eeg` (EDF/BDF/FIF), `ecg`, `wfdb`,
`mri` (NIfTI), `dicom`, `fhir`, `text`, `genomics`, `labs`, `tabular`. Each stores
its raw record verifiably, produces embedding space(s), and renders an LLM card.
They all compose in one file (`examples/all_modalities_demo.py`); run
`mind modalities` for the live catalog. Full table: [docs/FILE_TYPES.md](docs/FILE_TYPES.md).

Shipped processors are **dependency-light baseline representations** (the heavy
domain pipelines — full radiomics, learned encoders, sentence-transformers — are
optional upgrades that plug into the same protocol).

## Same items, three analysis scopes

Items carry identity (`subject_id`, `session_id`, `modality`, `study_id`,
`acquired_at`), so the *same recordings* can be analysed three ways via
`collections` (named subsets) + spaces:

```python
# GROUP study — cluster a cohort (cross-sectional)
analyze.group_study(mf, "eeg-cohort", where={"study_id": "S1", "modality": "eeg"},
                    space="bandpower", k=4)

# LONGITUDINAL — one patient's sessions over time -> trajectory + change detection
analyze.longitudinal(mf, "P1", space="bandpower", change_threshold=0.4)
#   -> ordered sessions, cosine distance per step, flagged changes, temporal edges

# MULTI-MODAL — one patient across modalities -> same_subject links
analyze.multimodal(mf, "P1")   # links the patient's EEG + MRI + labs items
```

## Two layers of safety: corruption + validation

**Corruption (byte integrity).** `items` and `vectors` are **write-once** —
ingesting data, attaching a new space, or re-clustering only ever *adds* rows.
Every payload, vector, waveform, and recipe is sha256-hashed;
`create_checkpoint("original")` freezes a **complete** manifest — payloads,
per-space vectors, waveforms, and virtual-vector recipes — plus a Merkle root.
`verify_checkpoint` re-hashes the live data and proves the *whole file* is
unchanged since the snapshot (it even catches a waveform swapped for a different
valid one).

```python
checkpoint.create_checkpoint(mf, "baseline")
# ... add patients, attach a connectivity space, re-cluster ...
checkpoint.verify_checkpoint(mf, "baseline")
# -> {'status': 'ok', 'changed': [], 'changed_vectors': [], 'missing': []}
```

**Tamper-evident provenance.** The append-only `history` log is **hash-chained**:
each entry hash-links to the previous one, so any silent edit, deletion,
insertion, or reorder of the audit trail breaks the chain. `verify_history`
detects it, and the head hash is pinned in `mind_meta` (export it to anchor
against a full rewrite).

**Authenticity (who attested).** On top of integrity, you can **sign** the
file's *state root* (uuid + schema + history head + latest checkpoint root).
`Ed25519Signer` gives public-key authenticity — sign with a private key, anyone
verifies with the public key stored in the file, nobody can forge; `HmacSigner`
is a stdlib fallback. `verify_signatures` reports whether each signature is valid
and whether the file is still in the signed state.

```python
sign.sign_state(mf, sign.Ed25519Signer())   # attest to the current state
sign.verify_signatures(mf)                   # valid? still the signed state?
```

**Cross-file trust.** A study/registry `.gnos` can reference many other `.gnos`
files (e.g. signed patient files), pinning each one's uuid + state root +
signing key. `verify_refs` re-opens each **read-only** and flags any that went
`missing`, was substituted (`uuid_mismatch`), drifted since registered
(`state_changed`), or lost its valid signature (`signature_invalid`).

```python
registry.link_file(study, "P1.gnos", role="patient", require_signed=True)
registry.verify_refs(study)   # {'status': 'ok'|'issues', 'refs': [...]}
```

**One call to verify everything.** Integrity is the format's first priority, so
`audit(mf)` runs *every* check — live-hash integrity, save points, fusion
provenance, lazy recipes, waveform reconstruction, the history chain, signatures,
and semantic validation — into a single verdict (`ok` / `issues` / `corrupted`). `audit(mf, deep=True)` also
resolves every virtual vector and reconstructs every waveform to prove the
recipes/codecs produce well-formed output.

```python
audit.audit(mf)   # {'status': 'ok', 'corrupted': [], 'checks': {...}}
```

**Validation (semantic/structural).** `validate(mf)` answers a different
question — *is the file well-formed and analysis-ready?*: per-space dimension
consistency, no NaN/Inf, every item embedded, no duplicate `(subject, session,
modality)`, referential integrity, and identity completeness for longitudinal
series.

```python
validate.validate(mf)   # -> {'status': 'ok'|'issues', 'errors': n, 'warnings': n, 'issues': [...]}
```

## Constellations — position-aware vectorization (4D)

Items can carry coordinate **axes** (`x, y, z, t` — your "4D value"). A
constellation enriches each item's semantic vector with its standardized
coordinates into a *position-aware* space, then connects a patient's points into
a graph across space and time. It's additive — a new space on the same items, so
originals stay intact.

```python
axes.set_axes_bulk(mf, item_ids, coords, ["x", "y", "z", "t"])
constellation.build_constellation_space(mf, "bandpower")   # -> "constellation" space = [L2(sem) ⊕ z(coords)]
constellation.build_constellation(mf, "P1")                # per-patient collection + graph
```

## The file manages itself — Planner → Reviewer → Renderer

An in-file, **deterministic** 3-agent loop (RecursiveMAS Tier-1 *Sequential*)
decides what the file should materialize for the clusters that matter, and does
it — logging every step in `plans` / `plan_actions`.

```python
allocate.orchestrate(mf)
# PLANNER proposes (cluster / knn / fuse / constellation / correlate / vacuum),
# REVIEWER approves under guards (write-once, min-vectors, size budget, validate clean),
# RENDERER executes + logs.  Run again and the file self-converges.
```

It computes **correlations inside the file** (`correlations` table): clustering
agreement (ARI), cross-space agreement, within-space feature correlation, and
subject cross-modality — e.g. *do the EEG and MRI views tell the same story?*

## Fast summary for AI — without unfolding the file

The flip side of all the laziness: an AI system can read a **cheap, resident
summary** of the file without resolving a single lazy vector or decompressing a
single waveform. Clustering already leaves centroids/labels/sizes resident, so
the summary is assembled from small tables only.

```python
summary.summary(mf)        # meta, counts, spaces+clusters, correlations, signals, ... (unfolds nothing)
summary.summary_text(mf)   # a few lines an agent can read at a glance
summary.file_vector(mf)    # ONE size-weighted centroid vector to place/compare the whole file
```

```
$ mind summary cohort.gnos
eeg .gnos (0.7.0) - 6 items, 6 subjects, 6 waveforms
  space bandpower dim10 [6]: cluster 0 x5, cluster 1 x1
```

## Made to be read by LLMs

An LLM never parses the raw binary. Every item renders to a compact **text card**
(normalized facts), and the whole file renders to a **digest** an LLM consumes
directly — while the raw bytes stay stored and integrity‑verified.

```python
render.card(mf, item_id)   # "[eeg] P1/t0: 19ch 10s@250Hz; rel power beta 0.38, gamma 0.33..."
render.digest(mf)          # file summary + one card per record
```

```
$ mind digest cohort.gnos --subject P1
- [eeg]      P1/t0: 19ch 10s@250Hz; rel power beta 0.38, gamma 0.33, alpha 0.10
- [ecg]      P1/t0: 1ch 10s@250Hz; rel power high 0.61, qrs 0.25, low 0.09
- [mri]      P1/t0: volume; intensity mean 0.50, entropy 5.31; centroid (4.5,4.4,4.5)
- [text]     P1/t0: 58M acute onset right-sided weakness and dysarthria; CT no hemorrhage
- [labs]     P1: glucose 7.8, crp 12, troponin 0.9
- [genomics] P1: 3 variants: 1:100:A:G, 7:55:G:A, X:12:C:T
```

See **[docs/medical_formats.md](docs/medical_formats.md)** for the full review of
medical data formats (DICOM, NIfTI, EDF/WFDB, FHIR, VCF/BAM, PDF, …) → the best
ingestion + LLM‑rendering approach for each.

## A living correlation surface (findings) + retrieval

Each data type is a "diamond" — already well‑structured. The file holds them
intact and **creates connections between them without changing the data**:
`findings.discover(mf)` scans for cross‑structure correlations and records them as
plain‑English statements an external LLM reasons over (the LLM stays outside the
file — it interprets; the file finds).

```python
findings.discover(mf)
#  - bandpower cluster 1: high beta, high delta
#  - labs cluster 1: high crp, high troponin
#  - bandpower and labs cluster structures agree across 8 subjects (ARI=1.00)
```

**Retrieval** turns a question into an LLM context bundle — real semantic search
(sentence‑transformers) over the cards, plus the discovered findings:

```python
from mindfile.embedders import SentenceTransformerEmbedder
emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")           # or CallableEmbedder(your_api_fn, dim)
proc = modality.TextProcessor(embedder=emb)                    # production embeddings, pluggable
retrieve.context(mf, "cardiac source of stroke", emb)          # summary + top records + findings, for an LLM
```

The allocator proposes `findings` as data grows, so the correlation surface stays
**live**. `mind findings --discover` / `mind retrieve --query "…"` from the CLI.

## Waveforms collapse on call too

An EEG/MEG recording is a `(channels × time)` grid — bulky but very compressible.
A signal-aware **lossless** codec (channel byte-shuffle + zlib, with integer
delta-along-time) stores the waveform **in the file** and reconstructs it
**byte-exact** on call. The waveform becomes the compact source of truth, and the
band-power vector is a lazy recipe recomputed from it on demand:

```python
signal.add_recording(mf, data, sfreq=500, ch_names=ch, subject_id="P1", embedder=EEGEmbedder())
#   -> stores the waveform losslessly + a virtual 'bandpower' vector
lazy.resolve(mf, item_id, "bandpower")   # recomputes band-power FROM the stored signal
signal.load_signal(mf, item_id)          # exact waveform back; verify_signals() checks all
```

## Lazy / "dynamic" derived values

Derived vectors (fused, constellation) can be stored as **recipes** instead of
bytes — a *virtual* value that resolves only when called, then optionally caches.
Recipes are deterministic, so a resolved vector is byte-identical to the eager
one and stays verifiable (the recipe's source hashes are checked by
`verify_defs`). Materialize to cache; dematerialize to reclaim space.

```python
fuse.fuse(mf, ["bandpower", "mri"], lazy=True)   # stores recipes, not fused bytes
lazy.resolve(mf, item_id, "fused")               # computes on call (collapses)
lazy.materialize_space(mf, "fused")              # cache all; dematerialize() frees them
```

You can also store a materialized derived vector as a **lossless compressed
code** (a "shorter value") that reconstructs byte-identically on access:

```python
lazy.compact_space(mf, "fused")     # zlib-code the derived vectors; resolve() decompresses exactly
# refuses primary spaces -> originals stay exact and verifiable
```

## File type

Files use the **`.gnos`** extension, but identity doesn't depend on the name: every
file is stamped with an in-content magic number (SQLite `application_id` = `GNOS`),
so `mindfile.is_mindfile(path)` recognizes it by content even if it's renamed —
and tools won't confuse it with the unrelated `.mind` mind-map files.

```python
from mindfile import is_mindfile
is_mindfile("patient.gnos")   # True — detected by content, not extension
```

## Install

```bash
pip install -e .                 # core: numpy + scipy + scikit-learn
pip install -e ".[auth]"         # + Ed25519 signatures (cryptography)
pip install -e ".[eeg]"          # + EDF/BDF/FIF ingest (mne)
pip install -e ".[imaging]"      # + NIfTI volumes + DICOM files/pixels (nibabel, pydicom)
pip install -e ".[waveforms]"    # + PhysioNet WFDB records (wfdb)
pip install -e ".[text]"         # + production text embeddings (sentence-transformers)
pip install -e ".[all,dev]"      # every reader + pytest
```

Requires Python ≥ 3.10. A `.gnos` file is a plain SQLite database — open it with
any SQLite tool. No server, no libSQL build required (a libSQL build with native
vectors is an optional acceleration, not a dependency).

## Quickstart

```bash
python examples/multimodal_demo.py  # CAPSTONE: 8 patients x 5 modalities -> fused subtyping + audit
python examples/patient_demo.py     # spaces + 3 scopes + validate + save point
python examples/synthetic_demo.py   # the "cluster splits over time" narrative
python examples/eeg_demo.py --limit 12   # real EDF data (needs `mne`)
```

### CLI

```bash
mind create scans.gnos --modality eeg
mind ingest scans.gnos --npy vectors.npy --ids ids.txt --space bandpower
mind cluster scans.gnos --space bandpower --k 3
mind checkpoint scans.gnos baseline
mind evolve scans.gnos --space bandpower --k 4    # records splits/merges
mind fuse scans.gnos --spaces bandpower,mri      # additive multimodal fusion
mind allocate scans.gnos                          # run Planner->Reviewer->Renderer
mind verify scans.gnos baseline    # byte integrity (exit 0 ok / 1 corrupted)
mind derivations scans.gnos        # prove fused points are backed by intact originals
mind validate scans.gnos           # semantic/structural checks
mind audit scans.gnos [--deep]     # ALL integrity + validation in one verdict
mind sign scans.gnos --keyfile lab.key   # Ed25519-attest the current state
mind signatures scans.gnos         # verify authenticity (public key in file)
mind link study.gnos P1.gnos --role patient --require-signed   # reference another file
mind refs study.gnos               # verify all referenced files
mind summary scans.gnos            # fast resident overview (unfolds nothing)
mind inspect scans.gnos            # meta, spaces, collections, correlations, save points, ...
```

## Architecture

```
src/mindfile/
  file.py          MindFile: open/create, header meta, append-only history
  schema.sql       canonical v2 schema (runtime source of truth)
  spaces.py        named embedding spaces (the "several ways of embedding")
  collections.py   named item subsets that scope an analysis
  embedders/       Embedder protocol; HashEmbedder (synthetic), EEGEmbedder (real)
  signal.py        lossless waveform store + lazy band-power features
  lazy.py          virtual derived vectors: resolve/materialize/compact
  summary.py       fast resident file summary for AI (unfolds nothing)
  ingest.py        write-once add_items / add_vectors / add_space_vectors (+ hashing + identity)
  index.py         per-space vector search (NumPy; libSQL ANN upgrade-ready)
  cluster.py       run_clustering -> immutable version, scoped to space (+ collection)
  evolve.py        recluster + split/merge lineage detection
  graph.py         kNN / temporal / same_subject edges
  query.py         search / cluster_of / explain
  analyze.py       group_study / longitudinal / multimodal scopes
  axes.py          per-item coordinates (x, y, z, t)
  constellation.py position-aware space + per-patient constellation graph
  fuse.py          additive multimodal fusion + provenance verification
  stats.py         in-file correlations (ARI / cross-space / feature / cross-modality)
  allocate.py      Planner -> Reviewer -> Renderer self-allocation
  checkpoint.py    multi-space save points + verification
  validate.py      semantic/structural validation
  agents.py        in-file self-update policies (Turso-registry style)
  cli.py           the `mind` command
```

The 3 allocation specialists are also mirrored as a registry scaffold under
`agents/` (planner / reviewer / renderer), per the project convention.

See `spec/SPEC.md` for the full schema and invariants.

## libSQL / Turso note

Embeddings are stored as portable float32 BLOBs and searched with an exact NumPy
cosine scan (fine at MVP scale). On a libSQL build with native vector support,
the `embedding` column migrates to `F32_BLOB(dim)` with a `libsql_vector_idx` ANN
index — **without changing the file format**. The file stays a valid SQLite/Turso
database throughout.

## License

MIT — see [LICENSE](LICENSE).

## Status & roadmap

**v1.0** — feature-complete reference implementation: a 79-test suite (`pytest`),
7 runnable demos (incl. an 8-patient × 5-modality capstone), real EEG. Capabilities: multi-space embeddings; identity +
collections; group / longitudinal / multi-modal analyses; per-space clustering +
split/merge evolution; additive multimodal fusion with provenance; position-aware
constellations (x,y,z,t); in-file correlations; a deterministic
Planner→Reviewer→Renderer self-allocator; **IVF approximate search** (reuses the
file's own clustering); lazy/virtual derived vectors + lossless compaction; a
lossless in-file EEG waveform store (+ opt-in lossy spectral codec) with
collapse-on-call features; a fast resident AI summary; and a full integrity +
authenticity stack — complete save points, tamper-evident hash-chained history,
Ed25519/HMAC signatures with **trusted-key pinning**, a cross-file registry, and
a one-call `audit`.

Next (Phase 2, optimizations — not gaps in the guarantee): native libSQL HNSW
backend (a drop-in for the IVF search); truly-lossless int16 EDF path; an
external anchor/notary for `history_head`; more domain embedders (MRI/NIfTI, text).
```
