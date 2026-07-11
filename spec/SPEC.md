# The `.gnos` format specification (v1.1.0, schema v14)

A `.gnos` file is an ordinary **SQLite** database (byte-compatible with
libSQL/Turso). The canonical schema is `src/mindfile/schema.sql`; this document
explains the blocks and the invariants a conforming reader/writer must uphold.

## Blocks (tables)

| Block | Table(s) | Purpose |
|-------|----------|---------|
| Header | `mind_meta` | `format_name/version`, `schema_version`, `file_uuid`, `created_at`, `modality` |
| Spaces | `spaces` | named embedding views; each has `name`, `dim`, `embedder_id`, `modality` |
| Data | `items` | payloads + identity (`subject_id`, `session_id`, `study_id`, `modality`, `acquired_at`) + `origin` (primary/derived) + `payload_sha256` |
| Vector | `vectors` | one embedding per `(item_id, space_id)` (float32 BLOB) + `vec_sha256` |
| Axes | `item_axes` | per-item coordinates `(x, y, z, t ...)` feeding the constellation space |
| Collections | `collections`, `collection_items` | named item subsets (cohort / longitudinal / multimodal / constellation) |
| Cluster | `clusterings`, `clusters`, `cluster_assignments` | clustering runs, versioned per space; clusters carry centroid/label/size + lineage |
| Knowledge | `edges` | graph: `knn` (per space) / `temporal` / `same_subject` / `fused_from` / `constellation` / `manual` |
| Derivations | `derivations`, `derivation_sources` | provenance for derived/fused items (source item+space+hash) |
| Lazy vectors | `vector_defs` | recipes for virtual (un-materialized) derived vectors, resolved on access |
| Compact codes | `vector_codes` | lossless compressed forms of derived vectors (kind `compressed`); reconstruct byte-identically. Only `spaces.derived=1` may be compacted |
| Signals | `signals` | lossless waveform store (channels x time); reconstructs byte-exact on call; band-power vectors are lazy `signal_feature` recipes recomputed from it. The `digital-cal` codec stores an EDF's exact **int16 digital** samples plus per-channel SI calibration (embedded in `code`, so `code_sha256` covers it) and reconstructs physical volts on call — more faithful to the source file and far smaller than storing pre-scaled floats |
| Events | `events` | task/stimulus markers (`onset` [s], `duration`, `condition`, `value`, `label`) attached to a signal recording, so event-locked dynamics (baseline-normalised **ERD**, hemispheric lateralisation) can be computed from the stored waveform on call |
| Allocation | `plans`, `plan_actions` | Planner→Reviewer→Renderer log (proposed/reviewed/rendered) |
| Correlations | `correlations` | in-file statistics (ARI / cross-space / feature / subject-cross-modality) |
| Signatures | `signatures` | Ed25519/HMAC signatures over the `state_root` (uuid+schema+history head+latest checkpoint root) — authenticity on top of integrity |
| Trusted keys | `trusted_keys` | pinned/expected signing keys; "valid" attestation means a current signature from one of these |
| File refs | `file_refs` | cross-file registry: pins another `.gnos`'s uuid + state_root + signing key; `verify_refs` detects missing/substituted/drifted/unsigned |
| Agent | `agents` | self-update policies + the 3 allocation roles |
| History | `history` | append-only event log, **hash-chained** (`prev_hash`/`entry_hash`) so the provenance trail is tamper-evident; head pinned in `mind_meta` |
| Save points | `checkpoints`, `checkpoint_items`, `checkpoint_vectors`, `checkpoint_signals`, `checkpoint_defs` | immutable content-hashed manifest of the COMPLETE state: payloads, vectors, waveforms, and virtual-vector recipes |

## Invariants

0. **Additive-only schema.** No schema version ever drops, renames, or retypes a
   column/table — only adds. This makes forward migration a pure reconciliation:
   `schema.migrate()` adds any missing tables/columns (with declared defaults)
   from the canonical `schema.sql`; `MindFile.open` runs it automatically when a
   file's `schema_version` is older than the library's.
1. **Write-once originals.** `items` and `vectors` rows are never UPDATEd or
   DELETEd in normal operation. Adding data, attaching a new space, or
   re-clustering is additive only.
2. **A vector is keyed by `(item_id, space_id)`.** An item may carry several
   embeddings; each space is fixed-dimension.
3. **Versioned clustering, per space.** Each `run_clustering` appends a
   `clusterings` row whose `version` increments within its `space_id` (and is
   scoped by optional `collection_id`). Earlier runs are immutable.
   `clusters.parent_cluster_id` records split/merge lineage.
4. **Append-only, hash-chained history.** Events: `ingest | embed | cluster |
   split | merge | relabel | link | trajectory | collection | checkpoint |
   signal | materialize | compact | allocate | fuse`. Each entry's
   `entry_hash = sha256(prev_hash || ts || event || detail)` links to the prior
   entry (genesis seeded from `file_uuid`); the head is pinned in
   `mind_meta('history_head')`. `verify_history` proves the trail is unedited.
5. **Content integrity.** `payload_sha256` = sha256(payload bytes);
   `vec_sha256` = sha256(float32 embedding bytes). A checkpoint's
   `manifest_sha256` = sha256 over sorted item rows (`I\t{id}\t{psha}`) followed
   by sorted vector rows (`V\t{id}\t{space_id}\t{vsha}`).

## Vector encoding

Embeddings and centroids are little-endian **float32** `BLOB`s. A libSQL build
with native vector support MAY migrate `vectors.embedding` to `F32_BLOB(dim)` and
add an ANN index via `libsql_vector_idx(embedding)`; this is an index/column-type
change only and does not alter the logical format.

## Two verification layers

- **Corruption** — `verify_checkpoint(name)` re-reads and re-hashes the current
  payload bytes (per item) and embedding bytes (per item+space) and compares to
  the frozen manifest; `status` is `ok` only if nothing changed, nothing is
  missing, and the manifest root is self-consistent. `verify_integrity()` checks
  every live row against its stored hash columns.
- **Validation** — `validate()` reports structural/semantic issues (per-space
  dim consistency, NaN/Inf, items with no vector, duplicate `(subject, session,
  modality)`, referential integrity, longitudinal identity completeness) at
  severities `error | warning | info`. This is independent of byte integrity.
