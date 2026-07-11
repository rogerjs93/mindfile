-- Canonical schema for the .mind AI-native file format v2 (runtime source of truth).
--
-- The file is an ordinary SQLite database, byte-compatible with libSQL / Turso.
-- v2 adds: multiple embedding SPACES per item, item IDENTITY (subject/session/
-- study/time) so the same items can be analysed as a cohort, a longitudinal
-- series, or a multi-modal patient bundle, and COLLECTIONS that scope analyses.
--
-- Design invariants:
--   * `items` and `vectors` are WRITE-ONCE (append-only); originals never mutate.
--   * A vector is identified by (item_id, space_id): an item may carry several
--     embeddings (e.g. bandpower / connectivity / encoder).
--   * Each clustering run is a new immutable version, scoped to one space (and
--     optionally one collection); version numbers increment within a space.
--   * `history` is append-only; `checkpoints` freeze content hashes for verify.

PRAGMA foreign_keys = ON;

-- HEADER -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mind_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- EMBEDDING SPACES (the "several ways of embedding") -----------------------
CREATE TABLE IF NOT EXISTS spaces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,   -- e.g. default | bandpower | connectivity | encoder
    embedder_id TEXT,
    dim         INTEGER NOT NULL,
    modality    TEXT,                   -- expected modality for this space (optional)
    derived     INTEGER NOT NULL DEFAULT 0,  -- 1 = computed (fused/constellation); only these may be compacted
    created_at  TEXT NOT NULL
);

-- DATA BLOCK (write-once) + IDENTITY ---------------------------------------
CREATE TABLE IF NOT EXISTS items (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id    TEXT,
    origin         TEXT NOT NULL DEFAULT 'primary',  -- primary (acquired) | derived (computed, e.g. fused)
    subject_id     TEXT,          -- patient / participant identity
    session_id     TEXT,          -- timepoint / visit label (longitudinal)
    study_id       TEXT,          -- group-study / dataset identity
    modality       TEXT,          -- eeg | mri | labs | text | ...
    acquired_at    TEXT,          -- acquisition time (orders longitudinal series)
    payload        BLOB,          -- inline bytes, or a path/url when payload_kind != 'inline'
    payload_kind   TEXT NOT NULL DEFAULT 'inline',
    payload_sha256 TEXT,
    meta           TEXT,          -- JSON
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_external ON items(external_id);
CREATE INDEX IF NOT EXISTS idx_items_subject  ON items(subject_id);
CREATE INDEX IF NOT EXISTS idx_items_study    ON items(study_id);
CREATE INDEX IF NOT EXISTS idx_items_hash     ON items(payload_sha256);

-- VECTOR BLOCK (write-once, per (item, space)) -----------------------------
CREATE TABLE IF NOT EXISTS vectors (
    item_id     INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    space_id    INTEGER NOT NULL REFERENCES spaces(id),
    embedder_id TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    embedding   BLOB NOT NULL,    -- float32 little-endian; libSQL upgrade -> F32_BLOB(dim)
    vec_sha256  TEXT,
    PRIMARY KEY (item_id, space_id)
);
CREATE INDEX IF NOT EXISTS idx_vectors_space ON vectors(space_id);

-- AXES (per-item coordinates: x, y, z, t ... -> "constellations") -----------
-- A data point's position in a coordinate frame. Combined with its semantic
-- vector this yields a position-aware "constellation" space (see fuse/concat).
CREATE TABLE IF NOT EXISTS item_axes (
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    axis    TEXT NOT NULL,        -- x | y | z | t | age | ...
    value   REAL NOT NULL,
    PRIMARY KEY (item_id, axis)
);

-- COLLECTIONS (named subsets that scope an analysis) -----------------------
CREATE TABLE IF NOT EXISTS collections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT UNIQUE NOT NULL,
    kind       TEXT,              -- cohort | longitudinal | multimodal | custom
    definition TEXT,              -- JSON filter used to (re)materialise members
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS collection_items (
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    item_id       INTEGER NOT NULL REFERENCES items(id),
    PRIMARY KEY (collection_id, item_id)
);

-- CLUSTER BLOCK (versioned per space, immutable per run) --------------------
CREATE TABLE IF NOT EXISTS clusterings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    space_id      INTEGER NOT NULL REFERENCES spaces(id),
    collection_id INTEGER REFERENCES collections(id),   -- NULL = all items in the space
    version       INTEGER NOT NULL,                     -- increments within (space)
    algo          TEXT NOT NULL,
    params        TEXT,
    embedder_id   TEXT,
    created_at    TEXT NOT NULL,
    parent_id     INTEGER REFERENCES clusterings(id),
    note          TEXT
);
CREATE INDEX IF NOT EXISTS idx_clusterings_space ON clusterings(space_id);

CREATE TABLE IF NOT EXISTS clusters (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    clustering_id     INTEGER NOT NULL REFERENCES clusterings(id),
    local_index       INTEGER NOT NULL,
    label             TEXT,
    centroid          BLOB,
    size              INTEGER,
    parent_cluster_id INTEGER REFERENCES clusters(id)
);
CREATE INDEX IF NOT EXISTS idx_clusters_run ON clusters(clustering_id);

CREATE TABLE IF NOT EXISTS cluster_assignments (
    clustering_id INTEGER NOT NULL REFERENCES clusterings(id),
    item_id       INTEGER NOT NULL REFERENCES items(id),
    cluster_id    INTEGER NOT NULL REFERENCES clusters(id),
    distance      REAL,
    PRIMARY KEY (clustering_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_assign_cluster ON cluster_assignments(cluster_id);

-- KNOWLEDGE BLOCK (graph) --------------------------------------------------
-- space_id: which embedding space a similarity edge came from (0 = none/manual).
-- dst_file: '' for within-file; reserved for future cross-file links.
CREATE TABLE IF NOT EXISTS edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    src_item_id INTEGER NOT NULL,
    dst_item_id INTEGER NOT NULL,
    dst_file    TEXT NOT NULL DEFAULT '',
    space_id    INTEGER NOT NULL DEFAULT 0,
    kind        TEXT NOT NULL,      -- knn | similar | temporal | same_subject | derived | manual
    weight      REAL,
    meta        TEXT,
    UNIQUE (src_item_id, dst_item_id, dst_file, space_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_item_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_item_id);

-- DERIVATIONS (provenance for fused / derived items) -----------------------
-- A derived item (origin='derived') is built additively from source items; the
-- originals are never modified. These tables record exactly which source
-- (item, space) vectors and their hashes produced each derived item, so the
-- fusion can be verified to still correspond to intact originals.
CREATE TABLE IF NOT EXISTS derivations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    derived_item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    out_space_id    INTEGER REFERENCES spaces(id),
    method          TEXT NOT NULL,        -- e.g. concat-l2-v1
    params          TEXT,                 -- JSON
    fusion_key      TEXT,                 -- JSON of the grouping key values
    source_sha256   TEXT NOT NULL,        -- fingerprint over sorted (src_item, space, vec_sha256)
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS derivation_sources (
    derivation_id  INTEGER NOT NULL REFERENCES derivations(id) ON DELETE CASCADE,
    source_item_id INTEGER NOT NULL,
    space_id       INTEGER NOT NULL,
    vec_sha256     TEXT,                  -- source vector hash frozen at fusion time
    PRIMARY KEY (derivation_id, source_item_id, space_id)
);

-- LAZY / VIRTUAL DERIVED VECTORS -------------------------------------------
-- A virtual vector is a *recipe* (how to compute a derived vector) with NO
-- materialized row in `vectors`. It is resolved on access and may be cached
-- (materialized) or dropped (dematerialized) to trade compute for space.
-- Deterministic recipes resolve to identical bytes, so they stay verifiable;
-- `source_sha256` fingerprints the inputs so drift is detectable.
CREATE TABLE IF NOT EXISTS vector_defs (
    item_id       INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    space_id      INTEGER NOT NULL REFERENCES spaces(id),
    kind          TEXT NOT NULL,    -- fused | constellation | ...
    recipe        TEXT NOT NULL,    -- JSON: method + sources + params
    source_sha256 TEXT,             -- fingerprint over the recipe's source vectors
    deterministic INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (item_id, space_id)
);

-- COMPACT CODES (lossless, derived vectors only) ---------------------------
-- A compacted derived vector is stored here as a compressed code (e.g. zlib of
-- the float32 bytes) instead of in `vectors`; it has a `vector_defs` recipe of
-- kind 'compressed' and reconstructs byte-identically on access. Primary spaces
-- are never compacted, so originals stay exact.
CREATE TABLE IF NOT EXISTS vector_codes (
    item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    space_id   INTEGER NOT NULL REFERENCES spaces(id),
    codec      TEXT NOT NULL,        -- zlib
    code       BLOB NOT NULL,        -- compressed bytes
    raw_dim    INTEGER NOT NULL,
    raw_sha256 TEXT,                 -- hash of the decompressed (logical) vector
    code_sha256 TEXT,                -- hash of the stored code
    created_at TEXT NOT NULL,
    PRIMARY KEY (item_id, space_id)
);

-- SIGNALS (lossless waveform store: channels x time) -----------------------
-- The raw EEG/MEG waveform for an item, stored as a signal-aware lossless code
-- (channel byte-shuffle + zlib, with integer delta-along-time for int dtypes).
-- Reconstructs byte-exact on call; band-power/feature vectors are then lazy
-- recipes (kind 'signal_feature') that recompute from this waveform on demand.
CREATE TABLE IF NOT EXISTS signals (
    item_id     INTEGER PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    codec       TEXT NOT NULL,        -- shuffle-zlib | delta-shuffle-zlib
    dtype       TEXT NOT NULL,        -- numpy dtype str, e.g. '<f4', '<i2'
    n_channels  INTEGER NOT NULL,
    n_samples   INTEGER NOT NULL,
    sfreq       REAL,
    ch_names    TEXT,                 -- JSON
    delta       INTEGER NOT NULL DEFAULT 0,
    code        BLOB NOT NULL,
    raw_sha256  TEXT,                 -- hash of the (lossless) or reconstructed (lossy) waveform
    code_sha256 TEXT,                 -- hash of the stored code
    lossy       INTEGER NOT NULL DEFAULT 0,  -- 1 = approximate (spectral) representation
    error       REAL,                -- relative reconstruction error for lossy codecs
    created_at  TEXT NOT NULL
);

-- EVENTS (task/stimulus markers for a signal recording; onset in seconds) -----
-- Turns a stored waveform into an analysable epoched recording so event-locked
-- dynamics (e.g. ERD, baseline-normalised) can be computed from it on call.
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY,
    item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    onset      REAL NOT NULL,        -- seconds from recording start
    duration   REAL NOT NULL DEFAULT 0.0,
    condition  TEXT,                 -- e.g. 'left' / 'right' trial type
    value      TEXT,                 -- marker code (e.g. stage within a trial)
    label      TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_item ON events(item_id);

-- TRUSTED KEYS (authenticity policy: which signers this file expects) --------
CREATE TABLE IF NOT EXISTS trusted_keys (
    key_id     TEXT PRIMARY KEY,     -- public-key fingerprint
    public_key BLOB NOT NULL,        -- ed25519 public key bytes
    label      TEXT,
    added_at   TEXT NOT NULL
);

-- FILE REFS (cross-file registry: a study file referencing patient files) ---
-- Pins the identity (uuid), exact state (state_root) and optionally the signing
-- key of another .mind file, so verify_refs detects if a referenced file goes
-- missing, is substituted, drifts from its registered state, or loses its
-- valid signature.
CREATE TABLE IF NOT EXISTS file_refs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    path       TEXT NOT NULL,        -- path to the referenced .mind (relative -> resolved at verify)
    file_uuid  TEXT,                 -- expected uuid
    state_root TEXT,                 -- expected state at link time
    key_id     TEXT,                 -- expected signing key (optional)
    public_key BLOB,                 -- expected public key (optional)
    role       TEXT,                 -- patient | recording | member | ...
    added_at   TEXT NOT NULL,
    note       TEXT
);

-- SIGNATURES (authenticity: who attested to this state) --------------------
-- A signature over the file's `state_root` (uuid + schema + history head +
-- latest checkpoint root). Ed25519 stores the public key so anyone can verify
-- without being able to forge; HMAC is a stdlib fallback needing the shared
-- secret to verify. Proves authenticity on top of integrity.
CREATE TABLE IF NOT EXISTS signatures (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    algo       TEXT NOT NULL,        -- ed25519 | hmac-sha256
    key_id     TEXT,                 -- label / public-key fingerprint
    public_key BLOB,                 -- ed25519 public key bytes (NULL for hmac)
    state_root TEXT NOT NULL,        -- the digest that was signed
    signature  BLOB NOT NULL,
    signed_at  TEXT NOT NULL,
    note       TEXT
);

-- AGENT BLOCK (self-update policy) -----------------------------------------
CREATE TABLE IF NOT EXISTS agents (
    name    TEXT PRIMARY KEY,
    purpose TEXT,
    trigger TEXT,
    action  TEXT,
    params  TEXT,
    status  TEXT
);

-- ALLOCATION (Planner -> Reviewer -> Renderer, RecursiveMAS Sequential) -----
-- The 3 deterministic roles exchange a plan: the Planner proposes actions, the
-- Reviewer approves/rejects them under guards, the Renderer executes the
-- approved ones. Every step is logged so the allocation is auditable in-file.
CREATE TABLE IF NOT EXISTS plans (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    status     TEXT NOT NULL,     -- planned | reviewed | rendered
    summary    TEXT
);
CREATE TABLE IF NOT EXISTS plan_actions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id       INTEGER NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL,
    action        TEXT NOT NULL,  -- cluster | knn | fuse | constellation | correlate | vacuum
    args          TEXT,           -- JSON
    plan_reason   TEXT,           -- why the Planner proposed it
    review_status TEXT,           -- approved | rejected
    review_reason TEXT,           -- the Reviewer's verdict
    render_status TEXT,           -- done | failed | skipped
    render_detail TEXT            -- JSON
);

-- CORRELATIONS (in-file statistics) ----------------------------------------
CREATE TABLE IF NOT EXISTS correlations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,     -- clustering_ari | cross_space | feature_corr | subject_cross_modality
    a_ref      TEXT,              -- e.g. clustering:1 | space:bandpower
    b_ref      TEXT,
    method     TEXT,
    value      REAL,
    n          INTEGER,
    detail     TEXT,              -- JSON
    created_at TEXT NOT NULL
);

-- FINDINGS (the living correlation surface — for external LLM analysis) ------
-- Auto-discovered relationships across the file's data structures, rendered as
-- plain-English statements an LLM can reason over (the LLM stays outside the
-- file; this is what it reads).
CREATE TABLE IF NOT EXISTS findings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,        -- cross_modal_agreement | cluster_signature | ...
    statement  TEXT NOT NULL,        -- LLM-readable sentence
    value      REAL,                 -- effect size / metric
    n          INTEGER,
    refs       TEXT,                 -- JSON (spaces/clusterings/clusters involved)
    detail     TEXT,                 -- JSON
    created_at TEXT NOT NULL
);

-- HISTORY (append-only, hash-chained -> tamper-evident provenance) ----------
-- Each entry hash-links to the previous: entry_hash = sha256(prev_hash || ts ||
-- event || detail). Any silent edit/deletion/insertion/reorder breaks the chain
-- (verify_history). The head hash is also pinned in mind_meta('history_head')
-- and can be exported/anchored externally for full-rewrite resistance.
CREATE TABLE IF NOT EXISTS history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    event      TEXT NOT NULL,
    detail     TEXT,
    prev_hash  TEXT,
    entry_hash TEXT
);

-- SAVE POINTS (immutable checkpoints; multi-space aware) -------------------
CREATE TABLE IF NOT EXISTS checkpoints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT UNIQUE NOT NULL,
    created_at      TEXT NOT NULL,
    item_count      INTEGER NOT NULL,
    vector_count    INTEGER NOT NULL,
    clustering_id   INTEGER REFERENCES clusterings(id),
    manifest_sha256 TEXT NOT NULL,
    note            TEXT
);
CREATE TABLE IF NOT EXISTS checkpoint_items (
    checkpoint_id  INTEGER NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
    item_id        INTEGER NOT NULL,
    payload_sha256 TEXT,
    cluster_id     INTEGER,
    PRIMARY KEY (checkpoint_id, item_id)
);
CREATE TABLE IF NOT EXISTS checkpoint_vectors (
    checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
    item_id       INTEGER NOT NULL,
    space_id      INTEGER NOT NULL,
    vec_sha256    TEXT,
    PRIMARY KEY (checkpoint_id, item_id, space_id)
);
-- A complete save point also pins the newer data types so verify_checkpoint
-- proves the WHOLE file (waveforms + lazy/compressed derived vectors) is
-- unchanged since the snapshot, not just the materialized vectors.
CREATE TABLE IF NOT EXISTS checkpoint_signals (
    checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
    item_id       INTEGER NOT NULL,
    raw_sha256    TEXT,             -- logical waveform hash
    code_sha256   TEXT,             -- stored code hash (any change flips this)
    PRIMARY KEY (checkpoint_id, item_id)
);
CREATE TABLE IF NOT EXISTS checkpoint_defs (
    checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
    item_id       INTEGER NOT NULL,
    space_id      INTEGER NOT NULL,
    kind          TEXT,
    source_sha256 TEXT,             -- virtual vector's recipe-source fingerprint
    PRIMARY KEY (checkpoint_id, item_id, space_id)
);
