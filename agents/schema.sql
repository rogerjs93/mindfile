-- Registry schema for the allocation agents (source of truth; registry.db is generated).
CREATE TABLE IF NOT EXISTS agents (
    name         TEXT PRIMARY KEY,
    purpose      TEXT,                 -- one-line
    model        TEXT,                 -- 'deterministic' here (no LLM)
    tools        TEXT,                 -- mindfile functions the role drives
    inputs       TEXT,
    outputs      TEXT,
    dependencies TEXT,
    status       TEXT,
    md_path      TEXT,
    role         TEXT,                 -- planner | reviewer | renderer
    seq          INTEGER               -- order in the Sequential pipeline
);

-- Sequential coordination edges (from_agent -> to_agent).
CREATE TABLE IF NOT EXISTS edges (
    from_agent TEXT,
    to_agent   TEXT,
    kind       TEXT                    -- sequential | loopback
);
