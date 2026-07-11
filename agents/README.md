# Allocation agents

The `.mind` file manages its own space allocation with a **three-specialist,
deterministic** pipeline — RecursiveMAS Tier-1 *Sequential* (Planner → Critic →
Solver), here **Planner → Reviewer → Renderer**. Behaviour lives in
`src/mindfile/allocate.py`; this folder is the version-controlled **config /
documentation** per the project convention.

```
Planner  ──►  Reviewer  ──►  Renderer
(propose)     (approve/guard) (execute + log)
```

- **planner** — inspects file state and proposes ordered actions (cluster / knn /
  fuse / constellation / correlate / vacuum).
- **reviewer** — approves or rejects each action under guards (write-once safety,
  min-vectors, already-done, size budget, clean `validate()`).
- **renderer** — executes approved actions and logs each into `plans` /
  `plan_actions`.

## Layout

```
agents/
  README.md            this file
  schema.sql           registry schema (source of truth)
  seed.sql             rows that (re)build registry.db
  registry.db          generated artifact (gitignored)
  planner/planner.md   one-line role summaries (orientation only, not prompts)
  reviewer/reviewer.md
  renderer/renderer.md
```

Rebuild the registry:  `sqlite3 agents/registry.db ".read agents/schema.sql" ".read agents/seed.sql"`

The same three roles are also registered *inside* each `.mind` file's `agents`
table by `allocate.register_roles()`, so a file carries its own allocation policy.
These specialists are **deterministic** (no model calls); Tier-2 latent agents
are intentionally not used here.
