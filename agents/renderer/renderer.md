# renderer

The Solver. Executes the approved actions in order — clustering, kNN graphs,
fusion, the constellation space + per-patient constellations, correlations, and
VACUUM — and logs every step (status + detail) into the file's `plans` /
`plan_actions` tables. A failed action is recorded and skipped; it never aborts
the rest of the render.

Implementation: `allocate.render()`.
