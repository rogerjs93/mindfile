# planner

Reads file state (spaces, vector counts, existing clusterings, axes,
correlations) and proposes an ordered list of allocation actions: cluster
un-clustered spaces, build missing kNN graphs, fuse modality spaces, build the
position-aware constellation space, compute correlations that have something to
compare, and VACUUM when free pages accumulate. Proposes only — never mutates.

Implementation: `allocate.plan()`.
