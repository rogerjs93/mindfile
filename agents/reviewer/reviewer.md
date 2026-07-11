# reviewer

The Critic. Approves or rejects each proposed action under guards: the file must
pass `validate()` (no errors), clustering needs ≥2 vectors and must not already
exist, kNN must not be already built, fusion must not be redone, and a size
budget (`max_new_vectors`) caps how many derived vectors an action may add.
Records a verdict + reason for every action.

Implementation: `allocate.review()`.
