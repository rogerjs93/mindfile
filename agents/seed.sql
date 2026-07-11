-- Human-editable rows that (re)build registry.db.
DELETE FROM agents;
DELETE FROM edges;

INSERT INTO agents (name, purpose, model, tools, inputs, outputs, dependencies, status, md_path, role, seq) VALUES
 ('planner',  'propose allocation actions from file state', 'deterministic',
  'spaces.list_spaces, cluster.latest_clustering_id, item_axes, correlations',
  'file state', 'ordered action plan', '', 'active', 'planner/planner.md', 'planner', 1),
 ('reviewer', 'approve/reject actions under guards', 'deterministic',
  'validate.validate, size budget, write-once checks',
  'proposed plan', 'approved/rejected plan', 'planner', 'active', 'reviewer/reviewer.md', 'reviewer', 2),
 ('renderer', 'execute approved actions and log them', 'deterministic',
  'cluster.run_clustering, graph.build_knn_edges, fuse.fuse, constellation.*, stats.*',
  'approved plan', 'materialized artifacts + plan_actions log', 'reviewer', 'active', 'renderer/renderer.md', 'renderer', 3);

INSERT INTO edges (from_agent, to_agent, kind) VALUES
 ('planner',  'reviewer', 'sequential'),
 ('reviewer', 'renderer', 'sequential');
