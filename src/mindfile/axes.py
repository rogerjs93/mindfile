"""Per-item coordinate axes (x, y, z, t ...) — the geometry of a constellation.

Axes are additive metadata on existing items; setting them never touches the
items' payloads or semantic vectors. They feed the position-aware constellation
space (see ``constellation.py``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .file import MindFile


def set_axes(mf: MindFile, item_id: int, coords: dict[str, float]) -> None:
    """Upsert coordinate axes for one item, e.g. {'x':1.0,'y':2.0,'z':3.0,'t':0}."""
    mf.conn.executemany(
        "INSERT INTO item_axes(item_id, axis, value) VALUES(?,?,?) "
        "ON CONFLICT(item_id, axis) DO UPDATE SET value = excluded.value",
        [(item_id, ax, float(v)) for ax, v in coords.items()],
    )
    mf.conn.commit()


def set_axes_bulk(
    mf: MindFile, item_ids: Sequence[int], coords: np.ndarray, axis_names: Sequence[str]
) -> None:
    """Set the same axes for many items at once. coords is (n, len(axis_names))."""
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (len(item_ids), len(axis_names)):
        raise ValueError("coords shape must be (len(item_ids), len(axis_names))")
    rows = []
    for i, item_id in enumerate(item_ids):
        for j, ax in enumerate(axis_names):
            rows.append((int(item_id), ax, float(coords[i, j])))
    mf.conn.executemany(
        "INSERT INTO item_axes(item_id, axis, value) VALUES(?,?,?) "
        "ON CONFLICT(item_id, axis) DO UPDATE SET value = excluded.value", rows)
    mf.conn.commit()


def get_axes(mf: MindFile, item_id: int) -> dict[str, float]:
    return {r["axis"]: r["value"] for r in mf.conn.execute(
        "SELECT axis, value FROM item_axes WHERE item_id = ?", (item_id,))}


def axes_matrix(
    mf: MindFile, item_ids: Sequence[int], axis_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (kept_item_ids, matrix) for the items that have ALL named axes."""
    kept, rows = [], []
    for item_id in item_ids:
        ax = get_axes(mf, int(item_id))
        if all(a in ax for a in axis_names):
            kept.append(int(item_id))
            rows.append([ax[a] for a in axis_names])
    ids = np.array(kept, dtype=np.int64)
    mat = np.array(rows, dtype=np.float32) if rows else np.empty((0, len(axis_names)), np.float32)
    return ids, mat
