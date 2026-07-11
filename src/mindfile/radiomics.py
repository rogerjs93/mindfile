"""Radiomics — first-order + GLCM/Haralick texture features for imaging.

Dependency-free (numpy). First-order captures intensity distribution; GLCM
(gray-level co-occurrence) captures spatial texture — the foundation of
radiomics. This is a genuine texture feature set, NOT the full IBSI/pyradiomics
suite (GLRLM/GLSZM/wavelet) and computes over the whole array, not a segmented
ROI — both are documented upgrades. Works for 2-D images and 3-D volumes.
"""

from __future__ import annotations

import numpy as np

FIRST_ORDER_LABELS = ["mean", "std", "min", "max", "median", "p10", "p90",
                      "energy", "entropy", "skew", "kurtosis", "nonzero_frac",
                      "shape0", "shape1", "shape2"]
TEXTURE_LABELS = ["contrast", "dissimilarity", "homogeneity", "asm", "glcm_entropy",
                  "correlation"]
RADIOMICS_LABELS = FIRST_ORDER_LABELS + TEXTURE_LABELS


def _entropy(values: np.ndarray) -> float:
    h, _ = np.histogram(values, bins=64)
    p = h / (h.sum() or 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def first_order_features(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    f = a.ravel()
    std = f.std() or 1.0
    feats = [f.mean(), f.std(), f.min(), f.max(), np.median(f),
             np.percentile(f, 10), np.percentile(f, 90), float((f ** 2).mean()),
             _entropy(f), float(((f - f.mean()) ** 3).mean() / std ** 3),
             float(((f - f.mean()) ** 4).mean() / std ** 4), float((f != 0).mean())]
    feats += list(a.shape[:3]) + [0.0] * (3 - min(3, a.ndim))
    return np.array(feats, dtype="float32")


def centroid(arr: np.ndarray) -> list[float]:
    a = np.asarray(arr, dtype=np.float64)
    w = np.abs(a)
    tot = w.sum() or 1.0
    idx = np.indices(a.shape)
    c = [float((idx[d] * w).sum() / tot) for d in range(min(3, a.ndim))]
    return (c + [0.0, 0.0, 0.0])[:3]


def _quantize(arr: np.ndarray, levels: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    lo, hi = a.min(), a.max()
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.int64)
    q = np.floor((a - lo) / (hi - lo) * (levels - 1e-9)).astype(np.int64)
    return np.clip(q, 0, levels - 1)


def _pair(q: np.ndarray, off) -> tuple[np.ndarray, np.ndarray]:
    """Aligned value pairs for a coordinate offset (valid region only)."""
    src, dst = [slice(None)] * q.ndim, [slice(None)] * q.ndim
    for ax, o in enumerate(off):
        if o > 0:
            src[ax], dst[ax] = slice(0, -o), slice(o, None)
        elif o < 0:
            src[ax], dst[ax] = slice(-o, None), slice(0, o)
    return q[tuple(src)].ravel(), q[tuple(dst)].ravel()


def glcm(arr: np.ndarray, levels: int = 16, offsets=None) -> np.ndarray:
    q = _quantize(arr, levels)
    if offsets is None:
        offsets = ([(1, 0, 0), (0, 1, 0), (0, 0, 1)] if q.ndim == 3
                   else [(1, 0), (0, 1), (1, 1), (1, -1)] if q.ndim == 2 else [(1,)])
    P = np.zeros((levels, levels), dtype=np.float64)
    for off in offsets:
        a, b = _pair(q, off)
        P += np.bincount(a * levels + b, minlength=levels * levels).reshape(levels, levels)
    P += P.T                       # symmetric
    s = P.sum()
    return P / s if s else P


def texture_features(P: np.ndarray) -> np.ndarray:
    n = P.shape[0]
    i = np.arange(n)
    I, J = np.meshgrid(i, i, indexing="ij")
    contrast = float((P * (I - J) ** 2).sum())
    dissim = float((P * np.abs(I - J)).sum())
    homog = float((P / (1.0 + (I - J) ** 2)).sum())
    asm = float((P ** 2).sum())
    ent = float(-(P * np.log2(P + 1e-12)).sum())
    mu_i, mu_j = float((I * P).sum()), float((J * P).sum())
    si = float(np.sqrt(((I - mu_i) ** 2 * P).sum()))
    sj = float(np.sqrt(((J - mu_j) ** 2 * P).sum()))
    corr = float(((I - mu_i) * (J - mu_j) * P).sum() / (si * sj)) if si > 0 and sj > 0 else 0.0
    return np.array([contrast, dissim, homog, asm, ent, corr], dtype="float32")


def radiomics_features(arr: np.ndarray, levels: int = 16) -> np.ndarray:
    """First-order (15) + GLCM texture (6) = 21 features."""
    return np.concatenate([first_order_features(arr),
                           texture_features(glcm(arr, levels))]).astype("float32")
