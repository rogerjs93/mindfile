"""Clinical reference ranges for lab analytes.

Turns a raw lab value into a clinically meaningful flag (low / normal / high)
without the LLM having to know reference intervals. Ranges are applied at render
and findings time from the analyte-keyed payload — no storage change.

The built-in table uses common adult reference intervals in the **canonical unit
noted per analyte**; values in a `labs` payload are assumed to be in those units.
Pass a custom ``ranges`` mapping (``{analyte: (low, high, unit)}``) to override
for a lab that reports differently.
"""

from __future__ import annotations

import math

# analyte -> (low, high, unit).  One-sided markers use 0 / large as the open end.
REFERENCE_RANGES: dict[str, tuple[float, float, str]] = {
    "crp": (0.0, 5.0, "mg/L"),
    "troponin": (0.0, 0.04, "ng/mL"),
    "d_dimer": (0.0, 0.5, "mg/L FEU"),
    "wbc": (4.0, 11.0, "10^9/L"),
    "hemoglobin": (12.0, 17.0, "g/dL"),
    "platelets": (150.0, 400.0, "10^9/L"),
    "sodium": (135.0, 145.0, "mmol/L"),
    "potassium": (3.5, 5.1, "mmol/L"),
    "creatinine": (0.6, 1.3, "mg/dL"),
    "egfr": (90.0, 200.0, "mL/min/1.73m2"),
    "glucose": (70.0, 99.0, "mg/dL"),
    "hba1c": (4.0, 5.6, "%"),
    "ldl": (0.0, 100.0, "mg/dL"),
    "hdl": (40.0, 200.0, "mg/dL"),
    "alt": (7.0, 56.0, "U/L"),
    "ast": (10.0, 40.0, "U/L"),
    "tsh": (0.4, 4.0, "mIU/L"),
    "inr": (0.8, 1.2, ""),
}

_ALIASES = {"troponin_t": "troponin", "troponin_i": "troponin", "trop": "troponin",
            "hb": "hemoglobin", "hgb": "hemoglobin", "plt": "platelets",
            "na": "sodium", "k": "potassium", "cr": "creatinine", "creat": "creatinine",
            "gluc": "glucose", "a1c": "hba1c", "ddimer": "d_dimer"}


def _key(analyte: str) -> str:
    k = str(analyte).strip().lower().replace(" ", "_")
    return _ALIASES.get(k, k)


def reference(analyte: str, ranges: dict | None = None):
    """Return ``(low, high, unit)`` for an analyte, or ``None`` if unknown."""
    return (ranges or REFERENCE_RANGES).get(_key(analyte))


def flag(analyte: str, value, ranges: dict | None = None) -> str | None:
    """``'low'`` / ``'normal'`` / ``'high'``, or ``None`` if the analyte is
    unknown or the value isn't a finite number."""
    r = reference(analyte, ranges)
    if r is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    if v < r[0]:
        return "low"
    if v > r[1]:
        return "high"
    return "normal"


def abnormal(row: dict, ranges: dict | None = None) -> list[tuple[str, str]]:
    """List ``(analyte, 'low'|'high')`` for the out-of-range values in a row."""
    out = []
    for a, v in row.items():
        f = flag(a, v, ranges)
        if f in ("low", "high"):
            out.append((a, f))
    return out
