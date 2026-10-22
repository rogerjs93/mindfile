"""LLM-view — render stored items as compact text an LLM can read directly.

The raw bytes stay stored and verifiable; an LLM reads **cards** (normalized
facts per item) and the file ``digest`` (summary + records), never the binary.
Each modality renders to a short factual line; unknown modalities fall back to a
generic line. This is what makes a `.gnos` file "easily read by LLMs".
"""

from __future__ import annotations

import json

from . import lazy as _lazy
from .file import MindFile

_EEG_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
_ECG_BANDS = ["low", "qrs", "high"]


def _decode(payload) -> str:
    if payload is None:
        return ""
    return bytes(payload).decode("utf-8", "replace") if isinstance(payload, (bytes, bytearray)) else str(payload)


def _resolve(mf, item_id, space):
    try:
        return _lazy.resolve(mf, item_id, space)
    except Exception:
        return None


def _band_summary(mf, item, space, labels):
    v = _resolve(mf, item["id"], space)
    sig = mf.conn.execute(
        "SELECT n_channels, n_samples, sfreq FROM signals WHERE item_id=?", (item["id"],)).fetchone()
    head = ""
    if sig:
        dur = sig["n_samples"] / (sig["sfreq"] or 1)
        head = f"{sig['n_channels']}ch {dur:.0f}s@{sig['sfreq']:.0f}Hz"
    if v is None:
        return head or "waveform"
    n = len(v) // 2
    pairs = sorted(zip(labels[:n], v[:n]), key=lambda t: -t[1])
    return f"{head}; rel power " + ", ".join(f"{l} {m:.2f}" for l, m in pairs[:3])


def _eeg(mf, it):
    return _band_summary(mf, it, "bandpower", _EEG_BANDS)


def _ecg(mf, it):
    base = _band_summary(mf, it, "ecg_bandpower", _ECG_BANDS)
    h = _resolve(mf, it["id"], "ecg_hrv")
    if h is not None and h[0] > 0:
        base = f"HR {h[0]:.0f} bpm, HRV(rmssd) {h[2]:.0f}ms; " + base
    return base


def _mri(mf, it):
    ax = {a["axis"]: a["value"] for a in mf.conn.execute(
        "SELECT axis, value FROM item_axes WHERE item_id=?", (it["id"],))}
    v = _resolve(mf, it["id"], "radiomics")
    # radiomics = first-order[0..14] + GLCM texture[15..20]; contrast=15, glcm_entropy=19
    stats = (f"intensity mean {v[0]:.2f}, entropy {v[8]:.2f}, texture contrast {v[15]:.2f}"
             if v is not None and len(v) >= 21 else
             (f"intensity mean {v[0]:.2f}, entropy {v[8]:.2f}" if v is not None else ""))
    pos = (f"centroid ({ax.get('x', 0):.1f},{ax.get('y', 0):.1f},{ax.get('z', 0):.1f})"
           if ax else "")
    return "volume; " + "; ".join(p for p in (stats, pos) if p)


def _text(mf, it):
    return _decode(it["payload"]).strip()[:400] or "(empty note)"


def _labs(mf, it):
    from . import labs
    try:
        d = json.loads(_decode(it["payload"]))
        tag = {"high": " [HIGH]", "low": " [LOW]"}
        return ", ".join(f"{k} {float(v):g}{tag.get(labs.flag(k, v), '')}"
                         for k, v in d.items())
    except Exception:
        return "labs"


def _genomics(mf, it):
    try:
        vs = json.loads(_decode(it["payload"]))
        return f"{len(vs)} variants: " + ", ".join(vs[:6]) + ("..." if len(vs) > 6 else "")
    except Exception:
        return "variants"


def _fhir(mf, it):
    from . import modality
    try:
        return modality.fhir_card(json.loads(_decode(it["payload"])))
    except Exception:
        return "FHIR resource"


def _dicom(mf, it):
    from . import modality
    try:
        return modality.dicom_card(json.loads(_decode(it["payload"])))
    except Exception:
        return "DICOM image"


def _wfdb(mf, it):
    r = mf.conn.execute(
        "SELECT n_channels, n_samples, sfreq, ch_names FROM signals WHERE item_id=?",
        (it["id"],)).fetchone()
    chs = ""
    if r and r["ch_names"]:
        try:
            names = json.loads(r["ch_names"])
            if names:
                chs = "signals " + ",".join(names[:5]) + ("..." if len(names) > 5 else "") + "; "
        except Exception:
            pass
    return chs + _band_summary(mf, it, "wfdb_spectral", ["low", "mid", "high"])


def _reconall(mf, it):
    from . import reconall
    try:
        man = json.loads(_decode(it["payload"]))
    except Exception:
        return "recon-all subject"
    ratio = man["total_bytes"] / max(man["stored_bytes"], 1)
    head = (f"FreeSurfer recon-all subject '{man['subject']}' ({man['profile']}); "
            f"{man['n_stored']}/{man['n_files']} files, "
            f"{man['total_bytes']/1e6:.0f}MB stored in {man['stored_bytes']/1e6:.0f}MB "
            f"({ratio:.2f}x)")
    try:
        st = reconall.read_stats(mf, it["id"])
    except Exception:
        return head
    bits = []
    aseg = st.get("aseg")
    if aseg:
        m = aseg["measures"]
        if m.get("eTIV") or m.get("ICV"):
            bits.append(f"eTIV {(m.get('eTIV') or m['ICV'])/1000:.0f}cm3")
        if m.get("CortexVol"):
            bits.append(f"cortical GM {m['CortexVol']/1000:.0f}cm3")
        for side, key in (("L", "Left-Hippocampus"), ("R", "Right-Hippocampus")):
            if aseg["structures"].get(key):
                bits.append(f"{side} hippocampus {aseg['structures'][key]:.0f}mm3")
    for hemi in ("lh", "rh"):
        mt = (st.get(hemi) or {}).get("measures", {}).get("MeanThickness")
        if mt:
            bits.append(f"{hemi} mean thickness {mt:.2f}mm")
    return head + ("; " + ", ".join(bits) if bits else "")


_RENDERERS = {"eeg": _eeg, "ecg": _ecg, "mri": _mri, "text": _text, "labs": _labs,
              "tabular": _labs, "genomics": _genomics, "fhir": _fhir, "dicom": _dicom,
              "wfdb": _wfdb, "reconall": _reconall}


def register_renderer(modality: str, fn) -> None:
    """Add/override the card renderer for a modality."""
    _RENDERERS[modality] = fn


def card(mf: MindFile, item_id: int) -> str:
    """A one-line, LLM-readable card for one item."""
    r = mf.conn.execute(
        "SELECT id, external_id, subject_id, session_id, modality, payload FROM items WHERE id=?",
        (item_id,)).fetchone()
    if r is None:
        return f"item {item_id}: not found"
    it = dict(r)
    who = it["subject_id"] or it["external_id"] or f"item{it['id']}"
    if it["session_id"]:
        who += f"/{it['session_id']}"
    body = _RENDERERS.get(it["modality"] or "", lambda mf, it: "stored item")(mf, it)
    return f"[{it['modality'] or 'generic'}] {who}: {body}"


def digest(mf: MindFile, *, subject_id: str | None = None, max_records: int = 200) -> str:
    """The whole file as text an LLM can read: summary + per-record cards."""
    from . import summary as _summary
    lines = [_summary.summary_text(mf), "", "## records"]
    sql = "SELECT id FROM items"
    params: tuple = ()
    if subject_id:
        sql += " WHERE subject_id = ?"
        params = (subject_id,)
    sql += " ORDER BY subject_id, modality, id LIMIT ?"
    rows = mf.conn.execute(sql, (*params, max_records)).fetchall()
    lines += ["- " + card(mf, r["id"]) for r in rows]
    from . import findings as _findings
    f = _findings.list_findings(mf)
    if f:
        lines += ["", "## findings (auto-discovered correlations)"] + [f"- {x['statement']}" for x in f]
    return "\n".join(lines)
