"""
Cross-sample feature alignment (cohort_with_qc).

align_features_detailed() does, in order:
  1. optional retention-time correction: each sample is aligned to a reference
     sample using unique, intense "anchor" features and a LOWESS drift curve;
  2. seeded grouping: the most intense unassigned feature seeds a group; the
     CLOSEST unassigned feature of every other sample within the m/z and RT
     tolerances joins it (at most one feature per sample per group). Features
     that lose out to a closer one in the same sample are left to form or join
     other groups instead of being silently dropped.

align_features_legacy() is the previous algorithm, kept only so results can be
compared (first-fit, quadratic, drops same-sample duplicates).

The returned table has the same layout as before: feature_id, mz, rt and one
intensity column per sample, NaN where the sample had no matching feature.
NaN still means "not detected" -- gap filling lives in gap_fill.py and goes to
a separate file so presence/absence stays meaningful downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass
class AlignmentParams:
    """
    Options for alignment. The anchor_* / lowess_* values only affect retention-time correction:
    anchors are intense features that are unique within a small m/z-RT window in both the sample and the reference.
    """
    mz_tolerance_ppm: float
    rt_tolerance_sec: float
    rt_correction: bool = True
    reference_sample: str | None = None     # None = sample with the most features
    anchor_mz_ppm: float = 5.0
    anchor_rt_search_sec: float = 30.0      # max drift considered when pairing anchors
    anchor_top_n: int = 3000                # most intense features of a sample used as anchor candidates
    min_anchors: int = 50                   # fewer unique anchors -> sample left uncorrected (warned)
    lowess_frac: float = 0.3


def _coords(mz: np.ndarray, rt: np.ndarray, mz_tol_ppm: float, rt_tol_sec: float) -> np.ndarray:
    """Scale (m/z, RT) so that one tolerance is one unit in each axis, for nearest-neighbour searches."""
    # d(ln mz) * 1e6 == ppm, so scaled Chebyshev distance <= 1  <=>  within BOTH tolerances
    return np.column_stack([np.log(mz) * 1e6 / mz_tol_ppm, rt / rt_tol_sec])


# --------------------------------------------------------------------------
# 1. retention-time correction
# --------------------------------------------------------------------------
def _fit_shift(sample: dict, ref_coords_tree: cKDTree, ref_rt: np.ndarray, p: AlignmentParams):
    """
    Fit one sample's retention-time drift against the reference with LOWESS over its anchor pairs.

    Returns:
        ((rt_grid, shift_seconds), n_anchors), or (None, n_anchors) when fewer than `min_anchors` unique anchors
        exist, in which case the sample is left uncorrected.
    """
    mz, rt, inten = sample["mz"], sample["rt"], sample["intensity"]
    top = np.argsort(-inten)[: p.anchor_top_n]
    c = _coords(mz[top], rt[top], p.anchor_mz_ppm, p.anchor_rt_search_sec)
    n_cand = ref_coords_tree.query_ball_point(c, r=1.0, p=np.inf, return_length=True)
    uniq = np.where(n_cand == 1)[0]
    if len(uniq) == 0:
        return None, 0
    _, nn = ref_coords_tree.query(c[uniq], k=1)
    # the reference feature must also be claimed by one sample feature only
    vals, counts = np.unique(nn, return_counts=True)
    ok = np.isin(nn, vals[counts == 1])
    uniq, nn = uniq[ok], nn[ok]
    if len(uniq) < p.min_anchors:
        return None, int(len(uniq))
    x = rt[top][uniq]
    y = ref_rt[nn] - x
    from statsmodels.nonparametric.smoothers_lowess import lowess
    frac = min(1.0, max(p.lowess_frac, 20.0 / len(x)))
    fit = lowess(y, x, frac=frac, it=3, delta=0.01 * (x.max() - x.min()), return_sorted=True)
    gx, inv = np.unique(fit[:, 0], return_inverse=True)
    gy = np.bincount(inv, weights=fit[:, 1]) / np.bincount(inv)
    return (gx, gy), int(len(x))


def correct_retention_times(per_sample: dict[str, dict], p: AlignmentParams):
    """per_sample: sid -> {"mz","rt","intensity"} arrays. Returns (corrected_rt dict, info dict)."""
    sids = list(per_sample)
    ref = p.reference_sample if p.reference_sample in per_sample else max(sids, key=lambda s: len(per_sample[s]["mz"]))
    ref_s = per_sample[ref]
    tree = cKDTree(_coords(ref_s["mz"], ref_s["rt"], p.anchor_mz_ppm, p.anchor_rt_search_sec))
    corrected, info = {}, {"reference_sample": ref, "samples": {}}
    for sid in sids:
        s = per_sample[sid]
        if sid == ref:
            corrected[sid] = s["rt"].copy()
            info["samples"][sid] = {"n_anchors": None, "applied": False, "median_shift_sec": 0.0, "max_abs_shift_sec": 0.0}
            continue
        fn, n_anchor = _fit_shift(s, tree, ref_s["rt"], p)
        if fn is None:
            corrected[sid] = s["rt"].copy()
            info["samples"][sid] = {"n_anchors": n_anchor, "applied": False, "median_shift_sec": 0.0, "max_abs_shift_sec": 0.0}
            continue
        shift = np.interp(s["rt"], fn[0], fn[1])
        shift = np.clip(shift, -p.anchor_rt_search_sec, p.anchor_rt_search_sec)
        corrected[sid] = s["rt"] + shift
        info["samples"][sid] = {
            "n_anchors": n_anchor, "applied": True,
            "median_shift_sec": float(np.median(shift)), "max_abs_shift_sec": float(np.abs(shift).max()),
            "curve_rt": fn[0].tolist(), "curve_shift": fn[1].tolist(),
        }
    return corrected, info


# --------------------------------------------------------------------------
# 2. seeded grouping
# --------------------------------------------------------------------------
def align_features_detailed(
    per_sample_features: dict[str, list[dict]],
    mz_tolerance_ppm: float,
    rt_tolerance_sec: float,
    rt_correction: bool = True,
    reference_sample: str | None = None,
    **anchor_kwargs,
) -> tuple[pd.DataFrame, dict]:
    """
    Align features of all samples into one table (see the module docstring for the algorithm).

    Args:
        per_sample_features: {sample_id: [{"mz", "rt" (seconds), "intensity", ...}, ...]}.
        mz_tolerance_ppm / rt_tolerance_sec: two features join a group only if within both tolerances.
        rt_correction: first correct retention-time drift against `reference_sample` (default: most features).
        **anchor_kwargs: extra AlignmentParams fields (anchor_mz_ppm, min_anchors, ...).

    Returns:
        (table, info): table has feature_id, mz, rt and one intensity column per sample (NaN = not detected);
        info holds the RT-correction details and the number of same-sample conflicts.
    """
    p = AlignmentParams(mz_tolerance_ppm, rt_tolerance_sec, rt_correction, reference_sample, **anchor_kwargs)
    sample_ids = list(per_sample_features.keys())
    arrays = {
        sid: {k: np.array([f[k] for f in feats], dtype=float) for k in ("mz", "rt", "intensity")}
        if feats else {k: np.array([], dtype=float) for k in ("mz", "rt", "intensity")}
        for sid, feats in per_sample_features.items()
    }
    info: dict = {"rt_correction": None, "n_same_sample_conflicts": 0}
    nonempty = {s: a for s, a in arrays.items() if len(a["mz"])}
    if rt_correction and len(nonempty) >= 2:
        corrected, rt_info = correct_retention_times(nonempty, p)
        info["rt_correction"] = rt_info
    else:
        corrected = {s: a["rt"] for s, a in nonempty.items()}

    mz = np.concatenate([arrays[s]["mz"] for s in sample_ids])
    rt_raw = np.concatenate([arrays[s]["rt"] for s in sample_ids])
    rt = np.concatenate([corrected.get(s, arrays[s]["rt"]) for s in sample_ids])
    inten = np.concatenate([arrays[s]["intensity"] for s in sample_ids])
    sidx = np.concatenate([np.full(len(arrays[s]["mz"]), i) for i, s in enumerate(sample_ids)])
    n = len(mz)
    if n == 0:
        return pd.DataFrame(columns=["feature_id", "mz", "rt"] + sample_ids), info

    coords = _coords(mz, rt, mz_tolerance_ppm, rt_tolerance_sec)
    tree = cKDTree(coords)
    assigned = np.zeros(n, dtype=bool)
    groups: list[list[int]] = []
    conflicts = 0
    for i in np.argsort(-inten, kind="stable"):
        if assigned[i]:
            continue
        nb = [j for j in tree.query_ball_point(coords[i], r=1.0, p=np.inf) if not assigned[j]]
        if len(nb) == 1:
            members = [i]
        else:
            d = np.linalg.norm(coords[nb] - coords[i], axis=1)
            best: dict[int, tuple[float, int]] = {}
            for j, dist in zip(nb, d):
                s = sidx[j]
                if s not in best or dist < best[s][0]:
                    best[s] = (dist, j)
            members = [j for _, j in best.values()]
            conflicts += len(nb) - len(members)
        assigned[members] = True
        groups.append(members)
    info["n_same_sample_conflicts"] = int(conflicts)

    cons_mz = np.array([np.median(mz[g]) for g in groups])
    order = np.argsort(cons_mz, kind="stable")
    rows = []
    for rank, gi in enumerate(order):
        g = groups[gi]
        row = {"feature_id": f"F{rank + 1:05d}", "mz": float(cons_mz[gi]), "rt": float(np.median(rt[g]))}
        for sid in sample_ids:
            row[sid] = np.nan
        for j in g:
            row[sample_ids[sidx[j]]] = float(inten[j])
        rows.append(row)
    return pd.DataFrame(rows, columns=["feature_id", "mz", "rt"] + sample_ids), info


def align_features(
    per_sample_features: dict[str, list[dict]],
    mz_tolerance_ppm: float,
    rt_tolerance_sec: float,
    rt_correction: bool = False,
) -> pd.DataFrame:
    """Backward-compatible entry point (same signature plus rt_correction, off by default)."""
    return align_features_detailed(per_sample_features, mz_tolerance_ppm, rt_tolerance_sec, rt_correction)[0]


# --------------------------------------------------------------------------
# previous algorithm, for comparison runs only
# --------------------------------------------------------------------------

def align_features_legacy(
    per_sample_features: dict[str, list[dict]],
    mz_tolerance_ppm: float,
    rt_tolerance_sec: float,
) -> pd.DataFrame:
    """
    Pools every sample's features, sorts by m/z, and groups features into
    consensus rows when they fall within mz_tolerance_ppm AND rt_tolerance_sec
    of a group's running centroid.

    Returns a DataFrame: one row per consensus feature, columns
    ['feature_id', 'mz', 'rt'] plus one intensity column per sample_id
    (NaN where that sample had no matching feature).
    """
    pooled = []
    for sample_id, features in per_sample_features.items():
        for f in features:
            pooled.append({"sample_id": sample_id, "mz": f["mz"], "rt": f["rt"], "intensity": f["intensity"]})

    pooled.sort(key=lambda f: f["mz"])

    groups: list[dict] = []  # each: {"members": [...], "mz_values": [...], "rt_values": [...]}

    for feat in pooled:
        placed = False
        for group in groups:
            group_mz = median(group["mz_values"])
            group_rt = median(group["rt_values"])
            mz_ppm_error = abs(feat["mz"] - group_mz) / group_mz * 1e6
            rt_diff = abs(feat["rt"] - group_rt)

            candidate_mz_values = group["mz_values"] + [feat["mz"]]
            span_ppm = (max(candidate_mz_values) - min(candidate_mz_values)) / group_mz * 1e6

            candidate_rt_values = group["rt_values"] + [feat["rt"]]
            span_rt_sec = max(candidate_rt_values) - min(candidate_rt_values)

            if (mz_ppm_error <= mz_tolerance_ppm
                    and rt_diff <= rt_tolerance_sec
                    and span_ppm <= mz_tolerance_ppm
                    and span_rt_sec <= rt_tolerance_sec):
                group["members"].append(feat)
                group["mz_values"].append(feat["mz"])
                group["rt_values"].append(feat["rt"])
                placed = True
                break
        if not placed:
            groups.append({"members": [feat], "mz_values": [feat["mz"]], "rt_values": [feat["rt"]]})

    rows = []
    all_sample_ids = list(per_sample_features.keys())
    for i, group in enumerate(groups):
        row = {
            "feature_id": f"F{i+1:05d}",
            "mz": median(group["mz_values"]),
            "rt": median(group["rt_values"]),
        }
        for sample_id in all_sample_ids:
            matches = [m["intensity"] for m in group["members"] if m["sample_id"] == sample_id]
            row[sample_id] = matches[0] if matches else None  # first match if duplicate hits in same sample
        rows.append(row)

    return pd.DataFrame(rows)