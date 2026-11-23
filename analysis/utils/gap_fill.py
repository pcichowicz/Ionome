"""
Gap filling: re-extract intensities for consensus features that were not
detected in a sample, straight from that sample's MS1 scans.

After alignment a NaN only means "this sample's feature picker found no peak
above the noise threshold" -- often the compound is there, just small. Reading the
extracted-ion signal at the consensus m/z and the (drift-corrected) expected RT
gives a measured value for imputation instead of an arbitrary constant, in the
spirit of XCMS fillPeaks.

The result is written to a SEPARATE table; the aligned table keeps NaN for
"not detected" so presence/absence filters (and the volcano's min_valid_per_group)
still mean what they say.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def expected_raw_rt(consensus_rt: np.ndarray, curve_rt: list[float] | None, curve_shift: list[float] | None) -> np.ndarray:
    """Invert the sample's RT correction (corrected = raw + shift(raw)) to find
    where a consensus-RT feature should sit in the sample's own raw scans."""
    if not curve_rt:
        return np.asarray(consensus_rt, dtype=float)
    x = np.asarray(curve_rt, dtype=float)
    corrected = x + np.asarray(curve_shift, dtype=float)
    order = np.argsort(corrected)
    return np.interp(consensus_rt, corrected[order], x[order])   # clamps at the ends


def _extract_max(scans: list[dict], mz_lo: np.ndarray, mz_hi: np.ndarray, rt_target: np.ndarray, rt_window: float) -> np.ndarray:
    """Max intensity within [mz_lo, mz_hi] over scans within +-rt_window of rt_target.
    0.0 where nothing was found. Vectorised over features, one pass over the scans."""
    m = len(rt_target)
    best = np.zeros(m)
    order = np.argsort(rt_target, kind="stable")
    rt_sorted = rt_target[order]
    lo_sorted, hi_sorted = mz_lo[order], mz_hi[order]
    best_sorted = np.zeros(m)
    for scan in scans:
        a = np.searchsorted(rt_sorted, scan["rt"] - rt_window, side="left")
        b = np.searchsorted(rt_sorted, scan["rt"] + rt_window, side="right")
        if b <= a:
            continue
        smz, sint = scan["mz"], scan["intensity"]
        if len(smz) == 0:
            continue
        if np.any(np.diff(smz) < 0):
            o = np.argsort(smz)
            smz, sint = smz[o], sint[o]
        lo = np.searchsorted(smz, lo_sorted[a:b], side="left")
        hi = np.searchsorted(smz, hi_sorted[a:b], side="right")
        hit = hi > lo
        if not hit.any():
            continue
        ext = np.append(sint, 0.0)                      # sentinel so hi == len(smz) is a valid index
        idx = np.empty(2 * int(hit.sum()), dtype=np.intp)
        idx[0::2], idx[1::2] = lo[hit], hi[hit]
        vals = np.maximum.reduceat(ext, idx)[0::2]
        seg = best_sorted[a:b]
        seg[hit] = np.maximum(seg[hit], vals)
    best[order] = best_sorted
    return best


def fill_gaps(
    table: pd.DataFrame,
    sample_cols: list[str],
    load_scans: Callable[[str], list[dict] | None],
    rt_curves: dict[str, dict],
    mz_ppm: float,
    rt_window_sec: float,
) -> tuple[pd.DataFrame, dict]:
    """Returns (filled_table, stats). Detected values are kept unchanged; missing
    ones are replaced by the extracted apex intensity, or stay NaN when no peak at all
    lies in the m/z window. load_scans(sample_id) -> MS1 scans (or None if unavailable)."""
    filled = table.copy()
    stats = {"per_sample": {}, "n_missing": 0, "n_filled": 0}
    for sid in sample_cols:
        miss = table[sid].isna().to_numpy()
        n_miss = int(miss.sum())
        if n_miss == 0:
            stats["per_sample"][sid] = {"n_missing": 0, "n_filled": 0}
            continue
        scans = load_scans(sid)
        if not scans:
            stats["per_sample"][sid] = {"n_missing": n_miss, "n_filled": 0, "skipped": "no scans"}
            stats["n_missing"] += n_miss
            continue
        mz = table.loc[miss, "mz"].to_numpy(dtype=float)
        crv = rt_curves.get(sid) or {}
        rt = expected_raw_rt(table.loc[miss, "rt"].to_numpy(dtype=float), crv.get("curve_rt"), crv.get("curve_shift"))
        tol = mz * mz_ppm / 1e6
        vals = _extract_max(scans, mz - tol, mz + tol, rt, rt_window_sec)
        vals[vals <= 0] = np.nan
        filled.loc[miss, sid] = vals
        n_f = int(np.isfinite(vals).sum())
        stats["per_sample"][sid] = {"n_missing": n_miss, "n_filled": n_f}
        stats["n_missing"] += n_miss
        stats["n_filled"] += n_f
    return filled, stats
