"""
Chromatogram-based feature detection (replaces the apex-only mass-trace picker
for method="chromatogram"; feature_utils.group_into_features stays as method="legacy").

Pipeline per sample:
  1. mass traces: peaks of consecutive MS1 scans within mass_error_ppm are linked into
     traces. Open traces are kept sorted by m/z and each scan's peaks are matched with
     one binary search (the legacy code compared every peak with every open trace).
  2. each trace's extracted-ion chromatogram is smoothed (Savitzky-Golay) and split at
     valleys: two apexes become two features when the valley between them drops below
     (1 - min_relative_prominence) of the smaller apex. Legacy merged any two peaks of the
     same m/z whose baseline stayed above the noise threshold (isomers!).
  3. each peak's base width is checked against peak_width = [min, max] seconds -- BOTH
     bounds (legacy used only the lower one, via min_scans).
  4. a feature records height (raw apex, as before), area (trapezoid over the peak),
     m/z (intensity-weighted over the peak, not the single apex scan), apex RT, peak
     start/end, scan count and S/N.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter

DEFAULTS = {
    "mass_error_ppm": 5.0,
    "peak_width": [10.0, 60.0],
    "max_gaps": 2,
    "min_scans": 2,
    "noise_threshold": 8000.0,
    "noise_mode": "absolute",          # "absolute" | "adaptive" (snr_threshold x the scan's median intensity)
    "snr_threshold": 3.0,
    "smoothing_points": 5,
    "min_relative_prominence": 0.2,
    "edge_fraction": 0.05,
    "max_width_action": "drop",        # "drop" | "keep" peaks wider than peak_width[1]
    "polarity": "negative",
}


def _scan_threshold(it: np.ndarray, p: dict) -> float:
    """Intensity floor for one scan: a fixed `noise_threshold`, or `snr_threshold` x the scan's median (adaptive mode)."""
    if p["noise_mode"] == "adaptive":
        return float(p["snr_threshold"] * np.median(it)) if len(it) else np.inf
    return float(p["noise_threshold"])


def build_mass_traces(scans: list[dict], p: dict):
    """Returns (trace_id, scan_idx, mz, intensity, scan_threshold) point arrays."""
    ppm = float(p["mass_error_ppm"]) / 1e6
    max_gaps = int(p["max_gaps"])
    open_mz = np.empty(0)
    open_id = np.empty(0, dtype=np.int64)
    open_missed = np.empty(0, dtype=np.int64)
    next_id = 0
    pt_trace, pt_scan, pt_mz, pt_int = [], [], [], []
    thresholds = np.empty(len(scans))

    for si, scan in enumerate(scans):
        mz_all = np.asarray(scan["mz"], dtype=float)
        it_all = np.asarray(scan["intensity"], dtype=float)
        thr = _scan_threshold(it_all, p)
        thresholds[si] = thr
        keep = it_all >= thr
        mz, it = mz_all[keep], it_all[keep]
        n = len(mz)
        extended = np.zeros(len(open_mz), dtype=bool)
        new_mask = np.ones(n, dtype=bool)

        if len(open_mz) and n:
            order = np.argsort(open_mz, kind="stable")
            om = open_mz[order]
            pos = np.searchsorted(om, mz)
            left = np.clip(pos - 1, 0, len(om) - 1)
            right = np.clip(pos, 0, len(om) - 1)
            dl, dr = np.abs(mz - om[left]), np.abs(mz - om[right])
            cand = np.where(dl <= dr, left, right)
            ok = np.minimum(dl, dr) <= mz * ppm
            idx = np.flatnonzero(ok)
            if len(idx):
                idx = idx[np.argsort(-it[idx], kind="stable")]       # strongest peak claims a trace first
                _, first = np.unique(cand[idx], return_index=True)
                win = idx[first]
                slot = order[cand[win]]
                open_mz[slot] = mz[win]
                open_missed[slot] = 0
                extended[slot] = True
                new_mask[win] = False
                pt_trace.append(open_id[slot]); pt_scan.append(np.full(len(win), si))
                pt_mz.append(mz[win]); pt_int.append(it[win])

        # traces not extended this scan accumulate a miss; too many misses close them
        open_missed[~extended] += 1
        alive = open_missed <= max_gaps
        open_mz, open_id, open_missed = open_mz[alive], open_id[alive], open_missed[alive]

        nn = int(new_mask.sum())
        if nn:
            ids = np.arange(next_id, next_id + nn)
            next_id += nn
            open_mz = np.concatenate([open_mz, mz[new_mask]])
            open_id = np.concatenate([open_id, ids])
            open_missed = np.concatenate([open_missed, np.zeros(nn, dtype=np.int64)])
            pt_trace.append(ids); pt_scan.append(np.full(nn, si))
            pt_mz.append(mz[new_mask]); pt_int.append(it[new_mask])

    if not pt_trace:
        e = np.empty(0)
        return e.astype(np.int64), e.astype(np.int64), e, e, thresholds
    return (np.concatenate(pt_trace), np.concatenate(pt_scan).astype(np.int64),
            np.concatenate(pt_mz), np.concatenate(pt_int), thresholds)


def _smooth(y: np.ndarray, points: int) -> np.ndarray:
    """Savitzky-Golay smoothing (2nd order) over an odd window of about `points`; never below zero; too-short input is returned unsmoothed."""
    w = points if points % 2 == 1 else points + 1
    if w < 3 or len(y) < w:
        return y.astype(float)
    return np.clip(savgol_filter(y, w, 2), 0.0, None)


def detect_features_chromatogram(scans: list[dict], params: dict) -> tuple[list[dict], dict]:
    """
    Detect peaks in one sample's MS1 scans (algorithm described in the module docstring).

    Args:
        scans: [{"rt": seconds, "mz": array, "intensity": array}, ...], one per MS1 scan.
        params: options from the feature_detection config section plus "polarity"; missing/None entries use DEFAULTS.

    Returns:
        (features, diagnostics): features are dicts with mz, rt (seconds), intensity and charge (-1 in negative
        mode, +1 otherwise); diagnostics counts traces and why candidates were rejected (too few scans, too narrow, too wide).
    """
    p = {**DEFAULTS, **{k: v for k, v in params.items() if v is not None}}
    lo_w, hi_w = float(p["peak_width"][0]), float(p["peak_width"][1])
    charge = -1 if str(p["polarity"]).lower().startswith("neg") else 1
    diag = {"n_scans": len(scans), "n_traces": 0, "n_candidate_peaks": 0, "n_traces_split": 0,
            "n_rejected_too_few_scans": 0, "n_rejected_too_narrow": 0, "n_rejected_too_wide": 0,
            "n_kept_too_wide": 0, "noise_mode": p["noise_mode"]}
    if len(scans) < 3:
        return [], diag
    scan_rt = np.array([s["rt"] for s in scans], dtype=float)
    dt = float(np.median(np.diff(np.sort(scan_rt))))
    min_scans = max(2, int(p["min_scans"]), int(round(lo_w / dt)))
    med = np.array([np.median(s["intensity"]) if len(s["intensity"]) else 0.0 for s in scans])
    diag["scan_median_intensity"] = float(np.median(med))
    diag["noise_threshold_over_scan_median"] = (float(p["noise_threshold"] / np.median(med))
                                                if np.median(med) > 0 else None)

    tid, sidx, mz, it, thr = build_mass_traces(scans, p)
    if len(tid) == 0:
        return [], diag
    counts = np.bincount(tid)
    diag["n_traces"] = int((counts > 0).sum())
    long_enough = counts[tid] >= min_scans
    diag["n_rejected_too_few_scans"] = int((counts[counts > 0] < min_scans).sum())
    tid, sidx, mz, it = tid[long_enough], sidx[long_enough], mz[long_enough], it[long_enough]
    order = np.lexsort((sidx, tid))
    tid, sidx, mz, it = tid[order], sidx[order], mz[order], it[order]
    bounds = np.flatnonzero(np.diff(tid)) + 1

    features = []
    for sl in np.split(np.arange(len(tid)), bounds):
        if len(sl) == 0:
            continue
        s, m, y_pts = sidx[sl], mz[sl], it[sl]
        s0, s1 = int(s[0]), int(s[-1])
        raw = np.zeros(s1 - s0 + 1)
        raw[s - s0] = y_pts
        wmz = np.zeros(s1 - s0 + 1)
        wmz[s - s0] = m
        times = scan_rt[s0:s1 + 1]
        sm = _smooth(raw, int(p["smoothing_points"]))
        cand, props = find_peaks(sm, prominence=0.0)
        if len(cand) == 0:
            continue
        rel = props["prominences"] >= float(p["min_relative_prominence"]) * sm[cand]
        peaks = cand[rel]
        if len(peaks) == 0:
            peaks = np.array([cand[np.argmax(sm[cand])]])
        diag["n_candidate_peaks"] += len(peaks)
        if len(peaks) > 1:
            diag["n_traces_split"] += 1
        # boundaries: valleys between neighbouring retained apexes; outer edges where the
        # smoothed signal falls under edge_fraction x apex (or the trace ends)
        cuts = [int(a + np.argmin(sm[a:b + 1])) for a, b in zip(peaks[:-1], peaks[1:])]
        for k, a in enumerate(peaks):
            left = cuts[k - 1] if k > 0 else 0
            right = cuts[k] if k < len(cuts) else len(sm) - 1
            floor = float(p["edge_fraction"]) * sm[a]
            while left < a and sm[left] < floor:
                left += 1
            while right > a and sm[right] < floor:
                right -= 1
            seg = slice(left, right + 1)
            n_pts = int((raw[seg] > 0).sum())
            width = (right - left + 1) * dt
            if n_pts < min_scans:
                diag["n_rejected_too_few_scans"] += 1
                continue
            if width < lo_w:
                diag["n_rejected_too_narrow"] += 1
                continue
            if width > hi_w:
                if p["max_width_action"] == "drop":
                    diag["n_rejected_too_wide"] += 1
                    continue
                diag["n_kept_too_wide"] += 1
            apex = left + int(np.argmax(raw[seg]))
            w = raw[seg]
            ok = w > 0
            mz_w = float(np.sum(wmz[seg][ok] * w[ok]) / np.sum(w[ok]))
            height = float(raw[apex])
            features.append({
                "mz": mz_w,
                "rt": float(times[apex]),
                "intensity": height,
                "charge": charge,
                "area": float(np.trapezoid(raw[seg], times[seg])) if right > left else float(height * dt),
                "rt_start": float(times[left]),
                "rt_end": float(times[right]),
                "n_scans": n_pts,
                "snr": float(height / thr[s0 + apex]) if thr[s0 + apex] > 0 else None,
            })
    features.sort(key=lambda f: f["mz"])
    diag["n_features"] = len(features)
    return features, diag
