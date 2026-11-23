"""
Adduct / isotope grouping for the aligned feature table (cohort_with_qc).

Same idea as adducts.annotate_adducts (features that co-elute, differ by a known
m/z relationship and whose intensities correlate across samples are the same
compound), implemented differently:

  * candidate pairs come from one binary search per (feature, relationship) instead
    of testing all ~N^2/2 pairs in Python (19.4k features: ~188M iterations before);
  * correlations are computed vectorised for the surviving pairs only, optionally on
    log intensities and on the gap-filled table (no sparse-NaN rows);
  * connected components use scipy;
  * LABELS come from an ion hypothesis, not from "most intense = [M-H]-":
    for each group every member is tried as every adduct to get a neutral mass M, the
    hypothesis that explains the most members wins. A group with no supporting
    evidence is labelled "assumed", never presented as identified.

Output columns: adduct_group_id, ion_type, neutral_mass, ion_evidence
(ion_evidence: "supported" | "assumed" | "none"; "none" = member the chosen hypothesis
cannot explain).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from analysis.utils.adducts import AdductDef, ISOTOPE_SPACING, MAX_ISOTOPE_ORDER
from analysis.utils.constants import NON_SAMPLE_COLS

# name -> definition. m/z = (multiplier * M + mass_shift) / charge
ADDUCT_REGISTRY: dict[str, AdductDef] = {a.name: a for a in [
    AdductDef("[M-H]-",       -1.007276),
    AdductDef("[M+Cl]-",      34.968853),
    AdductDef("[M+FA-H]-",    44.998201),
    AdductDef("[M+CH3COO]-",  59.013854),              # acetate: 12 entries in EMBL-MCF
    AdductDef("[M-H2O-H]-",  -19.017841),
    AdductDef("[M+Na-2H]-",   20.974669),
    AdductDef("[M-2H]2-",     -2.014552, 1, 2),        # doubly charged: 5 entries in EMBL-MCF
    AdductDef("[2M-H]-",      -1.007276, 2),
]}
DEFAULT_COHORT_ADDUCTS = list(ADDUCT_REGISTRY)
LEGACY_COHORT_ADDUCTS = ["[M-H]-", "[M+Cl]-", "[M+FA-H]-", "[M-H2O-H]-", "[2M-H]-"]   # = adducts.NEGATIVE_ADDUCTS


def build_adduct_table(names: list[str] | None) -> list[AdductDef]:
    """
    Adduct definitions for the given names (None = all in ADDUCT_REGISTRY).

    Raises:
        ValueError: an unknown name, or a list without "[M-H]-" (the default assumption for ungrouped features).
    """
    names = names or DEFAULT_COHORT_ADDUCTS
    unknown = [n for n in names if n not in ADDUCT_REGISTRY]
    if unknown:
        raise ValueError(f"unknown adduct(s) {unknown}; known: {list(ADDUCT_REGISTRY)}")
    if "[M-H]-" not in names:
        raise ValueError("the adduct list must contain '[M-H]-' (the default assumption in negative mode)")
    return [ADDUCT_REGISTRY[n] for n in names]


# --------------------------------------------------------------------------
# relationships between two features: m/z_b = slope * m/z_a + intercept
# --------------------------------------------------------------------------
def _relations(adduct_table: list[AdductDef]):
    """(role_a, role_b, slope, intercept), in the same priority order as the original
    _match_delta: isotopes, then monomer->monomer, then monomer->multimer."""
    rels = []
    for n in range(1, MAX_ISOTOPE_ORDER + 1):
        rels.append((None, f"isotope+{n}", 1.0, n * ISOTOPE_SPACING))
    monos = [a for a in adduct_table if a.multiplier == 1]
    for x in monos:
        for y in monos:
            if x.name != y.name:
                rels.append((x.name, y.name, *_affine(x, y)))
    for x in monos:
        for y in [a for a in adduct_table if a.multiplier > 1]:
            rels.append((x.name, y.name, *_affine(x, y)))
    return rels


def _affine(x: AdductDef, y: AdductDef) -> tuple[float, float]:
    """Slope and intercept so that mz_y = slope * mz_x + intercept for the same molecule seen as adduct x and as adduct y."""
    # M = (mz_a * z_x - s_x) / m_x ;  mz_b = (m_y * M + s_y) / z_y
    slope = (y.multiplier * x.charge) / (x.multiplier * y.charge)
    intercept = (y.mass_shift - y.multiplier * x.mass_shift / x.multiplier) / y.charge
    return slope, intercept


def find_candidate_edges(mz: np.ndarray, rt: np.ndarray, adduct_table: list[AdductDef],
                         mz_ppm: float, rt_tol: float):
    """Pairs (a, b), mz_a < mz_b (rows must be sorted by m/z), |rt_a - rt_b| <= rt_tol, whose m/z
    difference matches a known relationship. Returns (ia, ib, relation_index); one row per pair,
    the first matching relationship wins (as in the original)."""
    n = len(mz)
    tol = mz * mz_ppm / 1e6
    rels = _relations(adduct_table)
    ias, ibs, rks = [], [], []
    for k, (_, _, slope, intercept) in enumerate(rels):
        target = slope * mz + intercept
        lo = np.searchsorted(mz, target - tol, side="left")
        hi = np.searchsorted(mz, target + tol, side="right")
        cnt = np.maximum(hi - lo, 0)
        if cnt.sum() == 0:
            continue
        ia = np.repeat(np.arange(n), cnt)
        offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ib = np.repeat(lo, cnt) + offs
        keep = (ib > ia) & (np.abs(rt[ia] - rt[ib]) <= rt_tol)
        ias.append(ia[keep]); ibs.append(ib[keep]); rks.append(np.full(int(keep.sum()), k))
    if not ias:
        e = np.array([], dtype=int)
        return e, e, e, rels
    ia, ib, rk = np.concatenate(ias), np.concatenate(ibs), np.concatenate(rks)
    order = np.lexsort((rk, ib, ia))
    ia, ib, rk = ia[order], ib[order], rk[order]
    first = np.ones(len(ia), bool)
    first[1:] = (ia[1:] != ia[:-1]) | (ib[1:] != ib[:-1])
    return ia[first], ib[first], rk[first], rels


def pair_correlations(X: np.ndarray, ia: np.ndarray, ib: np.ndarray, min_samples: int, chunk: int = 200_000) -> np.ndarray:
    """Pearson r over the samples where both features have a value; NaN if fewer than
    min_samples shared samples or zero variance (those pairs are rejected)."""
    out = np.full(len(ia), np.nan)
    for s in range(0, len(ia), chunk):
        a, b = X[ia[s:s + chunk]], X[ib[s:s + chunk]]
        ok = ~np.isnan(a) & ~np.isnan(b)
        n = ok.sum(axis=1)
        a0, b0 = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ma, mb = a0.sum(1) / n, b0.sum(1) / n
            da, db = np.where(ok, a - ma[:, None], 0.0), np.where(ok, b - mb[:, None], 0.0)
            va, vb = (da ** 2).sum(1), (db ** 2).sum(1)
            r = (da * db).sum(1) / np.sqrt(va * vb)
        r[(n < min_samples) | (va <= 0) | (vb <= 0)] = np.nan
        out[s:s + chunk] = r
    return out


# --------------------------------------------------------------------------
# hypothesis-based labelling
# --------------------------------------------------------------------------
def _label_component(members, mz, mean_int, iso_label, adducts, ppm):
    """members: row positions. Returns {row: (label, neutral_mass, evidence)}."""
    iso = [i for i in members if i in iso_label]
    mono = [i for i in members if i not in iso_label] or list(members)
    best = None   # (score, M, {row: AdductDef})
    for i in mono:
        for A in adducts:
            M = (mz[i] * A.charge - A.mass_shift) / A.multiplier
            if M <= 0:
                continue
            explained: dict[str, tuple[float, int, AdductDef]] = {}
            for j in mono:
                cand = None
                for B in adducts:
                    err = abs(mz[j] - (B.multiplier * M + B.mass_shift) / B.charge)
                    if err <= mz[j] * ppm / 1e6 and (cand is None or err < cand[0]):
                        cand = (err, B)
                if cand and (cand[1].name not in explained or cand[0] < explained[cand[1].name][0]):
                    explained[cand[1].name] = (cand[0], j, cand[1])
            assign = {j: B for _, j, B in explained.values()}
            score = (len(assign), any(B.name == "[M-H]-" for B in assign.values()),
                     sum(mean_int[j] for j in assign))
            if best is None or score > best[0]:
                best = (score, M, assign)
    out = {}
    (n_expl, _, _), M, assign = best if best else ((0, False, 0.0), None, {})
    if n_expl >= 2:
        primary = max(assign, key=lambda j: mean_int[j])
        for j, B in assign.items():
            out[j] = (B.name + (" (primary)" if j == primary else ""), M, "supported")
        for j in mono:
            if j not in assign:
                out[j] = ("unassigned", np.nan, "none")
    else:
        # nothing corroborates any hypothesis: the most intense non-isotope ion is ASSUMED [M-H]-
        default = next(a for a in adducts if a.name == "[M-H]-")
        primary = max(mono, key=lambda j: mean_int[j])
        M = (mz[primary] * default.charge - default.mass_shift) / default.multiplier
        tag = "[M-H]- (assumed, ungrouped)" if len(members) == 1 else "[M-H]- (assumed, primary)"
        out[primary] = (tag, M, "assumed")
        for j in mono:
            if j != primary:
                out[j] = ("unassigned", np.nan, "none")
    group_evidence = "supported" if n_expl >= 2 else "assumed"
    for j in iso:
        out[j] = (iso_label[j], M, group_evidence)
    return out


# --------------------------------------------------------------------------
def annotate_adducts_v2(
    aligned: pd.DataFrame,
    mz_tolerance_ppm: float = 10.0,
    rt_tolerance_sec: float = 3.0,
    correlation_threshold: float = 0.75,
    adduct_table: list[AdductDef] | None = None,
    filled: pd.DataFrame | None = None,
    log_correlation: bool = True,
    min_correlation_samples: int = 6,
) -> tuple[pd.DataFrame, dict]:
    """
    Group co-eluting, correlated features that differ by a known adduct/isotope relation and label them.

    Args:
        aligned: aligned feature table, sorted by m/z (as feature_alignment writes it).
        mz_tolerance_ppm / rt_tolerance_sec: pair candidates must be within both.
        correlation_threshold: minimum intensity correlation across samples (0 to 1).
        adduct_table: relations to test; None = all registered adducts.
        filled: gap-filled table to correlate on instead of the raw (NaN-containing) one.
        log_correlation: correlate log intensities.
        min_correlation_samples: pairs sharing fewer samples cannot form a link.

    Returns:
        (annotated table, diagnostics): the table gains adduct_group_id, ion_type, neutral_mass and ion_evidence
        ("supported", "assumed" or "none").

    Raises:
        ValueError: `aligned` is not sorted by m/z.
    """
    adduct_table = adduct_table or build_adduct_table(None)
    df = aligned.reset_index(drop=True)
    if not np.all(np.diff(df["mz"].to_numpy()) >= 0):
        raise ValueError("annotate_adducts_v2 expects rows sorted by m/z (the alignment stage writes them that way)")
    sample_cols = [c for c in df.columns if c not in NON_SAMPLE_COLS]
    mz, rt = df["mz"].to_numpy(float), df["rt"].to_numpy(float)

    ia, ib, rk, rels = find_candidate_edges(mz, rt, adduct_table, mz_tolerance_ppm, rt_tolerance_sec)
    n_candidates = len(ia)

    src = df[sample_cols]
    if filled is not None:
        f = filled.set_index("feature_id")[sample_cols].reindex(df["feature_id"])
        src = f.set_axis(df.index)
    X = src.to_numpy(float)
    if log_correlation:
        with np.errstate(invalid="ignore", divide="ignore"):
            X = np.where(X > 0, np.log2(X), np.nan)
    r = pair_correlations(X, ia, ib, min_correlation_samples)
    keep = r >= correlation_threshold          # NaN compares False -> rejected
    ia, ib, rk = ia[keep], ib[keep], rk[keep]

    n = len(df)
    graph = coo_matrix((np.ones(len(ia)), (ia, ib)), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    # compact ids in order of first appearance (matches the original numbering)
    _, first_idx = np.unique(comp, return_index=True)
    remap = {c: k for k, c in enumerate(comp[np.sort(first_idx)])}
    gid = np.array([remap[c] for c in comp])

    iso_label = {}
    for a, b, k in zip(ia, ib, rk):
        if rels[k][0] is None:
            iso_label[int(b)] = rels[k][1]

    mean_int = np.nan_to_num(df[sample_cols].mean(axis=1, skipna=True).to_numpy(float), nan=0.0)
    ion_type = np.empty(n, dtype=object); neutral = np.full(n, np.nan); evidence = np.empty(n, dtype=object)
    order = np.argsort(gid, kind="stable")
    bounds = np.flatnonzero(np.diff(gid[order])) + 1
    for members in np.split(order, bounds):
        res = _label_component([int(m) for m in members], mz, mean_int, iso_label, adduct_table, mz_tolerance_ppm)
        for j, (label, M, ev) in res.items():
            ion_type[j], neutral[j], evidence[j] = label, M, ev

    out = df.copy()
    out["adduct_group_id"] = gid
    out["ion_type"] = ion_type
    out["neutral_mass"] = neutral
    out["ion_evidence"] = evidence
    stats = {
        "n_candidate_pairs": int(n_candidates), "n_edges": int(len(ia)),
        "n_groups_supported": int(out.loc[out["ion_evidence"] == "supported", "adduct_group_id"].nunique()),
        "n_groups_assumed": int(out.loc[out["ion_evidence"] == "assumed", "adduct_group_id"].nunique()),
        "n_unassigned_members": int((out["ion_evidence"] == "none").sum()),
    }
    return out, stats
