"""
Feature selection, normalization and matrix preparation shared by the
volcano, heatmap and PLS-DA stages.

Replaces direct calls to prepare_pca_matrix() in StatPlots.py. prepare_pca_matrix
itself is left untouched (summary_stats_plotting still uses it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StatsPrepParams:
    """
    Options for preparing the feature table for group statistics (volcano, heatmap, PLS-DA).

    Order of operations: drop blank-flagged features -> optional QC RSD filter -> keep one ion per adduct group ->
    (in prepare_stats_matrix) presence filter -> normalization -> fill gaps with half the matrix minimum -> log2(x + 1).
    """
    normalization: str = "pqn"            # "pqn" | "none"
    presence_mode: str = "per_group"      # "per_group" | "global"
    min_presence_frac: float = 0.5        # fraction of samples (of a group, or of all)
    drop_blank_flagged: bool = True
    collapse_adduct_groups: bool = True
    max_qc_rsd: Optional[float] = None    # percent; None = no RSD filter
    use_filled: bool = True               # use the gap-filled table where available

    @classmethod
    def legacy(cls, min_presence_frac: float = 0.5) -> "StatsPrepParams":
        """The earlier, simpler preparation (no normalization, global presence filter, no blank/adduct filtering); for comparison runs."""
        return cls("none", "global", min_presence_frac, False, False, None)


def select_features(
    table: pd.DataFrame,
    bio_cols: list[str],
    qc_cols: list[str],
    params: StatsPrepParams,
) -> tuple[pd.DataFrame, dict]:
    """Row-filter the final feature table. Returns (filtered_table, counts)."""
    counts = {"n_input": int(len(table))}
    out = table

    if params.drop_blank_flagged and "blank_flagged" in out.columns:
        out = out[~out["blank_flagged"].fillna(False).astype(bool)]
    counts["n_after_blank_filter"] = int(len(out))

    if params.max_qc_rsd is not None and len(qc_cols) >= 2:
        q = out[qc_cols]
        mean = q.mean(axis=1)
        rsd = q.std(axis=1) / mean * 100
        keep = (q.notna().sum(axis=1) >= 2) & (rsd <= params.max_qc_rsd)
        out = out[keep]
    counts["n_after_qc_rsd"] = int(len(out))

    if params.collapse_adduct_groups and "adduct_group_id" in out.columns and len(out):
        # one representative ion per group: highest median intensity over
        # biological samples (ties -> first row). Isotopes/adducts of the same
        # compound are not independent features.
        med = out[bio_cols].median(axis=1, skipna=True).fillna(-np.inf)
        order = med.sort_values(ascending=False, kind="stable").index
        keep_idx = out.loc[order].drop_duplicates("adduct_group_id", keep="first").index
        out = out.loc[out.index.isin(keep_idx)]
    counts["n_after_collapse"] = int(len(out))
    return out, counts


def pqn_normalize(intensities: pd.DataFrame, reference_cols: list[str] | None = None) -> pd.DataFrame:
    """Probabilistic quotient normalization on a features x samples frame of
    raw (linear) intensities. NaN stays NaN."""
    ref_cols = [c for c in (reference_cols or intensities.columns) if c in intensities.columns]
    reference = intensities[ref_cols].median(axis=1, skipna=True)
    reference = reference.where(reference > 0)
    quotients = intensities.div(reference, axis=0)
    factors = quotients.median(axis=0, skipna=True)
    factors = factors.where(factors > 0, 1.0).fillna(1.0)
    return intensities.div(factors, axis=1)


def prepare_stats_matrix(
    table: pd.DataFrame,
    sample_cols: list[str],
    role_map: dict[str, str],
    params: StatsPrepParams,
) -> pd.DataFrame:
    """samples x features log2 matrix, same shape contract as prepare_pca_matrix.

    Order: presence filter -> (PQN) -> min/2 imputation -> log2(x+1).
    `table` should already have been through select_features().
    """
    x = table[sample_cols].astype(float)
    notna = x.notna()

    if params.presence_mode == "per_group":
        groups: dict[str, list[str]] = {}
        for c in sample_cols:
            groups.setdefault(role_map.get(c, "?"), []).append(c)
        keep = pd.Series(False, index=x.index)
        for cols in groups.values():
            keep |= notna[cols].mean(axis=1) >= params.min_presence_frac
    else:
        keep = notna.mean(axis=1) >= params.min_presence_frac
    x = x[keep]

    if params.normalization == "pqn" and len(x):
        x = pqn_normalize(x)
    elif params.normalization not in ("pqn", "none"):
        raise ValueError(f"unknown normalization {params.normalization!r}")

    global_min = x.min().min()
    x = x.fillna(global_min / 2)
    return np.log2(x + 1).T
