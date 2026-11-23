"""
Library search for cohort_with_qc: precursor-indexed candidate lookup and a
stricter spectrum score than the legacy cosine_similarity().

Differences from spectral_matching.cosine_similarity (which the standards_only
stage keeps using, unchanged):
  * the precursor peak (+- precursor_exclusion_da) is removed from both
    spectra, so a spectrum that is "just the precursor" can't score high;
  * intensities are sqrt-weighted (intensity_power=0.5) so one dominant peak
    doesn't decide the score;
  * a minimum number of matched fragment peaks is required;
  * all candidates within tie_margin of the best are reported together
    (Leucine / Isoleucine) instead of the first one in file order winning.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from analysis.utils.spectral_matching import match_peaks

Peak = tuple[float, float]


@dataclass(frozen=True)
class SearchParams:
    """Settings of one library search: tolerances, score weighting and the rules described in the module docstring."""
    precursor_ppm: float = 10.0
    fragment_da: float = 0.02
    intensity_power: float = 0.5
    remove_precursor: bool = True
    precursor_exclusion_da: float = 1.0
    min_matched_peaks: int = 3
    tie_margin: float = 0.02


@dataclass
class LibraryHit:
    """One scored library candidate: the reference, its score (0 to 1) and how many fragment peaks matched."""
    ref: object            # ReferenceSpectrum
    score: float
    n_matched: int


def preprocess(peaks: list[Peak], precursor_mz: float, p: SearchParams) -> list[Peak]:
    """Drop non-positive peaks and (optionally) the precursor region, then apply the intensity power weighting."""
    out = []
    for mz, inten in peaks:
        if inten <= 0:
            continue
        if p.remove_precursor and abs(mz - precursor_mz) <= p.precursor_exclusion_da:
            continue
        out.append((mz, inten ** p.intensity_power))
    return out


def score_spectra(
    query_peaks: list[Peak], query_precursor: float,
    ref_peaks: list[Peak], ref_precursor: float,
    p: SearchParams,
) -> tuple[float, int]:
    """Returns (score in [0,1], n_matched_peaks). Score is 0.0 when fewer than
    min(min_matched_peaks, peaks in the reference, peaks in the query) fragment
    peaks match -- not enough evidence to call it a match."""
    q = preprocess(query_peaks, query_precursor, p)
    r = preprocess(ref_peaks, ref_precursor, p)
    if not q or not r:
        return 0.0, 0
    pairs = match_peaks(q, r, p.fragment_da)
    n = len(pairs)
    if n < min(p.min_matched_peaks, len(q), len(r)) or n == 0:
        return 0.0, n
    num = sum(q[qi][1] * r[ri][1] for qi, ri in pairs)
    qn = math.sqrt(sum(i * i for _, i in q))
    rn = math.sqrt(sum(i * i for _, i in r))
    return (min(num / (qn * rn), 1.0) if qn and rn else 0.0), n


class LibraryIndex:
    """Reference spectra sorted by precursor m/z; O(log n) candidate lookup."""

    def __init__(self, library: list):
        """Index a list of ReferenceSpectrum by precursor m/z so candidates are found by binary search."""
        self.library = sorted(library, key=lambda r: r.precursor_mz)
        self.mzs = np.array([r.precursor_mz for r in self.library], dtype=float)

    def _bounds(self, mz, ppm):
        """Index range of library entries within +-ppm of each m/z (works on a single value or an array)."""
        tol = np.asarray(mz) * ppm / 1e6
        return (np.searchsorted(self.mzs, np.asarray(mz) - tol, side="left"),
                np.searchsorted(self.mzs, np.asarray(mz) + tol, side="right"))

    def candidates(self, mz: float, ppm: float) -> list:
        """Reference spectra whose precursor m/z is within `ppm` of `mz`."""
        lo, hi = self._bounds(mz, ppm)
        return self.library[int(lo):int(hi)]

    def candidate_mask(self, mzs, ppm: float) -> np.ndarray:
        """Vectorised: which of these m/z values have >= 1 library candidate."""
        lo, hi = self._bounds(np.asarray(mzs, dtype=float), ppm)
        return hi > lo


def search(query_peaks: list[Peak], query_mz: float, index: LibraryIndex, p: SearchParams) -> list[LibraryHit]:
    """Score every candidate for the query; returns hits best first (ties broken by name, so results are deterministic)."""
    hits = []
    for ref in index.candidates(query_mz, p.precursor_ppm):
        score, n = score_spectra(query_peaks, query_mz, ref.peaks, ref.precursor_mz, p)
        hits.append(LibraryHit(ref, score, n))
    # deterministic: score desc, then name
    hits.sort(key=lambda h: (-h.score, h.ref.compound_name, h.ref.library_id))
    return hits


def tied_top_names(hits: list[LibraryHit], margin: float) -> list[str]:
    """Distinct names of all hits within `margin` of the best score (best first)."""
    if not hits or hits[0].score <= 0:
        return []
    best = hits[0].score
    names: list[str] = []
    for h in hits:
        if h.score >= best - margin and h.ref.compound_name not in names:
            names.append(h.ref.compound_name)
    return names


def chance_p_value(
    query_peaks: list[Peak], query_mz: float, observed: float,
    index: LibraryIndex, p: SearchParams, n_null: int, rng: random.Random,
) -> float | None:
    """How often does this query score >= `observed` against library entries
    whose precursor is NOT close to it? Add-one-corrected empirical p-value.
    A background-similarity estimate, not a proper FDR (the null is unrelated
    compounds, so it is optimistic for same-class lookalikes)."""
    pool = [r for r in index.library if abs(r.precursor_mz - query_mz) > 1.5 * 2]
    if not pool or n_null <= 0:
        return None
    sample = rng.sample(pool, min(n_null, len(pool)))
    ge = sum(1 for r in sample
             if score_spectra(query_peaks, query_mz, r.peaks, r.precursor_mz, p)[0] >= observed)
    return (1 + ge) / (1 + len(sample))
