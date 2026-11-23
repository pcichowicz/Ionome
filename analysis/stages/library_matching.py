"""
Library matching (stage 9 of standards_only, stage 8 of cohort_with_qc): identify compounds by their MS2 spectra.

Two stage classes, one per profile (SpectralLibraryMatchingStage, CohortLibraryMatchingStage), both named
"library_matching". For every compound/feature, extracts its MS2 spectrum from the sample's
mzML and scores it against every reference spectrum in a *pre-built,
independently downloaded* library (e.g. EMBL-MCF from curatr.mcf.embl.de/MS2/export/)

Failure handling
- No match found for one compound/feature -> recoverable, logged,
  library_match_id/match_score stay None, is_correct_match = False.
- Reference library file missing or unparseable -> fatal (nothing in
  this run can be validated without it).
- MS2 extraction failing for every confirmed compound in the whole run
  -> fatal (instrument/acquisition-mode problem, not a single-compound
  problem)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from analysis.utils.constants import NON_SAMPLE_COLS
from analysis.context import LCMSContext
from analysis.pipeline import FatalStageError
from analysis.utils.protocols import MS2SpectrumReader
from analysis.utils.spectral_library import ReferenceSpectrum, load_library, find_best_library_match, classify_confidence, empty_match_record
from core.raw_files import find_raw_file
from analysis.utils.mzml_parsing import load_ms1_scans, compute_ms1_coelution_purity
from analysis.utils.spectral_merging import merge_ms2_spectra
from analysis.utils.mzml_parsing import build_ms2_index, lookup_best_ms2
from analysis.utils.spectral_matching import names_match
import random
from analysis.utils.library_search import SearchParams, LibraryIndex, search, tied_top_names, chance_p_value, preprocess
from analysis.utils.library_qc import apply_library_qc
from analysis.utils.cache_utils import (
    compute_cache_key, hash_file, hash_files, load_filehash_cache, save_filehash_cache,
)

# Not used by any code at the moment. The standards stage (SpectralLibraryMatchingStage) has no cache_key(), so it
# always re-runs, which is safer than a stale match surviving a scoring change. The cohort stage hashes the reference
# library, and the pipeline adds a fingerprint of this module's code to every stage's key.
STAGE_LOGIC_VERSION = "v1"

class SpectralLibraryMatchingStage:
    """
    Match each confirmed standard's MS2 spectrum against the reference library (standards_only).

    Has no cache_key(), so it runs every time. Fatal: unreadable/empty library, or no MS2 for any compound.
    """
    _name = "library_matching"
    # requires: stages whose results this one reads (must be registered earlier; checked before anything runs).
    # runs_after: stages it only reads if present; if they are registered they must come earlier.
    requires: tuple[str, ...] = ('system_suitability',)

    def __init__(
        self,
        ms2_reader: MS2SpectrumReader,
        reference_library_path: Path,
        reference_library_format: str,
        precursor_mz_tolerance_ppm: float = 10.0,
        fragment_mz_tolerance_da: float = 0.02,
        min_match_score: float = 0.7,
    ):
        """
        Args:
            ms2_reader: extracts the best MS2 spectrum for a precursor m/z from a raw file.
            reference_library_path / reference_library_format: the spectral library file and its format ("msp" or "mgf").
            precursor_mz_tolerance_ppm: library entries within this of the compound's m/z are candidates [ppm].
            fragment_mz_tolerance_da: fragment peaks match within this [Da].
            min_match_score: cosine score (0 to 1) required to count as a match.
        """
        self.ms2_reader = ms2_reader
        self.reference_library_path = Path(reference_library_path)
        self.reference_library_format = reference_library_format
        self.precursor_mz_tolerance_ppm = precursor_mz_tolerance_ppm
        self.fragment_mz_tolerance_da = fragment_mz_tolerance_da
        self.min_match_score = min_match_score
        self._library: list[ReferenceSpectrum] | None = None

    def validate_input(self, context: LCMSContext) -> bool:
        """Fatal if the raw folder is not set or the reference library file does not exist."""
        if context.mzml_dir is None:
            raise FatalStageError("library_matching: mzml_directory not set")
        if not self.reference_library_path.exists():
            raise FatalStageError(
                f"library_matching: reference library not found at "
                f"{self.reference_library_path} -- download it from "
                f"curatr.mcf.embl.de/MS2/export/ and set "
                f"library_matching.reference_library_path in the config"
            )
        return True

    def _load_library(self) -> list[ReferenceSpectrum]:
        """Parse the reference library once and keep it; fatal if it is unreadable or has no usable entries."""
        if self._library is None:
            try:
                self._library = load_library(self.reference_library_path, self.reference_library_format)
            except Exception as exc:
                raise FatalStageError(
                    f"library_matching: reference library unreadable "
                    f"({self.reference_library_path}): {exc}"
                ) from exc
            if not self._library:
                raise FatalStageError(
                    f"library_matching: reference library at "
                    f"{self.reference_library_path} parsed to zero usable "
                    f"entries -- check reference_library_format matches the file"
                )
        return self._library

    def execute(self, context: LCMSContext) -> LCMSContext:
        """
        For each confirmed compound: extract its MS2 spectrum, search the library, and record the best match.

        A compound without a match is recoverable (logged, score stays None). If MS2 extraction fails for every
        compound in the run the stage is fatal.
        """
        library = self._load_library()
        ss_results = context.qc_metrics["system_suitability"]["results"]

        results: dict[str, dict] = {}
        warnings: list[str] = []
        input_files: list[str] = [str(self.reference_library_path)]
        n_attempted = 0
        n_with_ms2 = 0
        n_matched = 0
        n_correct = 0

        for sample_id, ss_result in ss_results.items():
            if ss_result.get("status") != "checked":
                continue

            confirmed_matches = [m for m in ss_result["matches"] if m["confirmed"]]
            if not confirmed_matches:
                continue

            mzml_path = find_raw_file(context.mzml_dir, sample_id)
            if mzml_path is None:
                warnings.append(f"{sample_id}: raw file (.mzML/.mzXML) missing, library matching skipped")
                continue
            input_files.append(str(mzml_path))

            sample_matches = []
            for match in confirmed_matches:
                n_attempted += 1
                compound = match["compound"]
                precursor_mz = match["matched_mz"]

                ms2_spectrum = self.ms2_reader.get_ms2_spectrum(
                    mzml_path, precursor_mz, precursor_tolerance_da=0.01
                )

                if not ms2_spectrum:
                    warnings.append(
                        f"{sample_id}: no MS2 spectrum extracted for {compound!r} "
                        f"(m/z {precursor_mz}) -- library matching skipped for this compound"
                    )
                    sample_matches.append(
                        empty_match_record(compound, precursor_mz, "no_ms2_spectrum")
                    )
                    continue

                n_with_ms2 += 1
                found = find_best_library_match(
                    ms2_spectrum,
                    precursor_mz,
                    library,
                    self.precursor_mz_tolerance_ppm,
                    self.fragment_mz_tolerance_da,
                )

                if found is None:
                    warnings.append(
                        f"{sample_id}: no library entry within "
                        f"{self.precursor_mz_tolerance_ppm} ppm of {compound!r} "
                        f"(m/z {precursor_mz}) -- no match found"
                    )
                    sample_matches.append(
                        {
                            "compound": compound,
                            "expected_mz": precursor_mz,
                            "ms2_spectrum": ms2_spectrum,
                            "n_ms2_peaks": len(ms2_spectrum),
                            "library_match_id": None,
                            "match_compound_name": None,
                            "match_score": None,
                            "match_adduct": None,
                            "known_identity": compound,
                            "is_correct_match": False,
                            "status": "no_precursor_candidates",
                        }
                    )
                    continue

                ref, score = found
                n_matched += 1
                passes_threshold = score >= self.min_match_score
                correct = passes_threshold and names_match(compound, ref.compound_name)
                if correct:
                    n_correct += 1
                elif passes_threshold:
                    warnings.append(
                        f"{sample_id}: top match for {compound!r} is "
                        f"{ref.compound_name!r} (score {score:.3f}) -- name mismatch"
                    )
                else:
                    warnings.append(
                        f"{sample_id}: best match for {compound!r} scored "
                        f"{score:.3f}, below threshold {self.min_match_score}"
                    )

                sample_matches.append(
                    {
                        "compound": compound,
                        "expected_mz": precursor_mz,
                        "ms2_spectrum": ms2_spectrum,
                        "n_ms2_peaks": len(ms2_spectrum),
                        "library_match_id": ref.library_id,
                        "match_compound_name": ref.compound_name,
                        "match_score": score,
                        "match_adduct": ref.adduct,
                        "known_identity": compound,
                        "is_correct_match": correct,
                        "status": "matched" if passes_threshold else "below_threshold",
                    }
                )

            results[sample_id] = {"status": "checked", "matches": sample_matches}

        if n_attempted > 0 and n_with_ms2 == 0:
            raise FatalStageError(
                "library_matching: could not extract an MS2 spectrum for any "
                f"confirmed compound across all {n_attempted} attempted -- "
                "likely an acquisition-mode/precursor-list problem, not a "
                "single-compound problem"
            )

        context.qc_metrics[self._name] = {
            "results": results,
            "n_attempted": n_attempted,
            "n_with_ms2": n_with_ms2,
            "n_matched": n_matched,
            "n_correct": n_correct,
            "validation_rate": (n_correct / n_attempted) if n_attempted else None,
            "reference_library_size": len(library),
        }
        context.log_step(
            self._name,
            parameters={
                "reference_library_path": str(self.reference_library_path),
                "reference_library_format": self.reference_library_format,
                "precursor_mz_tolerance_ppm": self.precursor_mz_tolerance_ppm,
                "fragment_mz_tolerance_da": self.fragment_mz_tolerance_da,
                "min_match_score": self.min_match_score,
            },
            metrics={
                "n_attempted": n_attempted,
                "n_with_ms2": n_with_ms2,
                "n_matched": n_matched,
                "n_correct": n_correct,
            },
            warnings=warnings,
            input_files=input_files,
        )
        return context

    def validate_output(self, context: LCMSContext) -> bool:
        """The stage left its metrics on the context."""
        return self._name in context.qc_metrics

class CohortLibraryMatchingStage:
    """
    Identify adduct groups of the annotated feature table by library search (cohort_with_qc).

    Order of work (cheap -> expensive):
      1. vectorised precursor-m/z prefilter against the library index -- only
         features with >= 1 library candidate are looked at further (typically a
         few % of the table), so MS2 and MS1 data is only read for those;
      2. per group, every candidate member ion is searched (not just the
         "primary"), so [2M-H]- / acetate / chloride library entries can match
         the ion they describe;
      3. MS2 comes from non-blank samples only, from spectra triggered near the
         feature's retention time, merged with a >=2-spectra rule on fragments;
      4. scoring: precursor removed, sqrt intensities, minimum matched peaks;
         near-ties are reported together.
    """
    _name = "library_matching"
    # requires: stages whose results this one reads (must be registered earlier; checked before anything runs).
    # runs_after: stages it only reads if present; if they are registered they must come earlier.
    requires: tuple[str, ...] = ('adduct_annotation',)

    def __init__(
        self,
        reference_library_path: Path,
        reference_library_format: str,
        precursor_mz_tolerance_ppm: float = 10.0,
        fragment_mz_tolerance_da: float = 0.02,
        min_match_score: float = 0.7,
        min_purity_high_confidence: float = 0.8,
        min_purity_tentative: float = 0.5,
        min_relative_intensity: float = 0.01,
        isolation_window_da: float = 1.0,
        ms2_precursor_tolerance_da: float = 0.01,
        ms2_rt_tolerance_sec: float | None = 15.0,
        min_matched_peaks: int = 3,
        intensity_power: float = 0.5,
        remove_precursor: bool = True,
        tie_margin: float = 0.02,
        min_ms2_spectra_for_peak: int = 2,
        library_qc: bool = True,
        n_null_refs: int = 0,
        blank_roles: tuple[str, ...] = ("blank", "solvent_blank"),
        seed: int = 0,
    ):
        """
        Args:
            reference_library_path / reference_library_format: spectral library file and its format ("msp" or "mgf").
            precursor_mz_tolerance_ppm: library candidates lie within this of the ion's m/z [ppm].
            fragment_mz_tolerance_da: fragment peaks match within this [Da].
            min_match_score: score (0 to 1) required to report a match.
            min_purity_high_confidence / min_purity_tentative: purity cut-offs for "high_confidence" / "tentative".
            min_relative_intensity: fragment peaks below this fraction of the base peak are dropped from the chosen MS2 spectrum.
            isolation_window_da: precursor isolation window used to compute purity [Da].
            ms2_precursor_tolerance_da / ms2_rt_tolerance_sec: which MS2 scans belong to a feature
                (precursor m/z and distance from the feature's retention time; None = ignore RT).
            min_matched_peaks: a "high_confidence" match resting on fewer matched fragment peaks is downgraded to "tentative".
            intensity_power: intensity weighting in the score (0.5 = square root).
            remove_precursor: drop the precursor peak before scoring.
            tie_margin: hits within this score of the best are reported together.
            min_ms2_spectra_for_peak: a fragment must appear in this many merged spectra to be kept.
            library_qc: check the library's formulas against its m/z values and correct mislabelled adducts.
            n_null_refs: number of unrelated library entries (precursor not close) scored to estimate a chance-match p-value (0 = off).
            blank_roles: sample roles whose spectra are never used.
            seed: random seed for the null references (results are reproducible).
        """
        self.reference_library_path = Path(reference_library_path)
        self.reference_library_format = reference_library_format
        self.precursor_mz_tolerance_ppm = precursor_mz_tolerance_ppm
        self.fragment_mz_tolerance_da = fragment_mz_tolerance_da
        self.min_match_score = min_match_score
        self.min_purity_high_confidence = min_purity_high_confidence
        self.min_purity_tentative = min_purity_tentative
        self.min_relative_intensity = min_relative_intensity
        self.isolation_window_da = isolation_window_da
        self.ms2_precursor_tolerance_da = ms2_precursor_tolerance_da
        self.ms2_rt_tolerance_sec = ms2_rt_tolerance_sec
        self.min_ms2_spectra_for_peak = min_ms2_spectra_for_peak
        self.library_qc = library_qc
        self.n_null_refs = n_null_refs
        self.blank_roles = tuple(blank_roles)
        self.seed = seed
        self.search_params = SearchParams(
            precursor_ppm=precursor_mz_tolerance_ppm,
            fragment_da=fragment_mz_tolerance_da,
            intensity_power=intensity_power,
            remove_precursor=remove_precursor,
            min_matched_peaks=min_matched_peaks,
            tie_margin=tie_margin,
        )
        self._library: list[ReferenceSpectrum] | None = None
        self._library_qc_report: dict | None = None

    def _annotated_path(self, context: LCMSContext) -> Path:
        """Annotated feature table written by adduct_annotation (this stage's input)."""
        return context.featurejson_dir / f"{context.study_id}_annotated_features.parquet"

    def cache_key(self, context: LCMSContext) -> str:
        """Everything the result depends on: parameters, the library file, the annotated table,
        the sample table and the raw files MS1/MS2 data is read from."""
        hash_cache = load_filehash_cache(context.cache_dir)
        raw_paths = [p for p in (find_raw_file(context.mzml_dir, sid)
                                 for sid in context.sample_metadata["sample_id"])
                     if p is not None and Path(p).exists()]
        params = {k: v for k, v in vars(self).items()
                  if not k.startswith("_") and k != "search_params"}
        payload = {
            "params": params,
            "search_params": vars(self.search_params),
            "reference_library_hash": hash_file(self.reference_library_path, hash_cache),
            "annotated_features_hash": hash_file(self._annotated_path(context), hash_cache),
            "sample_metadata_hash": hash_file(context.sample_metadata_path, hash_cache),
            "raw_hashes": hash_files(raw_paths, hash_cache),
        }
        save_filehash_cache(context.cache_dir, hash_cache)
        return compute_cache_key(payload)

    def validate_input(self, context: LCMSContext) -> bool:
        """Fatal if the annotated feature table or the reference library file is missing."""
        if not self._annotated_path(context).exists():
            raise FatalStageError(f"library_matching: annotated features not found at {self._annotated_path(context)}")
        if not self.reference_library_path.exists():
            raise FatalStageError(f"library_matching: reference library not found at {self.reference_library_path}")
        return True

    def _load_library(self) -> list[ReferenceSpectrum]:
        """Parse the library once (optionally sanity-checked and corrected by library_qc); fatal if it has no entries."""
        if self._library is None:
            library = load_library(self.reference_library_path, self.reference_library_format)
            if not library:
                raise FatalStageError("library_matching: reference library parsed to zero usable entries")
            if self.library_qc:
                library, self._library_qc_report = apply_library_qc(library)
            self._library = library
        return self._library

    def execute(self, context: LCMSContext) -> LCMSContext:
        """
        Search every adduct group that has a library candidate and record the best match per group.

        Result per group (keyed by str(adduct_group_id)): matched compound, score, matched peak count, confidence,
        purity and any tied names. MS1/MS2 data of each sample is read at most once and only when needed.
        Blank samples never contribute spectra.
        """
        library = self._load_library()
        index = LibraryIndex(library)
        p = self.search_params
        rng = random.Random(self.seed)
        table = pd.read_parquet(self._annotated_path(context))

        role_map = context.sample_metadata.set_index("sample_id")["sample_role"].to_dict()
        sample_cols = [c for c in table.columns if c not in NON_SAMPLE_COLS]
        # blank injections must not contribute MS2 spectra or purity values
        bio_cols = [c for c in sample_cols if role_map.get(c) not in self.blank_roles]

        warnings: list[str] = []
        if self._library_qc_report:
            for c in self._library_qc_report["corrected"]:
                warnings.append(f"library_matching: library entry {c['name']!r} adduct corrected {c['from']} -> {c['to']}")
            for e in self._library_qc_report["excluded"]:
                warnings.append(f"library_matching: library entry {e['name']!r} excluded ({e['problem']})")

        # references whose spectrum is only the precursor (Leucine, Glycine, ...) can
        # never be confirmed by fragment matching -- say so instead of silently never matching
        n_unmatchable = sum(1 for r in library if not preprocess(r.peaks, r.precursor_mz, p))
        n_few_peaks = sum(1 for r in library if 0 < len(preprocess(r.peaks, r.precursor_mz, p)) < p.min_matched_peaks)
        if n_unmatchable:
            warnings.append(
                f"library_matching: {n_unmatchable} of {len(library)} library spectra contain no fragment "
                f"peak besides the precursor and cannot be matched by MS2; {n_few_peaks} more have fewer than "
                f"{p.min_matched_peaks} (matched on all of their peaks). MS1 (m/z + RT) is the only evidence for these."
            )

        # ---- 1. cheap prefilter: which features can possibly match the library?
        cand_mask = index.candidate_mask(table["mz"].to_numpy(), self.precursor_mz_tolerance_ppm)
        candidates = table[cand_mask]
        n_blank_skipped = 0
        if "blank_flagged" in candidates.columns:
            flagged = candidates["blank_flagged"].fillna(False).astype(bool)
            n_blank_skipped = int(flagged.sum())
            candidates = candidates[~flagged]

        ms1_scan_cache: dict[str, list[dict]] = {}

        def get_ms1_scans(sid: str) -> list[dict]:
            """MS1 scans of a sample, read on first use and cached for the rest of the stage ([] if no raw file)."""
            if sid not in ms1_scan_cache:
                raw_path = find_raw_file(context.mzml_dir, sid)
                ms1_scan_cache[sid] = load_ms1_scans(raw_path) if raw_path else []
            return ms1_scan_cache[sid]

        ms2_index_cache: dict[str, list] = {}

        def get_ms2_index(sid: str) -> list:
            """MS2 scan index of a sample, read on first use and cached for the rest of the stage ([] if no raw file)."""
            if sid not in ms2_index_cache:
                raw_path = find_raw_file(context.mzml_dir, sid)
                ms2_index_cache[sid] = build_ms2_index(raw_path) if raw_path else []
            return ms2_index_cache[sid]

        results: dict[str, dict] = {}
        confidence_counts = {"high_confidence": 0, "tentative": 0, "low_confidence": 0, "no_match": 0}
        n_with_ms2 = 0
        n_matched = 0

        # groups with no library candidate at all: record and move on (no I/O)
        groups_all = table["adduct_group_id"].unique()
        groups_cand = set(candidates["adduct_group_id"].unique())
        for gid in groups_all:
            if gid not in groups_cand:
                results[str(gid)] = {"status": "no_precursor_candidates", "purity": None}
                confidence_counts["no_match"] += 1

        # ---- 2.-4. per group, search every candidate member ion
        for gid, members in tqdm(candidates.groupby("adduct_group_id"), total=len(groups_cand), desc=self._name):
            best = None   # (score, row, hits, n_spectra)
            any_ms2 = False
            for _, row in members.iterrows():
                present = [sid for sid in bio_cols if pd.notna(row[sid])]
                spectra = []
                for sid in present:
                    spec = lookup_best_ms2(
                        get_ms2_index(sid), row["mz"], self.ms2_precursor_tolerance_da,
                        self.min_relative_intensity,
                        target_rt_sec=row["rt"], rt_tolerance_sec=self.ms2_rt_tolerance_sec,
                    )
                    if spec:
                        spectra.append(spec)
                if not spectra:
                    continue
                any_ms2 = True
                merged = merge_ms2_spectra(
                    spectra, self.fragment_mz_tolerance_da,
                    min_spectra=self.min_ms2_spectra_for_peak if len(spectra) >= self.min_ms2_spectra_for_peak else 1,
                )
                if not merged:
                    continue
                hits = search(merged, row["mz"], index, p)
                top = hits[0].score if hits else 0.0
                if best is None or top > best[0]:
                    best = (top, row, hits, len(spectra), merged)

            if not any_ms2:
                results[str(gid)] = {"status": "no_ms2_available", "purity": None}
                confidence_counts["no_match"] += 1
                continue
            n_with_ms2 += 1
            if best is None or best[0] <= 0:
                results[str(gid)] = {"status": "no_fragment_match", "purity": None}
                confidence_counts["no_match"] += 1
                continue

            score, row, hits, n_spectra, merged = best
            top_hit = hits[0]
            tied = tied_top_names(hits, p.tie_margin)

            # purity only for the one feature we are actually reporting
            purities = []
            for sid in bio_cols:
                if pd.notna(row[sid]):
                    pur = compute_ms1_coelution_purity(get_ms1_scans(sid), row["mz"], row["rt"], self.isolation_window_da)
                    if pur is not None:
                        purities.append(pur)
            avg_purity = sum(purities) / len(purities) if purities else None

            confidence = classify_confidence(
                score, avg_purity, self.min_match_score,
                self.min_purity_high_confidence, self.min_purity_tentative,
            )
            # a match resting on fewer fragment peaks than min_matched_peaks (the reference
            # itself has only 1-2) is weak evidence however high the cosine is
            evidence_limited = top_hit.n_matched < p.min_matched_peaks
            if evidence_limited and confidence == "high_confidence":
                confidence = "tentative"
            confidence_counts[confidence] += 1
            if score >= self.min_match_score:
                n_matched += 1

            p_chance = None
            if self.n_null_refs and score >= self.min_match_score:
                p_chance = chance_p_value(merged, row["mz"], score, index, p, self.n_null_refs, rng)

            results[str(gid)] = {
                "status": "matched",
                "match_compound_name": " / ".join(tied),
                "match_score": score,
                "match_adduct": top_hit.ref.adduct,
                "n_matched_peaks": top_hit.n_matched,
                "evidence_limited": evidence_limited,
                "tied_candidates": tied,
                "n_contributing_samples": n_spectra,
                "purity": avg_purity,
                "confidence": confidence,
                "matched_feature_id": row["feature_id"],
                "matched_ion_type": row.get("ion_type"),
                "matched_mz": float(row["mz"]),
                "chance_p_value": p_chance,
            }

        context.qc_metrics[self._name] = {
            "results": results,
            "n_groups_attempted": int(len(groups_all)),
            "n_groups_with_library_candidate": int(len(groups_cand)),
            "n_groups_with_ms2": n_with_ms2,
            "n_matched": n_matched,
            "n_blank_flagged_skipped": n_blank_skipped,
            "confidence_counts": confidence_counts,
            "reference_library_size": len(library),
            "n_library_spectra_without_fragments": n_unmatchable,
            "library_qc": self._library_qc_report,
        }
        context.log_step(
            self._name,
            parameters={
                "precursor_mz_tolerance_ppm": self.precursor_mz_tolerance_ppm,
                "fragment_mz_tolerance_da": self.fragment_mz_tolerance_da,
                "min_match_score": self.min_match_score,
                "min_purity_high_confidence": self.min_purity_high_confidence,
                "min_purity_tentative": self.min_purity_tentative,
                "ms2_rt_tolerance_sec": self.ms2_rt_tolerance_sec,
                "min_matched_peaks": p.min_matched_peaks,
                "intensity_power": p.intensity_power,
                "remove_precursor": p.remove_precursor,
                "tie_margin": p.tie_margin,
                "min_ms2_spectra_for_peak": self.min_ms2_spectra_for_peak,
                "library_qc": self.library_qc,
                "n_null_refs": self.n_null_refs,
            },
            metrics={k: v for k, v in context.qc_metrics[self._name].items() if k != "results"},
            warnings=warnings,
            input_files=[str(self._annotated_path(context)), str(self.reference_library_path)],
        )
        return context

    def validate_output(self, context: LCMSContext) -> bool:
        """The stage left its metrics on the context."""
        return self._name in context.qc_metrics
