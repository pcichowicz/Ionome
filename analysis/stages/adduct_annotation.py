"""
Adduct annotation (stage 7 in both profiles): which features are different ions of the same compound?

Two stage classes, one per profile; both register under the name "adduct_annotation":

  * AdductAnnotationStage (standards_only): for each compound confirmed by system_suitability, compute the expected
    m/z of the other candidate adducts from its neutral mass and look for them among the sample's MS2 precursors.
  * CohortAdductAnnotationStage (cohort_with_qc): infers adduct / isotope / dimer relations directly from the aligned
    feature table (m/z difference + RT co-elution + intensity correlation across samples) and writes
    `features/<study>_annotated_features.parquet` with adduct_group_id, ion_type, neutral_mass and ion_evidence.

Limitation: the adduct tables are negative-mode only (no [M+H]+ and friends), whatever `study.polarity` says.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from analysis.utils.adducts import compute_adduct_mz, compute_neutral_mass
from analysis.context import LCMSContext
from analysis.utils.protocols import PrecursorReader
from analysis.pipeline import FatalStageError
from analysis.stages.system_suitability import find_best_match
from core.ingestion import find_raw_file
from analysis.utils.adducts import annotate_adducts, NEGATIVE_ADDUCTS, AdductDef
from analysis.utils.adduct_graph import annotate_adducts_v2, build_adduct_table
from analysis.utils.cache_utils import load_filehash_cache, save_filehash_cache, hash_file, compute_cache_key

CACHE_SCHEMA_VERSION = "v2"   # v2: hypothesis labelling / extended adduct table (cohort stage)

class AdductAnnotationStage:
    """Look for extra adducts of each confirmed standard (see the module docstring)."""
    _name: str = "adduct_annotation"
    # requires: stages whose results this one reads (must be registered earlier; checked before anything runs).
    # runs_after: stages it only reads if present; if they are registered they must come earlier.
    requires: tuple[str, ...] = ('system_suitability',)

    def __init__(
        self,
        reader: PrecursorReader,
        candidate_adducts: list[str],
        primary_adduct: str = "[M-H]-",
        ppm_tolerance: float = 5.0,
    ):
        """
        Args:
            reader: lists the MS2 precursor m/z values of a raw file.
            candidate_adducts: adduct names to look for, e.g. "[M+Cl]-"; must include `primary_adduct`.
            primary_adduct: the adduct system_suitability matched; its neutral mass seeds the search.
            ppm_tolerance: allowed m/z error [ppm].
        """
        self.reader = reader
        self.candidate_adducts = candidate_adducts
        self.primary_adduct = primary_adduct
        self.ppm_tolerance = ppm_tolerance

    def validate_input(self, context: LCMSContext) -> bool:
        """Fatal if the raw folder is not set."""
        if context.mzml_dir is None:
            raise FatalStageError("adduct_annotation: mzml_directory not set")
        return True

    def execute(self, context: LCMSContext) -> LCMSContext:
        """
        For every confirmed match: neutral mass from the primary adduct, expected m/z of each candidate adduct, and
        a search for it among the sample's precursors. Records the adducts found per compound.
        """
        ss_results = context.qc_metrics["system_suitability"]["results"]
        results: dict[str, dict] = {}
        warnings: list[str] = []
        n_compounds_checked = 0
        n_with_extra_adducts = 0

        for sample_id, ss_result in ss_results.items():
            if ss_result.get("status") != "checked":
                continue

            confirmed_matches = [m for m in ss_result["matches"] if m["confirmed"]]
            if not confirmed_matches:
                continue

            mzml_path = find_raw_file(context.mzml_dir, sample_id)
            if not mzml_path.exists():
                warnings.append(f"{sample_id}: mzML missing, adduct annotation skipped")
                continue

            observed_mzs = self.reader.get_precursor_mzs(mzml_path)
            compounds = []

            for match in confirmed_matches:
                n_compounds_checked += 1
                neutral_mass = compute_neutral_mass(match["matched_mz"], self.primary_adduct)

                adducts_detected = [
                    {
                    "adduct": self.primary_adduct,
                    "mz": match["matched_mz"],
                    "mass_error_ppm": match["mass_error_ppm"],
                    }
                ]

                for adduct in self.candidate_adducts:
                    if adduct == self.primary_adduct:
                        continue
                    expected_mz = compute_adduct_mz(neutral_mass, adduct)
                    found = find_best_match(expected_mz, observed_mzs, self.ppm_tolerance)
                    if found is not None:
                        matched_mz, ppm_error = found
                        adducts_detected.append(
                            {"adduct": adduct, "mz": matched_mz,
                             "mass_error_ppm": ppm_error}
                        )

                if len(adducts_detected) > 1:
                    n_with_extra_adducts += 1

                compounds.append(
                    {
                    "compound": match["compound"],
                    "neutral_mass_estimate": neutral_mass,
                    "adducts_detected": adducts_detected,
                    }
                )

            results[sample_id] = {"status": "checked", "compounds": compounds}

        context.qc_metrics[self._name] = {
            "results": results,
            "n_compounds_checked": n_compounds_checked,
            "n_with_extra_adducts": n_with_extra_adducts,
        }
        context.log_step(
            self._name,
            parameters={"candidate_adducts": self.candidate_adducts,
                        "ppm_tolerance": self.ppm_tolerance},
            metrics={"n_compounds_checked": n_compounds_checked,
                     "n_with_extra_adducts": n_with_extra_adducts},
            warnings=warnings,
        )
        return context

    def validate_output(self, context: LCMSContext) -> bool:
        """The stage left its metrics on the context."""
        return "adduct_annotation" in context.qc_metrics

class CohortAdductAnnotationStage:
    """
    Group aligned features into compounds and label their ion types (cohort_with_qc).

    method="hypothesis" (default): adduct_graph.annotate_adducts_v2 -- candidate pairs by
    binary search, correlation on log intensities of the gap-filled table, labels from the ion
    hypothesis that explains the most group members ("assumed" when nothing corroborates).
    method="legacy": the previous all-pairs implementation, kept for comparison runs.
    """
    _name: str = "adduct_annotation"
    # requires: stages whose results this one reads (must be registered earlier; checked before anything runs).
    # runs_after: stages it only reads if present; if they are registered they must come earlier.
    requires: tuple[str, ...] = ('feature_alignment',)

    def __init__(
        self,
        mz_tolerance_ppm: float = 10.0,
        rt_tolerance_sec: float = 15.0,
        correlation_threshold: float = 0.75,
        adduct_table: list[AdductDef] = NEGATIVE_ADDUCTS,
        method: str = "hypothesis",
        adduct_names: list[str] | None = None,     # None -> adduct_graph.DEFAULT_COHORT_ADDUCTS
        use_filled: bool = True,
        log_correlation: bool = True,
        min_correlation_samples: int = 6,
    ):
        """
        Args:
            mz_tolerance_ppm / rt_tolerance_sec: two features can be related only if within both tolerances.
            correlation_threshold: minimum intensity correlation across samples for a pair to count (0 to 1).
            adduct_table: relationships to test (legacy method only; the default method builds it from `adduct_names`).
            method: "hypothesis" (default) or "legacy" (previous all-pairs implementation, for comparison runs).
            adduct_names: adducts to consider; None = every adduct the stage knows (must include "[M-H]-").
            use_filled: correlate on the gap-filled table when it exists.
            log_correlation: correlate log intensities.
            min_correlation_samples: pairs sharing fewer samples than this cannot form a link.

        Raises:
            ValueError: unknown method or adduct name, or an adduct list without "[M-H]-".
        """
        if method not in ("hypothesis", "legacy"):
            raise ValueError(f"unknown adduct annotation method {method!r}")
        self.mz_tolerance_ppm = mz_tolerance_ppm
        self.rt_tolerance_sec = rt_tolerance_sec
        self.correlation_threshold = correlation_threshold
        self.method = method
        self.adduct_names = adduct_names
        self.adduct_table = adduct_table if method == "legacy" else build_adduct_table(adduct_names)
        self.use_filled = use_filled
        self.log_correlation = log_correlation
        self.min_correlation_samples = min_correlation_samples

    def _aligned_path(self, context: LCMSContext) -> Path:
        """Aligned feature table from feature_alignment."""
        return context.featurejson_dir / f"{context.study_id}_aligned_features.parquet"

    def _filled_path(self, context: LCMSContext) -> Path:
        """Gap-filled aligned table (optional input)."""
        return context.featurejson_dir / f"{context.study_id}_aligned_features_filled.parquet"

    def _annotated_path(self, context: LCMSContext) -> Path:
        """This stage's output: the aligned table plus adduct group / ion type columns and the blank flags."""
        return context.featurejson_dir / f"{context.study_id}_annotated_features.parquet"

    def _flags_path(self, context: LCMSContext) -> Path:
        """Blank-flags side table from blank_qc (optional input; merged into the annotated table)."""
        return context.featurejson_dir / f"{context.study_id}_blank_flags.parquet"

    def expected_outputs(self, context: LCMSContext) -> list[Path]:
        """The annotated table; if it is missing a cached result is stale."""
        return [self._annotated_path(context)]

    def _uses_filled(self, context: LCMSContext) -> bool:
        """True if the hypothesis method should (and can) correlate on the gap-filled table."""
        return self.method == "hypothesis" and self.use_filled and self._filled_path(context).exists()

    def cache_key(self, context: LCMSContext) -> str:
        """Method and thresholds, the adduct table, and the contents of the aligned, filled and blank-flag tables."""
        hash_cache = load_filehash_cache(context.cache_dir)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "method": self.method,
            "mz_tolerance_ppm": self.mz_tolerance_ppm,
            "rt_tolerance_sec": self.rt_tolerance_sec,
            "correlation_threshold": self.correlation_threshold,
            "log_correlation": self.log_correlation,
            "min_correlation_samples": self.min_correlation_samples,
            "adduct_table": [(a.name, a.mass_shift, a.multiplier, a.charge) for a in self.adduct_table],
            "aligned_features_hash": hash_file(self._aligned_path(context), hash_cache),
            "filled_features_hash": hash_file(self._filled_path(context), hash_cache) if self._uses_filled(context) else None,
            "blank_flags_hash": hash_file(self._flags_path(context), hash_cache) if self._flags_path(context).exists() else None,
        }
        save_filehash_cache(context.cache_dir, hash_cache)
        return compute_cache_key(payload)

    def validate_input(self, context: LCMSContext) -> bool:
        """Fatal if the aligned feature table does not exist."""
        if not self._aligned_path(context).exists():
            raise FatalStageError(
                f"adduct_annotation: aligned features file not found at {self._aligned_path(context)}"
            )
        return True

    def execute(self, context: LCMSContext) -> LCMSContext:
        """
        Annotate the aligned table, merge in the blank flags and write the annotated table.

        Warns when the largest adduct group has more than 10 members: that usually means loose tolerances are
        chaining unrelated features together.
        """
        aligned_table = pd.read_parquet(self._aligned_path(context))
        warnings: list[str] = []
        extra: dict = {}

        if self.method == "legacy":
            result = annotate_adducts(
                aligned_table,
                mz_tolerance_ppm=self.mz_tolerance_ppm,
                rt_tolerance_sec=self.rt_tolerance_sec,
                correlation_threshold=self.correlation_threshold,
                adduct_table=self.adduct_table,
            )
        else:
            filled = pd.read_parquet(self._filled_path(context)) if self._uses_filled(context) else None
            if self.use_filled and filled is None:
                warnings.append("adduct_annotation: no gap-filled table found, correlating on detected values only "
                                "(rows with < min_correlation_samples shared samples cannot form edges)")
            result, extra = annotate_adducts_v2(
                aligned_table,
                mz_tolerance_ppm=self.mz_tolerance_ppm,
                rt_tolerance_sec=self.rt_tolerance_sec,
                correlation_threshold=self.correlation_threshold,
                adduct_table=self.adduct_table,
                filled=filled,
                log_correlation=self.log_correlation,
                min_correlation_samples=self.min_correlation_samples,
            )
        if self._flags_path(context).exists():
            flags = pd.read_parquet(self._flags_path(context))[["feature_id", "blank_level", "blank_flagged"]]
            result = result.drop(columns=["blank_level", "blank_flagged"], errors="ignore").merge(
                flags, on="feature_id", how="left")
            result["blank_flagged"] = result["blank_flagged"].fillna(False).astype(bool)
        result.to_parquet(self._annotated_path(context))

        group_sizes = result.groupby("adduct_group_id").size()
        if int(group_sizes.max()) > 10:
            warnings.append(f"adduct_annotation: largest group has {int(group_sizes.max())} members; "
                            "check rt_tolerance_sec / mz_tolerance_ppm (chained false edges merge unrelated features)")
        context.qc_metrics[self._name] = {
            "method": self.method,
            "n_features": len(result),
            "n_groups": result["adduct_group_id"].nunique(),
            "n_singleton_groups": int((group_sizes == 1).sum()),
            "largest_group_size": int(group_sizes.max()),
            "group_size_distribution": group_sizes.value_counts().sort_index().to_dict(),
            **extra,
        }

        context.log_step(
            self._name,
            parameters={
                "method": self.method,
                "mz_tolerance_ppm": self.mz_tolerance_ppm,
                "rt_tolerance_sec": self.rt_tolerance_sec,
                "correlation_threshold": self.correlation_threshold,
                "log_correlation": self.log_correlation,
                "use_filled": self.use_filled,
                "adducts": [a.name for a in self.adduct_table],
            },
            metrics=context.qc_metrics[self._name],
            warnings=warnings,
            input_files=[str(self._aligned_path(context))],
            output_files=[str(self._annotated_path(context))],
        )
        return context

    def validate_output(self, context: LCMSContext) -> bool:
        """The metrics are set and the annotated table was written."""
        return self._name in context.qc_metrics and self._annotated_path(context).exists()
