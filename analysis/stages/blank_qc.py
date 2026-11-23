"""
Blank QC (stage 6 in both profiles): flag features that also appear in the solvent blank.

Two modes, chosen by the profile:
  * per_sample (standards_only): each sample's features are compared with the pooled blank features by m/z (ppm)
    and RT (seconds) tolerance. Results go to qc_metrics only.
  * feature_table (cohort_with_qc): on the aligned table, a feature is background when even its strongest sample
    does not exceed `blank_fold_threshold` x the blank level. Results go to the side table
    `features/<study>_blank_flags.parquet` (columns feature_id, blank_level, blank_flagged); the aligned table is
    never modified.

The blank samples are those with `sample_role == "solvent_blank"`. If there are none, nothing is flagged and a
warning is logged; the run continues.

Compares every standard/mix's detected features against the pooled feature
set from blanks. A matching feature is FLAGGED, not dropped for a spectral library, a
false negative -- dropping a real analyte peak because it happens to share
a background ion's m/z/RT -- is worse than a false positive here.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from analysis.context import LCMSContext
from analysis.utils.json_reader import JSONFeatureReader
from analysis.utils.constants import FEATURE_FILE_SUFFIX, NON_SAMPLE_COLS
from analysis.pipeline import FatalStageError
from analysis.utils.cache_utils import load_filehash_cache, save_filehash_cache, hash_file, compute_cache_key
from analysis.utils.blank_qc import is_blank_match

# Bump this whenever is_blank_match's matching logic changes -- file
# hashes alone won't catch an algorithm change.
CACHE_SCHEMA_VERSION = "v1"

class BlankQCStage:
    """Flag background features in either per-sample or feature-table mode (see the module docstring)."""
    _name = "blank_qc"
    # requires: stages whose results this one reads (must be registered earlier; checked before anything runs).
    # runs_after: stages it only reads if present; if they are registered they must come earlier.
    requires: tuple[str, ...] = ('feature_detection',)

    def __init__(
        self,
        reader: JSONFeatureReader,
        mz_tolerance_ppm: float = 20.0,
        rt_tolerance_sec: float = 20.0,
        blank_fold_threshold: float = 5.0,
        mode: str = "per_sample"

    ):
        """
        Args:
            reader: reads feature files (per_sample mode).
            mz_tolerance_ppm / rt_tolerance_sec: match window between a feature and a blank feature (per_sample mode).
            blank_fold_threshold: background if best sample signal < blank level x this factor (feature_table mode).
            mode: "per_sample" or "feature_table"; the latter requires the feature_alignment stage.
        """
        self.reader = reader
        self.mz_tolerance_ppm = mz_tolerance_ppm
        self.rt_tolerance_sec = rt_tolerance_sec
        self.blank_fold_threshold = blank_fold_threshold
        self.mode = mode
        if mode == "feature_table":
            self.requires = ("feature_alignment",)   # works on the aligned feature table

    def _aligned_path(self, context: LCMSContext) -> Path:
        """The aligned feature table written by feature_alignment."""
        return context.featurejson_dir / f"{context.study_id}_aligned_features.parquet"

    def _flags_path(self, context: LCMSContext) -> Path:
        """Side table (feature_id, blank_level, blank_flagged); the aligned table is never modified."""
        return context.featurejson_dir / f"{context.study_id}_blank_flags.parquet"

    def expected_outputs(self, context: LCMSContext) -> list[Path]:
        """The flags side table in feature_table mode; per_sample mode writes no files."""
        return [self._flags_path(context)] if self.mode == "feature_table" else []

    def cache_key(self, context: LCMSContext) -> str:
        """Mode, tolerances/threshold, the sample sheet and (feature_table) the aligned table's contents."""
        hash_cache = load_filehash_cache(context.cache_dir)

        if self.mode == "feature_table":
            payload = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "mode": self.mode,
                "blank_fold_threshold": self.blank_fold_threshold,
                "sample_metadata_hash": hash_file(context.sample_metadata_path, hash_cache),
                "aligned_features_hash": hash_file(self._aligned_path(context), hash_cache),
            }
        else:
            payload = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "mode": self.mode,
                "mz_tolerance_ppm": self.mz_tolerance_ppm,
                "rt_tolerance_sec": self.rt_tolerance_sec,
                "sample_metadata_hash": hash_file(context.sample_metadata_path, hash_cache),
                "feature_detection_qc": context.qc_metrics.get("feature_detection", {}),
            }

        save_filehash_cache(context.cache_dir, hash_cache)
        return compute_cache_key(payload)

    def validate_input(self, context: LCMSContext) -> bool:
        """Fatal if the features folder or sample sheet is not available."""
        if context.featurejson_dir is None:
            raise FatalStageError("blank_qc: featurejson_dir not set (run feature_detection first)")
        if context.sample_metadata is None:
            raise FatalStageError("blank_qc: sample_metadata not loaded on context")
        return True

    def _execute_per_sample(self, context: LCMSContext) -> LCMSContext:
        """
        standards_only: pool the features of all `solvent_blank` samples, then count, for each other sample, the
        features that match one of them within tolerance. Flags only; nothing is removed.
        """
        warnings: list[str] = []
        blank_ids = list(
            context.sample_metadata.loc[
                context.sample_metadata["sample_role"] == "solvent_blank", "sample_id"
            ]
        )

        if not blank_ids:
            warnings.append(
                "blank_qc: no samples with sample_role='solvent_blank' in sample_metadata; "
                "background flagging skipped entirely for this study"
            )
            context.qc_metrics[self._name] = {"results": {}, "n_blank_features": 0}
            context.log_step(
                self._name,
                parameters={"mz_tolerance_ppm": self.mz_tolerance_ppm,
                            "rt_tolerance_sec": self.rt_tolerance_sec},
                metrics={"n_blank_features": 0},
                warnings=warnings,
                input_files=[],
            )
            return context

        blank_features: list[dict] = []
        input_files: list[str] = []
        for sample_id in blank_ids:
            path = context.featurejson_dir / f"{sample_id}{FEATURE_FILE_SUFFIX}"
            if not path.exists():
                warnings.append(f"{sample_id}: featureJSON missing, excluded from blank background")
                continue
            blank_features.extend(self.reader.read_features(path))
            input_files.append(str(path))

        if not blank_features:
            warnings.append("blank_qc: no blank features could be loaded; background flagging skipped entirely")
            context.qc_metrics[self._name] = {"results": {},
                                              "n_blank_features": 0}
            context.log_step(
                self._name,
                parameters={"mz_tolerance_ppm": self.mz_tolerance_ppm,
                            "rt_tolerance_sec": self.rt_tolerance_sec},
                metrics={"n_blank_features": 0},
                warnings=warnings,
                input_files=input_files,
            )
            return context

        fd_results = context.qc_metrics.get("feature_detection", {}).get("results", {})
        results: dict[str, dict] = {}

        for _, row in context.sample_metadata.iterrows():
            sample_id = row["sample_id"]
            if row["sample_role"] in  ["blank", "solvent_blank"]:
                continue

            fd_result = fd_results.get(sample_id)
            if not fd_result or fd_result.get("status") != "ok":
                results[sample_id] = {"status": "skipped_no_features"}
                continue

            path = context.featurejson_dir / f"{sample_id}{FEATURE_FILE_SUFFIX}"
            if not path.exists():
                results[sample_id] = {"status": "skipped_missing_featurejson"}
                continue

            features = self.reader.read_features(path)
            flagged = []
            for feature in features:
                match = is_blank_match(feature, blank_features, self.mz_tolerance_ppm, self.rt_tolerance_sec)
                if match is not None:
                    flagged.append(
                        {
                        "feature_mz": feature["mz"],
                        "feature_rt": feature["rt"],
                        "matched_blank_mz": match["mz"],
                        "matched_blank_rt": match["rt"],
                        }
                    )

            results[sample_id] = {
                "status": "checked",
                "n_features": len(features),
                "n_flagged": len(flagged),
                "flagged": flagged,
            }
            if flagged:
                warnings.append(
                    f"{sample_id}: {len(flagged)}/{len(features)} features overlap with blank background"
                )

        context.qc_metrics[self._name] = {
            "results": results,
            "n_blank_features": len(blank_features),
        }
        context.log_step(
            self._name,
            parameters={"mz_tolerance_ppm": self.mz_tolerance_ppm,
                        "rt_tolerance_sec": self.rt_tolerance_sec},
            metrics={"n_blank_features": len(blank_features)},
            warnings=warnings,
            input_files=input_files,
        )
        return context

    def _execute_feature_table(self, context: LCMSContext) -> LCMSContext:
        """
        cohort_with_qc: blank level = max over blank columns; flag a feature when its best non-blank sample signal is
        below `blank_fold_threshold` x that level. Writes the flags side table; always writes it (all False when
        there is no blank) so later stages see the same columns either way.
        """
        warnings: list[str] = []
        aligned_path = self._aligned_path(context)
        table = pd.read_parquet(aligned_path)
        # tables written by older versions of this stage carry the flags in place; ignore them
        table = table.drop(columns=["blank_level", "blank_flagged"], errors="ignore")

        blank_ids = list(
            context.sample_metadata.loc[
                context.sample_metadata["sample_role"] == "solvent_blank", "sample_id"
            ]
        )
        blank_cols = [c for c in blank_ids if c in table.columns]
        sample_cols = [
            c for c in table.columns
            if c not in NON_SAMPLE_COLS and c not in blank_ids
        ]

        if not blank_cols:
            warnings.append(
                "blank_qc: no blank columns available in aligned feature table; "
                "background flagging skipped entirely for this study"
            )
            # still write the side table so downstream stages see the same columns either way
            pd.DataFrame({
                "feature_id": table["feature_id"],
                "blank_level": float("nan"),
                "blank_flagged": False,
            }).to_parquet(self._flags_path(context))
            context.qc_metrics["blank_qc"] = {"n_flagged": 0, "n_total": len(table), "blank_available": False}
            context.log_step(
                self._name,
                parameters={"blank_fold_threshold": self.blank_fold_threshold},
                metrics={"n_flagged": 0},
                warnings=warnings,
                input_files=[])
            return context

        blank_level = table[blank_cols].max(axis=1, skipna=True)
        best_sample_signal = table[sample_cols].max(axis=1, skipna=True)

        # A feature is background if even its best-showing sample never clears
        # blank_fold_threshold x the blank level
        flagged = (blank_level.notna()) & (best_sample_signal < blank_level * self.blank_fold_threshold)

        pd.DataFrame({
            "feature_id": table["feature_id"],
            "blank_level": blank_level,
            "blank_flagged": flagged.astype(bool),
        }).to_parquet(self._flags_path(context))

        n_flagged = int(flagged.sum())
        context.qc_metrics[self._name] = {
            "n_flagged": n_flagged,
            "n_total": len(table),
            "blank_available": True,
            "blank_fold_threshold": self.blank_fold_threshold,
        }
        context.log_step(
            self._name,
            parameters={"blank_fold_threshold": self.blank_fold_threshold},
            metrics={"n_flagged": n_flagged,
                     "n_total": len(table)},
            warnings=warnings,
            input_files=[str(aligned_path)],
            output_files=[str(self._flags_path(context))])
        return context

    def execute(self, context: LCMSContext) -> LCMSContext:
        """Dispatch to the per-sample or feature-table implementation."""
        if self.mode == "feature_table":
            return self._execute_feature_table(context)
        return self._execute_per_sample(context)

    def validate_output(self, context: LCMSContext) -> bool:
        """The metrics exist, and in feature_table mode the flags file was written."""
        if self.mode == "feature_table" and not self._flags_path(context).exists():
            return False
        return self._name in context.qc_metrics