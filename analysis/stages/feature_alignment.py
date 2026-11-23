"""
Feature alignment (stage 5 of cohort_with_qc only): one consensus feature table across all samples.

Builds a cross-sample consensus feature table from per-sample .featureJSON
files: optional LOWESS retention-time correction against a reference sample,
seeded one-feature-per-sample grouping, then (optionally) gap filling from the
raw MS1 data into a separate table.
Required before every cohort stage that needs a unified feature set (blank_qc, adduct_annotation, library_matching).
Files go to `<results>/features/`. `rt` in the table is in seconds.

Outputs
  <study>_aligned_features.parquet         NaN = not detected (unchanged meaning)
  <study>_aligned_features_filled.parquet  same layout; NaN replaced by the
                                           extracted-ion apex where a peak exists
  <study>_rt_correction.json               per-sample drift curves
"""
import json
from pathlib import Path

import numpy as np

from analysis.context import LCMSContext
from analysis.utils.feature_alignment import align_features_detailed, align_features_legacy
from analysis.utils.gap_fill import fill_gaps
from analysis.utils.cache_utils import load_filehash_cache, hash_files, save_filehash_cache, compute_cache_key
from analysis.utils.constants import FEATURE_FILE_SUFFIX
from analysis.utils.mzml_parsing import load_ms1_scans
from core.ingestion import find_raw_file

CACHE_SCHEMA_VERSION = "v2"   # v2: seeded algorithm, RT correction, gap-fill output

class FeatureAlignmentStage:
    """
    Align the per-sample features into the consensus table (and optionally gap-fill it).

    Cache key: tolerances and options + hashes of the feature files of the samples that detected features.
    """
    _name: str = "feature_alignment"
    # requires: stages whose results this one reads (must be registered earlier; checked before anything runs).
    # runs_after: stages it only reads if present; if they are registered they must come earlier.
    requires: tuple[str, ...] = ('feature_detection',)

    def __init__(
        self,
        mz_tolerance_ppm: float,
        rt_tolerance_sec: float,
        algorithm: str = "seeded",            # "seeded" | "legacy" (previous first-fit, for comparison runs)
        rt_correction: bool = True,
        reference_sample: str | None = None,  # None = sample with the most features
        gap_fill: bool = True,
        gap_fill_mz_ppm: float | None = None,         # default: mz_tolerance_ppm
        gap_fill_rt_window_sec: float | None = None,  # default: rt_tolerance_sec
    ):
        """
        Args:
            mz_tolerance_ppm / rt_tolerance_sec: features of different samples within BOTH are the same feature.
            algorithm: "seeded" (default) or "legacy" (previous matching, kept for comparison runs).
            rt_correction: correct retention-time drift between samples first (seeded only).
            reference_sample: sample the others are aligned to; None picks the one with the most features.
            gap_fill: also write a table where undetected values are re-extracted from the raw MS1 data.
            gap_fill_mz_ppm / gap_fill_rt_window_sec: extraction window; default to the alignment tolerances.

        Raises:
            ValueError: unknown `algorithm`.
        """
        if algorithm not in ("seeded", "legacy"):
            raise ValueError(f"unknown alignment algorithm {algorithm!r}")
        self.mz_tolerance_ppm = mz_tolerance_ppm
        self.rt_tolerance_sec = rt_tolerance_sec
        self.algorithm = algorithm
        self.rt_correction = rt_correction and algorithm == "seeded"
        self.reference_sample = reference_sample
        self.gap_fill = gap_fill
        self.gap_fill_mz_ppm = gap_fill_mz_ppm or mz_tolerance_ppm
        self.gap_fill_rt_window_sec = gap_fill_rt_window_sec or rt_tolerance_sec

    def expected_outputs(self, context: LCMSContext) -> list[Path]:
        """The aligned table (and the gap-filled table when enabled); a missing file invalidates the cached result."""
        outs = [self._aligned_path(context)]
        if self.gap_fill:
            outs.append(self._filled_path(context))
        return outs

    def _aligned_path(self, context: LCMSContext) -> Path:
        """Aligned table: NaN = not detected in that sample."""
        return context.featurejson_dir / f"{context.study_id}_aligned_features.parquet"

    def _filled_path(self, context: LCMSContext) -> Path:
        """Same layout as the aligned table, with gaps replaced by extracted-ion apex intensities."""
        return context.featurejson_dir / f"{context.study_id}_aligned_features_filled.parquet"

    def _rt_path(self, context: LCMSContext) -> Path:
        """Per-sample retention-time drift curves (JSON)."""
        return context.featurejson_dir / f"{context.study_id}_rt_correction.json"

    def cache_key(self, context: LCMSContext) -> str:
        """Hash of the alignment options and of the feature file of every sample whose detection succeeded."""
        hash_cache = load_filehash_cache(context.cache_dir)
        fd_results = context.qc_metrics.get("feature_detection", {}).get("results", {})
        ok_sample_ids = [sid for sid, r in fd_results.items() if r.get("status") == "ok"]
        featurejson_paths = [
            context.featurejson_dir / f"{sid}{FEATURE_FILE_SUFFIX}" for sid in ok_sample_ids
        ]
        featurejson_paths = [p for p in featurejson_paths if p.exists()]

        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "mz_tolerance_ppm": self.mz_tolerance_ppm,
            "rt_tolerance_sec": self.rt_tolerance_sec,
            "algorithm": self.algorithm,
            "rt_correction": self.rt_correction,
            "reference_sample": self.reference_sample,
            "gap_fill": self.gap_fill,
            "gap_fill_mz_ppm": self.gap_fill_mz_ppm,
            "gap_fill_rt_window_sec": self.gap_fill_rt_window_sec,
            "featurejson_hashes": hash_files(featurejson_paths, hash_cache),
        }
        save_filehash_cache(context.cache_dir, hash_cache)
        return compute_cache_key(payload)

    def validate_input(self, context: LCMSContext) -> bool:
        """At least one sample must have detected features."""
        fd_results = context.qc_metrics.get("feature_detection", {}).get("results", {})
        return any(r.get("status") == "ok" for r in fd_results.values())

    def execute(self, context: LCMSContext) -> LCMSContext:
        """
        Load each successful sample's features, align them, write the tables and record metrics.

        Samples whose detection failed or whose feature file is missing are skipped with a warning. Metrics include
        the number of consensus features, how many were found in all samples, and RT-correction / gap-fill summaries.
        """
        fd_results = context.qc_metrics["feature_detection"]["results"]

        per_sample_features = {}
        skipped = []
        for sample_id, result in fd_results.items():
            if result.get("status") != "ok":
                skipped.append(sample_id)
                continue
            path = context.featurejson_dir / f"{sample_id}{FEATURE_FILE_SUFFIX}"
            if not path.exists():
                skipped.append(sample_id)
                continue
            with open(path) as f:
                per_sample_features[sample_id] = json.load(f)

        warnings = [f"{sid}: skipped (missing/failed feature detection)" for sid in skipped]
        info: dict = {}
        if self.algorithm == "legacy":
            table = align_features_legacy(per_sample_features, self.mz_tolerance_ppm, self.rt_tolerance_sec)
        else:
            table, info = align_features_detailed(
                per_sample_features, self.mz_tolerance_ppm, self.rt_tolerance_sec,
                rt_correction=self.rt_correction, reference_sample=self.reference_sample,
            )
        table.to_parquet(self._aligned_path(context))

        sample_cols = list(per_sample_features.keys())
        n_detected = table[sample_cols].notna().sum(axis=1) if len(table) else None
        rt_info = info.get("rt_correction")
        rt_summary = None
        if rt_info:
            for sid, s in rt_info["samples"].items():
                if s["n_anchors"] is not None and not s["applied"]:
                    warnings.append(f"{sid}: RT correction skipped, only {s['n_anchors']} usable anchor features")
            rt_summary = {
                "reference_sample": rt_info["reference_sample"],
                "samples": {sid: {k: v for k, v in s.items() if not k.startswith("curve")}
                            for sid, s in rt_info["samples"].items()},
            }
            with open(self._rt_path(context), "w") as f:
                json.dump(rt_info, f)

        gap_stats = None
        output_files = [str(self._aligned_path(context))]
        if self.gap_fill and len(table):
            def load_scans(sid: str):
                """Raw MS1 scans of one sample for gap filling, or None if its raw file is missing."""
                raw = find_raw_file(context.mzml_dir, sid)
                return load_ms1_scans(raw) if raw is not None and Path(raw).exists() else None
            curves = (rt_info or {}).get("samples", {})
            filled, gap_stats = fill_gaps(
                table, sample_cols, load_scans, curves, self.gap_fill_mz_ppm, self.gap_fill_rt_window_sec,
            )
            filled.to_parquet(self._filled_path(context))
            output_files.append(str(self._filled_path(context)))
            for sid, s in gap_stats["per_sample"].items():
                if s.get("skipped"):
                    warnings.append(f"{sid}: gap filling skipped ({s['skipped']})")

        metrics = {
            "n_consensus_features": len(table),
            "n_samples_included": len(per_sample_features),
            "n_samples_skipped": len(skipped),
            "skipped_sample_ids": skipped,
            "algorithm": self.algorithm,
            "n_same_sample_conflicts": info.get("n_same_sample_conflicts"),
            "median_samples_detected_per_feature": float(n_detected.median()) if n_detected is not None else None,
            "n_features_in_all_samples": int((n_detected == len(sample_cols)).sum()) if n_detected is not None else None,
            "rt_correction": rt_summary,
            "gap_fill": None if gap_stats is None else {k: v for k, v in gap_stats.items() if k != "per_sample"},
        }
        context.qc_metrics[self._name] = metrics
        context.log_step(
            self._name,
            parameters={
                "mz_tolerance_ppm": self.mz_tolerance_ppm,
                "rt_tolerance_sec": self.rt_tolerance_sec,
                "algorithm": self.algorithm,
                "rt_correction": self.rt_correction,
                "reference_sample": self.reference_sample,
                "gap_fill": self.gap_fill,
                "gap_fill_mz_ppm": self.gap_fill_mz_ppm,
                "gap_fill_rt_window_sec": self.gap_fill_rt_window_sec,
            },
            metrics={k: v for k, v in metrics.items() if k != "skipped_sample_ids"},
            warnings=warnings,
            input_files=[str(context.featurejson_dir / f"{sid}{FEATURE_FILE_SUFFIX}") for sid in per_sample_features],
            output_files=output_files,
        )
        return context

    def validate_output(self, context: LCMSContext) -> bool:
        """The aligned table (and the filled one, if enabled) exist and the metrics are on the context."""
        ok = self._name in context.qc_metrics and self._aligned_path(context).exists()
        if ok and self.gap_fill:
            ok = self._filled_path(context).exists()
        return ok
