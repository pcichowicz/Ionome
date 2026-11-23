"""
Feature detection (stage 5 of standards_only, stage 4 of cohort_with_qc): find chromatographic peaks in every sample.

For each sample with a raw file it runs the detector (analysis.utils.peak_detection via PyteomicsFeatureDetector),
writes `<results>/features/<sample_id>.featurejson` (a list of mz / rt / intensity / charge records, rt in seconds)
and keeps a small summary per sample in `qc_metrics["feature_detection"]`.

A single sample that fails is logged and skipped; zero features in every sample is fatal (almost always a settings
or file problem, e.g. noise_threshold set too high for the instrument).
"""
from __future__ import annotations

from pathlib import Path

from analysis.context import LCMSContext
from analysis.utils.protocols import FeatureDetector
from analysis.pipeline import FatalStageError
from analysis.utils.constants import FEATURE_FILE_SUFFIX
from analysis.utils.feature_utils import summarize_features
from analysis.utils.cache_utils import load_filehash_cache, save_filehash_cache, hash_file, hash_files, compute_cache_key
from core.ingestion import find_raw_file

import tqdm

# Bump this whenever the peak-picking/feature-detection logic changes --
# file hashes alone won't catch an algorithm change.
CACHE_SCHEMA_VERSION = "v2"   # v2: chromatogram-based detection + optional Feature fields

class FeatureDetectionStage:
    """
    Run the detector over all samples and store each sample's features on disk.

    Cache key: detection parameters + hash of the sample sheet + hash of every raw file (+ upstream and code, added
    by the pipeline). Changing any detector option or replacing a raw file recomputes this stage.
    """
    _name:str = "feature_detection"
    # requires: stages whose results this one reads (must be registered earlier; checked before anything runs).
    # runs_after: stages it only reads if present; if they are registered they must come earlier.
    requires: tuple[str, ...] = ('ingestion',)

    def __init__(
        self,
        detector: FeatureDetector,
        output_dir: Path,
        params: dict | None = None,
    ):
        """
        Args:
            detector: object with `detect_features(raw_path, output_path, params)`.
            output_dir: where the .featurejson files go (the context's features/ folder).
            params: detector options, normally `config.feature_detection` plus the study polarity.
        """
        self.detector = detector
        self.output_dir = Path(output_dir)
        self.params = params or {}

    def cache_key(self, context: LCMSContext) -> str:
        """Hash of the parameters, the sample sheet and every raw file's contents (see the class docstring)."""
        hash_cache = load_filehash_cache(context.cache_dir)
        candidate_paths = (
            find_raw_file(context.mzml_dir, sid) for sid in context.sample_metadata["sample_id"]
        )
        mzml_paths = [p for p in candidate_paths if p.exists()]
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "params": self.params,
            "sample_metadata_hash": hash_file(context.sample_metadata_path, hash_cache),
            "mzml_hashes": hash_files(mzml_paths, hash_cache),
        }
        save_filehash_cache(context.cache_dir, hash_cache)
        return compute_cache_key(payload)

    def expected_outputs(self, context: LCMSContext) -> list[Path]:
        """The .featurejson file of each sample that has a raw file; a missing one makes a cached result stale."""
        outs = []
        for sid in context.sample_metadata["sample_id"]:
            raw = find_raw_file(context.mzml_dir, sid)
            if raw is not None and raw.exists():
                outs.append(self.output_dir / f"{sid}{FEATURE_FILE_SUFFIX}")
        return outs

    def validate_input(self, context: LCMSContext) -> bool:
        """Fatal if the raw folder or sample sheet is missing; creates the output folder."""
        if context.mzml_dir is None:
            raise FatalStageError("feature_detection: mzml_directory not set (run ingestion first)")
        if context.sample_metadata is None:
            raise FatalStageError("feature_detection: sample_metadata not loaded on context")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return True

    def execute(self, context: LCMSContext) -> LCMSContext:
        """
        Detect features per sample, tolerating individual failures.

        Records per-sample status ("ok" with a summary and detector diagnostics, "failed", or skipped) in
        qc_metrics, and warns about samples with zero features or peaks dropped for being too wide.

        Raises:
            FatalStageError: no sample produced any feature.
        """
        results: dict[str, dict] = {}
        warnings: list[str] = []
        n_with_features = 0
        n_attempted = 0
        input_files: list[str] = []
        output_files: list[str] = []

        #rough estimate of how long the feature detection will take
        for _, row in tqdm.tqdm(context.sample_metadata.iterrows(), total=len(context.sample_metadata),
                           desc=self._name):
            sample_id = row["sample_id"]
            mzml_path = find_raw_file(context.mzml_dir, sample_id)

            if not mzml_path.exists():
                warnings.append(f"{sample_id}: mzML missing, skipped")
                results[sample_id] = {"status": "skipped_missing_mzml"}
                continue

            n_attempted += 1
            output_path = self.output_dir / f"{sample_id}{FEATURE_FILE_SUFFIX}"
            input_files.append(str(mzml_path))

            try:
                features = self.detector.detect_features(mzml_path, output_path, self.params)
            except Exception as exc:  #  one bad file shouldn't kill the run
                warnings.append(f"{sample_id}: feature detection failed: {exc}")
                results[sample_id] = {"status": "failed", "error": str(exc)}
                continue

            summary = summarize_features(features)
            results[sample_id] = {"status": "ok", "summary": summary,
                                  "diagnostics": dict(getattr(self.detector, "last_diagnostics", {}) or {})}
            output_files.append(str(output_path))

            diag = results[sample_id]["diagnostics"]
            if diag.get("n_rejected_too_wide"):
                warnings.append(f"{sample_id}: {diag['n_rejected_too_wide']} peaks wider than peak_width[1] dropped")
            if summary["n_features"] > 0:
                n_with_features += 1
            else:
                warnings.append(f"{sample_id}: zero features detected")

        if n_attempted > 0 and n_with_features == 0:
            print("WARNINGS", warnings)
            raise FatalStageError(
                "feature_detection: zero features detected across all "
                f"{n_attempted} attempted samples -- likely a config or "
                "instrument-file problem, not a single-compound problem"
            )

        context.featurejson_dir = self.output_dir
        context.qc_metrics[self._name] = {
            "results": results,
            "n_attempted": n_attempted,
            "n_with_features": n_with_features,
        }

        context.log_step(
            self._name,
            parameters=self.params,
            metrics={"n_attempted": n_attempted,
                     "n_with_features": n_with_features},
            warnings=warnings,
            input_files=input_files,
            output_files=output_files
        )
        return context

    def validate_output(self, context: LCMSContext) -> bool:
        """The features folder is set on the context."""
        return context.featurejson_dir is not None