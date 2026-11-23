"""
Builds the stage list for a study's dataset profile and runs it.

`standards_only` runs 11 stages, `cohort_with_qc` runs 15; the lists are in `_build_standard_only_stages` and
`_build_cohort_with_qc_stages`, and the order matters (the pipeline checks each stage's `requires`).
All stage options come from the study's YAML (core.config_schema); nothing is hard-coded here.

Command line: `python -m analysis.pipeline_orchestrator <study_id>` runs the study but does NOT update the study
status in the database. Use scripts/register_study.py --run or the web app's Run button for that.
"""
import dataclasses
import json
import logging
import sys
from pathlib import Path

from core.config_schema import ExperimentConfig, ignored_settings, load_experiment_config
from analysis.context import LCMSContext
from analysis.pipeline import LCMSPipeline
from core.settings import settings
from core.ingestion import get_experiment_pk_sync
from analysis.stages.ingestion import MzMLIngestionStage
from analysis.stages.scan_summary import ScanSummaryCacheStage
from analysis.stages.chromatogram_plotting import ChromatogramPlottingStage
from analysis.stages.system_suitability import SystemSuitabilityStage
from analysis.stages.blank_qc import BlankQCStage
from analysis.stages.feature_detection import FeatureDetectionStage
from analysis.stages.feature_alignment import FeatureAlignmentStage
from analysis.stages.spectral_purity import SpectralPurityStage
from analysis.stages.adduct_annotation import AdductAnnotationStage, CohortAdductAnnotationStage
from analysis.stages.library_matching import SpectralLibraryMatchingStage, CohortLibraryMatchingStage
from analysis.stages.library_assembly import LibraryAssemblyStage, CohortLibraryAssemblyStage
from analysis.stages.summary_stats_plotting import SummaryStatsPlottingStage
from analysis.stages.export import ExportStage, CohortExportStage
from analysis.utils.json_reader import JSONFeatureReader
from analysis.utils.mzml_parsing import (
PyteomicsPrecursorReader,
PyteomicsFeatureDetector,
PyteomicsSpectralPurityReader,
PyteomicsMS2SpectrumReader
)
from analysis.utils.stats_prep import StatsPrepParams
from analysis.stages.StatPlots import VolcanoPlotStage, PLSDAStage, XICPlottingStage, HeatmapClusteringStage

logger = logging.getLogger(__name__)


def build_pipeline(config: ExperimentConfig, context: LCMSContext) -> LCMSPipeline:
    """Assemble the pipeline for `config.dataset_profile`; raises ValueError for an unknown profile."""
    pipeline = LCMSPipeline()

    if config.dataset_profile == "standards_only":
        stages = _build_standard_only_stages(config, context)
    elif config.dataset_profile == "cohort_with_qc":
        stages = _build_cohort_with_qc_stages(config, context)
    else:
        raise ValueError(f"Unknown dataset_profile: {config.dataset_profile}")

    for stage in stages:
        pipeline.register_stage(stage)
    return pipeline

def _library_path(config: ExperimentConfig, context: LCMSContext) -> Path:
    """reference_library_path is relative to base_dir unless absolute (as the YAML comments say)."""
    p = Path(config.library_matching.reference_library_path)
    return p if p.is_absolute() else context.base_dir / p


def _detection_params(config: ExperimentConfig) -> dict:
    """Detector options plus the study polarity (decides the charge sign of detected features)."""
    return {**config.feature_detection.model_dump(), "polarity": config.study.polarity.value}


def run_pipeline_for_study(study_id: str,
                           force_stages: set[str] | None = None,
                           on_stage_event= None) -> LCMSContext:
    """
    Load a registered study's config, build its pipeline and run it.

    Reads `<YAML_DIR>/<study_id>.yaml` and the study's database row, writes `resolved_config.json` (the exact
    settings used, plus options the profile ignores) next to the results, then runs all stages.

    Args:
        study_id: a study registered in the database (web app or scripts/register_study.py).
        force_stages: stage names to recompute regardless of the cache; {"*"} forces all.
        on_stage_event: optional (stage_name, status, error_msg) callback; the web app records progress with it.

    Returns:
        The final LCMSContext (qc_metrics, processing_log, final_feature_table).
    """
    config = load_experiment_config(settings.yaml_dir / f"{study_id}.yaml")
    experiment_pk = get_experiment_pk_sync(study_id)
    context = LCMSContext(study_id=study_id,experiment_id=experiment_pk, config=config)

    # Options the YAML sets but this profile never reads: say so instead of ignoring them silently.
    ignored = ignored_settings(config)
    tuned = ignored_settings(config, only_changed=True)   # changed from the default, yet read by nothing
    if tuned:
        logger.warning("config options not used by profile %s: %s", config.dataset_profile.value, tuned)
        print(f"  note: set in the YAML but not used by the {config.dataset_profile.value} profile: {', '.join(tuned)}")
    # What produced this run, kept next to the results (not in qc_metrics, which the report dumps).
    (context.results_dir / "resolved_config.json").write_text(json.dumps(
        {"config": config.model_dump(mode="json"), "ignored_settings": ignored}, indent=2))

    pipeline = build_pipeline(config, context)
    return pipeline.run(context, force_stages=force_stages, on_stage_event=on_stage_event)

def _build_standard_only_stages(config: ExperimentConfig, context: LCMSContext):
    """Stage list for pure-standard runs: per-sample suitability, blank, adduct, purity and library checks."""
    return [
        MzMLIngestionStage(mzml_dir = context.mzml_dir),
        ScanSummaryCacheStage(),
        ChromatogramPlottingStage(),
        SystemSuitabilityStage(
            reader = PyteomicsPrecursorReader(),
            ppm_tolerance = config.system_suitability.ppm_tolerance,
            applicable_roles = None
        ),
        FeatureDetectionStage(
            detector=PyteomicsFeatureDetector(),
            output_dir=Path(context.featurejson_dir),
            params=_detection_params(config),
        ),
        BlankQCStage(
            reader=JSONFeatureReader(),
            mz_tolerance_ppm=config.blank_qc.mz_tolerance_ppm,
            rt_tolerance_sec=config.blank_qc.rt_tolerance_sec,
        ),
        AdductAnnotationStage(
            reader=PyteomicsPrecursorReader(),
            candidate_adducts=config.adduct_annotation.candidate_adducts,
            primary_adduct=config.adduct_annotation.primary_adduct,
            ppm_tolerance=config.adduct_annotation.ppm_tolerance,
        ),
        SpectralPurityStage(
            reader=PyteomicsSpectralPurityReader(),
            isolation_window_da=config.spectral_purity.isolation_window_da,
            min_purity=config.spectral_purity.min_purity,
        ),
        SpectralLibraryMatchingStage(
            ms2_reader=PyteomicsMS2SpectrumReader(),
            reference_library_path=_library_path(config, context),
            reference_library_format=config.library_matching.reference_library_format,
            precursor_mz_tolerance_ppm=config.library_matching.precursor_mz_tolerance_ppm,
            fragment_mz_tolerance_da=config.library_matching.fragment_mz_tolerance_da,
            min_match_score=config.library_matching.min_match_score,
        ),
        LibraryAssemblyStage(
            reader=JSONFeatureReader()),
        ExportStage()
    ]

def _build_cohort_with_qc_stages(config: ExperimentConfig, context: LCMSContext):
    """Stage list for cohort runs: detection and alignment across samples, annotation, library matching, statistics."""
    stats = config.statistics
    prep = StatsPrepParams(**stats.prep_kwargs())
    align = config.feature_alignment
    adducts = config.adduct_annotation
    lib = config.library_matching
    purity = config.spectral_purity
    return [
        MzMLIngestionStage(mzml_dir = context.mzml_dir),
        ScanSummaryCacheStage(),
        ChromatogramPlottingStage(),
        FeatureDetectionStage(detector=PyteomicsFeatureDetector(),
                              output_dir=Path(context.featurejson_dir),
                              params=_detection_params(config)),
        FeatureAlignmentStage(
            mz_tolerance_ppm=align.alignment_mz_tolerance_ppm,
            rt_tolerance_sec=align.alignment_rt_tolerance_sec,
            algorithm=align.algorithm,
            rt_correction=align.rt_correction,
            reference_sample=align.reference_sample,
            gap_fill=align.gap_fill,
        ),
        BlankQCStage(reader=JSONFeatureReader(),
                     blank_fold_threshold=config.blank_qc.blank_fold_threshold,
                     mode="feature_table"),
        CohortAdductAnnotationStage(
            mz_tolerance_ppm=adducts.cohort_mz_tolerance_ppm,
            rt_tolerance_sec=adducts.cohort_rt_tolerance_sec,
            correlation_threshold=adducts.correlation_threshold,
            method=adducts.cohort_method,
            adduct_names=adducts.cohort_adducts,
            use_filled=adducts.use_filled,
            log_correlation=adducts.log_correlation,
        ),
        CohortLibraryMatchingStage(
            reference_library_path=_library_path(config, context),
            reference_library_format=lib.reference_library_format,
            precursor_mz_tolerance_ppm=lib.precursor_mz_tolerance_ppm,
            fragment_mz_tolerance_da=lib.fragment_mz_tolerance_da,
            min_match_score=lib.min_match_score,
            min_purity_high_confidence=purity.min_purity_high_confidence,
            min_purity_tentative=purity.min_purity_tentative,
            isolation_window_da=purity.isolation_window_da,
            ms2_precursor_tolerance_da=lib.ms2_precursor_tolerance_da,
            ms2_rt_tolerance_sec=lib.ms2_rt_tolerance_sec,
            min_matched_peaks=lib.min_matched_peaks,
            intensity_power=lib.intensity_power,
            remove_precursor=lib.remove_precursor,
            tie_margin=lib.tie_margin,
            min_ms2_spectra_for_peak=lib.min_ms2_spectra_for_peak,
            library_qc=lib.library_qc,
            n_null_refs=lib.n_null_refs,
        ),
        CohortLibraryAssemblyStage(),
        SummaryStatsPlottingStage(),
        # PLS-DA must not use class labels to choose features before cross-validation
        PLSDAStage(prep=dataclasses.replace(prep, presence_mode="global")),
        HeatmapClusteringStage(prep=prep),
        VolcanoPlotStage(prep=prep, test_method=stats.test_method),
        XICPlottingStage(),
        CohortExportStage(),
    ]

if __name__ == "__main__":
    run_pipeline_for_study(sys.argv[1])
