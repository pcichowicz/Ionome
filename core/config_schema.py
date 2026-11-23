"""
Experiment configuration (one YAML per study).

Every section rejects unknown keys, so a misspelled option fails at load time instead of being
silently ignored. All options added after v0.1 have defaults, so older YAMLs still load.

Some options only apply to one dataset profile. Setting one that the chosen profile does not use is
allowed, but `ignored_settings(config)` lists it so the run can warn (see pipeline_orchestrator).
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml

from core.safe_paths import STUDY_ID_PATTERN, is_safe_name


class _Section(BaseModel):
    """Base for every config section: unknown keys are rejected so typos fail at load time."""
    model_config = ConfigDict(extra="forbid")


class Polarity(str, Enum):
    """Ion mode the data was acquired in; decides which adducts and library entries make sense."""
    positive = "positive"
    negative = "negative"

class DatasetProfile(str, Enum):
    """
    Which pipeline runs: `standards_only` (pure-compound runs with known expected identities, 11 stages) or
    `cohort_with_qc` (biological cohort with groups, pooled QC and a blank, 15 stages).
    """
    standards_only = "standards_only"
    cohort_with_qc = "cohort_with_qc"

class LibraryFormat(str, Enum):
    """File format of the reference spectral library."""
    msp = "msp"
    mgf = "mgf"


class StudyInfo(_Section):
    """Identity of the study and how it was measured."""
    # becomes folder and file names under the raw-data, results and YAML directories
    id: str = Field(pattern=STUDY_ID_PATTERN)
    polarity: Polarity
    instrument: str          # free text, kept on the run context (LCMSContext.instrument)

class FilesInfo(_Section):
    """Names of input files inside the study's raw-data folder."""
    sample_metadata: str  # file name inside the study's raw-data folder; must be <study.id>_sample_metadata.csv

    @field_validator("sample_metadata")
    @classmethod
    def plain_csv_name(cls, v: str) -> str:
        """Reject folders and path tricks: the sheet must be a bare *.csv file name."""
        if not is_safe_name(v) or not v.endswith(".csv"):
            raise ValueError("files.sample_metadata must be a plain file name ending in .csv (no folders)")
        return v

class SystemSuitability(_Section):                      # standards_only
    """Checks that each standard's expected m/z is seen in its run."""
    ppm_tolerance: float = Field(gt=0)   # allowed m/z error [ppm]


class FeatureDetection(_Section):
    """Finding chromatographic peaks (features) in each raw file's MS1 scans."""
    mass_error_ppm: float = Field(gt=0)        # m/z tolerance for joining peaks of one ion across scans [ppm]
    peak_width: tuple[float, float]            # accepted peak base width [min, max] in seconds
    max_gaps: int = Field(gt=0)                # consecutive scans an ion may be missing before its trace ends
    min_scans: int = Field(ge=1)               # minimum scans in a peak (raised automatically to fit peak_width[0])
    noise_threshold: float = Field(gt=0)       # absolute intensity floor [counts]; higher = fewer, cleaner peaks
    method: Literal["chromatogram", "legacy"] = "chromatogram"   # "legacy" = previous detector, for comparison runs
    noise_mode: Literal["absolute", "adaptive"] = "absolute"     # adaptive: threshold = snr_threshold x scan median
    snr_threshold: float = Field(default=3.0, gt=0)          # noise_mode="adaptive"
    smoothing_points: int = Field(default=5, ge=1)           # width of the smoothing window applied before peak picking
    min_relative_prominence: float = Field(default=0.2, ge=0, lt=1)   # fraction of apex height a peak must stand out by
    max_width_action: Literal["drop", "keep"] = "drop"       # what to do with peaks wider than peak_width[1]

    @field_validator("peak_width")
    @classmethod
    def validate_peak_width(cls, peak_width: tuple[float, float]) -> tuple[float, float]:
        """The lower bound must be smaller than the upper bound."""
        lower, upper = peak_width
        if lower >= upper:
            raise ValueError(f"peak_width lower bound {lower} must be < upper bound {upper}")
        return peak_width


class FeatureAlignment(_Section):                       # cohort_with_qc
    """Matching the same feature across samples into one feature table."""
    alignment_mz_tolerance_ppm: float = Field(gt=0)    # features closer than this in m/z may be the same [ppm]
    alignment_rt_tolerance_sec: float = Field(gt=0)    # ... and in retention time [s]
    algorithm: Literal["seeded", "legacy"] = "seeded"  # "legacy" = previous first-fit matching, for comparison runs
    rt_correction: bool = True                         # correct retention-time drift between samples (seeded only)
    reference_sample: Optional[str] = None             # sample all others are aligned to; None = the one with most features
    gap_fill: bool = True                              # re-extract intensity where a sample had no detected peak


class BlankQC(_Section):
    """Flagging features that come from the background (solvent blank) rather than the sample."""
    mz_tolerance_ppm: Optional[float] = Field(default=None, gt=0)   # standards_only (per-sample matching)
    rt_tolerance_sec: Optional[float] = Field(default=None, gt=0)    # standards_only
    blank_fold_threshold: float = Field(default=5.0, gt=0)   # cohort_with_qc (feature-table mode)


class AdductAnnotation(_Section):
    """Grouping features that are different ions (adducts, isotopes) of the same compound."""
    # standards_only
    primary_adduct: Optional[str] = None
    ppm_tolerance: Optional[float] = Field(default=None, gt=0)
    candidate_adducts: Optional[list[str]] = None
    # cohort_with_qc
    cohort_mz_tolerance_ppm: float = Field(default=10.0, gt=0)
    cohort_rt_tolerance_sec: float = Field(default=3.0, gt=0)
    correlation_threshold: float = Field(default=0.75, ge=0, le=1)
    cohort_method: Literal["hypothesis", "legacy"] = "hypothesis"
    cohort_adducts: Optional[list[str]] = None          # None = every adduct the stage knows
    use_filled: bool = True
    log_correlation: bool = True

    @field_validator("candidate_adducts")
    @classmethod
    def primary_must_be_in_candidates(cls, candidates, info):
        primary = info.data.get("primary_adduct")
        if candidates is not None and primary is not None and primary not in candidates:
            raise ValueError(f"primary adduct {primary} not in candidates {candidates}, it must be included")
        return candidates


class SpectralPurity(_Section):
    """How clean the MS2 precursor isolation was (share of the isolated signal that belongs to the target)."""
    isolation_window_da: float = Field(gt=0)
    min_purity: Optional[float] = Field(default=None, ge=0, le=1)   # standards_only pass/fail
    # cohort_with_qc confidence tiers
    min_purity_high_confidence: float = Field(default=0.8, ge=0, le=1)
    min_purity_tentative: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def tiers_ordered(self):
        """The tentative cut-off may not exceed the high-confidence one."""
        if self.min_purity_tentative > self.min_purity_high_confidence:
            raise ValueError("min_purity_tentative must be <= min_purity_high_confidence")
        return self


class LibraryMatching(_Section):
    """Identifying features by comparing their MS2 spectra with a reference spectral library."""
    reference_library: str
    reference_library_source: str
    reference_library_path: str
    reference_library_format: LibraryFormat
    similarity_metric: Literal["cosine"] = "cosine"      # the only metric implemented
    precursor_mz_tolerance_ppm: float = Field(gt=0)
    fragment_mz_tolerance_da: float = Field(gt=0)
    min_match_score: float = Field(ge=0, le=1)
    validate_against_known_identity: bool = False         # standards_only (filename ground truth)
    # cohort_with_qc
    ms2_precursor_tolerance_da: float = Field(default=0.01, gt=0)
    ms2_rt_tolerance_sec: Optional[float] = Field(default=15.0, gt=0)   # null = no RT constraint
    min_matched_peaks: int = Field(default=3, ge=1)
    intensity_power: float = Field(default=0.5, gt=0, le=1)
    remove_precursor: bool = True
    tie_margin: float = Field(default=0.02, ge=0, le=1)
    min_ms2_spectra_for_peak: int = Field(default=2, ge=1)
    library_qc: bool = True
    n_null_refs: int = Field(default=0, ge=0)             # >0 adds a chance p-value to each match (slower)


_STATS_STAGE_ONLY = ("test_method",)   # configures a stage, not the matrix preparation


class Statistics(_Section):                              # cohort_with_qc
    """How the feature table is prepared for group statistics, and which test is used."""
    normalization: Literal["pqn", "none"] = "pqn"
    presence_mode: Literal["per_group", "global"] = "per_group"
    min_presence_frac: float = Field(default=0.5, gt=0, le=1)
    drop_blank_flagged: bool = True
    collapse_adduct_groups: bool = True
    max_qc_rsd: Optional[float] = Field(default=None, gt=0)
    use_filled: bool = True
    test_method: Literal["moderated", "welch"] = "moderated"

    def prep_kwargs(self) -> dict:
        """The fields StatsPrepParams accepts (passing model_dump() whole raises TypeError on test_method)."""
        return {k: v for k, v in self.model_dump().items() if k not in _STATS_STAGE_ONLY}


class ExperimentConfig(_Section):
    """
    The whole study configuration (one YAML file). Sections tagged `standards_only` / `cohort_with_qc` are
    required only by that profile; see `required_for_profile` and `ignored_settings`.
    """
    study: StudyInfo
    dataset_profile: DatasetProfile
    files: FilesInfo
    system_suitability: Optional[SystemSuitability] = None      # standards_only
    feature_detection: FeatureDetection
    feature_alignment: Optional[FeatureAlignment] = None        # cohort_with_qc
    blank_qc: BlankQC
    adduct_annotation: AdductAnnotation
    spectral_purity: SpectralPurity
    library_matching: LibraryMatching
    statistics: Statistics = Field(default_factory=Statistics)
    config_version: str

    @model_validator(mode="after")
    def metadata_file_matches_study(self):
        """The sample sheet name is fixed by the study id (see the comment below)."""
        # the pipeline always reads <study.id>_sample_metadata.csv (LCMSContext); a different name written by the
        # API would produce a study that can never run
        expected = f"{self.study.id}_sample_metadata.csv"
        if self.files.sample_metadata != expected:
            raise ValueError(f"files.sample_metadata must be {expected!r}, got {self.files.sample_metadata!r}")
        return self

    @model_validator(mode="after")
    def required_for_profile(self):
        """Options only one profile reads are optional in the YAML, but mandatory for that profile."""
        missing: list[str] = []
        if self.dataset_profile == DatasetProfile.standards_only:
            if self.system_suitability is None:
                missing.append("system_suitability")
            for section, names in _STANDARDS_REQUIRED.items():
                model = getattr(self, section)
                missing += [f"{section}.{n}" for n in names if getattr(model, n) is None]
        else:
            if self.feature_alignment is None:
                missing.append("feature_alignment")
        if missing:
            raise ValueError(f"dataset_profile {self.dataset_profile.value!r} requires: {', '.join(missing)}")
        return self


# ---------------------------------------------------------------------------
# What each profile reads. Anything outside it that the YAML sets is reported by ignored_settings().
# ---------------------------------------------------------------------------
_STANDARDS_REQUIRED = {
    "blank_qc": ("mz_tolerance_ppm", "rt_tolerance_sec"),
    "adduct_annotation": ("primary_adduct", "ppm_tolerance", "candidate_adducts"),
    "spectral_purity": ("min_purity",),
}
# whole sections / single options that cohort_with_qc never reads
_UNUSED_IN_COHORT = {
    "system_suitability": None,          # None = the whole section
    "blank_qc": {"mz_tolerance_ppm", "rt_tolerance_sec"},
    "adduct_annotation": {"primary_adduct", "ppm_tolerance", "candidate_adducts"},
    "spectral_purity": {"min_purity"},
}
# ... and standards_only never reads
_UNUSED_IN_STANDARDS = {
    "feature_alignment": None,
    "statistics": None,
    "blank_qc": {"blank_fold_threshold"},
    "adduct_annotation": {"cohort_mz_tolerance_ppm", "cohort_rt_tolerance_sec", "correlation_threshold",
                          "cohort_method", "cohort_adducts", "use_filled", "log_correlation"},
    "spectral_purity": {"min_purity_high_confidence", "min_purity_tentative"},
    "library_matching": {"ms2_precursor_tolerance_da", "ms2_rt_tolerance_sec", "min_matched_peaks",
                         "intensity_power", "remove_precursor", "tie_margin", "min_ms2_spectra_for_peak",
                         "library_qc", "n_null_refs"},
}


# read by no stage in either profile (kept so existing YAMLs load; remove it or implement it)
_UNUSED_EVERYWHERE = {"library_matching": {"validate_against_known_identity"}}


def ignored_settings(config: ExperimentConfig, only_changed: bool = False) -> list[str]:
    """
    "section" or "section.option" for everything the YAML writes out that the dataset profile does not
    use. Options left out of the YAML (defaults) are never listed.

    only_changed=True keeps only options whose value differs from the schema default, i.e. the ones
    someone tuned that do nothing. Whole sections and options that default to None (a template copied
    from another profile carries those) are then left out.
    """
    unused = (_UNUSED_IN_STANDARDS if config.dataset_profile == DatasetProfile.standards_only
              else _UNUSED_IN_COHORT)
    unused = {k: (None if v is None else set(v)) for k, v in unused.items()}
    for section, names in _UNUSED_EVERYWHERE.items():
        unused[section] = None if unused.get(section, set()) is None else unused.get(section, set()) | names
    out: list[str] = []
    for section, names in unused.items():
        model = getattr(config, section)
        if model is None or section not in config.model_fields_set:
            continue
        if names is None:
            if not only_changed:
                out.append(section)
            continue
        for n in sorted(names & model.model_fields_set):
            field = type(model).model_fields[n]
            if only_changed and (field.is_required() or field.default is None
                                 or getattr(model, n) == field.get_default(call_default_factory=True)):
                continue
            out.append(f"{section}.{n}")
    return out


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    """
    Load and validate a study YAML.

    Raises:
        pydantic.ValidationError: unknown keys, bad values or options missing for the chosen profile.
        FileNotFoundError: `config_path` does not exist.
    """
    with open(config_path, "r") as f:
        contents = yaml.safe_load(f)

    return ExperimentConfig(**contents)
