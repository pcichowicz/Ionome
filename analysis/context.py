"""
The run context: the one object handed from stage to stage.

It holds the study's configuration, its directories (all derived from the settings and the study id), the sample
sheet, and the small results each stage leaves behind (`qc_metrics`, `processing_log`). Large data stays on disk.
Everything it normally reads from settings/database can be injected, which is how the pipeline is unit-tested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import pandas as pd

from core.config_schema import ExperimentConfig
from analysis.utils.schemas import LibraryEntry
from core.paths import PROJECT_PATH, RESULT_SUBDIRS
from core.settings import settings

@dataclass
class LCMSContext:
    """
    Core context object that flows through every pipeline stage.

    LCMSContext carries file path and small in-memory metadata/results between
    stages -- NOT full DataFrames, spectra, or raw arrays. Large intermediate
    data (mzML, featureXML, etc.) lives on disk; the context just tracks where.
    This keeps memory bounded regardless of dataset size and makes every stage
    independently re-runnable/debuggable.
    """
    study_id: str
    experiment_id: int
    config: ExperimentConfig

    # Runtime Configurations (loads from yaml)
    dataset_profile: str = field(init=False)
    polarity: str = field(init=False)
    instrument: str = field(init=False)
    yaml_config: dict[str, Any] = field(default_factory=dict, repr=False)

    base_dir: Path = field(default=PROJECT_PATH)
    # Injectable: pass these (and sample_metadata / mzml_file_paths below) and the context does no settings,
    # database or CSV access at all -- that is how the pipeline is tested without any environment.
    # Left out, they are derived from the settings and the database as before.
    results_dir: Optional[Path] = None
    mzml_dir: Optional[Path] = None
    featurejson_dir: Path = field(init=False)
    library_dir: Path = field(init=False)
    qc_report_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)

    yaml_path: Path = field(init=False)
    sample_metadata_path: Path = field(init=False)
    aligned_features_path: Path = field(init=False)
    mzml_file_paths: Optional[list[Path]] = field(default=None, repr=False)

    # In memory Data / State
    sample_metadata: Optional[pd.DataFrame] = field(default=None, repr=False)
    qc_metrics: dict[str, Any] = field(default_factory=dict)
    processing_log: list[dict[str, Any]] = field(default_factory=list)
    # Cache chain: hash covering every stage that has run so far this invocation (set by the pipeline).
    # Each stage's cache key includes it, so a changed upstream stage invalidates everything after it.
    stage_chain: str = ""

    #Final outputs
    library_entries: list[LibraryEntry] = field(default_factory=list)
    final_feature_table: pd.DataFrame = field(default=None, repr=False)

    def __post_init__(self):
        """
        Resolves paths, creates dirs, loads sample metadata. Anything passed to the constructor is used as given;
        only what is missing is read from the settings, the metadata CSV and the database.
        """
        self.base_dir = Path(self.base_dir).resolve()
        self.results_dir = Path(self.results_dir) if self.results_dir is not None \
            else settings.lcms_results_dir / self.study_id
        self.mzml_dir = Path(self.mzml_dir) if self.mzml_dir is not None \
            else settings.lcms_raw_data_dir / self.study_id

        self.sample_metadata_path = self.mzml_dir / f"{self.study_id}_sample_metadata.csv"

        self.polarity = self.config.study.polarity
        self.instrument = self.config.study.instrument
        self.dataset_profile = self.config.dataset_profile

        if self.sample_metadata is None:
            self._validate_dirs()
            self.sample_metadata = pd.read_csv(self.sample_metadata_path)
        if self.mzml_file_paths is None:
            self.mzml_file_paths = self._load_mzml_file_paths()
        else:
            self.mzml_file_paths = [Path(p) for p in self.mzml_file_paths]

        #outputs dir
        self.featurejson_dir = self.results_dir / "features"
        self.library_dir = self.results_dir / "library"
        self.qc_report_dir = self.results_dir / "qc_reports"
        self.plots_dir = self.results_dir / "plots"
        self.logs_dir = self.results_dir / "logs"
        self.cache_dir = self.results_dir / "cache"

        self.make_dirs(subdirs=RESULT_SUBDIRS)

    def log_step(
        self,
        step_name: str,
        parameters: dict[str, Any],
        metrics: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        input_files: list[str] | None = None,
        output_files: list[str] | None = None,
    ) -> None:
        """
        Record a provenance entry for a completed (or partially-failed) stage.

        Every stage should call this exactly once, even on a recoverable
        failure
        """
        self.processing_log.append(
            {
            "stage": step_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parameters": parameters,
            "metrics": metrics or {},
            "warnings": warnings or [],
            "input_files": input_files or [],
            "output_dirs": output_files or [],
            }
        )

    def _validate_dirs(self):
        """Raise FileNotFoundError unless the study's sample sheet exists (it is created at registration)."""
        if not self.sample_metadata_path.exists():
            raise FileNotFoundError(f"ERROR: Metadata CSV missing at {self.sample_metadata_path}")

    def _load_mzml_file_paths(self) -> list[Path]:
        """Raw-file paths of the study's registered samples, read from the database."""
        from core.db import SyncSessionLocal   # imported here: a context that is given its paths never needs the DB
        from core.models import Sample
        with SyncSessionLocal() as session:
            samples = session.query(Sample).filter(
                Sample.experiment_id == self.experiment_id
            ).all()
        return [Path(s.raw_file_path) for s in samples if s.raw_file_path]

    def make_dirs(self, exist_ok: bool = True, subdirs: list[str] | None = None) -> None:
        """Create the study's results folder and the given sub-folders (features/, plots/, ...)."""
        self.results_dir.mkdir(exist_ok=exist_ok, parents=True)

        if subdirs:
            for s in subdirs:
                (self.results_dir / s).mkdir(exist_ok=exist_ok, parents=True)

    def resolve(self, *path_parts) -> Path:
        """Join the parts into a path under the results folder (absolute paths pass through); creates the parent folder."""
        p = Path(*[part for part in path_parts if part is not None])
        out = p if p.is_absolute() else self.results_dir / p
        out.parent.mkdir(exist_ok=True, parents=True)
        return out

    def path_for(self, stem: str, ext: str = "", subdir: str | None = None) -> Path:
        """`<results>/<subdir>/<stem><ext>`; the leading dot of `ext` is optional."""
        if ext and not ext.startswith("."):
            ext = f".{ext}"
        name = f"{stem}{ext}"
        return self.resolve(subdir, name)
