"""
Interfaces (typing.Protocol) that stages and their file readers implement.

Stages depend on these, not on a concrete parser, so a stage can be tested with a small fake reader. The real
implementations are the Pyteomics-based classes in analysis.utils.mzml_parsing.
"""
from pathlib import Path
from typing import Protocol
from analysis.context import LCMSContext

# ======== Pipeline =========== #
class PipelineStage(Protocol):
    """
    Every stage (ingestion, feature detection, ...) implements this.
    """

    _name: str                     # the pipeline, the cache and the logs all key on this
    requires: tuple[str, ...]      # optional on a stage; default is no prerequisites
    runs_after: tuple[str, ...]    # optional on a stage

    def validate_input(self, context: LCMSContext) -> bool:
        """
        Check prerequisites are present before running. Return False
        (not raise) for "not ready yet"; raise FatalStageError for
        "something is actually wrong".
        """
        ...

    def execute(self, context: LCMSContext) -> LCMSContext:
        """
        Do the work; return the (mutated) context. Raise
        RecoverableStageError or FatalStageError as appropriate.
        """
        ...

    def validate_output(self, context: LCMSContext) -> bool:
        """
        Sanity-check output before moving on to the next stage.
        """
        ...

# ======== System Suitability =========== #
class PrecursorReader(Protocol):
    """
    Reads the set of DDA precursor m/z values selected for MS2 in an mzML file.
    """

    def get_precursor_mzs(self, mzml_path: Path) -> list[float]:
        """Distinct precursor m/z values (floats) selected for MS2 in the file."""
        ...

# ======== Feature Detection =========== #
class FeatureDetector(Protocol):
    """Finds chromatographic features in one raw file."""

    def detect_features(
        self, mzml_path: Path, output_path: Path, params: dict
    ) -> list[dict]:
        """
        Run feature detection on mzml_path, write full results to
        output_path (e.g. featureJSON), and return a lightweight summary
        list: [{"mz": ..., "rt": ..., "intensity": ..., "charge": ...}, ...]
        """
        ...

# ======== Library Matching =========== #
class MS2SpectrumReader(Protocol):
    """
    Extracts a single representative MS2 peak list for a given
    precursor m/z
    """

    def get_ms2_spectrum(
        self, mzml_path: Path, precursor_mz: float, precursor_tolerance_da: float
    ) -> list[tuple[float, float]]:
        """
        Return [(mz, intensity), ...] for the best MS2 scan matching
        precursor_mz, or [] if no matching MS2 scan has any peaks.
        """
        ...
