"""
Ingestion (first stage in both profiles): check that every sample in the sample sheet has a raw file.

Starts from mzML/mzXML, not vendor RAW files. It (1) validates that each `sample_id` in the sheet has a raw file in
the study folder, (2) records what was found/missing in `qc_metrics["ingestion"]`, and (3) logs provenance, so every
later stage can assume the raw files are present.

Some samples missing is recoverable (later stages skip them); none found at all is fatal.
"""
from __future__ import annotations
from pathlib import Path

from analysis.context import LCMSContext
from analysis.pipeline import FatalStageError, RecoverableStageError
from core.ingestion import find_raw_file

class MzMLIngestionStage:
    """
    Verify raw-file presence for every sample in the sheet (stage 1 of both profiles).

    Reads: context.sample_metadata and the raw-file folder. Writes: qc_metrics["ingestion"] with n_expected, n_found
    and missing_sample_ids.
    """
    _name: str = "ingestion"
    # requires: stages whose results this one reads (must be registered earlier; checked before anything runs).
    # runs_after: stages it only reads if present; if they are registered they must come earlier.
    requires: tuple[str, ...] = ()

    def __init__(self, mzml_dir: Path):
        """`mzml_dir`: the study's raw-data folder (<LCMS_RAW_DATA_DIR>/<study_id>)."""
        self.mzml_dir = Path(mzml_dir)

    def validate_input(self, context: LCMSContext) -> bool:
        """Fatal if the sample sheet was not loaded or the raw-data folder does not exist."""
        if context.sample_metadata is None:
            raise FatalStageError("ingestion: sample_metadata not loaded on context")
        if not self.mzml_dir.exists():
            raise FatalStageError(f"ingestion: mzml_dir does not exist: {self.mzml_dir}")
        return True

    def execute(self, context: LCMSContext) -> LCMSContext:
        """
        Look up each sample's raw file and record the result.

        Raises:
            FatalStageError: no sample has a raw file.
            RecoverableStageError: some samples have none (the run continues without them).
        """
        sample_ids = list(context.sample_metadata["sample_id"])
        found: dict[str, Path] = {}
        missing: list[str] = []

        for sample_id in sample_ids:
            candidate = find_raw_file(self.mzml_dir, sample_id)
            if candidate is not None:   # find_raw_file returns None for a missing file
                found[sample_id] = candidate
            else:
                missing.append(sample_id)

        if not found:
            raise FatalStageError(
                f"{self._name}: no raw files (.mzML/.mzXML) found for any of {len(sample_ids)} samples "
                f"in {self.mzml_dir}"
            )

        context.qc_metrics[f"{self._name}"] = {
            "n_expected": len(sample_ids),
            "n_found": len(found),
            "missing_sample_ids": missing,
        }

        warnings = [f"missing raw file for sample: {s}" for s in missing]
        context.log_step(
            self._name,
            parameters={"mzml_dir": str(self.mzml_dir)},
            metrics={"n_found": len(found),
                     "n_missing": len(missing)},
            warnings=warnings,
            input_files=[str(p) for p in found.values()],
        )

        if missing:
            # Individual missing files are recoverable -- downstream stages
            # just won't have data for those sample_ids. If every sample is
            # missing, that's caught above as fatal.
            raise RecoverableStageError(
                f"{len(missing)}/{len(sample_ids)} samples missing mzML files: {missing}"
            )

        return context

    def validate_output(self, context: LCMSContext) -> bool:
        """The raw-data folder is set on the context."""
        return context.mzml_dir is not None