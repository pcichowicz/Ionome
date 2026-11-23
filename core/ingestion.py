"""
Study registration, step 2: turn the rows of a study's sample sheet into `Sample` database rows.

Each sheet row whose raw file exists becomes one `Sample` (with its path and a SHA-256 checksum). Rows already
registered are skipped, so re-running after adding files only adds the new ones. Rows without a raw file are
reported, never silently dropped and never fatal.
"""
import hashlib
from pathlib import Path

import pandas as pd

from core.db import SyncSessionLocal
from core.models import Sample, Experiment
# find_raw_file lives in core.raw_files; re-exported here so existing imports keep working
from core.raw_files import find_raw_file, is_raw_file, RAW_SUFFIXES  # noqa: F401
from core.status import STATUS


def _sha256_of_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in 8 KiB chunks so large raw files are not loaded into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_samples_from_csv(experiment_pk: int, csv_path: Path, mzml_dir: Path) -> dict:
    """
    Register the samples listed in a sample sheet for one study.

    Args:
        experiment_pk: primary key of the study's `Experiment` row.
        csv_path: the sample sheet; must contain a `sample_id` column (`sample_role` is stored as the description).
        mzml_dir: folder holding the raw files (<sample_id>.mzML or .mzXML).

    Returns:
        Counts and ids: n_created/created, n_skipped_existing, n_skipped_missing_file/skipped_missing_file.

    Raises:
        ValueError: the sheet has no `sample_id` column.
    """
    df = pd.read_csv(csv_path)
    if "sample_id" not in df.columns:
        raise ValueError(f"sample_metadata csv missing 'sample_id' column: {csv_path}")

    created, skipped_existing, skipped_missing_file = [], [], []

    with SyncSessionLocal() as session:
        for _, row_data in df.iterrows():
            sample_id = row_data["sample_id"]

            existing = (
                session.query(Sample)
                .filter(Sample.experiment_id == experiment_pk, Sample.sample_id == sample_id)
                .first()
            )
            if existing is not None:
                skipped_existing.append(sample_id)
                continue

            raw_file = find_raw_file(mzml_dir, sample_id)
            if raw_file is None:
                skipped_missing_file.append(sample_id)
                continue

            session.add(Sample(
                experiment_id=experiment_pk,
                sample_id=sample_id,
                description=row_data.get("sample_role"),
                raw_file_path=str(raw_file),
                checksum=_sha256_of_file(raw_file),
                status=STATUS.PENDING,
            ))
            created.append(sample_id)

        session.commit()

    return {
        "n_created": len(created), "created": created,
        "n_skipped_existing": len(skipped_existing),
        "n_skipped_missing_file": len(skipped_missing_file),
        "skipped_missing_file": skipped_missing_file,
    }

def get_experiment_pk_sync(study_id: str) -> int:
    """
    Resolve a study's string slug (config.study.id) to its integer Experiment primary key.

    Read-only: the Experiment row must already exist (created by registering the study, in the web app or with
    scripts/register_study.py).

    Raises:
        ValueError: the study is not registered.
    """
    with SyncSessionLocal() as session:
        experiment = session.query(Experiment).filter(
            Experiment.study_id == study_id
        ).one_or_none()
        if experiment is None:
            raise ValueError(
                f"No Experiment found for study_id={study_id!r} -- "
                f"has /studies/{study_id}/ingest-samples been called yet?"
            )
        return experiment.id