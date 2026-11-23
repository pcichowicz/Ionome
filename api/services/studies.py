"""
Study-level logic behind the study endpoints: dashboard summaries, raw-file scan, sample readiness and the intake
(create a study = validate, write YAML and sample sheet, register, ingest). `create_study_files_and_ingest` is also
used by scripts/register_study.py.
"""
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config_schema import ExperimentConfig
from core.ingestion import find_raw_file, ingest_samples_from_csv
from core.models import Experiment, Pipeline, Sample
from core.safe_paths import is_safe_name, safe_join
from core.settings import settings
from core.raw_files import list_raw_files
from core.status import STATUS
from api.schemas.study import (
    RawFileScanResult,
    SampleReadiness,
    SampleStatusCounts,
    StudyCreateResult,
    StudySummary,
)

# STATUS.DONE (core/status.py) is not used anywhere, so it has no counter. Give it one here if it ever gets a meaning.
_COUNT_FIELDS = {
    STATUS.PENDING: "pending",
    STATUS.RUNNING: "running",
    STATUS.SUCCESS: "success",
    STATUS.FAILED: "failed",
}


async def list_study_summaries(session: AsyncSession) -> list[StudySummary]:
    """One summary per study: status, sample counts per status and the time of the last run (3 queries in total)."""
    experiments = (await session.execute(select(Experiment))).scalars().all()

    sample_rows = (
        await session.execute(
            select(Sample.experiment_id, Sample.status, func.count())
            .group_by(Sample.experiment_id, Sample.status)
        )
    ).all()
    counts_by_experiment: dict[int, SampleStatusCounts] = {}
    for experiment_id, status, count in sample_rows:
        bucket = counts_by_experiment.setdefault(experiment_id, SampleStatusCounts())
        bucket.total += count
        field = _COUNT_FIELDS.get(status)
        if field is not None:
            setattr(bucket, field, getattr(bucket, field) + count)

    last_run_rows = (
        await session.execute(
            select(Pipeline.experiment_id, func.max(Pipeline.started_at))
            .group_by(Pipeline.experiment_id)
        )
    ).all()
    last_run_by_experiment = dict(last_run_rows)

    return [
        StudySummary(
            study_id=e.study_id,
            dataset_profile=_read_dataset_profile(e.config_path),
            status=e.status.value,
            sample_counts=counts_by_experiment.get(e.id, SampleStatusCounts()),
            created_at=e.created_at,
            last_run_at=last_run_by_experiment.get(e.id),
        )
        for e in experiments
    ]


def _read_dataset_profile(config_path: str) -> str | None:
    """The dataset_profile from the study's YAML (it is not stored in the database); None if the file is missing or unreadable."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f).get("dataset_profile")
    except (OSError, AttributeError):
        return None


def load_config_defaults() -> dict:
    """The stage parameters the wizard pre-fills: config/experiments/template.yaml, the single source of default values."""
    template_path = Path(__file__).resolve().parents[2] / "config" / "experiments" / "template.yaml"
    with open(template_path) as f:
        return yaml.safe_load(f)


def scan_raw_dir(study_id: str) -> RawFileScanResult:
    """
    Candidate sample ids: the file-name stems of the .mzML/.mzXML files in the study's raw-data folder.
    Needs no database row (it is wizard step 1, before the study exists).
    """
    raw_dir = safe_join(settings.lcms_raw_data_dir, study_id)
    if not raw_dir.is_dir():
        return RawFileScanResult(
            study_id=study_id, raw_data_dir=str(raw_dir), dir_exists=False, sample_ids=[]
        )
    sample_ids = sorted({p.stem for p in list_raw_files(raw_dir)})
    return RawFileScanResult(
        study_id=study_id, raw_data_dir=str(raw_dir), dir_exists=True, sample_ids=sample_ids
    )


async def get_sample_readiness(study_id: str, experiment_id: int, session: AsyncSession) -> SampleReadiness:
    """Compare the registered samples with the raw files on disk (feeds the "N of M samples present" banner)."""
    expected = sorted(
        (await session.execute(select(Sample.sample_id).where(Sample.experiment_id == experiment_id)))
        .scalars().all()
    )
    present = set(scan_raw_dir(study_id).sample_ids)
    return SampleReadiness(
        study_id=study_id,
        expected=expected,
        present=sorted(present),
        missing=sorted(set(expected) - present),
    )


class StudyCreateError(ValueError):
    """Intake validation failed (config, sample ids, missing raw files or duplicate study id); `errors` lists every problem as text."""
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def create_study_files_and_ingest(config_dict: dict, sample_rows: list[dict]) -> StudyCreateResult:
    """
    Create a study in one step: validate the config and sample rows, check that every sample has a raw file, then
    write `<yaml_dir>/<study_id>.yaml` and the sample sheet, add the Experiment row and ingest the samples.
    Nothing is written unless all checks pass; raises StudyCreateError listing every problem.
    Synchronous: async callers wrap it in asyncio.to_thread.
    """
    errors: list[str] = []

    try:
        config = ExperimentConfig(**config_dict)
    except Exception as exc:
        raise StudyCreateError([f"config validation failed: {exc}"]) from exc

    if not sample_rows:
        errors.append("no sample rows submitted")
    sample_ids = [row.get("sample_id") for row in sample_rows]
    if any(sid in (None, "") for sid in sample_ids):
        errors.append("every sample row needs a non-empty sample_id")

    study_id = config.study.id          # already restricted to [A-Za-z0-9_-] by the config schema
    raw_dir = safe_join(settings.lcms_raw_data_dir, study_id)
    # sample ids become file names (<id>.mzML, plots, ...): reject separators, "..", hidden names
    bad_ids = [sid for sid in sample_ids if sid and not is_safe_name(str(sid))]
    if bad_ids:
        errors.append("sample_id(s) with characters not allowed in file names (/, \\, .., leading dot): "
                      + ", ".join(repr(s) for s in bad_ids[:10]))
    duplicate_ids = sorted({str(s) for s in sample_ids if s and sample_ids.count(s) > 1})
    if duplicate_ids:
        errors.append(f"duplicate sample_id(s): {', '.join(duplicate_ids[:10])}")
    missing_files = [
        sid for sid in sample_ids
        if sid and sid not in bad_ids and find_raw_file(raw_dir, sid) is None
    ]
    if missing_files:
        errors.append(
            f"no raw file found for sample_id(s) in {raw_dir}: {', '.join(missing_files)}"
        )

    from core.db import SyncSessionLocal
    with SyncSessionLocal() as session:
        existing = session.query(Experiment).filter(Experiment.study_id == study_id).first()
    if existing is not None:
        errors.append(f"study_id {study_id!r} already exists")

    if errors:
        raise StudyCreateError(errors)

    csv_path = safe_join(raw_dir, config.files.sample_metadata)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample_rows).to_csv(csv_path, index=False)

    yaml_path = safe_join(settings.yaml_dir, f"{study_id}.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

    with SyncSessionLocal() as session:
        experiment = Experiment(
            study_id=study_id,
            config_path=str(yaml_path),
            status=STATUS.PENDING,
        )
        session.add(experiment)
        session.commit()
        session.refresh(experiment)
        experiment_id = experiment.id

    ingestion_result = ingest_samples_from_csv(experiment_id, csv_path, raw_dir)

    return StudyCreateResult(
        study_id=study_id,
        experiment_id=experiment_id,
        yaml_path=str(yaml_path),
        csv_path=str(csv_path),
        ingestion=ingestion_result,
    )
