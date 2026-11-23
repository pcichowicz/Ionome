"""
Find and read a study's result files under `<LCMS_RESULTS_DIR>/<study_id>/` for the results endpoints.
Nothing here trusts a client-supplied name: plot categories are an allowlist and file names are checked against the folder.
"""
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.results import PlotInfo, RunSummary, TablePreview
from core.models import Pipeline
from core.settings import settings

# The cohort export writes file names prefixed with the study id, the standards export writes plain names.
# The profile is not stored on the Experiment row, so each lookup simply tries both.

def resolve_qc_report_path(study_id: str) -> Path | None:
    """Path of the QC report JSON, or None if the study has none yet."""
    qc_dir = settings.lcms_results_dir / study_id / "qc_reports"
    for candidate in (qc_dir / f"{study_id}_qc_report.json", qc_dir / "qc_report.json"):
        if candidate.exists():
            return candidate
    return None


def resolve_table_csv_path(study_id: str) -> Path | None:
    """Path of the feature table CSV (cohort) or library CSV (standards), or None if not exported yet."""
    library_dir = settings.lcms_results_dir / study_id / "library"
    for candidate in (library_dir / f"{study_id}_feature_table.csv", library_dir / "library.csv"):
        if candidate.exists():
            return candidate
    return None


# The plots/<category>/ folders the stages write into. An explicit allowlist (instead of globbing plots/**) so a
# request can only reach these folders and never other files in the results directory.
# Add a category here when a new plotting stage writes to a new folder.
PLOT_CATEGORIES = [
    "chromatograms", "summary_stats", "volcano", "heatmap_clustering", "plsda", "xic",
]


def list_plots(study_id: str) -> list[PlotInfo]:
    """All PNGs in the allowlisted categories, in category order then file-name order."""
    plots_dir = settings.lcms_results_dir / study_id / "plots"
    found = []
    for category in PLOT_CATEGORIES:
        category_dir = plots_dir / category
        if not category_dir.is_dir():
            continue
        for png in sorted(category_dir.glob("*.png")):
            found.append(PlotInfo(
                category=category,
                filename=png.name,
                url=f"/studies/{study_id}/results/plots/{category}/{quote(png.name)}",
            ))
    return found


def resolve_plot_file(study_id: str, category: str, filename: str) -> Path | None:
    """
    Path of one plot for the plot endpoint, or None. Only returns a .png that exists directly inside an allowlisted
    category folder of this study; names like '../../x' are rejected because the resolved file's parent must be
    exactly that folder.
    """
    if category not in PLOT_CATEGORIES:
        return None
    category_dir = (settings.lcms_results_dir / study_id / "plots" / category).resolve()
    candidate = (category_dir / filename).resolve()
    if candidate.parent != category_dir or candidate.suffix != ".png" or not candidate.exists():
        return None
    return candidate


def get_table_preview(study_id: str, n_rows: int = 20) -> TablePreview | None:
    """First n_rows of the table plus the total row count (the whole file is not rendered); NaN becomes null."""
    csv_path = resolve_table_csv_path(study_id)
    if csv_path is None:
        return None
    preview_df = pd.read_csv(csv_path, nrows=n_rows)
    with open(csv_path) as f:
        total_rows = sum(1 for _ in f) - 1  # header line doesn't count
    return TablePreview(
        columns=list(preview_df.columns),
        rows=preview_df.where(pd.notna(preview_df), None).to_dict(orient="records"),
        total_rows=total_rows,
        download_url=f"/studies/{study_id}/results/table.csv",
    )


async def list_runs(experiment_id: int, session: AsyncSession) -> list[RunSummary]:
    """Group the per-stage rows by run. Run status: FAILED if any stage failed, else RUNNING if any is running, else SUCCESS."""
    rows = (
        await session.execute(
            select(Pipeline).where(Pipeline.experiment_id == experiment_id).order_by(Pipeline.started_at)
        )
    ).scalars().all()

    by_run: dict[str, list[Pipeline]] = {}
    for row in rows:
        by_run.setdefault(row.run_id, []).append(row)

    summaries = []
    for run_id, steps in by_run.items():
        statuses = {s.status.value for s in steps}
        if "FAILED" in statuses:
            status = "FAILED"
        elif "RUNNING" in statuses:
            status = "RUNNING"
        else:
            status = "SUCCESS"
        summaries.append(RunSummary(
            run_id=run_id,
            status=status,
            started_at=min(s.started_at for s in steps if s.started_at),
            finished_at=max((s.finished_at for s in steps if s.finished_at), default=None),
            n_steps=len(steps),
        ))
    summaries.sort(key=lambda r: r.started_at or "", reverse=True)
    return summaries
