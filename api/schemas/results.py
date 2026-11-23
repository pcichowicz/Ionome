"""
Response shapes of the results endpoints.
"""
from datetime import datetime
from pydantic import BaseModel


class PlotInfo(BaseModel):
    """One plot image: folder under plots/, file name and the URL that serves it."""
    category: str      # folder under plots/, e.g. "chromatograms", "volcano", "xic"
    filename: str
    url: str


class TablePreview(BaseModel):
    """Head of the feature table / library, for the in-browser preview.
    Full download is a separate endpoint -- 15-20k rows isn't rendered here."""
    columns: list[str]
    rows: list[dict]
    total_rows: int
    download_url: str


class RunSummary(BaseModel):
    """One run in the history list, condensed from its per-stage rows."""
    run_id: str
    status: str  # derived: FAILED if any step failed, RUNNING if any step still running, else SUCCESS
    started_at: datetime | None
    finished_at: datetime | None
    n_steps: int
