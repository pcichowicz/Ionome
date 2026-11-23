"""
Response shapes of the run endpoints.
"""
from datetime import datetime
from pydantic import BaseModel


class RunStepOut(BaseModel):
    """Status of one stage within a run."""
    step_name: str
    status: str
    error_msg: str | None
    started_at: datetime | None
    finished_at: datetime | None


class RunStartResult(BaseModel):
    """Returned when a run is accepted; the run_id is used to poll its stages."""
    study_id: str
    run_id: str
    status: str
