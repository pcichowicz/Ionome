"""
Response shape for a sample.
"""
from datetime import datetime
from pydantic import BaseModel


class SampleOut(BaseModel):
    """A registered sample; the checksum identifies the raw file that was ingested."""
    sample_id: str
    description: str | None
    status: str
    checksum: str
    ingested_at: datetime | None
