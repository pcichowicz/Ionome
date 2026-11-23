"""
FastAPI dependencies shared by the endpoints: the database session, the validated {study_id} path parameter and the Experiment row it names.
"""
from typing import Annotated

from fastapi import Depends, HTTPException, Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db import get_async_session
from core.models import Experiment
from core.safe_paths import STUDY_ID_PATTERN

SessionDep = Annotated[AsyncSession, Depends(get_async_session)]

# Every route with a {study_id} validates it here, before it is used in a path or a query.
StudyId = Annotated[str, Path(pattern=STUDY_ID_PATTERN, max_length=64)]


async def get_experiment_or_404(study_id: StudyId, session: SessionDep) -> Experiment:
    """Load the Experiment for the {study_id} in the URL, or answer 404."""
    result = await session.execute(select(Experiment).where(Experiment.study_id == study_id))
    experiment = result.scalar_one_or_none()
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"unknown study_id: {study_id}")
    return experiment


ExperimentDep = Annotated[Experiment, Depends(get_experiment_or_404)]
