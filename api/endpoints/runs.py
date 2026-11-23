"""
Start a pipeline run and follow its progress. The run itself happens in a worker process (services/runs.py); these routes only record state and answer quickly.
"""
import uuid

from fastapi import APIRouter, HTTPException

from api.deps import ExperimentDep
from api.schemas.run import RunStartResult, RunStepOut
from api.services.runs import claim_run, submit_run
from core.db import SyncSessionLocal
from core.models import Pipeline

router = APIRouter(prefix="/studies/{study_id}", tags=["runs"])


@router.post("/run", response_model=RunStartResult)
async def run_study(experiment: ExperimentDep):
    """Start a run: 409 if one is already active, otherwise return its run_id (poll GET /runs/{run_id})."""
    # Two runs of one study would write the same checkpoints, tables and plots at the same time.
    if not claim_run(experiment.study_id):
        raise HTTPException(status_code=409, detail=f"{experiment.study_id} is already running")
    run_id = str(uuid.uuid4())
    try:
        submit_run(experiment.study_id, experiment.id, run_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"could not start the run: {exc}") from exc
    return RunStartResult(study_id=experiment.study_id, run_id=run_id, status="PENDING")


@router.get("/status")
async def get_status(experiment: ExperimentDep):
    """Overall study status (PENDING / RUNNING / SUCCESS / FAILED)."""
    return {
        "study_id": experiment.study_id,
        "status": experiment.status.value,
        "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
    }


@router.get("/runs/{run_id}", response_model=list[RunStepOut])
async def get_run_status(experiment: ExperimentDep, run_id: str):
    """Per-stage status of one run, in start order (also carries the error message of a failed stage)."""
    with SyncSessionLocal() as session:
        rows = (
            session.query(Pipeline)
            .filter(Pipeline.run_id == run_id, Pipeline.experiment_id == experiment.id)
            .order_by(Pipeline.started_at)
            .all()
        )
    if not rows:
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
    return [
        RunStepOut(
            step_name=r.step_name,
            status=r.status.value,
            error_msg=r.error_msg,
            started_at=r.started_at,
            finished_at=r.finished_at,
        )
        for r in rows
    ]
