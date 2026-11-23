"""
Pipeline runs.

A run is CPU- and memory-heavy and can last many minutes, so it does not run in the API process (a
BackgroundTasks thread shares the process, its memory and its restarts with the web server). Each run goes to a
worker process from a small pool (LCMS_MAX_WORKERS). The API process only records state:

  claim_run          atomically moves an experiment to RUNNING, or says it already is (-> HTTP 409)
  submit_run         hands the run to the pool and attaches a callback for crashes
  recover_interrupted_runs   at startup, anything still RUNNING belongs to a process that no longer exists
"""
from __future__ import annotations

import importlib
import logging
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, UTC

from sqlalchemy import update

from core.db import SyncSessionLocal
from core.models import Experiment, Pipeline, Sample
from core.settings import settings
from core.status import STATUS

logger = logging.getLogger(__name__)

DEFAULT_RUNNER = "analysis.pipeline_orchestrator:run_pipeline_for_study"

_pool: ProcessPoolExecutor | None = None


# ---- state in the database (sync sessions only: these run in worker processes and pool threads too) ----------

def claim_run(study_id: str) -> bool:
    """True if this call moved the experiment to RUNNING; False if a run is already active."""
    with SyncSessionLocal() as session:
        result = session.execute(
            update(Experiment)
            .where(Experiment.study_id == study_id, Experiment.status != STATUS.RUNNING)
            .values(status=STATUS.RUNNING)
        )
        session.commit()
        return result.rowcount == 1


def set_experiment_status(study_id: str, status: STATUS) -> None:
    """Set the overall status of a study."""
    with SyncSessionLocal() as session:
        session.execute(update(Experiment).where(Experiment.study_id == study_id).values(status=status))
        session.commit()


def mark_samples(experiment_pk: int, status: STATUS, sample_ids: list[str] | None = None) -> None:
    """Set the status of all samples of an experiment, or only of `sample_ids`."""
    with SyncSessionLocal() as session:
        query = session.query(Sample).filter(Sample.experiment_id == experiment_pk)
        if sample_ids is not None:
            query = query.filter(Sample.sample_id.in_(sample_ids))
        query.update({Sample.status: status}, synchronize_session=False)
        session.commit()


def make_stage_event_recorder(run_id: str, experiment_pk: int):
    """
    Build the callback the pipeline calls on every stage start/finish: (stage_name, status, error_msg).
    It keeps one Pipeline row per stage and run (created on first event, updated afterwards, finished_at on SUCCESS/FAILED).
    """
    def _on_stage_event(stage_name: str, status: str, error_msg: str | None):
        with SyncSessionLocal() as session:
            row = (
                session.query(Pipeline)
                .filter(
                    Pipeline.run_id == run_id,
                    Pipeline.experiment_id == experiment_pk,
                    Pipeline.step_name == stage_name,
                )
                .one_or_none()
            )
            if row is None:
                row = Pipeline(run_id=run_id, experiment_id=experiment_pk, step_name=stage_name)
                session.add(row)
            row.status = STATUS[status]
            row.error_msg = error_msg
            if status in ("SUCCESS", "FAILED"):
                row.finished_at = datetime.now(UTC)
            session.commit()
    return _on_stage_event


def record_run_failure(run_id: str, experiment_pk: int, message: str) -> None:
    """A run that dies before or between stages (bad config, killed worker) has no failed stage row to show;
    give it one so the run history says what happened."""
    make_stage_event_recorder(run_id, experiment_pk)("pipeline", "FAILED", message[:2000])


def recover_interrupted_runs() -> int:
    """Startup: runs left RUNNING by a server that was stopped or crashed can never finish. Mark them failed."""
    with SyncSessionLocal() as session:
        stuck = session.query(Experiment).filter(Experiment.status == STATUS.RUNNING).all()
        now = datetime.now(UTC)
        for exp in stuck:
            exp.status = STATUS.FAILED
            session.query(Sample).filter(Sample.experiment_id == exp.id, Sample.status == STATUS.RUNNING) \
                .update({Sample.status: STATUS.FAILED}, synchronize_session=False)
            session.query(Pipeline).filter(Pipeline.experiment_id == exp.id, Pipeline.status == STATUS.RUNNING) \
                .update({Pipeline.status: STATUS.FAILED, Pipeline.finished_at: now,
                         Pipeline.error_msg: "interrupted: the server stopped while this step was running"},
                        synchronize_session=False)
        session.commit()
        return len(stuck)


# ---- the run itself (executes in a worker process) ----------------------------------------------------------

def _resolve(runner: str):
    """Turn "package.module:function" into the function (the runner is a string so it can be replaced in tests)."""
    module, _, attr = runner.partition(":")
    return getattr(importlib.import_module(module), attr)


def execute_pipeline_run(study_id: str, experiment_pk: int, run_id: str, runner: str = DEFAULT_RUNNER) -> None:
    """
    Always runs the full dataset-profile batch for the study (cohort-wide stages like alignment/stats aren't
    per-sample, so there is no partial-rerun mode); every sample is re-marked RUNNING regardless of prior status.
    The experiment is already RUNNING (claim_run). The pipeline package is imported here, not at module level,
    so the web process never loads the analysis stack.
    """
    mark_samples(experiment_pk, STATUS.RUNNING)
    on_event = make_stage_event_recorder(run_id, experiment_pk)
    try:
        context = _resolve(runner)(study_id, on_stage_event=on_event)
    except Exception as exc:
        logger.exception("pipeline run %s failed for %s", run_id, study_id)
        record_run_failure(run_id, experiment_pk, f"{type(exc).__name__}: {exc}")
        set_experiment_status(study_id, STATUS.FAILED)
        mark_samples(experiment_pk, STATUS.FAILED)
        return

    set_experiment_status(study_id, STATUS.SUCCESS)
    missing = context.qc_metrics.get("ingestion", {}).get("missing_sample_ids", [])
    all_ids = list(context.sample_metadata["sample_id"])
    found_ids = [sid for sid in all_ids if sid not in missing]
    if missing:
        mark_samples(experiment_pk, STATUS.FAILED, sample_ids=missing)
    mark_samples(experiment_pk, STATUS.SUCCESS, sample_ids=found_ids)


# ---- the pool (API process) ---------------------------------------------------------------------------------

def _get_pool() -> ProcessPoolExecutor:
    """The shared worker pool, created on first use."""
    global _pool
    if _pool is None:
        # "spawn": a forked copy of a running web server (open DB connections, event loop) is not safe to reuse
        _pool = ProcessPoolExecutor(max_workers=max(1, settings.lcms_max_workers),
                                    mp_context=multiprocessing.get_context("spawn"))
    return _pool


def _run_finished(study_id: str, experiment_pk: int, run_id: str, future: Future) -> None:
    """Only reached with an exception when the worker itself died (killed, out of memory): execute_pipeline_run
    handles ordinary pipeline errors inside the worker."""
    global _pool
    exc = future.exception()
    if exc is None:
        return
    logger.error("worker for run %s died: %r", run_id, exc)
    if type(exc).__name__ == "BrokenProcessPool":
        _pool = None           # a broken pool refuses new work; the next run builds a fresh one
    record_run_failure(run_id, experiment_pk, f"worker process died: {exc!r}")
    set_experiment_status(study_id, STATUS.FAILED)
    mark_samples(experiment_pk, STATUS.FAILED)


def submit_run(study_id: str, experiment_pk: int, run_id: str, runner: str = DEFAULT_RUNNER) -> None:
    """Queue the run on the pool. If that is impossible the run is recorded as failed and the exception is re-raised (HTTP 503)."""
    try:
        future = _get_pool().submit(execute_pipeline_run, study_id, experiment_pk, run_id, runner)
    except Exception as exc:
        record_run_failure(run_id, experiment_pk, f"could not start worker: {exc!r}")
        set_experiment_status(study_id, STATUS.FAILED)
        raise
    future.add_done_callback(lambda f: _run_finished(study_id, experiment_pk, run_id, f))


def shutdown_pool() -> None:
    """Stop the pool without waiting for running work (used by tests and clean shutdown)."""
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
