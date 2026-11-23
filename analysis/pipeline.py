"""
The pipeline runner: runs registered stages in order, with validation, caching and error handling.

For each stage `run_stage` does: check prerequisites -> validate input -> compute the cache key -> reuse the saved
checkpoint if nothing changed, otherwise execute -> validate output -> write the stage log and checkpoint.
Stage outcomes are reported through an optional callback (the web app uses it to record progress in the database).
Errors: `RecoverableStageError` lets the run continue; `FatalStageError` (or any unexpected exception) stops it.
"""
from __future__ import annotations
import logging
from pathlib import Path
import json
from typing import Callable, Optional

from analysis.context import LCMSContext
from analysis.utils.protocols import PipelineStage
from analysis.utils.cache_utils import (
    code_fingerprint, compute_cache_key, load_checkpoint, missing_outputs, save_checkpoint,
)
from analysis.utils.timer import timed

logger = logging.getLogger("LCMS PIPELINE")

class LCMSPipeline:
    """Registers stages, then runs them in order over a context."""

    def __init__(self) -> None:
        """Start with no stages; add them with `register_stage`."""
        self._stages: list[PipelineStage] = []

    def register_stage(self, stage: PipelineStage) -> "LCMSPipeline":
        """Append a stage (order of registration = order of execution); returns self so calls can be chained."""
        self._stages.append(stage)
        return self

    def validate_graph(self) -> list[str]:
        """
        Everything wrong with the stage list, found WITHOUT running anything: a stage without a name, two stages
        with the same name, a stage whose `requires` is not registered or comes after it, and `runs_after`
        stages that are registered but come later. Empty list = fine.
        """
        problems: list[str] = []
        position: dict[str, int] = {}
        for i, stage in enumerate(self._stages):
            name = getattr(stage, "_name", None)
            if not isinstance(name, str) or not name:
                problems.append(f"stage #{i + 1} ({type(stage).__name__}) has no _name")
                continue
            if name in position:
                problems.append(f"stage name {name!r} is registered twice (#{position[name] + 1} and #{i + 1})")
                continue
            position[name] = i
        for i, stage in enumerate(self._stages):
            name = getattr(stage, "_name", None)
            if name not in position or position[name] != i:
                continue
            for need in getattr(stage, "requires", ()):
                if need not in position:
                    problems.append(f"{name} requires {need!r}, which is not in this pipeline")
                elif position[need] > i:
                    problems.append(f"{name} requires {need!r}, but {need!r} is registered after it")
            for before in getattr(stage, "runs_after", ()):
                if before in position and position[before] > i:
                    problems.append(f"{name} must run after {before!r}, but {before!r} is registered after it")
        return problems

    def run(self,
            context: LCMSContext,
            force_stages: set[str] | None = None,
            on_stage_event: StageEventCallback | None = None) -> LCMSContext:
        """
        Run every registered stage in order after checking the stage graph.

        Args:
            context: the run context; stages read and extend it.
            force_stages: names of stages to recompute even if their cache matches; {"*"} forces all.
            on_stage_event: optional callback (stage_name, status, error_msg) for RUNNING / SUCCESS / FAILED.

        Returns:
            The context after the last stage.

        Raises:
            FatalStageError: the stage graph is invalid, or a stage failed fatally.
        """
        problems = self.validate_graph()
        if problems:   # before any compute: a mis-ordered pipeline should not run for an hour first
            raise FatalStageError("invalid pipeline:\n  - " + "\n  - ".join(problems))
        for stage in self._stages:
            print(f"Running stage: {stage._name}")
            force = bool(force_stages) and ("*" in force_stages or stage._name in force_stages)
            context = self.run_stage(stage, context, force=force, on_stage_event=on_stage_event)
            print("")
        return context

    @timed(lambda self, stage, context, *a, **kw: f"[{context.study_id}] {stage._name}")
    def run_stage(self,
                  stage: PipelineStage,
                  context: LCMSContext,
                  force: bool = False,
                  on_stage_event: StageEventCallback | None = None) -> LCMSContext:
        """
        Run a single stage with validation, error handling, and logging, and cache checkpoint.

        Kept separate from run() so a single stage can be re-executed on its own.
        """
        if on_stage_event:
            on_stage_event(stage._name, "RUNNING", None)
        logger.info("stage.start", extra={"stage": stage._name})

        missing_up = [n for n in getattr(stage, "requires", ()) if n not in context.qc_metrics]
        if missing_up:   # also guards run_by_name(), which skips the graph check
            msg = f"{stage._name}: needs results of {', '.join(missing_up)}, which have not run"
            if on_stage_event:
                on_stage_event(stage._name, "FAILED", msg)
            raise FatalStageError(msg)

        try:
            input_ok = stage.validate_input(context)
        except FatalStageError as exc:
            if on_stage_event:
                on_stage_event(stage._name, "FAILED", str(exc))
            raise
        if not input_ok:
            msg = f"{stage._name}: input validation failed"
            if on_stage_event:
                on_stage_event(stage._name, "FAILED", msg)
            raise FatalStageError(msg)

        cache_key_fn = getattr(stage, "cache_key", None)
        code = code_fingerprint(stage)
        key: str | None = None
        checkpoint = None

        if cache_key_fn is not None:
            # Always computed BEFORE execute (also when forced): a key taken after a stage has
            # changed its own inputs would never match on the next run.
            try:
                own_key = cache_key_fn(context)
                # the final key also covers every upstream stage and this stage's code
                key = compute_cache_key({
                    "stage": stage._name,
                    "stage_key": own_key,
                    "upstream": context.stage_chain,
                    "code": code,
                })
                if not force:
                    checkpoint = load_checkpoint(context.results_dir, stage._name)
            except Exception as exc:
                logger.warning("stage.cache_key_error", extra={"stage": stage._name, "error": str(exc)})
                print(f"  cache check failed for {stage._name} ({exc}) -- running normally")
                key = None
                checkpoint = None

        if key is not None and checkpoint is not None and checkpoint.get("key") == key:
            expected = getattr(stage, "expected_outputs", None)
            gone = missing_outputs(expected(context)) if expected is not None else []
            if gone:
                print(f"  cached result for {stage._name} is stale ({gone[0].name} missing) -- re-running")
            else:
                context.qc_metrics[stage._name] = checkpoint["qc_metrics"]
                if checkpoint.get("log_entry"):
                    context.processing_log.append(checkpoint["log_entry"])

                # Re-write the log file even on a cache hit, so results/logs/ reflects
                # every stage that ran THIS pipeline invocation, cached or not.
                save_stage_log(
                    context.results_dir,
                    stage._name,
                    checkpoint["qc_metrics"],
                    checkpoint.get("log_entry"),
                )
                context.stage_chain = key
                logger.info("stage.cache_hit", extra={"stage": stage._name})
                print(f"  cached (inputs and code unchanged) -- skipping {stage._name}")
                if on_stage_event:
                    on_stage_event(stage._name, "SUCCESS", None)
                return context

        try:
            context = stage.execute(context)
        except RecoverableStageError as exc:
            logger.warning(
                "stage.recoverable_error",
                extra={"stage": stage._name, "error": str(exc)},
            )
            context.log_step(stage._name, parameters={}, warnings=[str(exc)])
            context.stage_chain = compute_cache_key({"prev": context.stage_chain, "stage": stage._name,
                                                     "recoverable": str(exc)})
            if on_stage_event:
                on_stage_event(stage._name, "SUCCESS", str(exc))  # recoverable = stage still "completed"
            return context

        except FatalStageError as exc:
            if on_stage_event:
                on_stage_event(stage._name, "FAILED", str(exc))
            raise
        except Exception as exc:  # deliberate catch-all at the stage boundary
            if on_stage_event:
                on_stage_event(stage._name, "FAILED", str(exc))
            raise FatalStageError(f"{stage._name}: unexpected error: {exc}") from exc

        if not stage.validate_output(context):
            msg = f"{stage._name}: output validation failed"
            if on_stage_event:
                on_stage_event(stage._name, "FAILED", msg)
            raise FatalStageError(msg)

        # Always persist this stage's log + qc_metrics, regardless of caching support --
        # this is a human-readable audit trail, separate from the cache-hit checkpoint below.
        save_stage_log(
            context.results_dir,
            stage._name,
            context.qc_metrics.get(stage._name),
            context.processing_log[-1] if context.processing_log else None,
        )

        if key is not None:
            try:
                log_entry = context.processing_log[-1] if context.processing_log else None
                save_checkpoint(
                    context.results_dir,
                    stage._name,
                    key,
                    context.qc_metrics.get(stage._name),
                    log_entry,
                )
            except Exception as exc:
                # A checkpoint-writing problem should not undo a stage that just completed.
                logger.warning(
                    "stage.checkpoint_save_error",
                    extra={"stage": stage._name, "error": str(exc)},
                )
                print(f"  couldn't save checkpoint for {stage._name} ({exc}) -- will re-run next time")
        # stages without a cache key are assumed deterministic given upstream + their own code
        context.stage_chain = key if key is not None else compute_cache_key(
            {"prev": context.stage_chain, "stage": stage._name, "code": code})

        logger.info("stage.complete", extra={"stage": stage._name})
        if on_stage_event:
            on_stage_event(stage._name, "SUCCESS", None)
        return context

    def run_by_name(self, stage_name: str, context: LCMSContext) -> LCMSContext:
        """Re-run a single registered stage by name -- for debugging one
        stage at a time without re-running everything before it."""
        for stage in self._stages:
            if stage._name == stage_name:
                return self.run_stage(stage, context)
        raise ValueError(f"No registered stage named {stage_name!r}")

def save_stage_log(results_dir: Path, stage_name: str, qc_metrics: dict | None, log_entry: dict | None) -> None:
    """Write `<results>/logs/<stage>.json` (metrics + provenance entry): the human-readable audit trail of one stage."""
    out_dir = results_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stage_name}.json", "w") as f:
        json.dump({"stage": stage_name, "qc_metrics": qc_metrics, "log_entry": log_entry}, f, indent=2, default=str)

class StageError(Exception):
    """
    Base class for stage-level errors.
    """

class RecoverableStageError(StageError):
    """
    Stage failed for this input, but the pipeline should continue.

    Example: one standard's expected ion wasn't detected. Log it, flag
    it, move on -- don't kill the whole run over one compound.
    """

class FatalStageError(StageError):
    """
    Stage failed in a way that invalidates the whole run; pipeline stops.

    Example: the reference library file is missing, or every sample
    failed feature detection (instrument/config problem, not a
    single-compound problem)
    """

StageEventCallback = Callable[[str, str, Optional[str]], None]
# args: stage_name, status ("RUNNING" | "SUCCESS" | "FAILED"), error_msg or None