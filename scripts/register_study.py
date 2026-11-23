"""
Registers a study from files you already have on disk (no web server needed), and optionally runs it.

    uv run python scripts/register_study.py --config config/experiments/my_study.yaml
    uv run python scripts/register_study.py --config config/experiments/my_study.yaml --run
    uv run python scripts/register_study.py --config config/experiments/my_study.yaml --run --force feature_detection

Expects (paths come from your .env):
    <LCMS_RAW_DATA_DIR>/<study.id>/<files.sample_metadata>      the sample sheet (CSV)
    <LCMS_RAW_DATA_DIR>/<study.id>/<sample_id>.mzML (or .mzXML) one raw file per row of the sheet

What it does, every time you run it (safe to repeat):
  * validates the config, creates the database tables if needed
  * new study: creates it (same checks and writes as the 'New study' form)
  * existing study: copies the config file over the registered copy in YAML_DIR (so edits to the YAML you keep under
    config/experiments take effect) and registers samples that were added to the sheet since
  * with --run: runs the pipeline; unchanged stages are skipped from the cache, --force STAGE[,STAGE] (or '*') redoes them
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    """Returns the exit code: 0 ok, 1 the study could not be created or is already running, 2 bad setup (settings, config, sample sheet)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="study YAML, e.g. config/experiments/my_study.yaml")
    ap.add_argument("--run", action="store_true", help="run the pipeline after registering")
    ap.add_argument("--force", default="", help="comma-separated stages to recompute, or '*' for all (with --run)")
    a = ap.parse_args()

    from core.settings import SettingsError, settings
    try:
        raw_root, yaml_dir = settings.lcms_raw_data_dir, settings.yaml_dir
    except SettingsError as exc:
        print(exc, file=sys.stderr)
        return 2

    cfg_path = Path(a.config)
    if not cfg_path.is_file():
        print(f"config file not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg_dict = yaml.safe_load(cfg_path.read_text())

    from core.config_schema import load_experiment_config
    try:
        config = load_experiment_config(cfg_path)
    except Exception as exc:
        print(f"the config is not valid:\n{exc}", file=sys.stderr)
        return 2
    study_id = config.study.id

    raw_dir = raw_root / study_id
    sheet = raw_dir / config.files.sample_metadata
    if not sheet.is_file():
        print(f"sample sheet not found: {sheet}\n(raw files and the sheet go in {raw_dir}; check LCMS_RAW_DATA_DIR in .env "
              f"and files.sample_metadata in the config)", file=sys.stderr)
        return 2
    rows = pd.read_csv(sheet, dtype=str, keep_default_na=False).to_dict("records")

    # make sure the database tables exist (the web app does this at startup; the script may run before it ever has)
    from sqlalchemy import inspect
    from core.db import get_sync_engine
    from core.declarative_base import Base
    import core.models  # noqa: F401  (registers the tables)
    Base.metadata.create_all(get_sync_engine())

    from core.db import SyncSessionLocal
    from core.ingestion import ingest_samples_from_csv
    from core.models import Experiment

    with SyncSessionLocal() as session:
        exp = session.query(Experiment).filter(Experiment.study_id == study_id).first()
        exp_pk = exp.id if exp else None

    # new study: same validation and writes as the web form; existing study: refresh config and add new samples
    if exp_pk is None:
        from api.services.studies import StudyCreateError, create_study_files_and_ingest
        try:
            res = create_study_files_and_ingest(cfg_dict, rows)
        except StudyCreateError as exc:
            print("could not create the study:\n  - " + "\n  - ".join(exc.errors), file=sys.stderr)
            return 1
        ing = res.ingestion if isinstance(res.ingestion, dict) else dict(res.ingestion)
        exp_pk = res.experiment_id
        print(f"created study {study_id}: {ing['n_created']} samples registered; config copied to {res.yaml_path}")
    else:
        target = yaml_dir / f"{study_id}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cfg_path, target)
        ing = ingest_samples_from_csv(exp_pk, sheet, raw_dir)
        print(f"study {study_id} already registered: config re-copied to {target}; "
              f"{ing['n_created']} new samples registered, {ing['n_skipped_existing']} already known"
              + (f", {ing['n_skipped_missing_file']} without a raw file ({', '.join(map(str, ing['skipped_missing_file'][:5]))})"
                 if ing["n_skipped_missing_file"] else ""))

    if not a.run:
        print(f"next: uv run python scripts/{Path(__file__).name} --config {a.config} --run")
        return 0

    # Same bookkeeping as the web app's Run button (api.services.runs.execute_pipeline_run), so the study page,
    # the sample statuses and the per-stage run history show this run too.
    import uuid
    from api.services.runs import (claim_run, make_stage_event_recorder, mark_samples, record_run_failure,
                                   set_experiment_status)
    from analysis.pipeline_orchestrator import run_pipeline_for_study
    from core.status import STATUS

    force = {s.strip() for s in a.force.split(",") if s.strip()} or None
    if not claim_run(study_id):
        print(f"{study_id} is marked as already running (another run, or one that was killed). If nothing is "
              f"running, start the web app once (it resets interrupted runs) and try again.", file=sys.stderr)
        return 1
    run_id = str(uuid.uuid4())
    mark_samples(exp_pk, STATUS.RUNNING)
    try:
        context = run_pipeline_for_study(study_id, force_stages=force,
                                         on_stage_event=make_stage_event_recorder(run_id, exp_pk))
    except BaseException as exc:          # includes Ctrl-C: never leave the study stuck on RUNNING
        record_run_failure(run_id, exp_pk, f"{type(exc).__name__}: {exc}")
        set_experiment_status(study_id, STATUS.FAILED)
        mark_samples(exp_pk, STATUS.FAILED)
        raise
    set_experiment_status(study_id, STATUS.SUCCESS)
    missing = context.qc_metrics.get("ingestion", {}).get("missing_sample_ids", [])
    found = [sid for sid in context.sample_metadata["sample_id"] if sid not in missing]
    if missing:
        mark_samples(exp_pk, STATUS.FAILED, sample_ids=missing)
    mark_samples(exp_pk, STATUS.SUCCESS, sample_ids=found)
    print(f"done. results: {settings.lcms_results_dir / study_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
