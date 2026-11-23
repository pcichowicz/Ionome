"""Stands in for analysis.pipeline_orchestrator.run_pipeline_for_study in a worker process."""
import os
import time

import pandas as pd


class _Context:
    qc_metrics = {"ingestion": {"missing_sample_ids": []}}
    sample_metadata = pd.DataFrame({"sample_id": ["s1", "s2"]})


def run(study_id, force_stages=None, on_stage_event=None):
    for stage in ("ingestion", "feature_detection"):
        on_stage_event(stage, "RUNNING", None)
        time.sleep(0.3 if study_id.startswith("slow") else 0.05)
        if study_id.startswith("boom") and stage == "feature_detection":
            on_stage_event(stage, "FAILED", "detector exploded")
            raise RuntimeError("detector exploded")
        if study_id.startswith("kill") and stage == "feature_detection":
            os._exit(1)                          # the worker disappears (out of memory, SIGKILL, ...)
        on_stage_event(stage, "SUCCESS", None)
    return _Context()
