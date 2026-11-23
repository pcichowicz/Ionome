"""
Pipeline tests that need no database, no environment variables and no mocks: the context is built from injected
paths, the stages are the real ones, the raw files are tiny synthetic mzML/mzXML files (tests/synth_raw.py).
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

import synth_raw
from core.config_schema import ExperimentConfig
from core.raw_files import find_raw_file, is_raw_file, list_raw_files

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "config" / "experiments" / "template.yaml"


def make_config(profile: str, study_id: str = "S1") -> ExperimentConfig:
    d = yaml.safe_load(TEMPLATE.read_text())
    d["study"] = {"id": study_id, "polarity": "negative", "instrument": "Orbitrap"}
    d["files"] = {"sample_metadata": f"{study_id}_sample_metadata.csv"}
    d["dataset_profile"] = profile
    return ExperimentConfig(**d)


def make_context(tmp_path, profile="standards_only", samples=None, mzml_file_paths=None):
    from analysis.context import LCMSContext
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    samples = samples if samples is not None else pd.DataFrame(
        {"sample_id": ["a"], "sample_role": ["standard"], "expected_compounds": ['["X"]'], "expected_mz": ["[100.0]"]})
    return LCMSContext(
        study_id="S1", experiment_id=1, config=make_config(profile),
        results_dir=tmp_path / "results", mzml_dir=raw,
        sample_metadata=samples, mzml_file_paths=mzml_file_paths if mzml_file_paths is not None else [],
    )


# ---------------------------------------------------------------- importing needs nothing
def _clean_env_run(code: str, tmp_path):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "MPLCONFIGDIR": str(tmp_path / "mpl"),
           "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)


def test_importing_the_orchestrator_needs_no_environment(tmp_path):
    """Importing the pipeline needs no .env and creates no files."""
    before = {p for p in tmp_path.rglob("*")}
    r = _clean_env_run("import analysis.pipeline_orchestrator, core.db", tmp_path)
    assert r.returncode == 0, r.stderr[-800:]
    created = {p for p in tmp_path.rglob("*") if "mpl" not in p.parts and p.suffix != ".pyc"} - before
    assert not created, f"importing created files: {sorted(created)}"


def test_context_with_injected_inputs_never_touches_settings_or_db(tmp_path):
    """A context built with injected inputs works without settings or database."""
    code = f"""
import sys, pandas as pd
sys.path.insert(0, {str(ROOT / 'tests')!r})
from test_pipeline import make_context
from pathlib import Path
ctx = make_context(Path({str(tmp_path)!r}))
assert 'core.db' not in sys.modules, 'core.db was imported'
assert ctx.cache_dir.is_dir() and ctx.plots_dir.is_dir()
print('ok')
"""
    r = _clean_env_run(code, tmp_path)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]


# ---------------------------------------------------------------- raw files
def test_find_raw_file_formats_and_case(tmp_path):
    """find_raw_file finds .mzML and .mzXML in any letter case and prefers .mzML when both exist."""
    (tmp_path / "a.mzXML").write_text("x")
    (tmp_path / "b.mzML").write_text("x")
    (tmp_path / "c.MZML").write_text("x")
    (tmp_path / "both.mzXML").write_text("x")
    (tmp_path / "both.mzML").write_text("x")
    assert find_raw_file(tmp_path, "a").name == "a.mzXML"
    assert find_raw_file(tmp_path, "b").name == "b.mzML"
    c = find_raw_file(tmp_path, "c")
    assert c is not None and c.samefile(tmp_path / "c.MZML")
    assert c.name.lower() == "c.mzml"      # macOS and Windows ignore case, so the spelling returned may differ
    assert find_raw_file(tmp_path, "both").name == "both.mzML"      # mzML preferred
    assert find_raw_file(tmp_path, "nope") is None
    assert [p.name for p in list_raw_files(tmp_path)] == sorted(["a.mzXML", "b.mzML", "c.MZML", "both.mzML", "both.mzXML"])
    assert is_raw_file("x.MzXml") and not is_raw_file("x.csv")


@pytest.mark.parametrize("bad", ["../a", "/etc/passwd", "a/b", ""])
def test_find_raw_file_refuses_unsafe_ids(tmp_path, bad):
    """find_raw_file returns None for ids that are not plain file names."""
    (tmp_path / "a.mzML").write_text("x")
    assert find_raw_file(tmp_path / "sub", bad) is None
    assert find_raw_file(tmp_path, bad) is None


@pytest.mark.parametrize("fmt,writer", [("mzML", synth_raw.write_mzml), ("mzXML", synth_raw.write_mzxml)])
def test_readers_agree_across_formats(tmp_path, fmt, writer):
    """The scan, precursor and purity readers give the same answers for mzML and mzXML."""
    from analysis.utils.mzml_parsing import (
        PyteomicsPrecursorReader, PyteomicsSpectralPurityReader, build_ms2_index, load_ms1_scans)
    p = writer(tmp_path / f"a.{fmt}", synth_raw.SCANS)
    assert len(load_ms1_scans(p)) == 2
    assert PyteomicsPrecursorReader().get_precursor_mzs(p) == [100.0, 100.0]
    assert len(build_ms2_index(p)) == 2
    # scan 1: 800 / (800 + 200) = 0.8, scan 3: 900 / 1000 = 0.9 -> mean 0.85 (was mzML-only before)
    assert PyteomicsSpectralPurityReader().compute_precursor_purity(p, 100.0, 1.0) == pytest.approx(0.85)


# ---------------------------------------------------------------- stage graph
def _pipeline(profile, tmp_path):
    from analysis.pipeline_orchestrator import build_pipeline
    ctx = make_context(tmp_path, profile)
    return build_pipeline(ctx.config, ctx), ctx


@pytest.mark.parametrize("profile", ["standards_only", "cohort_with_qc"])
def test_shipped_pipelines_have_a_valid_graph(profile, tmp_path):
    """Both shipped profiles have all required stages in a valid order."""
    pipe, _ = _pipeline(profile, tmp_path)
    assert pipe.validate_graph() == []


def test_stage_lists_match_the_documented_counts(tmp_path):
    """The profiles register 11 (standards) and 15 (cohort) stages, and the cohort one has no system_suitability."""
    std, _ = _pipeline("standards_only", tmp_path)
    coh, _ = _pipeline("cohort_with_qc", tmp_path)
    assert len(std._stages) == 11 and len(coh._stages) == 15
    assert "system_suitability" not in [s._name for s in coh._stages]


def test_moving_a_stage_before_its_requirement_is_reported(tmp_path):
    """Putting a stage before one it requires is reported by validate_graph."""
    pipe, _ = _pipeline("cohort_with_qc", tmp_path)
    names = [s._name for s in pipe._stages]
    i, j = names.index("plsda"), names.index("library_assembly")
    pipe._stages[i], pipe._stages[j] = pipe._stages[j], pipe._stages[i]
    problems = pipe.validate_graph()
    assert any("plsda requires 'library_assembly'" in p for p in problems), problems


def test_missing_duplicate_and_unnamed_stages_are_all_reported_at_once(tmp_path):
    """A bad pipeline reports every problem (missing, duplicate, unnamed stage) in one list."""
    from analysis.pipeline import LCMSPipeline
    from analysis.stages.StatPlots import XICPlottingStage
    from analysis.stages.ingestion import MzMLIngestionStage
    pipe = LCMSPipeline()
    pipe.register_stage(MzMLIngestionStage(tmp_path)).register_stage(MzMLIngestionStage(tmp_path))
    pipe.register_stage(XICPlottingStage())
    pipe.register_stage(object())
    text = "\n".join(pipe.validate_graph())
    assert "registered twice" in text
    assert "requires 'library_assembly', which is not in this pipeline" in text
    assert "requires 'volcano_plot', which is not in this pipeline" in text
    assert "has no _name" in text


def test_an_invalid_pipeline_fails_before_any_stage_runs(tmp_path):
    """An invalid pipeline raises before any stage executes."""
    from analysis.pipeline import FatalStageError, LCMSPipeline

    ran = []

    class Probe:
        _name = "probe"
        requires = ()
        def validate_input(self, c): ran.append("probe"); return True
        def execute(self, c): return c
        def validate_output(self, c): return True

    class NeedsMissing(Probe):
        _name = "needs_missing"
        requires = ("nothing",)

    pipe = LCMSPipeline().register_stage(Probe()).register_stage(NeedsMissing())
    with pytest.raises(FatalStageError, match="invalid pipeline"):
        pipe.run(make_context(tmp_path))
    assert ran == []


def test_run_by_name_still_checks_requirements(tmp_path):
    """Running one stage by name still requires the results of its prerequisites."""
    from analysis.pipeline import FatalStageError
    pipe, ctx = _pipeline("cohort_with_qc", tmp_path)
    with pytest.raises(FatalStageError, match="needs results of library_assembly"):
        pipe.run_by_name("volcano_plot", ctx)


def test_blank_qc_requirement_follows_its_mode():
    """blank_qc requires feature_detection in standards mode and feature_alignment in feature-table mode."""
    from analysis.stages.blank_qc import BlankQCStage
    assert BlankQCStage(reader=None).requires == ("feature_detection",)
    assert BlankQCStage(reader=None, mode="feature_table").requires == ("feature_alignment",)


# ---------------------------------------------------------------- real stages, real files, both formats
@pytest.mark.parametrize("fmt,writer", [("mzML", synth_raw.write_mzml), ("mzXML", synth_raw.write_mzxml)])
def test_standards_front_half_runs_on_either_format(tmp_path, fmt, writer):
    """The first standards stages run end to end on both mzML and mzXML input."""
    from analysis.pipeline import LCMSPipeline
    from analysis.stages.ingestion import MzMLIngestionStage
    from analysis.stages.scan_summary import ScanSummaryCacheStage
    from analysis.stages.spectral_purity import SpectralPurityStage
    from analysis.stages.system_suitability import SystemSuitabilityStage
    from analysis.utils.mzml_parsing import PyteomicsPrecursorReader, PyteomicsSpectralPurityReader

    samples = pd.DataFrame({
        "sample_id": ["a", "ghost"], "sample_role": ["standard", "standard"],
        "expected_compounds": ['["X"]', '["X"]'], "expected_mz": ["[100.0]", "[100.0]"]})
    raw = tmp_path / "raw"
    raw.mkdir()
    p = writer(raw / f"a.{fmt}", synth_raw.SCANS)           # "ghost" has no file at all
    ctx = make_context(tmp_path, samples=samples, mzml_file_paths=[p])

    pipe = (LCMSPipeline()
            .register_stage(MzMLIngestionStage(raw))
            .register_stage(ScanSummaryCacheStage())
            .register_stage(SystemSuitabilityStage(reader=PyteomicsPrecursorReader(), ppm_tolerance=5.0))
            .register_stage(SpectralPurityStage(reader=PyteomicsSpectralPurityReader(), isolation_window_da=1.0)))
    ctx = pipe.run(ctx)

    q = ctx.qc_metrics
    assert q["ingestion"]["n_found"] == 1 and q["ingestion"]["missing_sample_ids"] == ["ghost"]   # used to crash
    assert (ctx.cache_dir / "scan_summary" / "a.parquet").exists()
    ss = q["system_suitability"]["results"]
    assert ss["a"]["matches"][0]["confirmed"] is True
    assert ss["ghost"]["status"] == "skipped_missing_mzml"
    assert q["spectral_purity"]["results"]["a"]["status"] != "skipped_missing_raw_file"
    json.dumps(q["spectral_purity"], default=str)


def test_xic_with_no_significant_features_completes(tmp_path):
    """Used to die on `self.name` (the attribute is _name) and surface as 'unexpected error'."""
    from analysis.stages.StatPlots import XICPlottingStage
    ctx = make_context(tmp_path, "cohort_with_qc")
    ctx.qc_metrics["volcano_plot"] = {}
    ctx.volcano_significant_features = {"A_vs_B": []}
    out = XICPlottingStage().execute(ctx)
    assert out.qc_metrics["xic_plotting"] == {"n_features_plotted": 0}
