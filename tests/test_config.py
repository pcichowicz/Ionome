"""Config schema tests: shipped YAMLs load, typos fail, profile rules hold. No pipeline imports needed."""
import copy
import dataclasses
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from core.config_schema import ExperimentConfig, Statistics, ignored_settings

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "experiments"
YAMLS = sorted(CONFIG_DIR.glob("*.yaml"))


def _load(path: Path) -> dict:
    d = yaml.safe_load(path.read_text())
    if path.name == "template.yaml":          # the template ships with blank study fields
        d["study"] = {"id": "t", "polarity": "negative", "instrument": "Orbitrap"}
        d["files"] = {"sample_metadata": "t_sample_metadata.csv"}
    return d


@pytest.fixture(params=YAMLS, ids=[p.stem for p in YAMLS])
def raw(request):
    return _load(request.param)


@pytest.fixture
def tmpl():
    """The template carries every option of both profiles, so it can be switched between them."""
    return _load(CONFIG_DIR / "template.yaml")


def test_shipped_yamls_load(raw):
    """Every YAML in config/experiments is a valid config."""
    ExperimentConfig(**raw)


def test_unknown_key_rejected_in_every_section(raw):
    """An unknown key in any section (or at top level) is rejected."""
    for section, value in raw.items():
        if isinstance(value, dict):
            bad = copy.deepcopy(raw)
            bad[section]["no_such_option"] = 1
            with pytest.raises(ValidationError, match="no_such_option"):
                ExperimentConfig(**bad)
    bad = copy.deepcopy(raw)
    bad["no_such_section"] = {}
    with pytest.raises(ValidationError):
        ExperimentConfig(**bad)


def test_misspelled_option_is_caught(raw):
    """A typo in an option name fails loudly instead of being silently ignored."""
    bad = copy.deepcopy(raw)
    bad["feature_detection"]["snr_treshold"] = 5.0       # the typo this schema exists to catch
    with pytest.raises(ValidationError, match="snr_treshold"):
        ExperimentConfig(**bad)


@pytest.mark.parametrize("profile,section", [("standards_only", "system_suitability"),
                                             ("cohort_with_qc", "feature_alignment")])
def test_profile_requires_its_sections(tmpl, profile, section):
    """Each profile fails validation when a section it needs is missing."""
    bad = copy.deepcopy(tmpl)
    bad["dataset_profile"] = profile
    bad.pop(section, None)
    with pytest.raises(ValidationError, match=section):
        ExperimentConfig(**bad)


def test_standards_profile_requires_standards_options(tmpl):
    """The standards_only profile needs its own options (e.g. blank_qc tolerances)."""
    bad = copy.deepcopy(tmpl)
    bad["dataset_profile"] = "standards_only"
    bad["blank_qc"].pop("mz_tolerance_ppm", None)
    bad["spectral_purity"].pop("min_purity", None)
    with pytest.raises(ValidationError, match="blank_qc.mz_tolerance_ppm"):
        ExperimentConfig(**bad)


def test_cohort_profile_does_not_need_standards_options(tmpl):
    """The cohort_with_qc profile loads without the standards-only options."""
    ok = copy.deepcopy(tmpl)
    ok["dataset_profile"] = "cohort_with_qc"
    ok.pop("system_suitability", None)
    for k in ("mz_tolerance_ppm", "rt_tolerance_sec"):
        ok["blank_qc"].pop(k, None)
    for k in ("primary_adduct", "ppm_tolerance", "candidate_adducts"):
        ok["adduct_annotation"].pop(k, None)
    ok["spectral_purity"].pop("min_purity", None)
    ExperimentConfig(**ok)


def test_value_rules(raw):
    """Invalid values (bad ranges, unknown method names, inconsistent thresholds) are rejected."""
    for section, key, value in [
        ("feature_detection", "peak_width", [60, 10]),
        ("feature_detection", "method", "xcms"),
        ("feature_alignment", "algorithm", "magic"),
        ("spectral_purity", "min_purity_tentative", 0.95),   # above the high-confidence tier
        ("statistics", "test_method", "anova"),
        ("library_matching", "similarity_metric", "dot"),
    ]:
        bad = copy.deepcopy(raw)
        bad.setdefault(section, {})[key] = value
        with pytest.raises(ValidationError):
            ExperimentConfig(**bad)


def test_primary_adduct_must_be_a_candidate(raw):
    """The primary adduct must be one of the candidate adducts."""
    bad = copy.deepcopy(raw)
    bad["adduct_annotation"]["primary_adduct"] = "[M+K-2H]-"
    bad["adduct_annotation"]["candidate_adducts"] = ["[M-H]-"]
    with pytest.raises(ValidationError, match="primary adduct"):
        ExperimentConfig(**bad)


def test_statistics_prep_kwargs_match_stats_prep_params():
    """StatsPrepParams(**statistics.model_dump()) raises TypeError on test_method; prep_kwargs() must not."""
    sp = pytest.importorskip("analysis.utils.stats_prep")
    fields = {f.name for f in dataclasses.fields(sp.StatsPrepParams)}
    kwargs = Statistics().prep_kwargs()
    assert set(kwargs) <= fields, set(kwargs) - fields
    sp.StatsPrepParams(**kwargs)


def test_ignored_settings(tmpl):
    """ignored_settings lists options the chosen profile does not read, and options no stage reads at all."""
    cohort = copy.deepcopy(tmpl)
    cohort["dataset_profile"] = "cohort_with_qc"
    cohort["spectral_purity"]["min_purity"] = 0.9
    assert "spectral_purity.min_purity" in ignored_settings(ExperimentConfig(**cohort))
    assert "spectral_purity.min_purity" not in ignored_settings(ExperimentConfig(**cohort), only_changed=True)  # no schema default
    cohort["library_matching"]["validate_against_known_identity"] = True   # read by no stage in any profile
    assert "library_matching.validate_against_known_identity" in ignored_settings(ExperimentConfig(**cohort), True)
    standards = copy.deepcopy(tmpl)
    standards["blank_qc"]["blank_fold_threshold"] = 9.0
    assert "blank_qc.blank_fold_threshold" in ignored_settings(ExperimentConfig(**standards), only_changed=True)
    standards["blank_qc"]["blank_fold_threshold"] = 5.0
    assert "blank_qc.blank_fold_threshold" not in ignored_settings(ExperimentConfig(**standards), only_changed=True)


@pytest.mark.parametrize("bad_id", ["../x", "/tmp/x", "a/b", "", ".x", "x y", "x" * 65])
def test_study_id_must_be_a_plain_name(tmpl, bad_id):
    """A study id with path or other unsafe characters is rejected."""
    bad = copy.deepcopy(tmpl)
    bad["study"]["id"] = bad_id
    bad["files"]["sample_metadata"] = f"{bad_id}_sample_metadata.csv"
    with pytest.raises(ValidationError):
        ExperimentConfig(**bad)


@pytest.mark.parametrize("bad_name", ["../t_sample_metadata.csv", "/etc/t_sample_metadata.csv", "other_sample_metadata.csv", "t.txt"])
def test_metadata_file_must_be_the_studys_own_csv(tmpl, bad_name):
    """files.sample_metadata must be a plain file name '<study_id>_sample_metadata.csv'."""
    bad = copy.deepcopy(tmpl)
    bad["study"]["id"] = "t"
    bad["files"]["sample_metadata"] = bad_name
    with pytest.raises(ValidationError):
        ExperimentConfig(**bad)
