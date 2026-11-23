"""The docs describe the pipelines that build_pipeline() really registers. Fails when someone changes a stage list and
forgets the docs; the diagrams are fixed with `python tools/gen_pipeline_diagrams.py`."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gen_pipeline_diagrams as gen   # noqa: E402

DOCS = ROOT / "docs"


@pytest.fixture(scope="module")
def table():
    return gen.stage_table()


def _names(table, profile):
    return [n for n, _ in table[profile]]


def test_diagrams_are_up_to_date():
    """The pipeline SVGs match what tools/gen_pipeline_diagrams.py generates from the code."""
    stale = [n for n, svg in gen.render().items() if (gen.OUT / n).read_text() != svg]
    assert not stale, f"{stale} out of date - run: python tools/gen_pipeline_diagrams.py"


def test_stage_order_text_matches_cohort_pipeline(table):
    """The stage order written on pipeline.html equals the real cohort pipeline."""
    html = (DOCS / "pipeline.html").read_text()
    m = re.search(r"Stage order:\s*<code>(.*?)</code>", html, re.S)
    assert m, "pipeline.html has no 'Stage order: <code>…</code>'"
    listed = [x.strip() for x in re.sub(r"\s+", " ", m.group(1)).split("&rarr;")]
    assert listed == _names(table, "cohort_with_qc")


def test_stage_counts_in_text_match_code(table):
    """Stage counts quoted in the docs equal the real number of cohort stages."""
    n = len(_names(table, "cohort_with_qc"))
    assert f"All {n} cohort-mode stages" in (DOCS / "stages.html").read_text()
    assert f"All {n} cohort-mode stages" in (DOCS / "development.html").read_text()
    assert f"Cohort-with-QC pipeline: {n} stages" in (DOCS / "pipeline.html").read_text()


def test_stages_page_lists_each_cohort_stage_once_in_run_order(table):
    """stages.html has exactly one card per cohort stage, in run order."""
    html = (DOCS / "stages.html").read_text()
    side = re.search(r'<nav class="sidebar">(.*?)</nav>', html, re.S).group(1)
    assert re.findall(r'href="#stage-[^"]+">([^<]+)<', side) == _names(table, "cohort_with_qc")
    for name in _names(table, "cohort_with_qc"):
        assert html.count(f'id="stage-{name.replace("_", "-")}"') == 1, name


def test_new_badges_match_the_code(table):
    """The 'new' and 'modified' badges on stages.html match how each stage differs from the standards profile."""
    html = (DOCS / "stages.html").read_text()
    for name, status in table["cohort_with_qc"]:
        card = re.search(rf'id="stage-{name.replace("_", "-")}">\s*<h3>(.*?)</h3>', html, re.S).group(1)
        assert ("badge new" in card) == (status == "new"), f"{name}: {card}"
        assert ("badge modified" in card) == (status == "modified"), f"{name}: {card}"
